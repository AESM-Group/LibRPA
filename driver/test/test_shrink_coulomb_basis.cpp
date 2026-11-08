// Regression test for the auxiliary-basis shrink dataset convention.
//
// When the compressed (shrink) auxiliary basis is enabled, the Coulomb
// matrices in the reader-v1 dataset are expanded in the COMPRESSED basis, not
// in the large basis that supplies the RI coefficients.  This mirrors the
// ABACUS convention and is what FHI-aims writes for
// `output librpa ... shrink_LU_inv_thr`.  The reader must therefore validate
// the on-disk Coulomb blocks against `basis_aux_shrink` whenever
// `use_shrink_abfs` is on, and against `basis_aux` otherwise.
//
// The test pins that contract by reading the same dataset with both switches:
//   * compressed-basis file: accepted with shrink, rejected without it
//   * large-basis file:      accepted without shrink, rejected with it
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../src/api/instance_manager.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../../src/utils/constants.h"
#include "../reader/reader_context.h"
#include "../reader/reader_coulomb.h"

namespace
{
constexpr int COULOMB_V1_MARKER = -20129433;

// Write a minimal reader-v1 Coulomb file holding one atom-pair block (0, 1).
// Layout: 6 int header, per-atom Naux, block table, then the real block.
void write_coulomb_v1(const std::filesystem::path &path,
                      const std::vector<int> &atom_naux,
                      const std::vector<double> &block)
{
    const int natoms = static_cast<int>(atom_naux.size());
    const int naux = std::accumulate(atom_naux.begin(), atom_naux.end(), 0);
    // marker, iq, Naux, real-valued flag, natoms, nblocks
    const int header[6] = {COULOMB_V1_MARKER, 1, naux, 0, natoms, 1};
    const std::int64_t offset = static_cast<std::int64_t>(sizeof(header)) +
                                static_cast<std::int64_t>(natoms) * sizeof(int) +
                                static_cast<std::int64_t>(sizeof(int) + sizeof(std::int64_t));
    const int ipair = 1;  // packed upper triangle: (0, 0), (0, 1), (1, 1)
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(header), sizeof(header));
    out.write(reinterpret_cast<const char *>(atom_naux.data()), natoms * sizeof(int));
    out.write(reinterpret_cast<const char *>(&ipair), sizeof(ipair));
    out.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
    out.write(reinterpret_cast<const char *>(block.data()),
              static_cast<std::streamsize>(block.size() * sizeof(double)));
    out.close();
    assert(out.good());
    assert(static_cast<std::int64_t>(std::filesystem::file_size(path)) ==
           offset + static_cast<std::int64_t>(block.size() * sizeof(double)));
}

// Large basis: 2 + 1 functions; compressed basis: 1 + 1 functions.
void set_up_handler(librpa::Handler &h)
{
    const double lattice[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double reciprocal[9] = {librpa_int::TWO_PI, 0, 0, 0, librpa_int::TWO_PI, 0, 0, 0,
                                  librpa_int::TWO_PI};
    h.set_latvec_and_G(lattice, reciprocal);
    h.set_scf_dimension(1, 1, 1, 1, 1);
    h.set_atoms({0, 0}, {0.0, 0.0, 0.0, 0.5, 0.5, 0.5});
    h.set_kgrids_kvec(1, 1, 1, std::vector<double>{0.0, 0.0, 0.0});
    h.set_ao_basis_aux({2, 1}, {{0, 0}, {0}});
    h.set_ao_basis_aux_shrink({1, 1}, {{0}, {0}});
}

void read_coulomb(const std::filesystem::path &dir, const std::string &prefix,
                  const bool is_cut, const bool use_shrink_basis)
{
    librpa::Handler h(MPI_COMM_SELF);
    set_up_handler(h);
    librpa::reader::ReaderState state;
    state.n_kpoints = 1;
    librpa::reader::ReaderContext ctx{h, state, {}, {},
                                      librpa_int::MpiCommHandler(MPI_COMM_SELF, true)};
    librpa::reader::read_Vq_full(ctx, dir.string(), prefix, is_cut,
                                 /*reader_version=*/1, use_shrink_basis);
}

void expect_read(const std::filesystem::path &dir, const std::string &prefix,
                 const bool is_cut, const bool use_shrink_basis,
                 const bool should_succeed, const std::string &label)
{
    bool succeeded = false;
    std::string error;
    try
    {
        read_coulomb(dir, prefix, is_cut, use_shrink_basis);
        succeeded = true;
    }
    catch (const std::exception &e)
    {
        error = e.what();
    }
    if (succeeded != should_succeed)
    {
        std::string message = label + ": expected the Coulomb read to " +
                              (should_succeed ? "succeed" : "be rejected");
        if (!error.empty()) message += " (error: " + error + ")";
        throw std::logic_error(message);
    }
}

void check_shrink_coulomb_basis_selection(const std::filesystem::path &root)
{
    const auto small_dir = root / "compressed";
    const auto large_dir = root / "large";
    std::filesystem::create_directory(small_dir);
    std::filesystem::create_directory(large_dir);

    write_coulomb_v1(small_dir / "coulomb_mat_1", {1, 1}, {0.25});
    write_coulomb_v1(small_dir / "coulomb_cut_1", {1, 1}, {0.125});
    write_coulomb_v1(large_dir / "coulomb_mat_1", {2, 1}, {0.25, -0.5});
    write_coulomb_v1(large_dir / "coulomb_cut_1", {2, 1}, {0.125, -0.25});

    // The compressed dataset is what a shrink run must consume.
    for (const bool is_cut: {false, true})
    {
        const std::string prefix = is_cut ? "coulomb_cut" : "coulomb_mat";
        expect_read(small_dir, prefix, is_cut, /*use_shrink_basis=*/true, true,
                    "compressed-basis " + prefix + " with shrink enabled");
        expect_read(small_dir, prefix, is_cut, /*use_shrink_basis=*/false, false,
                    "compressed-basis " + prefix + " without shrink");
        expect_read(large_dir, prefix, is_cut, /*use_shrink_basis=*/false, true,
                    "large-basis " + prefix + " without shrink");
        expect_read(large_dir, prefix, is_cut, /*use_shrink_basis=*/true, false,
                    "large-basis " + prefix + " with shrink enabled");
    }
}
}  // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
    librpa_int::global::init_global_io();
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("librpa_shrink_coulomb_" + std::to_string(stamp) + "_" +
                            std::to_string(rank));
    assert(std::filesystem::create_directory(directory));

    check_shrink_coulomb_basis_selection(directory);

    std::filesystem::remove_all(directory);
    if (rank == 0) std::printf("shrink Coulomb basis selection: passed\n");
    MPI_Finalize();
    return 0;
}
