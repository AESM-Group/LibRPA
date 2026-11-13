#include <cassert>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "../core/gw.h"
#include "../io/global_io.h"
#include "../io/output_matrix.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using namespace librpa_int;
namespace fs = std::filesystem;

// Byte positions in a SigcRF file, from the layout documented at RspaceFileHeader.
static constexpr std::streamoff POS_MAGIC = 0;
static constexpr std::streamoff POS_ENDIAN = 8;
static constexpr std::streamoff POS_VERSION = 12;
static constexpr std::streamoff POS_KIND = 16;
static constexpr std::streamoff POS_NPROCS = 20;
static constexpr std::streamoff POS_MYID = 24;
static constexpr std::streamoff POS_RUN_ID = 32;
static constexpr std::streamoff POS_FIRST_RECORD = 56;                         // 48 + omega
static constexpr std::streamoff POS_FIRST_OFFSET = POS_FIRST_RECORD + 7 * 4;  // after int[7]

static std::string sigc_file(const std::string &dir, const int iomega, const int myid)
{
    std::ostringstream name;
    name << dir << "/SigcRF_ispin_00_s_00_iomega_" << std::setfill('0') << std::setw(3) << iomega
         << "_myid_" << std::setw(5) << myid << ".bin";
    return name.str();
}

template <typename T>
static void patch(const std::string &fn, const std::streamoff pos, const T &value)
{
    std::fstream f(fn, std::ios::in | std::ios::out | std::ios::binary);
    assert(f);
    f.seekp(pos);
    f.write(reinterpret_cast<const char *>(&value), sizeof(value));
    assert(f);
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    {
        const auto &comm = global::mpi_comm_global_h;
        // Options: --write-dir D (default per rank count), --restart-dir D (default: the write
        // directory; another directory tests a restart with a different original rank count),
        // --clean (remove the write directory first).
        std::string directory = "test_sigc_output_" + std::to_string(comm.nprocs);
        std::string restart_directory;
        bool clean = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg = argv[i];
            if (arg == "--write-dir" && i + 1 < argc) directory = argv[++i];
            else if (arg == "--restart-dir" && i + 1 < argc) restart_directory = argv[++i];
            else if (arg == "--clean") clean = true;
            else assert(false && "unknown option");
        }
        if (restart_directory.empty()) restart_directory = directory;

        BlacsCtxtHandler blacs(comm.comm);
        blacs.init();
        blacs.set_square_grid();
        PeriodicBoundaryData pbc;
        pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        pbc.set_kgrids_kvec(2, 1, 1, {0, 0, 0, PI, 0, 0});
        AtomicBasis basis(std::vector<std::size_t>{2});
        AtomicBasis aux(std::vector<std::size_t>{1});
        SymmetryContext symmetry;
        MeanField mf(1, 2, 2, 2);
        for (int k = 0; k != 2; ++k)
        {
            mf.get_eigenvals()[0](k, 0) = -0.5;
            mf.get_eigenvals()[0](k, 1) = 0.7 + 0.1 * k;
            mf.get_weight()[0](k, 0) = 1.0;
            mf.get_weight()[0](k, 1) = 0.0;
            auto &wfc = mf.get_eigenvectors()[0][0][k];
            wfc.create(2, 2);
            wfc.zero_out();
            wfc(0, 0) = wfc(1, 1) = 1.0;
        }
        TFGrids grids(6);
        grids.generate_minimax(1.0, 8.0);
        KPointBlacsParallelContext kctx(KPointBlacsProcessShape{1, comm.nprocs}, comm.comm, 2);
        auto desc = kctx.create_array_desc(2, 2, 1, 1);
        auto desc_wc = kctx.create_array_desc(1, 1, 1, 1);
        Cs_LRI cs;
        cs.use_libri = true;
        if (comm.is_root())
        {
            RI::Tensor<double> c({1, 2, 2});
            c(0, 0, 0) = 0.3;
            c(0, 0, 1) = c(0, 1, 0) = 0.2;
            c(0, 1, 1) = 0.4;
            cs.data_libri[0][{0, {0, 0, 0}}] = c;
        }
        if (comm.is_root())
        {
            if (clean) fs::remove_all(directory);
            fs::create_directories(directory);
        }
        comm.barrier();
        G0W0 gw(mf, basis, pbc, symmetry, grids, kctx, kctx, desc, desc, false, false);
        gw.output_sigc_mat_rf = true;
        gw.output_sigc_ks_mat_kf = true;
        gw.output_dir = directory;
        std::map<double, std::map<Vector3_Order<double>, Matz>> wc;
        for (const auto omega : grids.get_freq_nodes())
            for (const auto &q : pbc.klist)
            {
                Matz full(1, 1, MAJOR::COL);
                full(0, 0) = -0.2 / (1.0 + omega * omega + q.x * q.x);
                wc[omega][q] = get_local_mat(full, desc_wc);
            }
        gw.build_spacetime(LIBRPA_ROUTING_LIBRI, aux, cs, wc, desc_wc);
        gw.build_sigc_matrix_KS_kgrid_blacs(blacs);

        // Headers describe the writer, and all files of the output share one run identifier.
        std::uint64_t run_id = 0;
        for (std::size_t f = 0; f != grids.size(); ++f)
        {
            const auto fn = sigc_file(directory, as_int(f), comm.myid);
            std::ifstream input(fn, std::ios::binary);
            const auto header = read_rspace_header(input, fn);
            assert(header.kind == RspaceKind::SigcRF && header.omega.has_value());
            assert(*header.omega == grids.get_freq_nodes()[f] && header.dtype == 1);
            assert(header.nprocs == comm.nprocs && header.myid == comm.myid);
            assert(header.n_blocks <= pbc.Rlist.size());
            if (f == 0) run_id = header.run_id;
            assert(header.run_id == run_id);
        }
        auto run_min = run_id, run_max = run_id;
        MPI_Allreduce(MPI_IN_PLACE, &run_min, 1, MPI_UINT64_T, MPI_MIN, comm.comm);
        MPI_Allreduce(MPI_IN_PLACE, &run_max, 1, MPI_UINT64_T, MPI_MAX, comm.comm);
        assert(run_min == run_max && run_min != 0);

        // Read a directory and compare the KS-basis result with the freshly built one.
        auto restart_and_compare = [&](const std::string &dir)
        {
            G0W0 restarted(mf, basis, pbc, symmetry, grids, kctx, kctx, desc, desc, false, false);
            restarted.output_sigc_ks_mat_kf = true;
            restarted.read_sigc(dir);
            restarted.build_sigc_matrix_KS_kgrid_blacs(blacs);
            double magnitude = 0.0;
            for (const auto &[spin, k_mats] : gw.sigc_is_ik_f_KS)
                for (const auto &[k, f_mats] : k_mats)
                    for (const auto &[f, expected] : f_mats)
                    {
                        const auto &actual = restarted.sigc_is_ik_f_KS.at(spin).at(k).at(f);
                        for (int i = 0; i != expected.nr(); ++i)
                            for (int j = 0; j != expected.nc(); ++j)
                            {
                                assert(std::abs(actual(i, j) - expected(i, j)) < 1e-12);
                                magnitude = std::max(magnitude, std::abs(expected(i, j)));
                            }
                    }
            MPI_Allreduce(MPI_IN_PLACE, &magnitude, 1, MPI_DOUBLE, MPI_MAX, comm.comm);
            assert(magnitude > 1e-8);
        };
        restart_and_compare(restart_directory);

        // Corrupted or foreign files must make read_sigc throw on every rank. Rank 0 alone
        // prepares a copy of the directory; all ranks then read it.
        auto expect_rejected = [&](const std::string &name, auto &&corrupt)
        {
            const auto bad_dir = directory + "_" + name;
            comm.barrier();
            if (comm.is_root())
            {
                fs::remove_all(bad_dir);
                fs::copy(directory, bad_dir, fs::copy_options::recursive);
                corrupt(bad_dir);
            }
            comm.barrier();
            G0W0 bad(mf, basis, pbc, symmetry, grids, kctx, kctx, desc, desc, false, false);
            bool rejected = false;
            try
            {
                bad.read_sigc(bad_dir);
            }
            catch (const std::exception &)
            {
                rejected = true;
            }
            assert(rejected);
        };
        // A file of the first frequency that holds a block, to corrupt its first record.
        auto first_nonempty = [&](const std::string &dir)
        {
            for (int id = 0; id != comm.nprocs; ++id)
            {
                const auto fn = sigc_file(dir, 0, id);
                std::ifstream input(fn, std::ios::binary);
                if (read_rspace_header(input, fn).n_blocks > 0) return fn;
            }
            assert(false && "no nonempty SigcRF file");
            return std::string();
        };
        expect_rejected("bad_R", [&](const std::string &dir)
                        { patch(first_nonempty(dir), POS_FIRST_RECORD, int{99}); });
        expect_rejected("bad_offset", [&](const std::string &dir)
                        { patch(first_nonempty(dir), POS_FIRST_OFFSET, std::size_t{1} << 40); });
        expect_rejected("bad_magic", [&](const std::string &dir)
                        { patch(sigc_file(dir, 0, 0), POS_MAGIC, 'X'); });
        expect_rejected("bad_endian", [&](const std::string &dir)
                        { patch(sigc_file(dir, 0, 0), POS_ENDIAN, std::uint32_t{0x04030201}); });
        expect_rejected("bad_version", [&](const std::string &dir)
                        { patch(sigc_file(dir, 0, 0), POS_VERSION, int{2}); });
        expect_rejected("bad_kind_first", [&](const std::string &dir)
                        { patch(sigc_file(dir, 0, 0), POS_KIND, int{0}); });
        expect_rejected("bad_kind_later", [&](const std::string &dir)
                        { patch(sigc_file(dir, 1, 0), POS_KIND, int{0}); });
        // The first file defines the set; more ranks than files of the directory is an error.
        expect_rejected("nprocs_first", [&](const std::string &dir)
                        { patch(sigc_file(dir, 0, 0), POS_NPROCS, int{99}); });
        expect_rejected("nprocs_later", [&](const std::string &dir)
                        { patch(sigc_file(dir, 1, 0), POS_NPROCS, int{comm.nprocs + 1}); });
        expect_rejected("myid_later", [&](const std::string &dir)
                        { patch(sigc_file(dir, 1, 0), POS_MYID, int{comm.nprocs}); });
        // A file from another run among the files of this run.
        expect_rejected("run_id_later", [&](const std::string &dir)
                        { patch(sigc_file(dir, 2, 0), POS_RUN_ID, run_id + 1); });
        expect_rejected("run_id_first", [&](const std::string &dir)
                        { patch(sigc_file(dir, 0, 0), POS_RUN_ID, run_id + 1); });
        expect_rejected("missing_shard", [&](const std::string &dir)
                        { fs::remove(sigc_file(dir, 2, comm.nprocs - 1)); });

        // Files of ranks beyond the writer's rank count are leftovers of another run: the
        // restart ignores them, even when they are garbage or a copy of a valid file.
        {
            const auto leftover_dir = directory + "_leftover";
            comm.barrier();
            if (comm.is_root())
            {
                fs::remove_all(leftover_dir);
                fs::copy(directory, leftover_dir, fs::copy_options::recursive);
                for (int f = 0; f != as_int(grids.size()); ++f)
                {
                    fs::copy_file(sigc_file(directory, f, 0),
                                  sigc_file(leftover_dir, f, comm.nprocs + 3));
                    fs::copy_file(sigc_file(directory, f, 0),
                                  sigc_file(leftover_dir, f, comm.nprocs + 9));
                }
                patch(sigc_file(leftover_dir, 0, comm.nprocs + 9), POS_RUN_ID, run_id + 1);
            }
            comm.barrier();
            restart_and_compare(leftover_dir);
        }
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
}
