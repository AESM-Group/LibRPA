#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "../core/coulmat.h"
#include "../core/exx.h"
#include "../io/global_io.h"
#include "../io/output_matrix.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"

using namespace librpa_int;

// The (isp = 0, bra, ket) channel of an EXX block container, or nullptr on a rank without it.
template <typename Map>
static const typename Map::mapped_type::mapped_type::mapped_type* find_channel(
    const Map& blocks, const int bra, const int ket)
{
    const auto isp = blocks.find(0);
    if (isp == blocks.end()) return nullptr;
    const auto b = isp->second.find(bra);
    if (b == isp->second.end()) return nullptr;
    const auto k = b->second.find(ket);
    return k == b->second.end() ? nullptr : &k->second;
}

static double exchange(const bool qstar, const bool clear_sector, const bool inversion = false,
                       const int n_spinor = 1)
{
    const auto& comm = global::mpi_comm_global_h;
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, {0, 0, 0, TWO_PI / 3, 0, 0},
                                    {{{0, 0, 0}}, {{1.0 / 3, 0, 0}, {-1.0 / 3, 0, 0}}});
    AtomicBasis basis(std::vector<std::size_t>{1});
    basis.set_l_shells({{0}});
    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec, pbc.G, {{0, 0}}, {{0, {0, 0, 0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    if (inversion)
    {
        SymmetryOperation inv;
        inv.rotation = Matrix3(-1, 0, 0, 0, -1, 0, 0, 0, -1);
        ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY, inv});
    }
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations(
        {-1, 0, LIBRPA_ANGULAR_ORDER_ABS_PM, LIBRPA_RSH_COEFF_M_1, LIBRPA_RSH_COEFF_1_M}, 0);
    ctx.build_kstar_member_rotations(0);
    ctx.set_available();
    if (!inversion && ctx.count_irreducible_blocks() != pbc.Rlist.size())
        throw std::runtime_error("Identity fixture must not reduce real-space sectors");
    if (inversion && ctx.count_irreducible_blocks() >= pbc.Rlist.size())
        throw std::runtime_error("Inversion fixture must reduce real-space sectors");
    if (clear_sector)
    {
        ctx.irreducible_sector.clear();
        ctx.rspace_sector_stars.clear();
    }

    MeanField mf(1, 2, 1, 1, n_spinor);
    for (int ik = 0; ik < 2; ++ik)
    {
        mf.get_weight()[0](ik, 0) = 2.0 / 3.0;
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        // One occupied state. For two components the weights 0.36 and 0.64 and the relative
        // phase i make all four (bra, ket) channels nonzero and the off-diagonal ones complex.
        for (int c = 0; c != n_spinor; ++c)
        {
            mf.get_eigenvectors()[0][c][ik].create(1, 1);
            mf.get_eigenvectors()[0][c][ik](0, 0) =
                n_spinor == 1 ? cplxdb{1.0, 0.0} : (c == 0 ? cplxdb{0.6, 0.0} : cplxdb{0.0, 0.8});
        }
    }
    Cs_LRI cs;
    atpair_k_cplx_mat_t vq;
    if (comm.myid == 0)
    {
        RI::Tensor<double> c({1, 1, 1});
        c(0, 0, 0) = 0.5;
        cs.data_libri[0][{0, {0, 0, 0}}] = c;
        for (const auto& q : pbc.klist)
        {
            auto v = std::make_shared<ComplexMatrix>(1, 1);
            (*v)(0, 0) = 2.0;
            vq[0][0][q] = v;
        }
    }
    bool replicated = false;
    const auto vr = FT_Vq(comm, basis, ctx, vq, pbc, true, qstar, &replicated);
    if (replicated != qstar)
        throw std::runtime_error("Unexpected Fourier output ownership receipt");
    // Use square BLACS groups: two ranks form two 1x1 groups, while four
    // ranks exercise the production 2x2 BLACS layout.
    const KPointBlacsProcessShape shape =
        comm.nprocs == 2 ? KPointBlacsProcessShape{2, 1} : KPointBlacsProcessShape{1, comm.nprocs};
    KPointBlacsParallelContext kctx(shape, comm.comm, 2);
    auto desc = kctx.create_array_desc(1, 1, 1, 1);
    // The band wave-function descriptor is unused by this fixture but must
    // still refer to a live object; the last flag enables the symmetry context.
    Exx exx(mf, basis, pbc, ctx, kctx, kctx, desc, desc, qstar, true);
    exx.output_dir = "test_exx_output_" + std::to_string(comm.nprocs) + "_" +
                     std::to_string(qstar) + std::to_string(clear_sector) +
                     std::to_string(inversion);
    if (comm.is_root()) std::filesystem::create_directories(exx.output_dir);
    comm.barrier();
    exx.output_exx_mat_r = true;
    exx.build(LIBRPA_ROUTING_LIBRI, basis, cs, vr, replicated);

    // Decode the file of one (bra, ket) channel, compare it with the in-memory blocks, and
    // return the sum over ranks of the R = 0 element, with the sign of the file.
    auto check_channel = [&](const auto& channel, auto scalar, const int bra, const int ket)
    {
        using T = decltype(scalar);
        constexpr int dtype = std::is_same_v<T, double> ? 0 : 1;
        const auto fn = rspace_matrix_filename(exx.output_dir, "ExxR", 0, bra, ket, comm.myid);
        std::ifstream input(fn, std::ios::binary);
        auto read = [&](auto& value)
        {
            if (!input.read(reinterpret_cast<char*>(&value), sizeof(value)))
                throw std::runtime_error("truncated EXX matrix output");
        };
        const auto header = read_rspace_header(input, fn);
        if (header.kind != RspaceKind::ExxR || header.nprocs != comm.nprocs ||
            header.myid != comm.myid)
            throw std::runtime_error("EXX output header does not describe its writer");
        if (header.dtype != dtype) throw std::runtime_error("EXX output has the wrong type");
        // All ranks of one output share the run identifier.
        auto run_min = header.run_id, run_max = header.run_id;
        MPI_Allreduce(MPI_IN_PLACE, &run_min, 1, MPI_UINT64_T, MPI_MIN, comm.comm);
        MPI_Allreduce(MPI_IN_PLACE, &run_max, 1, MPI_UINT64_T, MPI_MAX, comm.comm);
        if (run_min != run_max) throw std::runtime_error("EXX output files have different run ids");
        const auto n_blocks = static_cast<std::size_t>(header.n_blocks);
        std::vector<T> actual(pbc.Rlist.size(), T{}), expected(pbc.Rlist.size(), T{});
        std::vector<int> owners(pbc.Rlist.size(), 0);
        std::size_t expected_offset =
            rspace_header_bytes(header.kind) + n_blocks * (7 * sizeof(int) + sizeof(std::size_t));
        for (std::size_t b = 0; b != n_blocks; ++b)
        {
            int dims[7];
            std::size_t offset;
            read(dims);
            read(offset);
            if (dims[3] != 0 || dims[4] != 0 || dims[5] != 1 || dims[6] != 1 ||
                offset != expected_offset)
                throw std::runtime_error("invalid EXX block index");
            const auto next_record = input.tellg();
            const auto iR = pbc.get_R_index({dims[0], dims[1], dims[2]});
            input.seekg(offset);
            read(actual.at(iR));
            ++owners.at(iR);
            input.seekg(next_record);
            expected_offset += sizeof(T);
        }
        if (std::filesystem::file_size(fn) != expected_offset)
            throw std::runtime_error("invalid EXX file size");
        T local{};
        if (channel != nullptr)
            for (const auto& I : *channel)
                for (const auto& J : I.second)
                    for (const auto& [R, block] : J.second)
                    {
                        expected.at(pbc.get_R_index(R)) -= block(0, 0);
                        if (R == Vector3_Order<int>{0, 0, 0}) local += block(0, 0);
                    }
        // Real and imaginary parts are reduced separately, which is valid for both types.
        auto sum = [&](std::vector<T>& v)
        {
            MPI_Allreduce(MPI_IN_PLACE, v.data(), static_cast<int>(v.size() * sizeof(T) / sizeof(double)),
                          MPI_DOUBLE, MPI_SUM, comm.comm);
        };
        sum(actual);
        sum(expected);
        MPI_Allreduce(MPI_IN_PLACE, owners.data(), owners.size(), MPI_INT, MPI_SUM, comm.comm);
        double magnitude = 0.0;
        for (std::size_t r = 0; r != actual.size(); ++r)
        {
            if (owners[r] > 1 || std::abs(actual[r] - expected[r]) > 1e-12)
                throw std::runtime_error("EXX output block ownership or value mismatch");
            magnitude = std::max(magnitude, std::abs(actual[r]));
        }
        // A channel of zeros would hide a wrong payload stride.
        if (n_spinor > 1 && magnitude < 1e-8)
            throw std::runtime_error("EXX spinor channel is empty");
        // The off-diagonal channels of this fixture are purely imaginary.
        if constexpr (!std::is_same_v<T, double>)
            if (bra != ket)
            {
                double im = 0.0;
                for (const auto& v : actual) im = std::max(im, std::abs(v.imag()));
                if (im < 1e-8) throw std::runtime_error("EXX off-diagonal channel is not complex");
            }
        T total{};
        MPI_Allreduce(&local, &total, static_cast<int>(sizeof(T) / sizeof(double)), MPI_DOUBLE,
                      MPI_SUM, comm.comm);
        return total;
    };

    double result = 0.0;
    for (int bra = 0; bra != n_spinor; ++bra)
        for (int ket = 0; ket != n_spinor; ++ket)
        {
            if (n_spinor > 1)
            {
                const auto total = check_channel(find_channel(exx.exx_IJR_cplx, bra, ket),
                                                 cplxdb{}, bra, ket);
                if (bra == 0 && ket == 0) result = total.real();
            }
            else
                result = check_channel(find_channel(exx.exx_IJR, bra, ket), double{}, bra, ket);
        }
    return result;
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    int failure = 0;
    {
        const double direct = exchange(false, false);
        const double identity = exchange(true, false);
        const double full_sector = exchange(true, true);
        const double reduced = exchange(true, false, true);
        if (global::mpi_comm_global_h.is_root())
            std::cout << "OWNERSHIP_RESULT ranks=" << global::mpi_comm_global_h.nprocs
                      << " direct=" << direct << " identity=" << identity
                      << " full_sector=" << full_sector << " reduced=" << reduced << std::endl;
        // Two-component mean field: the complex writer runs, checked channel by channel.
        const double spinor = exchange(false, false, false, 2);
        failure = std::abs(spinor) < 1e-8 ||
                  std::abs(direct) < 1e-8 || std::abs(identity - direct) > 1e-11 ||
                  std::abs(full_sector - direct) > 1e-11 || std::abs(reduced - direct) > 1e-11;
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    return failure;
}
