#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "../core/coulmat.h"
#include "../core/exx.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"

using namespace librpa_int;

static double exchange(const bool qstar, const bool clear_sector, const bool inversion = false)
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

    MeanField mf(1, 2, 1, 1);
    for (int ik = 0; ik < 2; ++ik)
    {
        mf.get_weight()[0](ik, 0) = 2.0 / 3.0;
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvectors()[0][0][ik].create(1, 1);
        mf.get_eigenvectors()[0][0][ik](0, 0) = 1.0;
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
    if (replicated != (qstar && !clear_sector))
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

    std::ostringstream filename;
    filename << exx.output_dir << "/ExxR_ispin_00_s_00_myid_" << std::setfill('0') << std::setw(5)
             << comm.myid << ".bin";
    std::ifstream input(filename.str(), std::ios::binary);
    auto read = [&](auto& value)
    {
        if (!input.read(reinterpret_cast<char*>(&value), sizeof(value)))
            throw std::runtime_error("truncated EXX matrix output");
    };
    std::size_t n_blocks;
    int dtype;
    read(n_blocks);
    read(dtype);
    if (dtype != 0) throw std::runtime_error("EXX output must retain real double type");
    std::vector<double> actual(pbc.Rlist.size(), 0.0), expected(pbc.Rlist.size(), 0.0);
    std::vector<int> owners(pbc.Rlist.size(), 0);
    std::size_t expected_offset =
        sizeof(n_blocks) + sizeof(dtype) + n_blocks * (7 * sizeof(int) + sizeof(std::size_t));
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
        expected_offset += sizeof(double);
    }
    if (std::filesystem::file_size(filename.str()) != expected_offset)
        throw std::runtime_error("invalid EXX file size");
    for (const auto& spin : exx.exx_IJR)
        for (const auto& bra : spin.second)
            for (const auto& ket : bra.second)
                for (const auto& I : ket.second)
                    for (const auto& J : I.second)
                        for (const auto& [R, block] : J.second)
                            expected.at(pbc.get_R_index(R)) -= block(0, 0);
    MPI_Allreduce(MPI_IN_PLACE, actual.data(), actual.size(), MPI_DOUBLE, MPI_SUM, comm.comm);
    MPI_Allreduce(MPI_IN_PLACE, expected.data(), expected.size(), MPI_DOUBLE, MPI_SUM, comm.comm);
    MPI_Allreduce(MPI_IN_PLACE, owners.data(), owners.size(), MPI_INT, MPI_SUM, comm.comm);
    for (std::size_t r = 0; r != actual.size(); ++r)
        if (owners[r] > 1 || std::abs(actual[r] - expected[r]) > 1e-12)
            throw std::runtime_error("EXX output block ownership or value mismatch");
    double local = 0.0;
    for (const auto& s : exx.exx_IJR)
        for (const auto& b : s.second)
            for (const auto& k : b.second)
                for (const auto& i : k.second)
                    for (const auto& j : i.second)
                        for (const auto& r : j.second)
                            if (r.first == Vector3_Order<int>{0, 0, 0}) local += r.second(0, 0);
    double total = 0;
    comm.allreduce(&local, &total, 1, MPI_SUM);
    return total;
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
        failure = std::abs(direct) < 1e-8 || std::abs(identity - direct) > 1e-11 ||
                  std::abs(full_sector - direct) > 1e-11 || std::abs(reduced - direct) > 1e-11;
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    return failure;
}
