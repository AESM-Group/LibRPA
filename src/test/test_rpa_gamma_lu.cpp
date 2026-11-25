#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "../core/epsilon.h"
#include "../io/global_io.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using namespace librpa_int;

namespace
{
void require_all(bool condition, const BlacsCtxtHandler &blacs, const char *message)
{
    int local = condition ? 1 : 0, all = 0;
    MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, blacs.comm_h().comm);
    if (!all) throw std::runtime_error(message);
}

CorrEnergy gamma_energy(const BlacsCtxtHandler &blacs, const std::string &backend,
                       bool zero_diagonal_block, bool gpu = false)
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
    AtomicBasis wfc_basis({2}), aux_basis({2});
    MeanField mf(1, 1, 2, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](0, 1) = 1.0;
    TFGrids grids(1);
    grids.generate_evenspaced_tf(0.1, 0.1, 0.1, 0.1);
    KPointBlacsParallelContext kctx(
        KPointBlacsProcessShape(1, blacs.nprocs), blacs.comm_h().comm, 1);
    ArrayDesc wfc_desc(blacs);
    wfc_desc.init(2, 2, 1, 1, 0, 0);
    const SymmetryContext symmetry;
    Chi0 chi0(mf, wfc_basis, aux_basis, pbc, symmetry, grids, kctx, wfc_desc, false, false);
    using ResponseMap = std::map<double, std::map<Vector3_Order<double>,
                                                 atom_mapping<ComplexMatrix>::pair_t_old>>;
    auto &responses = const_cast<ResponseMap &>(chi0.get_chi0_q());
    atpair_k_cplx_mat_t coulomb;
    if (blacs.myid == 0)
    {
        const auto q = pbc.klist.front();
        auto v = std::make_shared<ComplexMatrix>(2, 2);
        (*v)(0, 0) = 1.0; (*v)(0, 1) = 1.0;
        (*v)(1, 0) = 1.0; (*v)(1, 1) = 2.0;
        coulomb[0][0][q] = v;
        auto &chi = responses[grids.get_freq_nodes().front()][q][0][0];
        chi.create(2, 2);
        chi(0, 0) = -1.0;
        chi(0, 1) = chi(1, 0) = zero_diagonal_block ? 2.0 : 0.0;
        chi(1, 1) = zero_diagonal_block ? -5.0 : -1.0;
    }
    return compute_RPA_correlation_blacs_2d_gamma_only(
        chi0, coulomb, {}, blacs, gpu, backend);
}

void test_gamma_lu(const BlacsCtxtHandler &blacs, bool gpu)
{
    // V and -chi0 are SPD, but I-V*chi0=[[0,3],[-3,9]]. Its determinant
    // is 9: conventional row pivoting succeeds while the first 1x1 block
    // of block-pivoted LU is singular on this 2x2 process grid.
    const auto native = gamma_energy(blacs, "scalapack", true);
    const double expected_native = (std::log(9.0) - 7.0) / TWO_PI;
    require_all(blacs.myid != 0 ||
                    (std::isfinite(native.value.real()) &&
                     std::abs(native.value - expected_native) < 1e-12),
                blacs, "native Gamma RPA disagrees with the independent determinant");
    for (int repeat = 0; repeat != 2; ++repeat)
    {
        bool rejected = false;
        try { gamma_energy(blacs, "ddla", true, gpu); }
        catch (const std::runtime_error &error)
        {
            const std::string message(error.what());
            rejected = message.find("RPA Gamma LU factorization failed with info=1") !=
                       std::string::npos;
        }
        require_all(rejected, blacs,
                    "failed Gamma block LU must be rejected on every owning rank");
    }
    // A successful solve after the rejected calls must remain usable. For
    // chi0=-I, det(I+V)=5 and Tr(V*chi0)=-3, with unit quadrature weight.
    const auto valid = gamma_energy(blacs, "ddla", false, gpu);
    require_all(blacs.myid != 0 ||
                    (std::isfinite(valid.value.real()) &&
                     std::abs(valid.value - (std::log(5.0) - 3.0) / TWO_PI) < 1e-12),
                blacs, "successful Gamma DDLA calculation changed after LU rejection");
}
}

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    int failed = 0;
    try
    {
        BlacsCtxtHandler blacs(MPI_COMM_WORLD);
        blacs.init();
        blacs.set_square_grid();
        if (blacs.nprows != 2 || blacs.npcols != 2)
            throw std::runtime_error("Gamma LU regression requires a 2x2 process grid");
        blacs.init_ddla_handle(ddla::DdlaBackend::CPU);
        const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        if (gpu) blacs.init_ddla_handle(ddla::DdlaBackend::GPU);
#else
        if (gpu) throw std::runtime_error("GPU test requires a CUDA/HIP build");
#endif
        test_gamma_lu(blacs, gpu);
        if (blacs.myid == 0) std::cout << "GAMMA_LU_STATUS_TEST_PASS\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        failed = 1;
    }
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    return any_failed;
}
