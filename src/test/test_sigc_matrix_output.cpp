#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "../core/gw.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using namespace librpa_int;

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    {
        const auto &comm = global::mpi_comm_global_h;
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
        const auto directory = "test_sigc_output_" + std::to_string(comm.nprocs);
        if (comm.is_root()) std::filesystem::create_directories(directory);
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
        for (std::size_t f = 0; f != grids.size(); ++f)
        {
            std::ostringstream filename;
            filename << directory << "/SigcRF_ispin_00_s_00_iomega_" << std::setfill('0')
                     << std::setw(3) << f << "_myid_" << std::setw(5) << comm.myid << ".bin";
            std::ifstream input(filename.str(), std::ios::binary);
            double omega;
            std::size_t count;
            int dtype;
            input.read(reinterpret_cast<char *>(&omega), sizeof(omega));
            input.read(reinterpret_cast<char *>(&count), sizeof(count));
            input.read(reinterpret_cast<char *>(&dtype), sizeof(dtype));
            assert(input && omega == grids.get_freq_nodes()[f] && dtype == 1);
            assert(count <= pbc.Rlist.size());
        }
        G0W0 restarted(mf, basis, pbc, symmetry, grids, kctx, kctx, desc, desc, false, false);
        restarted.output_sigc_ks_mat_kf = true;
        // An optional directory exercises restart with a different original MPI size.
        restarted.read_sigc(argc > 1 ? argv[1] : directory);
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
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
}
