#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <valarray>
#include <vector>

#include "../core/epsilon.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using namespace librpa_int;

namespace
{
Matz inverse(const Matz &input)
{
    auto matrix = input.copy();
    const int n = matrix.nr();
    std::vector<int> pivots(n);
    std::vector<std::complex<double>> work(n * n);
    int info = 0;
    LapackConnector::getrf_f(n, n, matrix.ptr(), n, pivots.data(), info);
    assert(info == 0);
    LapackConnector::getri_f(n, matrix.ptr(), n, pivots.data(), work.data(), n * n, info);
    assert(info == 0);
    return matrix;
}

Matz collect_matrix(const Matz &local, const ArrayDesc &desc)
{
    Matz result(desc.m(), desc.n(), MAJOR::COL);
    for (int i = 0; i != local.nr(); ++i)
        for (int j = 0; j != local.nc(); ++j)
            result(desc.indx_l2g_r(i), desc.indx_l2g_c(j)) = local(i, j);
    MPI_Allreduce(MPI_IN_PLACE, result.ptr(), result.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                  desc.comm());
    return result;
}

Matz collect_response(const Chi0 &response, double frequency, const Vector3_Order<double> &q,
                      int naux)
{
    Matz result(naux, naux, MAJOR::COL);
    int owners = 0;
    const auto &blocks = response.get_chi0_q();
    if (blocks.count(frequency) && blocks.at(frequency).count(q))
    {
        const auto &pairs = blocks.at(frequency).at(q);
        if (pairs.count(0) && pairs.at(0).count(0))
        {
            owners = 1;
            for (int i = 0; i != naux; ++i)
                for (int j = 0; j != naux; ++j) result(i, j) = pairs.at(0).at(0)(i, j);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &owners, 1, MPI_INT, MPI_SUM, response.comm_h.comm);
    MPI_Allreduce(MPI_IN_PLACE, result.ptr(), result.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                  response.comm_h.comm);
    assert(owners == 1);
    return result;
}

void check_matrix(const Matz &actual, const Matz &expected)
{
    double error = 0.0;
    for (std::size_t i = 0; i != actual.size(); ++i)
    {
        assert(std::isfinite(actual.ptr()[i].real()));
        assert(std::isfinite(actual.ptr()[i].imag()));
        error = std::max(error, std::abs(actual.ptr()[i] - expected.ptr()[i]));
    }
    if (error > 1e-10)
    {
        std::cerr << "strict-2D production Wc error: " << error << '\n';
        for (int i = 0; i != actual.nr(); ++i)
            for (int j = 0; j != actual.nc(); ++j)
                std::cerr << i << ',' << j << " actual=" << actual(i, j)
                          << " expected=" << expected(i, j) << '\n';
    }
    assert(error < 1e-10);
}

void test_complete_wc(const BlacsCtxtHandler &blacs_h, const Matz &rotation, bool use_cholesky)
{
    const int naux = rotation.nr();
    const int nbody = naux - 1;
    PeriodicBoundaryData pbc;
    pbc.set_latvec({4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 12.0});
    pbc.set_kgrids_kvec(2, 1, 1, {0.0, 0.0, 0.0, PI / 4.0, 0.0, 0.0});
    pbc.set_kq_mapping({0, 1});
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({static_cast<std::size_t>(naux)});
    MeanField mf(1, 2, 2, 2);
    for (int k = 0; k != 2; ++k)
    {
        mf.get_eigenvals()[0](k, 0) = -0.5;
        mf.get_eigenvals()[0](k, 1) = 0.5;
        mf.get_weight()[0](k, 0) = 1.0;
        mf.get_weight()[0](k, 1) = 0.0;
        auto &wfc = mf.get_eigenvectors()[0][0][k];
        wfc.create(2, 2);
        wfc.zero_out();
        wfc(0, 0) = wfc(1, 1) = 1.0;
    }
    TFGrids grids(6);
    grids.generate_minimax(1.0, 8.0);
    const auto frequencies = grids.get_freq_nodes();
    velocity_matrix_t velocity;
    initialize_velocity_matrix(velocity, 1, 2, 2);
    for (int k = 0; k != 2; ++k)
        for (int alpha = 0; alpha != 2; ++alpha)
            velocity[0][k][alpha](0, 1) = velocity[0][k][alpha](1, 0) = 0.2 + 0.1 * alpha;

    // Distinct regular eigenvalues and a mixed head eigenvector expose basis errors.
    Matz root(naux, naux, MAJOR::COL);
    for (int i = 0; i != naux; ++i) root(i, i) = 4.0 * std::pow(0.5, i);
    const auto scaled_eigenvectors = rotation * root;
    const auto sqrt_coulomb = scaled_eigenvectors * rotation.get_transpose(true);
    const auto coulomb = sqrt_coulomb * sqrt_coulomb;
    atpair_k_cplx_mat_t coulomb_blocks;
    // LibRI collects unique owner blocks; replicated inputs would be summed.
    for (const auto &q : pbc.klist_coul)
    {
        if (blacs_h.myid != 0) continue;
        auto block = std::make_shared<ComplexMatrix>(naux, naux);
        for (int i = 0; i != naux; ++i)
            for (int j = 0; j != naux; ++j) (*block)(i, j) = coulomb(i, j);
        coulomb_blocks[0][0][q] = block;
    }
    Cs_LRI coefficients;
    coefficients.use_libri = true;
    RI::Tensor<double> tensor({static_cast<std::size_t>(naux), 2UL, 2UL});
    const double transition[4] = {0.2, -0.1, 0.3, 0.15};
    for (int mu = 0; mu != naux; ++mu)
        for (int lambda = 0; lambda != naux; ++lambda)
        {
            const double value = rotation(mu, lambda).real() * transition[lambda];
            tensor(mu, 0, 1) += value;
            tensor(mu, 1, 0) += value;
        }
    if (blacs_h.myid == 0) coefficients.data_libri[0][{0, {0, 0, 0}}] = tensor;
    KPointBlacsParallelContext kctx(KPointBlacsProcessShape(1, blacs_h.nprocs, true),
                                    MPI_COMM_WORLD, 2);
    ArrayDesc desc_wfc(kctx.blacs_h);
    desc_wfc.init(2, 2, 1, 1, 0, 0);
    ArrayDesc desc(blacs_h);
    desc.init(naux, naux, 1, 1, 0, 0);
    SymmetryContext symmetry;
    Chi0 response(mf, basis_wfc, basis_abf, pbc, symmetry, grids, kctx, desc_wfc, false, false);
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    const std::vector<atpair_t> local_pairs =
        blacs_h.myid == 0 ? std::vector<atpair_t>{{0, 0}} : std::vector<atpair_t>{};
    response.build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_AUTO, coefficients, local_pairs, basis_abf,
                   no_shrink, blacs_h);

    diele_func headwing(mf, velocity, pbc.kfrac_list, basis_wfc, basis_abf, frequencies, 2, 2, 1,
                        naux, pbc, global::mpi_comm_global_h, blacs_h);
    headwing.configure_strict_2d_coulomb_head(true, TWO_PI);
    headwing.init(0.0, coulomb_blocks);
    headwing.cal_head();
    headwing.cal_wing(coefficients, 0.0, coulomb_blocks);
    diele_func reference = headwing;
    auto scaled_local = get_local_mat(scaled_eigenvectors, desc);
    reference.wing_mu_to_lambda(scaled_local, desc, naux);
    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(nbody, nbody, 0, 0);
    ArrayDesc desc_wing(blacs_h);
    desc_wing.init(nbody, 3, desc_body.mb(), 1, 0, 0);

    std::vector<double> qx(5000), qy(5000), weights(5000, TWO_PI / 5000), qmax(5000);
    for (int i = 0; i != 5000; ++i)
    {
        qx[i] = std::cos(TWO_PI * i / 5000);
        qy[i] = std::sin(TWO_PI * i / 5000);
        qmax[i] = TWO_PI * std::min(pbc.G.e11 / (4.0 * std::abs(qx[i])),
                                    pbc.G.e22 / (2.0 * std::abs(qy[i])));
    }
    const double gamma_area = TWO_PI * TWO_PI * pbc.G.e11 * pbc.G.e22 / 2.0;
    Matz regular_root(nbody, nbody, MAJOR::COL);
    for (int i = 0; i != nbody; ++i) regular_root(i, i) = root(i + 1, i + 1);
    std::map<double, std::map<Vector3_Order<double>, Matz>> expected;
    for (std::size_t f = 0; f != frequencies.size(); ++f)
    {
        auto wing = collect_matrix(reference.get_rpa_chi0v_wing(f), desc_wing);
        wing *= -1.0;
        assert(wing.absmax() > 1e-6);
        // The velocity inputs fix y/x = 1.5 and z = 0 independently of the
        // auxiliary basis. This detects a wrong Cartesian block distribution.
        for (int i = 0; i != nbody; ++i)
        {
            assert(std::abs(wing(i, 1) - 1.5 * wing(i, 0)) < 1e-12);
            assert(std::abs(wing(i, 2)) < 1e-12);
        }
        auto head = reference.get_rpa_chi0v_head(f);
        head *= -1.0;
        for (int i = 0; i != 3; ++i) head(i, i) += 1.0;
        for (const auto &q : pbc.klist_coul)
        {
            const auto chi = collect_response(response, frequencies[f], q, naux);
            auto epsilon = sqrt_coulomb * chi * sqrt_coulomb;
            epsilon *= -1.0;
            for (int i = 0; i != naux; ++i) epsilon(i, i) += 1.0;
            if (is_gamma_point(q))
            {
                const auto coulomb_epsilon = rotation.get_transpose(true) * epsilon * rotation;
                Matz body(nbody, nbody, MAJOR::COL);
                for (int i = 0; i != nbody; ++i)
                    for (int j = 0; j != nbody; ++j) body(i, j) = coulomb_epsilon(i + 1, j + 1);
                const auto body_inv = inverse(body);
                const auto bw = body_inv * wing;
                const auto wb = wing.get_transpose(true) * body_inv;
                const auto lind = head - wing.get_transpose(true) * bw;
                auto wc = strict_2d_average_wc_coulomb_basis(body_inv, bw, wb, lind, regular_root,
                                                             qx, qy, weights, qmax, gamma_area);
                wc = strict_2d_transform_pw_wc_to_auxiliary_basis(
                    wc, reference.get_strict_2d_pw_to_auxiliary_scale());
                expected[frequencies[f]][q] = rotation * wc * rotation.get_transpose(true);
            }
            else
            {
                auto eps_inv = inverse(epsilon);
                for (int i = 0; i != naux; ++i) eps_inv(i, i) -= 1.0;
                expected[frequencies[f]][q] = sqrt_coulomb * eps_inv * sqrt_coulomb;
            }
        }
    }
    const auto head_values = headwing.get_head_vec();
    const std::vector<std::complex<double>> epsmac(head_values.begin(), head_values.end());
    auto wc_coulomb_blocks = coulomb_blocks;
    const auto actual =
        compute_Wc_freq_q_blacs(response, coulomb_blocks, wc_coulomb_blocks, 0.0, true, 3, epsmac,
                                &headwing, blacs_h, desc, false, ".", use_cholesky);
    assert(actual.size() == frequencies.size());
    for (const auto &frequency : expected)
    {
        assert(actual.at(frequency.first).size() == pbc.klist_coul.size());
        for (const auto &q : frequency.second)
            check_matrix(collect_matrix(actual.at(frequency.first).at(q.first), desc), q.second);
    }
}
}  // namespace

int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    {
        BlacsCtxtHandler blacs_h(MPI_COMM_WORLD);
        blacs_h.init();
        blacs_h.set_square_grid();
        // Three regular channels make a 2x2 MPI grid expose Cartesian block
        // sizes of 1 versus 2, which a two-channel body cannot distinguish.
        Matz identity(4, 4, MAJOR::COL);
        for (int i = 0; i != 4; ++i) identity(i, i) = 1.0;
        Matz rotation(4, 4, MAJOR::COL);
        const double values[3][3] = {{0.36, -0.8, 0.48}, {0.48, 0.6, 0.64}, {-0.8, 0.0, 0.6}};
        for (int i = 0; i != 3; ++i)
            for (int j = 0; j != 3; ++j) rotation(i, j) = values[i][j];
        rotation(3, 3) = 1.0;
        for (bool use_cholesky : {false, true})
        {
            test_complete_wc(blacs_h, identity, use_cholesky);
            test_complete_wc(blacs_h, rotation, use_cholesky);
        }
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
