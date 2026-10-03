#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <map>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <valarray>

#include "../core/chi0.h"
#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../core/qpoint_view.h"
#include "../gpu/la_connector.h"
#include "../core/meanfield_mpi.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using librpa_int::ArrayDesc;
using librpa_int::AtomicBasis;
using librpa_int::atpair_k_cplx_mat_t;
using librpa_int::BlacsCtxtHandler;
using librpa_int::ComplexMatrix;
using librpa_int::diele_func;
using librpa_int::init_local_mat;
using librpa_int::KPointBlacsParallelContext;
using librpa_int::KPointBlacsProcessShape;
using librpa_int::MAJOR;
using librpa_int::Matrix3;
using librpa_int::Matz;
using librpa_int::matrix_m;
using librpa_int::MeanField;
using librpa_int::PeriodicBoundaryData;
using librpa_int::SpeciesBasisLayout;
using librpa_int::SymmetryContext;
using librpa_int::SymmetryKAtomRotation;
using librpa_int::SymmetryKStarMember;
using librpa_int::SymmetryOperation;
using librpa_int::Vector3_Order;
using librpa_int::atom_t;

namespace
{

void test_rspace_symmetry_requires_complete_band_space()
{
    MeanField truncated(1, 1, 2, 3);
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(truncated, -1));
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(truncated, 3));

    MeanField complete(1, 1, 3, 3);
    assert(librpa_int::rspace_symmetry_has_complete_band_space(complete, -1));
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(complete, 2));
}

void assert_complex_close(const std::complex<double> &actual, const std::complex<double> &expected,
                          const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        assert(false);
    }
}

void require_double_close(const double actual, const double expected, const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        std::abort();
    }
}

void fill_distributed_matrix(
    matrix_m<std::complex<double>> &matrix, const ArrayDesc &desc,
    const std::vector<std::vector<std::complex<double>>> &values)
{
    assert(static_cast<int>(values.size()) == desc.m());
    for (int i = 0; i != desc.m(); ++i)
    {
        assert(static_cast<int>(values[i].size()) == desc.n());
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != desc.n(); ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo >= 0) matrix(ilo, jlo) = values[i][j];
        }
    }
}

void fill_distributed_matrix(
    matrix_m<std::complex<double>> &matrix, const ArrayDesc &desc,
    const matrix_m<std::complex<double>> &values)
{
    assert(values.nr() == desc.m() && values.nc() == desc.n());
    for (int i_local = 0; i_local != desc.m_loc(); ++i_local)
    {
        const int i_global = desc.indx_l2g_r(i_local);
        for (int j_local = 0; j_local != desc.n_loc(); ++j_local)
        {
            const int j_global = desc.indx_l2g_c(j_local);
            matrix(i_local, j_local) = values(i_global, j_global);
        }
    }
}

void verify_distributed_inverse(
    const std::vector<std::vector<std::complex<double>>> &values,
    const BlacsCtxtHandler &blacs_h, const int block_size, const bool use_cholesky,
    const bool expect_empty_local_rank)
{
    const int n = static_cast<int>(values.size());
    ArrayDesc desc(blacs_h);
    desc.init(n, n, block_size, block_size, 0, 0);

    auto original = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    fill_distributed_matrix(original, desc, values);
    auto inverse = original.copy();

    int local_is_empty = inverse.size() == 0 ? 1 : 0;
    int any_empty = 0;
    MPI_Allreduce(&local_is_empty, &any_empty, 1, MPI_INT, MPI_MAX, desc.comm());
    if (expect_empty_local_rank) assert(any_empty == 1);

    librpa_int::invert_headwing_body_with_identity_solve(
        inverse, desc, blacs_h, use_cholesky, false);

    auto product = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    librpa_int::ScalapackConnector::pgemm_f(
        'N', 'N', n, n, n, std::complex<double>{1.0, 0.0}, original.ptr(), 1, 1,
        desc.desc, inverse.ptr(), 1, 1, desc.desc, std::complex<double>{0.0, 0.0},
        product.ptr(), 1, 1, desc.desc);

    double local_max_error = 0.0;
    for (int ilo = 0; ilo != desc.m_loc(); ++ilo)
    {
        const int i = desc.indx_l2g_r(ilo);
        for (int jlo = 0; jlo != desc.n_loc(); ++jlo)
        {
            const int j = desc.indx_l2g_c(jlo);
            const std::complex<double> expected = i == j ? 1.0 : 0.0;
            local_max_error =
                std::max(local_max_error, std::abs(product(ilo, jlo) - expected));
        }
    }
    double max_error = 0.0;
    MPI_Allreduce(&local_max_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, desc.comm());
    require_double_close(max_error, 0.0, 1.0e-10);
}

void test_headwing_body_inverse_uses_identity_solve(const BlacsCtxtHandler &square_blacs_h)
{
    const std::vector<std::vector<std::complex<double>>> pivoting_matrix{
        {{0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}},
        {{1.0, 0.0}, {4.0, 0.0}, {1.0, 0.2}, {0.0, 0.0}},
        {{0.0, 0.0}, {1.0, -0.2}, {5.0, 0.0}, {1.0, 0.0}},
        {{0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {6.0, 0.0}}};
    verify_distributed_inverse(pivoting_matrix, square_blacs_h, 1, false, false);

    BlacsCtxtHandler horizontal_blacs_h(MPI_COMM_WORLD);
    horizontal_blacs_h.init();
    horizontal_blacs_h.set_horizontal_grid();
    const std::vector<std::vector<std::complex<double>>> positive_definite_matrix{
        {{4.0, 0.0}, {1.0, 1.0}},
        {{1.0, -1.0}, {3.0, 0.0}}};
    verify_distributed_inverse(positive_definite_matrix, horizontal_blacs_h, 2, true,
                               horizontal_blacs_h.nprocs > 1);
}

void test_gamma_head_rank_one_matches_coulomb_basis_overwrite(
    const BlacsCtxtHandler &blacs_h)
{
    constexpr int n = 4;
    const double half = 0.5;
    const matrix_m<std::complex<double>> eigenvectors(
        {{{half, 0.0}, {half, 0.0}, {half, 0.0}, {half, 0.0}},
         {{half, 0.0}, {-half, 0.0}, {half, 0.0}, {-half, 0.0}},
         {{half, 0.0}, {half, 0.0}, {-half, 0.0}, {-half, 0.0}},
         {{half, 0.0}, {-half, 0.0}, {-half, 0.0}, {half, 0.0}}},
        MAJOR::COL);
    const std::array<double, n> eigenvalues{{9.0, 4.0, 1.0, 0.0}};
    const matrix_m<std::complex<double>> response(
        {{{0.13, 0.0}, {0.02, 0.03}, {-0.01, 0.02}, {0.04, -0.01}},
         {{0.02, -0.03}, {0.18, 0.0}, {0.03, -0.02}, {-0.02, 0.01}},
         {{-0.01, -0.02}, {0.03, 0.02}, {0.16, 0.0}, {0.01, 0.04}},
         {{0.04, 0.01}, {-0.02, -0.01}, {0.01, -0.04}, {0.11, 0.0}}},
        MAJOR::COL);
    const std::complex<double> corrected_head{2.35, -0.015};

    const auto eigenvectors_h = eigenvectors.get_transpose(true);
    auto scaled_eigenvectors = eigenvectors.copy();
    for (int i = 0; i != n; ++i)
        scaled_eigenvectors.scale_col(i, std::sqrt(eigenvalues.at(i)));
    const auto sqrt_coulomb = scaled_eigenvectors * eigenvectors_h;

    auto epsilon_direct = sqrt_coulomb * response * sqrt_coulomb;
    epsilon_direct *= -1.0;
    for (int i = 0; i != n; ++i) epsilon_direct(i, i) += 1.0;

    const auto epsilon_eigenbasis = eigenvectors_h * epsilon_direct * eigenvectors;
    const auto uncorrected_head_reference = epsilon_eigenbasis(0, 0);
    auto response_eigenbasis = eigenvectors_h * response * eigenvectors;
    for (int i = 0; i != n; ++i)
    {
        for (int j = 0; j != n; ++j)
        {
            response_eigenbasis(i, j) *=
                -std::sqrt(eigenvalues.at(i) * eigenvalues.at(j));
        }
    }
    response_eigenbasis(0, 0) = corrected_head - 1.0;
    auto epsilon_reference = eigenvectors * response_eigenbasis * eigenvectors_h;
    for (int i = 0; i != n; ++i) epsilon_reference(i, i) += 1.0;

    ArrayDesc desc(blacs_h);
    desc.init(n, n, 1, 2, 0, 0);
    auto eigen_local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    auto epsilon_local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    auto scratch_local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    fill_distributed_matrix(eigen_local, desc, eigenvectors);
    fill_distributed_matrix(epsilon_local, desc, epsilon_direct);

    ArrayDesc desc_1x1(blacs_h);
    desc_1x1.init(1, 1, 2, 2, 0, 0);
    auto h_local = init_local_mat<std::complex<double>>(desc_1x1, MAJOR::COL);
    if (desc_1x1.m_loc() == 0 || desc_1x1.n_loc() == 0) h_local.resize(1, 1);
    if (desc_1x1.m_loc() != 0 && desc_1x1.n_loc() != 0) h_local(0, 0) = {0.0, 0.0};

    // y = epsilon0 * x1
    librpa_int::LaConnector::pgemm(
        'N', 'N', n, 1, n, std::complex<double>{1.0, 0.0},
        epsilon_local.ptr(), 1, 1, desc,
        eigen_local.ptr(), 1, 1, desc,
        std::complex<double>{0.0, 0.0},
        scratch_local.ptr(), 1, 1, desc);

    // h = x1^H * y
    librpa_int::LaConnector::pgemm(
        'C', 'N', 1, 1, n, std::complex<double>{1.0, 0.0},
        eigen_local.ptr(), 1, 1, desc,
        scratch_local.ptr(), 1, 1, desc,
        std::complex<double>{0.0, 0.0},
        h_local.ptr(), 1, 1, desc_1x1);

    std::complex<double> h_scalar{0.0, 0.0};
    if (desc_1x1.is_src()) h_scalar = h_local(0, 0);
    const int h_root = blacs_h.get_pnum(0, 0);
    MPI_Bcast(&h_scalar, 1, MPI_CXX_DOUBLE_COMPLEX, h_root, desc_1x1.comm());

    assert_complex_close(h_scalar, uncorrected_head_reference, 1.0e-12);

    // epsilon += (H - h) * x1 * x1^H
    const std::complex<double> coeff = corrected_head - h_scalar;
    librpa_int::LaConnector::pgemm(
        'N', 'C', n, n, 1, coeff,
        eigen_local.ptr(), 1, 1, desc,
        eigen_local.ptr(), 1, 1, desc,
        std::complex<double>{1.0, 0.0},
        epsilon_local.ptr(), 1, 1, desc);

    double local_max_error = 0.0;
    for (int i_local = 0; i_local != desc.m_loc(); ++i_local)
    {
        const int i_global = desc.indx_l2g_r(i_local);
        for (int j_local = 0; j_local != desc.n_loc(); ++j_local)
        {
            const int j_global = desc.indx_l2g_c(j_local);
            local_max_error = std::max(
                local_max_error,
                std::abs(epsilon_local(i_local, j_local)
                         - epsilon_reference(i_global, j_global)));
        }
    }
    double max_error = 0.0;
    MPI_Allreduce(&local_max_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, desc.comm());
    require_double_close(max_error, 0.0, 1.0e-11);

    // Verify that modifying eigenvector columns other than column zero
    // cannot affect h or the rank-one update.
    auto eigen_modified = eigen_local.copy();
    for (int col = 1; col != n; ++col)
    {
        const int col_local = desc.indx_g2l_c(col);
        if (col_local < 0) continue;
        for (int i_local = 0; i_local != desc.m_loc(); ++i_local)
            eigen_modified(i_local, col_local) *= std::complex<double>{0.0, 2.0};
    }
    auto epsilon_modified = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    fill_distributed_matrix(epsilon_modified, desc, epsilon_direct);
    auto scratch2 = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    auto h2_local = init_local_mat<std::complex<double>>(desc_1x1, MAJOR::COL);
    if (desc_1x1.m_loc() == 0 || desc_1x1.n_loc() == 0) h2_local.resize(1, 1);
    if (desc_1x1.m_loc() != 0 && desc_1x1.n_loc() != 0) h2_local(0, 0) = {0.0, 0.0};

    librpa_int::LaConnector::pgemm(
        'N', 'N', n, 1, n, std::complex<double>{1.0, 0.0},
        epsilon_modified.ptr(), 1, 1, desc,
        eigen_modified.ptr(), 1, 1, desc,
        std::complex<double>{0.0, 0.0},
        scratch2.ptr(), 1, 1, desc);
    librpa_int::LaConnector::pgemm(
        'C', 'N', 1, 1, n, std::complex<double>{1.0, 0.0},
        eigen_modified.ptr(), 1, 1, desc,
        scratch2.ptr(), 1, 1, desc,
        std::complex<double>{0.0, 0.0},
        h2_local.ptr(), 1, 1, desc_1x1);

    std::complex<double> h2_scalar{0.0, 0.0};
    if (desc_1x1.is_src()) h2_scalar = h2_local(0, 0);
    MPI_Bcast(&h2_scalar, 1, MPI_CXX_DOUBLE_COMPLEX, h_root, desc_1x1.comm());
    assert_complex_close(h2_scalar, h_scalar, 1.0e-12);

    librpa_int::LaConnector::pgemm(
        'N', 'C', n, n, 1, coeff,
        eigen_modified.ptr(), 1, 1, desc,
        eigen_modified.ptr(), 1, 1, desc,
        std::complex<double>{1.0, 0.0},
        epsilon_modified.ptr(), 1, 1, desc);

    double local_mod_error = 0.0;
    for (int i_local = 0; i_local != desc.m_loc(); ++i_local)
    {
        for (int j_local = 0; j_local != desc.n_loc(); ++j_local)
        {
            local_mod_error = std::max(
                local_mod_error,
                std::abs(epsilon_modified(i_local, j_local)
                         - epsilon_local(i_local, j_local)));
        }
    }
    double mod_error = 0.0;
    MPI_Allreduce(&local_mod_error, &mod_error, 1, MPI_DOUBLE, MPI_MAX, desc.comm());
    require_double_close(mod_error, 0.0, 1.0e-11);
}

void test_gamma_head_rank_one_handles_empty_local_blocks(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init(1, 1, 1, 1, 0, 0);
    auto eigen_local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    auto epsilon_local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    auto scratch_local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    if (desc.m_loc() == 0 || desc.n_loc() == 0)
    {
        eigen_local.resize(1, 1);
        epsilon_local.resize(1, 1);
        scratch_local.resize(1, 1);
    }
    if (desc.m_loc() != 0 && desc.n_loc() != 0)
    {
        eigen_local(0, 0) = 1.0;
        epsilon_local(0, 0) = 0.8;
    }

    ArrayDesc desc_1x1(blacs_h);
    desc_1x1.init(1, 1, 1, 1, 0, 0);
    auto h_local = init_local_mat<std::complex<double>>(desc_1x1, MAJOR::COL);
    if (desc_1x1.m_loc() == 0 || desc_1x1.n_loc() == 0) h_local.resize(1, 1);
    if (desc_1x1.m_loc() != 0 && desc_1x1.n_loc() != 0) h_local(0, 0) = {0.0, 0.0};

    // y = epsilon0 * x1
    librpa_int::LaConnector::pgemm(
        'N', 'N', 1, 1, 1, std::complex<double>{1.0, 0.0},
        epsilon_local.ptr(), 1, 1, desc,
        eigen_local.ptr(), 1, 1, desc,
        std::complex<double>{0.0, 0.0},
        scratch_local.ptr(), 1, 1, desc);

    // h = x1^H * y
    librpa_int::LaConnector::pgemm(
        'C', 'N', 1, 1, 1, std::complex<double>{1.0, 0.0},
        eigen_local.ptr(), 1, 1, desc,
        scratch_local.ptr(), 1, 1, desc,
        std::complex<double>{0.0, 0.0},
        h_local.ptr(), 1, 1, desc_1x1);

    std::complex<double> h_scalar{0.0, 0.0};
    if (desc_1x1.is_src()) h_scalar = h_local(0, 0);
    const int h_root = blacs_h.get_pnum(0, 0);
    MPI_Bcast(&h_scalar, 1, MPI_CXX_DOUBLE_COMPLEX, h_root, desc_1x1.comm());

    assert_complex_close(h_scalar, std::complex<double>{0.8, 0.0}, 1.0e-12);

    // epsilon += (2.1 - h) * x1 * x1^H
    const std::complex<double> coeff = std::complex<double>{2.1, 0.0} - h_scalar;
    librpa_int::LaConnector::pgemm(
        'N', 'C', 1, 1, 1, coeff,
        eigen_local.ptr(), 1, 1, desc,
        eigen_local.ptr(), 1, 1, desc,
        std::complex<double>{1.0, 0.0},
        epsilon_local.ptr(), 1, 1, desc);

    int local_is_empty = (desc.m_loc() == 0 || desc.n_loc() == 0) ? 1 : 0;
    int any_empty = 0;
    MPI_Allreduce(&local_is_empty, &any_empty, 1, MPI_INT, MPI_MAX, desc.comm());
    if (desc.nprocs() > 1) assert(any_empty == 1);
    if (desc.m_loc() != 0 && desc.n_loc() != 0)
        assert_complex_close(epsilon_local(0, 0), std::complex<double>{2.1, 0.0}, 1.0e-12);
}

RI::Tensor<double> make_scalar_cs_tensor(const double value)
{
    auto data = std::make_shared<std::valarray<double>>(1);
    (*data)[0] = value;
    return RI::Tensor<double>({1UL, 1UL, 1UL}, data);
}

void test_kpoint_coordinate_mapping_selects_active_klist_from_full_source()
{
    const std::vector<Vector3_Order<double>> pyatb_full_kpoints{
        {0.0, 0.0, 0.0}, {0.125, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.375, 0.0, 0.0},
        {0.0, 0.125, 0.0}, {0.125, 0.125, 0.0}, {0.875, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> active_kpoints{
        {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.125, 0.125, 0.0}};

    const auto mapping = librpa_int::map_kpoints_by_coordinates(active_kpoints, pyatb_full_kpoints);

    assert((mapping == std::vector<int>{0, 2, 5}));

    const std::vector<Vector3_Order<double>> wrapped_active_kpoints{{-0.125, 0.0, 0.0}};
    const auto wrapped_mapping =
        librpa_int::map_kpoints_by_coordinates(wrapped_active_kpoints, pyatb_full_kpoints);
    assert((wrapped_mapping == std::vector<int>{6}));
}

void test_strict_2d_qmember_diagnostic_selects_one_periodic_member()
{
    const Vector3_Order<double> selected{0.0, 1.0 / 12.0, 0.0};
    const std::vector<Vector3_Order<double>> first_star{
        {0.0, 1.0 / 12.0, 0.0},
        {0.0, -1.0 / 12.0, 0.0},
        {1.0 / 12.0, 0.0, 0.0},
        {-1.0 / 12.0, 0.0, 0.0},
        {1.0 / 12.0, -1.0 / 12.0, 0.0},
        {-1.0 / 12.0, 1.0 / 12.0, 0.0},
    };

    for (const auto& q : first_star)
    {
        assert(librpa_int::strict_2d_qmember_diagnostic_keeps(q, selected, false));
    }
    assert(librpa_int::strict_2d_qmember_diagnostic_keeps(first_star.front(), selected, true));
    assert(librpa_int::strict_2d_qmember_diagnostic_keeps(
        Vector3_Order<double>{0.0, -11.0 / 12.0, 0.0}, selected, true));
    for (std::size_t i = 1; i != first_star.size(); ++i)
    {
        assert(!librpa_int::strict_2d_qmember_diagnostic_keeps(first_star[i], selected, true));
    }
}

void test_kstar_velocity_mapping_preserves_member_order_and_periodic_gauge()
{
    SymmetryContext ctx;
    librpa_int::SymmetryKStar gamma;
    gamma.k_ibz = {0.0, 0.0, 0.0};
    librpa_int::SymmetryKStarMember gamma_first;
    gamma_first.k_bz = {0.0, 0.0, 0.0};
    librpa_int::SymmetryKStarMember gamma_second;
    gamma_second.k_bz = {0.5, 0.0, 0.0};
    gamma.members = {gamma_first, gamma_second};
    librpa_int::SymmetryKStar quarter;
    quarter.k_ibz = {0.25, 0.0, 0.0};
    librpa_int::SymmetryKStarMember quarter_first;
    quarter_first.k_bz = {0.25, 0.0, 0.0};
    librpa_int::SymmetryKStarMember quarter_second;
    quarter_second.k_bz = {-0.25, 0.0, 0.0};
    quarter.members = {quarter_first, quarter_second};
    ctx.kstars = {gamma, quarter};

    const std::vector<Vector3_Order<double>> ibz_kpoints{{0.25, 0.0, 0.0}, {0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> full_bz_kpoints{
        {0.5, 0.0, 0.0}, {0.75, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}};

    const auto mapping = librpa_int::map_symmetry_kstar_members_to_source_kpoints(
        ctx, ibz_kpoints, full_bz_kpoints);

    assert((mapping == std::vector<std::vector<int>>{{3, 1}, {2, 0}}));
}

void test_replace_rpa_response_headwing_replaces_only_singular_channels(
    const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(4, 4, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 4; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) =
                std::complex<double>(0.01 * (i + 1) + 0.02 * (j + 1), 0.001 * (i - j));
        }
    }

    const matrix_m<std::complex<double>> head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{2.0, 0.0}, std::complex<double>{0.2, 0.1},
             std::complex<double>{0.3, -0.1}},
            {std::complex<double>{0.2, -0.1}, std::complex<double>{2.2, 0.0},
             std::complex<double>{0.4, 0.2}},
            {std::complex<double>{0.3, 0.1}, std::complex<double>{0.4, -0.2},
             std::complex<double>{2.4, 0.0}}},
        MAJOR::COL);
    const matrix_m<std::complex<double>> wing(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.11, 0.01}, std::complex<double>{0.12, 0.02},
             std::complex<double>{0.13, 0.03}},
            {std::complex<double>{0.21, 0.04}, std::complex<double>{0.22, 0.05},
             std::complex<double>{0.23, 0.06}},
            {std::complex<double>{0.31, 0.07}, std::complex<double>{0.32, 0.08},
             std::complex<double>{0.33, 0.09}}},
        MAJOR::COL);

    librpa_int::replace_rpa_response_headwing(response, head, wing, desc);

    const int head_row = desc.indx_g2l_r(0);
    const int head_col = desc.indx_g2l_c(0);
    if (head_row >= 0 && head_col >= 0)
    {
        assert_complex_close(response(head_row, head_col), std::complex<double>{2.2, 0.0}, 1e-12);
    }

    for (int lambda = 1; lambda != 4; ++lambda)
    {
        std::complex<double> expected_wing = 0.0;
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            expected_wing += wing(lambda - 1, alpha);
        }
        expected_wing /= 3.0;

        const int row_body = desc.indx_g2l_r(lambda);
        const int col_head = desc.indx_g2l_c(0);
        if (row_body >= 0 && col_head >= 0)
        {
            assert_complex_close(response(row_body, col_head), expected_wing, 1e-12);
        }

        const int row_head = desc.indx_g2l_r(0);
        const int col_body = desc.indx_g2l_c(lambda);
        if (row_head >= 0 && col_body >= 0)
        {
            assert_complex_close(response(row_head, col_body), std::conj(expected_wing), 1e-12);
        }
    }

    const int body_row = desc.indx_g2l_r(2);
    const int body_col = desc.indx_g2l_c(3);
    if (body_row >= 0 && body_col >= 0)
    {
        assert_complex_close(response(body_row, body_col), std::complex<double>{0.11, -0.001},
                             1e-12);
    }
}

void test_replace_rpa_response_head_only_keeps_numeric_wings(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(4, 4, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    matrix_m<std::complex<double>> original(4, 4, MAJOR::COL);
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        for (int j = 0; j != 4; ++j)
        {
            const auto value =
                std::complex<double>(0.1 * (i + 1) + 0.01 * (j + 1), 0.001 * (i - j));
            original(i, j) = value;
            if (ilo < 0) continue;
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) = value;
        }
    }

    const matrix_m<std::complex<double>> chi0v_head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.21, 0.0}, std::complex<double>{0.01, 0.02},
             std::complex<double>{-0.03, 0.04}},
            {std::complex<double>{0.05, -0.01}, std::complex<double>{0.24, 0.0},
             std::complex<double>{0.07, 0.03}},
            {std::complex<double>{-0.02, -0.04}, std::complex<double>{0.08, -0.03},
             std::complex<double>{0.27, 0.0}}},
        MAJOR::COL);

    librpa_int::replace_rpa_response_head_only(response, chi0v_head, desc);

    const auto expected_head = (chi0v_head(0, 0) + chi0v_head(1, 1) + chi0v_head(2, 2)) / 3.0;
    for (int i = 0; i != 4; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 4; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            const auto expected = (i == 0 && j == 0) ? expected_head : original(i, j);
            assert_complex_close(response(ilo, jlo), expected, 1e-12);
        }
    }
}

void test_rpa_trace_log_average_uses_directional_head_and_wing()
{
    const matrix_m<std::complex<double>> head(
        std::vector<std::vector<std::complex<double>>>{
            {std::complex<double>{0.2, 0.0}, std::complex<double>{0.0, 0.0},
             std::complex<double>{0.0, 0.0}},
            {std::complex<double>{0.0, 0.0}, std::complex<double>{0.3, 0.0},
             std::complex<double>{0.0, 0.0}},
            {std::complex<double>{0.0, 0.0}, std::complex<double>{0.0, 0.0},
             std::complex<double>{0.4, 0.0}}},
        MAJOR::COL);
    const std::complex<double> body{0.1, 0.0};
    const std::array<std::complex<double>, 3> wing{std::complex<double>{0.05, 0.0},
                                                   std::complex<double>{0.02, 0.0},
                                                   std::complex<double>{0.01, 0.0}};
    const std::complex<double> body_inv = 1.0 / (1.0 - body);
    const matrix_m<std::complex<double>> schur_l(
        std::vector<std::vector<std::complex<double>>>{
            {1.0 - head(0, 0) - std::conj(wing[0]) * body_inv * wing[0],
             -std::conj(wing[0]) * body_inv * wing[1], -std::conj(wing[0]) * body_inv * wing[2]},
            {-std::conj(wing[1]) * body_inv * wing[0],
             1.0 - head(1, 1) - std::conj(wing[1]) * body_inv * wing[1],
             -std::conj(wing[1]) * body_inv * wing[2]},
            {-std::conj(wing[2]) * body_inv * wing[0], -std::conj(wing[2]) * body_inv * wing[1],
             1.0 - head(2, 2) - std::conj(wing[2]) * body_inv * wing[2]}},
        MAJOR::COL);
    const std::vector<double> qx{1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0};
    const std::vector<double> qz{0.0, 0.0};
    const std::vector<double> weights{0.2, 0.6};
    const std::complex<double> trace_body = body;
    const std::complex<double> logdet_body = std::log(1.0 - body);

    const auto actual = librpa_int::compute_rpa_chi0v_headwing_trace_log_average(
        head, schur_l, trace_body, logdet_body, qx, qy, qz, weights);
    const auto direct_trace_log = [&](const double nx, const double ny, const double nz)
    {
        const auto directional_head = nx * (nx * head(0, 0) + ny * head(0, 1) + nz * head(0, 2)) +
                                      ny * (nx * head(1, 0) + ny * head(1, 1) + nz * head(1, 2)) +
                                      nz * (nx * head(2, 0) + ny * head(2, 1) + nz * head(2, 2));
        const auto directional_wing = nx * wing[0] + ny * wing[1] + nz * wing[2];
        const auto direct_det = (1.0 - directional_head) * (1.0 - body) -
                                std::conj(directional_wing) * directional_wing;
        return directional_head + body + std::log(direct_det);
    };
    const auto expected = weights[0] * direct_trace_log(qx[0], qy[0], qz[0]) +
                          weights[1] * direct_trace_log(qx[1], qy[1], qz[1]);

    assert_complex_close(actual, expected, 1e-12);
}

void test_rpa_headwing_regular_body_start_channel()
{
    librpa_int::RpaHeadwingSettings settings;

    settings.use_2d_dielectric = false;
    settings.rpa_headwing_body_start = 0;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = true;
    settings.rpa_headwing_body_start = 0;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = false;
    settings.rpa_headwing_body_start = 1;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 1);

    settings.use_2d_dielectric = true;
    settings.rpa_headwing_body_start = 4;
    assert(librpa_int::rpa_headwing_regular_body_start_channel(settings) == 4);
}

void test_rpa_headwing_gamma_cell_volume_uses_reciprocal_lattice()
{
    librpa_int::PeriodicBoundaryData pbc;
    pbc.latvec = librpa_int::Matrix3(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0);
    pbc.G = librpa_int::Matrix3(0.5, 0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 0.2);

    const double vol_3d = librpa_int::rpa_headwing_reciprocal_cell_volume(pbc, false);
    require_double_close(vol_3d, std::abs(pbc.G.Det()), 1e-14);

    const double vol_2d = librpa_int::rpa_headwing_reciprocal_cell_volume(pbc, true);
    const double expected_2d = std::abs(pbc.G.e11 * pbc.G.e22 - pbc.G.e12 * pbc.G.e21);
    require_double_close(vol_2d, expected_2d, 1e-14);

    pbc.set_period(4, 4, 4);
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, false),
                         vol_3d / 64.0, 1e-14);
    require_double_close(librpa_int::rpa_headwing_gamma_cell_volume(pbc, true),
                         vol_2d / 64.0, 1e-14);
}

void test_strict_2d_headwing_prefactors_use_inplane_area()
{
    constexpr double area = 15.0;
    require_double_close(librpa_int::strict_2d_head_prefactor(area), librpa_int::TWO_PI / area,
                         1e-14);
    require_double_close(librpa_int::strict_2d_wing_prefactor(area),
                         2.0 * std::sqrt(librpa_int::TWO_PI / area), 1e-14);
}

void test_strict_2d_auxiliary_normalization_is_computed_from_basis_metadata()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({19.390653825130212, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 30.0});

    constexpr double multipole_norm_squared = 2205.0673846924301;
    const auto normalization = librpa_int::strict_2d_coulomb_head_normalization(
        pbc, multipole_norm_squared);

    require_double_close(normalization.inplane_area_bohr2, 19.390653825130212, 1e-13);
    require_double_close(normalization.auxiliary_head_coefficient,
                         8978.8175111265446, 1e-10);
    require_double_close(normalization.pw_to_auxiliary_scale,
                         37.802423070695596, 1e-12);

    bool rejected = false;
    try
    {
        (void)librpa_int::strict_2d_coulomb_head_normalization(pbc, 0.0);
    }
    catch (const std::logic_error &)
    {
        rejected = true;
    }
    assert(rejected);
}

void test_strict_2d_gamma_cell_uses_physical_reciprocal_measure()
{
    constexpr double internal_q = 0.2;
    constexpr double internal_area = 0.07;
    require_double_close(librpa_int::strict_2d_physical_q(internal_q),
                         librpa_int::TWO_PI * internal_q, 1e-14);
    require_double_close(librpa_int::strict_2d_physical_gamma_cell_area(internal_area),
                         librpa_int::TWO_PI * librpa_int::TWO_PI * internal_area, 1e-14);
}

void test_strict_2d_radial_integrals_match_analytic_values()
{
    const std::complex<double> a{2.0, 0.0};
    constexpr double qmax = 0.5;
    const auto expected_i0 = (1.0 - std::log(2.0)) / 4.0;
    const auto expected_i1 = (0.5 - 1.0 + std::log(2.0)) / 8.0;

    assert_complex_close(librpa_int::strict_2d_radial_i0(a, qmax), expected_i0, 1e-14);
    assert_complex_close(librpa_int::strict_2d_radial_i1(a, qmax), expected_i1, 1e-14);
}

void test_strict_2d_radial_integrals_are_stable_at_zero_and_small_a()
{
    constexpr double qmax = 0.3;
    assert_complex_close(librpa_int::strict_2d_radial_i0(0.0, qmax), qmax * qmax / 2.0, 1e-15);
    assert_complex_close(librpa_int::strict_2d_radial_i1(0.0, qmax), qmax * qmax * qmax / 3.0,
                         1e-15);

    const std::complex<double> small_a{1.0e-10, -2.0e-10};
    const auto expected_i0 = qmax * qmax / 2.0 - small_a * std::pow(qmax, 3) / 3.0;
    const auto expected_i1 = std::pow(qmax, 3) / 3.0 - small_a * std::pow(qmax, 4) / 4.0;
    assert_complex_close(librpa_int::strict_2d_radial_i0(small_a, qmax), expected_i0, 1e-15);
    assert_complex_close(librpa_int::strict_2d_radial_i1(small_a, qmax), expected_i1, 1e-15);
}

void test_strict_2d_inverse_head_average_has_linear_q_screening()
{
    const std::complex<double> a{1.7, 0.0};
    constexpr double qmax = 0.2;
    const auto inverse_head_average =
        2.0 * librpa_int::strict_2d_radial_i0(a, qmax) / (qmax * qmax);
    const auto old_2d_formula = 1.0 / a;

    assert(std::abs(inverse_head_average - 1.0) < 0.2);
    assert(std::abs(inverse_head_average - old_2d_formula) > 0.1);
}

void test_strict_2d_finite_q_reference_matches_head_and_schur_limits()
{
    matrix_m<std::complex<double>> head(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    head(0, 0) = 1.8;
    head(0, 1) = 0.12;
    head(1, 0) = 0.12;
    head(1, 1) = 1.4;
    head(2, 2) = 1.0;
    lind(0, 0) = 1.65;
    lind(0, 1) = 0.08;
    lind(1, 0) = 0.08;
    lind(1, 1) = 1.30;
    lind(2, 2) = 1.0;

    const auto reference = librpa_int::strict_2d_finite_q_reference(head, lind, 3.0, 4.0);
    const double qx = 3.0 / 5.0;
    const double qy = 4.0 / 5.0;
    const auto expected_eps_coefficient =
        qx * (qx * head(0, 0) + qy * head(0, 1)) + qy * (qx * head(1, 0) + qy * head(1, 1)) - 1.0;
    const auto expected_a =
        qx * (qx * lind(0, 0) + qy * lind(0, 1)) + qy * (qx * lind(1, 0) + qy * lind(1, 1)) - 1.0;

    assert_complex_close(reference.epsilon_minus_identity_over_q, expected_eps_coefficient, 1e-14);
    assert_complex_close(reference.schur_a, expected_a, 1e-14);
    assert_complex_close(reference.wc_head_limit, -librpa_int::TWO_PI * expected_a, 1e-14);
}

void test_strict_2d_schur_coefficient_removes_identity()
{
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    lind(0, 0) = 1.4;
    lind(1, 1) = 1.9;
    lind(2, 2) = 1.0;

    constexpr double qx = 0.6;
    constexpr double qy = 0.8;
    const auto expected = 0.4 * qx * qx + 0.9 * qy * qy;
    assert_complex_close(librpa_int::strict_2d_schur_coefficient(lind, qx, qy), expected, 1e-14);
}

void test_strict_2d_screening_denominator_must_stay_on_physical_branch()
{
    librpa_int::validate_strict_2d_screening_denominator({0.7, 1.0e-12}, 0.4);

    bool rejected_zero = false;
    try
    {
        librpa_int::validate_strict_2d_screening_denominator(-2.5, 0.4);
    }
    catch (const std::logic_error &)
    {
        rejected_zero = true;
    }
    assert(rejected_zero);

    bool rejected_negative = false;
    try
    {
        librpa_int::validate_strict_2d_screening_denominator(-3.0, 0.4);
    }
    catch (const std::logic_error &)
    {
        rejected_negative = true;
    }
    assert(rejected_negative);
}

void test_strict_2d_gw_uses_full_coulomb_at_all_q()
{
    librpa_int::validate_strict_2d_gw_coulomb_choices(false, false, true);
    librpa_int::validate_strict_2d_gw_coulomb_choices(true, true, true);

    bool rejected_cut_coulomb_finite_q = false;
    try
    {
        librpa_int::validate_strict_2d_gw_coulomb_choices(true, true, false);
    }
    catch (const std::logic_error &)
    {
        rejected_cut_coulomb_finite_q = true;
    }
    if (!rejected_cut_coulomb_finite_q)
    {
        std::cerr << "strict 2D GW accepted non-Ewald finite-q Wc legs" << std::endl;
        std::abort();
    }

    bool rejected_cut_coulomb_basis = false;
    try
    {
        librpa_int::validate_strict_2d_gw_coulomb_choices(true, false, true);
    }
    catch (const std::logic_error &)
    {
        rejected_cut_coulomb_basis = true;
    }
    if (!rejected_cut_coulomb_basis)
    {
        std::cerr << "strict 2D GW accepted a non-Ewald dielectric basis" << std::endl;
        std::abort();
    }
}

void test_strict_2d_gw_routes_gamma_through_complete_wc_average()
{
    const auto require_route = [](const bool condition, const char *message) {
        if (!condition)
        {
            std::cerr << message << std::endl;
            std::abort();
        }
    };
    require_route(librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, true, true),
                  "strict 2D Gamma must use the complete-Wc route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(false, 3, true, true, true),
                  "disabled head/wing replacement must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 2, true, true, true),
                  "non-full head/wing dielectric mode must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, false, true, true),
                  "3D dielectric calculations must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, false, true),
                  "finite q must keep the standard route");
    require_route(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, true, false),
                  "missing head/wing data must keep the standard route");
}

void test_strict_2d_gw_fails_closed_for_incomplete_runtime_configuration()
{
    const auto require_condition = [](const bool condition, const char *message)
    {
        if (!condition)
        {
            std::cerr << message << std::endl;
            std::abort();
        }
    };
    require_condition(librpa_int::strict_2d_complete_wc_requested(true, 3, true),
                      "strict 2D complete-Wc request was not recognized");
    require_condition(!librpa_int::strict_2d_complete_wc_requested(false, 3, true),
                      "disabled replacement was classified as strict 2D");
    require_condition(!librpa_int::strict_2d_complete_wc_requested(true, 2, true),
                      "non-head/wing dielectric mode was classified as strict 2D");
    require_condition(!librpa_int::strict_2d_complete_wc_requested(true, 3, false),
                      "3D dielectric mode was classified as strict 2D");

    librpa_int::validate_strict_2d_complete_wc_runtime(false, false, false, false);
    librpa_int::validate_strict_2d_complete_wc_runtime(true, true, true, true);

    bool rejected_missing_metadata = false;
    try
    {
        librpa_int::validate_strict_2d_complete_wc_runtime(true, true, true, false);
    }
    catch (const std::logic_error &)
    {
        rejected_missing_metadata = true;
    }
    require_condition(rejected_missing_metadata,
                      "strict 2D GW silently accepted missing producer metadata");

    bool rejected_missing_data = false;
    try
    {
        librpa_int::validate_strict_2d_complete_wc_runtime(true, false, true, true);
    }
    catch (const std::logic_error &)
    {
        rejected_missing_data = true;
    }
    require_condition(rejected_missing_data,
                      "strict 2D GW silently accepted missing analytic head/wing data");

    bool rejected_dense_wc = false;
    try
    {
        librpa_int::validate_strict_2d_complete_wc_runtime(true, true, false, true);
    }
    catch (const std::logic_error &)
    {
        rejected_dense_wc = true;
    }
    require_condition(rejected_dense_wc,
                      "strict 2D GW silently accepted a path without complete-Wc support");
}

void test_strict_2d_diagnostic_schema_and_qpoint_order_are_stable()
{
    const auto count_columns = [](const std::string &header)
    { return 1 + static_cast<int>(std::count(header.begin(), header.end(), ',')); };
    if (count_columns(librpa_int::strict_2d_finite_q_diagnostics_header()) != 37 ||
        count_columns(librpa_int::strict_2d_gamma_wc_diagnostics_header()) != 19)
    {
        std::cerr << "strict 2D diagnostic CSV schema changed unexpectedly" << std::endl;
        std::abort();
    }

    const std::vector<Vector3_Order<double>> qpoints{
        {0.1, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.1, 0.0}};
    const auto unchanged = librpa_int::strict_2d_diagnostic_qpoint_order(qpoints, false);
    if (!(unchanged == qpoints))
    {
        std::cerr << "disabled strict 2D diagnostics changed q-point order" << std::endl;
        std::abort();
    }
    const auto ordered = librpa_int::strict_2d_diagnostic_qpoint_order(qpoints, true);
    if (!librpa_int::is_gamma_point(ordered.front()) || ordered.size() != qpoints.size())
    {
        std::cerr << "strict 2D diagnostics did not place Gamma first" << std::endl;
        std::abort();
    }
}

void test_strict_2d_block_metrics_separate_head_wings_and_body()
{
    librpa_int::Strict2dBlockMetricSums sums;
    librpa_int::accumulate_strict_2d_block_metric(sums, 0, 0, {2.0, -1.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 0, 1, {3.0, 4.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 2, 0, {0.0, 6.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 1, 1, {5.0, 12.0});
    librpa_int::accumulate_strict_2d_block_metric(sums, 2, 2, {8.0, 15.0});

    const auto metrics = librpa_int::finalize_strict_2d_block_metrics(sums);
    require_double_close(metrics.head.real(), 2.0, 1e-15);
    require_double_close(metrics.head.imag(), -1.0, 1e-15);
    require_double_close(metrics.head_body_frobenius, 5.0, 1e-15);
    require_double_close(metrics.body_head_frobenius, 6.0, 1e-15);
    require_double_close(metrics.body_body_frobenius, std::sqrt(13.0 * 13.0 + 17.0 * 17.0), 1e-15);
}

void test_strict_2d_alpha_reference_averages_bare_coulomb()
{
    constexpr double alpha = 0.25;
    constexpr double radius = 0.4;
    const double gamma_area = librpa_int::PI * radius * radius;
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax(4, radius);
    matrix_m<std::complex<double>> regular_body_sqrt(1, 1, MAJOR::COL);
    regular_body_sqrt(0, 0) = 2.0;

    const auto alpha_wc = librpa_int::strict_2d_alpha_wc_average_coulomb_basis(
        alpha, regular_body_sqrt, weights, qmax, gamma_area);
    require_double_close(alpha_wc(0, 0).real(), (alpha - 1.0) * 4.0 * librpa_int::PI / radius,
                         1e-13);
    require_double_close(alpha_wc(0, 0).imag(), 0.0, 1e-15);
    require_double_close(std::abs(alpha_wc(0, 1)), 0.0, 1e-15);
    require_double_close(std::abs(alpha_wc(1, 0)), 0.0, 1e-15);
    require_double_close(alpha_wc(1, 1).real(), (alpha - 1.0) * 4.0, 1e-13);
}

void test_strict_2d_pw_wc_transforms_to_auxiliary_coulomb_basis()
{
    constexpr double scale = 5.0;

    matrix_m<std::complex<double>> pw_wc(3, 3, MAJOR::COL);
    pw_wc(0, 0) = {2.0, -0.5};
    pw_wc(0, 1) = {3.0, 4.0};
    pw_wc(0, 2) = {-1.0, 0.25};
    pw_wc(1, 0) = std::conj(pw_wc(0, 1));
    pw_wc(2, 0) = std::conj(pw_wc(0, 2));
    pw_wc(1, 1) = {5.0, 0.0};
    pw_wc(1, 2) = {0.75, -0.2};
    pw_wc(2, 1) = std::conj(pw_wc(1, 2));
    pw_wc(2, 2) = {7.0, 0.0};

    const auto auxiliary_wc =
        librpa_int::strict_2d_transform_pw_wc_to_auxiliary_basis(pw_wc, scale);
    assert_complex_close(auxiliary_wc(0, 0), scale * scale * pw_wc(0, 0), 1e-13);
    for (int i = 1; i != 3; ++i)
    {
        assert_complex_close(auxiliary_wc(0, i), scale * pw_wc(0, i), 1e-13);
        assert_complex_close(auxiliary_wc(i, 0), scale * pw_wc(i, 0), 1e-13);
        for (int j = 1; j != 3; ++j)
            assert_complex_close(auxiliary_wc(i, j), pw_wc(i, j), 1e-13);
    }
}

void test_strict_2d_regular_coulomb_legs_are_projected_to_the_gamma_basis()
{
    constexpr double inverse_sqrt_two = 0.70710678118654752440;
    matrix_m<std::complex<double>> coulomb_sqrt(2, 2, MAJOR::COL);
    coulomb_sqrt(0, 0) = 4.0;
    coulomb_sqrt(1, 1) = 1.0;

    matrix_m<std::complex<double>> eigenvectors(2, 2, MAJOR::COL);
    eigenvectors(0, 0) = inverse_sqrt_two;
    eigenvectors(0, 1) = inverse_sqrt_two;
    eigenvectors(1, 0) = inverse_sqrt_two;
    eigenvectors(1, 1) = -inverse_sqrt_two;

    const auto projected =
        librpa_int::strict_2d_project_operator_to_coulomb_basis(coulomb_sqrt, eigenvectors);
    require_double_close(projected(0, 0).real(), 2.5, 1e-14);
    require_double_close(projected(0, 1).real(), 1.5, 1e-14);
    require_double_close(projected(1, 0).real(), 1.5, 1e-14);
    require_double_close(projected(1, 1).real(), 2.5, 1e-14);
}

void test_strict_2d_wc_blocks_match_dense_finite_q_inverse()
{
    const std::complex<double> body{1.6, 0.0};
    const std::complex<double> left_wing{0.25, 0.04};
    const std::complex<double> right_wing = std::conj(left_wing);
    const std::complex<double> head_coefficient{0.9, 0.0};
    const std::complex<double> body_inv = 1.0 / body;
    const std::complex<double> bw_direction = body_inv * left_wing;
    const std::complex<double> wb_direction = right_wing * body_inv;
    const std::complex<double> schur_a = head_coefficient - right_wing * body_inv * left_wing;
    constexpr double q = 0.17;
    constexpr double cut_body_sqrt = 1.3;

    matrix_m<std::complex<double>> body_inv_matrix(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> wb(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv_matrix(0, 0) = body_inv;
    bw(0, 0) = bw_direction;
    wb(0, 0) = wb_direction;
    cut_sqrt(0, 0) = cut_body_sqrt;

    const auto actual =
        librpa_int::strict_2d_wc_blocks_at_q(body_inv_matrix, bw, wb, schur_a, cut_sqrt, q);

    const std::complex<double> eps00 = 1.0 + q * head_coefficient;
    const std::complex<double> eps01 = std::sqrt(q) * right_wing;
    const std::complex<double> eps10 = std::sqrt(q) * left_wing;
    const std::complex<double> determinant = eps00 * body - eps01 * eps10;
    const std::complex<double> inv00 = body / determinant;
    const std::complex<double> inv01 = -eps01 / determinant;
    const std::complex<double> inv10 = -eps10 / determinant;
    const std::complex<double> inv11 = eps00 / determinant;
    const double head_sqrt = std::sqrt(librpa_int::TWO_PI / q);

    assert_complex_close(actual(0, 0), head_sqrt * head_sqrt * (inv00 - 1.0), 1e-13);
    assert_complex_close(actual(0, 1), head_sqrt * (inv01 * cut_body_sqrt), 1e-13);
    assert_complex_close(actual(1, 0), cut_body_sqrt * inv10 * head_sqrt, 1e-13);
    assert_complex_close(actual(1, 1), cut_body_sqrt * cut_body_sqrt * (inv11 - 1.0), 1e-13);
}

void test_strict_2d_wc_cell_average_matches_anisotropic_radial_quadrature()
{
    matrix_m<std::complex<double>> body_inv(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(1, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 1, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv(0, 0) = 0.72;
    bw_cart(0, 0) = {0.18, 0.03};
    bw_cart(0, 1) = {-0.07, 0.02};
    wb_cart(0, 0) = std::conj(bw_cart(0, 0));
    wb_cart(1, 0) = std::conj(bw_cart(0, 1));
    lind(0, 0) = 1.55;
    lind(0, 1) = 0.08;
    lind(1, 0) = 0.08;
    lind(1, 1) = 1.25;
    lind(2, 2) = 1.0;
    cut_sqrt(0, 0) = 1.4;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax{0.21, 0.14, 0.21, 0.14};
    double gamma_area = 0.0;
    for (std::size_t i = 0; i != qmax.size(); ++i)
        gamma_area += weights[i] * qmax[i] * qmax[i] / 2.0;

    const auto analytic = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_inv, bw_cart, wb_cart, lind, cut_sqrt, qx, qy, weights, qmax, gamma_area);

    matrix_m<std::complex<double>> numeric(2, 2, MAJOR::COL);
    constexpr int radial_points = 200000;
    for (std::size_t idir = 0; idir != qx.size(); ++idir)
    {
        matrix_m<std::complex<double>> bw_direction(1, 1, MAJOR::COL);
        matrix_m<std::complex<double>> wb_direction(1, 1, MAJOR::COL);
        bw_direction(0, 0) = bw_cart(0, 0) * qx[idir] + bw_cart(0, 1) * qy[idir];
        wb_direction(0, 0) = wb_cart(0, 0) * qx[idir] + wb_cart(1, 0) * qy[idir];
        const auto a = librpa_int::strict_2d_schur_coefficient(lind, qx[idir], qy[idir]);
        const double dq = qmax[idir] / radial_points;
        for (int ir = 0; ir != radial_points; ++ir)
        {
            const double q = (ir + 0.5) * dq;
            const auto point = librpa_int::strict_2d_wc_blocks_at_q(body_inv, bw_direction,
                                                                    wb_direction, a, cut_sqrt, q);
            const double measure = weights[idir] * q * dq / gamma_area;
            for (int i = 0; i != 2; ++i)
                for (int j = 0; j != 2; ++j) numeric(i, j) += measure * point(i, j);
        }
    }

    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j) assert_complex_close(analytic(i, j), numeric(i, j), 2e-11);
    assert(std::abs(analytic(0, 1)) < 1e-13);
    assert(std::abs(analytic(1, 0)) < 1e-13);
}

struct Point2d
{
    double x;
    double y;
};

double dot(const Point2d &point, const Point2d &normal)
{
    return point.x * normal.x + point.y * normal.y;
}

double cross(const Point2d &left, const Point2d &right)
{
    return left.x * right.y - left.y * right.x;
}

std::vector<Point2d> clip_polygon_halfplane(const std::vector<Point2d> &polygon,
                                            const Point2d &normal, const double bound)
{
    std::vector<Point2d> clipped;
    if (polygon.empty()) return clipped;
    Point2d previous = polygon.back();
    double previous_distance = dot(previous, normal) - bound;
    for (const auto &current : polygon)
    {
        const double current_distance = dot(current, normal) - bound;
        const bool previous_inside = previous_distance <= 1e-14;
        const bool current_inside = current_distance <= 1e-14;
        if (previous_inside != current_inside)
        {
            const double denominator = previous_distance - current_distance;
            if (std::abs(denominator) < 1e-18) std::abort();
            const double fraction = previous_distance / denominator;
            clipped.push_back({previous.x + fraction * (current.x - previous.x),
                               previous.y + fraction * (current.y - previous.y)});
        }
        if (current_inside) clipped.push_back(current);
        previous = current;
        previous_distance = current_distance;
    }
    return clipped;
}

std::pair<double, Point2d> polygon_area_centroid(const std::vector<Point2d> &polygon)
{
    if (polygon.size() < 3) return {0.0, {0.0, 0.0}};
    double twice_area = 0.0;
    Point2d weighted{0.0, 0.0};
    for (std::size_t i = 0; i != polygon.size(); ++i)
    {
        const auto &left = polygon[i];
        const auto &right = polygon[(i + 1) % polygon.size()];
        const double edge_cross = cross(left, right);
        twice_area += edge_cross;
        weighted.x += (left.x + right.x) * edge_cross;
        weighted.y += (left.y + right.y) * edge_cross;
    }
    if (!(twice_area > 0.0)) std::abort();
    return {0.5 * twice_area, {weighted.x / (3.0 * twice_area), weighted.y / (3.0 * twice_area)}};
}

std::vector<Point2d> gamma_cell_neighbors(const Point2d &g1, const Point2d &g2)
{
    std::vector<Point2d> neighbors;
    for (int i = -1; i <= 1; ++i)
        for (int j = -1; j <= 1; ++j)
            if (i != 0 || j != 0) neighbors.push_back({i * g1.x + j * g2.x, i * g1.y + j * g2.y});
    return neighbors;
}

std::vector<Point2d> gamma_voronoi_polygon(const Point2d &g1, const Point2d &g2)
{
    const double extent = 4.0 * std::max(std::hypot(g1.x, g1.y), std::hypot(g2.x, g2.y));
    std::vector<Point2d> polygon{
        {-extent, -extent}, {extent, -extent}, {extent, extent}, {-extent, extent}};
    for (const auto &neighbor : gamma_cell_neighbors(g1, g2))
        polygon = clip_polygon_halfplane(polygon, neighbor, 0.5 * dot(neighbor, neighbor));
    return polygon;
}

double gamma_cell_boundary(const Point2d &direction, const std::vector<Point2d> &neighbors)
{
    double qmax = std::numeric_limits<double>::infinity();
    for (const auto &neighbor : neighbors)
    {
        const double denominator = dot(direction, neighbor);
        if (denominator > 1e-14) qmax = std::min(qmax, 0.5 * dot(neighbor, neighbor) / denominator);
    }
    if (!(qmax > 0.0) || !std::isfinite(qmax)) std::abort();
    return qmax;
}

matrix_m<std::complex<double>> cartesian_gamma_subgrid_average(
    const matrix_m<std::complex<double>> &body_inv, const matrix_m<std::complex<double>> &bw_cart,
    const matrix_m<std::complex<double>> &wb_cart, const matrix_m<std::complex<double>> &lind,
    const matrix_m<std::complex<double>> &regular_body_sqrt, const std::vector<Point2d> &polygon,
    const int subdivisions, double &covered_area)
{
    if (subdivisions < 2) std::abort();
    double xmin = polygon.front().x, xmax = polygon.front().x;
    double ymin = polygon.front().y, ymax = polygon.front().y;
    for (const auto &point : polygon)
    {
        xmin = std::min(xmin, point.x);
        xmax = std::max(xmax, point.x);
        ymin = std::min(ymin, point.y);
        ymax = std::max(ymax, point.y);
    }
    const double dx = (xmax - xmin) / subdivisions;
    const double dy = (ymax - ymin) / subdivisions;
    const double gamma_area = polygon_area_centroid(polygon).first;
    matrix_m<std::complex<double>> average(body_inv.nr() + 1, body_inv.nc() + 1, MAJOR::COL);
    covered_area = 0.0;

    for (int ix = 0; ix != subdivisions; ++ix)
        for (int iy = 0; iy != subdivisions; ++iy)
        {
            const double xlo = xmin + ix * dx;
            const double xhi = xlo + dx;
            const double ylo = ymin + iy * dy;
            const double yhi = ylo + dy;
            auto cell = clip_polygon_halfplane(polygon, {1.0, 0.0}, xhi);
            cell = clip_polygon_halfplane(cell, {-1.0, 0.0}, -xlo);
            cell = clip_polygon_halfplane(cell, {0.0, 1.0}, yhi);
            cell = clip_polygon_halfplane(cell, {0.0, -1.0}, -ylo);
            const auto area_centroid = polygon_area_centroid(cell);
            const double area = area_centroid.first;
            const auto centroid = area_centroid.second;
            if (area == 0.0) continue;
            const double q = std::hypot(centroid.x, centroid.y);
            if (!(q > 1e-14)) std::abort();
            const double qx = centroid.x / q;
            const double qy = centroid.y / q;
            matrix_m<std::complex<double>> bw_direction(body_inv.nr(), 1, MAJOR::COL);
            matrix_m<std::complex<double>> wb_direction(1, body_inv.nc(), MAJOR::COL);
            for (int i = 0; i != body_inv.nr(); ++i)
            {
                bw_direction(i, 0) = bw_cart(i, 0) * qx + bw_cart(i, 1) * qy;
                wb_direction(0, i) = wb_cart(0, i) * qx + wb_cart(1, i) * qy;
            }
            const auto a = librpa_int::strict_2d_schur_coefficient(lind, qx, qy);
            const auto point = librpa_int::strict_2d_wc_blocks_at_q(
                body_inv, bw_direction, wb_direction, a, regular_body_sqrt, q);
            for (int i = 0; i != average.nr(); ++i)
                for (int j = 0; j != average.nc(); ++j)
                    average(i, j) += area * point(i, j) / gamma_area;
            covered_area += area;
        }
    return average;
}

double submatrix_frobenius(const matrix_m<std::complex<double>> &matrix, const int row_start,
                           const int column_start)
{
    double squared = 0.0;
    for (int i = row_start; i != matrix.nr(); ++i)
        for (int j = column_start; j != matrix.nc(); ++j) squared += std::norm(matrix(i, j));
    return std::sqrt(squared);
}

double matrix_hermiticity_residual(const matrix_m<std::complex<double>> &matrix)
{
    double residual = 0.0;
    for (int i = 0; i != matrix.nr(); ++i)
        for (int j = 0; j != matrix.nc(); ++j)
            residual = std::max(residual, std::abs(matrix(i, j) - std::conj(matrix(j, i))));
    return residual;
}

double wing_frobenius(const matrix_m<std::complex<double>> &matrix, const bool head_body)
{
    double squared = 0.0;
    for (int i = 1; i != matrix.nr(); ++i)
        squared += head_body ? std::norm(matrix(0, i)) : std::norm(matrix(i, 0));
    return std::sqrt(squared);
}

void test_strict_2d_wc_cell_average_matches_cartesian_voronoi_subgrid()
{
    matrix_m<std::complex<double>> body_inv(2, 2, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(2, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 2, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> regular_body_sqrt(2, 2, MAJOR::COL);
    body_inv(0, 0) = 0.72;
    body_inv(0, 1) = {0.03, 0.01};
    body_inv(1, 0) = std::conj(body_inv(0, 1));
    body_inv(1, 1) = 0.81;
    bw_cart(0, 0) = {0.18, 0.03};
    bw_cart(0, 1) = {-0.07, 0.02};
    bw_cart(1, 0) = {0.09, -0.01};
    bw_cart(1, 1) = {0.11, 0.04};
    wb_cart = bw_cart.get_transpose(true);
    lind(0, 0) = 1.55;
    lind(0, 1) = 0.08;
    lind(1, 0) = 0.08;
    lind(1, 1) = 1.25;
    lind(2, 2) = 1.0;
    regular_body_sqrt(0, 0) = 1.4;
    regular_body_sqrt(0, 1) = 0.05;
    regular_body_sqrt(1, 0) = 0.05;
    regular_body_sqrt(1, 1) = 1.1;

    const Point2d g1{0.08702149160639744, 0.05024962446042246};
    const Point2d g2{0.0, 0.1004992489208449};
    const auto neighbors = gamma_cell_neighbors(g1, g2);
    const auto polygon = gamma_voronoi_polygon(g1, g2);
    const double gamma_area = polygon_area_centroid(polygon).first;
    require_double_close(gamma_area, std::abs(cross(g1, g2)), 1e-14);

    constexpr int nangle = 5000;
    std::vector<double> qx(nangle), qy(nangle), weights(nangle), qmax(nangle);
    for (int i = 0; i != nangle; ++i)
    {
        const double angle = librpa_int::TWO_PI * i / nangle;
        qx[i] = std::cos(angle);
        qy[i] = std::sin(angle);
        weights[i] = librpa_int::TWO_PI / nangle;
        qmax[i] = gamma_cell_boundary({qx[i], qy[i]}, neighbors);
    }
    const auto analytic = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_inv, bw_cart, wb_cart, lind, regular_body_sqrt, qx, qy, weights, qmax, gamma_area);

    double covered_area = 0.0;
    const auto grid80 = cartesian_gamma_subgrid_average(
        body_inv, bw_cart, wb_cart, lind, regular_body_sqrt, polygon, 80, covered_area);
    require_double_close(covered_area, gamma_area, 1e-14);
    if (matrix_hermiticity_residual(analytic) >= 1e-12 ||
        matrix_hermiticity_residual(grid80) >= 1e-12)
        std::abort();

    const double head_relative = std::abs(grid80(0, 0) - analytic(0, 0)) / std::abs(analytic(0, 0));
    const double body_relative =
        submatrix_frobenius(grid80 - analytic, 1, 1) / submatrix_frobenius(analytic, 1, 1);
    const double wing_absolute =
        std::max(wing_frobenius(grid80, true), wing_frobenius(grid80, false));
    if (head_relative >= 1e-4 || body_relative >= 1e-4 || wing_absolute >= 1e-10) std::abort();
}

void test_strict_2d_wc_blocks_have_finite_small_q_limits()
{
    matrix_m<std::complex<double>> body_inv(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> wb(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv(0, 0) = 0.75;
    bw(0, 0) = {0.12, 0.03};
    wb(0, 0) = std::conj(bw(0, 0));
    cut_sqrt(0, 0) = 1.25;
    const std::complex<double> a{0.6, 0.0};
    constexpr double q = 1.0e-9;

    const auto wc = librpa_int::strict_2d_wc_blocks_at_q(body_inv, bw, wb, a, cut_sqrt, q);
    assert_complex_close(wc(0, 0), -librpa_int::TWO_PI * a, 3e-9);
    assert_complex_close(wc(1, 0), -std::sqrt(librpa_int::TWO_PI) * cut_sqrt(0, 0) * bw(0, 0),
                         3e-9);
    assert_complex_close(wc(0, 1), -std::sqrt(librpa_int::TWO_PI) * wb(0, 0) * cut_sqrt(0, 0),
                         3e-9);
    assert_complex_close(wc(1, 1), cut_sqrt(0, 0) * (body_inv(0, 0) - 1.0) * cut_sqrt(0, 0), 3e-9);
}

void test_strict_2d_wc_average_is_covariant_under_regular_body_rotation()
{
    matrix_m<std::complex<double>> body_inv(2, 2, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(2, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 2, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(2, 2, MAJOR::COL);
    body_inv(0, 0) = 0.70;
    body_inv(0, 1) = 0.04;
    body_inv(1, 0) = 0.04;
    body_inv(1, 1) = 0.82;
    bw_cart(0, 0) = 0.13;
    bw_cart(0, 1) = -0.05;
    bw_cart(1, 0) = 0.08;
    bw_cart(1, 1) = 0.11;
    wb_cart = bw_cart.get_transpose(true);
    lind(0, 0) = 1.4;
    lind(0, 1) = 0.06;
    lind(1, 0) = 0.06;
    lind(1, 1) = 1.7;
    lind(2, 2) = 1.0;
    cut_sqrt(0, 0) = 1.15;
    cut_sqrt(0, 1) = 0.03;
    cut_sqrt(1, 0) = 0.03;
    cut_sqrt(1, 1) = 0.95;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax(4, 0.18);
    const double gamma_area = librpa_int::TWO_PI * 0.18 * 0.18 / 2.0;
    const auto wc = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_inv, bw_cart, wb_cart, lind, cut_sqrt, qx, qy, weights, qmax, gamma_area);

    constexpr double angle = 0.37;
    matrix_m<std::complex<double>> rotation(2, 2, MAJOR::COL);
    rotation(0, 0) = std::cos(angle);
    rotation(0, 1) = -std::sin(angle);
    rotation(1, 0) = std::sin(angle);
    rotation(1, 1) = std::cos(angle);
    const auto rotation_h = rotation.get_transpose(true);
    const auto body_rotated = rotation_h * body_inv * rotation;
    const auto bw_rotated = rotation_h * bw_cart;
    const auto wb_rotated = wb_cart * rotation;
    const auto cut_rotated = rotation_h * cut_sqrt * rotation;
    const auto wc_rotated = librpa_int::strict_2d_average_wc_coulomb_basis(
        body_rotated, bw_rotated, wb_rotated, lind, cut_rotated, qx, qy, weights, qmax, gamma_area);

    matrix_m<std::complex<double>> full_rotation(3, 3, MAJOR::COL);
    full_rotation(0, 0) = 1.0;
    for (int i = 0; i != 2; ++i)
        for (int j = 0; j != 2; ++j) full_rotation(i + 1, j + 1) = rotation(i, j);
    const auto expected = full_rotation.get_transpose(true) * wc * full_rotation;
    for (int i = 0; i != 3; ++i)
        for (int j = 0; j != 3; ++j) assert_complex_close(wc_rotated(i, j), expected(i, j), 2e-13);
}

void test_strict_2d_wc_average_is_bounded_as_gamma_cell_shrinks()
{
    matrix_m<std::complex<double>> body_inv(1, 1, MAJOR::COL);
    matrix_m<std::complex<double>> bw_cart(1, 3, MAJOR::COL);
    matrix_m<std::complex<double>> wb_cart(3, 1, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> cut_sqrt(1, 1, MAJOR::COL);
    body_inv(0, 0) = 0.76;
    bw_cart(0, 0) = {0.14, 0.02};
    bw_cart(0, 1) = {-0.05, 0.01};
    wb_cart(0, 0) = std::conj(bw_cart(0, 0));
    wb_cart(1, 0) = std::conj(bw_cart(0, 1));
    lind(0, 0) = 1.45;
    lind(0, 1) = 0.04;
    lind(1, 0) = 0.04;
    lind(1, 1) = 1.30;
    lind(2, 2) = 1.0;
    cut_sqrt(0, 0) = 1.2;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> weights(4, librpa_int::TWO_PI / 4.0);
    const std::array<int, 4> meshes{12, 14, 16, 20};
    double previous_weighted_norm = std::numeric_limits<double>::infinity();

    for (const int mesh : meshes)
    {
        const double gamma_area = 1.0 / static_cast<double>(mesh * mesh);
        const double radial_extent = std::sqrt(2.0 * gamma_area / librpa_int::TWO_PI);
        const std::vector<double> qmax(4, radial_extent);
        const auto average = librpa_int::strict_2d_average_wc_coulomb_basis(
            body_inv, bw_cart, wb_cart, lind, cut_sqrt, qx, qy, weights, qmax, gamma_area);

        double average_norm_squared = 0.0;
        for (int i = 0; i != average.nr(); ++i)
            for (int j = 0; j != average.nc(); ++j)
                average_norm_squared += std::norm(average(i, j));
        const double average_norm = std::sqrt(average_norm_squared);
        assert(std::isfinite(average_norm));
        assert(average_norm < 10.0);

        const double weighted_norm = average_norm / static_cast<double>(mesh * mesh);
        assert(weighted_norm < previous_weighted_norm);
        previous_weighted_norm = weighted_norm;
    }
}

void test_rpa_chi0v_wing_desc_uses_global_rows(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(10, 10, 0, 0);

    ArrayDesc desc_full_wing(blacs_h);
    desc_full_wing.init(11, 3, desc_body.mb(), 1, 0, 0);

    const auto desc_wing = librpa_int::make_rpa_chi0v_wing_desc(
        desc_body, 1, desc_full_wing.m_loc(), desc_full_wing.n_loc());

    if (desc_body.nprows() > 1)
    {
        assert(desc_full_wing.m_loc() < desc_full_wing.m());
    }
    assert(desc_wing.m() == 11);
    assert(desc_wing.n() == 3);
    assert(desc_wing.mb() == desc_body.mb());
    assert(desc_wing.nb() == 1);
    assert(desc_wing.m_loc() == desc_full_wing.m_loc());
    assert(desc_wing.n_loc() == desc_full_wing.n_loc());
}

void test_headwing_spin_weights()
{
    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 2, false) - 0.75) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(2, false) - 1.0) < 1e-12);

    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 1, true) - 0.75) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(1, true) - 1.0) < 1e-12);

    assert(std::abs(librpa_int::headwing_transition_weight(1.0, 0.25, 1, false) - 0.375) < 1e-12);
    assert(std::abs(librpa_int::headwing_spin_prefactor(1, false) - 2.0) < 1e-12);
}

void test_wing_cartesian_gram_is_invariant_under_row_phases()
{
    ComplexMatrix wing(2, 3);
    wing(0, 0) = {1.0, 1.0};
    wing(0, 1) = {2.0, -1.0};
    wing(0, 2) = {0.0, -1.0};
    wing(1, 0) = {0.5, -2.0};
    wing(1, 1) = {-1.0, 0.25};
    wing(1, 2) = {3.0, 0.5};

    ComplexMatrix phased = wing;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        phased(0, alpha) *= std::complex<double>{0.0, 1.0};
        phased(1, alpha) *= std::complex<double>{-1.0, 0.0};
    }

    const auto gram = librpa_int::compute_wing_cartesian_gram(wing);
    const auto phased_gram = librpa_int::compute_wing_cartesian_gram(phased);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        for (int beta = 0; beta != 3; ++beta)
        {
            assert_complex_close(gram.at(alpha).at(beta), phased_gram.at(alpha).at(beta), 1e-12);
        }
    }
    assert_complex_close(gram.at(0).at(0), {6.25, 0.0}, 1e-12);
}

void test_velocity_matrix_initialization()
{
    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 2, 3, 4);

    assert(velocity.size() == 2);
    for (int ispin = 0; ispin != 2; ++ispin)
    {
        assert(velocity[ispin].size() == 3);
        for (int ik = 0; ik != 3; ++ik)
        {
            assert(velocity[ispin][ik].size() == 3);
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                assert(velocity[ispin][ik][alpha].nr == 4);
                assert(velocity[ispin][ik][alpha].nc == 4);
            }
        }
    }
}

void test_headwing_local_kpoints_prefers_kpoint_blacs_context()
{
    const auto all_k = librpa_int::headwing_local_kpoints(4, nullptr);
    assert((all_k == std::vector<int>{0, 1, 2, 3}));

    if (librpa_int::get_mpi_size(MPI_COMM_WORLD) != 4) return;

    KPointBlacsProcessShape shape(2, 2, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 4);
    const auto local_k = librpa_int::headwing_local_kpoints(4, &kctx);

    if (kctx.kpoint_group_id() == 0)
        assert((local_k == std::vector<int>{0, 2}));
    else
        assert((local_k == std::vector<int>{1, 3}));

    const auto mismatched = librpa_int::headwing_local_kpoints(5, &kctx);
    assert((mismatched == std::vector<int>{0, 1, 2, 3, 4}));
}

void test_headwing_world_fourier_uses_all_R_blocks_at_nonzero_k()
{
    AtomicBasis basis_wfc({1});
    AtomicBasis basis_abf({1});
    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    Cs_data.data_libri[0][{0, {0, 0, 0}}] = make_scalar_cs_tensor(2.0);
    Cs_data.data_libri[0][{0, {1, 0, 0}}] = make_scalar_cs_tensor(3.0);
    Cs_data.data_libri[0][{0, {2, 0, 0}}] = make_scalar_cs_tensor(-1.0);

    const auto targets = librpa_int::build_headwing_full_bz_fourier_targets(
        {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}});
    const auto Cs_IJ_k =
        librpa_int::fourier_headwing_cs_to_ijk(Cs_data, basis_wfc, basis_abf, targets);

    const auto &blocks = Cs_IJ_k.at(0);
    assert_complex_close(blocks.at({0, 0})(0, 0, 0), {4.0, 0.0}, 1e-12);
    assert_complex_close(blocks.at({0, 1})(0, 0, 0), {3.0, 3.0}, 1e-12);
}

void test_headwing_symmetry_fourier_target_ids_are_deterministic()
{
    librpa_int::SymmetryContext ctx;
    ctx.set_available();

    librpa_int::SymmetryKStar first;
    first.star_index = 0;
    first.k_ibz = {0.0, 0.0, 0.0};
    first.members.resize(2);
    first.members[0].k_bz = {0.0, 0.0, 0.0};
    first.members[1].k_bz = {0.5, 0.0, 0.0};
    ctx.kstars.push_back(first);

    librpa_int::SymmetryKStar second;
    second.star_index = 1;
    second.k_ibz = {0.25, 0.0, 0.0};
    second.members.resize(3);
    second.members[0].k_bz = {0.25, 0.0, 0.0};
    second.members[1].k_bz = {0.0, 0.25, 0.0};
    second.members[2].k_bz = {0.0, 0.0, 0.25};
    ctx.kstars.push_back(second);

    PeriodicBoundaryData pbc;
    const auto flattened = librpa_int::build_headwing_symmetry_fourier_targets(
        ctx, pbc, {first.k_ibz, second.k_ibz});

    assert((flattened.target_ids_by_ibz_member[0] == std::vector<int>{0, 1}));
    assert((flattened.target_ids_by_ibz_member[1] == std::vector<int>{2, 3, 4}));
    assert(flattened.targets.size() == 5);
    for (int target_id = 0; target_id != 5; ++target_id)
        assert(flattened.targets[target_id].target_id == target_id);
    assert(flattened.targets[0].owner_ik == 0);
    assert(flattened.targets[1].owner_ik == 0);
    assert(flattened.targets[2].owner_ik == 1);
    assert(flattened.targets[4].kfrac == second.members[2].k_bz);
}

void test_headwing_ijk_redistribution_is_owner_group_local()
{
    if (librpa_int::get_mpi_size(MPI_COMM_WORLD) != 4) return;

    KPointBlacsProcessShape shape(2, 2, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 4);
    const auto desc_nao_nao = kctx.create_array_desc(2, 2, 1, 1);
    const AtomicBasis basis_wfc(std::vector<std::size_t>{1, 1});
    const AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    const auto targets = librpa_int::build_headwing_full_bz_fourier_targets(
        {{0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.75, 0.0, 0.0}});
    const std::set<int> local_iks(kctx.kpoints_local().begin(), kctx.kpoints_local().end());
    const auto requests = librpa_int::build_headwing_cs_ijk_requests(
        basis_wfc, targets, kctx.kpoints_local(), desc_nao_nao);

    for (const auto &[J, target_id] : requests.second)
    {
        (void)J;
        assert(local_iks.count(targets.at(target_id).owner_ik) == 1);
    }

#ifdef LIBRPA_USE_LIBRI
    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    if (librpa_int::get_mpi_rank(MPI_COMM_WORLD) == 0)
    {
        for (int I = 0; I != 2; ++I)
            for (int J = 0; J != 2; ++J)
                Cs_data.data_libri[I][{J, {0, 0, 0}}] =
                    make_scalar_cs_tensor(1.0 + 2.0 * I + J);
    }
    const auto Cs_IJ_k = librpa_int::redistribute_headwing_cs_ijk(
        Cs_data, basis_wfc, basis_abf, targets, kctx.kpoints_local(), desc_nao_nao,
        librpa_int::global::mpi_comm_global_h);
    for (const auto &[I, Jtargets] : Cs_IJ_k)
    {
        assert(requests.first.count(I) == 1);
        for (const auto &[Jtarget, tensor] : Jtargets)
        {
            assert(requests.second.count(Jtarget) == 1);
            assert(local_iks.count(targets.at(Jtarget.second).owner_ik) == 1);
            assert(std::abs(tensor(0, 0, 0)) > 0.0);
        }
    }
#endif
}

void test_accumulate_wing_mu_for_pair_matches_original_formula()
{
    const std::vector<double> omega{0.5, 1.25};
    const std::array<std::complex<double>, 3> velocity{std::complex<double>{0.2, -0.1},
                                                       std::complex<double>{-0.3, 0.4},
                                                       std::complex<double>{0.15, 0.05}};
    const std::complex<double> c_mn{0.7, -0.2};
    const double egap = 1.8;
    const double factor1 = 0.6;
    const double factor2 = 0.125;

    std::array<std::complex<double>, 6> accumulated{};
    librpa_int::accumulate_wing_mu_for_pair(omega, velocity, c_mn, egap, factor1, factor2,
                                            accumulated.data());

    for (std::size_t iomega = 0; iomega != omega.size(); ++iomega)
    {
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            const auto denom = omega[iomega] * omega[iomega] + egap * egap;
            const auto expected = factor1 * std::conj(c_mn * velocity[alpha]) / denom +
                                  factor2 * c_mn * velocity[alpha] / denom;
            assert_complex_close(accumulated[iomega * 3 + alpha], expected, 1e-12);
        }
    }
}

SymmetryKStarMember make_headwing_wfc_atom_swap_member(
    const std::complex<double> &rot_0, const std::complex<double> &rot_1)
{
    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};

    SymmetryKAtomRotation atom_0;
    atom_0.atom_from = 0;
    atom_0.atom_to = 1;
    atom_0.atom_type = 0;
    atom_0.lmax = 0;
    atom_0.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_0.bloch_rsh_rotations[0](0, 0) = rot_0;

    SymmetryKAtomRotation atom_1;
    atom_1.atom_from = 1;
    atom_1.atom_to = 0;
    atom_1.atom_type = 0;
    atom_1.lmax = 0;
    atom_1.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_1.bloch_rsh_rotations[0](0, 0) = rot_1;

    member.atom_rotations = {atom_0, atom_1};
    return member;
}

void test_headwing_wfc_restore_applies_atom_permutation()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const auto member = make_headwing_wfc_atom_swap_member(
        {2.0, 0.5}, {3.0, -0.25});

    ComplexMatrix wfc_ibz(1, 2);
    wfc_ibz(0, 0) = {0.7, -0.2};
    wfc_ibz(0, 1) = {-0.4, 0.6};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, nullptr);

    assert_complex_close(wfc_bz(0, 0), wfc_ibz(0, 1) * std::complex<double>{3.0, -0.25},
                         1e-12);
    assert_complex_close(wfc_bz(0, 1), wfc_ibz(0, 0) * std::complex<double>{2.0, 0.5},
                         1e-12);
}

void test_headwing_wfc_restore_applies_time_reversal()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}};

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    member.k_bz = {0.0, 0.0, 0.0};
    member.time_reversal = true;
    SymmetryKAtomRotation atom;
    atom.atom_from = 0;
    atom.atom_to = 0;
    atom.atom_type = 0;
    atom.lmax = 0;
    atom.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom.bloch_rsh_rotations[0](0, 0) = {0.25, 0.75};
    member.atom_rotations.push_back(atom);

    ComplexMatrix wfc_ibz(1, 1);
    wfc_ibz(0, 0) = {0.6, -0.35};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, nullptr);

    assert_complex_close(wfc_bz(0, 0), std::conj(wfc_ibz(0, 0)) * std::complex<double>{0.25, 0.75},
                         1e-12);
}

void test_headwing_velocity_restore_uses_inverse_spatial_route()
{
    SymmetryContext ctx;
    ctx.lattice_vectors.Identity();
    SymmetryOperation operation;
    operation.rotation = Matrix3(0.0, -1.0, 0.0,
                                 1.0,  0.0, 0.0,
                                 0.0,  0.0, 1.0);
    ctx.rspace_operations = {operation};

    SymmetryKStarMember member;
    member.spatial_isym = 0;
    std::array<ComplexMatrix, 3> velocity_ibz;
    for (auto &component : velocity_ibz) component.create(1, 1);
    velocity_ibz[0](0, 0) = {1.0, 2.0};
    velocity_ibz[1](0, 0) = {3.0, -1.0};
    velocity_ibz[2](0, 0) = {-0.5, 0.25};

    const auto velocity_bz = librpa_int::rotate_headwing_velocity_to_kstar_member(
        ctx, member, velocity_ibz, 1, false);
    assert_complex_close(velocity_bz[0](0, 0), -velocity_ibz[1](0, 0), 1e-12);
    assert_complex_close(velocity_bz[1](0, 0), velocity_ibz[0](0, 0), 1e-12);
    assert_complex_close(velocity_bz[2](0, 0), velocity_ibz[2](0, 0), 1e-12);

    member.time_reversal = true;
    const auto velocity_bz_tr = librpa_int::rotate_headwing_velocity_to_kstar_member(
        ctx, member, velocity_ibz, 1, true);
    assert_complex_close(velocity_bz_tr[0](0, 0), std::conj(velocity_ibz[1](0, 0)), 1e-12);
    assert_complex_close(velocity_bz_tr[1](0, 0), -std::conj(velocity_ibz[0](0, 0)), 1e-12);
    assert_complex_close(velocity_bz_tr[2](0, 0), -std::conj(velocity_ibz[2](0, 0)), 1e-12);
}

void test_headwing_direct_full_bz_velocity_selects_kstar_member()
{
    librpa_int::velocity_matrix_t velocity_full;
    librpa_int::initialize_velocity_matrix(velocity_full, 1, 2, 1);
    velocity_full[0][0][0](0, 0) = {1.0, 0.0};
    velocity_full[0][1][0](0, 0) = {2.0, 0.0};
    velocity_full[0][1][1](0, 0) = {3.0, 0.0};
    velocity_full[0][1][2](0, 0) = {4.0, 0.0};

    const std::vector<std::vector<int>> member_source_ik{{1}};
    const auto &velocity = librpa_int::direct_full_bz_velocity_for_kstar_member(
        velocity_full, member_source_ik, 0, 0, 0);
    assert_complex_close(velocity[0](0, 0), {2.0, 0.0}, 1e-12);
    assert_complex_close(velocity[1](0, 0), {3.0, 0.0}, 1e-12);
    assert_complex_close(velocity[2](0, 0), {4.0, 0.0}, 1e-12);
}

void test_headwing_direct_full_bz_wfc_selects_same_kstar_member()
{
    MeanField wfc_full(1, 2, 1, 1);
    auto &wfc_k0 = wfc_full.get_eigenvectors()[0][0][0];
    wfc_k0.create(1, 1);
    wfc_k0(0, 0) = {1.0, 0.0};
    auto &wfc_k1 = wfc_full.get_eigenvectors()[0][0][1];
    wfc_k1.create(1, 1);
    wfc_k1(0, 0) = {0.0, 1.0};

    const std::vector<std::vector<int>> member_source_ik{{1}};
    const auto &wfc = librpa_int::direct_full_bz_wfc_for_kstar_member(
        wfc_full, member_source_ik, 0, 0, 0, 0);
    assert_complex_close(wfc(0, 0), {0.0, 1.0}, 1e-12);
}

void test_headwing_direct_full_bz_wfc_local_block(const BlacsCtxtHandler &blacs_h)
{
    ComplexMatrix wfc_full(5, 7);
    for (int iband = 0; iband != wfc_full.nr; ++iband)
        for (int iao = 0; iao != wfc_full.nc; ++iao)
            wfc_full(iband, iao) = {100.0 * iband + iao, iband - 0.1 * iao};

    ArrayDesc desc_wfc(blacs_h);
    desc_wfc.init(7, 5, 2, 3, 0, 0);
    const auto wfc_local = librpa_int::localize_direct_full_bz_wfc(wfc_full, desc_wfc);
    assert(wfc_local.nr == desc_wfc.n_loc());
    assert(wfc_local.nc == desc_wfc.m_loc());
    for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
    {
        const int iband = desc_wfc.indx_l2g_c(jloc);
        for (int iloc = 0; iloc != desc_wfc.m_loc(); ++iloc)
        {
            const int iao = desc_wfc.indx_l2g_r(iloc);
            assert_complex_close(wfc_local(jloc, iloc), wfc_full(iband, iao), 1e-12);
        }
    }
}

RI::Tensor<double> make_single_value_tensor(const double value)
{
    auto data = std::make_shared<std::valarray<double>>(1);
    (*data)[0] = value;
    return RI::Tensor<double>({1UL, 1UL, 1UL}, data);
}

void compare_local_blacs_matrices(
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &actual,
    const std::pair<ArrayDesc, matrix_m<std::complex<double>>> &expected,
    const double tolerance)
{
    assert(actual.first.m() == expected.first.m());
    assert(actual.first.n() == expected.first.n());
    assert(actual.first.m_loc() == expected.first.m_loc());
    assert(actual.first.n_loc() == expected.first.n_loc());
    for (int i = 0; i != actual.first.m_loc(); ++i)
    {
        for (int j = 0; j != actual.first.n_loc(); ++j)
        {
            assert_complex_close(actual.second(i, j), expected.second(i, j), tolerance);
        }
    }
}

void test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(
    const BlacsCtxtHandler &blacs_h)
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.5, 0.0, 0.0}}};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);

    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    const std::vector<SpeciesBasisLayout> layouts{layout};
    const std::map<librpa_int::atom_t, size_t> atom_nw{{0, 1}, {1, 1}};

    auto member = make_headwing_wfc_atom_swap_member({0.0, 1.0}, {-1.0, 0.0});
    member.k_bz = {0.5, 0.0, 0.0};

    MeanField mf_ibz(1, 1, 2, 2, 1);
    auto &wfc_ibz = mf_ibz.get_eigenvectors()[0][0][0];
    wfc_ibz.create(2, 2);
    wfc_ibz(0, 0) = {0.7, -0.2};
    wfc_ibz(0, 1) = {-0.4, 0.6};
    wfc_ibz(1, 0) = {0.3, 0.5};
    wfc_ibz(1, 1) = {-0.8, -0.1};

    const auto wfc_bz = librpa_int::rotate_headwing_wfc_to_kstar_member(
        ctx, member, layouts, atom_nw, {0.0, 0.0, 0.0}, wfc_ibz, &member.k_bz);

    MeanField mf_full(1, 1, 2, 2, 1);
    auto &wfc_full = mf_full.get_eigenvectors()[0][0][0];
    wfc_full.create(2, 2);
    for (int ib = 0; ib != 2; ++ib)
    {
        for (int iao = 0; iao != 2; ++iao)
        {
            wfc_full(ib, iao) = wfc_bz(ib, iao);
        }
    }

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc(std::vector<size_t>{1, 1});
    AtomicBasis basis_abf(std::vector<size_t>{1, 1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac_ibz{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df_ibz(mf_ibz, velocity, kfrac_ibz, basis_wfc, basis_abf, omega, 2, 2, 1, 1,
                      pbc, librpa_int::global::mpi_comm_global_h, blacs_h);
    diele_func df_full(mf_full, velocity, {member.k_bz}, basis_wfc, basis_abf, omega, 2, 2, 1, 1,
                       pbc, librpa_int::global::mpi_comm_global_h, blacs_h);

    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = make_single_value_tensor(1.0);
    Cs_IJ[0][{1, {0, 0, 0}}] = make_single_value_tensor(-0.35);
    Cs_IJ[0][{1, {1, 0, 0}}] = make_single_value_tensor(0.42);

    std::vector<std::vector<const ComplexMatrix *>> restored_wfc_ptrs(
        1, std::vector<const ComplexMatrix *>(1, &wfc_bz));
    const auto restored = df_ibz.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, blacs_h, member.k_bz, &restored_wfc_ptrs);
    const auto full = df_full.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, blacs_h, member.k_bz);

    compare_local_blacs_matrices(restored, full, 1e-12);
}

void test_kblacs_transform_matches_original_transform(const BlacsCtxtHandler &blacs_h)
{
    const int nprocs = librpa_int::get_mpi_size(MPI_COMM_WORLD);
    KPointBlacsProcessShape shape(1, nprocs, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 1);
    const auto desc_wfc = kctx.create_array_desc(2, 2, 2, 2);

    MeanField mf(1, 1, 2, 2, 1);
    auto &wfc = mf.get_eigenvectors()[0][0][0];
    wfc.create(2, 2);
    wfc(0, 0) = {0.7, 0.2};
    wfc(0, 1) = {-0.3, 0.4};
    wfc(1, 0) = {0.5, -0.1};
    wfc(1, 1) = {0.9, 0.3};

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.25, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h, true, &kctx, &desc_wfc);

    auto tensor_R0 = std::make_shared<std::valarray<double>>(4);
    (*tensor_R0)[0] = 1.0;
    (*tensor_R0)[1] = 0.2;
    (*tensor_R0)[2] = -0.4;
    (*tensor_R0)[3] = 0.8;
    auto tensor_R1 = std::make_shared<std::valarray<double>>(4);
    (*tensor_R1)[0] = 0.3;
    (*tensor_R1)[1] = -0.1;
    (*tensor_R1)[2] = 0.2;
    (*tensor_R1)[3] = 0.5;
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_R0);
    Cs_IJ[0][{0, {1, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_R1);

    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    Cs_data.data_libri = Cs_IJ;
    const auto targets = librpa_int::build_headwing_full_bz_fourier_targets(kfrac);
    const auto Cs_IJ_k =
        librpa_int::fourier_headwing_cs_to_ijk(Cs_data, basis_wfc, basis_abf, targets);

    auto original = df.transform_Cs2mnk(0, 0, Cs_IJ);
    auto kblacs = df.transform_Cs2mnk_kblacs(0, 0, Cs_IJ, kctx.blacs_h, kfrac[0]);
    auto prefourier = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);

    assert(original.first.m() == kblacs.first.m());
    assert(original.first.n() == kblacs.first.n());
    assert(original.first.m_loc() == kblacs.first.m_loc());
    assert(original.first.n_loc() == kblacs.first.n_loc());
    for (int i = 0; i != original.first.m_loc(); ++i)
    {
        for (int j = 0; j != original.first.n_loc(); ++j)
        {
            assert_complex_close(kblacs.second(i, j), original.second(i, j), 1e-12);
            assert_complex_close(prefourier.second(i, j), kblacs.second(i, j), 1e-12);
        }
    }

    std::vector<std::vector<ComplexMatrix>> override_storage(1);
    override_storage[0].resize(1);
    std::vector<std::vector<const ComplexMatrix *>> override_ptrs(1);
    override_ptrs[0].assign(1, nullptr);
    if (desc_wfc.is_src())
    {
        auto &override_wfc = override_storage[0][0];
        override_wfc.create(2, 2);
        override_wfc(0, 0) = {0.2, -0.4};
        override_wfc(0, 1) = {0.6, 0.1};
        override_wfc(1, 0) = {-0.5, 0.3};
        override_wfc(1, 1) = {0.4, -0.2};
        override_ptrs[0][0] = &override_wfc;
    }
    const auto real_override = df.transform_Cs2mnk_kblacs(
        0, 0, Cs_IJ, kctx.blacs_h, kfrac[0], &override_ptrs);
    const auto prefourier_override = df.transform_Cs2mnk_kblacs(
        0, 0, 0, Cs_IJ_k, kctx.blacs_h, &override_ptrs);
    int override_changed_local = 0;
    for (int i = 0; i != real_override.first.m_loc(); ++i)
    {
        for (int j = 0; j != real_override.first.n_loc(); ++j)
        {
            assert_complex_close(prefourier_override.second(i, j), real_override.second(i, j),
                                 1e-12);
            if (std::abs(real_override.second(i, j) - kblacs.second(i, j)) > 1e-12)
                override_changed_local = 1;
        }
    }
    int override_changed = 0;
    MPI_Allreduce(&override_changed_local, &override_changed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    assert(override_changed == 1);
}

void test_kblacs_transform_uses_rectangular_opt_128_blocks()
{
    constexpr int n_basis = 320;
    constexpr int n_states = 300;
    constexpr int n_ao_Mu = 160;

    const int nprocs = librpa_int::get_mpi_size(MPI_COMM_WORLD);
    KPointBlacsProcessShape shape(1, nprocs, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 1);
    const auto desc_wfc_full =
        kctx.create_array_desc(n_basis, n_states, n_basis, n_states);
    const int block_ao =
        librpa_int::get_capped_blacs_block_size(
            n_basis, librpa_int::wfc_gemm_block_size_opt, kctx.blacs_h);
    const int block_band =
        librpa_int::get_capped_blacs_block_size(
            n_states, librpa_int::wfc_gemm_block_size_opt, kctx.blacs_h);
    const auto desc_wfc =
        kctx.create_array_desc(n_basis, n_states, block_ao, block_band);

    MeanField mf(1, 1, n_states, n_basis, 1);
    if (desc_wfc_full.is_src())
    {
        auto &wfc = mf.get_eigenvectors()[0][0][0];
        wfc.create(n_states, n_basis);
        wfc.zero_out();
        for (int ib = 0; ib != n_states; ++ib) wfc(ib, ib) = {1.0, 0.0};
    }
    librpa_int::redistribute_meanfield_eigvecs_kblacs(
        mf, kctx, desc_wfc_full, desc_wfc, "test headwing");

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, n_states);
    AtomicBasis basis_wfc(std::vector<std::size_t>{n_ao_Mu, n_basis - n_ao_Mu});
    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, n_basis, n_states, 1,
                  2, pbc, librpa_int::global::mpi_comm_global_h, kctx.blacs_h, true, &kctx,
                  &desc_wfc);

    auto tensor_data = std::make_shared<std::valarray<std::complex<double>>>(
        std::complex<double>{0.0, 0.0}, static_cast<std::size_t>(n_ao_Mu) * n_ao_Mu);
    for (int i = 0; i != n_ao_Mu; ++i)
        (*tensor_data)[static_cast<std::size_t>(i) * n_ao_Mu + i] = {1.0, 0.0};

    librpa_int::HeadwingCsIJKMap Cs_IJ_k;
    Cs_IJ_k[0][{0, 0}] = RI::Tensor<std::complex<double>>(
        {1UL, static_cast<std::size_t>(n_ao_Mu), static_cast<std::size_t>(n_ao_Mu)},
        tensor_data);

    const auto transformed =
        df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);
    const auto &desc = transformed.first;
    const auto &matrix = transformed.second;
    assert(desc.m() == n_states && desc.n() == n_states);
    assert(desc.mb() == librpa_int::wfc_gemm_block_size_opt &&
           desc.nb() == librpa_int::wfc_gemm_block_size_opt);

    for (int ilo = 0; ilo != desc.m_loc(); ++ilo)
    {
        const int i = desc.indx_l2g_r(ilo);
        for (int jlo = 0; jlo != desc.n_loc(); ++jlo)
        {
            const int j = desc.indx_l2g_c(jlo);
            const std::complex<double> expected =
                i == j && i < n_ao_Mu ? std::complex<double>{2.0, 0.0}
                                       : std::complex<double>{0.0, 0.0};
            assert_complex_close(matrix(ilo, jlo), expected, 1e-12);
        }
    }
}

void test_transform_Cs2mnk_can_keep_spin_channels_separate(const BlacsCtxtHandler &blacs_h)
{
    const int nprocs = librpa_int::get_mpi_size(MPI_COMM_WORLD);
    KPointBlacsProcessShape shape(1, nprocs, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 1);
    const auto desc_wfc = kctx.create_array_desc(2, 2, 2, 2);

    MeanField mf(2, 1, 2, 2, 1);
    auto &wfc_up = mf.get_eigenvectors()[0][0][0];
    auto &wfc_dn = mf.get_eigenvectors()[1][0][0];
    wfc_up.create(2, 2);
    wfc_dn.create(2, 2);
    wfc_up(0, 0) = {0.7, 0.2};
    wfc_up(0, 1) = {-0.3, 0.4};
    wfc_up(1, 0) = {0.5, -0.1};
    wfc_up(1, 1) = {0.9, 0.3};
    wfc_dn(0, 0) = {0.2, -0.6};
    wfc_dn(0, 1) = {0.8, 0.1};
    wfc_dn(1, 0) = {-0.4, 0.5};
    wfc_dn(1, 1) = {0.6, -0.2};

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 2, 1, 2);
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 2, 2, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h, true, &kctx, &desc_wfc);

    auto tensor_data = std::make_shared<std::valarray<double>>(4);
    (*tensor_data)[0] = 1.0;
    (*tensor_data)[1] = 0.2;
    (*tensor_data)[2] = -0.4;
    (*tensor_data)[3] = 0.8;
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_data);
    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    Cs_data.data_libri = Cs_IJ;
    const auto Cs_IJ_k = librpa_int::fourier_headwing_cs_to_ijk(
        Cs_data, basis_wfc, basis_abf,
        librpa_int::build_headwing_full_bz_fourier_targets(kfrac));

    const auto all_spin = df.transform_Cs2mnk(0, 0, Cs_IJ);
    const auto spin_up = df.transform_Cs2mnk(0, 0, Cs_IJ, 0);
    const auto spin_dn = df.transform_Cs2mnk(0, 0, Cs_IJ, 1);
    const auto all_spin_k = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);
    const auto spin_up_k = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h, nullptr, 0);
    const auto spin_dn_k = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h, nullptr, 1);

    int spin_channels_differ_local = 0;
    for (int i = 0; i != all_spin.first.m_loc(); ++i)
    {
        for (int j = 0; j != all_spin.first.n_loc(); ++j)
        {
            assert_complex_close(all_spin.second(i, j), spin_up.second(i, j) + spin_dn.second(i, j),
                                 1e-12);
            assert_complex_close(all_spin_k.second(i, j),
                                 spin_up_k.second(i, j) + spin_dn_k.second(i, j), 1e-12);
            if (std::abs(spin_up_k.second(i, j) - spin_dn_k.second(i, j)) > 1e-12)
                spin_channels_differ_local = 1;
        }
    }
    int spin_channels_differ = 0;
    MPI_Allreduce(&spin_channels_differ_local, &spin_channels_differ, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    assert(spin_channels_differ == 1);
}

void test_head_initialization_does_not_require_coulomb_diagonalization(
    const BlacsCtxtHandler &blacs_h)
{
    MeanField mf(1, 1, 2, 1);
    mf.get_eigenvals()[0](0, 0) = -0.5;
    mf.get_eigenvals()[0](0, 1) = 0.5;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_weight()[0](0, 1) = 0.0;

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        velocity[0][0][alpha](1, 0) = std::complex<double>{0.1 * (alpha + 1), 0.0};
        velocity[0][0][alpha](0, 1) = std::complex<double>{0.1 * (alpha + 1), 0.0};
    }

    AtomicBasis basis_wfc({1});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};
    const atpair_k_cplx_mat_t empty_vq;

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 1, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h);

    df.init(0.0, empty_vq);
    df.cal_head();
    assert(df.get_head_vec().size() == 1);
}

void test_strict_2d_gamma_quadrature_is_ready_after_wing_initialization(
    const BlacsCtxtHandler &blacs_h)
{
    MeanField mf(1, 1, 2, 1);
    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 2);
    AtomicBasis basis_wfc({1});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 8.0});
    const std::vector<double> kvecs{0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    librpa_int::PI,
                                    0.0,
                                    librpa_int::PI,
                                    0.0,
                                    0.0,
                                    librpa_int::PI,
                                    librpa_int::PI,
                                    0.0};
    pbc.set_kgrids_kvec(2, 2, 1, kvecs);
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};
    const atpair_k_cplx_mat_t empty_vq;

    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 1, 2, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h);
    df.configure_strict_2d_coulomb_head(true, librpa_int::TWO_PI);
    assert(df.use_2d_dielectric);
    const auto normalization = librpa_int::strict_2d_coulomb_head_normalization(
        pbc, librpa_int::TWO_PI);
    require_double_close(df.get_strict_2d_pw_to_auxiliary_scale(),
                         normalization.pw_to_auxiliary_scale, 1e-14);
    df.init_wing(0.0, empty_vq);

    const double average = df.get_strict_2d_bare_coulomb_gamma_average();
    assert(std::isfinite(average));
    assert(average > 0.0);
}

// Dense old Coulomb-basis averaged inverse dielectric reference. sqrt(V) is an
// independent fixed input; U supplies the Coulomb eigenvectors (x1 = U[:,0] and
// the rotation back to the ABF basis). Returns eps_inv in the ABF basis.
Matz coulomb_basis_eps_inv_reference(
    const Matz &U, const Matz &sqrtV, const Matz &chi0,
    const Matz &wing_mu, const Matz &head,
    const std::vector<std::array<double, 3>> &q_pts, const std::vector<double> &q_rho,
    int n_nonsingular = -1)
{
    const int n = U.nr();
    if (n_nonsingular < 0) n_nonsingular = n;
    assert(n_nonsingular > 0 && n_nonsingular <= n);
    const int nl = n_nonsingular - 1;

    // sqrtveig = sqrt(V) * U  (= U * diag(sqrt(lambda)) when consistent)
    const auto sqrtveig = sqrtV * U;

    // E_coul = I - sqrtveig^H * chi0 * sqrtveig = U^H * E_abf * U
    auto E_coul = sqrtveig.get_transpose(true) * chi0 * sqrtveig;
    E_coul *= -1.0;
    for (int i = 0; i < n; ++i) E_coul(i, i) += 1.0;

    if (nl == 0)
    {
        // No body channels: L = H and the averaged inverse is a0 * P with
        // P = x1*x1^H. Avoids a zero-dimensional LAPACK inversion.
        const auto L00 = head(0, 0), L01 = head(0, 1), L02 = head(0, 2);
        const auto L10 = head(1, 0), L11 = head(1, 1), L12 = head(1, 2);
        const auto L20 = head(2, 0), L21 = head(2, 1), L22 = head(2, 2);
        std::complex<double> a0 = 0.0;
        for (std::size_t ileb = 0; ileb < q_pts.size(); ++ileb)
        {
            const double qx = q_pts[ileb][0], qy = q_pts[ileb][1], qz = q_pts[ileb][2];
            const auto qLq = qx * (qx * L00 + qy * L01 + qz * L02) +
                             qy * (qx * L10 + qy * L11 + qz * L12) +
                             qz * (qx * L20 + qy * L21 + qz * L22);
            a0 += q_rho[ileb] / qLq;
        }
        Matz x1(n, 1, MAJOR::COL);
        for (int i = 0; i < n; ++i) x1(i, 0) = U(i, 0);
        return a0 * (x1 * x1.get_transpose(true));
    }

    // body = E_coul[1:,1:], invert it via LU
    Matz body(nl, nl, MAJOR::COL);
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < nl; ++j)
            body(i, j) = E_coul(i + 1, j + 1);
    auto body_inv = body.copy();
    std::vector<int> ipiv(static_cast<std::size_t>(nl));
    std::vector<std::complex<double>> work(static_cast<std::size_t>(nl * nl));
    int info = 0;
    librpa_int::LapackConnector::getrf_f(
        nl, nl, body_inv.ptr(), nl, ipiv.data(), info);
    assert(info == 0);
    const int lwork = nl * nl;
    librpa_int::LapackConnector::getri_f(
        nl, body_inv.ptr(), nl, ipiv.data(), work.data(), lwork, info);
    assert(info == 0);

    // wing = sqrtveig[:,1:]^H * wing_mu  (nl x 3)
    Matz wing(nl, 3, MAJOR::COL);
    for (int i = 0; i < nl; ++i)
        for (int j = 0; j < 3; ++j)
        {
            std::complex<double> s = 0.0;
            for (int k = 0; k < n; ++k)
                s += std::conj(sqrtveig(k, i + 1)) * wing_mu(k, j);
            wing(i, j) = s;
        }

    auto Lind = head - wing.get_transpose(true) * body_inv * wing;
    auto bw = body_inv * wing;
    auto wb = wing.get_transpose(true) * body_inv;

    const auto L00 = Lind(0, 0), L01 = Lind(0, 1), L02 = Lind(0, 2);
    const auto L10 = Lind(1, 0), L11 = Lind(1, 1), L12 = Lind(1, 2);
    const auto L20 = Lind(2, 0), L21 = Lind(2, 1), L22 = Lind(2, 2);

    const int nleb = static_cast<int>(q_pts.size());
    Matz eps_inv_coul(n, n, MAJOR::COL);
    eps_inv_coul.zero_out();
    for (int ileb = 0; ileb < nleb; ++ileb)
    {
        const double qx = q_pts[ileb][0], qy = q_pts[ileb][1], qz = q_pts[ileb][2];
        const auto qLq = qx * (qx * L00 + qy * L01 + qz * L02) +
                         qy * (qx * L10 + qy * L11 + qz * L12) +
                         qz * (qx * L20 + qy * L21 + qz * L22);
        const auto w = q_rho[ileb] / qLq;
        eps_inv_coul(0, 0) += w;
        for (int i = 1; i < n_nonsingular; ++i)
            for (int j = 1; j < n_nonsingular; ++j)
            {
                const auto bwq = bw(i - 1, 0) * qx + bw(i - 1, 1) * qy + bw(i - 1, 2) * qz;
                const auto qwb = qx * wb(0, j - 1) + qy * wb(1, j - 1) + qz * wb(2, j - 1);
                eps_inv_coul(i, j) += w * bwq * qwb;
            }
    }
    for (int i = 1; i < n_nonsingular; ++i)
        for (int j = 1; j < n_nonsingular; ++j)
            eps_inv_coul(i, j) += body_inv(i - 1, j - 1);

    return U * eps_inv_coul * U.get_transpose(true);
}

// Run the production ABF-space helper on distributed matrices and compare to
// the dense old Coulomb-basis reference. Returns the observed max abs error.
double run_abf_case(const BlacsCtxtHandler &blacs_h, int n, int block_size,
                    const Matz &U, const Matz &sqrtV,
                    const Matz &chi0, const Matz &wing_mu,
                    const Matz &head,
                    const std::vector<std::array<double, 3>> &q_pts,
                    const std::vector<double> &q_rho, bool use_cholesky, double tol,
                    int n_nonsingular = -1)
{
    if (n_nonsingular < 0) n_nonsingular = n;
    const auto ref = coulomb_basis_eps_inv_reference(U, sqrtV, chi0, wing_mu, head,
                                                     q_pts, q_rho, n_nonsingular);
    // E = I - sqrt(V) * chi0 * sqrt(V)
    auto E = sqrtV * chi0 * sqrtV;
    E *= -1.0;
    for (int i = 0; i < n; ++i) E(i, i) += 1.0;

    ArrayDesc desc(blacs_h);
    desc.init(n, n, block_size, block_size, 0, 0);
    auto E_dist = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    auto sqrtV_dist = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    auto U_dist = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    fill_distributed_matrix(E_dist, desc, E);
    fill_distributed_matrix(sqrtV_dist, desc, sqrtV);
    fill_distributed_matrix(U_dist, desc, U);

    std::vector<double> qx(q_pts.size()), qy(q_pts.size()), qz(q_pts.size());
    for (std::size_t i = 0; i < q_pts.size(); ++i)
    {
        qx[i] = q_pts[i][0];
        qy[i] = q_pts[i][1];
        qz[i] = q_pts[i][2];
    }

    librpa_int::rewrite_eps_abf_space(E_dist, sqrtV_dist, U_dist, head, wing_mu, qx, qy,
                                      qz, q_rho, desc, blacs_h,
                                      static_cast<std::size_t>(n_nonsingular), 0.0,
                                      use_cholesky, false);

    double local_max_err = 0.0;
    for (int ilo = 0; ilo != desc.m_loc(); ++ilo)
    {
        const int ig = desc.indx_l2g_r(ilo);
        for (int jlo = 0; jlo != desc.n_loc(); ++jlo)
        {
            const int jg = desc.indx_l2g_c(jlo);
            local_max_err =
                std::max(local_max_err, std::abs(E_dist(ilo, jlo) - ref(ig, jg)));
        }
    }
    double max_err = 0.0;
    MPI_Allreduce(&local_max_err, &max_err, 1, MPI_DOUBLE, MPI_MAX, desc.comm());
    require_double_close(max_err, 0.0, tol);
    return max_err;
}

void test_abf_space_wing_rewrite_matches_coulomb_basis(const BlacsCtxtHandler &blacs_h)
{
    // 3D quadrature: 6 points on the unit sphere.
    const std::vector<std::array<double, 3>> q3d{
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    std::vector<double> rho3d(6);
    for (auto &r : rho3d) r = (4.0 * M_PI / 6.0) / 3.0;

    // 2D quadrature: 4 points on the unit circle.
    const std::vector<std::array<double, 3>> q2d{
        {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    std::vector<double> rho2d(4);
    for (auto &r : rho2d) r = (2.0 * M_PI / 4.0) / 2.0;

    constexpr int n = 4;
    const double half = 0.5;
    const Matz U(
        {{{half, 0.0}, {half, 0.0}, {half, 0.0}, {half, 0.0}},
         {{half, 0.0}, {-half, 0.0}, {half, 0.0}, {-half, 0.0}},
         {{half, 0.0}, {half, 0.0}, {-half, 0.0}, {-half, 0.0}},
         {{half, 0.0}, {-half, 0.0}, {-half, 0.0}, {half, 0.0}}},
        MAJOR::COL);
    const std::array<double, n> lambda{{9.0, 4.0, 1.0, 0.25}};
    Matz sqrtveig(n, n, MAJOR::COL);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            sqrtveig(i, j) = U(i, j) * std::sqrt(lambda[static_cast<size_t>(j)]);
    const auto sqrtV = sqrtveig * U.get_transpose(true);

    // Negative-semidefinite chi0 = -v v^H so E (and hence M) is Hermitian
    // positive definite and Cholesky is valid.
    Matz chi0(n, n, MAJOR::COL);
    {
        const std::array<std::complex<double>, n> v{{
            {0.20, 0.0}, {0.10, 0.05}, {-0.15, 0.0}, {0.05, -0.10}}};
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                chi0(i, j) = -v[static_cast<size_t>(i)] * std::conj(v[static_cast<size_t>(j)]);
    }

    const Matz wing_mu(
        {{{0.11, 0.01}, {0.12, 0.02}, {0.13, 0.03}},
         {{0.21, 0.04}, {0.22, 0.05}, {0.23, 0.06}},
         {{0.31, 0.07}, {0.32, 0.08}, {0.33, 0.09}},
         {{0.41, 0.10}, {0.42, 0.11}, {0.43, 0.12}}},
        MAJOR::COL);
    Matz zero_wing_mu(n, 3, MAJOR::COL);
    zero_wing_mu.zero_out();

    const Matz head(
        {{{2.0, 0.0}, {0.2, 0.1}, {0.3, -0.1}},
         {{0.2, -0.1}, {2.2, 0.0}, {0.4, 0.2}},
         {{0.3, 0.1}, {0.4, -0.2}, {2.4, 0.0}}},
        MAJOR::COL);

    // LU and Cholesky, 3D and 2D, nonzero and zero wing (square grid).
    run_abf_case(blacs_h, n, 1, U, sqrtV, chi0, wing_mu, head, q3d, rho3d, false, 1e-10);
    run_abf_case(blacs_h, n, 2, U, sqrtV, chi0, wing_mu, head, q3d, rho3d, true, 1e-10);
    run_abf_case(blacs_h, n, 1, U, sqrtV, chi0, wing_mu, head, q2d, rho2d, false, 1e-10);
    run_abf_case(blacs_h, n, 2, U, sqrtV, chi0, wing_mu, head, q2d, rho2d, true, 1e-10);
    run_abf_case(blacs_h, n, 1, U, sqrtV, chi0, zero_wing_mu, head, q3d, rho3d, false, 1e-10);
    run_abf_case(blacs_h, n, 2, U, sqrtV, chi0, zero_wing_mu, head, q3d, rho3d, true, 1e-10);

    // Wing with a deliberately large component parallel to x1 in each
    // Cartesian column. The Direct-Z path (A = D*Z, no explicit projection)
    // must still match the Coulomb-basis reference because D*x1 = 0.
    {
        auto wing_mu_large_x1 = wing_mu.copy();
        for (int alpha = 0; alpha != 3; ++alpha)
            for (int mu = 0; mu != n; ++mu)
                wing_mu_large_x1(mu, alpha) += 100.0 * U(mu, 0);
        run_abf_case(blacs_h, n, 1, U, sqrtV, chi0, wing_mu_large_x1, head, q3d, rho3d,
                     false, 1e-10);
        run_abf_case(blacs_h, n, 2, U, sqrtV, chi0, wing_mu_large_x1, head, q3d, rho3d,
                     true, 1e-10);
    }

    // Rectangular (horizontal) CPU BLACS grid; exercises empty thin local
    // blocks on ranks that own no rows/columns of the thin descriptors.
    {
        BlacsCtxtHandler horizontal_blacs_h(MPI_COMM_WORLD);
        horizontal_blacs_h.init();
        horizontal_blacs_h.set_horizontal_grid();
        run_abf_case(horizontal_blacs_h, n, 1, U, sqrtV, chi0, wing_mu, head, q3d, rho3d,
                     false, 1e-10);
        run_abf_case(horizontal_blacs_h, n, 1, U, sqrtV, chi0, wing_mu, head, q3d, rho3d,
                     true, 1e-10);
    }

    // n_abf == 1 exercises the n-by-1 thin descriptors at the minimal size.
    {
        Matz U1(1, 1, MAJOR::COL);
        U1(0, 0) = {1.0, 0.0};
        Matz sqrtV1(1, 1, MAJOR::COL);
        sqrtV1(0, 0) = {3.0, 0.0};
        Matz chi01(1, 1, MAJOR::COL);
        chi01(0, 0) = {-0.04, 0.0};
        Matz wing_mu1(1, 3, MAJOR::COL);
        wing_mu1(0, 0) = {0.10, 0.01};
        wing_mu1(0, 1) = {0.11, 0.02};
        wing_mu1(0, 2) = {0.12, 0.03};
        run_abf_case(blacs_h, 1, 1, U1, sqrtV1, chi01, wing_mu1, head, q3d, rho3d, false,
                     1e-10);
    }

    // Large orthonormal U with a block-cyclic layout wider than one block, so
    // ranks own multiple non-contiguous eigenvector columns and the thin
    // n_abf x m redistribution destination is genuinely block-cyclic. The
    // 16x16 Sylvester Hadamard matrix scaled by 1/4 has entries +-0.25, which
    // are exactly representable, so U U^H = I holds bit-exactly and the
    // reference comparison is not limited by the input's orthogonality.
    {
        constexpr int nh = 16;
        constexpr int blk = 4;
        Matz U16(nh, nh, MAJOR::COL);
        for (int i = 0; i < nh; ++i)
        {
            for (int j = 0; j < nh; ++j)
            {
                const int ii = i, jj = j;
                int parity = 0;
                for (int b = 0; b < 4; ++b)
                    parity ^= ((ii >> b) & 1) & ((jj >> b) & 1);
                U16(i, j) = std::complex<double>{parity ? -0.25 : 0.25, 0.0};
            }
        }

        const std::array<double, nh> lambda16{
            {64.0, 36.0, 25.0, 16.0, 12.0, 9.0, 7.0, 5.0,
             3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1}};

        // sqrt(V) = U * diag(sqrt(lambda)) * U^H, built with only the retained
        // channels so the filtered ones carry exactly zero Coulomb weight.
        auto build_sqrtV16 = [&](int n_nonsing) {
            Matz sqrtveig(nh, nh, MAJOR::COL);
            sqrtveig.zero_out();
            for (int i = 0; i < nh; ++i)
                for (int j = 0; j < n_nonsing; ++j)
                    sqrtveig(i, j) = U16(i, j) * std::sqrt(lambda16[static_cast<std::size_t>(j)]);
            return sqrtveig * U16.get_transpose(true);
        };

        Matz chi0_16(nh, nh, MAJOR::COL);
        chi0_16.zero_out();
        {
            // Negative-semidefinite chi0 = -v v^H keeps M Hermitian positive
            // definite, so the Cholesky path is valid.
            std::array<std::complex<double>, nh> v{};
            for (int i = 0; i < nh; ++i)
                v[static_cast<std::size_t>(i)] = {0.05 * (i + 1), 0.02 * ((i % 5) - 2)};
            for (int i = 0; i < nh; ++i)
                for (int j = 0; j < nh; ++j)
                    chi0_16(i, j) = -v[static_cast<std::size_t>(i)] *
                                    std::conj(v[static_cast<std::size_t>(j)]);
        }

        Matz wing_mu_16(nh, 3, MAJOR::COL);
        for (int i = 0; i < nh; ++i)
            for (int j = 0; j < 3; ++j)
                wing_mu_16(i, j) = {0.05 + 0.01 * i, 0.02 * (j + 1) - 0.03};

        // m = 3 (n_singular = 2) is the production regime
        // n_nonsingular >> n_singular; m = 11 > n/2 is the regime where the
        // low-rank form is admitted to do more flops than forming T directly.
        const int n_nons_16[] = {14, 6};
        for (int n_nons : n_nons_16)
        {
            const auto sqrtV16 = build_sqrtV16(n_nons);
            run_abf_case(blacs_h, nh, blk, U16, sqrtV16, chi0_16, wing_mu_16, head,
                         q3d, rho3d, false, 1e-10, n_nons);
            run_abf_case(blacs_h, nh, blk, U16, sqrtV16, chi0_16, wing_mu_16, head,
                         q3d, rho3d, true, 1e-10, n_nons);
        }
    }

    // Filtered Coulomb basis: the last eigenchannel has zero eigenvalue and is
    // excluded. Both inversion paths must agree with the old reduced Coulomb-
    // basis algorithm and produce no component in the filtered subspace.
    {
        constexpr int nr = n - 1;
        Matz sqrtveig_filtered(n, n, MAJOR::COL);
        sqrtveig_filtered.zero_out();
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < nr; ++j)
                sqrtveig_filtered(i, j) =
                    U(i, j) * std::sqrt(lambda[static_cast<std::size_t>(j)]);
        const auto sqrtV_filtered = sqrtveig_filtered * U.get_transpose(true);

        run_abf_case(blacs_h, n, 1, U, sqrtV_filtered, chi0, wing_mu, head,
                     q3d, rho3d, false, 1e-10, nr);
        run_abf_case(blacs_h, n, 2, U, sqrtV_filtered, chi0, wing_mu, head,
                     q3d, rho3d, true, 1e-10, nr);
        run_abf_case(blacs_h, n, 1, U, sqrtV_filtered, chi0, zero_wing_mu, head,
                     q2d, rho2d, false, 1e-10, nr);

        // Sensitivity to a non-Hermitian E. The rewrite contracts E only as
        // E*Y, so it is exact for Hermitian E and accurate to O(|E - E^H|)
        // otherwise: the result differs from the dense reference by 2*P*Delta
        // with Delta = (E - E^H)/2, so an anti-Hermitian perturbation of size d
        // must move the answer by O(d) -- here ~1e-12, far under the 1e-10
        // tolerance. This pins that scaling: a change that amplifies it (for
        // example reusing one product for two genuinely different operands)
        // fails here instead of silently biasing production results.
        {
            auto chi0_nonherm = chi0.copy();
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    chi0_nonherm(i, j) += std::complex<double>{1e-12 * (i - j), 0.0};
            const double err = run_abf_case(blacs_h, n, 1, U, sqrtV_filtered, chi0_nonherm,
                                            wing_mu, head, q3d, rho3d, false, 1e-10, nr);
            if (blacs_h.myid == 0)
                std::cout << "non-Hermitian E sensitivity (d=1e-12): max abs err = "
                          << err << std::endl;
        }

        // Retained head channel only: T = 0, M = I, D = 0, so the averaged
        // inverse reduces to a0*P. Exercises the k == 0 projector guard.
        {
            Matz sqrtveig_head_only(n, n, MAJOR::COL);
            sqrtveig_head_only.zero_out();
            for (int i = 0; i < n; ++i)
                sqrtveig_head_only(i, 0) = U(i, 0) * std::sqrt(lambda[0]);
            const auto sqrtV_head_only = sqrtveig_head_only * U.get_transpose(true);
            run_abf_case(blacs_h, n, 1, U, sqrtV_head_only, chi0, wing_mu, head,
                         q3d, rho3d, false, 1e-10, 1);
            run_abf_case(blacs_h, n, 2, U, sqrtV_head_only, chi0, wing_mu, head,
                         q3d, rho3d, true, 1e-10, 1);
        }
    }

    // Invariance: with E and sqrt(V) held fixed, changing Coulomb eigenvector
    // columns other than x1 must not change the ABF-space result.
    {
        auto U_mod = U.copy();
        for (int col = 1; col < n; ++col)
            for (int row = 0; row < n; ++row)
                U_mod(row, col) *= std::complex<double>{0.0, 2.0};

        auto E = sqrtV * chi0 * sqrtV;
        E *= -1.0;
        for (int i = 0; i < n; ++i) E(i, i) += 1.0;

        ArrayDesc desc(blacs_h);
        desc.init(n, n, 1, 1, 0, 0);
        auto E1 = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
        auto E2 = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
        auto sqrtV_dist = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
        auto U_orig_dist = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
        auto U_mod_dist = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
        fill_distributed_matrix(E1, desc, E);
        fill_distributed_matrix(E2, desc, E);
        fill_distributed_matrix(sqrtV_dist, desc, sqrtV);
        fill_distributed_matrix(U_orig_dist, desc, U);
        fill_distributed_matrix(U_mod_dist, desc, U_mod);

        std::vector<double> qx(6), qy(6), qz(6);
        for (std::size_t i = 0; i < q3d.size(); ++i)
        {
            qx[i] = q3d[i][0];
            qy[i] = q3d[i][1];
            qz[i] = q3d[i][2];
        }

        librpa_int::rewrite_eps_abf_space(E1, sqrtV_dist, U_orig_dist, head, wing_mu, qx, qy,
                                          qz, rho3d, desc, blacs_h,
                                          static_cast<std::size_t>(n), 0.0, false, false);
        librpa_int::rewrite_eps_abf_space(E2, sqrtV_dist, U_mod_dist, head, wing_mu, qx, qy,
                                          qz, rho3d, desc, blacs_h,
                                          static_cast<std::size_t>(n), 0.0, false, false);

        double local_max_err = 0.0;
        for (int ilo = 0; ilo != desc.m_loc(); ++ilo)
            for (int jlo = 0; jlo != desc.n_loc(); ++jlo)
                local_max_err =
                    std::max(local_max_err, std::abs(E1(ilo, jlo) - E2(ilo, jlo)));
        double max_err = 0.0;
        MPI_Allreduce(&local_max_err, &max_err, 1, MPI_DOUBLE, MPI_MAX, desc.comm());
        require_double_close(max_err, 0.0, 1e-10);
    }
}
}  // namespace

int main(int argc, char *argv[])
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    librpa_int::global::init_global_mpi(MPI_COMM_WORLD);
    librpa_int::global::init_global_io(false, "stdout", false);

    {
        BlacsCtxtHandler blacs_h(MPI_COMM_WORLD);
        blacs_h.init();
        blacs_h.set_square_grid();

        test_headwing_body_inverse_uses_identity_solve(blacs_h);
        test_replace_rpa_response_headwing_replaces_only_singular_channels(blacs_h);
        test_gamma_head_rank_one_matches_coulomb_basis_overwrite(blacs_h);
        test_gamma_head_rank_one_handles_empty_local_blocks(blacs_h);
        test_rspace_symmetry_requires_complete_band_space();
        test_kpoint_coordinate_mapping_selects_active_klist_from_full_source();
        test_strict_2d_qmember_diagnostic_selects_one_periodic_member();
        test_kstar_velocity_mapping_preserves_member_order_and_periodic_gauge();
        test_replace_rpa_response_head_only_keeps_numeric_wings(blacs_h);
        test_rpa_trace_log_average_uses_directional_head_and_wing();
        test_rpa_headwing_regular_body_start_channel();
        test_rpa_headwing_gamma_cell_volume_uses_reciprocal_lattice();
        test_strict_2d_headwing_prefactors_use_inplane_area();
        test_strict_2d_auxiliary_normalization_is_computed_from_basis_metadata();
        test_strict_2d_gamma_cell_uses_physical_reciprocal_measure();
        test_strict_2d_radial_integrals_match_analytic_values();
        test_strict_2d_radial_integrals_are_stable_at_zero_and_small_a();
        test_strict_2d_inverse_head_average_has_linear_q_screening();
        test_strict_2d_finite_q_reference_matches_head_and_schur_limits();
        test_strict_2d_schur_coefficient_removes_identity();
        test_strict_2d_screening_denominator_must_stay_on_physical_branch();
        test_strict_2d_gw_uses_full_coulomb_at_all_q();
        test_strict_2d_gw_routes_gamma_through_complete_wc_average();
        test_strict_2d_gw_fails_closed_for_incomplete_runtime_configuration();
        test_strict_2d_diagnostic_schema_and_qpoint_order_are_stable();
        test_strict_2d_block_metrics_separate_head_wings_and_body();
        test_strict_2d_alpha_reference_averages_bare_coulomb();
        test_strict_2d_pw_wc_transforms_to_auxiliary_coulomb_basis();
        test_strict_2d_regular_coulomb_legs_are_projected_to_the_gamma_basis();
        test_strict_2d_wc_blocks_match_dense_finite_q_inverse();
        test_strict_2d_wc_cell_average_matches_anisotropic_radial_quadrature();
        test_strict_2d_wc_cell_average_matches_cartesian_voronoi_subgrid();
        test_strict_2d_wc_blocks_have_finite_small_q_limits();
        test_strict_2d_wc_average_is_covariant_under_regular_body_rotation();
        test_strict_2d_wc_average_is_bounded_as_gamma_cell_shrinks();
        test_rpa_chi0v_wing_desc_uses_global_rows(blacs_h);
        test_headwing_spin_weights();
        test_wing_cartesian_gram_is_invariant_under_row_phases();
        test_velocity_matrix_initialization();
        test_headwing_local_kpoints_prefers_kpoint_blacs_context();
        test_headwing_world_fourier_uses_all_R_blocks_at_nonzero_k();
        test_headwing_symmetry_fourier_target_ids_are_deterministic();
        test_headwing_ijk_redistribution_is_owner_group_local();
        test_accumulate_wing_mu_for_pair_matches_original_formula();
        test_headwing_wfc_restore_applies_atom_permutation();
        test_headwing_wfc_restore_applies_time_reversal();
        test_headwing_velocity_restore_uses_inverse_spatial_route();
        test_headwing_direct_full_bz_velocity_selects_kstar_member();
        test_headwing_direct_full_bz_wfc_selects_same_kstar_member();
        test_headwing_direct_full_bz_wfc_local_block(blacs_h);
        test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(blacs_h);
        test_kblacs_transform_matches_original_transform(blacs_h);
        test_kblacs_transform_uses_rectangular_opt_128_blocks();
        test_transform_Cs2mnk_can_keep_spin_channels_separate(blacs_h);
        test_head_initialization_does_not_require_coulomb_diagonalization(blacs_h);
        test_abf_space_wing_rewrite_matches_coulomb_basis(blacs_h);
        test_strict_2d_gamma_quadrature_is_ready_after_wing_initialization(blacs_h);
    }

    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
