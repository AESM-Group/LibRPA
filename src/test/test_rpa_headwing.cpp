#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <valarray>

#include "librpa.h"
#include "../api/instance_manager.h"
#include "../core/chi0.h"
#include "../core/coulmat.h"
#include "../api/dataset.h"
#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../core/meanfield_mpi.h"
#include "../core/qpoint_view.h"
#include "../core/qsgw/headwing_velocity.h"
#include "../gpu/la_connector.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using librpa_int::ArrayDesc;
using librpa_int::AtomicBasis;
using librpa_int::atom_mapping;
using librpa_int::atpair_k_cplx_mat_t;
using librpa_int::atpair_R_mat_t;
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
using librpa_int::SymmetryQPointRestoreMode;
using librpa_int::TFGrids;
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

    // Six entangled spinors form a complete gapped three-AO spinor model.
    // Its resolution of identity, rather than the spatial AO count, fixes
    // which band selection can safely enter real-space restoration.
    MeanField spinor_complete(1, 1, 6, 3, 2);
    auto &wfc = spinor_complete.get_eigenvectors();
    for (int spinor = 0; spinor != 2; ++spinor)
    {
        wfc[0][spinor][0].create(6, 3);
        for (int band = 0; band != 6; ++band)
        {
            spinor_complete.get_eigenvals()[0](0, band) = band < 3 ? -0.5 : 0.5;
            for (int ao = 0; ao != 3; ++ao)
            {
                const double phase = librpa_int::TWO_PI * band * (3 * spinor + ao) / 6.0;
                wfc[0][spinor][0](band, ao) = std::polar(1.0 / std::sqrt(6.0), phase);
            }
        }
    }
    for (int row = 0; row != 6; ++row)
    {
        for (int col = 0; col != 6; ++col)
        {
            std::complex<double> projector = 0.0;
            for (int band = 0; band != 6; ++band)
                projector += wfc[0][row / 3][0](band, row % 3) *
                             std::conj(wfc[0][col / 3][0](band, col % 3));
            assert(std::abs(projector - (row == col ? 1.0 : 0.0)) < 1e-12);
        }
    }
    assert(librpa_int::rspace_symmetry_has_complete_band_space(spinor_complete, -1));
    assert(!librpa_int::rspace_symmetry_has_complete_band_space(spinor_complete, 3));
}

void test_qsgw_reconstruction_with_finite_reference_metric()
{
    using namespace librpa_int::qsgw;
    const Matrix3 lattice(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    const Vector3_Order<int> origin(0, 0, 0);
    const std::vector<Vector3_Order<double>> kpoints{{0.0, 0.0, 0.0}};
    MeanField reference(1, 1, 2, 2);
    auto &wfc = reference.get_eigenvectors()[0][0][0];
    wfc.create(2, 2);
    wfc.zero_out();
    wfc(0, 0) = 1.0 + 1e-7;
    wfc(1, 1) = (1.0 - 1e-7) / std::sqrt(1e-3);
    Matz coefficients(2, 2, MAJOR::ROW);
    coefficients.zero_out();
    coefficients(0, 0) = wfc(0, 0);
    coefficients(1, 1) = wfc(1, 1);
    HeadwingAoOperators operators;
    operators.dimension = 2;
    operators.overlap.emplace(
        origin, Matz({{{1.0, 0.0}, {0.0, 0.0}}, {{0.0, 0.0}, {1e-3, 0.0}}}, MAJOR::ROW));
    const Matz h({{{-0.5, 0.0}, {0.012, 0.008}}, {{0.012, -0.008}, {20.0, 0.0}}}, MAJOR::ROW);
    SpinKMatrixMap target;
    target[0][0] = h.copy();
    const auto result = reconstruct_qsgw_hamiltonian(reference, target, operators, kpoints,
                                                     Vector3_Order<int>(1, 1, 1), lattice);
    assert(result.reference_metric_residual > 1e-7);
    assert(result.reference_metric_residual < 1e-6);
    const Matz projected =
        transpose(coefficients, true) * result.hamiltonian[0].at(origin) * coefficients;
    double error = 0.0;
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) error = std::max(error, std::abs(projected(i, j) - h(i, j)));
    assert(error < 1e-12);
    // The former S C embedding amplifies this accepted metric defect beyond
    // the unchanged 1e-4 eV frame gate, especially in the high-energy band.
    const Matz su = operators.overlap.at(origin) * coefficients;
    const Matz old_projected =
        transpose(coefficients, true) * su * h * transpose(su, true) * coefficients;
    assert(std::abs(old_projected(1, 1) - h(1, 1)) > 1e-4 / librpa_int::HA2EV);
    assert(target[0][0](1, 1) == h(1, 1));
}

void test_qsgw_native_baseline_preserves_aliased_tail()
{
    using namespace librpa_int::qsgw;
    const Matrix3 lattice(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    const std::vector<Vector3_Order<double>> kpoints{
        {0.0, 0.0, 0.0}, {0.25, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.75, 0.0, 0.0}};
    const Vector3_Order<double> probe(0.13, 0.0, 0.0);
    const auto sampled = [](const LocalizedMatrix &model, const Vector3_Order<double> &k)
    {
        Matz result(2, 2, MAJOR::ROW);
        result.zero_out();
        for (const auto &[r, block] : model)
        {
            const auto weight =
                std::polar(1.0, librpa_int::TWO_PI * (k.x * r.x + k.y * r.y + k.z * r.z));
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) result(i, j) += weight * block(i, j);
        }
        return result;
    };
    // Test both distinct collinear channels and complete AO-major spinors.
    for (const int spinors : {1, 2})
    {
        const int spins = spinors == 1 ? 2 : 1;
        MeanField reference(spins, 4, 2, 2 / spinors, spinors);
        HeadwingAoOperators operators;
        operators.dimension = 2;
        operators.overlap.emplace(
            Vector3_Order<int>(0, 0, 0),
            Matz({{{1.2, 0.0}, {0.07, 0.02}}, {{0.07, -0.02}, {0.9, 0.0}}}, MAJOR::ROW));
        if (spinors == 2)
        {
            auto &overlap = operators.overlap.at(Vector3_Order<int>(0, 0, 0));
            overlap(0, 1) = overlap(1, 0) = 0.0;
            overlap(1, 1) = overlap(0, 0);
        }
        std::vector<LocalizedMatrix> native(spins), current(spins);
        SpinKMatrixMap target;
        for (int spin = 0; spin < spins; ++spin)
        {
            native[spin].emplace(Vector3_Order<int>(0, 0, 0),
                                 Matz({{{-0.7 + 0.1 * spin, 0.0}, {0.0, 0.0}},
                                       {{0.0, 0.0}, {0.9 + 0.1 * spin, 0.0}}},
                                      MAJOR::ROW));
            // R=3 aliases to R=-1 on the four-point mesh. Whole-H IFT loses
            // its actual range, while native-baseline reconstruction keeps it.
            Matz tail({{{0.04, 0.015}, {0.018, 0.024}}, {{-0.01, 0.03}, {-0.03, 0.01}}},
                      MAJOR::COL);
            for (std::size_t i = 0; i < tail.size(); ++i) tail.ptr()[i] *= 1.0 + 0.2 * spin;
            native[spin].emplace(Vector3_Order<int>(3, 0, 0), tail);
            native[spin].emplace(Vector3_Order<int>(-3, 0, 0), transpose(tail, true));
            for (const auto &[r, block] : native[spin]) current[spin].emplace(r, block.copy());
            const Matz onsite({{{0.05, 0.0}, {0.012, 0.008}}, {{0.012, -0.008}, {-0.02, 0.0}}},
                              MAJOR::ROW);
            current[spin].at(Vector3_Order<int>(0, 0, 0)) += onsite;
            Matz correction({{{0.01, 0.0}, {0.0, 0.005}}, {{0.0, 0.0}, {-0.006, 0.0}}}, MAJOR::ROW);
            current[spin].emplace(Vector3_Order<int>(1, 0, 0), correction);
            current[spin].emplace(Vector3_Order<int>(-1, 0, 0), transpose(correction, true));
            for (int ik = 0; ik < 4; ++ik)
            {
                const auto solution =
                    solve_localized_velocity(native[spin], operators, kpoints[ik], lattice);
                const auto &coefficients = solution.ket_coefficients;
                for (int band = 0; band < 2; ++band)
                    reference.get_eigenvals()[spin](ik, band) = solution.eigenvalues_ha[band];
                for (int component = 0; component < spinors; ++component)
                {
                    auto &wfc = reference.get_eigenvectors()[spin][component][ik];
                    wfc.create(2, 2 / spinors);
                    for (int band = 0; band < 2; ++band)
                        for (int ao = 0; ao < 2 / spinors; ++ao)
                            wfc(band, ao) = coefficients(ao * spinors + component, band);
                }
                target[spin][ik] = transpose(coefficients, true) *
                                   sampled(current[spin], kpoints[ik]) * coefficients;
            }
        }
        const auto whole = reconstruct_qsgw_hamiltonian(reference, target, operators, kpoints,
                                                        Vector3_Order<int>(4, 1, 1), lattice);
        operators.native_dft_hamiltonian = native;
        auto preserved = reconstruct_qsgw_hamiltonian(reference, target, operators, kpoints,
                                                      Vector3_Order<int>(4, 1, 1), lattice);
        double off_grid_error = 0.0, whole_error = 0.0, derivative_error = 0.0;
        for (int spin = 0; spin < spins; ++spin)
        {
            const auto expected =
                solve_localized_velocity(current[spin], operators, probe, lattice);
            const auto actual =
                solve_localized_velocity(preserved.hamiltonian[spin], operators, probe, lattice);
            const auto aliased =
                solve_localized_velocity(whole.hamiltonian[spin], operators, probe, lattice);
            for (int band = 0; band < 2; ++band)
            {
                off_grid_error = std::max(off_grid_error, std::abs(actual.eigenvalues_ha[band] -
                                                                   expected.eigenvalues_ha[band]));
                whole_error = std::max(whole_error, std::abs(aliased.eigenvalues_ha[band] -
                                                             expected.eigenvalues_ha[band]));
            }
            // Evaluate derivatives in the same complete frame to avoid phase
            // ambiguity in off-diagonal velocity comparisons.
            const auto projected = compute_localized_velocity_in_frame(
                preserved.hamiltonian[spin], operators, probe, lattice, expected.eigenvalues_ha,
                expected.ket_coefficients);
            for (int a = 0; a < 3; ++a)
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        derivative_error = std::max(derivative_error,
                                                    std::abs(projected.velocity_ha_bohr[a](i, j) -
                                                             expected.velocity_ha_bohr[a](i, j)));
            for (int ik = 0; ik < 4; ++ik)
            {
                const Matz grid_difference = sampled(preserved.hamiltonian[spin], kpoints[ik]) -
                                             sampled(current[spin], kpoints[ik]);
                for (std::size_t entry = 0; entry < grid_difference.size(); ++entry)
                    if (std::abs(grid_difference.ptr()[entry]) >= 1e-12) std::abort();
            }
            // The output must not share mutable storage with the baseline.
            const auto original_tail = native[spin].at(Vector3_Order<int>(3, 0, 0))(0, 0);
            preserved.hamiltonian[spin].at(Vector3_Order<int>(3, 0, 0))(0, 0) += 0.125;
            if (operators.native_dft_hamiltonian[spin].at(Vector3_Order<int>(3, 0, 0))(0, 0) !=
                original_tail)
                std::abort();
            for (const auto &[r, block] : native[spin])
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        if (operators.native_dft_hamiltonian[spin].at(r)(i, j) != block(i, j))
                            std::abort();
        }
        if (!std::isfinite(off_grid_error) || !std::isfinite(derivative_error) ||
            off_grid_error >= 1e-12 || derivative_error >= 1e-12 || whole_error <= 1e-3)
            std::abort();
        if (librpa_int::global::mpi_comm_global_h.is_root())
            std::cout << "QSGW native-baseline spinors=" << spinors
                      << " path_E_Ha=" << off_grid_error << " velocity_HaBohr=" << derivative_error
                      << " whole_H_alias_E_Ha=" << whole_error << std::endl;
    }
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

void require_condition(const bool condition, const char *message)
{
    if (!condition)
    {
        std::cerr << message << std::endl;
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

    constexpr double small_qmax = 0.3;
    assert_complex_close(librpa_int::strict_2d_radial_i0(0.0, small_qmax),
                         small_qmax * small_qmax / 2.0, 1e-15);
    assert_complex_close(librpa_int::strict_2d_radial_i1(0.0, small_qmax),
                         small_qmax * small_qmax * small_qmax / 3.0, 1e-15);
    const std::complex<double> small_a{1.0e-10, -2.0e-10};
    const auto small_expected_i0 = small_qmax * small_qmax / 2.0 -
                                   small_a * std::pow(small_qmax, 3) / 3.0;
    const auto small_expected_i1 = std::pow(small_qmax, 3) / 3.0 -
                                   small_a * std::pow(small_qmax, 4) / 4.0;
    assert_complex_close(librpa_int::strict_2d_radial_i0(small_a, small_qmax),
                         small_expected_i0, 1e-15);
    assert_complex_close(librpa_int::strict_2d_radial_i1(small_a, small_qmax),
                         small_expected_i1, 1e-15);
}

void test_strict_2d_radial_log_integral_matches_numeric_quadrature()
{
    constexpr std::complex<double> a{0.37, 0.0};
    constexpr double qmax = 0.61;
    constexpr int n = 200000;
    const double dq = qmax / n;
    std::complex<double> numeric = 0.0;
    for (int i = 0; i != n; ++i)
    {
        const double q = (i + 0.5) * dq;
        numeric += q * std::log(1.0 + a * q) * dq;
    }
    assert_complex_close(librpa_int::strict_2d_radial_log_integral(a, qmax), numeric, 2e-13);
}

void test_strict_2d_radial_log_integral_is_stable_at_zero_and_small_a()
{
    assert_complex_close(librpa_int::strict_2d_radial_log_integral(0.0, 0.7), 0.0, 1e-15);
    constexpr std::complex<double> a{1.0e-10, 0.0};
    constexpr double qmax = 0.7;
    const auto expected = a * std::pow(qmax, 3) / 3.0 - a * a * std::pow(qmax, 4) / 8.0;
    assert_complex_close(librpa_int::strict_2d_radial_log_integral(a, qmax), expected, 1e-24);
}

void test_strict_2d_rpa_trace_log_average_matches_dense_radial_quadrature()
{
    matrix_m<std::complex<double>> head(3, 3, MAJOR::COL);
    matrix_m<std::complex<double>> lind(3, 3, MAJOR::COL);
    head(0, 0) = -0.28;
    head(0, 1) = 0.03;
    head(1, 0) = 0.03;
    head(1, 1) = -0.16;
    lind(0, 0) = 1.41;
    lind(0, 1) = 0.04;
    lind(1, 0) = 0.04;
    lind(1, 1) = 1.23;
    lind(2, 2) = 1.0;

    const std::vector<double> qx{1.0, 0.0, -1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0, 0.0, -1.0};
    const std::vector<double> angular_weights(4, librpa_int::TWO_PI / 4.0);
    const std::vector<double> qmax{0.21, 0.14, 0.21, 0.14};
    double gamma_area = 0.0;
    for (std::size_t i = 0; i != qmax.size(); ++i)
        gamma_area += angular_weights[i] * qmax[i] * qmax[i] / 2.0;

    const std::complex<double> trace_body{0.12, 0.0};
    const std::complex<double> logdet_body = std::log(0.88);
    const auto analytic = librpa_int::compute_strict_2d_rpa_chi0v_trace_log_average(
        head, lind, trace_body, logdet_body, qx, qy, angular_weights, qmax, gamma_area);

    std::complex<double> numeric = 0.0;
    constexpr int radial_points = 200000;
    for (std::size_t idir = 0; idir != qx.size(); ++idir)
    {
        const double nx = qx[idir];
        const double ny = qy[idir];
        const auto directional_head =
            nx * (nx * head(0, 0) + ny * head(0, 1)) + ny * (nx * head(1, 0) + ny * head(1, 1));
        const auto a = librpa_int::strict_2d_schur_coefficient(lind, nx, ny);
        const double dq = qmax[idir] / radial_points;
        for (int ir = 0; ir != radial_points; ++ir)
        {
            const double q = (ir + 0.5) * dq;
            const auto integrand =
                trace_body + logdet_body + directional_head * q + std::log(1.0 + a * q);
            numeric += angular_weights[idir] * q * dq * integrand / gamma_area;
        }
    }
    assert_complex_close(analytic, numeric, 2e-11);
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
    require_condition(librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, true, true),
                  "strict 2D Gamma must use the complete-Wc route");
    require_condition(!librpa_int::use_strict_2d_complete_wc_gamma_route(false, 3, true, true, true),
                  "disabled head/wing replacement must keep the standard route");
    require_condition(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 2, true, true, true),
                  "non-full head/wing dielectric mode must keep the standard route");
    require_condition(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, false, true, true),
                  "3D dielectric calculations must keep the standard route");
    require_condition(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, false, true),
                  "finite q must keep the standard route");
    require_condition(!librpa_int::use_strict_2d_complete_wc_gamma_route(true, 3, true, true, false),
                  "missing head/wing data must keep the standard route");
}

void test_strict_2d_gw_fails_closed_for_incomplete_runtime_configuration()
{
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

void test_rpa_chi0v_wing_desc_matches_producer_layout(const BlacsCtxtHandler &blacs_h)
{
    const auto require = [](const bool condition)
    {
        if (!condition) std::abort();
    };
    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(10, 10, 0, 0);

    ArrayDesc desc_full_wing(blacs_h);
    desc_full_wing.init(11, 3, desc_body.mb(), 1, 0, 0);

    const auto desc_wing = librpa_int::make_rpa_chi0v_wing_desc(
        desc_body, 1, desc_full_wing.m_loc(), desc_full_wing.n_loc());

    if (desc_body.nprows() > 1)
    {
        require(desc_full_wing.m_loc() < desc_full_wing.m());
    }
    require(desc_wing.m() == 11);
    require(desc_wing.n() == 3);
    require(desc_wing.mb() == desc_body.mb());
    require(desc_wing.nb() == 1);
    require(desc_wing.m_loc() == desc_full_wing.m_loc());
    require(desc_wing.n_loc() == desc_full_wing.n_loc());
    require(std::equal(desc_wing.desc, desc_wing.desc + 9, desc_full_wing.desc));
    require(desc_wing.l2g_r() == desc_full_wing.l2g_r());
    require(desc_wing.l2g_c() == desc_full_wing.l2g_c());
}

void test_rpa_headwing_matrix_descriptor_contract(const BlacsCtxtHandler &blacs_h)
{
    const auto require = [](const bool condition)
    {
        if (!condition) std::abort();
    };
    ArrayDesc storage_desc(blacs_h);
    storage_desc.init(8, 8, 2, 2, 0, 0);
    const auto matrix = init_local_mat<std::complex<double>>(storage_desc, MAJOR::COL);

    ArrayDesc same_desc(blacs_h);
    same_desc.init(8, 8, 2, 2, 0, 0);
    require(librpa_int::rpa_headwing_matrix_matches_descriptor(matrix, storage_desc, same_desc));

    ArrayDesc wrong_desc(blacs_h);
    wrong_desc.init(8, 8, 4, 4, 0, 0);
    require(!librpa_int::rpa_headwing_matrix_matches_descriptor(matrix, storage_desc, wrong_desc));
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

    // C_BZ = C_IBZ * R, with R[atom_to, atom_from] = rotation(atom_from).
    // The independent Si density fixture tests this placement against producer data.
    assert_complex_close(wfc_bz(0, 0), wfc_ibz(0, 1) * std::complex<double>{2.0, 0.5}, 1e-12);
    assert_complex_close(wfc_bz(0, 1), wfc_ibz(0, 0) * std::complex<double>{3.0, -0.25}, 1e-12);
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

// Complete nonorthogonal SOC frames. The reference below acts directly on
// AO-major ket columns; it does not call a production symmetry transform.
std::pair<ComplexMatrix, ComplexMatrix> make_ibz_soc_metric_frame(const double phase)
{
    ComplexMatrix metric(4, 4), frame(4, 4);
    const double a = 1.2, d = 0.9;
    const std::complex<double> b{0.17, 0.11};
    for (int spinor = 0; spinor != 2; ++spinor)
    {
        metric(spinor, spinor) = a * a;
        metric(spinor, 2 + spinor) = a * b;
        metric(2 + spinor, spinor) = a * std::conj(b);
        metric(2 + spinor, 2 + spinor) = std::norm(b) + d * d;
        for (int band = 0; band != 4; ++band)
        {
            const auto q0 = std::polar(
                0.5, librpa_int::TWO_PI * spinor * band / 4.0 + phase * spinor + 0.13 * band);
            const auto q1 = std::polar(0.5, librpa_int::TWO_PI * (2 + spinor) * band / 4.0 +
                                                phase * (2 + spinor) + 0.13 * band);
            frame(spinor, band) = (q0 - b * q1 / d) / a;
            frame(2 + spinor, band) = q1 / d;
        }
    }
    return {metric, frame};
}

ComplexMatrix ibz_soc_degenerate_gauge()
{
    ComplexMatrix gauge(4, 4);
    for (int pair = 0; pair != 2; ++pair)
    {
        const double c = std::cos(0.43 + 0.21 * pair);
        const auto s = std::polar(std::sin(0.43 + 0.21 * pair), 0.37 + 0.19 * pair);
        gauge(2 * pair, 2 * pair) = c;
        gauge(2 * pair, 2 * pair + 1) = s;
        gauge(2 * pair + 1, 2 * pair) = -std::conj(s);
        gauge(2 * pair + 1, 2 * pair + 1) = c;
    }
    return gauge;
}

std::array<ComplexMatrix, 2> ibz_soc_frame_to_wfc(const ComplexMatrix &frame)
{
    std::array<ComplexMatrix, 2> result{ComplexMatrix(4, 2), ComplexMatrix(4, 2)};
    for (int ao = 0; ao != 2; ++ao)
        for (int spinor = 0; spinor != 2; ++spinor)
            for (int band = 0; band != 4; ++band)
                result[spinor](band, ao) = frame(2 * ao + spinor, band);
    return result;
}

std::array<ComplexMatrix, 3> ibz_soc_band_velocity(const double shift)
{
    std::array<ComplexMatrix, 3> result;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        result[alpha].create(4, 4);
        for (int row = 0; row != 4; ++row)
        {
            result[alpha](row, row) = 0.02 * (1 + row + alpha);
            for (int col = row + 1; col != 4; ++col)
            {
                const auto value = std::polar(0.08 * (1 + alpha + row) + shift,
                                              0.27 * row - 0.39 * col + 0.21 * alpha);
                result[alpha](row, col) = value;
                result[alpha](col, row) = std::conj(value);
            }
        }
    }
    return result;
}

void test_ibz_soc_wfc_velocity_matches_nonorthogonal_ao_oracle()
{
    SymmetryContext ctx;
    ctx.set_available();
    ctx.lattice_vectors.Identity();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.13, 0.2, 0.0}}, {1, {0.37, 0.4, 0.0}}};
    ctx.basis_convention.bloch_phase = 1;
    ctx.basis_convention.bloch_ratom = 1;
    SymmetryOperation operation;
    // Cubic C3 about (1,1,1), stored in the full -> representative direction.
    operation.rotation = Matrix3(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    operation.use_row_convention = false;
    ctx.rspace_operations = {operation};
    const std::vector<SpeciesBasisLayout> layouts{SpeciesBasisLayout("X", {0})};
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const auto p0 = std::polar(1.0, 0.31), p1 = std::polar(1.0, -0.52);
    // The SOC-locked inverse lift exp(+i pi sigma.(1,1,1)/3/sqrt(3)).
    const std::array<std::complex<double>, 4> u{
        std::complex<double>{0.5, 0.5}, std::complex<double>{0.5, 0.5},
        std::complex<double>{-0.5, 0.5}, std::complex<double>{0.5, -0.5}};
    const std::array<std::complex<double>, 4> inverse{std::conj(u[0]), std::conj(u[2]),
                                                      std::conj(u[1]), std::conj(u[3])};
    const Vector3_Order<double> representative{0.0, 0.0, 0.0}, target{1.0, 0.0, 0.0};
    const auto source = make_ibz_soc_metric_frame(0.29);
    const auto &frame = source.second;
    double maximum_error = 0.0, wrong_spin_witness = 0.0;
    for (const bool antiunitary : {false, true})
    {
        ctx.spin_operations = {
            {0, inverse, antiunitary, librpa_int::SymmetrySpinActionSource::DerivedFromSpatialSOC}};
        ctx.kspace_actions = {{0, {0}}};
        auto member = make_headwing_wfc_atom_swap_member(p0, p1);
        member.action_id = 0;
        member.time_reversal = antiunitary;
        member.k_bz = representative;
        // Dense source-to-target coefficient action, independently assembled.
        ComplexMatrix action(4, 4);
        for (int ao = 0; ao != 2; ++ao)
            for (int spinor = 0; spinor != 2; ++spinor)
                for (int source_spinor = 0; source_spinor != 2; ++source_spinor)
                {
                    auto value = (ao == 0 ? p0 : p1) * u[2 * spinor + source_spinor];
                    int target_spinor = spinor;
                    if (antiunitary)
                    {
                        value = std::conj(value);
                        target_spinor = 1 - spinor;
                        if (target_spinor == 1) value *= -1.0;
                    }
                    value *= std::polar(1.0, -librpa_int::TWO_PI * (ao == 0 ? 0.13 : 0.37));
                    action(2 * ao + target_spinor, 2 * (1 - ao) + source_spinor) = value;
                }
        const auto metric_target =
            action * (antiunitary ? conj(source.first) : source.first) * transpose(action, true);
        const auto expected = action * (antiunitary ? conj(frame) : frame);
        const auto source_wfc = ibz_soc_frame_to_wfc(frame);
        const auto actual = librpa_int::rotate_headwing_spinor_wfc_to_kstar_member(
            ctx, member, layouts, atom_nw, representative, source_wfc, &target);
        for (int band = 0; band != 4; ++band)
            for (int ao = 0; ao != 2; ++ao)
                for (int spinor = 0; spinor != 2; ++spinor)
                    maximum_error = std::max(
                        maximum_error,
                        std::abs(actual[spinor](band, ao) - expected(2 * ao + spinor, band)));
        const auto metric = transpose(expected, true) * metric_target * expected;
        for (int row = 0; row != 4; ++row)
            for (int col = 0; col != 4; ++col)
            {
                maximum_error =
                    std::max(maximum_error, std::abs(metric(row, col) - (row == col ? 1.0 : 0.0)));
                wrong_spin_witness =
                    std::max(wrong_spin_witness, std::abs(expected(row, col) - frame(row, col)));
            }

        const auto velocity = ibz_soc_band_velocity(0.01);
        const auto restored_velocity = librpa_int::rotate_headwing_velocity_to_kstar_member(
            ctx, member, velocity, 4, antiunitary);
        const std::array<int, 3> input_axis{1, 2, 0};
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            const auto source_operator = source.first * frame * velocity[input_axis[alpha]] *
                                         transpose(frame, true) * source.first;
            auto target_operator = action *
                                   (antiunitary ? conj(source_operator) : source_operator) *
                                   transpose(action, true);
            target_operator *= antiunitary ? -1.0 : 1.0;
            const auto projected = transpose(expected, true) * target_operator * expected;
            for (int row = 0; row != 4; ++row)
                for (int col = 0; col != 4; ++col)
                    maximum_error = std::max(
                        maximum_error,
                        std::abs(projected(row, col) - restored_velocity[alpha](row, col)));
        }
    }
    if (librpa_int::global::mpi_comm_global_h.is_root())
        std::cout << "IBZ SOC complex WFC/velocity/metric oracle error: " << maximum_error
                  << "; nontrivial spin/gauge witness: " << wrong_spin_witness << std::endl;
    assert(maximum_error < 2e-12);
    assert(wrong_spin_witness > 0.1);
}

RI::Tensor<double> make_single_value_tensor(double value);

void test_ibz_soc_headwing_matches_complete_complex_bz(const BlacsCtxtHandler &blacs_h)
{
#ifdef LIBRPA_USE_LIBRI
    constexpr bool test_wing = true;
#else
    // The analytic wing needs LibRI tensor-map communication. Keep the same
    // nonzero head, unequal-star occupations and complex-gauge oracle active
    // without that backend; its communication stubs cannot supply a wing.
    constexpr bool test_wing = false;
#endif
    const std::vector<double> lattice{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    PeriodicBoundaryData full_pbc, ibz_pbc;
    full_pbc.set_latvec(lattice);
    full_pbc.set_kgrids_kvec(1, 3, 1,
                             {0.0, 0.0, 0.0, 0.0, librpa_int::TWO_PI / 3.0, 0.0, 0.0,
                              2.0 * librpa_int::TWO_PI / 3.0, 0.0});
    ibz_pbc.set_latvec(lattice);
    ibz_pbc.set_irreducible_kgrids_kvec(
        1, 3, 1, {0.0, 0.0, 0.0, 0.0, librpa_int::TWO_PI / 3.0, 0.0},
        {{{0.0, 0.0, 0.0}}, {{0.0, 1.0 / 3.0, 0.0}, {0.0, 2.0 / 3.0, 0.0}}});
    SymmetryContext ctx;
    ctx.set_available();
    ctx.lattice_vectors.Identity();
    ctx.atom_to_type = {{0, 0}, {1, 0}};
    ctx.input_coord_frac = {{0, {0.0, 0.0, 0.0}}, {1, {0.0, 0.25, 0.0}}};
    ctx.basis_convention.bloch_phase = 1;
    ctx.basis_convention.bloch_ratom = 1;
    SymmetryOperation identity;
    identity.rotation.Identity();
    ctx.rspace_operations = {identity};
    const std::array<std::complex<double>, 4> spin_identity{1.0, 0.0, 0.0, 1.0};
    ctx.spin_operations = {
        {0, spin_identity, false, librpa_int::SymmetrySpinActionSource::DerivedFromSpatialSOC},
        {0, spin_identity, true, librpa_int::SymmetrySpinActionSource::DerivedFromSpatialSOC}};
    ctx.kspace_actions = {{0, {0}}, {1, {1}}};
    const auto make_member = [](const Vector3_Order<double> &k, const bool anti)
    {
        SymmetryKStarMember member;
        member.spatial_isym = 0;
        member.action_id = anti ? 1 : 0;
        member.time_reversal = anti;
        member.k_bz = k;
        for (int iatom = 0; iatom != 2; ++iatom)
        {
            SymmetryKAtomRotation atom;
            atom.atom_from = atom.atom_to = iatom;
            atom.atom_type = 0;
            atom.lmax = 0;
            atom.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
            atom.bloch_rsh_rotations[0](0, 0) = 1.0;
            member.atom_rotations.push_back(atom);
        }
        return member;
    };
    librpa_int::SymmetryKStar gamma, pair;
    gamma.star_index = 0;
    gamma.k_ibz = {0.0, 0.0, 0.0};
    gamma.members = {make_member(gamma.k_ibz, false)};
    pair.star_index = 1;
    pair.k_ibz = {0.0, 1.0 / 3.0, 0.0};
    pair.members = {make_member(pair.k_ibz, false), make_member({0.0, -1.0 / 3.0, 0.0}, true)};
    ctx.kstars = {gamma, pair};
    ctx.kstar_grid_mapping = {{0, 0, {{0.0, 0.0, 0.0}}},
                              {1, 1, {{0.0, 1.0 / 3.0, 0.0}, {0.0, 2.0 / 3.0, 0.0}}}};

    AtomicBasis basis_wfc(std::vector<size_t>{1, 1});
    AtomicBasis basis_abf(std::vector<size_t>{1, 1});
    basis_wfc.set_l_shells({{0}, {0}});
    basis_abf.set_l_shells({{0}, {0}});
    librpa_int::Cs_LRI cs;
    cs.use_libri = true;
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        cs.data_libri[0][{0, {0, 0, 0}}] = make_single_value_tensor(0.8);
        cs.data_libri[0][{1, {0, 0, 0}}] = make_single_value_tensor(0.19);
        cs.data_libri[0][{1, {0, 1, 0}}] = make_single_value_tensor(-0.11);
        cs.data_libri[1][{0, {0, 0, 0}}] = make_single_value_tensor(-0.27);
        cs.data_libri[1][{1, {0, 0, 0}}] = make_single_value_tensor(0.63);
    }
    const std::vector<double> frequencies{0.17, 0.61, 1.3};
    const atpair_k_cplx_mat_t empty_vq;
    enum class InputRoute
    {
        SourceOnly,
        StarWeighted,
        FullCache
    };
    ArrayDesc sqrt_desc(blacs_h);
    sqrt_desc.init_1b1p(2, 2, 0, 0);
    auto sqrt_v = init_local_mat<std::complex<double>>(sqrt_desc, MAJOR::COL);
    fill_distributed_matrix(sqrt_v, sqrt_desc, {{1.0, 0.0}, {0.0, std::sqrt(0.5)}});
    const auto gauge = ibz_soc_degenerate_gauge();
    std::vector<Matz> baseline_head, baseline_wing;
    double maximum_error = 0.0, head_magnitude = 0.0, wing_magnitude = 0.0;
    double regular_wing_magnitude = 0.0;
    for (const bool change_degenerate_gauge : {false, true})
    {
        MeanField full_mf(1, 3, 4, 2, 2), ibz_mf(1, 2, 4, 2, 2);
        librpa_int::velocity_matrix_t full_velocity, ibz_velocity;
        librpa_int::initialize_velocity_matrix(full_velocity, 1, 3, 4);
        librpa_int::initialize_velocity_matrix(ibz_velocity, 1, 2, 4);
        for (int source_ik = 0; source_ik != 2; ++source_ik)
        {
            const auto source = make_ibz_soc_metric_frame(0.29 + 0.17 * source_ik);
            const auto frame = change_degenerate_gauge ? source.second * gauge : source.second;
            auto velocity = ibz_soc_band_velocity(0.01 + 0.015 * source_ik);
            if (change_degenerate_gauge)
                for (auto &component : velocity)
                    component = transpose(gauge, true) * component * gauge;
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                ibz_velocity[0][source_ik][alpha] = velocity[alpha];
                full_velocity[0][source_ik][alpha] = velocity[alpha];
            }
            const auto wfc = ibz_soc_frame_to_wfc(frame);
            for (int spinor = 0; spinor != 2; ++spinor)
            {
                // Different whole components live on different ranks. The
                // reconstruction must gather the paired source frame first.
                const int owner = (source_ik + spinor) % librpa_int::get_mpi_size(MPI_COMM_WORLD);
                if (librpa_int::global::mpi_comm_global_h.myid == owner)
                    ibz_mf.get_eigenvectors()[0][spinor][source_ik] = wfc[spinor];
                full_mf.get_eigenvectors()[0][spinor][source_ik] = wfc[spinor];
            }
            for (int band = 0; band != 4; ++band)
            {
                const double energy = band < 2 ? -0.7 - 0.03 * source_ik : 0.9 + 0.05 * source_ik;
                const double occupation = band < 2 ? 1.0 : 0.0;
                ibz_mf.get_eigenvals()[0](source_ik, band) = energy;
                ibz_mf.get_weight()[0](source_ik, band) = occupation / 2.0;
                full_mf.get_eigenvals()[0](source_ik, band) = energy;
                full_mf.get_weight()[0](source_ik, band) = occupation / 3.0;
                if (source_ik != 1) continue;
                full_mf.get_eigenvals()[0](2, band) = energy;
                full_mf.get_weight()[0](2, band) = occupation / 3.0;
            }
            if (source_ik != 1) continue;
            // Independent target frame: Theta followed by reciprocal-basis
            // re-gauging from -1/3 to 2/3. No production helper generates it.
            for (int spinor = 0; spinor != 2; ++spinor)
            {
                auto &target = full_mf.get_eigenvectors()[0][spinor][2];
                target.create(4, 2);
                for (int ao = 0; ao != 2; ++ao)
                    for (int band = 0; band != 4; ++band)
                        target(band, ao) =
                            (spinor == 0 ? 1.0 : -1.0) *
                            std::polar(1.0, -librpa_int::TWO_PI * (ao == 0 ? 0.0 : 0.25)) *
                            std::conj(frame(2 * ao + 1 - spinor, band));
            }
            for (int alpha = 0; alpha != 3; ++alpha)
                full_velocity[0][2][alpha] =
                    std::complex<double>{-1.0, 0.0} * conj(velocity[alpha]);
        }
        // The main SCF mean field retains geometric star weights times f;
        // the separate head/wing eigenframe uses f/N_IBZ. Exercise this
        // boundary with unequal star sizes while preserving both inputs.
        MeanField star_weighted_mf = ibz_mf;
        for (int source_ik = 0; source_ik != 2; ++source_ik)
            for (int band = 0; band != 4; ++band)
                star_weighted_mf.get_weight()[0](source_ik, band) =
                    band < 2 ? static_cast<double>(source_ik + 1) / 3.0 : 0.0;
        diele_func full(full_mf, full_velocity, full_pbc.kfrac_list, basis_wfc, basis_abf,
                        frequencies, 2, 4, 1, 2, full_pbc, librpa_int::global::mpi_comm_global_h,
                        blacs_h);
        full.use_soc = true;
        full.init(0.0, empty_vq);
        full.cal_head();
        std::vector<Matz> full_wing;
        if (test_wing)
        {
            full.cal_wing(cs, 0.0, empty_vq);
            // The Coulomb-basis transform releases the auxiliary tensors.
            for (const auto &tensor : full.get_wing_auxiliary()) full_wing.push_back(tensor.copy());
            full.wing_mu_to_lambda(sqrt_v, sqrt_desc, 2);
        }
        const auto &full_head = full.get_head_tensors();

        // All routes share the same independent complete-grid reference.
        // FullCache retains the older consumer path alongside source-only IBZ.
        for (const auto route :
             {InputRoute::SourceOnly, InputRoute::StarWeighted, InputRoute::FullCache})
        {
            double route_error = 0.0;
            diele_func reduced(route == InputRoute::StarWeighted ? star_weighted_mf : ibz_mf,
                               ibz_velocity, ibz_pbc.kfrac_list, basis_wfc, basis_abf, frequencies,
                               2, 4, 1, 2, ibz_pbc, librpa_int::global::mpi_comm_global_h, blacs_h);
            reduced.use_soc = reduced.use_symmetry = true;
            reduced.set_symmetry_context(ctx);
            reduced.atom_nw = basis_wfc.get_atom_nb_map();
            reduced.coord_frac = ctx.input_coord_frac;
            if (route == InputRoute::StarWeighted)
                reduced.normalize_ibz_star_weighted_occupations();
            if (route == InputRoute::FullCache)
                reduced.set_direct_full_bz_headwing_inputs(full_velocity, full_mf, {{0}, {1, 2}});
            assert(reduced.has_direct_full_bz_headwing_inputs() ==
                   (route == InputRoute::FullCache));
            for (int source_ik = 0; source_ik != 2; ++source_ik)
                for (int band = 0; band != 4; ++band)
                {
                    assert_complex_close(ibz_mf.get_weight()[0](source_ik, band),
                                         band < 2 ? 0.5 : 0.0, 1e-15);
                    assert_complex_close(star_weighted_mf.get_weight()[0](source_ik, band),
                                         band < 2 ? static_cast<double>(source_ik + 1) / 3.0 : 0.0,
                                         1e-15);
                    assert_complex_close(
                        reduced.get_meanfield_df().get_weight()[0](source_ik, band),
                        band < 2 ? 0.5 : 0.0, 1e-15);
                }
            reduced.init(0.0, empty_vq);
            reduced.cal_head();
            if (test_wing) reduced.cal_wing(cs, 0.0, empty_vq);
            const auto &ibz_head = reduced.get_head_tensors();
            const auto &ibz_wing = reduced.get_wing_auxiliary();
            assert(full_head.size() == frequencies.size() && ibz_head.size() == frequencies.size());
            if (test_wing)
                assert(full_wing.size() == frequencies.size() &&
                       ibz_wing.size() == frequencies.size());
            for (std::size_t frequency = 0; frequency != frequencies.size(); ++frequency)
            {
                for (int row = 0; row != 3; ++row)
                    for (int col = 0; col != 3; ++col)
                    {
                        head_magnitude = std::max(
                            head_magnitude,
                            std::abs(full_head[frequency](row, col) - (row == col ? 1.0 : 0.0)));
                        route_error =
                            std::max(route_error, std::abs(full_head[frequency](row, col) -
                                                           ibz_head[frequency](row, col)));
                        if (change_degenerate_gauge)
                            route_error =
                                std::max(route_error, std::abs(full_head[frequency](row, col) -
                                                               baseline_head[frequency](row, col)));
                    }
                if (test_wing)
                    for (int mu = 0; mu != 2; ++mu)
                        for (int alpha = 0; alpha != 3; ++alpha)
                        {
                            wing_magnitude =
                                std::max(wing_magnitude, std::abs(full_wing[frequency](mu, alpha)));
                            route_error =
                                std::max(route_error, std::abs(full_wing[frequency](mu, alpha) -
                                                               ibz_wing[frequency](mu, alpha)));
                            if (change_degenerate_gauge)
                                route_error = std::max(
                                    route_error, std::abs(full_wing[frequency](mu, alpha) -
                                                          baseline_wing[frequency](mu, alpha)));
                        }
            }
            if (test_wing)
            {
                reduced.wing_mu_to_lambda(sqrt_v, sqrt_desc, 2);
                for (std::size_t frequency = 0; frequency != frequencies.size(); ++frequency)
                {
                    const auto x = full.get_rpa_chi0v_wing(frequency);
                    const auto y = reduced.get_rpa_chi0v_wing(frequency);
                    assert(x.nr() == y.nr() && x.nc() == y.nc());
                    for (int row = 0; row != x.nr(); ++row)
                        for (int col = 0; col != x.nc(); ++col)
                        {
                            route_error =
                                std::max(route_error, std::abs(x(row, col) - y(row, col)));
                            regular_wing_magnitude =
                                std::max(regular_wing_magnitude, std::abs(x(row, col)));
                        }
                }
            }
            maximum_error = std::max(maximum_error, route_error);
            assert(route_error < (route == InputRoute::FullCache ? 1e-12 : 2e-11));
        }
        if (test_wing)
        {
            // A complete BZ with single-member stars needs no reconstruction.
            // Feed genuinely distributed complex AO/band blocks without a
            // separate full-BZ cache, as in the collinear shrink-wing driver.
            SymmetryContext full_ctx = ctx;
            full_ctx.kstars.clear();
            full_ctx.kstar_grid_mapping.clear();
            for (int ik = 0; ik != 3; ++ik)
            {
                librpa_int::SymmetryKStar star;
                star.star_index = ik;
                star.k_ibz = full_pbc.kfrac_list[ik];
                star.members = {make_member(star.k_ibz, false)};
                full_ctx.kstars.push_back(star);
                full_ctx.kstar_grid_mapping.push_back({ik, ik, {star.k_ibz}});
            }
            KPointBlacsParallelContext kctx(
                KPointBlacsProcessShape(1, librpa_int::get_mpi_size(MPI_COMM_WORLD), true),
                MPI_COMM_WORLD, 3);
            const auto desc_wfc = kctx.create_array_desc(2, 4, 1, 1);
            MeanField distributed_mf = full_mf;
            for (int ik = 0; ik != 3; ++ik)
                for (int spinor = 0; spinor != 2; ++spinor)
                {
                    const auto &source = full_mf.get_eigenvectors()[0][spinor][ik];
                    ComplexMatrix local(desc_wfc.n_loc(), desc_wfc.m_loc(), false);
                    for (int band = 0; band != desc_wfc.n_loc(); ++band)
                        for (int ao = 0; ao != desc_wfc.m_loc(); ++ao)
                            local(band, ao) =
                                source(desc_wfc.indx_l2g_c(band), desc_wfc.indx_l2g_r(ao));
                    distributed_mf.get_eigenvectors()[0][spinor][ik] = std::move(local);
                }
            diele_func distributed(distributed_mf, full_velocity, full_pbc.kfrac_list, basis_wfc,
                                   basis_abf, frequencies, 2, 4, 1, 2, full_pbc,
                                   librpa_int::global::mpi_comm_global_h, blacs_h, true, &kctx,
                                   &desc_wfc);
            distributed.use_soc = distributed.use_symmetry = true;
            distributed.set_symmetry_context(full_ctx);
            distributed.atom_nw = basis_wfc.get_atom_nb_map();
            distributed.coord_frac = full_ctx.input_coord_frac;
            assert(!distributed.has_direct_full_bz_headwing_inputs());
            distributed.init(0.0, empty_vq);
            distributed.cal_head();
            distributed.cal_wing(cs, 0.0, empty_vq);
            double distributed_error = 0.0;
            for (std::size_t frequency = 0; frequency != frequencies.size(); ++frequency)
            {
                const auto &head = distributed.get_head_tensors()[frequency];
                const auto &wing = distributed.get_wing_auxiliary()[frequency];
                for (int alpha = 0; alpha != 3; ++alpha)
                {
                    for (int beta = 0; beta != 3; ++beta)
                        distributed_error = std::max(
                            distributed_error,
                            std::abs(head(alpha, beta) - full_head[frequency](alpha, beta)));
                    for (int mu = 0; mu != 2; ++mu)
                        distributed_error =
                            std::max(distributed_error,
                                     std::abs(wing(mu, alpha) - full_wing[frequency](mu, alpha)));
                }
            }
            distributed.wing_mu_to_lambda(sqrt_v, sqrt_desc, 2);
            for (std::size_t frequency = 0; frequency != frequencies.size(); ++frequency)
            {
                const auto expected = full.get_rpa_chi0v_wing(frequency);
                const auto actual = distributed.get_rpa_chi0v_wing(frequency);
                for (int row = 0; row != expected.nr(); ++row)
                    for (int col = 0; col != expected.nc(); ++col)
                        distributed_error = std::max(
                            distributed_error, std::abs(actual(row, col) - expected(row, col)));
            }
            MPI_Allreduce(MPI_IN_PLACE, &distributed_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (librpa_int::global::mpi_comm_global_h.is_root())
                std::cout << "SOC full-BZ k-BLACS single-member symmetry error: "
                          << distributed_error << std::endl;
            assert(distributed_error < 2e-11);
            maximum_error = std::max(maximum_error, distributed_error);
        }
        if (!change_degenerate_gauge)
            for (std::size_t frequency = 0; frequency != frequencies.size(); ++frequency)
            {
                baseline_head.push_back(full_head[frequency].copy());
                if (test_wing) baseline_wing.push_back(full_wing[frequency].copy());
            }
    }
    MPI_Allreduce(MPI_IN_PLACE, &maximum_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &regular_wing_magnitude, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        std::cout << "SOC full-cache/source-IBZ/weighted " << (test_wing ? "head/wing" : "head")
                  << " and degenerate-gauge error: " << maximum_error
                  << "; nonzero head: " << head_magnitude;
        if (test_wing)
            std::cout << "; nonzero wing: " << wing_magnitude
                      << "; nonzero regular wing: " << regular_wing_magnitude;
        std::cout << std::endl;
    }
    assert(head_magnitude > 1e-6);
    if (test_wing) assert(wing_magnitude > 1e-4 && regular_wing_magnitude > 1e-6);
    assert(maximum_error < 2e-11);
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

#ifdef LIBRPA_USE_LIBRI
void test_complex_soc_bubble_matches_lindhard()
{
    using namespace librpa_int;
    using Z = std::complex<double>;
    Dataset ds(MPI_COMM_WORLD);
    ds.basis_wfc.set(std::vector<std::size_t>{3});
    ds.basis_aux.set(std::vector<std::size_t>{2});
    ds.mf.set(1, 4, 6, 3, 2);
    ds.mf.get_efermi() = 0;
    ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    const double pi = std::acos(-1.0);
    ds.pbc.set_kgrids_kvec(4, 1, 1, {0, 0, 0, pi / 2, 0, 0, pi, 0, 0, -pi / 2, 0, 0});
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, 4);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(6, 3, 6, 3);
    ds.tfg.reset(24);
    ds.tfg.generate_minimax(0.4, 4.0);

    const double entries[2][9] = {{0.4, 0.3, 0.4, 0.3, 0.7, 0.2, 0.4, 0.2, 0.9},
                                  {0.9, 0.13, -0.17, 0.13, 0.1, 0.07, -0.17, 0.07, 0.4}};
    std::array<ComplexMatrix, 2> density{ComplexMatrix(3, 3), ComplexMatrix(3, 3)};
    for (int mu = 0; mu < 2; ++mu)
        for (int i = 0; i < 9; ++i) density[mu].c[i] = entries[mu][i];
    ComplexMatrix base(6, 6);
    const double c = std::cos(0.31), t = std::sin(0.31);
    base(0, 0) = base(3, 1) = c;
    base(4, 0) = base(1, 1) = Z(0, t);
    base(1, 2) = base(4, 3) = c;
    base(3, 2) = base(0, 3) = Z(0, t);
    base(2, 4) = base(5, 5) = 1;
    std::vector<ComplexMatrix> coefficients;
    const double energies[] = {-0.7, 0.3, 1.1};
    for (int k = 0; k < 4; ++k)
    {
        auto current = base;
        // Nonconstant complex Bloch phases give nonzero translated Green blocks.
        for (int component = 0; component < 2; ++component)
            for (int band = 0; band < 6; ++band)
                current(3 * component + 1, band) *= std::exp(Z(0, pi * k / 2));
        coefficients.push_back(current);
        for (int component = 0; component < 2; ++component)
        {
            ComplexMatrix wfc(6, 3);
            for (int band = 0; band < 6; ++band)
                for (int ao = 0; ao < 3; ++ao) wfc(band, ao) = current(3 * component + ao, band);
            ds.mf.get_eigenvectors()[0][component][k] = wfc;
        }
        for (int band = 0; band < 6; ++band)
        {
            ds.mf.get_eigenvals()[0](k, band) = energies[band / 2];
            ds.mf.get_weight()[0](k, band) = band < 2 ? 0.25 : 0;
        }
    }
    ds.cs_data.use_libri = true;
    if (ds.comm_h.is_root())
    {
        auto values = std::make_shared<std::valarray<double>>(0.0, 18);
        for (int mu = 0; mu < 2; ++mu)
            for (int i = 0; i < 9; ++i) (*values)[9 * mu + i] = 0.5 * entries[mu][i];
        ds.cs_data.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({2, 3, 3}, values);
    }
    const auto assigned =
        dispatch_upper_triangular_tasks(1, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
                                        ds.blacs_h.myprow, ds.blacs_h.mypcol);
    const std::vector<atpair_t> pairs(assigned.begin(), assigned.end());
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    Chi0 chi(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context, ds.tfg,
             ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
    chi.gf_threshold = chi.libri_threshold_C = chi.libri_threshold_G = 0;
    chi.build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_AUTO, ds.cs_data, pairs, ds.basis_aux, no_shrink,
              ds.blacs_h);
    double maximum = 0;
    for (const double frequency : ds.tfg.get_freq_nodes())
        for (int iq = 0; iq < 4; ++iq)
        {
            ComplexMatrix local(2, 2), total(2, 2), exact(2, 2);
            const auto f = chi.get_chi0_q().find(frequency);
            if (f != chi.get_chi0_q().end())
            {
                const auto q = f->second.find(ds.pbc.klist[iq]);
                if (q != f->second.end() && q->second.count(0) && q->second.at(0).count(0))
                    local = q->second.at(0).at(0);
            }
            MPI_Allreduce(local.c, total.c, 4, MPI_C_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
            for (int k = 0; k < 4; ++k)
                for (int n = 0; n < 6; ++n)
                    for (int m = 0; m < 6; ++m)
                    {
                        const double occupation_difference =
                            (n < 2 ? 1.0 : 0.0) - (m < 2 ? 1.0 : 0.0);
                        if (!occupation_difference) continue;
                        const int kq = (k + iq) % 4;
                        std::array<Z, 2> vertex{};
                        for (int mu = 0; mu < 2; ++mu)
                            for (int spin = 0; spin < 2; ++spin)
                                for (int i = 0; i < 3; ++i)
                                    for (int j = 0; j < 3; ++j)
                                        vertex[mu] += std::conj(coefficients[k](3 * spin + i, n)) *
                                                      density[mu](i, j) *
                                                      coefficients[kq](3 * spin + j, m);
                        const Z factor =
                            occupation_difference /
                            (4.0 * (Z(0, frequency) + energies[n / 2] - energies[m / 2]));
                        for (int mu = 0; mu < 2; ++mu)
                            for (int nu = 0; nu < 2; ++nu)
                                exact(mu, nu) += factor * vertex[mu] * std::conj(vertex[nu]);
                    }
            for (int i = 0; i < 4; ++i)
            {
                maximum = std::max(maximum, std::abs(total.c[i] - exact.c[i]));
                assert(std::isfinite(total.c[i].real()) && std::isfinite(total.c[i].imag()));
                assert(std::abs(total.c[i] - exact.c[i]) < 2.0e-8);
            }
        }
    if (ds.comm_h.is_root()) std::cout << "Complex SOC charge Lindhard maximum " << maximum << '\n';
}
#endif

void test_soc_charge_vertex_is_spin_trace_and_rotation_invariant(const BlacsCtxtHandler &blacs_h)
{
    const int nprocs = librpa_int::get_mpi_size(MPI_COMM_WORLD);
    KPointBlacsProcessShape shape(1, nprocs, true);
    KPointBlacsParallelContext kctx(shape, MPI_COMM_WORLD, 1);
    const auto desc_wfc = kctx.create_array_desc(2, 4, 2, 4);

    // A complete complex spinor frame in AO-major/spinor-minor order.
    Matz coefficients(4, 4, MAJOR::ROW);
    for (int row = 0; row != 4; ++row)
        for (int band = 0; band != 4; ++band)
            coefficients(row, band) =
                std::polar(0.5, librpa_int::TWO_PI * row * band / 4.0 + 0.31 * row + 0.21 * band);
    Matz rotated(4, 4, MAJOR::ROW);
    const double c = std::cos(0.63);
    const auto s = std::polar(std::sin(0.63), 0.41);
    for (int ao = 0; ao != 2; ++ao)
        for (int band = 0; band != 4; ++band)
        {
            rotated(2 * ao, band) =
                c * coefficients(2 * ao, band) + s * coefficients(2 * ao + 1, band);
            rotated(2 * ao + 1, band) =
                -std::conj(s) * coefficients(2 * ao, band) + c * coefficients(2 * ao + 1, band);
        }
    MeanField mf(1, 1, 4, 2, 2);
    MeanField mf_rotated(1, 1, 4, 2, 2);
    for (int spinor = 0; spinor != 2; ++spinor)
    {
        auto &wfc = mf.get_eigenvectors()[0][spinor][0];
        auto &wfc_rotated = mf_rotated.get_eigenvectors()[0][spinor][0];
        wfc.create(4, 2);
        wfc_rotated.create(4, 2);
        for (int band = 0; band != 4; ++band)
            for (int ao = 0; ao != 2; ++ao)
            {
                wfc(band, ao) = coefficients(2 * ao + spinor, band);
                wfc_rotated(band, ao) = rotated(2 * ao + spinor, band);
            }
    }

    librpa_int::velocity_matrix_t velocity;
    librpa_int::initialize_velocity_matrix(velocity, 1, 1, 4);
    AtomicBasis basis_wfc({2});
    AtomicBasis basis_abf({1});
    PeriodicBoundaryData pbc;
    const std::vector<Vector3_Order<double>> kfrac{{0.0, 0.0, 0.0}};
    const std::vector<double> omega{0.5};
    diele_func df(mf, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 4, 1, 1, pbc,
                  librpa_int::global::mpi_comm_global_h, blacs_h, true, &kctx, &desc_wfc);
    diele_func df_rotated(mf_rotated, velocity, kfrac, basis_wfc, basis_abf, omega, 2, 4, 1, 1, pbc,
                          librpa_int::global::mpi_comm_global_h, blacs_h, true, &kctx, &desc_wfc);

    auto tensor_data = std::make_shared<std::valarray<double>>(4);
    (*tensor_data)[0] = 0.9;
    (*tensor_data)[1] = 0.31;
    (*tensor_data)[2] = -0.27;
    (*tensor_data)[3] = 0.63;
    std::map<int, std::map<librpa_int::libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    Cs_IJ[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1UL, 2UL, 2UL}, tensor_data);
    librpa_int::Cs_LRI Cs_data;
    Cs_data.use_libri = true;
    Cs_data.data_libri = Cs_IJ;
    const auto Cs_IJ_k = librpa_int::fourier_headwing_cs_to_ijk(
        Cs_data, basis_wfc, basis_abf, librpa_int::build_headwing_full_bz_fourier_targets(kfrac));

    // Independent dense charge operator: (A + A^dagger) tensor I_spin.
    // Both local RI centres coincide in this one-atom control.
    Matz charge(4, 4, MAJOR::ROW);
    std::fill(charge.ptr(), charge.ptr() + charge.size(), std::complex<double>{});
    for (int row = 0; row != 4; ++row)
        for (int col = 0; col != 4; ++col)
            if (row % 2 == col % 2)
                charge(row, col) = (*tensor_data)[2 * (row / 2) + col / 2] +
                                   (*tensor_data)[2 * (col / 2) + row / 2];
    const Matz expected = transpose(coefficients, true) * charge * coefficients;
    const auto world = df.transform_Cs2mnk(0, 0, Cs_IJ);
    const auto world_rotated = df_rotated.transform_Cs2mnk(0, 0, Cs_IJ);
    const auto kblacs = df.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);
    const auto kblacs_rotated = df_rotated.transform_Cs2mnk_kblacs(0, 0, 0, Cs_IJ_k, kctx.blacs_h);
    double local_error = 0.0;
    const std::array<const char *, 4> route_names{"world", "world_SU2", "kBLACS", "kBLACS_SU2"};
    int route = 0;
    for (const auto *actual : {&world, &world_rotated, &kblacs, &kblacs_rotated})
    {
        double local_route_error = 0.0;
        for (int row = 0; row != actual->first.m_loc(); ++row)
            for (int col = 0; col != actual->first.n_loc(); ++col)
            {
                const double difference =
                    std::abs(actual->second(row, col) - expected(actual->first.indx_l2g_r(row),
                                                                 actual->first.indx_l2g_c(col)));
                if (!std::isfinite(difference)) MPI_Abort(MPI_COMM_WORLD, 1);
                local_route_error = std::max(local_route_error, difference);
            }
        double route_error = 0.0;
        MPI_Allreduce(&local_route_error, &route_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (librpa_int::global::mpi_comm_global_h.is_root())
            std::cout << "SOC charge " << route_names[route] << " error: " << route_error
                      << std::endl;
        ++route;
        local_error = std::max(local_error, local_route_error);
    }
    double error = 0.0;
    MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (librpa_int::global::mpi_comm_global_h.is_root())
        std::cout << "SOC charge spin trace / SU(2) maximum error: " << error << std::endl;
    if (!std::isfinite(error) || error >= 1e-12) MPI_Abort(MPI_COMM_WORLD, 1);
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

    ArrayDesc desc_head_direction(blacs_h);
    desc_head_direction.init(n, 1, block_size, block_size, 0, 0);
    auto head_direction = init_local_mat<std::complex<double>>(desc_head_direction, MAJOR::COL);
    librpa_int::ScalapackConnector::pgemr2d_f(
        n, 1, U_dist.ptr(), 1, 1, desc.desc,
        head_direction.ptr(), 1, 1, desc_head_direction.desc, blacs_h.ictxt);

    std::vector<double> qx(q_pts.size()), qy(q_pts.size()), qz(q_pts.size());
    for (std::size_t i = 0; i < q_pts.size(); ++i)
    {
        qx[i] = q_pts[i][0];
        qy[i] = q_pts[i][1];
        qz[i] = q_pts[i][2];
    }

    librpa_int::rewrite_eps_abf_space(E_dist, sqrtV_dist, U_dist,
                                      head_direction, desc_head_direction, head, wing_mu, qx, qy,
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

    // Without filtering, only the explicit head direction matters; changing
    // any Coulomb eigenvector column must leave the result unchanged.
    {
        auto U_mod = U.copy();
        for (int col = 0; col < n; ++col)
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

        ArrayDesc desc_head_direction(blacs_h);
        desc_head_direction.init(n, 1, 2, 1, 0, 0);
        auto head_direction = init_local_mat<std::complex<double>>(desc_head_direction, MAJOR::COL);
        librpa_int::ScalapackConnector::pgemr2d_f(
            n, 1, U_orig_dist.ptr(), 1, 1, desc.desc,
            head_direction.ptr(), 1, 1, desc_head_direction.desc, blacs_h.ictxt);

        std::vector<double> qx(6), qy(6), qz(6);
        for (std::size_t i = 0; i < q3d.size(); ++i)
        {
            qx[i] = q3d[i][0];
            qy[i] = q3d[i][1];
            qz[i] = q3d[i][2];
        }

        librpa_int::rewrite_eps_abf_space(E1, sqrtV_dist, U_orig_dist,
                                          head_direction, desc_head_direction, head, wing_mu, qx, qy,
                                          qz, rho3d, desc, blacs_h,
                                          static_cast<std::size_t>(n), 0.0, false, false);
        librpa_int::rewrite_eps_abf_space(E2, sqrtV_dist, U_mod_dist,
                                          head_direction, desc_head_direction, head, wing_mu, qx, qy,
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

void add_wq_blocks(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old &wq,
    const Vector3_Order<double> &q, const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, source] : row)
        {
            auto &target = wq[atom_i][atom_j][q];
            target = matrix_m<std::complex<double>>(source.nr, source.nc, MAJOR::ROW);
            for (int i = 0; i != source.nr; ++i)
                for (int j = 0; j != source.nc; ++j) target(i, j) = source(i, j);
        }
    }
}

void add_vq_blocks(atpair_k_cplx_mat_t &vq, const Vector3_Order<double> &q,
                   const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, source] : row)
        {
            vq[atom_i][atom_j][q] = std::make_shared<ComplexMatrix>(source);
        }
    }
}

librpa_int::symmetry_atom_block_matrix_map_t make_bn_hermitian_wq_blocks(
    const std::vector<int> &n_by_atom, const double index)
{
    const int n = std::accumulate(n_by_atom.begin(), n_by_atom.end(), 0);
    ComplexMatrix dense(n, n);
    for (int i = 0; i != n; ++i)
    {
        dense(i, i) = {1.0 + 0.09 * index + 0.04 * (i + 1), 0.0};
        for (int j = i + 1; j != n; ++j)
        {
            dense(i, j) = {0.023 * (i + 1) * (j + 2) + 0.011 * index,
                           0.017 * (j - i) + 0.007 * index};
            dense(j, i) = std::conj(dense(i, j));
        }
    }

    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    std::vector<int> offsets{0};
    for (const int n_atom : n_by_atom) offsets.push_back(offsets.back() + n_atom);
    for (atom_t atom_i = 0; atom_i != 2; ++atom_i)
    {
        for (atom_t atom_j = 0; atom_j != 2; ++atom_j)
        {
            auto &block = blocks[atom_i][atom_j];
            block.create(n_by_atom.at(atom_i), n_by_atom.at(atom_j));
            for (int i = 0; i != n_by_atom.at(atom_i); ++i)
                for (int j = 0; j != n_by_atom.at(atom_j); ++j)
                    block(i, j) = dense(offsets.at(atom_i) + i, offsets.at(atom_j) + j);
        }
    }
    return blocks;
}

std::vector<std::vector<int>> make_bn_test_shells(const int basis_case)
{
    if (basis_case == 0) return {{0}, {0}};
    if (basis_case == 1) return {{0, 1}, {0, 1}};
    if (basis_case == 2)
        return {{0, 0, 0, 1, 1, 1, 2, 2}, {0, 0, 0, 1, 1, 1, 2, 2}};
    if (basis_case == 3)
        return {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2,
                 3, 3, 3, 3, 3, 4, 4, 4},
                {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3,
                 4, 4, 4}};
    throw std::runtime_error("unknown BN basis test case");
}

std::vector<int> basis_counts_from_shells(const std::vector<std::vector<int>> &shells)
{
    std::vector<int> counts;
    for (const auto &atom_shells : shells)
    {
        int count = 0;
        for (const int l : atom_shells) count += 2 * l + 1;
        counts.push_back(count);
    }
    return counts;
}

void test_spacetime_fourier_phases_form_k_minus_q_convolution()
{
    constexpr int mesh = 3;
    const std::array<std::complex<double>, mesh> wc_q{{{1.2, -0.3}, {-0.4, 0.7}, {0.9, 0.2}}};
    const std::array<std::complex<double>, mesh> weighted_green_k{{{0.13, 0.04},
                                                                    {-0.08, 0.02},
                                                                    {0.05, -0.06}}};
    std::array<std::complex<double>, mesh> wc_R{};
    std::array<std::complex<double>, mesh> green_R{};
    for (int iR = 0; iR != mesh; ++iR)
        for (int iq = 0; iq != mesh; ++iq)
        {
            const double angle = -librpa_int::TWO_PI * static_cast<double>(iq * iR) / mesh;
            const std::complex<double> phase{std::cos(angle), std::sin(angle)};
            wc_R[iR] += wc_q[iq] * phase / static_cast<double>(mesh);
            green_R[iR] += weighted_green_k[iq] * phase;
        }
    for (int ik = 0; ik != mesh; ++ik)
    {
        std::complex<double> spacetime_sigma_k{};
        std::complex<double> direct_k_minus_q{};
        std::complex<double> direct_k_plus_q{};
        for (int iR = 0; iR != mesh; ++iR)
        {
            const double angle = librpa_int::TWO_PI * static_cast<double>(ik * iR) / mesh;
            const std::complex<double> phase{std::cos(angle), std::sin(angle)};
            spacetime_sigma_k += wc_R[iR] * green_R[iR] * phase;
        }
        for (int iq = 0; iq != mesh; ++iq)
        {
            direct_k_minus_q += wc_q[iq] * weighted_green_k[(ik - iq + mesh) % mesh];
            direct_k_plus_q += wc_q[iq] * weighted_green_k[(ik + iq) % mesh];
        }
        assert_complex_close(spacetime_sigma_k, direct_k_minus_q, 1.0e-13);
        if (std::abs(spacetime_sigma_k - direct_k_plus_q) < 1.0e-5)
            throw std::runtime_error("synthetic data do not distinguish k-q from k+q");
    }
}

std::vector<SymmetryOperation> make_bn_hexagonal_operations()
{
    const std::array<std::array<int, 9>, 12> rotations{{
        {{1, 0, 0, 0, 1, 0, 0, 0, 1}}, {{0, 1, 0, -1, -1, 0, 0, 0, -1}},
        {{-1, -1, 0, 1, 0, 0, 0, 0, 1}}, {{1, 0, 0, 0, 1, 0, 0, 0, -1}},
        {{0, 1, 0, -1, -1, 0, 0, 0, 1}}, {{-1, -1, 0, 1, 0, 0, 0, 0, -1}},
        {{-1, 0, 0, 1, 1, 0, 0, 0, 1}}, {{1, 1, 0, 0, -1, 0, 0, 0, -1}},
        {{0, -1, 0, -1, 0, 0, 0, 0, 1}}, {{-1, 0, 0, 1, 1, 0, 0, 0, -1}},
        {{1, 1, 0, 0, -1, 0, 0, 0, 1}}, {{0, -1, 0, -1, 0, 0, 0, 0, -1}}}};
    std::vector<SymmetryOperation> operations;
    operations.reserve(rotations.size());
    for (const auto &rotation : rotations)
    {
        SymmetryOperation operation;
        operation.rotation = Matrix3(rotation[0], rotation[1], rotation[2], rotation[3],
                                     rotation[4], rotation[5], rotation[6], rotation[7],
                                     rotation[8]);
        operation.translation = {0.0, 0.0, 0.0};
        operation.use_row_convention = true;
        operations.push_back(operation);
    }
    return operations;
}

void set_bn_hexagonal_lattice(PeriodicBoundaryData &pbc)
{
    const double sqrt_three = std::sqrt(3.0);
    pbc.set_latvec({0.5, -0.5 * sqrt_three, 0.0, 0.5, 0.5 * sqrt_three, 0.0, 0.0, 0.0,
                    8.0});
}

SymmetryContext make_bn_hexagonal_context(const PeriodicBoundaryData &pbc, const int max_l = 0)
{
    SymmetryContext ctx;
    ctx.set_crystal_structure(pbc.latvec, pbc.G, {{0, 0}, {1, 1}},
                              {{0, {1.0 / 3.0, 2.0 / 3.0, 0.5}},
                               {1, {2.0 / 3.0, 1.0 / 3.0, 0.5}}});
    ctx.set_rspace_operations(make_bn_hexagonal_operations());
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations(
        {-1, 0, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M, LIBRPA_RSH_COEFF_1_M},
        max_l);
    ctx.build_kstar_member_rotations(max_l);
    return ctx;
}

PeriodicBoundaryData make_bn_hexagonal_full_pbc(const int mesh = 3)
{
    PeriodicBoundaryData pbc;
    set_bn_hexagonal_lattice(pbc);
    std::vector<double> kvecs;
    for (const auto &kfrac : librpa_int::build_uniform_kmesh_frac({mesh, mesh, 1}))
    {
        const auto kvec = kfrac * pbc.G;
        kvecs.insert(kvecs.end(), {librpa_int::TWO_PI * kvec.x, librpa_int::TWO_PI * kvec.y,
                                   librpa_int::TWO_PI * kvec.z});
    }
    pbc.set_kgrids_kvec(mesh, mesh, 1, kvecs);
    return pbc;
}

PeriodicBoundaryData make_bn_hexagonal_reduced_pbc(const SymmetryContext &ctx,
                                                   const int mesh = 3)
{
    PeriodicBoundaryData pbc;
    set_bn_hexagonal_lattice(pbc);
    std::vector<double> kvecs_ibz;
    std::vector<std::vector<Vector3_Order<double>>> full_kstars;
    for (const auto &star : ctx.kstars)
    {
        const auto kvec = librpa_int::restrict_fractional_coordinate(star.k_ibz) * pbc.G;
        kvecs_ibz.insert(kvecs_ibz.end(), {librpa_int::TWO_PI * kvec.x,
                                           librpa_int::TWO_PI * kvec.y,
                                           librpa_int::TWO_PI * kvec.z});
        full_kstars.emplace_back();
        for (const auto &member : star.members)
            full_kstars.back().push_back(
                librpa_int::restrict_fractional_coordinate(member.k_bz) * pbc.G);
    }
    pbc.set_irreducible_kgrids_kvec(mesh, mesh, 1, kvecs_ibz, full_kstars);
    return pbc;
}

std::size_t find_fractional_kpoint_index(
    const std::vector<Vector3_Order<double>> &kpoints,
    const Vector3_Order<double> &target)
{
    for (std::size_t ik = 0; ik != kpoints.size(); ++ik)
        if (librpa_int::same_fractional_kpoint(kpoints[ik], target, 1e-8)) return ik;
    throw std::runtime_error("failed to find a full-grid fractional k-point in the test");
}

void assert_wq_rspace_maps_close(
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &actual,
    const atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old
        &expected)
{
    for (const auto &[atom_i, expected_row] : expected)
    {
        assert(actual.count(atom_i) != 0);
        for (const auto &[atom_j, expected_Rs] : expected_row)
        {
            assert(actual.at(atom_i).count(atom_j) != 0);
            for (const auto &[R, expected_block] : expected_Rs)
            {
                assert(actual.at(atom_i).at(atom_j).count(R) != 0);
                const auto &actual_block = actual.at(atom_i).at(atom_j).at(R);
                if (std::abs(actual_block(0, 0) - expected_block(0, 0)) >= 1e-12)
                {
                    std::cerr << "atom_pair=(" << atom_i << "," << atom_j << ") R=("
                              << R.x << "," << R.y << "," << R.z << ")" << std::endl;
                }
                assert_complex_close(actual_block(0, 0), expected_block(0, 0), 1e-12);
            }
        }
    }
}

void assert_vr_rspace_maps_close(const atpair_R_mat_t &actual,
                                 const atpair_R_mat_t &expected)
{
    assert(actual.size() == expected.size());
    for (const auto &[atom_i, expected_row] : expected)
    {
        assert(actual.count(atom_i) != 0);
        assert(actual.at(atom_i).size() == expected_row.size());
        for (const auto &[atom_j, expected_Rs] : expected_row)
        {
            assert(actual.at(atom_i).count(atom_j) != 0);
            assert(actual.at(atom_i).at(atom_j).size() == expected_Rs.size());
            for (const auto &[R, expected_block_ptr] : expected_Rs)
            {
                assert(actual.at(atom_i).at(atom_j).count(R) != 0);
                const auto &actual_block = *actual.at(atom_i).at(atom_j).at(R);
                const auto &expected_block = *expected_block_ptr;
                if (actual_block.nr != expected_block.nr || actual_block.nc != expected_block.nc)
                    throw std::runtime_error("V(R) atom block dimensions differ");
                for (int i = 0; i != expected_block.nr; ++i)
                {
                    for (int j = 0; j != expected_block.nc; ++j)
                    {
                        if (std::abs(actual_block(i, j) - expected_block(i, j)) >= 1e-12)
                        {
                            std::cerr << "V(R) atom_pair=(" << atom_i << "," << atom_j
                                      << ") R=(" << R.x << "," << R.y << "," << R.z
                                      << ") block=(" << i << "," << j << ") actual="
                                      << actual_block(i, j) << " expected=" << expected_block(i, j)
                                      << std::endl;
                        }
                        assert(std::abs(actual_block(i, j) - expected_block(i, j)) < 1e-12);
                    }
                }
            }
        }
    }
}

void test_bn_qstar_wq_to_wr_matches_explicit_full_bz_for_mesh(
    const int mesh, const int basis_case, const bool clear_rspace_sector = false)
{
    const auto pbc_full = make_bn_hexagonal_full_pbc(mesh);
    const auto ctx_full = make_bn_hexagonal_context(pbc_full);
    const auto pbc_sym = make_bn_hexagonal_reduced_pbc(ctx_full, mesh);
    const auto shells = make_bn_test_shells(basis_case);
    const auto counts = basis_counts_from_shells(shells);
    const int max_l = *std::max_element(shells.front().begin(), shells.front().end());
    auto ctx = make_bn_hexagonal_context(pbc_sym, max_l);

    assert(ctx.count_kstar_members() == static_cast<std::size_t>(mesh * mesh));

    AtomicBasis basis_abf(
        std::vector<std::size_t>{static_cast<std::size_t>(counts.at(0)),
                                 static_cast<std::size_t>(counts.at(1))});
    basis_abf.set_l_shells(shells);
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{
        {0, static_cast<std::size_t>(counts.at(0))},
        {1, static_cast<std::size_t>(counts.at(1))}};
    const std::set<std::pair<atom_t, atom_t>> target_pairs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    const auto full_targets =
        librpa_int::build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, pbc_sym.kfrac_list);
    const bool use_full_targets = full_targets.size() == ctx.kstars.size();

    for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
    {
        const auto &star = ctx.kstars[istar];
        const auto mapping_iter =
            std::find_if(ctx.kstar_grid_mapping.begin(), ctx.kstar_grid_mapping.end(),
                         [istar](const librpa_int::SymmetryKStarGridMappingEntry &entry)
                         { return entry.star_list_index == static_cast<int>(istar); });
        assert(mapping_iter != ctx.kstar_grid_mapping.end());
        const double index = static_cast<double>(istar + 1);
        const auto blocks_ibz = make_bn_hermitian_wq_blocks(counts, index);
        add_wq_blocks(wq_sym, pbc_sym.klist.at(istar), blocks_ibz);

        const auto closure = librpa_int::build_symmetry_upper_atom_pair_closure(star, target_pairs);
        const auto symmetrized = librpa_int::symmetrize_symmetry_ibz_kspace_operator_blocks(
            ctx, layouts, star.k_ibz, blocks_ibz, atom_nabf, &closure);
        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const auto target = librpa_int::restrict_fractional_coordinate(
                use_full_targets
                    ? full_targets.at(istar).at(imember)
                    : Vector3_Order<double>{pbc_sym.latvec *
                                            mapping_iter->member_q_bz_keys.at(imember)});
            const auto rotated = librpa_int::rotate_symmetry_kspace_operator_blocks(
                ctx, layouts, star.members[imember], symmetrized, atom_nabf, star.k_ibz,
                star.members[imember].time_reversal, &target_pairs, &target);
            const auto ifull = find_fractional_kpoint_index(pbc_full.kfrac_list, target);
            add_wq_blocks(wq_full, pbc_full.klist.at(ifull), rotated);
        }
    }

    const double collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
        for (auto &[atom_j, q_blocks] : row)
            for (auto &[q, block] : q_blocks) block *= collective_scale;

    if (clear_rspace_sector)
    {
        ctx.irreducible_sector.clear();
        ctx.rspace_sector_stars.clear();
    }

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry,
                              wq_full, dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual =
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
                              dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);
    assert_wq_rspace_maps_close(actual, expected);
}

void test_bn_qstar_wq_to_wr_matches_explicit_full_bz_on_odd_and_even_meshes()
{
    for (const int basis_case : {0, 1, 3})
    {
        test_bn_qstar_wq_to_wr_matches_explicit_full_bz_for_mesh(3, basis_case);
        test_bn_qstar_wq_to_wr_matches_explicit_full_bz_for_mesh(4, basis_case);
    }
}

void test_bn_qstar_wq_to_wr_without_rspace_sector_matches_explicit_full_bz()
{
    test_bn_qstar_wq_to_wr_matches_explicit_full_bz_for_mesh(3, 1, true);
}

void test_bn_qstar_vq_to_vr_without_rspace_sector_matches_explicit_full_bz()
{
    constexpr int mesh = 3;
    constexpr int basis_case = 1;
    const auto pbc_full = make_bn_hexagonal_full_pbc(mesh);
    const auto ctx_full = make_bn_hexagonal_context(pbc_full);
    const auto pbc_sym = make_bn_hexagonal_reduced_pbc(ctx_full, mesh);
    const auto shells = make_bn_test_shells(basis_case);
    const auto counts = basis_counts_from_shells(shells);
    const int max_l = *std::max_element(shells.front().begin(), shells.front().end());
    auto ctx = make_bn_hexagonal_context(pbc_sym, max_l);

    AtomicBasis basis_abf(
        std::vector<std::size_t>{static_cast<std::size_t>(counts.at(0)),
                                 static_cast<std::size_t>(counts.at(1))});
    basis_abf.set_l_shells(shells);
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{
        {0, static_cast<std::size_t>(counts.at(0))},
        {1, static_cast<std::size_t>(counts.at(1))}};
    const std::set<std::pair<atom_t, atom_t>> target_pairs{{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    atpair_k_cplx_mat_t vq_sym;
    atpair_k_cplx_mat_t vq_full;
    const auto full_targets =
        librpa_int::build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, pbc_sym.kfrac_list);
    const bool use_full_targets = full_targets.size() == ctx.kstars.size();

    for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
    {
        const auto &star = ctx.kstars[istar];
        const auto mapping_iter =
            std::find_if(ctx.kstar_grid_mapping.begin(), ctx.kstar_grid_mapping.end(),
                         [istar](const librpa_int::SymmetryKStarGridMappingEntry &entry)
                         { return entry.star_list_index == static_cast<int>(istar); });
        assert(mapping_iter != ctx.kstar_grid_mapping.end());
        const double index = static_cast<double>(istar + 1);
        const auto blocks_ibz = make_bn_hermitian_wq_blocks(counts, index);
        add_vq_blocks(vq_sym, pbc_sym.klist.at(istar), blocks_ibz);

        const auto closure = librpa_int::build_symmetry_upper_atom_pair_closure(star, target_pairs);
        const auto symmetrized = librpa_int::symmetrize_symmetry_ibz_kspace_operator_blocks(
            ctx, layouts, star.k_ibz, blocks_ibz, atom_nabf, &closure);
        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const auto target = librpa_int::restrict_fractional_coordinate(
                use_full_targets
                    ? full_targets.at(istar).at(imember)
                    : Vector3_Order<double>{pbc_sym.latvec *
                                            mapping_iter->member_q_bz_keys.at(imember)});
            const auto rotated = librpa_int::rotate_symmetry_kspace_operator_blocks(
                ctx, layouts, star.members[imember], symmetrized, atom_nabf, star.k_ibz,
                star.members[imember].time_reversal, &target_pairs, &target);
            const auto ifull = find_fractional_kpoint_index(pbc_full.kfrac_list, target);
            add_vq_blocks(vq_full, pbc_full.klist.at(ifull), rotated);
        }
    }

    const double collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : vq_sym)
        for (auto &[atom_j, q_blocks] : row)
            for (auto &[q, block] : q_blocks) *block *= collective_scale;

    ctx.irreducible_sector.clear();
    ctx.rspace_sector_stars.clear();
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Vq(librpa_int::global::mpi_comm_global_h, basis_abf,
                                             no_symmetry, vq_full, pbc_full, true, false);
    const auto actual = librpa_int::FT_Vq(librpa_int::global::mpi_comm_global_h, basis_abf, ctx,
                                           vq_sym, pbc_sym, true, true);
    assert_vr_rspace_maps_close(actual, expected);
}

ComplexMatrix restore_bn_test_wfc_to_member(const SymmetryContext &ctx,
                                            const std::vector<SpeciesBasisLayout> &layouts,
                                            const std::map<atom_t, size_t> &atom_nw,
                                            const librpa_int::SymmetryKStar &star,
                                            const SymmetryKStarMember &member,
                                            const ComplexMatrix &wfc_ibz)
{
    const auto rotation = librpa_int::build_symmetry_kspace_rotation_matrix(
        ctx, layouts, member, atom_nw, star.k_ibz, member.time_reversal, &member.k_bz);
    return member.time_reversal ? librpa_int::conj(wfc_ibz) * librpa_int::conj(rotation)
                                : wfc_ibz * rotation;
}

void test_bn_kstar_green_function_matches_explicit_full_bz_for_mesh(const int mesh,
                                                                     const int basis_case)
{
    const auto pbc_full = make_bn_hexagonal_full_pbc(mesh);
    const auto ctx_full = make_bn_hexagonal_context(pbc_full);
    const auto pbc_sym = make_bn_hexagonal_reduced_pbc(ctx_full, mesh);
    const auto shells = make_bn_test_shells(basis_case);
    const auto counts = basis_counts_from_shells(shells);
    const int max_l = *std::max_element(shells.front().begin(), shells.front().end());
    auto ctx = make_bn_hexagonal_context(pbc_sym, max_l);

    const int n_aos = std::accumulate(counts.begin(), counts.end(), 0);
    AtomicBasis basis_wfc(
        std::vector<std::size_t>{static_cast<std::size_t>(counts.at(0)),
                                 static_cast<std::size_t>(counts.at(1))});
    basis_wfc.set_l_shells(shells);
    const auto layouts = basis_wfc.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nw{
        {0, static_cast<std::size_t>(counts.at(0))},
        {1, static_cast<std::size_t>(counts.at(1))}};
    const int n_full_kpoints = mesh * mesh;

    MeanField mf_ibz(1, static_cast<int>(ctx.kstars.size()), n_aos, n_aos);
    MeanField mf_full(1, n_full_kpoints, n_aos, n_aos);
    mf_ibz.get_efermi() = 0.0;
    mf_full.get_efermi() = 0.0;
    std::vector<int> full_owner(static_cast<std::size_t>(n_full_kpoints), -1);

    for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
    {
        const auto &star = ctx.kstars[istar];
        const double star_weight =
            static_cast<double>(star.members.size()) / static_cast<double>(n_full_kpoints);
        for (int iband = 0; iband != n_aos; ++iband)
        {
            const bool occupied = iband < n_aos / 2;
            mf_ibz.get_eigenvals()[0](static_cast<int>(istar), iband) =
                (occupied ? -0.8 - 0.13 * iband : 0.6 + 0.17 * (iband - n_aos / 2)) +
                0.03 * static_cast<double>(istar);
            mf_ibz.get_weight()[0](static_cast<int>(istar), iband) =
                occupied ? 2.0 * star_weight : 0.0;
        }
        auto &wfc_ibz = mf_ibz.get_eigenvectors()[0][0][istar];
        wfc_ibz.create(n_aos, n_aos);
        for (int iband = 0; iband != n_aos; ++iband)
            for (int iao = 0; iao != n_aos; ++iao)
                wfc_ibz(iband, iao) =
                    std::complex<double>{
                        0.11 * (iband + 1) + 0.07 * (iao + 1),
                        0.013 * static_cast<double>((iband + 1) * (iao + 2)) +
                            0.009 * static_cast<double>(istar + 1)} /
                    static_cast<double>(n_aos);

        for (const auto &member : star.members)
        {
            const auto ifull = find_fractional_kpoint_index(
                pbc_full.kfrac_list, librpa_int::restrict_fractional_coordinate(member.k_bz));
            assert(full_owner[ifull] < 0);
            full_owner[ifull] = static_cast<int>(istar);
            for (int iband = 0; iband != n_aos; ++iband)
            {
                mf_full.get_eigenvals()[0](static_cast<int>(ifull), iband) =
                    mf_ibz.get_eigenvals()[0](static_cast<int>(istar), iband);
                mf_full.get_weight()[0](static_cast<int>(ifull), iband) =
                    iband < n_aos / 2 ? 2.0 / static_cast<double>(n_full_kpoints) : 0.0;
            }
            mf_full.get_eigenvectors()[0][0][ifull] =
                restore_bn_test_wfc_to_member(ctx, layouts, atom_nw, star, member, wfc_ibz);
        }
    }
    assert(std::find(full_owner.begin(), full_owner.end(), -1) == full_owner.end());

    const std::vector<double> taus{-0.35, 0.27};
    const auto &Rs = pbc_full.Rlist;
    const auto expected = mf_full.get_gf_cplx_imagtimes_Rs(0, 0, 0, pbc_full.kfrac_list, taus, Rs);
    const auto actual = librpa_int::get_symmetry_restored_gf_cplx_imagtimes_Rs(
        ctx, layouts, mf_ibz, 0, 0, 0, pbc_sym.kfrac_list, taus, Rs, atom_nw);

    for (const auto tau : taus)
    {
        for (const auto &R : Rs)
        {
            const auto diff = actual.at(tau).at(R) - expected.at(tau).at(R);
            double max_abs = 0.0;
            for (int i = 0; i != diff.nr; ++i)
                for (int j = 0; j != diff.nc; ++j)
                    max_abs = std::max(max_abs, std::abs(diff(i, j)));
            if (max_abs >= 1e-12)
            {
                std::cerr << "BN k-star GF mismatch: mesh=" << mesh << " tau=" << tau << " R=" << R
                          << " basis_case=" << basis_case << " max_abs=" << max_abs << std::endl;
                throw std::runtime_error("BN k-star Green function differs from explicit full BZ");
            }
        }
    }
}

void test_bn_kstar_green_function_matches_explicit_full_bz_on_odd_and_even_meshes()
{
    for (const int basis_case : {0, 1, 2})
    {
        test_bn_kstar_green_function_matches_explicit_full_bz_for_mesh(3, basis_case);
        test_bn_kstar_green_function_matches_explicit_full_bz_for_mesh(4, basis_case);
    }
}

void add_scalar_wq_block(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const atom_t atom_i,
    const atom_t atom_j,
    const Vector3_Order<double> &q,
    const std::complex<double> value)
{
    auto &block = wq[atom_i][atom_j][q];
    block = matrix_m<std::complex<double>>(1, 1, MAJOR::ROW);
    block(0, 0) = value;
}

librpa_int::symmetry_atom_block_matrix_map_t scalar_wq_to_blocks(
    const std::map<atom_t, std::map<atom_t, std::complex<double>>> &values)
{
    librpa_int::symmetry_atom_block_matrix_map_t blocks;
    for (const auto &[atom_i, row] : values)
    {
        for (const auto &[atom_j, value] : row)
        {
            blocks[atom_i][atom_j] = ComplexMatrix(1, 1);
            blocks[atom_i][atom_j](0, 0) = value;
        }
    }
    return blocks;
}

void add_scalar_wq_blocks(
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        &wq,
    const Vector3_Order<double> &q,
    const librpa_int::symmetry_atom_block_matrix_map_t &blocks)
{
    for (const auto &[atom_i, row] : blocks)
    {
        for (const auto &[atom_j, block] : row)
        {
            add_scalar_wq_block(wq, atom_i, atom_j, q, block(0, 0));
        }
    }
}

PeriodicBoundaryData make_wq_full_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0,
        librpa_int::TWO_PI * 2.0 / 3.0, 0.0, 0.0};
    pbc.set_kgrids_kvec(3, 1, 1, kvecs);
    return pbc;
}

PeriodicBoundaryData make_wq_reduced_pbc()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    const std::vector<double> kvecs_ibz{
        0.0, 0.0, 0.0,
        librpa_int::TWO_PI / 3.0, 0.0, 0.0};
    const std::vector<std::vector<Vector3_Order<double>>> full_kstars{
        {{0.0, 0.0, 0.0}},
        {{1.0 / 3.0, 0.0, 0.0}, {-1.0 / 3.0, 0.0, 0.0}}};
    pbc.set_irreducible_kgrids_kvec(3, 1, 1, kvecs_ibz, full_kstars);
    return pbc;
}

SymmetryContext make_two_atom_inversion_context(const PeriodicBoundaryData &pbc)
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(
        lattice, lattice,
        {{0, 0}, {1, 0}},
        {{0, {0.25, 0.0, 0.0}}, {1, {0.75, 0.0, 0.0}}});

    SymmetryOperation identity;
    identity.rotation.Identity();
    identity.translation = {0.0, 0.0, 0.0};

    SymmetryOperation inversion;
    inversion.rotation = Matrix3(-1.0, 0.0, 0.0,
                                  0.0, 1.0, 0.0,
                                  0.0, 0.0, 1.0);
    inversion.translation = {0.0, 0.0, 0.0};

    ctx.set_rspace_operations({identity, inversion});
    ctx.set_available();
    ctx.build_periodic_mappings(pbc, pbc.Rlist);
    ctx.build_rsh_rotations({-1,
                             0,
                             LIBRPA_ANGULAR_ORDER_NATURAL,
                             LIBRPA_RSH_COEFF_1_M,
                             LIBRPA_RSH_COEFF_1_M},
                            0);
    ctx.build_kstar_member_rotations(0);
    return ctx;
}

Matz dense_wq_from_scalar_blocks(const librpa_int::symmetry_atom_block_matrix_map_t &blocks,
                                 const ArrayDesc &desc)
{
    Matz mat(desc.m_loc(), desc.n_loc(), MAJOR::COL);
    for (int i_local = 0; i_local < desc.m_loc(); ++i_local)
    {
        const int atom_i = desc.indx_l2g_r(i_local);
        for (int j_local = 0; j_local < desc.n_loc(); ++j_local)
        {
            const int atom_j = desc.indx_l2g_c(j_local);
            mat(i_local, j_local) = blocks.at(static_cast<atom_t>(atom_i))
                                        .at(static_cast<atom_t>(atom_j))(0, 0);
        }
    }
    return mat;
}

void assert_dense_wq_rspace_maps_close(
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &actual,
    const std::map<double, std::map<Vector3_Order<int>, Matz>> &expected)
{
    for (const auto &[freq, expected_Rs] : expected)
    {
        assert(actual.count(freq) != 0);
        for (const auto &[R, expected_mat] : expected_Rs)
        {
            assert(actual.at(freq).count(R) != 0);
            const auto diff = actual.at(freq).at(R) - expected_mat;
            double max_abs = 0.0;
            for (int i = 0; i < diff.nr(); ++i)
            {
                for (int j = 0; j < diff.nc(); ++j)
                {
                    max_abs = std::max(max_abs, std::abs(diff(i, j)));
                }
            }
            if (max_abs >= 1e-12)
            {
                std::cerr << "freq=" << freq << " R=(" << R.x << "," << R.y << "," << R.z
                          << ") max_abs=" << max_abs << std::endl;
                for (int i = 0; i < diff.nr(); ++i)
                {
                    for (int j = 0; j < diff.nc(); ++j)
                    {
                        std::cerr << "  (" << i << "," << j << ") actual="
                                  << actual.at(freq).at(R)(i, j) << " expected="
                                  << expected_mat(i, j) << " diff=" << diff(i, j) << std::endl;
                    }
                }
            }
            assert(max_abs < 1e-12);
        }
    }
}

void test_wq_to_wr_symmetry_reduced_q_matches_full_bz()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    const auto q_gamma_sym = pbc_sym.klist.at(0);
    const auto q_rep_sym = pbc_sym.klist.at(1);
    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
    add_scalar_wq_blocks(wq_sym, q_gamma_sym, gamma_blocks);
    add_scalar_wq_blocks(wq_sym, q_rep_sym, rep_blocks);
    const double symmetry_collective_scale =
        1.0 / static_cast<double>(librpa_int::global::mpi_comm_global_h.nprocs);
    for (auto &[atom_i, row] : wq_sym)
    {
        for (auto &[atom_j, q_blocks] : row)
        {
            for (auto &[q, block] : q_blocks)
            {
                block *= symmetry_collective_scale;
            }
        }
    }

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
}

void test_wq_to_wr_symmetry_collective_handles_empty_local_rank()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        const auto gamma_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
            {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
        const auto rep_blocks = scalar_wq_to_blocks({
            {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
            {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
        const auto inversion_minus_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
            {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});

        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_sym, pbc_sym.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);
    }

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto actual = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_sym,
        dummy_tfg, pbc_sym, pbc_sym.Rlist, false, "", true);

    assert_wq_rspace_maps_close(actual, expected);
    if (!librpa_int::global::mpi_comm_global_h.is_root())
    {
        assert(expected.empty());
        assert(actual.empty());
    }
}

void test_wq_to_wr_full_grid_k_input_restores_from_ibz_stars()
{
    // FHI-aims hands the shrink run the full-BZ SCF k list, so the symmetry
    // context folds that list into IBZ q-stars while pbc.klist keeps every
    // full-grid point.  The W(R) restore must pick the star representatives
    // and still reproduce the direct full-BZ transform.
    const auto pbc_full = make_wq_full_pbc();
    auto ctx_full = make_two_atom_inversion_context(pbc_full);
    assert(ctx_full.kstars.size() < pbc_full.kfrac_list.size());
    assert(ctx_full.count_kstar_members() == pbc_full.kfrac_list.size());
    assert(ctx_full.kstar_grid_mapping.size() == pbc_full.kfrac_list.size());

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});

    // The production shrink path evaluates Wc only at the IBZ q points.
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_full;
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_ibz;
    const auto representative_indices =
        librpa_int::build_symmetry_full_grid_kstar_representative_indices(
            ctx_full, pbc_full.kfrac_list);
    assert(representative_indices.size() == 2);
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        const auto gamma_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
            {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
        const auto rep_blocks = scalar_wq_to_blocks({
            {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
            {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});
        const auto inversion_minus_blocks = scalar_wq_to_blocks({
            {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
            {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});

        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(0), gamma_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(1), rep_blocks);
        add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), inversion_minus_blocks);

        for (const auto ik: representative_indices)
        {
            const auto &kfrac = pbc_full.kfrac_list.at(ik);
            if (std::abs(kfrac.x) < 1e-8)
            {
                add_scalar_wq_blocks(wq_ibz, pbc_full.klist.at(ik), gamma_blocks);
            }
            else if (kfrac.x < 0.5)
            {
                add_scalar_wq_blocks(wq_ibz, pbc_full.klist.at(ik), rep_blocks);
            }
            else
            {
                add_scalar_wq_blocks(wq_ibz, pbc_full.klist.at(ik), inversion_minus_blocks);
            }
        }
    }

    const TFGrids dummy_tfg;
    SymmetryContext no_symmetry;
    const auto expected = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, no_symmetry, wq_full,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", false);
    const auto restored = librpa_int::FT_Wc_q2R(
        librpa_int::global::mpi_comm_global_h, basis_abf, ctx_full, wq_ibz,
        dummy_tfg, pbc_full, pbc_full.Rlist, false, "", true);

    assert_wq_rspace_maps_close(restored, expected);
    if (!librpa_int::global::mpi_comm_global_h.is_root())
    {
        assert(expected.empty());
        assert(restored.empty());
    }
}

void test_wq_to_wr_full_grid_k_input_rejects_unresolved_representatives()
{
    // A full-BZ k list whose k-stars cannot be folded onto distinct
    // representative k-points must be reported, not silently treated as a
    // reduced input.  Falling through would walk the grid mapping once per
    // member k-point and repeat every star, inflating W(R) without a warning.
    const auto pbc_full = make_wq_full_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_full);
    assert(ctx.kstars.size() < pbc_full.kfrac_list.size());

    // Collapse every star onto the same representative k-point so the shared
    // builder cannot resolve a distinct index per star and returns an empty list.
    for (auto& star : ctx.kstars)
    {
        star.k_ibz = pbc_full.kfrac_list.at(0);
    }
    assert(librpa_int::build_symmetry_full_grid_kstar_representative_indices(
               ctx, pbc_full.kfrac_list)
               .empty());

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_ibz;
    if (librpa_int::global::mpi_comm_global_h.is_root())
    {
        add_scalar_wq_blocks(wq_ibz, pbc_full.klist.at(0),
                             scalar_wq_to_blocks({{0, {{0, {1.5, 0.0}}}},
                                                  {1, {{1, {1.5, 0.0}}}}}));
    }

    const TFGrids dummy_tfg;
    bool raised = false;
    std::string message;
    try
    {
        librpa_int::FT_Wc_q2R(librpa_int::global::mpi_comm_global_h, basis_abf, ctx, wq_ibz,
                              dummy_tfg, pbc_full, pbc_full.Rlist, false, "", true);
    }
    catch (const std::exception& ex)
    {
        raised = true;
        message = ex.what();
    }
    assert(raised);
    assert(message.find("no usable k-star representatives") != std::string::npos);
}

void test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(const BlacsCtxtHandler &blacs_h)
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_two_atom_inversion_context(pbc_sym);
    const auto qpoint_view = build_symmetry_qpoint_view(ctx, pbc_sym, true);
    assert(qpoint_view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);

    AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    basis_abf.set_l_shells({{0}, {0}});
    const auto layouts = basis_abf.build_species_basis_layouts(ctx.atom_to_type);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}};
    ArrayDesc ad_Wc(blacs_h);
    ad_Wc.init(2, 2, 2, 2, 0, 0);

    const auto gamma_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.5, 0.0}}, {1, {0.4, 0.0}}}},
        {1, {{0, {0.4, 0.0}}, {1, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({
        {0, {{0, {2.1, 0.0}}, {1, {-0.7, 0.5}}}},
        {1, {{0, {-0.7, -0.5}}, {1, {1.4, 0.0}}}}});

    constexpr double freq = 0.25;
    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_sym;
    wq_sym[freq][pbc_sym.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_sym[freq][pbc_sym.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);

    std::map<double, std::map<Vector3_Order<double>, Matz>> wq_full;
    wq_full[freq][pbc_full.klist.at(0)] = dense_wq_from_scalar_blocks(gamma_blocks, ad_Wc);
    wq_full[freq][pbc_full.klist.at(1)] = dense_wq_from_scalar_blocks(rep_blocks, ad_Wc);
    const auto inversion_minus_blocks = scalar_wq_to_blocks({
        {0, {{0, {1.4, 0.0}}, {1, {-0.7, -0.5}}}},
        {1, {{0, {-0.7, 0.5}}, {1, {2.1, 0.0}}}}});
    wq_full[freq][pbc_full.klist.at(2)] =
        dense_wq_from_scalar_blocks(inversion_minus_blocks, ad_Wc);

    const auto expected = librpa_int::FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, wq_full, pbc_full, false);
    const auto actual = librpa_int::FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, wq_sym, pbc_sym, false,
        &qpoint_view, &ctx, &basis_abf, &ad_Wc);

    assert_dense_wq_rspace_maps_close(actual, expected);
}

void test_gamma_only_dense_wq_fourier_weight_scales_as_inverse_bvk_cells()
{
    constexpr double frequency = 0.25;
    const std::complex<double> gamma_value{2.4, -0.3};
    const std::array<int, 4> meshes{12, 14, 16, 20};

    for (const int mesh : meshes)
    {
        PeriodicBoundaryData pbc;
        pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        std::vector<double> kvecs;
        kvecs.reserve(static_cast<std::size_t>(3 * mesh * mesh));
        for (int ix = 0; ix != mesh; ++ix)
        {
            for (int iy = 0; iy != mesh; ++iy)
            {
                kvecs.push_back(librpa_int::TWO_PI * ix / mesh);
                kvecs.push_back(librpa_int::TWO_PI * iy / mesh);
                kvecs.push_back(0.0);
            }
        }
        pbc.set_kgrids_kvec(mesh, mesh, 1, kvecs);

        std::map<double, std::map<Vector3_Order<double>, Matz>> wq;
        if (librpa_int::global::mpi_comm_global_h.is_root())
        {
            Matz gamma(1, 1, MAJOR::COL);
            gamma(0, 0) = gamma_value;
            wq[frequency][pbc.klist.at(0)] = gamma;
        }

        const auto wr =
            librpa_int::FT_Wc_freq_q(librpa_int::global::mpi_comm_global_h, wq, pbc, false);
        if (librpa_int::global::mpi_comm_global_h.is_root())
        {
            const Vector3_Order<int> center{0, 0, 0};
            const auto expected = gamma_value / static_cast<double>(mesh * mesh);
            assert_complex_close(wr.at(frequency).at(center)(0, 0), expected, 1e-13);
        }
        else
        {
            assert(wr.empty());
        }
    }
}

#ifdef LIBRPA_USE_LIBRI
void test_native_optical_uses_handler_communicator()
{
    const int world_rank = librpa_int::get_mpi_rank(MPI_COMM_WORLD);
    const int color = world_rank % 2;
    MPI_Comm subgroup;
    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &subgroup);
    int subgroup_rank = 0;
    MPI_Comm_rank(subgroup, &subgroup_rank);
    double error = 0.0;
    const auto check = [&](const std::complex<double> actual, const std::complex<double> expected)
    {
        const double difference = std::abs(actual - expected);
        error = !std::isfinite(difference) ? std::numeric_limits<double>::infinity()
                                           : std::max(error, difference);
    };
    const double scale = color + 1.0;
    const std::array<std::complex<double>, 3> dipole{scale * std::complex<double>{0.2, 0.1},
                                                     scale * std::complex<double>{0.3, -0.15},
                                                     scale * std::complex<double>{-0.4, 0.05}};
    librpa::OpticalResult snapshot(nullptr, librpa_delete_optical_result);
    {
        librpa::Handler handler(subgroup);
        const auto ds = librpa_int::api::get_dataset_instance(handler);
        ds->mf = MeanField(1, 1, 2, 2);
        ds->mf.get_eigenvals()[0](0, 0) = -0.5;
        ds->mf.get_eigenvals()[0](0, 1) = 0.5;
        ds->mf.get_weight()[0](0, 0) = 2.0;
        ds->mf.get_weight()[0](0, 1) = 0.0;
        auto &wfc = ds->mf.get_eigenvectors()[0][0][0];
        wfc.create(2, 2);
        wfc(0, 0) = wfc(1, 1) = 1.0;
        ds->basis_wfc = AtomicBasis({2});
        ds->basis_aux = AtomicBasis({1});
        ds->pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
        ds->pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});

        // Distinct subgroup dipoles and charge vertices must not cross-sum.
        // The two-level model has C=S=I, gap=1 Ha and v_01=-i*r_01.
        const std::array<double, 8> h_values{-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0};
        const std::array<double, 8> s_values{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
        std::array<std::array<double, 8>, 3> r_values{};
        LibrpaLocalizedBlock h_block{{0, 0, 0}, h_values.data(), 1};
        LibrpaLocalizedBlock s_block{{0, 0, 0}, s_values.data(), 1};
        std::array<LibrpaLocalizedBlock, 3> r_blocks{};
        LibrpaLocalizedOperator h_operator{2, 1, &h_block};
        LibrpaNativeVelocityInput input{};
        input.n_spins = 1;
        input.hamiltonian = &h_operator;
        input.overlap = {2, 1, &s_block};
        for (int axis = 0; axis != 3; ++axis)
        {
            r_values[axis][2] = dipole[axis].real();
            r_values[axis][3] = dipole[axis].imag();
            r_blocks[axis] = {{0, 0, 0}, r_values[axis].data(), 1};
            input.position[axis] = {2, 1, &r_blocks[axis]};
        }
        ds->cs_data.use_libri = true;
        if (ds->comm_h.is_root())
        {
            auto tensor = std::make_shared<std::valarray<double>>(4);
            (*tensor)[1] = 0.125 * scale;
            (*tensor)[2] = 0.25 * scale;
            ds->cs_data.data_libri[0][{0, {0, 0, 0}}] =
                RI::Tensor<double>({1UL, 2UL, 2UL}, tensor);
        }
        librpa::Options opts;
        opts.option_dielect_func = 3;
        opts.replace_w_head = LIBRPA_SWITCH_ON;
        opts.use_scalapack_gw_wc = LIBRPA_SWITCH_ON;
        opts.tfgrids_type = LIBRPA_TFGRID_EVEN_SPACED_TF;
        opts.nfreq = 2;
        opts.tfgrids_freq_min = 0.5;
        opts.tfgrids_freq_interval = 0.75;
        handler.prepare_native_velocity(opts, ds->comm_h.is_root() ? &input : nullptr);
        snapshot = handler.get_optical_result();
        const auto check_wing = [&]
        {
            const auto &frequencies = ds->p_headwing->get_frequency_nodes();
            const auto &wings = ds->p_headwing->get_wing_auxiliary();
            for (std::size_t iw = 0; iw != frequencies.size(); ++iw)
                for (int axis = 0; axis != 3; ++axis)
                {
                    const auto v01 = std::complex<double>{0.0, -1.0} * dipole[axis];
                    const auto expected = -4.0 * std::sqrt(4.0 * librpa_int::PI) *
                                          (0.375 * scale) * v01 /
                                          (1.0 + frequencies[iw] * frequencies[iw]);
                    check(wings[iw](0, axis), expected);
                }
        };
        check_wing();
        const auto &frequencies = ds->p_headwing->get_frequency_nodes();
        const auto &heads = ds->p_headwing->get_head_tensors();
        for (int axis = 0; axis != 3; ++axis)
        {
            const auto v01 = std::complex<double>{0.0, -1.0} * dipole[axis];
            check(ds->velocity_matrix[0][0][axis](0, 1), v01);
            check(ds->velocity_matrix[0][0][axis](1, 0), std::conj(v01));
            for (std::size_t iw = 0; iw != frequencies.size(); ++iw)
                for (int beta = 0; beta != 3; ++beta)
                {
                    const auto expected = (axis == beta ? 1.0 : 0.0) +
                                          16.0 * librpa_int::PI * std::conj(dipole[axis]) *
                                              dipole[beta] /
                                              (1.0 + frequencies[iw] * frequencies[iw]);
                    check(heads[iw](axis, beta), expected);
                }
        }

        // Exercise the other wing reduction with the same analytic reference.
        SymmetryContext context;
        context.set_available();
        context.atom_to_type = {{0, 0}};
        ds->basis_wfc.set_l_shells({{0, 0}});
        librpa_int::SymmetryKStar star;
        star.k_ibz = {0.0, 0.0, 0.0};
        star.members.resize(1);
        star.members[0].k_bz = star.k_ibz;
        star.members[0].atom_rotations.resize(1);
        star.members[0].atom_rotations[0].atom_from = 0;
        star.members[0].atom_rotations[0].atom_to = 0;
        context.kstars = {star};
        ds->p_headwing->set_symmetry_context(context);
        ds->p_headwing->use_symmetry = true;
        ds->p_headwing->atom_nw = {{0, 2}};
        ds->p_headwing->set_direct_full_bz_headwing_inputs(ds->velocity_matrix, ds->mf, {{0}});
        ds->p_headwing->cal_wing(ds->cs_data, 0.0, ds->vq);
        check_wing();
    }
    // The public snapshot owns numerical arrays independently of the handler.
    // Check its storage layout and surviving values against the same model.
    require_condition(snapshot && snapshot->n_spins == 1 && snapshot->n_kpoints == 1 &&
                          snapshot->n_states == 2 && snapshot->n_frequencies == 2 &&
                          snapshot->head_rows == 3 && snapshot->head_columns == 3 &&
                          snapshot->wing_rows == 1 && snapshot->wing_columns == 3,
                      "Native optical snapshot has incompatible model dimensions");
    const auto view = librpa_get_optical_view(snapshot.get());
    const auto packed = [](const double *values, int entry)
    { return std::complex<double>{values[2 * entry], values[2 * entry + 1]}; };
    for (int iw = 0; iw != view.n_frequencies; ++iw)
    {
        const double frequency = 0.5 + 0.75 * iw;
        check(view.frequencies_ha[iw], frequency);
        for (int axis = 0; axis != 3; ++axis)
        {
            const auto v01 = std::complex<double>{0.0, -1.0} * dipole[axis];
            check(packed(view.velocity_ri, axis * 4 + 1), v01);
            check(packed(view.velocity_ri, axis * 4 + 2), std::conj(v01));
            const auto expected_wing = -4.0 * std::sqrt(4.0 * librpa_int::PI) * (0.375 * scale) *
                                       v01 / (1.0 + frequency * frequency);
            check(packed(view.wing_ri, iw * 3 + axis), expected_wing);
            for (int beta = 0; beta != 3; ++beta)
            {
                const auto expected_head =
                    (axis == beta ? 1.0 : 0.0) + 16.0 * librpa_int::PI * std::conj(dipole[axis]) *
                                                     dipole[beta] / (1.0 + frequency * frequency);
                check(packed(view.head_ri, iw * 9 + axis * 3 + beta), expected_head);
            }
        }
    }
    if (subgroup_rank == 0)
        std::cout << "Native optical subgroup " << color << " max error " << error << '\n';
    MPI_Comm_free(&subgroup);
    double maximum = 0.0;
    MPI_Allreduce(&error, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    require_condition(std::isfinite(maximum) && maximum < 1e-12,
                      "Native head/wing mixed independent handler communicators");
}
#endif

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

#ifdef LIBRPA_USE_LIBRI
        test_native_optical_uses_handler_communicator();
#endif
        test_headwing_body_inverse_uses_identity_solve(blacs_h);
        test_replace_rpa_response_headwing_replaces_only_singular_channels(blacs_h);
        test_gamma_head_rank_one_matches_coulomb_basis_overwrite(blacs_h);
        test_gamma_head_rank_one_handles_empty_local_blocks(blacs_h);
        test_rspace_symmetry_requires_complete_band_space();
        test_qsgw_reconstruction_with_finite_reference_metric();
        test_qsgw_native_baseline_preserves_aliased_tail();
        test_kpoint_coordinate_mapping_selects_active_klist_from_full_source();
        test_kstar_velocity_mapping_preserves_member_order_and_periodic_gauge();
        test_replace_rpa_response_head_only_keeps_numeric_wings(blacs_h);
        test_rpa_trace_log_average_uses_directional_head_and_wing();
        test_rpa_headwing_regular_body_start_channel();
        test_rpa_headwing_gamma_cell_volume_uses_reciprocal_lattice();
        test_strict_2d_headwing_prefactors_use_inplane_area();
        test_strict_2d_auxiliary_normalization_is_computed_from_basis_metadata();
        test_strict_2d_gamma_cell_uses_physical_reciprocal_measure();
        test_strict_2d_radial_integrals_match_analytic_values();
        test_strict_2d_radial_log_integral_matches_numeric_quadrature();
        test_strict_2d_radial_log_integral_is_stable_at_zero_and_small_a();
        test_strict_2d_rpa_trace_log_average_matches_dense_radial_quadrature();
        test_strict_2d_inverse_head_average_has_linear_q_screening();
        test_strict_2d_schur_coefficient_removes_identity();
        test_strict_2d_screening_denominator_must_stay_on_physical_branch();
        test_strict_2d_gw_uses_full_coulomb_at_all_q();
        test_strict_2d_gw_routes_gamma_through_complete_wc_average();
        test_strict_2d_gw_fails_closed_for_incomplete_runtime_configuration();
        test_strict_2d_pw_wc_transforms_to_auxiliary_coulomb_basis();
        test_strict_2d_wc_blocks_match_dense_finite_q_inverse();
        test_strict_2d_wc_cell_average_matches_anisotropic_radial_quadrature();
        test_strict_2d_wc_cell_average_matches_cartesian_voronoi_subgrid();
        test_strict_2d_wc_blocks_have_finite_small_q_limits();
        test_strict_2d_wc_average_is_covariant_under_regular_body_rotation();
        test_strict_2d_wc_average_is_bounded_as_gamma_cell_shrinks();
        test_rpa_chi0v_wing_desc_matches_producer_layout(blacs_h);
        test_rpa_headwing_matrix_descriptor_contract(blacs_h);
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
        test_ibz_soc_wfc_velocity_matches_nonorthogonal_ao_oracle();
        test_ibz_soc_headwing_matches_complete_complex_bz(blacs_h);
        test_headwing_direct_full_bz_velocity_selects_kstar_member();
        test_headwing_direct_full_bz_wfc_selects_same_kstar_member();
        test_headwing_direct_full_bz_wfc_local_block(blacs_h);
        test_kblacs_transform_with_restored_wfc_matches_full_bz_atom_permutation(blacs_h);
        test_kblacs_transform_matches_original_transform(blacs_h);
        test_kblacs_transform_uses_rectangular_opt_128_blocks();
        test_transform_Cs2mnk_can_keep_spin_channels_separate(blacs_h);
        test_soc_charge_vertex_is_spin_trace_and_rotation_invariant(blacs_h);
#ifdef LIBRPA_USE_LIBRI
        test_complex_soc_bubble_matches_lindhard();
#endif
        test_head_initialization_does_not_require_coulomb_diagonalization(blacs_h);
        test_spacetime_fourier_phases_form_k_minus_q_convolution();
        test_bn_qstar_wq_to_wr_matches_explicit_full_bz_on_odd_and_even_meshes();
        test_bn_qstar_wq_to_wr_without_rspace_sector_matches_explicit_full_bz();
        test_bn_qstar_vq_to_vr_without_rspace_sector_matches_explicit_full_bz();
        test_bn_kstar_green_function_matches_explicit_full_bz_on_odd_and_even_meshes();
        test_wq_to_wr_symmetry_reduced_q_matches_full_bz();
        test_wq_to_wr_symmetry_collective_handles_empty_local_rank();
        test_wq_to_wr_full_grid_k_input_restores_from_ibz_stars();
        test_wq_to_wr_full_grid_k_input_rejects_unresolved_representatives();
        test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(blacs_h);
        test_gamma_only_dense_wq_fourier_weight_scales_as_inverse_bvk_cells();
        test_abf_space_wing_rewrite_matches_coulomb_basis(blacs_h);
    }

    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
