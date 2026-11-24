#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <vector>

#include "../core/epsilon.h"
#include "../core/qpoint_view.h"
#include "../io/global_io.h"
#include "../math/lebedev_laikov.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../utils/constants.h"
#include "mpi_test_config.h"

using namespace librpa_int;

namespace
{

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

void test_gamma_s_head_vector(const BlacsCtxtHandler &blacs_h)
{
    const std::vector<double> frequencies{0.1, 1.0}, dielectric{12.0, 4.0};
    assert(interpolate_dielec_func(5, frequencies, dielectric, frequencies) == dielectric);
    assert(interpolate_dielec_func(0, frequencies, dielectric, frequencies) == dielectric);
    AtomicBasis basis(std::vector<std::size_t>{5, 5});
    basis.set_l_shells({{0, 1, 0}, {0, 1, 0}});
    const std::vector<double> expected{0.5, 0, 0, 0, -0.5, 0.5, 0, 0, 0, -0.5};
    matrix_m<std::complex<double>> dense(10, 10, MAJOR::COL);
    for (int j = 0; j < 10; ++j)
        for (int i = 0; i < 10; ++i)
            dense(i, j) = (i == j ? (expected[i] == 0 ? 20.0 : 1.0) : 0.0)
                          + 6.0 * expected[i] * expected[j];
    // The full leading mode is p-like; nonzero s-p wings also make the desired
    // zero-padded vector different from any full Coulomb eigenvector.
    dense(0, 1) = dense(1, 0) = 0.2;
    const auto serial = compute_gamma_s_head_vector(basis, dense);
    ArrayDesc desc(blacs_h);
    desc.init(10, 10, 2, 2, 0, 0);
    auto local = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int j = 0; j < desc.n_loc(); ++j)
        for (int i = 0; i < desc.m_loc(); ++i)
        {
            const int gi = desc.indx_l2g_r(i), gj = desc.indx_l2g_c(j);
            // Distributed Coulomb assembly only guarantees the upper triangle.
            local(i, j) = gi <= gj ? dense(gi, gj) : std::complex<double>{99.0, 0.0};
        }
    const auto parallel = compute_gamma_s_head_vector(basis, local, &desc);
    for (const auto &actual : {serial, parallel})
    {
        double overlap = 0.0, norm = 0.0;
        for (int i = 0; i < 10; ++i)
        {
            overlap += actual[i] * expected[i];
            norm += actual[i] * actual[i];
            if (expected[i] == 0.0) assert(actual[i] == 0.0);
        }
        assert(std::abs(norm - 1.0) < 1e-12);
        assert(std::abs(std::abs(overlap) - 1.0) < 1e-12);
    }
    for (bool missing_metadata : {false, true})
    {
        AtomicBasis invalid(std::vector<std::size_t>{3});
        if (!missing_metadata) invalid.set_l_shells({{1}});
        matrix_m<std::complex<double>> matrix(3, 3, MAJOR::COL);
        bool rejected = false;
        try { compute_gamma_s_head_vector(invalid, matrix); }
        catch (const std::exception &) { rejected = true; }
        assert(rejected);
    }
}

void test_head_only_trace_logdet_can_use_reduced_response(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(3, 3, 0, 0);

    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo < 0) continue;
            response(ilo, jlo) =
                std::complex<double>{0.02 * (i + 1) + 0.01 * (j + 1), 0.002 * (i - j)};
        }
    }

    const auto actual = librpa_int::compute_rpa_response_trace_logdet_blacs_2d(response, desc);

    const matrix_m<std::complex<double>> dense_response(
        std::vector<std::vector<std::complex<double>>>{
            {{0.03, 0.0}, {0.04, -0.002}, {0.05, -0.004}},
            {{0.05, 0.002}, {0.06, 0.0}, {0.07, -0.002}},
            {{0.07, 0.004}, {0.08, 0.002}, {0.09, 0.0}}},
        MAJOR::COL);
    std::complex<double> trace = 0.0;
    for (int i = 0; i != 3; ++i)
    {
        trace += dense_response(i, i);
    }
    const auto a = 1.0 - dense_response(0, 0);
    const auto b = -dense_response(0, 1);
    const auto c = -dense_response(0, 2);
    const auto d = -dense_response(1, 0);
    const auto e = 1.0 - dense_response(1, 1);
    const auto f = -dense_response(1, 2);
    const auto g = -dense_response(2, 0);
    const auto h = -dense_response(2, 1);
    const auto i = 1.0 - dense_response(2, 2);
    const auto det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    const auto expected = trace + std::log(det);

    assert_complex_close(actual, expected, 1e-12);

    ArrayDesc desc_full(blacs_h);
    desc_full.init_square_blk(5, 5, 0, 0);
    auto full_response = init_local_mat<std::complex<double>>(desc_full, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int ilo = desc_full.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int jlo = desc_full.indx_g2l_c(j);
            if (jlo < 0) continue;
            full_response(ilo, jlo) = dense_response(i, j);
        }
    }
    const auto full_actual =
        librpa_int::compute_rpa_response_trace_logdet_blacs_2d(full_response, desc_full);

    assert_complex_close(full_actual, actual, 1e-12);
}

void test_trace_logdet_accounts_for_complex_row_pivot(const BlacsCtxtHandler &blacs_h)
{
    ArrayDesc desc(blacs_h);
    desc.init_square_blk(2, 2, 0, 0);

    // response = I - A, where A = [[1, 2i], [-2i, 5]] is positive definite
    // with det(A) = 1. PZGETRF must pivot the first row because |-2i| > |1|.
    const std::complex<double> response_dense[2][2] = {{{0.0, 0.0}, {0.0, -2.0}},
                                                       {{0.0, 2.0}, {-4.0, 0.0}}};
    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i != 2; ++i)
    {
        const int ilo = desc.indx_g2l_r(i);
        if (ilo < 0) continue;
        for (int j = 0; j != 2; ++j)
        {
            const int jlo = desc.indx_g2l_c(j);
            if (jlo >= 0) response(ilo, jlo) = response_dense[i][j];
        }
    }

    const auto actual = librpa_int::compute_rpa_response_trace_logdet_blacs_2d(response, desc);
    assert_complex_close(actual, {-4.0, 0.0}, 1e-12);
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

/*!
 * Single-atom s-orbital toy with an explicit antiunitary-only "magnetic"
 * table {E, Theta}: the q-star of q=1/3 gains the member -1/3 through Theta
 * (k -> -k, no spatial rotation), so W(-1/3) must be the plain complex
 * conjugate of W(1/3). Verifies that the member time_reversal flag produced
 * by the explicit spin-operation table drives the conjugation in the
 * symmetry W(q)->W(R) accumulation.
 */
SymmetryContext make_single_atom_antiunitary_context(const PeriodicBoundaryData &pbc)
{
    SymmetryContext ctx;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    ctx.set_crystal_structure(
        lattice, lattice,
        {{0, 0}},
        {{0, {0.0, 0.0, 0.0}}});

    SymmetryOperation identity;
    identity.rotation.Identity();
    identity.translation = {0.0, 0.0, 0.0};
    ctx.set_rspace_operations({identity});

    SymmetrySpinOperation spin_unitary;  // E, identity spin
    SymmetrySpinOperation spin_anti;     // Theta: k -> -k, conjugation
    spin_anti.antiunitary = true;
    ctx.set_symmetry_spin_operations({spin_unitary, spin_anti}, false);

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

void test_wq_to_wr_antiunitary_member_conjugates()
{
    const auto pbc_full = make_wq_full_pbc();
    const auto pbc_sym = make_wq_reduced_pbc();
    auto ctx = make_single_atom_antiunitary_context(pbc_sym);

    AtomicBasis basis_abf(std::vector<std::size_t>{1});
    basis_abf.set_l_shells({{0}});

    const auto q_gamma_sym = pbc_sym.klist.at(0);
    const auto q_rep_sym = pbc_sym.klist.at(1);
    const auto gamma_blocks = scalar_wq_to_blocks({{0, {{0, {1.5, 0.0}}}}});
    const auto rep_blocks = scalar_wq_to_blocks({{0, {{0, {2.1, -0.7}}}}});

    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old
        wq_sym;
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
    // Theta maps q=1/3 to q=-1/3 with conjugation and no orbital rotation.
    const auto tr_blocks = scalar_wq_to_blocks({{0, {{0, {2.1, 0.7}}}}});
    add_scalar_wq_blocks(wq_full, pbc_full.klist.at(2), tr_blocks);

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

void test_dense_wc_real_ct_matches_legacy_real_part()
{
    const auto pbc = make_wq_full_pbc();
    TFGrids tfg(2);
    tfg.generate_evenspaced_tf(0.1, 0.2, 0.3, 0.4);

    std::map<double, std::map<Vector3_Order<double>, Matz>> input;
    const auto freq_nodes = tfg.get_freq_nodes();
    for (std::size_t ifreq = 0; ifreq != freq_nodes.size(); ++ifreq)
    {
        for (std::size_t iq = 0; iq != pbc.klist.size(); ++iq)
        {
            Matz mat(2, 2, MAJOR::COL);
            for (std::size_t i = 0; i != mat.size(); ++i)
            {
                const double real = 1.0 + ifreq + 0.1 * i;
                const double imag = 0.2 + 0.05 * i;
                if (iq == 0)
                    mat.ptr()[i] = {real, 0.0};
                else if (iq == 1)
                    mat.ptr()[i] = {real, imag};
                else
                    mat.ptr()[i] = {real, -imag};
            }
            input[freq_nodes[ifreq]][pbc.klist[iq]] = std::move(mat);
        }
    }

    auto legacy_input = input;
    auto real_input = input;
    const auto legacy = librpa_int::CT_FT_Wc_freq_q(
        librpa_int::global::mpi_comm_global_h, legacy_input, pbc, tfg, true);
    const auto actual = librpa_int::CT_FT_Wc_freq_q_real(
        librpa_int::global::mpi_comm_global_h, real_input, pbc, tfg);
    assert(legacy_input.empty());
    assert(real_input.empty());

    for (const auto &[tau, expected_Rs] : legacy)
    {
        assert(actual.count(tau) != 0);
        for (const auto &[R, expected] : expected_Rs)
        {
            assert(actual.at(tau).count(R) != 0);
            const auto &result = actual.at(tau).at(R);
            assert(result.major() == expected.major());
            assert(result.size() == expected.size());
            for (std::size_t i = 0; i != result.size(); ++i)
                assert(std::abs(result.ptr()[i] - expected.ptr()[i].real()) < 1e-12);
        }
    }
}

void test_dense_wc_real_ct_projects_large_imaginary_residual()
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0,
                    0.0, 1.0, 0.0,
                    0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(1, 1, 1, {0.0, 0.0, 0.0});
    TFGrids tfg(1);
    tfg.generate_evenspaced_tf(0.1, 0.1, 0.2, 0.1);

    std::map<double, std::map<Vector3_Order<double>, Matz>> input;
    Matz gamma(1, 1, MAJOR::COL);
    gamma(0, 0) = {1.0, 1e-4};
    input[tfg.get_freq_nodes().front()][pbc.klist.front()] = std::move(gamma);

    const auto projected = librpa_int::CT_FT_Wc_freq_q_real(
        librpa_int::global::mpi_comm_global_h, input, pbc, tfg);
    assert(input.empty());
    assert(std::abs(projected.at(tfg.get_time_nodes().front()).at({0, 0, 0})(0, 0) - 1.0) <
           1e-12);
}

#ifdef LIBRPA_USE_LIBRI

Matz serial_inverse(const Matz &input)
{
    const int n = input.nr();
    auto factor = input.copy();
    Matz inverse(n, n, MAJOR::COL);
    for (int i = 0; i != n; ++i) inverse(i, i) = 1.0;
    for (int k = 0; k != n; ++k)
    {
        int pivot = k;
        for (int i = k + 1; i != n; ++i)
            if (std::abs(factor(i, k)) > std::abs(factor(pivot, k))) pivot = i;
        assert(std::abs(factor(pivot, k)) > 1e-12);
        for (int j = 0; j != n; ++j)
        {
            std::swap(factor(k, j), factor(pivot, j));
            std::swap(inverse(k, j), inverse(pivot, j));
        }
        const auto diagonal = factor(k, k);
        for (int j = 0; j != n; ++j)
        {
            factor(k, j) /= diagonal;
            inverse(k, j) /= diagonal;
        }
        for (int i = 0; i != n; ++i)
        {
            if (i == k) continue;
            const auto multiplier = factor(i, k);
            for (int j = 0; j != n; ++j)
            {
                factor(i, j) -= multiplier * factor(k, j);
                inverse(i, j) -= multiplier * inverse(k, j);
            }
        }
    }
    return inverse;
}

void test_dense_wc_layout(const BlacsCtxtHandler &blacs_h)
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(3, 1, 1,
                        {0.0, 0.0, 0.0, 2.0 * PI / 3.0, 0.0, 0.0, -2.0 * PI / 3.0, 0.0, 0.0});
    AtomicBasis wfc_basis({2}), aux_basis(std::vector<std::size_t>{1, 1});
    MeanField mf(1, 3, 2, 2);
    for (int ik = 0; ik != 3; ++ik)
    {
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvals()[0](ik, 1) = 1.0;
    }
    TFGrids grids(3);
    grids.generate_evenspaced_tf(0.1, 0.1, 0.25, 0.1);
    KPointBlacsParallelContext kctx(KPointBlacsProcessShape(1, blacs_h.nprocs), MPI_COMM_WORLD, 3);
    ArrayDesc desc_wfc(blacs_h), desc_aux(blacs_h);
    desc_wfc.init(2, 2, 1, 1, 0, 0);
    desc_aux.init(2, 2, 1, 1, 0, 0);
    const SymmetryContext context;
    Chi0 chi0(mf, wfc_basis, aux_basis, pbc, context, grids, kctx, desc_wfc, false, false);
    using ResponseMap =
        std::map<double, std::map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>>;
    // Inject a fixed response to test the production Wc paths independently of chi0 construction.
    auto &response_blocks = const_cast<ResponseMap &>(chi0.get_chi0_q());
    atpair_k_cplx_mat_t bare_blocks, cut_blocks, parallel_cut_blocks;
    std::map<double, std::map<Vector3_Order<double>, Matz>> expected;
    for (std::size_t iq = 0; iq != pbc.klist.size(); ++iq)
    {
        const auto &q = pbc.klist[iq];
        const double phase = iq == 0 ? 0.0 : (iq == 1 ? 1.0 : -1.0);
        // Positive-definite, noncommuting square roots. At +/-q the matrices are conjugates.
        const Matz s({{{2.0, 0.0}, {0.2, 0.3 * phase}}, {{0.2, -0.3 * phase}, {1.5, 0.0}}},
                     MAJOR::COL);
        const Matz t({{{1.1, 0.0}, {-0.1, 0.2 * phase}}, {{-0.1, -0.2 * phase}, {0.8, 0.0}}},
                     MAJOR::COL);
        Matz bare(2, 2, MAJOR::COL), cut(2, 2, MAJOR::COL);
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j)
                for (int k = 0; k != 2; ++k)
                {
                    bare(i, j) += s(i, k) * s(k, j);
                    cut(i, j) += t(i, k) * t(k, j);
                }
        for (int i = 0; i != 2; ++i)
            for (int j = i; j != 2; ++j)
            {
                bare_blocks[i][j][q] = std::make_shared<ComplexMatrix>(1, 1);
                cut_blocks[i][j][q] = std::make_shared<ComplexMatrix>(1, 1);
                parallel_cut_blocks[i][j][q] = std::make_shared<ComplexMatrix>(1, 1);
                (*bare_blocks[i][j][q])(0, 0) = bare(i, j);
                (*cut_blocks[i][j][q])(0, 0) = cut(i, j);
                (*parallel_cut_blocks[i][j][q])(0, 0) = cut(i, j);
            }
        for (std::size_t iw = 0; iw != grids.get_freq_nodes().size(); ++iw)
        {
            const double freq = grids.get_freq_nodes()[iw];
            Matz response(
                {{{-0.3, 0.0}, {0.04, 0.05 * phase}}, {{0.04, -0.05 * phase}, {-0.2, 0.0}}},
                MAJOR::COL);
            response *= iw == 2 ? 0.0 : 1.0 / (iw + 1);
            for (int i = 0; i != 2; ++i)
                for (int j = i; j != 2; ++j)
                {
                    auto &block = response_blocks[freq][q][i][j];
                    block.create(1, 1);
                    block(0, 0) = response(i, j);
                }
            // Wc = T [(I - S chi0 S)^(-1) - I] T. Use scalar products and
            // the explicit 2x2 inverse, independent of the production BLAS/eigensolver calls.
            Matz eps(2, 2, MAJOR::COL);
            for (int i = 0; i != 2; ++i)
                for (int j = 0; j != 2; ++j)
                {
                    eps(i, j) = i == j ? 1.0 : 0.0;
                    for (int k = 0; k != 2; ++k)
                        for (int l = 0; l != 2; ++l)
                            eps(i, j) -= s(i, k) * response(k, l) * s(l, j);
                }
            const auto det = eps(0, 0) * eps(1, 1) - eps(0, 1) * eps(1, 0);
            Matz inverse({{eps(1, 1) / det - 1.0, -eps(0, 1) / det},
                          {-eps(1, 0) / det, eps(0, 0) / det - 1.0}},
                         MAJOR::COL);
            auto &wc = expected[freq][q];
            wc = Matz(2, 2, MAJOR::COL);
            for (int i = 0; i != 2; ++i)
                for (int j = 0; j != 2; ++j)
                    for (int k = 0; k != 2; ++k)
                        for (int l = 0; l != 2; ++l) wc(i, j) += t(i, k) * inverse(k, l) * t(l, j);
        }
    }
    const auto dense = compute_Wc_freq_q(chi0, bare_blocks, cut_blocks, 0.0, {}, false, ".", 0);
    // Dense input is replicated; ScaLAPACK collects atom blocks additively from their owners.
    if (blacs_h.myid != 0)
    {
        bare_blocks.clear();
        parallel_cut_blocks.clear();
        response_blocks.clear();
    }
    const auto parallel =
        compute_Wc_freq_q_blacs(chi0, bare_blocks, parallel_cut_blocks, 0.0, false, 0, {}, nullptr,
                                blacs_h, desc_aux, false, ".", false, false, false);
    double local_error = 0.0;
    const auto compare =
        [&](const std::complex<double> actual, const std::complex<double> reference)
    {
        if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
            !std::isfinite(reference.real()) || !std::isfinite(reference.imag()))
            local_error = std::numeric_limits<double>::infinity();
        else
            local_error = std::max(local_error, std::abs(actual - reference));
    };
    for (const auto &[freq, q_wc] : expected)
        for (const auto &[q, reference] : q_wc)
        {
            const auto &actual = dense.at(freq).at(q);
            assert(actual.major() == MAJOR::COL);
            for (int i = 0; i != 2; ++i)
                for (int j = 0; j != 2; ++j) compare(actual(i, j), reference(i, j));
            const auto &distributed = parallel.at(freq).at(q);
            for (int i = 0; i != desc_aux.m_loc(); ++i)
                for (int j = 0; j != desc_aux.n_loc(); ++j)
                    compare(distributed(i, j),
                            reference(desc_aux.indx_l2g_r(i), desc_aux.indx_l2g_c(j)));
        }
    double max_error = 0.0;
    MPI_Allreduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (blacs_h.myid == 0)
        std::cout << "Dense/ScaLAPACK Wc analytic control: max_error=" << max_error << std::endl;
    assert(max_error < 1e-12);
}

double rectangular_gamma_volume_ratio()
{
    // The fixture's 2x1x1 mesh in a unit reciprocal cube has Gamma box
    // half-widths (1/4, 1/2, 1/2) and volume 1/2. Finite angular quadrature
    // does not give exactly 1 even at zero response; retain that independent
    // head residual rather than changing the production quadrature.
    const auto grid = lebedev_laikov::grid(5810);
    double volume = 0.0;
    for (std::size_t i = 0; i != grid.weights.size(); ++i)
    {
        const double inverse_radius = std::max(
            {4.0 * std::abs(grid.x[i]), 2.0 * std::abs(grid.y[i]), 2.0 * std::abs(grid.z[i])});
        volume +=
            4.0 * PI * grid.weights[i] / (3.0 * inverse_radius * inverse_radius * inverse_radius);
    }
    return volume / 0.5;
}

void verify_gamma_wc_retained_identity(const BlacsCtxtHandler &blacs_h, const bool rank_deficient,
                                       const bool zero_response, const int dielectric_option,
                                       const bool use_cholesky, const int block_size = 1)
{
    constexpr int n = 4;
    const int retained = rank_deficient ? n - 1 : n;
    const std::array<double, n> eigenvalues{4.0, 2.0, 1.0, rank_deficient ? -0.2 : 0.5};
    const Matz eigenvectors({{{0.5, 0.0}, {0.5, 0.0}, {0.5, 0.0}, {0.5, 0.0}},
                             {{0.5, 0.0}, {-0.5, 0.0}, {0.5, 0.0}, {-0.5, 0.0}},
                             {{0.5, 0.0}, {0.5, 0.0}, {-0.5, 0.0}, {-0.5, 0.0}},
                             {{0.5, 0.0}, {-0.5, 0.0}, {-0.5, 0.0}, {0.5, 0.0}}},
                            MAJOR::COL);
    Matz diagonal(n, n, MAJOR::COL), sqrt_diagonal(n, n, MAJOR::COL);
    Matz cut_root_channels(n, n, MAJOR::COL), response_channels(n, n, MAJOR::COL);
    const double cut_diagonal[n]{1.2, 1.1, 1.4, 0.9};
    for (int i = 0; i != n; ++i)
    {
        diagonal(i, i) = eigenvalues[i];
        sqrt_diagonal(i, i) = std::sqrt(std::max(0.0, eigenvalues[i]));
        cut_root_channels(i, i) = cut_diagonal[i];
    }
    // The positive cut root couples retained and discarded bare-Coulomb
    // channels. Projecting the cut interaction too would fail this control.
    cut_root_channels(1, 3) = cut_root_channels(3, 1) = 0.21;
    cut_root_channels(2, 3) = cut_root_channels(3, 2) = -0.13;
    if (!zero_response)
    {
        response_channels(1, 1) = -0.07;
        response_channels(2, 2) = -0.09;
        response_channels(3, 3) = -0.11;
        response_channels(1, 2) = {0.012, 0.009};
        response_channels(2, 1) = {0.012, -0.009};
        response_channels(2, 3) = {0.006, -0.005};
        response_channels(3, 2) = {0.006, 0.005};
    }
    const auto adjoint = transpose(eigenvectors, true);
    const auto bare = eigenvectors * diagonal * adjoint;
    const auto cut_root = eigenvectors * cut_root_channels * adjoint;
    const auto cut = cut_root * cut_root;
    const auto response = eigenvectors * response_channels * adjoint;
    auto dielectric_channels = sqrt_diagonal * response_channels * sqrt_diagonal;
    dielectric_channels *= -1.0;
    for (int i = 0; i != n; ++i) dielectric_channels(i, i) += 1.0;
    const auto inverse = serial_inverse(dielectric_channels);

    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    pbc.set_kgrids_kvec(2, 1, 1, {0.0, 0.0, 0.0, PI, 0.0, 0.0});
    AtomicBasis wfc_basis({2}), aux_basis({n});
    MeanField mf(1, 2, 2, 2);
    for (int ik = 0; ik != 2; ++ik)
    {
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvals()[0](ik, 1) = 1.0;
    }
    TFGrids grids(1);
    grids.generate_evenspaced_tf(0.1, 0.1, 0.25, 0.1);
    KPointBlacsParallelContext kctx(KPointBlacsProcessShape(1, blacs_h.nprocs), MPI_COMM_WORLD, 2);
    ArrayDesc desc_wfc(blacs_h), desc_aux(blacs_h);
    desc_wfc.init(2, 2, 1, 1, 0, 0);
    desc_aux.init(n, n, block_size, block_size, 0, 0);
    const SymmetryContext context;
    Chi0 chi0(mf, wfc_basis, aux_basis, pbc, context, grids, kctx, desc_wfc, false, false);
    atpair_k_cplx_mat_t bare_blocks, cut_blocks;
    using ResponseMap =
        std::map<double, std::map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>>;
    // Inject fixed numerical response data in this non-const fixture so the
    // complete production Wc path is exercised without coupling this test to
    // Green-function contraction or a new production input API.
    auto &response_blocks = const_cast<ResponseMap &>(chi0.get_chi0_q());
    if (blacs_h.myid == 0)
    {
        for (const auto &q : pbc.klist)
        {
            auto &v = bare_blocks[0][0][q];
            auto &vc = cut_blocks[0][0][q];
            v = std::make_shared<ComplexMatrix>(n, n);
            vc = std::make_shared<ComplexMatrix>(n, n);
            auto &chi = response_blocks[grids.get_freq_nodes().front()][q][0][0];
            chi.create(n, n);
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != n; ++j)
                {
                    (*v)(i, j) = bare(i, j);
                    (*vc)(i, j) = cut(i, j);
                    chi(i, j) = response(i, j);
                }
        }
    }
    velocity_matrix_t velocity;
    initialize_velocity_matrix(velocity, 1, 2, 2);
    diele_func headwing(mf, velocity, pbc.kfrac_list, wfc_basis, aux_basis, grids.get_freq_nodes(),
                        2, 2, 1, n, pbc, global::mpi_comm_global_h, blacs_h);
    headwing.init(0.0, bare_blocks);
    headwing.cal_head();
    headwing.init_wing(0.0, bare_blocks);
    headwing.set_0_wing();
    const auto actual =
        compute_Wc_freq_q_blacs(chi0, bare_blocks, cut_blocks, 0.0, true, dielectric_option,
                                std::vector<std::complex<double>>{1.0}, &headwing, blacs_h,
                                desc_aux, false, ".", use_cholesky, false, false);

    for (std::size_t iq = 0; iq != pbc.klist.size(); ++iq)
    {
        auto expected_channels = inverse.copy();
        if (iq == 0 && dielectric_option == 3)
        {
            expected_channels(0, 0) = rectangular_gamma_volume_ratio();
            for (int i = 0; i != n; ++i)
                for (int j = 0; j != n; ++j)
                    if (i >= retained || j >= retained) expected_channels(i, j) = 0.0;
            for (int i = 0; i != retained; ++i) expected_channels(i, i) -= 1.0;
        }
        else
            for (int i = 0; i != n; ++i) expected_channels(i, i) -= 1.0;
        const auto expected = cut_root * eigenvectors * expected_channels * adjoint * cut_root;
        const auto &result = actual.at(grids.get_freq_nodes().front()).at(pbc.klist[iq]);
        double local_error = 0.0;
        for (int i = 0; i != desc_aux.m_loc(); ++i)
            for (int j = 0; j != desc_aux.n_loc(); ++j)
            {
                const auto actual_value = result(i, j);
                const auto expected_value = expected(desc_aux.indx_l2g_r(i),
                                                     desc_aux.indx_l2g_c(j));
                const double error = std::abs(actual_value - expected_value);
                // Reduce a failure on any rank before asserting, including non-finite data.
                if (!std::isfinite(actual_value.real()) || !std::isfinite(actual_value.imag()) ||
                    !std::isfinite(expected_value.real()) ||
                    !std::isfinite(expected_value.imag()) || !std::isfinite(error))
                    local_error = std::numeric_limits<double>::infinity();
                else
                    local_error = std::max(local_error, error);
            }
        double max_error = 0.0;
        MPI_Allreduce(&local_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (blacs_h.myid == 0)
            std::cout << "Gamma Wc dense control: rank=" << retained
                      << " zero_response=" << zero_response << " option=" << dielectric_option
                      << " cholesky=" << use_cholesky << " block_size=" << block_size
                      << " iq=" << iq << " max_error=" << max_error << std::endl;
        assert(max_error < 1e-10);
    }
}

void test_gamma_wc_retained_identity(const BlacsCtxtHandler &blacs_h)
{
    for (const bool rank_deficient : {true, false})
        for (const bool zero_response : {true, false})
            verify_gamma_wc_retained_identity(blacs_h, rank_deficient, zero_response, 3, false);
    verify_gamma_wc_retained_identity(blacs_h, true, false, 3, true);
    verify_gamma_wc_retained_identity(blacs_h, true, false, 1, false);
    // The filtered eigenvector starts inside a block, rather than at its edge.
    verify_gamma_wc_retained_identity(blacs_h, true, false, 3, false, 2);
}

#endif

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

        test_gamma_s_head_vector(blacs_h);
        test_head_only_trace_logdet_can_use_reduced_response(blacs_h);
        test_trace_logdet_accounts_for_complex_row_pivot(blacs_h);
        test_wq_to_wr_symmetry_reduced_q_matches_full_bz();
        test_wq_to_wr_antiunitary_member_conjugates();
        test_wq_to_wr_symmetry_collective_handles_empty_local_rank();
        test_dense_wq_to_wr_symmetry_reduced_q_matches_full_bz(blacs_h);
        test_dense_wc_real_ct_matches_legacy_real_part();
        test_dense_wc_real_ct_projects_large_imaginary_residual();
#ifdef LIBRPA_USE_LIBRI
        test_dense_wc_layout(blacs_h);
        test_gamma_wc_retained_identity(blacs_h);
#endif
    }

    librpa_int::global::finalize_global_io();
    librpa_int::global::finalize_global_mpi();
    MPI_Finalize();
    return 0;
}
