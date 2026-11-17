#ifdef NDEBUG
#undef NDEBUG
#endif

/*!
 * @file test_pair_density_vertex.cpp
 * @brief Regression tests for projector-consistent complex cRPA D and U.
 */

#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <valarray>
#include <vector>

#include "../core/pair_density_vertex.h"
#include "../utils/dev_options.h"
#include "../utils/profiler.h"
#include "crpa_atom_block_reference.h"

using namespace librpa_int;
using namespace crpa_test;
using Complex = std::complex<double>;

namespace
{

bool close(const Complex a, const Complex b, const double tol = 1.0e-11)
{
    return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}

void assert_matrix_close(const ComplexMatrix &a, const ComplexMatrix &b, const double tol = 1.0e-11)
{
    assert(a.nr == b.nr && a.nc == b.nc);
    for (int i = 0; i != a.size; ++i) assert(close(a.c[i], b.c[i], tol));
}

RI::Tensor<double> make_cs(const int nmu, const int ni, const int nj,
                           const std::vector<double> &values)
{
    assert(values.size() == static_cast<std::size_t>(nmu * ni * nj));
    auto data = std::make_shared<std::valarray<double>>(values.data(), values.size());
    return RI::Tensor<double>(
        {static_cast<std::size_t>(nmu), static_cast<std::size_t>(ni), static_cast<std::size_t>(nj)},
        data);
}

PeriodicBoundaryData make_pbc(const int nx)
{
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    pbc.set_period(nx, 1, 1);
    return pbc;
}

std::vector<Vector3_Order<double>> uniform_qx(const int nx)
{
    std::vector<Vector3_Order<double>> q;
    for (int i = 0; i != nx; ++i) q.push_back({static_cast<double>(i) / nx, 0.0, 0.0});
    return q;
}

SiteOrbitalGroup site(const std::string &label, const int atom, const int norb)
{
    SiteOrbitalGroup result;
    result.label = label;
    result.atom_index = atom;
    result.orb_start = 0;
    result.n_orbitals = norb;
    for (int i = 0; i != norb; ++i) result.orbital_labels.push_back("o" + std::to_string(i));
    return result;
}

void test_raw_same_center_half_coefficient()
{
    const auto pbc = make_pbc(1);
    const auto qpoints = uniform_qx(1);
    const AtomicBasis basis({1});
    ComplexMatrix t(1, 1);
    t(0, 0) = 1.0;
    PairDensityCsMap Cs;
    Cs[0][{0, {0, 0, 0}}] = make_cs(1, 1, 1, {0.5});
    const auto vertex = build_site_pair_density_vertex_from_T(0, site("half", 0, 1), {t}, Cs, basis,
                                                              basis, pbc, qpoints);
    const auto &D = vertex.q_blocks.at(qpoints[0]);
    PairInteractionAtomBlocks W;
    W[0][0] = ComplexMatrix(1, 1);
    W[0][0](0, 0) = 1.0;
    const auto U = contract_ordered_pair_vertex(D, W, 1);
    std::cout << "raw Cs=0.5: D=" << D.at(0)(0, 0) << " U=" << U(0, 0) << " expected D=1 U=1"
              << std::endl;
    assert(close(D.at(0)(0, 0), 1.0));
    assert(close(U(0, 0), 1.0));
}

void test_dense_complex_Tdagger_C_T()
{
    test_raw_same_center_half_coefficient();
    const auto pbc = make_pbc(1);
    const auto qpoints = uniform_qx(1);
    const AtomicBasis basis_wfc({2});
    const AtomicBasis basis_abf({2});

    ComplexMatrix T(2, 2);
    T(0, 0) = {0.7, 0.2};
    T(0, 1) = {-0.1, 0.4};
    T(1, 0) = {0.3, -0.5};
    T(1, 1) = {0.8, 0.1};

    PairDensityCsMap Cs;
    // Tensor storage order is (mu,i,j).
    Cs[0][{0, {0, 0, 0}}] = make_cs(2, 2, 2, {1.0, 0.2, -0.4, 0.7, -0.3, 0.6, 0.5, 1.2});

    const auto result = build_site_pair_density_vertex_from_T(0, site("dense", 0, 2), {T}, Cs,
                                                              basis_wfc, basis_abf, pbc, qpoints);
    const auto &D = result.q_blocks.at(qpoints[0]).at(0);
    assert(D.nr == 4 && D.nc == 2);

    ComplexMatrix expected(4, 2);
    const auto &C = Cs.at(0).begin()->second;
    for (int mu = 0; mu != 2; ++mu)
    {
        // Raw first-end coefficients complete the AO matrix as C + C^T.
        ComplexMatrix Cfull(2, 2);
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j) Cfull(i, j) = C(mu, i, j) + C(mu, j, i);
        const auto projected = transpose(T, true) * Cfull * T;
        for (int a = 0; a != 2; ++a)
            for (int b = 0; b != 2; ++b) expected(a * 2 + b, mu) = projected(a, b);
    }
    assert_matrix_close(D, expected);
}

void test_random_complex_vertex_against_literal_sum(
    const int n_spinor = 1, const BlacsCtxtHandler *cpu_grid = nullptr)
{
    // Unequal AO/auxiliary blocks, translated Cs outside the BvK cell,
    // multiple contributions to each output, and complex orbital phases.
    // The reference performs the original scalar five-index sum directly.
    const auto pbc = make_pbc(3);
    auto qpoints = uniform_qx(3);
    for (auto &q : qpoints) q.x += 0.137;
    const AtomicBasis basis_wfc(std::vector<std::size_t>{2, 3});
    const AtomicBasis basis_abf(std::vector<std::size_t>{3, 2});
    constexpr int norb = 3;
    std::mt19937 generator(71023);
    std::uniform_real_distribution<double> random(-0.8, 0.8);
    std::vector<ComplexMatrix> T_R;
    for (std::size_t ir = 0; ir != pbc.Rlist.size(); ++ir)
    {
        ComplexMatrix T(5 * n_spinor, norb);
        for (int i = 0; i != T.size; ++i) T.c[i] = {random(generator), random(generator)};
        T_R.push_back(T);
    }
    // An absent atom block also exercises the production support shortcut.
    for (int s = 0; s != n_spinor; ++s)
        for (int i = 0; i != 2; ++i)
            for (int a = 0; a != norb; ++a) T_R[1](s * 5 + i, a) = 0.0;
    PairDensityCsMap Cs;
    for (int I = 0; I != 2; ++I)
        for (int J = 0; J != 2; ++J)
            for (const int r : {-5, 0, 4})
            {
                const int nmu = basis_abf.get_atom_nb(I);
                const int ni = basis_wfc.get_atom_nb(I), nj = basis_wfc.get_atom_nb(J);
                std::vector<double> values(nmu * ni * nj);
                for (auto &value : values) value = random(generator);
                Cs[I][{J, {r, 0, 0}}] = make_cs(nmu, ni, nj, values);
            }
    const auto actual = build_site_pair_density_vertex_from_T(
        0, site("random-complex", 0, norb), T_R, Cs, basis_wfc, basis_abf, pbc, qpoints, cpu_grid);
    for (const auto &q : qpoints)
        for (int I = 0; I != 2; ++I)
        {
            ComplexMatrix expected(norb * norb, basis_abf.get_atom_nb(I));
            const int ibegin = basis_wfc.get_part_range().at(I);
            for (const auto &[JR, C] : Cs.at(I))
            {
                const int J = JR.first;
                const int jbegin = basis_wfc.get_part_range().at(J);
                for (int ia = 0; ia != 3; ++ia)
                {
                    // Canonical Rlist is [-1,0,1]; derive its wrapped index
                    // independently of the production wrap/get_R_index helpers.
                    const int ib = ((ia + JR.second[0]) % 3 + 3) % 3;
                    const Complex phase = std::polar(1.0, -2.0 * M_PI * q.x * (ia - 1));
                    for (int a = 0; a != norb; ++a)
                        for (int b = 0; b != norb; ++b)
                            for (int mu = 0; mu != expected.nc; ++mu)
                                for (int i = 0; i != basis_wfc.get_atom_nb(I); ++i)
                                    for (int j = 0; j != basis_wfc.get_atom_nb(J); ++j)
                                        for (int s = 0; s != n_spinor; ++s)
                                            expected(a * norb + b, mu) +=
                                                phase * C(mu, i, j) *
                                                (std::conj(T_R[ia](s * 5 + ibegin + i, a)) *
                                                     T_R[ib](s * 5 + jbegin + j, b) +
                                                 std::conj(T_R[ib](s * 5 + jbegin + j, a)) *
                                                     T_R[ia](s * 5 + ibegin + i, b));
                }
            }
            assert_matrix_close(actual.q_blocks.at(q).at(I), expected, 2.0e-12);
        }
}

#ifdef LIBRPA_USE_DDLA
void test_mixed_coefficient_memory_gate(const BlacsCtxtHandler &cpu_grid,
                                       const bool eligible, const int n_spinor)
{
    const int ni = eligible ? 4 : 2, nj = eligible ? 5 : 3;
    const int nmu = eligible ? 16 : 2, norb = eligible ? 2 : 3;
    const int nao = ni + nj, nij = ni * nj, npair = norb * norb;
    const auto bytes = ddla::gemm_mixed_workspace_bytes<double>(
        cpu_grid.ddla_cpu_handle, nmu, npair, nij, false);
    const auto removed = static_cast<std::size_t>(nmu) * nij * sizeof(Complex);
    assert((bytes <= removed) == eligible);
    const auto pbc = make_pbc(1);
    const auto qpoints = uniform_qx(1);
    const AtomicBasis ao(std::vector<std::size_t>{static_cast<std::size_t>(ni),
                                                static_cast<std::size_t>(nj)});
    const AtomicBasis abf(std::vector<std::size_t>{static_cast<std::size_t>(nmu), 1});
    ComplexMatrix t(n_spinor * nao, norb);
    for (int s = 0; s != n_spinor; ++s)
        for (int i = 0; i != nao; ++i)
            for (int a = 0; a != norb; ++a)
                t(s * nao + i, a) = {0.13 + 0.07 * i - 0.03 * a, 0.09 * i + 0.11 * a + 0.17 * s};
    std::vector<double> coefficients(nmu * nij);
    for (int i = 0; i != nmu * nij; ++i) coefficients[i] = std::sin(0.17 * (i + 1));
    PairDensityCsMap Cs;
    Cs[0][{1, {0, 0, 0}}] = make_cs(nmu, ni, nj, coefficients);
    auto saved_profiler = std::move(global::profiler);
    global::profiler = Profiler{};
    const auto vertex = build_site_pair_density_vertex_from_T(
        0, site("memory-gate", 0, norb), {t}, Cs, ao, abf, pbc, qpoints, &cpu_grid);
    const auto profile = global::profiler.get_profile_string(LIBRPA_VERBOSE_DEBUG);
    assert((profile.find("pair_vertex_mixed_coefficients") != std::string::npos) == eligible);
    ComplexMatrix expected(npair, nmu);
    for (int a = 0; a != norb; ++a)
        for (int b = 0; b != norb; ++b)
            for (int mu = 0; mu != nmu; ++mu)
                for (int i = 0; i != ni; ++i)
                    for (int j = 0; j != nj; ++j)
                        for (int s = 0; s != n_spinor; ++s)
                            expected(a * norb + b, mu) += coefficients[mu * nij + i * nj + j] *
                                (std::conj(t(s * nao + i, a)) * t(s * nao + ni + j, b) +
                                 std::conj(t(s * nao + ni + j, a)) * t(s * nao + i, b));
    assert_matrix_close(vertex.q_blocks.at(qpoints[0]).at(0), expected, 2e-12);
    // The same eligible shape with no borrowed handle must safely use legacy GEMM.
    global::profiler = Profiler{};
    const auto absent = build_site_pair_density_vertex_from_T(
        0, site("absent-handle", 0, norb), {t}, Cs, ao, abf, pbc, qpoints);
    assert(global::profiler.get_profile_string(LIBRPA_VERBOSE_DEBUG)
               .find("pair_vertex_mixed_coefficients") == std::string::npos);
    assert_matrix_close(absent.q_blocks.at(qpoints[0]).at(0), expected, 2e-12);
    global::profiler = std::move(saved_profiler);
}
#endif

void test_nonzero_R_wrap_and_minus_fourier_phase()
{
    const auto pbc = make_pbc(3);
    const auto qpoints = uniform_qx(3);
    const AtomicBasis basis_wfc({1});
    const AtomicBasis basis_abf({1});

    std::vector<ComplexMatrix> T_R;
    // Rlist for period 3 is {-1,0,+1}.
    for (const Complex value : std::vector<Complex>{{1.0, 0.0}, {2.0, 1.0}, {-0.5, 0.3}})
    {
        ComplexMatrix T(1, 1);
        T(0, 0) = value;
        T_R.push_back(T);
    }
    PairDensityCsMap Cs;
    Cs[0][{0, {1, 0, 0}}] = make_cs(1, 1, 1, {1.0});

    const auto result = build_site_pair_density_vertex_from_T(0, site("phase", 0, 1), T_R, Cs,
                                                              basis_wfc, basis_abf, pbc, qpoints);

    std::vector<Complex> D_A(3);
    for (int iA = 0; iA != 3; ++iA)
    {
        const int iB = (iA + 1) % 3;
        D_A[iA] = 2.0 * std::real(std::conj(T_R[iA](0, 0)) * T_R[iB](0, 0));
    }
    bool sign_test_is_discriminating = false;
    for (int iq = 0; iq != 3; ++iq)
    {
        Complex minus{0.0, 0.0};
        Complex plus{0.0, 0.0};
        for (int iA = 0; iA != 3; ++iA)
        {
            const int A = pbc.Rlist[iA].x;
            const double arg = 2.0 * M_PI * qpoints[iq].x * A;
            minus += Complex(std::cos(-arg), std::sin(-arg)) * D_A[iA];
            plus += Complex(std::cos(arg), std::sin(arg)) * D_A[iA];
        }
        const auto actual = result.q_blocks.at(qpoints[iq]).at(0)(0, 0);
        assert(close(actual, minus));
        if (std::abs(minus - plus) > 1.0e-5) sign_test_is_discriminating = true;
    }
    assert(sign_test_is_discriminating);
}

void test_multiple_atoms_and_sites_remain_separate()
{
    const auto pbc = make_pbc(1);
    const auto qpoints = uniform_qx(1);
    const AtomicBasis basis_wfc(std::vector<std::size_t>{1, 1});
    const AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    PairDensityCsMap Cs;
    Cs[0][{0, {0, 0, 0}}] = make_cs(1, 1, 1, {1.0});
    Cs[0][{1, {0, 0, 0}}] = make_cs(1, 1, 1, {2.0});
    Cs[1][{0, {0, 0, 0}}] = make_cs(1, 1, 1, {3.0});
    Cs[1][{1, {0, 0, 0}}] = make_cs(1, 1, 1, {4.0});

    ComplexMatrix T0(2, 1), T1(2, 1), Tmix(2, 1);
    T0(0, 0) = 1.0;
    T1(1, 0) = 1.0;
    Tmix(0, 0) = 1.0;
    Tmix(1, 0) = Complex{0.0, 1.0};

    const auto s0 = build_site_pair_density_vertex_from_T(0, site("A", 0, 1), {T0}, Cs, basis_wfc,
                                                          basis_abf, pbc, qpoints);
    const auto s1 = build_site_pair_density_vertex_from_T(1, site("B", 1, 1), {T1}, Cs, basis_wfc,
                                                          basis_abf, pbc, qpoints);
    const auto sm = build_site_pair_density_vertex_from_T(2, site("mixed", 0, 1), {Tmix}, Cs,
                                                          basis_wfc, basis_abf, pbc, qpoints);

    const auto &D0 = s0.q_blocks.at(qpoints[0]);
    const auto &D1 = s1.q_blocks.at(qpoints[0]);
    assert(D0.size() == 1 && D0.count(0) == 1 && close(D0.at(0)(0, 0), 2.0));
    assert(D1.size() == 1 && D1.count(1) == 1 && close(D1.at(1)(0, 0), 8.0));
    assert(s0.label != s1.label && s0.site_index != s1.site_index);

    const auto &Dm = sm.q_blocks.at(qpoints[0]);
    assert(close(Dm.at(0)(0, 0), 2.0));
    assert(close(Dm.at(1)(0, 0), 8.0));
}

void test_complex_pair_reversal_and_hermiticity()
{
    constexpr int norb = 2;
    constexpr int npair = 4;
    PairVertexAtomBlocks D;
    D.emplace(0, ComplexMatrix(npair, 2));
    D.emplace(1, ComplexMatrix(npair, 1));
    for (int p = 0; p != npair; ++p)
    {
        D.at(0)(p, 0) = Complex{0.2 + 0.1 * p, -0.3 + 0.07 * p};
        D.at(0)(p, 1) = Complex{-0.1 + 0.04 * p, 0.5 - 0.03 * p};
        D.at(1)(p, 0) = Complex{0.6 - 0.05 * p, 0.2 + 0.09 * p};
    }

    PairInteractionAtomBlocks W;
    W[0][0] = ComplexMatrix(2, 2);
    W[0][0](0, 0) = 2.0;
    W[0][0](1, 1) = 1.5;
    W[0][0](0, 1) = Complex{0.3, -0.2};
    W[0][0](1, 0) = std::conj(W[0][0](0, 1));
    W[0][1] = ComplexMatrix(2, 1);
    W[0][1](0, 0) = Complex{0.4, 0.25};
    W[0][1](1, 0) = Complex{-0.2, 0.1};
    W[1][1] = ComplexMatrix(1, 1);
    W[1][1](0, 0) = 0.9;

    const auto U = contract_ordered_pair_vertex(D, W, norb);
    ComplexMatrix expected(npair, npair);
    for (int a = 0; a != norb; ++a)
        for (int b = 0; b != norb; ++b)
            for (int c = 0; c != norb; ++c)
                for (int d = 0; d != norb; ++d)
                {
                    const int ab = a * norb + b;
                    const int ba = b * norb + a;
                    const int cd = c * norb + d;
                    for (int mu = 0; mu != 2; ++mu)
                        for (int nu = 0; nu != 2; ++nu)
                            expected(ab, cd) += std::conj(D.at(0)(ba, mu)) * W.at(0).at(0)(mu, nu) *
                                                D.at(0)(cd, nu);
                    for (int mu = 0; mu != 2; ++mu)
                        expected(ab, cd) +=
                            std::conj(D.at(0)(ba, mu)) * W.at(0).at(1)(mu, 0) * D.at(1)(cd, 0);
                    for (int nu = 0; nu != 2; ++nu)
                        expected(ab, cd) += std::conj(D.at(1)(ba, 0)) *
                                            std::conj(W.at(0).at(1)(nu, 0)) * D.at(0)(cd, nu);
                    expected(ab, cd) += std::conj(D.at(1)(ba, 0)) * 0.9 * D.at(1)(cd, 0);
                }
    assert_matrix_close(U, expected);

    for (int a = 0; a != norb; ++a)
        for (int b = 0; b != norb; ++b)
            for (int c = 0; c != norb; ++c)
                for (int d = 0; d != norb; ++d)
                {
                    const int ab = a * norb + b;
                    const int cd = c * norb + d;
                    const int dc = d * norb + c;
                    const int ba = b * norb + a;
                    assert(close(std::conj(U(ab, cd)), U(dc, ba), 2.0e-11));
                }
}

void test_real_D_reduces_to_D_W_Dt()
{
    constexpr int norb = 2;
    PairVertexAtomBlocks D;
    D.emplace(0, ComplexMatrix(4, 2));
    const double values[4][2] = {{1.0, 0.2}, {0.3, -0.4}, {0.3, -0.4}, {0.7, 0.8}};
    for (int p = 0; p != 4; ++p)
        for (int mu = 0; mu != 2; ++mu) D.at(0)(p, mu) = values[p][mu];
    PairInteractionAtomBlocks W;
    W[0][0] = ComplexMatrix(2, 2);
    W[0][0](0, 0) = 2.0;
    W[0][0](0, 1) = 0.5;
    W[0][0](1, 0) = 0.5;
    W[0][0](1, 1) = 1.2;

    const auto U = contract_ordered_pair_vertex(D, W, norb);
    const auto legacy = D.at(0) * W.at(0).at(0) * transpose(D.at(0), false);
    assert_matrix_close(U, legacy);
}

void test_upper_triangle_mirror_added_once()
{
    PairVertexAtomBlocks D;
    D.emplace(0, ComplexMatrix(1, 1));
    D.emplace(1, ComplexMatrix(1, 1));
    D.at(0)(0, 0) = Complex{2.0, 0.0};
    D.at(1)(0, 0) = Complex{3.0, 1.0};
    PairInteractionAtomBlocks W;
    W[0][1] = ComplexMatrix(1, 1);
    W[0][1](0, 0) = Complex{0.4, -0.2};

    const auto U = contract_ordered_pair_vertex(D, W, 1);
    const Complex expected =
        std::conj(D.at(0)(0, 0)) * W.at(0).at(1)(0, 0) * D.at(1)(0, 0) +
        std::conj(D.at(1)(0, 0)) * std::conj(W.at(0).at(1)(0, 0)) * D.at(0)(0, 0);
    assert(close(U(0, 0), expected));
    assert(!close(U(0, 0), 2.0 * expected));
}

void test_q_weighted_sum()
{
    const auto pbc = make_pbc(3);
    const auto qpoints = uniform_qx(3);
    const AtomicBasis basis_wfc({1});
    const AtomicBasis basis_abf({1});
    std::vector<ComplexMatrix> T_R(3, ComplexMatrix(1, 1));
    T_R[0](0, 0) = {1.0, 0.2};
    T_R[1](0, 0) = {-0.4, 0.7};
    T_R[2](0, 0) = {0.3, -0.1};
    PairDensityCsMap Cs;
    Cs[0][{0, {0, 0, 0}}] = make_cs(1, 1, 1, {1.3});
    const auto vertex = build_site_pair_density_vertex_from_T(0, site("weighted", 0, 1), T_R, Cs,
                                                              basis_wfc, basis_abf, pbc, qpoints);

    const double weights[3] = {0.2, 0.3, 0.5};
    Complex weighted{0.0, 0.0};
    Complex explicit_sum{0.0, 0.0};
    for (int iq = 0; iq != 3; ++iq)
    {
        PairInteractionAtomBlocks W;
        W[0][0] = ComplexMatrix(1, 1);
        W[0][0](0, 0) = 1.0 + 0.4 * iq;
        const auto &Dq = vertex.q_blocks.at(qpoints[iq]);
        weighted += weights[iq] * contract_ordered_pair_vertex(Dq, W, 1)(0, 0);
        const auto d = Dq.at(0)(0, 0);
        explicit_sum += weights[iq] * std::conj(d) * W[0][0](0, 0) * d;
    }
    assert(close(weighted, explicit_sum));
}

void test_fourier_contraction_matches_real_space_convolution()
{
    const auto pbc = make_pbc(3);
    const auto qpoints = uniform_qx(3);
    const AtomicBasis basis_wfc({1});
    const AtomicBasis basis_abf({1});
    std::vector<ComplexMatrix> T_R(3, ComplexMatrix(1, 1));
    T_R[0](0, 0) = {0.8, -0.2};
    T_R[1](0, 0) = {-0.3, 0.6};
    T_R[2](0, 0) = {0.5, 0.4};
    constexpr double cs = 1.3;
    PairDensityCsMap Cs;
    Cs[0][{0, {1, 0, 0}}] = make_cs(1, 1, 1, {cs});
    const auto vertex = build_site_pair_density_vertex_from_T(0, site("convolution", 0, 1), T_R, Cs,
                                                              basis_wfc, basis_abf, pbc, qpoints);

    auto wrap_x = [](int x)
    {
        while (x < -1) x += 3;
        while (x > 1) x -= 3;
        return x;
    };
    std::map<int, Complex> D_A;
    for (int iA = 0; iA != 3; ++iA)
    {
        const int A = pbc.Rlist[static_cast<std::size_t>(iA)].x;
        const int iB = pbc.get_R_index({wrap_x(A + 1), 0, 0});
        assert(iB >= 0);
        D_A[A] = 2.0 * cs *
                 std::real(std::conj(T_R[static_cast<std::size_t>(iA)](0, 0)) *
                           T_R[static_cast<std::size_t>(iB)](0, 0));
    }

    const Complex w_plus{0.25, -0.17};
    const std::map<int, Complex> W_R{{-1, std::conj(w_plus)}, {0, Complex{1.4, 0.0}}, {1, w_plus}};
    Complex reciprocal{0.0, 0.0};
    for (const auto &q : qpoints)
    {
        Complex Wq{0.0, 0.0};
        for (const auto &[R, value] : W_R)
        {
            const double arg = 2.0 * M_PI * q.x * R;
            Wq += Complex{std::cos(arg), std::sin(arg)} * value;
        }
        PairInteractionAtomBlocks W;
        W[0][0] = ComplexMatrix(1, 1);
        W[0][0](0, 0) = Wq;
        reciprocal += contract_ordered_pair_vertex(vertex.q_blocks.at(q), W, 1)(0, 0) / 3.0;
    }

    Complex real_space{0.0, 0.0};
    for (const auto &[A, D_left] : D_A)
        for (const auto &[R, W] : W_R) real_space += std::conj(D_left) * W * D_A.at(wrap_x(A + R));
    assert(close(reciprocal, real_space, 2.0e-11));
}

void test_distributed_contraction_ownership(const MpiCommHandler &comm_h)
{
    PairVertexAtomBlocks D;
    D.emplace(0, ComplexMatrix(1, 1));
    D.emplace(1, ComplexMatrix(1, 1));
    D.at(0)(0, 0) = Complex{2.0, 0.0};
    D.at(1)(0, 0) = Complex{3.0, 1.0};

    PairInteractionAtomBlocks W_full;
    W_full[0][0] = ComplexMatrix(1, 1);
    W_full[0][0](0, 0) = 1.1;
    W_full[0][1] = ComplexMatrix(1, 1);
    W_full[0][1](0, 0) = Complex{0.4, -0.2};
    W_full[1][1] = ComplexMatrix(1, 1);
    W_full[1][1](0, 0) = 0.8;
    const auto expected = contract_ordered_pair_vertex(D, W_full, 1);

    const std::vector<std::pair<int, int>> pairs{{0, 0}, {0, 1}, {1, 1}};
    auto distribute = [&](const int omitted_pair)
    {
        PairInteractionAtomBlocks local;
        for (int ipair = 0; ipair != static_cast<int>(pairs.size()); ++ipair)
        {
            if (ipair == omitted_pair || ipair % comm_h.nprocs != comm_h.myid) continue;
            const auto [I, J] = pairs[static_cast<std::size_t>(ipair)];
            local[I][J] = W_full.at(I).at(J);
        }
        return local;
    };

    const auto actual = contract_distributed_ordered_pair_vertex(D, distribute(-1), 1, comm_h,
                                                                 "distributed ownership test");
    assert_matrix_close(actual, expected);
}

#ifdef LIBRPA_USE_LIBRI
CorrelatedSubspace make_two_site_subspace(const PeriodicBoundaryData &pbc)
{
    SiteOrbitalGroup A = site("A", 0, 1);
    SiteOrbitalGroup B = site("B", 1, 1);
    B.orb_start = 1;
    CorrelatedSubspace subspace({A, B}, {{0.0, 0.0, 0.0}}, pbc.Rlist, 2, 2, 1);
    ComplexMatrix I(2, 2);
    I.set_as_identity_matrix();
    subspace.set_S_k(0, I);
    subspace.set_W_k(0, I);
    subspace.compute_T_R();
    return subspace;
}

void test_production_wrapper_mpi(const MpiCommHandler &comm_h)
{
    const auto pbc = make_pbc(1);
    const auto qpoints = uniform_qx(1);
    const AtomicBasis basis_wfc(std::vector<std::size_t>{1, 1});
    const AtomicBasis basis_abf(std::vector<std::size_t>{1, 1});
    auto subspace = make_two_site_subspace(pbc);

    Cs_LRI Cs;
    Cs.use_libri = true;
    struct Block
    {
        int I;
        int J;
        double value;
    };
    const std::vector<Block> blocks{{0, 0, 1.0}, {0, 1, 2.0}, {1, 0, 3.0}, {1, 1, 4.0}};
    for (std::size_t ib = 0; ib != blocks.size(); ++ib)
        if (static_cast<int>(ib % comm_h.nprocs) == comm_h.myid)
            Cs.data_libri[blocks[ib].I][{blocks[ib].J, {0, 0, 0}}] =
                make_cs(1, 1, 1, {blocks[ib].value});

    PairDensityVertex vertex(subspace, Cs, basis_wfc, basis_abf, pbc, qpoints, comm_h);
    assert(vertex.sites().size() == 2);
    assert(vertex.site(0).label == "A" && vertex.site(1).label == "B");
    const auto d_A = vertex.site(0).q_blocks.at(qpoints[0]).at(0)(0, 0);
    const auto d_B = vertex.site(1).q_blocks.at(qpoints[0]).at(1)(0, 0);
    assert(close(d_A, 2.0));
    assert(close(d_B, 8.0));

    double local[4] = {d_A.real(), d_A.imag(), d_B.real(), d_B.imag()};
    std::vector<double> all(static_cast<std::size_t>(4 * comm_h.nprocs));
    MPI_Allgather(local, 4, MPI_DOUBLE, all.data(), 4, MPI_DOUBLE, comm_h.comm);
    for (int rank = 0; rank != comm_h.nprocs; ++rank)
        for (int i = 0; i != 4; ++i) assert(close(all[4 * rank + i], local[i]));
}
#endif

void test_analytic_cross_end_complex_phase()
{
    const auto pbc = make_pbc(4);
    const auto qpoints = uniform_qx(4);
    const AtomicBasis basis(std::vector<std::size_t>{1, 1});
    std::vector<ComplexMatrix> t(4, ComplexMatrix(2, 2));
    t[pbc.get_R_index({0, 0, 0})](0, 0) = Complex{0.0, 1.0};
    t[pbc.get_R_index({1, 0, 0})](1, 1) = 1.0;
    PairDensityCsMap Cs;
    Cs[0][{1, {1, 0, 0}}] = make_cs(1, 1, 1, {2.0});
    Cs[1][{0, {-1, 0, 0}}] = make_cs(1, 1, 1, {3.0});
    const auto vertex = build_site_pair_density_vertex_from_T(0, site("cross-end", 0, 2), t, Cs,
                                                              basis, basis, pbc, qpoints);
    const auto &D = vertex.q_blocks.at(qpoints[1]);
    // Independent values: D_ab=(-2i,-3), D_ba=(2i,3) at q=1/4.
    assert(close(D.at(0)(1, 0), Complex{0.0, -2.0}));
    assert(close(D.at(1)(1, 0), -3.0));
    assert(close(D.at(0)(2, 0), Complex{0.0, 2.0}));
    assert(close(D.at(1)(2, 0), 3.0));
    PairInteractionAtomBlocks W;
    W[0][0] = ComplexMatrix(1, 1);
    W[1][1] = ComplexMatrix(1, 1);
    W[0][1] = ComplexMatrix(1, 1);
    W[0][0](0, 0) = W[1][1](0, 0) = 1.0;
    W[0][1](0, 0) = Complex{0.0, 0.5};
    const auto U = contract_ordered_pair_vertex(D, W, 2);
    std::cout << "cross-end: U_ab,ba=" << U(1, 2) << " U_ba,ab=" << U(2, 1)
              << " U_ab,ab=" << U(1, 1) << " expected 19,19,-19" << std::endl;
    assert(close(U(1, 2), 19.0));
    assert(close(U(2, 1), 19.0));
    assert(close(U(1, 1), -19.0));
}

#ifdef LIBRPA_USE_LIBRI
void test_s_aware_non_even_wrapper(const MpiCommHandler &comm_h)
{
    const auto pbc = make_pbc(3);
    const std::vector<Vector3_Order<double>> kpoints{
        {-1.0 / 3.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {1.0 / 3.0, 0.0, 0.0}};
    const AtomicBasis basis(std::vector<std::size_t>{1, 1});
    auto A = site("S-aware-I", 0, 1);
    auto B = site("S-aware-J", 1, 1);
    B.orb_start = 1;
    CorrelatedSubspace subspace({A, B}, kpoints, pbc.Rlist, 2, 2, 1);
    const double a = 2.0 * std::sqrt(5.0) / 3.0;
    const double b = std::sqrt(5.0) / 3.0;
    std::vector<ComplexMatrix> analytic_t(3, ComplexMatrix(2, 2));
    analytic_t[1](0, 0) = analytic_t[1](1, 1) = a;
    analytic_t[0](0, 1) = b;
    analytic_t[2](1, 0) = b;
    for (int ik = 0; ik != 3; ++ik)
    {
        const Complex phase = std::polar(1.0, 2.0 * M_PI * kpoints[ik].x);
        ComplexMatrix S(2, 2), W(2, 2), phi(2, 2);
        S.set_as_identity_matrix();
        S(0, 1) = -0.8 * phase;
        S(1, 0) = std::conj(S(0, 1));
        W.set_as_identity_matrix();
        subspace.set_S_k(ik, S);
        subspace.set_W_k(ik, W);
        phi(0, 0) = phi(1, 1) = a;
        phi(0, 1) = b * phase;
        phi(1, 0) = std::conj(phi(0, 1));
        assert_matrix_close(subspace.build_phi(ik), phi);
        // Independently recover Phi from the physical-position coefficients.
        ComplexMatrix recovered(2, 2);
        for (int iA = 0; iA != 3; ++iA)
        {
            const Complex factor = std::polar(1.0, -2.0 * M_PI * kpoints[ik].x * pbc.Rlist[iA].x);
            for (int i = 0; i != 4; ++i) recovered.c[i] += factor * analytic_t[iA].c[i];
        }
        assert_matrix_close(recovered, phi);
    }
    subspace.compute_T_R();
    std::vector<ComplexMatrix> physical_t;
    std::vector<ComplexMatrix> wrong_t;
    for (const auto &R : pbc.Rlist)
    {
        physical_t.push_back(subspace.get_T(pbc.get_R_index({-R.x, 0, 0})));
        wrong_t.push_back(subspace.get_T(pbc.get_R_index(R)));
    }
    for (int iA = 0; iA != 3; ++iA) assert_matrix_close(physical_t[iA], analytic_t[iA]);
    // Real-space metric: S_II(0)=S_JJ(0)=1, S_IJ(+1)=S_JI(-1)=-0.8.
    auto metric_norm = [&](const std::vector<ComplexMatrix> &t)
    {
        Complex norm{0.0, 0.0};
        for (int iA = 0; iA != 3; ++iA)
            for (int iB = 0; iB != 3; ++iB)
            {
                const int delta = (pbc.Rlist[iB].x - pbc.Rlist[iA].x + 3) % 3;
                for (int i = 0; i != 2; ++i)
                    for (int j = 0; j != 2; ++j)
                    {
                        double overlap = (i == j && delta == 0) ? 1.0 : 0.0;
                        if ((i == 0 && j == 1 && delta == 1) || (i == 1 && j == 0 && delta == 2))
                            overlap = -0.8;
                        norm += std::conj(t[iA](i, 0)) * overlap * t[iB](j, 0);
                    }
            }
        return norm;
    };
    assert(close(metric_norm(physical_t), 1.0));
    assert(close(metric_norm(wrong_t), 25.0 / 9.0));

    PairDensityCsMap full;
    full[0][{0, {0, 0, 0}}] = make_cs(1, 1, 1, {0.5});
    full[1][{1, {0, 0, 0}}] = make_cs(1, 1, 1, {0.5});
    full[0][{1, {1, 0, 0}}] = make_cs(1, 1, 1, {2.0});
    full[1][{0, {-1, 0, 0}}] = make_cs(1, 1, 1, {3.0});
    Cs_LRI local;
    local.use_libri = true;
    int iblock = 0;
    for (const auto &[I, blocks] : full)
        for (const auto &[JR, value] : blocks)
            if (iblock++ % comm_h.nprocs == comm_h.myid) local.data_libri[I][JR] = value;
    PairDensityVertex vertex(subspace, local, basis, basis, pbc, kpoints, comm_h);
    const auto &D = vertex.site(0).q_blocks.at(kpoints[2]);
    const Complex expected_J = (65.0 / 9.0) * std::polar(1.0, -2.0 * M_PI / 3.0);
    std::cout << "S-aware wrapper: D_I=" << D.at(0)(0, 0) << " D_J=" << D.at(1)(0, 0)
              << " expected " << 20.0 / 3.0 << "," << expected_J
              << " metric norms=" << metric_norm(physical_t) << "," << metric_norm(wrong_t)
              << std::endl;
    assert(close(D.at(0)(0, 0), 20.0 / 3.0));
    assert(close(D.at(1)(0, 0), expected_J));
    PairInteractionAtomBlocks interaction;
    interaction[0][0] = ComplexMatrix(1, 1);
    interaction[1][1] = ComplexMatrix(1, 1);
    interaction[0][0](0, 0) = interaction[1][1](0, 0) = 1.0;
    assert(close(contract_ordered_pair_vertex(D, interaction, 1)(0, 0), 7825.0 / 81.0));
    const auto serial =
        build_site_pair_density_vertex(subspace, 0, full, basis, basis, pbc, kpoints);
    for (const auto &q : kpoints)
        for (const auto &[I, block] : serial.q_blocks.at(q))
            assert_matrix_close(block, vertex.site(0).q_blocks.at(q).at(I));
    double values[4] = {D.at(0)(0, 0).real(), D.at(0)(0, 0).imag(), D.at(1)(0, 0).real(),
                        D.at(1)(0, 0).imag()};
    std::vector<double> all(static_cast<std::size_t>(4 * comm_h.nprocs));
    MPI_Allgather(values, 4, MPI_DOUBLE, all.data(), 4, MPI_DOUBLE, comm_h.comm);
    for (int rank = 0; rank != comm_h.nprocs; ++rank)
        for (int i = 0; i != 4; ++i) assert(close(all[4 * rank + i], values[i]));
}

void test_wrapper_nyquist_and_twisted_mesh(const MpiCommHandler &comm_h)
{
    const AtomicBasis basis({1});
    PairDensityCsMap full;
    full[0][{0, {0, 0, 0}}] = make_cs(1, 1, 1, {0.5});
    const auto pbc = make_pbc(4);
    const auto kpoints = uniform_qx(4);
    CorrelatedSubspace subspace({site("Nyquist", 0, 1)}, kpoints, pbc.Rlist, 1, 1, 1);
    ComplexMatrix S(1, 1), W(1, 1);
    S(0, 0) = 1.0;
    for (int ik = 0; ik != 4; ++ik)
    {
        W(0, 0) = std::polar(1.0, 4.0 * M_PI * kpoints[ik].x);
        subspace.set_S_k(ik, S);
        subspace.set_W_k(ik, W);
    }
    subspace.compute_T_R();
    assert(close(subspace.get_T(pbc.get_R_index({-2, 0, 0}))(0, 0), 1.0));
    const auto vertex =
        build_site_pair_density_vertex(subspace, 0, full, basis, basis, pbc, kpoints);
    for (int iq = 0; iq != 4; ++iq)
        assert(close(vertex.q_blocks.at(kpoints[iq]).at(0)(0, 0), iq % 2 ? -1.0 : 1.0));

    const auto pbc3 = make_pbc(3);
    auto twisted = uniform_qx(3);
    for (auto &k : twisted) k.x += 1.0 / 6.0;
    CorrelatedSubspace shifted({site("twisted", 0, 1)}, twisted, pbc3.Rlist, 1, 1, 1);
    W(0, 0) = 1.0;
    for (int ik = 0; ik != 3; ++ik)
    {
        shifted.set_S_k(ik, S);
        shifted.set_W_k(ik, W);
    }
    shifted.compute_T_R();
    bool rejected_twist = false;
    try
    {
        (void)build_site_pair_density_vertex(shifted, 0, full, basis, basis, pbc3, twisted);
    }
    catch (const std::exception &error)
    {
        rejected_twist = std::string(error.what()).find("twisted") != std::string::npos;
    }
    assert(rejected_twist);
    Cs_LRI local;
    local.use_libri = true;
    if (comm_h.is_root()) local.data_libri = full;

    // The explicit physical-t kernel still accepts a shifted full-q dual grid.
    std::vector<ComplexMatrix> t(3, ComplexMatrix(1, 1));
    t[1](0, 0) = 1.0;
    const auto pure = build_site_pair_density_vertex_from_T(0, site("pure-shifted-q", 0, 1), t,
                                                            full, basis, basis, pbc3, twisted);
    for (const auto &q : twisted) assert(close(pure.q_blocks.at(q).at(0)(0, 0), 1.0));
}
#endif

ComplexMatrix auxiliary_matrix2(Complex a, Complex b, Complex c, Complex d)
{
    ComplexMatrix result(2, 2);
    result(0, 0) = a;
    result(0, 1) = b;
    result(1, 0) = c;
    result(1, 1) = d;
    return result;
}

SitePairDensityVertex auxiliary_parent(const ComplexMatrix &rows, const AtomicBasis &basis,
                                       const int norb = 1)
{
    SitePairDensityVertex parent;
    parent.site_index = 3;
    parent.atom_index = 0;
    parent.n_orbitals = norb;
    parent.label = "auxiliary-adapter";
    for (int a = 0; a != norb; ++a) parent.orbital_labels.push_back("a" + std::to_string(a));
    for (const double qx : {1.0 / 3.0, -1.0 / 3.0})
        for (int I = 0; I != static_cast<int>(basis.n_atoms); ++I)
        {
            const int width = static_cast<int>(basis.get_atom_nb(I));
            if (width == 0) continue;
            const int begin = static_cast<int>(basis.get_part_range().at(I));
            auto &D = parent.q_blocks[{qx, 0.0, 0.0}][I];
            D = ComplexMatrix(norb * norb, width);
            for (int a = 0; a != norb; ++a)
                for (int b = 0; b != norb; ++b)
                    for (int mu = 0; mu != width; ++mu)
                        D(a * norb + b, mu) = qx > 0.0 ? rows(a * norb + b, begin + mu)
                                                       : std::conj(rows(b * norb + a, begin + mu));
        }
    return parent;
}

PairInteractionAtomBlocks auxiliary_interaction(const ComplexMatrix &matrix,
                                                const AtomicBasis &basis)
{
    PairInteractionAtomBlocks blocks;
    for (int I = 0; I != static_cast<int>(basis.n_atoms); ++I)
        for (int J = I; J != static_cast<int>(basis.n_atoms); ++J)
        {
            const int ni = static_cast<int>(basis.get_atom_nb(I));
            const int nj = static_cast<int>(basis.get_atom_nb(J));
            if (ni == 0 || nj == 0) continue;
            auto &block = blocks[I][J];
            block = ComplexMatrix(ni, nj);
            for (int i = 0; i != ni; ++i)
                for (int j = 0; j != nj; ++j)
                    block(i, j) =
                        matrix(basis.get_part_range().at(I) + i, basis.get_part_range().at(J) + j);
        }
    return blocks;
}

void assert_same_site_vertex(const SitePairDensityVertex &a, const SitePairDensityVertex &b)
{
    assert(a.site_index == b.site_index && a.atom_index == b.atom_index);
    assert(a.label == b.label && a.orbital_labels == b.orbital_labels);
    assert(a.n_orbitals == b.n_orbitals && a.q_blocks.size() == b.q_blocks.size());
    for (const auto &[q, blocks] : a.q_blocks)
    {
        assert(blocks.size() == b.q_blocks.at(q).size());
        for (const auto &[I, D] : blocks) assert_matrix_close(D, b.q_blocks.at(q).at(I), 0.0);
    }
}

void test_auxiliary_square_covariance(const MpiCommHandler &comm_h)
{
    const Complex imaginary{0.0, 1.0};
    const Vector3_Order<double> q{1.0 / 3.0, 0.0, 0.0}, minus_q{-1.0 / 3.0, 0.0, 0.0};
    const AtomicBasis basis(std::vector<std::size_t>{1, 1});
    const auto L = auxiliary_matrix2(1.0, imaginary, 0.0, 2.0);
    const auto K = auxiliary_matrix2(1.0, -0.5 * imaginary, 0.0, 0.5);
    const auto identity = auxiliary_matrix2(1.0, 0.0, 0.0, 1.0);
    assert_matrix_close(L * K, identity);
    const auto v = auxiliary_matrix2(2.0, 0.0, 0.0, 3.0);
    const auto P0 = auxiliary_matrix2(-2.0, 0.0, 0.0, -3.0);
    const auto Pd = auxiliary_matrix2(-1.0, 0.0, 0.0, -1.0);
    const auto Pr = P0 - Pd;
    const auto Wr = auxiliary_matrix2(2.0 / 3.0, 0.0, 0.0, 3.0 / 7.0);
    const auto active_v = transpose(K, true) * v * K;
    const auto active_P0 = L * P0 * transpose(L, true);
    const auto active_Pd = L * Pd * transpose(L, true);
    const auto active_Pr = L * Pr * transpose(L, true);
    const auto active_Wr = transpose(K, true) * Wr * K;
    assert_matrix_close(active_v, auxiliary_matrix2(2.0, -imaginary, imaginary, 1.25));
    assert_matrix_close(active_Pr,
                        auxiliary_matrix2(-3.0, -4.0 * imaginary, 4.0 * imaginary, -8.0));
    assert_matrix_close(active_P0 - active_Pd, active_Pr);
    assert_matrix_close(
        active_Wr, auxiliary_matrix2(2.0 / 3.0, -imaginary / 3.0, imaginary / 3.0, 23.0 / 84.0));
    assert_matrix_close(Wr, v + v * Pr * Wr);
    assert_matrix_close(active_Wr, active_v + active_v * active_Pr * active_Wr);

    ComplexMatrix rows(1, 2);
    rows(0, 0) = 1.0;
    rows(0, 1) = imaginary;
    const auto parent = auxiliary_parent(rows, basis);
    const auto parent_before = parent;
    const std::map<Vector3_Order<double>, ComplexMatrix> transforms{{q, L}, {minus_q, conj(L)}};
    const auto transforms_before = transforms;
    const auto active =
        transform_site_pair_density_vertex_auxiliary_basis(parent, basis, basis, transforms);
    assert(active.site_index == parent.site_index && active.atom_index == parent.atom_index);
    assert(active.n_orbitals == parent.n_orbitals && active.label == parent.label);
    assert(active.orbital_labels == parent.orbital_labels);
    assert(active.q_blocks.at(q).size() == 2);  // Keep the cancelled atom-0 block.
    assert(close(active.q_blocks.at(q).at(0)(0, 0), 0.0));
    assert(close(active.q_blocks.at(q).at(1)(0, 0), 2.0 * imaginary));
    Complex weighted_bare{0.0, 0.0}, weighted_total{0.0, 0.0};
    for (const auto &[point, blocks] : active.q_blocks)
    {
        const bool positive = point.x > 0.0;
        const double weight = positive ? 0.4 : 0.6;
        const auto v_blocks = auxiliary_interaction(positive ? active_v : conj(active_v), basis);
        const auto wr_blocks = auxiliary_interaction(positive ? active_Wr : conj(active_Wr), basis);
        const auto bare = contract_ordered_pair_vertex(blocks, v_blocks, 1)(0, 0);
        const auto total = contract_ordered_pair_vertex(blocks, wr_blocks, 1)(0, 0);
        const auto parent_bare = contract_ordered_pair_vertex(
            parent.q_blocks.at(point), auxiliary_interaction(v, basis), 1)(0, 0);
        const auto parent_total = contract_ordered_pair_vertex(
            parent.q_blocks.at(point), auxiliary_interaction(Wr, basis), 1)(0, 0);
        assert(close(bare, 5.0) && close(bare, parent_bare));
        assert(close(total, 23.0 / 21.0) && close(total, parent_total));
        weighted_bare += weight * bare;
        weighted_total += weight * total;
        PairInteractionAtomBlocks local_W;
        int block_index = 0;
        for (const auto &[I, Js] : wr_blocks)
            for (const auto &[J, W] : Js)
                if (block_index++ % comm_h.nprocs == comm_h.myid) local_W[I][J] = W;
        if (comm_h.nprocs == 4 && comm_h.myid == 3) assert(local_W.empty());
        assert(close(contract_distributed_ordered_pair_vertex(blocks, local_W, 1, comm_h,
                                                              "active auxiliary square")(0, 0),
                     total));
    }
    assert(close(weighted_bare, 5.0) && close(weighted_total, 23.0 / 21.0));
    const auto wrong_rows = rows * transpose(L, true);
    assert(close(wrong_rows(0, 0), 2.0) && close(wrong_rows(0, 1), 2.0 * imaginary));
    const auto wrong = auxiliary_parent(wrong_rows, basis);
    const auto wrong_bare = contract_ordered_pair_vertex(
        wrong.q_blocks.at(q), auxiliary_interaction(active_v, basis), 1)(0, 0);
    const auto wrong_total = contract_ordered_pair_vertex(
        wrong.q_blocks.at(q), auxiliary_interaction(active_Wr, basis), 1)(0, 0);
    assert(close(wrong_bare, 21.0) && close(wrong_total, 45.0 / 7.0));

    // Probes pull back by L^dagger, unlike the density ROW adapter's L^T.
    ComplexMatrix x(2, 1);
    x(0, 0) = 1.0;
    x(1, 0) = imaginary;
    const auto probe = transpose(L, true) * x;
    const auto wrong_probe = transpose(L, false) * x;
    assert_matrix_close(probe, x);
    const auto quadratic = transpose(x, true) * active_Pr * x;
    assert(close(quadratic(0, 0), -3.0));
    assert(close((transpose(probe, true) * Pr * probe)(0, 0), -3.0));
    assert(close((transpose(wrong_probe, true) * Pr * wrong_probe)(0, 0), -19.0));
    assert_same_site_vertex(parent, parent_before);
    for (const auto &[point, matrix] : transforms)
        assert_matrix_close(matrix, transforms_before.at(point), 0.0);
    if (comm_h.is_root())
        std::cout << "aux square: d=(0,2i) bare=" << weighted_bare << " total=" << weighted_total
                  << " wrong-adjoint=" << wrong_bare << "," << wrong_total
                  << " probe=-3 wrong-transpose=-19" << std::endl;
}

void test_auxiliary_rectangular_projection(const MpiCommHandler &comm_h)
{
    const Complex imaginary{0.0, 1.0};
    const Vector3_Order<double> q{1.0 / 3.0, 0.0, 0.0}, minus_q{-1.0 / 3.0, 0.0, 0.0};
    const AtomicBasis parent_basis(std::vector<std::size_t>{1, 1});
    const AtomicBasis active_basis(std::vector<std::size_t>{1, 0});
    ComplexMatrix L(1, 2), rows(1, 2);
    L(0, 0) = 1.0 / std::sqrt(2.0);
    L(0, 1) = imaginary / std::sqrt(2.0);
    rows(0, 0) = 1.0;
    rows(0, 1) = 2.0 * imaginary;
    const auto parent = auxiliary_parent(rows, parent_basis);
    const auto active = transform_site_pair_density_vertex_auxiliary_basis(
        parent, parent_basis, active_basis, {{q, L}, {minus_q, conj(L)}});
    assert(active.q_blocks.at(q).size() == 1 && active.q_blocks.at(q).count(1) == 0);
    assert(close(active.q_blocks.at(q).at(0)(0, 0), -1.0 / std::sqrt(2.0)));
    const auto v = auxiliary_matrix2(2.0, 0.0, 0.0, 3.0);
    const auto P0 = auxiliary_matrix2(-2.0, 0.0, 0.0, -3.0);
    const auto Pd = auxiliary_matrix2(-1.0, 0.0, 0.0, -1.0);
    const auto Pr = P0 - Pd;
    const auto active_v = L * v * transpose(L, true);
    const auto active_Pr = L * Pr * transpose(L, true);
    assert_matrix_close(active_Pr, L * P0 * transpose(L, true) - L * Pd * transpose(L, true));
    assert(close(active_v(0, 0), 2.5) && close(active_Pr(0, 0), -1.5));
    ComplexMatrix active_Wr(1, 1);
    active_Wr(0, 0) = 10.0 / 19.0;
    assert_matrix_close(active_Wr, active_v + active_v * active_Pr * active_Wr);
    const auto bare = contract_ordered_pair_vertex(
        active.q_blocks.at(q), auxiliary_interaction(active_v, active_basis), 1)(0, 0);
    const auto total = contract_ordered_pair_vertex(
        active.q_blocks.at(q), auxiliary_interaction(active_Wr, active_basis), 1)(0, 0);
    assert(close(bare, 1.25) && close(total, 5.0 / 19.0));
    const auto full_total = contract_ordered_pair_vertex(
        parent.q_blocks.at(q),
        auxiliary_interaction(auxiliary_matrix2(2.0 / 3.0, 0.0, 0.0, 3.0 / 7.0), parent_basis),
        1)(0, 0);
    assert(close(full_total, 50.0 / 21.0) && !close(total, full_total));
    const auto wrong = auxiliary_parent(rows * transpose(L, true), active_basis);
    const auto wrong_total = contract_ordered_pair_vertex(
        wrong.q_blocks.at(q), auxiliary_interaction(active_Wr, active_basis), 1)(0, 0);
    assert(close(wrong_total, 45.0 / 19.0));
    assert(close(abs2(rows) - std::norm(active.q_blocks.at(q).at(0)(0, 0)), 4.5));
    if (comm_h.is_root())
        std::cout << "aux rectangular: bare=" << bare << " total=" << total
                  << " full=" << full_total << " wrong-adjoint=" << wrong_total
                  << " analytic density norm loss=4.5" << std::endl;
}

void test_rectangular_intershell_ordered_pair_contraction(const MpiCommHandler &comm_h)
{
    // Deliberately complex 5-orbital (Ni) and 3-orbital (O) vertices.
    // This independently checks the production bra pair reversal [ba] and
    // distinguishes a 25x9 intershell tensor from an onsite square tensor.
    constexpr int n_ni = 5;
    constexpr int n_o = 3;
    ComplexMatrix D_ni(n_ni * n_ni, 2);
    ComplexMatrix D_o(n_o * n_o, 3);
    ComplexMatrix W(2, 3);
    for (int row = 0; row != D_ni.nr; ++row)
        for (int col = 0; col != D_ni.nc; ++col)
            D_ni(row, col) = Complex{0.13 * (row + 1) - 0.07 * col, 0.05 * (row - col + 1)};
    for (int row = 0; row != D_o.nr; ++row)
        for (int col = 0; col != D_o.nc; ++col)
            D_o(row, col) = Complex{-0.11 * (row + 1) + 0.09 * col, 0.04 * (2 * row + col + 1)};
    W(0, 0) = {0.7, -0.2};
    W(0, 1) = {-0.1, 0.3};
    W(0, 2) = {0.2, 0.4};
    W(1, 0) = {-0.5, 0.1};
    W(1, 1) = {0.6, -0.3};
    W(1, 2) = {0.8, 0.2};

    const auto U_ni_o = contract_rectangular_ordered_pair_vertex(D_ni, W, D_o, n_ni, n_o);
    assert(U_ni_o.nr == 25 && U_ni_o.nc == 9);
    const int a = 1, b = 4, c = 2, d = 1;
    Complex expected{0.0, 0.0};
    for (int mu = 0; mu != W.nr; ++mu)
        for (int nu = 0; nu != W.nc; ++nu)
            expected += std::conj(D_ni(b * n_ni + a, mu)) * W(mu, nu) * D_o(c * n_o + d, nu);
    assert(close(U_ni_o(a * n_ni + b, c * n_o + d), expected));

    const auto U_o_ni =
        contract_rectangular_ordered_pair_vertex(D_o, transpose(W, true), D_ni, n_o, n_ni);
    assert(U_o_ni.nr == 9 && U_o_ni.nc == 25);
    // From U_LR(ab,cd)=conj(D_L(ba)) W D_R(cd), the reversed-site
    // relation is U_RL(dc,ba)=conj(U_LR(ab,cd)).
    for (int a0 = 0; a0 != n_ni; ++a0)
        for (int b0 = 0; b0 != n_ni; ++b0)
            for (int c0 = 0; c0 != n_o; ++c0)
                for (int d0 = 0; d0 != n_o; ++d0)
                    assert(close(U_o_ni(d0 * n_o + c0, b0 * n_ni + a0),
                                 std::conj(U_ni_o(a0 * n_ni + b0, c0 * n_o + d0))));

    PairVertexAtomBlocks left{{0, D_ni}};
    PairVertexAtomBlocks right{{1, D_o}};
    PairInteractionAtomBlocks upper;
    // W is distributed, while the two vertices are replicated. Use the last
    // rank so MPI runs also check that a non-root contribution reaches root.
    if (comm_h.myid == comm_h.nprocs - 1) upper[0][1] = W;
    const auto U_distributed = contract_distributed_rectangular_ordered_pair_vertex(
        left, right, upper, n_ni, n_o, comm_h, "synthetic Ni-O");
    assert_matrix_close(U_distributed, U_ni_o);

    // Reversing atom order exercises the W^dagger path in the distributed API.
    PairVertexAtomBlocks reverse_left{{1, D_o}};
    PairVertexAtomBlocks reverse_right{{0, D_ni}};
    const auto U_reverse_distributed = contract_distributed_rectangular_ordered_pair_vertex(
        reverse_left, reverse_right, upper, n_o, n_ni, comm_h, "synthetic O-Ni");
    assert_matrix_close(U_reverse_distributed, U_o_ni);
}

void test_auxiliary_pair_reversal_and_sparse_support()
{
    const Vector3_Order<double> q{1.0 / 3.0, 0.0, 0.0}, minus_q{-1.0 / 3.0, 0.0, 0.0};
    const AtomicBasis basis(std::vector<std::size_t>{1, 1});
    const auto L = auxiliary_matrix2(1.0, Complex{0.0, 1.0}, 0.0, 2.0);
    ComplexMatrix rows(4, 2);
    for (int ab = 0; ab != 4; ++ab)
        for (int mu = 0; mu != 2; ++mu) rows(ab, mu) = Complex{0.2 + ab + mu, -0.3 + 0.7 * ab - mu};
    const auto parent = auxiliary_parent(rows, basis, 2);
    const std::map<Vector3_Order<double>, ComplexMatrix> transforms{{q, L}, {minus_q, conj(L)}};
    const auto active =
        transform_site_pair_density_vertex_auxiliary_basis(parent, basis, basis, transforms);
    for (int a = 0; a != 2; ++a)
        for (int b = 0; b != 2; ++b)
            for (int I = 0; I != 2; ++I)
                assert(close(active.q_blocks.at(minus_q).at(I)(a * 2 + b, 0),
                             std::conj(active.q_blocks.at(q).at(I)(b * 2 + a, 0))));

    // Sparse parent support is zero, not missing data. L can create a new atom's support.
    auto sparse = parent;
    sparse.q_blocks.at(q).erase(0);
    const auto sparse_before = sparse;
    auto extra = transforms;
    extra[{0.0, 0.0, 0.0}] = ComplexMatrix();  // Extra q keys are not required.
    const auto sparse_active =
        transform_site_pair_density_vertex_auxiliary_basis(sparse, basis, basis, extra);
    assert(sparse_active.q_blocks.at(q).size() == 2);
    for (int ab = 0; ab != 4; ++ab)
        assert(close(sparse_active.q_blocks.at(q).at(0)(ab, 0),
                     Complex{0.0, 1.0} * sparse.q_blocks.at(q).at(1)(ab, 0)));
    assert_same_site_vertex(sparse, sparse_before);
    sparse.q_blocks.at(q).clear();
    const auto zero =
        transform_site_pair_density_vertex_auxiliary_basis(sparse, basis, basis, transforms);
    assert(zero.q_blocks.at(q).size() == 2);
    for (const auto &[I, block] : zero.q_blocks.at(q)) assert(close(abs2(block), 0.0));
}

}  // namespace

int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MpiCommHandler comm_h(MPI_COMM_WORLD, true);

    const bool saved_mixed = global::dev_opts.use_mixed_type_ddla;
    std::vector<bool> mixed_modes{false};
    std::unique_ptr<BlacsCtxtHandler> cpu_grid;
#ifdef LIBRPA_USE_DDLA
    cpu_grid = std::make_unique<BlacsCtxtHandler>(MPI_COMM_SELF);
    cpu_grid->init();
    cpu_grid->set_grid(1, 1);
    cpu_grid->init_ddla_handle(ddla::DdlaBackend::CPU);
    mixed_modes.push_back(true);
#endif
    for (const bool mixed : mixed_modes)
    {
        global::dev_opts.use_mixed_type_ddla = mixed;
        test_dense_complex_Tdagger_C_T();
        test_random_complex_vertex_against_literal_sum(1, cpu_grid.get());
        test_random_complex_vertex_against_literal_sum(2, cpu_grid.get());
#ifdef LIBRPA_USE_DDLA
        if (mixed)
            for (const bool eligible : {false, true})
                for (const int spinor : {1, 2})
                    test_mixed_coefficient_memory_gate(*cpu_grid, eligible, spinor);
#endif
        test_nonzero_R_wrap_and_minus_fourier_phase();
        test_multiple_atoms_and_sites_remain_separate();
        test_complex_pair_reversal_and_hermiticity();
        test_real_D_reduces_to_D_W_Dt();
        test_upper_triangle_mirror_added_once();
        test_q_weighted_sum();
        test_fourier_contraction_matches_real_space_convolution();
        test_distributed_contraction_ownership(comm_h);
        test_analytic_cross_end_complex_phase();
        test_auxiliary_square_covariance(comm_h);
        test_auxiliary_rectangular_projection(comm_h);
        test_rectangular_intershell_ordered_pair_contraction(comm_h);
        test_auxiliary_pair_reversal_and_sparse_support();
#ifdef LIBRPA_USE_LIBRI
        test_production_wrapper_mpi(comm_h);
        test_s_aware_non_even_wrapper(comm_h);
        test_wrapper_nyquist_and_twisted_mesh(comm_h);
#endif
    }
    global::dev_opts.use_mixed_type_ddla = saved_mixed;
    cpu_grid.reset();

    if (comm_h.is_root()) std::cout << "PairDensityVertex tests passed\n";
    MPI_Finalize();
    return 0;
}
