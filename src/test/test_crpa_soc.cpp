#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../api/instance_manager.h"
#include "../core/crpa.h"
#include "../core/crpa_window_input.h"
#include "../core/crpa_workflow.h"
#include "../api/dataset.h"
#include "../api/dataset_helper.h"
#include "../core/dielecmodel.h"
#include "../core/meanfield_mpi.h"
#include "../core/orbital_weights.h"
#include "../core/symmetry_spin_kernel.h"
#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"
#include "librpa_input.h"

using namespace librpa_int;

namespace
{
using Z = std::complex<double>;

void close(Z actual, Z expected, double tolerance = 2e-8)
{
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(actual - expected) > tolerance)
    {
        std::cerr << "SOC actual=" << actual << " expected=" << expected << '\n';
        throw std::runtime_error("spinor cRPA numerical comparison failed");
    }
}

BasisConvention orbital_convention()
{
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    return convention;
}

void compare_meanfields(const MeanField& actual, const MeanField& reference)
{
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
        for (int k = 0; k < reference.get_n_kpoints(); ++k)
        {
            for (int band = 0; band < reference.get_n_bands(); ++band)
            {
                close(actual.get_eigenvals()[spin](k, band),
                      reference.get_eigenvals()[spin](k, band), 0);
                close(actual.get_weight()[spin](k, band), reference.get_weight()[spin](k, band), 0);
            }
            for (int component = 0; component < reference.get_n_spinor(); ++component)
            {
                const auto& a = *actual.find_wfc(spin, component, k);
                const auto& r = *reference.find_wfc(spin, component, k);
                for (int i = 0; i < r.size; ++i) close(a.c[i], r.c[i], 0);
            }
        }
}

double compare_tensors(const CrpaResult& actual, const CrpaResult& reference)
{
    double maximum = 0;
    for (std::size_t site = 0; site < reference.tensors.size(); ++site)
    {
        const auto& a = actual.tensors[site];
        const auto& r = reference.tensors[site];
        for (int i = 0; i < r.bare.size; ++i)
        {
            maximum = std::max(maximum, std::abs(a.bare.c[i] - r.bare.c[i]));
            close(a.bare.c[i], r.bare.c[i], 1e-12);
        }
        for (std::size_t iw = 0; iw < reference.frequencies.size(); ++iw)
            for (const auto& matrices :
                 {std::pair{&a.u[iw], &r.u[iw]}, std::pair{&a.w[iw], &r.w[iw]}})
                for (int i = 0; i < matrices.first->size; ++i)
                {
                    maximum =
                        std::max(maximum, std::abs(matrices.first->c[i] - matrices.second->c[i]));
                    close(matrices.first->c[i], matrices.second->c[i], 1e-11);
                }
    }
    return maximum;
}

void compare_responses(const Chi0QMap& actual, const Chi0QMap& reference)
{
    if (actual.size() != reference.size())
        throw std::runtime_error("borrowed cRPA response frequency map changed");
    for (const auto& [frequency, qs] : reference)
    {
        const auto& aq = actual.at(frequency);
        if (aq.size() != qs.size())
            throw std::runtime_error("borrowed cRPA response q map changed");
        for (const auto& [q, atoms] : qs)
        {
            const auto& aa = aq.at(q);
            if (aa.size() != atoms.size())
                throw std::runtime_error("borrowed cRPA response atom map changed");
            for (const auto& [left, rights] : atoms)
            {
                const auto& ar = aa.at(left);
                if (ar.size() != rights.size())
                    throw std::runtime_error("borrowed cRPA response atom-pair map changed");
                for (const auto& [right, block] : rights)
                {
                    const auto& ab = ar.at(right);
                    if (ab.nr != block.nr || ab.nc != block.nc)
                        throw std::runtime_error("borrowed cRPA response block changed shape");
                    for (int i = 0; i < block.size; ++i) close(ab.c[i], block.c[i], 0);
                }
            }
        }
    }
}

double compare_u_only(const CrpaResult& actual, const CrpaResult& reference)
{
    if (actual.frequencies != reference.frequencies ||
        actual.tensors.size() != reference.tensors.size())
        throw std::runtime_error("shared-response cRPA returned incomplete tensors");
    double maximum = 0;
    for (std::size_t site = 0; site < reference.tensors.size(); ++site)
    {
        const auto& a = actual.tensors[site];
        const auto& r = reference.tensors[site];
        if (!a.w.empty() || a.u.size() != r.u.size() || a.bare.nr != r.bare.nr ||
            a.bare.nc != r.bare.nc || a.site_index != r.site_index || a.spin_left != r.spin_left ||
            a.spin_right != r.spin_right)
            throw std::runtime_error("U-only route returned W or changed the local frames");
        for (int i = 0; i < r.bare.size; ++i)
        {
            close(a.bare.c[i], r.bare.c[i], 1e-12);
            maximum = std::max(maximum, std::abs(a.bare.c[i] - r.bare.c[i]));
        }
        for (std::size_t iw = 0; iw < r.u.size(); ++iw)
            for (int i = 0; i < r.u[iw].size; ++i)
            {
                close(a.u[iw].c[i], r.u[iw].c[i], 1e-11);
                maximum = std::max(maximum, std::abs(a.u[iw].c[i] - r.u[iw].c[i]));
            }
    }
    return maximum;
}

// Known 2x2 spectral projectors specify S and its square root independently
// of the production eigensolver. The imaginary off-diagonal distinguishes
// ordinary transpose from conjugate transpose in the stored C^T coefficients.
struct OrbitalMetric
{
    ComplexMatrix overlap{5, 5}, root{5, 5}, inverse{5, 5};

    OrbitalMetric()
    {
        overlap.set_as_identity_matrix();
        root.set_as_identity_matrix();
        inverse.set_as_identity_matrix();
        const double coupling = 0.31;
        overlap(0, 3) = Z(0, coupling);
        overlap(3, 0) = Z(0, -coupling);
        for (int sign : {-1, 1})
        {
            const double value = 1 + sign * coupling;
            const double a = std::sqrt(value) / 2, b = 1 / (2 * std::sqrt(value));
            root(0, 0) += a - 0.5;
            root(3, 3) += a - 0.5;
            root(0, 3) += Z(0, sign * a);
            root(3, 0) -= Z(0, sign * a);
            inverse(0, 0) += b - 0.5;
            inverse(3, 3) += b - 0.5;
            inverse(0, 3) += Z(0, sign * b);
            inverse(3, 0) -= Z(0, sign * b);
        }
    }
};

void rotate_rows(ComplexMatrix& q, int first, int second, double angle, double phase)
{
    const double c = std::cos(angle);
    const Z s = std::polar(std::sin(angle), phase);
    for (int band = 0; band < q.nc; ++band)
    {
        const Z a = q(first, band), b = q(second, band);
        q(first, band) = c * a + s * b;
        q(second, band) = -std::conj(s) * a + c * b;
    }
}

void test_spinor_character_nonorthogonal_metric_and_pdos()
{
    AtomicBasis basis(std::vector<std::size_t>{2, 3});
    basis.set_l_shells({{0, 0}, {1}});
    const auto convention = orbital_convention();
    OrbitalMetric metric;
    MeanField mf(1, 2, 10, 5, 2);
    std::vector<std::array<ComplexMatrix, 2>> orthogonal, coefficients;
    double coherence = 0, noncollinearity = 0;
    for (int k = 0; k < 2; ++k)
    {
        ComplexMatrix q(10, 10);
        q.set_as_identity_matrix();
        // Spatial and spin rotations with independent phases make the local
        // spin directions noncollinear, while Q remains exactly unitary.
        rotate_rows(q, 0, 3, 0.23 + 0.09 * k, 0.71);
        rotate_rows(q, 5, 8, 0.41 + 0.05 * k, -0.37);
        rotate_rows(q, 0, 5, 0.32, 0.61);
        rotate_rows(q, 1, 7, 0.48, -0.82);
        rotate_rows(q, 3, 6, 0.27, 0.42);
        rotate_rows(q, 4, 9, 0.38, -0.21);
        std::array<ComplexMatrix, 2> parts{ComplexMatrix(5, 10), ComplexMatrix(5, 10)};
        for (int component = 0; component < 2; ++component)
            for (int ao = 0; ao < 5; ++ao)
                for (int band = 0; band < 10; ++band)
                    parts[component](ao, band) =
                        q(5 * component + ao, band) * std::polar(1.0, 0.17 * (1 + k + band));
        orthogonal.push_back(parts);
        const auto polarization = [&](int ao)
        {
            const Z up = parts[0](ao, 0), down = parts[1](ao, 0);
            const double density = std::norm(up) + std::norm(down);
            return std::array<double, 3>{2 * std::real(std::conj(up) * down) / density,
                                         2 * std::imag(std::conj(up) * down) / density,
                                         (std::norm(up) - std::norm(down)) / density};
        };
        const auto first = polarization(0), second = polarization(3);
        double cross_squared = 0;
        for (int axis = 0; axis < 3; ++axis)
        {
            const double cross = first[(axis + 1) % 3] * second[(axis + 2) % 3] -
                                 first[(axis + 2) % 3] * second[(axis + 1) % 3];
            cross_squared += cross * cross;
        }
        noncollinearity = std::max(noncollinearity, std::sqrt(cross_squared));
        std::array<ComplexMatrix, 2> c{metric.inverse * parts[0], metric.inverse * parts[1]};
        coefficients.push_back(c);
        for (int component = 0; component < 2; ++component)
            mf.get_eigenvectors()[0][component][k] = transpose(c[component], false);
        for (int band = 0; band < 10; ++band)
        {
            mf.get_eigenvals()[0](k, band) = -0.7 + 0.19 * band + 0.03 * k;
            mf.get_weight()[0](k, band) = band < 4 ? 0.5 : 0;
            for (int ao = 0; ao < 5; ++ao)
                coherence = std::max(coherence, std::abs(std::imag(std::conj(parts[0](ao, band)) *
                                                                   parts[1](ao, band))));
        }
    }
    if (coherence < 0.1 || noncollinearity < 0.2)
        throw std::runtime_error("spinor orbital fixture lacks complex noncollinear coherence");
    const auto original = mf;
    const auto original_overlap = metric.overlap;
    const std::vector<double> kw{0.27, 0.73};
    const auto result = analyze_orbital_weights(basis, convention, {0, 1}, mf,
                                                {metric.overlap, metric.overlap}, kw, 1e-11);
    if (result.n_spins != 1 || result.n_spinor != 2 || result.states.size() != 20 ||
        result.groups.size() != 2)
        throw std::runtime_error("spinor orbitals became independent spin channels");
    double maximum_mulliken = 0, maximum_lowdin = 0;
    for (int k = 0; k < 2; ++k)
    {
        const std::array<ComplexMatrix, 2> sc{metric.root * orthogonal[k][0],
                                              metric.root * orthogonal[k][1]};
        for (int band = 0; band < 10; ++band)
        {
            const auto& state = result.states[10 * k + band];
            std::array<double, 2> group_m{}, group_l{};
            for (int ao = 0; ao < 5; ++ao)
            {
                double expected_m = 0, expected_l = 0;
                for (int component = 0; component < 2; ++component)
                {
                    expected_m += std::real(std::conj(coefficients[k][component](ao, band)) *
                                            sc[component](ao, band));
                    expected_l += std::norm(orthogonal[k][component](ao, band));
                }
                close(state.mulliken_ao[ao], expected_m, 2e-12);
                close(state.lowdin_ao[ao], expected_l, 2e-12);
                maximum_mulliken =
                    std::max(maximum_mulliken, std::abs(state.mulliken_ao[ao] - expected_m));
                maximum_lowdin =
                    std::max(maximum_lowdin, std::abs(state.lowdin_ao[ao] - expected_l));
                group_m[ao < 2 ? 0 : 1] += expected_m;
                group_l[ao < 2 ? 0 : 1] += expected_l;
            }
            close(state.norm, 1, 2e-12);
            for (int group = 0; group < 2; ++group)
            {
                close(state.mulliken[group], group_m[group], 2e-12);
                close(state.lowdin[group], group_l[group], 2e-12);
            }
        }
    }
    compare_meanfields(mf, original);
    for (int i = 0; i < metric.overlap.size; ++i)
        close(metric.overlap.c[i], original_overlap.c[i], 0);

    // A general complex SU(2) change of spin representation must preserve
    // total charge weights, including the cancellation of component coherences.
    MeanField rotated = mf;
    const Z alpha = std::polar(std::cos(0.43), 0.29);
    const Z beta = std::polar(std::sin(0.43), -0.57);
    for (int k = 0; k < 2; ++k)
        for (int ao = 0; ao < 5; ++ao)
            for (int band = 0; band < 10; ++band)
            {
                const Z up = coefficients[k][0](ao, band), down = coefficients[k][1](ao, band);
                rotated.get_eigenvectors()[0][0][k](band, ao) = alpha * up + beta * down;
                rotated.get_eigenvectors()[0][1][k](band, ao) =
                    -std::conj(beta) * up + std::conj(alpha) * down;
            }
    const auto rotated_result = analyze_orbital_weights(
        basis, convention, {0, 1}, rotated, {metric.overlap, metric.overlap}, kw, 1e-11);
    double maximum_rotation = 0;
    for (std::size_t i = 0; i < result.states.size(); ++i)
    {
        const auto& first = result.states[i];
        const auto& second = rotated_result.states[i];
        for (int ao = 0; ao < 5; ++ao)
        {
            close(first.mulliken_ao[ao], second.mulliken_ao[ao], 2e-12);
            close(first.lowdin_ao[ao], second.lowdin_ao[ao], 2e-12);
            maximum_rotation =
                std::max(maximum_rotation, std::abs(first.lowdin_ao[ao] - second.lowdin_ao[ao]));
        }
        for (int group = 0; group < 2; ++group)
        {
            close(first.mulliken[group], second.mulliken[group], 2e-12);
            close(first.lowdin[group], second.lowdin[group], 2e-12);
        }
    }
    const double step = 0.005, sigma = 0.23;
    std::vector<double> grid;
    for (int i = 0; i <= 1600; ++i) grid.push_back(-4 + i * step);
    std::vector<double> mulliken, lowdin;
    gaussian_orbital_pdos(result, grid, sigma, mulliken, lowdin);
    std::vector<double> rotated_m, rotated_l;
    gaussian_orbital_pdos(rotated_result, grid, sigma, rotated_m, rotated_l);
    for (std::size_t i = 0; i < lowdin.size(); ++i)
    {
        close(mulliken[i], rotated_m[i], 2e-12);
        close(lowdin[i], rotated_l[i], 2e-12);
    }
    std::vector<std::array<ComplexMatrix, 2>> sc_values;
    for (const auto& q : orthogonal) sc_values.push_back({metric.root * q[0], metric.root * q[1]});
    double integral_m = 0, integral_l = 0;
    for (std::size_t e = 0; e < grid.size(); ++e)
    {
        std::array<double, 2> expected_m{}, expected_l{};
        for (int k = 0; k < 2; ++k)
            for (int band = 0; band < 10; ++band)
            {
                const double delta = (grid[e] - mf.get_eigenvals()[0](k, band)) / sigma;
                const double kernel = kw[k] * std::exp(-0.5 * delta * delta) /
                                      (sigma * std::sqrt(2 * std::acos(-1.0)));
                for (int ao = 0; ao < 5; ++ao)
                    for (int component = 0; component < 2; ++component)
                    {
                        expected_l[ao < 2 ? 0 : 1] +=
                            kernel * std::norm(orthogonal[k][component](ao, band));
                        expected_m[ao < 2 ? 0 : 1] +=
                            kernel * std::real(std::conj(coefficients[k][component](ao, band)) *
                                               sc_values[k][component](ao, band));
                    }
            }
        for (int group = 0; group < 2; ++group)
        {
            close(mulliken[2 * e + group], expected_m[group], 2e-12);
            close(lowdin[2 * e + group], expected_l[group], 2e-12);
        }
        const double quadrature = (e == 0 || e + 1 == grid.size()) ? 0.5 * step : step;
        integral_m += quadrature * (mulliken[2 * e] + mulliken[2 * e + 1]);
        integral_l += quadrature * (lowdin[2 * e] + lowdin[2 * e + 1]);
    }
    close(integral_m, 10, 2e-11);
    close(integral_l, 10, 2e-11);
    if (global::myid_global == 0)
        std::cout << "Spinor orbital analytic complex metric: Mulliken max=" << maximum_mulliken
                  << "; Lowdin max=" << maximum_lowdin << "; SU(2) max=" << maximum_rotation
                  << "; Gaussian integrated states=" << integral_l << '\n';
}

void test_spinor_character_scalar_packing_limit()
{
    AtomicBasis basis(std::vector<std::size_t>{2, 3});
    basis.set_l_shells({{0, 0}, {1}});
    const auto convention = orbital_convention();
    OrbitalMetric metric;
    MeanField scalar(2, 2, 5, 5), spinor(1, 2, 10, 5, 2);
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
        {
            ComplexMatrix q(5, 5);
            q.set_as_identity_matrix();
            rotate_rows(q, 0, 3, 0.19 + 0.13 * spin + 0.07 * k, 0.42);
            const auto c = metric.inverse * q;
            scalar.get_eigenvectors()[spin][0][k] = transpose(c, false);
            ComplexMatrix packed(10, 5);
            for (int band = 0; band < 5; ++band)
            {
                const double energy = -0.7 + 0.42 * band + 0.05 * spin + 0.03 * k;
                const double occupation = band < 2 ? 0.5 : 0;
                scalar.get_eigenvals()[spin](k, band) = energy;
                scalar.get_weight()[spin](k, band) = occupation;
                spinor.get_eigenvals()[0](k, 2 * band + spin) = energy;
                spinor.get_weight()[0](k, 2 * band + spin) = occupation;
                for (int ao = 0; ao < 5; ++ao) packed(2 * band + spin, ao) = c(ao, band);
            }
            spinor.get_eigenvectors()[0][spin][k] = packed;
        }
    const auto original_scalar = scalar, original_spinor = spinor;
    const std::vector<double> kw{0.27, 0.73};
    const auto a = analyze_orbital_weights(basis, convention, {0, 1}, scalar,
                                           {metric.overlap, metric.overlap}, kw, 1e-11);
    const auto b = analyze_orbital_weights(basis, convention, {0, 1}, spinor,
                                           {metric.overlap, metric.overlap}, kw, 1e-11);
    if (a.n_spins != 2 || a.n_spinor != 1 || b.n_spins != 1 || b.n_spinor != 2)
        throw std::runtime_error("scalar to spinor packing lost physical spin dimensions");
    for (int spin = 0; spin < 2; ++spin)
        for (int k = 0; k < 2; ++k)
            for (int band = 0; band < 5; ++band)
            {
                const auto& sa = a.states[(spin * 2 + k) * 5 + band];
                const auto& sb = b.states[10 * k + 2 * band + spin];
                close(sa.energy, sb.energy, 0);
                close(sa.occupation, sb.occupation, 0);
                close(sa.norm, sb.norm, 2e-12);
                for (int ao = 0; ao < 5; ++ao)
                {
                    close(sa.mulliken_ao[ao], sb.mulliken_ao[ao], 2e-12);
                    close(sa.lowdin_ao[ao], sb.lowdin_ao[ao], 2e-12);
                }
                for (int group = 0; group < 2; ++group)
                {
                    close(sa.mulliken[group], sb.mulliken[group], 2e-12);
                    close(sa.lowdin[group], sb.lowdin[group], 2e-12);
                }
            }
    const std::vector<double> grid{-0.45, 0.0, 0.71};
    std::vector<double> scalar_m, scalar_l, spinor_m, spinor_l;
    gaussian_orbital_pdos(a, grid, 0.23, scalar_m, scalar_l);
    gaussian_orbital_pdos(b, grid, 0.23, spinor_m, spinor_l);
    for (std::size_t energy = 0; energy < grid.size(); ++energy)
        for (int group = 0; group < 2; ++group)
        {
            const auto offset = 2 * energy + group;
            close(spinor_m[offset], scalar_m[offset] + scalar_m[6 + offset], 2e-12);
            close(spinor_l[offset], scalar_l[offset] + scalar_l[6 + offset], 2e-12);
        }
    compare_meanfields(scalar, original_scalar);
    compare_meanfields(spinor, original_spinor);
    if (global::myid_global == 0)
        std::cout << "Spinor orbital zero-SOC packing: charge weights and PDOS equal summed "
                     "collinear channels\n";
}

struct Fixture
{
    Dataset ds{MPI_COMM_WORLD};
    std::unique_ptr<Chi0> chi;
    std::unique_ptr<CorrelatedSubspace> frame;
    std::vector<atpair_t> pairs;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    std::array<ComplexMatrix, 2> density{ComplexMatrix(3, 3), ComplexMatrix(3, 3)};
    ComplexMatrix coefficients, coulomb{2, 2};
    std::vector<ComplexMatrix> coefficients_k;
    int components, bands;

    Fixture(bool spinor, double soc_angle, double representation_angle = 0,
            bool dispersive_phase = false)
        : coefficients(spinor ? 6 : 3, spinor ? 6 : 3),
          components(spinor ? 2 : 1),
          bands(spinor ? 6 : 3)
    {
        ds.basis_wfc.set(std::vector<std::size_t>{3});
        ds.basis_wfc.set_l_shells({{0, 0, 0}});
        ds.basis_aux.set(std::vector<std::size_t>{2});
        ds.mf.set(1, 4, bands, 3, components);
        ds.mf.get_efermi() = 0;
        ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        const double pi = std::acos(-1.0);
        ds.pbc.set_kgrids_kvec(4, 1, 1, {0, 0, 0, pi / 2, 0, 0, pi, 0, 0, -pi / 2, 0, 0});
        ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD,
                                4);
        ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(bands, 3, bands, 3);
        ds.desc_abf.reset_handler(ds.blacs_h);
        ds.desc_abf.init_1b1p(2, 2, 0, 0);
        ds.tfg.reset(24);
        ds.tfg.generate_minimax(0.4, 4.0);
        const double f0[] = {0.4, 0.3, 0.4, 0.3, 0.7, 0.2, 0.4, 0.2, 0.9};
        const double f1[] = {0.9, 0.13, -0.17, 0.13, 0.1, 0.07, -0.17, 0.07, 0.4};
        for (int i = 0; i < 9; ++i)
        {
            density[0].c[i] = f0[i];
            density[1].c[i] = f1[i];
        }
        coulomb(0, 0) = 2;
        coulomb(1, 1) = 1.3;
        coulomb(0, 1) = coulomb(1, 0) = 0.2;

        if (!spinor)
            coefficients.set_as_identity_matrix();
        else
        {
            // Kramers pairs (0,1), (2,3), (4,5), with complex inter-orbital SOC.
            const double c = std::cos(soc_angle), s = std::sin(soc_angle);
            coefficients(0, 0) = coefficients(3, 1) = c;
            coefficients(4, 0) = coefficients(1, 1) = Z(0, s);
            coefficients(1, 2) = coefficients(4, 3) = c;
            coefficients(3, 2) = coefficients(0, 3) = Z(0, s);
            coefficients(2, 4) = coefficients(5, 5) = 1;
            // Change the spin representation of both KS and output columns.
            const auto original = coefficients;
            const double u = std::cos(representation_angle), v = std::sin(representation_angle);
            for (int i = 0; i < 3; ++i)
                for (int n = 0; n < bands; ++n)
                {
                    coefficients(i, n) = u * original(i, n) + Z(0, v) * original(3 + i, n);
                    coefficients(3 + i, n) = Z(0, v) * original(i, n) + u * original(3 + i, n);
                }
        }
        const double energies[] = {-0.7, 0.3, 1.1};
        SiteOrbitalGroup site{"spinor-test", 0, bands, 0, {}};
        frame = std::make_unique<CorrelatedSubspace>(std::vector<SiteOrbitalGroup>{site},
                                                     ds.pbc.kfrac_list, ds.pbc.Rlist,
                                                     3 * components, bands, 1);
        ComplexMatrix overlap(3 * components, 3 * components);
        overlap.set_as_identity_matrix();
        for (int k = 0; k < 4; ++k)
        {
            auto ck = coefficients;
            if (dispersive_phase)
                for (int s = 0; s < components; ++s)
                    for (int n = 0; n < bands; ++n) ck(3 * s + 1, n) *= std::exp(Z(0, pi * k / 2));
            coefficients_k.push_back(ck);
            for (int s = 0; s < components; ++s)
            {
                ComplexMatrix wfc(bands, 3);
                for (int n = 0; n < bands; ++n)
                    for (int i = 0; i < 3; ++i) wfc(n, i) = ck(3 * s + i, n);
                ds.mf.get_eigenvectors()[0][s][k] = wfc;
            }
            for (int n = 0; n < bands; ++n)
            {
                ds.mf.get_eigenvals()[0](k, n) = energies[n / components];
                ds.mf.get_weight()[0](k, n) = (n < components ? 1.0 : 0.0) * (spinor ? 0.25 : 0.5);
            }
            frame->set_S_k(k, overlap);
            frame->set_W_k(k, coefficients);
            frame->build_spin_k(0, k, transpose(coefficients, false));
            if (ds.comm_h.is_root())
                ds.vq[0][0][ds.pbc.klist[k]] = std::make_shared<ComplexMatrix>(coulomb);
        }
        frame->compute_T_R();
        if (ds.comm_h.is_root())
        {
            ds.cs_data.use_libri = true;
            auto values = std::make_shared<std::valarray<double>>(0.0, 18);
            for (int mu = 0; mu < 2; ++mu)
                for (int i = 0; i < 9; ++i) (*values)[9 * mu + i] = 0.5 * density[mu].c[i].real();
            ds.cs_data.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({2, 3, 3}, values);
        }
        const auto assigned = dispatch_upper_triangular_tasks(1, ds.blacs_h.myid, ds.blacs_h.nprows,
                                                              ds.blacs_h.npcols, ds.blacs_h.myprow,
                                                              ds.blacs_h.mypcol);
        pairs.assign(assigned.begin(), assigned.end());
        chi = std::make_unique<Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
                                     ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
        chi->gf_threshold = chi->libri_threshold_C = chi->libri_threshold_G = 0;
    }

    Chi0::BandSelection selection() const
    {
        std::vector<double> mask(bands, 0);
        for (int n = 0; n < 2 * components; ++n) mask[n] = 1;
        return Chi0::BandSelection(1, std::vector<std::vector<double>>(4, mask));
    }

    Z vertex(int mu, int n, int m) const
    {
        Z result{};
        for (int s = 0; s < components; ++s)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    result += std::conj(coefficients(3 * s + i, n)) * density[mu](i, j) *
                              coefficients(3 * s + j, m);
        return result;
    }

    Z transition(int mu, int n, int m, int k, int kq) const
    {
        Z result{};
        for (int s = 0; s < components; ++s)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    result += std::conj(coefficients_k[k](3 * s + i, n)) * density[mu](i, j) *
                              coefficients_k[kq](3 * s + j, m);
        return result;
    }

    ComplexMatrix response(double frequency, bool excluded, int iq = 0,
                           const Chi0::BandSelection* weights = nullptr) const
    {
        ComplexMatrix result(2, 2);
        for (int k = 0; k < 4; ++k)
            for (int n = 0; n < bands; ++n)
                for (int m = 0; m < bands; ++m)
                {
                    if (excluded && (n >= 2 * components || m >= 2 * components)) continue;
                    const double occupation_difference =
                        (n < components ? 1.0 : 0.0) - (m < components ? 1.0 : 0.0);
                    if (occupation_difference == 0) continue;
                    const int kq = (k + iq) % 4;
                    const double gap =
                        ds.mf.get_eigenvals()[0](kq, m) - ds.mf.get_eigenvals()[0](k, n);
                    const Z factor = (components == 1 ? 2.0 : 1.0) * occupation_difference /
                                     (4.0 * (Z(0, frequency) - gap)) *
                                     (weights ? (*weights)[0][k][n] * (*weights)[0][kq][m] : 1.0);
                    for (int mu = 0; mu < 2; ++mu)
                        for (int nu = 0; nu < 2; ++nu)
                            result(mu, nu) += factor * transition(mu, n, m, k, kq) *
                                              std::conj(transition(nu, n, m, k, kq));
                }
        return result;
    }

    void compare_response(bool excluded, const Chi0::BandSelection* weights = nullptr)
    {
        if (weights)
            chi->set_band_selection(*weights);
        else if (excluded)
            chi->set_band_selection(selection());
        else
            chi->clear_band_selection();
        chi->build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_AUTO, ds.cs_data, pairs, ds.basis_aux,
                   no_shrink, ds.blacs_h);
        for (double frequency : ds.tfg.get_freq_nodes())
            for (int iq = 0; iq < 4; ++iq)
            {
                const auto& q = ds.pbc.klist[iq];
                ComplexMatrix local(2, 2), total(2, 2);
                const auto f = chi->get_chi0_q().find(frequency);
                if (f != chi->get_chi0_q().end())
                {
                    const auto it = f->second.find(q);
                    if (it != f->second.end() && it->second.count(0) && it->second.at(0).count(0))
                        local = it->second.at(0).at(0);
                }
                MPI_Allreduce(local.c, total.c, 4, MPI_C_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
                const auto exact = response(frequency, excluded, iq, weights);
                for (int i = 0; i < 4; ++i) close(total.c[i], exact.c[i]);
            }
        chi->clear_band_selection();
    }

    ComplexMatrix screen(const ComplexMatrix& response) const
    {
        ComplexMatrix dielectric(2, 2);
        dielectric.set_as_identity_matrix();
        dielectric -= coulomb * response;
        const Z determinant =
            dielectric(0, 0) * dielectric(1, 1) - dielectric(0, 1) * dielectric(1, 0);
        ComplexMatrix inverse(2, 2);
        inverse(0, 0) = dielectric(1, 1) / determinant;
        inverse(1, 1) = dielectric(0, 0) / determinant;
        inverse(0, 1) = -dielectric(0, 1) / determinant;
        inverse(1, 0) = -dielectric(1, 0) / determinant;
        return inverse * coulomb;
    }

    CrpaResult run()
    {
        CrpaContext context{*chi,      ds.cs_data, ds.basis_aux, pairs,
                            no_shrink, ds.vq,      ds.blacs_h,   ds.desc_abf};
        return compute_crpa_onsite(context, selection(), {frame.get()});
    }

    CrpaResult run_staged()
    {
        CrpaContext context{*chi,      ds.cs_data, ds.basis_aux, pairs,
                            no_shrink, ds.vq,      ds.blacs_h,   ds.desc_abf};
        // Keep a nonempty Pd resident, and a different incoming selector. This
        // exposes response replacement or selector leakage between the stages.
        compare_response(true);
        const auto resident = chi->get_chi0_q();
        auto incoming = selection();
        for (auto& row : incoming[0]) row[0] = 0;
        chi->set_band_selection(incoming);
        const auto original = ds.mf;
        const auto projection = prepare_crpa_local_projection(context, selection(), {frame.get()});
        const auto full = build_crpa_total_response(context);
        const auto original_full = full;
        auto excluded = build_crpa_excluded_response(context, selection());
        auto constrained = form_crpa_constrained_response(context, full, std::move(excluded));
        const auto uc = screen_crpa_response(context, std::move(constrained));
        const auto wc = screen_crpa_response(context, full);
        const auto u = contract_crpa_interaction(context, projection, uc);
        const auto w = contract_crpa_interaction(context, projection, wc);
        auto result = projection.bare_result;
        for (std::size_t i = 0; i < result.tensors.size(); ++i)
        {
            result.tensors[i].u = u[i];
            result.tensors[i].w = w[i];
        }
        compare_responses(full, original_full);
        compare_responses(chi->get_chi0_q(), resident);
        if (chi->get_band_selection() != incoming)
            throw std::runtime_error("staged cRPA leaked its response selector");
        compare_meanfields(ds.mf, original);
        return result;
    }

    CrpaResult run_shared(bool resident_alias)
    {
        CrpaContext context{*chi,      ds.cs_data, ds.basis_aux, pairs,
                            no_shrink, ds.vq,      ds.blacs_h,   ds.desc_abf};
        const auto original = ds.mf;
        auto full = build_crpa_total_response(context);
        const auto original_full = full;
        if (resident_alias)
            chi->swap_chi0_q(full);
        else
            compare_response(true);
        const auto resident = chi->get_chi0_q();
        auto incoming = selection();
        for (auto& row : incoming[0]) row[0] = 0;
        chi->set_band_selection(incoming);
        if (ds.comm_h.is_root())
            std::cout << "Shared-P0 U-only begin: " << (components == 2 ? "SOC" : "scalar")
                      << (resident_alias ? " resident alias" : " detached response") << std::endl;
        const auto result = evaluate_crpa_onsite_from_response(
            context, selection(), {frame.get()}, resident_alias ? chi->get_chi0_q() : full);
        if (ds.comm_h.is_root()) std::cout << "Shared-P0 U-only end\n";
        if (!resident_alias) compare_responses(full, original_full);
        compare_responses(chi->get_chi0_q(), resident);
        if (chi->get_band_selection() != incoming)
            throw std::runtime_error("shared-response cRPA leaked its response selector");
        compare_meanfields(ds.mf, original);
        return result;
    }

    void compare_tensors(const CrpaResult& result) const
    {
        for (std::size_t iw = 0; iw < result.frequencies.size(); ++iw)
        {
            const double frequency = result.frequencies[iw];
            std::array<ComplexMatrix, 3> kernels{coulomb, ComplexMatrix(2, 2), ComplexMatrix(2, 2)};
            for (int iq = 0; iq < 4; ++iq)
            {
                const auto p0 = response(frequency, false, iq), pd = response(frequency, true, iq);
                kernels[1] += 0.25 * screen(p0 - pd);
                kernels[2] += 0.25 * screen(p0);
            }
            const auto& tensor = result.tensors.at(0);
            const std::array<const ComplexMatrix*, 3> actual{
                &tensor.bare, &tensor.u[iw], tensor.w.empty() ? nullptr : &tensor.w[iw]};
            for (int a = 0; a < bands; ++a)
                for (int b = 0; b < bands; ++b)
                    for (int c = 0; c < bands; ++c)
                        for (int d = 0; d < bands; ++d)
                            for (int family = 0; family < (tensor.w.empty() ? 2 : 3); ++family)
                            {
                                Z expected{};
                                for (int mu = 0; mu < 2; ++mu)
                                    for (int nu = 0; nu < 2; ++nu)
                                        expected += std::conj(vertex(mu, b, a)) *
                                                    kernels[family](mu, nu) * vertex(nu, c, d);
                                close((*actual[family])(a * bands + b, c * bands + d), expected);
                            }
        }
    }
};

void test_zero_soc_limit()
{
    Fixture scalar(false, 0), spinor(true, 0);
    scalar.compare_response(false);
    scalar.compare_response(true);
    spinor.compare_response(false);
    spinor.compare_response(true);
    const auto reference = scalar.run(), packed = spinor.run();
    scalar.compare_tensors(reference);
    spinor.compare_tensors(packed);
    const auto staged = scalar.run_staged();
    scalar.compare_tensors(staged);
    const auto scalar_shared = scalar.run_shared(false), packed_shared = spinor.run_shared(true);
    scalar.compare_tensors(scalar_shared);
    spinor.compare_tensors(packed_shared);
    const double shared_maximum =
        std::max(compare_u_only(scalar_shared, reference), compare_u_only(packed_shared, packed));
    const double staged_maximum = compare_tensors(staged, reference);
    if (scalar.ds.comm_h.is_root())
        std::cout << "Scalar/zero-SOC cRPA composition: staged v/U/W max=" << staged_maximum
                  << "; shared-P0 v/U max=" << shared_maximum
                  << "; caller response, resident response, selector, E/C/f unchanged\n";
    for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b)
            for (int c = 0; c < 6; ++c)
                for (int d = 0; d < 6; ++d)
                    for (std::size_t iw = 0; iw < reference.frequencies.size(); ++iw)
                    {
                        const bool same_components = a % 2 == b % 2 && c % 2 == d % 2;
                        const int row = (a / 2) * 3 + b / 2, col = (c / 2) * 3 + d / 2;
                        for (int family = 0; family < 3; ++family)
                        {
                            const auto& s = reference.tensors[0];
                            const auto& p = packed.tensors[0];
                            const auto& x =
                                family == 0 ? s.bare : (family == 1 ? s.u[iw] : s.w[iw]);
                            const auto& y =
                                family == 0 ? p.bare : (family == 1 ? p.u[iw] : p.w[iw]);
                            close(y(a * 6 + b, c * 6 + d), same_components ? x(row, col) : Z{});
                        }
                    }
}

struct MagneticModel
{
    Matz hamiltonian{6, 6, MAJOR::COL};
    Matz eigenvectors;
    std::vector<double> eigenvalues;
    std::vector<double> positions;
    std::vector<int> operations;
    std::vector<int> antiunitary{0, 0, 0, 1, 1, 1};
    const double lattice[9]{6.0, 0.0, 0.0, 3.0, 3.0 * std::sqrt(3.0), 0.0, 0.0, 0.0, 8.0};
    const double reciprocal[9]{TWO_PI / 6.0,
                               -TWO_PI / (6.0 * std::sqrt(3.0)),
                               0.0,
                               0.0,
                               TWO_PI / (3.0 * std::sqrt(3.0)),
                               0.0,
                               0.0,
                               0.0,
                               TWO_PI / 8.0};

    MagneticModel()
    {
        hamiltonian.zero_out();
        for (int atom = 0; atom != 3; ++atom)
        {
            const double angle = TWO_PI * atom / 3.0;
            positions.insert(positions.end(), {0.8 * std::cos(angle), 0.8 * std::sin(angle), 0.0});
            hamiltonian(atom, atom + 3) = std::polar(1.0, -angle);
            hamiltonian(atom + 3, atom) = std::conj(hamiltonian(atom, atom + 3));
            for (int spinor = 0; spinor != 2; ++spinor)
            {
                const int next = (atom + 1) % 3;
                const Z hopping{0.15, spinor == 0 ? 0.04 : -0.04};
                hamiltonian(atom + 3 * spinor, next + 3 * spinor) = hopping;
                hamiltonian(next + 3 * spinor, atom + 3 * spinor) = std::conj(hopping);
            }
        }
        eigsh(hamiltonian, eigenvalues, eigenvectors);
        if (eigenvalues[3] - eigenvalues[2] <= 1.0)
            throw std::runtime_error("magnetic model must be gapped");

        // Row fractional C3, C3^2 and their horizontal-mirror partners.
        const int rotations[3][9]{{1, 0, 0, 0, 1, 0, 0, 0, 1},
                                  {-1, 1, 0, -1, 0, 0, 0, 0, 1},
                                  {0, -1, 0, 1, -1, 0, 0, 0, 1}};
        for (int mirror = 0; mirror != 2; ++mirror)
            for (int rotation = 0; rotation != 3; ++rotation)
                for (int index = 0; index != 9; ++index)
                    operations.push_back(index == 8 && mirror ? -1 : rotations[rotation][index]);

        // Independent Cartesian Hamiltonian covariance, including Theta Mz.
        for (int mirror = 0; mirror != 2; ++mirror)
            for (int rotation = 0; rotation != 3; ++rotation)
            {
                Matz action(6, 6, MAJOR::COL);
                action.zero_out();
                const double angle = TWO_PI * rotation / 3.0 + mirror * PI;
                for (int atom = 0; atom != 3; ++atom)
                    for (int spinor = 0; spinor != 2; ++spinor)
                    {
                        const Z u = std::polar(1.0, (spinor == 0 ? -0.5 : 0.5) * angle);
                        const int target_atom = (atom + rotation) % 3;
                        if (mirror)
                            action(target_atom + 3 * (1 - spinor), atom + 3 * spinor) =
                                (spinor == 0 ? -1.0 : 1.0) * std::conj(u);
                        else
                            action(target_atom + 3 * spinor, atom + 3 * spinor) = u;
                    }
                for (int i = 0; i != 6; ++i)
                    for (int j = 0; j != 6; ++j)
                    {
                        Z transformed = 0.0;
                        for (int a = 0; a != 6; ++a)
                            for (int b = 0; b != 6; ++b)
                                transformed +=
                                    action(i, a) *
                                    (mirror ? std::conj(hamiltonian(a, b)) : hamiltonian(a, b)) *
                                    std::conj(action(j, b));
                        close(transformed, hamiltonian(i, j), 1e-12);
                    }
            }
    }
};

void test_magnetic_qstar_screening()
{
    // Complex periodic Kramers state: the antiunitary q member is actually
    // restored, while both selected GF branches retain all original k points.
    Fixture f(true, 0.37, 0, true);
    const auto full = f.run();
    LibrpaOptions options;
    librpa_init_options(&options);
    options.use_symmetry_rpa = LIBRPA_SWITCH_ON;
    int legacy_rejected = 0, all_legacy_rejected = 0;
    try
    {
        initialize_ds_chi0_qstar(f.ds, options);
    }
    catch (const std::exception&)
    {
        legacy_rejected = 1;
    }
    MPI_Allreduce(&legacy_rejected, &all_legacy_rejected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_legacy_rejected)
        throw std::runtime_error("SOC cRPA accepted an implicit time-reversal declaration");
    f.ds.basis_aux.set_l_shells({{0, 0}});
    auto& ctx = f.ds.symmetry_context;
    ctx.set_crystal_structure(f.ds.pbc.latvec, f.ds.pbc.G, {{0, 0}}, {{0, {0, 0, 0}}});
    ctx.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    ctx.build_periodic_mappings(f.ds.pbc, f.ds.pbc.Rlist);
    ctx.set_available();
    const auto implicit_view = build_symmetry_qpoint_view(ctx, f.ds.pbc, true);
    if (implicit_view.restore_mode != SymmetryQPointRestoreMode::NONE ||
        implicit_view.representatives.size() != 4)
        throw std::runtime_error("a legacy identity-only table silently enabled time reversal");
    SymmetrySpinOperation identity;
    identity.spin_source = SymmetrySpinActionSource::DerivedFromSpatialSOC;
    ctx.set_symmetry_spin_operations({identity}, true);
    ctx.build_periodic_mappings(f.ds.pbc, f.ds.pbc.Rlist);
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    ctx.build_rsh_rotations(convention, 0);
    ctx.build_kstar_member_rotations(0);
    ctx.set_available();
    f.chi = std::make_unique<Chi0>(f.ds.mf, f.ds.basis_wfc, f.ds.basis_aux, f.ds.pbc, ctx, f.ds.tfg,
                                   f.ds.scfk_blacs_ctxt, f.ds.desc_wfc_kb_full, false, true);
    f.chi->qstar_only_symmetry = true;
    f.chi->gf_threshold = f.chi->libri_threshold_C = f.chi->libri_threshold_G = 0;
    if (f.chi->active_qpoints().size() != 3)
        throw std::runtime_error("grey-group cRPA did not reduce four q points to three");
    bool has_antiunitary = false;
    for (const auto& star : ctx.kstars)
        for (const auto& member : star.members) has_antiunitary |= member.time_reversal;
    if (!has_antiunitary) throw std::runtime_error("antiunitary cRPA member was not exercised");

    // Two complete Kramers pairs form a closed retained subspace even though
    // the AO-spinor coefficient frame is rectangular (six rows, four states).
    MeanField retained(1, 4, 4, 3, 2);
    for (int k = 0; k < 4; ++k)
    {
        for (int component = 0; component < 2; ++component)
        {
            ComplexMatrix c(4, 3);
            for (int band = 0; band < 4; ++band)
                for (int ao = 0; ao < 3; ++ao)
                    c(band, ao) = (*f.ds.mf.find_wfc(0, component, k))(band, ao);
            retained.get_eigenvectors()[0][component][k] = std::move(c);
        }
        for (int band = 0; band < 4; ++band)
        {
            retained.get_eigenvals()[0](k, band) = f.ds.mf.get_eigenvals()[0](k, band);
            retained.get_weight()[0](k, band) = f.ds.mf.get_weight()[0](k, band);
        }
    }
    validate_symmetry_band_weights(retained, f.ds.basis_wfc, f.ds.pbc, ctx);
    auto reduced_pbc = f.ds.pbc;
    reduced_pbc.klist.resize(3);
    reduced_pbc.kfrac_list.resize(3);
    reduced_pbc.weight_k = {0.25, 0.5, 0.25};
    MeanField retained_ibz(1, 3, 4, 3, 2);
    for (int k = 0; k < 3; ++k)
    {
        for (int component = 0; component < 2; ++component)
            retained_ibz.get_eigenvectors()[0][component][k] =
                *retained.find_wfc(0, component, k);
        for (int band = 0; band < 4; ++band)
        {
            retained_ibz.get_eigenvals()[0](k, band) = retained.get_eigenvals()[0](k, band);
            retained_ibz.get_weight()[0](k, band) =
                retained.get_weight()[0](k, band) * 4 * reduced_pbc.weight_k[k];
        }
    }
    validate_symmetry_band_weights(retained_ibz, f.ds.basis_wfc, reduced_pbc, ctx);
    if (retained_ibz.get_n_kpoints() != 3 || reduced_pbc.kfrac_list.size() != 3)
        throw std::runtime_error("covariance validation changed the live IBZ grid");

    // A deliberately star-dependent KS selector must fail on every MPI rank,
    // without silently replacing the original transition model.
    auto invalid = f.selection();
    invalid[0][3][0] = 1 - invalid[0][3][0];
    int rejected = 0, all_rejected = 0;
    try
    {
        f.chi->set_band_selection(invalid);
    }
    catch (const std::exception&)
    {
        rejected = 1;
    }
    MPI_Allreduce(&rejected, &all_rejected, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_rejected)
        throw std::runtime_error("noncovariant cRPA selector was not collectively rejected");

    for (bool excluded : {false, true})
    {
        if (excluded)
            f.chi->set_band_selection(f.selection());
        else
            f.chi->clear_band_selection();
        f.chi->build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_AUTO, f.ds.cs_data, f.pairs,
                     f.ds.basis_aux, f.no_shrink, f.ds.blacs_h);
        for (double frequency : f.ds.tfg.get_freq_nodes())
            for (const auto& q : f.chi->active_qpoints())
            {
                const int iq =
                    std::distance(f.ds.pbc.klist.begin(),
                                  std::find(f.ds.pbc.klist.begin(), f.ds.pbc.klist.end(), q));
                ComplexMatrix local(2, 2), total(2, 2);
                const auto& response = f.chi->get_chi0_q();
                const auto fi = response.find(frequency);
                if (fi != response.end())
                {
                    const auto qi = fi->second.find(q);
                    if (qi != fi->second.end() && qi->second.count(0) && qi->second.at(0).count(0))
                        local = qi->second.at(0).at(0);
                }
                MPI_Allreduce(local.c, total.c, total.size, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                              MPI_COMM_WORLD);
                const auto expected = f.response(frequency, excluded, iq);
                for (int i = 0; i < total.size; ++i) close(total.c[i], expected.c[i]);
            }
    }
    const auto reduced = f.run();
    double maximum = 0;
    for (std::size_t node = 0; node < full.frequencies.size(); ++node)
        for (const auto& pair : {std::pair{&full.tensors[0].u[node], &reduced.tensors[0].u[node]},
                                 std::pair{&full.tensors[0].w[node], &reduced.tensors[0].w[node]}})
            for (int i = 0; i < pair.first->size; ++i)
            {
                maximum = std::max(maximum, std::abs(pair.first->c[i] - pair.second->c[i]));
                close(pair.first->c[i], pair.second->c[i], 2e-11);
            }
    // Fractional endpoints, not a binary band mask. The exact spectral sum
    // checks Pd before comparing all components of the screened local tensors.
    auto weights = f.selection();
    for (int k = 0; k < 4; ++k) weights[0][k] = {0.82, 0.82, 0.57, 0.57, 0.13, 0.13};
    const auto calculate_weighted = [&](bool symmetry, bool spectral)
    {
        Chi0 chi(f.ds.mf, f.ds.basis_wfc, f.ds.basis_aux, f.ds.pbc, ctx, f.ds.tfg,
                 f.ds.scfk_blacs_ctxt, f.ds.desc_wfc_kb_full, false, symmetry);
        chi.qstar_only_symmetry = true;
        chi.gf_threshold = chi.libri_threshold_C = chi.libri_threshold_G = 0;
        CrpaContext context{chi,         f.ds.cs_data, f.ds.basis_aux, f.pairs,
                            f.no_shrink, f.ds.vq,      f.ds.blacs_h,   f.ds.desc_abf};
        if (spectral)
        {
            const auto pd = build_crpa_excluded_response(context, weights);
            for (const auto frequency : f.ds.tfg.get_freq_nodes())
                for (const auto& q : chi.active_qpoints())
                {
                    ComplexMatrix local(2, 2), total(2, 2);
                    if (pd.count(frequency))
                    {
                        const auto& qs = pd.at(frequency);
                        if (qs.count(q) && qs.at(q).count(0) && qs.at(q).at(0).count(0))
                            local = qs.at(q).at(0).at(0);
                    }
                    MPI_Allreduce(local.c, total.c, total.size, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                                  MPI_COMM_WORLD);
                    const auto iq =
                        std::distance(f.ds.pbc.klist.begin(),
                                      std::find(f.ds.pbc.klist.begin(), f.ds.pbc.klist.end(), q));
                    const auto expected = f.response(frequency, false, iq, &weights);
                    for (int i = 0; i < total.size; ++i) close(total.c[i], expected.c[i]);
                }
        }
        const auto all = compute_crpa_onsite(context, weights, {f.frame.get()});
        const auto p0 = build_crpa_total_response(context);
        const auto shared =
            evaluate_crpa_onsite_from_response(context, weights, {f.frame.get()}, p0);
        compare_u_only(shared, all);
        return all;
    };
    const auto weighted_full = calculate_weighted(false, true);
    const auto weighted_reduced = calculate_weighted(true, true);
    maximum = std::max(maximum, compare_tensors(weighted_reduced, weighted_full));

    // Same-index arrays are insufficient at a TR-invariant point: the
    // diagonal weight operator must also commute with its Kramers sewing.
    auto little_group_bad = weights;
    for (auto& k : little_group_bad[0]) k[0] = 0.21;
    bool little_rejected = false;
    try
    {
        f.chi->set_band_selection(little_group_bad);
    }
    catch (const std::exception&)
    {
        little_rejected = true;
    }
    if (!little_rejected)
        throw std::runtime_error("a target breaking the magnetic little group was accepted");

    // At just one member, permute two complete Kramers pairs and rotate the
    // degenerate rows by a complex unitary. The physical weighted response
    // stays unchanged although the same-index weight arrays now differ.
    const std::array<int, 6> order{2, 3, 0, 1, 4, 5};
    const auto old_mf = f.ds.mf;
    const auto old_weights = weights[0][3];
    for (int band = 0; band < 6; ++band)
    {
        weights[0][3][band] = old_weights[order[band]];
        f.ds.mf.get_eigenvals()[0](3, band) = old_mf.get_eigenvals()[0](3, order[band]);
        f.ds.mf.get_weight()[0](3, band) = old_mf.get_weight()[0](3, order[band]);
        for (int component = 0; component < 2; ++component)
            for (int ao = 0; ao < 3; ++ao)
                f.ds.mf.get_eigenvectors()[0][component][3](band, ao) =
                    old_mf.get_eigenvectors().at(0).at(component).at(3)(order[band], ao);
    }
    for (int component = 0; component < 2; ++component)
        rotate_rows(f.ds.mf.get_eigenvectors()[0][component][3], 0, 1, 0.31, 0.42);
    const auto permuted = calculate_weighted(true, false);
    maximum = std::max(maximum, compare_tensors(permuted, weighted_full));
    f.ds.mf.get_eigenvals()[0](3, 0) += 0.01;
    bool energy_rejected = false;
    try
    {
        f.chi->set_band_selection(weights);
    }
    catch (const std::exception&)
    {
        energy_rejected = true;
    }
    if (!energy_rejected)
        throw std::runtime_error("a live spectrum breaking the magnetic group was accepted");
    f.ds.mf = old_mf;

    if (f.ds.comm_h.is_root())
        std::cout << "SOC magnetic q-star: 4 -> 3 q, antiunitary restore, all tensor max="
                  << maximum << '\n';
}

void test_noncollinear_magnetic_crpa()
{
    MagneticModel model;
    Dataset ds(MPI_COMM_WORLD);
    ds.basis_wfc.set(std::vector<std::size_t>{1, 1, 1});
    ds.basis_wfc.set_l_shells({{0}, {0}, {0}});
    ds.basis_aux.set(std::vector<std::size_t>{1, 1, 1});
    ds.basis_aux.set_l_shells({{0}, {0}, {0}});
    ds.mf.set(1, 9, 6, 3, 2);
    ds.mf.get_efermi() = (model.eigenvalues[2] + model.eigenvalues[3]) * 0.5;
    ds.pbc.set_latvec_and_G(std::vector<double>(model.lattice, model.lattice + 9),
                            std::vector<double>(model.reciprocal, model.reciprocal + 9));
    std::vector<double> kvec;
    for (int x = -1; x <= 1; ++x)
        for (int y = -1; y <= 1; ++y)
        {
            const Vector3_Order<double> cart{Vector3_Order<double>{x / 3.0, y / 3.0, 0} * ds.pbc.G};
            // The public BvK input includes 2pi, whereas internal pbc.G does not.
            kvec.insert(kvec.end(), {TWO_PI * cart.x, TWO_PI * cart.y, TWO_PI * cart.z});
        }
    ds.pbc.set_kgrids_kvec(3, 3, 1, kvec);
    ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD, 9);
    ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(6, 3, 6, 3);
    ds.desc_abf.reset_handler(ds.blacs_h);
    ds.desc_abf.init_1b1p(3, 3, 0, 0);
    ds.tfg.reset(24);
    ds.tfg.generate_minimax(0.4, 4.0);
    std::vector<SiteOrbitalGroup> sites;
    ComplexMatrix trial(6, 6), overlap(6, 6);
    overlap.set_as_identity_matrix();
    for (int a = 0; a < 3; ++a)
    {
        sites.push_back({"magnetic-site-" + std::to_string(a), 2 * a, 2, a, {}});
        trial(a, 2 * a) = trial(a + 3, 2 * a + 1) = 1;
        ds.atoms.types[a] = 0;
        const Vector3_Order<double> cart{model.positions[3 * a], model.positions[3 * a + 1], 0};
        const Vector3_Order<double> frac{cart * ds.pbc.latvec.Inverse()};
        ds.atoms.coords_frac[a] = Vector3<double>{frac.x, frac.y, frac.z};
    }
    CorrelatedSubspace frame(sites, ds.pbc.kfrac_list, ds.pbc.Rlist, 6, 6, 1);
    for (int k = 0; k < 9; ++k)
    {
        for (int spin = 0; spin < 2; ++spin)
        {
            ComplexMatrix c(6, 3);
            for (int band = 0; band < 6; ++band)
                for (int a = 0; a < 3; ++a) c(band, a) = model.eigenvectors(a + 3 * spin, band);
            ds.mf.get_eigenvectors()[0][spin][k] = c;
        }
        for (int band = 0; band < 6; ++band)
        {
            ds.mf.get_eigenvals()[0](k, band) = model.eigenvalues[band];
            ds.mf.get_weight()[0](k, band) = band < 3 ? 1.0 / 9 : 0;
        }
        frame.set_S_k(k, overlap);
        frame.set_W_k(k, trial);
        if (ds.comm_h.is_root())
            for (int a = 0; a < 3; ++a)
                for (int b = a; b < 3; ++b)
                {
                    auto block = std::make_shared<ComplexMatrix>(1, 1);
                    (*block)(0, 0) = a == b ? 1.2 : 0.08;
                    ds.vq[a][b][ds.pbc.klist[k]] = block;
                }
    }
    frame.compute_T_R();
    if (ds.comm_h.is_root())
    {
        ds.cs_data.use_libri = true;
        for (int a = 0; a < 3; ++a)
        {
            auto value = std::make_shared<std::valarray<double>>(0.5, 1);
            ds.cs_data.data_libri[a][{a, {0, 0, 0}}] = RI::Tensor<double>({1, 1, 1}, value);
        }
    }
    const auto assigned =
        dispatch_upper_triangular_tasks(3, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols,
                                        ds.blacs_h.myprow, ds.blacs_h.mypcol);
    std::vector<atpair_t> pairs(assigned.begin(), assigned.end());
    std::map<Vector3_Order<double>, ComplexMatrix> transform;
    Chi0::BandSelection selection(1, std::vector<std::vector<double>>(9, {1, 1, 1, 1, 0, 0}));
    const auto calculate = [&](bool symmetry, bool check_shared = false)
    {
        Chi0 chi(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context, ds.tfg,
                 ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, symmetry);
        chi.qstar_only_symmetry = true;
        chi.gf_threshold = chi.libri_threshold_C = chi.libri_threshold_G = 0;
        CrpaContext context{chi,       ds.cs_data, ds.basis_aux, pairs,
                            transform, ds.vq,      ds.blacs_h,   ds.desc_abf};
        const auto result = compute_crpa_onsite(context, selection, {&frame});
        if (check_shared)
        {
            const auto original_mf = ds.mf;
            auto full_response = build_crpa_total_response(context);
            chi.swap_chi0_q(full_response);
            const auto resident = chi.get_chi0_q();
            auto incoming = selection;
            for (auto& row : incoming[0]) row[0] = 0;
            chi.set_band_selection(incoming);
            if (ds.comm_h.is_root())
                std::cout << "Shared-P0 U-only begin: noncollinear magnetic "
                          << (symmetry ? "q-star" : "full q") << " resident alias" << std::endl;
            const auto shared =
                evaluate_crpa_onsite_from_response(context, selection, {&frame}, chi.get_chi0_q());
            if (ds.comm_h.is_root()) std::cout << "Shared-P0 U-only end\n";
            const double maximum = compare_u_only(shared, result);
            compare_responses(chi.get_chi0_q(), resident);
            if (chi.get_band_selection() != incoming)
                throw std::runtime_error("magnetic shared-response cRPA leaked its selector");
            compare_meanfields(ds.mf, original_mf);
            if (ds.comm_h.is_root())
                std::cout << "Noncollinear magnetic shared-P0 " << (symmetry ? "q-star" : "full q")
                          << ": all v/U max=" << maximum
                          << "; resident P0, selector, E/C/f unchanged\n";
        }
        return std::pair{result, chi.active_qpoints().size()};
    };
    const auto full = calculate(false, true);
    const auto original_mf = ds.mf;
    ComplexMatrix spatial_overlap(3, 3);
    spatial_overlap.set_as_identity_matrix();
    const auto orbital = analyze_orbital_weights(
        ds.basis_wfc, orbital_convention(), {0, 0, 0}, ds.mf,
        std::vector<ComplexMatrix>(9, spatial_overlap), std::vector<double>(9, 1.0 / 9), 1e-11);
    for (int k = 0; k < 9; ++k)
        for (int band = 0; band < 6; ++band)
            for (int atom = 0; atom < 3; ++atom)
            {
                const double charge = std::norm(model.eigenvectors(atom, band)) +
                                      std::norm(model.eigenvectors(atom + 3, band));
                const auto& state = orbital.states[6 * k + band];
                close(state.mulliken_ao[atom], charge, 2e-12);
                close(state.lowdin_ao[atom], charge, 2e-12);
                close(state.mulliken[atom], charge, 2e-12);
                close(state.lowdin[atom], charge, 2e-12);
            }
    compare_meanfields(ds.mf, original_mf);
    const auto analyzed = calculate(false);
    const double analysis_tensor_maximum = compare_tensors(analyzed.first, full.first);
    if (ds.comm_h.is_root())
        std::cout << "Noncollinear magnetic orbital analysis: 3 atom weights; eigenpairs "
                     "unchanged; all v/U/W max="
                  << analysis_tensor_maximum << '\n';
    ds.basis_convention.bloch_phase = -1;
    ds.basis_convention.bloch_ratom = 0;
    ds.basis_convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    ds.basis_convention.coeff_m_negative = ds.basis_convention.coeff_m_positive =
        LIBRPA_RSH_COEFF_1_M;
    for (int i = 0; i < 6; ++i)
    {
        std::array<double, 9> rotation;
        for (int j = 0; j < 9; ++j) rotation[j] = model.operations[9 * i + j];
        ds.spg_symops.push_back({rotation, {0, 0, 0}, true});
        SymmetrySpinOperation op;
        op.spatial_id = i;
        op.antiunitary = model.antiunitary[i];
        op.spin_source = SymmetrySpinActionSource::DerivedFromSpatialSOC;
        ds.spg_spin_ops.push_back(op);
    }
    ds.spg_spin_ops_explicit = true;
    ds.spg_grey_group = false;
    initialize_symmetry_context(ds, true);
    const auto reduced = calculate(true, true);
    if (ds.symmetry_context.spin_operations.size() != 6 || reduced.second >= full.second)
        throw std::runtime_error("explicit magnetic table was widened or did not reduce q points");
    double maximum = 0;
    for (int site = 0; site < full.first.tensors.size(); ++site)
        for (int node = 0; node < full.first.frequencies.size(); ++node)
            for (const auto pair : {std::pair{&full.first.tensors[site].u[node],
                                              &reduced.first.tensors[site].u[node]},
                                    std::pair{&full.first.tensors[site].w[node],
                                              &reduced.first.tensors[site].w[node]}})
                for (int i = 0; i < pair.first->size; ++i)
                {
                    maximum = std::max(maximum, std::abs(pair.first->c[i] - pair.second->c[i]));
                    close(pair.first->c[i], pair.second->c[i], 2e-11);
                }
    if (ds.comm_h.is_root())
        std::cout << "Noncollinear magnetic cRPA: " << full.second << " -> " << reduced.second
                  << " q; 3 unitary + 3 antiunitary; complete U/W max=" << maximum << '\n';
}

void test_soc_and_spin_representation()
{
    Fixture original(true, 0.37), rotated(true, 0.37, 0.61);
    original.compare_response(false);
    original.compare_response(true);
    rotated.compare_response(false);
    rotated.compare_response(true);
    const auto first = original.run(), second = rotated.run();
    original.compare_tensors(first);
    rotated.compare_tensors(second);
    const auto staged = original.run_staged();
    const auto shared = original.run_shared(false), aliased = original.run_shared(true);
    original.compare_tensors(staged);
    original.compare_tensors(shared);
    original.compare_tensors(aliased);
    const double staged_maximum = compare_tensors(staged, first);
    const double shared_maximum =
        std::max(compare_u_only(shared, first), compare_u_only(aliased, first));
    if (original.ds.comm_h.is_root())
        std::cout << "Complex SOC cRPA composition: staged v/U/W max=" << staged_maximum
                  << "; shared-P0 v/U max=" << shared_maximum
                  << "; borrowed and resident response, selector, E/C/f unchanged\n";
    for (std::size_t iw = 0; iw < first.frequencies.size(); ++iw)
        for (int i = 0; i < first.tensors[0].bare.size; ++i)
        {
            close(first.tensors[0].bare.c[i], second.tensors[0].bare.c[i], 1e-12);
            close(first.tensors[0].u[iw].c[i], second.tensors[0].u[iw].c[i], 1e-11);
            close(first.tensors[0].w[iw].c[i], second.tensors[0].w[iw].c[i], 1e-11);
        }

    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    ComplexMatrix spatial(3, 3);
    spatial.set_as_identity_matrix();
    const auto original_mf = original.ds.mf;
    const auto orbital = analyze_orbital_weights(
        original.ds.basis_wfc, convention, {0}, original.ds.mf,
        std::vector<ComplexMatrix>(4, spatial), std::vector<double>(4, 0.25), 1e-11);
    for (int k = 0; k < 4; ++k)
        for (int band = 0; band < 6; ++band)
            for (int ao = 0; ao < 3; ++ao)
            {
                const double charge = std::norm(original.coefficients(ao, band)) +
                                      std::norm(original.coefficients(ao + 3, band));
                const auto& state = orbital.states[6 * k + band];
                close(state.mulliken_ao[ao], charge, 2e-12);
                close(state.lowdin_ao[ao], charge, 2e-12);
            }
    compare_meanfields(original.ds.mf, original_mf);
    const auto analyzed = original.run();
    const double analysis_tensor_maximum = compare_tensors(analyzed, first);
    if (original.ds.comm_h.is_root())
        std::cout << "SOC cRPA orbital analysis: eigenpairs unchanged; all v/U/W max="
                  << analysis_tensor_maximum << '\n';
    CrpaWindowOptions options;
    options.correlated_species = "X";
    options.parent_orbitals = options.output_orbitals = "s";
    options.response_bands = {0, 1, 2, 3};
    options.orbital_bands = {0, 1, 4, 5};
    const auto built = build_crpa_window_input(original.ds.basis_wfc, convention, {0}, {"X"},
                                               std::vector<ComplexMatrix>(4, spatial),
                                               original.ds.pbc.kfrac_list, original.ds.mf, options);
    if (built.sites[0].n_orbitals != 2 ||
        built.sites[0].orbital_labels != std::vector<std::string>{"s:up", "s:down"})
        throw std::runtime_error("SOC output labels/count are inconsistent");
    // Analytic spectral projection gives precisely the first Kramers pair.
    for (const auto& phi : built.orbitals_spin_k.at(0))
        for (int i = 0; i < 6; ++i)
            for (int a = 0; a < 2; ++a) close(phi(i, a), original.coefficients(i, a), 1e-12);

    // An independently specified nonorthogonal AO change: C=R*C0,
    // S=R^(-dagger)R^(-1). The complete B projects back to the original trials.
    ComplexMatrix r(3, 3), inverse(3, 3);
    r.set_as_identity_matrix();
    inverse.set_as_identity_matrix();
    r(0, 1) = 0.2;
    r(1, 2) = 0.3;
    inverse(0, 1) = -0.2;
    inverse(1, 2) = -0.3;
    inverse(0, 2) = 0.06;
    const auto metric = transpose(inverse, true) * inverse;
    auto nonorthogonal = original.ds.mf;
    for (int k = 0; k < 4; ++k)
        for (int s = 0; s < 2; ++s)
            nonorthogonal.get_eigenvectors()[0][s][k] =
                nonorthogonal.get_eigenvectors()[0][s][k] * transpose(r, false);
    options.orbital_bands = {0, 1, 2, 3, 4, 5};
    const auto changed = build_crpa_window_input(
        original.ds.basis_wfc, convention, {0}, {"X"}, std::vector<ComplexMatrix>(4, metric),
        original.ds.pbc.kfrac_list, nonorthogonal, options);
    for (const auto& phi : changed.orbitals_spin_k.at(0))
        for (int i = 0; i < 6; ++i)
            for (int a = 0; a < 2; ++a) close(phi(i, a), i == 3 * a ? 1.0 : 0.0, 1e-12);
    // A public SOC setter with a supplied SU(2) matrix must preserve that
    // matrix through initialization. The null case instead derives identity
    // from the spatial identity. Compare the resulting density rotation to an
    // independent dense calculation, including the grey-group Theta partner.
    std::unique_ptr<LibrpaHandler, decltype(&librpa_destroy_handler)> public_handler(
        librpa_create_handler(MPI_COMM_WORLD), &librpa_destroy_handler);
    auto* handle = public_handler.get();
    const int rotation[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double lattice[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    const double reciprocal[] = {TWO_PI, 0, 0, 0, TWO_PI, 0, 0, 0, TWO_PI};
    const double origin[] = {0, 0, 0};
    const int species = 0;
    librpa_set_scf_dimension(handle, 1, 1, 2, 1, 2);
    librpa_set_latvec_and_G(handle, lattice, reciprocal);
    librpa_set_atoms(handle, 1, &species, origin);
    librpa_set_kgrids_kvec(handle, 1, 1, 1, 1, origin, nullptr);
    const double c = std::cos(0.43), s = std::sin(0.43);
    const double supplied_u[] = {c, 0, 0, s, 0, s, c, 0};
    ComplexMatrix density(2, 2), supplied(2, 2), theta(2, 2);
    density(0, 0) = 0.7;
    density(1, 1) = 0.3;
    density(0, 1) = Z(0.13, 0.24);
    density(1, 0) = std::conj(density(0, 1));
    supplied(0, 0) = supplied(1, 1) = c;
    supplied(0, 1) = supplied(1, 0) = Z(0, s);
    theta(0, 1) = 1;
    theta(1, 0) = -1;
    SpinorBlocks4<ComplexMatrix> blocks{ComplexMatrix(1, 1), ComplexMatrix(1, 1),
                                        ComplexMatrix(1, 1), ComplexMatrix(1, 1)};
    blocks.b00(0, 0) = density(0, 0);
    blocks.b01(0, 0) = density(0, 1);
    blocks.b10(0, 0) = density(1, 0);
    blocks.b11(0, 0) = density(1, 1);
    for (bool provided : {false, true})
    {
        librpa_set_symmetry_spin_operations(handle, 1, 1, rotation, nullptr, nullptr,
                                            provided ? supplied_u : nullptr, 2, 1);
        const auto public_ds = api::get_dataset_instance(handle);
        initialize_symmetry_context(*public_ds, false);
        for (const auto& operation : public_ds->symmetry_context.spin_operations)
        {
            auto expected = provided ? supplied * density * transpose(supplied, true) : density;
            if (operation.antiunitary) expected = theta * conj(expected) * transpose(theta, true);
            const auto transformed = transform_spinor_bilinear(
                operation, blocks, [](std::size_t, const ComplexMatrix& value) { return value; },
                BilinearConvention::SourceToTarget_DXDdag);
            close(transformed.b00(0, 0), expected(0, 0), 1e-12);
            close(transformed.b01(0, 0), expected(0, 1), 1e-12);
            close(transformed.b10(0, 0), expected(1, 0), 1e-12);
            close(transformed.b11(0, 0), expected(1, 1), 1e-12);
        }
    }
}

void test_soc_bloch_phases()
{
    Fixture periodic(true, 0.37, 0, true);
    periodic.compare_response(false);
    periodic.compare_response(true);
    periodic.compare_tensors(periodic.run());
}

void test_constrained_soc_optical()
{
    Fixture f(true, 0.37);
    const double pi = std::acos(-1.0);
    initialize_velocity_matrix(f.ds.velocity_matrix, 1, 4, f.bands);
    // Hermitian imaginary AO velocity operators, odd under time reversal.
    for (int k = 0; k < 4; ++k)
        for (int alpha = 0; alpha < 3; ++alpha)
            for (int n = 0; n < f.bands; ++n)
                for (int m = 0; m < f.bands; ++m)
                    for (int s = 0; s < 2; ++s)
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j)
                                f.ds.velocity_matrix[0][k][alpha](n, m) +=
                                    std::conj(f.coefficients(3 * s + i, n)) *
                                    Z(0, 0.025 * (alpha + 1) * (j - i)) *
                                    f.coefficients(3 * s + j, m);
    const auto& frequencies = f.ds.tfg.get_freq_nodes();
    auto fractional = f.selection();
    for (auto& row : fractional[0]) row = {.27, .68, .45, .82, .11, .62};
    f.compare_response(false, &fractional);
    const auto make_optical = [&](const MeanField& mf, int constrained)
    {
        auto optical = std::make_unique<diele_func>(mf, f.ds.velocity_matrix, f.ds.pbc.kfrac_list,
                                                    f.ds.basis_wfc, f.ds.basis_aux, frequencies, 3,
                                                    6, 1, 2, f.ds.pbc, f.ds.comm_h, f.ds.blacs_h);
        optical->use_soc = true;
        if (constrained)
            optical->set_excluded_band_pairs(constrained == 1 ? f.selection() : fractional);
        optical->normalize_gamma_cell_weights = true;
        optical->init(0, f.ds.vq);
        optical->cal_head();
        optical->cal_wing(f.ds.cs_data, 0, f.ds.vq);
        return optical;
    };
    const auto full = make_optical(f.ds.mf, false);
    const auto retained = make_optical(f.ds.mf, true);
    const auto weighted = make_optical(f.ds.mf, 2);
    double excluded_signal = 0;
    for (std::size_t iw = 0; iw < frequencies.size(); ++iw)
        for (int which = 0; which < 3; ++which)
        {
            ComplexMatrix head(3, 3), wing(2, 3);
            head.set_as_identity_matrix();
            for (int k = 0; k < 4; ++k)
                for (int n = 0; n < 2; ++n)
                    for (int m = 2; m < 6; ++m)
                    {
                        if (which == 1 && m < 4) continue;
                        const double remainder =
                            which == 2 ? 1 - fractional[0][k][n] * fractional[0][k][m] : 1;
                        const double delta =
                            f.ds.mf.get_eigenvals()[0](k, m) - f.ds.mf.get_eigenvals()[0](k, n);
                        const double denominator =
                            delta * delta + frequencies[iw] * frequencies[iw];
                        for (int alpha = 0; alpha < 3; ++alpha)
                        {
                            const auto& va = f.ds.velocity_matrix[0][k][alpha];
                            for (int beta = 0; beta < 3; ++beta)
                                head(alpha, beta) += 4 * pi * 0.5 * remainder * va(m, n) *
                                                     f.ds.velocity_matrix[0][k][beta](n, m) /
                                                     (delta * denominator);
                            for (int mu = 0; mu < 2; ++mu)
                                wing(mu, alpha) -= 2 * std::sqrt(4 * pi) * 0.25 * remainder *
                                                   std::conj(f.vertex(mu, n, m) * va(m, n)) /
                                                   denominator;
                        }
                    }
            const auto& actual = which == 2 ? weighted : (which ? retained : full);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    close(actual->get_head_tensors().at(iw)(i, j), head(i, j), 2e-12);
                    excluded_signal = std::max(excluded_signal,
                                               std::abs(full->get_head_tensors().at(iw)(i, j) -
                                                        retained->get_head_tensors().at(iw)(i, j)));
                }
            for (int mu = 0; mu < 2; ++mu)
                for (int alpha = 0; alpha < 3; ++alpha)
                    close(actual->get_wing_auxiliary().at(iw)(mu, alpha), wing(mu, alpha), 2e-12);
        }
    if (excluded_signal < 1e-4) throw std::runtime_error("optical Pd control was trivial");

    // A zero-response vacuum must give the retained identity after cell averaging.
    auto vacuum_mf = f.ds.mf;
    vacuum_mf.get_weight()[0].zero_out();
    const auto vacuum = make_optical(vacuum_mf, false);
    auto identity = init_local_mat<Z>(f.ds.desc_abf, MAJOR::COL);
    for (int i = 0; i < identity.nr(); ++i)
        for (int j = 0; j < identity.nc(); ++j)
            identity(i, j) = f.ds.desc_abf.indx_l2g_r(i) == f.ds.desc_abf.indx_l2g_c(j) ? 1.0 : 0.0;
    auto inverse = identity.copy();
    ArrayDesc head_desc(f.ds.blacs_h);
    head_desc.init_1b1p(2, 1, 0, 0);
    auto head_direction = init_local_mat<Z>(head_desc, MAJOR::COL);
    if (head_desc.indx_g2l_r(0) >= 0 && head_desc.indx_g2l_c(0) >= 0)
        head_direction(head_desc.indx_g2l_r(0), head_desc.indx_g2l_c(0)) = 1;
    vacuum->rewrite_eps_abf_space(inverse, 0, identity, identity, head_direction, head_desc,
                                  f.ds.desc_abf, 2, 0, false, false);
    for (int i = 0; i < inverse.nr(); ++i)
        for (int j = 0; j < inverse.nc(); ++j) close(inverse(i, j), identity(i, j), 2e-12);
}
}  // namespace

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    try
    {
        test_spinor_character_nonorthogonal_metric_and_pdos();
        test_spinor_character_scalar_packing_limit();
        test_zero_soc_limit();
        test_soc_and_spin_representation();
        test_soc_bloch_phases();
        test_magnetic_qstar_screening();
        test_noncollinear_magnetic_crpa();
        test_constrained_soc_optical();
    }
    catch (const std::exception& failure)
    {
        std::cerr << failure.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    std::cout << "test_crpa_soc: passed\n";
}
