#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

#include "../core/analycont.h"
#include "../core/qsgw/correlation_potential.h"

using librpa_int::AnalyContPade;
using librpa_int::cplxdb;
using librpa_int::Matz;
using librpa_int::MeanField;
using librpa_int::qsgw::build_qsgw_correlation_potential;
using librpa_int::qsgw::CorrelationPotentialMode;
using librpa_int::qsgw::CorrelationPotentialSettings;

namespace
{

void assert_close(const cplxdb actual, const cplxdb expected, const double tolerance = 1.0e-13)
{
    assert(std::abs(actual - expected) < tolerance);
}

std::vector<cplxdb> make_imagfreqs(const std::vector<double>& frequencies)
{
    std::vector<cplxdb> result;
    for (const double frequency : frequencies)
    {
        result.emplace_back(0.0, frequency);
    }
    return result;
}

std::map<double, Matz> make_sigma(const std::vector<double>& frequencies)
{
    Matz constant(2, 2);
    constant(0, 0) = 2.123456789e-7;
    constant(0, 1) = cplxdb(1.234567891e-7, 0.456789123e-7);
    constant(1, 0) = std::conj(constant(0, 1));
    constant(1, 1) = 4.234567891e-7;

    Matz slope(2, 2);
    slope(0, 0) = 0.712345678e-7;
    slope(0, 1) = cplxdb(-0.312345678e-7, 0.212345678e-7);
    slope(1, 0) = std::conj(slope(0, 1));
    slope(1, 1) = -0.423456789e-7;

    std::map<double, Matz> sigma;
    for (const double frequency : frequencies)
    {
        sigma[frequency] = constant + cplxdb(0.0, frequency) * slope;
    }
    return sigma;
}

cplxdb evaluate_element_with_upstream_pade(const std::vector<double>& source_frequencies,
                                           const std::map<double, Matz>& sigma, const int row,
                                           const int column, const cplxdb target,
                                           const std::vector<cplxdb>& resample_frequencies,
                                           const int n_params_anacon,
                                           const int n_params_anacon_resample)
{
    const auto source_imagfreqs = make_imagfreqs(source_frequencies);
    std::vector<cplxdb> source_data;
    for (const double frequency : source_frequencies)
    {
        source_data.push_back(sigma.at(frequency)(row, column));
    }

    if (resample_frequencies.empty())
    {
        return AnalyContPade(n_params_anacon, source_imagfreqs, source_data).get(target);
    }

    const AnalyContPade source_pade(n_params_anacon_resample, source_imagfreqs, source_data);
    std::vector<cplxdb> resampled_data;
    for (const cplxdb frequency : resample_frequencies)
    {
        resampled_data.push_back(source_pade.get(frequency));
    }
    return AnalyContPade(n_params_anacon, resample_frequencies, resampled_data).get(target);
}

Matz evaluate_matrix(const std::vector<double>& source_frequencies,
                     const std::map<double, Matz>& sigma, const double energy,
                     const std::vector<cplxdb>& resample_frequencies, const int n_params_anacon,
                     const int n_params_anacon_resample)
{
    Matz result(2, 2);
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            result(row, column) = evaluate_element_with_upstream_pade(
                source_frequencies, sigma, row, column, cplxdb(energy, 0.0), resample_frequencies,
                n_params_anacon, n_params_anacon_resample);
        }
    }
    return 0.5 * (result + transpose(result, true));
}

Matz expected_potential(const MeanField& meanfield, const std::vector<double>& source_frequencies,
                        const std::map<double, Matz>& sigma, const CorrelationPotentialMode mode,
                        const std::vector<cplxdb>& resample_frequencies, const int n_params_anacon,
                        const int n_params_anacon_resample)
{
    const double efermi = meanfield.get_efermi();
    const double e0 = meanfield.get_eigenvals()[0](0, 0) - efermi;
    const double e1 = meanfield.get_eigenvals()[0](0, 1) - efermi;
    const Matz sigma0 = evaluate_matrix(source_frequencies, sigma, e0, resample_frequencies,
                                        n_params_anacon, n_params_anacon_resample);
    const Matz sigma1 = evaluate_matrix(source_frequencies, sigma, e1, resample_frequencies,
                                        n_params_anacon, n_params_anacon_resample);
    const Matz sigma_fermi = evaluate_matrix(source_frequencies, sigma, 0.0, resample_frequencies,
                                             n_params_anacon, n_params_anacon_resample);

    Matz expected(2, 2);
    if (mode == CorrelationPotentialMode::ModeB)
    {
        expected(0, 0) = sigma0(0, 0);
        expected(1, 1) = sigma1(1, 1);
        expected(0, 1) = sigma_fermi(0, 1);
        expected(1, 0) = sigma_fermi(1, 0);
        return expected;
    }

    expected(0, 0) = sigma0(0, 0);
    expected(1, 1) = sigma1(1, 1);
    expected(0, 1) = 0.5 * (sigma0(0, 1) + sigma1(0, 1));
    expected(1, 0) = 0.5 * (sigma1(1, 0) + sigma0(1, 0));
    return expected;
}

void run_mode_test(const CorrelationPotentialMode mode,
                   const std::vector<cplxdb>& resample_frequencies, const int n_params_anacon = -1,
                   const int n_params_anacon_resample = -1)
{
    MeanField meanfield(1, 1, 2, 2, 1);
    meanfield.get_eigenvals()[0](0, 0) = -0.4;
    meanfield.get_eigenvals()[0](0, 1) = 0.6;
    meanfield.get_efermi() = 0.1;
    const std::vector<double> source_frequencies{0.2, 1.0};
    const auto sigma = make_sigma(source_frequencies);

    CorrelationPotentialSettings settings;
    settings.mode = mode;
    settings.n_params_anacon = n_params_anacon;
    settings.n_params_anacon_resample = n_params_anacon_resample;
    settings.resample_imagfreqs = resample_frequencies;
    const Matz actual =
        build_qsgw_correlation_potential(meanfield, source_frequencies, sigma, 0, 0, settings);
    const Matz expected =
        expected_potential(meanfield, source_frequencies, sigma, mode, resample_frequencies,
                           n_params_anacon, n_params_anacon_resample);

    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            assert_close(actual(row, column), expected(row, column));
        }
    }
    assert(std::abs(actual(0, 0)) > 1.0e-8);
}

void test_mode_a_and_b_match_upstream_pade_without_qsgw_rounding()
{
    run_mode_test(CorrelationPotentialMode::ModeA, {});
    run_mode_test(CorrelationPotentialMode::ModeB, {});
}

void test_optional_resampling_matches_the_upstream_two_stage_pade()
{
    run_mode_test(CorrelationPotentialMode::ModeB, {cplxdb(0.0, 0.3), cplxdb(0.0, 0.8)});
}

void test_two_stage_parameter_counts_follow_upstream_order()
{
    const std::vector<cplxdb> resample_frequencies{cplxdb(0.0, 0.3), cplxdb(0.0, 0.65),
                                                   cplxdb(0.0, 0.9)};

    MeanField meanfield(1, 1, 2, 2, 1);
    meanfield.get_eigenvals()[0](0, 0) = -0.4;
    meanfield.get_eigenvals()[0](0, 1) = 0.6;
    meanfield.get_efermi() = 0.1;
    const std::vector<double> source_frequencies{0.2, 0.6, 1.0};
    auto sigma = make_sigma(source_frequencies);
    sigma.at(0.6)(0, 0) += 1.1e-7;
    sigma.at(0.6)(0, 1) += cplxdb(0.4e-7, -0.2e-7);
    sigma.at(0.6)(1, 0) += cplxdb(-0.3e-7, 0.1e-7);
    sigma.at(0.6)(1, 1) -= 0.7e-7;

    CorrelationPotentialSettings settings;
    settings.mode = CorrelationPotentialMode::ModeB;
    settings.n_params_anacon = 2;
    settings.n_params_anacon_resample = 3;
    settings.resample_imagfreqs = resample_frequencies;
    const Matz actual =
        build_qsgw_correlation_potential(meanfield, source_frequencies, sigma, 0, 0, settings);
    const Matz correct =
        expected_potential(meanfield, source_frequencies, sigma, CorrelationPotentialMode::ModeB,
                           resample_frequencies, 2, 3);
    const Matz swapped =
        expected_potential(meanfield, source_frequencies, sigma, CorrelationPotentialMode::ModeB,
                           resample_frequencies, 3, 2);
    for (int row = 0; row < 2; ++row)
    {
        for (int column = 0; column < 2; ++column)
        {
            assert_close(actual(row, column), correct(row, column));
        }
    }
    assert(std::abs(correct(0, 0) - swapped(0, 0)) > 1.0e-12);
}

Matz analytic_sigma(const int dimension, const int spin, const int kpoint, const cplxdb energy)
{
    Matz result(dimension, dimension);
    for (int row = 0; row < dimension; ++row)
    {
        for (int column = 0; column < dimension; ++column)
        {
            const cplxdb residue(0.3 + 0.17 * row + 0.09 * column + 0.05 * spin + 0.03 * kpoint,
                                 0.12 + 0.04 * row - 0.08 * column + 0.03 * spin - 0.02 * kpoint);
            const cplxdb pole(-2.1 - 0.19 * row + 0.14 * column + 0.07 * spin - 0.04 * kpoint,
                              -0.5 - 0.08 * row - 0.11 * column);
            result(row, column) = residue / (energy - pole);
        }
    }
    return result;
}

Matz analytic_hermitian_sigma(const int dimension, const int spin, const int kpoint,
                              const double energy)
{
    const Matz sigma = analytic_sigma(dimension, spin, kpoint, energy);
    return 0.5 * (sigma + transpose(sigma, true));
}

Matz analytic_static_potential(const MeanField& meanfield, const int spin, const int kpoint,
                               const CorrelationPotentialMode mode)
{
    const int dimension = meanfield.get_n_bands();
    const Matz sigma_fermi = analytic_hermitian_sigma(dimension, spin, kpoint, 0.0);
    Matz expected(dimension, dimension);
    if (mode == CorrelationPotentialMode::ModeB)
    {
        expected = sigma_fermi.copy();
    }

    // Spectral-projector form of the staticization formula, evaluated from the
    // known rational function rather than another analytic-continuation fit.
    // A: sum_i (P_i H(e_i) + H(e_i) P_i) / 2;
    // B: H(0) + sum_i P_i (H(e_i) - H(0)) P_i, where e_i = epsilon_i - E_F.
    for (int band = 0; band < dimension; ++band)
    {
        Matz projector(dimension, dimension);
        projector(band, band) = 1.0;
        const double energy =
            meanfield.get_eigenvals()[spin](kpoint, band) - meanfield.get_efermi();
        const Matz sigma_state = analytic_hermitian_sigma(dimension, spin, kpoint, energy);
        if (mode == CorrelationPotentialMode::ModeA)
        {
            expected = expected + 0.5 * (projector * sigma_state + sigma_state * projector);
        }
        else
        {
            expected = expected + projector * (sigma_state - sigma_fermi) * projector;
        }
    }
    return expected;
}

void run_complex_rational_staticization_test(MeanField& meanfield)
{
    meanfield.get_efermi() = 0.37;
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < meanfield.get_n_kpoints(); ++kpoint)
        {
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
            {
                meanfield.get_eigenvals()[spin](kpoint, band) =
                    -0.8 + 0.63 * band + 0.17 * spin + 0.11 * kpoint;
            }
        }
    }
    const std::vector<double> source_frequencies{0.15, 0.4, 0.9, 1.7, 3.2};
    const std::vector<std::vector<cplxdb>> resample_grids{
        {}, {cplxdb(0.0, 0.25), cplxdb(0.0, 0.65), cplxdb(0.0, 1.2), cplxdb(0.0, 2.4)}};
    const int dimension = meanfield.get_n_bands();

    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < meanfield.get_n_kpoints(); ++kpoint)
        {
            std::map<double, Matz> sigma;
            for (const double frequency : source_frequencies)
            {
                sigma.emplace(frequency,
                              analytic_sigma(dimension, spin, kpoint, cplxdb(0.0, frequency)));
            }
            const Matz sigma_zero = analytic_sigma(dimension, spin, kpoint, 0.0);
            assert(std::abs(sigma_zero(0, 1) - std::conj(sigma_zero(1, 0))) > 1.0e-2);
            assert(std::abs(sigma.begin()->second(0, 1) - std::conj(sigma.begin()->second(1, 0))) >
                   1.0e-2);

            for (const auto mode :
                 {CorrelationPotentialMode::ModeA, CorrelationPotentialMode::ModeB})
            {
                const Matz expected = analytic_static_potential(meanfield, spin, kpoint, mode);
                assert(std::abs(expected(0, 1).imag()) > 1.0e-3);
                for (const auto& resample_frequencies : resample_grids)
                {
                    for (const int final_count : {-1, 2})
                    {
                        for (const int source_count : {-1, 2})
                        {
                            if (resample_frequencies.empty() && source_count != -1)
                            {
                                continue;
                            }
                            CorrelationPotentialSettings settings;
                            settings.mode = mode;
                            settings.n_params_anacon = final_count;
                            settings.n_params_anacon_resample = source_count;
                            settings.resample_imagfreqs = resample_frequencies;
                            const Matz actual = build_qsgw_correlation_potential(
                                meanfield, source_frequencies, sigma, spin, kpoint, settings);
                            for (int row = 0; row < dimension; ++row)
                            {
                                for (int column = 0; column < dimension; ++column)
                                {
                                    assert_close(actual(row, column), expected(row, column),
                                                 2.0e-12);
                                    assert_close(actual(row, column),
                                                 std::conj(actual(column, row)));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void test_complex_spinor_staticization_against_analytic_rational_reference()
{
    MeanField meanfield(1, 2, 4, 2, 2);
    run_complex_rational_staticization_test(meanfield);
}

void test_spin_and_kpoint_selection_against_analytic_rational_reference()
{
    MeanField meanfield(2, 3, 3, 3, 1);
    run_complex_rational_staticization_test(meanfield);
}

}  // namespace

void test_ridge_selection_with_noisy_samples()
{
    MeanField mf(1, 1, 2, 2);
    mf.get_efermi() = 0.23;
    mf.get_eigenvals()[0](0, 0) = -0.6;
    mf.get_eigenvals()[0](0, 1) = 0.7;
    const std::vector<double> freqs{0.2, 0.5, 1, 2, 4, 8};
    const double noise[] = {0.003, -0.002, 0.001, -0.004, 0.002, -0.001};
    const cplxdb residues[2][2] = {{{0.4, 0}, {0.05, 0.03}}, {{0.05, -0.03}, {0.3, 0}}};
    std::map<double, Matz> sigma;
    for (int f = 0; f != freqs.size(); ++f)
    {
        Matz matrix(2, 2, librpa_int::MAJOR::ROW);
        const auto value = 1.0 / cplxdb{2, freqs[f]} + noise[f] * cplxdb{1, 0.3};
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j) matrix(i, j) = residues[i][j] * value;
        sigma.emplace(freqs[f], matrix);
    }
    // Independent NumPy solve of the documented penalized coefficient objective.
    // Non-rational samples distinguish ridge from interpolation. Unequal source
    // and final counts distinguish the two resampling stages.
    const double references[2][3] = {{0.8642652159877527, 0.4059229961160846, 0.5024598494842251},
                                     {0.8795711963916500, 0.4048744621627424, 0.5034232973694422}};
    for (int resample = 0; resample != 2; ++resample)
    {
        CorrelationPotentialSettings settings;
        settings.anacon_method = LIBRPA_ANACON_RIDGE;
        settings.n_params_anacon = 3;
        settings.n_params_anacon_resample = 4;
        if (resample) settings.resample_imagfreqs = {{0, .3}, {0, .8}, {0, 1.5}, {0, 3}, {0, 6}};
        const auto potential = build_qsgw_correlation_potential(mf, freqs, sigma, 0, 0, settings);
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j)
                assert(std::abs(potential(i, j) -
                                residues[i][j] * references[resample][i == j ? i : 2]) < 1e-10);
    }
}

void test_selected_methods_against_rational_matrix()
{
    MeanField mf(1, 1, 2, 2);
    mf.get_efermi() = 0.23;
    mf.get_eigenvals()[0](0, 0) = -0.6;
    mf.get_eigenvals()[0](0, 1) = 0.7;
    const std::vector<double> freqs{0.2, 0.5, 1.0, 2.0, 4.0, 8.0};
    const cplxdb residues[2][2] = {{{0.4, 0}, {0.05, 0.03}}, {{0.05, -0.03}, {0.3, 0}}};
    std::map<double, Matz> sigma;
    for (const double freq : freqs)
    {
        Matz matrix(2, 2, librpa_int::MAJOR::ROW);
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j) matrix(i, j) = residues[i][j] / cplxdb{2.0, freq};
        sigma.emplace(freq, matrix);
    }
    // Compare the opt-in methods with the analytic input function.
    for (const auto method : {LIBRPA_ANACON_AAA, LIBRPA_ANACON_RIDGE})
        for (const auto mode : {CorrelationPotentialMode::ModeA, CorrelationPotentialMode::ModeB})
            for (const bool resample : {false, true})
            {
                CorrelationPotentialSettings settings;
                settings.anacon_method = method;
                settings.mode = mode;
                settings.n_params_anacon = settings.n_params_anacon_resample = 3;
                if (resample)
                    settings.resample_imagfreqs = {{0, 0.3}, {0, 0.8}, {0, 1.5}, {0, 3.0}};
                const auto potential =
                    build_qsgw_correlation_potential(mf, freqs, sigma, 0, 0, settings);
                for (int i = 0; i != 2; ++i)
                    for (int j = 0; j != 2; ++j)
                    {
                        const double ei = mf.get_eigenvals()[0](0, i) - mf.get_efermi();
                        const double ej = mf.get_eigenvals()[0](0, j) - mf.get_efermi();
                        const auto expected =
                            residues[i][j] * (i == j ? 1.0 / (2.0 + ei)
                                              : mode == CorrelationPotentialMode::ModeB
                                                  ? 0.5
                                                  : 0.5 * (1.0 / (2.0 + ei) + 1.0 / (2.0 + ej)));
                        assert(std::abs(potential(i, j) - expected) < 2.0e-7);
                        assert(std::abs(potential(i, j) - std::conj(potential(j, i))) < 1.0e-14);
                    }
            }
}

int main()
{
    test_selected_methods_against_rational_matrix();
    test_ridge_selection_with_noisy_samples();
    test_mode_a_and_b_match_upstream_pade_without_qsgw_rounding();
    test_optional_resampling_matches_the_upstream_two_stage_pade();
    test_two_stage_parameter_counts_follow_upstream_order();
    test_complex_spinor_staticization_against_analytic_rational_reference();
    test_spin_and_kpoint_selection_against_analytic_rational_reference();
    return 0;
}
