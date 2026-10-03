#include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "../core/dielecmodel.h"
#include "../core/sternheimer_rpa.h"
#include "../utils/constants.h"

namespace
{

void require_close(const std::complex<double> &actual, const std::complex<double> &expected,
                   const double tolerance)
{
    if (std::abs(actual - expected) >= tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected
                  << " diff=" << std::abs(actual - expected) << std::endl;
        std::abort();
    }
}

void test_sternheimer_pi_and_trace_log_match_diagonal_reference()
{
    librpa_int::ComplexMatrix coulomb(2, 2);
    coulomb(0, 0) = {4.0, 0.0};
    coulomb(1, 1) = {9.0, 0.0};

    librpa_int::ComplexMatrix response_m(2, 2);
    response_m(0, 0) = {-0.8, 0.0};
    response_m(1, 1) = {-0.9, 0.0};

    const auto pi = librpa_int::compute_sternheimer_pi_from_m(coulomb, response_m, 1e-12);
    require_close(pi(0, 0), {-0.2, 0.0}, 1e-12);
    require_close(pi(1, 1), {-0.1, 0.0}, 1e-12);
    require_close(pi(0, 1), {0.0, 0.0}, 1e-12);
    require_close(pi(1, 0), {0.0, 0.0}, 1e-12);

    const std::complex<double> expected_integrand = std::log(std::complex<double>(1.2, 0.0)) - 0.2 +
                                                    std::log(std::complex<double>(1.1, 0.0)) - 0.1;
    require_close(librpa_int::compute_rpa_trace_log_integrand(pi), expected_integrand, 1e-12);

    const auto result = librpa_int::compute_sternheimer_rpa_frequency(coulomb, response_m, 3, 0.5,
                                                                      0.25, 1.0, 1e-12);
    require_close(result.integrand, expected_integrand, 1e-12);
    require_close(result.energy, expected_integrand * 0.25 / librpa_int::TWO_PI, 1e-12);
}

void test_sternheimer_headwing_frequency_uses_one_based_response_labels()
{
    assert(librpa_int::sternheimer_headwing_frequency_index(1, 12) == 0);
    assert(librpa_int::sternheimer_headwing_frequency_index(12, 12) == 11);
    for (const int invalid : {0, 13})
    {
        bool threw = false;
        try
        {
            (void)librpa_int::sternheimer_headwing_frequency_index(invalid, 12);
        }
        catch (const std::out_of_range &)
        {
            threw = true;
        }
        assert(threw);
    }
}

void test_sternheimer_headwing_uses_response_frequency_grid()
{
    const std::vector<std::pair<int, double>> metadata{
        {1, 0.01939223117160}, {2, 0.06206988944744}, {1, 0.01939223117160}, {2, 0.06206988944744}};
    const auto frequencies = librpa_int::sternheimer_frequency_grid_from_metadata(metadata, 2);
    assert(frequencies.size() == 2);
    assert(frequencies[0] == metadata[0].second);
    assert(frequencies[1] == metadata[1].second);

    for (const auto &invalid :
         {std::vector<std::pair<int, double>>{{1, 0.01}, {2, 0.02}, {2, 0.03}},
          std::vector<std::pair<int, double>>{{1, 0.01}},
          std::vector<std::pair<int, double>>{{1, 0.02}, {2, 0.01}}})
    {
        bool threw = false;
        try
        {
            (void)librpa_int::sternheimer_frequency_grid_from_metadata(invalid, 2);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        assert(threw);
    }
}

void test_sternheimer_headwing_accepts_recomputed_frequency_roundoff()
{
    const std::vector<std::pair<int, double>> metadata{
        {1, 0.6787171755480910}, {2, 77.73973043128609},
        {1, 0.6787171755493192}, {2, 77.73973043142676}};
    const auto frequencies = librpa_int::sternheimer_frequency_grid_from_metadata(metadata, 2);
    assert(frequencies.size() == 2);
}

void test_sternheimer_head_only_replaces_gamma_head()
{
    librpa_int::ComplexMatrix coulomb(2, 2);
    coulomb(0, 0) = {9.0, 0.0};
    coulomb(1, 1) = {4.0, 0.0};

    librpa_int::ComplexMatrix response_m(2, 2);
    response_m(0, 0) = {-3.6, 0.0};
    response_m(1, 1) = {-0.8, 0.0};

    librpa_int::SternheimerRpaHeadwingInput headwing;
    headwing.mode = "head_only";
    headwing.head = librpa_int::ComplexMatrix(3, 3);
    headwing.head(0, 0) = {-0.1, 0.0};
    headwing.head(1, 1) = {-0.1, 0.0};
    headwing.head(2, 2) = {-0.1, 0.0};

    const auto result = librpa_int::compute_sternheimer_rpa_frequency_headwing(
        coulomb, response_m, headwing, 1, 0.5, 0.25, 1.0, 1e-12);
    const auto expected = std::log(std::complex<double>(1.1, 0.0)) - 0.1 +
                          std::log(std::complex<double>(1.2, 0.0)) - 0.2;
    require_close(result.integrand, expected, 1e-12);
}

void test_sternheimer_headwing_dense_projection_matches_direct_reference()
{
    librpa_int::ComplexMatrix coulomb(3, 3);
    coulomb(0, 0) = {9.0, 0.0};
    coulomb(1, 1) = {4.0, 0.0};
    coulomb(2, 2) = {1.0, 0.0};

    librpa_int::ComplexMatrix response_m(3, 3);
    response_m(0, 0) = {-2.7, 0.0};
    response_m(1, 1) = {-0.8, 0.0};
    response_m(2, 2) = {-0.1, 0.0};
    response_m(0, 1) = {0.12, 0.03};
    response_m(1, 0) = std::conj(response_m(0, 1));
    response_m(0, 2) = {0.03, -0.06};
    response_m(2, 0) = std::conj(response_m(0, 2));
    response_m(1, 2) = {0.04, 0.02};
    response_m(2, 1) = std::conj(response_m(1, 2));

    librpa_int::SternheimerRpaHeadwingInput headwing;
    headwing.mode = "head_only";
    headwing.head = librpa_int::ComplexMatrix(3, 3);
    headwing.head(0, 0) = {-0.03, 0.0};
    headwing.head(1, 1) = {-0.05, 0.0};
    headwing.head(2, 2) = {-0.07, 0.0};

    const auto result = librpa_int::compute_sternheimer_rpa_frequency_headwing(
        coulomb, response_m, headwing, 1, 0.5, 0.25, 1.0, 1e-12);

    librpa_int::ComplexMatrix expected_pi(3, 3);
    const double eigenvalues[] = {9.0, 4.0, 1.0};
    for (int i = 0; i != 3; ++i)
    {
        for (int j = 0; j != 3; ++j)
        {
            expected_pi(i, j) = response_m(i, j) / std::sqrt(eigenvalues[i] * eigenvalues[j]);
        }
    }
    expected_pi(0, 0) = {-0.05, 0.0};
    require_close(result.integrand, librpa_int::compute_rpa_trace_log_integrand(expected_pi),
                  1e-12);
}

void test_sternheimer_qavg_uses_analytic_head_and_wing()
{
    librpa_int::ComplexMatrix coulomb(2, 2);
    coulomb(0, 0) = {9.0, 0.0};
    coulomb(1, 1) = {4.0, 0.0};

    librpa_int::ComplexMatrix response_m(2, 2);
    response_m(0, 0) = {-3.6, 0.0};
    response_m(1, 1) = {-0.8, 0.0};

    librpa_int::SternheimerRpaHeadwingInput headwing;
    headwing.mode = "qavg";
    headwing.head = librpa_int::ComplexMatrix(3, 3);
    headwing.head(0, 0) = {-0.1, 0.0};
    headwing.head(1, 1) = {-0.1, 0.0};
    headwing.head(2, 2) = {-0.1, 0.0};
    headwing.wing_mu = librpa_int::ComplexMatrix(2, 3);
    headwing.wing_mu(1, 0) = {0.025, 0.0};
    headwing.directions = {{{1.0, 0.0, 0.0}, 1.0}};

    const auto result = librpa_int::compute_sternheimer_rpa_frequency_headwing(
        coulomb, response_m, headwing, 1, 0.5, 0.25, 1.0, 1e-12);
    const double schur = 1.1 - 0.05 * 0.05 / 1.2;
    const auto expected = std::complex<double>(-0.3, 0.0) +
                          std::log(std::complex<double>(1.2, 0.0)) +
                          std::log(std::complex<double>(schur, 0.0));
    require_close(result.integrand, expected, 1e-12);
}

void test_sternheimer_qavg_matches_standard_rpa_headwing_average()
{
    librpa_int::ComplexMatrix coulomb(2, 2);
    coulomb(0, 0) = {9.0, 0.0};
    coulomb(1, 1) = {4.0, 0.0};

    librpa_int::ComplexMatrix response_m(2, 2);
    response_m(0, 0) = {-3.6, 0.0};
    response_m(1, 1) = {-0.8, 0.0};

    librpa_int::SternheimerRpaHeadwingInput headwing;
    headwing.mode = "qavg";
    headwing.head = librpa_int::ComplexMatrix(3, 3);
    headwing.head(0, 0) = {-0.1, 0.0};
    headwing.head(1, 1) = {-0.2, 0.0};
    headwing.head(2, 2) = {-0.3, 0.0};
    headwing.wing_mu = librpa_int::ComplexMatrix(2, 3);
    headwing.wing_mu(1, 0) = {0.025, 0.010};
    headwing.wing_mu(1, 1) = {-0.015, 0.005};
    headwing.directions = {{{1.0, 0.0, 0.0}, 0.25}, {{0.0, 1.0, 0.0}, 0.75}};

    const auto sternheimer = librpa_int::compute_sternheimer_rpa_frequency_headwing(
        coulomb, response_m, headwing, 1, 0.5, 0.25, 1.0, 1e-12);

    const std::complex<double> body{-0.2, 0.0};
    const std::complex<double> body_inverse = 1.0 / (1.0 - body);
    const std::array<std::complex<double>, 3> wing{
        2.0 * headwing.wing_mu(1, 0), 2.0 * headwing.wing_mu(1, 1), 2.0 * headwing.wing_mu(1, 2)};
    librpa_int::matrix_m<std::complex<double>> standard_head(
        std::vector<std::vector<std::complex<double>>>{
            {headwing.head(0, 0), headwing.head(0, 1), headwing.head(0, 2)},
            {headwing.head(1, 0), headwing.head(1, 1), headwing.head(1, 2)},
            {headwing.head(2, 0), headwing.head(2, 1), headwing.head(2, 2)}},
        librpa_int::MAJOR::COL);
    librpa_int::matrix_m<std::complex<double>> standard_schur(3, 3, librpa_int::MAJOR::COL);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        for (int beta = 0; beta != 3; ++beta)
        {
            standard_schur(alpha, beta) = (alpha == beta ? 1.0 : 0.0) - headwing.head(alpha, beta) -
                                          std::conj(wing[alpha]) * body_inverse * wing[beta];
        }
    }
    const std::vector<double> qx{1.0, 0.0};
    const std::vector<double> qy{0.0, 1.0};
    const std::vector<double> qz{0.0, 0.0};
    const std::vector<double> weights{0.25, 0.75};
    const auto standard = librpa_int::compute_rpa_chi0v_headwing_trace_log_average(
        standard_head, standard_schur, body, std::log(1.0 - body), qx, qy, qz, weights);

    require_close(sternheimer.integrand, standard, 1e-12);
}

void test_sternheimer_qavg_body_start_zero_keeps_all_positive_finite_part_channels()
{
    librpa_int::ComplexMatrix coulomb(3, 3);
    coulomb(0, 0) = {-1.0, 0.0};
    coulomb(1, 1) = {4.0, 0.0};
    coulomb(2, 2) = {9.0, 0.0};

    librpa_int::ComplexMatrix response_m(3, 3);
    response_m(1, 1) = {-0.8, 0.0};
    response_m(2, 2) = {-1.8, 0.0};

    librpa_int::SternheimerRpaHeadwingInput headwing;
    headwing.mode = "qavg";
    headwing.body_start = 0;
    headwing.head = librpa_int::ComplexMatrix(3, 3);
    headwing.wing_mu = librpa_int::ComplexMatrix(3, 3);
    headwing.directions = {{{1.0, 0.0, 0.0}, 1.0}};

    const auto result = librpa_int::compute_sternheimer_rpa_frequency_headwing(
        coulomb, response_m, headwing, 1, 0.5, 0.25, 1.0, 1e-12);
    const auto one_body_channel = std::log(std::complex<double>(1.2, 0.0)) - 0.2;
    require_close(result.integrand, 2.0 * one_body_channel, 1e-12);
}
}  // namespace

int main()
{
    test_sternheimer_pi_and_trace_log_match_diagonal_reference();
    test_sternheimer_headwing_frequency_uses_one_based_response_labels();
    test_sternheimer_headwing_uses_response_frequency_grid();
    test_sternheimer_headwing_accepts_recomputed_frequency_roundoff();
    test_sternheimer_head_only_replaces_gamma_head();
    test_sternheimer_headwing_dense_projection_matches_direct_reference();
    test_sternheimer_qavg_uses_analytic_head_and_wing();
    test_sternheimer_qavg_matches_standard_rpa_headwing_average();
    test_sternheimer_qavg_body_start_zero_keeps_all_positive_finite_part_channels();
    return 0;
}
