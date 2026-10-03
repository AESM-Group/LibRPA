#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

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

void test_two_q_weighted_trace_log_sum()
{
    librpa_int::ComplexMatrix coulomb_q1(1, 1);
    coulomb_q1(0, 0) = {4.0, 0.0};
    librpa_int::ComplexMatrix response_q1(1, 1);
    response_q1(0, 0) = {-0.8, 0.0};

    librpa_int::ComplexMatrix coulomb_q2(1, 1);
    coulomb_q2(0, 0) = {9.0, 0.0};
    librpa_int::ComplexMatrix response_q2(1, 1);
    response_q2(0, 0) = {-0.9, 0.0};

    const auto q1 = librpa_int::compute_sternheimer_rpa_frequency(coulomb_q1, response_q1, 1, 0.5,
                                                                  0.4, 0.25, 1.0e-12);
    const auto q2 = librpa_int::compute_sternheimer_rpa_frequency(coulomb_q2, response_q2, 1, 0.5,
                                                                  0.4, 0.75, 1.0e-12);

    const auto actual = librpa_int::sum_sternheimer_rpa_energies({q1, q2});
    const auto integrand_q1 = std::log(std::complex<double>(1.2, 0.0)) - 0.2;
    const auto integrand_q2 = std::log(std::complex<double>(1.1, 0.0)) - 0.1;
    const auto expected = 0.4 * (0.25 * integrand_q1 + 0.75 * integrand_q2) / librpa_int::TWO_PI;
    require_close(actual, expected, 1.0e-12);
}

}  // namespace

int main()
{
    test_two_q_weighted_trace_log_sum();
    return 0;
}
