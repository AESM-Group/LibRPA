#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <vector>

namespace driver
{

struct RpaQContribution
{
    std::array<double, 3> q = {0.0, 0.0, 0.0};
    std::complex<double> energy = {0.0, 0.0};
};

struct RpaQTotals
{
    std::complex<double> including_gamma = {0.0, 0.0};
    std::complex<double> gamma = {0.0, 0.0};
    std::complex<double> excluding_gamma = {0.0, 0.0};
    std::size_t gamma_count = 0;
};

bool is_rpa_gamma_point(const std::array<double, 3> &q, double threshold = 1.0e-5);

RpaQTotals sum_rpa_q_contributions(const std::vector<RpaQContribution> &contributions);

std::complex<double> select_rpa_q_total(const RpaQTotals &totals, bool use_rpa_gamma);

}  // namespace driver
