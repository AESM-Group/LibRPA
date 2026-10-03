#include "rpa_qsum.h"

#include <cmath>

namespace driver
{

bool is_rpa_gamma_point(const std::array<double, 3> &q, const double threshold)
{
    return std::abs(q[0]) < threshold && std::abs(q[1]) < threshold && std::abs(q[2]) < threshold;
}

RpaQTotals sum_rpa_q_contributions(const std::vector<RpaQContribution> &contributions)
{
    RpaQTotals totals;
    for (const auto &contribution : contributions)
    {
        totals.including_gamma += contribution.energy;
        if (is_rpa_gamma_point(contribution.q))
        {
            totals.gamma += contribution.energy;
            ++totals.gamma_count;
        }
        else
        {
            totals.excluding_gamma += contribution.energy;
        }
    }
    return totals;
}

std::complex<double> select_rpa_q_total(const RpaQTotals &totals, const bool use_rpa_gamma)
{
    return use_rpa_gamma ? totals.including_gamma : totals.excluding_gamma;
}

}  // namespace driver
