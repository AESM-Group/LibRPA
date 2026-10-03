#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <vector>

#include "../rpa_qsum.h"

namespace
{

void require_close(const std::complex<double> &actual, const std::complex<double> &expected)
{
    assert(std::abs(actual - expected) < 1.0e-14);
}

void test_separates_gamma_without_renormalizing_remaining_qpoints()
{
    const std::vector<driver::RpaQContribution> contributions = {
        {{0.0, 0.0, 0.0}, {-2.5, 0.0}},
        {{0.25, 0.0, 0.0}, {-0.1, 0.0}},
        {{0.5, 0.0, 0.0}, {-0.2, 0.0}},
    };

    const auto totals = driver::sum_rpa_q_contributions(contributions);

    assert(totals.gamma_count == 1);
    require_close(totals.including_gamma, {-2.8, 0.0});
    require_close(totals.gamma, {-2.5, 0.0});
    require_close(totals.excluding_gamma, {-0.3, 0.0});
}

void test_selects_requested_total()
{
    driver::RpaQTotals totals;
    totals.including_gamma = {-2.8, 0.0};
    totals.excluding_gamma = {-0.3, 0.0};

    require_close(driver::select_rpa_q_total(totals, true), {-2.8, 0.0});
    require_close(driver::select_rpa_q_total(totals, false), {-0.3, 0.0});
}

}  // namespace

int main()
{
    test_separates_gamma_without_renormalizing_remaining_qpoints();
    test_selects_requested_total();
    return 0;
}
