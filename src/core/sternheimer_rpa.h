#pragma once

#include <array>
#include <complex>
#include <string>
#include <utility>
#include <vector>

#include "../math/complexmatrix.h"

namespace librpa_int
{

struct SternheimerRpaFrequencyResult
{
    int ifreq = 0;
    double omega = 0.0;
    double weight = 0.0;
    double qweight = 1.0;
    std::complex<double> trace_pi = {0.0, 0.0};
    std::complex<double> logdet = {0.0, 0.0};
    std::complex<double> integrand = {0.0, 0.0};
    std::complex<double> energy = {0.0, 0.0};
};

struct SternheimerRpaAngularPoint
{
    std::array<double, 3> direction{0.0, 0.0, 0.0};
    double weight = 0.0;
};

// Analytic q->0 response in the same convention used by the ordinary RPA
// head/wing path. head and wing_mu are chi0*v quantities; wing_mu is still in
// the auxiliary-basis representation and is transformed with sqrt(V) here.
struct SternheimerRpaHeadwingInput
{
    std::string mode = "qavg";
    int body_start = 1;
    ComplexMatrix head;
    ComplexMatrix wing_mu;
    std::vector<SternheimerRpaAngularPoint> directions;
};

int sternheimer_headwing_frequency_index(int response_ifreq, int nfreq);

std::vector<double> sternheimer_frequency_grid_from_metadata(
    const std::vector<std::pair<int, double>> &metadata, int expected_nfreq);

ComplexMatrix compute_sternheimer_pi_from_m(const ComplexMatrix &coulomb,
                                            const ComplexMatrix &response_m,
                                            double sqrt_coulomb_threshold);

std::complex<double> compute_rpa_trace_log_integrand(const ComplexMatrix &pi);

SternheimerRpaFrequencyResult compute_sternheimer_rpa_frequency(const ComplexMatrix &coulomb,
                                                                const ComplexMatrix &response_m,
                                                                int ifreq, double omega,
                                                                double weight, double qweight,
                                                                double sqrt_coulomb_threshold);

SternheimerRpaFrequencyResult compute_sternheimer_rpa_frequency_headwing(
    const ComplexMatrix &coulomb, const ComplexMatrix &response_m,
    const SternheimerRpaHeadwingInput &headwing, int ifreq, double omega, double weight,
    double qweight, double sqrt_coulomb_threshold);

std::complex<double> sum_sternheimer_rpa_energies(
    const std::vector<SternheimerRpaFrequencyResult> &results);

}  // namespace librpa_int
