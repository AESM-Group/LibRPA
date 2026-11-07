#pragma once

#include <map>
#include <vector>

#include "../../math/matrix_m.h"
#include "../meanfield.h"
#include "librpa_enums.h"

namespace librpa_int
{
namespace qsgw
{

enum class CorrelationPotentialMode
{
    ModeA,
    ModeB,
};

struct CorrelationPotentialSettings
{
    CorrelationPotentialMode mode = CorrelationPotentialMode::ModeB;
    LibrpaAnalyContMethod anacon_method = LIBRPA_ANACON_PADE;
    double anacon_aaa_tolerance = 1.0e-13;
    double anacon_ridge_lambda = 1.0e-12;
    double anacon_ridge_den_weight = 10.0;
    int n_params_anacon = -1;
    int n_params_anacon_resample = -1;
    std::vector<cplxdb> resample_imagfreqs;
};

Matz build_qsgw_correlation_potential(
    const MeanField& meanfield,
    const std::vector<double>& source_frequencies,
    const std::map<double, Matz>& sigma_imaginary_axis,
    int spin,
    int kpoint,
    const CorrelationPotentialSettings& settings = {});

} // namespace qsgw
} // namespace librpa_int
