#pragma once

#include "crpa.h"
#include "librpa_enums.h"

namespace librpa_int
{
struct CrpaContinuedTensor
{
    int atom_index, spin_left, spin_right, n_orbitals;
    std::vector<ComplexMatrix> values;
    std::vector<std::vector<int>> status;
    std::vector<std::vector<double>> spread;
    double max_source_residual = 0, max_holdout_residual = 0;
};

//! Selected rational continuation of Delta U(z)=U(z)-v, z=i*nu. No fermionic shift.
//! Status bits: 1 primary failure, 2 subset failure, 4 subset spread,
//! 8 source/holdout residual, 16 insufficient nodes, 32 nonfinite evaluation.
//! These are numerical diagnostics, not passivity or material convergence certificates.
std::vector<CrpaContinuedTensor> continue_crpa_interactions(
    const CrpaResult& result, const std::vector<std::complex<double>>& queries, int n_parameters,
    double tolerance, LibrpaAnalyContMethod method = LIBRPA_ANACON_PADE,
    double ridge_lambda = 1.0e-12, double ridge_den_weight = 10.0, double aaa_tolerance = 1.0e-13);
}  // namespace librpa_int
