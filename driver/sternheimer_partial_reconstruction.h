#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "../src/core/atomic_basis.h"
#include "../src/core/sternheimer_rpa.h"
#include "../src/core/sternheimer_symmetry.h"
#include "../src/math/vector3_order.h"
#include "reader_sternheimer_partial.h"
#include "reader_sternheimer_qpoints.h"

namespace librpa_int
{
struct SymmetryContext;
}

namespace driver
{

struct SternheimerReconstructedResponse
{
    int iq = 0;
    int ifreq = 0;
    double omega = 0.0;
    double weight = 0.0;
    double q_weight = 0.0;
    int full_k_count = 0;
    int representative_k_count = 0;
    int little_group_order = 0;
    librpa_int::ComplexMatrix matrix;
    std::vector<librpa_int::SternheimerFixedQKResponse> kresolved_responses;
    std::vector<librpa_int::SternheimerQStarResponse> qstar_responses;
};

struct SternheimerFixedQSymmetryDiagnostic
{
    int representative_ik_full = -1;
    int member_ik_full = -1;
    librpa_int::SternheimerSymmetryRoute inverse_route;
    librpa_int::ComplexMatrix transform;
};

struct SternheimerQStarRpaAudit
{
    librpa_int::SternheimerRpaFrequencyResult result;
    int qstar_size = 0;
    double max_integrand_difference = 0.0;
};

std::vector<librpa_int::SternheimerQStarResponse> reconstruct_sternheimer_full_q_matrices_from_ibz(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &ibz_qpoints,
    const std::vector<librpa_int::ComplexMatrix> &ibz_matrices, int lmax);

std::vector<SternheimerReconstructedResponse> reconstruct_sternheimer_partial_responses(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const std::vector<librpa_int::Vector3_Order<double>> &full_qpoints,
    const std::vector<SternheimerQPoint> &qpoints, const SternheimerPartialResponseGroups &groups,
    int expected_nfreq, bool use_rpa_gamma, int lmax,
    const std::vector<SternheimerFixedQRouteRecord> *fixed_q_routes = nullptr,
    bool fixed_q_matrix_only = false,
    const std::vector<SternheimerQStarRouteRecord> *qstar_routes = nullptr);

std::vector<librpa_int::SternheimerFixedQKOrbit> build_sternheimer_fixed_q_k_orbits_from_routes(
    const librpa_int::SpaceGroupSymOps &spatial_operations,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const librpa_int::Vector3_Order<double> &q,
    const std::vector<SternheimerFixedQRouteRecord> &routes, double tolerance = 1.0e-8);

std::vector<librpa_int::SternheimerQStarResponse> build_sternheimer_qstar_responses_from_routes(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_qpoints, int representative_iq,
    const librpa_int::Vector3_Order<double> &q_representative,
    const librpa_int::ComplexMatrix &representative_response,
    const std::vector<SternheimerQStarRouteRecord> &routes, int lmax, double tolerance = 1.0e-8);

std::vector<SternheimerFixedQSymmetryDiagnostic> build_sternheimer_fixed_q_symmetry_diagnostics(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const librpa_int::Vector3_Order<double> &q, int lmax,
    const std::vector<SternheimerFixedQRouteRecord> *fixed_q_routes = nullptr);

void write_sternheimer_fixed_q_symmetry_diagnostics(
    const std::string &path, int iq,
    const std::vector<SternheimerFixedQSymmetryDiagnostic> &diagnostics);

SternheimerQStarRpaAudit compute_sternheimer_qstar_rpa_frequency(
    const SternheimerReconstructedResponse &response,
    const std::vector<librpa_int::SternheimerQStarResponse> &coulomb_qstar,
    double sqrt_coulomb_threshold,
    const librpa_int::SternheimerRpaHeadwingInput *headwing = nullptr,
    double invariance_tolerance = 1.0e-6);

}  // namespace driver
