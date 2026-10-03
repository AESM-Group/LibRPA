/*!
 * @file sternheimer_symmetry.h
 * @brief Fixed-q symmetry reduction for partial Sternheimer responses.
 */
#pragma once

#include <cstddef>
#include <map>
#include <vector>

#include "atomic_basis.h"
#include "symmetry_types.h"
#include "../math/complexmatrix.h"
#include "../math/symmetry.h"
#include "../math/vector3_order.h"

namespace librpa_int
{

struct SymmetryContext;

// The operation maps a canonical fractional k point to another canonical
// representative modulo fold_G. Time reversal applies an additional k -> -k.
struct SternheimerSymmetryRoute
{
    int spatial_isym = -1;
    bool time_reversal = false;
    Vector3_Order<int> fold_G{0, 0, 0};
};

struct SternheimerFixedQKOrbitMember
{
    int ik_full = -1;
    // Maps this member back to the orbit representative, matching the
    // inverse-route convention of SymmetryKStarMember.
    SternheimerSymmetryRoute inverse_route;
};

struct SternheimerFixedQKOrbit
{
    int representative_ik_full = -1;
    std::vector<SternheimerFixedQKOrbitMember> members;
};

struct SternheimerFixedQKResponse
{
    int ik_full = -1;
    ComplexMatrix matrix;
};

struct SternheimerQStarResponse
{
    int star_index = -1;
    int member_index = -1;
    int spatial_isym = -1;
    bool time_reversal = false;
    Vector3_Order<double> q{0.0, 0.0, 0.0};
    ComplexMatrix matrix;
};

std::vector<SternheimerSymmetryRoute> build_sternheimer_fixed_q_little_group(
    const SpaceGroupSymOps &spatial_operations,
    const Vector3_Order<double> &q,
    double tolerance = 1e-8);

std::vector<SternheimerFixedQKOrbit> build_sternheimer_fixed_q_k_orbits(
    const SpaceGroupSymOps &spatial_operations,
    const std::vector<Vector3_Order<double>> &full_kpoints,
    const Vector3_Order<double> &q,
    double tolerance = 1e-8);

ComplexMatrix rotate_sternheimer_partial_response(
    const SymmetryContext &symmetry,
    const std::vector<SpeciesBasisLayout> &layouts,
    const std::map<atom_t, std::size_t> &atom_nabf,
    const SternheimerSymmetryRoute &inverse_route,
    const Vector3_Order<double> &q,
    const ComplexMatrix &partial_response,
    int lmax);

ComplexMatrix reconstruct_sternheimer_fixed_q_response(
    const SymmetryContext &symmetry,
    const std::vector<SpeciesBasisLayout> &layouts,
    const std::map<atom_t, std::size_t> &atom_nabf,
    const Vector3_Order<double> &q,
    const std::vector<SternheimerFixedQKOrbit> &orbits,
    const std::map<int, ComplexMatrix> &representative_responses,
    int lmax,
    std::vector<SternheimerFixedQKResponse> *kresolved_responses = nullptr);

std::vector<SternheimerQStarResponse> reconstruct_sternheimer_qstar_responses(
    const SymmetryContext &symmetry,
    const std::vector<SpeciesBasisLayout> &layouts,
    const std::map<atom_t, std::size_t> &atom_nabf,
    const Vector3_Order<double> &q_representative,
    const ComplexMatrix &representative_response,
    int lmax);

}  // namespace librpa_int
