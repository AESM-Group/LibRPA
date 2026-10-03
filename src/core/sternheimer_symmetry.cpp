/*!
 * @file sternheimer_symmetry.cpp
 * @brief Fixed-q symmetry reduction for partial Sternheimer responses.
 */
#include "sternheimer_symmetry.h"

#include "symmetry_context.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace librpa_int
{
namespace
{

void require_hermitian_response(const ComplexMatrix &matrix,
                                const std::string &label,
                                const double relative_tolerance = 1e-10)
{
    if (matrix.nr != matrix.nc)
    {
        throw std::runtime_error(label + " must be square");
    }

    double max_value = 0.0;
    double max_residual = 0.0;
    for (int row = 0; row != matrix.nr; ++row)
    {
        for (int column = 0; column != matrix.nc; ++column)
        {
            max_value = std::max(max_value, std::abs(matrix(row, column)));
            max_residual = std::max(
                max_residual,
                std::abs(matrix(row, column) - std::conj(matrix(column, row))));
        }
    }
    if (max_residual > relative_tolerance * std::max(1.0, max_value))
    {
        throw std::runtime_error(label + " is not Hermitian");
    }
}

Vector3_Order<double> apply_route(const SpaceGroupSymOps &operations,
                                  const SternheimerSymmetryRoute &route,
                                  const Vector3_Order<double> &k)
{
    if (route.spatial_isym < 0
        || route.spatial_isym >= static_cast<int>(operations.size()))
    {
        throw std::runtime_error("Sternheimer symmetry route has an invalid spatial index");
    }
    auto transformed = apply_space_group_rotation_to_kpoint(
        operations.at(static_cast<std::size_t>(route.spatial_isym)), k);
    return route.time_reversal ? transformed * -1.0 : transformed;
}

FoldedKPoint fold_route_to_grid(const SpaceGroupSymOps &operations,
                                const SternheimerSymmetryRoute &route,
                                const Vector3_Order<double> &k,
                                const std::vector<Vector3_Order<double>> &full_kpoints,
                                const double tolerance)
{
    return fold_fractional_kpoint_to_targets(
        apply_route(operations, route, k), full_kpoints, tolerance);
}

SternheimerSymmetryRoute find_inverse_route(
    const SpaceGroupSymOps &operations,
    const std::vector<SternheimerSymmetryRoute> &little_group,
    const Vector3_Order<double> &member,
    const Vector3_Order<double> &representative,
    const double tolerance)
{
    for (const auto &route : little_group)
    {
        FoldedKPoint folded;
        if (!try_fold_fractional_kpoint_to_target(
                apply_route(operations, route, member), representative, tolerance, folded))
        {
            continue;
        }
        auto inverse_route = route;
        inverse_route.fold_G = folded.fold_G;
        return inverse_route;
    }
    throw std::runtime_error(
        "Fixed-q k orbit has no little-group route from member to representative");
}

}  // namespace

std::vector<SternheimerSymmetryRoute> build_sternheimer_fixed_q_little_group(
    const SpaceGroupSymOps &spatial_operations,
    const Vector3_Order<double> &q,
    const double tolerance)
{
    if (spatial_operations.empty())
    {
        throw std::runtime_error("Cannot build a fixed-q little group without spatial operations");
    }

    std::vector<SternheimerSymmetryRoute> little_group;
    little_group.reserve(2 * spatial_operations.size());
    for (const bool time_reversal : {false, true})
    {
        for (std::size_t isym = 0; isym != spatial_operations.size(); ++isym)
        {
            SternheimerSymmetryRoute route;
            route.spatial_isym = static_cast<int>(isym);
            route.time_reversal = time_reversal;
            FoldedKPoint folded;
            if (!try_fold_fractional_kpoint_to_target(
                    apply_route(spatial_operations, route, q), q, tolerance, folded))
            {
                continue;
            }
            route.fold_G = folded.fold_G;
            little_group.push_back(route);
        }
    }
    if (little_group.empty())
    {
        throw std::runtime_error("Fixed-q little group is empty; the identity operation is missing");
    }
    return little_group;
}

std::vector<SternheimerFixedQKOrbit> build_sternheimer_fixed_q_k_orbits(
    const SpaceGroupSymOps &spatial_operations,
    const std::vector<Vector3_Order<double>> &full_kpoints,
    const Vector3_Order<double> &q,
    const double tolerance)
{
    if (full_kpoints.empty())
    {
        throw std::runtime_error("Cannot build fixed-q k orbits for an empty k grid");
    }
    const auto little_group =
        build_sternheimer_fixed_q_little_group(spatial_operations, q, tolerance);

    std::vector<bool> assigned(full_kpoints.size(), false);
    std::vector<SternheimerFixedQKOrbit> orbits;
    for (std::size_t seed = 0; seed != full_kpoints.size(); ++seed)
    {
        if (assigned[seed])
        {
            continue;
        }

        std::set<int> member_indices;
        std::deque<int> pending;
        member_indices.insert(static_cast<int>(seed));
        pending.push_back(static_cast<int>(seed));
        while (!pending.empty())
        {
            const int ik = pending.front();
            pending.pop_front();
            for (const auto &route : little_group)
            {
                const auto folded = fold_route_to_grid(
                    spatial_operations,
                    route,
                    full_kpoints.at(static_cast<std::size_t>(ik)),
                    full_kpoints,
                    tolerance);
                if (member_indices.insert(folded.target_k_index).second)
                {
                    pending.push_back(folded.target_k_index);
                }
            }
        }

        SternheimerFixedQKOrbit orbit;
        orbit.representative_ik_full = *member_indices.begin();
        orbit.members.reserve(member_indices.size());
        const auto &representative =
            full_kpoints.at(static_cast<std::size_t>(orbit.representative_ik_full));
        for (const int member_index : member_indices)
        {
            if (assigned.at(static_cast<std::size_t>(member_index)))
            {
                throw std::runtime_error("Fixed-q k orbits overlap");
            }
            assigned[static_cast<std::size_t>(member_index)] = true;
            orbit.members.push_back({
                member_index,
                find_inverse_route(spatial_operations,
                                   little_group,
                                   full_kpoints.at(static_cast<std::size_t>(member_index)),
                                   representative,
                                   tolerance)});
        }
        orbits.push_back(std::move(orbit));
    }

    if (!std::all_of(assigned.begin(), assigned.end(), [](const bool value) { return value; }))
    {
        throw std::runtime_error("Fixed-q k orbits do not cover the full k grid");
    }
    return orbits;
}

ComplexMatrix rotate_sternheimer_partial_response(
    const SymmetryContext &symmetry,
    const std::vector<SpeciesBasisLayout> &layouts,
    const std::map<atom_t, std::size_t> &atom_nabf,
    const SternheimerSymmetryRoute &inverse_route,
    const Vector3_Order<double> &q,
    const ComplexMatrix &partial_response,
    const int lmax)
{
    require_hermitian_response(partial_response, "Sternheimer partial response");
    const auto member = build_symmetry_kspace_operation_member(
        symmetry,
        inverse_route.spatial_isym,
        inverse_route.time_reversal,
        q,
        q,
        lmax);
    auto rotated = rotate_symmetry_kspace_matrix(symmetry,
                                                 layouts,
                                                 member,
                                                 partial_response,
                                                 atom_nabf,
                                                 q);
    require_hermitian_response(rotated, "Rotated Sternheimer partial response");
    return rotated;
}

ComplexMatrix reconstruct_sternheimer_fixed_q_response(
    const SymmetryContext &symmetry,
    const std::vector<SpeciesBasisLayout> &layouts,
    const std::map<atom_t, std::size_t> &atom_nabf,
    const Vector3_Order<double> &q,
    const std::vector<SternheimerFixedQKOrbit> &orbits,
    const std::map<int, ComplexMatrix> &representative_responses,
    const int lmax,
    std::vector<SternheimerFixedQKResponse> *kresolved_responses)
{
    if (orbits.empty())
    {
        throw std::runtime_error("Cannot reconstruct a fixed-q response from no k orbits");
    }

    std::set<int> used_representatives;
    ComplexMatrix reconstructed;
    bool initialized = false;
    std::vector<SternheimerFixedQKResponse> kresolved;
    for (const auto &orbit : orbits)
    {
        if (!used_representatives.insert(orbit.representative_ik_full).second)
        {
            throw std::runtime_error("Duplicate fixed-q k-orbit representative");
        }
        const auto response_iter =
            representative_responses.find(orbit.representative_ik_full);
        if (response_iter == representative_responses.end())
        {
            throw std::runtime_error(
                "Fixed-q reconstruction is missing representative partial response ik=" +
                std::to_string(orbit.representative_ik_full));
        }
        if (!initialized)
        {
            reconstructed = ComplexMatrix(response_iter->second.nr, response_iter->second.nc);
            reconstructed.zero_out();
            initialized = true;
        }
        if (response_iter->second.nr != reconstructed.nr
            || response_iter->second.nc != reconstructed.nc)
        {
            throw std::runtime_error(
                "Fixed-q representative partial responses have inconsistent dimensions");
        }
        for (const auto &member : orbit.members)
        {
            SternheimerFixedQKResponse response;
            response.ik_full = member.ik_full;
            response.matrix = rotate_sternheimer_partial_response(symmetry,
                                                                   layouts,
                                                                   atom_nabf,
                                                                   member.inverse_route,
                                                                   q,
                                                                   response_iter->second,
                                                                   lmax);
            reconstructed += response.matrix;
            kresolved.push_back(std::move(response));
        }
    }
    if (representative_responses.size() != used_representatives.size())
    {
        throw std::runtime_error("Fixed-q reconstruction received unused partial responses");
    }
    require_hermitian_response(reconstructed, "Reconstructed fixed-q Sternheimer response");
    std::sort(kresolved.begin(), kresolved.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.ik_full < rhs.ik_full;
    });
    for (std::size_t index = 1; index != kresolved.size(); ++index)
    {
        if (kresolved[index - 1].ik_full == kresolved[index].ik_full)
        {
            throw std::runtime_error("Fixed-q reconstruction produced duplicate k-resolved responses");
        }
    }
    if (kresolved_responses != nullptr)
    {
        *kresolved_responses = std::move(kresolved);
    }
    return reconstructed;
}

std::vector<SternheimerQStarResponse> reconstruct_sternheimer_qstar_responses(
    const SymmetryContext &symmetry,
    const std::vector<SpeciesBasisLayout> &layouts,
    const std::map<atom_t, std::size_t> &atom_nabf,
    const Vector3_Order<double> &q_representative,
    const ComplexMatrix &representative_response,
    const int lmax)
{
    require_hermitian_response(
        representative_response, "Representative q-star Sternheimer response");
    const auto &star = find_symmetry_kstar_for_kpoint(
        symmetry.kstars, q_representative, "Sternheimer q-stars");
    if (!same_fractional_kpoint(star.k_ibz, q_representative, 1e-8))
    {
        throw std::runtime_error(
            "Sternheimer q-star input must use the canonical star representative");
    }

    std::vector<SternheimerQStarResponse> restored;
    restored.reserve(star.members.size());
    for (std::size_t member_index = 0; member_index != star.members.size(); ++member_index)
    {
        const auto &member = star.members[member_index];
        if (std::any_of(restored.begin(), restored.end(), [&member](const auto &existing) {
                return same_fractional_kpoint(existing.q, member.k_bz, 1e-8);
            }))
        {
            throw std::runtime_error("Sternheimer q-star contains a duplicate full-q member");
        }

        const Vector3_Order<double> *target =
            same_fractional_kpoint(member.k_bz, star.k_ibz, 1e-8) ? nullptr : &member.k_bz;
        auto matrix = rotate_symmetry_kspace_matrix(symmetry,
                                                    layouts,
                                                    member,
                                                    representative_response,
                                                    atom_nabf,
                                                    star.k_ibz,
                                                    false,
                                                    target);
        require_hermitian_response(matrix, "Restored q-star Sternheimer response");
        restored.push_back({star.star_index,
                            static_cast<int>(member_index),
                            member.spatial_isym,
                            member.time_reversal,
                            member.k_bz,
                            std::move(matrix)});
    }
    if (restored.empty())
    {
        throw std::runtime_error("Sternheimer q-star has no members");
    }
    return restored;
}

}  // namespace librpa_int
