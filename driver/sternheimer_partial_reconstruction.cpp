#include "sternheimer_partial_reconstruction.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "../src/core/symmetry_context.h"
#include "rpa_qsum.h"

namespace driver
{
namespace
{

std::vector<int> ordered_atom_naux(const std::map<librpa_int::atom_t, std::size_t> &atom_nabf)
{
    std::vector<int> result(atom_nabf.size(), -1);
    for (const auto &[atom, count] : atom_nabf)
    {
        const auto index = static_cast<std::size_t>(atom);
        if (index >= result.size() || count == 0)
        {
            throw std::runtime_error(
                "Sternheimer auxiliary basis has invalid atom indices or dimensions");
        }
        result[index] = static_cast<int>(count);
    }
    if (std::find(result.begin(), result.end(), -1) != result.end())
    {
        throw std::runtime_error("Sternheimer auxiliary basis atom indices are not contiguous");
    }
    return result;
}

librpa_int::Vector3_Order<double> q_vector(const SternheimerQPoint &point)
{
    return {point.q[0], point.q[1], point.q[2]};
}

}  // namespace

std::vector<librpa_int::SternheimerQStarResponse> reconstruct_sternheimer_full_q_matrices_from_ibz(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &ibz_qpoints,
    const std::vector<librpa_int::ComplexMatrix> &ibz_matrices, const int lmax)
{
    if (ibz_qpoints.empty() || ibz_qpoints.size() != ibz_matrices.size())
    {
        throw std::runtime_error(
            "Sternheimer IBZ q points and matrices must have the same non-zero count");
    }

    std::size_t expected_full_q_count = 0;
    for (const auto &star : symmetry.kstars)
    {
        expected_full_q_count += star.members.size();
    }

    std::vector<librpa_int::SternheimerQStarResponse> restored;
    restored.reserve(expected_full_q_count);
    for (std::size_t index = 0; index != ibz_qpoints.size(); ++index)
    {
        auto star = librpa_int::reconstruct_sternheimer_qstar_responses(
            symmetry, layouts, atom_nabf, ibz_qpoints[index], ibz_matrices[index], lmax);
        for (auto &member : star)
        {
            if (std::any_of(
                    restored.cbegin(), restored.cend(),
                    [&member](const auto &existing)
                    { return librpa_int::same_fractional_kpoint(existing.q, member.q, 1.0e-8); }))
            {
                throw std::runtime_error(
                    "Sternheimer IBZ matrix reconstruction produced a duplicate full-q point");
            }
            restored.push_back(std::move(member));
        }
    }
    if (restored.size() != expected_full_q_count)
    {
        throw std::runtime_error("Sternheimer IBZ matrices do not cover the full q grid");
    }
    return restored;
}

std::vector<librpa_int::SternheimerFixedQKOrbit> build_sternheimer_fixed_q_k_orbits_from_routes(
    const librpa_int::SpaceGroupSymOps &spatial_operations,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const librpa_int::Vector3_Order<double> &q,
    const std::vector<SternheimerFixedQRouteRecord> &routes, const double tolerance)
{
    if (spatial_operations.empty() || full_kpoints.empty() || routes.empty() || tolerance <= 0.0)
    {
        throw std::runtime_error(
            "Explicit fixed-q Sternheimer routes require non-empty symmetry, k-grid, and route "
            "data");
    }

    const int iq = routes.front().iq;
    std::vector<bool> covered(full_kpoints.size(), false);
    std::map<int, librpa_int::SternheimerFixedQKOrbit> orbits_by_representative;
    for (const auto &record : routes)
    {
        if (record.iq != iq || record.member_ik_full < 0 ||
            record.member_ik_full >= static_cast<int>(full_kpoints.size()) ||
            record.representative_ik_full < 0 ||
            record.representative_ik_full >= static_cast<int>(full_kpoints.size()))
        {
            throw std::runtime_error(
                "Explicit fixed-q Sternheimer route has inconsistent q or k indices");
        }
        if (covered.at(static_cast<std::size_t>(record.member_ik_full)))
        {
            throw std::runtime_error(
                "Explicit fixed-q Sternheimer routes contain a duplicate full-k member");
        }
        const auto &route = record.inverse_route;
        if (route.spatial_isym < 0 ||
            route.spatial_isym >= static_cast<int>(spatial_operations.size()))
        {
            throw std::runtime_error(
                "Explicit fixed-q Sternheimer route has an invalid spatial operation");
        }

        auto apply_route = [&](const librpa_int::Vector3_Order<double> &point)
        {
            auto transformed = librpa_int::apply_space_group_rotation_to_kpoint(
                spatial_operations.at(static_cast<std::size_t>(route.spatial_isym)), point);
            return route.time_reversal ? transformed * -1.0 : transformed;
        };
        librpa_int::FoldedKPoint q_folded;
        if (!librpa_int::try_fold_fractional_kpoint_to_target(apply_route(q), q, tolerance,
                                                              q_folded))
        {
            throw std::runtime_error("Explicit fixed-q Sternheimer route does not preserve q");
        }

        librpa_int::FoldedKPoint k_folded;
        if (!librpa_int::try_fold_fractional_kpoint_to_target(
                apply_route(full_kpoints.at(static_cast<std::size_t>(record.member_ik_full))),
                full_kpoints.at(static_cast<std::size_t>(record.representative_ik_full)), tolerance,
                k_folded))
        {
            throw std::runtime_error(
                "Explicit fixed-q Sternheimer route does not map member k to its representative");
        }
        if (k_folded.fold_G.x != route.fold_G.x || k_folded.fold_G.y != route.fold_G.y ||
            k_folded.fold_G.z != route.fold_G.z)
        {
            throw std::runtime_error(
                "Explicit fixed-q Sternheimer route reciprocal fold disagrees with the k mapping");
        }

        covered[static_cast<std::size_t>(record.member_ik_full)] = true;
        auto &orbit = orbits_by_representative[record.representative_ik_full];
        orbit.representative_ik_full = record.representative_ik_full;
        orbit.members.push_back({record.member_ik_full, route});
    }
    if (!std::all_of(covered.cbegin(), covered.cend(), [](const bool value) { return value; }))
    {
        throw std::runtime_error(
            "Explicit fixed-q Sternheimer routes do not cover the full k grid");
    }

    std::vector<librpa_int::SternheimerFixedQKOrbit> orbits;
    orbits.reserve(orbits_by_representative.size());
    for (auto &entry : orbits_by_representative)
    {
        const int representative = entry.first;
        auto &orbit = entry.second;
        const auto self = std::find_if(orbit.members.cbegin(), orbit.members.cend(),
                                       [representative](const auto &member)
                                       { return member.ik_full == representative; });
        if (self == orbit.members.cend())
        {
            throw std::runtime_error(
                "Explicit fixed-q Sternheimer orbit does not contain its representative");
        }
        std::sort(orbit.members.begin(), orbit.members.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.ik_full < rhs.ik_full; });
        orbits.push_back(std::move(orbit));
    }
    return orbits;
}

std::vector<librpa_int::SternheimerQStarResponse> build_sternheimer_qstar_responses_from_routes(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_qpoints, const int representative_iq,
    const librpa_int::Vector3_Order<double> &q_representative,
    const librpa_int::ComplexMatrix &representative_response,
    const std::vector<SternheimerQStarRouteRecord> &routes, const int lmax, const double tolerance)
{
    if (representative_iq <= 0 || full_qpoints.empty() || routes.empty() || tolerance <= 0.0)
    {
        throw std::runtime_error(
            "Explicit Sternheimer q-star reconstruction requires a representative, full q grid, "
            "and routes");
    }
    std::vector<librpa_int::SternheimerQStarResponse> restored;
    restored.reserve(routes.size());
    bool contains_representative = false;
    for (const auto &record : routes)
    {
        if (record.representative_iq != representative_iq || record.member_iq <= 0 ||
            record.member_iq > static_cast<int>(full_qpoints.size()))
        {
            throw std::runtime_error(
                "Explicit Sternheimer q-star route has inconsistent q indices");
        }
        const auto &route = record.inverse_route;
        if (route.spatial_isym < 0 ||
            route.spatial_isym >= static_cast<int>(symmetry.rspace_operations.size()))
        {
            throw std::runtime_error(
                "Explicit Sternheimer q-star route has an invalid spatial operation");
        }
        const auto &member_q = full_qpoints.at(static_cast<std::size_t>(record.member_iq - 1));
        auto transformed = librpa_int::apply_space_group_rotation_to_kpoint(
            symmetry.rspace_operations.at(static_cast<std::size_t>(route.spatial_isym)), member_q);
        if (route.time_reversal)
        {
            transformed = transformed * -1.0;
        }
        librpa_int::FoldedKPoint folded;
        if (!librpa_int::try_fold_fractional_kpoint_to_target(transformed, q_representative,
                                                              tolerance, folded))
        {
            throw std::runtime_error(
                "Explicit Sternheimer q-star route does not map member q to its representative");
        }
        if (folded.fold_G.x != route.fold_G.x || folded.fold_G.y != route.fold_G.y ||
            folded.fold_G.z != route.fold_G.z)
        {
            throw std::runtime_error(
                "Explicit Sternheimer q-star route reciprocal fold disagrees with the q mapping");
        }
        if (record.member_iq == representative_iq)
        {
            contains_representative = true;
        }

        const auto member = librpa_int::build_symmetry_kspace_operation_member(
            symmetry, route.spatial_isym, route.time_reversal, member_q, q_representative, lmax);
        const auto *target = record.member_iq == representative_iq ? nullptr : &member_q;
        auto matrix = librpa_int::rotate_symmetry_kspace_matrix(symmetry, layouts, member,
                                                                representative_response, atom_nabf,
                                                                q_representative, false, target);
        restored.push_back({representative_iq - 1, static_cast<int>(restored.size()),
                            route.spatial_isym, route.time_reversal, member_q, std::move(matrix)});
    }
    if (!contains_representative)
    {
        throw std::runtime_error(
            "Explicit Sternheimer q-star orbit does not contain its representative");
    }
    return restored;
}

std::vector<SternheimerReconstructedResponse> reconstruct_sternheimer_partial_responses(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const std::vector<librpa_int::Vector3_Order<double>> &full_qpoints,
    const std::vector<SternheimerQPoint> &qpoints, const SternheimerPartialResponseGroups &groups,
    const int expected_nfreq, const bool use_rpa_gamma, const int lmax,
    const std::vector<SternheimerFixedQRouteRecord> *fixed_q_routes, const bool fixed_q_matrix_only,
    const std::vector<SternheimerQStarRouteRecord> *qstar_routes)
{
    if (expected_nfreq <= 0)
    {
        throw std::runtime_error("Sternheimer partial reconstruction requires positive nfreq");
    }
    if (qpoints.empty() || full_kpoints.empty() || full_qpoints.empty())
    {
        throw std::runtime_error(
            "Sternheimer partial reconstruction requires non-empty q, full-k, and full-q grids");
    }
    if (lmax < 0)
    {
        throw std::runtime_error("Sternheimer partial reconstruction requires ABF l-shell data");
    }

    const auto expected_atom_naux = ordered_atom_naux(atom_nabf);
    if (fixed_q_routes != nullptr)
    {
        std::set<int> qpoint_iq;
        for (const auto &point : qpoints)
        {
            qpoint_iq.insert(point.iq);
        }
        for (const auto &route : *fixed_q_routes)
        {
            if (qpoint_iq.count(route.iq) == 0)
            {
                throw std::runtime_error(
                    "Explicit fixed-q Sternheimer routes contain unexpected iq=" +
                    std::to_string(route.iq));
            }
        }
    }
    if (!fixed_q_matrix_only)
    {
        if (qstar_routes == nullptr)
        {
            std::size_t represented_full_q_count = 0;
            for (const auto &point : qpoints)
            {
                const auto &qstar = librpa_int::find_symmetry_kstar_for_kpoint(
                    symmetry.kstars, q_vector(point), "Sternheimer q-star coverage");
                represented_full_q_count += qstar.members.size();
            }
            if (represented_full_q_count != full_qpoints.size())
            {
                throw std::runtime_error(
                    "Sternheimer q-star representatives do not cover the full q grid");
            }
        }
        else
        {
            std::set<int> representative_iq;
            for (const auto &point : qpoints)
            {
                representative_iq.insert(point.iq);
            }
            std::vector<bool> covered(full_qpoints.size(), false);
            for (const auto &route : *qstar_routes)
            {
                if (representative_iq.count(route.representative_iq) == 0 || route.member_iq <= 0 ||
                    route.member_iq > static_cast<int>(full_qpoints.size()))
                {
                    throw std::runtime_error(
                        "Explicit Sternheimer q-star route has inconsistent representative or "
                        "member indices");
                }
                if (covered.at(static_cast<std::size_t>(route.member_iq - 1)))
                {
                    throw std::runtime_error(
                        "Explicit Sternheimer q-star routes contain a duplicate full-q member");
                }
                covered[static_cast<std::size_t>(route.member_iq - 1)] = true;
            }
            if (!std::all_of(covered.cbegin(), covered.cend(),
                             [](const bool value) { return value; }))
            {
                throw std::runtime_error(
                    "Explicit Sternheimer q-star routes do not cover the full q grid");
            }
        }
    }

    std::set<std::pair<int, int>> used_groups;
    std::vector<SternheimerReconstructedResponse> reconstructed;
    for (const auto &point : qpoints)
    {
        if (!use_rpa_gamma && is_rpa_gamma_point(point.q))
        {
            continue;
        }

        const auto q = q_vector(point);
        std::vector<SternheimerQStarRouteRecord> point_qstar_routes;
        if (qstar_routes != nullptr)
        {
            std::copy_if(qstar_routes->cbegin(), qstar_routes->cend(),
                         std::back_inserter(point_qstar_routes),
                         [&point](const auto &route)
                         { return route.representative_iq == point.iq; });
            if (!fixed_q_matrix_only && point_qstar_routes.empty())
            {
                throw std::runtime_error(
                    "Explicit Sternheimer q-star routes are missing representative iq=" +
                    std::to_string(point.iq));
            }
        }
        std::vector<SternheimerFixedQRouteRecord> point_routes;
        if (fixed_q_routes != nullptr)
        {
            std::copy_if(fixed_q_routes->cbegin(), fixed_q_routes->cend(),
                         std::back_inserter(point_routes),
                         [&point](const auto &route) { return route.iq == point.iq; });
            if (point_routes.empty())
            {
                throw std::runtime_error("Explicit fixed-q Sternheimer routes are missing iq=" +
                                         std::to_string(point.iq));
            }
        }
        const auto little_group =
            fixed_q_routes == nullptr
                ? librpa_int::build_sternheimer_fixed_q_little_group(symmetry.rspace_operations, q)
                : std::vector<librpa_int::SternheimerSymmetryRoute>();
        const auto orbits = fixed_q_routes == nullptr
                                ? librpa_int::build_sternheimer_fixed_q_k_orbits(
                                      symmetry.rspace_operations, full_kpoints, q)
                                : build_sternheimer_fixed_q_k_orbits_from_routes(
                                      symmetry.rspace_operations, full_kpoints, q, point_routes);
        std::set<std::pair<int, bool>> route_operations;
        for (const auto &route : point_routes)
        {
            route_operations.emplace(route.inverse_route.spatial_isym,
                                     route.inverse_route.time_reversal);
        }
        for (int ifreq = 1; ifreq <= expected_nfreq; ++ifreq)
        {
            const auto key = std::make_pair(point.iq, ifreq);
            const auto group_iter = groups.find(key);
            if (group_iter == groups.end())
            {
                throw std::runtime_error("Sternheimer partial responses are missing (iq, ifreq)=(" +
                                         std::to_string(point.iq) + ", " + std::to_string(ifreq) +
                                         ")");
            }
            used_groups.insert(key);
            const auto &group = group_iter->second;
            if (group.iq != point.iq || group.ifreq != ifreq)
            {
                throw std::runtime_error(
                    "Sternheimer partial response group key disagrees with its metadata");
            }
            if (group.atom_naux != expected_atom_naux)
            {
                throw std::runtime_error(
                    "Sternheimer partial response atom_naux does not match the active ABF basis");
            }
            if (!std::isfinite(group.omega) || !std::isfinite(group.weight) || group.weight <= 0.0)
            {
                throw std::runtime_error(
                    "Sternheimer partial response group has invalid frequency metadata");
            }

            std::vector<librpa_int::SternheimerFixedQKResponse> kresolved_responses;
            auto matrix = librpa_int::reconstruct_sternheimer_fixed_q_response(
                symmetry, layouts, atom_nabf, q, orbits, group.representatives, lmax,
                &kresolved_responses);
            std::vector<librpa_int::SternheimerQStarResponse> qstar_responses;
            double q_weight = 0.0;
            if (!fixed_q_matrix_only)
            {
                qstar_responses = qstar_routes == nullptr
                                      ? librpa_int::reconstruct_sternheimer_qstar_responses(
                                            symmetry, layouts, atom_nabf, q, matrix, lmax)
                                      : build_sternheimer_qstar_responses_from_routes(
                                            symmetry, layouts, atom_nabf, full_qpoints, point.iq, q,
                                            matrix, point_qstar_routes, lmax);
                q_weight = static_cast<double>(qstar_responses.size()) /
                           static_cast<double>(full_qpoints.size());
                const double q_weight_scale =
                    std::max({1.0, std::abs(point.weight), std::abs(q_weight)});
                if (std::abs(point.weight - q_weight) > 1.0e-12 * q_weight_scale)
                {
                    throw std::runtime_error(
                        "Sternheimer q-star weight disagrees with q-point manifest for iq=" +
                        std::to_string(point.iq));
                }
            }
            reconstructed.push_back(
                {point.iq, ifreq, group.omega, group.weight, q_weight,
                 static_cast<int>(full_kpoints.size()), static_cast<int>(orbits.size()),
                 fixed_q_routes == nullptr ? static_cast<int>(little_group.size())
                                           : static_cast<int>(route_operations.size()),
                 std::move(matrix), std::move(kresolved_responses), std::move(qstar_responses)});
        }
    }

    if (used_groups.size() != groups.size())
    {
        throw std::runtime_error(
            "Sternheimer partial manifest contains unexpected q/frequency response groups");
    }
    return reconstructed;
}

std::vector<SternheimerFixedQSymmetryDiagnostic> build_sternheimer_fixed_q_symmetry_diagnostics(
    const librpa_int::SymmetryContext &symmetry,
    const std::vector<librpa_int::SpeciesBasisLayout> &layouts,
    const std::map<librpa_int::atom_t, std::size_t> &atom_nabf,
    const std::vector<librpa_int::Vector3_Order<double>> &full_kpoints,
    const librpa_int::Vector3_Order<double> &q, const int lmax,
    const std::vector<SternheimerFixedQRouteRecord> *fixed_q_routes)
{
    const auto orbits = fixed_q_routes == nullptr
                            ? librpa_int::build_sternheimer_fixed_q_k_orbits(
                                  symmetry.rspace_operations, full_kpoints, q)
                            : build_sternheimer_fixed_q_k_orbits_from_routes(
                                  symmetry.rspace_operations, full_kpoints, q, *fixed_q_routes);
    std::vector<SternheimerFixedQSymmetryDiagnostic> diagnostics;
    for (const auto &orbit : orbits)
    {
        for (const auto &member : orbit.members)
        {
            const auto operation_member = librpa_int::build_symmetry_kspace_operation_member(
                symmetry, member.inverse_route.spatial_isym, member.inverse_route.time_reversal, q,
                q, lmax);
            diagnostics.push_back({
                orbit.representative_ik_full,
                member.ik_full,
                member.inverse_route,
                librpa_int::build_symmetry_kspace_operator_transform_matrix(
                    symmetry, layouts, operation_member, atom_nabf, q),
            });
        }
    }
    return diagnostics;
}

void write_sternheimer_fixed_q_symmetry_diagnostics(
    const std::string &path, const int iq,
    const std::vector<SternheimerFixedQSymmetryDiagnostic> &diagnostics)
{
    if (iq <= 0)
    {
        throw std::runtime_error("Fixed-q symmetry diagnostic requires a positive q index");
    }
    std::ofstream output(path);
    if (!output)
    {
        throw std::runtime_error("Cannot open fixed-q symmetry diagnostic file " + path);
    }
    output << std::scientific << std::setprecision(17);
    output << "# route iq representative_ik member_ik spatial_isym time_reversal fold_Gx fold_Gy "
              "fold_Gz\n";
    output << "# matrix nrow ncol followed by: row column real imag\n";
    for (const auto &diagnostic : diagnostics)
    {
        const auto &route = diagnostic.inverse_route;
        output << "route " << iq << ' ' << diagnostic.representative_ik_full << ' '
               << diagnostic.member_ik_full << ' ' << route.spatial_isym << ' '
               << static_cast<int>(route.time_reversal) << ' ' << route.fold_G.x << ' '
               << route.fold_G.y << ' ' << route.fold_G.z << '\n';
        output << "matrix " << diagnostic.transform.nr << ' ' << diagnostic.transform.nc << '\n';
        for (int row = 0; row != diagnostic.transform.nr; ++row)
        {
            for (int column = 0; column != diagnostic.transform.nc; ++column)
            {
                const auto value = diagnostic.transform(row, column);
                output << row << ' ' << column << ' ' << value.real() << ' ' << value.imag()
                       << '\n';
            }
        }
    }
    if (!output.good())
    {
        throw std::runtime_error("Failed to write fixed-q symmetry diagnostic file " + path);
    }
}

SternheimerQStarRpaAudit compute_sternheimer_qstar_rpa_frequency(
    const SternheimerReconstructedResponse &response,
    const std::vector<librpa_int::SternheimerQStarResponse> &coulomb_qstar,
    const double sqrt_coulomb_threshold, const librpa_int::SternheimerRpaHeadwingInput *headwing,
    const double invariance_tolerance)
{
    if (response.qstar_responses.empty() || response.qstar_responses.size() != coulomb_qstar.size())
    {
        throw std::runtime_error("Sternheimer q-star response and Coulomb member counts disagree");
    }
    if (invariance_tolerance < 0.0)
    {
        throw std::runtime_error("Sternheimer q-star invariance tolerance must be non-negative");
    }

    SternheimerQStarRpaAudit audit;
    audit.qstar_size = static_cast<int>(response.qstar_responses.size());
    bool initialized = false;
    std::set<std::pair<int, int>> used_coulomb_members;
    for (const auto &response_member : response.qstar_responses)
    {
        const auto key = std::make_pair(response_member.star_index, response_member.member_index);
        const auto coulomb_iter =
            std::find_if(coulomb_qstar.begin(), coulomb_qstar.end(),
                         [&key](const auto &member)
                         { return std::make_pair(member.star_index, member.member_index) == key; });
        if (coulomb_iter == coulomb_qstar.end() || !used_coulomb_members.insert(key).second ||
            !librpa_int::same_fractional_kpoint(response_member.q, coulomb_iter->q, 1.0e-8))
        {
            throw std::runtime_error(
                "Sternheimer q-star response and Coulomb members cannot be paired");
        }

        const auto member_result =
            headwing == nullptr
                ? librpa_int::compute_sternheimer_rpa_frequency(
                      coulomb_iter->matrix, response_member.matrix, response.ifreq, response.omega,
                      response.weight, response.q_weight, sqrt_coulomb_threshold)
                : librpa_int::compute_sternheimer_rpa_frequency_headwing(
                      coulomb_iter->matrix, response_member.matrix, *headwing, response.ifreq,
                      response.omega, response.weight, response.q_weight, sqrt_coulomb_threshold);
        if (!initialized)
        {
            audit.result = member_result;
            initialized = true;
            continue;
        }

        const double difference = std::abs(member_result.integrand - audit.result.integrand);
        audit.max_integrand_difference = std::max(audit.max_integrand_difference, difference);
        const double scale =
            std::max({1.0, std::abs(member_result.integrand), std::abs(audit.result.integrand)});
        if (difference > invariance_tolerance * scale)
        {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "Sternheimer q-star trace-log invariance check failed: iq=" << response.iq
                    << ", ifreq=" << response.ifreq << ", reference_member=("
                    << response.qstar_responses.front().star_index << ','
                    << response.qstar_responses.front().member_index << ")"
                    << ", failing_member=(" << response_member.star_index << ','
                    << response_member.member_index << ")"
                    << ", q=(" << response_member.q.x << ',' << response_member.q.y << ','
                    << response_member.q.z << ")"
                    << ", reference_integrand=" << audit.result.integrand
                    << ", member_integrand=" << member_result.integrand
                    << ", difference=" << difference
                    << ", tolerance_scale=" << invariance_tolerance * scale;
            throw std::runtime_error(message.str());
        }
    }
    return audit;
}

}  // namespace driver
