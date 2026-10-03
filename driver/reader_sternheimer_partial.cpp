#include "reader_sternheimer_partial.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

#include "../src/io/fs.h"

namespace driver
{
namespace
{

std::string strip_comment(const std::string &line)
{
    const auto hash = line.find('#');
    const auto bang = line.find('!');
    return line.substr(0, std::min(hash, bang));
}

std::string resolve_response_path(const std::string &manifest_path,
                                  const std::string &response_file)
{
    const auto unresolved =
        librpa_int::is_absolute_path(response_file)
            ? response_file
            : librpa_int::join_path(librpa_int::parent_path(manifest_path), response_file);
    librpa_int::require_readable_file(unresolved);
    return std::filesystem::weakly_canonical(unresolved).string();
}

}  // namespace

void validate_sternheimer_partial_task_contract(const std::string &qpoint_manifest,
                                                const std::string &partial_manifest,
                                                const std::string &route_manifest)
{
    if (!partial_manifest.empty() && qpoint_manifest.empty())
    {
        throw std::runtime_error("fn_sternheimer_partial_manifest requires fn_sternheimer_qpoints");
    }
    if (!route_manifest.empty() && partial_manifest.empty())
    {
        throw std::runtime_error(
            "fn_sternheimer_symmetry_routes requires fn_sternheimer_partial_manifest");
    }
}

std::vector<SternheimerPartialResponse> read_sternheimer_partial_manifest(const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path);
    std::vector<SternheimerPartialResponse> records;
    std::set<std::tuple<int, int, int>> seen_keys;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        std::istringstream row(strip_comment(line));
        row >> std::ws;
        if (row.eof())
        {
            continue;
        }

        SternheimerPartialResponse record;
        std::string response_file;
        if (!(row >> record.iq >> record.ik_full >> record.ifreq >> response_file))
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " must contain iq ik_full ifreq response_file");
        }
        std::string extra;
        if (row >> extra)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " contains extra fields");
        }
        if (record.iq <= 0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a positive one-based iq");
        }
        if (record.ik_full < 0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a non-negative zero-based ik_full");
        }
        if (record.ifreq <= 0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a positive one-based ifreq");
        }

        const auto key = std::make_tuple(record.iq, record.ik_full, record.ifreq);
        if (!seen_keys.insert(key).second)
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number) +
                ": duplicate (iq, ik_full, ifreq)=(" + std::to_string(record.iq) + ", " +
                std::to_string(record.ik_full) + ", " + std::to_string(record.ifreq) + ")");
        }
        record.response_path = resolve_response_path(path, response_file);
        records.push_back(std::move(record));
    }

    if (records.empty())
    {
        throw std::runtime_error(path + ": Sternheimer partial manifest is empty");
    }
    return records;
}

std::vector<SternheimerFullKPoint> read_sternheimer_full_kpoint_manifest(
    const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path);
    std::map<int, SternheimerFullKPoint> points_by_index;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line))
    {
        ++line_number;
        std::istringstream row(strip_comment(line));
        row >> std::ws;
        if (row.eof())
        {
            continue;
        }

        SternheimerFullKPoint point;
        if (!(row >> point.ik_full >> point.k[0] >> point.k[1] >> point.k[2]))
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " must contain ik_full kx ky kz");
        }
        std::string extra;
        if (row >> extra)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " contains extra fields");
        }
        if (point.ik_full < 0 || !std::isfinite(point.k[0]) || !std::isfinite(point.k[1])
            || !std::isfinite(point.k[2]))
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " contains an invalid full-k-point record");
        }
        if (!points_by_index.emplace(point.ik_full, point).second)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " duplicates ik_full=" + std::to_string(point.ik_full));
        }
    }

    if (points_by_index.empty())
    {
        throw std::runtime_error(path + ": Sternheimer full-k-point manifest is empty");
    }
    std::vector<SternheimerFullKPoint> points;
    points.reserve(points_by_index.size());
    for (int ik = 0; ik != static_cast<int>(points_by_index.size()); ++ik)
    {
        const auto iter = points_by_index.find(ik);
        if (iter == points_by_index.end())
        {
            throw std::runtime_error(path + ": full-k-point indices are not contiguous from zero");
        }
        points.push_back(iter->second);
    }
    return points;
}

std::vector<SternheimerFixedQRouteRecord> read_sternheimer_fixed_q_route_manifest(
    const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path);
    std::vector<SternheimerFixedQRouteRecord> records;
    std::set<std::pair<int, int>> seen_members;
    std::string line;
    int line_number = 0;
    bool read_version = false;
    while (std::getline(input, line))
    {
        ++line_number;
        std::istringstream row(strip_comment(line));
        row >> std::ws;
        if (row.eof())
        {
            continue;
        }
        if (!read_version)
        {
            std::string keyword;
            int version = 0;
            std::string extra;
            if (!(row >> keyword >> version) || keyword != "version" || (row >> extra))
            {
                throw std::runtime_error(
                    path + ": first data row must contain version 1");
            }
            if (version != 1)
            {
                throw std::runtime_error(
                    path + ": unsupported Sternheimer fixed-q route version "
                    + std::to_string(version));
            }
            read_version = true;
            continue;
        }

        SternheimerFixedQRouteRecord record;
        int time_reversal = -1;
        int fold_x = 0;
        int fold_y = 0;
        int fold_z = 0;
        if (!(row >> record.iq >> record.representative_ik_full >> record.member_ik_full
                  >> record.inverse_route.spatial_isym >> time_reversal
                  >> fold_x >> fold_y >> fold_z))
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number)
                + " must contain iq representative_ik member_ik spatial_isym "
                  "time_reversal fold_Gx fold_Gy fold_Gz");
        }
        std::string extra;
        if (row >> extra)
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number) + " contains extra fields");
        }
        if (record.iq <= 0 || record.representative_ik_full < 0
            || record.member_ik_full < 0 || record.inverse_route.spatial_isym < 0
            || (time_reversal != 0 && time_reversal != 1))
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number)
                + " contains an invalid fixed-q route");
        }
        const auto key = std::make_pair(record.iq, record.member_ik_full);
        if (!seen_members.insert(key).second)
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number)
                + ": duplicate (iq, member_ik)=(" + std::to_string(record.iq) + ", "
                + std::to_string(record.member_ik_full) + ")");
        }
        record.inverse_route.time_reversal = time_reversal != 0;
        record.inverse_route.fold_G = {fold_x, fold_y, fold_z};
        records.push_back(std::move(record));
    }
    if (!read_version || records.empty())
    {
        throw std::runtime_error(path + ": Sternheimer fixed-q route manifest is empty");
    }
    std::sort(records.begin(), records.end(), [](const auto &lhs, const auto &rhs) {
        return std::tie(lhs.iq, lhs.member_ik_full, lhs.representative_ik_full)
               < std::tie(rhs.iq, rhs.member_ik_full, rhs.representative_ik_full);
    });
    return records;
}

std::vector<SternheimerQStarRouteRecord> read_sternheimer_qstar_route_manifest(
    const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path);
    std::vector<SternheimerQStarRouteRecord> records;
    std::set<int> seen_members;
    std::string line;
    int line_number = 0;
    bool read_version = false;
    while (std::getline(input, line))
    {
        ++line_number;
        std::istringstream row(strip_comment(line));
        row >> std::ws;
        if (row.eof())
        {
            continue;
        }
        if (!read_version)
        {
            std::string keyword;
            int version = 0;
            std::string extra;
            if (!(row >> keyword >> version) || keyword != "version" || (row >> extra))
            {
                throw std::runtime_error(path + ": first data row must contain version 1");
            }
            if (version != 1)
            {
                throw std::runtime_error(
                    path + ": unsupported Sternheimer q-star route version "
                    + std::to_string(version));
            }
            read_version = true;
            continue;
        }

        SternheimerQStarRouteRecord record;
        int time_reversal = -1;
        int fold_x = 0;
        int fold_y = 0;
        int fold_z = 0;
        if (!(row >> record.representative_iq >> record.member_iq
                  >> record.inverse_route.spatial_isym >> time_reversal
                  >> fold_x >> fold_y >> fold_z))
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number)
                + " must contain representative_iq member_iq spatial_isym "
                  "time_reversal fold_Gx fold_Gy fold_Gz");
        }
        std::string extra;
        if (row >> extra)
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number) + " contains extra fields");
        }
        if (record.representative_iq <= 0 || record.member_iq <= 0
            || record.inverse_route.spatial_isym < 0
            || (time_reversal != 0 && time_reversal != 1))
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number)
                + " contains an invalid q-star route");
        }
        if (!seen_members.insert(record.member_iq).second)
        {
            throw std::runtime_error(
                path + ": line " + std::to_string(line_number)
                + ": duplicate member_iq=" + std::to_string(record.member_iq));
        }
        record.inverse_route.time_reversal = time_reversal != 0;
        record.inverse_route.fold_G = {fold_x, fold_y, fold_z};
        records.push_back(std::move(record));
    }
    if (!read_version || records.empty())
    {
        throw std::runtime_error(path + ": Sternheimer q-star route manifest is empty");
    }
    std::sort(records.begin(), records.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.member_iq < rhs.member_iq;
    });
    return records;
}

SternheimerPartialResponseGroups read_sternheimer_partial_response_groups(
    const std::vector<SternheimerPartialResponse> &records)
{
    if (records.empty())
    {
        throw std::runtime_error("Cannot group an empty Sternheimer partial-response list");
    }

    SternheimerPartialResponseGroups groups;
    for (const auto &record : records)
    {
        auto response = read_sternheimer_chi0_v1_matrix_file(record.response_path);
        if (response.iq != record.iq)
        {
            throw std::runtime_error(record.response_path +
                                     ": binary iq=" + std::to_string(response.iq) +
                                     " does not match manifest iq=" + std::to_string(record.iq));
        }
        if (response.ifreq != record.ifreq)
        {
            throw std::runtime_error(
                record.response_path + ": binary ifreq=" + std::to_string(response.ifreq) +
                " does not match manifest ifreq=" + std::to_string(record.ifreq));
        }
        if (!std::isfinite(response.omega) || !std::isfinite(response.weight) ||
            response.weight <= 0.0)
        {
            throw std::runtime_error(record.response_path +
                                     ": invalid Sternheimer frequency metadata");
        }

        const auto key = std::make_pair(record.iq, record.ifreq);
        auto [group_iter, inserted] = groups.try_emplace(key);
        auto &group = group_iter->second;
        if (inserted)
        {
            group.iq = response.iq;
            group.ifreq = response.ifreq;
            group.omega = response.omega;
            group.weight = response.weight;
            group.atom_naux = response.atom_naux;
        }
        else
        {
            const auto close = [](const double lhs, const double rhs) {
                return std::abs(lhs - rhs) <= 1e-12 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
            };
            if (!close(group.omega, response.omega))
            {
                throw std::runtime_error(record.response_path +
                                         ": inconsistent omega within (iq, ifreq) group");
            }
            if (!close(group.weight, response.weight))
            {
                throw std::runtime_error(
                    record.response_path +
                    ": inconsistent frequency weight within (iq, ifreq) group");
            }
            if (group.atom_naux != response.atom_naux)
            {
                throw std::runtime_error(record.response_path +
                                         ": inconsistent atom_naux within (iq, ifreq) group");
            }
        }

        if (!group.representatives.emplace(record.ik_full, std::move(response.matrix)).second)
        {
            throw std::runtime_error("Duplicate Sternheimer representative response ik_full=" +
                                     std::to_string(record.ik_full) +
                                     " for iq=" + std::to_string(record.iq) +
                                     ", ifreq=" + std::to_string(record.ifreq));
        }
    }
    return groups;
}

}  // namespace driver
