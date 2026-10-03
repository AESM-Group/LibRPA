#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "reader_sternheimer.h"
#include "../src/core/sternheimer_symmetry.h"

namespace driver
{

struct SternheimerPartialResponse
{
    int iq = 0;
    int ik_full = -1;
    int ifreq = 0;
    std::string response_path;
};

struct SternheimerPartialResponseGroup
{
    int iq = 0;
    int ifreq = 0;
    double omega = 0.0;
    double weight = 0.0;
    std::vector<int> atom_naux;
    std::map<int, librpa_int::ComplexMatrix> representatives;
};

using SternheimerPartialResponseGroups =
    std::map<std::pair<int, int>, SternheimerPartialResponseGroup>;

struct SternheimerFullKPoint
{
    int ik_full = -1;
    std::array<double, 3> k{};
};

struct SternheimerFixedQRouteRecord
{
    int iq = 0;
    int representative_ik_full = -1;
    int member_ik_full = -1;
    librpa_int::SternheimerSymmetryRoute inverse_route;
};

struct SternheimerQStarRouteRecord
{
    int representative_iq = 0;
    int member_iq = 0;
    librpa_int::SternheimerSymmetryRoute inverse_route;
};

void validate_sternheimer_partial_task_contract(const std::string &qpoint_manifest,
                                                const std::string &partial_manifest,
                                                const std::string &route_manifest = "");

std::vector<SternheimerPartialResponse> read_sternheimer_partial_manifest(const std::string &path);

std::vector<SternheimerFullKPoint> read_sternheimer_full_kpoint_manifest(
    const std::string &path);

std::vector<SternheimerFixedQRouteRecord> read_sternheimer_fixed_q_route_manifest(
    const std::string &path);

std::vector<SternheimerQStarRouteRecord> read_sternheimer_qstar_route_manifest(
    const std::string &path);

SternheimerPartialResponseGroups read_sternheimer_partial_response_groups(
    const std::vector<SternheimerPartialResponse> &records);

}  // namespace driver
