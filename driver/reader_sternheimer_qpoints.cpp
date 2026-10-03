#include "reader_sternheimer_qpoints.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../src/io/fs.h"
#include "reader_sternheimer.h"
#include "rpa_qsum.h"

namespace driver
{
namespace
{

std::string strip_comment(const std::string &line)
{
    const auto hash = line.find('#');
    const auto bang = line.find('!');
    const auto comment = std::min(hash, bang);
    return line.substr(0, comment);
}

}  // namespace

std::vector<SternheimerQPoint> read_sternheimer_qpoint_manifest(const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path);
    std::vector<SternheimerQPoint> qpoints;
    std::set<int> seen_iq;
    double weight_sum = 0.0;
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

        SternheimerQPoint point;
        if (!(row >> point.iq >> point.q[0] >> point.q[1] >> point.q[2] >> point.weight))
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " must contain iq qx qy qz qweight");
        }
        std::string extra;
        if (row >> extra)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " contains extra fields");
        }
        if (point.iq <= 0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a positive one-based iq");
        }
        if (!seen_iq.insert(point.iq).second)
        {
            throw std::runtime_error(path + ": duplicate iq=" + std::to_string(point.iq));
        }
        if (!std::isfinite(point.q[0]) || !std::isfinite(point.q[1]) || !std::isfinite(point.q[2]))
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " contains a non-finite q coordinate");
        }
        if (!std::isfinite(point.weight) || point.weight <= 0.0)
        {
            throw std::runtime_error(path + ": line " + std::to_string(line_number) +
                                     " requires a positive q weight");
        }
        weight_sum += point.weight;
        qpoints.push_back(point);
    }

    if (qpoints.empty())
    {
        throw std::runtime_error(path + ": Sternheimer q-point manifest is empty");
    }
    if (std::abs(weight_sum - 1.0) > 1.0e-10)
    {
        throw std::runtime_error(
            path + ": q-point weights must sum to 1; actual sum=" + std::to_string(weight_sum));
    }
    return qpoints;
}

void validate_sternheimer_gamma_contract(const std::vector<SternheimerQPoint> &qpoints,
                                         const bool use_rpa_gamma)
{
    if (use_rpa_gamma)
    {
        return;
    }
    const auto gamma_count =
        std::count_if(qpoints.begin(), qpoints.end(),
                      [](const auto &point) { return is_rpa_gamma_point(point.q); });
    if (gamma_count != 1)
    {
        throw std::runtime_error(
            "use_rpa_gamma=false requires exactly one Gamma row in the Sternheimer q manifest");
    }
}

void validate_sternheimer_qpoint_input_files(const std::vector<SternheimerQPoint> &qpoints,
                                             const std::string &dir_path,
                                             const std::string &coulomb_prefix,
                                             const std::string &response_prefix,
                                             const int expected_nfreq, const bool use_rpa_gamma)
{
    validate_sternheimer_gamma_contract(qpoints, use_rpa_gamma);
    if (expected_nfreq <= 0)
    {
        throw std::runtime_error("Sternheimer nfreq must be positive");
    }
    for (const auto &point : qpoints)
    {
        if (!use_rpa_gamma && is_rpa_gamma_point(point.q))
        {
            continue;
        }
        validate_coulomb_v1_full_matrix_file(dir_path, coulomb_prefix, point.iq);
        validate_sternheimer_chi0_v1_files(dir_path, response_prefix, point.iq, expected_nfreq);
    }
}

void validate_sternheimer_partial_qpoint_input_files(
    const std::vector<SternheimerQPoint> &qpoints,
    const std::string &dir_path,
    const std::string &coulomb_prefix,
    const bool use_rpa_gamma)
{
    validate_sternheimer_gamma_contract(qpoints, use_rpa_gamma);
    for (const auto &point : qpoints)
    {
        if (!use_rpa_gamma && is_rpa_gamma_point(point.q))
        {
            continue;
        }
        validate_coulomb_v1_full_matrix_file(dir_path, coulomb_prefix, point.iq);
    }
}

}  // namespace driver
