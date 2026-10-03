#pragma once

#include <array>
#include <string>
#include <vector>

namespace driver
{

struct SternheimerQPoint
{
    int iq = 0;
    // Fractional reciprocal coordinates used for audit output; iq selects the v1 files.
    std::array<double, 3> q = {0.0, 0.0, 0.0};
    double weight = 0.0;
};

std::vector<SternheimerQPoint> read_sternheimer_qpoint_manifest(const std::string &path);

void validate_sternheimer_gamma_contract(const std::vector<SternheimerQPoint> &qpoints,
                                         bool use_rpa_gamma);

void validate_sternheimer_qpoint_input_files(const std::vector<SternheimerQPoint> &qpoints,
                                             const std::string &dir_path,
                                             const std::string &coulomb_prefix,
                                             const std::string &response_prefix, int expected_nfreq,
                                             bool use_rpa_gamma = true);

void validate_sternheimer_partial_qpoint_input_files(
    const std::vector<SternheimerQPoint> &qpoints,
    const std::string &dir_path,
    const std::string &coulomb_prefix,
    bool use_rpa_gamma = true);

}  // namespace driver
