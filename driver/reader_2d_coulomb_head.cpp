#include "reader_2d_coulomb_head.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace
{

double parse_finite_double(const std::string &token, const std::string &key,
                           const std::string &source_name)
{
    std::size_t consumed = 0;
    double value = 0.0;
    try
    {
        value = std::stod(token, &consumed);
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(source_name + ": invalid " + key);
    }
    if (consumed != token.size() || !std::isfinite(value))
        throw std::runtime_error(source_name + ": invalid " + key);
    return value;
}

}  // namespace

Strict2dCoulombHeadMetadata read_strict_2d_coulomb_head_metadata(std::istream &input,
                                                                 const std::string &source_name)
{
    Strict2dCoulombHeadMetadata metadata;
    bool have_version = false;
    bool have_area = false;
    bool have_multipole_norm = false;

    std::string line;
    while (std::getline(input, line))
    {
        const auto comment = line.find_first_of("#!");
        if (comment != std::string::npos) line.erase(comment);
        std::replace(line.begin(), line.end(), '=', ' ');

        std::istringstream fields(line);
        std::string key;
        std::string value;
        if (!(fields >> key >> value)) continue;

        if (key == "version")
        {
            const double parsed = parse_finite_double(value, key, source_name);
            metadata.version = static_cast<int>(parsed);
            if (parsed != metadata.version)
                throw std::runtime_error(source_name + ": invalid version");
            have_version = true;
        }
        else if (key == "area_parallel_bohr2")
        {
            metadata.inplane_area_bohr2 = parse_finite_double(value, key, source_name);
            have_area = true;
        }
        else if (key == "multipole_norm_squared")
        {
            metadata.auxiliary_monopole_norm_squared = parse_finite_double(value, key, source_name);
            have_multipole_norm = true;
        }
    }

    if (!have_version || metadata.version != 1)
        throw std::runtime_error(source_name + ": unsupported or missing metadata version");
    if (!have_area || !(metadata.inplane_area_bohr2 > 0.0))
        throw std::runtime_error(source_name + ": missing positive area_parallel_bohr2");
    if (!have_multipole_norm || !(metadata.auxiliary_monopole_norm_squared > 0.0))
        throw std::runtime_error(source_name + ": missing positive multipole_norm_squared");
    return metadata;
}

Strict2dCoulombHeadMetadata read_strict_2d_coulomb_head_metadata(const std::string &file_path)
{
    std::ifstream input(file_path);
    if (!input.good())
        throw std::runtime_error("Failed to open strict 2D Coulomb metadata " + file_path);
    return read_strict_2d_coulomb_head_metadata(input, file_path);
}
