#pragma once

#include <istream>
#include <string>

struct Strict2dCoulombHeadMetadata
{
    int version = 0;
    double inplane_area_bohr2 = 0.0;
    double auxiliary_monopole_norm_squared = 0.0;
};

Strict2dCoulombHeadMetadata read_strict_2d_coulomb_head_metadata(std::istream &input,
                                                                 const std::string &source_name);
Strict2dCoulombHeadMetadata read_strict_2d_coulomb_head_metadata(const std::string &file_path);
