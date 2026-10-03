#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include "../reader_2d_coulomb_head.h"
#include "../../src/core/dielecmodel.h"

namespace
{

void test_abacus_reader_v1_metadata_is_parsed_without_equals_signs()
{
    std::istringstream input{
        "# ABACUS reader-v1 strict 2D Coulomb head normalization\n"
        "version 1\n"
        "area_parallel_bohr2 19.390653825130212\n"
        "multipole_norm_squared 2205.0673846924301\n"};

    const auto metadata = read_strict_2d_coulomb_head_metadata(input, "memory");
    assert(metadata.version == 1);
    assert(std::abs(metadata.inplane_area_bohr2 - 19.390653825130212) < 1e-14);
    assert(std::abs(metadata.auxiliary_monopole_norm_squared - 2205.0673846924301) < 1e-12);

    librpa_int::PeriodicBoundaryData pbc;
    pbc.set_latvec({metadata.inplane_area_bohr2, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 30.0});
    const auto normalization = librpa_int::strict_2d_coulomb_head_normalization(
        pbc, metadata.auxiliary_monopole_norm_squared);
    assert(std::abs(normalization.auxiliary_head_coefficient - 8978.8175111265446) < 1e-10);
    assert(std::abs(normalization.pw_to_auxiliary_scale - 37.802423070695596) < 1e-12);
}

void test_canonical_metadata_is_parsed_with_equals_signs()
{
    std::istringstream input{
        "version = 1\n"
        "area_parallel_bohr2 = 31.347506240476704\n"
        "multipole_norm_squared = 196.9576792074629\n"};

    const auto metadata = read_strict_2d_coulomb_head_metadata(input, "memory");
    assert(metadata.version == 1);
    assert(std::abs(metadata.inplane_area_bohr2 - 31.347506240476704) < 1e-14);
    assert(std::abs(metadata.auxiliary_monopole_norm_squared - 196.9576792074629) < 1e-13);
}

void test_missing_multipole_norm_is_rejected()
{
    std::istringstream input{"version 1\narea_parallel_bohr2 10.0\n"};
    bool rejected = false;
    try
    {
        (void)read_strict_2d_coulomb_head_metadata(input, "memory");
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    assert(rejected);
}

}  // namespace

int main()
{
    test_abacus_reader_v1_metadata_is_parsed_without_equals_signs();
    test_canonical_metadata_is_parsed_with_equals_signs();
    test_missing_multipole_norm_is_rejected();
    return 0;
}
