#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <functional>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

#include "../core/pbc.h"
#include "../core/sternheimer_symmetry.h"
#include "../core/symmetry_context.h"
#include "../math/symmetry.h"

using namespace librpa_int;

namespace
{

void require_throws(const std::function<void()> &operation, const std::string &fragment)
{
    try
    {
        operation();
    }
    catch (const std::exception &error)
    {
        if (std::string(error.what()).find(fragment) != std::string::npos)
        {
            return;
        }
        std::cerr << "unexpected error: " << error.what() << std::endl;
        std::abort();
    }
    std::cerr << "expected an exception containing: " << fragment << std::endl;
    std::abort();
}

void assert_matrix_close(const ComplexMatrix &actual,
                         const ComplexMatrix &expected,
                         const double tolerance = 1e-12)
{
    assert(actual.nr == expected.nr);
    assert(actual.nc == expected.nc);
    for (int row = 0; row != actual.nr; ++row)
    {
        for (int col = 0; col != actual.nc; ++col)
        {
            assert(std::abs(actual(row, col) - expected(row, col)) < tolerance);
        }
    }
}

ComplexMatrix make_complex_hermitian_matrix()
{
    ComplexMatrix matrix(2, 2);
    matrix(0, 0) = {1.0, 0.0};
    matrix(0, 1) = {2.0, 3.0};
    matrix(1, 0) = {2.0, -3.0};
    matrix(1, 1) = {4.0, 0.0};
    return matrix;
}

SpeciesBasisLayout make_s_layout()
{
    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    return layout;
}

SymmetryContext make_two_atom_swap_context()
{
    SymmetryContext context;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    context.set_crystal_structure(lattice,
                                  lattice,
                                  {{0, 0}, {1, 0}},
                                  {{0, {0.25, 0.0, 0.0}},
                                   {1, {0.75, 0.0, 0.0}}});
    SpaceGroupSymOp mirror;
    mirror.rotation = Matrix3(-1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0);
    context.set_rspace_operations({SpaceGroupSymOp::IDENTITY, mirror});
    context.basis_convention = {-1,
                                0,
                                LIBRPA_ANGULAR_ORDER_NATURAL,
                                LIBRPA_RSH_COEFF_1_M,
                                LIBRPA_RSH_COEFF_1_M};
    return context;
}

SpaceGroupSymOps make_fcc_fractional_operations()
{
    const Matrix3 lattice(0.0, 0.5, 0.5,
                          0.5, 0.0, 0.5,
                          0.5, 0.5, 0.0);
    const std::array<std::array<int, 3>, 6> permutations{{
        {{0, 1, 2}},
        {{0, 2, 1}},
        {{1, 0, 2}},
        {{1, 2, 0}},
        {{2, 0, 1}},
        {{2, 1, 0}},
    }};
    const std::array<int, 2> signs{{-1, 1}};
    SpaceGroupSymOps operations;
    for (const auto &permutation : permutations)
    {
        for (const int sx : signs)
        {
            for (const int sy : signs)
            {
                for (const int sz : signs)
                {
                    const std::array<int, 3> sign{{sx, sy, sz}};
                    std::array<int, 9> values{};
                    for (int col = 0; col != 3; ++col)
                    {
                        values[3 * permutation[col] + col] = sign[col];
                    }
                    const Matrix3 col_cartesian(values[0], values[1], values[2],
                                                values[3], values[4], values[5],
                                                values[6], values[7], values[8]);
                    SpaceGroupSymOp operation;
                    operation.rotation =
                        lattice * col_cartesian.Transpose() * lattice.Inverse();
                    operation.use_row_convention = true;
                    operations.push_back(operation);
                }
            }
        }
    }
    assert(operations.size() == 48);
    return operations;
}

Vector3_Order<double> apply_k_route(const SpaceGroupSymOps &operations,
                                    const SternheimerSymmetryRoute &route,
                                    const Vector3_Order<double> &k)
{
    auto transformed = apply_space_group_rotation_to_kpoint(
        operations.at(static_cast<std::size_t>(route.spatial_isym)), k);
    return route.time_reversal ? transformed * -1.0 : transformed;
}

void assert_complete_disjoint_coverage(const std::vector<SternheimerFixedQKOrbit> &orbits,
                                       const SpaceGroupSymOps &operations,
                                       const std::vector<Vector3_Order<double>> &full_kpoints,
                                       const Vector3_Order<double> &q)
{
    std::vector<int> visits(full_kpoints.size(), 0);
    for (const auto &operation : build_sternheimer_fixed_q_little_group(operations, q))
    {
        const auto mapped_q = apply_k_route(operations, operation, q);
        assert(same_fractional_kpoint(mapped_q, q, 1e-10));
    }

    for (const auto &orbit : orbits)
    {
        assert(!orbit.members.empty());
        assert(orbit.representative_ik_full >= 0);
        assert(orbit.representative_ik_full < static_cast<int>(full_kpoints.size()));
        assert(orbit.members.front().ik_full == orbit.representative_ik_full);
        for (const auto &member : orbit.members)
        {
            assert(member.ik_full >= 0);
            assert(member.ik_full < static_cast<int>(full_kpoints.size()));
            ++visits.at(static_cast<std::size_t>(member.ik_full));

            const auto mapped_member = apply_k_route(
                operations, member.inverse_route,
                full_kpoints.at(static_cast<std::size_t>(member.ik_full)));
            assert(same_fractional_kpoint(
                mapped_member,
                full_kpoints.at(static_cast<std::size_t>(orbit.representative_ik_full)),
                1e-10));
        }
    }
    assert(std::all_of(visits.begin(), visits.end(), [](const int count) { return count == 1; }));
}

void test_gamma_uses_full_space_group_and_covers_grid()
{
    const auto operations = make_fcc_fractional_operations();
    const auto full_kpoints = build_uniform_kmesh_frac({4, 4, 4});
    const Vector3_Order<double> q{0.0, 0.0, 0.0};
    const auto little_group = build_sternheimer_fixed_q_little_group(operations, q);
    assert(little_group.size() == 2 * operations.size());

    const auto orbits = build_sternheimer_fixed_q_k_orbits(operations, full_kpoints, q);
    assert(orbits.size() == 8);
    assert_complete_disjoint_coverage(orbits, operations, full_kpoints, q);
}

void test_generic_q_keeps_only_identity_routes()
{
    const SpaceGroupSymOps operations{SpaceGroupSymOp::IDENTITY,
                                      SpaceGroupSymOp::INVERSE,
                                      SpaceGroupSymOp::C41_Z};
    const auto full_kpoints = build_uniform_kmesh_frac({4, 4, 4});
    const Vector3_Order<double> q{0.25, 0.5, 0.0};
    const auto little_group = build_sternheimer_fixed_q_little_group(operations, q);
    assert(!little_group.empty());
    const auto orbits = build_sternheimer_fixed_q_k_orbits(operations, full_kpoints, q);
    assert_complete_disjoint_coverage(orbits, operations, full_kpoints, q);
}

void test_boundary_q_accepts_reciprocal_folding_and_time_reversal()
{
    const SpaceGroupSymOps operations{SpaceGroupSymOp::IDENTITY};
    const auto full_kpoints = build_uniform_kmesh_frac({4, 4, 4});
    const Vector3_Order<double> q{0.5, 0.0, 0.0};
    const auto little_group = build_sternheimer_fixed_q_little_group(operations, q);
    assert(little_group.size() == 2);
    assert(!little_group[0].time_reversal);
    assert(little_group[1].time_reversal);

    const auto orbits = build_sternheimer_fixed_q_k_orbits(operations, full_kpoints, q);
    assert(orbits.size() == 36);
    assert(std::any_of(orbits.begin(), orbits.end(), [](const auto &orbit) {
        return orbit.members.size() == 2;
    }));
    assert_complete_disjoint_coverage(orbits, operations, full_kpoints, q);
}

void test_si_k444_qstar_representatives_require_154_k_orbits()
{
    const auto operations = make_fcc_fractional_operations();
    const auto full_kpoints = build_uniform_kmesh_frac({4, 4, 4});

    SpaceGroupSymOps qstar_operations = operations;
    for (const auto &operation : operations)
    {
        auto antiunitary = operation;
        antiunitary.rotation = antiunitary.rotation * -1.0;
        qstar_operations.push_back(antiunitary);
    }
    const auto qstars = build_kpoint_stars(full_kpoints, qstar_operations, 1e-10);
    assert(qstars.size() == 8);

    std::vector<int> representative_counts;
    int total = 0;
    for (const auto &qstar : qstars)
    {
        const auto &q = qstar.members.at(
            static_cast<std::size_t>(qstar.representative_k_index)).kpoint;
        const auto orbits = build_sternheimer_fixed_q_k_orbits(operations, full_kpoints, q);
        assert_complete_disjoint_coverage(orbits, operations, full_kpoints, q);
        representative_counts.push_back(static_cast<int>(orbits.size()));
        total += static_cast<int>(orbits.size());
    }

    std::sort(representative_counts.begin(), representative_counts.end());
    assert((representative_counts == std::vector<int>{8, 13, 13, 16, 18, 20, 26, 40}));
    assert(total == 154);
}

void test_unitary_partial_rotation_swaps_atom_blocks()
{
    const auto context = make_two_atom_swap_context();
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}, {1, 1}};
    const SternheimerSymmetryRoute inverse_route{1, false, {0, 0, 0}};
    const auto actual = rotate_sternheimer_partial_response(
        context,
        layouts,
        atom_nabf,
        inverse_route,
        {0.0, 0.0, 0.0},
        make_complex_hermitian_matrix(),
        0);

    ComplexMatrix expected(2, 2);
    expected(0, 0) = {4.0, 0.0};
    expected(0, 1) = {2.0, -3.0};
    expected(1, 0) = {2.0, 3.0};
    expected(1, 1) = {1.0, 0.0};
    assert_matrix_close(actual, expected);
}

void test_antiunitary_partial_rotation_conjugates_once()
{
    const auto context = make_two_atom_swap_context();
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}, {1, 1}};
    const SternheimerSymmetryRoute inverse_route{0, true, {0, 0, 0}};
    const auto source = make_complex_hermitian_matrix();
    const auto actual = rotate_sternheimer_partial_response(
        context, layouts, atom_nabf, inverse_route, {0.0, 0.0, 0.0}, source, 0);
    assert_matrix_close(actual, conj(source));
}

void test_fixed_q_reconstruction_sums_every_orbit_member_once()
{
    const auto context = make_two_atom_swap_context();
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}, {1, 1}};
    const SternheimerFixedQKOrbit orbit{
        0,
        {{0, {0, false, {0, 0, 0}}},
         {1, {1, false, {0, 0, 0}}}}};
    const auto source = make_complex_hermitian_matrix();
    const auto reconstructed = reconstruct_sternheimer_fixed_q_response(
        context,
        layouts,
        atom_nabf,
        {0.0, 0.0, 0.0},
        {orbit},
        {{0, source}},
        0);

    ComplexMatrix expected(2, 2);
    expected(0, 0) = {5.0, 0.0};
    expected(0, 1) = {4.0, 0.0};
    expected(1, 0) = {4.0, 0.0};
    expected(1, 1) = {5.0, 0.0};
    assert_matrix_close(reconstructed, expected);

    require_throws(
        [&]() {
            reconstruct_sternheimer_fixed_q_response(
                context, layouts, atom_nabf, {0.0, 0.0, 0.0}, {orbit}, {}, 0);
        },
        "missing representative partial response");
}

void test_rejects_nonhermitian_partial_response()
{
    const auto context = make_two_atom_swap_context();
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}, {1, 1}};
    ComplexMatrix nonhermitian(2, 2);
    nonhermitian(0, 0) = {1.0, 0.0};
    nonhermitian(0, 1) = {2.0, 3.0};
    nonhermitian(1, 0) = {2.0, 3.0};
    nonhermitian(1, 1) = {4.0, 0.0};

    require_throws(
        [&]() {
            rotate_sternheimer_partial_response(
                context,
                layouts,
                atom_nabf,
                {0, false, {0, 0, 0}},
                {0.0, 0.0, 0.0},
                nonhermitian,
                0);
        },
        "not Hermitian");
}

void test_restores_every_qstar_matrix_member()
{
    auto context = make_two_atom_swap_context();
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}, {1, 1}};
    const Vector3_Order<double> q_representative{0.25, 0.0, 0.0};
    const Vector3_Order<double> q_member{0.75, 0.0, 0.0};

    SymmetryKStar star;
    star.star_index = 7;
    star.k_ibz = q_representative;
    star.members.push_back(build_symmetry_kspace_operation_member(
        context, 0, false, q_representative, q_representative, 0));
    star.members.push_back(build_symmetry_kspace_operation_member(
        context, 1, false, q_member, q_representative, 0));
    context.kstars = {star};

    const auto source = make_complex_hermitian_matrix();
    const auto restored = reconstruct_sternheimer_qstar_responses(
        context, layouts, atom_nabf, q_representative, source, 0);
    assert(restored.size() == 2);
    assert(restored[0].star_index == 7);
    assert(restored[0].member_index == 0);
    assert(same_fractional_kpoint(restored[0].q, q_representative, 1e-12));
    assert_matrix_close(restored[0].matrix, source);
    assert(restored[1].member_index == 1);
    assert(same_fractional_kpoint(restored[1].q, q_member, 1e-12));

    const auto expected_member = rotate_symmetry_kspace_matrix(
        context,
        layouts,
        star.members[1],
        source,
        atom_nabf,
        q_representative,
        false,
        &q_member);
    assert_matrix_close(restored[1].matrix, expected_member);

    require_throws(
        [&]() {
            reconstruct_sternheimer_qstar_responses(
                context, layouts, atom_nabf, {0.0, 0.0, 0.0}, source, 0);
        },
        "Failed to match");
}

}  // namespace

int main()
{
    test_gamma_uses_full_space_group_and_covers_grid();
    test_generic_q_keeps_only_identity_routes();
    test_boundary_q_accepts_reciprocal_folding_and_time_reversal();
    test_si_k444_qstar_representatives_require_154_k_orbits();
    test_unitary_partial_rotation_swaps_atom_blocks();
    test_antiunitary_partial_rotation_conjugates_once();
    test_fixed_q_reconstruction_sums_every_orbit_member_once();
    test_rejects_nonhermitian_partial_response();
    test_restores_every_qstar_matrix_member();
    return 0;
}
