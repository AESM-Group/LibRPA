#include <cassert>
#include <complex>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../sternheimer_partial_reconstruction.h"
#include "../../src/core/symmetry_context.h"

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

ComplexMatrix scalar_matrix(const double value)
{
    ComplexMatrix matrix(1, 1);
    matrix(0, 0) = {value, 0.0};
    return matrix;
}

SpeciesBasisLayout make_s_layout()
{
    SpeciesBasisLayout layout;
    layout.label = "X";
    layout.set({0});
    return layout;
}

SymmetryContext make_one_atom_context(const Vector3_Order<double> &q)
{
    SymmetryContext context;
    const Matrix3 lattice(1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0);
    context.set_crystal_structure(lattice, lattice, {{0, 0}}, {{0, {0.0, 0.0, 0.0}}});
    context.set_rspace_operations({SpaceGroupSymOp::IDENTITY});
    context.basis_convention = {-1,
                                0,
                                LIBRPA_ANGULAR_ORDER_NATURAL,
                                LIBRPA_RSH_COEFF_1_M,
                                LIBRPA_RSH_COEFF_1_M};
    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = q;
    star.members.push_back(build_symmetry_kspace_operation_member(context, 0, false, q, q, 0));
    context.kstars = {star};
    return context;
}

driver::SternheimerPartialResponseGroup make_group(
    const int iq,
    const int ifreq,
    const double omega,
    const double weight,
    std::map<int, ComplexMatrix> representatives)
{
    driver::SternheimerPartialResponseGroup group;
    group.iq = iq;
    group.ifreq = ifreq;
    group.omega = omega;
    group.weight = weight;
    group.atom_naux = {1};
    group.representatives = std::move(representatives);
    return group;
}

void test_reconstructs_all_frequencies_and_reports_orbit_counts()
{
    const Vector3_Order<double> q{0.25, 0.0, 0.0};
    const Vector3_Order<double> q_member{0.75, 0.0, 0.0};
    auto context = make_one_atom_context(q);
    context.kstars.front().members.push_back(build_symmetry_kspace_operation_member(
        context, 0, true, q_member, q, 0));
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0},
                                                          {0.5, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.25, 0.0, 0.0}, 1.0}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(2, 1),
                   make_group(2, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-1.0)}, {1, scalar_matrix(-2.0)}}));
    groups.emplace(std::make_pair(2, 2),
                   make_group(2, 2, 1.5, 0.25,
                              {{0, scalar_matrix(-4.0)}, {1, scalar_matrix(-5.0)}}));

    const auto reconstructed = driver::reconstruct_sternheimer_partial_responses(
        context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 2, true, 0);
    assert(reconstructed.size() == 2);
    assert(reconstructed[0].iq == 2);
    assert(reconstructed[0].ifreq == 1);
    assert(reconstructed[0].full_k_count == 2);
    assert(reconstructed[0].representative_k_count == 2);
    assert(reconstructed[0].little_group_order == 1);
    assert(std::abs(reconstructed[0].matrix(0, 0) - std::complex<double>(-3.0, 0.0))
           < 1.0e-12);
    assert(reconstructed[0].kresolved_responses.size() == 2);
    assert(reconstructed[0].kresolved_responses[0].ik_full == 0);
    assert(std::abs(reconstructed[0].kresolved_responses[0].matrix(0, 0)
                    - std::complex<double>(-1.0, 0.0))
           < 1.0e-12);
    assert(reconstructed[0].kresolved_responses[1].ik_full == 1);
    assert(std::abs(reconstructed[0].kresolved_responses[1].matrix(0, 0)
                    - std::complex<double>(-2.0, 0.0))
           < 1.0e-12);
    assert(reconstructed[0].qstar_responses.size() == 2);
    assert(std::abs(reconstructed[1].matrix(0, 0) - std::complex<double>(-9.0, 0.0))
           < 1.0e-12);
}

void test_boundary_q_time_reversal_reduces_two_kpoints_to_one_representative()
{
    const Vector3_Order<double> q{0.5, 0.0, 0.0};
    const Vector3_Order<double> gamma{0.0, 0.0, 0.0};
    auto context = make_one_atom_context(q);
    SymmetryKStar gamma_star;
    gamma_star.star_index = 1;
    gamma_star.k_ibz = gamma;
    gamma_star.members.push_back(build_symmetry_kspace_operation_member(
        context, 0, false, gamma, gamma, 0));
    context.kstars.push_back(std::move(gamma_star));
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.25, 0.0, 0.0},
                                                          {0.75, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{3, {0.5, 0.0, 0.0}, 0.5},
                                                         {4, {0.0, 0.0, 0.0}, 0.5}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(3, 1),
                   make_group(3, 1, 0.5, 0.125, {{0, scalar_matrix(-2.0)}}));
    groups.emplace(std::make_pair(4, 1),
                   make_group(4, 1, 0.5, 0.125, {{0, scalar_matrix(-1.0)}}));

    const auto reconstructed = driver::reconstruct_sternheimer_partial_responses(
        context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0);
    assert(reconstructed.size() == 2);
    assert(reconstructed[0].representative_k_count == 1);
    assert(reconstructed[0].full_k_count == 2);
    assert(std::abs(reconstructed[0].matrix(0, 0) - std::complex<double>(-4.0, 0.0))
           < 1.0e-12);
}

void test_explicit_routes_can_retain_two_discrete_hamiltonian_orbits()
{
    const Vector3_Order<double> q{0.0, 0.0, 0.0};
    const auto context = make_one_atom_context(q);
    const std::vector<Vector3_Order<double>> full_kpoints{{0.25, 0.0, 0.0},
                                                          {0.75, 0.0, 0.0}};
    const std::vector<driver::SternheimerFixedQRouteRecord> routes{
        {1, 0, 0, {0, false, {0, 0, 0}}},
        {1, 1, 1, {0, false, {0, 0, 0}}},
    };

    const auto orbits = driver::build_sternheimer_fixed_q_k_orbits_from_routes(
        context.rspace_operations, full_kpoints, q, routes);

    assert(orbits.size() == 2);
    assert(orbits[0].representative_ik_full == 0);
    assert(orbits[0].members.size() == 1);
    assert(orbits[0].members[0].ik_full == 0);
    assert(orbits[1].representative_ik_full == 1);
    assert(orbits[1].members.size() == 1);
    assert(orbits[1].members[0].ik_full == 1);

    auto wrong_fold = routes;
    wrong_fold[1].inverse_route.fold_G = {1, 0, 0};
    require_throws(
        [&]() {
            driver::build_sternheimer_fixed_q_k_orbits_from_routes(
                context.rspace_operations, full_kpoints, q, wrong_fold);
        },
        "reciprocal fold disagrees");
}

void test_matrix_only_reconstructs_one_q_without_claiming_full_q_coverage()
{
    const Vector3_Order<double> q{0.0, 0.0, 0.0};
    const auto context = make_one_atom_context(q);
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.25, 0.0, 0.0},
                                                          {0.75, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{1, {0.0, 0.0, 0.0}, 1.0}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(1, 1),
                   make_group(1, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-1.0)}, {1, scalar_matrix(-2.0)}}));
    const std::vector<driver::SternheimerFixedQRouteRecord> routes{
        {1, 0, 0, {0, false, {0, 0, 0}}},
        {1, 1, 1, {0, false, {0, 0, 0}}},
    };

    const auto reconstructed = driver::reconstruct_sternheimer_partial_responses(
        context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0,
        &routes, true);

    assert(reconstructed.size() == 1);
    assert(reconstructed[0].representative_k_count == 2);
    assert(reconstructed[0].kresolved_responses.size() == 2);
    assert(reconstructed[0].qstar_responses.empty());
    assert(reconstructed[0].q_weight == 0.0);
    assert(std::abs(reconstructed[0].matrix(0, 0) - std::complex<double>(-3.0, 0.0))
           < 1.0e-12);
}

void test_fixed_q_symmetry_diagnostic_reports_route_and_transform()
{
    auto context = make_one_atom_context({0.0, 0.0, 0.0});
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0}};

    const auto diagnostics = driver::build_sternheimer_fixed_q_symmetry_diagnostics(
        context, layouts, atom_nabf, full_kpoints, {0.0, 0.0, 0.0}, 0);

    assert(diagnostics.size() == 1);
    assert(diagnostics[0].representative_ik_full == 0);
    assert(diagnostics[0].member_ik_full == 0);
    assert(diagnostics[0].inverse_route.spatial_isym == 0);
    assert(!diagnostics[0].inverse_route.time_reversal);
    assert(std::abs(diagnostics[0].transform(0, 0) - std::complex<double>(1.0, 0.0))
           < 1.0e-12);
}

void test_fixed_q_symmetry_diagnostic_writer_records_route_and_transform()
{
    driver::SternheimerFixedQSymmetryDiagnostic diagnostic;
    diagnostic.representative_ik_full = 1;
    diagnostic.member_ik_full = 4;
    diagnostic.inverse_route = {3, true, {1, -1, 0}};
    diagnostic.transform = scalar_matrix(2.0);
    const auto path = std::filesystem::temp_directory_path()
                      / "librpa_sternheimer_fixed_q_symmetry_diagnostic_test.dat";

    driver::write_sternheimer_fixed_q_symmetry_diagnostics(path.string(), 2, {diagnostic});

    std::ifstream input(path);
    const std::string text((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    assert(text.find("route 2 1 4 3 1 1 -1 0") != std::string::npos);
    assert(text.find("matrix 1 1") != std::string::npos);
    assert(text.find("0 0 2") != std::string::npos);
    std::filesystem::remove(path);
}

void test_derives_qweight_from_qstar_without_overwriting_frequency_weight()
{
    const Vector3_Order<double> q{0.25, 0.0, 0.0};
    const Vector3_Order<double> q_member{0.75, 0.0, 0.0};
    auto context = make_one_atom_context(q);
    context.kstars.front().members.push_back(build_symmetry_kspace_operation_member(
        context, 0, true, q_member, q, 0));
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0},
                                                          {0.5, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{4, {0.25, 0.0, 0.0}, 1.0}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(4, 1),
                   make_group(4, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-1.0)}, {1, scalar_matrix(-2.0)}}));

    const auto reconstructed = driver::reconstruct_sternheimer_partial_responses(
        context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0);
    assert(reconstructed.size() == 1);
    assert(reconstructed[0].qstar_responses.size() == 2);
    assert(std::abs(reconstructed[0].weight - 0.125) < 1.0e-15);
    assert(std::abs(reconstructed[0].q_weight - 1.0) < 1.0e-15);
}

void test_rejects_normalized_manifest_weights_that_disagree_with_qstars()
{
    const Vector3_Order<double> gamma{0.0, 0.0, 0.0};
    const Vector3_Order<double> boundary{0.5, 0.0, 0.0};
    auto context = make_one_atom_context(gamma);
    SymmetryKStar boundary_star;
    boundary_star.star_index = 1;
    boundary_star.k_ibz = boundary;
    boundary_star.members.push_back(build_symmetry_kspace_operation_member(
        context, 0, false, boundary, boundary, 0));
    context.kstars.push_back(std::move(boundary_star));

    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0},
                                                          {0.5, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{1, {0.0, 0.0, 0.0}, 0.75},
                                                         {2, {0.5, 0.0, 0.0}, 0.25}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(1, 1),
                   make_group(1, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-1.0)}, {1, scalar_matrix(-2.0)}}));
    groups.emplace(std::make_pair(2, 1),
                   make_group(2, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-3.0)}, {1, scalar_matrix(-4.0)}}));

    require_throws(
        [&]() {
            driver::reconstruct_sternheimer_partial_responses(
                context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0);
        },
        "q-star weight");
}

void test_rejects_qstars_that_do_not_cover_the_full_q_grid()
{
    const Vector3_Order<double> gamma{0.0, 0.0, 0.0};
    const auto context = make_one_atom_context(gamma);
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0},
                                                          {0.5, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{1, {0.0, 0.0, 0.0}, 0.5}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(1, 1),
                   make_group(1, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-1.0)}, {1, scalar_matrix(-2.0)}}));

    require_throws(
        [&]() {
            driver::reconstruct_sternheimer_partial_responses(
                context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0);
        },
        "do not cover the full q grid");
}

void test_qstar_rpa_audit_uses_internal_weight_and_checks_every_member()
{
    driver::SternheimerReconstructedResponse response;
    response.iq = 2;
    response.ifreq = 1;
    response.omega = 0.5;
    response.weight = 0.125;
    response.q_weight = 1.0;
    response.matrix = scalar_matrix(-0.4);
    response.qstar_responses = {
        {0, 0, 0, false, {0.25, 0.0, 0.0}, scalar_matrix(-0.4)},
        {0, 1, 0, true, {0.75, 0.0, 0.0}, scalar_matrix(-0.4)}};
    const std::vector<SternheimerQStarResponse> coulomb_qstar{
        {0, 0, 0, false, {0.25, 0.0, 0.0}, scalar_matrix(2.0)},
        {0, 1, 0, true, {0.75, 0.0, 0.0}, scalar_matrix(2.0)}};

    const auto audit = driver::compute_sternheimer_qstar_rpa_frequency(
        response, coulomb_qstar, 1.0e-10);
    assert(audit.qstar_size == 2);
    assert(audit.result.qweight == 1.0);
    assert(audit.max_integrand_difference < 1.0e-14);

    auto inconsistent_coulomb = coulomb_qstar;
    inconsistent_coulomb[1].matrix = scalar_matrix(3.0);
    require_throws(
        [&]() {
            driver::compute_sternheimer_qstar_rpa_frequency(
                response, inconsistent_coulomb, 1.0e-10);
        },
        "q-star trace-log invariance");
}

void test_default_qstar_tolerance_accepts_dense_linear_algebra_noise()
{
    driver::SternheimerReconstructedResponse response;
    response.iq = 2;
    response.ifreq = 1;
    response.omega = 0.5;
    response.weight = 0.125;
    response.q_weight = 1.0;
    response.matrix = scalar_matrix(-0.4);
    response.qstar_responses = {
        {0, 0, 0, false, {0.25, 0.0, 0.0}, scalar_matrix(-0.4)},
        {0, 1, 0, true, {0.75, 0.0, 0.0}, scalar_matrix(-0.4)}};
    const std::vector<SternheimerQStarResponse> coulomb_qstar{
        {0, 0, 0, false, {0.25, 0.0, 0.0}, scalar_matrix(2.0)},
        {0, 1, 0, true, {0.75, 0.0, 0.0}, scalar_matrix(2.00000002)}};

    const auto audit = driver::compute_sternheimer_qstar_rpa_frequency(
        response, coulomb_qstar, 1.0e-10);
    assert(audit.max_integrand_difference > 1.0e-10);
    assert(audit.max_integrand_difference < 1.0e-8);
}

void test_recovers_target_q_coulomb_from_ibz_representative()
{
    const Vector3_Order<double> q_ibz{0.25, 0.0, 0.0};
    const Vector3_Order<double> q_target{0.75, 0.0, 0.0};
    auto context = make_one_atom_context(q_ibz);
    context.kstars.front().members.push_back(build_symmetry_kspace_operation_member(
        context, 0, true, q_target, q_ibz, 0));
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};

    const auto recovered = driver::reconstruct_sternheimer_full_q_matrices_from_ibz(
        context,
        layouts,
        atom_nabf,
        {q_ibz},
        {scalar_matrix(2.0)},
        0);
    assert(recovered.size() == 2);
    const auto target = std::find_if(recovered.cbegin(), recovered.cend(), [&](const auto &member) {
        return same_fractional_kpoint(member.q, q_target, 1.0e-8);
    });
    assert(target != recovered.cend());
    assert(std::abs(target->matrix(0, 0) - std::complex<double>(2.0, 0.0)) < 1.0e-12);
}

void test_explicit_discrete_qstar_routes_define_coverage_and_weights()
{
    const auto context = make_one_atom_context({0.0, 0.0, 0.0});
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0},
                                                          {0.5, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{1, {0.0, 0.0, 0.0}, 0.5},
                                                         {2, {0.5, 0.0, 0.0}, 0.5}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(1, 1),
                   make_group(1, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-1.0)}, {1, scalar_matrix(-2.0)}}));
    groups.emplace(std::make_pair(2, 1),
                   make_group(2, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-3.0)}, {1, scalar_matrix(-4.0)}}));
    const std::vector<driver::SternheimerFixedQRouteRecord> fixed_q_routes{
        {1, 0, 0, {0, false, {0, 0, 0}}},
        {1, 1, 1, {0, false, {0, 0, 0}}},
        {2, 0, 0, {0, false, {0, 0, 0}}},
        {2, 1, 1, {0, false, {0, 0, 0}}},
    };
    const std::vector<driver::SternheimerQStarRouteRecord> qstar_routes{
        {1, 1, {0, false, {0, 0, 0}}},
        {2, 2, {0, false, {0, 0, 0}}},
    };

    const auto reconstructed = driver::reconstruct_sternheimer_partial_responses(
        context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0,
        &fixed_q_routes, false, &qstar_routes);

    assert(reconstructed.size() == 2);
    assert(reconstructed[0].qstar_responses.size() == 1);
    assert(reconstructed[1].qstar_responses.size() == 1);
    assert(std::abs(reconstructed[0].q_weight - 0.5) < 1.0e-15);
    assert(std::abs(reconstructed[1].q_weight - 0.5) < 1.0e-15);
    assert(std::abs(reconstructed[0].matrix(0, 0) - std::complex<double>(-3.0, 0.0))
           < 1.0e-12);
    assert(std::abs(reconstructed[1].matrix(0, 0) - std::complex<double>(-7.0, 0.0))
           < 1.0e-12);

    auto wrong_fold = qstar_routes;
    wrong_fold[1].inverse_route.fold_G = {1, 0, 0};
    require_throws(
        [&]() {
            driver::reconstruct_sternheimer_partial_responses(
                context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0,
                &fixed_q_routes, false, &wrong_fold);
        },
        "q-star route reciprocal fold disagrees");

    auto extra_fixed_q_routes = fixed_q_routes;
    extra_fixed_q_routes.push_back({99, 0, 0, {0, false, {0, 0, 0}}});
    require_throws(
        [&]() {
            driver::reconstruct_sternheimer_partial_responses(
                context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, groups, 1, true, 0,
                &extra_fixed_q_routes, false, &qstar_routes);
        },
        "unexpected iq=99");
}

void test_qstar_routes_use_the_full_q_grid_for_member_indices()
{
    const auto context = make_one_atom_context({0.5, 0.0, 0.0});
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0},
                                                          {0.5, 0.0, 0.0}};
    const std::vector<Vector3_Order<double>> full_qpoints{{0.5, 0.0, 0.0},
                                                          {0.25, 0.0, 0.0},
                                                          {-0.5, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{1, {0.5, 0.0, 0.0}, 2.0 / 3.0},
                                                         {2, {0.25, 0.0, 0.0}, 1.0 / 3.0}};
    driver::SternheimerPartialResponseGroups groups;
    groups.emplace(std::make_pair(1, 1),
                   make_group(1, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-1.0)}, {1, scalar_matrix(-2.0)}}));
    groups.emplace(std::make_pair(2, 1),
                   make_group(2, 1, 0.5, 0.125,
                              {{0, scalar_matrix(-3.0)}, {1, scalar_matrix(-4.0)}}));
    const std::vector<driver::SternheimerQStarRouteRecord> routes{
        {1, 1, {0, false, {0, 0, 0}}},
        {2, 2, {0, false, {0, 0, 0}}},
        {1, 3, {0, true, {0, 0, 0}}},
    };

    const auto reconstructed = driver::reconstruct_sternheimer_partial_responses(
        context, layouts, atom_nabf, full_kpoints, full_qpoints, qpoints, groups, 1, true, 0,
        nullptr, false, &routes);

    assert(reconstructed.size() == 2);
    assert(reconstructed[0].qstar_responses.size() == 2);
    assert(reconstructed[1].qstar_responses.size() == 1);
    assert(std::abs(reconstructed[0].q_weight - 2.0 / 3.0) < 1.0e-15);
    assert(std::abs(reconstructed[1].q_weight - 1.0 / 3.0) < 1.0e-15);
}

void test_rejects_missing_representative_and_frequency()
{
    const Vector3_Order<double> q{0.0, 0.0, 0.0};
    const auto context = make_one_atom_context(q);
    const std::vector<SpeciesBasisLayout> layouts{make_s_layout()};
    const std::map<atom_t, std::size_t> atom_nabf{{0, 1}};
    const std::vector<Vector3_Order<double>> full_kpoints{{0.0, 0.0, 0.0}};
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.0, 0.0, 0.0}, 1.0}};

    driver::SternheimerPartialResponseGroups missing_representative;
    missing_representative.emplace(
        std::make_pair(2, 1), make_group(2, 1, 0.5, 0.125, {}));
    require_throws(
        [&]() {
            driver::reconstruct_sternheimer_partial_responses(
                context,
                layouts,
                atom_nabf,
                full_kpoints,
                full_kpoints,
                qpoints,
                missing_representative,
                1,
                true,
                0);
        },
        "missing representative partial response");

    require_throws(
        [&]() {
            driver::reconstruct_sternheimer_partial_responses(
                context, layouts, atom_nabf, full_kpoints, full_kpoints, qpoints, {}, 1, true, 0);
        },
        "missing (iq, ifreq)=(2, 1)");
}

}  // namespace

int main()
{
    test_reconstructs_all_frequencies_and_reports_orbit_counts();
    test_boundary_q_time_reversal_reduces_two_kpoints_to_one_representative();
    test_explicit_routes_can_retain_two_discrete_hamiltonian_orbits();
    test_matrix_only_reconstructs_one_q_without_claiming_full_q_coverage();
    test_fixed_q_symmetry_diagnostic_reports_route_and_transform();
    test_fixed_q_symmetry_diagnostic_writer_records_route_and_transform();
    test_derives_qweight_from_qstar_without_overwriting_frequency_weight();
    test_rejects_normalized_manifest_weights_that_disagree_with_qstars();
    test_rejects_qstars_that_do_not_cover_the_full_q_grid();
    test_qstar_rpa_audit_uses_internal_weight_and_checks_every_member();
    test_default_qstar_tolerance_accepts_dense_linear_algebra_noise();
    test_recovers_target_q_coulomb_from_ibz_representative();
    test_explicit_discrete_qstar_routes_define_coverage_and_weights();
    test_qstar_routes_use_the_full_q_grid_for_member_indices();
    test_rejects_missing_representative_and_frequency();
    return 0;
}
