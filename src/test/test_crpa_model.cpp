#include <cmath>
#include <iostream>
#include <stdexcept>

#include "../core/crpa_continuation.h"
#include "../core/crpa_model.h"
#include "../core/pair_density_vertex.h"
#include "librpa_compute.h"

using namespace librpa_int;
using Z = std::complex<double>;

namespace
{
void close(Z value, Z reference, double tolerance = 1e-8)
{
    if (!std::isfinite(std::abs(value)) || std::abs(value - reference) > tolerance)
    {
        std::cerr << "actual=" << value << " expected=" << reference << " tolerance=" << tolerance
                  << '\n';
        throw std::runtime_error("cRPA model numerical comparison failed");
    }
}

void check_rotation_and_fermion_order()
{
    ComplexMatrix left(2, 2), right(2, 2), vertex(4, 2), bare(2, 2);
    const double theta = .37;
    left(0, 0) = left(1, 1) = std::cos(theta);
    left(0, 1) = left(1, 0) = Z{0, std::sin(theta)};
    right(0, 0) = right(1, 1) = std::cos(.61);
    right(0, 1) = std::sin(.61);
    right(1, 0) = -std::sin(.61);
    bare(0, 0) = 3;
    bare(1, 1) = 2;
    bare(0, 1) = bare(1, 0) = .2;
    vertex(0, 0) = 1;
    vertex(3, 1) = 1;
    vertex(1, 0) = Z{.3, .2};
    vertex(2, 0) = std::conj(vertex(1, 0));
    const auto transform_vertex = [&](const ComplexMatrix& rotation)
    {
        ComplexMatrix output(4, 2);
        for (int a = 0; a < 2; ++a)
            for (int b = 0; b < 2; ++b)
                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        for (int mu = 0; mu < 2; ++mu)
                            output(a * 2 + b, mu) +=
                                std::conj(rotation(i, a)) * rotation(j, b) * vertex(i * 2 + j, mu);
        return output;
    };
    const auto original = contract_rectangular_ordered_pair_vertex(vertex, bare, vertex, 2, 2);
    const auto independent = contract_rectangular_ordered_pair_vertex(
        transform_vertex(left), bare, transform_vertex(right), 2, 2);
    ComplexMatrix rotated(4, 4), fermion(4, 4);
    librpa_crpa_rotate_tensor(
        2, 2, reinterpret_cast<const double*>(original.c), reinterpret_cast<const double*>(left.c),
        reinterpret_cast<const double*>(right.c), reinterpret_cast<double*>(rotated.c));
    for (int i = 0; i < rotated.size; ++i) close(rotated.c[i], independent.c[i], 1e-12);
    librpa_crpa_tensor_to_fermion(2, reinterpret_cast<const double*>(rotated.c),
                                  reinterpret_cast<double*>(fermion.c));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k)
                for (int l = 0; l < 2; ++l)
                    close(fermion(i * 2 + j, k * 2 + l), rotated(i * 2 + k, j * 2 + l), 0);
}

void check_bosonic_continuation()
{
    CrpaResult input;
    input.sites = {{"test", 0, 2, 0, {"a", "b"}}};
    input.frequencies = {.03, .08, .17, .31, .55, .9, 1.6, 3.0};
    CrpaOnsiteTensor tensor{0, 0, 0, ComplexMatrix(4, 4), {}, {}};
    ComplexMatrix residue(4, 4);
    for (int i = 0; i < residue.size; ++i)
    {
        tensor.bare.c[i] = Z{1.2 + .03 * i, .02 * (i % 3)};
        residue.c[i] = Z{.2 + .001 * i, .01 * (i % 2)};
    }
    const auto exact = [&](Z z, int i) { return tensor.bare.c[i] + residue.c[i] / (z * z - .49); };
    for (double frequency : input.frequencies)
    {
        ComplexMatrix value(4, 4);
        for (int i = 0; i < value.size; ++i) value.c[i] = exact(Z{0, frequency}, i);
        tensor.u.push_back(std::move(value));
    }
    input.tensors.push_back(tensor);
    const std::vector<Z> queries{{0, 0}, {.4, .05}, {0, .45}, {0, 100}};
    const auto result = continue_crpa_interactions(input, queries, -1, 1e-7);
    for (int q = 0; q < queries.size(); ++q)
        for (int i = 0; i < residue.size; ++i)
        {
            close(result[0].values[q].c[i], exact(queries[q], i), 1e-8);
            if (result[0].status[q][i])
                throw std::runtime_error("exact bosonic pole failed diagnostics");
        }
    // Four coefficients include the quadratic denominator of this bosonic pole.
    for (const auto method : {LIBRPA_ANACON_AAA, LIBRPA_ANACON_RIDGE})
    {
        const auto selected = continue_crpa_interactions(input, queries, 4, 1e-7, method);
        for (int q = 0; q < queries.size(); ++q)
            for (int i = 0; i < residue.size; ++i)
                close(selected[0].values[q].c[i], exact(queries[q], i), 2e-7);
    }
    // A perturbed node must be visible in independent-subset diagnostics.
    input.tensors[0].u[3].c[0] += .03;
    const auto perturbed = continue_crpa_interactions(input, {{0, 0}}, -1, 1e-7);
    if (!(perturbed[0].status[0][0] & (2 | 4 | 8)))
        throw std::runtime_error("inconsistent response node escaped continuation diagnostics");
    // Constant U=v (Pd=P0 limit) must survive reciprocal-difference zero pivots.
    for (auto& value : input.tensors[0].u) value = tensor.bare;
    const auto constant = continue_crpa_interactions(input, queries, -1, 1e-7);
    for (int q = 0; q < queries.size(); ++q)
        for (int i = 0; i < residue.size; ++i)
            close(constant[0].values[q].c[i], tensor.bare.c[i], 0);
}

void check_two_sublattice_model()
{
    // Two Ni(d) and two O(p), with opposite collinear occupations on the Ni sites.
    AtomicBasis basis(std::vector<std::size_t>{5, 5, 3, 3});
    basis.set_l_shells({{2}, {2}, {1}, {1}});
    BasisConvention convention;
    convention.bloch_phase = -1;
    convention.bloch_ratom = 0;
    convention.order = LIBRPA_ANGULAR_ORDER_ABS_PM;
    convention.coeff_m_negative = LIBRPA_RSH_COEFF_M_1;
    convention.coeff_m_positive = LIBRPA_RSH_COEFF_1_M;
    ComplexMatrix identity(16, 16);
    identity.set_as_identity_matrix();
    MeanField mf(2, 1, 16, 16);
    for (int spin = 0; spin < 2; ++spin)
    {
        mf.get_eigenvectors()[spin][0][0] = identity;
        for (int band = 0; band < 16; ++band)
        {
            const bool occupied = band >= 10 || (spin == 0 ? band < 5 : band >= 5 && band < 10);
            mf.get_eigenvals()[spin](0, band) = occupied ? -.5 : .4;
            mf.get_weight()[spin](0, band) = occupied ? 1 : 0;
        }
    }
    CrpaWindowOptions options;
    options.correlated_species = "Ni";
    options.ligand_species = "O";
    options.parent_orbitals = "dp";
    options.output_orbitals = "d";
    options.response_bands = {0, 1};
    for (int b = 0; b < 16; ++b) options.orbital_bands.push_back(b);
    PeriodicBoundaryData pbc;
    pbc.kfrac_list = {{0, 0, 0}};
    pbc.Rlist = {{0, 0, 0}};
    auto model = prepare_crpa_model(basis, convention, {0, 0, 1, 1}, {"Ni", "O"}, {identity}, pbc,
                                    mf, options);
    const auto snapshot = snapshot_crpa_model(model, mf);
    if (snapshot.parent_sites.size() != 4 || snapshot.output_sites.size() != 2)
        throw std::runtime_error("AFM sites were collapsed or ligand sites lost");
    for (int spin = 0; spin < 2; ++spin)
    {
        const auto& state = snapshot.spin_k[spin][0];
        ComplexMatrix projector(10, 16);
        librpa_crpa_project_orbitals(16, 1, 16, 10,
                                     reinterpret_cast<const double*>(state.coefficients.c),
                                     reinterpret_cast<const double*>(identity.c),
                                     reinterpret_cast<const double*>(state.output_phi.c),
                                     reinterpret_cast<double*>(projector.c));
        for (int i = 0; i < projector.size; ++i)
            close(projector.c[i], state.output_projector.c[i], 0);
        for (int site = 0; site < 2; ++site)
            for (int orbital = 0; orbital < 5; ++orbital)
            {
                const int index = 5 * site + orbital;
                close(state.output_density(index, index), spin == site ? 1 : 0, 0);
                close(state.output_h(index, index), spin == site ? -.5 : .4, 0);
            }
    }
}
}  // namespace

int main()
{
    check_rotation_and_fermion_order();
    check_bosonic_continuation();
    check_two_sublattice_model();
    std::cout << "Complex density rotation and bosonic static/real-axis continuation passed\n";
}
