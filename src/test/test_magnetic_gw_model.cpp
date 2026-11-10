// Gapped three-site 120-degree SOC model. This is a numerical kernel control,
// not a PBE material reference. The full contractions are its independent
// oracle for the magnetic irreducible-sector contractions.
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "../api/instance_manager.h"
#include "../core/chi0.h"
#include "../api/dataset_helper.h"
#include "../core/epsilon.h"
#include "../math/matrix_m.h"
#include "../math/utils_matrix_mpi.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"
#include "librpa.hpp"

using namespace librpa_int;
using Complex = std::complex<double>;

namespace
{

struct Model
{
    Matz hamiltonian{6, 6, MAJOR::COL};
    Matz eigenvectors;
    std::vector<double> eigenvalues;
    std::vector<double> positions;
    std::vector<int> operations;
    std::vector<int> antiunitary{0, 0, 0, 1, 1, 1};
    bool dispersive = false;
    bool rotated_spin_basis = false;
    std::vector<double> provided_spin_u;
    Matz basis{6, 6, MAJOR::COL};
    const double lattice[9]{6.0, 0.0, 0.0, 3.0, 3.0 * std::sqrt(3.0), 0.0, 0.0, 0.0, 8.0};
    const double reciprocal[9]{TWO_PI / 6.0,
                               -TWO_PI / (6.0 * std::sqrt(3.0)),
                               0.0,
                               0.0,
                               TWO_PI / (3.0 * std::sqrt(3.0)),
                               0.0,
                               0.0,
                               0.0,
                               TWO_PI / 8.0};

    Model(const bool with_dispersion = false, const bool with_rotated_basis = false)
        : dispersive(with_dispersion), rotated_spin_basis(with_rotated_basis)
    {
        basis.zero_out();
        // A global SU(2) basis change about y; it commutes with canonical Theta.
        const double angle = rotated_spin_basis ? 0.71 : 0.0;
        for (int atom = 0; atom != 3; ++atom)
        {
            basis(atom, atom) = basis(atom + 3, atom + 3) = std::cos(angle / 2);
            basis(atom, atom + 3) = -std::sin(angle / 2);
            basis(atom + 3, atom) = std::sin(angle / 2);
        }
        hamiltonian.zero_out();
        for (int atom = 0; atom != 3; ++atom)
        {
            const double angle = TWO_PI * atom / 3.0;
            positions.insert(positions.end(), {0.8 * std::cos(angle), 0.8 * std::sin(angle), 0.0});
            hamiltonian(atom, atom + 3) = std::polar(1.0, -angle);
            hamiltonian(atom + 3, atom) = std::conj(hamiltonian(atom, atom + 3));
            for (int spinor = 0; spinor != 2; ++spinor)
            {
                const int next = (atom + 1) % 3;
                const Complex hopping{0.15, spinor == 0 ? 0.04 : -0.04};
                hamiltonian(atom + 3 * spinor, next + 3 * spinor) = hopping;
                hamiltonian(next + 3 * spinor, atom + 3 * spinor) = std::conj(hopping);
            }
        }
        eigsh(hamiltonian, eigenvalues, eigenvectors);
        assert(eigenvalues[3] - eigenvalues[2] > 1.0);

        // Row fractional C3, C3^2 and their horizontal-mirror partners.
        const int rotations[3][9]{{1, 0, 0, 0, 1, 0, 0, 0, 1},
                                  {-1, 1, 0, -1, 0, 0, 0, 0, 1},
                                  {0, -1, 0, 1, -1, 0, 0, 0, 1}};
        for (int mirror = 0; mirror != 2; ++mirror)
            for (int rotation = 0; rotation != 3; ++rotation)
                for (int index = 0; index != 9; ++index)
                    operations.push_back(index == 8 && mirror ? -1 : rotations[rotation][index]);

        // Provided SOC actions live in the same spin basis as the wavefunctions.
        for (int mirror = 0; mirror != 2; ++mirror)
            for (int rotation = 0; rotation != 3; ++rotation)
            {
                const double angle = TWO_PI * rotation / 3.0 + mirror * PI;
                const Complex u[2]{std::polar(1.0, -0.5 * angle), std::polar(1.0, 0.5 * angle)};
                for (int a = 0; a != 2; ++a)
                    for (int b = 0; b != 2; ++b)
                    {
                        Complex rotated = 0.0;
                        for (int c = 0; c != 2; ++c)
                            rotated += basis(3 * a, 3 * c) * u[c] * std::conj(basis(3 * b, 3 * c));
                        provided_spin_u.insert(provided_spin_u.end(),
                                               {rotated.real(), rotated.imag()});
                    }
            }
    }

    Matz h_at(const Vector3_Order<double> &k) const
    {
        Matz h = hamiltonian.copy();
        if (dispersive)
        {
            // Real inter-cell hopping, invariant under Cartesian C3 and Theta Mz.
            // Unlike an identity shift, this changes the occupied projector.
            const double f = std::cos(TWO_PI * k.x) + std::cos(TWO_PI * k.y) +
                             std::cos(TWO_PI * (k.x - k.y)) + 0.8 * std::cos(TWO_PI * k.z);
            for (int atom = 0; atom != 3; ++atom)
                for (int spin = 0; spin != 2; ++spin)
                {
                    const int next = (atom + 1) % 3;
                    h(atom + 3 * spin, next + 3 * spin) += 0.035 * f;
                    h(next + 3 * spin, atom + 3 * spin) += 0.035 * f;
                }
        }
        return basis * h * transpose(basis, true);
    }

    void diagonalize(const Vector3_Order<double> &k, std::vector<double> &values,
                     Matz &vectors) const
    {
        eigsh(h_at(k), values, vectors);
        assert(values[3] - values[2] > 1.0);
    }

    Matz action(const int rotation, const bool mirror) const
    {
        Matz a(6, 6, MAJOR::COL);
        a.zero_out();
        const double angle = TWO_PI * rotation / 3.0 + mirror * PI;
        for (int atom = 0; atom != 3; ++atom)
            for (int spin = 0; spin != 2; ++spin)
            {
                const Complex u = std::polar(1.0, (spin == 0 ? -0.5 : 0.5) * angle);
                const int target = (atom + rotation) % 3;
                if (mirror)
                    a(target + 3 * (1 - spin), atom + 3 * spin) =
                        (spin == 0 ? -1.0 : 1.0) * std::conj(u);
                else
                    a(target + 3 * spin, atom + 3 * spin) = u;
            }
        // The antiunitary matrix changes by B A B^T, the unitary one by B A B^dag.
        return basis * a * (mirror ? transpose(basis) : transpose(basis, true));
    }

    void check_covariance(const std::vector<Vector3_Order<double>> &points) const
    {
        double residual = 0.0;
        for (const auto &k : points)
            for (int mirror = 0; mirror != 2; ++mirror)
                for (int rotation = 0; rotation != 3; ++rotation)
                {
                    // Dual action derived explicitly from the row fractional C3 matrix.
                    Vector3_Order<double> target = k;
                    for (int n = 0; n != rotation; ++n)
                        target = {-target.y, target.x - target.y, target.z};
                    if (mirror) target = {-target.x, -target.y, target.z};
                    const auto source = h_at(k);
                    Matz operand = source.copy();
                    if (mirror)
                        for (int i = 0; i != 6; ++i)
                            for (int j = 0; j != 6; ++j) operand(i, j) = std::conj(source(i, j));
                    const auto transformed = action(rotation, mirror) * operand *
                                             transpose(action(rotation, mirror), true);
                    const auto expected = h_at(target);
                    for (int i = 0; i != 6; ++i)
                        for (int j = 0; j != 6; ++j)
                            residual =
                                std::max(residual, std::abs(transformed(i, j) - expected(i, j)));
                }
        if (global::mpi_comm_global_h.is_root())
            std::cout << "Independent Hamiltonian covariance residual: " << residual << std::endl;
        assert(residual < 1e-12);
    }
};

// Count input stars by the model's explicit geometric action, independently of LibRPA.
int independent_ibz_size(const int n, const int nz)
{
    std::set<std::array<int, 3>> remaining;
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
            for (int c = 0; c < nz; ++c) remaining.insert({a, b, c});
    const auto mod = [n](int x) { return (x % n + n) % n; };
    int stars = 0;
    while (!remaining.empty())
    {
        const auto k = *remaining.begin();
        ++stars;
        for (int mirror = 0; mirror < 2; ++mirror)
        {
            int a = k[0], b = k[1];
            for (int rotation = 0; rotation < 3; ++rotation)
            {
                remaining.erase({mod(mirror ? -a : a), mod(mirror ? -b : b), k[2]});
                const int next_a = -b;
                b = a - b;
                a = next_a;
            }
        }
    }
    return stars;
}

// Capture the full finite response operators for one bounded model pair.
struct ResponseOperators
{
    std::vector<Complex> chi0;
    std::vector<Complex> wc;
};

atpair_k_cplx_mat_t independent_coulomb_copy(const atpair_k_cplx_mat_t &source)
{
    atpair_k_cplx_mat_t result;
    for (const auto &[i, js] : source)
        for (const auto &[j, qs] : js)
            for (const auto &[q, block] : qs)
                result[i][j][q] = std::make_shared<ComplexMatrix>(*block);
    return result;
}

int response_full_q_index(const PeriodicBoundaryData &pbc, const Vector3_Order<double> &q)
{
    const Vector3_Order<double> frac{pbc.latvec * q};
    int result = -1;
    for (int k = 0; k != static_cast<int>(pbc.kfrac_list_full.size()); ++k)
        if (same_fractional_kpoint(frac, pbc.kfrac_list_full[k], 1e-5))
        {
            assert(result == -1);
            result = k;
        }
    assert(result >= 0);
    return result;
}

std::vector<Complex> collect_response_tiles(
    const std::map<double, std::map<Vector3_Order<double>, Matz>> &tiles, const Dataset &ds)
{
    const auto &desc = ds.desc_abf;
    const int nq = ds.pbc.get_n_cells_bvk(), nf = ds.tfg.get_n_grids();
    const int n = ds.basis_aux.nb_total;
    std::vector<Complex> result(nf * nq * n * n, 0.0);
    std::vector<int> owners(result.size(), 0);
    for (const auto &[frequency, qs] : tiles)
    {
        const int f = ds.tfg.get_freq_index(frequency);
        assert(f >= 0 && f < nf);
        for (const auto &[q, tile] : qs)
        {
            const int k = response_full_q_index(ds.pbc, q);
            for (int i = 0; i != tile.nr(); ++i)
                for (int j = 0; j != tile.nc(); ++j)
                {
                    const int row = desc.indx_l2g_r(i), col = desc.indx_l2g_c(j);
                    assert(row >= 0 && row < n && col >= 0 && col < n);
                    const int index = ((f * nq + k) * n + row) * n + col;
                    result[index] += tile(i, j);
                    ++owners[index];
                }
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, result.data(), result.size(), MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                  ds.comm_h.comm);
    MPI_Allreduce(MPI_IN_PLACE, owners.data(), owners.size(), MPI_INT, MPI_SUM, ds.comm_h.comm);
    for (const int count : owners) assert(count == 1);
    return result;
}

ResponseOperators capture_response_operators(Dataset &ds, const LibrpaOptions &options)
{
    // Deliberately bounded to the current three-site, one-ABF-per-site model.
    assert(ds.basis_aux.n_atoms == 3 && ds.basis_aux.nb_total == 3);
    for (int atom = 0; atom != 3; ++atom) assert(ds.basis_aux.get_atom_nb(atom) == 1);
    assert(options.use_shrink_abfs == LIBRPA_SWITCH_OFF);
    assert(options.use_scalapack_gw_wc == LIBRPA_SWITCH_ON);
    assert(options.replace_w_head == LIBRPA_SWITCH_OFF);
    assert(ds.mf.get_n_kpoints() == ds.pbc.get_n_cells_bvk());
    initialize_ds_chi0(ds, options);
    auto &chi = *ds.p_chi0;
    std::map<Vector3_Order<double>, ComplexMatrix> empty_sinvS;
    chi.build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_AUTO, ds.cs_data, ds.atpairs_local,
              ds.basis_aux, empty_sinvS, ds.blacs_h);
    const auto view = chi.qpoint_view();  // independent snapshot before consumption
    assert(view.restore_mode == SymmetryQPointRestoreMode::NONE ||
           view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL);
    const std::map<atom_t, size_t> atom_nabf{{0, 1}, {1, 1}, {2, 1}};
    std::map<double, std::map<Vector3_Order<double>, Matz>> chi_tiles;
    for (const double f : ds.tfg.get_freq_nodes())
        for (const auto &q : view.representatives)
        {
            symmetry_atom_block_matrix_map_t local_blocks;
            std::vector<int> owners(9, 0);
            const auto f_it = chi.get_chi0_q().find(f);
            if (f_it != chi.get_chi0_q().end())
            {
                const auto q_it = f_it->second.find(q);
                if (q_it != f_it->second.end())
                    for (const auto &[i, js] : q_it->second)
                        for (const auto &[j, block] : js)
                        {
                            assert(i <= j && block.nr == 1 && block.nc == 1);
                            local_blocks[i][j] = block;  // ComplexMatrix copies deeply
                            ++owners[3 * i + j];
                            if (i != j) ++owners[3 * j + i];
                        }
            }
            const auto local_dense = build_dense_symmetry_hermitian_matrix_from_local_blocks(
                local_blocks, atom_nabf, false);  // retain raw diagonal entries
            ComplexMatrix dense(3, 3);
            allreduce_ComplexMatrix(local_dense, dense, ds.comm_h.comm);
            MPI_Allreduce(MPI_IN_PLACE, owners.data(), owners.size(), MPI_INT, MPI_SUM,
                          ds.comm_h.comm);
            for (const int count : owners) assert(count == 1);
            Matz tile(ds.desc_abf.m_loc(), ds.desc_abf.n_loc(), MAJOR::COL);
            for (int i = 0; i != tile.nr(); ++i)
                for (int j = 0; j != tile.nc(); ++j)
                    tile(i, j) = dense(ds.desc_abf.indx_l2g_r(i), ds.desc_abf.indx_l2g_c(j));
            chi_tiles[f][q] = std::move(tile);
        }
    // Isolate shared_ptr Coulomb storage so the diagnostic cannot pollute the
    // later API run. BLACS consumes chi(freq,q); chi snapshot above is independent.
    auto eps_coul = independent_coulomb_copy(options.use_fullcoul_eps ? ds.vq : ds.vq_cut);
    auto wc_coul = independent_coulomb_copy(options.use_fullcoul_wc ? ds.vq : ds.vq_cut);
    auto wc_tiles = compute_Wc_freq_q_blacs(
        chi, eps_coul, wc_coul, options.sqrt_coulomb_threshold, false, options.option_dielect_func,
        {}, nullptr, ds.blacs_h, ds.desc_abf, false, options.output_dir, options.use_cholesky_gw_wc,
        options.use_gpu_replace_scalapack, options.use_elpa_sqrt_coulomb);
    if (view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL)
    {
        for (auto &[f, qs] : chi_tiles)
            qs = restore_symmetry_dense_wq_map(qs, ds.pbc, view, ds.symmetry_context, ds.basis_aux,
                                               ds.desc_abf);
        for (auto &[f, qs] : wc_tiles)
            qs = restore_symmetry_dense_wq_map(qs, ds.pbc, view, ds.symmetry_context, ds.basis_aux,
                                               ds.desc_abf);
    }
    ResponseOperators result;
    result.chi0 = collect_response_tiles(chi_tiles, ds);
    result.wc = collect_response_tiles(wc_tiles, ds);
    ds.p_chi0.reset();  // no existing G0W0 object at this insertion point
    return result;
}

struct Observables
{
    double ec = 0.0;
    std::vector<double> exx;
    std::vector<double> qp;
    std::vector<Complex> sigma_operator;
    std::vector<Complex> exx_rspace;
    ResponseOperators response;
};

// Hold the computed G/W fixed and supply an analytic diagonal Sigma to the
// actual public band APIs. This isolates continuation failures from GW building.
void check_public_anacon_failures(librpa::Handler &handler, librpa_int::Dataset &ds,
                                  librpa::Options options)
{
    auto saved = std::move(ds.p_g0w0->sigc_diag_is_ik_f_KS);
    auto &diagonal = ds.p_g0w0->sigc_diag_is_ik_f_KS;
    const double fermi = ds.mf_band.get_efermi();
    const double saved_energy = ds.mf_band.get_eigenvals()[0](0, 0);
    const double pole = 0.5;
    ds.mf_band.get_eigenvals()[0](0, 0) = fermi + pole;
    for (const double freq : ds.tfg.get_freq_nodes())
        diagonal[0][0][freq] = {1.0 / Complex{-pole, freq}, Complex{0.2, 0}};
    options.anacon_method = LIBRPA_ANACON_AAA;
    options.n_params_anacon = 3;
    options.anacon_nfreq = -1;
    options.anacon_tfgrids_type = LIBRPA_TFGRID_UNSET;
    options.option_qpe_solver = 2;
    options.override_qpe_solver_nan = LIBRPA_SWITCH_ON;
    options.sf_sigc_omega_shift = 0;
    options.sf_gf_omega_shift = 0.1;
    // Every rank requests data, including ranks not owning the cached matrix.
    const std::vector<int> points{0};
    const std::vector<double> zero(2, 0);
    for (const bool hedin : {false, true})
    {
        options.use_hedin_shift = hedin ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
        options.istate_ref_hedin_shift = 0;
        const auto qp = handler.get_g0w0_qpe_band_k(options, 1, points, 0, 2, zero, zero);
        assert(std::isnan(qp.eqp[0]) && std::isnan(qp.sigc[0].real()));
        if (hedin) assert(std::isnan(qp.eqp[1]));
        else assert(std::abs(qp.eqp[1] - ds.mf_band.get_eigenvals()[0](0, 1) - 0.2) < 1e-12);
    }
    options.use_hedin_shift = LIBRPA_SWITCH_OFF;
    const std::vector<double> omegas{fermi + pole - 0.2, fermi + pole, fermi + pole + 0.2};
    const auto sf = handler.get_g0w0_spectral_function_with_sigc_band_k(
        options, 1, points, 0, 2, omegas, zero, zero);
    for (int state = 0; state != 2; ++state)
        for (int f = 0; f != 3; ++f)
        {
            const int index = 3 * state + f;
            if (state == 0 && f == 1)
            {
                assert(std::isnan(sf.sigc[index].real()) && std::isnan(sf.sigc[index].imag()));
                assert(std::isnan(sf.spectral_function[index]));
            }
            else
            {
                const Complex expected = state == 0 ? 1.0 / (omegas[f] - fermi - pole) : 0.2;
                assert(std::abs(sf.sigc[index] - expected) < 1e-11);
                const Complex gf_inv = Complex{omegas[f], 0.1}
                    - ds.mf_band.get_eigenvals()[0](0, state) - expected;
                assert(std::abs(sf.spectral_function[index] + (1.0 / (PI * gf_inv)).imag()) < 1e-11);
            }
        }
    ds.mf_band.get_eigenvals()[0](0, 0) = saved_energy;
    diagonal = std::move(saved);
}

Observables calculate(const Model &model, const bool symmetry, const bool irreducible_input = false,
                      const bool band_projection = false, const int switch_pattern = -1,
                      const int mesh_order = 3, const int mesh_z = 1, const int spin_source = 2)
{
    assert(mesh_order >= 3 && mesh_z >= 1);
    const int full_points = mesh_order * mesh_order * mesh_z;
    const int mesh_start = mesh_order == 3 ? -1 : 0;
    librpa::Handler handler(MPI_COMM_WORLD);
    // MeanField dimensions can be installed only once; count stars independently first.
    handler.set_scf_dimension(
        1, irreducible_input ? independent_ibz_size(mesh_order, mesh_z) : full_points, 6, 3, 2);
    handler.set_ao_basis_wfc({1, 1, 1}, {{0}, {0}, {0}});
    handler.set_ao_basis_aux({1, 1, 1}, {{0}, {0}, {0}});
    handler.set_latvec_and_G(model.lattice, model.reciprocal);
    handler.set_atoms({0, 0, 0}, model.positions);
    const auto ds = api::get_dataset_instance(handler);
    initialize_ds_atpairs_local(*ds, LIBRPA_ROUTING_LIBRI);
    handler.set_basis_convention(-1, 0, LIBRPA_ANGULAR_ORDER_ABS_PM, LIBRPA_RSH_COEFF_1_M,
                                 LIBRPA_RSH_COEFF_1_M);
    handler.set_symmetry_spin_operations(
        6, 1, model.operations.data(), nullptr, model.antiunitary.data(),
        model.rotated_spin_basis ? model.provided_spin_u.data() : nullptr, spin_source, 0);
    std::vector<double> kvecs;
    std::vector<Vector3_Order<double>> full_fractional;
    for (int a = mesh_start; a < mesh_start + mesh_order; ++a)
        for (int b = mesh_start; b < mesh_start + mesh_order; ++b)
            for (int c = 0; c != mesh_z; ++c)
            {
                full_fractional.emplace_back(double(a) / mesh_order, double(b) / mesh_order,
                                             double(c) / mesh_z);
                for (int axis = 0; axis != 3; ++axis)
                    kvecs.push_back(a * model.reciprocal[axis] / mesh_order +
                                    b * model.reciprocal[3 + axis] / mesh_order +
                                    c * model.reciprocal[6 + axis] / mesh_z);
            }
    model.check_covariance(full_fractional);
    handler.set_kgrids_kvec(mesh_order, mesh_order, mesh_z, kvecs);
    if (irreducible_input)
    {
        assert(symmetry && band_projection);
        initialize_symmetry_context(*ds, true);
        if (mesh_order == 3 && mesh_z == 1) assert(ds->symmetry_context.kstars.size() == 3);
        std::vector<double> representatives;
        std::vector<std::vector<Vector3_Order<double>>> stars;
        std::vector<int> multiplicities;
        for (const auto &star : ds->symmetry_context.kstars)
        {
            const auto cart = star.k_ibz * ds->pbc.G;
            representatives.insert(representatives.end(),
                                   {TWO_PI * cart.x, TWO_PI * cart.y, TWO_PI * cart.z});
            stars.emplace_back();
            for (const auto &member : star.members) stars.back().push_back(member.k_bz * ds->pbc.G);
            multiplicities.push_back(star.members.size());
        }
        std::sort(multiplicities.begin(), multiplicities.end());
        if (mesh_order == 3 && mesh_z == 1) assert((multiplicities == std::vector<int>{1, 2, 6}));
        assert(representatives.size() / 3 == independent_ibz_size(mesh_order, mesh_z));
        ds->pbc.set_irreducible_kgrids_kvec(mesh_order, mesh_order, mesh_z, representatives, stars);
    }
    const int loaded_kpoints = ds->mf.get_n_kpoints();
    std::vector<double> energies(6 * loaded_kpoints), occupations(6 * loaded_kpoints, 0.0);
    std::vector<Complex> up(18), down(18);
    double occupied_max = -1e10, empty_min = 1e10;
    std::vector<Matz> full_vectors;
    std::vector<std::vector<double>> full_values;
    for (const auto &k : full_fractional)
    {
        std::vector<double> values;
        Matz vectors;
        model.diagonalize(k, values, vectors);
        occupied_max = std::max(occupied_max, values[2]);
        empty_min = std::min(empty_min, values[3]);
        full_vectors.push_back(std::move(vectors));
        full_values.push_back(std::move(values));
    }
    assert(empty_min - occupied_max > 1.0);
    const double fermi = 0.5 * (occupied_max + empty_min);
    for (int k = 0; k != loaded_kpoints; ++k)
    {
        std::vector<double> values;
        Matz vectors;
        model.diagonalize(ds->pbc.kfrac_list.at(k), values, vectors);
        for (int band = 0; band != 6; ++band)
        {
            energies[6 * k + band] = values[band];
            occupations[6 * k + band] = band < 3 ? loaded_kpoints * ds->pbc.weight_k.at(k) : 0.0;
            for (int ao = 0; ao != 3; ++ao)
            {
                up[3 * band + ao] = vectors(ao, band);
                down[3 * band + ao] = vectors(ao + 3, band);
            }
        }
        handler.set_wfc_spinor_packed(k, 6, 3, up.data(), down.data());
        // Atom-pair Coulomb inputs are distributed; each block has one owner.
        for (int i = 0; i != 3; ++i)
            for (int j = i; j != 3; ++j)
                if (std::find(ds->atpairs_local.begin(), ds->atpairs_local.end(), atpair_t{i, j}) !=
                    ds->atpairs_local.end())
                {
                    // A positive invariant q_z factor gives V(R_z=+/-1) != 0,
                    // so the antiunitary real-space EXX restore is numerically active.
                    const double modulation =
                        model.dispersive && mesh_z > 1
                            ? 1.0 + 0.2 * std::cos(TWO_PI * ds->pbc.kfrac_list.at(k).z)
                            : 1.0;
                    assert(modulation >= 0.8 - 1e-14);
                    const Complex interaction = (i == j ? 1.0 : 0.15) * modulation;
                    handler.set_aux_bare_coulomb_k_atom_pair_packed(k, i, j, 1, 1, &interaction,
                                                                    0.0);
                    handler.set_aux_cut_coulomb_k_atom_pair_packed(k, i, j, 1, 1, &interaction,
                                                                   0.0);
                }
    }
    handler.set_wg_ekb_efermi(1, loaded_kpoints, 6, occupations.data(), energies.data(), fermi);
    double electrons = 0.0;
    for (int k = 0; k != loaded_kpoints; ++k)
        for (int band = 0; band != 6; ++band)
        {
            const double weight = ds->mf.get_weight()[0](k, band);
            assert(std::abs(weight - (band < 3 ? ds->pbc.weight_k.at(k) : 0.0)) < 1e-14);
            electrons += weight;
        }
    assert(std::abs(electrons - 3.0) < 1e-14);
    if (symmetry)
    {
        initialize_symmetry_context(*ds, true);
        const auto layouts =
            ds->basis_wfc.build_species_basis_layouts(ds->symmetry_context.atom_to_type);
        auto targets = irreducible_input ? build_symmetry_kstar_member_kfrac_targets(
                                               ds->symmetry_context, ds->pbc)
                                         : build_symmetry_full_grid_kstar_member_kfrac_targets(
                                               ds->symmetry_context, ds->pbc.kfrac_list);
        const auto representative_indices =
            irreducible_input ? symmetry_kstar_representative_indices_t{}
                              : build_symmetry_full_grid_kstar_representative_indices(
                                    ds->symmetry_context, ds->pbc.kfrac_list);
        double density_error = 0.0, nonorigin_density = 0.0, anti_density = 0.0,
               self_inverse_density = 0.0;
        std::size_t anti_members = 0, covered_members = 0;
        for (const auto &pair : ds->symmetry_context.rspace_sector_stars)
            for (const auto &sector : pair.second)
                for (const auto &member : sector.second)
                {
                    ++covered_members;
                    if (symmetry_rspace_restore_member_is_antiunitary(ds->symmetry_context, member))
                    {
                        ++anti_members;
                        Complex expected = 0.0;
                        for (int k = 0; k != full_points; ++k)
                            for (int band = 0; band != 3; ++band)
                                expected +=
                                    std::polar(1.0 / full_points,
                                               -TWO_PI * (full_fractional[k] * member.full_R)) *
                                    full_vectors[k](member.full_atom_pair.first, band) *
                                    std::conj(full_vectors[k](member.full_atom_pair.second, band));
                        anti_density = std::max(anti_density, std::abs(expected));
                    }
                }
        assert(covered_members == 9 * full_points);
        for (const auto &R : ds->pbc.Rlist)
        {
            const auto restored = get_symmetry_restored_dmat_cplx_R_spinor(
                ds->symmetry_context, layouts, ds->mf, 0, ds->pbc.kfrac_list, R,
                ds->basis_wfc.get_atom_nb_map(), &targets,
                irreducible_input ? nullptr : &representative_indices);
            const ComplexMatrix *channels[4]{&restored.b00, &restored.b01, &restored.b10,
                                             &restored.b11};
            for (int channel = 0; channel != 4; ++channel)
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j)
                    {
                        Complex expected = 0.0;
                        for (int k = 0; k != full_points; ++k)
                            for (int band = 0; band != 3; ++band)
                                expected += std::polar(1.0 / full_points,
                                                       -TWO_PI * (full_fractional[k] * R)) *
                                            full_vectors[k](i + 3 * (channel / 2), band) *
                                            std::conj(full_vectors[k](j + 3 * (channel % 2), band));
                        if (R != Vector3_Order<int>{0, 0, 0})
                            nonorigin_density = std::max(nonorigin_density, std::abs(expected));
                        if (mesh_z % 2 == 0 && R.x == 0 && R.y == 0 && R.z == -mesh_z / 2)
                            self_inverse_density =
                                std::max(self_inverse_density, std::abs(expected));
                        const double error = std::abs((*channels[channel])(i, j) - expected);
                        if (error > density_error)
                        {
                            density_error = error;
                            if (global::mpi_comm_global_h.is_root())
                                std::cout << "IBZ density boundary: R=" << R
                                          << " channel=" << channel << " i=" << i << " j=" << j
                                          << " expected=" << expected
                                          << " restored=" << (*channels[channel])(i, j)
                                          << " residual=" << error << std::endl;
                        }
                    }
        }
        if (global::mpi_comm_global_h.is_root())
            std::cout << "Independent full-DFT density residual: " << density_error << std::endl;
        if (global::mpi_comm_global_h.is_root())
            std::cout << "Real-space coverage: full=" << covered_members << " anti=" << anti_members
                      << " nonorigin_density=" << nonorigin_density
                      << " anti_density=" << anti_density
                      << " self_inverse_density=" << self_inverse_density << std::endl;
        if (model.dispersive && mesh_z > 1)
            assert(anti_members > 0 && nonorigin_density > 1e-5 && anti_density > 1e-5);
        if (model.dispersive && mesh_z % 2 == 0) assert(self_inverse_density > 1e-8);
        assert(density_error < 1e-12);
    }
    const int origin[3]{0, 0, 0};
    const double coefficient = 0.5;  // two equivalent RI centers for an on-site product
    for (int atom = 0; atom != 3; ++atom)
        if (atom % global::mpi_comm_global_h.nprocs == global::mpi_comm_global_h.myid)
            handler.set_lri_coeff(LIBRPA_ROUTING_LIBRI, atom, atom, 1, 1, 1, origin, &coefficient);

    librpa::Options options;
    options.parallel_routing = LIBRPA_ROUTING_LIBRI;
    options.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    options.nfreq = 6;
    options.n_params_anacon = 6;
    options.minimax_emin = 1.0;
    options.minimax_emax = 4.0;
    options.use_shrink_abfs = LIBRPA_SWITCH_OFF;
    options.use_shrink_chi = LIBRPA_SWITCH_OFF;
    options.option_qpe_solver = 2;
    options.sqrt_coulomb_threshold = 0.0;
    const int pattern = switch_pattern >= 0 ? switch_pattern : (symmetry ? 7 : 0);
    options.use_symmetry_exx = pattern & 4 ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
    options.use_symmetry_rpa = pattern & 2 ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
    options.use_symmetry_gw = pattern & 1 ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
    options.set_output_dir(symmetry ? "model-magnetic" : "model-full");

    Observables output;
    std::vector<Complex> ec_contributions(loaded_kpoints);
    output.ec = handler.get_rpa_correlation_energy(options, ec_contributions);
    // The RPA API publishes the energy on the global root.
    MPI_Bcast(&output.ec, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (options.use_symmetry_rpa == LIBRPA_SWITCH_ON)
    {
        const auto ds = api::get_dataset_instance(handler);
        assert(ds->symmetry_context.count_kstar_members() == full_points);
        assert(ds->symmetry_context.count_irreducible_blocks() < 3 * 3 * full_points);
        assert(rspace_symmetry_has_complete_band_space(ds->mf, -1));
    }
    if (model.dispersive && !model.rotated_spin_basis && mesh_order == 3 && mesh_z == 3 &&
        !irreducible_input && switch_pattern < 0)
        output.response = capture_response_operators(*ds, options);
    handler.build_exx(options);
    if (model.dispersive && mesh_z > 1)
    {
        // Native EXX stores the positive D*V contraction; projection applies
        // its minus sign. Collect the physical -D*V operator before projection.
        const auto &Rs = ds->pbc.Rlist;
        std::map<Vector3_Order<int>, std::size_t> r_indices;
        for (std::size_t ir = 0; ir != Rs.size(); ++ir) r_indices.emplace(Rs[ir], ir);
        const auto index_of =
            [&r_indices](const int channel, const int i, const int j, const Vector3_Order<int> &R)
        { return (r_indices.at(R) * 4 + channel) * 9 + 3 * i + j; };
        output.exx_rspace.assign(36 * Rs.size(), 0.0);
        std::vector<int> owners(output.exx_rspace.size(), 0);
        for (const auto &[spin, bras] : ds->p_exx->exx_IJR_cplx)
            for (const auto &[bra, kets] : bras)
                for (const auto &[ket, atoms_i] : kets)
                    for (const auto &[i, atoms_j] : atoms_i)
                        for (const auto &[j, cells] : atoms_j)
                            for (const auto &[R, block] : cells)
                            {
                                assert(spin == 0 && block.nr() == 1 && block.nc() == 1);
                                const auto index = index_of(2 * bra + ket, i, j, R);
                                output.exx_rspace[index] -= block(0, 0);
                                ++owners[index];
                            }
        MPI_Allreduce(MPI_IN_PLACE, output.exx_rspace.data(), output.exx_rspace.size(),
                      MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, owners.data(), owners.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        const auto modulo = [](int value, int n) { return (value % n + n) % n; };
        double oracle_error = 0.0;
        int nonzero_entries = 0, max_owners = 0;
        int min_nonzero_contributors = global::mpi_comm_global_h.nprocs + 1;
        int max_nonzero_contributors = 0;
        for (const auto &R : Rs)
        {
            // Exact finite-mesh Fourier coefficients of 1+0.2*cos(2*pi*q_z).
            const double vr = modulo(R.x, mesh_order) == 0 && modulo(R.y, mesh_order) == 0
                                  ? (modulo(R.z, mesh_z) == 0 ? 1.0 : 0.0) +
                                        (modulo(R.z, mesh_z) == 1 ? 0.1 : 0.0) +
                                        (modulo(R.z, mesh_z) == mesh_z - 1 ? 0.1 : 0.0)
                                  : 0.0;
            for (int channel = 0; channel != 4; ++channel)
                for (int i = 0; i != 3; ++i)
                    for (int j = 0; j != 3; ++j)
                    {
                        Complex density = 0.0;
                        for (int k = 0; k != full_points; ++k)
                            for (int band = 0; band != 3; ++band)
                                density += std::polar(1.0 / full_points,
                                                      -TWO_PI * (full_fractional[k] * R)) *
                                           full_vectors[k](i + 3 * (channel / 2), band) *
                                           std::conj(full_vectors[k](j + 3 * (channel % 2), band));
                        const Complex expected = -(i == j ? 1.0 : 0.15) * vr * density;
                        const auto index = index_of(channel, i, j, R);
                        oracle_error =
                            std::max(oracle_error, std::abs(output.exx_rspace[index] - expected));
                        max_owners = std::max(max_owners, owners[index]);
                        // Hs are additive native contributions, rather than a
                        // BLACS tile with a unique element owner. Exact zeros
                        // may have structural buffers on several ranks. The
                        // independent oracle checks the globally summed value.
                        if (std::abs(expected) > 1e-8)
                        {
                            ++nonzero_entries;
                            assert(owners[index] > 0);
                            min_nonzero_contributors =
                                std::min(min_nonzero_contributors, owners[index]);
                            max_nonzero_contributors =
                                std::max(max_nonzero_contributors, owners[index]);
                        }
                    }
        }
        int anti_nonzero_entries = 0;
        double anti_exx_norm = 0.0;
        if (options.use_symmetry_exx == LIBRPA_SWITCH_ON)
            for (const auto &pair : ds->symmetry_context.rspace_sector_stars)
                for (const auto &sector : pair.second)
                    for (const auto &member : sector.second)
                        if (symmetry_rspace_restore_member_is_antiunitary(ds->symmetry_context,
                                                                          member))
                            for (int channel = 0; channel != 4; ++channel)
                            {
                                const auto index =
                                    index_of(channel, member.full_atom_pair.first,
                                             member.full_atom_pair.second, member.full_R);
                                const double norm = std::abs(output.exx_rspace[index]);
                                anti_exx_norm = std::max(anti_exx_norm, norm);
                                if (norm > 1e-8) ++anti_nonzero_entries;
                            }
        if (global::mpi_comm_global_h.is_root())
            std::cout << "Independent real-space EXX oracle: residual=" << oracle_error
                      << " nonzero=" << nonzero_entries << " max_contributors=" << max_owners
                      << " nonzero_contributors=" << min_nonzero_contributors << ':'
                      << max_nonzero_contributors
                      << " anti_restore=" << (options.use_symmetry_exx == LIBRPA_SWITCH_ON)
                      << " anti_nonzero=" << anti_nonzero_entries << " anti_norm=" << anti_exx_norm
                      << std::endl;
        assert(oracle_error < 1e-12 && nonzero_entries > 0);
        if (options.use_symmetry_exx == LIBRPA_SWITCH_ON)
            assert(anti_nonzero_entries > 0 && anti_exx_norm > 1e-6);
    }
    std::vector<int> points;
    if (global::mpi_comm_global_h.is_root())
    {
        points.resize(full_points);
        std::iota(points.begin(), points.end(), 0);
    }
    if (band_projection)
    {
        std::vector<double> fractional_points, band_energies(6 * full_points),
            band_occupations(6 * full_points, 0.0);
        for (const auto &k : full_fractional)
            fractional_points.insert(fractional_points.end(), {k.x, k.y, k.z});
        handler.set_band_kvec(full_points, fractional_points.data());
        for (int k = 0; k != full_points; ++k)
            for (int band = 0; band != 6; ++band)
            {
                band_energies[6 * k + band] = full_values[k][band];
                band_occupations[6 * k + band] = band < 3 ? 1.0 : 0.0;
            }
        handler.set_band_occ_eigval(1, full_points, 6, band_occupations.data(),
                                    band_energies.data());
        for (int k = 0; k != full_points; ++k)
        {
            for (int band = 0; band != 6; ++band)
                for (int ao = 0; ao != 3; ++ao)
                {
                    up[3 * band + ao] = full_vectors[k](ao, band);
                    down[3 * band + ao] = full_vectors[k](ao + 3, band);
                }
            handler.set_wfc_band_spinor_packed(k, 6, 3, up.data(), down.data());
        }
        output.exx = handler.get_exx_pot_band_k(options, 1, points, 0, 6);
    }
    else
        output.exx = handler.get_exx_pot_kgrid(options, 1, points, 0, 6);
    handler.build_g0w0_sigma(options);
    // Retain the matrices already used for projection, without writing files.
    ds->p_g0w0->output_sigc_ks_mat_kf = true;
    const auto vxc = std::vector<double>(output.exx.size(), 0.0);
    const auto qp = band_projection
                        ? handler.get_g0w0_qpe_band_k(options, 1, points, 0, 6, vxc, output.exx)
                        : handler.get_g0w0_qpe_kgrid(options, 1, points, 0, 6, vxc, output.exx);
    output.qp = qp.eqp;
    output.sigma_operator.assign(full_points * 6 * 6 * 6, 0.0);
    std::vector<int> owners(output.sigma_operator.size(), 0);
    const auto &desc = ds->p_g0w0->desc_sigc_is_ik_f_KS;
    for (const auto &[spin, kpoints] : ds->p_g0w0->sigc_is_ik_f_KS)
        for (const auto &[kpoint, frequencies] : kpoints)
            for (const auto &[frequency, block] : frequencies)
            {
                assert(spin == 0);
                const int ifreq = ds->tfg.get_freq_index(frequency);
                for (int i = 0; i != block.nr(); ++i)
                    for (int j = 0; j != block.nc(); ++j)
                    {
                        const int row = desc.indx_l2g_r(i), col = desc.indx_l2g_c(j);
                        assert(row >= 0 && col >= 0);
                        const int index = ((kpoint * 6 + ifreq) * 6 + row) * 6 + col;
                        output.sigma_operator.at(index) = block(i, j);
                        owners.at(index) += 1;
                    }
            }
    MPI_Allreduce(MPI_IN_PLACE, output.sigma_operator.data(), output.sigma_operator.size(),
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, owners.data(), owners.size(), MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    for (const int count : owners) assert(count == 1);
    static bool checked_continuation = false;
    if (!checked_continuation && band_projection && !symmetry)
    {
        check_public_anacon_failures(handler, *ds, options);
        checked_continuation = true;
    }
    return output;
}

template <typename T>
double difference(const std::vector<T> &a, const std::vector<T> &b)
{
    assert(a.size() == b.size());
    double residual = 0.0;
    for (std::size_t index = 0; index != a.size(); ++index)
    {
        assert(std::isfinite(std::abs(a[index])) && std::isfinite(std::abs(b[index])));
        residual = std::max(residual, std::abs(a[index] - b[index]));
    }
    double maximum = 0.0;
    MPI_Allreduce(&residual, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return maximum;
}

void check_dispersive_model(const bool rotated, const int source, const int mesh, const int mesh_z)
{
    const Model dispersive(true, rotated);
    const auto full = calculate(dispersive, false, false, true, -1, mesh, mesh_z, source);
    double sigma_norm = 0.0;
    for (const auto value : full.sigma_operator) sigma_norm = std::max(sigma_norm, std::abs(value));
    assert(sigma_norm > 1e-8 && std::isfinite(full.ec) && std::abs(full.ec) > 1e-8);
    const auto magnetic = calculate(dispersive, true, false, true, -1, mesh, mesh_z, source);
    std::vector<Observables> comparisons{magnetic};
    if (mesh == 3)
        comparisons.push_back(calculate(dispersive, true, true, true, -1, mesh, mesh_z, source));
    for (const auto &result : comparisons)
    {
        const auto *reduced = &result;
        const double ec = std::abs(full.ec - reduced->ec);
        const double exx = difference(full.exx, reduced->exx);
        const double exx_r = difference(full.exx_rspace, reduced->exx_rspace);
        const double sigma = difference(full.sigma_operator, reduced->sigma_operator);
        const double qp = difference(full.qp, reduced->qp);
        double chi_error = 0.0, wc_error = 0.0;
        if (!reduced->response.chi0.empty())
        {
            assert(full.response.chi0.size() == 6 * 27 * 9 &&
                   full.response.wc.size() == 6 * 27 * 9);
            chi_error = difference(full.response.chi0, reduced->response.chi0);
            wc_error = difference(full.response.wc, reduced->response.wc);
            double chi_norm = 0.0, wc_norm = 0.0;
            for (const auto value : full.response.chi0)
                chi_norm = std::max(chi_norm, std::abs(value));
            for (const auto value : full.response.wc) wc_norm = std::max(wc_norm, std::abs(value));
            assert(chi_norm > 1e-8 && wc_norm > 1e-8);
            if (global::mpi_comm_global_h.is_root())
                std::cout << "Complete response operator residuals: Chi0=" << chi_error
                          << " Wc=" << wc_error << " elements=" << full.response.chi0.size()
                          << " Chi0_norm=" << chi_norm << " Wc_norm=" << wc_norm << " owners=1"
                          << std::endl;
            assert(chi_error < 1e-12 && wc_error < 1e-12);
        }
        if (global::mpi_comm_global_h.is_root())
            std::cout << "Dispersive model mesh=" << mesh << "x" << mesh << "x" << mesh_z
                      << " rotated=" << rotated << " source=" << source << " Ec=" << ec
                      << " EXX=" << exx << " EXX_R=" << exx_r << " Sigma=" << sigma << " QP=" << qp
                      << std::endl;
        assert(ec < 1e-11 && exx < 1e-11 && exx_r < 1e-12 && sigma < 1e-11 && qp < 1e-10);
    }
    // Isolate the six mixed stage settings on this nonzero-R model; the
    // full and magnetic calculations above cover the two endpoint settings.
    if (!rotated && mesh == 3 && mesh_z == 3)
        for (int pattern = 1; pattern != 7; ++pattern)
        {
            const auto split =
                calculate(dispersive, true, false, true, pattern, mesh, mesh_z, source);
            const double ec = std::abs(full.ec - split.ec);
            const double exx = difference(full.exx, split.exx);
            const double exx_r = difference(full.exx_rspace, split.exx_rspace);
            const double sigma = difference(full.sigma_operator, split.sigma_operator);
            const double qp = difference(full.qp, split.qp);
            if (global::mpi_comm_global_h.is_root())
                std::cout << "Dispersive switch pattern " << pattern << " residuals (Ha): Ec=" << ec
                          << " EXX=" << exx << " EXX_R=" << exx_r << " Sigma=" << sigma
                          << " QP=" << qp << std::endl;
            assert(ec < 1e-11 && exx < 1e-11 && exx_r < 1e-12 && sigma < 1e-11 && qp < 1e-10);
        }
}

}  // namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    librpa::init_global(LIBRPA_SWITCH_OFF, "stdout", LIBRPA_SWITCH_OFF);
    {
        if (argc > 1)
        {
            const bool rotated = std::string(argv[1]) == "--provided-soc";
            const int source = argc > 2 ? std::stoi(argv[2]) : 2;
            const int mesh = argc > 3 ? std::stoi(argv[3]) : 3;
            check_dispersive_model(rotated, source, mesh, mesh);
        }
        else
        {
            // The odd grid exercises all stage combinations and IBZ input;
            // the even grid adds self-inverse cells, and the rotated frame
            // distinguishes supplied SOC actions from spatially derived ones.
            check_dispersive_model(false, 2, 3, 3);
            check_dispersive_model(false, 2, 4, 4);
            check_dispersive_model(true, 2, 3, 3);
        }
    }
    librpa::finalize_global();
    MPI_Finalize();
}
