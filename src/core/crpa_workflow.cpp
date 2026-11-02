#include "crpa_workflow.h"

#include <algorithm>
#include <cmath>
#include <memory>

#include "../io/global_io.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/error.h"
#include "../utils/profiler.h"
#include "epsilon.h"
#include "pair_density_vertex.h"
#include "../math/utils_matrix_m_mpi.h"

namespace librpa_int
{
namespace
{

// Keep the timing hierarchy balanced even when collective validation throws.
struct CrpaTimer
{
    const char *name;
    explicit CrpaTimer(const char *name, const char *description) : name(name)
    {
        global::profiler.start(name, description);
    }
    ~CrpaTimer() { global::profiler.stop(name); }
};

// A borrowed resident P0 may alias the map returned by get_chi0_q().
// Move its storage aside while a stage uses Chi0's native response machinery.
struct CrpaChi0State
{
    Chi0 &chi;
    Chi0::BandSelection selection;
    Chi0QMap response;

    explicit CrpaChi0State(Chi0 &chi)
        : chi(chi), selection(chi.get_band_selection()), response(chi.take_chi0_q())
    {
    }
    ~CrpaChi0State()
    {
        chi.swap_chi0_q(response);
        chi.swap_band_selection(selection);
    }
};

void require_collectively(bool valid, const MpiCommHandler &comm, const char *message)
{
    const int failed = !valid;
    int any_failed = 0;
    comm.allreduce(&failed, &any_failed, 1, MPI_MAX);
    if (any_failed) throw LIBRPA_INVALID_ARGUMENT(message);
}

bool finite(const ComplexMatrix &matrix)
{
    for (int i = 0; i != matrix.size; ++i)
        if (!std::isfinite(matrix.c[i].real()) || !std::isfinite(matrix.c[i].imag())) return false;
    return true;
}

ComplexMatrix flatten_vertex(const SitePairDensityVertex &vertex, const Vector3_Order<double> &q,
                             const AtomicBasis &basis)
{
    ComplexMatrix result(vertex.n_orbitals * vertex.n_orbitals, basis.nb_total);
    const auto &offsets = basis.get_part_range();
    for (const auto &[atom, block] : vertex.q_blocks.at(q))
        for (int pair = 0; pair != block.nr; ++pair)
            for (int mu = 0; mu != block.nc; ++mu)
                result(pair, offsets.at(atom) + mu) = block(pair, mu);
    return result;
}

// Reconstruct only one q at a time. Original producer blocks stay untouched.
ComplexMatrix retained_coulomb(const CrpaContext &context, const Vector3_Order<double> &q)
{
    const auto &basis = context.chi0.atbasis_abf;
    const auto &comm = context.chi0.comm_h;
    const auto &offsets = basis.get_part_range();
    const int n = basis.nb_total;
    ComplexMatrix local(n, n), total(n, n);
    std::vector<int> owners(basis.n_atoms * basis.n_atoms, 0), all_owners(owners.size());
    bool valid = true;
    for (const auto &[i, js] : context.coulomb)
        for (const auto &[j, qs] : js)
        {
            if (i < 0 || j < i || j >= basis.n_atoms)
            {
                valid = false;
                continue;
            }
            const auto it = qs.find(q);
            if (it == qs.end() || !it->second)
            {
                valid = false;
                continue;
            }
            const auto &block = *it->second;
            if (block.nr != basis.get_atom_nb(i) || block.nc != basis.get_atom_nb(j) ||
                !finite(block))
            {
                valid = false;
                continue;
            }
            owners[i * basis.n_atoms + j] = 1;
            for (int a = 0; a != block.nr; ++a)
                for (int b = 0; b != block.nc; ++b)
                {
                    local(offsets.at(i) + a, offsets.at(j) + b) = block(a, b);
                    if (i != j)
                        local(offsets.at(j) + b, offsets.at(i) + a) = std::conj(block(a, b));
                }
        }
    require_collectively(valid, comm, "cRPA: invalid Coulomb block or missing q");
    comm.allreduce(owners.data(), all_owners.data(), owners.size(), MPI_SUM);
    for (int i = 0; i != basis.n_atoms; ++i)
        for (int j = i; j != basis.n_atoms; ++j)
            if (basis.get_atom_nb(i) && basis.get_atom_nb(j))
                valid = valid && all_owners[i * basis.n_atoms + j] == 1;
    require_collectively(valid, comm, "cRPA: Coulomb blocks need exactly one MPI owner");
    comm.reduce(local.c, total.c, total.size, 0, MPI_SUM);
    std::string error;
    if (comm.is_root())
    {
        try
        {
            // Same lambda >= threshold convention as native sqrt(v).
            // Power one retains the bare interaction in precisely that subspace.
            total = power_hemat(total, 1.0, false, true, context.sqrt_coulomb_threshold);
            if (!finite(total)) error = "non-finite retained Coulomb";
        }
        catch (const std::exception &e)
        {
            error = e.what();
        }
    }
    require_collectively(error.empty(), comm, "cRPA: retained Coulomb construction failed");
    comm.bcast(total.c, total.size, 0);
    return total;
}

atpair_k_cplx_mat_t copy_coulomb(const atpair_k_cplx_mat_t &input)
{
    atpair_k_cplx_mat_t result;
    for (const auto &[i, js] : input)
        for (const auto &[j, qs] : js)
            for (const auto &[q, block] : qs)
                result[i][j][q] = std::make_shared<ComplexMatrix>(*block);
    return result;
}

}  // namespace

ComplexMatrix contract_crpa_blacs(const ComplexMatrix &left_vertex,
                                  const ComplexMatrix &right_vertex, const Matz &kernel,
                                  const ArrayDesc &descriptor, int n_left, int n_right,
                                  const MpiCommHandler &comm)
{
    require_collectively(
        descriptor.initialized() && n_left > 0 && n_right > 0 &&
            left_vertex.nr == n_left * n_left && right_vertex.nr == n_right * n_right &&
            left_vertex.nc == descriptor.m() && right_vertex.nc == descriptor.n() &&
            kernel.nr() == descriptor.m_loc() && kernel.nc() == descriptor.n_loc(),
        comm, "cRPA: inconsistent distributed contraction dimensions");
    ComplexMatrix local(n_left * n_left, n_right * n_right);
    bool valid = finite(left_vertex) && finite(right_vertex);
    for (int row = 0; row != kernel.nr(); ++row)
        for (int col = 0; col != kernel.nc(); ++col)
            valid = valid && std::isfinite(kernel(row, col).real()) &&
                    std::isfinite(kernel(row, col).imag());
    require_collectively(valid, comm, "cRPA: non-finite distributed contraction input");
    if (kernel.nr() && kernel.nc())
    {
        ComplexMatrix left(left_vertex.nr, kernel.nr());
        ComplexMatrix right(right_vertex.nr, kernel.nc());
        ComplexMatrix block(kernel.nr(), kernel.nc());
        for (int row = 0; row != kernel.nr(); ++row)
        {
            for (int pair = 0; pair != left.nr; ++pair)
                left(pair, row) = left_vertex(pair, descriptor.indx_l2g_r(row));
            for (int col = 0; col != kernel.nc(); ++col) block(row, col) = kernel(row, col);
        }
        for (int col = 0; col != kernel.nc(); ++col)
            for (int pair = 0; pair != right.nr; ++pair)
                right(pair, col) = right_vertex(pair, descriptor.indx_l2g_c(col));
        local = contract_rectangular_ordered_pair_vertex(left, block, right, n_left, n_right);
    }
    ComplexMatrix result(local.nr, local.nc);
    comm.allreduce(local.c, result.c, result.size, MPI_SUM);
    return result;
}

CrpaPreparedModel prepare_crpa_model(const AtomicBasis &basis, const BasisConvention &convention,
                                     const std::vector<int> &atom_types,
                                     const std::vector<std::string> &species_labels,
                                     const std::vector<ComplexMatrix> &spatial_overlap,
                                     const PeriodicBoundaryData &pbc, const MeanField &meanfield,
                                     const CrpaWindowOptions &options)
{
    CrpaPreparedModel prepared;
    prepared.windows = build_crpa_window_input(basis, convention, atom_types, species_labels,
                                               spatial_overlap, pbc.kfrac_list, meanfield, options);
    prepared.selection.resize(meanfield.get_n_spins());
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
    {
        auto frame = std::make_unique<CorrelatedSubspace>(
            prepared.windows.sites, pbc.kfrac_list, pbc.Rlist,
            meanfield.get_n_aos() * meanfield.get_n_spinor(), meanfield.get_n_bands(), 1,
            options.gram_abs, options.gram_rel, options.gram_cond_max, options.s_abs, options.s_rel,
            options.s_cond_max, options.residual_tol, options.t_roundtrip_tol);
        prepared.selection[spin].resize(meanfield.get_n_kpoints());
        for (int k = 0; k < meanfield.get_n_kpoints(); ++k)
        {
            frame->set_S_k(k, crpa_spinor_overlap(spatial_overlap[k], meanfield.get_n_spinor()));
            frame->set_W_k(k, prepared.windows.orbitals_spin_k[spin][k]);
            if (options.use_orbital_weights)
                prepared.selection[spin][k] = prepared.windows.response_weights[spin][k];
            else
            {
                prepared.selection[spin][k].assign(meanfield.get_n_bands(), 0);
                for (const int band : prepared.windows.response_bands[spin][k])
                    prepared.selection[spin][k][band] = 1;
            }
        }
        frame->compute_T_R();
        prepared.frames.push_back(std::move(frame));
    }
    return prepared;
}

CrpaLocalProjection prepare_crpa_local_projection(CrpaContext &context,
                                                  const Chi0::BandSelection &selection,
                                                  const std::vector<CorrelatedSubspace *> &frames)
{
    auto &chi = context.chi0;
    const auto &comm = chi.comm_h;
    const auto &basis = chi.atbasis_abf;
    const auto &desc = context.active_auxiliary_descriptor;
    require_collectively(
        !frames.empty() && frames.size() == chi.mf.get_n_spins() && desc.initialized() &&
            desc.m() == basis.nb_total && desc.n() == basis.nb_total &&
            chi.mf.get_n_kpoints() == chi.pbc.get_n_cells_bvk() &&
            (!chi.use_symmetry_context || chi.qstar_only_symmetry) &&
            std::isfinite(context.sqrt_coulomb_threshold) && context.sqrt_coulomb_threshold >= 0.0,
        comm, "cRPA requires matching spin frames, full q grid and active descriptor");
    for (const auto *frame : frames)
        require_collectively(frame != nullptr, comm, "cRPA: null output spin frame");
    // Validate the requested mask before doing an expensive full-response build.
    CrpaChi0State response_state(chi);
    chi.set_band_selection(selection);
    require_collectively(!selection.empty(), comm,
                         "cRPA requires an explicit band-selection table");
    chi.clear_band_selection();

    CrpaLocalProjection projection;
    auto &result = projection.bare_result;
    result.sites = frames.front()->get_sites();
    result.frequencies = chi.tfg.get_freq_nodes();
    require_collectively(!result.sites.empty() && !result.frequencies.empty() &&
                             std::all_of(result.frequencies.begin(), result.frequencies.end(),
                                         [](double f) { return std::isfinite(f) && f > 0.0; }),
                         comm, "cRPA requires output sites and original positive frequency nodes");
    const auto &qs = chi.pbc.klist;
    const double q_weight = 1.0 / chi.pbc.get_n_cells_bvk();
    require_collectively(qs.size() == chi.pbc.get_n_cells_bvk(), comm,
                         "cRPA requires complete original q points for orbital contraction");
    global::lib_printf_root(
        "cRPA: screening %zu representative q points; contracting %zu full q points\n",
        chi.active_qpoints().size(), qs.size());

    // Build and transform each physical spin's output frame once.
    auto &vertices = projection.vertices;
    {
        CrpaTimer vertex_timer("crpa_vertices",
                               "Build and transform physical-spin pair-density vertices");
        for (auto *frame : frames)
        {
            require_collectively(frame->get_sites().size() == result.sites.size(), comm,
                                 "cRPA: spin frames have different site counts");
            for (int site = 0; site != result.sites.size(); ++site)
                require_collectively(
                    frame->get_sites()[site].n_orbitals == result.sites[site].n_orbitals &&
                        frame->get_sites()[site].atom_index == result.sites[site].atom_index,
                    comm, "cRPA: spin frames refer to different sites or orbital counts");
            PairDensityVertex parent(*frame, context.coefficients, chi.atbasis_wfc,
                                     context.parent_auxiliary_basis, chi.pbc, qs, comm);
            vertices.emplace_back();
            for (const auto &site : parent.sites())
            {
                const auto active = context.transform_auxiliary_basis
                                        ? transform_site_pair_density_vertex_auxiliary_basis(
                                              site, context.parent_auxiliary_basis, basis,
                                              context.auxiliary_transform)
                                        : site;
                vertices.back().emplace_back();
                for (const auto &q : qs)
                    vertices.back().back().emplace(q, flatten_vertex(active, q, basis));
            }
        }
    }
    for (int site = 0; site != result.sites.size(); ++site)
        for (int left = 0; left != frames.size(); ++left)
            for (int right = 0; right != frames.size(); ++right)
            {
                const int pairs = result.sites[site].n_orbitals * result.sites[site].n_orbitals;
                result.tensors.push_back({site, left, right, ComplexMatrix(pairs, pairs), {}, {}});
            }
    if (chi.qpoint_view().restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL)
    {
        // IBZ producers need only supply representative bare Coulomb blocks.
        // Filter once in the active basis, then restore that same subspace.
        std::map<Vector3_Order<double>, Matz> representative;
        for (const auto &q : chi.active_qpoints())
        {
            const auto bare = retained_coulomb(context, q);
            const Matz dense(bare.nr, bare.nc, bare.c, MAJOR::ROW, MAJOR::COL);
            representative.emplace(q, get_local_mat(dense, desc));
        }
        const auto bare = restore_symmetry_dense_wq_map(representative, chi.pbc, chi.qpoint_view(),
                                                        chi.symmetry_context, basis, desc);
        for (const auto &q : qs)
            for (auto &tensor : result.tensors)
            {
                const int n = result.sites[tensor.site_index].n_orbitals;
                tensor.bare += q_weight * contract_crpa_blacs(
                                              vertices[tensor.spin_left][tensor.site_index].at(q),
                                              vertices[tensor.spin_right][tensor.site_index].at(q),
                                              bare.at(q), desc, n, n, comm);
            }
    }
    else
    {
        for (const auto &q : qs)
        {
            const auto bare = retained_coulomb(context, q);
            for (auto &tensor : result.tensors)
            {
                const int n = result.sites[tensor.site_index].n_orbitals;
                ComplexMatrix block(n * n, n * n);
                if (comm.is_root())
                    block = contract_rectangular_ordered_pair_vertex(
                        vertices[tensor.spin_left][tensor.site_index].at(q), bare,
                        vertices[tensor.spin_right][tensor.site_index].at(q), n, n);
                comm.bcast(block.c, block.size, 0);
                tensor.bare += q_weight * block;
            }
        }
    }

    return projection;
}

Chi0QMap build_crpa_total_response(CrpaContext &context)
{
    auto &chi = context.chi0;
    // Pd needs time nodes. Reject here, before P0 is built.
    require_collectively(chi.tfg.has_time_grids(), chi.comm_h,
                         "cRPA requires a grid with time nodes, such as tfgrids_type = minimax");
    CrpaChi0State response_state(chi);
    global::lib_printf_root("cRPA: building P0 on original KS poles\n");
    chi.clear_band_selection();
    chi.build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_SPACE_TIME, context.coefficients,
              context.local_atom_pairs,
              context.parent_auxiliary_basis, context.auxiliary_transform, context.blacs);
    return chi.take_chi0_q();
}

Chi0QMap build_crpa_excluded_response(CrpaContext &context, const Chi0::BandSelection &selection)
{
    auto &chi = context.chi0;
    CrpaChi0State response_state(chi);
    require_collectively(!selection.empty(), chi.comm_h,
                         "cRPA requires an explicit band-selection table");
    global::lib_printf_root("cRPA: building Pd with selected occupied and empty branches\n");
    chi.set_band_selection(selection);
    chi.build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_SPACE_TIME, context.coefficients,
              context.local_atom_pairs,
              context.parent_auxiliary_basis, context.auxiliary_transform, context.blacs);
    return chi.take_chi0_q();
}

Chi0QMap form_crpa_constrained_response(CrpaContext &context, const Chi0QMap &full_response,
                                        Chi0QMap excluded_response)
{
    auto &chi = context.chi0;
    const bool resident_full = &full_response == &chi.get_chi0_q();
    CrpaChi0State response_state(chi);
    chi.swap_chi0_q(excluded_response);
    chi.replace_chi0_q_by_difference(resident_full ? response_state.response : full_response);
    return chi.take_chi0_q();
}

CrpaScreenedResponse screen_crpa_response(CrpaContext &context, Chi0QMap response,
                                          diele_func *optical)
{
    CrpaTimer screen_timer("crpa_screen", "Native screening of an owning cRPA response");
    auto &chi = context.chi0;
    const auto &comm = chi.comm_h;
    const auto &basis = chi.atbasis_abf;
    const auto &desc = context.active_auxiliary_descriptor;
    const auto &qs = chi.pbc.klist;
    CrpaChi0State response_state(chi);
    chi.swap_chi0_q(response);
    // The native Wc implementation owns the inverse dielectric calculation.
    // Raw v is supplied only once to its spectral filter. The local bare term
    // uses the identical threshold and never restores discarded directions.
    auto coulomb_wc = copy_coulomb(context.coulomb);
    std::vector<std::complex<double>> epsmac;
    if (optical)
        for (double value : optical->get_head_vec()) epsmac.emplace_back(value);
    auto wc = compute_Wc_freq_q_blacs(chi, context.coulomb, coulomb_wc,
                                      context.sqrt_coulomb_threshold, optical != nullptr,
                                      optical ? 3 : 0, epsmac, optical, context.blacs, desc, false);
    if (chi.qpoint_view().restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL)
        for (auto &[frequency, kernels] : wc)
        {
            const auto restored = restore_symmetry_dense_wq_map(kernels, chi.pbc, chi.qpoint_view(),
                                                                chi.symmetry_context, basis, desc);
            // Preserve the input's exact q keys after fractional-coordinate restoration.
            std::map<Vector3_Order<double>, Matz> full;
            for (const auto &q : qs)
            {
                auto match = restored.end();
                for (auto it = restored.begin(); it != restored.end(); ++it)
                    if (same_fractional_kpoint(Vector3_Order<double>{chi.pbc.latvec * it->first},
                                               Vector3_Order<double>{chi.pbc.latvec * q}, 1e-8))
                    {
                        require_collectively(match == restored.end(), comm,
                                             "cRPA q-star restore has duplicate members");
                        match = it;
                    }
                require_collectively(match != restored.end(), comm,
                                     "cRPA q-star restore misses a full-grid member");
                full.emplace(q, match->second);
            }
            kernels.swap(full);
        }
    return wc;
}

std::vector<std::vector<ComplexMatrix>> contract_crpa_interaction(
    CrpaContext &context, const CrpaLocalProjection &projection,
    const CrpaScreenedResponse &screened_response)
{
    const auto &chi = context.chi0;
    const auto &result = projection.bare_result;
    const auto &vertices = projection.vertices;
    const auto &desc = context.active_auxiliary_descriptor;
    const auto &comm = chi.comm_h;
    const double q_weight = 1.0 / chi.pbc.get_n_cells_bvk();
    std::vector<std::vector<ComplexMatrix>> values;
    values.reserve(result.tensors.size());
    for (const auto &tensor : result.tensors)
    {
        values.emplace_back();
        const int n = result.sites[tensor.site_index].n_orbitals;
        for (double frequency : result.frequencies)
        {
            ComplexMatrix value(tensor.bare);
            for (const auto &q : chi.pbc.klist)
                value += q_weight * contract_crpa_blacs(
                                        vertices[tensor.spin_left][tensor.site_index].at(q),
                                        vertices[tensor.spin_right][tensor.site_index].at(q),
                                        screened_response.at(frequency).at(q), desc, n, n, comm);
            require_collectively(finite(value), comm, "cRPA: non-finite screened interaction");
            values.back().push_back(std::move(value));
        }
    }
    return values;
}

CrpaResult evaluate_crpa_onsite_from_response(CrpaContext &context,
                                              const Chi0::BandSelection &selection,
                                              const std::vector<CorrelatedSubspace *> &frames,
                                              const Chi0QMap &full_response)
{
    CrpaTimer total_timer("crpa_shared_response", "Onsite cRPA U from an existing full response");
    auto projection = prepare_crpa_local_projection(context, selection, frames);
    auto excluded = build_crpa_excluded_response(context, selection);
    auto constrained = form_crpa_constrained_response(context, full_response, std::move(excluded));
    global::lib_printf_root("cRPA: screening Pr=P0-Pd for U from shared P0\n");
    auto wc = screen_crpa_response(context, std::move(constrained), context.constrained_headwing);
    auto values = contract_crpa_interaction(context, projection, wc);
    auto result = std::move(projection.bare_result);
    for (std::size_t tensor = 0; tensor < result.tensors.size(); ++tensor)
        result.tensors[tensor].u = std::move(values[tensor]);
    return result;
}

}  // namespace librpa_int
