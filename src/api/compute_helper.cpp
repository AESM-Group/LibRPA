#include "compute_helper.h"

#include <librpa_enums.h>
#include "librpa_compute.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <set>
#include <string>

#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../core/meanfield_mpi.h"
#include "../math/utils_matrix_mpi.h"
#include "../core/native_velocity.h"
#include "../mpi/collective_stage.h"
#include "dataset_helper.h"
#include "../io/global_io.h"
#include "instance_manager.h"
#include "../io/stl_io_helper.h"
#include "../utils/error.h"

namespace librpa_int::api
{

// Build a deduplicated list of requested k-points across ranks. This lets
// extraction APIs exchange only the k-resolved quantities callers asked for.
std::vector<int> collect_requested_iks(const MpiCommHandler &comm_h, const int n_kpts_this,
                                       const int *iks_this, const int n_kpoints)
{
    std::vector<int> nks_all(comm_h.nprocs, 0);
    nks_all[comm_h.myid] = n_kpts_this;
    comm_h.allreduce(MPI_IN_PLACE, nks_all.data(), comm_h.nprocs, MPI_SUM);

    const int nk_max = *std::max_element(nks_all.cbegin(), nks_all.cend());
    if (nk_max == 0) return {};

    std::vector<int> iks_all(comm_h.nprocs * nk_max, -1);
    int bad_ik_local = 0;
    int bad_ik_value_local = -1;
    for (int i = 0; i != n_kpts_this; ++i)
    {
        const int ik = iks_this[i];
        if (ik < 0 || ik >= n_kpoints)
        {
            bad_ik_local = 1;
            bad_ik_value_local = ik;
            continue;
        }
        iks_all[comm_h.myid * nk_max + i] = ik;
    }

    int bad_ik = 0;
    int bad_ik_value = -1;
    comm_h.allreduce(&bad_ik_local, &bad_ik, 1, MPI_MAX);
    comm_h.allreduce(&bad_ik_value_local, &bad_ik_value, 1, MPI_MAX);
    if (bad_ik)
        throw LIBRPA_RUNTIME_ERROR("requested k-point index out of range: " +
                                  std::to_string(bad_ik_value));

    comm_h.allreduce(MPI_IN_PLACE, iks_all.data(), static_cast<int>(iks_all.size()), MPI_MAX);

    std::set<int> unique_iks;
    for (int pid = 0; pid != comm_h.nprocs; ++pid)
    {
        for (int i = 0; i != nks_all[pid]; ++i)
        {
            unique_iks.insert(iks_all[pid * nk_max + i]);
        }
    }
    return {unique_iks.cbegin(), unique_iks.cend()};
}

AtomPairBvKRemap<atom_t> build_band_bvk_remap(const Atoms &atoms,
                                              const PeriodicBoundaryData &pbc,
                                              const int remap_convention)
{
    using global::ofs_myid;

    if (remap_convention != 0 && remap_convention != 1)
        throw LIBRPA_RUNTIME_ERROR("Invalid BvK remap convention: " + std::to_string(remap_convention));

    const bool is_ready = pbc.is_latt_set() && atoms.is_set() && atoms.is_frac_set() && !pbc.Rlist.empty();
    AtomPairBvKRemap<atom_t> remap;
    if (is_ready)
    {
        remap.build(atoms.coords_frac, pbc.Rlist, pbc.period, pbc.latvec, remap_convention);
    }

    ofs_myid << "Final BvK remapping for band interpolation:\n";
    ofs_myid << "| remap convention  : " << remap_convention << "\n";
    ofs_myid << "| input ready       : " << (is_ready ? "true" : "false") << "\n";
    ofs_myid << "| lattice set       : " << (pbc.is_latt_set() ? "true" : "false") << "\n";
    ofs_myid << "| atoms set         : " << (atoms.is_set() ? "true" : "false") << "\n";
    ofs_myid << "| atom frac set     : " << (atoms.is_frac_set() ? "true" : "false") << "\n";
    ofs_myid << "| period            : " << pbc.period << "\n";
    ofs_myid << "| Rlist size        : " << pbc.Rlist.size() << "\n";
    ofs_myid << "| atom-pair entries : " << remap.size() << "\n";
    for (const auto &[atom_pair, R_map]: remap.data())
    {
        for (const auto &[R, R_bvks]: R_map)
        {
            // Only print 1-to-1 remapping with debug
            if (R_bvks.size() < 2 && !global::should_output(LIBRPA_VERBOSE_DEBUG)) continue;
            ofs_myid << "| " << atom_pair << " " << R << " -> " << R_bvks << "\n";
        }
    }
    ofs_myid << std::flush;
    return remap;
}

}

namespace
{
using namespace librpa_int;

void append_matrix(std::vector<double> &values, const Matz &matrix)
{
    for (int row = 0; row < matrix.nr(); ++row)
        for (int column = 0; column < matrix.nc(); ++column)
        {
            const auto value = matrix(row, column);
            values.push_back(value.real());
            values.push_back(value.imag());
        }
}

const double *copy_optical_buffer(const std::vector<double> &values)
{
    if (values.empty()) return nullptr;
    auto buffer = std::make_unique<double[]>(values.size());
    std::copy(values.begin(), values.end(), buffer.get());
    return buffer.release();
}
}  // namespace

LibrpaOpticalResult *librpa_get_optical_result(const LibrpaHandler *h)
{
    const auto ds = librpa_int::api::get_dataset_instance(h);
    if (!ds->p_headwing) throw LIBRPA_RUNTIME_ERROR("Analytic head/wing is not initialized");
    std::unique_ptr<LibrpaOpticalResult, decltype(&librpa_delete_optical_result)> result(
        new LibrpaOpticalResult{}, librpa_delete_optical_result);
    auto &view = *result;
    std::vector<double> velocity, head, wing;
    view.n_spins = ds->mf.get_n_spins();
    view.n_kpoints = ds->mf.get_n_kpoints();
    view.n_states = ds->mf.get_n_states();
    for (const auto &spin : ds->velocity_matrix)
        for (const auto &k : spin)
            for (const auto &axis : k)
                for (int row = 0; row < axis.nr; ++row)
                    for (int column = 0; column < axis.nc; ++column)
                    {
                        velocity.push_back(axis(row, column).real());
                        velocity.push_back(axis(row, column).imag());
                    }
    const auto &frequencies = ds->p_headwing->get_frequency_nodes();
    view.n_frequencies = frequencies.size();
    const auto &heads = ds->p_headwing->get_head_tensors();
    const auto &wings = ds->p_headwing->get_wing_auxiliary();
    if (!heads.empty())
    {
        view.head_rows = heads.front().nr();
        view.head_columns = heads.front().nc();
        for (const auto &matrix : heads) append_matrix(head, matrix);
    }
    if (!wings.empty())
    {
        view.wing_rows = wings.front().nr();
        view.wing_columns = wings.front().nc();
        for (const auto &matrix : wings) append_matrix(wing, matrix);
    }
    view.velocity_ri = copy_optical_buffer(velocity);
    view.frequencies_ha = copy_optical_buffer(frequencies);
    view.head_ri = copy_optical_buffer(head);
    view.wing_ri = copy_optical_buffer(wing);
    return result.release();
}

LibrpaOpticalView librpa_get_optical_view(const LibrpaOpticalResult *result)
{
    if (!result) throw LIBRPA_INVALID_ARGUMENT("Optical result must not be null");
    return *result;
}

void librpa_delete_optical_result(LibrpaOpticalResult *result)
{
    if (!result) return;
    delete[] result->velocity_ri;
    delete[] result->frequencies_ha;
    delete[] result->head_ri;
    delete[] result->wing_ri;
    delete result;
}

void librpa_prepare_native_velocity(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                    const LibrpaNativeVelocityInput *root_input,
                                    int prepare_optical)
{
    using namespace librpa_int;
    const auto ds = api::get_dataset_instance(h);
    const auto &comm = ds->comm_h;
    collective_all_stage(
        comm, "Native velocity options",
        [&]
        {
            if (!p_opts)
                throw LIBRPA_INVALID_ARGUMENT("Native velocity requires numerical options");
            if (p_opts->option_dielect_func != 3 && p_opts->option_dielect_func != 4)
                throw LIBRPA_INVALID_ARGUMENT(
                    "Native velocity API options must use numerical values 3/4");
            if (prepare_optical && p_opts->replace_w_head != LIBRPA_SWITCH_ON)
                throw LIBRPA_INVALID_ARGUMENT(
                    "Native optical preparation requires replace_w_head enabled");
            if (prepare_optical && p_opts->option_dielect_func == 3 &&
                p_opts->use_scalapack_gw_wc != LIBRPA_SWITCH_ON)
                throw LIBRPA_INVALID_ARGUMENT(
                    "Native head/wing preparation with option_dielect_func = 3 requires "
                    "use_scalapack_gw_wc enabled to consume analytic wings");
        });
    const auto &opts = *p_opts;
    // A cached full-BZ frame belongs to the inputs that built it. Drop it before
    // reading the new ones, so a rebuild cannot select a stale frame.
    ds->headwing_full_bz_mf = MeanField{};
    ds->headwing_full_bz_pbc = PeriodicBoundaryData{};
    ds->headwing_full_bz_velocity.clear();
    ds->headwing_full_bz_ready = false;

    // The differentiation needs one complete AO x band block per k point. A
    // k-point-parallel layout can split a single block across the ranks of its
    // k-group, so rebuild the replicated layout whenever the stored one is
    // split. This depends on the wave-function layout, not on whether the SCF
    // grid happens to be the full mesh.
    MeanField source = ds->mf;
    if (ds->eigvecs_kpara_2d_ready() && ds->scfk_blacs_ctxt.is_initialized() &&
        ds->desc_wfc_kb.is_initialized() && ds->desc_wfc_kb_full.is_initialized() &&
        (ds->desc_wfc_kb.mb() < ds->desc_wfc_kb.m() ||
         ds->desc_wfc_kb.nb() < ds->desc_wfc_kb.n()))
    {
        int failed = 0;
        try
        {
            redistribute_meanfield_eigvecs_kblacs(source, ds->scfk_blacs_ctxt, ds->desc_wfc_kb,
                                                  ds->desc_wfc_kb_full,
                                                  "native head/wing");
        }
        catch (const std::exception &)
        {
            failed = 1;
        }
        // Each k-group reports on its own communicator; agree on the whole
        // handler before any rank enters the next collective.
        int any_failed = 0;
        comm.allreduce(&failed, &any_failed, 1, MPI_MAX);
        if (any_failed)
            throw LIBRPA_RUNTIME_ERROR(
                "Native head/wing cannot replicate the split k-point-parallel "
                "wave-function blocks");
    }

    const SymmetryContext *symmetry = nullptr;
    collective_all_stage(
        comm, "Native magnetic velocity frame",
        [&]
        {
            if (ds->mf.get_n_kpoints() == ds->pbc.get_n_cells_bvk()) return;
            // A symmetry-reduced SCF grid is accepted whenever the k-stars can
            // restore the full mesh; the expansion below supplies the frame that
            // the per-k differentiation needs.
            initialize_symmetry_context(*ds, true);
            require_symmetry_shell_layouts(*ds, "native IBZ head/wing");
            symmetry = &ds->symmetry_context;
        });
    // A symmetry-reduced SCF grid cannot feed the head/wing velocity directly: the
    // per-k blocks only exist on the IBZ representatives. Expand the loaded
    // eigenframe to the full mesh first, exactly as QSGW does for its reference
    // state, then differentiate H/S/r in that full frame. No eigenproblem is
    // solved, and the caller keeps its own IBZ mean field for P0 and W.
    MeanField headwing_meanfield;
    PeriodicBoundaryData headwing_pbc;
    const MeanField *velocity_mf = &ds->mf;
    const PeriodicBoundaryData *velocity_pbc = &ds->pbc;
    if (symmetry != nullptr && ds->mf.get_n_kpoints() != ds->pbc.get_n_cells_bvk())
    {
        MeanField replicated = source;
        for (int ispin = 0; ispin < replicated.get_n_spins(); ++ispin)
            for (int ispinor = 0; ispinor < replicated.get_n_spinor(); ++ispinor)
                for (int ik = 0; ik < replicated.get_n_kpoints(); ++ik)
                {
                    auto *local = replicated.find_wfc(ispin, ispinor, ik);
                    const bool complete = local != nullptr &&
                                          local->nr == replicated.get_n_states() &&
                                          local->nc == replicated.get_n_aos();
                    const int owner = find_mpi_owner_rank(complete, comm.comm);
                    if (owner < 0)
                        throw LIBRPA_RUNTIME_ERROR(
                            "Native IBZ expansion is missing a complete source WFC block");
                    auto &block = replicated.get_eigenvectors()[ispin][ispinor][ik];
                    broadcast_ComplexMatrix(block, owner, comm.comm);
                }
        const auto expanded =
            expand_symmetry_meanfield(replicated, ds->basis_wfc, ds->pbc, *symmetry);
        headwing_meanfield = expanded.meanfield;
        headwing_pbc = ds->pbc;
        headwing_pbc.klist = headwing_pbc.klist_full;
        headwing_pbc.kfrac_list = headwing_pbc.kfrac_list_full;
        headwing_pbc.weight_k.assign(
            static_cast<std::size_t>(headwing_meanfield.get_n_kpoints()),
            1.0 / static_cast<double>(headwing_meanfield.get_n_kpoints()));
        headwing_pbc.k_to_kfull.resize(static_cast<std::size_t>(headwing_meanfield.get_n_kpoints()));
        std::iota(headwing_pbc.k_to_kfull.begin(), headwing_pbc.k_to_kfull.end(), 0);
        headwing_pbc.kfull_to_k = headwing_pbc.k_to_kfull;
        headwing_pbc.kfull_to_k_relation.assign(
            static_cast<std::size_t>(headwing_meanfield.get_n_kpoints()), KFullToKRelation::DIRECT);
        ds->headwing_full_bz_mf = headwing_meanfield;
        ds->headwing_full_bz_pbc = headwing_pbc;
        ds->headwing_full_bz_ready = true;
        velocity_mf = &ds->headwing_full_bz_mf;
        velocity_pbc = &ds->headwing_full_bz_pbc;
        if (comm.is_root())
            global::lib_printf(
                "G0W0 native velocity: expanded %d IBZ states to %d full-grid states; "
                "no eigenproblem solved\n",
                ds->mf.get_n_kpoints(), headwing_meanfield.get_n_kpoints());
    }
    auto result = build_native_velocity(*velocity_mf, *velocity_pbc, comm, root_input, nullptr);
    if (ds->headwing_full_bz_ready)
    {
        // Repack the full-BZ velocity so head/wing can consume it on that mesh.
        const int n_hw_spins = ds->headwing_full_bz_mf.get_n_spins();
        const int n_hw_kpoints = ds->headwing_full_bz_mf.get_n_kpoints();
        const int n_hw_states = ds->headwing_full_bz_mf.get_n_states();
        const std::size_t count = static_cast<std::size_t>(n_hw_spins) *
                                  static_cast<std::size_t>(n_hw_kpoints) * 3 *
                                  static_cast<std::size_t>(n_hw_states) *
                                  static_cast<std::size_t>(n_hw_states);
        if (result.velocity.size() != count)
            throw LIBRPA_RUNTIME_ERROR("Full-BZ native velocity has an unexpected size");
        initialize_velocity_matrix(ds->headwing_full_bz_velocity, n_hw_spins, n_hw_kpoints,
                                   n_hw_states);
        std::size_t cursor = 0;
        for (int spin = 0; spin < n_hw_spins; ++spin)
            for (int ik = 0; ik < n_hw_kpoints; ++ik)
                for (int alpha = 0; alpha < 3; ++alpha)
                    for (int row = 0; row < n_hw_states; ++row)
                        for (int column = 0; column < n_hw_states; ++column, ++cursor)
                            ds->headwing_full_bz_velocity[spin][ik][alpha](row, column) =
                                result.velocity[cursor];
    }
    // The public velocity matrix is indexed on the caller's k list. When the
    // differentiation ran on the expanded mesh, hand the setter only the
    // representative k-points, in the caller's order.
    std::vector<cplxdb> ibz_velocity;
    const std::vector<cplxdb> *setter_velocity = &result.velocity;
    if (ds->headwing_full_bz_ready)
    {
        const int n_spins = ds->mf.get_n_spins();
        const int n_kpoints = ds->mf.get_n_kpoints();
        const int n_states = ds->mf.get_n_states();
        const std::size_t per_k = static_cast<std::size_t>(3) * n_states * n_states;
        ibz_velocity.resize(static_cast<std::size_t>(n_spins) * n_kpoints * per_k);
        for (int spin = 0; spin < n_spins; ++spin)
            for (int ik = 0; ik < n_kpoints; ++ik)
            {
                const int ifull = fold_fractional_kpoint_to_targets(
                                      ds->pbc.kfrac_list[ik], ds->pbc.kfrac_list_full, 1e-8)
                                      .target_k_index;
                if (ifull < 0)
                    throw LIBRPA_RUNTIME_ERROR(
                        "Native IBZ velocity cannot locate a representative k-point");
                const auto *src = result.velocity.data() +
                                  ((static_cast<std::size_t>(spin) *
                                        ds->headwing_full_bz_mf.get_n_kpoints() + ifull) *
                                   per_k);
                std::copy(src, src + per_k,
                          ibz_velocity.begin() + (static_cast<std::size_t>(spin) * n_kpoints + ik) *
                                                    per_k);
            }
        setter_velocity = &ibz_velocity;
    }
    collective_all_stage(comm, "G0W0 native velocity setter",
                         [&]
                         {
                             librpa_set_velocity_matrix_packed(
                                 h, ds->mf.get_n_spins(), ds->mf.get_n_kpoints(),
                                 ds->mf.get_n_states(),
                                 reinterpret_cast<const double *>(setter_velocity->data()));
                         });
    if (comm.is_root())
        global::lib_printf(
            "G0W0 native velocity: metric %.8e frame_Ha %.8e hermiticity_Ha %.8e "
            "diagonal_imag_HaBohr %.8e; SCF E/C/occupations retained\n",
            result.residuals[0], result.residuals[1], result.residuals[2], result.residuals[3]);
    auto &coulomb_normalization = result.coulomb_normalization;

    if (!prepare_optical) return;

    // Preserve the numerical grid/head/wing lifecycle behind the public API.
    initialize_ds_global_ddla(*ds, opts);
    initialize_ds_tfgrids(*ds, opts);
    initialize_ds_headwing(*ds, opts, opts.option_dielect_func == 3);
    comm.bcast(coulomb_normalization.data(), 2, 0);
    collective_all_stage(
        comm, "Native head/wing Coulomb normalization",
        [&]
        {
            if (opts.use_2d_dielectric == LIBRPA_SWITCH_ON)
            {
                const auto normalization =
                    strict_2d_coulomb_head_normalization(ds->pbc, coulomb_normalization[1]);
                const double area = coulomb_normalization[0];
                if (!std::isfinite(area) || area <= 0.0 ||
                    std::abs(normalization.inplane_area_bohr2 - area) >
                        1.0e-10 * std::max(normalization.inplane_area_bohr2, area))
                    throw LIBRPA_INVALID_ARGUMENT(
                        "Coulomb normalization area disagrees with the lattice");
                ds->p_headwing->configure_strict_2d_coulomb_head(true, coulomb_normalization[1]);
            }
            else
                ds->p_headwing->configure_strict_2d_coulomb_head(false);
        });
}
