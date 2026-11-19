// Public API headers
#include "librpa_enums.h"

// Internal headers
#include "../utils/error.h"
#include "../utils/profiler.h"
#include "../io/global_io.h"
#include "../math/rsh.h"
#include "../math/symmetry.h"
#include "dataset_helper.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace librpa_int
{

namespace
{

int max_basis_l(const std::vector<const AtomicBasis*>& bases)
{
    int lmax = -1;
    for (const auto* basis : bases)
    {
        if (basis != nullptr && basis->has_l_shells())
        {
            lmax = std::max(lmax, basis->get_max_l());
        }
    }
    return lmax;
}

} // namespace

void initialize_symmetry_context(Dataset &ds, const bool build_shell_rotations)
{
    // Metadata-only setup is also used while reading full-BZ inputs. Numerical
    // spinor restoration, however, must never inherit the legacy identity-spin
    // grey group: its spatial operations do not specify the SOC spin action.
    if (build_shell_rotations) reject_spinor_symmetry_speedup(ds, "numerical");
    auto &spg_symops = ds.spg_symops;

    if (spg_symops.empty())
    {
        spg_symops.push_back(SpaceGroupSymOp::IDENTITY);
    }

    auto &ctx = ds.symmetry_context;
    ctx.clear();
    ctx.set_crystal_structure(ds.pbc.latvec, ds.pbc.G, ds.atoms.types, ds.atoms.coords_frac);
    ctx.set_rspace_operations(spg_symops);
    if (ds.spg_spin_ops_explicit)
    {
        ctx.set_symmetry_spin_operations(ds.spg_spin_ops, ds.spg_grey_group);
        // Phase 8: collinear/scalar storage can only hold operations whose
        // effective spin action maps each channel onto itself; anything else
        // is rejected here instead of silently mis-restored downstream.
        validate_spin_operations_for_storage(
            ctx, ds.mf.get_n_spins(), ds.mf.get_n_spinor());
    }
    ctx.build_periodic_mappings(ds.pbc, ds.pbc.Rlist);

    auto mark_available = [&ctx]() {
        ctx.set_available();
        ctx.print_summary(global::ofs_myid);
    };

    if (!build_shell_rotations)
    {
        mark_available();
        return;
    }

    const int lmax = max_basis_l({&ds.basis_wfc, &ds.basis_aux, &ds.basis_aux_shrink});
    if (lmax < 0)
    {
        mark_available();
        return;
    }

    auto basis_convention =
        is_basis_convention_set(ctx.basis_convention) ? ctx.basis_convention : ds.basis_convention;
    if (!is_basis_convention_set(basis_convention))
    {
        throw LIBRPA_RUNTIME_ERROR("Cannot initialize symmetry context without basis convention");
    }
    ds.basis_convention = basis_convention;
    ctx.build_rsh_rotations(basis_convention, lmax);
    ctx.build_kstar_member_rotations(lmax);
    mark_available();
}

void require_symmetry_shell_layouts(const Dataset &ds, const char *calculation)
{
    const auto &ctx = ds.symmetry_context;
    if (!ctx.available || ctx.rspace_operations.empty())
    {
        return;
    }

    if (!ds.basis_wfc.has_l_shells() || !ds.basis_aux.has_l_shells())
    {
        throw LIBRPA_RUNTIME_ERROR(
            std::string("Cannot use ") + calculation
            + " symmetry without l-shell basis layouts for AO and ABF species");
    }
}

void reject_spinor_symmetry_speedup(const Dataset &ds, const char *calculation)
{
    if (ds.mf.get_n_spinor() <= 1)
    {
        return;
    }
    // Spinor symmetry requires an explicit spatial/spin operation table in the
    // input dataset. Keep rejecting legacy requests with no spin-space action
    // so that an ordinary spatial table cannot silently produce wrong channels.
    if (ds.spg_spin_ops_explicit)
    {
        return;
    }
    throw LIBRPA_RUNTIME_ERROR(
        std::string("Cannot use ") + calculation
        + " symmetry speed-up with spinor wave functions without an explicit spin-space"
        + " operation table; regenerate the dataset with spin_symmetry metadata");
}

void initialize_ds_tfgrids(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_tfgrids");
    ds.tfg.reset(opts.nfreq);
    double emin = opts.tfgrids_freq_min;
    double eintv = opts.tfgrids_freq_interval;
    double emax = opts.tfgrids_freq_max;
    double tmin = opts.tfgrids_time_min;
    double tintv = opts.tfgrids_time_interval;
    double regulation = opts.minimax_regulation;
    if (opts.tfgrids_type == LIBRPA_TFGRID_MINIMAX)
    {
        double emin_mf, emax_mf;
        ds.mf.get_E_min_max(emin_mf, emax_mf);
        emax = opts.minimax_emax > 0 ? opts.minimax_emax : emax_mf;
        emin = opts.minimax_emin > 0 ? opts.minimax_emin : emin_mf;
    }
    ds.tfg.generate(opts.tfgrids_type, emin, eintv, emax, tmin, tintv, regulation);
    global::profiler.stop("initialize_ds_tfgrids");
}

void initialize_ds_global_ddla(Dataset &ds, const LibrpaOptions &opts)
{
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON && ds.blacs_h.ddla_handle == nullptr)
    {
        global::profiler.start("initialize_ds_global_ddla");
        ds.blacs_h.init_ddla_handle();
        global::profiler.stop("initialize_ds_global_ddla");
    }
#else
    (void)ds;
    (void)opts;
#endif
}

static void collect_atpairs_all(Dataset &ds)
{
    const auto &comm_h = ds.comm_h;
    const auto &atpairs_local = ds.atpairs_local;
    const int np_this = atpairs_local.size();
    std::vector<int> np_all(comm_h.nprocs, 0);
    np_all[comm_h.myid] = np_this;
    int np_max;
    comm_h.allreduce(&np_this, &np_max, 1, MPI_MAX);
    comm_h.allreduce(MPI_IN_PLACE, np_all.data(), comm_h.nprocs, MPI_SUM);
    if (np_max == 0) return;
    std::vector<size_t> pairs_all(comm_h.nprocs * np_max * 2, 0);
    const int st = comm_h.myid * np_max * 2;
    for (int ip = 0; ip < np_this; ip++)
    {
        pairs_all[st + ip * 2] = atpairs_local[ip].first;
        pairs_all[st + ip * 2 + 1] = atpairs_local[ip].second;
    }
    comm_h.allreduce(MPI_IN_PLACE, pairs_all.data(), pairs_all.size(), MPI_SUM);
    for (int pid = 0; pid < comm_h.nprocs; pid++)
    {
        if (pid == comm_h.myid)
        {
            std::set<atpair_t> atpairs_this(atpairs_local.cbegin(), atpairs_local.cend());
            ds.atpairs_unique_all.emplace(pid, atpairs_this);
        }
        else
        {
            const int np_this = np_all[pid];
            std::set<atpair_t> atpairs_this;
            const int st = pid * np_max * 2;
            for (int ip = 0; ip < np_this; ip++)
            {
                atpairs_this.insert({pairs_all[st + ip * 2], pairs_all[st + ip * 2 + 1]});
            }
            ds.atpairs_unique_all.emplace(pid, atpairs_this);
        }
    }
}

void initialize_ds_atpairs_local(Dataset &ds, LibrpaParallelRouting routing)
{
    global::profiler.start(__FUNCTION__);

    ds.atpairs_local.clear();
    const int n_atoms_basis_wfc = ds.basis_wfc.n_atoms;
    const int n_atoms_basis_aux = ds.basis_aux.n_atoms;
    const int n_atoms_struc = ds.atoms.size();
    const int n_atoms =
        n_atoms_struc > 0 ? n_atoms_struc : std::max(n_atoms_basis_aux, n_atoms_basis_wfc);
    if (n_atoms == 0)
        throw LIBRPA_RUNTIME_ERROR(
            "Number of atoms can not be extracted, please set structure or basis first");

    if (routing == LIBRPA_ROUTING_AUTO)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "internal error: routing should be decided before initialize_ds_atpairs_local, not "
            "AUTO");
    }
    else if (routing == LIBRPA_ROUTING_ATOMPAIR || routing == LIBRPA_ROUTING_LIBRI)
    {
        auto tri_local_atpair = librpa_int::dispatch_upper_triangular_tasks(
            n_atoms, ds.blacs_h.myid, ds.blacs_h.nprows, ds.blacs_h.npcols, ds.blacs_h.myprow,
            ds.blacs_h.mypcol);
        for (const auto &p : tri_local_atpair) ds.atpairs_local.emplace_back(p);
    }
    else
    {
        ds.atpairs_local = generate_atom_pair_from_nat(n_atoms, false);
    }
    collect_atpairs_all(ds);
    global::profiler.stop(__FUNCTION__);
}

bool resolve_use_symmetry_exx(const LibrpaOptions &opts, const PeriodicBoundaryData &pbc)
{
    if (opts.use_symmetry_exx == LIBRPA_SWITCH_ON)
        return true;
    // Same predicate as the GW/RPA guard in build_pbc_qpoint_view: the loaded
    // SCF k-list is symmetry-reduced when it is shorter than the full BZ grid.
    if (static_cast<int>(pbc.klist.size()) < pbc.get_n_cells_bvk())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "The SCF k-grid is symmetry-reduced, but use_symmetry_exx is off. Exchange from "
            "a reduced k-grid without the symmetry restore is inexact (star members are "
            "folded into weights without basis rotations, and the Coulomb star expansion "
            "only supports the inversion partner). Either set use_symmetry_exx = t, or "
            "supply an unreduced full-BZ k-list from a producer run with symmetry disabled. "
            "Refusing to compute silently wrong exchange.");
    }
    return false;
}

void initialize_ds_exx(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_exx");
    const bool use_symmetry = resolve_use_symmetry_exx(opts, ds.pbc);
    if (use_symmetry)
    {
        initialize_symmetry_context(ds, true);
        require_symmetry_shell_layouts(ds, "EXX");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    const ArrayDesc &desc_scf_wfc =
        ds.eigvecs_kpara_2d_ready() ? ds.desc_wfc_kb : ds.desc_wfc_kb_full;
    ds.p_exx = std::make_unique<librpa_int::Exx>(ds.mf, ds.basis_wfc, ds.pbc, ds.symmetry_context,
                                                 ds.scfk_blacs_ctxt, ds.bandk_blacs_ctxt,
                                                 desc_scf_wfc, ds.desc_band_wfc_kb,
                                                 is_eigvec_k_distributed,
                                                 use_symmetry);
    ds.p_exx->libri_threshold_C = opts.libri_exx_threshold_C;
    ds.p_exx->libri_threshold_D = opts.libri_exx_threshold_D;
    ds.p_exx->libri_threshold_V = opts.libri_exx_threshold_V;
    ds.p_exx->output_dir = opts.output_dir;
    ds.p_exx->output_exx_mat_k = opts.output_exx_mat_k == LIBRPA_SWITCH_ON;
    ds.p_exx->output_exx_mat_r = opts.output_exx_mat_r == LIBRPA_SWITCH_ON;
    global::profiler.stop("initialize_ds_exx");
}

static void initialize_ds_chi0_impl(Dataset &ds, const LibrpaOptions &opts, bool qstar_only)
{
    global::profiler.start("initialize_ds_chi0");
    if (opts.n_bands_exclude_chi0 < 0 || opts.n_bands_exclude_chi0 > ds.mf.get_n_bands())
        throw LIBRPA_RUNTIME_ERROR(
            "n_bands_exclude_chi0 must be between 0 and the number of input bands");
    const bool use_symmetry = opts.use_symmetry_rpa == LIBRPA_SWITCH_ON;
    if (qstar_only && (ds.mf.get_n_kpoints() != ds.pbc.get_n_cells_bvk() ||
                       opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON))
        throw LIBRPA_INVALID_ARGUMENT(
            "q-star-only response requires replicated full-grid KS states");
    if (qstar_only && use_symmetry && ds.mf.get_n_spinor() == 2 && !ds.spg_spin_ops_explicit)
        throw LIBRPA_INVALID_ARGUMENT(
            "SOC cRPA symmetry requires an explicit magnetic operation table and grey-group "
            "declaration");
    if (use_symmetry)
    {
        initialize_symmetry_context(ds, true);
        require_symmetry_shell_layouts(ds, "RPA/chi0");
    }
    if (use_symmetry && opts.use_shrink_abfs == LIBRPA_SWITCH_ON)
    {
        const auto view = build_symmetry_qpoint_view(ds.symmetry_context, ds.pbc, true);
        if (view.restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL)
            ds.sinvS = restore_symmetry_rectangular_q_map(
                ds.sinvS, ds.pbc, view, ds.symmetry_context, ds.basis_aux_shrink, ds.basis_aux);
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    const ArrayDesc &desc_scf_wfc =
        ds.eigvecs_kpara_2d_ready() ? ds.desc_wfc_kb : ds.desc_wfc_kb_full;
    if (opts.use_shrink_abfs == LIBRPA_SWITCH_ON)
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(
            ds.mf, ds.basis_wfc, ds.basis_aux_shrink, ds.pbc, ds.symmetry_context, ds.tfg,
            ds.scfk_blacs_ctxt, desc_scf_wfc, is_eigvec_k_distributed, use_symmetry);
    else
        ds.p_chi0 = std::make_unique<librpa_int::Chi0>(
            ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context, ds.tfg,
            ds.scfk_blacs_ctxt, desc_scf_wfc, is_eigvec_k_distributed, use_symmetry);
    ds.p_chi0->qstar_only_symmetry = qstar_only;
    ds.p_chi0->gf_threshold = opts.gf_threshold;
    ds.p_chi0->libri_collect_s0_chunk = opts.libri_chi0_collect_s0_chunk;
    ds.p_chi0->libri_collect_max_bytes = opts.libri_chi0_collect_max_bytes;
    ds.p_chi0->nbands_G = opts.n_bands_chi0;
    ds.p_chi0->n_bands_exclude = opts.n_bands_exclude_chi0;
    ds.p_chi0->libri_threshold_C = opts.libri_chi0_threshold_C;
    ds.p_chi0->libri_threshold_G = opts.libri_chi0_threshold_G;
    global::profiler.stop("initialize_ds_chi0");
}

void initialize_ds_chi0(Dataset &ds, const LibrpaOptions &opts)
{
    initialize_ds_chi0_impl(ds, opts, false);
}

void initialize_ds_chi0_qstar(Dataset &ds, const LibrpaOptions &opts)
{
    initialize_ds_chi0_impl(ds, opts, true);
}

void initialize_ds_g0w0(Dataset &ds, const LibrpaOptions &opts)
{
    global::profiler.start("initialize_ds_g0w0");
    if (opts.n_bands_exclude_sigc < 0 || opts.n_bands_exclude_sigc > ds.mf.get_n_bands())
        throw LIBRPA_RUNTIME_ERROR("n_bands_exclude_sigc must be between 0 and the number of input bands");
    const bool use_symmetry = opts.use_symmetry_gw == LIBRPA_SWITCH_ON;
    if (use_symmetry)
    {
        initialize_symmetry_context(ds, true);
        require_symmetry_shell_layouts(ds, "GW");
    }
    const bool is_eigvec_k_distributed = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    const ArrayDesc &desc_scf_wfc =
        ds.eigvecs_kpara_2d_ready() ? ds.desc_wfc_kb : ds.desc_wfc_kb_full;
    // global::ofs_myid << "is_eigvec_k_distributed " << is_eigvec_k_distributed << std::endl;
    ds.p_g0w0 = std::make_unique<librpa_int::G0W0>(
        ds.mf, ds.basis_wfc, ds.pbc, ds.symmetry_context, ds.tfg, ds.scfk_blacs_ctxt,
        ds.bandk_blacs_ctxt, desc_scf_wfc, ds.desc_band_wfc_kb, is_eigvec_k_distributed,
        use_symmetry, ds.p_chi0 ? &ds.p_chi0->qpoint_view() : nullptr);
    ds.p_g0w0->nbands_G = opts.n_bands_sigc;
    ds.p_g0w0->n_bands_exclude = opts.n_bands_exclude_sigc;
    ds.p_g0w0->libri_threshold_C = opts.libri_g0w0_threshold_C;
    ds.p_g0w0->libri_threshold_G = opts.libri_g0w0_threshold_G;
    ds.p_g0w0->libri_threshold_Wc = opts.libri_g0w0_threshold_Wc;
    ds.p_g0w0->output_dir = opts.output_dir;
    ds.p_g0w0->output_sigc_ks_kf = opts.output_gw_sigc_ks_kf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_ks_mat_kf = opts.output_gw_sigc_ks_mat_kf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->istate_output_mat_start = opts.istate_output_mat_start;
    ds.p_g0w0->istate_output_mat_end = opts.istate_output_mat_end;
    ds.p_g0w0->output_sigc_mat_kf = opts.output_gw_sigc_mat_kf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat_rt = opts.output_gw_sigc_mat_rt == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_sigc_mat_rf = opts.output_gw_sigc_mat_rf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_wc_rf = opts.output_wc_rf == LIBRPA_SWITCH_ON;
    ds.p_g0w0->output_wc_rf_atom_pair = opts.output_wc_rf_atom_pair == LIBRPA_SWITCH_ON;
    ds.p_g0w0->ifreq_output_wc_start = opts.ifreq_output_wc_start;
    ds.p_g0w0->ifreq_output_wc_end = opts.ifreq_output_wc_end;
    global::profiler.stop("initialize_ds_g0w0");
}

std::unique_ptr<diele_func> make_analytic_headwing(Dataset &ds, const LibrpaOptions &opts,
                                                   bool need_wing,
                                                   const HeadwingBandSelection *excluded)
{
    if (ds.velocity_matrix.empty())
    {
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing requested but velocity matrix is not set");
    }

    // Native velocity on a symmetry-reduced SCF grid expands the IBZ states to the
    // full mesh and differentiates H/S/r there. When that frame exists, head/wing
    // run on the plain full-BZ mesh: no k-star reconstruction, no IBZ weights.
    const bool use_full_bz_frame = ds.headwing_full_bz_ready;
    MeanField &mf = use_full_bz_frame ? ds.headwing_full_bz_mf : ds.mf;
    const auto &headwing_kfrac = use_full_bz_frame ? ds.headwing_full_bz_pbc.kfrac_list
                                                  : ds.pbc.kfrac_list;
    if (!mf.initialized())
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing meanfield is not initialized");

    if (static_cast<int>(headwing_kfrac.size()) != mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR(
            "analytic head/wing k-point list is inconsistent with meanfield");

    const auto &headwing_basis_aux =
        opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.basis_aux_shrink : ds.basis_aux;
    if (!headwing_basis_aux.initialized())
        throw LIBRPA_RUNTIME_ERROR("analytic head/wing auxiliary basis is not initialized");

    const bool use_headwing_symmetry =
        !use_full_bz_frame && (opts.use_symmetry_rpa == LIBRPA_SWITCH_ON ||
                               opts.use_symmetry_gw == LIBRPA_SWITCH_ON);
    if (excluded && (use_headwing_symmetry || opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON ||
                     opts.use_2d_dielectric == LIBRPA_SWITCH_ON))
        throw LIBRPA_INVALID_ARGUMENT(
            "constrained optical input requires replicated full BZ and 3D");
    if (use_headwing_symmetry)
    {
        initialize_symmetry_context(ds, true);
        require_symmetry_shell_layouts(ds, "analytic head/wing");
    }

    const auto &freqs = ds.tfg.get_freq_nodes();
    const bool use_kpara_eigvec = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    const ArrayDesc &desc_scf_wfc =
        ds.eigvecs_kpara_2d_ready() ? ds.desc_wfc_kb : ds.desc_wfc_kb_full;
    const auto &headwing_pbc = use_full_bz_frame ? ds.headwing_full_bz_pbc : ds.pbc;
    // The full-BZ frame is a plain mesh with no k-point-parallel distribution, so
    // the SCF k-BLACS context does not describe it.
    const bool headwing_kpara = use_full_bz_frame ? false : use_kpara_eigvec;
    auto optical = std::make_unique<diele_func>(
        mf, use_full_bz_frame ? ds.headwing_full_bz_velocity : ds.velocity_matrix, headwing_kfrac,
        ds.basis_wfc, headwing_basis_aux, freqs, mf.get_n_aos(), mf.get_n_states(),
        mf.get_n_spins(), headwing_basis_aux.nb_total, headwing_pbc, ds.comm_h, ds.blacs_h,
        headwing_kpara, use_full_bz_frame ? nullptr : &ds.scfk_blacs_ctxt, &desc_scf_wfc);
    optical->use_2d_dielectric = opts.use_2d_dielectric == LIBRPA_SWITCH_ON;
    optical->use_soc = mf.get_n_spinor() > 1;
    optical->debug = global::should_output(LIBRPA_VERBOSE_DEBUG);
    if (use_headwing_symmetry && ds.symmetry_context.available &&
        !ds.symmetry_context.kstars.empty())
    {
        optical->set_symmetry_context(ds.symmetry_context);
        optical->use_symmetry = true;
        optical->atom_nw = ds.basis_wfc.get_atom_nb_map();
        optical->coord_frac = ds.symmetry_context.input_coord_frac;
    }
    optical->normalize_ibz_star_weighted_occupations();
    if (excluded) optical->set_excluded_band_pairs(*excluded);
    optical->init(opts.sqrt_coulomb_threshold, ds.vq);
    optical->cal_head();
    optical->test_head();

    if (need_wing)
    {
        const auto &headwing_cs =
            opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.cs_data_shrink : ds.cs_data;
        optical->cal_wing(headwing_cs, opts.sqrt_coulomb_threshold, ds.vq);
        optical->test_wing();
    }

    return optical;
}

void initialize_ds_headwing(Dataset &ds, const LibrpaOptions &opts, const bool need_wing)
{
    if (opts.replace_w_head != LIBRPA_SWITCH_ON ||
        (opts.option_dielect_func != 3 && opts.option_dielect_func != 4 &&
         opts.option_dielect_func != 53))
    {
        return;
    }

    global::profiler.start("initialize_ds_headwing");

    if (ds.p_headwing && (!need_wing || ds.p_headwing->has_wing()))
    {
        global::profiler.stop("initialize_ds_headwing");
        return;
    }

    if (ds.p_headwing && need_wing)
    {
        const auto &headwing_cs =
            opts.use_shrink_abfs == LIBRPA_SWITCH_ON ? ds.cs_data_shrink : ds.cs_data;
        ds.p_headwing->cal_wing(headwing_cs, opts.sqrt_coulomb_threshold, ds.vq);
        ds.p_headwing->test_wing();
        global::profiler.stop("initialize_ds_headwing");
        return;
    }

    ds.p_headwing = make_analytic_headwing(ds, opts, need_wing);
    ds.epsmacs_imagfreq = ds.p_headwing->get_head_vec();
    ds.omegas_imagfreq = ds.tfg.get_freq_nodes();

    global::profiler.stop("initialize_ds_headwing");
}

}  // namespace librpa_int
