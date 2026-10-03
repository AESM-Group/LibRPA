// Public API headers
#include "librpa_compute.h"

// Standard C++ headers
#include <algorithm>
#include <iomanip>
#include <limits>
#include <map>
#include <ostream>
#include <set>
#include <string>
#include <vector>

// Headers for API
#include "compute_helper.h"
#include "dataset_helper.h"
#include "instance_manager.h"

// Internal headers
#include "../core/analycont.h"
#include "../core/coulmat.h"
#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../core/meanfield.h"
#include "../core/qpe_solver.h"
#include "../core/timefreq.h"
#include "../io/fs.h"
#include "../io/global_io.h"
#include "../math/complexmatrix.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/constants.h"
#include "../utils/error.h"
// #include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "../utils/utils_mem.h"

#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#endif

using librpa_int::global::profiler;
using librpa_int::global::ofs_myid;
using librpa_int::global::lib_printf;

namespace
{

struct GwAnaconGrid
{
    std::vector<librpa_int::cplxdb> imagfreqs;
    bool resample;
};

// Map a compact collected k-list back to its buffer position.
std::map<int, int> make_ik_pos_map(const std::vector<int> &iks)
{
    std::map<int, int> ik_pos;
    for (int i = 0; i != static_cast<int>(iks.size()); ++i)
    {
        ik_pos.emplace(iks[i], i);
    }
    return ik_pos;
}

// Linearized index for the collected SigC diagonal buffer:
// [spin][collected k][frequency][requested state].
int sigc_diag_index(const int isp, const int ik_collect, const int ifreq, const int i_state,
                    const int nk_collect, const int nfreq, const int n_states_calc)
{
    return (((isp * nk_collect + ik_collect) * nfreq + ifreq) * n_states_calc + i_state);
}

// Linearized index for public spectral-function output:
// [spin][requested k][requested state][real-frequency].
int spectral_function_index(const int isp, const int ik_this, const int i_state,
                            const int iomega, const int n_kpts_this,
                            const int n_states_calc, const int n_omegas)
{
    return (((isp * n_kpts_this + ik_this) * n_states_calc + i_state) * n_omegas
            + iomega);
}

std::vector<librpa_int::cplxdb> make_imagfreqs(const std::vector<double> &freq_nodes)
{
    std::vector<librpa_int::cplxdb> imagfreqs;
    imagfreqs.reserve(freq_nodes.size());
    for (const auto &freq: freq_nodes)
    {
        imagfreqs.emplace_back(0.0, freq);
    }
    return imagfreqs;
}

std::vector<librpa_int::cplxdb> make_real_omegas(const int n_omegas,
                                                  const double *omegas)
{
    std::vector<librpa_int::cplxdb> real_omegas;
    real_omegas.reserve(n_omegas);
    for (int iomega = 0; iomega != n_omegas; ++iomega)
    {
        real_omegas.emplace_back(omegas[iomega], 0.0);
    }
    return real_omegas;
}

GwAnaconGrid make_gw_anacon_grid(
    const LibrpaOptions &opts, const librpa_int::MeanField &mf,
    const librpa_int::TFGrids &source_tfg, const std::vector<double> &source_freq_nodes)
{
    if (opts.anacon_nfreq == 0)
    {
        throw LIBRPA_RUNTIME_ERROR("anacon_nfreq must not be zero");
    }

    const int source_nfreq = static_cast<int>(source_freq_nodes.size());
    const auto source_grid_type = source_tfg.get_grid_type();
    const bool use_new_nfreq = opts.anacon_nfreq > 0
                                && opts.anacon_nfreq != source_nfreq;
    const bool use_new_grid_type = opts.anacon_tfgrids_type != LIBRPA_TFGRID_UNSET
                                   && opts.anacon_tfgrids_type != source_grid_type;
    if (!use_new_nfreq && !use_new_grid_type)
    {
        return {make_imagfreqs(source_freq_nodes), false};
    }

    const int nfreq = use_new_nfreq ? opts.anacon_nfreq : source_nfreq;

    const auto grid_type = use_new_grid_type
                               ? opts.anacon_tfgrids_type
                               : source_grid_type;
    if (grid_type == LIBRPA_TFGRID_UNSET)
    {
        throw LIBRPA_RUNTIME_ERROR("anacon_tfgrids_type is unset and source grid type is unavailable");
    }

    librpa_int::TFGrids tfg(nfreq);
    double emin = opts.tfgrids_freq_min;
    double eintv = opts.tfgrids_freq_interval;
    double emax = opts.tfgrids_freq_max;
    const double tmin = opts.tfgrids_time_min;
    const double tintv = opts.tfgrids_time_interval;
    const double regulation = opts.minimax_regulation;
    if (grid_type == LIBRPA_TFGRID_MINIMAX)
    {
        double emin_mf, emax_mf;
        mf.get_E_min_max(emin_mf, emax_mf);
        emax = opts.minimax_emax > 0 ? opts.minimax_emax : emax_mf;
        emin = opts.minimax_emin > 0 ? opts.minimax_emin : emin_mf;
    }
    tfg.generate(grid_type, emin, eintv, emax, tmin, tintv, regulation);

    const auto freq_nodes = tfg.get_freq_nodes();
    return {make_imagfreqs(freq_nodes), true};
}

int gw_anacon_resample_n_params(const LibrpaOptions &opts)
{
    return opts.n_params_anacon_resample < 0
               ? opts.n_params_anacon
               : opts.n_params_anacon_resample;
}

librpa_int::AnalyContPade make_gw_anacon_pade(
    const int n_params_anacon,
    const int n_params_anacon_resample,
    const std::vector<librpa_int::cplxdb> &source_imagfreqs,
    const std::vector<librpa_int::cplxdb> &source_data,
    const GwAnaconGrid &anacon_grid)
{
    if (n_params_anacon == 0)
    {
        throw LIBRPA_RUNTIME_ERROR("n_params_anacon must not be zero");
    }
    if (!anacon_grid.resample)
    {
        return librpa_int::AnalyContPade(n_params_anacon, source_imagfreqs, source_data);
    }
    if (n_params_anacon_resample == 0)
    {
        throw LIBRPA_RUNTIME_ERROR("n_params_anacon_resample must not be zero");
    }

    librpa_int::AnalyContPade source_pade(
        n_params_anacon_resample, source_imagfreqs, source_data);
    std::vector<librpa_int::cplxdb> data;
    data.reserve(anacon_grid.imagfreqs.size());
    for (const auto &freq: anacon_grid.imagfreqs)
    {
        data.emplace_back(source_pade.get(freq));
    }
    return librpa_int::AnalyContPade(n_params_anacon, anacon_grid.imagfreqs, data);
}

std::vector<librpa_int::cplxdb> extract_sigc_state(
    const std::vector<librpa_int::cplxdb> &sigc_diag, const int isp,
    const int ik_collect, const int i_state, const int nk_collect, const int nfreq,
    const int n_states_calc)
{
    std::vector<librpa_int::cplxdb> sigc_state;
    sigc_state.reserve(nfreq);
    for (int ifreq = 0; ifreq != nfreq; ++ifreq)
    {
        const int idx = sigc_diag_index(isp, ik_collect, ifreq, i_state,
                                        nk_collect, nfreq, n_states_calc);
        sigc_state.emplace_back(sigc_diag[idx]);
    }
    return sigc_state;
}

std::vector<double> compute_hedin_shifts(
    const LibrpaOptions &opts, const librpa_int::MeanField &mf,
    const std::vector<librpa_int::cplxdb> &sigc_diag,
    const std::vector<int> &iks_collect, const std::vector<double> &freq_nodes,
    const std::vector<librpa_int::cplxdb> &imagfreqs,
    const GwAnaconGrid &anacon_grid, const int n_spins, const int n_kpts_this,
    const int *iks_this, const int i_state_low, const int n_states_calc,
    const double *vxc, const double *vexx, const char *api_name)
{
    std::vector<double> shifts(n_spins * n_kpts_this * n_states_calc, 0.0);
    if (opts.use_hedin_shift != LIBRPA_SWITCH_ON) return shifts;
    if (shifts.empty()) return shifts;
    if (iks_collect.empty())
        throw LIBRPA_RUNTIME_ERROR("failed to collect inputs for Hedin shift in "
                                  + std::string(api_name));

    const int i_state_ref_abs = opts.istate_ref_hedin_shift;
    if (i_state_ref_abs >= mf.get_n_states())
        throw LIBRPA_RUNTIME_ERROR("Hedin shift reference state index out of range: "
                                  + std::to_string(i_state_ref_abs));
    if (i_state_ref_abs >= 0
        && (i_state_ref_abs < i_state_low || i_state_ref_abs >= i_state_low + n_states_calc))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Hedin shift absolute reference state " + std::to_string(i_state_ref_abs)
            + " is outside selected absolute state range [" + std::to_string(i_state_low)
            + ", " + std::to_string(i_state_low + n_states_calc) + ") in "
            + api_name);
    }

    const auto ik_pos = make_ik_pos_map(iks_collect);
    const int nk_collect = static_cast<int>(iks_collect.size());
    const int nfreq = static_cast<int>(freq_nodes.size());

    for (int isp = 0; isp != n_spins; ++isp)
    {
        const int start_isp = isp * n_kpts_this * n_states_calc;
        for (int ik_this = 0; ik_this != n_kpts_this; ++ik_this)
        {
            const int ik = iks_this[ik_this];
            const auto ik_pos_it = ik_pos.find(ik);
            if (ik_pos_it == ik_pos.end())
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "Hedin shift k-point " + std::to_string(ik)
                    + " is not among requested k-points in " + api_name);
            }
            const int ik_collect = ik_pos_it->second;
            const int start_k = start_isp + ik_this * n_states_calc;

            const auto compute_shift = [&](const int i_ref, const int state_ref) {
                const int ref_idx = start_k + i_ref;
                const auto sigc_ref_state = extract_sigc_state(
                    sigc_diag, isp, ik_collect, i_ref, nk_collect, nfreq, n_states_calc);
                const auto pade = make_gw_anacon_pade(
                    opts.n_params_anacon, gw_anacon_resample_n_params(opts),
                    imagfreqs, sigc_ref_state, anacon_grid);
                const auto sigc_ref = pade.get(librpa_int::cplxdb{
                    mf.get_eigenvals()[isp](ik, state_ref) - mf.get_efermi(), 0.0});
                return sigc_ref.real() + vexx[ref_idx] - vxc[ref_idx];
            };

            if (i_state_ref_abs >= 0)
            {
                const double shift = compute_shift(i_state_ref_abs - i_state_low,
                                                   i_state_ref_abs);
                for (int i = 0; i != n_states_calc; ++i) shifts[start_k + i] = shift;
                continue;
            }

            for (int i = 0; i != n_states_calc; ++i)
            {
                shifts[start_k + i] = compute_shift(i, i_state_low + i);
            }
        }
    }

    ofs_myid << "Computed Hedin shifts in " << api_name
             << (i_state_ref_abs >= 0 ? " from one reference state per k-point"
                                      : " per state")
             << std::endl;
    return shifts;
}

void validate_spectral_function_inputs(const int n_omegas, const double *omegas,
                                       const double *vxc, const double *vexx,
                                       const double *spectral_function)
{
    if (n_omegas < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("n_omegas must be non-negative");
    }
    if (n_omegas == 0)
    {
        return;
    }
    if (n_omegas > 0 && omegas == nullptr)
    {
        throw LIBRPA_RUNTIME_ERROR("omegas is null");
    }
    if (vxc == nullptr)
    {
        throw LIBRPA_RUNTIME_ERROR("vxc is null");
    }
    if (vexx == nullptr)
    {
        throw LIBRPA_RUNTIME_ERROR("vexx is null");
    }
    if (spectral_function == nullptr)
    {
        throw LIBRPA_RUNTIME_ERROR("spectral_function is null");
    }
}

void evaluate_spectral_function_diagonal(
    const librpa_int::MeanField &mf, const int n_params_anacon,
    const int n_params_anacon_resample,
    const std::vector<librpa_int::cplxdb> &sigc_diag,
    const std::vector<int> &iks_collect, const std::vector<double> &freq_nodes,
    const std::vector<librpa_int::cplxdb> &imagfreqs,
    const GwAnaconGrid &anacon_grid, const int n_spins, const int n_kpts_this,
    const int *iks_this, const int i_state_low, const int n_states_calc,
    const int n_omegas, const double *omegas, const double *vxc, const double *vexx,
    const double sigc_omega_imag_shift, const double gf_omega_imag_shift,
    double *spectral_function, double *sigc)
{
    const auto ik_pos = make_ik_pos_map(iks_collect);
    const int nk_collect = static_cast<int>(iks_collect.size());
    const int nfreq = static_cast<int>(freq_nodes.size());
    const auto real_omegas = make_real_omegas(n_omegas, omegas);
    const double efermi = mf.get_efermi();

    for (int isp = 0; isp != n_spins; ++isp)
    {
        const int start_isp = isp * n_kpts_this * n_states_calc;
        for (int ik_this = 0; ik_this != n_kpts_this; ++ik_this)
        {
            const int start_k = start_isp + ik_this * n_states_calc;
            const int ik = iks_this[ik_this];
            const int ik_collect = ik_pos.at(ik);
            for (int i = 0; i != n_states_calc; ++i)
            {
                const int i_state = i_state_low + i;
                const double eks_state = mf.get_eigenvals()[isp](ik, i_state);
                const double vxc_state = vxc[start_k + i];
                const double exx_state = vexx[start_k + i];
                const auto sigc_state = extract_sigc_state(
                    sigc_diag, isp, ik_collect, i, nk_collect, nfreq, n_states_calc);
                const auto pade = make_gw_anacon_pade(
                    n_params_anacon, n_params_anacon_resample,
                    imagfreqs, sigc_state, anacon_grid);
                for (int iomega = 0; iomega != n_omegas; ++iomega)
                {
                    const auto omega_gf = real_omegas[iomega]
                                          + librpa_int::cplxdb(0.0, gf_omega_imag_shift);
                    const auto omega_sigc = real_omegas[iomega]
                                            + librpa_int::cplxdb(0.0, sigc_omega_imag_shift);
                    const auto sigc_omega = pade.get(omega_sigc - efermi);
                    const auto gf_inv = omega_gf - eks_state + vxc_state - exx_state
                                        - sigc_omega;
                    const int idx = spectral_function_index(
                        isp, ik_this, i, iomega, n_kpts_this, n_states_calc, n_omegas);
                    if (sigc != nullptr)
                    {
                        sigc[2 * idx] = sigc_omega.real();
                        sigc[2 * idx + 1] = sigc_omega.imag();
                    }
                    spectral_function[idx] = -((1.0 / librpa_int::PI) / gf_inv).imag();
                }
            }
        }
    }
}

void ensure_band_sigc_ks_blacs(librpa_int::Dataset &ds, const LibrpaOptions &opts,
                               const std::vector<int> &iks_output)
{
    if (ds.is_band_calc_done && ds.p_g0w0
        && ds.p_g0w0->sigc_kspace_source().rfind("band_", 0) == 0
        && !ds.p_g0w0->sigc_diag_is_ik_f_KS.empty())
    {
        return;
    }

    if (!ds.is_band_calc_done && ds.p_exx)
    {
        ds.p_exx->reset_kspace();
    }
    ds.p_g0w0->reset_kspace();
    const auto bvk_remap = librpa_int::api::build_band_bvk_remap(
        ds.atoms, ds.pbc, opts.option_bvk_remap);
    const bool use_band_kblacs_wfc = opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON;
    if (use_band_kblacs_wfc)
    {
        ds.redistribute_band_eigvecs_kpara_2d();
    }
    ds.p_g0w0->build_sigc_matrix_KS_band_blacs(ds.mf_band.get_eigenvectors(),
                                               ds.kfrac_band_list,
                                               bvk_remap, ds.blacs_h,
                                               opts.use_gpu_replace_scalapack,
                                               &iks_output);
    ds.is_band_calc_done = true;
}

void write_sigc_matrices_KS_binary(librpa_int::Dataset &ds, const std::string &output_dir,
                                   const std::string &source)
{
    profiler.start("g0w0_export_sigc_KS", "Export self-energy in KS basis");
    ds.p_g0w0->write_sigc_matrices_KS_binary(output_dir, source);
    profiler.stop("g0w0_export_sigc_KS");
}

std::map<double, librpa_int::atom_mapping<std::map<librpa_int::Vector3_Order<double>,
                                                   librpa_int::Matz>>::pair_t_old>
collect_w_blacs_to_atom_pairs(
    const std::map<double, std::map<librpa_int::Vector3_Order<double>, librpa_int::Matz>>
        &Wc_freq_q,
    const librpa_int::AtomicBasis &basis_aux,
    const librpa_int::ArrayDesc &desc_abf,
    const librpa_int::MpiCommHandler &comm_h)
{
    using namespace librpa_int;

    if (!desc_abf.initialized())
        throw LIBRPA_RUNTIME_ERROR("Cannot collect Wc to atom pairs: ABF descriptor is not initialized");

    const auto atpairs_local = dispatch_upper_triangular_tasks(
        basis_aux.n_atoms, desc_abf.myid(), desc_abf.nprows(), desc_abf.npcols(),
        desc_abf.myprow(), desc_abf.mypcol());

    std::pair<std::set<int>, std::set<int>> target_atoms;
    for (const auto &atom_pair : atpairs_local)
    {
        target_atoms.first.insert(atom_pair.first);
        target_atoms.second.insert(atom_pair.second);
    }

    std::map<int, std::vector<int>> local_rows_by_atom;
    std::map<int, std::vector<int>> local_cols_by_atom;
    int atom = 0;
    int orbital = 0;
    for (int iloc = 0; iloc != desc_abf.m_loc(); ++iloc)
    {
        basis_aux.get_local_index(desc_abf.indx_l2g_r(iloc), atom, orbital);
        local_rows_by_atom[atom].push_back(orbital);
    }
    for (int iloc = 0; iloc != desc_abf.n_loc(); ++iloc)
    {
        basis_aux.get_local_index(desc_abf.indx_l2g_c(iloc), atom, orbital);
        local_cols_by_atom[atom].push_back(orbital);
    }

    std::map<double, atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old>
        Wc_freq_q_atom_pair;
    for (const auto &[freq, q_Wc] : Wc_freq_q)
    {
        for (const auto &[q, Wc_blacs] : q_Wc)
        {
            const std::array<double, 3> qa{q.x, q.y, q.z};
            std::map<int, std::map<int, Matz>> Wc_blocks;
            map_block_to_IJ_storage_new(Wc_blocks, basis_aux, local_rows_by_atom,
                                        local_cols_by_atom, Wc_blacs, desc_abf, MAJOR::ROW);

            std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<cplxdb>>>
                Wc_tensors;
            for (const auto &[iatom, jatom_Wc] : Wc_blocks)
            {
                const auto ni = basis_aux.get_atom_nb(iatom);
                for (const auto &[jatom, Wc_block] : jatom_Wc)
                {
                    const auto nj = basis_aux.get_atom_nb(jatom);
                    Wc_tensors[iatom][{jatom, qa}] =
                        RI::Tensor<cplxdb>({ni, nj}, Wc_block.sptr());
                }
            }

            const auto Wc_collected = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                comm_h.comm, Wc_tensors, target_atoms.first, target_atoms.second);
            for (const auto &[iatom, jatom] : atpairs_local)
            {
                const auto ni = basis_aux.get_atom_nb(iatom);
                const auto nj = basis_aux.get_atom_nb(jatom);
                Wc_freq_q_atom_pair[freq][iatom][jatom][q] =
                    Matz(ni, nj, Wc_collected.at(iatom).at({jatom, qa}).data, MAJOR::ROW);
            }
        }
    }

    return Wc_freq_q_atom_pair;
}

int solve_qpe_with_option(const int option_qpe_solver,
                          const librpa_int::AnalyContPade &pade,
                          const double e_mf,
                          const double e_fermi,
                          const double vxc,
                          const double sigma_x,
                          double &e_qp,
                          librpa_int::cplxdb &sigc,
                          const double diff_init,
                          const double thres,
                          const int n_iter_max,
                          const double damp_fac,
                          const bool use_adaptive_damp,
                          const bool use_legacy_nonadaptive_update)
{
    switch (option_qpe_solver)
    {
    case 0:
        return librpa_int::qpe_solver_pade_self_consistent(
            pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, diff_init, thres,
            n_iter_max, damp_fac, use_adaptive_damp, use_legacy_nonadaptive_update);
    case 1:
        return librpa_int::qpe_solver_pade_quasi_newton(
            pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, diff_init, thres,
            n_iter_max, damp_fac, use_adaptive_damp);
    case 2:
    {
        librpa_int::cplxdb sigc_deriv;
        double qp_weight = std::numeric_limits<double>::quiet_NaN();
        const int info = librpa_int::qpe_solver_pade_perturbative(
            pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, sigc_deriv, qp_weight);
        if (info == 0)
        {
            // The public API returns SigC and existing drivers reconstruct
            // e_qp = e_mf - vxc + sigma_x + Re SigC from it.  For perturbative
            // mode, fold the renormalization into an effective real SigC.
            sigc.real(e_qp - e_mf + vxc - sigma_x);
        }
        return info;
    }
    default:
        throw LIBRPA_RUNTIME_ERROR(
            "Invalid option_qpe_solver: " + std::to_string(option_qpe_solver)
            + ". Available values are 0 (damped residual-mixing), 1 (quasi-Newton), and 2 (perturbative).");
    }
}

// Publish SigC diagonal values from the rank that owns each rotated k-block.
// The caller later uses these values locally with its matching vxc/vexx input.
std::vector<librpa_int::cplxdb> collect_sigc_diag_to_callers(
    const std::map<int, std::map<int, std::map<double, std::vector<librpa_int::cplxdb>>>> &sigc_diag_is_ik_f_KS,
    const std::vector<double> &freqs,
    const librpa_int::MpiCommHandler &comm_h,
    const bool publish_local_values,
    const int n_spins,
    const int n_kpoints,
    const int n_kpts_this,
    const int *iks_this,
    const int i_state_low,
    const int n_states_calc,
    std::vector<int> &iks_collect)
{
    iks_collect =
        librpa_int::api::collect_requested_iks(comm_h, n_kpts_this, iks_this, n_kpoints);
    if (iks_collect.empty()) return {};

    const int nk_collect = static_cast<int>(iks_collect.size());
    const int nfreq = static_cast<int>(freqs.size());
    const int n_owner = n_spins * nk_collect;
    const int n_value = n_owner * nfreq * n_states_calc;

    std::vector<int> owners(n_owner, 0);
    std::vector<int> owners_sum(n_owner, 0);
    std::vector<librpa_int::cplxdb> values(n_value, librpa_int::cplxdb{0.0, 0.0});
    std::vector<librpa_int::cplxdb> values_sum(n_value, librpa_int::cplxdb{0.0, 0.0});

    if (publish_local_values)
    {
        for (int isp = 0; isp != n_spins; ++isp)
        {
            const auto it_sp = sigc_diag_is_ik_f_KS.find(isp);
            if (it_sp == sigc_diag_is_ik_f_KS.cend()) continue;
            for (int ik_collect = 0; ik_collect != nk_collect; ++ik_collect)
            {
                const int ik = iks_collect[ik_collect];
                const auto it_k = it_sp->second.find(ik);
                if (it_k == it_sp->second.cend()) continue;

                owners[isp * nk_collect + ik_collect] = 1;
                for (int ifreq = 0; ifreq != nfreq; ++ifreq)
                {
                    const double freq = freqs[ifreq];
                    const auto it_freq = it_k->second.find(freq);
                    if (it_freq == it_k->second.cend())
                    {
                        throw LIBRPA_RUNTIME_ERROR(
                            "fail to locate sigc diagonal at ik = " + std::to_string(ik) +
                            " freq = " + std::to_string(freq));
                    }
                    const auto &diag = it_freq->second;
                    if (static_cast<int>(diag.size()) < i_state_low + n_states_calc)
                    {
                        throw LIBRPA_RUNTIME_ERROR(
                            "sigc diagonal size insufficient at ik = " + std::to_string(ik) +
                            " freq = " + std::to_string(freq));
                    }
                    for (int i = 0; i != n_states_calc; ++i)
                    {
                        const int idx = sigc_diag_index(isp, ik_collect, ifreq, i,
                                                        nk_collect, nfreq, n_states_calc);
                        const int i_state = i_state_low + i;
                        values[idx] = diag[i_state];
                    }
                }
            }
        }
    }

    comm_h.allreduce(values.data(), values_sum.data(), n_value, MPI_SUM);
    comm_h.allreduce(owners.data(), owners_sum.data(), n_owner, MPI_SUM);

    for (int isp = 0; isp != n_spins; ++isp)
    {
        for (int ik_collect = 0; ik_collect != nk_collect; ++ik_collect)
        {
            const int owner_count = owners_sum[isp * nk_collect + ik_collect];
            if (owner_count != 1)
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "failed to locate a unique SigC value owner for spin = " +
                    std::to_string(isp) + " ik = " + std::to_string(iks_collect[ik_collect]) +
                    ", owner count = " + std::to_string(owner_count));
            }
        }
    }

    return values_sum;
}

constexpr int analycont_source_dump_precision = 11;

void dump_analycont_source_data(std::ostream &os, const librpa_int::AnalyCont &ac,
                                const char *api_name, const int isp, const int ik,
                                const int i_state)
{
    const auto &xs = ac.get_source_xs();
    const auto &data = ac.get_source_data();
    const auto flags = os.flags();
    const auto precision = os.precision();

    os << "Source data used by analytic continuation in " << api_name
       << " for spin " << isp + 1 << " kpoint " << ik + 1
       << " state " << i_state + 1 << std::endl;
    os << "n_analycont_source_data = " << data.size() << std::endl;
    os << "i x_re x_im y_re y_im" << std::endl;
    os << std::scientific << std::setprecision(analycont_source_dump_precision);
    const int n = std::min(static_cast<int>(xs.size()), static_cast<int>(data.size()));
    for (int i = 0; i != n; ++i)
    {
        os << i + 1 << " " << xs[i].real() << " " << xs[i].imag() << " "
           << data[i].real() << " " << data[i].imag() << std::endl;
    }
    os.flags(flags);
    os.precision(precision);
}

} // namespace

void librpa_build_g0w0_sigma(LibrpaHandler* h, const LibrpaOptions *p_opts)
{
    using namespace librpa_int;
    using librpa_int::global::lib_printf;

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &opts = *p_opts;
    const bool debug = global::should_output(LIBRPA_VERBOSE_DEBUG);
    pds->is_band_calc_done = false;
    initialize_ds_global_ddla(*pds, opts);

    profiler.start("api_build_g0w0_sigma");

    // Prepare time-frequency grids
    initialize_ds_tfgrids(*pds, opts);
    if (opts.replace_w_head == LIBRPA_SWITCH_ON && opts.option_dielect_func == 0
        && !pds->epsmacs_imagfreq.empty()
        && pds->epsmacs_imagfreq.size() != pds->tfg.get_freq_nodes().size())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "option_dielect_func=0 requires dielectric function data on the current "
            "frequency grid: got " + std::to_string(pds->epsmacs_imagfreq.size()) +
            " input points for " + std::to_string(pds->tfg.get_freq_nodes().size()) +
            " target points. Regenerate dielecfunc_out for this grid or set "
            "option_dielect_func to an interpolation/fitting mode.");
    }

    // Decide actual routing
    LibrpaParallelRouting routing = opts.parallel_routing;
    if (routing == LIBRPA_ROUTING_AUTO)
    {
        const int n_atoms = pds->atoms.size();
        routing = decide_auto_routing(n_atoms, opts.nfreq * pds->pbc.get_n_cells_bvk());
    }

    if (opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON)
    {
        if (routing == LIBRPA_ROUTING_LIBRI)
            pds->redistribute_eigvecs_kpara_2d();
        else
            pds->redistribute_eigvecs_kpara();
    }

    auto ensure_exx = [&]()
    {
        if (pds->p_exx) return;

        profiler.start("g0w0_exx", "Build exchange self-energy");
        initialize_ds_exx(*pds, opts);
        const bool use_shrink_abfs = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
        const auto &basis_aux_exx = use_shrink_abfs ? pds->basis_aux_shrink : pds->basis_aux;
        const auto &cs_data_exx = use_shrink_abfs ? pds->cs_data_shrink : pds->cs_data;
        const auto &coul = opts.use_fullcoul_exx ? pds->vq : pds->vq_cut;
        profiler.start("ft_vq_cut", "Fourier transform truncated Coulomb");
        const auto VR = librpa_int::FT_Vq(
            pds->comm_h, basis_aux_exx, pds->symmetry_context, coul, pds->pbc, true,
            opts.use_symmetry_exx == LIBRPA_SWITCH_ON);
        profiler.stop("ft_vq_cut");

        profiler.start("g0w0_exx_real_work");
        pds->p_exx->build(routing, basis_aux_exx, cs_data_exx, VR);
        // pds->p_exx->build_KS_kgrid_blacs(pds->blacs_h);
        profiler.stop("g0w0_exx_real_work");
        profiler.stop("g0w0_exx");
        pds->comm_h.barrier();
        release_free_mem();
    };

    // Determine the atom pairs that this process is responsible for
    initialize_ds_atpairs_local(*pds, routing);
    // Redistribute 2D Coulomb matrices to atom-pair blocks if they are parsed
    pds->redistribute_coulomb_blacs2ap();

    initialize_ds_headwing(*pds, opts, opts.option_dielect_func == 3);

    // Initialize response function object
    initialize_ds_chi0(*pds, opts);
    auto &chi0 = *(pds->p_chi0);

    if (opts.read_sigc_mat_rf == LIBRPA_SWITCH_ON)
    {
        ensure_exx();
        initialize_ds_g0w0(*pds, opts);
        const char *restart_from_dir = opts.restart_from_dir[0] ? opts.restart_from_dir : opts.output_dir;
        pds->p_g0w0->read_sigc(restart_from_dir);
        profiler.stop("api_build_g0w0_sigma");
        return;
    }

    profiler.start("chi0_build", "Build response function chi0");
    const bool use_shrink_abfs = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
    const bool shrink_chi_from_full_abfs =
        use_shrink_abfs && opts.use_shrink_chi == LIBRPA_SWITCH_ON;
    const auto &cs_data_chi0 =
        use_shrink_abfs && !shrink_chi_from_full_abfs ? pds->cs_data_shrink : pds->cs_data;
    const auto &basis_aux_chi0 =
        use_shrink_abfs && !shrink_chi_from_full_abfs ? pds->basis_aux_shrink : pds->basis_aux;
    std::map<Vector3_Order<double>, ComplexMatrix> empty_sinvS;
    auto &sinvS_chi0 = shrink_chi_from_full_abfs ? pds->sinvS : empty_sinvS;
    chi0.build(routing, cs_data_chi0, pds->atpairs_local, basis_aux_chi0, sinvS_chi0,
               pds->blacs_h);
    profiler.stop("chi0_build");
    pds->comm_h.barrier();

    if (debug)
    { // debug, check chi0
        for (const auto &chi0q: chi0.get_chi0_q())
        {
            const int ifreq = chi0.tfg.get_freq_index(chi0q.first);
            for (const auto &[q, IJchi0]: chi0q.second)
            {
                const int iq = pds->pbc.get_k_index_full(q);
                for (const auto &[I, J_chi0]: IJchi0)
                {
                    for (const auto &[J, chi0]: J_chi0)
                    {
                        std::stringstream ss;
                        ss << "chi0fq_ifreq_" << ifreq << "_iq_" << iq << "_I_" << I << "_J_" << J << "_id_" << pds->comm_h.myid << ".mtx";
                        print_complex_matrix_mm(chi0, librpa_int::path_as_directory(opts.output_dir) + ss.str(), 1e-15);
                    }
                }
            }
        }
    }
    pds->comm_h.barrier();

    ensure_exx();

    profiler.start("g0w0_wc", "Build screened interaction");

    std::vector<double> epsmac_LF_imagfreq_re;
    if (opts.replace_w_head == LIBRPA_SWITCH_ON && pds->epsmacs_imagfreq.size() > 0)
    {
        if (opts.option_dielect_func == 3 || opts.option_dielect_func == 4)
        {
            if (pds->epsmacs_imagfreq.size() != chi0.tfg.get_freq_nodes().size())
                throw LIBRPA_RUNTIME_ERROR("analytic head/wing dielectric data must match chi0 frequency grid");
            epsmac_LF_imagfreq_re = pds->epsmacs_imagfreq;
        }
        else
        {
            epsmac_LF_imagfreq_re = interpolate_dielec_func(opts.option_dielect_func,
                                                            pds->omegas_imagfreq,
                                                            pds->epsmacs_imagfreq,
                                                            chi0.tfg.get_freq_nodes());
        }
        if (debug)
        {
            if (pds->comm_h.is_root())
            {
                lib_printf("Dielectric function parsed as head correction:\n");
                for (int i = 0; i < opts.nfreq; i++)
                    lib_printf("%d %f %f\n", i+1, chi0.tfg.get_freq_nodes()[i], epsmac_LF_imagfreq_re[i]);
            }
            pds->comm_h.barrier();
        }
    }
    std::vector<std::complex<double>> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(), epsmac_LF_imagfreq_re.cend());

    const bool strict_2d_complete_wc = strict_2d_complete_wc_requested(
        opts.replace_w_head == LIBRPA_SWITCH_ON, opts.option_dielect_func,
        opts.use_2d_dielectric == LIBRPA_SWITCH_ON);
    validate_strict_2d_complete_wc_runtime(strict_2d_complete_wc, !epsmac_LF_imagfreq.empty(),
                                           opts.use_scalapack_gw_wc == LIBRPA_SWITCH_ON,
                                           pds->p_headwing != nullptr &&
                                               pds->p_headwing->strict_2d_coulomb_head_configured());
    validate_strict_2d_gw_coulomb_choices(strict_2d_complete_wc,
                                          opts.use_fullcoul_eps == LIBRPA_SWITCH_ON,
                                          opts.use_fullcoul_wc == LIBRPA_SWITCH_ON);

    std::map<double, std::map<Vector3_Order<double>, librpa_int::matrix_m<std::complex<double>>>>
        Wc_freq_q;
    const auto &wc_desc_abf = use_shrink_abfs ? pds->desc_abf_shrink : pds->desc_abf;
    const auto &coul_eps = opts.use_fullcoul_eps ? pds->vq : pds->vq_cut;
    auto &coul_wc = opts.use_fullcoul_wc ? pds->vq : pds->vq_cut;
    if (opts.use_scalapack_gw_wc == LIBRPA_SWITCH_ON)
    {
        bool replace_w_head = opts.replace_w_head == LIBRPA_SWITCH_ON;
        Wc_freq_q = compute_Wc_freq_q_blacs(
            chi0, coul_eps, coul_wc, opts.sqrt_coulomb_threshold, replace_w_head,
            opts.option_dielect_func, epsmac_LF_imagfreq, pds->p_headwing.get(), pds->blacs_h,
            wc_desc_abf, debug, opts.output_dir, opts.use_cholesky_gw_wc,
            opts.use_gpu_replace_scalapack, opts.use_elpa_sqrt_coulomb,
            opts.output_2d_finite_q_diagnostics == LIBRPA_SWITCH_ON);
    }
    else
    {
        Wc_freq_q = compute_Wc_freq_q(chi0, coul_eps, coul_wc, opts.sqrt_coulomb_threshold,
                                      epsmac_LF_imagfreq, debug, opts.output_dir);
    }

    std::map<double, atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old>
        Wc_freq_q_atom_pair;
    if (use_shrink_abfs)
    {
        profiler.start("collect_Wc_blacs_to_atom_pairs",
                       "Collect compressed Wc from BLACS to atom pairs");
        Wc_freq_q_atom_pair =
            collect_w_blacs_to_atom_pairs(Wc_freq_q, pds->basis_aux_shrink,
                                          pds->desc_abf_shrink, pds->comm_h);
        profiler.stop("collect_Wc_blacs_to_atom_pairs");

        Wc_freq_q.clear();
    }
    profiler.stop("g0w0_wc");

    if (global::should_output(LIBRPA_VERBOSE_DEBUG))
    { // debug, check Wc
        for (const auto &[freq, q_Wc]: Wc_freq_q)
        {
            const int ifreq = chi0.tfg.get_freq_index(freq);
            for (const auto &[q, Wc]: q_Wc)
            {
                const int iq = pds->pbc.get_k_index_full(q);
                std::stringstream ss;
                ss << "Wcfq_ifreq_" << ifreq << "_iq_" << iq << ".csc";
                const auto fn = librpa_int::path_as_directory(opts.output_dir) + ss.str();
                write_matrix_elsi_csc_parallel(fn, Wc, pds->desc_abf, 1e-15);
            }
        }
    }

    initialize_ds_g0w0(*pds, opts);
    profiler.start("g0w0_sigc_IJ", "Build real-space correlation self-energy");
    // HACK: choice of space-time is hard-coded. May need to change when more approaches are implemented
    pds->p_g0w0->build_spacetime(
        routing, pds->basis_aux, pds->cs_data, Wc_freq_q, pds->desc_abf,
        use_shrink_abfs ? &Wc_freq_q_atom_pair : nullptr,
        use_shrink_abfs ? &pds->sinvS : nullptr,
        use_shrink_abfs ? &pds->basis_aux_shrink : nullptr,
        use_shrink_abfs ? &pds->basis_aux : nullptr,
        use_shrink_abfs ? &pds->blacs_h : nullptr,
        use_shrink_abfs
            ? &(pds->eigvecs_kpara_2d_ready() ? pds->desc_wfc_kb
                                              : pds->desc_wfc_kb_full)
            : nullptr);
    profiler.stop("g0w0_sigc_IJ");
    release_free_mem();

    profiler.stop("api_build_g0w0_sigma");
}

void librpa_get_g0w0_qpe_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
                               const int n_kpts_this, const int *iks_this, int i_state_low,
                               int i_state_high, const double *vxc, const double *vexx,
                               double *sigc_re, double *sigc_im, double *eqp)
{
    using namespace librpa_int;

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &opts = *p_opts;
    initialize_ds_global_ddla(*pds, opts);
    i_state_low = std::max(0, i_state_low);
    i_state_high = std::min(pds->mf.get_n_states(), i_state_high);
    if (n_spins != pds->mf.get_n_spins())
    {
        ofs_myid << "n_spins != pds->mf.get_n_spins(): " << n_spins << " != " << pds->mf.get_n_spins() << std::endl;
        throw LIBRPA_RUNTIME_ERROR("parsed nspins is not consitent with the SCF starting poing");
    }
    if (i_state_high <= i_state_low) return;
    const int n_states_calc = i_state_high - i_state_low;

    if (!pds->p_g0w0) librpa_build_g0w0_sigma(h, p_opts);

    profiler.start("api_get_g0w0_sigc_kgrid");

    // Decide actual routing
    LibrpaParallelRouting routing = opts.parallel_routing;
    if (routing == LIBRPA_ROUTING_AUTO)
    {
        const int n_atoms = pds->atoms.size();
        routing = decide_auto_routing(n_atoms, opts.nfreq * pds->pbc.get_n_cells_bvk());
    }

    profiler.start("g0w0_sigc_rotate_KS", "Correlation self-energy in K-S space");
    pds->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(pds->blacs_h, opts.use_gpu_replace_scalapack);
    pds->is_band_calc_done = false;
    profiler.stop("g0w0_sigc_rotate_KS");
    if (opts.output_gw_sigc_ks_mat_kf == LIBRPA_SWITCH_ON)
        write_sigc_matrices_KS_binary(*pds, opts.output_dir, "kgrid");

    std::vector<int> iks_collect;
    const auto freq_nodes = pds->tfg.get_freq_nodes();
    const bool publish_local_values =
        opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON || pds->blacs_h.myid == 0;
    const auto sigc_diag = collect_sigc_diag_to_callers(
        pds->p_g0w0->sigc_diag_is_ik_f_KS, freq_nodes, pds->comm_h, publish_local_values,
        n_spins, pds->mf.get_n_kpoints(), n_kpts_this, iks_this, i_state_low,
        n_states_calc, iks_collect);
    const auto ik_pos = make_ik_pos_map(iks_collect);
    const int nk_collect = static_cast<int>(iks_collect.size());
    const int nfreq = static_cast<int>(freq_nodes.size());

    profiler.start("g0w0_solve_qpe", "Solve quasi-particle equation");
    const auto efermi = pds->mf.get_efermi();
    const auto imagfreqs = make_imagfreqs(freq_nodes);
    const auto anacon_grid = make_gw_anacon_grid(opts, pds->mf, pds->tfg, freq_nodes);
    const auto hedin_shifts = compute_hedin_shifts(
        opts, pds->mf, sigc_diag, iks_collect, freq_nodes, imagfreqs, anacon_grid,
        n_spins, n_kpts_this, iks_this, i_state_low, n_states_calc, vxc, vexx,
        "librpa_get_g0w0_qpe_kgrid");

    const auto thres_qpe = opts.qpe_solver_thres;
    const auto n_iter_max = opts.qpe_solver_n_iter_max;
    const auto damp_fac = opts.qpe_solver_damp_factor;
    const auto option_qpe_solver = opts.option_qpe_solver;
    const double diff_init = option_qpe_solver == 0 ? 1.0e-3 : 0.0;
    const bool use_adaptive_damp = opts.use_qpe_adaptive_damp == LIBRPA_SWITCH_ON;
    const bool use_legacy_nonadaptive_update =
            opts.use_qpe_legacy_update == LIBRPA_SWITCH_ON;
    const bool override_qpe_solver_nan = opts.override_qpe_solver_nan == LIBRPA_SWITCH_ON;

    for (int isp = 0; isp < n_spins; isp++)
    {
        const int start_isp = isp * n_kpts_this * n_states_calc;
        // ofs_myid << exx_isp << endl;
        for (int ik_this = 0; ik_this < n_kpts_this; ik_this++)
        {
            const int start_k = start_isp + ik_this * n_states_calc;
            const int ik = *(iks_this + ik_this);
            const int ik_collect = ik_pos.at(ik);
            ofs_myid << "Start QPE solver for spin " << isp + 1
                             << " kpoint " << ik + 1 << std::endl;
            for (int i = 0; i < n_states_calc; i++)
            {
                const int i_state = i + i_state_low;
                const auto &eks_state = pds->mf.get_eigenvals()[isp](ik, i_state);
                const auto &exx_state = vexx[start_k+i];
                const auto &vxc_state = vxc[start_k+i];
                std::vector<cplxdb> sigc_state;
                for (int ifreq = 0; ifreq != nfreq; ++ifreq)
                {
                    const int idx = sigc_diag_index(isp, ik_collect, ifreq, i,
                                                    nk_collect, nfreq, n_states_calc);
                    sigc_state.emplace_back(sigc_diag[idx]);
                }
                double e_qp;
                cplxdb sigc;
                sigc_re[start_k+i] = std::numeric_limits<double>::quiet_NaN();
                sigc_im[start_k+i] = std::numeric_limits<double>::quiet_NaN();
                eqp[start_k+i] = std::numeric_limits<double>::quiet_NaN();
                const auto pade = make_gw_anacon_pade(
                    opts.n_params_anacon, gw_anacon_resample_n_params(opts),
                    imagfreqs, sigc_state, anacon_grid);
                int flag_qpe_solver = solve_qpe_with_option(
                    option_qpe_solver, pade, eks_state, efermi + hedin_shifts[start_k+i],
                    vxc_state, exx_state, e_qp, sigc, diff_init, thres_qpe,
                    n_iter_max, damp_fac, use_adaptive_damp,
                    use_legacy_nonadaptive_update);
                if (flag_qpe_solver != 0)
                {
                    ofs_myid << "Warning! QPE solver failed for spin " << isp + 1
                                     << " kpoint " << ik + 1 << " state " << i_state + 1
                                     << std::endl;
                    dump_analycont_source_data(ofs_myid, pade,
                                               "librpa_get_g0w0_sigc_kgrid", isp, ik, i_state);
                    if (override_qpe_solver_nan)
                    {
                        ofs_myid << "Using final unconverged QPE result because "
                                    "override_qpe_solver_nan is on"
                                 << std::endl;
                    }
                }
                if (flag_qpe_solver == 0 || override_qpe_solver_nan)
                {
                    sigc_re[start_k+i] = sigc.real();
                    sigc_im[start_k+i] = sigc.imag();
                    eqp[start_k+i] = e_qp;
                }
            }
        }
    }
    profiler.stop("g0w0_solve_qpe");

    profiler.stop("api_get_g0w0_sigc_kgrid");
}

void librpa_get_g0w0_sigc_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
                                const int n_kpts_this, const int *iks_this, int i_state_low,
                                int i_state_high, const double *vxc, const double *vexx,
                                double *sigc_re, double *sigc_im)
{
    const int n_states_calc = std::max(0, i_state_high - i_state_low);
    std::vector<double> eqp(n_spins * n_kpts_this * n_states_calc);
    librpa_get_g0w0_qpe_kgrid(h, p_opts, n_spins, n_kpts_this, iks_this, i_state_low,
                              i_state_high, vxc, vexx, sigc_re, sigc_im, eqp.data());
}

void librpa_get_g0w0_spectral_function_kgrid(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_this, const int *iks_this, int i_state_low, int i_state_high,
    const int n_omegas, const double *omegas, const double *vxc, const double *vexx,
    double *spectral_function, double *sigc)
{
    using namespace librpa_int;

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &opts = *p_opts;
    initialize_ds_global_ddla(*pds, opts);
    if (n_omegas < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("n_omegas must be non-negative");
    }

    i_state_low = std::max(0, i_state_low);
    i_state_high = std::min(pds->mf.get_n_states(), i_state_high);
    if (n_spins != pds->mf.get_n_spins())
    {
        ofs_myid << "n_spins != pds->mf.get_n_spins(): " << n_spins << " != "
                         << pds->mf.get_n_spins() << std::endl;
        throw LIBRPA_RUNTIME_ERROR("parsed nspins is not consitent with the SCF starting poing");
    }
    if (n_omegas == 0 || n_kpts_this == 0 || i_state_high <= i_state_low) return;
    const int n_states_calc = i_state_high - i_state_low;
    validate_spectral_function_inputs(n_omegas, omegas, vxc, vexx, spectral_function);

    if (!pds->p_g0w0) librpa_build_g0w0_sigma(h, p_opts);

    profiler.start("api_get_g0w0_spectral_function_kgrid");

    profiler.start("g0w0_sigc_rotate_KS", "Correlation self-energy in K-S space");
    pds->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(pds->blacs_h,
                                                  opts.use_gpu_replace_scalapack);
    pds->is_band_calc_done = false;
    profiler.stop("g0w0_sigc_rotate_KS");
    if (opts.output_gw_sigc_ks_mat_kf == LIBRPA_SWITCH_ON)
        write_sigc_matrices_KS_binary(*pds, opts.output_dir, "kgrid");

    std::vector<int> iks_collect;
    const auto freq_nodes = pds->tfg.get_freq_nodes();
    const bool publish_local_values =
        opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON || pds->blacs_h.myid == 0;
    const auto sigc_diag = collect_sigc_diag_to_callers(
        pds->p_g0w0->sigc_diag_is_ik_f_KS, freq_nodes, pds->comm_h, publish_local_values,
        n_spins, pds->mf.get_n_kpoints(), n_kpts_this, iks_this, i_state_low,
        n_states_calc, iks_collect);
    const auto imagfreqs = make_imagfreqs(freq_nodes);
    const auto anacon_grid = make_gw_anacon_grid(opts, pds->mf, pds->tfg, freq_nodes);

    profiler.start("g0w0_spectral_function", "Compute spectral function");
    evaluate_spectral_function_diagonal(
        pds->mf, opts.n_params_anacon, gw_anacon_resample_n_params(opts),
        sigc_diag, iks_collect, freq_nodes, imagfreqs, anacon_grid,
        n_spins, n_kpts_this, iks_this, i_state_low, n_states_calc, n_omegas,
        omegas, vxc, vexx, opts.sf_sigc_omega_shift, opts.sf_gf_omega_shift,
        spectral_function, sigc);
    profiler.stop("g0w0_spectral_function");

    profiler.stop("api_get_g0w0_spectral_function_kgrid");
}

void librpa_get_g0w0_qpe_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
                                const int n_kpts_band_this, const int *iks_band_this,
                                int i_state_low, int i_state_high, const double *vxc_band,
                                const double *vexx_band, double *sigc_band_re,
                                double *sigc_band_im, double *eqp_band)
{
    using namespace librpa_int;

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &opts = *p_opts;
    initialize_ds_global_ddla(*pds, opts);
    if (!pds->is_band_data_set || pds->mf_band.get_n_spins() == 0)
        throw LIBRPA_RUNTIME_ERROR("Meanfield data for band calculation is not set");
    i_state_low = std::max(0, i_state_low);
    i_state_high = std::min(pds->mf_band.get_n_states(), i_state_high);
    const int n_spins_band = pds->mf_band.get_n_spins();
    if (n_spins != n_spins_band)
    {
        ofs_myid << "n_spins != pds->mf_band.get_n_spins(): " << n_spins << " != " << n_spins_band << std::endl;
        throw LIBRPA_RUNTIME_ERROR("parsed n_spins is not consitent with the band input data");
    }
    if (i_state_high <= i_state_low) return;
    const int n_states_calc = i_state_high - i_state_low;

    if (!pds->p_g0w0) librpa_build_g0w0_sigma(h, p_opts);

    profiler.start("api_get_g0w0_sigc_band_k");

    // Decide actual routing
    LibrpaParallelRouting routing = opts.parallel_routing;
    if (routing == LIBRPA_ROUTING_AUTO)
    {
        const int n_atoms = pds->atoms.size();
        routing = decide_auto_routing(n_atoms, opts.nfreq * pds->pbc.get_n_cells_bvk());
    }

    const auto iks_output =
        librpa_int::api::collect_requested_iks(pds->comm_h, n_kpts_band_this,
                                              iks_band_this, pds->mf_band.get_n_kpoints());

    profiler.start("g0w0_sigc_rotate_KS", "Correlation self-energy in K-S space");
    ensure_band_sigc_ks_blacs(*pds, opts, iks_output);
    profiler.stop("g0w0_sigc_rotate_KS");
    if (opts.output_gw_sigc_ks_mat_kf == LIBRPA_SWITCH_ON)
        write_sigc_matrices_KS_binary(*pds, opts.output_dir,
                                      "band_" + std::to_string(pds->band_data_id));

    std::vector<int> iks_collect;
    const auto freq_nodes = pds->tfg.get_freq_nodes();
    const bool publish_local_values =
        opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON || pds->blacs_h.myid == 0;
    const auto sigc_diag = collect_sigc_diag_to_callers(
        pds->p_g0w0->sigc_diag_is_ik_f_KS, freq_nodes, pds->comm_h, publish_local_values,
        n_spins, pds->mf_band.get_n_kpoints(), n_kpts_band_this, iks_band_this,
        i_state_low, n_states_calc, iks_collect);
    const auto ik_pos = make_ik_pos_map(iks_collect);
    const int nk_collect = static_cast<int>(iks_collect.size());
    const int nfreq = static_cast<int>(freq_nodes.size());

    profiler.start("g0w0_solve_qpe", "Solve quasi-particle equation");
    const auto efermi = pds->mf_band.get_efermi();
    const auto imagfreqs = make_imagfreqs(freq_nodes);
    const auto anacon_grid = make_gw_anacon_grid(opts, pds->mf_band, pds->tfg, freq_nodes);
    const auto hedin_shifts = compute_hedin_shifts(
        opts, pds->mf_band, sigc_diag, iks_collect, freq_nodes, imagfreqs, anacon_grid,
        n_spins, n_kpts_band_this, iks_band_this, i_state_low, n_states_calc,
        vxc_band, vexx_band, "librpa_get_g0w0_qpe_band_k");

    const auto thres_qpe = opts.qpe_solver_thres;
    const auto n_iter_max = opts.qpe_solver_n_iter_max;
    const auto damp_fac = opts.qpe_solver_damp_factor;
    const auto option_qpe_solver = opts.option_qpe_solver;
    const double diff_init = option_qpe_solver == 0 ? 1.0e-3 : 0.0;
    const bool use_adaptive_damp = opts.use_qpe_adaptive_damp == LIBRPA_SWITCH_ON;
    const bool use_legacy_nonadaptive_update =
            opts.use_qpe_legacy_update == LIBRPA_SWITCH_ON;
    const bool override_qpe_solver_nan = opts.override_qpe_solver_nan == LIBRPA_SWITCH_ON;

    for (int isp = 0; isp < n_spins; isp++)
    {
        const int start_isp = isp * n_kpts_band_this * n_states_calc;
        // ofs_myid << exx_isp << endl;
        for (int ik_this = 0; ik_this < n_kpts_band_this; ik_this++)
        {
            const int start_k = start_isp + ik_this * n_states_calc;
            const int ik = *(iks_band_this + ik_this);
            const int ik_collect = ik_pos.at(ik);
            ofs_myid << "Start QPE solver for spin " << isp + 1
                             << " kpoint " << ik + 1 << std::endl;
            for (int i = 0; i < n_states_calc; i++)
            {
                const int i_state = i + i_state_low;
                const auto &eks_state = pds->mf_band.get_eigenvals()[isp](ik, i_state);
                const auto &exx_state = vexx_band[start_k+i];
                const auto &vxc_state = vxc_band[start_k+i];
                std::vector<cplxdb> sigc_state;
                for (int ifreq = 0; ifreq != nfreq; ++ifreq)
                {
                    const int idx = sigc_diag_index(isp, ik_collect, ifreq, i,
                                                    nk_collect, nfreq, n_states_calc);
                    sigc_state.emplace_back(sigc_diag[idx]);
                }
                double e_qp;
                cplxdb sigc;
                sigc_band_re[start_k+i] = std::numeric_limits<double>::quiet_NaN();
                sigc_band_im[start_k+i] = std::numeric_limits<double>::quiet_NaN();
                eqp_band[start_k+i] = std::numeric_limits<double>::quiet_NaN();
                const auto pade = make_gw_anacon_pade(
                    opts.n_params_anacon, gw_anacon_resample_n_params(opts),
                    imagfreqs, sigc_state, anacon_grid);
                int flag_qpe_solver = solve_qpe_with_option(
                    option_qpe_solver, pade, eks_state, efermi + hedin_shifts[start_k+i],
                    vxc_state, exx_state, e_qp, sigc, diff_init, thres_qpe,
                    n_iter_max, damp_fac, use_adaptive_damp,
                    use_legacy_nonadaptive_update);
                if (flag_qpe_solver != 0)
                {
                    ofs_myid << "Warning! QPE solver failed for spin " << isp + 1
                                     << " kpoint " << ik + 1 << " state " << i_state + 1
                                     << std::endl;
                    dump_analycont_source_data(ofs_myid, pade,
                                               "librpa_get_g0w0_sigc_band_k", isp, ik, i_state);
                    if (override_qpe_solver_nan)
                    {
                        ofs_myid
                            << "Using final unconverged QPE result because override_qpe_solver_nan is on"
                            << std::endl;
                    }
                }
                if (flag_qpe_solver == 0 || override_qpe_solver_nan)
                {
                    sigc_band_re[start_k+i] = sigc.real();
                    sigc_band_im[start_k+i] = sigc.imag();
                    eqp_band[start_k+i] = e_qp;
                }
            }
        }
    }
    profiler.stop("g0w0_solve_qpe");

    profiler.stop("api_get_g0w0_sigc_band_k");
}

void librpa_get_g0w0_sigc_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
                                 const int n_kpts_band_this, const int *iks_band_this,
                                 int i_state_low, int i_state_high, const double *vxc_band,
                                 const double *vexx_band, double *sigc_band_re,
                                 double *sigc_band_im)
{
    const int n_states_calc = std::max(0, i_state_high - i_state_low);
    std::vector<double> eqp_band(n_spins * n_kpts_band_this * n_states_calc);
    librpa_get_g0w0_qpe_band_k(h, p_opts, n_spins, n_kpts_band_this, iks_band_this,
                               i_state_low, i_state_high, vxc_band, vexx_band, sigc_band_re,
                               sigc_band_im, eqp_band.data());
}

void librpa_get_g0w0_spectral_function_band_k(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_band_this, const int *iks_band_this, int i_state_low,
    int i_state_high, const int n_omegas, const double *omegas,
    const double *vxc_band, const double *vexx_band, double *spectral_function_band,
    double *sigc_band)
{
    using namespace librpa_int;

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &opts = *p_opts;
    initialize_ds_global_ddla(*pds, opts);
    if (n_omegas < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("n_omegas must be non-negative");
    }
    if (!pds->is_band_data_set || pds->mf_band.get_n_spins() == 0)
        throw LIBRPA_RUNTIME_ERROR("Meanfield data for band calculation is not set");

    i_state_low = std::max(0, i_state_low);
    i_state_high = std::min(pds->mf_band.get_n_states(), i_state_high);
    const int n_spins_band = pds->mf_band.get_n_spins();
    if (n_spins != n_spins_band)
    {
        ofs_myid << "n_spins != pds->mf_band.get_n_spins(): " << n_spins << " != "
                         << n_spins_band << std::endl;
        throw LIBRPA_RUNTIME_ERROR("parsed n_spins is not consitent with the band input data");
    }
    if (n_omegas == 0 || n_kpts_band_this == 0 || i_state_high <= i_state_low) return;
    const int n_states_calc = i_state_high - i_state_low;
    validate_spectral_function_inputs(n_omegas, omegas, vxc_band, vexx_band,
                                      spectral_function_band);

    if (!pds->p_g0w0) librpa_build_g0w0_sigma(h, p_opts);

    profiler.start("api_get_g0w0_spectral_function_band_k");

    const auto iks_output =
        librpa_int::api::collect_requested_iks(pds->comm_h, n_kpts_band_this,
                                              iks_band_this, pds->mf_band.get_n_kpoints());

    profiler.start("g0w0_sigc_rotate_KS", "Correlation self-energy in K-S space");
    ensure_band_sigc_ks_blacs(*pds, opts, iks_output);
    profiler.stop("g0w0_sigc_rotate_KS");
    if (opts.output_gw_sigc_ks_mat_kf == LIBRPA_SWITCH_ON)
        write_sigc_matrices_KS_binary(*pds, opts.output_dir,
                                      "band_" + std::to_string(pds->band_data_id));

    std::vector<int> iks_collect;
    const auto freq_nodes = pds->tfg.get_freq_nodes();
    const bool publish_local_values =
        opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON || pds->blacs_h.myid == 0;
    const auto sigc_diag = collect_sigc_diag_to_callers(
        pds->p_g0w0->sigc_diag_is_ik_f_KS, freq_nodes, pds->comm_h, publish_local_values,
        n_spins, pds->mf_band.get_n_kpoints(), n_kpts_band_this, iks_band_this,
        i_state_low, n_states_calc, iks_collect);
    const auto imagfreqs = make_imagfreqs(freq_nodes);
    const auto anacon_grid = make_gw_anacon_grid(opts, pds->mf_band, pds->tfg, freq_nodes);

    profiler.start("g0w0_spectral_function", "Compute spectral function");
    evaluate_spectral_function_diagonal(
        pds->mf_band, opts.n_params_anacon, gw_anacon_resample_n_params(opts),
        sigc_diag, iks_collect, freq_nodes, imagfreqs, anacon_grid,
        n_spins, n_kpts_band_this, iks_band_this, i_state_low, n_states_calc,
        n_omegas, omegas, vxc_band, vexx_band, opts.sf_sigc_omega_shift,
        opts.sf_gf_omega_shift, spectral_function_band, sigc_band);
    profiler.stop("g0w0_spectral_function");

    profiler.stop("api_get_g0w0_spectral_function_band_k");
}
