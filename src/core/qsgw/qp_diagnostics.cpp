#include "qp_diagnostics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <ostream>
#include <string>

#include "../../io/global_io.h"
#include "../../utils/error.h"
#include "../../utils/profiler.h"
#include "../analycont.h"
#include "../qpe_solver.h"

using librpa_int::global::ofs_myid;
using librpa_int::global::profiler;

namespace
{
// QSGW's entering-state diagnostic preserves the original G0W0 continuation,
// Hedin shift, solver options and solver-status reporting without moving that API.
struct GwAnaconGrid
{
    std::vector<librpa_int::cplxdb> imagfreqs;
    bool resample;
};

std::map<int, int> make_ik_pos_map(const std::vector<int> &iks)
{
    std::map<int, int> ik_pos;
    for (int i = 0; i != static_cast<int>(iks.size()); ++i)
    {
        ik_pos.emplace(iks[i], i);
    }
    return ik_pos;
}

int sigc_diag_index(const int isp, const int ik_collect, const int ifreq, const int i_state,
                    const int nk_collect, const int nfreq, const int n_states_calc)
{
    return (((isp * nk_collect + ik_collect) * nfreq + ifreq) * n_states_calc + i_state);
}

std::vector<librpa_int::cplxdb> make_imagfreqs(const std::vector<double> &freq_nodes)
{
    std::vector<librpa_int::cplxdb> imagfreqs;
    imagfreqs.reserve(freq_nodes.size());
    for (const auto &freq : freq_nodes)
    {
        imagfreqs.emplace_back(0.0, freq);
    }
    return imagfreqs;
}

GwAnaconGrid make_gw_anacon_grid(const LibrpaOptions &opts, const librpa_int::MeanField &mf,
                                 const librpa_int::TFGrids &source_tfg,
                                 const std::vector<double> &source_freq_nodes)
{
    if (opts.anacon_nfreq == 0)
    {
        throw LIBRPA_RUNTIME_ERROR("anacon_nfreq must not be zero");
    }

    const int source_nfreq = static_cast<int>(source_freq_nodes.size());
    const auto source_grid_type = source_tfg.get_grid_type();
    const bool use_new_nfreq = opts.anacon_nfreq > 0 && opts.anacon_nfreq != source_nfreq;
    const bool use_new_grid_type = opts.anacon_tfgrids_type != LIBRPA_TFGRID_UNSET &&
                                   opts.anacon_tfgrids_type != source_grid_type;
    if (!use_new_nfreq && !use_new_grid_type)
    {
        return {make_imagfreqs(source_freq_nodes), false};
    }

    const int nfreq = use_new_nfreq ? opts.anacon_nfreq : source_nfreq;

    const auto grid_type = use_new_grid_type ? opts.anacon_tfgrids_type : source_grid_type;
    if (grid_type == LIBRPA_TFGRID_UNSET)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "anacon_tfgrids_type is unset and source grid type is unavailable");
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
    return opts.n_params_anacon_resample < 0 ? opts.n_params_anacon : opts.n_params_anacon_resample;
}

std::unique_ptr<librpa_int::AnalyCont> make_gw_anacon(
    const LibrpaOptions &opts, const int n_params_anacon, const int n_params_anacon_resample,
    const std::vector<librpa_int::cplxdb> &source_imagfreqs,
    const std::vector<librpa_int::cplxdb> &source_data, const GwAnaconGrid &anacon_grid)
{
    if (n_params_anacon == 0)
    {
        throw LIBRPA_RUNTIME_ERROR("n_params_anacon must not be zero");
    }
    if (!anacon_grid.resample)
    {
        return librpa_int::make_analycont(opts.anacon_method, n_params_anacon, source_imagfreqs,
                                          source_data, opts.anacon_ridge_lambda,
                                          opts.anacon_ridge_den_weight, opts.anacon_aaa_tolerance);
    }
    if (n_params_anacon_resample == 0)
    {
        throw LIBRPA_RUNTIME_ERROR("n_params_anacon_resample must not be zero");
    }

    const auto source_pade = librpa_int::make_analycont(
        opts.anacon_method, n_params_anacon_resample, source_imagfreqs, source_data,
        opts.anacon_ridge_lambda, opts.anacon_ridge_den_weight, opts.anacon_aaa_tolerance);
    std::vector<librpa_int::cplxdb> data;
    data.reserve(anacon_grid.imagfreqs.size());
    for (const auto &freq : anacon_grid.imagfreqs)
    {
        data.emplace_back(source_pade->get(freq));
    }
    return librpa_int::make_analycont(opts.anacon_method, n_params_anacon, anacon_grid.imagfreqs,
                                      data, opts.anacon_ridge_lambda, opts.anacon_ridge_den_weight,
                                      opts.anacon_aaa_tolerance);
}

std::vector<librpa_int::cplxdb> extract_sigc_state(const std::vector<librpa_int::cplxdb> &sigc_diag,
                                                   const int isp, const int ik_collect,
                                                   const int i_state, const int nk_collect,
                                                   const int nfreq, const int n_states_calc)
{
    std::vector<librpa_int::cplxdb> sigc_state;
    sigc_state.reserve(nfreq);
    for (int ifreq = 0; ifreq != nfreq; ++ifreq)
    {
        const int idx =
            sigc_diag_index(isp, ik_collect, ifreq, i_state, nk_collect, nfreq, n_states_calc);
        sigc_state.emplace_back(sigc_diag[idx]);
    }
    return sigc_state;
}

std::vector<double> compute_hedin_shifts(
    const LibrpaOptions &opts, const librpa_int::MeanField &mf,
    const std::vector<librpa_int::cplxdb> &sigc_diag, const std::vector<int> &iks_collect,
    const std::vector<double> &freq_nodes, const std::vector<librpa_int::cplxdb> &imagfreqs,
    const GwAnaconGrid &anacon_grid, const int n_spins, const int n_kpts_this, const int *iks_this,
    const int i_state_low, const int n_states_calc, const double *vxc, const double *vexx,
    const char *api_name)
{
    std::vector<double> shifts(n_spins * n_kpts_this * n_states_calc, 0.0);
    if (opts.use_hedin_shift != LIBRPA_SWITCH_ON) return shifts;
    if (shifts.empty()) return shifts;
    if (iks_collect.empty())
        throw LIBRPA_RUNTIME_ERROR("failed to collect inputs for Hedin shift in " +
                                   std::string(api_name));

    const int i_state_ref_abs = opts.istate_ref_hedin_shift;
    if (i_state_ref_abs >= mf.get_n_states())
        throw LIBRPA_RUNTIME_ERROR("Hedin shift reference state index out of range: " +
                                   std::to_string(i_state_ref_abs));
    if (i_state_ref_abs >= 0 &&
        (i_state_ref_abs < i_state_low || i_state_ref_abs >= i_state_low + n_states_calc))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Hedin shift absolute reference state " + std::to_string(i_state_ref_abs) +
            " is outside selected absolute state range [" + std::to_string(i_state_low) + ", " +
            std::to_string(i_state_low + n_states_calc) + ") in " + api_name);
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
                throw LIBRPA_RUNTIME_ERROR("Hedin shift k-point " + std::to_string(ik) +
                                           " is not among requested k-points in " + api_name);
            }
            const int ik_collect = ik_pos_it->second;
            const int start_k = start_isp + ik_this * n_states_calc;

            const auto compute_shift = [&](const int i_ref, const int state_ref)
            {
                const int ref_idx = start_k + i_ref;
                const auto sigc_ref_state = extract_sigc_state(sigc_diag, isp, ik_collect, i_ref,
                                                               nk_collect, nfreq, n_states_calc);
                try
                {
                    const auto pade =
                        make_gw_anacon(opts, opts.n_params_anacon, gw_anacon_resample_n_params(opts),
                                       imagfreqs, sigc_ref_state, anacon_grid);
                    const auto sigc_ref = pade->get(librpa_int::cplxdb{
                        mf.get_eigenvals()[isp](ik, state_ref) - mf.get_efermi(), 0.0});
                    return sigc_ref.real() + vexx[ref_idx] - vxc[ref_idx];
                }
                catch (const librpa_int::AnalyContError &error)
                {
                    librpa_int::report_analycont_failure(api_name, isp, ik, state_ref,
                                                         imagfreqs, sigc_ref_state, error);
                    return std::numeric_limits<double>::quiet_NaN();
                }
            };

            if (i_state_ref_abs >= 0)
            {
                const double shift = compute_shift(i_state_ref_abs - i_state_low, i_state_ref_abs);
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
             << (i_state_ref_abs >= 0 ? " from one reference state per k-point" : " per state")
             << std::endl;
    return shifts;
}

int solve_qpe_with_option(const int option_qpe_solver, const librpa_int::AnalyCont &pade,
                          const double e_mf, const double e_fermi, const double vxc,
                          const double sigma_x, double &e_qp, librpa_int::cplxdb &sigc,
                          const double diff_init, const double thres, const int n_iter_max,
                          const double damp_fac, const bool use_adaptive_damp,
                          const bool use_legacy_nonadaptive_update)
{
    switch (option_qpe_solver)
    {
        case 0:
            return librpa_int::qpe_solver_pade_self_consistent(
                pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, diff_init, thres, n_iter_max,
                damp_fac, use_adaptive_damp, use_legacy_nonadaptive_update);
        case 1:
            return librpa_int::qpe_solver_pade_quasi_newton(pade, e_mf, e_fermi, vxc, sigma_x, e_qp,
                                                            sigc, diff_init, thres, n_iter_max,
                                                            damp_fac, use_adaptive_damp);
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
                "Invalid option_qpe_solver: " + std::to_string(option_qpe_solver) +
                ". Available values are 0 (damped residual-mixing), 1 (quasi-Newton), and 2 "
                "(perturbative).");
    }
}

}  // namespace

void librpa_int::qsgw::solve_qsgw_qpe_diagonal(
    const MeanField &mf, const TFGrids &tfg, const LibrpaOptions &opts,
    const std::vector<cplxdb> &sigc_diag, const std::vector<int> &iks_collect, int n_kpts_this,
    const int *iks_this, int i_state_low, int n_states_calc, const double *vxc, const double *vexx,
    double *sigc_re, double *sigc_im, double *eqp, const char *context, int *solver_info)
{
    const int n_spins = mf.get_n_spins();
    const auto freq_nodes = tfg.get_freq_nodes();
    const auto ik_pos = make_ik_pos_map(iks_collect);
    const int nk_collect = static_cast<int>(iks_collect.size());
    const int nfreq = static_cast<int>(freq_nodes.size());
    if (sigc_diag.size() != static_cast<std::size_t>(n_spins) * nk_collect * nfreq * n_states_calc)
        throw LIBRPA_RUNTIME_ERROR("Incomplete G0W0 diagonal self-energy input");
    profiler.start("g0w0_solve_qpe", "Solve quasi-particle equation");
    const auto efermi = mf.get_efermi();
    const auto imagfreqs = make_imagfreqs(freq_nodes);
    const auto anacon_grid = make_gw_anacon_grid(opts, mf, tfg, freq_nodes);
    const auto hedin_shifts = compute_hedin_shifts(
        opts, mf, sigc_diag, iks_collect, freq_nodes, imagfreqs, anacon_grid, n_spins, n_kpts_this,
        iks_this, i_state_low, n_states_calc, vxc, vexx, context);

    const auto thres_qpe = opts.qpe_solver_thres;
    const auto n_iter_max = opts.qpe_solver_n_iter_max;
    const auto damp_fac = opts.qpe_solver_damp_factor;
    const auto option_qpe_solver = opts.option_qpe_solver;
    const double diff_init = option_qpe_solver == 0 ? 1.0e-3 : 0.0;
    const bool use_adaptive_damp = opts.use_qpe_adaptive_damp == LIBRPA_SWITCH_ON;
    const bool use_legacy_nonadaptive_update = opts.use_qpe_legacy_update == LIBRPA_SWITCH_ON;
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
            ofs_myid << "Start QPE solver for spin " << isp + 1 << " kpoint " << ik + 1
                     << std::endl;
            for (int i = 0; i < n_states_calc; i++)
            {
                const int i_state = i + i_state_low;
                const auto &eks_state = mf.get_eigenvals()[isp](ik, i_state);
                const auto &exx_state = vexx[start_k + i];
                const auto &vxc_state = vxc[start_k + i];
                std::vector<cplxdb> sigc_state;
                for (int ifreq = 0; ifreq != nfreq; ++ifreq)
                {
                    const int idx = sigc_diag_index(isp, ik_collect, ifreq, i, nk_collect, nfreq,
                                                    n_states_calc);
                    sigc_state.emplace_back(sigc_diag[idx]);
                }
                double e_qp;
                cplxdb sigc;
                sigc_re[start_k + i] = std::numeric_limits<double>::quiet_NaN();
                sigc_im[start_k + i] = std::numeric_limits<double>::quiet_NaN();
                eqp[start_k + i] = std::numeric_limits<double>::quiet_NaN();
                if (!std::isfinite(hedin_shifts[start_k + i]))
                {
                    if (solver_info) solver_info[start_k + i] = 1;
                    continue;
                }
                try
                {
                    const auto pade =
                        make_gw_anacon(opts, opts.n_params_anacon, gw_anacon_resample_n_params(opts),
                                       imagfreqs, sigc_state, anacon_grid);
                    int flag_qpe_solver = solve_qpe_with_option(
                        option_qpe_solver, *pade, eks_state, efermi + hedin_shifts[start_k + i],
                        vxc_state, exx_state, e_qp, sigc, diff_init, thres_qpe, n_iter_max, damp_fac,
                        use_adaptive_damp, use_legacy_nonadaptive_update);
                    if (solver_info) solver_info[start_k + i] = flag_qpe_solver;
                    if (flag_qpe_solver != 0)
                    {
                        ofs_myid << "Warning! QPE solver failed for spin " << isp + 1 << " kpoint "
                                 << ik + 1 << " state " << i_state + 1 << std::endl;
                        dump_analycont_source_data(ofs_myid, *pade, context, isp, ik, i_state);
                        if (override_qpe_solver_nan)
                        {
                            ofs_myid << "Using final unconverged QPE result because "
                                        "override_qpe_solver_nan is on"
                                     << std::endl;
                        }
                    }
                    if (flag_qpe_solver == 0 || override_qpe_solver_nan)
                    {
                        sigc_re[start_k + i] = sigc.real();
                        sigc_im[start_k + i] = sigc.imag();
                        eqp[start_k + i] = e_qp;
                    }
                }
                catch (const librpa_int::AnalyContError &error)
                {
                    if (solver_info) solver_info[start_k + i] = 1;
                    librpa_int::report_analycont_failure(context, isp, ik, i_state,
                                                         imagfreqs, sigc_state, error);
                }
            }
        }
    }
    profiler.stop("g0w0_solve_qpe");
}
