// Public headers (prefixed by librpa)
#include "librpa_enums.h"
#include "librpa_options.h"

#include "../io/fs.h"
#include "../utils/error.h"

#include <string>
#include <cstring>

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla_connector.h>
#endif

namespace
{

void set_directory_option(char *dst, const char *src, const char *name, bool allow_empty)
{
    if (src == nullptr)
    {
        throw LIBRPA_RUNTIME_ERROR(std::string(name) + " is null");
    }

    if (allow_empty && src[0] == '\0')
    {
        dst[0] = '\0';
        return;
    }

    std::string dir = librpa_int::path_as_directory(src);
    if (dir.size() >= LIBRPA_MAX_STRLEN)
    {
        throw LIBRPA_RUNTIME_ERROR(
            std::string(name) + " is too long; maximum length is "
            + std::to_string(LIBRPA_MAX_STRLEN - 1)
            + " characters including the appended trailing slash");
    }

    std::memcpy(dst, dir.c_str(), dir.size() + 1);
}

}

// C APIs
void librpa_init_options(LibrpaOptions *opts)
{
    librpa_set_output_dir(opts, "librpa.d");
    librpa_set_restart_from_dir(opts, "");

    opts->parallel_routing = LIBRPA_ROUTING_AUTO;
    opts->vq_threshold = 0.0e0;
    opts->use_kpara_scf_eigvec = LIBRPA_SWITCH_OFF;

    opts->tfgrids_type = LIBRPA_TFGRID_UNSET;
    opts->nfreq = 16;
    opts->tfgrids_freq_min = 0.005;
    opts->tfgrids_freq_interval = 0.0;
    opts->tfgrids_freq_max = 1000.0;
    opts->tfgrids_time_min = 0.005;
    opts->tfgrids_time_interval = 0.0;

    opts->minimax_emin = -1.0;
    opts->minimax_emax = -1.0;
    opts->minimax_regulation = 0.0;

    opts->use_fullcoul_eps = LIBRPA_SWITCH_ON;
    opts->use_fullcoul_exx = LIBRPA_SWITCH_OFF;
    opts->use_fullcoul_wc = LIBRPA_SWITCH_OFF;
    opts->use_symmetry_exx = LIBRPA_SWITCH_OFF;
    opts->use_symmetry_gw = LIBRPA_SWITCH_OFF;
    opts->use_symmetry_rpa = LIBRPA_SWITCH_OFF;
    opts->output_abacus_gw_gf = LIBRPA_SWITCH_OFF;

    opts->n_bands_chi0 = -1;
    opts->n_bands_sigc = -1;
    opts->n_bands_exclude_chi0 = 0;
    opts->n_bands_exclude_sigc = 0;
    opts->option_bvk_remap = 1;

    opts->gf_threshold = 0.0e0;
    opts->libri_chi0_collect_s0_chunk = 0;
    opts->libri_chi0_collect_max_bytes = 0;
    opts->use_scalapack_ecrpa = LIBRPA_SWITCH_ON;

    opts->use_shrink_abfs = LIBRPA_SWITCH_OFF;
    opts->use_shrink_chi = LIBRPA_SWITCH_ON;

    opts->n_params_anacon = -1;
    opts->n_params_anacon_resample = -1;
    opts->anacon_tfgrids_type = LIBRPA_TFGRID_UNSET;
    opts->anacon_nfreq = -1;
    opts->option_qpe_solver = 0;
    opts->qpe_solver_thres = 1.0e-6;
    opts->qpe_solver_n_iter_max = 10000;
    opts->qpe_solver_damp_factor = 0.1;
    opts->use_qpe_adaptive_damp = LIBRPA_SWITCH_OFF;
    opts->use_qpe_legacy_update = LIBRPA_SWITCH_OFF;
    opts->override_qpe_solver_nan = LIBRPA_SWITCH_OFF;
    opts->use_hedin_shift = LIBRPA_SWITCH_OFF;
    opts->istate_ref_hedin_shift = -1;
    opts->sf_gf_omega_shift = 0.01;
    opts->sf_sigc_omega_shift = 0.01;
    opts->use_scalapack_gw_wc = LIBRPA_SWITCH_ON;
    opts->use_cholesky_gw_wc = LIBRPA_SWITCH_OFF;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    int deviceCount = 0;
    auto info = ddla::deviceGetDeviceCount(&deviceCount);
    if(info == ddla::deviceSuccess && deviceCount > 0)
        opts->use_gpu_replace_scalapack = LIBRPA_SWITCH_ON;
    else
        opts->use_gpu_replace_scalapack = LIBRPA_SWITCH_OFF;
#else
    opts->use_gpu_replace_scalapack = LIBRPA_SWITCH_OFF;
#endif
#ifdef LIBRPA_USE_ELPA
    opts->use_elpa_sqrt_coulomb = LIBRPA_SWITCH_ON;
#else
    opts->use_elpa_sqrt_coulomb = LIBRPA_SWITCH_OFF;
#endif
    opts->replace_w_head = LIBRPA_SWITCH_OFF;
    opts->option_dielect_func = 0;
    opts->use_2d_dielectric = LIBRPA_SWITCH_OFF;
    opts->rpa_headwing_body_start = 0;
    opts->rpa_headwing_mode[0] = '\0';
    std::strncpy(opts->rpa_headwing_mode, "qavg", LIBRPA_MAX_STRLEN);
    opts->rpa_headwing_mode[LIBRPA_MAX_STRLEN - 1] = '\0';
    opts->sqrt_coulomb_threshold = 0.0e0;
    opts->read_sigc_mat_rf = LIBRPA_SWITCH_OFF;

    opts->libri_chi0_threshold_C = 0.0e0;
    opts->libri_chi0_threshold_G = 0.0e0;
    opts->libri_exx_threshold_C = 0.0e0;
    opts->libri_exx_threshold_D = 0.0e0;
    opts->libri_exx_threshold_V = 0.0e0;
    opts->libri_g0w0_threshold_C = 0.0e0;
    opts->libri_g0w0_threshold_G = 0.0e0;
    opts->libri_g0w0_threshold_Wc = 0.0e0;

    opts->output_gw_sigc_ks_kf = LIBRPA_SWITCH_OFF;
    opts->output_gw_sigc_ks_mat_kf = LIBRPA_SWITCH_OFF;
    opts->output_exx_ks_mat_k = LIBRPA_SWITCH_OFF;
    opts->output_exx_mat_k = LIBRPA_SWITCH_OFF;
    opts->istate_output_mat_start = 0;
    opts->istate_output_mat_end = -1;
    opts->output_gw_sigc_mat_kf = LIBRPA_SWITCH_OFF;
    opts->output_gw_sigc_mat_rt = LIBRPA_SWITCH_OFF;
    opts->output_gw_sigc_mat_rf = LIBRPA_SWITCH_OFF;
    opts->output_wc_rf = LIBRPA_SWITCH_OFF;
    opts->output_wc_rf_atom_pair = LIBRPA_SWITCH_ON;
    opts->output_2d_finite_q_diagnostics = LIBRPA_SWITCH_OFF;
    opts->ifreq_output_wc_start = 0;
    opts->ifreq_output_wc_end = -1;
}

void librpa_set_output_dir(LibrpaOptions *opts, const char *output_dir)
{
    set_directory_option(opts->output_dir, output_dir, "output_dir", false);
}

void librpa_set_restart_from_dir(LibrpaOptions *opts, const char *restart_from_dir)
{
    set_directory_option(opts->restart_from_dir, restart_from_dir, "restart_from_dir", true);
}
