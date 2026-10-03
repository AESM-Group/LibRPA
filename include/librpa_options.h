#pragma once
/**
 * @file librpa_options.h
 * @brief Runtime options for LibRPA calculations.
 *
 * This file defines the LibrpaOptions structure that controls runtime behavior
 * for RPA, exact exchange, and G0W0 calculations.
 */

#include "librpa_enums.h"

// C APIs
#ifdef __cplusplus
extern "C" {
#endif

/** Maximum length for string parameters (e.g., output directory path). */
#define LIBRPA_MAX_STRLEN 200

// NOTE: in case the data layout of LibrpaOptions is changed,
// its Fortran binding should be adapted accordingly.

/**
 * @brief Runtime options structure for LibRPA calculations.
 *
 * This structure contains all configurable parameters for controlling
 * RPA, exact exchange (EXX), and G0W0 calculations. Initialize with
 * librpa_init_options() and modify as needed before passing to
 * computation functions.
 *
 * @note The data layout must match the Fortran binding.
 *       Use utilities/check_librpa_options.py for inter-binding check.
 */
typedef struct
{
    /* ============================================================================= */
    /* Common runtime control */

    //! Output directory for results.
    //! @par Details
    //! Set with `librpa_set_output_dir()` or `librpa::Options::set_output_dir()`
    //! so directory normalization is applied.
    //! @par Default
    //! librpa.d/
    char output_dir[LIBRPA_MAX_STRLEN];

    //! Directory to read restart checkpoint files from. Empty uses output_dir.
    //! @par Details
    //! Set with `librpa_set_restart_from_dir()` or `librpa::Options::set_restart_from_dir()`
    //! so directory normalization is applied.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    char restart_from_dir[LIBRPA_MAX_STRLEN];

    //! Scheme of parallelization.
    /*!
     * Available driver values: `auto`, `atompair`, `rtau`, `libri`.
     *
     * @par Default
     * LIBRPA_ROUTING_AUTO
     */
    LibrpaParallelRouting parallel_routing;

    //! Threshold for real-space Coulomb matrices.
    //! @par Default
    //! 0.0
    double vq_threshold;

    //! Flag to specify parallel distribution of eigenvectors of SCF starting point.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_kpara_scf_eigvec;

    //! Type of time/frequency grids.
    /*!
     * Available driver values: `GL`, `GC-I`, `GL-II`, `minimax`, `evenspaced`, `evenspaced_tf`.
     * The driver maps the unset API value to `minimax`.
     *
     * @par Default
     * LIBRPA_TFGRID_UNSET
     */
    LibrpaTimeFreqGrid tfgrids_type;

    //! Number of time/frequency points.
    //! @par Default
    //! 16
    int nfreq;

    /* ============================================================================= */
    /* Parameters for time and freqeuncy grids.
     * Different grid types will use none/part/all of the parameters. */
    //! Minimum frequency for grid generation, in Hartree.
    //! @par Default
    //! 0.005
    double tfgrids_freq_min;

    //! Frequency interval for even-spaced grids, in Hartree.
    //! @par Default
    //! 0.0
    double tfgrids_freq_interval;

    //! Maximum frequency for grid generation, in Hartree.
    //! @par Default
    //! 1000.0
    double tfgrids_freq_max;

    //! Minimum time for grid generation, in Hartree^-1.
    //! @par Default
    //! 0.005
    double tfgrids_time_min;

    //! Time interval for even-spaced grids, in Hartree^-1.
    //! @par Default
    //! 0.0
    double tfgrids_time_interval;

    //! Minimal transition energy when generating minimax grids.
    /*!
     * When set negative (the default), the minimal energy is decided by the mean-field input
     * and set to the energy gap.
     * For now, it is suggested to manually set this option for gapless systems.
     *
     * @par Default
     * -1.0
     * @par Status
     * Experimental
     */
    double minimax_emin;

    //! Maximal transition energy when generating minimax grids.
    /*!
     * This option is introduced for test of supercell to fix the minimax frequency points
     *
     * @par Default
     * -1.0
     * @par Status
     * Experimental
     */
    double minimax_emax;

    //! Regulation for minimax grids generation.
    //! @par Default
    //! 0.0
    //! @par Status
    //! Experimental
    double minimax_regulation;

    /* ============================================================================= */
    /* Coulomb matrices usage */

    //! Switch of using full Coulomb interaction in exact-exchange operator.
    //! @par Default
    //! false
    LibrpaSwitch use_fullcoul_exx;

    //! Switch of using full Coulomb interaction in \f$\varepsilon = 1 - v \chi^0\f$
    //! @par Default
    //! true
    LibrpaSwitch use_fullcoul_eps;

    //! Switch of using full Coulomb interaction in \f$W^c = (\varepsilon^{-1} - 1) v\f$.
    //! @par Default
    //! false
    LibrpaSwitch use_fullcoul_wc;

    //! Switch of using the symmetry context in exact-exchange paths.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_symmetry_exx;

    //! Switch of using the symmetry context in GW paths.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_symmetry_gw;

    //! Switch of using the symmetry context in RPA/chi0 paths.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_symmetry_rpa;

    //! Switch of outputting ABACUS-compatible GW Green's-function data.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_abacus_gw_gf;

    /* ============================================================================= */
    /* Sum of states */

    //! Maximal number of bands for computing response function.
    //! @par Default
    //! -1 (< 0 for all bands)
    //! @par Status
    //! Experimental
    int n_bands_chi0;

    //! Maximal number of bands for computing correlation self-energy.
    //! @par Default
    //! -1 (< 0 for all bands)
    //! @par Status
    //! Experimental
    int n_bands_sigc;

    //! Number of lowest bands excluded from both Green's functions forming the response function.
    //!
    //! Applies independently at each input k-point and spin, for both signs of imaginary time.
    //! Valid range: 0 through the number of input bands. The input band ordering is retained.
    //! Does not modify occupations, EXX, output state ranges, or analytic/external head corrections.
    //! @par Default
    //! 0
    int n_bands_exclude_chi0;

    //! Number of lowest bands excluded from the Green's function multiplying Wc in the correlation self-energy.
    //!
    //! Applies independently at each input k-point and spin, for both signs of imaginary time.
    //! Valid range: 0 through the number of input bands. The input band ordering is retained.
    //! Does not modify occupations, EXX, output state ranges, or analytic/external head corrections.
    //! @par Default
    //! 0
    int n_bands_exclude_sigc;

    //! BvK remapping convention for band interpolation.
    /*!
     * Available values:
     * - 0: map each atom-pair R to one nearest BvK image.
     * - 1: Wigner-Seitz remapping; split over all nearest BvK images at equal distance.
     *
     * @par Default
     * 1
     */
    int option_bvk_remap;

    /* ============================================================================= */
    /* RPA specific */

    //! Threshold for real-space Green's function in response function calculation.
    //! @par Default
    //! 0.0
    double gf_threshold;

    //! Number of first-index atoms per LibRI chi0 collection chunk; 0 keeps one-shot collection.
    //! @par Default
    //! 0
    //! @par Status
    //! Experimental
    int libri_chi0_collect_s0_chunk;

    //! Maximum estimated local chi0 tensor bytes per LibRI collection chunk; 0 keeps one-shot collection.
    //! @par Default
    //! 0
    //! @par Status
    //! Experimental
    long long libri_chi0_collect_max_bytes;

    //! Flag to control whether to use ScaLAPACK to compute RPA correlation energy.
    //! @par Default
    //! true
    LibrpaSwitch use_scalapack_ecrpa;

    /* ============================================================================= */
    /* ABF compression */

    //! Flag to use a compressed auxiliary basis.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_shrink_abfs;

    //! Flag to compress response function using shrinked basis.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    LibrpaSwitch use_shrink_chi;

    /* ============================================================================= */
    /* GW specific */

    //! Number of parameters for analytic continuation.
    //! @par Default
    //! -1 (uses all `nfreq` data)
    int n_params_anacon;

    //! Number of parameters for the source-grid Pade used to resample GW analytic continuation.
    //! Negative values use `n_params_anacon`.
    //! This parameter only works when resampling imaginary-frequency grids is enabled via
    //! a different frequency grids type (`anacon_tfgrids_type`) or number of grids points (`anacon_nfreq`)
    //! from the direct evaluation of \f$\Sigma_c(i\omega)\f$.
    //! @par Default
    //! -1
    //! @par Status
    //! Experimental
    int n_params_anacon_resample;

    //! Type of optional frequency grid used for GW analytic continuation.
    /*!
     * When unset and `anacon_nfreq` does not request a different grid size,
     * analytic continuation uses the original GW self-energy grid.
     *
     * @par Default
     * LIBRPA_TFGRID_UNSET
     * @par Status
     * Experimental
     */
    LibrpaTimeFreqGrid anacon_tfgrids_type;

    //! Number of points in the optional GW analytic-continuation frequency grid.
    /*!
     * Negative values use the source grid `nfreq` when the analytic-continuation
     * grid is built. Zero is invalid. Positive values select that many points;
     * if this equals the source `nfreq` and no new grid type is selected, the
     * original grid is reused.
     *
     * @par Default
     * -1
     * @par Status
     * Experimental
     */
    int anacon_nfreq;

    //! Quasi-particle equation solver option.
    /*!
     * Available values:
     * - 0: damped residual-mixing self-consistent solver
     * - 1: quasi-Newton self-consistent solver using the Pade derivative
     * - 2: perturbative solver linearized at the mean-field energy
     *
     * @par Default
     * 0
     */
    int option_qpe_solver;

    //! Convergence threshold for the quasi-particle equation solver, in Hartree.
    //! @par Default
    //! 1.0e-6
    double qpe_solver_thres;

    //! Maximum number of iterations for the quasi-particle equation solver; must be positive.
    //! @par Default
    //! 10000
    int qpe_solver_n_iter_max;

    //! Damping factor for quasi-particle equation solver updates.
    //! Used as the initial and maximum factor when adaptive damping is enabled.
    //! @par Default
    //! 0.1
    double qpe_solver_damp_factor;

    //! If enabled, adapt the QPE damping factor during the solve.
    //! @par Default
    //! false
    LibrpaSwitch use_qpe_adaptive_damp;

    //! Test-only switch to recover the legacy non-adaptive QPE update.
    //! Ignored when adaptive damping is enabled.
    //! @par Default
    //! false
    //! @par Status
    //! Diagnostic
    LibrpaSwitch use_qpe_legacy_update;

    //! If enabled, keep the final unconverged QPE iterate instead of outputting NaN.
    //! @par Default
    //! false
    //! @par Status
    //! Diagnostic
    LibrpaSwitch override_qpe_solver_nan;

    //! Compute and apply Hedin shifts in QP solves.
    //! @par Details
    //! \f[
    //! \Delta_{rks} = \mathrm{Re}\,\Sigma_c(\epsilon_{rks} - E_F)
    //!        + \Sigma_x(rks) - v_{xc}(rks)
    //! \f]
    //! and evaluate the correlation self-energy in QP solves at
    //! \f[
    //! \Sigma_c(E - E_F - \Delta_{rks}).
    //! \f]
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_hedin_shift;

    //! Absolute zero-based reference state index for Hedin shifts.
    //! @par Details
    //! This index is not relative to i_state_low. A negative value applies a per-state shift; a nonnegative
    //! value uses that state at each k-point and spin channel for all states in
    //! the same channel.
    //! @par Default
    //! -1
    //! @par Status
    //! Experimental
    int istate_ref_hedin_shift;

    //! Broadening/shift used for Green's function in spectral-function output, in Hartree.
    //! @par Default
    //! 0.01
    //! @par Status
    //! Experimental
    double sf_gf_omega_shift;

    //! Broadening/shift used for correlation self-energy in spectral-function output, in Hartree.
    //! @par Default
    //! 0.01
    //! @par Status
    //! Experimental
    double sf_sigc_omega_shift;

    //! Flag of using ScaLAPACK for computing Wc from chi0.
    //! @par Default
    //! true
    LibrpaSwitch use_scalapack_gw_wc;

    //! Flag of using cholesky factorization for computing Wc from chi0.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_cholesky_gw_wc;

    //! Flag of using GPU for computing Wc from chi0.
    //! @par Default
    //! Build-dependent: true when a CUDA/HIP build detects a device, false otherwise.
    //! @par Status
    //! Experimental
    LibrpaSwitch use_gpu_replace_scalapack;

    //! Flag of using ELPA for sqrt Coulomb matrix.
    //! @par Default
    //! Build-dependent: true with `LIBRPA_USE_ELPA`, false otherwise.
    //! @par Status
    //! Experimental
    LibrpaSwitch use_elpa_sqrt_coulomb;

    //! Flag of replacing head of screened interaction by macroscopic dielectric function.
    //! @par Default
    //! false
    LibrpaSwitch replace_w_head;

    //! Option of computing dielectric function on imaginary axis.
    /*!
     * Available values:
     * - 0: use data directly read from input
     * - 1: cubic-spline interpolation from input data
     * - 2: dielectric model fitting from input data
     * - 3: analytic head and wing correction
     * - 4: analytic head correction
     *
     * @par Default
     * 0
     */
    int option_dielect_func;

    //! Switch of using 2D dielectric function.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch use_2d_dielectric;

    //! First regular Coulomb-eigenbasis channel used by RPA head/wing correction.
    //! Zero uses channel 1 in the current analytic 3D/2D head/wing path.
    //! Minus one is reserved for Sternheimer finite-part Coulomb input and keeps
    //! every retained positive Coulomb eigenchannel as a regular body channel.
    //! @par Default
    //! 0
    //! @par Status
    //! Experimental
    int rpa_headwing_body_start;

    //! Flag of reading NAO correlation self-energy matrix in real-space/frequency form.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch read_sigc_mat_rf;

    //! RPA Gamma correction mode: "qavg" keeps the current head/wing q-average,
    //! "head_only" replaces only the analytic head before the ordinary trace-log.
    char rpa_headwing_mode[LIBRPA_MAX_STRLEN];

    //! Threshold of eigenvalues to perform square root of Coulomb matrices.
    //! @par Default
    //! 0.0
    double sqrt_coulomb_threshold;

    /* ============================================================================= */
    /* LibRI related */

    //! Threshold of real-space LRI triple coefficients to compute response function using LibRI.
    //! @par Default
    //! 0.0
    double libri_chi0_threshold_C;

    //! Threshold of real-space Green's function to compute response function using LibRI.
    //! @par Default
    //! 0.0
    double libri_chi0_threshold_G;

    //! Threshold of real-space LRI triple coefficients to compute exact exchange using LibRI.
    //! @par Default
    //! 0.0
    double libri_exx_threshold_C;

    //! Threshold of real-space density matrices to compute exact exchange using LibRI.
    //! @par Default
    //! 0.0
    double libri_exx_threshold_D;

    //! Threshold of real-space Coulomb matrices to compute exact exchange using LibRI.
    //! @par Default
    //! 0.0
    double libri_exx_threshold_V;

    //! Threshold of real-space LRI triple coefficients to compute GW correlation self-energy using LibRI.
    //! @par Default
    //! 0.0
    double libri_g0w0_threshold_C;

    //! Threshold of real-space Green's function to compute GW correlation self-energy using LibRI.
    //! @par Default
    //! 0.0
    double libri_g0w0_threshold_G;

    //! Threshold of correlation screened Coulomb matrix to compute GW correlation self-energy using LibRI.
    //! @par Default
    //! 0.0
    double libri_g0w0_threshold_Wc;

    /* Output controls */
    //! Output KS-diagonal correlation self-energy in k-space, imaginary frequency domain.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_gw_sigc_ks_kf;

    //! Output correlation self-energy matrix in KS basis (k-space, imaginary frequency domain).
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_gw_sigc_ks_mat_kf;

    //! Output exact-exchange matrix in KS basis and k-space.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_exx_ks_mat_k;

    //! Output exact-exchange matrix in NAO basis and k-space as dense binary files.
    //! Files in output_dir are named ExxK_kgrid_ispin_S_ik_K.bin or
    //! ExxK_band_B_ispin_S_ik_K.bin, with zero-based indices. Spinor blocks
    //! include _spinor_BRA_KET before _ik. Values are in Hartree and include
    //! the exchange minus sign. All NAOs are exported, independently of the
    //! KS-state output range, when the k-space EXX potential is constructed.
    //! Binary layout: native-endian int32 dimension and int32 scalar byte size (8),
    //! then row-major real/imaginary float64 pairs, without threshold filtering.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_exx_mat_k;

    //! First zero-based KS state included in both dimensions when exporting the
    //! KS-basis exact-exchange and correlation self-energy matrices. Matrix
    //! construction is unaffected.
    //! @par Default
    //! 0
    //! @par Status
    //! Experimental
    int istate_output_mat_start;

    //! Half-open KS-state end index when exporting KS-basis matrices; negative
    //! means all remaining states.
    //! @par Default
    //! -1
    //! @par Status
    //! Experimental
    int istate_output_mat_end;

    //! Output correlation self-energy matrix in NAO basis (k-space, imaginary frequency domain).
    //! Dense binary files in output_dir use the same layout as output_exx_mat_k:
    //! SigcKF_kgrid_ispin_S_ik_K_ifreq_F.bin or
    //! SigcKF_band_B_ispin_S_ik_K_ifreq_F.bin, with zero-based indices.
    //! Spinor blocks include _spinor_BRA_KET before _ik. Values are in Hartree;
    //! all NAOs are exported without threshold filtering.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_gw_sigc_mat_kf;

    //! Output correlation self-energy matrix in NAO basis (real space, imaginary time domain).
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_gw_sigc_mat_rt;

    //! Output correlation self-energy matrix in NAO basis (real space, imaginary frequency domain).
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_gw_sigc_mat_rf;

    //! Output Wc matrix in ABFs (real space, imaginary frequency domain).
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    LibrpaSwitch output_wc_rf;

    //! Output Wc(R, iw) as atom-pair blocks instead of full matrices.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    LibrpaSwitch output_wc_rf_atom_pair;

    //! First zero-based Wc frequency index to output.
    //! @par Default
    //! 0
    //! @par Status
    //! Experimental
    int ifreq_output_wc_start;

    //! Half-open Wc frequency output end index; negative means all remaining frequencies.
    //! @par Default
    //! -1
    //! @par Status
    //! Experimental
    int ifreq_output_wc_end;

} LibrpaOptions;

/**
 * @brief Initialize runtime options to default values.
 *
 * Sets all options to their default settings. Must be called before
 * modifying options and passing to computation functions.
 *
 * @param[out] opts Pointer to the options structure to initialize.
 */
void librpa_init_options(LibrpaOptions *opts);

/**
 * @brief Set the output directory for LibRPA results.
 *
 * @param[in,out] opts      Pointer to the options structure.
 * @param[in]     output_dir Path to directory where output files will be written.
 */
void librpa_set_output_dir(LibrpaOptions *opts, const char *output_dir);

/**
 * @brief Set the directory to read restart checkpoint files from.
 *
 * @param[in,out] opts           Pointer to the options structure.
 * @param[in]     restart_from_dir Path to directory containing restart files. Empty uses output_dir.
 */
void librpa_set_restart_from_dir(LibrpaOptions *opts, const char *restart_from_dir);

#ifdef __cplusplus
}
#endif
