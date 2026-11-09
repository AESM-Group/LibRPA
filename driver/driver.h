#pragma once

#include "librpa.hpp"
#include "reader/reader_context.h"
#include "librpa_enums.h"

#include <string>
#include <vector>

#include "../src/math/vector3_order.h"

namespace driver
{

// Runtime options specific to the driver
struct DriverParams
{
    static constexpr int default_i_state_high = 999999;

    //! Task type.
    //! @par Default
    //! required (`rpa`, `g0w0`, `exx`, `exx_band`, `hartree`)
    std::string task;

    //! Physical constants source.
    //! @par Default
    //! `internal` (`internal`, `aims`)
    std::string constants_choice;

    //! Preset for producer-specific input filenames and head/wing velocity readers.
    //! `aims` is an alias for `fhi-aims`.
    //! `abacus-legacy` is for input before [ABACUS PR #7849](https://github.com/deepmodeling/abacus-develop/pull/7849)
    //! (merge commit `42f8ad905d56` on `develop`).
    //! @par Default
    //! `fhi-aims` (`fhi-aims`, `aims`, `abacus`, `abacus-legacy`)
    std::string input_preset;

    //! Input directory to find and read the AO dataset.
    //! @par Default
    //! `./`
    std::string input_dir;

    //! Verbosity level for driver output.
    //!
    //! It is parsed to the global API function `set_output_level`.
    //!
    //! @par Default
    //! `info` (`silent`, `critical`, `info`, `warn`, `debug`)
    LibrpaVerbose output_level;

    //! Read wavefunctions in spinor format.
    //! @par Default
    //! false
    bool use_spinor_wfc;

    //! Prefix of localized RI coefficient files.
    //! @par Default
    //! `Cs_data`
    std::string prefix_lri_coeff;

    //! Prefix of compressed-auxiliary-basis RI coefficient files.
    //! @par Default
    //! `Cs_shrinked_data`
    //! @par Status
    //! Experimental
    std::string prefix_lri_coeff_shrink;

    //! Dedicated unwrapped, full-precision LRI coefficient prefix for Hartree
    //! feedback. Uses existing binary v1 blocks with true physical translations
    //! and the auxiliary basis selected by use_shrink_abfs; no normal Cs fallback.
    //! @par Default
    //! empty (required when qsgw_update_hartree=true)
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string prefix_lri_coeff_hartree;

    //! Prefix of shrink auxiliary-basis transform files.
    //! @par Default
    //! `shrink_sinvS_`
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string prefix_shrink_sinvS;

    //! Prefix of bare Coulomb matrix files.
    //! @par Default
    //! `coulomb_mat`
    std::string prefix_coul_full;

    //! Bare three-dimensional Coulomb prefix for task=hartree or QSGW Hartree feedback, with reciprocal
    //! G=0 set to zero. Must be supplied explicitly; no EXX metric fallback.
    //! @par Default
    //! empty (required for task=hartree or qsgw_update_hartree=true)
    //! @par Status
    //! Experimental
    std::string prefix_coul_hartree;

    //! Optional complete complex ABACUS Hartree H(R) CSR reference.
    //! Relative paths are resolved from input_dir. No gauge shift is fitted.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string hartree_fn_reference;

    //! Reference CSR energy conversion to Hartree (0.5 for Rydberg input).
    //! @par Default
    //! 0.5
    //! @par Status
    //! Experimental
    double hartree_reference_scale;

    //! Positive maximum complex AO-entry comparison tolerance, in Hartree.
    //! A negative value only reports differences; a positive gate requires a reference.
    //! @par Default
    //! -1 (report only)
    //! @par Status
    //! Experimental
    double hartree_check_tolerance;

    //! Prefix of Sternheimer chi0 v1 response files.
    //! @par Default
    //! `v1_sternheimer_chi0_iq_`
    //! @par Status
    //! Experimental
    std::string prefix_sternheimer_chi0;

    //! Sternheimer q-point manifest filename with rows `iq qx qy qz qweight`.
    //! An empty value preserves the single-q `sternheimer_iq` mode.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string fn_sternheimer_qpoints;

    //! Sternheimer partial-response manifest with rows
    //! `iq ik_full ifreq response_file`.
    //! An empty value preserves aggregate Sternheimer response input.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string fn_sternheimer_partial_manifest;

    //! Explicit fixed-q inverse routes provided by the producer for reconstruction
    //! under the symmetry subgroup preserved by the discrete Hamiltonian.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string fn_sternheimer_symmetry_routes;

    //! Explicit inverse routes from every full-q member to the discrete
    //! q-star representative provided by the producer.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string fn_sternheimer_qstar_routes;

    //! Optional prefix for reconstructed full-q Sternheimer chi0 v1 files.
    //! An empty value disables the diagnostic output.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string prefix_sternheimer_reconstructed;

    //! Optional prefix for fixed-q k-resolved Sternheimer response matrices.
    //! An empty value disables the diagnostic output.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string prefix_sternheimer_kresolved;

    //! Optional prefix for fixed-q Sternheimer symmetry route diagnostics.
    //! An empty value disables the diagnostic output.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    std::string prefix_sternheimer_symmetry_diagnostic;

    //! One-based q index for Sternheimer-RPA postprocessing.
    //! @par Default
    //! 1
    //! @par Status
    //! Experimental
    int sternheimer_iq;

    //! q-point weight for Sternheimer-RPA postprocessing.
    //! @par Default
    //! 1.0
    //! @par Status
    //! Experimental
    double sternheimer_qweight;

    //! Include the Gamma-point contribution in the reported RPA q sum.
    //! @par Default
    //! true
    bool use_rpa_gamma;

    //! Reconstruct only fixed-q aggregate and k-resolved response matrices.
    //! This diagnostic mode skips q-star coverage, Coulomb input, and RPA energy.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    bool sternheimer_matrix_only;

    //! Prefix of truncated Coulomb matrix files.
    //! @par Default
    //! `coulomb_cut`
    std::string prefix_coul_cut;

    //! Prefix of SCF Kohn-Sham eigenvector files.
    //! @par Default
    //! `KS_eigenvector`
    std::string prefix_eigvecs_scf;

    //! Structure-data filename.
    //! @par Default
    //! `stru_out`
    std::string fn_stru;

    //! Brillouin-zone sampling filename.
    //! @par Default
    //! `bz_sampling_out`
    std::string fn_bz_sampling;

    //! Deprecated combined basis-set fallback filename.
    //! @par Default
    //! `basis_out`
    //! @deprecated Use `fn_basis_wfc` and `fn_basis_aux` instead.
    std::string fn_basis;

    //! Wave-function basis metadata filename.
    //! @par Default
    //! `basis_wfc_out`
    std::string fn_basis_wfc;

    //! Auxiliary-basis metadata filename.
    //! @par Default
    //! `basis_aux_out`
    std::string fn_basis_aux;

    //! Compressed auxiliary-basis metadata filename used with `use_shrink_abfs = true`.
    //! @par Default
    //! `basis_aux_shrink_out`
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string fn_basis_aux_shrink;

    //! SCF eigenvalue and occupation filename.
    //! @par Default
    //! `band_out`
    std::string fn_eigocc_scf;

    //! Dielectric-function filename for dielectric-head correction.
    //! @par Default
    //! `dielecfunc_out`
    std::string fn_dielfunc;

    //! SCF exchange-correlation potential filename.
    //! @par Default
    //! `vxc_out`
    std::string fn_vxc_scf;

    //! Prefix of velocity/momentum-matrix files.
    //! @par Default
    //! `mommat_ks_kpt_`
    std::string prefix_velocity;

    //! G0W0 analytic head/wing velocity source: file (existing reader) or native.
    //! Native uses the original DFT H(R)/S(R)/r(R) and the loaded SCF E/C frame.
    //! @par Default
    //! file
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string g0w0_headwing_velocity;

    //! Original DFT H(R) CSR for spin 0, relative to input_dir or absolute.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string g0w0_fn_hr;

    //! Original DFT H(R) CSR for spin 1 in a collinear two-spin calculation.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string g0w0_fn_hr_spin1;

    //! Dimensionless S(R) CSR with its original full real-space support.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string g0w0_fn_sr;

    //! Three-component r(R) CSR with its original full real-space support.
    //! @par Default
    //! empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string g0w0_fn_rr;

    //! H(R) input energy unit: Ha, Ry, or eV, converted once into Hartree.
    //! @par Default
    //! Ry
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string g0w0_hr_unit;

    //! r(R) input length unit: Bohr or Angstrom, converted once into Bohr.
    //! @par Default
    //! Bohr
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string g0w0_rr_unit;

    //! Band-path k-point metadata filename.
    //! For QSGW, presence under input_dir enables the existing band channel
    //! and requires matching band E/C and full Vxc; absence runs only the
    //! SCF grid. This input-driven behavior uses task=qsgw.
    //! Keep band-channel presence and inputs unchanged across a restart.
    //! @par Default
    //! `band_kpath_info`
    std::string fn_band_kpath_info;

    //! Coulomb reader-version selector.
    /*!
     * Available values:
     * - -1: auto-detect from the first file matching the prefix.
     * - 0: legacy text or binary rectangular matrix blocks.
     * - 1: binary v1 files with atom-pair blocks.
     *
     * @par Default
     * -1
     * @par Status
     * Experimental
     */
    int version_coul_reader;

    //! Local RI coefficient reader-version selector.
    /*!
     * Available values:
     * - -1: auto-detect from the first file matching the prefix.
     * - 0: legacy text or legacy binary files.
     * - 1: binary v1 files with a block table and payload offsets.
     *
     * @par Default
     * -1
     * @par Status
     * Experimental
     */
    int version_lri_reader;

    //! Screening threshold when reading RI coefficient data.
    //! @par Default
    //! 1e-6
    double cs_threshold;

    //! Output quasiparticle energies for external BSE workflows.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    bool output_energy_qp;

    //! First state index for printed QP energies, inclusive.
    /*!
     * Negative values enable automatic selection from a Fermi-level/gap energy window.
     * The current lower bound is \f$E_F - \frac{1}{2}E_g - 0.5 \text{Ha}\f$.
     *
     * @par Default
     * 0
     */
    int i_state_low;

    //! Last state index for printed QP energies, exclusive.
    /*!
     * Negative values enable automatic selection from a Fermi-level/gap energy window.
     * The current upper bound is \f$E_F + \frac{1}{2}E_g + 0.5 \text{Ha}\f$.
     * Values larger than the number of states are truncated.
     *
     * @par Default
     * 999999
     */
    int i_state_high;

    //! Output GW energies for HamGNN machine-learning workflows.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    bool output_hamgnn;

    //! Use PyATB mean-field data for dielectric head/wing calculations.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    bool use_pyatb;

    //! Output GW spectral-function data.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    bool output_gw_spec_func;

    //! Starting frequency for spectral-function output.
    //! @par Default
    //! 0.0
    //! @par Status
    //! Experimental
    double sf_omega_start;

    //! Ending frequency for spectral-function output.
    //! @par Default
    //! 1.0
    //! @par Status
    //! Experimental
    double sf_omega_end;

    //! Frequency step for spectral-function output.
    //! @par Default
    //! 0.1
    //! @par Status
    //! Experimental
    double sf_omega_step;

    //! First state index for spectral-function output, inclusive.
    //! Negative values inherit i_state_low after QP state-range normalization.
    //! @par Default
    //! -1
    //! @par Status
    //! Experimental
    int sf_state_start;

    //! Last state index for spectral-function output, exclusive.
    //! Negative values inherit i_state_high after QP state-range normalization.
    //! @par Default
    //! -1
    //! @par Status
    //! Experimental
    int sf_state_end;

    //! Full SCF Vxc matrix prefix, relative to `input_dir` unless absolute.
    //! Used by QSGW; defaults to `xc_matr` (aims) or `vxc` (ABACUS).
    //! @par Default
    //! Empty (producer default)
    //! @par Status
    //! Experimental
    std::string prefix_vxc_scf;

    //! Full band Vxc matrix prefix, relative to `input_dir` unless absolute.
    //! Used by QSGW; defaults to `band_vxc_mat` (aims) or `band_vxc` (ABACUS).
    //! @par Default
    //! Empty (producer default)
    //! @par Status
    //! Experimental
    std::string prefix_vxc_band;

    //! Basis of the full Vxc matrices: `state` or `nao` (ABACUS AO input).
    //! Applies to both SCF and band matrices. The ABACUS `_nao` filename suffix
    //! does not determine the matrix basis: standard out_mat_xc exports states.
    //! @par Default
    //! `state`
    //! @par Status
    //! Experimental
    std::string qsgw_vxc_basis;

    //! Fixed-basis Hamiltonian mixing (`none` or `linear`).
    //! After reading SCF input, `linear` is allowed only for one loaded k point
    //! on a one-point full BvK mesh. Multi-k meshes automatically use `none`.
    //! Band-path points do not affect this selection. A Gamma-only solid also
    //! enters the single-k branch; select `none` for a direct update there.
    //! @par Default
    //! `none`
    //! @par Status
    //! Experimental
    std::string qsgw_mixer;

    //! Linear Hamiltonian mixing fraction for the single-k SCF branch.
    //! @par Default
    //! 0.2
    //! @par Status
    //! Experimental
    double qsgw_mixing_beta;

    //! Earliest global QSGW iteration at which convergence may terminate the run.
    //! @par Default
    //! 1
    //! @par Status
    //! Experimental
    int qsgw_min_iter;

    //! Maximum global QSGW iteration, including restored updates.
    //! @par Default
    //! 10
    //! @par Status
    //! Experimental
    int qsgw_max_iter;

    //! Number of unoccupied states retained by the QSGW H_QSGW cut.
    //! Applies to the SCF grid and optional band path when cut mode 1 is selected.
    //! @par Default
    //! 10
    //! @par Status
    //! Experimental
    int qsgw_unoccupied_keep;

    //! QSGW H_QSGW cut: 0 uncut, 1 shifted KS diagonal (formerly mode 2).
    //! Applies to the SCF grid and optional band path; HR exports the actual cut H.
    //! @par Default
    //! 0
    //! @par Status
    //! Experimental
    int qsgw_cut_mode;

    //! Hamiltonian energy shift applied to cut states in mode 1, in Hartree.
    //! The omitted default is 2 eV converted using the selected constants.
    //! Applies to the SCF grid and optional band path.
    //! @par Default
    //! 2.0 / 27.211386245988 (2 eV in Hartree with internal constants)
    //! @par Status
    //! Experimental
    double qsgw_cut_shift_ha;

    //! Write per-iteration matrices used by numerical regression tests.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    bool qsgw_write_iteration_matrices;

    //! Include V_H[D_t-D_DFT] in each QSGW update, using current physical occupations.
    //! Requires a dedicated prefix_coul_hartree bare 3D G=0-zero metric matching
    //! the selected auxiliary basis. Ordinary GW Coulomb inputs remain required.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool qsgw_update_hartree;

    //! Solve the diagonal G0W0 quasiparticle equation on every entering QSGW state.
    //! Reuses the existing EXX/Sigma matrices in the current eigenbasis and the
    //! G0W0 analytic-continuation/QPE options. Writes qsgw_qp.csv for the SCF grid
    //! and optional band path, plus energy_qp_N and GW_band_spin_S_N.dat
    //! in the ordinary G0W0 file formats for each update.
    //! These diagnostic roots do not enter the QSGW update.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool qsgw_output_qp;

    //! Export the complete SCF-grid Hamiltonian before each QSGW diagonalization.
    //! Writes full complex Ry CSR as hrsS_nao_qsgw_iter_NNNN.csr, with H padded
    //! to the actual H/S/r support union. Preserves native DFT hrs1_nao.csr
    //! (also hrs2_nao.csr for two physical collinear channels) and transforms
    //! only the complete current correction. Native DFT H input is in Ry;
    //! S/r use qsgw_fn_sr and qsgw_fn_rr defaults. Requires a complete unshifted
    //! mesh, independently of head/wing and band inputs. SOC retains the full
    //! complex spinor matrix. current_hr differentiates this same native-baseline
    //! reconstruction; its checkpoints bind the native DFT H as well as S/r.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool qsgw_export_hamiltonian_for_pyatb;

    //! Analytic head/wing velocity convention: reference (legacy head-only),
    //! rotate_reference (fixed initial operator in current eigenstates), or
    //! current_hr (velocity of the complete current native-baseline Hamiltonian,
    //! matching formal HR export; requires native hrs1/hrs2_nao.csr in Ry).
    //! Standalone option_dielect_func=6 selects current_hr if omitted and
    //! rejects an explicit reference or rotate_reference. It resolves to
    //! numerical option 3, with the same current-HR calculation and scope.
    //! The QSGW option-6 preset selects current_hr; explicit option 4 keeps reference.
    //! @par Default
    //! reference
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string qsgw_headwing_velocity;

    //! S(R) input for current_hr or HR export: native ABACUS CSR, dimensionless, with its
    //! full real-space support; path relative to input_dir or absolute.
    //! @par Default
    //! srs1_nao.csr
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string qsgw_fn_sr;

    //! Three-component r(R) input for current_hr or HR export: native ABACUS CSR in Bohr,
    //! with its full real-space support; path relative to input_dir or absolute.
    //! @par Default
    //! rr.csr
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string qsgw_fn_rr;

    //! Checkpoint to restore at a completed QSGW iteration; relative to cwd.
    //! @par Default
    //! `none`
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string qsgw_restart_file;

    //! Checkpoint written after every complete update; relative to output_dir.
    //! Use `none` to disable checkpoint output.
    //! @par Default
    //! `qsgw_checkpoint.bin`
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string qsgw_checkpoint_file;

    //! Maximum eigenvalue change used for convergence, in eV.
    //! @par Default
    //! 1.0e-4
    //! @par Status
    //! Experimental
    double qsgw_convergence_tolerance_ev;

    //! Compute cRPA U during G0W0 (task=g0w0 or g0w0_band) or QSGW (task=qsgw).
    //! G0W0 evaluates U once; QSGW evaluates selected entering states and the final state.
    //! Requires crpa_analysis_only=false; use task=crpa for standalone screening.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool crpa_enabled = false;
    //! Electronic-state interval for cRPA during QSGW; the final state is always evaluated.
    //! @par Default
    //! 1
    //! @par Status
    //! Experimental
    //! @since Unreleased
    int crpa_interval = 1;
    //! Import only cRPA model/analysis parameters from an existing native input file.
    //! Main-input cRPA assignments override imported settings; shared GW options stay in the main
    //! input.
    //! @par Default
    //! none
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_model_file = "none";
    //! Native spatial or interleaved SOC S(R) CSR input, relative to the main input_dir.
    //! @par Default
    //! SR.csr
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_overlap_file = "SR.csr";
    //! Explicit species labels in numeric structure-type order, separated by spaces.
    //! Required for task=crpa, including analysis-only runs, or crpa_enabled=true.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_species_labels;
    //! Species supplying first-radial correlated atomic trials for the declared parent frame.
    //! Required for standalone task=crpa screening and coupled crpa_enabled=true calculations.
    //! Not required when crpa_analysis_only=true.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_correlated_species;
    //! Species supplying the first radial p shell of a dp parent frame.
    //! Must differ from crpa_correlated_species; spectator species are not selected.
    //! Required for standalone or coupled screening when crpa_parent_orbitals=dp.
    //! Not required when crpa_analysis_only=true.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_ligand_species;
    //! Atomic trials projected into B and jointly metric-normalized: s, p, sp, d, dp, eg or t2g.
    //! eg/t2g refer to global Cartesian axes, using the first radial shell.
    //! @par Default
    //! d
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_parent_orbitals = "d";
    //! Reported subset of the jointly normalized parent frame: s, p, sp, d, dp, eg or t2g.
    //! Must be a subset of crpa_parent_orbitals; it does not select response bands A.
    //! @par Default
    //! d
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_output_orbitals = "d";
    //! Response rule: bands uses binary A windows/lists; orbital_weights uses
    //! probabilities from the actual output orbitals, without an A selector.
    //! @par Default
    //! bands
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_response_mode = "bands";
    //! Transport the complete parent Phi through B after each QSGW update, without screening.
    //! false; the first call uses the declared atomic trials. Restart retains the parent.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool crpa_track_orbitals = false;
    //! Smallest retained probability of a previous parent direction; default 0.99.
    //! A loss stops the calculation and requires a wider declared B.
    //! @par Default
    //! 0.99
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_tracking_min_capture = 0.99;
    //! A: inclusive absolute-Hartree lower/upper pairs selecting original KS states
    //! on both occupied and empty branches of Pd. Independent of output window B.
    //! For screening with crpa_response_mode=bands, specify exactly one of this parameter
    //! and crpa_response_bands. Optional in analysis-only bands mode. Must be empty for
    //! crpa_response_mode=orbital_weights, including analysis-only runs.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::vector<double> crpa_response_windows_ha;
    //! B: inclusive absolute-Hartree lower/upper pairs for projecting atomic trials.
    //! Does not change KS poles, occupations, or the response selection A.
    //! For screening in either response mode, specify exactly one of this parameter
    //! and crpa_orbital_bands. Not required when crpa_analysis_only=true.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::vector<double> crpa_orbital_windows_ha;
    //! A: distinct zero-based original KS indices shared by all spin/k points.
    //! For screening with crpa_response_mode=bands, specify exactly one of this parameter
    //! and crpa_response_windows_ha. Optional in analysis-only bands mode. Must be empty for
    //! crpa_response_mode=orbital_weights, including analysis-only runs.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::vector<int> crpa_response_bands;
    //! B: distinct zero-based original KS indices shared by all spin/k points.
    //! For screening in either response mode, specify exactly one of this parameter
    //! and crpa_orbital_windows_ha. Not required when crpa_analysis_only=true.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::vector<int> crpa_orbital_bands;
    //! Positive metric/KS orthogonality tolerance. Frobenius residual gates use
    //! this value times sqrt(matrix dimension); input KS states are never repaired.
    //! @par Default
    //! 1.0e-10
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_residual_tol = 1.0e-10;

    //! Output Mulliken/Lowdin weights and Gaussian PDOS before cRPA screening.
    //! Uses existing scalar/spinor KS states and spatial S; includes every radial shell.
    //! Spinor weights sum both components in one stored spin channel.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool crpa_output_analysis;
    //! Save current E/f, complex parent/output projectors, H/density and U in compact NumPy
    //! archives.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool crpa_output_model = true;
    //! Native model output container: npz or human-readable dat files. Both retain the same model.
    //! @par Default
    //! npz
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_output_format = "npz";
    //! Include C, S and AO coefficients Phi in the model archives for changing the local frame.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool crpa_output_raw = false;
    //! Estimate U(0) with native Padé of U-v; retain entrywise numerical status and fit spread.
    //! @par Default
    //! true
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool crpa_output_static = true;
    //! Absolute tolerance for static/real-axis subset and node residual diagnostics, in eV.
    //! @par Default
    //! 0.01
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_fit_tol_ev = 0.01;
    //! Real frequencies in eV relative to bosonic zero; an empty list skips real-axis output.
    //! @par Default
    //! Empty
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::vector<double> crpa_real_frequencies_ev;
    //! Positive imaginary broadening for requested real-frequency interactions, in eV.
    //! @par Default
    //! 0.05
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_eta_ev = 0.05;
    //! Analyze orbital character only; skip RI/Coulomb input and screening.
    //! A/B selections and correlated species are not required in this mode.
    //! Requires task=crpa and crpa_output_analysis=true; incompatible with crpa_enabled=true.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    //! @since Unreleased
    bool crpa_analysis_only;
    //! PDOS energy zero: vbm (highest occupied KS pole) or fermi.
    //! @par Default
    //! vbm
    //! @par Status
    //! Experimental
    //! @since Unreleased
    std::string crpa_pdos_reference;
    //! Lower PDOS grid edge in eV relative to crpa_pdos_reference.
    //! @par Default
    //! -10
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_pdos_min_ev;
    //! Upper PDOS grid edge in eV relative to crpa_pdos_reference.
    //! @par Default
    //! 14
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_pdos_max_ev;
    //! PDOS grid spacing in eV.
    //! @par Default
    //! 0.02
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_pdos_step_ev;
    //! Gaussian standard deviation in eV (not FWHM).
    //! @par Default
    //! 0.15
    //! @par Status
    //! Experimental
    //! @since Unreleased
    double crpa_pdos_sigma_ev;

    std::string format();

    //! Apply producer-specific defaults to input filename and prefix parameters.
    void apply_input_preset();

    DriverParams();
};

extern DriverParams driver_params;
extern librpa::reader::ReaderState reader_state;

extern const std::string input_filename;

// Types of each atom, read from structure file and also used to generate basis list
extern std::vector<int> &atom_types;
extern size_t &n_atoms;

// Dimension information, used across a few read_data functions
extern int &n_spins;
extern int &n_kpoints;
extern int &n_ibz_kpoints;
extern int &n_kpoints_band;
extern int &n_states;
extern int &n_basis_wfc;
extern int &n_basis_ao;
extern int &n_spinor;
extern std::vector<size_t> &nbs_wfc;
extern std::vector<size_t> &nbs_aux;
extern std::vector<size_t> &nbs_aux_shrink;

// Used for parallel distribution of input SCF KS eigenvectors over k-points
extern std::vector<int> &iks_eigvec_this;
extern std::vector<int> &iks_band_eigvec_this;

extern std::vector<std::pair<size_t, size_t>> local_atpair;
extern std::vector<librpa_int::Vector3_Order<double>> &ibz_kpoints;
extern std::vector<librpa_int::Vector3_Order<double>> &kfrac_band;

// Store basis convention label
extern bool &is_basis_convention_read;
extern std::string &basis_convention_label;

// Working handle
extern librpa::Handler h;

// Working runtime options
extern librpa::Options opts;

// Original dielectric-method request for driver diagnostics, before preset
// resolution. This is not an input keyword or a numerical API option.
extern int requested_option_dielect_func;

// TODO: consider move to public API
std::string format_runtime_options(const librpa::Options &opts) noexcept;

LibrpaAnalyContMethod get_anacon_method(const std::string &method);
std::string get_anacon_string(LibrpaAnalyContMethod method) noexcept;

LibrpaTimeFreqGrid get_tfgrid_type(const std::string& grid_str);

std::string get_tfgrid_string(const LibrpaTimeFreqGrid& grid_type) noexcept;

inline LibrpaSwitch get_switch(bool switch_bool) noexcept
{
    return switch_bool ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
}

inline bool get_bool(LibrpaSwitch switch_in) noexcept
{
    return switch_in == LIBRPA_SWITCH_ON;
}

LibrpaParallelRouting get_parallel_routing(const std::string& routing_str_low);

std::string get_routing_string(LibrpaParallelRouting routing);

LibrpaMethod get_method(const std::string& method_str_low);

std::string get_method_string(LibrpaMethod method);

LibrpaVerbose get_verbose(const std::string& verbose_str_low);

std::string get_verbose_string(LibrpaVerbose verbose);

}
