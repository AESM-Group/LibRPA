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
    //! required (`rpa`, `g0w0`, `exx`, `exx_band`)
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

    //! Prefix of shrink auxiliary-basis transform files.
    //! @par Default
    //! `shrink_sinvS_`
    //! @par Status
    //! Experimental
    std::string prefix_shrink_sinvS;

    //! Prefix of bare Coulomb matrix files.
    //! @par Default
    //! `coulomb_mat`
    std::string prefix_coul_full;

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

    //! Band-path k-point metadata filename.
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
    //! @par Default
    //! `none`
    //! @par Status
    //! Experimental
    std::string qsgw_mixer;

    //! Linear Hamiltonian mixing fraction.
    //! @par Default
    //! 0.2
    //! @par Status
    //! Experimental
    double qsgw_mixing_beta;

    //! Minimum number of QSGW updates before convergence may terminate the run.
    //! @par Default
    //! 1
    //! @par Status
    //! Experimental
    int qsgw_min_iter;

    //! Maximum number of QSGW updates.
    //! @par Default
    //! 10
    //! @par Status
    //! Experimental
    int qsgw_max_iter;

    //! Number of unoccupied states retained by qsgw_band H_QSGW cut.
    //! @par Default
    //! 10
    //! @par Status
    //! Experimental
    int qsgw_band0_unoccupied_keep;

    //! qsgw_band H_QSGW cut: 0 uncut, 1 KS diagonal, 2 shifted KS diagonal.
    //! @par Default
    //! 0
    //! @par Status
    //! Experimental
    int qsgw_band0_cut_mode;

    //! Hamiltonian energy shift applied to cut states in mode 2, in Hartree.
    //! @par Default
    //! 20.0
    //! @par Status
    //! Experimental
    double qsgw_band0_cut_shift_ha;

    //! Write per-iteration matrices used by numerical regression tests.
    //! @par Default
    //! false
    //! @par Status
    //! Experimental
    bool qsgw_write_iteration_matrices;

    //! Maximum eigenvalue change used for convergence, in eV.
    //! @par Default
    //! 1.0e-4
    //! @par Status
    //! Experimental
    double qsgw_convergence_tolerance_ev;

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

// TODO: consider move to public API
std::string format_runtime_options(const librpa::Options &opts) noexcept;

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

LibrpaVerbose get_verbose(const std::string& verbose_str_low);

std::string get_verbose_string(LibrpaVerbose verbose);

}
