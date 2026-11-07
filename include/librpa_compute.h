#pragma once
/**
 * @file librpa_compute.h
 * @brief Computing data APIs for LibRPA.
 */

#include "librpa_handler.h"
#include "librpa_input.h"
#include "librpa_options.h"

// C APIs
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Construct and return frequency grids for numerical integration.
 *
 * Generates the frequency points and quadrature weights based on the grid type
 * specified in the options (e.g., Gauss-Legendre, Minimax, etc.).
 *
 * @param[in]  h        Handler.
 * @param[in]  p_opts   Runtime options.
 * @param[out] omegas  Array of frequency points (size: opts->nfreq).
 * @param[out] weights Array of quadrature weights (size: opts->nfreq).
 */
void librpa_get_imaginary_frequency_grids(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                          double *omegas, double *weights);

/**
 * @brief Compute the physical, spin-summed periodic Hartree potential in the spatial AO basis.
 *
 * Collective on the handler communicator. Rebuilds from the current eigenvectors
 * and occupations on every call, without EXX spin factors or an occupation cutoff.
 * Requires a complete Gamma-centred full-BZ grid, matched full/shrunk RI data, and a bare q=0
 * Coulomb metric supplied through the normal Coulomb-input API. Its electrostatic
 * zero mode must be chosen explicitly by the host (e.g. G=0 omitted). An EXX
 * singularity-corrected q=0 metric is not interchangeable with that metric.
 * Uses the native RI Fourier orientation exp(+2*pi*i*k.R); mean-field and RI
 * inputs must share this orientation. Atom-position phase must be zero.
 * Requires official LibRI Hartree support and complete directed coefficients
 * loaded with LIBRPA_ROUTING_LIBRI, with unique MPI ownership of each Cs block.
 * Strict 2D is not yet supported.
 *
 * @param h Handler.
 * @param p_opts Options; use_shrink_abfs selects both RI and auxiliary metric basis.
 * @param n_kpoints Number of requested SCF k indices on this rank; may be zero.
 * @param kpoints Zero-based indices in the loaded full SCF grid.
 * @param n_aos Spatial AO dimension.
 * @param values_re Real output in Hartree, row-major [requested k][AO row][AO column].
 * @param values_im Imaginary output, same layout. Each buffer holds n_kpoints*n_aos*n_aos doubles.
 */
void librpa_compute_hartree_ao(LibrpaHandler* h, const LibrpaOptions* p_opts,
                              int n_kpoints, const int* kpoints, int n_aos,
                              double* values_re, double* values_im);


/**
 * @brief Compute RPA correlation energy.
 *
 * Calculates the RPA correlation energy using the input data (wavefunctions,
 * Coulomb matrices, etc.) that has been set via the input parsing functions.
 *
 * @param[in]  h                           Handler.
 * @param[in]  p_opts                      Runtime options.
 * @param[in]  n_ibz_kpoints                Number of irreducible k-points.
 * @param[out] rpa_corr_ibzk_contrib_re   Real part of correlation energy per IBZ k-point.
 * @param[out] rpa_corr_ibzk_contrib_im   Imaginary part of correlation energy per IBZ k-point.
 * @return Total RPA correlation energy (real part).
 */
double librpa_get_rpa_correlation_energy(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                         int n_ibz_kpoints, double *rpa_corr_ibzk_contrib_re,
                                         double *rpa_corr_ibzk_contrib_im);

//! Build exact-exchange matrix
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 */
void librpa_build_exx(LibrpaHandler *h, const LibrpaOptions *p_opts);

//! Obtain exact-exchange potential for selected states.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_this      Number of k-points to compute on this process.
 * @param[in]  iks_this         (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[out] vexx             Exact-exchange potential for selected states.
 *                              It should be at least as long as n_spins * n_kpts_local * (i_state_high - i_state_low).
 */
void librpa_get_exx_pot_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                              const int n_spins, const int n_kpts_this, const int *iks_this,
                              int i_state_low, int i_state_high, double *vexx);

//! Obtain exact-exchange potential for selected states at band k-points.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_band_this Number of k-points to compute on this process.
 * @param[in]  iks_band_this    (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of band k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[out] vexx_band        Exact-exchange potential for selected states at band k-points.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 */
void librpa_get_exx_pot_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                               const int n_spins, const int n_kpts_band_this, const int *iks_band_this,
                               int i_state_low, int i_state_high, double *vexx_band);

//! Build self-energy matrix of G0W0, including the correlation and exchange contributions.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 */
void librpa_build_g0w0_sigma(LibrpaHandler *h, const LibrpaOptions *p_opts);

//! Obtain correlation self-energies for selected states.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_this      Number of k-points to compute on this process.
 * @param[in]  iks_this         (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[in]  vxc              exchange-correlation potential of the selected states.
 * @param[in]  vexx             Exact-exchange potential for the selected states.
 *                              It should be at least as long as n_spins * n_kpoints_local * (i_state_high - i_state_low).
 *                              It can be obtained using librpa_get_exx_pot_kgrid.
 * @param[out] sigc_re          Real-part of the correlation self-energy for the selected states.
 *                              For option_qpe_solver=2, this is the perturbative effective real
 *                              contribution that reconstructs the perturbative QP energy.
 *                              It should be at least as long as n_spins * n_kpoints_local * (i_state_high - i_state_low).
 * @param[out] sigc_im          Same as sigc_re, but for the imaginary part.
 */
void librpa_get_g0w0_sigc_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                const int n_spins, const int n_kpts_this,
                                const int *iks_this, int i_state_low, int i_state_high,
                                const double *vxc, const double *vexx, double *sigc_re, double *sigc_im);

//! Obtain correlation self-energies and QP energies for selected states.
/**
 * Same inputs and SigC outputs as librpa_get_g0w0_sigc_kgrid.
 * If p_opts->use_hedin_shift is enabled, the QP solver evaluates
 * \f[
 * \Sigma_c(E - E_F - \Delta).
 * \f]
 * If p_opts->istate_ref_hedin_shift is negative, Delta is computed separately
 * for each selected state. Otherwise, that absolute zero-based reference state
 * at each k-point and spin provides Delta for all states in the same channel.
 *
 * @param[in]  h           LibRPA handler.
 * @param[in]  p_opts      Runtime options.
 * @param[in]  n_spins     Number of spin channels.
 * @param[in]  n_kpts_this Number of k-points computed on this process.
 * @param[in]  iks_this    Global indices of those k-points.
 * @param[in]  i_state_low First state index (inclusive).
 * @param[in]  i_state_high Last state index (exclusive).
 * @param[in]  vxc         XC potential for the selected states.
 * @param[in]  vexx        Exact-exchange potential for the selected states.
 * @param[out] sigc_re     Real part of the correlation self-energy.
 * @param[out] sigc_im     Imaginary part of the correlation self-energy.
 * @param[out] eqp              Quasi-particle energy solved by the selected QPE solver.
 *                              It should be at least as long as sigc_re/sigc_im.
 */
void librpa_get_g0w0_qpe_kgrid(LibrpaHandler *h, const LibrpaOptions *p_opts,
                               const int n_spins, const int n_kpts_this,
                               const int *iks_this, int i_state_low, int i_state_high,
                               const double *vxc, const double *vexx, double *sigc_re,
                               double *sigc_im, double *eqp);

//! Obtain spectral functions for selected k-grid states.
/**
 * Computes spectral functions under the diagonal approximation of the Green's
 * function:
 * \f$A_{n\mathbf{k}}(\omega) = -\pi^{-1}\mathrm{Im}\,G_{n\mathbf{k}}(\omega)\f$.
 *
 * @param[in]  h                       Pointer to LibRPA handler.
 * @param[in]  p_opts                  Pointer to runtime options.
 * @param[in]  n_spins                 Number of spin channels.
 * @param[in]  n_kpts_this             Number of k-points to compute on this process.
 * @param[in]  iks_this                Global k-point indices computed on this process.
 * @param[in]  i_state_low             First state index (inclusive).
 * @param[in]  i_state_high            Last state index (exclusive).
 * @param[in]  n_omegas                Number of real-frequency points.
 * @param[in]  omegas                  Real-frequency points, in Hartree.
 * @param[in]  vxc                     XC potential for selected states.
 * @param[in]  vexx                    Exact-exchange potential for selected states.
 * @param[out] spectral_function       Spectral functions. This output array should be at least as long as
 *                                     n_spins * n_kpts_this * (i_state_high - i_state_low) * n_omegas.
 *                                     The order is [spin][k-point][state][omega].
 * @param[out] sigc                    Optional continued correlation self-energy. Pass nullptr if this output is
 *                                     not needed. If present, this is a packed complex<double> buffer with the
 *                                     same [spin][k-point][state][omega] order as spectral_function.
 */
void librpa_get_g0w0_spectral_function_kgrid(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_this, const int *iks_this, int i_state_low, int i_state_high,
    const int n_omegas, const double *omegas, const double *vxc, const double *vexx,
    double *spectral_function, double *sigc);

//! Obtain correlation self-energies for selected states at band k-points.
/**
 * @param[in]  h                Pointer to LibRPA handler.
 * @param[in]  p_opts           Pointer to runtime options.
 * @param[in]  n_spins          Number of spin channels.
 * @param[in]  n_kpts_band_this Number of k-points to compute on this process.
 * @param[in]  iks_band_this    (Global) index of k-points that this process compute.
 *                              Each process can have different indices.
 *                              Must be a subset of k-points at which the eigenvetors are parsed.
 * @param[in]  i_state_low      Index of the first state to compute the potential (inclusive)
 * @param[in]  i_state_high     Index of the last state to compute the potential (exclusive)
 * @param[in]  vxc_band         exchange-correlation potential of the selected states at band k-points.
 * @param[in]  vexx_band        Exact-exchange potential for the selected states at band k-points.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 *                              It can be obtained using librpa_get_exx_pot_kgrid.
 * @param[out] sigc_band_re     Real-part of the correlation self-energy for the selected states.
 *                              For option_qpe_solver=2, this is the perturbative effective real
 *                              contribution that reconstructs the perturbative QP energy.
 *                              It should be at least as long as n_spins * n_kpts_band_this * (i_state_high - i_state_low).
 * @param[out] sigc_band_im     Same as sigc_band_re, but for the imaginary part.
 */
void librpa_get_g0w0_sigc_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                 const int n_spins, const int n_kpts_band_this,
                                 const int *iks_band_this, int i_state_low, int i_state_high,
                                 const double *vxc_band, const double *vexx_band, double *sigc_band_re, double *sigc_band_im);

//! Obtain correlation self-energies and QP energies for selected band-k states.
/**
 * Same inputs and SigC outputs as librpa_get_g0w0_sigc_band_k.
 * If p_opts->use_hedin_shift is enabled, the QP solver evaluates
 * \f[
 * \Sigma_c(E - E_F - \Delta).
 * \f]
 * If p_opts->istate_ref_hedin_shift is negative, Delta is computed separately
 * for each selected band state. Otherwise, that absolute zero-based reference
 * state at each band k-point and spin provides Delta for all states in the same channel.
 *
 * @param[in]  h                 LibRPA handler.
 * @param[in]  p_opts            Runtime options.
 * @param[in]  n_spins           Number of spin channels.
 * @param[in]  n_kpts_band_this  Number of band k-points computed on this process.
 * @param[in]  iks_band_this     Global indices of those band k-points.
 * @param[in]  i_state_low       First state index (inclusive).
 * @param[in]  i_state_high      Last state index (exclusive).
 * @param[in]  vxc_band          XC potential for the selected band states.
 * @param[in]  vexx_band         Exact-exchange potential for the selected band states.
 * @param[out] sigc_band_re      Real part of the correlation self-energy.
 * @param[out] sigc_band_im      Imaginary part of the correlation self-energy.
 * @param[out] eqp_band         Quasi-particle energy solved by the selected QPE solver.
 *                              It should be at least as long as sigc_band_re/sigc_band_im.
 */
void librpa_get_g0w0_qpe_band_k(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                const int n_spins, const int n_kpts_band_this,
                                const int *iks_band_this, int i_state_low, int i_state_high,
                                const double *vxc_band, const double *vexx_band,
                                double *sigc_band_re, double *sigc_band_im,
                                double *eqp_band);

//! Obtain spectral functions for selected band-k states.
/**
 * Same convention as librpa_get_g0w0_spectral_function_kgrid, but evaluates
 * states at band k-points.
 *
 * @param[in]  h                       Pointer to LibRPA handler.
 * @param[in]  p_opts                  Pointer to runtime options.
 * @param[in]  n_spins                 Number of spin channels.
 * @param[in]  n_kpts_band_this        Number of band k-points to compute on this process.
 * @param[in]  iks_band_this           Global band-k indices computed on this process.
 * @param[in]  i_state_low             First state index (inclusive).
 * @param[in]  i_state_high            Last state index (exclusive).
 * @param[in]  n_omegas                Number of real-frequency points.
 * @param[in]  omegas                  Real-frequency points, in Hartree.
 * @param[in]  vxc_band                XC potential for selected band states.
 * @param[in]  vexx_band               Exact-exchange potential for selected band states.
 * @param[out] spectral_function_band  Spectral functions. This output array should be at least as long as
 *                                     n_spins * n_kpts_band_this * (i_state_high - i_state_low) * n_omegas.
 *                                     The order is [spin][band k-point][state][omega].
 * @param[out] sigc_band               Optional continued correlation self-energy. Pass nullptr if this output is
 *                                     not needed. If present, this is a packed complex<double> buffer with the
 *                                     same [spin][band k-point][state][omega] order as spectral_function_band.
 */
void librpa_get_g0w0_spectral_function_band_k(
    LibrpaHandler *h, const LibrpaOptions *p_opts, const int n_spins,
    const int n_kpts_band_this, const int *iks_band_this, int i_state_low,
    int i_state_high, const int n_omegas, const double *omegas,
    const double *vxc_band, const double *vexx_band, double *spectral_function_band,
    double *sigc_band);

/** Collective velocity computation and setter; optional analytic head/wing
 * preparation reuses the ordinary numerical API. Options use API values 3/4,
 * never driver preset 6. Optical preparation requires replace_w_head enabled.
 * Option 3 also requires use_scalapack_gw_wc to consume the analytic wing.
 * Input is borrowed only for this call; input structures are in librpa_input.h.
 */
void librpa_prepare_native_velocity(LibrpaHandler *h, const LibrpaOptions *options,
                                    const LibrpaNativeVelocityInput *root_input,
                                    int prepare_optical);

/** Independently owned, rank-local snapshot of numerical optical outputs.
 * Obtain this structure with librpa_get_optical_result and release it only with
 * librpa_delete_optical_result. Do not modify its fields, free its arrays, or
 * pass a copied structure to the deletion function. All arrays are read-only
 * and remain valid until that original result is deleted, including after the
 * handler is changed or destroyed. Empty arrays have null pointers.
 *
 * Complex arrays are row-major, interleaved [real,imag]. Velocity is
 * [n_spins][n_kpoints][3][n_states][n_states][2], in Hartree*Bohr;
 * frequencies_ha contains n_frequencies imaginary-frequency nodes in Hartree.
 * Head and wing are [n_frequencies][rows][columns][2], with their respective
 * dimensions below. Head is dimensionless; the auxiliary-basis wing uses
 * LibRPA's Coulomb-normalized response convention. No MPI gathering is done.
 */
typedef struct LibrpaOpticalResult
{
    int n_spins, n_kpoints, n_states, n_frequencies;
    int head_rows, head_columns, wing_rows, wing_columns;
    const double *velocity_ri;
    const double *frequencies_ha;
    const double *head_ri;
    const double *wing_ri;
} LibrpaOpticalResult;

/** Source-compatible borrowed view. Copying it does not copy or own the arrays. */
typedef LibrpaOpticalResult LibrpaOpticalView;

/** Noncollective snapshot; analytic head/wing preparation must be complete. */
LibrpaOpticalResult *librpa_get_optical_result(const LibrpaHandler *h);
/** Return a borrowed copy of the public fields, with the original result's lifetime. */
LibrpaOpticalView librpa_get_optical_view(const LibrpaOpticalResult *result);
/** Release a result and its arrays. Passing NULL is permitted. */
void librpa_delete_optical_result(LibrpaOpticalResult *result);

typedef enum
{
    LIBRPA_QSGW_ITERATIONS,
    LIBRPA_QSGW_EIGENVALUES,
    LIBRPA_QSGW_MATRICES,
    LIBRPA_QSGW_QUASIPARTICLES,
    LIBRPA_QSGW_ENERGY_QP,
    LIBRPA_QSGW_GW_BAND,
    LIBRPA_QSGW_KS_BAND,
    LIBRPA_QSGW_EXX_BAND,
    LIBRPA_QSGW_QSGW_BAND,
    LIBRPA_QSGW_HAMILTONIAN,
    LIBRPA_QSGW_HEAD_HAMILTONIAN,
    LIBRPA_QSGW_HARTREE
} LibrpaQsgwOutputKind;
/** Borrowed output on rank zero only, valid during the callback. spin is
 * zero-based; text_size excludes the terminating zero. The callback presents
 * output only and must not mutate or recursively use the handler/task. */
typedef struct
{
    LibrpaQsgwOutputKind kind;
    int iteration, spin, append;
    const char *text;
    size_t text_size;
    const LibrpaLocalizedOperator *hamiltonian;
} LibrpaQsgwOutput;
typedef int (*LibrpaQsgwOutputSink)(const LibrpaQsgwOutput *, void *user_data);

/** All energies are Hartree unless the field explicitly says eV. Strings and
 * numeric inputs are copied by the call receiving them. Checkpoints preserve
 * the immutable input eigenframe; original producer inputs remain required.
 * Keep the handler alive until the task is deleted. After preparation keep
 * its SCF eigenframe/grid fixed; finish band/Coulomb/velocity input before
 * run. Do not modify or reenter the handler during run or its callbacks. Call prepare and
 * set_input collectively before the single run call. */
LibrpaQsgwTask *librpa_create_qsgw_task(LibrpaHandler *, const LibrpaOptions *,
                                        const LibrpaQsgwControls *);
/** Collective preparation. If IBZ expansion is triggered, the handler's SCF
 * mean field and k grid are replaced by full-grid states and existing compute
 * objects are invalidated. Mean-field getters then expose the expanded grid.
 * Supply Vxc for the original source grid and gauge; retain its dimensions
 * before preparation if needed. */
void librpa_prepare_qsgw_task(LibrpaQsgwTask *);
void librpa_set_qsgw_output_sink(LibrpaQsgwTask *, LibrpaQsgwOutputSink, void *user_data);
void librpa_run_qsgw_task(LibrpaQsgwTask *);
void librpa_delete_qsgw_task(LibrpaQsgwTask *);

/** Query the numerical dimensions needed to prepare a QSGW task's host input.
 * band_path selects band input when nonzero. This call is noncollective.
 */
void librpa_get_meanfield_dimensions(const LibrpaHandler *h, int band_path, int *nspins, int *nkpts,
                                     int *nstates, int *naos, int *nspinor);

/** Orbital output structures are caller-owned views. Their array and string
 * pointers borrow read-only storage from the indexed result handle. Keep the
 * result alive while using these pointers; callback results and their views
 * expire when the callback returns. AO arrays have n_aos entries and group
 * arrays have n_groups entries, as returned by librpa_orbital_info. */
    typedef struct
    {
        int n_spins, n_spinor, n_kpoints, n_bands, n_aos, n_groups;
        double fermi_ha, highest_occupied_ha;
        double min_overlap_eigenvalue, max_norm_error, max_sum_error, max_sqrt_residual;
    } LibrpaOrbitalInfo;

    typedef struct
    {
        int atom_index, type_index, l;
        const char* channel;  // s, p, eg, t2g, or lN; global Cartesian axes.
    } LibrpaOrbitalGroup;

    typedef struct
    {
        double energy_ha, occupation, kweight, norm;
        const double* mulliken_ao;
        const double* lowdin_ao;
        const double* mulliken_group;
        const double* lowdin_group;
    } LibrpaOrbitalState;

    /** Collective orbital analysis from loaded eigenstates and spatial S(k).
     * S is packed [k][AO row][AO column][real/imag], replicated on every rank.
     * Requires complete scalar or two-component spinor coefficients, shell metadata, explicit basis
     * conventions, and normalized loaded k weights. Evaluates S^(1/2), never H.
     * Mulliken = Re[c* (Sc)]; Lowdin = |S^(1/2)c|^2/(c-dagger S c).
     * For spinors, sum both components in each spatial AO and normalize by their
     * combined S norm. n_spins is one; no extra spin-degeneracy factor is added.
     * All radial shells contribute to atom/l groups, with d split into eg/t2g.
     * Does not alter the handler, eigenstates, cRPA windows or response weights.
     * Each rank owns its result and must delete it. Views remain valid until deletion.
     */
    LibrpaOrbitalResult* librpa_analyze_orbitals(const LibrpaHandler* h, int n_kpoints, int n_aos,
                                                 const double* overlap_k_ri, double residual_tol);
    void librpa_delete_orbital_result(LibrpaOrbitalResult* result);
    void librpa_orbital_info(const LibrpaOrbitalResult* result, LibrpaOrbitalInfo* info);
    void librpa_orbital_group(const LibrpaOrbitalResult* result, int group,
                              LibrpaOrbitalGroup* view);
    void librpa_orbital_state(const LibrpaOrbitalResult* result, int spin, int k, int band,
                              LibrpaOrbitalState* view);

    /** Local Gaussian PDOS from an existing result; no collective or files.
     * Grid is absolute Hartree and sigma is Hartree. Output arrays, supplied by
     * the caller, each have n_spins*n_energies*n_groups doubles in [spin][E][group]
     * order and units states/(Ha cell stored-spin-channel). Uses loaded k weights;
     * applies neither occupation nor an extra spin-degeneracy factor.
     */
    void librpa_orbital_pdos(const LibrpaOrbitalResult* result, int n_energies,
                             const double* energy_grid_ha, double sigma_ha, double* mulliken,
                             double* lowdin);

/** cRPA accessor structures are caller-owned views. All array and string
 * pointers borrow read-only result storage and share the result handle
 * lifetime documented in librpa_handler.h. Caller-supplied output buffers
 * remain owned by the caller. */
    /** Owned snapshot dimensions and static geometry; views survive handler changes.
     * All indices are zero based. Geometry uses Bohr and fractional k coordinates.
     * n_parent/n_output include both trial components for SOC.
     */
    typedef struct
    {
        int n_spins, n_spinor, n_kpoints, n_bands, n_aos, n_atoms;
        int n_parent, n_output, n_parent_sites, n_output_sites;
        double fermi_ha;
        const double* lattice_bohr;   /* [3][3], row lattice vectors */
        const double* positions_bohr; /* [atom][3] */
        const int* atom_types;
        const int* ao_counts;       /* spatial AO count per atom */
        const double* k_fractional; /* [k][3] */
        const double* k_weights;    /* sum to one */
        const int* output_columns;  /* indices in joint parent frame */
        int n_shells;
        const int* shell_offsets;    /* [atom+1], indexing shell_l */
        const int* shell_l;          /* radial shells in producer AO order */
        const int* basis_convention; /* phase, ratom, order, coeff_negative, coeff_positive */
    } LibrpaCrpaModelInfo;

    /** Row-major complex views packed as [real,imag]; energies use Hartree.
     * C is [n_spinor*n_aos][n_bands], S is spatial [n_aos][n_aos].
     * Phi is [n_spinor*n_aos][n_orbitals], P is [n_orbitals][n_bands].
     * H is [n_orbitals][n_orbitals]; parent and output have separate dimensions.
     * Occupations are physical per-state values, without the k-point weight.
     * C/S/output-Phi pointers are null unless retain_raw_model was enabled.
     * Parent Phi is always retained for numerical subspace transport.
     */
    typedef struct
    {
        const double* energies_ha;
        const double* occupations;
        const double* coefficients_ri;
        const double* overlap_ri;
        const double* parent_phi_ri;
        const double* output_phi_ri;
        const double* parent_projector_ri;
        const double* output_projector_ri;
        const double* parent_h_ha_ri;
        const double* output_h_ha_ri;
        const double* parent_density_ri;
        const double* output_density_ri;
        double parent_capture_min;
    } LibrpaCrpaModelK;

    void librpa_crpa_model_info(const LibrpaCrpaResult* result, LibrpaCrpaModelInfo* info);
    void librpa_crpa_model_k(const LibrpaCrpaResult* result, int spin, int kpoint,
                             LibrpaCrpaModelK* view);
    void librpa_crpa_model_site(const LibrpaCrpaResult* result, int parent, int site, int* atom,
                                int* column_start, int* n_orbitals);
    const char* librpa_crpa_model_label(const LibrpaCrpaResult* result, int parent, int site,
                                        int orbital);
    /** Weighted onsite H and density in one site's actual frame, supplied output buffers [M][M][2].
     */
    void librpa_crpa_model_local(const LibrpaCrpaResult* result, int parent, int spin, int site,
                                 double* h_ha_ri, double* density_ri);
    /** Full parent/output hopping at integer lattice R, [M][M][2], Hartree.
     * Uses sum_k weight_k exp(-i 2*pi k.R) H(k). R need not be a stored BvK representative. */
    void librpa_crpa_model_hopping(const LibrpaCrpaResult* result, int parent, int spin,
                                   const int* lattice_R, double* h_ha_ri);
    /** Noncollective numerical helpers; arrays are row-major complex pairs.
     * P=[orbital][band], diagonal uses the caller's energy or occupation convention. */
    void librpa_crpa_project_one_body(int n_orbitals, int n_bands, const double* projector_ri,
                                      const double* diagonal, double* operator_ri);
    /** Construct P from an explicit metric-orthonormal frame without a KS solve.
     * C=[component-major AO][band], Phi=[component-major AO][orbital], S is spatial.
     * Returns P=[orbital][band]. The orbital construction/window is chosen by the caller. */
    void librpa_crpa_project_orbitals(int n_aos, int n_spinor, int n_bands, int n_orbitals,
                                      const double* coefficients_ri, const double* overlap_ri,
                                      const double* phi_ri, double* projector_ri);
    /** P_new=R^dagger P_old for Phi_new=Phi_old R; use the same R on the corresponding U legs. */
    void librpa_crpa_rotate_projector(int n_bands, int n_old, int n_new, const double* projector_ri,
                                      const double* rotation_ri, double* rotated_projector_ri);
    /** Phi_new=Phi_old R, R=[old][new] is an isometry; rotate both density legs.
     * No screening response is changed. Use the matching rotated projector/H. */
    void librpa_crpa_rotate_tensor(int n_old, int n_new, const double* tensor_ri,
                                   const double* left_rotation_ri, const double* right_rotation_ri,
                                   double* rotated_ri);
    /** Density U_ab,cd -> fermionic V_acbd: V_ijkl=U_ik,jl.
     * V multiplies (1/2) c_i^dagger c_j^dagger c_l c_k, with the stored spin-frame pair.
     * No antisymmetrization or Kanamori/Slater approximation is performed. */
    void librpa_crpa_tensor_to_fermion(int n_orbitals, const double* density_ri,
                                       double* fermion_ri);

    /** Ordered-column averages of U_mm,mm, U_mm,nn and U_mn,nm (m!=n).
     * Input is [M][M][M][M][2]; output is three complex pairs in the same energy unit.
     * Off-diagonal means are zero for M=1. Spinor columns use the same ordered means,
     * without a scalar-shell Hund interpretation or an added degeneracy factor. */
    void librpa_crpa_tensor_averages(int n_orbitals, const double* tensor_ri,
                                     double* averages_ha_ri);

    typedef struct
    {
        int atom_index, spin_left, spin_right, n_orbitals;
        const double* values_ha_ri;
        const int* status;              /* [M^4], zero means numerical diagnostics pass */
        const double* subset_spread_ha; /* [M^4] */
        double max_source_residual_ha, max_holdout_residual_ha;
    } LibrpaCrpaContinuedTensor;
    /** Default ridge continuation of U-v at original z=i*nu; restores v, no fermionic shift.
     * Queries are [real,imag] in Hartree: z=0 for static extrapolation, or Im(z)>0.
     * n_parameters=-1 uses all nodes. tolerance_ha controls numerical diagnostics.
     * Noncollective; returns independently owned values, not an added response node.
     * Status bits: 1 primary failure, 2 subset failure, 4 excessive subset spread,
     * 8 excessive source/holdout residual, 16 fewer than four nodes, 32 nonfinite value.
     * No passivity, causality, basis or mesh convergence certificate is implied. */
    LibrpaCrpaContinuedResult* librpa_continue_crpa(const LibrpaCrpaResult* result, int n_queries,
                                                    const double* queries_ha_ri, int n_parameters,
                                                    double tolerance_ha);
    /** Continue Delta U with the selected analytic-continuation options.
     * Same units, ownership, query and status conventions as librpa_continue_crpa.
     * options must be initialized and non-null. Uses anacon_method,
     * n_params_anacon, anacon_aaa_tolerance and the ridge penalty controls.
     * librpa_continue_crpa retains its signature and uses default ridge settings.
     */
    LibrpaCrpaContinuedResult* librpa_continue_crpa_with_options(const LibrpaCrpaResult* result,
                                                                 int n_queries,
                                                                 const double* queries_ha_ri,
                                                                 const LibrpaOptions* options,
                                                                 double tolerance_ha);

    void librpa_delete_crpa_continued_result(LibrpaCrpaContinuedResult* result);
    int librpa_crpa_continued_size(const LibrpaCrpaContinuedResult* result);
    void librpa_crpa_continued_tensor(const LibrpaCrpaContinuedResult* result, int tensor,
                                      int query, LibrpaCrpaContinuedTensor* view);

    /** Read-only tensor view. Arrays remain valid until the result is deleted.
     * Arrays contain M^4 complex entries, each packed as real/imag doubles.
     * Matrix row is the ordered pair (a,b), column is (c,d): a*M+b, c*M+d.
     * Energies and interactions are Hartree; frequency is positive imaginary nu.
     */
    typedef struct
    {
        int atom_index;
        int spin_left;
        int spin_right;
        int n_orbitals;
        double frequency_ha;
        const double* bare_ri;
        const double* partially_screened_ri;
        /** Null for U-only results; no fully screened interaction was computed. */
        const double* fully_screened_ri;
    } LibrpaCrpaTensor;


    /** Read full fractional k coordinates in the current KS eigenvector order. */
    void librpa_get_crpa_kgrid(const LibrpaHandler* h, int n_kpoints, double* fractional_k);
    /** Prepare current local frames, weights and one-body model only. Collective.
     * Builds no response, W or self-energy; returns zero interaction tensors.
     * Useful for cheap parent transport between selected screening evaluations. */
    LibrpaCrpaResult* librpa_prepare_crpa_model(const LibrpaHandler* h,
                                                const LibrpaOptions* options,
                                                const LibrpaCrpaInput* input);

    /** Collective calculation on the handler communicator; no input/output files.
     * All ranks must call with identical options and replicated LibrpaCrpaInput.
     * Supports scalar or two-component spinor full-grid LIBRI with replicated KS eigenvectors and
     * native positive minimax nodes. With use_symmetry_rpa enabled, only q-space screening is
     * reduced; kernels are restored before full-grid local contraction. Original full-k states
     * are required. No static node or analytic continuation is added. Each rank owns its result
     * independently and must delete it.
     */
    LibrpaCrpaResult* librpa_compute_crpa_window(const LibrpaHandler* h,
                                                 const LibrpaOptions* options,
                                                 const LibrpaCrpaInput* input);

    /** Build the ordinary GW self-energy and cRPA U for the same entering state.
     * P0 is built once by GW and borrowed by cRPA; only Pd and constrained
     * screening are added. GW P0, optical data, W and self-energy conventions
     * remain unchanged. Requires full-grid LIBRI and rejects a SigC file-read shortcut.
     * Input has the same full-grid and optional q-star screening contract as the standalone call.
     * The owned result contains bare/U; fully_screened_ri is null.
     * cRPA screens and projects with full Coulomb independently of the GW
     * eps/Wc/EXX Coulomb flags, which keep their original GW meaning.
     */
    LibrpaCrpaResult* librpa_build_g0w0_sigma_crpa(LibrpaHandler* h, const LibrpaOptions* options,
                                                   const LibrpaCrpaInput* input);

    /** Response-only bare/U calculation for the handler's current E/C/f.
     * Builds P0 and Pd and screens Pr, without full-W screening or self-energy.
     * Suitable for the final updated QSGW state. Uses current velocity data
     * when optical correction is enabled. Returns an independently owned
     * result whose fully_screened_ri is null. The cRPA solve uses full Coulomb
     * regardless of the independent GW Coulomb flags.
     */
    LibrpaCrpaResult* librpa_compute_crpa_u_window(const LibrpaHandler* h,
                                                   const LibrpaOptions* options,
                                                   const LibrpaCrpaInput* input);

    /** Noncollective save of this result's parent frame in the native tracking
     * checkpoint format. The caller selects the writing rank. Energy context
     * may be empty for a state export, or include every spin/k/band pole. */
    void librpa_write_crpa_tracking_checkpoint(const LibrpaCrpaResult* result, const char* filename,
                                               int iteration, const char* definition,
                                               int n_energies, const double* energies_ha);


    /** Borrowed numerical output on every rank, valid only during the call.
     * The sink may query results and write files, but must not delete these
     * views or recursively use the handler/task. The orbital result may be
     * null when analysis is unavailable. Return zero on success. */
    typedef int (*LibrpaCrpaOutputSink)(const LibrpaCrpaResult*, const LibrpaOrbitalResult*,
                                        int state_index, int gw_update, int final_state,
                                        void* user_data);
    /** Attach after librpa_prepare_qsgw_task, before run. Input and controls
     * are copied; the sink and user_data must remain valid until run returns. */
    void librpa_set_qsgw_crpa(LibrpaQsgwTask* task, const LibrpaCrpaInput* input,
                              const LibrpaQsgwCrpaControls* controls, LibrpaCrpaOutputSink sink,
                              void* user_data);

    void librpa_delete_crpa_result(LibrpaCrpaResult* result);
    int librpa_crpa_result_size(const LibrpaCrpaResult* result);
    void librpa_crpa_result_tensor(const LibrpaCrpaResult* result, int index,
                                   LibrpaCrpaTensor* tensor);

    /** Retrieve A (response_window != 0) or B selected original-KS band indices.
     * Band indices are zero based and the returned array is owned by result.
     */
    void librpa_crpa_result_bands(const LibrpaCrpaResult* result, int response_window, int spin,
                                  int kpoint, int* count, const int** bands);
    /** Actual endpoint weights [band], owned by result. Binary or fractional,
     * depending on the input mode. Weighted mode has no binary A band list. */
    void librpa_crpa_result_weights(const LibrpaCrpaResult* result, int spin, int kpoint,
                                    int* count, const double** weights);

    /** Trial label for a local output column; owned by result until deletion.
     * SOC suffix up/down identifies a trial, not a conserved spin quantum number.
     * SOC tensor spin_left/right are channel zero; spinor indices are a,b,c,d. */
    const char* librpa_crpa_result_orbital_label(const LibrpaCrpaResult* result, int atom,
                                                 int orbital);

#ifdef __cplusplus
}
#endif
