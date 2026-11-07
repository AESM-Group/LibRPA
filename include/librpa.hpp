#pragma once
/**
 * @file librpa.hpp
 * @brief C++ wrapper API for LibRPA.
 *
 * This file provides a convenient C++ interface to LibRPA functionality.
 * It wraps the C API with STL containers for easier use.
 *
 * Usage:
 * @code
 * #include "librpa.hpp"
 * @endcode
 *
 * @note This header includes librpa_options.h and librpa_handler.h which provide the underlying C types.
 */

#include "librpa_enums.h"
#include "librpa_options.h"
#include "librpa_handler.h"
#include "librpa_compute.h"

#include <array>
#include <vector>
#include <complex>
#include <memory>

/**
 * @brief Main namespace for LibRPA C++ API.
 */
namespace librpa
{

using QsgwTask = std::unique_ptr<LibrpaQsgwTask, decltype(&librpa_delete_qsgw_task)>;
using OpticalResult = std::unique_ptr<LibrpaOpticalResult, decltype(&librpa_delete_optical_result)>;
using OrbitalResult = std::unique_ptr<LibrpaOrbitalResult, decltype(&librpa_delete_orbital_result)>;
using CrpaResult = std::unique_ptr<LibrpaCrpaResult, decltype(&librpa_delete_crpa_result)>;
using CrpaContinuedResult =
    std::unique_ptr<LibrpaCrpaContinuedResult, decltype(&librpa_delete_crpa_continued_result)>;
/** Noncollective native continuation. Energies are Hartree; default diagnostic tolerance is 0.01
 * eV. */
CrpaContinuedResult continue_crpa(const LibrpaCrpaResult *result,
                                  const std::vector<std::complex<double>> &queries,
                                  int n_parameters = -1,
                                  double tolerance_ha = 0.0003674932217565499);

/** Mean direct, inter-orbital direct and exchange tensor elements, in Hartree. */
std::array<std::complex<double>, 3> crpa_tensor_averages(
    int n_orbitals, const std::vector<std::complex<double>> &tensor);

/* Enums */
/**
 * @brief Parallel routing strategy (C++ alias).
 * @see LibrpaParallelRouting
 */
typedef LibrpaParallelRouting ParallelRouting;

typedef LibrpaTimeFreqGrid TimeFreqGrid;  ///< Time/frequency grid type (C++ alias)

typedef LibrpaSwitch Switch;  ///< Boolean switch type (C++ alias)

typedef LibrpaKind Kind;  ///< DFT code kind type (C++ alias, reserved)

typedef LibrpaVerbose Verbose;  ///< Verbosity level type (C++ alias)

/** @brief G0W0 correlation self-energies and quasiparticle energies. */
struct G0W0QpeResult
{
    std::vector<std::complex<double>> sigc;  ///< Correlation self-energies.
    std::vector<double> eqp;                 ///< Quasiparticle energies.
};

/** @brief G0W0 spectral functions and continued correlation self-energies. */
struct G0W0SpectralFunctionResult
{
    std::vector<std::complex<double>> sigc;  ///< Continued correlation self-energies.
    std::vector<double> spectral_function;   ///< Spectral-function values.
};


/* Options */
/**
 * @brief C++ wrapper for runtime options.
 *
 * Provides a convenient C++ interface to LibrpaOptions with constructor
 * initialization and setter methods.
 *
 * @note Straightforward inheritance. DO NOT add extra member variables here,
 *       which will break the inherited data layout. New control options
 *       should be put under the LibrpaOptions C structure.
 */
class Options : public ::LibrpaOptions
{
public:
    /** @brief Construct with default options. */
    Options() { ::librpa_init_options(this); }

    /**
     * @brief Set output directory.
     *
     * @param[in] output_dir   Path for output data files.
     */
    void set_output_dir(const char *output_dir) { ::librpa_set_output_dir(this, output_dir); }

    /**
     * @brief Set directory to read restart checkpoint files from.
     *
     * @param[in] restart_from_dir   Path containing restart checkpoint files.
     */
    void set_restart_from_dir(const char *restart_from_dir)
    {
        ::librpa_set_restart_from_dir(this, restart_from_dir);
    }
};

/** Continue with explicit method and penalty options; otherwise the same contract. */
CrpaContinuedResult continue_crpa(const LibrpaCrpaResult *result,
                                  const std::vector<std::complex<double>> &queries,
                                  const Options &options,
                                  double tolerance_ha = 0.0003674932217565499);

/* Global environment functions */

/**
 * @brief Get build information string.
 * @return C-string containing build information.
 * @see librpa_get_build_info
 */
const char* get_build_info(void);

/**
 * @brief Get major version number.
 * @return Major version (X in X.Y.Z).
 * @see librpa_get_major_version
 */
int get_major_version(void);

/**
 * @brief Get minor version number.
 * @return Minor version (Y in X.Y.Z).
 * @see librpa_get_minor_version
 */
int get_minor_version(void);

/**
 * @brief Get patch version number.
 * @return Patch version (Z in X.Y.Z).
 * @see librpa_get_patch_version
 */
int get_patch_version(void);

/**
 * @brief Initialize the global LibRPA environment.
 *
 * Must be called after MPI_Init() and before any other LibRPA functions.
 *
 * @param[in] switch_redirect_stdout If true, redirect stdout to a file.
 * @param[in] redirect_path          Path for redirected output (default: "stdout").
 * @param[in] switch_process_output  If true, enable per-process output (default: true).
 * @see librpa_init_global
 */
void init_global(Switch switch_redirect_stdout = LIBRPA_SWITCH_OFF, const char *redirect_path = "stdout",
                 Switch switch_process_output = LIBRPA_SWITCH_ON);

/**
 * @brief Set the global LibRPA stdout verbosity.
 * @param[in] output_level Verbosity level.
 * @see librpa_set_output_level
 */
void set_output_level(Verbose output_level);

/**
 * @brief Get the global LibRPA stdout verbosity.
 * @return Current verbosity level.
 * @see librpa_get_output_level
 */
Verbose get_output_level(void);

/**
 * @brief Finalize the global LibRPA environment.
 *
 * Must be called after all LibRPA operations are complete.
 * @see librpa_finalize_global
 */
void finalize_global(void);

/**
 * @brief Run internal self-tests.
 * @see librpa_test
 */
void test(void);

/**
 * @brief Print profiling information.
 * @see librpa_print_profile
 */
void print_profile(void);

/* C++ handler definition, input and compute methods declarations */

/**
 * @brief Main handler class for LibRPA computations.
 *
 * This class wraps the C handler with RAII semantics for convenient
 * resource management. It provides methods to set input data and
 * perform RPA/EXX/G0W0 calculations.
 *
 * Typical usage:
 * @code{.cpp}
 * librpa::init_global();
 *
 * librpa::Handler h(MPI_COMM_WORLD);
 * h.set_scf_dimension(nspins, nkpts, nstates, nbasis);
 * // ... set more input data ...
 *
 * double ec_rpa = h.get_rpa_correlation_energy(opts, ibzk_contrib);
 * h.free();
 *
 * librpa::finalize_global();
 * @endcode
 */
class Handler
{
private:
    LibrpaHandler *h_ = nullptr;
public:
    /** @brief Default constructor (creates null handler). */
    Handler(): h_(nullptr) {};

    /**
     * @brief Construct and initialize handler with MPI communicator.
     * @param[in] comm MPI communicator for parallel computation.
     */
    Handler(MPI_Comm comm);

    /**
     * @brief Get underlying C handler.
     * @return Non-owning pointer to the wrapped C handler.
     */
    const LibrpaHandler *get_c_handler() const { return h_; }

    /** Collective scalar/spinor orbital analysis from existing states and spatial S(k).
     * Returns an owning result; no H solve, file I/O or change to eigenstate/response data.
     */
    OrbitalResult analyze_orbitals(int n_kpoints, int n_aos,
                                   const std::vector<double> &overlap_k_ri,
                                   double residual_tol) const;

    /**
     * @brief Initialize handler with given MPI communicator.
     * @param[in] comm MPI communicator.
     */
    void init(MPI_Comm comm);

    /**
     * @brief Free handler and release resources.
     */
    void free();

    /**
     * @brief Defatul destructor - automatically frees handler if not already freed.
     *
     * @note Rely on free() to explicitly release resources.
     */
    ~Handler();

    /** Create an owning QSGW task; numerical state and checkpoints belong to the library.
     * controls.update_hartree defaults on. When enabled, supply dedicated raw
     * LRI blocks and the bare q=0 metric through LibrpaQsgwInput, leaving GW
     * input unchanged. */
    QsgwTask create_qsgw_task(const Options &opts, const LibrpaQsgwControls &controls) const;

    /* Input (set) functions */

    /**
     * @brief Set mean-field wavefunction dimensions.
     * @param[in] nspins Number of spin channels.
     * @param[in] nkpts Number of loaded SCF k-points.
     * @param[in] nstates Number of electronic states.
     * @param[in] nbasis Number of basis functions.
     * @param[in] nspinor Number of spinor components per wavefunction.
     */
    void set_scf_dimension(int nspins, int nkpts, int nstates, int nbasis, int nspinor = 1);

    /**
     * @brief Set occupation numbers, eigenvalues, and Fermi level.
     * @param[in] nspins Number of spin channels.
     * @param[in] nkpts Number of k-points.
     * @param[in] nstates Number of states.
     * @param[in] wg Occupation numbers.
     * @param[in] ekb Eigenvalues.
     * @param[in] efermi Fermi level.
     */
    void set_wg_ekb_efermi(int nspins, int nkpts, int nstates, const double *wg, const double *ekb,
                           double efermi);

    /**
     * @brief Set wavefunction coefficients from separate real and imaginary arrays.
     * @param[in] ispin Spin index.
     * @param[in] ik K-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc_real Real parts of the coefficients.
     * @param[in] wfc_imag Imaginary parts of the coefficients.
     */
    void set_wfc(int ispin, int ik, int nstates_local, int nbasis_local, const double *wfc_real,
                 const double *wfc_imag);

    /**
     * @brief Set spinor wavefunction coefficients from separate real and imaginary arrays.
     * @param[in] ik K-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc_up_real Real parts of the spin-up coefficients.
     * @param[in] wfc_up_imag Imaginary parts of the spin-up coefficients.
     * @param[in] wfc_dn_real Real parts of the spin-down coefficients.
     * @param[in] wfc_dn_imag Imaginary parts of the spin-down coefficients.
     */
    void set_wfc_spinor(int ik, int nstates_local, int nbasis_local,
                        const double* wfc_up_real, const double* wfc_up_imag,
                        const double* wfc_dn_real, const double* wfc_dn_imag);

    /**
     * @brief Set wavefunction coefficients from a packed complex array.
     * @param[in] ispin Spin index.
     * @param[in] ik K-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc Complex coefficients.
     */
    void set_wfc_packed(int ispin, int ik, int nstates_local, int nbasis_local,
                        const std::complex<double> *wfc);

    /**
     * @brief Set spinor wavefunction coefficients from packed complex arrays.
     * @param[in] ik K-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc_up Spin-up coefficients.
     * @param[in] wfc_dn Spin-down coefficients.
     */
    void set_wfc_spinor_packed(int ik, int nstates_local, int nbasis_local,
                               const std::complex<double> *wfc_up,
                               const std::complex<double> *wfc_dn);

    /**
     * @brief Set the wavefunction atomic-orbital basis.
     * @param[in] nbs_wfc Number of basis functions per atom.
     * @param[in] l_shells Optional angular momenta grouped by atom.
     */
    void set_ao_basis_wfc(const std::vector<size_t> &nbs_wfc,
                          const std::vector<std::vector<int>> &l_shells = {});

    /**
     * @brief Set the auxiliary atomic-orbital basis.
     * @param[in] nbs_aux Number of auxiliary basis functions per atom.
     * @param[in] l_shells Optional angular momenta grouped by atom.
     */
    void set_ao_basis_aux(const std::vector<size_t> &nbs_aux,
                          const std::vector<std::vector<int>> &l_shells = {});

    /**
     * @brief Set the compressed auxiliary atomic-orbital basis.
     * @param[in] nbs_aux_shrink Number of compressed auxiliary functions per atom.
     * @param[in] l_shells Optional angular momenta grouped by atom.
     */
    void set_ao_basis_aux_shrink(const std::vector<size_t> &nbs_aux_shrink,
                                 const std::vector<std::vector<int>> &l_shells = {});

    /**
     * @brief Set basis-convention metadata used by symmetry reductions.
     * @param[in] bloch_phase Bloch-sum phase sign.
     * @param[in] bloch_ratom Atom-position coefficient in the Bloch phase.
     * @param[in] order Angular basis ordering.
     * @param[in] nega_m Real-spherical-harmonic convention for negative m.
     * @param[in] posi_m Real-spherical-harmonic convention for positive m.
     */
    void set_basis_convention(int bloch_phase, int bloch_ratom, LibrpaAngularOrder order,
                              LibrpaRshCoeff nega_m, LibrpaRshCoeff posi_m);

    /**
     * @brief Set real-space symmetry operations.
     * @param[in] n_symops Number of operations.
     * @param[in] row_conv Nonzero for row-fractional rotations.
     * @param[in] rotmats Flattened rotation matrices.
     * @param[in] trans Optional flattened fractional translations.
     */
    void set_symmetry_operations(int n_symops, int row_conv, const int* rotmats,
                                 const double* trans = nullptr);

    /**
     * @brief Set spatial and spin-space symmetry operations for spinor data.
     * @param[in] n_ops Number of operations.
     * @param[in] row_conv Nonzero for row-fractional rotations.
     * @param[in] rotmats Flattened spatial rotation matrices.
     * @param[in] trans Optional flattened fractional translations.
     * @param[in] antiunitary Optional antiunitary flags, one per operation.
     * @param[in] spin_u Optional explicit 2x2 spin matrices, eight doubles per operation.
     * @param[in] spin_action_source Spin action convention (0 identity, 1 explicit, 2 derived).
     * @param[in] grey_group Nonzero for a grey magnetic group.
     */
    void set_symmetry_spin_operations(int n_ops, int row_conv, const int* rotmats,
                                      const double* trans = nullptr,
                                      const int* antiunitary = nullptr,
                                      const double* spin_u = nullptr,
                                      int spin_action_source = 0,
                                      int grey_group = 0);

    /**
     * @brief Set direct and reciprocal lattice vectors.
     * @param[in] lat_mat Direct lattice vectors in Bohr.
     * @param[in] G_mat Reciprocal lattice vectors in Bohr^-1.
     */
    void set_latvec_and_G(const double lat_mat[9], const double G_mat[9]);

    /**
     * @brief Set atom types and Cartesian coordinates.
     * @param[in] types Species indices.
     * @param[in] pos_cart Cartesian coordinates in Bohr.
     */
    void set_atoms(const std::vector<int> &types, const std::vector<double> &pos_cart);

    /**
     * @brief Set loaded SCF k-points from arrays.
     * @param[in] nk1 Grid size along direction 1.
     * @param[in] nk2 Grid size along direction 2.
     * @param[in] nk3 Grid size along direction 3.
     * @param[in] nkpts Number of loaded SCF k-points.
     * @param[in] kvecs Cartesian k-point vectors.
     * @param[in] kweights Optional weights, normalized to sum to one.
     */
    void set_kgrids_kvec(int nk1, int nk2, int nk3, int nkpts,
                         const double *kvecs, const double *kweights = nullptr);

    /**
     * @brief Set loaded SCF k-points from vectors.
     * @param[in] nk1 Grid size along direction 1.
     * @param[in] nk2 Grid size along direction 2.
     * @param[in] nk3 Grid size along direction 3.
     * @param[in] kvecs Cartesian k-point vectors.
     * @param[in] kweights Optional weights, normalized to sum to one.
     */
    void set_kgrids_kvec(int nk1, int nk2, int nk3, const std::vector<double> &kvecs,
                         const std::vector<double> &kweights = {});

    /**
     * @brief Set the mapping from loaded SCF k-points to Coulomb q-points.
     * @param[in] map_q_ks Representative q-point index for each loaded k-point.
     */
    void set_kq_mapping(const std::vector<int> &map_q_ks);

    /** Set explicit full-grid coordinates (Bohr^-1) and 0-based SCF representatives.
     * Local operation without collectives; preserves loaded SCF dimensions.
     */
    void set_irreducible_kgrids_kvec(int nk1, int nk2, int nk3,
                                     const std::vector<double> &kvecs_scf,
                                     const std::vector<double> &kvecs_full,
                                     const std::vector<int> &full_to_scf);

    /**
     * @brief Set local RI coefficients.
     * @param[in] routing Parallel routing strategy.
     * @param[in] I First atom index.
     * @param[in] J Second atom index.
     * @param[in] nbasis_i Basis size on atom I.
     * @param[in] nbasis_j Basis size on atom J.
     * @param[in] naux_mu Auxiliary basis size.
     * @param[in] R Lattice vector.
     * @param[in] Cs_in RI coefficients.
     * @param[in] shrink_aux Nonzero selects the compressed auxiliary basis.
     */
    void set_lri_coeff(LibrpaParallelRouting routing, int I, int J, int nbasis_i, int nbasis_j,
                       int naux_mu, const int R[3], const double *Cs_in,
                       int shrink_aux = 0);

    /**
     * @brief Set bare Coulomb matrix elements in atom-pair format.
     * @param[in] ik K-point index.
     * @param[in] I First atom index.
     * @param[in] J Second atom index.
     * @param[in] naux_mu Row size.
     * @param[in] naux_nu Column size.
     * @param[in] Vq_real_in Real parts of the matrix.
     * @param[in] Vq_imag_in Imaginary parts of the matrix.
     * @param[in] vq_threshold Screening threshold.
     */
    void set_aux_bare_coulomb_k_atom_pair(int ik, int I, int J, int naux_mu, int naux_nu,
                                          const double *Vq_real_in, const double *Vq_imag_in,
                                          double vq_threshold);

    /**
     * @brief Set bare Coulomb matrix elements from a packed complex atom-pair matrix.
     * @param[in] ik K-point index.
     * @param[in] I First atom index.
     * @param[in] J Second atom index.
     * @param[in] naux_mu Row size.
     * @param[in] naux_nu Column size.
     * @param[in] Vq Complex matrix elements.
     * @param[in] vq_threshold Screening threshold.
     */
    void set_aux_bare_coulomb_k_atom_pair_packed(int ik, int I, int J, int naux_mu, int naux_nu,
                                                 const std::complex<double> *Vq,
                                                 double vq_threshold);

    /**
     * @brief Set truncated Coulomb matrix elements in atom-pair format.
     * @param[in] ik K-point index.
     * @param[in] I First atom index.
     * @param[in] J Second atom index.
     * @param[in] naux_mu Row size.
     * @param[in] naux_nu Column size.
     * @param[in] Vq_real_in Real parts of the matrix.
     * @param[in] Vq_imag_in Imaginary parts of the matrix.
     * @param[in] vq_threshold Screening threshold.
     */
    void set_aux_cut_coulomb_k_atom_pair(int ik, int I, int J, int naux_mu, int naux_nu,
                                         const double *Vq_real_in, const double *Vq_imag_in,
                                         double vq_threshold);

    /**
     * @brief Set truncated Coulomb matrix elements from a packed complex atom-pair matrix.
     * @param[in] ik K-point index.
     * @param[in] I First atom index.
     * @param[in] J Second atom index.
     * @param[in] naux_mu Row size.
     * @param[in] naux_nu Column size.
     * @param[in] Vq Complex matrix elements.
     * @param[in] vq_threshold Screening threshold.
     */
    void set_aux_cut_coulomb_k_atom_pair_packed(int ik, int I, int J, int naux_mu, int naux_nu,
                                                const std::complex<double> *Vq,
                                                double vq_threshold);

    /**
     * @brief Set a bare Coulomb matrix block.
     * @param[in] ik K-point index.
     * @param[in] mu_begin First row index.
     * @param[in] mu_end Row end index.
     * @param[in] nu_begin First column index.
     * @param[in] nu_end Column end index.
     * @param[in] Vq_real_in Real parts of the block.
     * @param[in] Vq_imag_in Imaginary parts of the block.
     */
    void set_aux_bare_coulomb_k_2d_block(int ik, int mu_begin, int mu_end, int nu_begin, int nu_end,
                                         const double *Vq_real_in, const double *Vq_imag_in);

    /**
     * @brief Set a bare Coulomb matrix block from packed complex values.
     * @param[in] ik K-point index.
     * @param[in] mu_begin First row index.
     * @param[in] mu_end Row end index.
     * @param[in] nu_begin First column index.
     * @param[in] nu_end Column end index.
     * @param[in] Vq Complex block values.
     */
    void set_aux_bare_coulomb_k_2d_block_packed(int ik, int mu_begin, int mu_end,
                                                int nu_begin, int nu_end,
                                                const std::complex<double> *Vq);

    /**
     * @brief Set a truncated Coulomb matrix block.
     * @param[in] ik K-point index.
     * @param[in] mu_begin First row index.
     * @param[in] mu_end Row end index.
     * @param[in] nu_begin First column index.
     * @param[in] nu_end Column end index.
     * @param[in] Vq_real_in Real parts of the block.
     * @param[in] Vq_imag_in Imaginary parts of the block.
     */
    void set_aux_cut_coulomb_k_2d_block(int ik, int mu_begin, int mu_end, int nu_begin, int nu_end,
                                        const double *Vq_real_in, const double *Vq_imag_in);

    /**
     * @brief Set a truncated Coulomb matrix block from packed complex values.
     * @param[in] ik K-point index.
     * @param[in] mu_begin First row index.
     * @param[in] mu_end Row end index.
     * @param[in] nu_begin First column index.
     * @param[in] nu_end Column end index.
     * @param[in] Vq Complex block values.
     */
    void set_aux_cut_coulomb_k_2d_block_packed(int ik, int mu_begin, int mu_end,
                                               int nu_begin, int nu_end,
                                               const std::complex<double> *Vq);

    /**
     * @brief Set the dielectric function on the imaginary-frequency axis.
     * @param[in] omegas_imag Imaginary-frequency points.
     * @param[in] dielect_func Dielectric-function values.
     */
    void set_dielect_func_imagfreq(const std::vector<double> &omegas_imag,
                                   const std::vector<double> &dielect_func);

    /**
     * @brief Set velocity matrices from separate real and imaginary arrays.
     * @param[in] n_spins Number of spin channels.
     * @param[in] n_kpts Number of k-points.
     * @param[in] n_states Number of states.
     * @param[in] velocity_real Real parts of the matrices.
     * @param[in] velocity_imag Imaginary parts of the matrices.
     */
    void set_velocity_matrix(int n_spins, int n_kpts, int n_states,
                             const double *velocity_real, const double *velocity_imag);

    /**
     * @brief Set velocity matrices from packed complex values.
     * @param[in] n_spins Number of spin channels.
     * @param[in] n_kpts Number of k-points.
     * @param[in] n_states Number of states.
     * @param[in] velocity Complex matrix values.
     */
    void set_velocity_matrix_packed(int n_spins, int n_kpts, int n_states,
                                    const std::complex<double> *velocity);

    //! Collective original-H/S/r velocity in the loaded SCF frame; input on rank zero.
    void prepare_native_velocity(const Options &opts, const LibrpaNativeVelocityInput *root_input,
                                 bool prepare_optical = true);
    //! Noncollective owning snapshot of this rank's numerical optical results.
    OpticalResult get_optical_result() const;

    /**
     * @brief Set k-points for band-structure calculations.
     * @param[in] n_kpts_band Number of band k-points.
     * @param[in] kfrac_band Fractional band k-point coordinates.
     */
    void set_band_kvec(int n_kpts_band, const double *kfrac_band);

    /**
     * @brief Set occupation numbers and eigenvalues for band k-points.
     * @param[in] n_spins Number of spin channels.
     * @param[in] n_kpts_band Number of band k-points.
     * @param[in] n_states Number of states.
     * @param[in] occ Occupation numbers.
     * @param[in] eig Eigenvalues.
     */
    void set_band_occ_eigval(int n_spins, int n_kpts_band, int n_states, const double *occ,
                             const double *eig);

    /**
     * @brief Set band-k wavefunctions from separate real and imaginary arrays.
     * @param[in] ispin Spin index.
     * @param[in] ik_band Band k-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc_real Real parts of the coefficients.
     * @param[in] wfc_imag Imaginary parts of the coefficients.
     */
    void set_wfc_band(int ispin, int ik_band, int nstates_local, int nbasis_local,
                      const double *wfc_real, const double *wfc_imag);

    /**
     * @brief Set spinor band-k wavefunctions from separate real and imaginary arrays.
     * @param[in] ik_band Band k-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc_up_real Real parts of the spin-up coefficients.
     * @param[in] wfc_up_imag Imaginary parts of the spin-up coefficients.
     * @param[in] wfc_dn_real Real parts of the spin-down coefficients.
     * @param[in] wfc_dn_imag Imaginary parts of the spin-down coefficients.
     */
    void set_wfc_band_spinor(int ik_band, int nstates_local, int nbasis_local,
                             const double* wfc_up_real, const double* wfc_up_imag,
                             const double* wfc_dn_real, const double* wfc_dn_imag);

    /**
     * @brief Set band-k wavefunctions from packed complex values.
     * @param[in] ispin Spin index.
     * @param[in] ik_band Band k-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc Complex coefficients.
     */
    void set_wfc_band_packed(int ispin, int ik_band, int nstates_local, int nbasis_local,
                             const std::complex<double> *wfc);

    /**
     * @brief Set spinor band-k wavefunctions from packed complex values.
     * @param[in] ik_band Band k-point index.
     * @param[in] nstates_local Local number of states.
     * @param[in] nbasis_local Local number of basis functions.
     * @param[in] wfc_up Spin-up coefficients.
     * @param[in] wfc_dn Spin-down coefficients.
     */
    void set_wfc_band_spinor_packed(int ik_band, int nstates_local, int nbasis_local,
                                    const std::complex<double> *wfc_up,
                                    const std::complex<double> *wfc_dn);

    /** @brief Reset band structure data. */
    void reset_band_data();

    /* Compute (build/get) functions */

    /**
     * @brief Construct imaginary-frequency grids.
     * @param[in] opts Runtime options.
     * @param[out] omegas Frequency points.
     * @param[out] weights Quadrature weights.
     */
    void get_imaginary_frequency_grids(const Options &opts,
                                       std::vector<double> &omegas, std::vector<double> &weights);

    /** @brief Compute RPA correlation energy.
     *
     * @param[in] opts Runtime options.
     * @param[out] rpa_corr_ibzk_contrib Complex RPA correlation contribution per k-point.
     * @return Total RPA correlation energy.
     */
    double get_rpa_correlation_energy(const Options &opts,
                                      std::vector<std::complex<double>> &rpa_corr_ibzk_contrib);

    /** @brief Build exact-exchange matrix in real space.
     *
     * @param[in] opts  Runtime options.
     * */
    void build_exx(const Options &opts);

    /** @brief Physical spin-summed Hartree AO matrices on selected full-grid k points.
     * Collective; see librpa_compute_hartree_ao for metric/zero-mode requirements.
     * Returns Hartree-unit complex values, row-major [requested k][AO row][AO column].
     */
    std::vector<std::complex<double>> compute_hartree_ao(
        const Options& opts, const std::vector<int>& iks_this, int n_aos);

    /** @brief Get exact-exchange potential for k-grid states.
     *
     * @param[in] opts         Runtime options.
     * @param[in] n_spins      Number of spin channels.
     * @param[in] iks_this     List of k-point indices computed on this process.
     * @param[in] i_state_low  First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @return Exact-exchange potentials for selected states.
     */
    std::vector<double>
    get_exx_pot_kgrid(const Options &opts, const int n_spins, const std::vector<int> &iks_this,
                      int i_state_low, int i_state_high);

    /** @brief Get exact-exchange potential for band k-points.
     *
     * @param[in] opts           Runtime options.
     * @param[in] n_spins        Number of spin channels.
     * @param[in] iks_band_this  List of band k-point indices on this process.
     * @param[in] i_state_low    First state index (inclusive).
     * @param[in] i_state_high   Last state index (exclusive).
     *
     * @return Exact-exchange potentials for band states.
     */
    std::vector<double>
    get_exx_pot_band_k(const Options &opts, const int n_spins, const std::vector<int> &iks_band_this,
                       int i_state_low, int i_state_high);

    /** @brief Build G0W0 self-energy matrix in real space.
     *
     * @param[in] opts  Runtime options.
     * */
    void build_g0w0_sigma(const Options &opts);

    /** Build GW Sigma and U from one full-response build of the current state. */
    CrpaResult build_g0w0_sigma_crpa(const Options &opts, const LibrpaCrpaInput &input);

    /** Compute current-state bare/U/W; the returned snapshot owns its storage. */
    CrpaResult compute_crpa_window(const Options &opts, const LibrpaCrpaInput &input) const;

    /** Compute current-state bare/U without full W or Sigma. */
    CrpaResult compute_crpa_u_window(const Options &opts, const LibrpaCrpaInput &input) const;
    //! Current frames/weights/model without response, W, or Sigma.
    CrpaResult prepare_crpa_model(const Options &opts, const LibrpaCrpaInput &input) const;

    /** @brief Get G0W0 correlation self-energy for k-grid states.
     *
     * @param[in] opts         Runtime options.
     * @param[in] n_spins      Number of spin channels.
     * @param[in] iks_this     List of k-point indices on this process.
     * @param[in] i_state_low  First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @param[in] vxc          XC potential for selected states.
     * @param[in] vexx         Exact-exchange potential for selected states.
     *
     * @return Correlation self-energy for selected states.
     */
    std::vector<std::complex<double>>
    get_g0w0_sigc_kgrid(const Options &opts, const int n_spins, const std::vector<int> &iks_this,
                        int i_state_low, int i_state_high, const std::vector<double> &vxc, const std::vector<double> &vexx);

    /**
     * @brief Get G0W0 correlation self-energies and quasiparticle energies for k-grid states.
     * @param[in] opts Runtime options.
     * @param[in] n_spins Number of spin channels.
     * @param[in] iks_this K-point indices computed on this process.
     * @param[in] i_state_low First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @param[in] vxc XC potential for selected states.
     * @param[in] vexx Exact-exchange potential for selected states.
     * @return Correlation self-energies and quasiparticle energies.
     */
    G0W0QpeResult
    get_g0w0_qpe_kgrid(const Options &opts, const int n_spins, const std::vector<int> &iks_this,
                       int i_state_low, int i_state_high, const std::vector<double> &vxc,
                       const std::vector<double> &vexx);

    /** @brief Get G0W0 spectral functions for k-grid states.
     *
     * The returned data are ordered as [spin][k-point][state][omega].
     * @param[in] opts Runtime options.
     * @param[in] n_spins Number of spin channels.
     * @param[in] iks_this K-point indices computed on this process.
     * @param[in] i_state_low First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @param[in] omegas Real-frequency points.
     * @param[in] vxc XC potential for selected states.
     * @param[in] vexx Exact-exchange potential for selected states.
     * @return Spectral-function values.
     */
    std::vector<double>
    get_g0w0_spectral_function_kgrid(
        const Options &opts, const int n_spins, const std::vector<int> &iks_this,
        int i_state_low, int i_state_high, const std::vector<double> &omegas,
        const std::vector<double> &vxc, const std::vector<double> &vexx);

    /**
     * @brief Get G0W0 spectral functions and continued self-energies for k-grid states.
     * @param[in] opts Runtime options.
     * @param[in] n_spins Number of spin channels.
     * @param[in] iks_this K-point indices computed on this process.
     * @param[in] i_state_low First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @param[in] omegas Real-frequency points.
     * @param[in] vxc XC potential for selected states.
     * @param[in] vexx Exact-exchange potential for selected states.
     * @return Spectral functions and continued correlation self-energies.
     */
    G0W0SpectralFunctionResult
    get_g0w0_spectral_function_with_sigc_kgrid(
        const Options &opts, const int n_spins, const std::vector<int> &iks_this,
        int i_state_low, int i_state_high, const std::vector<double> &omegas,
        const std::vector<double> &vxc, const std::vector<double> &vexx);

    /** @brief Get G0W0 correlation self-energy for band k-points.
     *
     * @param[in] opts            Runtime options.
     * @param[in] n_spins         Number of spin channels.
     * @param[in] iks_band_this   List of band k-point indices on this process.
     * @param[in] i_state_low     First state index (inclusive).
     * @param[in] i_state_high    Last state index (exclusive).
     * @param[in] vxc_band        XC potential for band states.
     * @param[in] vexx_band       Exact-exchange potential for band states.
     *
     * @return Correlation self-energy for band states.
     */
    std::vector<std::complex<double>>
    get_g0w0_sigc_band_k(const Options &opts, const int n_spins, const std::vector<int> &iks_band_this,
                         int i_state_low, int i_state_high, const std::vector<double> &vxc_band, const std::vector<double> &vexx_band);

    /**
     * @brief Get G0W0 correlation self-energies and quasiparticle energies for band-k states.
     * @param[in] opts Runtime options.
     * @param[in] n_spins Number of spin channels.
     * @param[in] iks_band_this Band k-point indices computed on this process.
     * @param[in] i_state_low First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @param[in] vxc_band XC potential for selected band states.
     * @param[in] vexx_band Exact-exchange potential for selected band states.
     * @return Correlation self-energies and quasiparticle energies.
     */
    G0W0QpeResult
    get_g0w0_qpe_band_k(const Options &opts, const int n_spins,
                        const std::vector<int> &iks_band_this, int i_state_low,
                        int i_state_high, const std::vector<double> &vxc_band,
                        const std::vector<double> &vexx_band);

    /** @brief Get G0W0 spectral functions for band k-point states.
     *
     * The returned data are ordered as [spin][band k-point][state][omega].
     * @param[in] opts Runtime options.
     * @param[in] n_spins Number of spin channels.
     * @param[in] iks_band_this Band k-point indices computed on this process.
     * @param[in] i_state_low First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @param[in] omegas Real-frequency points.
     * @param[in] vxc_band XC potential for selected band states.
     * @param[in] vexx_band Exact-exchange potential for selected band states.
     * @return Spectral-function values.
     */
    std::vector<double>
    get_g0w0_spectral_function_band_k(
        const Options &opts, const int n_spins, const std::vector<int> &iks_band_this,
        int i_state_low, int i_state_high, const std::vector<double> &omegas,
        const std::vector<double> &vxc_band, const std::vector<double> &vexx_band);

    /**
     * @brief Get G0W0 spectral functions and continued self-energies for band-k states.
     * @param[in] opts Runtime options.
     * @param[in] n_spins Number of spin channels.
     * @param[in] iks_band_this Band k-point indices computed on this process.
     * @param[in] i_state_low First state index (inclusive).
     * @param[in] i_state_high Last state index (exclusive).
     * @param[in] omegas Real-frequency points.
     * @param[in] vxc_band XC potential for selected band states.
     * @param[in] vexx_band Exact-exchange potential for selected band states.
     * @return Spectral functions and continued correlation self-energies.
     */
    G0W0SpectralFunctionResult
    get_g0w0_spectral_function_with_sigc_band_k(
        const Options &opts, const int n_spins, const std::vector<int> &iks_band_this,
        int i_state_low, int i_state_high, const std::vector<double> &omegas,
        const std::vector<double> &vxc_band, const std::vector<double> &vexx_band);

    /* Utility functions */
};

}
