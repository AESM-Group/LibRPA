!> @file librpa_f03.f90
!> @brief Fortran 2003 binding for LibRPA
!>
!> This module provides Fortran interfaces to LibRPA functionality for
!> performing RPA correlation energy, exact exchange, and G0W0 calculations.
!>
!> ## Usage
!>
!> Typical workflow:
!> @code{.f90}
!> use librpa_f03
!> implicit none
!>
!> type(LibrpaOptions) :: opts
!> type(LibrpaHandler) :: h
!>
!> ! Initialize LibRPA environment
!> call librpa_init_global()
!>
!> ! Initialize options
!> call opts%init()
!> call librpa_set_output_level(LIBRPA_VERBOSE_INFO)
!>
!> ! Create handler
!> call h%init(MPI_COMM_WORLD)
!>
!> ! Set input data
!> call h%set_scf_dimension(nspins, nkpts, nstates, nbasis)
!> ! ... set more input data ...
!>
!> ! Perform calculation
!> Ec = h%get_rpa_correlation_energy(h, opts, nkpts_ibz, contrib_ibzk(:))
!>
!> ! Clean up
!> call h%free()
!> call librpa_finalize_global()
!> @endcode

!> @brief Fortran 2003 module for LibRPA API
module librpa_f03
   implicit none

   private

   !=======================================================================
   ! Public types, constants, and functions
   !=======================================================================
   public :: LibrpaOptions
   public :: LibrpaHandler

   !> @brief Double precision kind for user data.
   !>
   integer, parameter, public :: dp = 8

   !> @brief Maximum length for string parameters.
   integer, parameter, public :: LIBRPA_MAX_STRLEN = 200

   !> @name Verbosity levels
   !> @brief Controls the amount of output during computation.
   !> @{
   integer, parameter, public :: LIBRPA_VERBOSE_DEBUG = 4      !< Debug output
   integer, parameter, public :: LIBRPA_VERBOSE_INFO = 3       !< Informational messages
   integer, parameter, public :: LIBRPA_VERBOSE_WARN = 2       !< Warnings and above
   integer, parameter, public :: LIBRPA_VERBOSE_CRITICAL = 1   !< Critical errors only
   integer, parameter, public :: LIBRPA_VERBOSE_SILENT = 0     !< No output
   !> @}

   !> @brief Undefined or unset value for integer parameters.
   integer, parameter :: LIBRPA_UNSET = -101

   !> @brief Automatic selection value. LibRPA will choose appropriate setting.
   integer, parameter :: LIBRPA_AUTO = -51

   !> @name Parallel routing strategies
   !> @brief Specifies how computation is distributed across MPI processes.
   !> @{
   integer, parameter, public :: LIBRPA_ROUTING_UNSET = LIBRPA_UNSET  !< Unset
   integer, parameter, public :: LIBRPA_ROUTING_AUTO = LIBRPA_AUTO    !< Auto-select
   integer, parameter, public :: LIBRPA_ROUTING_RTAU = 0              !< Real-space tau decomposition
   integer, parameter, public :: LIBRPA_ROUTING_ATOMPAIR = 1          !< Atom-pair parallelization
   integer, parameter, public :: LIBRPA_ROUTING_LIBRI = 2             !< Use LibRI for RI basis
   !> @}

   !> @name Time-frequency grid types
   !> @brief Different grid types for numerical integration.
   !> @{
   integer, parameter, public :: LIBRPA_TFGRID_UNSET = LIBRPA_UNSET     !< Unset
   integer, parameter, public :: LIBRPA_TFGRID_GL = 0                   !< Gauss-Legendre
   integer, parameter, public :: LIBRPA_TFGRID_GCI= 1                   !< Gauss-Chebyshev type I
   integer, parameter, public :: LIBRPA_TFGRID_GCII = 2                 !< Gauss-Chebyshev type II
   integer, parameter, public :: LIBRPA_TFGRID_MINIMAX = 3              !< Minimax grid
   integer, parameter, public :: LIBRPA_TFGRID_EVENSPACED = 4           !< Evenly spaced
   integer, parameter, public :: LIBRPA_TFGRID_EVENSPACED_TF = 5        !< Evenly spaced in time-frequency
   !> @}

   !> @name Angular basis ordering conventions
   !> @brief Ordering of real spherical harmonics inside one angular-momentum shell.
   !> @{
   integer, parameter, public :: LIBRPA_ANGULAR_ORDER_UNSET = LIBRPA_UNSET  !< Unknown or not specified
   integer, parameter, public :: LIBRPA_ANGULAR_ORDER_NATURAL = 0           !< -l, -l+1, ..., l-1, l
   integer, parameter, public :: LIBRPA_ANGULAR_ORDER_ABS_PM = 1            !< 0, 1, -1, 2, -2, ...
   integer, parameter, public :: LIBRPA_ANGULAR_ORDER_OPENMX = 2            !< OpenMX ordering
   integer, parameter, public :: LIBRPA_ANGULAR_ORDER_PYSCF = 3             !< PySCF ordering
   !> @}

   !> @name Real spherical harmonic coefficient conventions
   !> @brief Coefficient-pair conventions for nonzero real spherical harmonic branches.
   !> @{
   integer, parameter, public :: LIBRPA_RSH_COEFF_UNSET = LIBRPA_UNSET  !< Unknown or not specified
   integer, parameter, public :: LIBRPA_RSH_COEFF_1_M = 0               !< {1, (-1)^m}
   integer, parameter, public :: LIBRPA_RSH_COEFF_M_1 = 1               !< {(-1)^m, 1}
   !> @}

   public :: librpa_init_global
   public :: librpa_finalize_global
   public :: librpa_set_output_level
   public :: librpa_get_output_level
   public :: librpa_get_major_version
   public :: librpa_get_minor_version
   public :: librpa_get_patch_version
   public :: librpa_test
   public :: librpa_print_profile

   !> @brief High-level Fortran wrapper for runtime options.
   !>
   !> This type provides a Fortran-friendly interface to LibRPA options.
   !> Initialize with init() method or call librpa_init_options() before use.
   !> For full defaults and support status, see the runtime parameters guide.
   !>
   !> @note Keep these members synchronized with the C LibrpaOptions struct.
   type :: LibrpaOptions
      !> Output directory for result files.
      character(len=LIBRPA_MAX_STRLEN) :: output_dir
      !> Experimental: directory to read restart checkpoint files from; empty uses output_dir.
      character(len=LIBRPA_MAX_STRLEN) :: restart_from_dir
      !> Parallel distribution strategy; use LIBRPA_ROUTING_* constants.
      integer :: parallel_routing
      !> Real-space Coulomb matrix screening threshold.
      real(dp) :: vq_threshold
      !> Experimental: use k-point-parallel distribution of SCF eigenvectors.
      logical :: use_kpara_scf_eigvec
      !> Time-frequency integration grid type; use LIBRPA_TFGRID_* constants.
      integer :: tfgrids_type
      !> Number of frequency integration grid points.
      integer :: nfreq
      !> Minimum frequency for grid generation, in Hartree.
      real(dp) :: tfgrids_freq_min
      !> Frequency interval for even-spaced grids, in Hartree.
      real(dp) :: tfgrids_freq_interval
      !> Maximum frequency for grid generation, in Hartree.
      real(dp) :: tfgrids_freq_max
      !> Minimum time for grid generation, in Hartree^-1.
      real(dp) :: tfgrids_time_min
      !> Time interval for even-spaced grids, in Hartree^-1.
      real(dp) :: tfgrids_time_interval
      !> Experimental: minimum transition energy for minimax grid generation.
      real(dp) :: minimax_emin
      !> Experimental: maximum transition energy for minimax grid generation.
      real(dp) :: minimax_emax
      !> Experimental: regulation parameter for minimax transformation matrix.
      real(dp) :: minimax_regulation
      !> Experimental: use full Coulomb interaction in \f$\varepsilon = 1 - v \chi^0\f$.
      logical :: use_fullcoul_eps
      !> Experimental: use full Coulomb interaction in the exact-exchange operator.
      logical :: use_fullcoul_exx
      !> Experimental: use full Coulomb interaction in \f$W^c = (\varepsilon^{-1} - 1) v\f$.
      logical :: use_fullcoul_wc
      !> Experimental: use the symmetry context in exact-exchange paths.
      logical :: use_symmetry_exx
      !> Experimental: use the symmetry context in GW paths.
      logical :: use_symmetry_gw
      !> Experimental: use the symmetry context in RPA/chi0 paths.
      logical :: use_symmetry_rpa
      !> Experimental: output ABACUS-compatible GW Green's-function data.
      logical :: output_abacus_gw_gf
      !> Experimental: maximum number of bands for response-function construction.
      integer :: n_bands_chi0
      !> Experimental: maximum number of bands for correlation self-energy construction.
      integer :: n_bands_sigc
      !> Lowest bands excluded from response Green's functions at each k-point and spin.
      integer :: n_bands_exclude_chi0
      !> Lowest bands excluded from correlation self-energy Green's functions at each k-point and spin.
      integer :: n_bands_exclude_sigc
      !> BvK remapping option for band interpolation: 0 single nearest image, 1 Wigner-Seitz.
      integer :: option_bvk_remap
      !> Real-space Green's function screening threshold for response function.
      real(dp) :: gf_threshold
      !> Number of first-index atoms per LibRI chi0 collection chunk.
      integer :: libri_chi0_collect_s0_chunk
      !> Maximum estimated local chi0 tensor bytes per LibRI collection chunk.
      !> Use ScaLAPACK to calculate \f$E_\text{c}^{\text{RPA}}\f$.
      logical :: use_scalapack_ecrpa
      !> Experimental: use a compressed auxiliary basis.
      logical :: use_shrink_abfs
      !> Experimental: build response matrices in the compressed auxiliary basis.
      logical :: use_shrink_chi
      !> Number of parameters for analytic continuation.
      integer :: n_params_anacon
      !> Parameters for the source-grid Pade used to resample GW analytic continuation.
      integer :: n_params_anacon_resample
      !> Optional GW analytic-continuation frequency grid type; unset uses original grid.
      integer :: anacon_tfgrids_type
      !> Optional number of points for GW analytic-continuation frequency grid.
      integer :: anacon_nfreq
      !> Quasi-particle equation solver: 0 damped residual-mixing, 1 quasi-Newton, 2 perturbative.
      integer :: option_qpe_solver
      !> Convergence threshold for the quasi-particle equation solver, in Hartree.
      real(dp) :: qpe_solver_thres
      !> Maximum number of iterations for the quasi-particle equation solver; must be positive.
      integer :: qpe_solver_n_iter_max
      !> Damping factor for quasi-particle equation solver updates.
      !> Used as the initial and maximum factor when adaptive damping is enabled.
      real(dp) :: qpe_solver_damp_factor
      !> Adapt the quasi-particle equation damping factor during the solve.
      logical :: use_qpe_adaptive_damp
      !> Test-only: recover legacy non-adaptive update for QPE solver 0.
      !> Ignored when adaptive damping is enabled.
      logical :: use_qpe_legacy_update
      !> Keep the final unconverged QPE iterate instead of outputting NaN.
      logical :: override_qpe_solver_nan
      !> Use Hedin's poor-man's self-consistency
      logical :: use_hedin_shift
      !> Absolute zero-based reference state index for Hedin shifts; negative means per-state shift.
      integer :: istate_ref_hedin_shift
      !> Broadening/shift used for Green's function in spectral-function output, in Hartree.
      real(dp) :: sf_gf_omega_shift
      !> Broadening/shift used for correlation self-energy in spectral-function output, in Hartree.
      real(dp) :: sf_sigc_omega_shift
      !> Use ScaLAPACK for computing \f$W^c\f$ from \f$\chi^0\f$.
      logical :: use_scalapack_gw_wc
      !> Experimental: use Cholesky factorization for computing \f$W^c\f$ from \f$\chi^0\f$.
      logical :: use_cholesky_gw_wc
      !> Experimental: use GPU to replace scalapack for calculation
      logical :: use_gpu_replace_scalapack
      !> Experimental: use elpa for sqrt coulomb matrix
      logical :: use_elpa_sqrt_coulomb
      !> Experimental: replace dielectric matrix head by the macroscopic dielectric function.
      logical :: replace_w_head
      !> Experimental: dielectric-function handling on the imaginary axis.
      integer :: option_dielect_func
      !> Experimental: use the 2D dielectric-function branch where supported.
      logical :: use_2d_dielectric
      !> First regular Coulomb-eigenbasis channel used by RPA head/wing correction.
      integer :: rpa_headwing_body_start
      !> Experimental: read NAO correlation self-energy matrix in real-space/frequency form.
      logical :: read_sigc_mat_rf
      !> RPA Gamma correction mode: "qavg" or "head_only".
      character(len=LIBRPA_MAX_STRLEN) :: rpa_headwing_mode
      !> Threshold for eigenvalues when taking the square root of Coulomb matrices.
      real(dp) :: sqrt_coulomb_threshold
      !> LibRI threshold of LRI triple coefficients for response function.
      real(dp) :: libri_chi0_threshold_C
      !> LibRI threshold of Green's function for response function.
      real(dp) :: libri_chi0_threshold_G
      !> LibRI threshold of LRI triple coefficients for exact exchange.
      real(dp) :: libri_exx_threshold_C
      !> LibRI threshold of density matrices for exact exchange.
      real(dp) :: libri_exx_threshold_D
      !> LibRI threshold of Coulomb matrices for exact exchange.
      real(dp) :: libri_exx_threshold_V
      !> LibRI threshold of LRI triple coefficients for G0W0 correlation self-energy.
      real(dp) :: libri_g0w0_threshold_C
      !> LibRI threshold of Green's function for G0W0 correlation self-energy.
      real(dp) :: libri_g0w0_threshold_G
      !> LibRI threshold of screened Coulomb matrix for G0W0 correlation self-energy.
      real(dp) :: libri_g0w0_threshold_Wc
      !> Output KS-diagonal correlation self-energy in k-space, imaginary frequency domain.
      logical :: output_gw_sigc_ks_kf
      !> Experimental: output KS-basis correlation self-energy matrix in k-space and imaginary frequencies.
      logical :: output_gw_sigc_ks_mat_kf
      !> Experimental: output the exact-exchange matrix in the KS basis and k-space.
      logical :: output_exx_ks_mat_k
      !> Experimental: output the NAO-basis exact-exchange matrix in k-space as dense binary files.
      logical :: output_exx_mat_k
      !> First zero-based KS state included when exporting KS-basis matrices.
      integer :: istate_output_mat_start
      !> Half-open KS-basis matrix export end index; negative means all remaining states.
      integer :: istate_output_mat_end
      !> Experimental: output dense binary NAO-basis correlation self-energy matrices in k-space and imaginary frequencies.
      logical :: output_gw_sigc_mat_kf
      !> Experimental: output NAO-basis correlation self-energy matrix in real space and imaginary time.
      logical :: output_gw_sigc_mat_rt
      !> Experimental: output NAO-basis correlation self-energy matrix in real space and imaginary frequencies.
      logical :: output_gw_sigc_mat_rf
      !> Experimental: output \f$W^c\f$ matrix in real space and imaginary frequency.
      logical :: output_wc_rf
      !> Experimental: output \f$W^c(R,i\omega)\f$ as atom-pair block files.
      logical :: output_wc_rf_atom_pair
      !> Experimental: output strict-2D finite-q head projection diagnostics.
      logical :: output_2d_finite_q_diagnostics
      !> First zero-based \f$W^c\f$ frequency index to output.
      integer :: ifreq_output_wc_start
      !> Half-open \f$W^c\f$ frequency output end index; negative means all remaining frequencies.
      integer :: ifreq_output_wc_end

      contains
         procedure :: init => librpa_init_options
         procedure :: set_output_dir => librpa_set_output_dir
         procedure :: set_restart_from_dir => librpa_set_restart_from_dir
   end type LibrpaOptions

   !> @brief High-level Fortran wrapper for LibRPA handler.
   !>
   !> This type encapsulates the LibRPA handler and provides member procedures
   !> for setting input data and performing calculations.
   !>
   !> Usage:
   !> @code{.f90}
   !> type(LibrpaHandler) :: h
   !> call h%create(MPI_COMM_WORLD)
   !> call h%set_scf_dimension(nspins, nkpts, nstates, nbasis)
   !> ! ... set more input ...
   !> Ec = h%get_rpa_correlation_energy(opts)
   !> call h%destroy()
   !> @endcode
   type :: LibrpaHandler
      contains
         ! Initialization and destruction
         procedure :: init => librpa_create_handler
         procedure :: free => librpa_destroy_handler
         ! Input
         procedure :: set_scf_dimension => librpa_set_scf_dimension
         procedure :: set_wg_ekb_efermi => librpa_set_wg_ekb_efermi
         procedure :: set_wfc => librpa_set_wfc
         procedure :: set_wfc_spinor => librpa_set_wfc_spinor
         procedure :: set_ao_basis_wfc => librpa_set_ao_basis_wfc
         procedure :: set_ao_basis_aux => librpa_set_ao_basis_aux
         procedure :: set_ao_basis_aux_shrink => librpa_set_ao_basis_aux_shrink
         procedure :: set_basis_convention => librpa_set_basis_convention
         procedure :: set_symmetry_operations => librpa_set_symmetry_operations
         procedure :: set_latvec_and_G => librpa_set_latvec_and_G
         procedure :: set_atoms => librpa_set_atoms
         procedure :: set_kgrids_kvec => librpa_set_kgrids_kvec
         procedure :: set_kq_mapping => librpa_set_kq_mapping
         procedure :: set_lri_coeff => librpa_set_lri_coeff
         procedure :: set_aux_bare_coulomb_k_atom_pair => librpa_set_aux_bare_coulomb_k_atom_pair
         procedure :: set_aux_cut_coulomb_k_atom_pair => librpa_set_aux_cut_coulomb_k_atom_pair
         procedure :: set_aux_bare_coulomb_k_2d_block => librpa_set_aux_bare_coulomb_k_2d_block
         procedure :: set_aux_cut_coulomb_k_2d_block => librpa_set_aux_cut_coulomb_k_2d_block
         procedure :: set_dielect_func_imagfreq => librpa_set_dielect_func_imagfreq
         procedure :: set_band_kvec => librpa_set_band_kvec
         procedure :: set_wfc_band => librpa_set_wfc_band
         procedure :: set_wfc_band_spinor => librpa_set_wfc_band_spinor
         procedure :: set_band_occ_eigval => librpa_set_band_occ_eigval
         procedure :: reset_band_data => librpa_reset_band_data
         ! Compute
         procedure :: get_imaginary_frequency_grids => librpa_get_imaginary_frequency_grids
         procedure :: get_rpa_correlation_energy => librpa_get_rpa_correlation_energy
         procedure :: build_exx => librpa_build_exx
         procedure :: get_exx_pot_kgrid => librpa_get_exx_pot_kgrid
         procedure :: get_exx_pot_band_k => librpa_get_exx_pot_band_k
         procedure :: build_g0w0_sigma => librpa_build_g0w0_sigma
         procedure :: get_g0w0_sigc_kgrid => librpa_get_g0w0_sigc_kgrid
         procedure :: get_g0w0_spectral_function_kgrid => librpa_get_g0w0_spectral_function_kgrid
         procedure :: get_g0w0_sigc_band_k => librpa_get_g0w0_sigc_band_k
         procedure :: get_g0w0_spectral_function_band_k => librpa_get_g0w0_spectral_function_band_k
   end type LibrpaHandler

contains

   ! Can be customized by actual hosting code
   subroutine error_on_call(func)
      implicit none
      character(len=*), intent(in) :: func
      character(len=200) :: info_str

      write(info_str,'(1X,A,A,A)') '* You have called a librpa_f03_stub routine: ', &
         trim(func), '. Make sure you have linked LibRPA library.'
      write(*,'(A)') info_str
      stop
   end subroutine error_on_call

   !=======================================================================
   ! Usually no need change things below
   !=======================================================================

   !> @brief Initialize runtime options to default values.
   !>
   !> Sets all options to their default settings. Must be called before
   !> modifying options and passing to computation functions.
   !>
   !> @param[in,out] opts Options structure to initialize.
   subroutine librpa_init_options(opts)
      implicit none
      class(LibrpaOptions), intent(inout) :: opts
      call error_on_call("librpa_init_options")
   end subroutine librpa_init_options

   !> @brief Set the output directory for LibRPA results.
   !>
   !> @param[in,out] opts       Options structure.
   !> @param[in]     output_dir Path to directory for output files.
   subroutine librpa_set_output_dir(opts, output_dir)
      implicit none
      class(LibrpaOptions), intent(inout) :: opts
      character(len=*), intent(in) :: output_dir
      call error_on_call("librpa_set_output_dir")
   end subroutine librpa_set_output_dir

   !> @brief Set the directory to read restart checkpoint files from.
   !>
   !> @param[in,out] opts           Options structure.
   !> @param[in]     restart_from_dir Path to directory containing restart files.
   subroutine librpa_set_restart_from_dir(opts, restart_from_dir)
      implicit none
      class(LibrpaOptions), intent(inout) :: opts
      character(len=*), intent(in) :: restart_from_dir
      call error_on_call("librpa_set_restart_from_dir")
   end subroutine librpa_set_restart_from_dir

   !> @brief Initialize the global computing environment of LibRPA
   !>
   !> It should be called after MPI initialization and before other LibRPA functions.
   !>
   !> @param[in] sw_redirect    Switch of redirecting standard output (default false)
   !> @param[in] redirect_path  Path of redirected output, only used when `sw_redirect` is true
   !> @param[in] sw_process     Switch of writing per-process output (default true)
   subroutine librpa_init_global(sw_redirect, redirect_path, sw_process)
      implicit none
      logical, intent(in), optional :: sw_redirect, sw_process
      character(len=*), intent(in), optional :: redirect_path
      call error_on_call("librpa_init_global")
   end subroutine librpa_init_global

   !> @brief Release all internal data and finalize the global computing environment of LibRPA
   !>
   !> It should be called after all LibRPA operations are finished.
   subroutine librpa_finalize_global()
      implicit none
      call error_on_call("librpa_finalize_global")
   end subroutine librpa_finalize_global

   !> @brief Set global LibRPA stdout verbosity.
   !> @param[in] output_level Verbosity level.
   subroutine librpa_set_output_level(output_level)
      implicit none
      integer, intent(in) :: output_level
      call error_on_call("librpa_set_output_level")
   end subroutine librpa_set_output_level

   !> @brief Get global LibRPA stdout verbosity.
   !> @return Current verbosity level.
   integer function librpa_get_output_level() result(output_level)
      implicit none
      output_level = -1
      call error_on_call("librpa_get_output_level")
   end function librpa_get_output_level

   !> @brief Get major version number.
   !> @return Major version (X in X.Y.Z).
   integer function librpa_get_major_version() result(v)
      implicit none
      v = -1
      call error_on_call("librpa_get_major_version")
   end function librpa_get_major_version

   !> @brief Get minor version number.
   !> @return Minor version (Y in X.Y.Z).
   integer function librpa_get_minor_version() result(v)
      implicit none
      v = -1
      call error_on_call("librpa_get_minor_version")
   end function librpa_get_minor_version

   !> @brief Get patch version number.
   !> @return Patch version (Z in X.Y.Z).
   integer function librpa_get_patch_version() result(v)
      implicit none
      v = -1
      call error_on_call("librpa_get_patch_version")
   end function librpa_get_patch_version

   !> @brief Run internal self-tests.
   !>
   !> Performs basic sanity checks on LibRPA functionality.
   !> Useful for debugging issues.
   subroutine librpa_test()
      implicit none
      call error_on_call("librpa_test")
   end subroutine librpa_test

   !> @brief Print profiling information.
   !>
   !> Outputs timing and memory usage statistics.
   subroutine librpa_print_profile()
      implicit none
      call error_on_call("librpa_print_profile")
   end subroutine librpa_print_profile

   !> @brief Create a new LibRPA handler instance.
   !>
   !> Allocates and initializes a new LibRPA handler associated with the given
   !> MPI communicator.
   !>
   !> @param[in,out] this  Handler to create.
   !> @param[in]     comm  MPI communicator (e.g., MPI_COMM_WORLD).
   subroutine librpa_create_handler(this, comm)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: comm
      call error_on_call("librpa_create_handler")
   end subroutine librpa_create_handler

   !> @brief Destroy a LibRPA handler instance.
   !>
   !> Frees all internal resources associated with the handler.
   !>
   !> @param[in,out] this  Handler to destroy.
   subroutine librpa_destroy_handler(this)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      call error_on_call("librpa_destroy_handler")
   end subroutine librpa_destroy_handler

   !> @brief Set SCF wavefunction dimension.
   !>
   !> @param[in,out] this     Handler.
   !> @param[in]     nspins   Number of spin channels.
   !> @param[in]     nkpts    Number of k-points.
   !> @param[in]     nstates  Number of electronic states.
   !> @param[in]     nbasis   Number of basis functions.
   !> @param[in]     nspinor  Number of spin components per wavefunction (default 1)
   subroutine librpa_set_scf_dimension(this, nspins, nkpts, nstates, nbasis, nspinor)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: nspins, nkpts, nstates, nbasis
      integer, intent(in), optional :: nspinor
      call error_on_call("librpa_set_scf_dimension")
   end subroutine librpa_set_scf_dimension

   !> @brief Set occupation numbers, eigenvalues, and Fermi level.
   !>
   !> @param[in,out] this     Handler.
   !> @param[in]     nspins   Number of spin channels.
   !> @param[in]     nkpts    Number of k-points.
   !> @param[in]     nstates  Number of electronic states.
   !> @param[in]     wg       Occupation numbers (nstates x nkpts x nspins).
   !> @param[in]     ekb      Eigenvalues (nstates x nkpts x nspins).
   !> @param[in]     efermi   Fermi level.
   subroutine librpa_set_wg_ekb_efermi(this, nspins, nkpts, nstates, wg, ekb, efermi)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: nspins, nkpts, nstates
      real(dp), intent(in) :: wg(nstates, nkpts, nspins)
      real(dp), intent(in) :: ekb(nstates, nkpts, nspins)
      real(dp), intent(in) :: efermi
      call error_on_call("librpa_set_wg_ekb_efermi")
   end subroutine librpa_set_wg_ekb_efermi

   !> @brief Set the wave-function expansion coefficients
   !>
   !> @param[in,out] this           Handler.
   !> @param[in]     ispin          Spin index (starting from 1) of the wave function.
   !> @param[in]     ik             (Global) k-point index (starting from 1) of the wave function.
   !> @param[in]     nstates_local  Local dimension (number of states) of the parsed wave function.
   !> @param[in]     nbasis_local   Local dimension (number of basis functions) of the parsed wave function.
   !> @param[in]     wfc_cplx       Complex-valued wave function to parse.
   subroutine librpa_set_wfc(this, ispin, ik, nstates_local, nbasis_local, wfc_cplx)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ispin, ik, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_cplx(nbasis_local, nstates_local)
      call error_on_call("librpa_set_wfc")
   end subroutine librpa_set_wfc

   !> @brief Set the wave-function expansion coefficients, spinor format
   !>
   !> @param[in,out] this           Handler.
   !> @param[in]     ik             (Global) k-point index (starting from 1) of the wave function.
   !> @param[in]     nstates_local  Local dimension (number of states) of the parsed wave function.
   !> @param[in]     nbasis_local   Local dimension (number of basis functions) of the parsed wave function.
   !> @param[in]     wfc_up_cplx    Complex-valued wave function to parse (spin-up component).
   !> @param[in]     wfc_dn_cplx    Complex-valued wave function to parse (spin-down component).
   subroutine librpa_set_wfc_spinor(this, ik, nstates_local, nbasis_local, wfc_up_cplx, wfc_dn_cplx)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_up_cplx(nbasis_local, nstates_local)
      complex(dp), intent(in), target :: wfc_dn_cplx(nbasis_local, nstates_local)
      call error_on_call("librpa_set_wfc_spinor")
   end subroutine librpa_set_wfc_spinor

   !> @brief Set the wave-function atomic basis
   !>
   !> @param[in,out] this     Handler.
   !> @param[in]     natoms   Number of atoms.
   !> @param[in]     nbs_wfc  Number of wave-function basis functions on each atom.
   !> @param[in]     nshells  Optional number of angular shells on each atom.
   !> @param[in]     l_shells Optional concatenated angular momenta, grouped by atom.
   subroutine librpa_set_ao_basis_wfc(this, natoms, nbs_wfc, nshells, l_shells)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: natoms
      integer, intent(in) :: nbs_wfc(natoms)
      integer, intent(in), optional :: nshells(natoms)
      integer, intent(in), optional :: l_shells(*)
      call error_on_call("librpa_set_ao_basis_wfc")
   end subroutine librpa_set_ao_basis_wfc

   !> @brief Set the auxiliary atomic basis
   !>
   !> @param[in,out] this     Handler.
   !> @param[in]     natoms   Number of atoms.
   !> @param[in]     nbs_aux  Number of auxiliary basis functions on each atom.
   !> @param[in]     nshells  Optional number of angular shells on each atom.
   !> @param[in]     l_shells Optional concatenated angular momenta, grouped by atom.
   subroutine librpa_set_ao_basis_aux(this, natoms, nbs_aux, nshells, l_shells)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: natoms
      integer, intent(in) :: nbs_aux(natoms)
      integer, intent(in), optional :: nshells(natoms)
      integer, intent(in), optional :: l_shells(*)
      call error_on_call("librpa_set_ao_basis_aux")
   end subroutine librpa_set_ao_basis_aux

   !> @brief Set the shrink auxiliary atomic basis
   !>
   !> @param[in,out] this            Handler.
   !> @param[in]     natoms          Number of atoms.
   !> @param[in]     nbs_aux_shrink  Number of shrink auxiliary basis functions on each atom.
   !> @param[in]     nshells         Optional number of angular shells on each atom.
   !> @param[in]     l_shells        Optional concatenated angular momenta, grouped by atom.
   subroutine librpa_set_ao_basis_aux_shrink(this, natoms, nbs_aux_shrink, nshells, l_shells)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: natoms
      integer, intent(in) :: nbs_aux_shrink(natoms)
      integer, intent(in), optional :: nshells(natoms)
      integer, intent(in), optional :: l_shells(*)
      call error_on_call("librpa_set_ao_basis_aux_shrink")
   end subroutine librpa_set_ao_basis_aux_shrink

   !> @brief Set basis convention metadata used by symmetry-based reductions
   !>
   !> @param[in,out] this    Handler.
   !> @param[in]     bloch_phase Bloch-sum phase sign, either +1 or -1.
   !> @param[in]     bloch_ratom Coefficient of atom position in the Bloch-sum phase, one of -1, 0, or +1.
   !> @param[in]     order   Angular basis ordering convention.
   !> @param[in]     nega_m  Real-spherical-harmonic coefficient convention for m < 0.
   !> @param[in]     posi_m  Real-spherical-harmonic coefficient convention for m > 0.
   subroutine librpa_set_basis_convention(this, bloch_phase, bloch_ratom, order, nega_m, posi_m)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: bloch_phase
      integer, intent(in) :: bloch_ratom
      integer, intent(in) :: order
      integer, intent(in) :: nega_m
      integer, intent(in) :: posi_m
      call error_on_call("librpa_set_basis_convention")
   end subroutine librpa_set_basis_convention

   !> @brief Set real-space symmetry operations
   !>
   !> @param[in,out] this       Handler.
   !> @param[in]     n_symops   Number of symmetry operations.
   !> @param[in]     row_conv   True if rotations use the row-fractional convention.
   !> @param[in]     rotmats    Rotation matrices, one 9-element column per operation.
   !> @param[in]     trans      Optional fractional translations, one 3-element column per operation.
   subroutine librpa_set_symmetry_operations(this, n_symops, row_conv, rotmats, trans)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: n_symops
      logical, intent(in) :: row_conv
      integer, dimension(9, n_symops), intent(in) :: rotmats
      real(dp), dimension(3, n_symops), intent(in), optional :: trans
      call error_on_call("librpa_set_symmetry_operations")
   end subroutine librpa_set_symmetry_operations

   !> @brief Set the direct and reciprocal lattice vectors
   !>
   !> Each column is a lattice/reciprocal lattice vector.
   !>
   !> @param[in,out] this      Handler.
   !> @param[in]     latt      Lattice vectors (in Bohr).
   !> @param[in]     recplatt  Reciprocal lattice vectors (in Bohr^-1).
   !>
   subroutine librpa_set_latvec_and_G(this, latt, recplatt)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      real(dp), dimension(3, 3), intent(in) :: latt, recplatt
      call error_on_call("librpa_set_latvec_and_G")
   end subroutine librpa_set_latvec_and_G

   !> @brief Set types and coordinates of the atoms in the model
   !>
   !> @param[in,out] this       Handler.
   !> @param[in]     natoms     Number of atoms.
   !> @param[in]     types      Species type of each atom.
   !> @param[in]     posi_cart  Cartesian coordinates of each atom.
   !>
   subroutine librpa_set_atoms(this, natoms, types, posi_cart)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: natoms
      integer, dimension(natoms), intent(in) :: types
      real(dp), dimension(3, natoms), intent(in) :: posi_cart
      call error_on_call("librpa_set_atoms")
   end subroutine librpa_set_atoms

   !> @brief Set k-point grid vectors.
   !>
   !> @param[in,out] this  Handler.
   !> @param[in]     nk1    Number of k-points along direction 1.
   !> @param[in]     nk2    Number of k-points along direction 2.
   !> @param[in]     nk3    Number of k-points along direction 3.
   !> @param[in]     nkpts  Number of loaded SCF k-points.
   !> @param[in]     kvecs    K-point vectors (3 x nkpts, Cartesian).
   !> @param[in]     kweights Optional k-point weights, normalized internally to sum to one.
   !>
   subroutine librpa_set_kgrids_kvec(this, nk1, nk2, nk3, nkpts, kvecs, kweights)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: nk1, nk2, nk3, nkpts
      real(dp), intent(in) :: kvecs(3, nkpts)
      real(dp), intent(in), optional :: kweights(nkpts)
      call error_on_call("librpa_set_kgrids_kvec")
   end subroutine librpa_set_kgrids_kvec

   !> @brief Set the mapping from loaded SCF k-points to Coulomb q-points
   !>
   !> Example: four loaded k-points where the first two and last are Coulomb q-points,
   !>          and the third point maps to the second q-point, then map_q_ks should be (1, 2, 2, 4)
   !>
   !> @param[in,out] this      Handler.
   !> @param[in]     nkpts     Number of loaded SCF k-points.
   !> @param[in]     map_q_ks  Mapping from each loaded SCF k-point to a q-point.
   !>
   subroutine librpa_set_kq_mapping(this, nkpts, map_q_ks)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: nkpts
      integer, dimension(nkpts), intent(in) :: map_q_ks
      call error_on_call("librpa_set_kq_mapping")
   end subroutine librpa_set_kq_mapping

   !> @brief Set the local RI coefficients
   !>
   !> @param[in,out] this     Handler.
   !> @param[in]     routing  Parallel routing, should be one of the `LIBRPA_ROUTING_*` parameters.
   !> @param[in]     i_atom   Index of atom I (starting from 1).
   !> @param[in]     j_atom   Index of atom J (starting from 1).
   !> @param[in]     nao_i    Number of wave-function basis functions on atom I.
   !> @param[in]     nao_j    Number of wave-function basis functions on atom J.
   !> @param[in]     naux_i   Number of auxiliary basis functions on atom I.
   !> @param[in]     r        Index of unit cell in the crystal, with (0,0,0) at the origin.
   !> @param[in]     coeff    Local RI coefficients associated with atom pair I-J, with auxiliary basis on I.
   !> @param[in]     shrink_aux If present and true, parse coefficients to the shrink auxiliary basis.
   !>
   subroutine librpa_set_lri_coeff(this, routing, i_atom, j_atom, nao_i, nao_j, naux_i, r, coeff, shrink_aux)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: routing, i_atom, j_atom, nao_i, nao_j, naux_i
      integer, dimension(3), intent(in) :: r
      real(dp), contiguous, intent(in) :: coeff(:, :, :)
      logical, intent(in), optional :: shrink_aux
      call error_on_call("librpa_set_lri_coeff")
   end subroutine librpa_set_lri_coeff

   !> @brief Set bare Coulomb matrix elements (atom-pair format).
   !>
   !> @param[in,out] this         Handler.
   !> @param[in]     ik           K-point index (1-based).
   !> @param[in]     i_atom       Atom I index (1-based).
   !> @param[in]     j_atom       Atom J index (1-based).
   !> @param[in]     naux_i       Number of aux functions for i.
   !> @param[in]     naux_j       Number of aux functions for j.
   !> @param[in]     vq           Coulomb matrix (naux_i x naux_j, complex).
   !> @param[in]     vq_threshold  Threshold for screening.
   subroutine librpa_set_aux_bare_coulomb_k_atom_pair &
         (this, ik, i_atom, j_atom, naux_i, naux_j, vq, vq_threshold)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik, i_atom, j_atom, naux_i, naux_j
      complex(dp), intent(in) :: vq(naux_i, naux_j)
      real(dp), intent(in) :: vq_threshold
      call error_on_call("librpa_set_aux_bare_coulomb_k_atom_pair")
   end subroutine librpa_set_aux_bare_coulomb_k_atom_pair

   !> @brief Set truncated Coulomb matrix elements (atom-pair format).
   !>
   !> @param[in,out] this         Handler.
   !> @param[in]     ik           K-point index (1-based).
   !> @param[in]     i_atom       Atom I index (1-based).
   !> @param[in]     j_atom       Atom J index (1-based).
   !> @param[in]     naux_i       Number of aux functions for i.
   !> @param[in]     naux_j       Number of aux functions for j.
   !> @param[in]     vq           Coulomb matrix (naux_i x naux_j, complex).
   !> @param[in]     vq_threshold  Threshold for screening.
   subroutine librpa_set_aux_cut_coulomb_k_atom_pair &
         (this, ik, i_atom, j_atom, naux_i, naux_j, vq, vq_threshold)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik, i_atom, j_atom, naux_i, naux_j
      complex(dp), intent(in) :: vq(naux_i, naux_j)
      real(dp), intent(in) :: vq_threshold
      call error_on_call("librpa_set_aux_cut_coulomb_k_atom_pair")
   end subroutine librpa_set_aux_cut_coulomb_k_atom_pair

   !> @brief Set bare Coulomb matrix elements (2D block format).
   !>
   !> @param[in,out] this        Handler.
   !> @param[in]     ik          K-point index (1-based).
   !> @param[in]     mu_begin    Starting mu index (1-based).
   !> @param[in]     mu_end      Ending mu index (inclusive).
   !> @param[in]     nu_begin    Starting nu index (1-based).
   !> @param[in]     nu_end      Ending nu index (inclusive).
   !> @param[in]     vq          Coulomb matrix (complex).
   subroutine librpa_set_aux_bare_coulomb_k_2d_block &
         (this, ik, mu_begin, mu_end, nu_begin, nu_end, vq)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik, mu_begin, mu_end, nu_begin, nu_end
      complex(dp), intent(in) :: vq(mu_end-mu_begin+1, nu_end-nu_begin+1)
      call error_on_call("librpa_set_aux_bare_coulomb_k_2d_block")
   end subroutine librpa_set_aux_bare_coulomb_k_2d_block

   !> @brief Set truncated Coulomb matrix elements (2D block format).
   !>
   !> @param[in,out] this        Handler.
   !> @param[in]     ik          K-point index (1-based).
   !> @param[in]     mu_begin    Starting mu index (1-based).
   !> @param[in]     mu_end      Ending mu index (inclusive).
   !> @param[in]     nu_begin    Starting nu index (1-based).
   !> @param[in]     nu_end      Ending nu index (inclusive).
   !> @param[in]     vq          Coulomb matrix (complex).
   subroutine librpa_set_aux_cut_coulomb_k_2d_block &
         (this, ik, mu_begin, mu_end, nu_begin, nu_end, vq)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik, mu_begin, mu_end, nu_begin, nu_end
      complex(dp), intent(in) :: vq(mu_end-mu_begin+1, nu_end-nu_begin+1)
      call error_on_call("librpa_set_aux_cut_coulomb_k_2d_block")
   end subroutine librpa_set_aux_cut_coulomb_k_2d_block

   !> @brief Set dielectric function on imaginary frequency axis.
   !> @param[in,out] this            Handler.
   !> @param[in]     nfreq           Number of frequency points.
   !> @param[in]     omegas_imag     Imaginary frequency values.
   !> @param[in]     dielect_func    Dielectric function values.
   subroutine librpa_set_dielect_func_imagfreq(this, nfreq, omegas_imag, dielect_func)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: nfreq
      real(dp), dimension(nfreq), intent(in) :: omegas_imag
      real(dp), dimension(nfreq), intent(in) :: dielect_func
      call error_on_call("librpa_set_dielect_func_imagfreq")
   end subroutine librpa_set_dielect_func_imagfreq

   !> @brief Set k-points for band structure calculations.
   !>
   !> @param[in,out] this        Handler.
   !> @param[in]     nkpts_band  Number of band k-points.
   !> @param[in]     kfrac_band  Band k-point coordinates (3 x nkpts_band, fractional).
   subroutine librpa_set_band_kvec(this, nkpts_band, kfrac_band)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: nkpts_band
      real(dp), intent(in) :: kfrac_band(3, nkpts_band)
      call error_on_call("librpa_set_band_kvec")
   end subroutine librpa_set_band_kvec

   !> @brief Set occupation numbers and eigenvalues for band k-points.
   !>
   !> @param[in,out] this         Handler.
   !> @param[in]     nspins       Number of spin channels.
   !> @param[in]     nkpts_band   Number of band k-points.
   !> @param[in]     nstates      Number of states.
   !> @param[in]     occ          Occupation numbers (nstates x nkpts_band x nspins).
   !> @param[in]     eig          Eigenvalues (nstates x nkpts_band x nspins).
   subroutine librpa_set_band_occ_eigval(this, nspins, nkpts_band, nstates, occ, eig)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: nspins, nkpts_band, nstates
      real(dp), intent(in) :: occ(nstates, nkpts_band, nspins)
      real(dp), intent(in) :: eig(nstates, nkpts_band, nspins)
      call error_on_call("librpa_set_band_occ_eigval")
   end subroutine librpa_set_band_occ_eigval

   !> @brief Set the wave-function expansion coefficients for band calculation
   !>
   !> @param[in,out] this           Handler.
   !> @param[in]     ispin          Spin index (starting from 1) of the wave function.
   !> @param[in]     ik_band        (Global) k-point index (starting from 1) of the wave function.
   !> @param[in]     nstates_local  Local dimension (number of states) of the parsed wave function.
   !> @param[in]     nbasis_local   Local dimension (number of basis functions) of the parsed wave function.
   !> @param[in]     wfc_cplx       Complex-valued wave function to parse.
   subroutine librpa_set_wfc_band(this, ispin, ik_band, nstates_local, nbasis_local, wfc_cplx)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ispin, ik_band, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_cplx(nbasis_local, nstates_local)
      call error_on_call("librpa_set_wfc_band")
   end subroutine librpa_set_wfc_band

   !> @brief Set the wave-function expansion coefficients for band calculation, spinor format
   !>
   !> @param[in,out] this           Handler.
   !> @param[in]     ik_band        (Global) k-point index (starting from 1) of the wave function.
   !> @param[in]     nstates_local  Local dimension (number of states) of the parsed wave function.
   !> @param[in]     nbasis_local   Local dimension (number of basis functions) of the parsed wave function.
   !> @param[in]     wfc_up_cplx    Complex-valued wave function to parse (spin-up component).
   !> @param[in]     wfc_dn_cplx    Complex-valued wave function to parse (spin-down component).
   subroutine librpa_set_wfc_band_spinor(this, ik_band, nstates_local, nbasis_local, wfc_up_cplx, wfc_dn_cplx)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik_band, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_up_cplx(nbasis_local, nstates_local)
      complex(dp), intent(in), target :: wfc_dn_cplx(nbasis_local, nstates_local)
      call error_on_call("librpa_set_wfc_band_spinor")
   end subroutine librpa_set_wfc_band_spinor

   !> @brief Reset band structure data.
   !> @param[in,out] this  Handler.
   subroutine librpa_reset_band_data(this)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      call error_on_call("librpa_reset_band_data")
   end subroutine librpa_reset_band_data

   !> @brief Construct and return frequency grids.
   !>
   !> @param[in,out] this    Handler.
   !> @param[in,out] opts    Runtime options.
   !> @param[out]    omegas  Frequency values.
   !>
   !> @param[out]    weights Quadrature weights.
   subroutine librpa_get_imaginary_frequency_grids(this, opts, omegas, weights)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      real(dp), allocatable, intent(inout) :: omegas(:), weights(:)
      call error_on_call("librpa_get_imaginary_frequency_grids")
   end subroutine librpa_get_imaginary_frequency_grids

   !> @brief Compute RPA correlation energy.
   !>
   !> @param[in,out] this          Handler.
   !> @param[in,out] opts          Runtime options.
   !> @param[in]     nkpts_ibz     Number of irreducible k-points.
   !> @param[out]    contrib_ibzk  Complex correlation contribution per k-point.
   !>
   !> @return Total RPA correlation energy.
   real(dp) function librpa_get_rpa_correlation_energy(this, opts, nkpts_ibz, contrib_ibzk) result(e)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      integer, intent(in) :: nkpts_ibz
      complex(dp), dimension(nkpts_ibz), intent(inout) :: contrib_ibzk
      e = 0.0d0
      call error_on_call("librpa_get_rpa_correlation_energy")
   end function librpa_get_rpa_correlation_energy

   !> @brief Build exact-exchange matrix in real space.
   !> @param[in,out] this  Handler.
   !> @param[in,out] opts  Runtime options.
   subroutine librpa_build_exx(this, opts)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      call error_on_call("librpa_build_exx")
   end subroutine librpa_build_exx

   !> @brief Get exact-exchange potential for k-grid states.
   !> @param[in,out] this          Handler.
   !> @param[in,out] opts          Runtime options.
   !> @param[in]     n_spins       Number of spin channels.
   !> @param[in]     n_kpts_this   Number of k-points on this process.
   !> @param[in]     iks_this      List of k-point indices (1-based).
   !> @param[in]     i_state_low   First state index (1-based, inclusive).
   !> @param[in]     i_state_high  Last state index (1-based, inclusive).
   !> @param[out]    vexx          Exact-exchange potentials.
   subroutine librpa_get_exx_pot_kgrid(this, opts, n_spins, n_kpts_this, iks_this, &
                                       i_state_low, i_state_high, vexx)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      integer, contiguous, dimension(:), intent(in) :: iks_this
      integer, intent(in) :: n_spins, n_kpts_this, i_state_low, i_state_high
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(inout) :: vexx
      call error_on_call("librpa_get_exx_pot_kgrid")
   end subroutine librpa_get_exx_pot_kgrid

   !> @brief Get exact-exchange potential for band k-points.
   !> @param[in,out] this              Handler.
   !> @param[in,out] opts              Runtime options.
   !> @param[in]     n_spins           Number of spin channels.
   !> @param[in]     n_kpts_band_this  Number of band k-points on this process.
   !> @param[in]     iks_band_this     List of band k-point indices (1-based).
   !> @param[in]     i_state_low       First state index (1-based, inclusive).
   !> @param[in]     i_state_high      Last state index (1-based, inclusive).
   !> @param[out]    vexx_band         Exact-exchange potentials for band k-points.
   subroutine librpa_get_exx_pot_band_k(this, opts, n_spins, n_kpts_band_this, iks_band_this, &
                                        i_state_low, i_state_high, vexx_band)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      integer, contiguous, dimension(:), intent(in) :: iks_band_this
      integer, intent(in) :: n_spins, n_kpts_band_this, i_state_low, i_state_high
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(inout) :: vexx_band
      call error_on_call("librpa_get_exx_pot_band_k")
   end subroutine librpa_get_exx_pot_band_k

   !> @brief Build G0W0 self-energy matrix in real space.
   !> @param[in,out] this  Handler.
   !> @param[in,out] opts  Runtime options.
   subroutine librpa_build_g0w0_sigma(this, opts)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      call error_on_call("librpa_build_g0w0_sigma")
   end subroutine librpa_build_g0w0_sigma

   !> @brief Get G0W0 correlation self-energy for k-grid states.
   !> @param[in,out] this           Handler.
   !> @param[in,out] opts           Runtime options.
   !> @param[in]     n_spins        Number of spin channels.
   !> @param[in]     n_kpts_this    Number of k-points on this process.
   !> @param[in]     iks_this       List of k-point indices (1-based).
   !> @param[in]     i_state_low    First state index (1-based, inclusive).
   !> @param[in]     i_state_high   Last state index (1-based, inclusive).
   !> @param[in]     vxc            XC potential for selected states.
   !> @param[in]     vexx           Exact-exchange potential for selected states.
   !> @param[out]    sigc           Correlation self-energy (complex).
   subroutine librpa_get_g0w0_sigc_kgrid(this, opts, n_spins, n_kpts_this, iks_this, &
                                         i_state_low, i_state_high, vxc, vexx, sigc)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      integer, contiguous, dimension(:), intent(in) :: iks_this
      integer, intent(in) :: n_spins, n_kpts_this, i_state_low, i_state_high
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(in) :: vxc
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(in) :: vexx
      complex(dp), dimension(i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(inout) :: sigc
      call error_on_call("librpa_get_g0w0_sigc_kgrid")
   end subroutine librpa_get_g0w0_sigc_kgrid

   !> @brief Get G0W0 spectral functions for k-grid states.
   !> @param[in,out] this                   Handler.
   !> @param[in,out] opts                   Runtime options.
   !> @param[in]     n_spins                Number of spin channels.
   !> @param[in]     n_kpts_this            Number of k-points on this process.
   !> @param[in]     iks_this               List of k-point indices (1-based).
   !> @param[in]     i_state_low            First state index (1-based, inclusive).
   !> @param[in]     i_state_high           Last state index (1-based, inclusive).
   !> @param[in]     omegas                 Real-frequency points in Hartree.
   !> @param[in]     vxc                    XC potential for selected states.
   !> @param[in]     vexx                   Exact-exchange potential for selected states.
   !> @param[out]    spectral_function      Spectral function values, ordered as (omega, state, k, spin).
   !> @param[out]    sigc                   Optional continued correlation self-energy, ordered as (omega, state, k, spin).
   subroutine librpa_get_g0w0_spectral_function_kgrid(this, opts, n_spins, n_kpts_this, iks_this, &
                                                      i_state_low, i_state_high, omegas, vxc, vexx, &
                                                      spectral_function, sigc)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      integer, contiguous, dimension(:), intent(in) :: iks_this
      integer, intent(in) :: n_spins, n_kpts_this, i_state_low, i_state_high
      real(dp), contiguous, dimension(:), intent(in) :: omegas
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(in) :: vxc
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(in) :: vexx
      real(dp), dimension(size(omegas), i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(inout) :: spectral_function
      complex(dp), dimension(size(omegas), i_state_high - i_state_low + 1, n_kpts_this, n_spins), intent(inout), optional :: sigc
      call error_on_call("librpa_get_g0w0_spectral_function_kgrid")
   end subroutine librpa_get_g0w0_spectral_function_kgrid

   !> @brief Get G0W0 correlation self-energy for band k-points.
   !> @param[in,out] this              Handler.
   !> @param[in,out] opts              Runtime options.
   !> @param[in]     n_spins           Number of spin channels.
   !> @param[in]     n_kpts_band_this  Number of band k-points on this process.
   !> @param[in]     iks_band_this     List of band k-point indices (1-based).
   !> @param[in]     i_state_low       First state index (1-based, inclusive).
   !> @param[in]     i_state_high      Last state index (1-based, inclusive).
   !> @param[in]     vxc_band          XC potential for band states.
   !> @param[in]     vexx_band         Exact-exchange potential for band states.
   !> @param[out]    sigc_band        Correlation self-energy for band (complex).
   subroutine librpa_get_g0w0_sigc_band_k(this, opts, n_spins, n_kpts_band_this, iks_band_this, &
                                          i_state_low, i_state_high, vxc_band, vexx_band, sigc_band)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      integer, contiguous, dimension(:), intent(in) :: iks_band_this
      integer, intent(in) :: n_spins, n_kpts_band_this, i_state_low, i_state_high
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(in) :: vxc_band
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(in) :: vexx_band
      complex(dp), dimension(i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(inout) :: sigc_band
      call error_on_call("librpa_get_g0w0_sigc_band_k")
   end subroutine librpa_get_g0w0_sigc_band_k

   !> @brief Get G0W0 spectral functions for band k-points.
   !> @param[in,out] this                   Handler.
   !> @param[in,out] opts                   Runtime options.
   !> @param[in]     n_spins                Number of spin channels.
   !> @param[in]     n_kpts_band_this       Number of band k-points on this process.
   !> @param[in]     iks_band_this          List of band k-point indices (1-based).
   !> @param[in]     i_state_low            First state index (1-based, inclusive).
   !> @param[in]     i_state_high           Last state index (1-based, inclusive).
   !> @param[in]     omegas                 Real-frequency points in Hartree.
   !> @param[in]     vxc_band               XC potential for selected band states.
   !> @param[in]     vexx_band              Exact-exchange potential for selected band states.
   !> @param[out]    spectral_function_band Spectral function values, ordered as (omega, state, k, spin).
   !> @param[out]    sigc_band              Optional continued correlation self-energy, ordered as (omega, state, k, spin).
   subroutine librpa_get_g0w0_spectral_function_band_k(this, opts, n_spins, n_kpts_band_this, iks_band_this, &
                                                       i_state_low, i_state_high, omegas, vxc_band, vexx_band, &
                                                       spectral_function_band, sigc_band)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      integer, contiguous, dimension(:), intent(in) :: iks_band_this
      integer, intent(in) :: n_spins, n_kpts_band_this, i_state_low, i_state_high
      real(dp), contiguous, dimension(:), intent(in) :: omegas
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(in) :: vxc_band
      real(dp), dimension(i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(in) :: vexx_band
      real(dp), dimension(size(omegas), i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(inout) :: spectral_function_band
      complex(dp), dimension(size(omegas), i_state_high - i_state_low + 1, n_kpts_band_this, n_spins), intent(inout), optional :: sigc_band
      call error_on_call("librpa_get_g0w0_spectral_function_band_k")
   end subroutine librpa_get_g0w0_spectral_function_band_k

end module librpa_f03
