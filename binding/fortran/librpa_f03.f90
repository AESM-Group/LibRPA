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

   use iso_c_binding, only: c_char, c_ptr, c_int, c_double, c_long_long, c_null_ptr, c_size_t, c_loc
   implicit none

   private

   !=======================================================================
   ! Public types, constants, and functions
   !=======================================================================
   public :: LibrpaOptions
   public :: LibrpaHandler

   !> @brief Double precision kind for user data.
   !>
   !> Control it through -DLIBRPA_FORTRAN_DP at compile time.
   integer, parameter, public :: dp = ${LIBRPA_FORTRAN_DP}

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

   !=======================================================================
   !> @name Switch values
   !> @brief Boolean switch as integer type (compatible with C).
   !> @{
   integer(c_int), parameter :: LIBRPA_SWITCH_OFF = 0  !< Disabled/off
   integer(c_int), parameter :: LIBRPA_SWITCH_ON = 1  !< Enabled/on
   !> @}

   !> @brief Buffer for redirect path string (internal use).
   character(kind=c_char), allocatable, target, save :: redirect_path_buf(:)

   !> @brief Imaginary unit (i = sqrt(-1)).
   complex(dp), parameter :: CIMAG = (0.0d0, 1.0d0)

   !============================================================================
   !> @name Type definitions
   !============================================================================

   !===== C-side options type =====
   !> @brief C-compatible options structure (internal use).
   !>
   !> Must have the same data layout as the struct defined in include/librpa_options.h.
   !> @note Do not use directly; use the Fortran wrapper type LibrpaOptions instead.
   type, bind(c) :: LibrpaOptions_c
      ! Common runtime control
      character(kind=c_char, len=1) :: output_dir(LIBRPA_MAX_STRLEN)
      character(kind=c_char, len=1) :: restart_from_dir(LIBRPA_MAX_STRLEN)
      integer(c_int) :: parallel_routing
      real(c_double) :: vq_threshold
      integer(c_int) :: use_kpara_scf_eigvec
      integer(c_int) :: tfgrids_type
      integer(c_int) :: nfreq
      real(c_double) :: tfgrids_freq_min
      real(c_double) :: tfgrids_freq_interval
      real(c_double) :: tfgrids_freq_max
      real(c_double) :: tfgrids_time_min
      real(c_double) :: tfgrids_time_interval

      real(c_double) :: minimax_emin
      real(c_double) :: minimax_emax
      real(c_double) :: minimax_regulation

      integer(c_int) :: use_fullcoul_exx
      integer(c_int) :: use_fullcoul_eps
      integer(c_int) :: use_fullcoul_wc
      integer(c_int) :: use_symmetry_exx
      integer(c_int) :: use_symmetry_gw
      integer(c_int) :: use_symmetry_rpa
      integer(c_int) :: output_abacus_gw_gf

      integer(c_int) :: n_bands_chi0
      integer(c_int) :: n_bands_sigc
      integer(c_int) :: n_bands_exclude_chi0
      integer(c_int) :: n_bands_exclude_sigc
      integer(c_int) :: option_bvk_remap

      ! RPA specific
      real(c_double) :: gf_threshold
      integer(c_int) :: libri_chi0_collect_s0_chunk
      integer(c_long_long) :: libri_chi0_collect_max_bytes
      integer(c_int) :: use_scalapack_ecrpa

      integer(c_int) :: use_shrink_abfs
      integer(c_int) :: use_shrink_chi

      ! GW specific
      integer(c_int) :: n_params_anacon
      integer(c_int) :: n_params_anacon_resample
      integer(c_int) :: anacon_tfgrids_type
      integer(c_int) :: anacon_nfreq
      integer(c_int) :: option_qpe_solver
      real(c_double) :: qpe_solver_thres
      integer(c_int) :: qpe_solver_n_iter_max
      real(c_double) :: qpe_solver_damp_factor
      integer(c_int) :: use_qpe_adaptive_damp
      integer(c_int) :: use_qpe_legacy_update
      integer(c_int) :: override_qpe_solver_nan
      integer(c_int) :: use_hedin_shift
      integer(c_int) :: istate_ref_hedin_shift
      real(c_double) :: sf_gf_omega_shift
      real(c_double) :: sf_sigc_omega_shift
      integer(c_int) :: use_scalapack_gw_wc
      integer(c_int) :: use_cholesky_gw_wc
      integer(c_int) :: use_gpu_replace_scalapack
      integer(c_int) :: use_elpa_sqrt_coulomb
      integer(c_int) :: replace_w_head
      integer(c_int) :: option_dielect_func
      integer(c_int) :: use_2d_dielectric
      integer(c_int) :: rpa_headwing_body_start
      integer(c_int) :: read_sigc_mat_rf
      character(kind=c_char, len=1) :: rpa_headwing_mode(LIBRPA_MAX_STRLEN)
      real(c_double) :: sqrt_coulomb_threshold
      real(c_double) :: libri_chi0_threshold_C
      real(c_double) :: libri_chi0_threshold_G
      real(c_double) :: libri_exx_threshold_C
      real(c_double) :: libri_exx_threshold_D
      real(c_double) :: libri_exx_threshold_V
      real(c_double) :: libri_g0w0_threshold_C
      real(c_double) :: libri_g0w0_threshold_G
      real(c_double) :: libri_g0w0_threshold_Wc

      ! Output controls
      integer(c_int) :: output_gw_sigc_ks_kf
      integer(c_int) :: output_gw_sigc_ks_mat_kf
      integer(c_int) :: output_exx_ks_mat_k
      integer(c_int) :: output_exx_mat_k
      integer(c_int) :: istate_output_mat_start
      integer(c_int) :: istate_output_mat_end
      integer(c_int) :: output_gw_sigc_mat_kf
      integer(c_int) :: output_gw_sigc_mat_rt
      integer(c_int) :: output_gw_sigc_mat_rf
      integer(c_int) :: output_wc_rf
      integer(c_int) :: output_wc_rf_atom_pair
      integer(c_int) :: ifreq_output_wc_start
      integer(c_int) :: ifreq_output_wc_end
   end type LibrpaOptions_c

   !> @brief High-level Fortran wrapper for runtime options.
   !>
   !> This type provides a Fortran-friendly interface to LibRPA options.
   !> Initialize with init() method or call librpa_init_options() before use.
   !> For full defaults and support status, see the runtime parameters guide.
   !>
   !> @note Keep these members synchronized with the C LibrpaOptions struct.
   type :: LibrpaOptions
      type(LibrpaOptions_c), private :: opts_c

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
      integer(c_long_long) :: libri_chi0_collect_max_bytes
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
      !> -1 keeps all retained positive channels for Sternheimer finite-part Coulomb input.
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
      !> First zero-based \f$W^c\f$ frequency index to output.
      integer :: ifreq_output_wc_start
      !> Half-open \f$W^c\f$ frequency output end index; negative means all remaining frequencies.
      integer :: ifreq_output_wc_end

      contains
         procedure :: init => librpa_init_options
         procedure :: set_output_dir => librpa_set_output_dir
         procedure :: set_restart_from_dir => librpa_set_restart_from_dir
   end type LibrpaOptions

   !> \cond INTERNAL
   interface
      subroutine librpa_init_options_c(opts_c) bind(c, name="librpa_init_options")
         import :: LibrpaOptions_c
         type(LibrpaOptions_c) :: opts_c
      end subroutine librpa_init_options_c

      subroutine librpa_set_output_dir_c(opts_c, output_dir) bind(c, name="librpa_set_output_dir")
         import :: LibrpaOptions_c, c_char
         type(LibrpaOptions_c) :: opts_c
         character(kind=c_char, len=1), dimension(*), intent(in) :: output_dir
      end subroutine librpa_set_output_dir_c
   end interface
   !> \endcond

   integer, parameter :: SYNC_OPTS_C2F = 1
   integer, parameter :: SYNC_OPTS_F2C = -1

   private :: SYNC_OPTS_F2C
   private :: SYNC_OPTS_C2F

   ! Global environment
   interface
      subroutine librpa_init_global_c(f_comm, sw_redirect, path, sw_process) &
            bind(c, name="librpa_init_global_fortran")
         import :: c_int, c_char
         integer(c_int) :: f_comm
         integer(c_int), value :: sw_redirect
         character(kind=c_char), dimension(*), intent(in) :: path
         integer(c_int), value :: sw_process
      end subroutine librpa_init_global_c

      subroutine librpa_finalize_global_c() bind(c, name="librpa_finalize_global")
      end subroutine librpa_finalize_global_c

      subroutine librpa_set_output_level_c(output_level) bind(c, name="librpa_set_output_level")
         import :: c_int
         integer(c_int), value :: output_level
      end subroutine librpa_set_output_level_c

      function librpa_get_output_level_c() bind(c, name="librpa_get_output_level")
         import :: c_int
         integer(c_int) :: librpa_get_output_level_c
      end function librpa_get_output_level_c

      subroutine librpa_test_c() bind(c, name="librpa_test")
      end subroutine librpa_test_c

      subroutine librpa_print_profile_c() bind(c, name="librpa_print_profile")
      end subroutine librpa_print_profile_c
   end interface

   ! Version information
   interface
      function librpa_get_major_version_c() bind(c, name="librpa_get_major_version")
         import :: c_int
         integer(c_int) :: librpa_get_major_version_c
      end function librpa_get_major_version_c

      function librpa_get_minor_version_c() bind(c, name="librpa_get_minor_version")
         import :: c_int
         integer(c_int) :: librpa_get_minor_version_c
      end function librpa_get_minor_version_c

      function librpa_get_patch_version_c() bind(c, name="librpa_get_patch_version")
         import :: c_int
         integer(c_int) :: librpa_get_patch_version_c
      end function librpa_get_patch_version_c
   end interface

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
      type(c_ptr), private :: ptr_c_handle = c_null_ptr
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

   !> \cond INTERNAL
   interface
      function librpa_create_handler_c(f_comm) bind(c, name="librpa_create_handler_fortran")
         import :: c_ptr, c_int
         integer(c_int) :: f_comm
         type(c_ptr) :: librpa_create_handler_c
      end function librpa_create_handler_c

      subroutine librpa_destroy_handler_c(h) bind(c, name="librpa_destroy_handler")
         import :: c_ptr
         type(c_ptr), value :: h
      end subroutine librpa_destroy_handler_c
   end interface
   !> \endcond

   ! Input functions interface
   !> \cond INTERNAL
   interface
      subroutine librpa_set_scf_dimension_c(h, nspins, nkpts, nstates, nbasis, nspinor) &
            bind(c, name="librpa_set_scf_dimension")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: nspins, nkpts, nstates, nbasis, nspinor
      end subroutine librpa_set_scf_dimension_c

      subroutine librpa_set_wg_ekb_efermi_c(h, nspins, nkpts, nstates, wg, ekb, efermi) &
            bind(c, name="librpa_set_wg_ekb_efermi")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: nspins, nkpts, nstates
         real(c_double), dimension(*), intent(in) :: wg
         real(c_double), dimension(*), intent(in) :: ekb
         real(c_double), value :: efermi
      end subroutine librpa_set_wg_ekb_efermi_c

      subroutine librpa_set_wfc_c(h, ispin, ik, nstates_local, nbasis_local, wfc_real, wfc_imag) &
            bind(c, name="librpa_set_wfc")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ispin, ik, nstates_local, nbasis_local
         real(c_double), dimension(*), intent(in) :: wfc_real
         real(c_double), dimension(*), intent(in) :: wfc_imag
      end subroutine librpa_set_wfc_c

      subroutine librpa_set_wfc_spinor_c(h, ik, nstates_local, nbasis_local, wfc_up_real, wfc_up_imag, wfc_dn_real, wfc_dn_imag) &
            bind(c, name="librpa_set_wfc_spinor")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik, nstates_local, nbasis_local
         real(c_double), dimension(*), intent(in) :: wfc_up_real, wfc_up_imag
         real(c_double), dimension(*), intent(in) :: wfc_dn_real, wfc_dn_imag
      end subroutine librpa_set_wfc_spinor_c

      subroutine librpa_set_wfc_packed_c(h, ispin, ik, nstates_local, nbasis_local, wfc) &
            bind(c, name="librpa_set_wfc_packed")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: ispin, ik, nstates_local, nbasis_local
         type(c_ptr), value :: wfc
      end subroutine librpa_set_wfc_packed_c

      subroutine librpa_set_wfc_spinor_packed_c(h, ik, nstates_local, nbasis_local, wfc_up, wfc_dn) &
            bind(c, name="librpa_set_wfc_spinor_packed")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: ik, nstates_local, nbasis_local
         type(c_ptr), value :: wfc_up, wfc_dn
      end subroutine librpa_set_wfc_spinor_packed_c

      subroutine librpa_set_ao_basis_wfc_c(h, natoms, nbs_wfc, nshells, l_shells) &
            bind(c, name="librpa_set_ao_basis_wfc")
         import :: c_ptr, c_int, c_size_t
         type(c_ptr), value :: h
         integer(c_int), value :: natoms
         integer(c_size_t), dimension(*), intent(in) :: nbs_wfc
         type(c_ptr), value :: nshells, l_shells
      end subroutine librpa_set_ao_basis_wfc_c

      subroutine librpa_set_ao_basis_aux_c(h, natoms, nbs_aux, nshells, l_shells) &
            bind(c, name="librpa_set_ao_basis_aux")
         import :: c_ptr, c_int, c_size_t
         type(c_ptr), value :: h
         integer(c_int), value :: natoms
         integer(c_size_t), dimension(*), intent(in) :: nbs_aux
         type(c_ptr), value :: nshells, l_shells
      end subroutine librpa_set_ao_basis_aux_c

      subroutine librpa_set_ao_basis_aux_shrink_c(h, natoms, nbs_aux_shrink, nshells, l_shells) &
            bind(c, name="librpa_set_ao_basis_aux_shrink")
         import :: c_ptr, c_int, c_size_t
         type(c_ptr), value :: h
         integer(c_int), value :: natoms
         integer(c_size_t), dimension(*), intent(in) :: nbs_aux_shrink
         type(c_ptr), value :: nshells, l_shells
      end subroutine librpa_set_ao_basis_aux_shrink_c

      subroutine librpa_set_basis_convention_c(h, bloch_phase, bloch_ratom, order, nega_m, posi_m) &
            bind(c, name="librpa_set_basis_convention")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: bloch_phase, bloch_ratom, order, nega_m, posi_m
      end subroutine librpa_set_basis_convention_c

      subroutine librpa_set_symmetry_operations_c(h, n_symops, row_conv, rotmats, trans) &
            bind(c, name="librpa_set_symmetry_operations")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: n_symops, row_conv
         type(c_ptr), value :: rotmats, trans
      end subroutine librpa_set_symmetry_operations_c

      subroutine librpa_set_latvec_and_G_c(h, latt, recplatt) &
            bind(c, name="librpa_set_latvec_and_G")
         import :: c_ptr, c_double
         type(c_ptr), value :: h
         real(c_double), dimension(9), intent(in) :: latt, recplatt
      end subroutine librpa_set_latvec_and_G_c

      subroutine librpa_set_atoms_c(h, natoms, types, posi_cart) &
            bind(c, name="librpa_set_atoms")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: natoms
         integer(c_int), dimension(*), intent(in) :: types
         real(c_double), dimension(*), intent(in) :: posi_cart
      end subroutine librpa_set_atoms_c

      subroutine librpa_set_kgrids_kvec_c(h, nk1, nk2, nk3, nkpts, kvecs, kweights) &
            bind(c, name="librpa_set_kgrids_kvec")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: nk1, nk2, nk3, nkpts
         real(c_double), dimension(*), intent(in) :: kvecs
         type(c_ptr), value :: kweights
      end subroutine librpa_set_kgrids_kvec_c

      subroutine librpa_set_kq_mapping_c(h, nkpts, map_q_ks) &
            bind(c, name="librpa_set_kq_mapping")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: nkpts
         integer(c_int), dimension(*), intent(in) :: map_q_ks
      end subroutine librpa_set_kq_mapping_c

      subroutine librpa_set_lri_coeff_c(h, routing, i_atom, j_atom, nao_i, nao_j, naux_i, &
                                        r, coeff, shrink_aux) &
            bind(c, name="librpa_set_lri_coeff")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: routing, i_atom, j_atom, nao_i, nao_j, naux_i
         integer(c_int), dimension(3), intent(in) :: r
         real(c_double), dimension(*), intent(in) :: coeff
         integer(c_int), value :: shrink_aux
      end subroutine librpa_set_lri_coeff_c

      subroutine librpa_set_aux_bare_coulomb_k_atom_pair_c &
            (h, ik, i_atom, j_atom, naux_i, naux_j, vq_real, vq_imag, vq_threshold) &
            bind(c, name="librpa_set_aux_bare_coulomb_k_atom_pair")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik, i_atom, j_atom, naux_i, naux_j
         real(c_double), dimension(*), intent(in) :: vq_real, vq_imag
         real(c_double), value :: vq_threshold
      end subroutine librpa_set_aux_bare_coulomb_k_atom_pair_c

      subroutine librpa_set_aux_bare_coulomb_k_atom_pair_packed_c &
            (h, ik, i_atom, j_atom, naux_i, naux_j, vq, vq_threshold) &
            bind(c, name="librpa_set_aux_bare_coulomb_k_atom_pair_packed")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik, i_atom, j_atom, naux_i, naux_j
         type(c_ptr), value :: vq
         real(c_double), value :: vq_threshold
      end subroutine librpa_set_aux_bare_coulomb_k_atom_pair_packed_c

      subroutine librpa_set_aux_cut_coulomb_k_atom_pair_c &
            (h, ik, i_atom, j_atom, naux_i, naux_j, vq_real, vq_imag, vq_threshold) &
            bind(c, name="librpa_set_aux_cut_coulomb_k_atom_pair")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik, i_atom, j_atom, naux_i, naux_j
         real(c_double), dimension(*), intent(in) :: vq_real, vq_imag
         real(c_double), value :: vq_threshold
      end subroutine librpa_set_aux_cut_coulomb_k_atom_pair_c

      subroutine librpa_set_aux_cut_coulomb_k_atom_pair_packed_c &
            (h, ik, i_atom, j_atom, naux_i, naux_j, vq, vq_threshold) &
            bind(c, name="librpa_set_aux_cut_coulomb_k_atom_pair_packed")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik, i_atom, j_atom, naux_i, naux_j
         type(c_ptr), value :: vq
         real(c_double), value :: vq_threshold
      end subroutine librpa_set_aux_cut_coulomb_k_atom_pair_packed_c

      subroutine librpa_set_aux_bare_coulomb_k_2d_block_c &
            (h, ik, mu_begin, mu_end, nu_begin, nu_end, vq_real, vq_imag) &
            bind(c, name="librpa_set_aux_bare_coulomb_k_2d_block")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik, mu_begin, mu_end, nu_begin, nu_end
         real(c_double), dimension(*), intent(in) :: vq_real, vq_imag
      end subroutine librpa_set_aux_bare_coulomb_k_2d_block_c

      subroutine librpa_set_aux_bare_coulomb_k_2d_block_packed_c &
            (h, ik, mu_begin, mu_end, nu_begin, nu_end, vq) &
            bind(c, name="librpa_set_aux_bare_coulomb_k_2d_block_packed")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: ik, mu_begin, mu_end, nu_begin, nu_end
         type(c_ptr), value :: vq
      end subroutine librpa_set_aux_bare_coulomb_k_2d_block_packed_c

      subroutine librpa_set_aux_cut_coulomb_k_2d_block_c &
            (h, ik, mu_begin, mu_end, nu_begin, nu_end, vq_real, vq_imag) &
            bind(c, name="librpa_set_aux_cut_coulomb_k_2d_block")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik, mu_begin, mu_end, nu_begin, nu_end
         real(c_double), dimension(*), intent(in) :: vq_real, vq_imag
      end subroutine librpa_set_aux_cut_coulomb_k_2d_block_c

      subroutine librpa_set_aux_cut_coulomb_k_2d_block_packed_c &
            (h, ik, mu_begin, mu_end, nu_begin, nu_end, vq) &
            bind(c, name="librpa_set_aux_cut_coulomb_k_2d_block_packed")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: ik, mu_begin, mu_end, nu_begin, nu_end
         type(c_ptr), value :: vq
      end subroutine librpa_set_aux_cut_coulomb_k_2d_block_packed_c

      subroutine librpa_set_dielect_func_imagfreq_c(h, nfreq, omegas_imag, dielect_func) &
            bind(c, name="librpa_set_dielect_func_imagfreq")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: nfreq
         real(c_double), dimension(*), intent(in) :: omegas_imag
         real(c_double), dimension(*), intent(in) :: dielect_func
      end subroutine librpa_set_dielect_func_imagfreq_c

      subroutine librpa_set_band_kvec_c(h, nkpts_band, kfrac_band) &
            bind(c, name="librpa_set_band_kvec")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: nkpts_band
         real(c_double), dimension(*), intent(in) :: kfrac_band
      end subroutine librpa_set_band_kvec_c

      subroutine librpa_set_band_occ_eigval_c(h, nspins, nkpts_band, nstates, occ, eig) &
            bind(c, name="librpa_set_band_occ_eigval")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: nspins, nkpts_band, nstates
         real(c_double), dimension(*), intent(in) :: occ
         real(c_double), dimension(*), intent(in) :: eig
      end subroutine librpa_set_band_occ_eigval_c

      subroutine librpa_set_wfc_band_c(h, ispin, ik_band, nstates_local, nbasis_local, wfc_real, wfc_imag) &
            bind(c, name="librpa_set_wfc_band")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ispin, ik_band, nstates_local, nbasis_local
         real(c_double), dimension(*), intent(in) :: wfc_real
         real(c_double), dimension(*), intent(in) :: wfc_imag
      end subroutine librpa_set_wfc_band_c

      subroutine librpa_set_wfc_band_spinor_c(h, ik_band, nstates_local, nbasis_local, wfc_up_real, wfc_up_imag, wfc_dn_real, wfc_dn_imag) &
            bind(c, name="librpa_set_wfc_band_spinor")
         import :: c_ptr, c_int, c_double
         type(c_ptr), value :: h
         integer(c_int), value :: ik_band, nstates_local, nbasis_local
         real(c_double), dimension(*), intent(in) :: wfc_up_real, wfc_up_imag
         real(c_double), dimension(*), intent(in) :: wfc_dn_real, wfc_dn_imag
      end subroutine librpa_set_wfc_band_spinor_c

      subroutine librpa_set_wfc_band_packed_c(h, ispin, ik_band, nstates_local, nbasis_local, wfc) &
            bind(c, name="librpa_set_wfc_band_packed")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: ispin, ik_band, nstates_local, nbasis_local
         type(c_ptr), value :: wfc
      end subroutine librpa_set_wfc_band_packed_c

      subroutine librpa_set_wfc_band_spinor_packed_c(h, ik_band, nstates_local, nbasis_local, wfc_up, wfc_dn) &
            bind(c, name="librpa_set_wfc_band_spinor_packed")
         import :: c_ptr, c_int
         type(c_ptr), value :: h
         integer(c_int), value :: ik_band, nstates_local, nbasis_local
         type(c_ptr), value :: wfc_up, wfc_dn
      end subroutine librpa_set_wfc_band_spinor_packed_c

      subroutine librpa_reset_band_data_c(h) bind(c, name="librpa_reset_band_data")
         import :: c_ptr
         type(c_ptr), value :: h
      end subroutine librpa_reset_band_data_c
   end interface
   !> \endcond

   ! Compute functions interface
   !> \cond INTERNAL
   interface
      subroutine librpa_get_imaginary_frequency_grids_c(h, opts, omegas, weights) &
            bind(c, name="librpa_get_imaginary_frequency_grids")
         import :: LibrpaOptions_c, c_ptr, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         real(c_double), dimension(*), intent(inout) :: omegas, weights
      end subroutine librpa_get_imaginary_frequency_grids_c

      function librpa_get_rpa_correlation_energy_c(h, opts, nkpts_ibz, contrib_ibzk_re, contrib_ibzk_im) &
            bind(c, name="librpa_get_rpa_correlation_energy")
         import :: LibrpaOptions_c, c_ptr, c_int, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         integer(c_int), intent(in), value :: nkpts_ibz
         real(c_double), dimension(*), intent(inout) :: contrib_ibzk_re, contrib_ibzk_im
         real(c_double) :: librpa_get_rpa_correlation_energy_c
      end function librpa_get_rpa_correlation_energy_c

      subroutine librpa_build_exx_c(h, opts) bind(c, name="librpa_build_exx")
         import :: LibrpaOptions_c, c_ptr
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
      end subroutine librpa_build_exx_c

      subroutine librpa_get_exx_pot_kgrid_c(h, opts, n_spins, n_kpts_this, iks_this, &
                                            i_state_low, i_state_high, vexx) &
            bind(c, name="librpa_get_exx_pot_kgrid")
         import :: LibrpaOptions_c, c_ptr, c_int, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         integer(c_int), value :: n_spins, n_kpts_this, i_state_low, i_state_high
         integer(c_int), dimension(*), intent(in) :: iks_this
         real(c_double), dimension(*), intent(inout) :: vexx
      end subroutine librpa_get_exx_pot_kgrid_c

      subroutine librpa_get_exx_pot_band_k_c(h, opts, n_spins, n_kpts_band_this, iks_band_this, &
                                             i_state_low, i_state_high, vexx_band) &
            bind(c, name="librpa_get_exx_pot_band_k")
         import :: LibrpaOptions_c, c_ptr, c_int, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         integer(c_int), value :: n_spins, n_kpts_band_this, i_state_low, i_state_high
         integer(c_int), dimension(*), intent(in) :: iks_band_this
         real(c_double), dimension(*), intent(inout) :: vexx_band
      end subroutine librpa_get_exx_pot_band_k_c

      subroutine librpa_build_g0w0_sigma_c(h, opts) bind(c, name="librpa_build_g0w0_sigma")
         import :: LibrpaOptions_c, c_ptr
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
      end subroutine librpa_build_g0w0_sigma_c

      subroutine librpa_get_g0w0_sigc_kgrid_c(h, opts, n_spins, n_kpts_this, iks_this, &
                                             i_state_low, i_state_high, vxc, vexx, sigc_re, sigc_im) &
            bind(c, name="librpa_get_g0w0_sigc_kgrid")
         import :: LibrpaOptions_c, c_ptr, c_int, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         integer(c_int), value :: n_spins, n_kpts_this, i_state_low, i_state_high
         integer(c_int), dimension(*), intent(in) :: iks_this
         real(c_double), dimension(*), intent(in) :: vxc, vexx
         real(c_double), dimension(*), intent(inout) :: sigc_re, sigc_im
      end subroutine librpa_get_g0w0_sigc_kgrid_c

      subroutine librpa_get_g0w0_spectral_function_kgrid_c(h, opts, n_spins, n_kpts_this, iks_this, &
                                                           i_state_low, i_state_high, n_omegas, omegas, &
                                                           vxc, vexx, spectral_function, sigc) &
            bind(c, name="librpa_get_g0w0_spectral_function_kgrid")
         import :: LibrpaOptions_c, c_ptr, c_int, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         integer(c_int), value :: n_spins, n_kpts_this, i_state_low, i_state_high, n_omegas
         integer(c_int), dimension(*), intent(in) :: iks_this
         real(c_double), dimension(*), intent(in) :: omegas, vxc, vexx
         real(c_double), dimension(*), intent(inout) :: spectral_function
         type(c_ptr), value :: sigc
      end subroutine librpa_get_g0w0_spectral_function_kgrid_c

      subroutine librpa_get_g0w0_sigc_band_k_c(h, opts, n_spins, n_kpts_band_this, iks_band_this, &
                                               i_state_low, i_state_high, vxc_band, vexx_band, sigc_band_re, sigc_band_im) &
            bind(c, name="librpa_get_g0w0_sigc_band_k")
         import :: LibrpaOptions_c, c_ptr, c_int, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         integer(c_int), value :: n_spins, n_kpts_band_this, i_state_low, i_state_high
         integer(c_int), dimension(*), intent(in) :: iks_band_this
         real(c_double), dimension(*), intent(in) :: vxc_band, vexx_band
         real(c_double), dimension(*), intent(inout) :: sigc_band_re, sigc_band_im
      end subroutine librpa_get_g0w0_sigc_band_k_c

      subroutine librpa_get_g0w0_spectral_function_band_k_c(h, opts, n_spins, n_kpts_band_this, iks_band_this, &
                                                            i_state_low, i_state_high, n_omegas, omegas, &
                                                            vxc_band, vexx_band, spectral_function_band, &
                                                            sigc_band) &
            bind(c, name="librpa_get_g0w0_spectral_function_band_k")
         import :: LibrpaOptions_c, c_ptr, c_int, c_double
         type(c_ptr), value :: h
         type(LibrpaOptions_c), intent(in) :: opts
         integer(c_int), value :: n_spins, n_kpts_band_this, i_state_low, i_state_high, n_omegas
         integer(c_int), dimension(*), intent(in) :: iks_band_this
         real(c_double), dimension(*), intent(in) :: omegas, vxc_band, vexx_band
         real(c_double), dimension(*), intent(inout) :: spectral_function_band
         type(c_ptr), value :: sigc_band
      end subroutine librpa_get_g0w0_spectral_function_band_k_c
   end interface
   !> \endcond

   ! Helper to communicate runtime options between C and Fortran types
   !> \cond INTERNAL
   interface sync_opt
      module procedure sync_opt_string
      module procedure sync_opt_switch
      module procedure sync_opt_int
      module procedure sync_opt_long_long
      module procedure sync_opt_dp
   end interface
   !> \endcond

   private :: sync_opt_string
   private :: sync_opt_switch
   private :: sync_opt_int
   private :: sync_opt_long_long
   private :: sync_opt_dp

contains

   ! Synchronize C/C++ and Fortran strings
   subroutine sync_opt_string(f_string, c_string, direction)
      implicit none
      character(len=*), intent(inout) :: f_string
      character(len=1, kind=c_char), dimension(*), intent(inout) :: c_string
      integer, intent(in) :: direction

      if (direction .eq. SYNC_OPTS_C2F) then
         call c_f_string_chars(c_string, f_string)
      else if (direction .eq. SYNC_OPTS_F2C) then
         call f_c_string_chars(f_string, c_string, trim_f=.true.)
      end if
   end subroutine sync_opt_string

   ! Synchronize LibrpaSwitch with Fortran logical
   subroutine sync_opt_switch(f_logical, c_switch, direction)
      implicit none
      integer(kind=c_int), intent(inout) :: c_switch
      logical, intent(inout) :: f_logical
      integer, intent(in) :: direction

      if (direction .eq. SYNC_OPTS_C2F) then
         f_logical = (c_switch .eq. LIBRPA_SWITCH_ON)
      else if (direction .eq. SYNC_OPTS_F2C) then
         c_switch = LIBRPA_SWITCH_OFF
         if (f_logical) c_switch = LIBRPA_SWITCH_ON
      end if
   end subroutine sync_opt_switch

   ! Synchronize C/C++ int with Fortran integer
   subroutine sync_opt_int(f_integer, c_int_value, direction)
      implicit none
      integer, intent(inout) :: f_integer
      integer(kind=c_int), intent(inout) :: c_int_value
      integer, intent(in) :: direction

      if (direction .eq. SYNC_OPTS_C2F) then
         f_integer = int(c_int_value)
      else if (direction .eq. SYNC_OPTS_F2C) then
         c_int_value = int(f_integer, kind=c_int)
      end if
   end subroutine sync_opt_int

   ! Synchronize C/C++ long long with Fortran integer(c_long_long)
   subroutine sync_opt_long_long(f_integer, c_long_long_value, direction)
      implicit none
      integer(kind=c_long_long), intent(inout) :: f_integer
      integer(kind=c_long_long), intent(inout) :: c_long_long_value
      integer, intent(in) :: direction

      if (direction .eq. SYNC_OPTS_C2F) then
         f_integer = c_long_long_value
      else if (direction .eq. SYNC_OPTS_F2C) then
         c_long_long_value = f_integer
      end if
   end subroutine sync_opt_long_long

   ! Synchronize C/C++ and Fortran double precision numbers
   subroutine sync_opt_dp(f_dp, c_double_value, direction)
      implicit none
      real(dp), intent(inout) :: f_dp
      real(c_double), intent(inout) :: c_double_value
      integer, intent(in) :: direction

      if (direction .eq. SYNC_OPTS_C2F) then
         f_dp = real(c_double_value, kind=dp)
      else if (direction .eq. SYNC_OPTS_F2C) then
         c_double_value = real(f_dp, kind=c_double)
      end if
   end subroutine sync_opt_dp

   !> @brief Copy a Fortran character varaible to C char array
   !!
   !> Adapted from https://fortranwiki.org/fortran/show/c_interface_module
   subroutine f_c_string_chars(f_string, c_string, c_string_len, trim_f)
      use iso_c_binding, only: c_null_char
      implicit none

      character(len=*), intent(in) :: f_string
      character(len=1, kind=c_char), dimension(*), intent(out) :: c_string
      ! Max string length, INCLUDING THE TERMINAL NUL
      integer, intent(in), optional :: c_string_len
      logical, intent(in), optional :: trim_f

      integer :: i, strlen

      if (present(trim_f)) then
         if (trim_f) then
            strlen = len(trim(f_string))
         else
            strlen = len(f_string)
         end if
      else
         strlen = len(f_string)
      end if
      ! print*, "strlen ", strlen
      if (present(c_string_len)) then
         if (c_string_len <= 0) return
         strlen = min(strlen, c_string_len - 1)
      end if

      do i = 1, strlen
         c_string(i) = f_string(i:i)
      end do

      c_string(strlen + 1) = c_null_char
   end subroutine f_c_string_chars

   !> @brief Copy a C string, passed as a char-array reference, to a Fortran string.
   !!
   !> copied from https://fortranwiki.org/fortran/show/c_interface_module
   subroutine c_f_string_chars(c_string, f_string)
      use iso_c_binding, only: c_null_char
      implicit none

      character(len=1, kind=c_char), intent(in) :: c_string(*)
      character(len=*), intent(out) :: f_string
      integer :: i
      i=1
      do while (c_string(i) /= c_null_char .and. i <= len(f_string))
         f_string(i:i) = c_string(i)
         i = i + 1
      end do
      if (i < len(f_string)) f_string(i:) = ' '
   end subroutine c_f_string_chars


   !> @brief Convert Fortran logical to C integer as boolean
   subroutine f_c_bool(f_logical, c_bi)
      implicit none
      logical, intent(in) :: f_logical
      integer(kind=c_int), intent(out) :: c_bi

      if (f_logical) then
         c_bi = LIBRPA_SWITCH_ON
      else
         c_bi = LIBRPA_SWITCH_OFF
      endif
   end subroutine f_c_bool

   !> @brief Convert C integer as boolean to Fortran logical
   subroutine c_f_bool(c_bi, f_logical)
      implicit none
      integer(kind=c_int), intent(in) :: c_bi
      logical, intent(out) :: f_logical

      f_logical = (c_bi .eq. LIBRPA_SWITCH_ON)
   end subroutine c_f_bool

   ! Synchronize option values between the Fortran object and the containing C object
   ! Everytime opts_c used through any C interface, its value should be synchronized from opts
   !   call sync_opts(opts, SYNC_OPTS_F2C)
   subroutine sync_opts(opts, direction)
      type(LibrpaOptions), intent(inout) :: opts
      integer, intent(in) :: direction

      if ((direction .ne. SYNC_OPTS_C2F) .and. (direction .ne. SYNC_OPTS_F2C)) then
         stop "internal error - illegal direction value"
      end if

      call sync_opt(opts%output_dir,              opts%opts_c%output_dir,              direction)
      call sync_opt(opts%restart_from_dir,        opts%opts_c%restart_from_dir,        direction)
      call sync_opt(opts%parallel_routing,        opts%opts_c%parallel_routing,        direction)
      call sync_opt(opts%vq_threshold,            opts%opts_c%vq_threshold,            direction)
      call sync_opt(opts%use_kpara_scf_eigvec,    opts%opts_c%use_kpara_scf_eigvec,    direction)
      call sync_opt(opts%tfgrids_type,            opts%opts_c%tfgrids_type,            direction)
      call sync_opt(opts%nfreq,                   opts%opts_c%nfreq,                   direction)
      call sync_opt(opts%tfgrids_freq_min,        opts%opts_c%tfgrids_freq_min,        direction)
      call sync_opt(opts%tfgrids_freq_interval,   opts%opts_c%tfgrids_freq_interval,   direction)
      call sync_opt(opts%tfgrids_freq_max,        opts%opts_c%tfgrids_freq_max,        direction)
      call sync_opt(opts%tfgrids_time_min,        opts%opts_c%tfgrids_time_min,        direction)
      call sync_opt(opts%tfgrids_time_interval,   opts%opts_c%tfgrids_time_interval,   direction)
      call sync_opt(opts%minimax_emin,            opts%opts_c%minimax_emin,            direction)
      call sync_opt(opts%minimax_emax,            opts%opts_c%minimax_emax,            direction)
      call sync_opt(opts%minimax_regulation,      opts%opts_c%minimax_regulation,      direction)
      call sync_opt(opts%use_fullcoul_eps,        opts%opts_c%use_fullcoul_eps,        direction)
      call sync_opt(opts%use_fullcoul_exx,        opts%opts_c%use_fullcoul_exx,        direction)
      call sync_opt(opts%use_fullcoul_wc,         opts%opts_c%use_fullcoul_wc,         direction)
      call sync_opt(opts%use_symmetry_exx, opts%opts_c%use_symmetry_exx, direction)
      call sync_opt(opts%use_symmetry_gw,  opts%opts_c%use_symmetry_gw,  direction)
      call sync_opt(opts%use_symmetry_rpa, opts%opts_c%use_symmetry_rpa, direction)
      call sync_opt(opts%output_abacus_gw_gf,     opts%opts_c%output_abacus_gw_gf,     direction)
      call sync_opt(opts%n_bands_chi0,            opts%opts_c%n_bands_chi0,            direction)
      call sync_opt(opts%n_bands_sigc,            opts%opts_c%n_bands_sigc,            direction)
      call sync_opt(opts%n_bands_exclude_chi0,    opts%opts_c%n_bands_exclude_chi0, direction)
      call sync_opt(opts%n_bands_exclude_sigc,    opts%opts_c%n_bands_exclude_sigc, direction)
      call sync_opt(opts%option_bvk_remap,        opts%opts_c%option_bvk_remap,        direction)
      call sync_opt(opts%gf_threshold,            opts%opts_c%gf_threshold,            direction)
      call sync_opt(opts%libri_chi0_collect_s0_chunk, opts%opts_c%libri_chi0_collect_s0_chunk, direction)
      call sync_opt(opts%libri_chi0_collect_max_bytes, opts%opts_c%libri_chi0_collect_max_bytes, direction)
      call sync_opt(opts%use_scalapack_ecrpa,     opts%opts_c%use_scalapack_ecrpa,     direction)
      call sync_opt(opts%use_shrink_abfs,         opts%opts_c%use_shrink_abfs,         direction)
      call sync_opt(opts%use_shrink_chi,          opts%opts_c%use_shrink_chi,          direction)
      call sync_opt(opts%n_params_anacon,         opts%opts_c%n_params_anacon,         direction)
      call sync_opt(opts%n_params_anacon_resample, opts%opts_c%n_params_anacon_resample, direction)
      call sync_opt(opts%anacon_tfgrids_type,     opts%opts_c%anacon_tfgrids_type,     direction)
      call sync_opt(opts%anacon_nfreq,            opts%opts_c%anacon_nfreq,            direction)
      call sync_opt(opts%option_qpe_solver,       opts%opts_c%option_qpe_solver,       direction)
      call sync_opt(opts%qpe_solver_thres,        opts%opts_c%qpe_solver_thres,        direction)
      call sync_opt(opts%qpe_solver_n_iter_max,   opts%opts_c%qpe_solver_n_iter_max,   direction)
      call sync_opt(opts%qpe_solver_damp_factor,  opts%opts_c%qpe_solver_damp_factor,  direction)
      call sync_opt(opts%use_qpe_adaptive_damp,   opts%opts_c%use_qpe_adaptive_damp,   direction)
      call sync_opt(opts%use_qpe_legacy_update, opts%opts_c%use_qpe_legacy_update, direction)
      call sync_opt(opts%override_qpe_solver_nan, opts%opts_c%override_qpe_solver_nan, direction)
      call sync_opt(opts%use_hedin_shift,         opts%opts_c%use_hedin_shift, direction)
      call sync_opt(opts%istate_ref_hedin_shift,  opts%opts_c%istate_ref_hedin_shift,  direction)
      call sync_opt(opts%sf_gf_omega_shift,       opts%opts_c%sf_gf_omega_shift,       direction)
      call sync_opt(opts%sf_sigc_omega_shift,     opts%opts_c%sf_sigc_omega_shift,     direction)
      call sync_opt(opts%option_dielect_func,     opts%opts_c%option_dielect_func,     direction)
      call sync_opt(opts%use_2d_dielectric,       opts%opts_c%use_2d_dielectric,       direction)
      call sync_opt(opts%rpa_headwing_body_start, opts%opts_c%rpa_headwing_body_start, direction)
      call sync_opt(opts%read_sigc_mat_rf,        opts%opts_c%read_sigc_mat_rf,        direction)
      call sync_opt(opts%rpa_headwing_mode,       opts%opts_c%rpa_headwing_mode,       direction)
      call sync_opt(opts%use_scalapack_gw_wc,     opts%opts_c%use_scalapack_gw_wc,     direction)
      call sync_opt(opts%use_cholesky_gw_wc,      opts%opts_c%use_cholesky_gw_wc,      direction)
      call sync_opt(opts%use_gpu_replace_scalapack, opts%opts_c%use_gpu_replace_scalapack, direction)
      call sync_opt(opts%use_elpa_sqrt_coulomb,   opts%opts_c%use_elpa_sqrt_coulomb,   direction)
      call sync_opt(opts%sqrt_coulomb_threshold,  opts%opts_c%sqrt_coulomb_threshold,  direction)
      call sync_opt(opts%replace_w_head,          opts%opts_c%replace_w_head,          direction)
      call sync_opt(opts%libri_chi0_threshold_C,  opts%opts_c%libri_chi0_threshold_C,  direction)
      call sync_opt(opts%libri_chi0_threshold_G,  opts%opts_c%libri_chi0_threshold_G,  direction)
      call sync_opt(opts%libri_exx_threshold_C,   opts%opts_c%libri_exx_threshold_C,   direction)
      call sync_opt(opts%libri_exx_threshold_D,   opts%opts_c%libri_exx_threshold_D,   direction)
      call sync_opt(opts%libri_exx_threshold_V,   opts%opts_c%libri_exx_threshold_V,   direction)
      call sync_opt(opts%libri_g0w0_threshold_C,  opts%opts_c%libri_g0w0_threshold_C,  direction)
      call sync_opt(opts%libri_g0w0_threshold_G,  opts%opts_c%libri_g0w0_threshold_G,  direction)
      call sync_opt(opts%libri_g0w0_threshold_Wc, opts%opts_c%libri_g0w0_threshold_Wc, direction)
      call sync_opt(opts%output_gw_sigc_ks_kf,    opts%opts_c%output_gw_sigc_ks_kf,    direction)
      call sync_opt(opts%output_gw_sigc_ks_mat_kf, opts%opts_c%output_gw_sigc_ks_mat_kf, direction)
      call sync_opt(opts%output_exx_ks_mat_k,     opts%opts_c%output_exx_ks_mat_k,     direction)
      call sync_opt(opts%output_exx_mat_k,        opts%opts_c%output_exx_mat_k,        direction)
      call sync_opt(opts%istate_output_mat_start, opts%opts_c%istate_output_mat_start, direction)
      call sync_opt(opts%istate_output_mat_end,   opts%opts_c%istate_output_mat_end,   direction)
      call sync_opt(opts%output_gw_sigc_mat_kf,   opts%opts_c%output_gw_sigc_mat_kf,   direction)
      call sync_opt(opts%output_gw_sigc_mat_rt,   opts%opts_c%output_gw_sigc_mat_rt,   direction)
      call sync_opt(opts%output_gw_sigc_mat_rf,   opts%opts_c%output_gw_sigc_mat_rf,   direction)
      call sync_opt(opts%output_wc_rf,            opts%opts_c%output_wc_rf,            direction)
      call sync_opt(opts%output_wc_rf_atom_pair,  opts%opts_c%output_wc_rf_atom_pair,  direction)
      call sync_opt(opts%ifreq_output_wc_start,   opts%opts_c%ifreq_output_wc_start,   direction)
      call sync_opt(opts%ifreq_output_wc_end,     opts%opts_c%ifreq_output_wc_end,     direction)
   end subroutine

   !> @brief Initialize runtime options to default values.
   !>
   !> Sets all options to their default settings. Must be called before
   !> modifying options and passing to computation functions.
   !>
   !> @param[in,out] opts Options structure to initialize.
   subroutine librpa_init_options(opts)
      implicit none
      class(LibrpaOptions), intent(inout) :: opts
      call librpa_init_options_c(opts%opts_c)
      call sync_opts(opts, SYNC_OPTS_C2F)
   end subroutine librpa_init_options

   !> @brief Set the output directory for LibRPA results.
   !>
   !> @param[in,out] opts       Options structure.
   !> @param[in]     output_dir Path to directory for output files.
   subroutine librpa_set_output_dir(opts, output_dir)
      implicit none
      class(LibrpaOptions), intent(inout) :: opts
      character(len=*), intent(in) :: output_dir
      opts%output_dir = trim(output_dir)
      call sync_opt(opts%output_dir, opts%opts_c%output_dir, SYNC_OPTS_F2C)
   end subroutine librpa_set_output_dir

   !> @brief Set the directory to read restart checkpoint files from.
   !>
   !> @param[in,out] opts           Options structure.
   !> @param[in]     restart_from_dir Path to directory containing restart files.
   subroutine librpa_set_restart_from_dir(opts, restart_from_dir)
      implicit none
      class(LibrpaOptions), intent(inout) :: opts
      character(len=*), intent(in) :: restart_from_dir
      opts%restart_from_dir = trim(restart_from_dir)
      call sync_opt(opts%restart_from_dir, opts%opts_c%restart_from_dir, SYNC_OPTS_F2C)
   end subroutine librpa_set_restart_from_dir

   !> @brief Initialize the global computing environment of LibRPA
   !>
   !> It should be called after MPI initialization and before other LibRPA functions.
   !>
   !> @param[in] sw_redirect    Switch of redirecting standard output (default false)
   !> @param[in] redirect_path  Path of redirected output, only used when `sw_redirect` is true
   !> @param[in] sw_process     Switch of writing per-process output (default true)
   subroutine librpa_init_global(sw_redirect, redirect_path, sw_process)
      use iso_c_binding, only: c_null_char
      use mpi, only: MPI_COMM_WORLD
      implicit none

      logical, intent(in), optional :: sw_redirect, sw_process
      character(len=*), intent(in), optional :: redirect_path

      character(len=*), parameter :: def = "stdout"
      integer(c_int) :: s1, s2, f_comm
      character(kind=c_char), allocatable, target :: path_c(:)
      character(len=:), allocatable :: tmp
      integer :: n, i

      s1 = LIBRPA_SWITCH_OFF
      if (present(sw_redirect)) then
         if (sw_redirect) s1 = LIBRPA_SWITCH_ON
      end if

      s2 = LIBRPA_SWITCH_ON
      if (present(sw_process)) then
         if (.not. sw_process) s2 = LIBRPA_SWITCH_OFF
      end if

      if (present(redirect_path)) then
        tmp = trim(redirect_path)
      else
        tmp = trim(def)
      end if

      n = len(tmp)
      if (allocated(redirect_path_buf)) deallocate(redirect_path_buf)
      allocate(redirect_path_buf(n+1))
      do i = 1, n
         redirect_path_buf(i) = tmp(i:i)
      end do
      redirect_path_buf(n+1) = c_null_char

      f_comm = int(MPI_COMM_WORLD, kind=c_int)

      call librpa_init_global_c(f_comm, s1, redirect_path_buf, s2)

      !call librpa_init_global_c(s1, redirect_path_buf, s2)
      !if (allocated(path_c)) deallocate(path_c)
   end subroutine librpa_init_global

   !> @brief Release all internal data and finalize the global computing environment of LibRPA
   !>
   !> It should be called after all LibRPA operations are finished.
   subroutine librpa_finalize_global()
      implicit none
      call librpa_finalize_global_c()
      if (allocated(redirect_path_buf)) deallocate(redirect_path_buf)
   end subroutine librpa_finalize_global

   !> @brief Set global LibRPA stdout verbosity.
   !> @param[in] output_level Verbosity level.
   subroutine librpa_set_output_level(output_level)
      implicit none
      integer, intent(in) :: output_level
      call librpa_set_output_level_c(int(output_level, c_int))
   end subroutine librpa_set_output_level

   !> @brief Get global LibRPA stdout verbosity.
   !> @return Current verbosity level.
   integer function librpa_get_output_level() result(output_level)
      implicit none
      output_level = librpa_get_output_level_c()
   end function librpa_get_output_level

   !> @brief Get major version number.
   !> @return Major version (X in X.Y.Z).
   integer function librpa_get_major_version() result(v)
      implicit none
      v = librpa_get_major_version_c()
   end function librpa_get_major_version

   !> @brief Get minor version number.
   !> @return Minor version (Y in X.Y.Z).
   integer function librpa_get_minor_version() result(v)
      implicit none
      v = librpa_get_minor_version_c()
   end function librpa_get_minor_version

   !> @brief Get patch version number.
   !> @return Patch version (Z in X.Y.Z).
   integer function librpa_get_patch_version() result(v)
      implicit none
      v = librpa_get_patch_version_c()
   end function librpa_get_patch_version

   !> @brief Run internal self-tests.
   !>
   !> Performs basic sanity checks on LibRPA functionality.
   !> Useful for debugging issues.
   subroutine librpa_test()
      implicit none
      call librpa_test_c()
   end subroutine librpa_test

   !> @brief Print profiling information.
   !>
   !> Outputs timing and memory usage statistics.
   subroutine librpa_print_profile()
      implicit none
      call librpa_print_profile_c()
   end subroutine librpa_print_profile

   !> @brief Create a new LibRPA handler instance.
   !>
   !> Allocates and initializes a new LibRPA handler associated with the given
   !> MPI communicator.
   !>
   !> @param[in,out] this  Handler to create.
   !> @param[in]     comm  MPI communicator (e.g., MPI_COMM_WORLD).
   subroutine librpa_create_handler(this, comm)
      use iso_c_binding, only: c_associated
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: comm
      integer(c_int) :: f_comm

      if (c_associated(this%ptr_c_handle)) call this%free()

      f_comm = int(comm, kind=c_int)
      this%ptr_c_handle = librpa_create_handler_c(f_comm)
      ! this%ptr_c_handle = librpa_create_handler_c(mpi_comm_f2c(comm))
      ! this%ptr_c_handle = librpa_create_handler_c(comm)
   end subroutine librpa_create_handler

   !> @brief Destroy a LibRPA handler instance.
   !>
   !> Frees all internal resources associated with the handler.
   !>
   !> @param[in,out] this  Handler to destroy.
   subroutine librpa_destroy_handler(this)
      use iso_c_binding, only: c_associated
      implicit none
      class(LibrpaHandler), intent(inout) :: this

      if (c_associated(this%ptr_c_handle)) then
         call librpa_destroy_handler_c(this%ptr_c_handle)
         this%ptr_c_handle = c_null_ptr
      end if
   end subroutine librpa_destroy_handler

   ! Input functions

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

      integer(c_int) :: nspins_c, nkpts_c, nstates_c, nbasis_c, nspinor_c
      ! integer(c_int) :: st_istate_c, nstates_local_c, st_ibasis_c, nbasis_local_c

      nspins_c = int(nspins, kind=c_int)
      nkpts_c = int(nkpts, kind=c_int)
      nstates_c = int(nstates, kind=c_int)
      nbasis_c = int(nbasis, kind=c_int)
      if (present(nspinor)) then
         nspinor_c = int(nspinor, kind=c_int)
      else
         nspinor_c = 1
      end if
      ! st_istate_c = int(st_istate, kind=c_int) - 1
      ! nstates_local_c = int(nstates_local, kind=c_int)
      ! st_ibasis_c = int(st_ibasis, kind=c_int) - 1
      ! nbasis_local_c = int(nbasis_local, kind=c_int)

      call librpa_set_scf_dimension_c(this%ptr_c_handle, nspins_c, nkpts_c, nstates_c, nbasis_c, nspinor_c)
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

      integer(c_int) :: nspins_c, nkpts_c, nstates_c
      real(c_double), allocatable :: wg_c(:,:,:), ekb_c(:,:,:)

      nspins_c = int(nspins, kind=c_int)
      nkpts_c = int(nkpts, kind=c_int)
      nstates_c = int(nstates, kind=c_int)

      if (dp == c_double) then
         call librpa_set_wg_ekb_efermi_c(this%ptr_c_handle, nspins_c, nkpts_c, nstates_c, wg, ekb, real(efermi, kind=c_double))
      else
         allocate(wg_c(nstates, nkpts, nspins))
         allocate(ekb_c(nstates, nkpts, nspins))
         wg_c = real(wg, kind=c_double)
         ekb_c = real(ekb, kind=c_double)
         call librpa_set_wg_ekb_efermi_c(this%ptr_c_handle, nspins_c, nkpts_c, nstates_c, wg_c, ekb_c, real(efermi, kind=c_double))
         deallocate(wg_c, ekb_c)
      end if
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
      use iso_c_binding, only: c_int, c_double, c_loc
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ispin, ik, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_cplx(nbasis_local, nstates_local)

      real(c_double), allocatable :: wfc_real(:,:), wfc_imag(:,:)
      integer(c_int) :: ispin_c, ik_c, nstates_local_c, nbasis_local_c

      ispin_c = int(ispin-1, kind=c_int)
      ik_c = int(ik-1, kind=c_int)
      nstates_local_c = int(nstates_local, kind=c_int)
      nbasis_local_c = int(nbasis_local, kind=c_int)

      if (dp == c_double) then
         ! Fast path without create intermediate Fortran arrays
         call librpa_set_wfc_packed_c(&
            this%ptr_c_handle, ispin_c, ik_c, &
            nstates_local_c, nbasis_local_c, c_loc(wfc_cplx))
      else
         allocate(wfc_real(nbasis_local, nstates_local))
         allocate(wfc_imag(nbasis_local, nstates_local))
         wfc_real = real(wfc_cplx, kind=c_double)
         wfc_imag = real(aimag(wfc_cplx), kind=c_double)
         call librpa_set_wfc_c(this%ptr_c_handle, ispin_c, ik_c, &
            nstates_local_c, nbasis_local_c, wfc_real, wfc_imag)
         deallocate(wfc_real, wfc_imag)
      end if
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
      use iso_c_binding, only: c_int, c_double, c_loc
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_up_cplx(nbasis_local, nstates_local)
      complex(dp), intent(in), target :: wfc_dn_cplx(nbasis_local, nstates_local)

      real(c_double), allocatable :: wfc_up_real(:,:), wfc_up_imag(:,:)
      real(c_double), allocatable :: wfc_dn_real(:,:), wfc_dn_imag(:,:)
      integer(c_int) :: ik_c, nstates_local_c, nbasis_local_c

      ik_c = int(ik-1, kind=c_int)
      nstates_local_c = int(nstates_local, kind=c_int)
      nbasis_local_c = int(nbasis_local, kind=c_int)

      if (dp == c_double) then
         ! Fast path without create intermediate Fortran arrays
         call librpa_set_wfc_spinor_packed_c(&
            this%ptr_c_handle, ik_c, &
            nstates_local_c, nbasis_local_c, c_loc(wfc_up_cplx), c_loc(wfc_dn_cplx))
      else
         allocate(wfc_up_real(nbasis_local, nstates_local))
         allocate(wfc_up_imag(nbasis_local, nstates_local))
         allocate(wfc_dn_real(nbasis_local, nstates_local))
         allocate(wfc_dn_imag(nbasis_local, nstates_local))
         wfc_up_real = real(wfc_up_cplx, kind=c_double)
         wfc_up_imag = real(aimag(wfc_up_cplx), kind=c_double)
         wfc_dn_real = real(wfc_dn_cplx, kind=c_double)
         wfc_dn_imag = real(aimag(wfc_dn_cplx), kind=c_double)
         call librpa_set_wfc_spinor_c(this%ptr_c_handle, ik_c, &
            nstates_local_c, nbasis_local_c, wfc_up_real, wfc_up_imag, wfc_dn_real, wfc_dn_imag)
         deallocate(wfc_up_real, wfc_up_imag)
         deallocate(wfc_dn_real, wfc_dn_imag)
      end if
   end subroutine librpa_set_wfc_spinor

   subroutine set_ao_basis(h, natoms, nbs, basis_kind, nshells, l_shells)
      implicit none
      type(LibrpaHandler), intent(inout) :: h
      integer, intent(in) :: natoms
      integer, intent(in) :: nbs(natoms)
      integer, intent(in) :: basis_kind
      integer, intent(in), optional :: nshells(natoms)
      integer, intent(in), optional :: l_shells(*)

      integer(c_size_t), allocatable :: nbs_c(:)
      integer(c_int), allocatable, target :: nshells_c(:)
      integer(c_int), allocatable, target :: l_shells_c(:)
      integer(c_int) :: natoms_c
      integer :: ishell, total_shells
      type(c_ptr) :: nshells_ptr, l_shells_ptr

      allocate(nbs_c(natoms))
      nbs_c = int(nbs, kind=c_size_t)
      natoms_c = int(natoms, kind=c_int)
      nshells_ptr = c_null_ptr
      l_shells_ptr = c_null_ptr
      if (present(nshells)) then
         allocate(nshells_c(natoms))
         nshells_c = int(nshells, kind=c_int)
         if (natoms > 0) nshells_ptr = c_loc(nshells_c(1))

         total_shells = sum(nshells)
         if (present(l_shells) .and. total_shells > 0) then
            allocate(l_shells_c(total_shells))
            do ishell = 1, total_shells
               l_shells_c(ishell) = int(l_shells(ishell), kind=c_int)
            end do
            l_shells_ptr = c_loc(l_shells_c(1))
         end if
      end if

      if (basis_kind == 2) then
         call librpa_set_ao_basis_aux_shrink_c(h%ptr_c_handle, natoms_c, nbs_c, nshells_ptr, l_shells_ptr)
      else if (basis_kind == 1) then
         call librpa_set_ao_basis_aux_c(h%ptr_c_handle, natoms_c, nbs_c, nshells_ptr, l_shells_ptr)
      else
         call librpa_set_ao_basis_wfc_c(h%ptr_c_handle, natoms_c, nbs_c, nshells_ptr, l_shells_ptr)
      end if
      if (allocated(l_shells_c)) deallocate(l_shells_c)
      if (allocated(nshells_c)) deallocate(nshells_c)
      deallocate(nbs_c)
   end subroutine set_ao_basis

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

      call set_ao_basis(this, natoms, nbs_wfc, 0, nshells, l_shells)
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

      call set_ao_basis(this, natoms, nbs_aux, 1, nshells, l_shells)
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

      call set_ao_basis(this, natoms, nbs_aux_shrink, 2, nshells, l_shells)
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

      call librpa_set_basis_convention_c(this%ptr_c_handle, &
         int(bloch_phase, kind=c_int), int(bloch_ratom, kind=c_int), int(order, kind=c_int), &
         int(nega_m, kind=c_int), int(posi_m, kind=c_int))
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

      integer(c_int), allocatable, target :: rotmats_c(:, :)
      real(c_double), allocatable, target :: trans_c(:, :)
      integer(c_int) :: n_symops_c, row_conv_c
      type(c_ptr) :: rotmats_ptr, trans_ptr

      n_symops_c = int(n_symops, kind=c_int)
      row_conv_c = 0
      if (row_conv) row_conv_c = 1
      rotmats_ptr = c_null_ptr
      trans_ptr = c_null_ptr

      if (n_symops > 0) then
         allocate(rotmats_c(9, n_symops))
         rotmats_c = int(rotmats, kind=c_int)
         rotmats_ptr = c_loc(rotmats_c(1, 1))
         if (present(trans)) then
            allocate(trans_c(3, n_symops))
            trans_c = real(trans, kind=c_double)
            trans_ptr = c_loc(trans_c(1, 1))
         end if
      end if

      call librpa_set_symmetry_operations_c(this%ptr_c_handle, &
         n_symops_c, row_conv_c, rotmats_ptr, trans_ptr)

      if (allocated(trans_c)) deallocate(trans_c)
      if (allocated(rotmats_c)) deallocate(rotmats_c)
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

      real(c_double) :: latt_c(3,3), recplatt_c(3,3)

      latt_c = real(latt, kind=c_double)
      recplatt_c = real(recplatt, kind=c_double)
      call librpa_set_latvec_and_G_c(this%ptr_c_handle, latt_c, recplatt_c)
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

      integer(c_int) :: natoms_c
      integer(c_int), allocatable :: types_c(:)
      real(c_double), allocatable :: posi_cart_c(:,:)

      allocate(types_c(natoms))

      types_c = int(types, kind=c_int)
      natoms_c = int(natoms, kind=c_int)

      if (dp == c_double) then
         call librpa_set_atoms_c(this%ptr_c_handle, natoms_c, types_c, posi_cart)
      else
         allocate(posi_cart_c(3, natoms))
         posi_cart_c = real(posi_cart, kind=c_double)
         call librpa_set_atoms_c(this%ptr_c_handle, natoms_c, types_c, posi_cart_c)
         deallocate(posi_cart_c)
      end if
      deallocate(types_c)
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

      integer(c_int) :: nk1_c, nk2_c, nk3_c, nkpts_c
      real(c_double), allocatable :: kvecs_c(:,:)
      real(c_double), allocatable, target :: kweights_c(:)
      type(c_ptr) :: kweights_ptr

      nk1_c = int(nk1, kind=c_int)
      nk2_c = int(nk2, kind=c_int)
      nk3_c = int(nk3, kind=c_int)
      nkpts_c = int(nkpts, kind=c_int)
      kweights_ptr = c_null_ptr
      if (present(kweights)) then
         allocate(kweights_c(nkpts))
         kweights_c = real(kweights, kind=c_double)
         kweights_ptr = c_loc(kweights_c)
      end if

      if (dp == c_double) then
         call librpa_set_kgrids_kvec_c(this%ptr_c_handle, nk1_c, nk2_c, nk3_c, &
                                       nkpts_c, kvecs, kweights_ptr)
      else
         allocate(kvecs_c(3, nkpts))
         kvecs_c = real(kvecs, kind=c_double)
         call librpa_set_kgrids_kvec_c(this%ptr_c_handle, nk1_c, nk2_c, nk3_c, &
                                       nkpts_c, kvecs_c, kweights_ptr)
         deallocate(kvecs_c)
      end if
      if (allocated(kweights_c)) deallocate(kweights_c)
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

      integer :: ik
      integer(c_int) :: nkpts_c
      integer(c_int), allocatable :: map_q_ks_c(:)

      allocate(map_q_ks_c(nkpts))
      map_q_ks_c = int(map_q_ks, kind=c_int) - 1
      nkpts_c = int(nkpts, kind=c_int)
      call librpa_set_kq_mapping_c(this%ptr_c_handle, nkpts_c, map_q_ks_c)
      deallocate(map_q_ks_c)
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

      integer(c_int) :: r_c(3)
      integer(c_int) :: routing_c, i_atom_c, j_atom_c, nao_i_c, nao_j_c, naux_i_c, shrink_aux_c
      real(c_double), allocatable :: coeff_c(:,:,:)

      ! Sanity check
      if (size(coeff,1) /= naux_i .or. size(coeff,2) /= nao_j .or. size(coeff,3) /= nao_i) then
         write(*,*) "wrong coeff shape: input (", naux_i, nao_i , nao_j, ") | ", &
                    "internal (", size(coeff,1), size(coeff,2), size(coeff,3), ")"
         error stop "librpa_set_lri_coeff: coeff has wrong shape"
      end if

      ! Check if the routing is a valid LIBRPA_ROUTING_ parameter
      select case (routing)
         case (LIBRPA_ROUTING_AUTO)
         case (LIBRPA_ROUTING_RTAU)
         case (LIBRPA_ROUTING_ATOMPAIR)
         case (LIBRPA_ROUTING_LIBRI)
         case default
            write(*,*) "Invalid routing parameter:", routing
            error stop
      end select

      r_c = int(r, kind=c_int)
      routing_c = int(routing, c_int)
      i_atom_c = int(i_atom-1, c_int)
      j_atom_c = int(j_atom-1, c_int)
      nao_i_c = int(nao_i, c_int)
      nao_j_c = int(nao_j, c_int)
      naux_i_c = int(naux_i, c_int)
      shrink_aux_c = 0
      if (present(shrink_aux)) then
         if (shrink_aux) shrink_aux_c = 1
      end if
      if (dp == c_double) then
         call librpa_set_lri_coeff_c(this%ptr_c_handle, &
               routing_c, i_atom_c, j_atom_c, nao_i_c, nao_j_c, naux_i_c, r_c, coeff, shrink_aux_c)
      else
         allocate(coeff_c(naux_i, nao_j, nao_i))
         coeff_c = real(coeff, kind=c_double)
         call librpa_set_lri_coeff_c(this%ptr_c_handle, &
               routing_c, i_atom_c, j_atom_c, nao_i_c, nao_j_c, naux_i_c, r_c, coeff_c, shrink_aux_c)
         deallocate(coeff_c)
      end if
   end subroutine librpa_set_lri_coeff

   !> @brief Internal: set Coulomb matrix (atom-pair format).
   !>
   !> @param[in,out] h       Handler.
   !> @param[in]     ik       K-point index (1-based).
   !> @param[in]     i_atom   Atom I index (1-based).
   !> @param[in]     j_atom   Atom J index (1-based).
   !> @param[in]     naux_i   Number of aux functions for i.
   !> @param[in]     naux_j   Number of aux functions for j.
   !> @param[in]     vq       Coulomb matrix (complex).
   !> @param[in]     vq_threshold Threshold.
   !> @param[in]     is_cut   True for truncated Coulomb.
   subroutine set_aux_coulomb_k_atom_pair(h, ik, i_atom, j_atom, naux_i, naux_j, vq, vq_threshold, is_cut)
      use iso_c_binding, only: c_int, c_double, c_double_complex, c_loc
      type(LibrpaHandler), intent(inout) :: h
      integer, intent(in) :: ik, i_atom, j_atom, naux_i, naux_j
      complex(dp), intent(in) :: vq(naux_i, naux_j)
      real(dp), intent(in) :: vq_threshold
      logical, intent(in) :: is_cut

      integer(c_int) :: ik_c, i_atom_c, j_atom_c, naux_i_c, naux_j_c
      complex(c_double_complex), allocatable, target :: vq_c(:,:)
      real(c_double) :: thres_c

      ik_c = int(ik-1, kind=c_int)
      i_atom_c = int(i_atom-1, kind=c_int)
      j_atom_c = int(j_atom-1, kind=c_int)
      naux_i_c = int(naux_i, kind=c_int)
      naux_j_c = int(naux_j, kind=c_int)
      allocate(vq_c(naux_j, naux_i))
      vq_c = transpose(cmplx(vq, kind=c_double_complex))
      thres_c = real(vq_threshold, kind=c_double)
      if (is_cut) then
         call librpa_set_aux_cut_coulomb_k_atom_pair_packed_c(h%ptr_c_handle, &
            ik_c, i_atom_c, j_atom_c, naux_i_c, naux_j_c, c_loc(vq_c), thres_c)
      else
         call librpa_set_aux_bare_coulomb_k_atom_pair_packed_c(h%ptr_c_handle, &
            ik_c, i_atom_c, j_atom_c, naux_i_c, naux_j_c, c_loc(vq_c), thres_c)
      end if

      deallocate(vq_c)
   end subroutine set_aux_coulomb_k_atom_pair

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

      call set_aux_coulomb_k_atom_pair(this, ik, i_atom, j_atom, naux_i, naux_j, vq, vq_threshold, .false.)
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

      call set_aux_coulomb_k_atom_pair(this, ik, i_atom, j_atom, naux_i, naux_j, vq, vq_threshold, .true.)
   end subroutine librpa_set_aux_cut_coulomb_k_atom_pair

   !> @brief Internal: set Coulomb matrix (2D block format).
   subroutine set_aux_coulomb_k_2d_block(h, ik, mu_begin, mu_end, nu_begin, nu_end, vq, is_cut)
      use iso_c_binding, only: c_int, c_double_complex, c_loc
      implicit none
      type(LibrpaHandler), intent(inout) :: h
      integer, intent(in) :: ik, mu_begin, mu_end, nu_begin, nu_end
      complex(dp), intent(in) :: vq(mu_end-mu_begin+1, nu_end-nu_begin+1)
      logical, intent(in) :: is_cut

      integer(c_int) :: ik_c, mb, me, nb, ne
      complex(c_double_complex), allocatable, target :: vq_c(:,:)

      ik_c = int(ik-1, kind=c_int)
      ! The C interface uses included beginning and excluded ending.
      ! In Fortran, it is more common that both of two ends are included
      mb = int(mu_begin-1, kind=c_int)
      me = int(mu_end, kind=c_int)
      nb = int(nu_begin-1, kind=c_int)
      ne = int(nu_end, kind=c_int)

      allocate(vq_c(ne-nb, me-mb))
      vq_c = transpose(cmplx(vq, kind=c_double_complex))
      if (is_cut) then
         call librpa_set_aux_cut_coulomb_k_2d_block_packed_c(h%ptr_c_handle, ik_c, mb, me, nb, ne, c_loc(vq_c))
      else
         call librpa_set_aux_bare_coulomb_k_2d_block_packed_c(h%ptr_c_handle, ik_c, mb, me, nb, ne, c_loc(vq_c))
      end if
      deallocate(vq_c)
   end subroutine set_aux_coulomb_k_2d_block

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

      call set_aux_coulomb_k_2d_block(this, ik, mu_begin, mu_end, nu_begin, nu_end, vq, .false.)
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

      call set_aux_coulomb_k_2d_block(this, ik, mu_begin, mu_end, nu_begin, nu_end, vq, .true.)
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

      integer(c_int) :: nfreq_c
      real(c_double), allocatable :: omegas_c(:), df_c(:)

      nfreq_c = int(nfreq, kind=c_int)
      if (dp == c_double) then
         call librpa_set_dielect_func_imagfreq_c(this%ptr_c_handle, nfreq_c, omegas_imag, dielect_func)
      else
         allocate(omegas_c(nfreq))
         allocate(df_c(nfreq))
         omegas_c(:) = real(omegas_imag(:), kind=c_double)
         df_c(:) = real(dielect_func(:), kind=c_double)
         call librpa_set_dielect_func_imagfreq_c(this%ptr_c_handle, nfreq_c, omegas_c, df_c)
         deallocate(omegas_c, df_c)
      end if
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

      integer(c_int) :: nkb_c
      real(c_double), allocatable :: kfb_c(:,:)

      nkb_c = int(nkpts_band, kind=c_int)

      if (dp == c_double) then
         call librpa_set_band_kvec_c(this%ptr_c_handle, nkb_c, kfrac_band)
      else
         allocate(kfb_c(3, nkpts_band))
         kfb_c = real(kfrac_band, kind=c_double)
         call librpa_set_band_kvec_c(this%ptr_c_handle, nkb_c, kfb_c)
         deallocate(kfb_c)
      end if
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

      integer(c_int) :: nspins_c, nkpts_band_c, nstates_c
      real(c_double), allocatable :: occ_c(:,:,:), eig_c(:,:,:)

      nspins_c = int(nspins, kind=c_int)
      nkpts_band_c = int(nkpts_band, kind=c_int)
      nstates_c = int(nstates, kind=c_int)

      if (dp == c_double) then
         call librpa_set_band_occ_eigval_c(this%ptr_c_handle, nspins_c, nkpts_band_c, nstates_c, occ, eig)
      else
         allocate(occ_c(nstates, nkpts_band, nspins))
         allocate(eig_c(nstates, nkpts_band, nspins))
         occ_c = real(occ, kind=c_double)
         eig_c = real(eig, kind=c_double)
         call librpa_set_band_occ_eigval_c(this%ptr_c_handle, nspins_c, nkpts_band_c, nstates_c, occ_c, eig_c)
         deallocate(occ_c, eig_c)
      end if
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
      use iso_c_binding, only: c_int, c_double, c_loc
      implicit none

      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ispin, ik_band, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_cplx(nbasis_local, nstates_local)

      real(c_double), allocatable :: wfc_real(:,:), wfc_imag(:,:)
      integer(c_int) :: ispin_c, ikb_c, nstates_local_c, nbasis_local_c

      ispin_c = int(ispin-1, kind=c_int)
      ikb_c = int(ik_band-1, kind=c_int)
      nstates_local_c = int(nstates_local, kind=c_int)
      nbasis_local_c = int(nbasis_local, kind=c_int)

      if (dp == c_double) then
         ! Fast path without create intermediate Fortran arrays
         call librpa_set_wfc_band_packed_c(&
            this%ptr_c_handle, ispin_c, ikb_c, &
            nstates_local_c, nbasis_local_c, c_loc(wfc_cplx))
      else
         allocate(wfc_real(nbasis_local, nstates_local))
         allocate(wfc_imag(nbasis_local, nstates_local))
         wfc_real = real(wfc_cplx, kind=c_double)
         wfc_imag = real(aimag(wfc_cplx), kind=c_double)
         call librpa_set_wfc_band_c(this%ptr_c_handle, ispin_c, ikb_c, &
            nstates_local_c, nbasis_local_c, wfc_real, wfc_imag)
         deallocate(wfc_real, wfc_imag)
      end if
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
      use iso_c_binding, only: c_int, c_double, c_loc
      implicit none

      class(LibrpaHandler), intent(inout) :: this
      integer, intent(in) :: ik_band, nstates_local, nbasis_local
      complex(dp), intent(in), target :: wfc_up_cplx(nbasis_local, nstates_local)
      complex(dp), intent(in), target :: wfc_dn_cplx(nbasis_local, nstates_local)

      real(c_double), allocatable :: wfc_up_real(:,:), wfc_up_imag(:,:)
      real(c_double), allocatable :: wfc_dn_real(:,:), wfc_dn_imag(:,:)
      integer(c_int) :: ikb_c, nstates_local_c, nbasis_local_c

      ikb_c = int(ik_band-1, kind=c_int)
      nstates_local_c = int(nstates_local, kind=c_int)
      nbasis_local_c = int(nbasis_local, kind=c_int)

      if (dp == c_double) then
         ! Fast path without create intermediate Fortran arrays
         call librpa_set_wfc_band_spinor_packed_c(&
            this%ptr_c_handle, ikb_c, &
            nstates_local_c, nbasis_local_c, c_loc(wfc_up_cplx), c_loc(wfc_dn_cplx))
      else
         allocate(wfc_up_real(nbasis_local, nstates_local))
         allocate(wfc_up_imag(nbasis_local, nstates_local))
         allocate(wfc_dn_real(nbasis_local, nstates_local))
         allocate(wfc_dn_imag(nbasis_local, nstates_local))
         wfc_up_real = real(wfc_up_cplx, kind=c_double)
         wfc_up_imag = real(aimag(wfc_up_cplx), kind=c_double)
         wfc_dn_real = real(wfc_dn_cplx, kind=c_double)
         wfc_dn_imag = real(aimag(wfc_dn_cplx), kind=c_double)
         call librpa_set_wfc_band_spinor_c(this%ptr_c_handle, ikb_c, &
            nstates_local_c, nbasis_local_c, wfc_up_real, wfc_up_imag, wfc_dn_real, wfc_dn_imag)
         deallocate(wfc_up_real, wfc_up_imag)
         deallocate(wfc_dn_real, wfc_dn_imag)
      end if
   end subroutine librpa_set_wfc_band_spinor

   !> @brief Reset band structure data.
   !> @param[in,out] this  Handler.
   subroutine librpa_reset_band_data(this)
      implicit none
      class(LibrpaHandler), intent(inout) :: this

      call librpa_reset_band_data_c(this%ptr_c_handle)
   end subroutine librpa_reset_band_data

   !> @name Compute functions
   !> @brief Functions for performing RPA, EXX, and G0W0 calculations.
   !> @{

   !> @brief Construct and return frequency grids.
   !>
   !> @param[in,out] this    Handler.
   !> @param[in,out] opts    Runtime options.
   !> @param[out]    omegas  Frequency values.
   !>
   !> @param[out]    weights Quadrature weights.
   subroutine librpa_get_imaginary_frequency_grids(this, opts, omegas, weights)
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts
      real(dp), allocatable, intent(inout) :: omegas(:), weights(:)

      real(c_double), allocatable :: omegas_c(:), weights_c(:)

      if (allocated(omegas)) then
         if (size(omegas) .ne. opts%nfreq) deallocate(omegas)
      end if
      if (.not. allocated(omegas)) allocate(omegas(opts%nfreq))

      if (allocated(weights)) then
         if (size(weights) .ne. opts%nfreq) deallocate(weights)
      end if
      if (.not. allocated(weights)) allocate(weights(opts%nfreq))

      ! write(*, *) "size(omegas): ", size(omegas), " size(weights): ", size(weights)

      call sync_opts(opts, SYNC_OPTS_F2C)
      if (c_double == dp) then
         call librpa_get_imaginary_frequency_grids_c(this%ptr_c_handle, opts%opts_c, omegas, weights)
      else
         allocate(omegas_c(opts%nfreq))
         allocate(weights_c(opts%nfreq))
         call librpa_get_imaginary_frequency_grids_c(this%ptr_c_handle, opts%opts_c, omegas_c, weights_c)
         omegas(:) = real(omegas_c(:), kind=dp)
         weights(:) = real(weights_c(:), kind=dp)
         deallocate(omegas_c, weights_c)
      end if
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

      integer(c_int) :: nkpts_ibz_c
      real(c_double), allocatable :: contrib_ibzk_re(:), contrib_ibzk_im(:)

      call sync_opts(opts, SYNC_OPTS_F2C)

      nkpts_ibz_c = int(nkpts_ibz, kind=c_int)
      allocate(contrib_ibzk_re(nkpts_ibz), contrib_ibzk_im(nkpts_ibz))
      e = librpa_get_rpa_correlation_energy_c(this%ptr_c_handle, opts%opts_c, &
                                              nkpts_ibz_c, contrib_ibzk_re, contrib_ibzk_im)
      if (dp == c_double) then
         contrib_ibzk = contrib_ibzk_re + contrib_ibzk_im * CIMAG
      else
         contrib_ibzk = real(contrib_ibzk_re, kind=dp) + real(contrib_ibzk_im, kind=dp) * CIMAG
      end if
      deallocate(contrib_ibzk_re, contrib_ibzk_im)
   end function librpa_get_rpa_correlation_energy

   !> @brief Build exact-exchange matrix in real space.
   !> @param[in,out] this  Handler.
   !> @param[in,out] opts  Runtime options.
   subroutine librpa_build_exx(this, opts)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts

      call sync_opts(opts, SYNC_OPTS_F2C)
      call librpa_build_exx_c(this%ptr_c_handle, opts%opts_c)
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

      integer(c_int), allocatable :: iks_this_c(:)
      real(c_double), allocatable :: vexx_c(:,:,:)
      integer(c_int) :: n_spins_c, n_kpts_this_c, i_state_low_c, i_state_high_c
      integer :: n_states_calc

      n_spins_c = int(n_spins, kind=c_int)
      i_state_low_c = int(i_state_low - 1, kind=c_int)
      ! i_state_high is not included in the C interface, so no minus 1 here
      i_state_high_c = int(i_state_high, kind=c_int)

      n_kpts_this_c = int(n_kpts_this, kind=c_int)
      allocate(iks_this_c(max(1, n_kpts_this)))
      if (n_kpts_this > 0) then
         iks_this_c = int(iks_this(1:n_kpts_this), kind=c_int) - 1
         ! write(*,*) "size(iks_local) ", size(iks_local)
         ! write(*,*) "size(iks_local_c) ", size(iks_local_c)
         ! write(*,*) "iks_local_c ", iks_local_c
      end if

      call sync_opts(opts, SYNC_OPTS_F2C)
      if (dp == c_double) then
         call librpa_get_exx_pot_kgrid_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                         iks_this_c, i_state_low_c, i_state_high_c, vexx)
      else
         n_states_calc = i_state_high - i_state_low + 1
         allocate(vexx_c(n_states_calc, max(n_kpts_this, 1), n_spins))
         call librpa_get_exx_pot_kgrid_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                         iks_this_c, i_state_low_c, i_state_high_c, vexx_c)
         if (n_kpts_this > 0) vexx = real(vexx_c, kind=dp)
         deallocate(vexx_c)
      end if

      deallocate(iks_this_c)
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

      integer(c_int), allocatable :: iks_band_this_c(:)
      real(c_double), allocatable :: vexx_band_c(:,:,:)
      integer(c_int) :: n_spins_c, n_kpts_this_c, i_state_low_c, i_state_high_c
      integer :: n_states_calc

      n_spins_c = int(n_spins, kind=c_int)
      i_state_low_c = int(i_state_low - 1, kind=c_int)
      ! i_state_high is not included in the C interface, so no minus 1 here
      i_state_high_c = int(i_state_high, kind=c_int)

      n_kpts_this_c = int(n_kpts_band_this, kind=c_int)
      allocate(iks_band_this_c(max(1, n_kpts_band_this)))
      if (n_kpts_band_this > 0) then
         iks_band_this_c = int(iks_band_this(1:n_kpts_band_this), kind=c_int) - 1
         ! write(*,*) "size(iks_local) ", size(iks_local)
         ! write(*,*) "size(iks_local_c) ", size(iks_local_c)
         ! write(*,*) "iks_local_c ", iks_local_c
      end if

      call sync_opts(opts, SYNC_OPTS_F2C)
      if (dp == c_double) then
         call librpa_get_exx_pot_band_k_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                          iks_band_this_c, i_state_low_c, i_state_high_c, vexx_band)
      else
         n_states_calc = i_state_high - i_state_low + 1
         allocate(vexx_band_c(n_states_calc, max(n_kpts_band_this, 1), n_spins))
         call librpa_get_exx_pot_band_k_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                          iks_band_this_c, i_state_low_c, i_state_high_c, vexx_band_c)
         if (n_kpts_band_this > 0) vexx_band = real(vexx_band_c, kind=dp)
         deallocate(vexx_band_c)
      end if

      deallocate(iks_band_this_c)
   end subroutine librpa_get_exx_pot_band_k

   !> @brief Build G0W0 self-energy matrix in real space.
   !> @param[in,out] this  Handler.
   !> @param[in,out] opts  Runtime options.
   subroutine librpa_build_g0w0_sigma(this, opts)
      implicit none
      class(LibrpaHandler), intent(inout) :: this
      type(LibrpaOptions), intent(inout) :: opts

      call sync_opts(opts, SYNC_OPTS_F2C)
      call librpa_build_g0w0_sigma_c(this%ptr_c_handle, opts%opts_c)
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

      integer(c_int), allocatable :: iks_this_c(:)
      real(c_double), allocatable :: vxc_c(:,:,:), vexx_c(:,:,:)
      real(c_double), allocatable :: sigc_re_c(:,:,:), sigc_im_c(:,:,:)
      integer(c_int) :: n_spins_c, n_kpts_this_c, i_state_low_c, i_state_high_c
      integer :: n_states_calc

      n_spins_c = int(n_spins, kind=c_int)
      i_state_low_c = int(i_state_low - 1, kind=c_int)
      i_state_high_c = int(i_state_high, kind=c_int)

      n_kpts_this_c = int(n_kpts_this, kind=c_int)
      if (n_kpts_this > 0) then
         allocate(iks_this_c(n_kpts_this))
         iks_this_c = int(iks_this(1:n_kpts_this), kind=c_int) - 1
      else
         allocate(iks_this_c(1))
      end if

      n_states_calc = i_state_high - i_state_low + 1
      allocate(sigc_re_c(n_states_calc, max(n_kpts_this, 1), n_spins))
      allocate(sigc_im_c(n_states_calc, max(n_kpts_this, 1), n_spins))

      call sync_opts(opts, SYNC_OPTS_F2C)
      if (dp == c_double) then
         call librpa_get_g0w0_sigc_kgrid_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                           iks_this_c, i_state_low_c, i_state_high_c, vxc, vexx, sigc_re_c, sigc_im_c)
      else
         allocate(vexx_c(n_states_calc, max(n_kpts_this, 1), n_spins))
         allocate(vxc_c(n_states_calc, max(n_kpts_this, 1), n_spins))
         if (n_kpts_this > 0) then
            vxc_c = real(vxc, kind=c_double)
            vexx_c = real(vexx, kind=c_double)
         end if
         call librpa_get_g0w0_sigc_kgrid_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                           iks_this_c, i_state_low_c, i_state_high_c, vxc_c, vexx_c, sigc_re_c, sigc_im_c)
         if (allocated(vxc_c)) deallocate(vxc_c)
         if (allocated(vexx_c)) deallocate(vexx_c)
      end if

      if (n_kpts_this > 0) then
         sigc(:,:,:) = cmplx(sigc_re_c, sigc_im_c, kind=dp)
      end if

      deallocate(sigc_re_c)
      deallocate(sigc_im_c)
      deallocate(iks_this_c)
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

      integer(c_int), allocatable :: iks_this_c(:)
      real(c_double), allocatable :: omegas_c(:)
      real(c_double), allocatable :: vxc_c(:,:,:), vexx_c(:,:,:)
      real(c_double), allocatable, target :: sigc_c(:)
      real(c_double), allocatable :: spectral_function_c(:,:,:,:)
      type(c_ptr) :: sigc_ptr
      integer(c_int) :: n_spins_c, n_kpts_this_c, i_state_low_c, i_state_high_c, n_omegas_c
      integer :: n_states_calc, n_omegas, n_sigc, n_sigc_buffer

      n_spins_c = int(n_spins, kind=c_int)
      i_state_low_c = int(i_state_low - 1, kind=c_int)
      i_state_high_c = int(i_state_high, kind=c_int)
      n_omegas = size(omegas)
      n_omegas_c = int(n_omegas, kind=c_int)

      n_kpts_this_c = int(n_kpts_this, kind=c_int)
      if (n_kpts_this > 0) then
         allocate(iks_this_c(n_kpts_this))
         iks_this_c = int(iks_this(1:n_kpts_this), kind=c_int) - 1
      else
         allocate(iks_this_c(1))
      end if

      n_states_calc = i_state_high - i_state_low + 1

      call sync_opts(opts, SYNC_OPTS_F2C)
      n_sigc = n_omegas * n_states_calc * n_kpts_this * n_spins
      sigc_ptr = c_null_ptr
      if (present(sigc)) then
         n_sigc_buffer = max(n_omegas, 1) * max(n_states_calc, 1) * max(n_kpts_this, 1) * max(n_spins, 1)
         allocate(sigc_c(2 * n_sigc_buffer))
         sigc_ptr = c_loc(sigc_c(1))
      end if
      if (dp == c_double) then
         call librpa_get_g0w0_spectral_function_kgrid_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                                        iks_this_c, i_state_low_c, i_state_high_c, n_omegas_c, &
                                                        omegas, vxc, vexx, spectral_function, sigc_ptr)
         if (present(sigc) .and. n_sigc > 0) then
            sigc(:,:,:,:) = cmplx(reshape(real(sigc_c(1:2*n_sigc-1:2), kind=dp), shape(sigc)), &
                                  reshape(real(sigc_c(2:2*n_sigc:2), kind=dp), shape(sigc)), kind=dp)
         end if
      else
         allocate(omegas_c(max(n_omegas, 1)))
         allocate(vexx_c(n_states_calc, max(n_kpts_this, 1), n_spins))
         allocate(vxc_c(n_states_calc, max(n_kpts_this, 1), n_spins))
         allocate(spectral_function_c(max(n_omegas, 1), n_states_calc, max(n_kpts_this, 1), n_spins))
         if (n_omegas > 0) then
            omegas_c(1:n_omegas) = real(omegas, kind=c_double)
         end if
         if (n_kpts_this > 0) then
            vxc_c = real(vxc, kind=c_double)
            vexx_c = real(vexx, kind=c_double)
         end if
         call librpa_get_g0w0_spectral_function_kgrid_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_this_c, &
                                                        iks_this_c, i_state_low_c, i_state_high_c, n_omegas_c, &
                                                        omegas_c, vxc_c, vexx_c, spectral_function_c, &
                                                        sigc_ptr)
         if (n_omegas > 0 .and. n_kpts_this > 0) then
            spectral_function(:,:,:,:) = real(spectral_function_c(1:n_omegas,:,:,:), kind=dp)
         end if
         if (present(sigc) .and. n_sigc > 0) then
            sigc(:,:,:,:) = cmplx(reshape(real(sigc_c(1:2*n_sigc-1:2), kind=dp), shape(sigc)), &
                                  reshape(real(sigc_c(2:2*n_sigc:2), kind=dp), shape(sigc)), kind=dp)
         end if
         deallocate(omegas_c)
         deallocate(vxc_c)
         deallocate(vexx_c)
         deallocate(spectral_function_c)
      end if

      if (allocated(sigc_c)) deallocate(sigc_c)
      deallocate(iks_this_c)
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

      integer(c_int), allocatable :: iks_band_this_c(:)
      real(c_double), allocatable :: vxc_c(:,:,:), vexx_c(:,:,:)
      real(c_double), allocatable :: sigc_re_c(:,:,:), sigc_im_c(:,:,:)
      integer(c_int) :: n_spins_c, n_kpts_band_this_c, i_state_low_c, i_state_high_c
      integer :: n_states_calc

      n_spins_c = int(n_spins, kind=c_int)
      i_state_low_c = int(i_state_low - 1, kind=c_int)
      i_state_high_c = int(i_state_high, kind=c_int)

      n_kpts_band_this_c = int(n_kpts_band_this, kind=c_int)
      if (n_kpts_band_this > 0) then
         allocate(iks_band_this_c(n_kpts_band_this))
         iks_band_this_c = int(iks_band_this(1:n_kpts_band_this), kind=c_int) - 1
      else
         allocate(iks_band_this_c(1))
      end if

      n_states_calc = i_state_high - i_state_low + 1
      allocate(sigc_re_c(n_states_calc, max(n_kpts_band_this, 1), n_spins))
      allocate(sigc_im_c(n_states_calc, max(n_kpts_band_this, 1), n_spins))

      call sync_opts(opts, SYNC_OPTS_F2C)
      if (dp == c_double) then
         call librpa_get_g0w0_sigc_band_k_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_band_this_c, &
                                            iks_band_this_c, i_state_low_c, i_state_high_c, vxc_band, vexx_band, sigc_re_c, sigc_im_c)
      else
         allocate(vexx_c(n_states_calc, max(n_kpts_band_this, 1), n_spins))
         allocate(vxc_c(n_states_calc, max(n_kpts_band_this, 1), n_spins))
         if (n_kpts_band_this > 0) then
            vxc_c = real(vxc_band, kind=c_double)
            vexx_c = real(vexx_band, kind=c_double)
         end if
         call librpa_get_g0w0_sigc_band_k_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_band_this_c, &
                                            iks_band_this_c, i_state_low_c, i_state_high_c, vxc_c, vexx_c, sigc_re_c, sigc_im_c)
         if (allocated(vxc_c)) deallocate(vxc_c)
         if (allocated(vexx_c)) deallocate(vexx_c)
      end if

      if (n_kpts_band_this > 0) then
         sigc_band(:,:,:) = cmplx(sigc_re_c, sigc_im_c, kind=dp)
      end if

      deallocate(sigc_re_c)
      deallocate(sigc_im_c)
      deallocate(iks_band_this_c)
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

      integer(c_int), allocatable :: iks_band_this_c(:)
      real(c_double), allocatable :: omegas_c(:)
      real(c_double), allocatable :: vxc_c(:,:,:), vexx_c(:,:,:)
      real(c_double), allocatable, target :: sigc_c(:)
      real(c_double), allocatable :: spectral_function_c(:,:,:,:)
      type(c_ptr) :: sigc_ptr
      integer(c_int) :: n_spins_c, n_kpts_band_this_c, i_state_low_c, i_state_high_c, n_omegas_c
      integer :: n_states_calc, n_omegas, n_sigc, n_sigc_buffer

      n_spins_c = int(n_spins, kind=c_int)
      i_state_low_c = int(i_state_low - 1, kind=c_int)
      i_state_high_c = int(i_state_high, kind=c_int)
      n_omegas = size(omegas)
      n_omegas_c = int(n_omegas, kind=c_int)

      n_kpts_band_this_c = int(n_kpts_band_this, kind=c_int)
      if (n_kpts_band_this > 0) then
         allocate(iks_band_this_c(n_kpts_band_this))
         iks_band_this_c = int(iks_band_this(1:n_kpts_band_this), kind=c_int) - 1
      else
         allocate(iks_band_this_c(1))
      end if

      n_states_calc = i_state_high - i_state_low + 1

      call sync_opts(opts, SYNC_OPTS_F2C)
      n_sigc = n_omegas * n_states_calc * n_kpts_band_this * n_spins
      sigc_ptr = c_null_ptr
      if (present(sigc_band)) then
         n_sigc_buffer = max(n_omegas, 1) * max(n_states_calc, 1) * max(n_kpts_band_this, 1) * max(n_spins, 1)
         allocate(sigc_c(2 * n_sigc_buffer))
         sigc_ptr = c_loc(sigc_c(1))
      end if
      if (dp == c_double) then
         call librpa_get_g0w0_spectral_function_band_k_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_band_this_c, &
                                                         iks_band_this_c, i_state_low_c, i_state_high_c, n_omegas_c, &
                                                         omegas, vxc_band, vexx_band, spectral_function_band, &
                                                         sigc_ptr)
         if (present(sigc_band) .and. n_sigc > 0) then
            sigc_band(:,:,:,:) = cmplx(reshape(real(sigc_c(1:2*n_sigc-1:2), kind=dp), shape(sigc_band)), &
                                       reshape(real(sigc_c(2:2*n_sigc:2), kind=dp), shape(sigc_band)), kind=dp)
         end if
      else
         allocate(omegas_c(max(n_omegas, 1)))
         allocate(vexx_c(n_states_calc, max(n_kpts_band_this, 1), n_spins))
         allocate(vxc_c(n_states_calc, max(n_kpts_band_this, 1), n_spins))
         allocate(spectral_function_c(max(n_omegas, 1), n_states_calc, max(n_kpts_band_this, 1), n_spins))
         if (n_omegas > 0) then
            omegas_c(1:n_omegas) = real(omegas, kind=c_double)
         end if
         if (n_kpts_band_this > 0) then
            vxc_c = real(vxc_band, kind=c_double)
            vexx_c = real(vexx_band, kind=c_double)
         end if
         call librpa_get_g0w0_spectral_function_band_k_c(this%ptr_c_handle, opts%opts_c, n_spins_c, n_kpts_band_this_c, &
                                                         iks_band_this_c, i_state_low_c, i_state_high_c, n_omegas_c, &
                                                         omegas_c, vxc_c, vexx_c, spectral_function_c, &
                                                         sigc_ptr)
         if (n_omegas > 0 .and. n_kpts_band_this > 0) then
            spectral_function_band(:,:,:,:) = real(spectral_function_c(1:n_omegas,:,:,:), kind=dp)
         end if
         if (present(sigc_band) .and. n_sigc > 0) then
            sigc_band(:,:,:,:) = cmplx(reshape(real(sigc_c(1:2*n_sigc-1:2), kind=dp), shape(sigc_band)), &
                                       reshape(real(sigc_c(2:2*n_sigc:2), kind=dp), shape(sigc_band)), kind=dp)
         end if
         deallocate(omegas_c)
         deallocate(vxc_c)
         deallocate(vexx_c)
         deallocate(spectral_function_c)
      end if

      if (allocated(sigc_c)) deallocate(sigc_c)
      deallocate(iks_band_this_c)
   end subroutine librpa_get_g0w0_spectral_function_band_k
   !> @}

end module librpa_f03
