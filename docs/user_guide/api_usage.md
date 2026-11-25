# API Usage

## Summary of user API functions

The user-facing APIs that can be called from within a DFT code are declared in
{librpa}`librpa.h` (C), {librpa}`librpa.hpp` (C++), and {librpa}`librpa_f03.f90` (Fortran 2003).

These APIs fall into three groups:

- global environment management (initialization and finalization)
- input parsing
- computation

They are centered around two management structures:

- {librpa}`LibrpaHandler`, which manages input parsing and computations
- {librpa}`LibrpaOptions`, which stores runtime parameters

```{note}
Some advanced options and diagnostic output paths are marked `Experimental` in the
[runtime parameter guide](runtime_parameters). They are exposed for development and
validation workflows, but are not as broadly tested as the default RPA, EXX, and G0W0 paths.
```

## Handler object

The {librpa}`LibrpaHandler` (C) / {librpa}`librpa::Handler` (C++) / {librpa}`type(LibrpaHandler) <librpahandler>` (Fortran) is the central object that manages the LibRPA computation. It encapsulates all input data and computation state.

The typical workflow is:
1. Create a handler using the MPI communicator
2. Set input data via setter methods
3. Call computation functions
4. Free the handler when done

**C:**
```c
LibrpaHandler *h = librpa_create_handler(MPI_COMM_WORLD);
// ... set input data and compute ...
librpa_destroy_handler(h);
```

**C++:**
```cpp
librpa::Handler h(MPI_COMM_WORLD);
// ... set input data and compute ...
h.free();
```

**Fortran:**
```fortran
type(LibrpaHandler) :: h
call h%init(MPI_COMM_WORLD)
! ... set input data and compute ...
call h%free()
```

(environment-setup)=
## Environment setup

Runtime environment for LibRPA should be properly setup to parse the input data and call
computation functions. They are managed by the environment setup APIs (C/C++/Fortran):

- {librpa}`librpa_init_global` / {librpa}`librpa::init_global`
- {librpa}`librpa_finalize_global` / {librpa}`librpa::finalize_global`
- {librpa}`librpa_set_output_level` / {librpa}`librpa::set_output_level`
- {librpa}`librpa_get_output_level` / {librpa}`librpa::get_output_level`
- {librpa}`librpa_init_options` / {librpa}`librpa::Options`
- {librpa}`librpa_set_output_dir` / {librpa}`librpa::Options::set_output_dir`
- {librpa}`librpa_set_restart_from_dir` / {librpa}`librpa::Options::set_restart_from_dir`

Any calls to input parsing and computation APIs should be surrounded by calls to `librpa_init_global` and `librpa_finalize_global`.

To configure runtime parameters for computation, an {librpa}`LibrpaOptions` object should be created, modified and passed to computation functions.
Parameters are attributes of `LibrpaOptions` object, and can be initialized using `librpa_init_options`.
For API calls, set the time-frequency grid type explicitly before computation.
For example, to configure a minimax grid, the number of frequency points, and Green's function threshold:
```c
// C API
LibrpaOptions opts;
librpa_init_options(&opts);
opts.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
opts.nfreq = 16;
opts.gf_threshold = 1.e-3;
librpa_set_output_level(LIBRPA_VERBOSE_INFO);
// Optional: read restart data from a directory distinct from output_dir.
librpa_set_restart_from_dir(&opts, "previous_run");
```

```cpp
// C++ API
librpa::Options opts;
opts.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
opts.nfreq = 16;
opts.gf_threshold = 1.e-3;
librpa::set_output_level(LIBRPA_VERBOSE_INFO);
opts.set_restart_from_dir("previous_run");
```

```fortran
! Fortran API
type(LibrpaOptions) :: opts
call opts%init()
opts%tfgrids_type = LIBRPA_TFGRID_MINIMAX
opts%nfreq = 16
opts%gf_threshold = 1.0d-3
call librpa_set_output_level(LIBRPA_VERBOSE_INFO)
call opts%set_restart_from_dir("previous_run")
```

The API uses enum/integer constants from the public headers, while the driver reads string values
such as `tfgrids_type = minimax` and maps them to the same internal options.
Driver-only parameters, such as `task` and `input_dir`, are not part of `LibrpaOptions`.

## Input parsing

Input parsing APIs common to all tasks include (C/C++/Fortran):

- {librpa}`librpa_set_scf_dimension` / {librpa}`Handler::set_scf_dimension` / {librpa}`handler%set_scf_dimension <librpahandler::set_scf_dimension>`
- {librpa}`librpa_set_wg_ekb_efermi` / {librpa}`Handler::set_wg_ekb_efermi` / {librpa}`handler%set_wg_ekb_efermi <librpahandler::set_wg_ekb_efermi>`
- {librpa}`librpa_set_wfc` / {librpa}`Handler::set_wfc` / {librpa}`handler%set_wfc <librpahandler::set_wfc>`
- {librpa}`librpa_set_wfc_spinor` / {librpa}`Handler::set_wfc_spinor` / {librpa}`handler%set_wfc_spinor <librpahandler::set_wfc_spinor>`
- {librpa}`librpa_set_ao_basis_wfc` / {librpa}`Handler::set_ao_basis_wfc` / {librpa}`handler%set_ao_basis_wfc <librpahandler::set_ao_basis_wfc>`
- {librpa}`librpa_set_ao_basis_aux` / {librpa}`Handler::set_ao_basis_aux` / {librpa}`handler%set_ao_basis_aux <librpahandler::set_ao_basis_aux>`
- {librpa}`librpa_set_basis_convention` / {librpa}`Handler::set_basis_convention` / {librpa}`handler%set_basis_convention <librpahandler::set_basis_convention>`
- {librpa}`librpa_set_symmetry_operations` / {librpa}`Handler::set_symmetry_operations` / {librpa}`handler%set_symmetry_operations <librpahandler::set_symmetry_operations>`
- {librpa}`librpa_set_latvec_and_G` / {librpa}`Handler::set_latvec_and_G` / {librpa}`handler%set_latvec_and_G <librpahandler::set_latvec_and_g>`
- {librpa}`librpa_set_atoms` / {librpa}`Handler::set_atoms` / {librpa}`handler%set_atoms <librpahandler::set_atoms>`
- {librpa}`librpa_set_kgrids_kvec` / {librpa}`Handler::set_kgrids_kvec` / {librpa}`handler%set_kgrids_kvec <librpahandler::set_kgrids_kvec>`
- {librpa}`librpa_set_kq_mapping` / {librpa}`Handler::set_kq_mapping` / {librpa}`handler%set_kq_mapping <librpahandler::set_kq_mapping>`
- {librpa}`librpa_set_lri_coeff` / {librpa}`Handler::set_lri_coeff` / {librpa}`handler%set_lri_coeff <librpahandler::set_lri_coeff>`
- {librpa}`librpa_set_aux_bare_coulomb_k_atom_pair` / {librpa}`Handler::set_aux_bare_coulomb_k_atom_pair` / {librpa}`handler%set_aux_bare_coulomb_k_atom_pair <librpahandler::set_aux_bare_coulomb_k_atom_pair>`
- {librpa}`librpa_set_aux_cut_coulomb_k_atom_pair` / {librpa}`Handler::set_aux_cut_coulomb_k_atom_pair` / {librpa}`handler%set_aux_cut_coulomb_k_atom_pair <librpahandler::set_aux_cut_coulomb_k_atom_pair>`
- {librpa}`librpa_set_aux_bare_coulomb_k_2d_block` / {librpa}`Handler::set_aux_bare_coulomb_k_2d_block` / {librpa}`handler%set_aux_bare_coulomb_k_2d_block <librpahandler::set_aux_bare_coulomb_k_2d_block>`
- {librpa}`librpa_set_aux_cut_coulomb_k_2d_block` / {librpa}`Handler::set_aux_cut_coulomb_k_2d_block` / {librpa}`handler%set_aux_cut_coulomb_k_2d_block <librpahandler::set_aux_cut_coulomb_k_2d_block>`
- {librpa}`librpa_set_band_kvec` / {librpa}`Handler::set_band_kvec` / {librpa}`handler%set_band_kvec <librpahandler::set_band_kvec>`
- {librpa}`librpa_set_band_occ_eigval` / {librpa}`Handler::set_band_occ_eigval` / {librpa}`handler%set_band_occ_eigval <librpahandler::set_band_occ_eigval>`
- {librpa}`librpa_set_wfc_band` / {librpa}`Handler::set_wfc_band` / {librpa}`handler%set_wfc_band <librpahandler::set_wfc_band>`
- {librpa}`librpa_set_wfc_band_spinor` / {librpa}`Handler::set_wfc_band_spinor` / {librpa}`handler%set_wfc_band_spinor <librpahandler::set_wfc_band_spinor>`

These functions parse information from the hosting DFT code to LibRPA, such as:
- dimension of lattice vectors, orbital basis and auxiliary basis
- k-mesh points, eigenvalues and occupation numbers
- wave functions of mean-field calculation
- local RI coefficients
- Coulomb matrices in auxiliary basis
- data for band structure calculation

They have to be called after the runtime environment is initialized,
see [Environment setup](#environment-setup) above.

`set_kgrids_kvec` takes the k-grid dimensions, the number of loaded SCF k-points,
their Cartesian vectors, and optional weights. If the loaded SCF k-points are
symmetry-reduced for the Coulomb calculation, call `set_kq_mapping` with the
0-based mapping from each loaded SCF k-point to its representative Coulomb q-point.

The wavefunction and auxiliary-basis setters accept optional angular-shell metadata.
Symmetry-enabled calculations should also provide the basis
convention and real-space symmetry operations through `set_basis_convention` and
`set_symmetry_operations`; the convention values must match those used by the calling
electronic-structure code.

The C and C++ APIs additionally provide packed-complex Coulomb setters:

- {librpa}`librpa_set_aux_bare_coulomb_k_atom_pair_packed` /
  {librpa}`Handler::set_aux_bare_coulomb_k_atom_pair_packed`
- {librpa}`librpa_set_aux_cut_coulomb_k_atom_pair_packed` /
  {librpa}`Handler::set_aux_cut_coulomb_k_atom_pair_packed`
- {librpa}`librpa_set_aux_bare_coulomb_k_2d_block_packed` /
  {librpa}`Handler::set_aux_bare_coulomb_k_2d_block_packed`
- {librpa}`librpa_set_aux_cut_coulomb_k_2d_block_packed` /
  {librpa}`Handler::set_aux_cut_coulomb_k_2d_block_packed`

The C functions take interleaved real/imaginary `double` buffers; the C++ wrappers
take `std::complex<double>` buffers. The Fortran Coulomb setters already accept native
complex arrays and therefore do not have separate `_packed` names.

Analytic head/wing calculations can receive velocity matrices through
{librpa}`librpa_set_velocity_matrix` or
{librpa}`librpa_set_velocity_matrix_packed`; the corresponding C++ methods are
{librpa}`Handler::set_velocity_matrix` and
{librpa}`Handler::set_velocity_matrix_packed`. These setters are not currently
exposed by the Fortran wrapper. Matrix elements are ordered as
`[spin][k-point][Cartesian component][row state][column state]`.

For spinor workflows, initialize the handler with two spinor components, then use the spinor
wavefunction setters. Internally, LibRPA selects the normal or spinor path from the mean-field
`n_spinor` attribute through `get_n_spinor()`.
Driver users can select spinor-format input with `use_spinor_wfc`; API users should pass
`nspinor = 2` to `set_scf_dimension` instead.

## Computation

After data have been correctly set up by input parsing functions, computation functions can be called to get quantities of interest:

**General**
- {librpa}`librpa_get_imaginary_frequency_grids`
  / {librpa}`Handler::get_imaginary_frequency_grids`
  / {librpa}`handler%get_imaginary_frequency_grids <librpahandler::get_imaginary_frequency_grids>`

**RPA correlation energy:**
- {librpa}`librpa_get_rpa_correlation_energy`
  / {librpa}`Handler::get_rpa_correlation_energy`
  / {librpa}`handler%get_rpa_correlation_energy <librpahandler::get_rpa_correlation_energy>`

**Exact exchange:**
- {librpa}`librpa_build_exx` / {librpa}`Handler::build_exx` / {librpa}`handler%build_exx <librpahandler::build_exx>`
- {librpa}`librpa_get_exx_pot_kgrid` / {librpa}`Handler::get_exx_pot_kgrid` / {librpa}`handler%get_exx_pot_kgrid <librpahandler::get_exx_pot_kgrid>`
- {librpa}`librpa_get_exx_pot_band_k` / {librpa}`Handler::get_exx_pot_band_k` / {librpa}`handler%get_exx_pot_band_k <librpahandler::get_exx_pot_band_k>`

**G0W0 self-energy:**
- {librpa}`librpa_build_g0w0_sigma` / {librpa}`Handler::build_g0w0_sigma` / {librpa}`handler%build_g0w0_sigma <librpahandler::build_g0w0_sigma>`
- {librpa}`librpa_get_g0w0_sigc_kgrid` / {librpa}`Handler::get_g0w0_sigc_kgrid` / {librpa}`handler%get_g0w0_sigc_kgrid <librpahandler::get_g0w0_sigc_kgrid>`
- {librpa}`librpa_get_g0w0_sigc_band_k` / {librpa}`Handler::get_g0w0_sigc_band_k` / {librpa}`handler%get_g0w0_sigc_band_k <librpahandler::get_g0w0_sigc_band_k>`

**G0W0 quasiparticle energies:**
- {librpa}`librpa_get_g0w0_qpe_kgrid` / {librpa}`Handler::get_g0w0_qpe_kgrid`
- {librpa}`librpa_get_g0w0_qpe_band_k` / {librpa}`Handler::get_g0w0_qpe_band_k`

These functions return the correlation self-energy together with quasiparticle
energies obtained using `opts.option_qpe_solver`. The C++ methods return a
{librpa}`librpa::G0W0QpeResult`. They are not currently exposed by the Fortran
wrapper; Fortran users can retrieve correlation self-energies through the
`get_g0w0_sigc_*` methods.

**G0W0 spectral function:**
- {librpa}`librpa_get_g0w0_spectral_function_kgrid` / {librpa}`Handler::get_g0w0_spectral_function_kgrid` / {librpa}`handler%get_g0w0_spectral_function_kgrid <librpahandler::get_g0w0_spectral_function_kgrid>`
- {librpa}`librpa_get_g0w0_spectral_function_band_k` / {librpa}`Handler::get_g0w0_spectral_function_band_k` / {librpa}`handler%get_g0w0_spectral_function_band_k <librpahandler::get_g0w0_spectral_function_band_k>`

The C spectral-function APIs take an optional packed complex self-energy output as the last argument.
Pass `nullptr` when only the spectral function is needed.
The C++ `get_g0w0_spectral_function_*` methods return only spectral-function values;
use {librpa}`Handler::get_g0w0_spectral_function_with_sigc_kgrid` or
{librpa}`Handler::get_g0w0_spectral_function_with_sigc_band_k` to receive a
{librpa}`librpa::G0W0SpectralFunctionResult`. The Fortran methods expose the
continued self-energy as an optional final argument.

## Schematic examples

### C++

The following C++ code gives an impression of how LibRPA APIs should be called in the host DFT code:

```cpp
#include <librpa.hpp>
#include <mpi.h>

int main()
{
    MPI_Init(...);
    // If LibRI is enabled, MPI thread support should be set to MPI_THREAD_MULTIPLE
    // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    // Before: setup and SCF calculation of hosting program

    // *** Initialize LibRPA runtime environment ***
    librpa::init_global();

    // *** Set up runtime parameters ***
    librpa::Options opts;
    opts.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    opts.nfreq = 16;
    opts.gf_threshold = 1e-3;
    librpa::set_output_level(LIBRPA_VERBOSE_INFO);

    // *** Create handler ***
    librpa::Handler h(MPI_COMM_WORLD);

    // *** Input parsing ***
    h.set_scf_dimension(nspins, nkpts, nstates, nbasis);
    h.set_wg_ekb_efermi(nspins, nkpts, nstates, wg, ekb, efermi);
    h.set_latvec_and_G(lat_mat, G_mat);
    h.set_kgrids_kvec(nk1, nk2, nk3, nkpts, kvecs, kweights);
    h.set_kq_mapping(map_q_ks);
    h.set_ao_basis_wfc(nbs_wfc);
    h.set_ao_basis_aux(nbs_aux);

    // Optional symmetry metadata
    h.set_basis_convention(bloch_phase, bloch_ratom, angular_order,
                           negative_m_convention, positive_m_convention);
    h.set_symmetry_operations(n_symops, row_conv, rotmats, translations);

    // Loop over atoms and R cells to set LRI coefficients
    for (int i = 0; i < natoms; ++i) {
        for (int j = 0; j < natoms; ++j) {
            for (auto& R : r_cells) {
                h.set_lri_coeff(LIBRPA_ROUTING_AUTO, i, j, nbasis_i, nbasis_j, naux_mu, R.data(), Cs.data());
            }
        }
    }

    // Loop over k-points to set Coulomb matrices
    for (int ik = 0; ik < nkpts; ++ik) {
        h.set_aux_bare_coulomb_k_atom_pair(ik, i, j, naux_i, naux_j, Vq_real.data(), Vq_imag.data(), vq_threshold);
        // For std::complex<double> input, use
        // h.set_aux_bare_coulomb_k_atom_pair_packed(..., Vq.data(), vq_threshold);
    }

    // *** Compute RPA correlation energy ***
    std::vector<std::complex<double>> rpa_contrib_ibzk(nkpts_ibz);
    double Ec = h.get_rpa_correlation_energy(opts, rpa_contrib_ibzk);

    // *** Compute exact exchange ***
    h.build_exx(opts);
    std::vector<double> vexx = h.get_exx_pot_kgrid(opts, nspins, iks_this, i_state_low, i_state_high);

    // *** Compute G0W0 self-energy ***
    h.build_g0w0_sigma(opts);
    std::vector<std::complex<double>> sigc = h.get_g0w0_sigc_kgrid(opts, nspins, iks_this, i_state_low, i_state_high, vxc, vexx);
    librpa::G0W0QpeResult qpe = h.get_g0w0_qpe_kgrid(
        opts, nspins, iks_this, i_state_low, i_state_high, vxc, vexx);
    librpa::G0W0SpectralFunctionResult spectral =
        h.get_g0w0_spectral_function_with_sigc_kgrid(
            opts, nspins, iks_this, i_state_low, i_state_high, omegas_real, vxc, vexx);

    // *** Clean up ***
    h.free();
    librpa::finalize_global();

    MPI_Finalize();
}
```

### Fortran

```fortran
use librpa_f03
use mpi

implicit none

type(LibrpaOptions) :: opts
type(LibrpaHandler) :: h
integer :: ierr
integer :: comm, nkpts_ibz
complex(dp), allocatable :: contrib_ibzk(:)

call MPI_Init(ierr)
comm = MPI_COMM_WORLD

! Initialize LibRPA
call librpa_init_global()

! Set up options
call opts%init()
opts%tfgrids_type = LIBRPA_TFGRID_MINIMAX
opts%nfreq = 16
opts%gf_threshold = 1.0d-3

! Initialize handler
call h%init(comm)

! Input parsing
call h%set_scf_dimension(nspins, nkpts, nstates, nbasis)
call h%set_wg_ekb_efermi(nspins, nkpts, nstates, wg, ekb, efermi)
call h%set_latvec_and_G(lat, recplatt)
call h%set_kgrids_kvec(nk1, nk2, nk3, nkpts, kpoints, kweights)
call h%set_kq_mapping(nkpts, map_q_ks)
call h%set_basis_convention(bloch_phase, bloch_ratom, angular_order, &
                            negative_m_convention, positive_m_convention)
call h%set_symmetry_operations(n_symops, row_conv, rotmats, translations)
! ... more input calls ...

! Compute RPA correlation energy
allocate(contrib_ibzk(nkpts_ibz))
Ec = h%get_rpa_correlation_energy(opts, nkpts_ibz, contrib_ibzk)

! Clean up
call h%free()

! Finalize LibRPA
call librpa_finalize_global()

call MPI_Finalize(ierr)
```


## Native optical velocity from in-memory operators

The experimental C/C++ optical API accepts numerical real-space H/S/r blocks
through `LibrpaNativeVelocityInput` in `librpa_input.h`. Computation and result
access are declared in `librpa_compute.h`. The driver selects
files and converts units before this call; no filenames or producer presets
enter the library. Supply H in Hartree, S dimensionless, and position in Bohr.
The public header specifies Fourier, spinor and matrix-storage conventions.

`librpa_prepare_native_velocity` is collective over the handler communicator.
Only its rank zero supplies the borrowed operator views; other ranks pass a
null input pointer. The operators remain valid until the call returns. The
calculation preserves the loaded SCF energies, occupations and eigenvectors,
sets the velocity through the existing public setter, and optionally prepares
analytic head/wing. Pass numerical option 3, 4, or 53. For
option 3 or 53 with optical preparation, the shared Wc implementation consumes
the analytic wing automatically. For strict 2D, also supply the area and
auxiliary monopole normalization decoded from the matching producer export.

`librpa_get_optical_result` provides a noncollective, rank-local owning snapshot
of the computed velocity, frequency nodes, head and auxiliary wing matrices.
Its public `LibrpaOpticalResult` fields describe row-major, interleaved-complex
arrays and their dimensions. Read those fields directly or obtain a borrowed
copy with `librpa_get_optical_view`; both refer to the same read-only arrays.
The snapshot survives changes to, or destruction of, its handler. Release only
the original result with `librpa_delete_optical_result`; do not free individual
arrays or delete a copied view. A snapshot does not gather a global
matrix from distributed ranks. The C++ Handler wrappers preserve the same
collective and ownership rules. These optical convenience functions currently
have C and C++ interfaces; no high-level Fortran wrapper is advertised.

## QSGW task API

The experimental C API declares QSGW controls and their initialization in
`librpa_options.h`, numerical input structures and their setter in `librpa_input.h`,
and task execution, output views and callback registration in `librpa_compute.h`.
The latter creates an opaque task with `librpa_create_qsgw_task`;
`Handler::create_qsgw_task` provides C++ ownership.
Initialize its controls with `librpa_init_qsgw_controls`, prepare its numerical
state, supply Vxc/native-operator buffers with `librpa_set_qsgw_input`, then run
`librpa_run_qsgw_task`. The same loaded grid/path and EXX/GW options apply.
Vxc and native operators are supplied on root and copied by the setter. All
collective task calls involve the handler communicator. Keep the handler alive
until task deletion. After preparation preserve the SCF eigenframe/grid; finish
band/Coulomb/velocity input before running. Do not modify or reenter the handler
during a run or its output callbacks.

An optional rank-zero output sink receives borrowed text and localized operator
views valid only during the callback; it may present or save those results.
Numerical iteration, projection, state updates and checkpoint restore/save occur
inside the library. Supply ordinary producer data numerically, using the existing
input APIs. The task reads only its own checkpoints. This task interface does
not yet have a Fortran wrapper.

`librpa_get_meanfield_dimensions` is a noncollective, read-only query for
preparing SCF-grid or band-path matrix buffers. QSGW controls, checkpoint
behavior, units and current numerical limits are described in [QSGW](qsgw.md).
