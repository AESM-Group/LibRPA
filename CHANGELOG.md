# Changelog

## [Unreleased]

### Added

- Added `output_exx_mat_r` for binary real-space NAO EXX blocks. EXX and Sigma_c
  share a typed, indexed binary layout and write one file per MPI rank and channel.

- Documented the reader-v1 `librpa_2d_coulomb_head.txt` contract used by
  strict-2D complete-Wc calculations.
- Added a small serial/MPI regression for the strict-2D production Wc path.

- Added `n_bands_exclude_chi0` and `n_bands_exclude_sigc` to independently
  exclude the lowest input bands from response and correlation self-energy
  Green's functions. Both default to zero and leave EXX unchanged.

### Deprecated

- Deprecated legacy `SigcRF_*.dat` checkpoints. Restart still reads them with a
  warning; new checkpoints use the indexed `.bin` format. Legacy reading will
  be removed in a future release.

### Fixed

- Kept strict-2D Gamma body, wing, and Coulomb legs in the same eigenbasis,
  aligned the MPI wing layout, and restored the returned Wc redistribution.

- Applied `n_bands_sigc` to correlation self-energy Green functions in serial,
  k-point/BLACS parallel, and symmetry-restored routes; rejected upper bounds
  exceeding the number of input bands.

- Applied the `n_bands_chi0` mask consistently in serial, k-point/BLACS
  parallel, and symmetry-restored response Green-function construction.
  Preserved k-parallel Green-function blocks in the atom-pair/R-tau routes
  and zero-response frequency entries in the LibRI route.
- Preserved one-sided imaginary-time self-energy contributions and handled
  identically zero self-energy in analytic continuation when bands are excluded.

- Fixed out-of-bounds access in binary v1 LRI and Coulomb readers when input
  files contain zero data blocks.

## [0.8.0] - 2026-09-23

### Breaking changes

- Changed `output_gw_sigc_mat_kf` NAO self-energy exports from Matrix Market
  `.mtx` to dense binary `.bin`. Update readers of `SigcKF_*` output to the
  format documented in `librpa_options.h`.
- Changed the default number of time/frequency points, `nfreq`, from 6 to 16.

### Added

- Added experimental fixed-basis `qsgw` and `qsgw_band` tasks with native
  FHI-aims/ABACUS Vxc input, Hamiltonian mixing and truncation, and iteration
  diagnostics. Same-grid analytic head correction is supported; Hartree,
  independent-grid head, and wing updates remain unsupported.
- Added `input_preset` for `fhi-aims` (alias `aims`), `abacus`, and
  `abacus-legacy`, plus configurable `prefix_velocity`. The legacy preset
  retains unsuffixed ABACUS filenames and the ABACUS velocity reader.
- Added `output_exx_mat_k` for dense binary NAO-basis EXX matrices on the
  k-grid and band path, including spinor blocks.
- Added binary regression-data extraction with selectable field groups.
- Added detailed profiling of Green-function construction, response
  collection, matrix operations, and MPI waits.

### Changed

- Changed CPU LibRI EXX, response, and GW work distribution to use weights
  based on the number of basis functions on each atom, and updated bundled LibRI.
- Changed driver file-reader organization through a major refactor, moving
  shared readers into `driver/reader` and the experimental
  `librpa_file_reader` library, independent of driver state and available with
  `LIBRPA_ENABLE_DRIVER=OFF`. The `file_reader.hpp` API lets external programs,
  such as LibBSE, reuse LibRPA dataset formats and file parsing, with configurable
  input filenames and prefixes.

### Fixed

- Fixed Gamma head/wing corrections for filtered Coulomb subspaces,
  including the case with only the head channel retained, and avoided forming
  the full body projector during matrix construction.
- Fixed missing ELSI CSC size and index validation and unaligned value reads.
- Fixed missing KS, EXX, and GW band-edge and gap analysis on the k-grid and band
  path, retaining the mean-field occupied/empty band partition and adding
  warnings for higher empty GW states below the mean-field Fermi level.
- Fixed suppressed RPA, EXX, and G0W0 result tables, timing summaries, and
  MPI/build information at `warning` and `critical` output levels.
- Fixed output-level filtering for structural and time/frequency-grid
  diagnostics and collective messages.
- Fixed missing propagation of library feature definitions to CMake
  consumers, keeping exposed C++ data types consistent with the library build.

## [0.7.0] - 2026-08-02

### Breaking changes

- API `librpa_set_kgrids_kvec` changed to accept the number of loaded SCF
  k-points explicitly and optional k-point weights. C++/Fortran bindings
  changed accordingly: `set_kgrids_kvec`.
- API `librpa_set_ibz_mapping` was renamed to `librpa_set_kq_mapping`.
  C++/Fortran bindings changed accordingly: `set_ibz_mapping` was renamed to
  `set_kq_mapping`.

### Added

- New public APIs:
  - Global output control: `librpa_set_output_level` and
    `librpa_get_output_level`.
  - Restart setup: `librpa_set_restart_from_dir`.
  - Auxiliary-basis and symmetry metadata: `librpa_set_ao_basis_aux_shrink`,
    `librpa_set_basis_convention`, and `librpa_set_symmetry_operations`.
  - Packed-complex Coulomb input:
    `librpa_set_aux_bare_coulomb_k_atom_pair_packed`,
    `librpa_set_aux_cut_coulomb_k_atom_pair_packed`,
    `librpa_set_aux_bare_coulomb_k_2d_block_packed`, and
    `librpa_set_aux_cut_coulomb_k_2d_block_packed`.
  - Velocity-matrix input: `librpa_set_velocity_matrix` and
    `librpa_set_velocity_matrix_packed`.
  - GW quasiparticle and spectral-function retrieval:
    `librpa_get_g0w0_qpe_kgrid`, `librpa_get_g0w0_qpe_band_k`,
    `librpa_get_g0w0_spectral_function_kgrid`, and
    `librpa_get_g0w0_spectral_function_band_k`.
- Added CUDA and HIP acceleration for EXX, RPA, GW, and head/wing calculations
  through LibDDLA, GPU-enabled LibRI, and optional ELPA routines.
- Added support for bundled or external ELPA and bundled or external LibDDLA
  installations.
- Added two-level k-point/BLACS parallelism for eigenvector redistribution,
  density and Green functions, EXX, response, and GW calculations.
- Added versioned binary readers for Coulomb matrices, LRI coefficients, KS
  eigenvectors, and velocity matrices, with automatic format detection and
  legacy-data conversion utilities.
- Added selectable iterative, quasi-Newton, and perturbative quasiparticle
  solvers, adaptive damping, fallback output, Hedin's shift, and independent
  analytic-continuation grids and resampling.
- Added symmetry-operation input APIs, basis-shell metadata, k-star
  reconstruction, and configurable Born-von Kármán remapping.
- Added analytic GW and RPA head/wing workflows, response-collection chunking,
  and sparse-block handling.
- Added restart loading from a separate directory and outputs for atom-pair
  screened interactions, k-frequency self-energies, ranged EXX/self-energy
  matrices, and binary spectral functions.

### Changed

- Changed driver input handling to auto-detect supported dataset formats and
  exposed input filenames and prefixes as runtime parameters.
- Changed the default Born-von Kármán remapping to consider multiple periodic
  images.
- Changed the iterative quasiparticle solver to residual mixing and tightened
  its default convergence threshold.
- Improved memory use and scaling in response Fourier transforms, screened
  interactions, self-energy rotation, and symmetry-enabled calculations.
- Made the driver fail cleanly on missing input files, unknown tasks, and
  initialization or task exceptions while finalizing LibRPA and MPI as applicable.
- Made the required MPI thread-support level configurable at build time and
  added startup reporting of the requested and provided levels.

### Deprecated

- Deprecated `task = g0w0_band`; use `task = g0w0`. The old value remains an
  accepted alias.
- Deprecated the monolithic `basis_out`, `symrot_k`, and
  `irreducible_sector.txt` inputs in favor of split basis data and parsed
  symmetry operations.
- Retained several renamed runtime parameters as compatibility aliases; the
  preferred names are listed in the runtime-parameter documentation.

### Fixed

- Fixed symmetry reconstruction and k-parallel execution for EXX, RPA, GW,
  spinor wavefunctions, and head/wing corrections, including an MPI hang.
- Fixed head/wing normalization, shrink-basis selection, GPU execution, and
  dielectric-solver error reporting.
- Fixed small-basis calculations with large MPI task counts, non-LibRI
  response routes, distributed wavefunction layouts, and multiple input/output
  regressions.

## [0.6.0] - 2026-05-25

### Added

- Extended spin-orbit coupling and spinor-wavefunction support across input
  readers, public APIs, response construction, EXX, GW, and band calculations.
- Added experimental head/wing corrections, including spin-polarized and
  two-dimensional dielectric workflows.
- Added experimental auxiliary-basis and response compression, followed by
  screened-interaction unfolding.
- Added QSGW backend and task infrastructure for future development. The QSGW
  task remained disabled in this release.
- Introduced a regression-test backend with new ABACUS and FHI-aims cases,
  result comparisons, labels, and MPI smoke coverage.
- Added new runtime controls, including minimax regulation and PyATB input
  selection.

### Changed

- Updated the C++, C, and Fortran interfaces and their documentation for the
  spinor workflows.
- Replaced the GPL Lebedev quadrature component with a local implementation
  under a compatible LGPL license.
- Cleaned up obsolete code paths and third-party components.

### Fixed

- Corrected spinor Green-function, response, EXX, self-energy, and
  quasiparticle calculations.
- Fixed several distributed-matrix, MPI, input-reading, and non-LibRI build
  failures.

## [0.5.0] - 2026-03-28

### Added

- Added C/C++ and Fortran APIs for packed wavefunctions, band-path data, EXX
  band potentials, G0W0 self-energies, and quasiparticle energies.
- Added atom-pair-to-BLACS and BLACS-to-atom-pair matrix redistribution with
  reusable index scheduling.
- Added k-parallel Green-function construction and self-energy rotation for
  G0W0 calculations.
- Added APIs for imaginary-frequency grids and dielectric-function input.
- Added per-process profiling output and embedded Git reference information.

### Changed

- Migrated the EXX and G0W0 driver tasks to the public API workflows.
- Bundled LibRI and LibComm instead of requiring them as Git submodules.
- Overhauled the installation, driver, API, tutorial, and reference
  documentation.
- Improved OpenMPI and ScaLAPACK discovery and documented `SCALAPACK_DIR`.

### Fixed

- Corrected quasiparticle energies with multiple MPI tasks and without
  k-distributed eigenvectors.
- Fixed failures when the MPI task count exceeded the number of atom pairs.
- Fixed non-square process-grid handling and several Fortran-binding issues.

## [0.4.0-gw-benchmark] - 2026-03-18

This tag is the benchmark snapshot used for the LibRPA GW paper.

### Added

- Added EXX and G0W0 band-structure workflows, including band-path input,
  k-grid quasiparticle energies, and KS/EXX/GW band output.
- Added spectral-function calculations based on analytic continuation, with an
  optional imaginary-frequency shift.
- Added automatic binary-format detection and readers for coefficient and
  Coulomb-matrix inputs.
- Added a configurable driver input directory and general ELSI CSC matrix
  readers and writers.
- Expanded regression coverage for atoms, molecules, and periodic solids.

### Changed

- Reduced memory use in Green-function, response, screened-interaction, and
  self-energy calculations and improved their MPI/OpenMP scalability.
- Optimized Fourier transforms, atom-pair redistribution, and coefficient and
  Coulomb-matrix input.
- Adapted LibRPA to the updated LibRI RPA/GW interface.

### Fixed

- Corrected EXX/GW basis rotations when the band and state counts differ.
- Fixed band interpolation, distributed indexing, and small-system behavior
  with large MPI task counts.
- Fixed multiple memory leaks and non-LibRI build regressions.

## [0.3.0] - 2024-11-22

### Added

- Added RPA correlation-energy, EXX, and initial G0W0 quasiparticle workflows,
  including analytic continuation and quasiparticle-equation solvers.
- Added public C/C++ APIs and a Fortran binding for initializing LibRPA,
  supplying mean-field data, and retrieving RPA and EXX results.
- Added ScaLAPACK two-dimensional block-matrix infrastructure and LibRI
  atom-pair routing with MPI/OpenMP parallelism.
- Added binary coefficient input, real-space screened-interaction output, and
  driver runtime controls.
- Added Sphinx and Doxygen documentation, GitHub Pages deployment, tutorials,
  and regression tests.

### Changed

- Reorganized LibRPA into a shared library, driver, bindings, and tests.
- Adopted CMake as the supported build system and added platform-specific build
  examples.
- Integrated LibRI, LibComm, cereal, and GreenX time-frequency grids.

### Fixed

- Corrected numerous response, RPA, EXX, GW, time-frequency transformation,
  and parallel-routing errors.
- Fixed macOS builds, non-LibRI builds, output redirection, and binary-input
  behavior.

## [0.01] - 2024-04-16

Initial tagged snapshot.

### Added

- Added the initial standalone space-time response and RPA correlation-energy
  workflow with an FHI-aims file interface.
- Added MPI/OpenMP execution, minimax time-frequency transformations, LibRI
  response construction, and ScaLAPACK-based RPA calculations.
- Added CMake and Make build paths, runtime parameter parsing, profiling,
  regression cases, and initial continuous integration.

### Fixed

- Corrected early Green-function, response, spin, and distributed RPA-energy
  calculations.

[Unreleased]: https://github.com/AESM-Group/LibRPA/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/AESM-Group/LibRPA/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/AESM-Group/LibRPA/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/AESM-Group/LibRPA/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AESM-Group/LibRPA/compare/v0.4.0-gw-benchmark...v0.5.0
[0.4.0-gw-benchmark]: https://github.com/AESM-Group/LibRPA/compare/v0.3.0...v0.4.0-gw-benchmark
[0.3.0]: https://github.com/AESM-Group/LibRPA/compare/v0.01...v0.3.0
[0.01]: https://github.com/AESM-Group/LibRPA/releases/tag/v0.01
