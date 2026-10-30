# Changelog

## [Unreleased]

## [0.9.0] - 2026-10-31

### Breaking changes

- Replaced `task = qsgw_band` with `task = qsgw` for grid and band-path
  calculations. Used band-path input metadata to select the optional band channel
  ([#67](https://github.com/AESM-Group/LibRPA/pull/67)).
- Renamed `qsgw_band0_*` controls to `qsgw_unoccupied_keep`, `qsgw_cut_mode`,
  and `qsgw_cut_shift_ha`. Assigned the shifted-reference cut to mode 1 and rejected the former mode 2.
  Changed the default cut shift to 2 eV. Kept explicit
  `qsgw_cut_shift_ha` values in Hartree
  ([#67](https://github.com/AESM-Group/LibRPA/pull/67)).
- Enabled complete native real-space QSGW Hamiltonian export by default. Required
  `qsgw_export_hamiltonian_for_pyatb = false` for inputs without matching
  native ABACUS H(R), S(R), and r(R) files
  ([#67](https://github.com/AESM-Group/LibRPA/pull/67)).
- Enabled QSGW Hartree density feedback by default. Required dedicated
  `prefix_coul_hartree` and `prefix_lri_coeff_hartree` inputs for this feature.
  Kept fixed-Hartree calculations available through
  `qsgw_update_hartree = false`
  ([#73](https://github.com/AESM-Group/LibRPA/pull/73)).
- Updated `input_preset = abacus` to current reader-v1 filenames and prefixes.
  Kept older exports accessible through `abacus-legacy` or explicit filename overrides
  ([#49](https://github.com/AESM-Group/LibRPA/pull/49)).
- Removed the separate bundled `thirdparty/LibRI_GPU` tree and unified CPU/GPU
  builds under `thirdparty/LibRI`. Used LibDDLA for CUDA and HIP tensor
  contractions. Required external GPU-enabled LibRI headers to support
  `__GPU_RI`, the DDLA backend, and `RI::LRI_Cal_Mode::GPU`.
  Required LibRI and one GPU backend for `LIBRPA_USE_LIBRI_GPU`.
  Kept CPU tensor contractions when this option is off
  ([#76](https://github.com/AESM-Group/LibRPA/pull/76)).
- Stopped installing the experimental `file_reader.hpp` and
  `librpa_file_reader.hpp` headers and the `librpa_file_reader` library. The
  headers include headers under `src/` and work only from a LibRPA source tree.
  Built `librpa_file_reader` as a static library with position-independent code,
  so the installed driver does not depend on it at run time
  ([#78](https://github.com/AESM-Group/LibRPA/pull/78)).

### Added

- Added experimental `task = crpa` for bare, partially screened, and fully
  screened onsite interaction tensors, with independent response and local-orbital
  spaces, band or orbital-projector response weights, and scalar/spinor support.
  Added orbital-character and PDOS analysis, G0W0/QSGW coupling, and local-frame
  tracking with QSGW checkpoint continuation. Extended
  `option_dielect_func = 6` to native optical calculations in cRPA
  ([#71](https://github.com/AESM-Group/LibRPA/pull/71)).
- Extended experimental fixed-basis QSGW with checkpoint restart, per-update
  quasiparticle solutions, and live same-grid analytic head/wing updates.
  Added `option_dielect_func = 6` for current-HR QSGW velocities and a public
  C/C++ task API available without the standalone driver. Added regression
  checks for iteration and restart workflows
  ([#67](https://github.com/AESM-Group/LibRPA/pull/67)).
- Added experimental standalone periodic Hartree AO matrices through LibRI.
  Limited standalone Hartree and QSGW Hartree feedback to a complete regular
  Gamma-containing 3D SCF grid with `bloch_ratom = 0`. Required a dedicated
  bare q=0 Coulomb metric with G=0 set to zero
  ([#73](https://github.com/AESM-Group/LibRPA/pull/73)).
- Added native optical velocities from H(R), S(R), and r(R), public velocity
  preparation APIs, and driver preset `option_dielect_func = 6` for G0W0
  analytic head/wing calculations. Added numerical checks for native velocities
  ([#70](https://github.com/AESM-Group/LibRPA/pull/70)).
- Added explicit `spin_symmetry` input and spatial/spin operation tables
  for SOC symmetry restoration
  ([#43](https://github.com/AESM-Group/LibRPA/pull/43)).
- Added unitary/antiunitary magnetic reconstruction for SOC exchange, optical
  head/wing calculations, and cRPA. Validated compatible states and
  correlated subspaces
  ([#72](https://github.com/AESM-Group/LibRPA/pull/72)).
- Added experimental strict-2D analytic head/wing screening and Gamma-cell
  RPA integration. Documented the reader-v1 `librpa_2d_coulomb_head.txt`
  contract for complete Wc calculations. Aligned Gamma body, wing, and
  Coulomb legs in the same eigenbasis and corrected Wc redistribution
  ([#44](https://github.com/AESM-Group/LibRPA/pull/44)).
- Added compact serial/MPI regressions for strict-2D complete Wc and
  Delta-Sternheimer head/wing calculations
  ([#49](https://github.com/AESM-Group/LibRPA/pull/49)).
- Added experimental `task = sternheimer_rpa` for Delta-Sternheimer RPA input
  and response reconstruction from aggregate or partial producer data, with
  symmetry routes and external frequency grids. Kept the consumer-side v1
  input format provisional
  ([#45](https://github.com/AESM-Group/LibRPA/pull/45)).
- Added s-subspace dielectric-head replacement (`option_dielect_func = 5`)
  and analytic head/wing correction (`53`). Limited option `53` to the
  ScaLAPACK Wc path for 3D G0W0 grid and band calculations
  ([#64](https://github.com/AESM-Group/LibRPA/pull/64)).
- Added `output_exx_mat_r` for binary real-space NAO EXX blocks, with
  MPI-independent filenames. Reused the Sigma_c real-space block
  writer and checkpoint layout for EXX
  ([#48](https://github.com/AESM-Group/LibRPA/pull/48)).
- Added `n_bands_exclude_chi0` and `n_bands_exclude_sigc` to independently
  exclude the lowest input bands from response and correlation self-energy
  Green functions. Set both defaults to zero and left EXX unchanged
  ([#41](https://github.com/AESM-Group/LibRPA/pull/41)).
- Added optional explicit full-grid k-point mappings to the BZ input format
  ([#53](https://github.com/AESM-Group/LibRPA/pull/53)).

### Changed

- Updated the design diagrams and added architecture views, with LaTeX and
  PlantUML sources and build targets to regenerate the figures
  ([#75](https://github.com/AESM-Group/LibRPA/pull/75)).
- Removed the obsolete `gw_old` and `meanfield_old` source and header files
  ([#77](https://github.com/AESM-Group/LibRPA/pull/77)).
- Updated bundled LibRI to `a0661cb`
  ([#74](https://github.com/AESM-Group/LibRPA/pull/74)).
- Enabled CPU fine-grained locking when supported by the selected LibRI
  headers
  ([#50](https://github.com/AESM-Group/LibRPA/pull/50)).
- Moved QSGW iteration state and checkpoint handling into the library core.
  Kept input parsing and output formatting in the driver
  ([#67](https://github.com/AESM-Group/LibRPA/pull/67)).
- Strengthened regression comparisons to reject nonfinite results and expanded
  symmetry, MPI, and input-contract checks
  ([#63](https://github.com/AESM-Group/LibRPA/pull/63)).

### Fixed

- Fixed allocator cleanup in bundled LibRI for macOS
  ([#76](https://github.com/AESM-Group/LibRPA/pull/76)).
- Fixed driver test linking with MPI and math libraries in shared-library builds
  ([#77](https://github.com/AESM-Group/LibRPA/pull/77)).
- Used external frequency nodes directly for Delta-Sternheimer head/wing
  calculations, avoiding redundant minimax-grid generation
  ([#55](https://github.com/AESM-Group/LibRPA/pull/55)).
- Corrected Gamma screening to subtract the retained Coulomb projector
  after channel filtering. Added numerical checks for filtered-Coulomb screening
  ([#70](https://github.com/AESM-Group/LibRPA/pull/70)).
- Corrected distributed head/wing layouts and RPA log-determinant handling
  ([#51](https://github.com/AESM-Group/LibRPA/pull/51)).
- Corrected strict-2D Gamma-cell RPA integration by integrating the
  trace-log radially
  ([#62](https://github.com/AESM-Group/LibRPA/pull/62)).
- Fixed Padé construction for zero or repeated samples using a checked
  barycentric fallback that preserves the selected samples and Padé degree
  class. Corrected scalar distributed matrix powers and rejected nonfinite
  minimax grids and transform coefficients before use. Added numerical
  checks for these paths
  ([#70](https://github.com/AESM-Group/LibRPA/pull/70)).
- Corrected SOC charge vertices and reversed Green-function buffers
  ([#69](https://github.com/AESM-Group/LibRPA/pull/69)).
- Corrected k-star wavefunction rotations and supercell symmetry handling
  ([#54](https://github.com/AESM-Group/LibRPA/pull/54)).
- Restored supercell q-star Coulomb reconstruction and ensured unique
  Coulomb ownership across MPI ranks
  ([#59](https://github.com/AESM-Group/LibRPA/pull/59)).
- Corrected the operator transformation of real-space self-energy
  ([#56](https://github.com/AESM-Group/LibRPA/pull/56)).
- Restored occupations using per-IBZ k-point weights
  ([#52](https://github.com/AESM-Group/LibRPA/pull/52)).
- Canonicalized BvK k-point coordinates
  ([#40](https://github.com/AESM-Group/LibRPA/pull/40)).
- Handled empty local wavefunction blocks in scalar and spinor k-point/BLACS
  calculations
  ([#60](https://github.com/AESM-Group/LibRPA/pull/60)).
- Rejected symmetry-reduced SCF k-grids when `use_symmetry_exx = false`
  instead of producing incomplete exchange results
  ([#68](https://github.com/AESM-Group/LibRPA/pull/68)).
- Applied `n_bands_chi0` and `n_bands_sigc` to response and correlation
  self-energy Green functions across serial and k-point/BLACS-parallel routes.
  Rejected band upper bounds exceeding the input size. Preserved
  k-parallel blocks, zero-response frequency entries,
  and one-sided or zero self-energies when bands are excluded
  ([#41](https://github.com/AESM-Group/LibRPA/pull/41)).
- Adapted scalar and spinor symmetry restoration helpers to band masks
  and cutoffs
  ([#43](https://github.com/AESM-Group/LibRPA/pull/43)).
- Fixed distributed Wc Hermitian completion and compressed auxiliary-basis
  unfolding
  ([#42](https://github.com/AESM-Group/LibRPA/pull/42)).
- Fixed distributed Wc export scheduling to keep MPI collectives aligned
  across ranks with different local frequency data
  ([#43](https://github.com/AESM-Group/LibRPA/pull/43)).
- Fixed out-of-bounds access in binary v1 LRI and Coulomb readers when input
  files contain zero data blocks
  ([#39](https://github.com/AESM-Group/LibRPA/pull/39)).
- Corrected the ABACUS preset Vxc filename
  ([#53](https://github.com/AESM-Group/LibRPA/pull/53)).

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

[Unreleased]: https://github.com/AESM-Group/LibRPA/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/AESM-Group/LibRPA/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/AESM-Group/LibRPA/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/AESM-Group/LibRPA/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/AESM-Group/LibRPA/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/AESM-Group/LibRPA/compare/v0.4.0-gw-benchmark...v0.5.0
[0.4.0-gw-benchmark]: https://github.com/AESM-Group/LibRPA/compare/v0.3.0...v0.4.0-gw-benchmark
[0.3.0]: https://github.com/AESM-Group/LibRPA/compare/v0.01...v0.3.0
[0.01]: https://github.com/AESM-Group/LibRPA/releases/tag/v0.01
