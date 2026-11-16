# Experimental fixed-basis QSGW

Use `task = qsgw` for both molecular and periodic calculations. The library task
constructs self-energies in the immutable initial KS basis and updates the
eigenvalues and wavefunctions each iteration.
Its selected Mode B staticization evaluates the fixed-reference diagonal
entries at the current state energies and the off-diagonal entries at the
current Fermi energy; see the source-only note `docs/theory/qsgw/staticization.md`.
This fixed-reference prescription is retained throughout iteration and restart.
It is distinct from applying Mode B in the current eigenbasis and rotating the
result back, even with exact analytic continuation.
The effective Hamiltonian replaces the initial DFT exchange-correlation
potential with exchange and a Hermitian correlation potential. Hartree
density feedback is enabled by default and described below. Same-grid analytic head and wing updates are supported
with an explicit velocity convention; independent-grid updates are rejected.

QSGW reuses the G0W0 input readers and EXX/Sigma kernels. The driver reads
producer files and calls the public opaque-task API in `librpa_compute.h`, using
controls from `librpa_options.h` and numerical input from `librpa_input.h`.
Iteration, numerical state and checkpoint handling live in `src/core/qsgw`;
the driver's output callback writes the requested tables and native CSR files.
The task API is available without building the command-line driver.

## Hartree density feedback

`qsgw_update_hartree` defaults to `true`. Every update includes the change in
the physical Hartree potential and requires the following dedicated inputs.
To retain a fixed-Hartree trajectory or use inputs without these exports, set
`qsgw_update_hartree = false` explicitly:

```ini
qsgw_update_hartree = true
parallel_routing = libri
prefix_coul_hartree = v1_coulomb_hartree_
prefix_lri_coeff_hartree = v1_Cs_hartree_shrinked_data
use_shrink_abfs = true
```

The target Hamiltonian becomes
`H_DFT - Vxc_DFT + V_H[D_t-D_DFT] + Sigma_x,t + Vc,t`. The density is rebuilt
from the entering eigenvectors and their physical k-weighted occupations,
summing spin channels and diagonal spinor components. The Hartree difference
is the same scalar spatial AO operator for both collinear spin channels. The
initial DFT state has zero density difference. This adds feedback between QSGW
updates; it does not add an inner Hartree SCF loop or change the mixing/cut rules.

`prefix_coul_hartree` must contain the matched **bare three-dimensional q=0
auxiliary metric with G=0 set to zero**, in Hartree. Its zero mode fixes the
background convention; the driver applies no potential shift or density fit.
The ordinary full/cut Coulomb files are still required for GW and remain in
their original storage. Hartree input never replaces those GW matrices.

`prefix_lri_coeff_hartree` must contain both directed LRI fitting endpoints
with their **unwrapped physical lattice translations**, at full output
precision, in the existing binary v1 format. These dedicated full-precision inputs are kept separate from ordinary GW
coefficients. Hartree path matrices are interpolated from the mesh operator
using the same `option_bvk_remap` convention as EXX; no auxiliary-potential
extension to LibRI is required.
The raw coefficients and dedicated metric must match the auxiliary basis
selected by `use_shrink_abfs`. For a full-basis run, use the producer's full
`v1_Cs_hartree_data` family instead of the shrunk family shown above. Both
dedicated prefixes are required whenever feedback is enabled; neither has a
fallback to the ordinary GW input.

This first periodic feedback route requires LibRI, a complete regular Gamma-containing
three-dimensional SCF grid, and `bloch_ratom=0`; the library task checks these
constraints collectively. AUTO selects LibRI before data
input; explicit RTAU and atom-pair routes are rejected collectively. Supply the
matching band-path E/C/Vxc payload when the path channel is enabled.

`qsgw_hartree.csv` reports the entering/updated Hartree difference for the grid
and optional band channel, with AO and projected maximum/Frobenius diagnostics
in eV. Enabling feedback does not establish material accuracy or convergence.

For library clients, `LibrpaQsgwControls.update_hartree` enables the same task
through the C API and the C++ `Handler::create_qsgw_task` wrapper. Supply the
dedicated root matrix and `LibrpaQsgwLriBlock` array in `LibrpaQsgwInput`; the
task copies these inputs independently of the handler's GW data. The existing
Fortran module exposes standalone Hartree calculation but does not yet expose
the opaque QSGW task API.

## Quasiparticle roots during iteration

To also solve the diagonal G0W0 quasiparticle equation on each entering state:

```ini
task = qsgw
qsgw_output_qp = true
```

The default is `true`; use `qsgw_output_qp = false` to disable these diagnostics.
Results are written to `output_dir/qsgw_qp.csv` for the
SCF grid and, when present, the band path. State 0 is the original DFT reference;
GW update 1 evaluates its ordinary G0W0 roots. Update `t+1` evaluates roots for
entering state `t`, using that state's energies, Fermi level and eigenvectors.
The newly produced final state has no QP rows until it enters another GW build;
the driver does not perform an extra final self-energy calculation for this output.

The collected exchange and frequency-dependent Sigma matrices are rotated from
the initial KS basis to the current eigenbasis. The subtracted static potential is
`Vstatic_t = H_t - H_DFT + Vxc_DFT - V_H[D_t-D_DFT]` when Hartree feedback is
enabled (the last term is zero otherwise), including the actual mixing/cut correction.
The diagonal equation is therefore
`E = epsilon_t - Vstatic_t + Sigma_x,t + Re Sigma_c,t(E - mu_t)`.
The initial state uses the original uncut DFT operator. This output reuses the
G0W0 continuation, resampling, Hedin-shift and `option_qpe_solver` settings; it
does not rebuild W or alter QSGW staticization, mixing, cuts or occupations.
The full QSGW Vxc input is sufficient; no separate diagonal `vxc_out` is needed.

The CSV records the reference and QP energies, static potential, exchange,
correlation and `solver_info` (`0` means solver success), all energies in eV.
Failed roots remain NaN unless `override_qpe_solver_nan = true` requests the
last unconverged value; their nonzero status is still retained. For perturbative
solver 2 the real correlation column is the same effective value used by the
G0W0 API to reconstruct the renormalized QP energy. `i_state_low`/`i_state_high`
select the zero-based, upper-exclusive band range. Restart output starts at the
restored entering state and recovers its saved eigenvector gauge.

With a band path, each GW build additionally writes
`output_dir/GW_band_spin_S_N.dat`, where `S` is the one-based spin channel
and `N` is the update number. The native table contains point index,
fractional coordinates, and occupation/energy pairs for the selected band
window; energies are in eV. Its QSGW-owned formatter preserves the ordinary G0W0 layout:
no header, point index width 5, fractional coordinates with seven decimal
places, and occupation/energy pairs with five decimal places (width 15).
Thus `GW_band_spin_1_1.dat` contains the initial
DFT-start G0W0 roots, while `QSGW_band_spin_1_1.dat` contains the newly
constructed QSGW state's eigenvalues. Update `N` roots belong to entering
state `N-1`. Later GW tables use that iteration's current eigenstates and
static subtraction rather than repeating DFT-start G0W0. NaN roots are
retained, with their solver status available in the CSV. Restarted runs
continue the update numbering. State indices, the selected one-based band
range and solver statuses remain available in the CSV rather than a custom
band-file header.

Band-path occupation columns in the QP CSV and all band tables retain the
original producer occupations. The path energies and Fermi level are updated,
but path occupation weights are not refilled. These columns therefore do not
identify the occupied subspace after a path state crosses the current Fermi
level. SCF-grid occupations are updated each iteration and determine screening.

Every enabled QP update also writes `output_dir/energy_qp_N` on the SCF
grid in the ordinary G0W0 `energy_qp` layout. Its header, k-point and
spin blocks, band numbering and occupation convention match G0W0, with
columns `state`, `occ_num`, `e_gs(Ha)`, and `e_qp(Ha)`. The energies use
scientific notation with ten decimal places. Here `e_gs` is the entering
state's mean-field energy. This per-update file is controlled by
`qsgw_output_qp`; ordinary G0W0 retains its separate `output_energy_qp`
switch and unnumbered filename. With no band path, the grid CSV and
`energy_qp_N` are still written.

## Optional band-path channel

The driver selects the band channel from the input files. If
`fn_band_kpath_info` (default `band_kpath_info`) exists under `input_dir`, QSGW
runs the full existing band-path channel alongside the SCF grid. Supply the
matching band energies, wavefunctions, and **full** band Vxc matrices; the
metadata file does not substitute for these inputs. Missing or inconsistent
band data are errors. If the metadata file is absent, QSGW runs only the SCF
grid. A periodic calculation without a band path is valid, and the driver
does not infer a path from molecular or periodic geometry.

`qsgw` is the only QSGW task name. Older inputs containing
`task = qsgw_band` must change that value to `task = qsgw`; the former task has
been removed. Band-path selection still follows the input metadata. Option 6
uses the unchanged current-HR prescription described below.

`qsgw_unoccupied_keep`, `qsgw_cut_mode`, and `qsgw_cut_shift_ha` select the cut.
The former `qsgw_band0_*` names are removed and rejected. An explicitly selected cut
applies to the SCF grid and the optional band path, including grid-only runs.

## Vxc input

QSGW needs the full DFT Vxc matrix, including off-diagonal elements; the diagonal
Vxc tables used by G0W0 are insufficient. The driver reads the full matrices
directly from `input_dir`, using spin and k-point filenames with these defaults:

| Producer | SCF example (spin 1, k point 1) | Band example (spin 1, k point 1) |
| --- | --- | --- |
| FHI-aims | `xc_matr_spin_1_kpt_000001.csc` | `band_vxc_mat_spin_1_k_00001.csc` |
| ABACUS, one spin channel | `vxck1_nao.txt` | `band_vxck1_nao.txt` |
| ABACUS, two spin channels | `vxck1s1_nao.txt` | `band_vxck1s1_nao.txt` |
| ABACUS, SOC/noncollinear (`nspin=4`) | `vxck1s4_nao.txt` | `band_vxck1s4_nao.txt` |

No additional file list or manifest is required. Optional `prefix_vxc_scf` and
`prefix_vxc_band` override the prefixes shown above (`xc_matr`/`band_vxc_mat`
for FHI-aims and `vxc`/`band_vxc` for ABACUS). Prefixes are relative to
`input_dir` unless absolute. For a separate ABACUS band output directory,
`prefix_vxc_band = OUT.band/vxc` reads its native `vxck1_nao.txt`, etc., without
renaming; the default `band_vxc` prefix distinguishes files staged alongside
SCF input. Spin and k indices are one-based and must follow
the same ordering as the corresponding SCF or band reference. Vxc files do not
provide an independent k-coordinate check.
Gamma-only ABACUS output can omit the k suffix; the reader also accepts
`vxc_nao.txt`, `vxcs1_nao.txt`, or `vxcs4_nao.txt` for the corresponding mode.

Use `constants_choice = aims` for FHI-aims ELSI matrices in Hartree, and
`constants_choice = internal` for ABACUS complex text matrices in Ry, which are
converted to Hartree. The default `qsgw_vxc_basis = state` reads matrices in the
initial KS-state basis. This includes standard ABACUS `out_mat_xc` output,
despite its `_nao` filename suffix. Set `qsgw_vxc_basis = nao` only for genuine
ABACUS AO matrices; the driver then projects them with the reference
wavefunctions. This setting applies to both SCF and band matrices. Matrix
dimensions, finiteness and Hermiticity are checked on input.
For native ABACUS SOC `out_mat_xc`, keep the default state basis: the producer
writes the complete complex KS-state matrix. A genuine SOC AO input instead
has dimension `2 * N_AO`, ordered `(AO0 up, AO0 down, AO1 up, AO1 down, ...)`;
its explicit `nao` projection includes all four spinor blocks.

## Hamiltonian cut

`qsgw_cut_mode` accepts `0` (uncut, the default) and `1` (shifted DFT
reference diagonal, formerly mode 2). The former unshifted cut is removed;
the old numeric value `2` is rejected. For mode 1, the fixed-reference matrix retains the leading block whose size
is the current occupied-state count plus `qsgw_unoccupied_keep`
(default 10), bounded by the complete state dimension. Every other diagonal
is restored to its initial DFT energy plus `qsgw_cut_shift_ha`
(default 2 eV, converted to about 0.07349864435131 Ha); all other off-diagonal entries are zero. The matrix and
reference basis remain complete.

The selected cut applies when assembling the SCF-grid update and again after
mixing; an optional band-path update uses the same rule. The HR exporter
serializes this actual post-cut operator immediately before diagonalization.
To select the shifted-cut HR requested for topology, use:

```ini
qsgw_cut_mode = 1
```

HR export is enabled by default. The omitted keep and shift defaults are
10 empty states and exactly 2 eV under the selected constants. Explicit
`qsgw_cut_shift_ha` values remain in Hartree; its suffix states the input unit.
A fresh iteration `0000` remains the actual entering PBE
operator; update files `0001`, `0002`, etc. contain the
mode-1 shifted operator. The complete DFT/IFT preserves that operator on the original
SCF mesh, including the shifted high states. This identity does not prescribe
or establish interpolation accuracy at points outside the mesh.

## Analytic head and wing feedback

Explicit magnetic symmetry can be combined with current-HR velocity and coupled
cRPA. The live E/C/f state and cRPA endpoint weights must preserve every supplied
operation. Paired IBZ source E/C/f and Vxc are expanded internally into a complete
unshifted mesh, using the unitary/antiunitary ket action and reciprocal gauge.
Source occupations are normalized once to f/N_full; source Vxc remains paired
with its source KS gauge. Complete AO-spinor/state frames and all three symmetry
switches are required. This does not introduce another eigenproblem or accept
an arbitrary magnetic odd-frequency response.

Unpolarized scalar inputs originally supplied on the complete full mesh, with
complete square AO/state frames and an implicit spatial-operation table, retain
the historical spatial-symmetry reconstruction and direct/full-grid fallback
checks. The strict magnetic sewing check applies to explicit spin operations,
spinful storage and inputs expanded from paired IBZ sources. Finite E/C/f and
storage-shape checks apply to every live state. Retaining the historical scalar
route does not establish the numerical accuracy of its symmetry approximation.


The native-velocity preset moved from 5 to **6**. Use 6 for native QSGW
feedback. Value 5 follows upstream's input-dielectric s-subspace head
correction for G0W0; it is not an alias for native-velocity feedback.
The preset renumbering does not change the head/wing equations or feedback.

The standalone `qsgw` driver accepts `option_dielect_func = 6`
to recompute velocity internally from the complete current Hamiltonian and
then apply the existing analytic head/wing prescription at every iteration.
This preset selects `qsgw_headwing_velocity = current_hr` and resolves to
numerical option 3. An omitted selector or explicit `current_hr` is accepted;
explicit `reference` and `rotate_reference` are rejected for option 6. It
requires `replace_w_head = true` and `use_scalapack_gw_wc = true`.
The log records requested option 6 and effective option 3.
The native mode remains experimental.

Existing option 3 provides both analytic head and wing, and option 4 provides
head only, with `replace_w_head = true`. Option 3 requires
`use_scalapack_gw_wc = true`: the LAPACK Wc route does not consume analytic
wings and is rejected. Option 4 retains its head-only routes.
These options retain their explicit
`qsgw_headwing_velocity` selection:

| Value | Velocity used by each entering update |
| --- | --- |
| `reference` (default) | Legacy fixed initial matrix entries, option 4 only |
| `rotate_reference` | Fixed initial velocity operator rotated into the current eigenstates |
| `current_hr` | Velocity recomputed from the complete current AO Hamiltonian and current eigenpairs |

The last two choices clear the previous head/wing data and rebuild them from
the actual live energies, wavefunctions, occupations and frequency nodes before
each body/self-energy calculation. The analytic and body nodes are checked individually, including
when automatic minimax endpoints change. `rotate_reference` needs the original
same-grid velocity input. It does not differentiate the updated Hamiltonian.

For option 6, supply dimensionless native ABACUS `S(R)` and Bohr `r(R)` CSR
files, with their complete independent real-space supports, in addition to the
ordinary QSGW inputs:

```ini
task = qsgw
input_preset = abacus
replace_w_head = true
option_dielect_func = 6
use_scalapack_gw_wc = true
use_kpara_scf_eigvec = false
use_symmetry_rpa = false
use_symmetry_gw = false
use_symmetry_exx = false
```

The previous equivalent selection remains supported with the same inputs:

```ini
option_dielect_func = 3
qsgw_headwing_velocity = current_hr
```

`qsgw_fn_sr` defaults to `srs1_nao.csr` and `qsgw_fn_rr` to `rr.csr`; ordinary
ABACUS files need no filename overrides. Paths are relative to `input_dir`
unless absolute. This mode requires a full,
unshifted uniform mesh, replicated wavefunctions and a complete square AO/state
basis, with full-grid inputs or the paired IBZ expansion described above. The native SOC reader accepts a
complete two-component spinor reference with matching full-dimension S/r,
one physical spin channel and `2 * N_AO` states. Both reference and live
coefficients use AO-major/spinor-minor ordering; the entire complex Hamiltonian,
including cross-spinor entries, enters the same velocity projector. For this
SOC route requires explicit magnetic metadata when symmetry is enabled;
k-parallel input remains unsupported. S/r and the native DFT baseline remain fixed.
This mode also reads matching `hrs1_nao.csr` in Ry, plus `hrs2_nao.csr` for a
second physical collinear channel, even if HR export is disabled. At each
entering iteration, the driver embeds the complete reference-basis Hamiltonian
with the dual reference frame defined below, preserves the original DFT H(R),
and inverse-transforms
only `H_AO - H_DFT_native` on the input mesh. It differentiates this same
native-baseline reconstruction used by formal HR export, in the already
selected current E/C frame. Correction Nyquist images have equal weights.
A fresh first update uses the actual entering PBE operator; subsequent
updates use the preceding completed mixed/postcut operator. No independent
velocity file is used. The native DFT H must match the immutable reference
on the complete input mesh.

The implementation follows the executed PyATB H/S/r velocity convention, with
energies in Ha and Cartesian derivatives/position in Bohr. It checks the
reference and live S metrics, current eigenpair frame and Fourier round trip.
The AO basis and S/r operators remain fixed throughout QSGW.

The AO embedding uses the dual reference frame
`D = S C (C-dagger S C)^-1`, giving `H_AO = D h D-dagger`.
This preserves `C-dagger H_AO C = h` at finite export precision. The former
`S C h C-dagger S` expression instead projected back as `M h M`, with
`M = C-dagger S C`, amplifying small metric errors in high-energy bands.
The existing metric, native-reference and current-frame tolerances remain
unchanged; an inconsistent native DFT H(R) still fails its reference check.

For a modern scalar ABACUS export with matching dense SCF H(k)/S(k), small
CSR sparsification losses can be restored explicitly before staging inputs:

```sh
python3 utilities/restore_abacus_csr_precision.py \
  --hr OUT.material/hrs1_nao.csr --sr OUT.material/srs1_nao.csr \
  --dense-dir OUT.material --bz dataset/bz_sampling_out \
  --output-dir restored-csr
```

Use the resulting H/S files together, with the unchanged original SCF
eigenpairs, position operator and other dataset files. This utility adds
only the inverse-transformed dense-minus-CSR residual to the original
real-space model, retains native tails and splits even BvK boundary images
equally. It rejects discrepancies above `1e-8` in the native units (H in Ry),
checks Hermiticity and full-mesh closure, and writes new files without an
entry cutoff. H uses complex CSR values; scalar S retains real values for
the native and PyATB readers. Imaginary S(R) entries above `1e-11` are rejected;
closure is checked again after removing smaller rounding noise.
The dense matrices must cover the identical complete unshifted
mesh. This repairs mesh precision; the unique off-mesh range of discarded
terms cannot be recovered from mesh samples. It does not establish off-mesh
accuracy or replace material convergence tests.

The shared internal C++ kernel in `src/core/headwing_velocity.h` accepts
localized matrices rather than filenames. Its Fourier and upper-triangle
operations are shared through `src/math/localized_operator.h`. QSGW reconstruction
and live refresh remain in `src/core/qsgw/headwing_velocity.h`.
For model controls, its QSGW-specific
`solve_localized_velocity(H, operators, k_fractional, lattice_bohr)` performs
one generalized eigensolve and returns all `eigenvalues_ha`, AO-row/band-column
`ket_coefficients`, and three `velocity_ha_bohr` matrices in that same frame.
H contains full complex entries in Ha; S is dimensionless and r is in Bohr.
The lattice contains row vectors in Bohr, and fractional k uses
`exp(+i 2*pi*k.R)` while the derivative is Cartesian, in inverse Bohr.
S/dS and r use the documented upper-triangle completion, followed by
Hermitian upper-triangle completion of the band velocity. Native S/dS and
velocity diagonals are explicitly real, as in the existing live path;
PyATB retains complex diagonal roundoff. The consistency claim concerns
physically Hermitian models, with the residuals reported, and does not
extend to arbitrary invalid complex diagonal inputs.

For the iterative consumer, `compute_localized_velocity_in_frame` accepts
the already selected complete entering E/C and uses the same numerical core.
`compute_live_headwing_velocity` calls this form after building the current
native-baseline H(R), preserving the live orbital frame used by the RI response.
Consequently each head/wing refresh needs no external PyATB process and no
second eigensolve. These are internal numerical interfaces; no public C or
Fortran binding is added. Scalar full-matrix controls cover the model solve
and the retained three entering Si frames. SOC uses the same complex model
solve and projector. Symmetry sewing and fractional-occupation response require
separate validation beyond the full-BZ integer-occupation controls.

The SOC controls include complex nonorthogonal and degenerate spinor models,
all 220 states at four selected Bi2Se3 points, and two actual current-HR updates
on a four-state model compared with compiled PyATB `tb(4)`. They retain all
Cartesian complex velocity entries after S-metric frame transport, with declared
energy/length constants reconciled explicitly. The analytic wing's RI charge
vertex sums equal spinor components, as required for a spin-independent charge
operator; `test_rpa_headwing` checks its complete projection and global SU(2)
rotation invariance. These finite-model controls do not establish a converged
material spectrum or an accepted off-mesh interpolant.

Fourier round-trip and eigenpair-frame checks establish agreement only on the
input mesh. Before using `current_hr` for physical band, velocity or topology
predictions, compare the interpolant with direct producer bands away from
that mesh and check mesh convergence. The earlier whole-H 26-state Si 4x4x4
control passed native-mesh closure while producing spurious off-mesh states,
with a five-eV-window discrepancy up to 116.74 eV. The native-baseline route
now preserves the producer's H0(R) and is shared with method-2 HR export;
this changes the former derivative/feedback map. Its grid closure and
same-model velocity checks still do not establish material off-mesh accuracy
or mesh convergence.

With `qsgw_write_iteration_matrices = true`, traces include all three velocity
components, the head tensors and auxiliary-space wing arrays. With HR export
enabled, the entering Hamiltonian differentiated for update N is the formal
`hrs<S>_nao_qsgw_iter_<N-1>.csr` snapshot in **Ry**; use the matching raw S/r.
On a fresh band/cut calculation, update 1 still screens the original PBE state;
the cut reference used for residuals does not replace those entering eigenpairs.

### Strict two-dimensional screening

Option 6 can use the strict-2D complete Gamma-cell Wc route. In addition to
the native H/S/r inputs above, supply the matching producer-generated
[`librpa_2d_coulomb_head.txt`](../develop/dataset_format.md#librpa_2d_coulomb_headtxt)
and select:

```ini
use_2d_dielectric = true
use_fullcoul_eps = true
use_fullcoul_wc = true
use_shrink_abfs = false
sqrt_coulomb_threshold = 0
```

Keep `replace_w_head = true` and `use_scalapack_gw_wc = true`. The metadata
must describe the same full auxiliary basis and Coulomb data as the ordinary
standalone dataset. The driver applies the producer normalization to every
new analytic head/wing object, including after checkpoint continuation.
The cell area and auxiliary monopole normalization remain fixed; the live
energies, wavefunctions, velocity, frequency grid, head, wing and screening
are rebuilt for each update. Native G0W0 applies the same normalization.
The public task input accepts the numerical in-plane area and auxiliary monopole norm; the driver reads them from the existing producer metadata file.

## Iteration and output

After reading and validating the SCF data, the driver selects the mixing route
from the global loaded SCF k-point count and full BvK mesh. The single-k route
requires both one SCF k point and a full mesh product of 1; it permits the
requested `qsgw_mixer = none` or `linear`. All other inputs use the periodic
route and force effective `none` with beta 1, even if `linear` was requested.
An IBZ containing one point from a larger full mesh therefore uses the
periodic route. The log records requested/effective mixer, SCF count and mesh.

The default mixer remains `none`; selecting a single-k input does not enable
mixing automatically. A single-k supercell mixing example is:

```ini
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
```

For single-k `linear`, `H_next = H_in + beta * (H_out - H_in)` with
`beta = qsgw_mixing_beta` (default 0.2); beta 1 gives the direct-update result.
`none` accepts the new Hamiltonian directly. Grid and band path use the same
effective fraction; band-path presence and its point count do not select the
mixing route. This is an input convention for single-k supercells, not a
physical isolation test: a Gamma-only solid also enters the single-k route
and can select `none`. It does not change geometry, Coulomb, or head/wing.
The obsolete `qsgw_system_type` key must be removed from inputs; specifying it
is an input error. The band Hamiltonian cut is applied before and after
mixing, so excluded states obey the cut exactly.

`qsgw_iterations.dat` records the iteration number, largest eigenvalue change
(eV), Hamiltonian residual L2 and maximum norms (Ha), Fermi energy and gap (eV),
electron count, mixing fraction and convergence flag. Residuals are measured
before mixing on the SCF grid; the maximum is the largest complex-entry
magnitude. The convergence flag and stopping gate use only the largest eigenvalue
change and `qsgw_min_iter`; the Hamiltonian residual is reported separately and
does not enter this gate.
`qsgw_eigenvalues.dat` contains every grid eigenvalue and, when enabled, every
path eigenvalue at initialization and after each update. With band inputs,
each completed iteration also writes `KS_band_spin_S_N.dat`,
`EXX_band_spin_S_N.dat`, and `QSGW_band_spin_S_N.dat` under `output_dir`, where
S is the one-based spin index and N is the iteration number. These tables
provide the data for band plotting; the driver does not generate PNG figures.
Optional matrix diagnostics are enabled by
`qsgw_write_iteration_matrices`.

## Complete Hamiltonian export for PyATB

QSGW retains a complete real-space Hamiltonian at every update by default,
independently of head/wing corrections, matrix diagnostics and band-path inputs.
To disable this output, set:

```ini
qsgw_export_hamiltonian_for_pyatb = false
```

The export switch defaults to `true`. The input filenames default to
`qsgw_fn_sr = srs1_nao.csr` and `qsgw_fn_rr = rr.csr`; specify overrides only
for a different filename or path. Export also reads the original DFT
`hrs1_nao.csr` from `input_dir`, and `hrs2_nao.csr` for a second physical
collinear spin channel. Scalar and SOC use one original H file. These are
the regular ABACUS H/S/r output names. The original DFT H files must be
available and in Ry; no fallback to whole-H interpolation is made if they
are absent. Existing molecular or FHI-aims inputs without this matching native
ABACUS H/S/r set must explicitly disable export with
`qsgw_export_hamiltonian_for_pyatb = false`.
An overlap-only
`get_s` output named `sr_nao.csr` needs an explicit S override.
S(R) is dimensionless and r(R) is in
Bohr; filenames are relative to `input_dir` unless absolute. They retain their
original real-space support. The exporter requires a complete square AO
reference and a full, unshifted uniform SCF mesh. Shifted-mesh export
is rejected before the first GW update; paired IBZ input uses the expansion above; the old
`qsgw_hr_export_full_mp_rgrid` selector is not needed or supported.

After mixing and the cut actually selected by the task, but **before
diagonalization**, the driver writes
`output_dir/hrsS_nao_qsgw_iter_NNNN.csr`, with a one-based spin index S and
global iteration N. This is the exact fixed-reference operator passed to
diagonalization, including all retained and cut-state blocks. Its AO
embedding uses the immutable initial KS coefficients and S, so it does not
require the new eigenvectors. A fresh calculation also writes iteration 0000
for the actual entering PBE Hamiltonian. A restart writes its restored state
under the saved iteration number and then continues global numbering.

The selected export reconstruction preserves the matching native DFT
Hamiltonian and inverse-transforms only the complete current correction:

$$
H_{\rm export}(R)=H_{0,\rm native}(R)
+\mathcal F^{-1}\!\left[
H_{\rm current}^{AO}(k_i)-H_{0,\rm native}^{AO}(k_i)
\right](R).
$$

The correction includes the Hartree density difference together with exchange,
correlation, DFT Vxc subtraction, any selected post-mixing cut and all high-state
blocks. Hartree and EXX therefore pass through the same total-Hamiltonian
reconstruction; there is no separate Hartree export or second Hartree addition.
The original DFT H must match the loaded immutable reference on the full
SCF grid. Every complex entry and native H/S/r support is retained; the
existing Fourier roundtrip check still compares the complete exported
operator with the current AO grid matrices. Native baseline preservation
improves the tested Si path but does not establish off-mesh convergence.

HR export and `current_hr` now use the same native-baseline reconstruction.
The completed iteration-N output and the entering velocity model for update
N+1 have the same complete H(R), apart from file units and exact-zero support
padding. A fresh iteration 0000 and update 1 likewise share the entering PBE
operator. This changes k derivatives and head/wing feedback relative to the
former whole-H velocity reconstruction, even though both close on the same
original-grid H. Current-HR checkpoints therefore use a new velocity contract
and cannot silently resume the old whole-H map.

The output mode follows the loaded reference's spin-channel and spinor
dimensions automatically. The same export switch covers all three native modes:

| Native ABACUS mode | Reference channels / spinor components | Files at iteration NNNN | Matrix dimension per file |
| --- | --- | --- | --- |
| Non-spin-polarized, `nspin=1` | 1 / 1 | `hrs1_nao_qsgw_iter_NNNN.csr` | `N_AO` |
| Collinear spin-polarized, `nspin=2` | 2 / 1 | `hrs1_nao_qsgw_iter_NNNN.csr` and `hrs2_nao_qsgw_iter_NNNN.csr` | `N_AO` for each channel |
| SOC/noncollinear, `nspin=4` | 1 / 2 | `hrs1_nao_qsgw_iter_NNNN.csr` | `2 * N_AO`, including cross-spinor blocks |

For collinear input, the two HR files use their own channel's Hamiltonian
and reference coefficients; they share the unchanged `srs1_nao.csr` and
`rr.csr` inputs. Their headers record two spin directions and indices 1 and 2.
Scalar and SOC headers record one spin direction and index 1. SOC output
retains both spinor components and their coupling in one complete matrix.

The files use the modern native ABACUS HContainer matrix grammar:
`--- Ionic Step N ---`, spin/basis/R counts, `# CSR Format`, and the native
values/column-indices/row-pointers markers. They contain full complex CSR
entries in **Ry**, with 17 significant digits and no imaginary projection or
sparsity threshold. The matrix file does not embed an ABACUS cell/atom block;
the consumer takes geometry from the matching original input.
For SOC (`use_spinor_wfc = true`, also accepted as `use_soc = true`), one
`hrs1_nao_qsgw_iter_NNNN.csr` retains the complete `2 * N_AO` matrix,
including off-diagonal spin blocks, in the native interleaved ordering.
The S reader accepts both legacy and modern native CSR; native SOC S is
complex, while `rr.csr` uses real entries at the same full spinor dimension.
No spatial-AO-only S/r is replicated implicitly.
H is padded with exact-zero blocks to the complete H/S/r support
union; S/r are neither trimmed nor folded into the BvK mesh. Each complete
CSR file is published through a temporary-file rename. Iteration 0 is the
original DFT operator; iteration N enters the screening for update N+1.

HR N is an operator snapshot written before diagonalization. If a later
diagonalization or occupation update fails, this file can exist without a
completed iteration N checkpoint. Successful iterations continue to write
the normal checkpoint after occupations update; its default filename still
retains only the latest completed iteration. Per-iteration HR output does
not create per-iteration checkpoint archives or replace the native input
required for continuation.

`utilities/pyatb_qsgw.py` supplies `load_qsgw_model`: `nspin=1` loads a scalar
model or a selected collinear channel, and `nspin=4` loads the complete SOC
model as `tb(4)`. It uses PyATB's existing complex-H reader in Ry, the native
real or complex S reader as appropriate, and the real r reader. All five
operators are aligned on their full support union with exact-zero padding.
With `utilities` on the Python import path, a scalar example is:

```python
from pyatb_qsgw import load_qsgw_model

model = load_qsgw_model(
    "librpa.d/hrs1_nao_qsgw_iter_0001.csr",
    lattice_constant_angstrom=5.43,
    lattice_vector=[[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    n_kpoints=64,
    nspin=1,
)
```

Use the actual producer lattice, not the illustrative numbers above;
`n_kpoints` is the solver's k-point batch capacity. For SOC, set `nspin=4`
and supply the matching SOC geometry and complete spinor S/r files.
For collinear output, load each HR file separately with `nspin=1` and the
same original scalar S/r and geometry.
This adapter requires NumPy, SciPy and an installed PyATB and does not
implement a topology algorithm. Its energy conversion follows the installed PyATB
Ry reader; the earlier eV-based velocity-oracle tolerance is not a new
numerical validation of this Ry export. Ordinary scalar PyATB inputs that
expect real CSR tokens cannot directly parse the complex H file. Native
mesh reconstruction alone also does not accept off-mesh material bands,
an occupied projector, a gap or a topological invariant.

## Checkpoint continuation

QSGW saves the last completed update by default in
`output_dir/qsgw_checkpoint.bin` (normally `librpa.d/qsgw_checkpoint.bin`).
The optional `qsgw_checkpoint_file` selects another filename relative to
`output_dir`, or an absolute path; `none` disables saving. The driver writes
a temporary file and renames it only after the complete state has been written.
Interruption during a later update therefore leaves the previous completed
checkpoint available. Work inside an unfinished self-energy calculation is
not restored.

To continue, retain the original producer input and numerical settings, use
a fresh working/output directory, and set:

```ini
qsgw_restart_file = /path/to/initial/librpa.d/qsgw_checkpoint.bin
qsgw_max_iter = 3
```

The restart path is relative to the working directory unless absolute.
Restoration is explicit: a checkpoint already present in `output_dir` does
not start a restart unless selected with `qsgw_restart_file`. The default
`none` starts from the original input.

Iteration limits are global. Restoring completed iteration 1 with a maximum
of 3 computes updates 2 and 3. The maximum must exceed the saved iteration;
the minimum and stopping tolerance may also change. New traces begin at the
restored boundary, so a separate output directory preserves earlier traces.

With `qsgw_update_hartree=true`, the format adds 200 to the corresponding
checkpoint version and checks the dedicated bare q=0 metric, unwrapped
LRI coefficients, BvK convention and actual atom-pair image mapping. A restart
cannot switch Hartree feedback on or off. Earlier +100 checkpoints used the
direct-R Hartree path and require a fresh start. The initial DFT density remains
the reference; it is reconstructed from the original inputs before the saved
live state is restored.

The checkpoint retains the grid and optional band-path Hamiltonians in the
immutable initial KS basis, live eigenvalues, wavefunctions, occupation weights,
Fermi energy and the last iteration summary. Directly restoring wavefunctions
preserves their gauge at the boundary. Screening and exchange are rebuilt
for each new update; the original reference remains the basis for Vxc and
self-energy projection. Native RI and Coulomb files remain required, as do
the velocity input or S/r files selected by the velocity convention. Preserve
the band channel across a restart: a checkpoint created with band inputs
requires the same path metadata, band E/C, and full band Vxc inputs; a
grid-only checkpoint cannot acquire a band channel by adding those files.

The reader checks dimensions, reference eigenpairs, Vxc, k coordinates/weights,
cut policy, effective fixed mixing fraction, frequency count and head mode.
The automatically selected route uses the existing effective-beta checkpoint
check without changing the checkpoint format. Older mixed single-k molecular
calculations can continue with the same mixer and beta, without a system-type
setting. Periodic-route checkpoints with effective beta below 1 cannot
continue under the new policy because the existing beta check rejects the
change; effective-beta-1 checkpoints remain compatible.
Other producer files and numerical controls must also remain unchanged; the
checkpoint does not validate every input. MPI size can change because these
tasks require replicated wavefunctions. This experimental binary format
rejects incompatible byte order and accepts legacy fixed-mode versions 1--3.
The rotated-reference velocity map writes version 5 without cut and version
9 with shifted mode 1, preserving its immutable-velocity contract. The
native-baseline `current_hr` map writes version 10 without cut and version 11
with shifted mode 1. It binds the actual native DFT H(R), S/r, geometry and
velocity convention. Older whole-H current-HR versions 5/9 cannot continue
under this changed map. A restart cannot switch maps or replace these
numerical operators. Version 4 belongs to a separate equation/mixer branch
and is not interpreted here.
Reference-velocity shifted mode 1 retains version 8, on both grid-only and
band-enabled routes. Older cut checkpoints, including
the former unshifted mode 1 and the previous mode-2 versions 6/7, cannot be
silently reinterpreted under this numbering. Saved keep/shift settings must
still match explicitly when a default
has changed.
This implementation supports `qsgw_mixer = none` and `linear`; adaptive
automatic-mixing history requires the separate mixer-aware extension.

For the saved payload, MPI restoration order and binary-format maintenance,
see the [developer design notes](../develop/design.md#qsgw-iteration-state-and-checkpoints).

## Numerical checks

The small regression suite covers two updates of molecular H2O from FHI-aims
with linear mixing, Si k333 with analytic head correction, and
FHI-aims H2O through `qsgw` with a band path, mixing and Hamiltonian truncation.
An ABACUS H2O case checks one full update through the native complex Vxc input
and Ry conversion against results from before the input simplification.
The band case uses the default 16 minimax
frequencies and the same Gamma reference on grid and path. Tests compare actual
iteration energies, Hamiltonian residuals, electron count and eigenvalue
trajectories with the existing numeric table comparator used by G0W0.

`regression_tests/verification/verify_qsgw_restart.py` compares uninterrupted trajectories
with independently started continuation processes for Si without mixing and
molecular H2O with a band path and Hamiltonian cut. Earlier controls also
covered Si with linear mixing; those results remain historical, and that case
is excluded from the active verifier under the single-k mixing policy.
The verifier checks all eigenvalues,
summary fields, reference-basis operators and exact restored WFC/occupations;
later eigenvector phases and degenerate rotations are excluded. Use a fresh
`--work-dir`, the built `--binary`, and explicit `--ranks` in the matching MPI
runtime. These controls test numerical continuation, not material convergence.

A manual Si k666 band case is under
`regression_tests/manual/Si_k666_qsgw_band`. These cases check numerical workflows;
they do not establish convergence of material predictions.

`regression_tests/verification/verify_qsgw_headwing.py` compares three uninterrupted updates
with a one-plus-two restart, including saved velocities, head tensors, wings,
self-energies, effective Hamiltonians and restored boundary states. It also
checks the actual analytic/body frequency nodes. Select `--case aims-rotate`
for the bundled initial-operator control or `--case native-current --dataset`
for a complete full-mesh native ABACUS dataset; the manual Si k444 input gives
the latter's operator and velocity convention. These are feedback and restart
controls, separate from independent velocity and off-mesh band acceptance.
