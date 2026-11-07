# Local interactions from constrained RPA

`task = crpa` computes bare, partially screened and fully screened onsite
interaction tensors on the native positive imaginary-frequency grid. The two
Green-function branches use the same original KS states with binary or
fractional endpoint weights to form `Pd`; the screened interaction uses `Pr = P0 - Pd`. KS energies, occupations
and the frequency grid are unchanged.

## Orbital analysis before screening

By default, `crpa_output_analysis = true` evaluates Mulliken and symmetric
Lowdin AO weights from the loaded scalar or two-component spinor KS
coefficients and supplied S(k).
Here \(C\) contains the already exported KS eigenvectors in the atomic-orbital
basis, with \(C_{\mu n}\) the coefficient of AO \(\mu\) in band \(n\).
For scalar states each column \(c_n\) obeys \(Hc_n=\varepsilon_n Sc_n\).
For spinor states the overlap metric is \(I_{\rm spinor}\otimes S\), and
\(\sum_s c_{ns}^\dagger Sc_{ns}=1\); k and stored-channel indices are
suppressed here.
No Hamiltonian is read and no physical KS eigenproblem is solved. S alone is
diagonalized to evaluate its positive square root. The weight definitions are

$$
w^M_\mu=\sum_s\operatorname{Re}[c_{\mu s}^*(Sc_s)_\mu],\qquad
w^L_\mu=\frac{\sum_s |(S^{1/2}c_s)_\mu|^2}
{\sum_s c_s^\dagger Sc_s}.
$$

The existing `basis_wfc_out` shell metadata and explicit basis convention
identify atom/l groups. All radial functions are summed, with d split into
global-axis eg and t2g channels. No `pbands.xml`, `Orbital` or `PDOS.dat`
reference file is required. This AO character diagnostic is independent of
the first-radial projected local frame used for the interaction tensors.

The driver writes the following before screening:

- `crpa_orbital_groups.dat`: zero-based group/atom indices, species, l and channel.
- `crpa_orbital_weights.dat`: each spin/k/band/group weight in both conventions,
  absolute and relative energy, occupation, k weight and original S norm.
- `crpa_pdos.dat`: both Gaussian PDOS conventions on one energy grid, with
  separate stored spin channels, in states/(eV cell stored-spin-channel).
- `crpa_orbital_analysis.txt`: definitions, energy reference, Gaussian sigma,
  normalization and overlap-square-root diagnostics.

For a two-component input, weights sum both components of the same KS state
and are written in one stored spin channel (`spin = 0`). This index does not
identify a conserved spin projection. Scalar collinear inputs retain their
separate spin-up and spin-down channels. No spin-projected vector or spin
texture is computed by this orbital-character analysis.

PDOS uses the loaded normalized k weights, all loaded bands and no occupation
or additional spin-degeneracy factor. Mulliken weights can be negative; they
are not clipped. The default grid is -10 to 14 eV in steps of 0.02 eV, relative
to the highest occupied KS energy (`crpa_pdos_reference = vbm`), with Gaussian
standard deviation `crpa_pdos_sigma_ev = 0.15`. Set the reference to `fermi`
for a Fermi-level zero, and control the range with `crpa_pdos_min_ev`,
`crpa_pdos_max_ev` and `crpa_pdos_step_ev`. A finite grid covers only that
energy range; it is not a complete-spectrum normalization integral.
Energy conversion uses the configured LibRPA constants. A producer's printed
eV column can use a slightly different Hartree/eV constant; comparisons must
match this convention as well as poles, k weights and Gaussian sigma.

For a first pass that helps select A/B windows, use:

```text
task = crpa
input_preset = abacus
input_dir = ./input/
fn_stru = stru_out
fn_eigocc_scf = band_out
crpa_species_labels = Ni O
crpa_overlap_file = srs1_nao.csr
crpa_residual_tol = 1e-8
crpa_analysis_only = true
use_spinor_wfc = false
use_kpara_scf_eigvec = false
use_symmetry_rpa = false
```

This mode reads structure/k metadata, the split wave-function basis, energies,
coefficients and S; it skips auxiliary-basis, RI and Coulomb inputs entirely.
For SOC or a noncollinear spinor input, change `use_spinor_wfc` to `true`.
Either a spatial S(R) export or a validated interleaved SOC export can be
used. Optical velocity inputs are also skipped in analysis-only mode.
No correlated species or A/B selections are required. After inspecting the
analysis, supply the normal cRPA selections and set `crpa_analysis_only = false`
to run screening. These diagnostic AO weights never select windows automatically
or enter Pd. The separate orbital-weight response mode below uses the actual
interaction frame. `crpa_output_analysis = false` skips the additional diagnostic
analysis during an interaction run, without disabling the response weights.

## Response selection and local frames

With the default `crpa_response_mode = bands`, the binary response subspace
**A** and orbital-construction band space **B** are independent. Each can be selected by inclusive absolute-Hartree intervals (lower/upper pairs)
or by an explicit zero-based KS band list shared by all spin/k points. PDOS and
KS-band inspection determine the material-specific selection.

For each spin, atomic trials are projected into B and orthonormalized using
the AO overlap. `crpa_parent_orbitals` selects `s`, `p`, `sp`, `d`, `dp`, `eg` or `t2g` trials;
`crpa_output_orbitals` selects a subset of the resulting common frame. Thus
`d|dp` and `eg|dp` retain columns of the jointly normalized d+p frame. The
first radial shell is used; `eg` and `t2g` refer to the global Cartesian axes.
`sp` combines `s, pz, px, py` on each correlated atom in one output site, retaining
the cross-shell direct and exchange interactions in its four-index tensor. In
contrast, `dp` combines correlated-atom d orbitals with distinct ligand-atom p
orbitals. Selecting `s` or `p` from an `sp` parent retains columns of that same
jointly normalized frame, rather than constructing separate local frames.
An explicit ligand species prevents spectator atoms such as Li or Sr from
being interpreted as oxygen.

An example using the existing AFM NiO d|dp windows is:

```text
task = crpa
input_preset = abacus
parallel_routing = libri
use_symmetry_rpa = false
use_kpara_scf_eigvec = false
use_fullcoul_eps = true
use_fullcoul_wc = true
replace_w_head = false
option_dielect_func = 0
n_bands_chi0 = -1
tfgrids_type = minimax
nfreq = 24
crpa_overlap_file = srs1_nao.csr
crpa_species_labels = Ni O
crpa_correlated_species = Ni
crpa_ligand_species = O
crpa_parent_orbitals = dp
crpa_output_orbitals = d
crpa_response_windows_ha = 0.422538107206142 0.6066522113061734
crpa_orbital_windows_ha = 0.231441631892736 0.6066522113061734
crpa_residual_tol = 1e-8
```

If a group is isolated at each k but overlaps other bands in its global energy
range, specify `crpa_response_bands = 20 21 22` and/or
`crpa_orbital_bands = 20 21 22`. Omit the respective `_windows_ha` input when
using a band list. This selects the original KS indices directly; no orbital
character weighting or modified eigenvalues enter the response.

### Fractional orbital-projector response

Set `crpa_response_mode = orbital_weights` to define the excluded response from
the **actual output frame**, rather than a binary A window. Supply only B
(`crpa_orbital_windows_ha` or `crpa_orbital_bands`) and the parent/output orbitals;
omit both `crpa_response_windows_ha` and `crpa_response_bands`. The NAO trials
are still projected into B and jointly Lowdin normalized. No Wannier
localization or new physical eigenproblem is introduced.

For each physical spin and k, the output frame obeys
\(\Phi^\dagger S\Phi=I\). Its endpoint probabilities are

$$
P_{an\mathbf k}=\Phi_{a\mathbf k}^\dagger S_{\mathbf k}C_{n\mathbf k},\qquad
w_{n\mathbf k}=\frac{\sum_a|P_{an\mathbf k}|^2}
{C_{n\mathbf k}^\dagger S_{\mathbf k}C_{n\mathbf k}}.
$$

The denominator handles the accepted finite-precision normalization of input C.
For SOC, S denotes the lifted spinor metric. Both Green-function branches are
weighted, giving the original KS transition \((n\mathbf k,m\mathbf{k+q})\)
a factor \(w_{n\mathbf k}w_{m\mathbf{k+q}}\) in Pd. Pr therefore uses
\(1-w_{n\mathbf k}w_{m\mathbf{k+q}}\). The same retained factor enters
analytic optical head and wings. KS poles, physical occupations, density
vertices and the full GW response are unchanged. Small positive probabilities
are retained continuously; there is no character threshold or filling of band-index
holes. Validated roundoff above one alone is clipped to one.

For NiO d|dp, for example:

```text
crpa_response_mode = orbital_weights
crpa_parent_orbitals = dp
crpa_output_orbitals = d
crpa_orbital_bands = 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25
```

These band indices are illustrative, not a transferable NiO prescription.
B must span all requested trials without a rank defect at every spin/k.
The d|dp response uses d output columns from the joint dp frame; dp|dp uses
all dp output columns. A d|d frame generally defines a different projector.
This weighted construction is a distinct screening model from binary
energy-window exclusion, even when its windows are numerically identical.
Collinear magnetic and full SOC states are supported within the existing
even-frequency response scope. Explicit q-star symmetry can be used when the
chosen weighted target preserves that group, as described below.
By default QSGW rebuilds atomic-trial frames from current C when cRPA is requested.
B retains its supplied window/list rule.

For iteration with an explicitly reviewed, sufficiently wide candidate B, enable

```ini
crpa_track_orbitals = true
crpa_tracking_min_capture = 0.99
```

The first frame uses the declared atomic trials. Subsequent frames transport the
complete previous parent, including ligand columns, through the current KS B:
`X = C_B C_B^dagger S Phi_previous`, followed by metric polar normalization.
The orbital dimension and labels stay fixed while KS band character may reorder.
Every previous parent direction must retain at least the stated probability,
measured by the smallest eigenvalue of `X^dagger S X`. Original Gram/metric
rank and condition gates remain in force. A loss requires a wider reviewed B;
the code does not silently change B, remove orbitals or lower rank thresholds.
Full rank alone does not establish a stable model: compare frames and U under
candidate-window enlargement.

QSGW performs this inexpensive model-only transport after every eigenstate update,
independently of `crpa_interval`; no response, W or self-energy is computed by
transport. The C API `librpa_prepare_crpa_model` and the C++ handler expose the
same operation. Response probabilities always use the actual current C and Phi.
Tracking writes `crpa_tracking.dat` with each output state and a `.crpa` companion
to the QSGW checkpoint. Keep that companion for continuation: the model definition,
parent dimensions, iteration and complete checkpoint eigenvalues are checked.
Restarting a tracked trajectory without its parent record is rejected.

Species labels follow the numeric species order in the structure input.
`crpa_overlap_file` is a native S(R) CSR file, relative to `input_dir`.
Spatial exports and interleaved SOC exports are accepted; SOC exports must
have identical diagonal spin blocks and zero off-diagonal blocks.
Scalar or two-component spinor wavefunctions, the full k/q mesh and LIBRI routing are required. A
compressed auxiliary basis additionally requires `use_shrink_abfs = true`,
`use_shrink_chi = true` and the corresponding native transformation inputs.

Outputs under `output_dir` are:

- `crpa_tensors.dat`: complete ordered onsite v/U/W tensors for every output
  spin pair and positive imaginary frequency, in eV.
- `crpa_summary.dat`: intraorbital, distinct-orbital direct and exchange
  averages, with explicit atom/spin indices. Exchange averages are not Slater J.
- `crpa_windows.dat`: actual selected KS indices for binary A and for B;
  weighted mode has no binary A list.
- `crpa_response_weights.dat`: the actual endpoint probability for every spin/k/band
  in either response mode.
- `crpa_orbitals.dat`: atom, local column and atomic-trial label.
- `crpa_metadata.txt`: orbital conventions, input selections and units.

The response calculation has no direct zero-frequency node. By default the
driver additionally estimates U(0) using the internal continuation API described
below. The first minimax node must not be identified with U(0).

### Native model output and continuation

`crpa_output_model = true` and `crpa_output_static = true` are defaults. Each
calculated electronic state writes `crpa_model.npz`, a standard uncompressed
NumPy/ZIP64 archive, alongside the existing text tensors. Geometry, k points,
weights, AO conventions and shell layout are stored once in
`output_dir/crpa_static.npz`; the state archive contains its relative path.
Move that file together with the state directories when copying the model.

Set `crpa_output_format = dat` to write the same model as readable text instead
of NumPy archives. This option changes driver output only; the response and
orbital-selection rules are unchanged. `crpa_model_info.dat` records dimensions,
units, spin frames and state identity; `crpa_structure.dat` and
`crpa_kpoints.dat` retain geometry and sampling. `crpa_bands.dat` contains
energies, physical occupations and the actual A endpoint probabilities;
`crpa_projectors.dat` contains parent/output P; `crpa_model_k.dat` contains
H(k) and density. `parent=1` identifies the joint construction frame and
`parent=0` the actual output frame. The orbital/site mapping is in
`crpa_model_orbitals.dat`. Energies and interactions in DAT output use eV,
lengths use Bohr, and all indices are zero based.

The existing `crpa_tensors.dat` retains every imaginary-node tensor entry;
`crpa_continued_tensors.dat` additionally retains every requested static/real
entry with its numerical status and subset spread. The metadata and orbital
analysis notes use `.dat` extensions in this mode. Optional raw export writes
`crpa_coefficients.dat`, `crpa_overlap.dat` and `crpa_local_orbitals.dat`.
QSGW uses the same files inside each `crpa/state_NNNN` directory.

The state archive contains physical occupations, absolute-Hartree Bloch
energies, the parent/output projectors P, projected H(k) and density matrices,
weighted onsite H and density blocks, the actual A/B band selections and `A_weights` (empty A lists in weighted mode), orbital
labels, site offsets, and the full ordered v/U tensors on original positive
imaginary nodes. Standalone results also contain the full-RPA local W tensors.
QSGW metadata identifies the entering or final electronic state. These are the
static state's eigenvalues, distinct from diagonal QP roots in `qsgw_qp.csv`.

The numerical library constructs P = Phi-dagger S C and H = P diag(E)
P-dagger once using the actual frame that formed the U vertices. No additional
KS eigenproblem is solved. `crpa_local_model.dat` reports onsite H in eV and
physical density matrices, without subtracting a DMFT double-counting term.
Use Bloch E and P for projection-based DMFT: replacing the Bloch Hamiltonian
by a smaller projected H can discard its coupling to omitted bands.

The compact default does not retain or write full C/S/Phi. Set
`crpa_output_raw = true` to include them for rebuilding orbital projectors.
The corresponding C API input is `retain_raw_model = LIBRPA_SWITCH_ON`;
otherwise raw model-view pointers are null. This avoids duplicating every
wavefunction and metric on each MPI rank just to write the compact model.
Retaining C/S alone does not permit an arbitrary new U projection: that also
requires the matching constrained kernel and RI data. A full auxiliary-space
Wr/RI cache is not included in this archive.

Static extraction uses the selected `anacon_method` (default Padé) on Delta U = U-v with source
arguments z=i*nu, evaluates at z=0, then restores v. It uses the original imaginary-frequency grid
and no electronic Fermi/Hedin shift. This is an extrapolation, not a direct
static response calculation. `crpa_continuation_summary.dat` and the archive
retain per-entry status, subset spread and source/holdout residuals. The
default absolute diagnostic tolerance is `crpa_fit_tol_ev = 0.01` eV.
Nonfinite or unstable estimates are flagged; they do not replace the original
U(i*nu). Numerical fit checks do not certify passivity, causality or material
convergence. Set `crpa_output_static = false` to omit the static estimate.

Real-axis continuation uses the same internal function and is requested by an
explicit frequency list, for example:

```text
crpa_real_frequencies_ev = 0 1 3
crpa_eta_ev = 0.05
```

The real frequencies are bosonic energies measured from zero. Evaluation is
at omega+i*eta; no real-axis grid is produced when the list is empty. The
archive's `continued_z` gives the actual complex queries in Hartree and
`continued_U_*`, `continued_status_*`, `continued_spread_*` contain all entries.
When static output is enabled the first query is exactly zero; subsequent
queries correspond to the requested real frequencies.

Library callers use `librpa_crpa_model_info`, `librpa_crpa_model_k`,
`librpa_crpa_model_local` and `librpa_crpa_model_hopping` to retrieve the owned
state and one-body model. `librpa_crpa_project_orbitals` and
`librpa_crpa_project_one_body` also operate on explicit numerical arrays.
`librpa_crpa_rotate_projector` and `librpa_crpa_rotate_tensor` apply matched
isometries to a fixed local model. Changing the excluded response requires
new screening. `librpa_crpa_tensor_to_fermion` converts density U_ab,cd to
V_ijkl=U_ik,jl for a (1/2) c_i-dagger c_j-dagger c_l c_k Hamiltonian; it does
not impose a Kanamori or Slater approximation. `librpa_crpa_tensor_averages`
returns the complex ordered-column means U_mm,mm, U_mm,nn and U_mn,nm;
the two off-diagonal means are zero for a one-orbital frame. The continuation API
`librpa_continue_crpa` returns an owned result for arbitrary upper-half-plane
queries, including zero. These numerical operations are noncollective.

For an iterative library calculation, create and prepare a public QSGW task
before supplying the full-grid cRPA overlap. Initialize
`LibrpaQsgwCrpaControls` with `librpa_init_qsgw_crpa_controls`, then attach it
with `librpa_set_qsgw_crpa`; the task supplies its owning numerical state internally.
The task copies the numerical input and controls. Core cRPA code evaluates
the resident entering-state response, transports the parent after every
completed update, restores its checkpoint companion, and builds the final
response-only state. Its output sink runs on every rank and receives borrowed
cRPA and optional orbital-analysis views. The sink may query these views and
write output, but must not delete them or recursively invoke the handler/task.
Keep the sink's user data alive until the task returns. Ordinary input reading
and DAT/NPZ output remain in the driver.

`utilities/export_crpa_dmft.py` performs only unit/layout conversion into a
TRIQS `HDFArchive` (requires TRIQS's Python `h5` module):

```bash
python3 utilities/export_crpa_dmft.py output/crpa/state_0000/crpa_model.npz nio.h5
```

It preserves all output sites as separate impurity shells and both AFM spin
blocks, writes the band-basis Hamiltonian and complex projectors, and retains
full native interaction tensors with their conventions. The interaction group
also stores the original positive imaginary-frequency nodes as `nu`, converted
to eV with the same factor as U; `frequency_unit` is `eV`. It does not choose
a double-counting functional, impose AFM equivalence, or initialize an impurity
solver. Mixed-angular-momentum `sp` sites require a custom DMFT shell definition.

For SOC, set `use_spinor_wfc = true`. Each spatial trial is lifted to both
spin components, projected into the original full-spinor B states, and jointly
normalized in the metric `I_spinor tensor S`. A p site therefore has six columns,
a d site ten, and an sp site eight. Within each site, up trials precede down
trials; these labels identify the trial orientation and are not conserved spin
quantum numbers. The response has one channel of occupied spinor states with
occupation at most one, without an additional factor of two. It retains all
four Green-function spin-block products.

SOC emits one complete spinor four-index tensor per site/frequency. Its
`spin_left = spin_right = 0` denotes that one channel; spinor information is
carried by `a,b,c,d`. The density vertex sums same-component products before
screening. The complete-spinor summary exchange average includes opposite
trial-spin columns and is not a scalar-shell Hund parameter. Inspect resolved
entries using the emitted orbital labels. Keep entire degenerate groups when
choosing A or B.

For a gapped three-dimensional input, `replace_w_head = true` with
`option_dielect_func = 3` uses analytic head and wings from the current
one-particle eigenpairs and their velocity matrices. The driver preset
`option_dielect_func = 6` selects the same full head/wing calculation and
computes velocity from native H(R), S(R) and r(R), using the existing
`g0w0_fn_hr`, `g0w0_fn_sr`, `g0w0_fn_rr` and unit parameters. With option 3,
select either that native source or the existing `velocity_matrix` file. Full
screening uses every physical transition; constrained screening omits a pair
only when both original endpoints belong to A. The RI optical charge vertex
sums equal spinor components. Only the angular Gamma-cell weights are
normalized; the regular dielectric body is unchanged. Optical sums retain all
original full-BZ KS states and velocities even when q-star screening is enabled.
Distributed KS inputs and the two-dimensional optical route remain excluded.

With `replace_w_head = false` and `option_dielect_func = 0`, an uncorrected
periodic result remains a finite-mesh diagnostic until the Gamma contribution
is bounded. Using a full-response optical head unchanged for nonzero Pd would
screen with the wrong response. Small-case optical tests do not establish
material mesh or basis convergence.

Library callers use `LibrpaCrpaInput` and its initializer from `librpa_input.h`,
and the QSGW companion controls and initializer from `librpa_options.h`.
Numerical result views, accessors, continuation and task attachment are declared
in `librpa_compute.h`; `librpa_handler.h` declares the indexed result handles.
Supply S(k) and the window/species data, call `librpa_compute_crpa_window` collectively,
then inspect the owned result and release it with `librpa_delete_crpa_result`.
The standalone computation accepts numerical data and returns an owned snapshot.
The separate tracking-checkpoint function writes library state; the iterative
companion restores and saves its checkpoint at the QSGW completed-state boundary.

The orbital-analysis functions in `librpa_compute.h` compute an owned
result through `librpa_analyze_orbitals`, expose spatial AO/group weights and
state metadata including the number of spinor components, and evaluate
Gaussian PDOS in Hartree units. C++ callers can use `Handler::analyze_orbitals`,
which returns an owning result. Neither path modifies the handler, KS states,
response masks or local output frames. Orbital analysis uses the original full
k-grid states even when cRPA screening applies q-star symmetry reduction.

Every participating rank must supply identical numerical options, species,
selections and S(k) arrays. Input buffers remain owned by the caller and must
stay valid until the collective call returns. Each rank receives its own result
and releases it locally. Accessor arrays and strings remain valid until that
owned result is deleted. QSGW output callbacks instead receive borrowed result
handles and views that must not be deleted or retained after the callback returns.
The driver supplies this replicated input by reading the same native source on
every rank. KS orthogonality and frame checks validate
the supplied data; they do not reorthogonalize or repair the KS eigenvectors.

The scalar nspin=1 response retains the native spin-degeneracy convention.
For nspin=2, screening includes both physical spin channels, while output
vertices use their respective spin-resolved local frames. The reported left and
right spin indices identify those frames, not two independently screened U's.
Finite-temperature or metallic static-limit accuracy requires separate response
and convergence controls; selecting positive minimax nodes alone does not establish it.

With `use_symmetry_rpa = true`, the first symmetry route retains full original
k-grid Green functions and reduces only response q sampling and dense screening.
Every screened kernel is restored through the supplied unitary/antiunitary
operation table before contraction with the full-grid local spinor vertices.
The bare term also uses all original q points. Multiplying one representative
orbital tensor by star multiplicity would not preserve its individual components.
The actual diagonal endpoint-weight operator must transform covariantly with
E/C/f. The implementation obtains the band sewing matrix from the supplied
complete KS frames and checks E_t M = M E_s, f_t M = M f_s and p_t M = M p_s,
including operations in each little group. Complex phases, band permutations
and degenerate mixing that preserves p are allowed; weights are not averaged
or replaced by a binary envelope. Violations fail collectively. In particular,
arbitrary mixing of degenerate states with unequal p is incompatible with this
diagonal-weight approximation and cannot use q reduction.

The driver accepts an explicit `spin_symmetry` structure tail; library callers
use `librpa_set_symmetry_spin_operations`. SOC symmetry rejects a legacy spatial-only
declaration. A grey-group declaration appends time reversal only when the supplied
state has that symmetry. Current-HR QSGW can expand paired IBZ source E/C/f and
Vxc into its internal full-grid frame before coupled cRPA; standalone library
cRPA continues to require complete replicated KS frames. Auxiliary compression
maps use separate full/compressed shell actions, with conjugation on antiunitary
routes, and existing full-q maps are checked rather than averaged. Bare and
screened kernels are restored before contraction. This does not extend the
native even-frequency response to arbitrary magnetic odd-frequency components.

## Compose or reuse numerical stages

The internal C++ stage declarations are in `src/core/crpa_workflow.h`, with
their implementations together in `src/core/crpa_workflow.cpp`. The public
standalone API prepares the model through these stages, and `compute_crpa_onsite`
is the composition that retains the existing complete v/U/W calculation.
The driver continues to own native-file reading and output writing.

| Stage | Result |
| --- | --- |
| `prepare_crpa_model` | Owning B, binary A or fractional weights, and scalar/spinor local frames |
| `prepare_crpa_local_projection` | RI density vertices and retained bare interaction |
| `build_crpa_total_response` | Owning P0 on the current state and actual grid |
| `build_crpa_excluded_response` | Owning Pd with both endpoint probabilities |
| `form_crpa_constrained_response` | Owning Pr = P0 - Pd |
| `screen_crpa_response` | Native correlation-only Wc from an owning response |
| `contract_crpa_interaction` | Local bare + Wc tensors, without changing Wc |

For a caller that already holds P0, the combination
`evaluate_crpa_onsite_from_response(context, selection, frames, p0)` computes
only bare and U. It builds Pd and screens Pr; it does not build P0 or screen
the full response. Its returned `CrpaResult` has empty W vectors. The caller
can use the projection/contraction stages separately for an already computed
full Wc when a local fully screened comparison is wanted.

The borrowed P0, resident Chi0 response map and incoming band selector are
preserved, including when P0 aliases `context.chi0.get_chi0_q()`. Response
stages return owning maps; screening consumes its map by value/move. Temporary
Green-function workspaces may rebuild. The numerical stages do not update
the mean field or reinitialize the Dataset, frequency grid or descriptors.

Reuse requires the same current E/C/f, native frequency nodes, response q view,
active auxiliary space, MPI atom-pair layout and Coulomb/optical conventions.
Call before the GW backend releases its P0. The public collective
`librpa_build_g0w0_sigma_crpa` combines the ordinary GW build with U-only
constrained screening immediately after its P0 is available. The corresponding
C++ method is `Handler::build_g0w0_sigma_crpa`. It returns an owned cRPA result
while retaining the usual GW EXX and self-energy objects. The public
`librpa_compute_crpa_u_window` / `Handler::compute_crpa_u_window` builds a
current-state P0 and U without an additional full-W or self-energy calculation.
The standalone `librpa_compute_crpa_window` / `Handler::compute_crpa_window`
returns complete v/U/W. The C++ wrapper owns the result through `CrpaResult`
and releases it automatically; accessor views borrow that result's storage.

## Compute U during G0W0 or QSGW

For a single G0W0 calculation, enable the shared cRPA workflow with:

```text
task = g0w0
option_dielect_func = 6
crpa_enabled = true
crpa_model_file = crpa-model.in
```

The calculation reuses G0W0's P0 and the original PBE eigenpairs for U, while
retaining the usual GW screened interaction and quasiparticle calculation.
It writes the cRPA tensors, orbital weights and PDOS to
`output_dir/crpa/state_0000`. The metadata records the unchanged reference
state (`state_index = 0`, `gw_update = 0`) as final. The interval option
applies to QSGW iteration; G0W0 evaluates U once.

Add the following to a working QSGW input using the native full head/wing preset:

```text
task = qsgw
option_dielect_func = 6
crpa_enabled = true
crpa_model_file = crpa-model.in
```

The model file contains only the cRPA choices already used by a standalone
calculation: species labels, correlated/ligand species, overlap, parent/output
orbitals and the A/B selectors. An existing standalone cRPA input can itself
be the model file. Its shared GW settings are not imported. The model-file
path is relative to the main input file; native data paths are relative to the
main `input_dir`. Main-input cRPA assignments override the imported model;
selecting bands replaces an inherited energy-window selection and conversely.
Do not explicitly specify both selector forms for the same A or B.

`crpa_interval = 1` evaluates every entering electronic state. To sample every
second state, set `crpa_interval = 2`. The final state is always evaluated,
including when it does not fall on the interval. Ordinary QSGW supplies its
current E, C and occupations; each selected calculation rebuilds the model
frame and Pd from that state. The overlap, atomic trial definition and input
selection rule remain fixed. An energy-window rule may select different bands
as energies move; explicit band indices retain the stated indices.

Outputs are written below `output_dir/crpa/state_0000`, `state_0001`, etc.
State 0 is the initial reference; update 1 produces state 1. Metadata records
the electronic state, GW update and whether this is the final state. A selected
entering state borrows the GW P0 before full screening. After the last QSGW
update, a response-only call evaluates the final U with the final eigenpairs
and current-HR velocity. That call does not perform another self-energy update.
A checkpoint restart keeps the saved state index and the same interval rule.

These outputs contain bare and partially screened tensors and their direct/
exchange summaries. The U-only route omits W columns. Orbital weights and PDOS
are enabled by the existing `crpa_output_analysis` default and use the same
current eigenvectors; no extra physical KS solve is required.

The coupled path currently requires replicated full-BZ states, native minimax
frequencies and the LibRI response path. Explicit magnetic q reduction is
supported for compatible live states and targets. The QSGW task can expand
paired IBZ inputs internally; the direct coupled API requires full-grid KS
frames. G0W0 with `crpa_enabled = true` also requires complete replicated KS
inputs; its IBZ-only head/wing reconstruction does not expand the cRPA KS frame.
It rejects `read_sigc_mat_rf`, which bypasses P0. Use full head/wing option 3 (or driver
preset 6) for a corrected three-dimensional calculation. Head-only option 4
does not provide the constrained optical wings required by this implementation.
Driver preset 6 is normalized to option 3 before the library call.

The cRPA kernel uses the dataset's full Coulomb interaction for its own bare
term and constrained screening. GW retains its separately selected full/cut
Coulomb treatment. P0 reuse requires the same response, not an identical final
screened interaction. cRPA normalizes the angular Gamma-cell weights in its
onsite kernel; the GW optical convention remains the one selected by the
ordinary GW calculation. Consequently a cRPA fully screened local comparison
and GW W should be compared only with explicitly matched optical conventions.

Method selection is shared with GW/QSGW; see [analytic continuation](analytic_continuation.md).
The original C API `librpa_continue_crpa` now uses default Pade settings with
its requested coefficient count. Use `librpa_continue_crpa_with_options` for
explicit Padé/AAA/ridge selection and penalties. C++ has the corresponding
`continue_crpa` overload taking `Options`. The bosonic variable remains z=i*nu,
without a fermionic energy shift or change to the original response tensors.
