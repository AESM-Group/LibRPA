# Periodic Hartree AO matrices

`task = hartree` builds the electrostatic Hartree operator from the currently
loaded SCF eigenvectors and their physical, k-weighted occupations. The result
is the complete complex matrix in the spatial AO basis, in Hartree. The
operator sums physical spin channels and diagonal spinor components and is
the same scalar spatial operator for both collinear spins.

This endpoint is a one-shot build and comparison. It does not update a QSGW
Hamiltonian or introduce a Hartree inner self-consistency loop.

## Build support

The default build prepares the unmodified LibRI revision
`ab81ac939d820b1350c1e00f0e09e8e1edfc3d96`. No separate LibRI installation or auxiliary-potential
extension is needed. With `LIBRPA_USE_LIBRI=ON`, CMake checks for
`RI/physics/Hartree.h` in the resolved headers, including external headers.
This header enables standalone Hartree and QSGW feedback.
CMake reports this capability; unsupported Hartree requests fail explicitly.
See {ref}`Installation <install-local-dependencies>` for dependency selection.

Band-path matrices use the ordinary mesh Hartree result: inverse Fourier
transform to AO real-space blocks, then the same atom-pair BvK image averaging
(`option_bvk_remap`) and forward Fourier convention as EXX. This finite-mesh
interpolant reproduces the input mesh; away from it, convergence depends on
the mesh and the BvK convention. The dedicated RI input remains full precision,
but its unwrapped translations are not used to define a separate path operator.

## Input

Use the ordinary structure, full-BZ mean-field, wave-function/auxiliary-basis,
and RI-coefficient inputs, with an explicitly supplied Hartree metric prefix:

```text
task = hartree
input_preset = abacus-legacy
input_dir = ../dataset
version_coul_reader = 1
version_lri_reader = 1
prefix_lri_coeff = v1_Cs_data
prefix_lri_coeff_shrink = v1_Cs_shrinked_data
prefix_shrink_sinvS = v1_shrink_sinvS_
prefix_coul_hartree = v1_coulomb_hartree_
parallel_routing = libri
use_shrink_abfs = false
cs_threshold = 0
vq_threshold = 0
```

This example matches the reference producer's `out_librpa_reader_version 1`
filenames. The `abacus-legacy` preset selects its structure/basis/eigenstate
names; the versioned RI filenames are overridden explicitly above.
This first-comparison example uses unshrunk auxiliary data without extra
consumer-side coefficient screening. Use `use_shrink_abfs=true` only with
the matching shrunk producer metric and coefficients.

For the ABACUS reference producer, use a converged LCAO SCF calculation with
`symmetry -1`, a Gamma-centered regular k mesh, `rpa 1`,
`out_librpa_reader_version 1` and `out_mat_h_vh 1`. Stage its ordinary
structure/basis/eigenstate and selected C files, the dedicated
`v1_coulomb_hartree_iq_*_rank0.dat` file and the native `vh1_nao.csr`
from the output directory into `input_dir`. The metric and C files must
come from the same full/shrunk producer setup. The native CSR is in Ry;
the metric is already in Ha. This compares LibRPA's first build from the
exported DFT state to ABACUS's post-SCF Hartree operator for that state.

`prefix_coul_hartree` has no preset/default fallback. Its files must contain
the **bare three-dimensional Coulomb auxiliary matrix at zero lattice
momentum transfer**, with the reciprocal-space `G=0` term set to zero. That
choice fixes the background/potential convention. A single q=0 v1 file is
sufficient; the task does not require Coulomb files at every nonzero q. The
metric must match the selected full/shrunk auxiliary basis and the RI
coefficients. Supplying this prefix asserts that physical kernel convention;
a generic Coulomb filename does not establish it.

The ordinary EXX `prefix_coul_full` and `prefix_coul_cut` inputs are not used
as Hartree fallbacks. A truncated, erfc-screened, optical-head-replaced or
other modified metric must not be substituted for the required bare metric.
The first implementation requires a complete Gamma-centered BZ mean-field grid, the supported
three-dimensional periodic geometry, and no atomic-position Bloch phase
(`bloch_ratom=0`). The producer's remaining Bloch convention must match the
computation API and native comparison; these restrictions are not a claim of
support for arbitrary basis gauges. Only `parallel_routing=libri` is supported.
`parallel_routing=auto` selects `libri` before any data input. Explicit `rtau`
and `atompair` routes are rejected collectively at that point. The `rtau`
setter uses a different coefficient representation that this Hartree backend
does not consume; `atompair` additionally omits one directed LRI endpoint.
A shifted SCF mesh is not supported by this public input path: the existing
Coulomb-index setter resolves q indices through the SCF grid, so q=0 must be
an actual supplied grid point.

## Output and native reference comparison

Root writes `hartree_ao.dat`, retaining every complex AO entry at every loaded
SCF k point. Rows are `k i j real imag`, with zero-based indices and Hartree
units. These are full spatial AO matrices rather than KS diagonal values.

To compare a native ABACUS Hartree CSR export, add:

```text
hartree_fn_reference = vh1_nao.csr
hartree_reference_scale = 0.5
hartree_check_tolerance = -1
```

The reference filename is relative to `input_dir`, unless absolute. The
explicit scale converts its energy to Hartree; `0.5` is appropriate for
Rydberg input, while Hartree input requires `1.0`. The reader retains both
complex triangles and all supplied lattice translations. It constructs
`V_H(k) = sum_R exp(+2 pi i k.R) V_H(R)` on the actual SCF grid and compares
every complex entry without fitting a constant or an overlap-proportional
potential shift. The reference and exported auxiliary metric must therefore
share the same electrostatic background convention and density.

`hartree_comparison.dat` records the maximum absolute entry difference and
Frobenius norm for each k point and over the complete matrix bundle. All
differences are in Hartree. A negative `hartree_check_tolerance` only reports
the discrepancies. A positive value explicitly gates the global maximum
entry error and requires a reference; a failure is propagated to all MPI
ranks. The code supplies no universal material-accuracy tolerance.

Agreement can be limited by the RI approximation, radial/plane-wave grids,
finite real-space support, CSR export precision, or a mismatch between the
final grid density and the density reconstructed from occupations. A completed
build or a report-only comparison does not by itself establish numerical
acceptance or physical convergence.

## QSGW density-difference feedback

The optional feedback adds `V_H[P_live - P_original_DFT]` explicitly to the
QSGW Hamiltonian before cuts/mixing and removes the same component from the
entering-state QP static subtraction. It is enabled by default; set
`qsgw_update_hartree = false` for fixed-Hartree runs. In addition
to normal QSGW input, supply the matching bare q=0 metric and coefficients
with **physical, unwrapped translations**:

```ini
qsgw_update_hartree = true
prefix_coul_hartree = v1_coulomb_hartree_iq_
prefix_lri_coeff_hartree = v1_Cs_hartree_shrinked_data_
```

Use `v1_Cs_hartree_data_` for the full, unshrunk auxiliary basis. The raw-C
prefix must match the auxiliary basis selected by `use_shrink_abfs`;
the coefficients are folded consistently onto the integration mesh. Band paths
use the resulting AO operator with the EXX BvK interpolation convention. The supported public
route uses LibRI, bare 3D Coulomb, a complete unshifted Gamma-containing mesh,
and `bloch_ratom=0`. Disabled behavior retains its previous requirements.

`qsgw_hartree.csv` reports the entering and post-update density differences,
AO/fixed-initial-KS matrix norms and grid matrices; path output contains KS
diagonals. A fresh DFT entering state has zero correction by definition. After
one QSGW update, the reported updated-state value is the correction for the
next Hamiltonian; its evaluation does not perform a second GW build.

The immutable original DFT density remains the reference across restart.
Checkpoints reject changes in the enabled flag, reference, metric or raw C
using the checkpoint's numerical input tolerance. The BvK convention and
actual atom-pair image mapping are also bound. Checkpoints from the earlier
direct-R Hartree path require a fresh start. The C/C++ opaque QSGW API
accepts these inputs; the standalone Fortran Hartree API does not expose a
Fortran QSGW task interface.
