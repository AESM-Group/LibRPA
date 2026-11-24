# Driver Usage

## Data preparation

To run the LibRPA driver for RPA/*GW* calculation, you need to export necessary
data from atomic-basis first-principles code.
The data include

- Atomistic structure
- one-particle orbital energies, occupation numbers and wave functions
- Bloch vectors used in periodic calculation
- Triple co-efficients of resolution of identity (RI)
- Coulomb matrix under the auxiliary basis for RI

### FHI-aims

Export of data required by LibRPA from FHI-aims is supported since the release version
231212 and also in the latest master branch. You can switch it on by adding to your `control.in` file:

```text
output librpa
# legacy switch before 240507 release
print_librpa_input .true.
```

It will dump necessary data files for many-body computation tasks.
Please refer to the FHI-aims manual for more details of the `output librpa` tag.

### ABACUS

Similar to the case with FHI-aims, the latest master branch of ABACUS supports the export of files required by LibRPA.
You only need to add the following line to the `INPUT` file of ABACUS:

```text
rpa 1
```

## LibRPA job

### Input file

Before triggering the LibRPA driver, an input file named `librpa.in` is required
at the working directory. An example is below:

```ini
task = rpa
input_dir = .
nfreq = 16
cs_threshold = 1e-4
```

The driver will read the parameters defined in the file and run the calculation
Please refer to the [guide page of runtime parameters](runtime_parameters)
for more information about the driver and API parameters.

By default, `input_preset = fhi-aims` (also accepted as `aims`) preserves the
historical driver filenames,
including `stru_out`, `basis_out`, `band_out`, `mommat_ks_kpt_*.dat`,
`Cs_data_*`, and `coulomb_mat_*` under `input_dir`. For current ABACUS output,
select:

```ini
input_preset = abacus
```

This selects the short reader-v1 names: `stru_out.txt`, `band_out.txt`,
`vxc.txt`, `bz_sample.txt`, `basis_map.txt`, `wfc_basis.txt`,
`aux_basis.txt`, `aux_basis_s.txt`, `KS_wfc_*`, `Cs_*`, `Cs_shrink_*`,
`sinvS_*`, `V_cut_*`, and `V_full_*`. It also changes `prefix_velocity` to
`velocity_matrix`.

PyATB head/wing sidecars retain their `KS_eigenvector_*` filename contract.
LibRPA reads those files independently of the `KS_wfc_*` ABACUS preset.

For ABACUS datasets produced before
[PR #7849](https://github.com/deepmodeling/abacus-develop/pull/7849)
(merge commit [`42f8ad905d56`](https://github.com/deepmodeling/abacus-develop/commit/42f8ad905d567e965ae53fc554498bdc7e9ccbcc) on `develop`), select:

```ini
input_preset = abacus-legacy
```

If a host code exports the same data under different names, set the corresponding `fn_*` or `prefix_*` driver parameters in `librpa.in`:

```ini
fn_eigocc_scf = band_out_alt
prefix_lri_coeff = Cs_data_alt
prefix_coul_full = coulomb_mat_alt
prefix_coul_cut = coulomb_cut_alt
```

If `stru_out` contains symmetry operations, the driver builds a symmetry context from
the structure, basis metadata, and k-point grids. Runtime `use_symmetry_*` switches
then decide whether calculation paths use that context.

### QSGW with an optional band path

Use `task = qsgw` for molecular and periodic QSGW. The driver also computes the
band-path channel when the file selected by `fn_band_kpath_info` (default
`band_kpath_info`) exists under `input_dir`; this requires matching band
energies, wavefunctions, and full Vxc matrices. Without the metadata file,
only the SCF grid is computed, including for periodic inputs. The driver
does not infer geometry or create a path. `qsgw` is the only QSGW task name;
older inputs must switch their task value to `qsgw`.

Mixing is selected automatically after SCF input validation. One global SCF k
point with a full BvK mesh product of 1 permits `none` or `linear`; other SCF
meshes force effective `none` with beta 1, including a one-point IBZ of a larger
mesh. The default mixer remains `none`. For a single-k supercell, for example:

```ini
qsgw_mixer = linear
qsgw_mixing_beta = 0.2
```

The log records requested/effective mixer, SCF count and mesh. This input
convention does not detect physical isolation: a Gamma-only solid also enters
the single-k route and can use `none`. Band-path presence/count does not affect
the route; grid and path use the same effective beta. Geometry, Coulomb and
head/wing settings remain unchanged. Remove the obsolete `qsgw_system_type`
key from input files.

With band inputs, the existing band-cut options remain active and each
iteration writes `KS_band_spin_S_N.dat`, `EXX_band_spin_S_N.dat`, and
`QSGW_band_spin_S_N.dat` for plotting. Native per-update QP diagnostics are
written when `qsgw_output_qp = true`; see [QSGW](qsgw.md#quasiparticle-roots-during-iteration). The driver writes energy data rather
than PNG figures. Continue a checkpoint with the same band-channel presence
and original band inputs. See [QSGW](qsgw.md) for the full Vxc input, analytic
head/wing feedback, output and restart requirements.

### Native G0W0 head/wing velocity

For analytic head/wing correction, the standalone `g0w0` driver and its
`g0w0_band` alias take the velocity from `option_velocity`. `option_velocity =
native` builds it internally from the original DFT H(R), S(R), and r(R) instead
of reading an external velocity matrix. Velocity is evaluated in the SCF
eigenframe already loaded by the driver, retaining the SCF energies,
wavefunctions, occupations, and frequency/response prescription. No velocity
export or second eigensolve is required. Every analytic option (3, 4, 53)
accepts both sources.

An ABACUS example, in addition to the ordinary RI/Coulomb and mean-field inputs, is:

```ini
task = g0w0
input_preset = abacus
replace_w_head = true
option_dielect_func = 3
option_velocity = native
use_scalapack_gw_wc = true
g0w0_fn_hr = OUT.ABACUS/data-HR-sparse_SPIN0.csr
g0w0_fn_sr = OUT.ABACUS/data-SR-sparse_SPIN0.csr
g0w0_fn_rr = OUT.ABACUS/data-rR-sparse.csr
g0w0_hr_unit = Ry
g0w0_rr_unit = Bohr
use_kpara_scf_eigvec = false
use_symmetry_rpa = false
use_symmetry_gw = false
use_symmetry_exx = false
```

Native velocity requires `replace_w_head = true`; with option 3 or 53 it also
requires `use_scalapack_gw_wc = true` to consume the analytic wings. Direct
library/API callers use option 3 and either supply velocity through the public
setter or construct it with `librpa_prepare_native_velocity`.

The source selector is `option_velocity`, with `pyatb` (default) and `native`:

```ini
option_dielect_func = 3
option_velocity = native
```

Use filenames that match the producer. For two collinear spin channels, also set
`g0w0_fn_hr_spin1` to the second spin's Hamiltonian. Paths can be absolute or relative
to `input_dir`; H, S and r retain their independent real-space supports. H units
can be `Ha`, `Ry` or `eV`, and position units `Bohr` or `Angstrom`. Units are decoded
once into Ha/Bohr. S is dimensionless, and the structure reader supplies the lattice
in Bohr. Option 4 uses the same native velocity for the existing analytic head-only
prescription. The `g0w0_band` task alias accepts the same options.

The current native mode requires complete square AO/band frames, with scalar or SOC
wavefunctions. Both a uniform full-BZ mesh and a symmetry-reduced grid whose k stars
cover the full mesh are accepted; SOC additionally needs the magnetic spin operations.
K-point-parallel SCF eigenvectors are accepted too: the head/wing frame is replicated
from the owning ranks first, and a run whose blocks cannot be completed stops there
instead of differentiating a partial frame. It checks the metric residual against 1e-6, Hamiltonian frame and
Hermiticity residuals against 1e-4 eV, and the imaginary velocity diagonal against
1e-8 Ha Bohr. These input checks establish frame consistency within their tolerances;
they do not certify material or basis convergence.

For options 3, 4, and 53, the default `option_velocity = pyatb` preserves the
existing reader: it looks for a velocity matrix produced by an external optical
code under `pyatb_librpa_df/`. When ABACUS writes the real-space H(R), S(R) and
r(R) blocks, prefer `option_velocity = native` — LibRPA then differentiates
those blocks itself and no external velocity file is needed. With
`output_level = debug`, either path writes `g0w0_velocity.csv` and
`g0w0_headwing.csv` under `output_dir`, containing complete complex velocity, head
and auxiliary-wing arrays. The head/wing snapshot is written before Wc consumes the
auxiliary wing. These diagnostic files do not change the response calculation.

### Run the calculation

After setting up the input file `librpa.in`, the LibRPA driver can be called by issuing:

```bash
/path/to/LibRPA/build/chi0_main.exe
```

To run with multiple processes using MPI, you can simply invoke the relevant MPI driver,
for example

```bash
mpirun -np <nprocs> /path/to/LibRPA/build/chi0_main.exe
```

where `<nprocs>` is the number of MPI processes.
In addition, you want to run with multiple threads, you need to specify the
environment variable `OMP_NUM_THREADS`. For example

```bash
export OMP_NUM_THREADS=4
mpirun -np 4 /path/to/LibRPA/build/chi0_main.exe
```

This will run the LibRPA calculation using 4 MPI processes, each with 4 OpenMP threads.

## Task-specific guides

```{toctree}
:maxdepth: 1

Conventional k-space RPA <conventional_rpa>
QSGW <qsgw>
cRPA <crpa>
Hartree AO operator <hartree>
```
