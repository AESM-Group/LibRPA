# Dataset Format

This page documents the formats of the input data files required by the LibRPA driver.

## Driver input names

The standalone driver reads dataset files from `input_dir`. The default
`input_preset = fhi-aims` keeps the historical filenames. Setting
`input_preset = abacus` selects the current reader-v1 names: `stru_out.txt`,
`band_out.txt`, `vxc.txt`, `bz_sample.txt`, `basis_map.txt`,
`wfc_basis.txt`, `aux_basis.txt`, `aux_basis_s.txt`, `KS_wfc_*`, `Cs_*`,
`Cs_shrink_*`, `sinvS_*`, `V_cut_*`, and `V_full_*`, together with the
`velocity_matrix` prefix.
For the historical FHI-aims and `abacus-legacy` presets, single-file inputs
are named [`stru_out`](#stru-out),
[`bz_sampling_out`](#bz-sampling-out), [`basis_wfc_out`](#basis-files),
[`basis_aux_out`](#basis-files), and [`band_out`](#band-out).
Additional optional single-file inputs use
[`dielecfunc_out`](#dielecfunc-out), [`vxc_out`](#vxc-out), and
[`band_kpath_info`](#band-kpath-info).
These exact filenames can be changed in `librpa.in` with `fn_stru`,
`fn_bz_sampling`, `fn_basis_wfc`, `fn_basis_aux`, `fn_eigocc_scf`,
`fn_dielfunc`, `fn_vxc_scf`, and `fn_band_kpath_info`. Velocity/momentum files
are selected with `prefix_velocity`.

The combined [`basis_out`](#basis-out) file selected by `fn_basis` is
deprecated and is read only as a fallback when split basis files are absent.
When `use_shrink_abfs = t`, historical datasets use `basis_aux_shrink_out`, or
the filename selected by `fn_basis_aux_shrink`, for the compressed auxiliary
basis. Current ABACUS reader-v1 datasets use `aux_basis_s.txt` through the
same setting. The old `fn_basis_shrink` input key is still accepted as an
alias.

For the historical FHI-aims and `abacus-legacy` presets, multi-file inputs are
selected by prefix. The defaults are [`Cs_data`](#cs-data) for localized RI coefficients,
[`Cs_shrinked_data`](#cs-data) for compressed-auxiliary-basis RI coefficients,
[`coulomb_mat`](#coulomb-mat) for bare Coulomb matrices,
[`coulomb_cut`](#coulomb-cut) for truncated Coulomb matrices, and
[`KS_eigenvector`](#ks-eigenvector-scf) for SCF Kohn-Sham eigenvectors.
These prefixes can be changed with `prefix_lri_coeff`,
`prefix_lri_coeff_shrink`, `prefix_coul_full`, `prefix_coul_cut`, and
`prefix_eigvecs_scf`.
For example, `prefix_coul_full = coulomb_mat` matches files such as
`coulomb_mat_0.txt`.
For ABACUS reader-v1 shrink datasets, keep the full and shrink coefficient
families distinct, for example `prefix_lri_coeff = Cs_` and
`prefix_lri_coeff_shrink = Cs_shrink_`. The LRI reader rejects
identical full/shrink prefixes and filters the other family when one prefix is
a leading substring of the other.

## `librpa_2d_coulomb_head.txt`

Strict-2D complete-Wc runs require the standalone reader-v1 dataset to provide
this producer-generated metadata file in `input_dir`. It records the in-plane
cell area and the squared auxiliary-basis monopole norm used to normalize the
analytic Gamma head. The file is consumed by the reader; it is not inferred
from the LibRPA basis files.

The parser accepts comments, optional `=`, and arbitrary whitespace. The
required keys are:

```text
version = 1
area_parallel_bohr2 = 19.390653825130212
multipole_norm_squared = 2205.0673846924301
```

`area_parallel_bohr2` is the positive in-plane cell area in Bohr$^2$.
`multipole_norm_squared` is the positive squared monopole norm in the
producer's auxiliary-basis convention. The values must come from the same
full auxiliary basis and producer run as `basis_aux_out`, Coulomb matrices,
and head/wing data. A missing, non-positive, or unsupported-version value is
rejected before the strict-2D complete-Wc route starts.

### Minimal complete-Wc driver settings

The following strict-2D settings select the complete Gamma head/wing Wc route
for the standalone driver. The ordinary dataset, frequency, band, and GW
settings still need to be supplied for the calculation:

```ini
task = g0w0
input_dir = ./reader-v1
output_dir = ./librpa.out
parallel_routing = libri
replace_w_head = t
option_dielect_func = 3
use_2d_dielectric = t
use_fullcoul_eps = t
use_fullcoul_wc = t
use_shrink_abfs = f
sqrt_coulomb_threshold = 0
rpa_headwing_mode = qavg
```

The `input_dir` must contain the metadata file described above together with
the full auxiliary basis, full Coulomb matrices, and producer head/wing data.
The complete-Wc route requires full Coulomb for both the dielectric basis and
the finite-q external legs, and it currently requires the ScaLAPACK Wc path.

For native velocity, use `option_velocity = native` with the native G0W0 H/S/r
inputs, or with `task = qsgw` and `qsgw_headwing_velocity = current_hr`. Retain the strict-2D
settings above. Both drivers apply this same producer metadata to their
analytic head/wing object; QSGW reapplies it after each rebuild and on restart.
Only the normalization is fixed: QSGW recomputes the response from the live
entering state and frequency grid. See [QSGW](../user_guide/qsgw.md#strict-two-dimensional-screening).

Producer support is defined by this reader-v1 contract: a producer is
supported when it writes `librpa_2d_coulomb_head.txt` and the matching full
auxiliary-basis, Coulomb, and head/wing files from the same run. Older output
without this metadata is not inferred or accepted; regenerate it with a
producer that implements the contract.

In this version the public C/C++ and Fortran API options do not provide a
metadata injection field. Strict-2D complete-Wc is therefore supported through
the standalone reader-v1 driver, which reads the file and configures the
head/wing object. A direct API request without that driver configuration is
rejected before Wc construction.

(stru-out)=
## `stru_out`

The file `stru_out` contains structural information.

Its contents are arranged in the following order:

- **Lattice vectors**: 3 lines, each containing 3 floating-point numbers, in units of Bohr.
- **Reciprocal lattice vectors**: 3 lines, each containing 3 floating-point numbers, in units of Bohr${}^{-1}$.
- **Number of atoms**: 1 line containing the number of atoms in the unit cell, `n_atoms`.
- **Atomic coordinates and types**: `n_atoms` lines, each containing the Cartesian coordinates and the atom type of one atom.

After the atom rows, `stru_out` may end or may contain an optional tail. The
tail can contain a symmetry block directly. In older files, the legacy
Brillouin-zone section below may come before the symmetry block.

The symmetry block starts with one line containing the number of operations and
the fractional-coordinate convention:

```text
n_symops row
```

or

```text
n_symops col
```

Use `row` for operations applied as `x' = x * R + t`. Use `col` for operations
applied as `x' = R * x + t`, which is the Spglib convention. Each of the next
`n_symops` lines contains one integer rotation matrix and one fractional
translation:

```text
R11 R12 R13 R21 R22 R23 R31 R32 R33 t1 t2 t3
```

Include the identity operation when providing the block; symmetry-enabled
calculations require it.

An optional spin-operation block may follow the spatial operation rows:

```text
spin_symmetry grey source
```

The `grey` value is nonzero when the listed operations are the unitary block of
a grey magnetic group; LibRPA then adds the corresponding antiunitary copy.
`source` selects the spin action convention:

- `0`: `Identity`; no spin matrix is read.
- `1`: `ExplicitSpinSpace`; each operation row must provide an explicit SU(2)
  matrix.
- `2`: `DerivedFromSpatialSOC`; no spin matrix is read from this tail, and
  LibRPA derives the spin action from the spatial operation.

The spin block contains one row for every spatial operation. Each row starts
with an `antiunitary` flag (`0` or nonzero). For `source = 1`, append eight
floating-point values containing the row-major 2x2 SU(2) matrix as
`(real, imaginary)` pairs:

```text
antiunitary
```

or

```text
antiunitary U11_re U11_im U12_re U12_im U21_re U21_im U22_re U22_im
```

Whitespace separates tokens; line breaks in these examples are for
readability only. The spin block is passed to the same symmetry-operation API
as programmatically supplied datasets.

Older `stru_out` files may also contain the following Brillouin-zone sampling
entries. LibRPA no longer reads k-points from `stru_out`; datasets must provide
[`bz_sampling_out`](#bz-sampling-out).

- **k-point grid dimensions**: 1 line containing three integers, `nkx`, `nky`, and `nkz`. The total number of k-points in the full grid is
  ```text
  nkpts = nkx * nky * nkz
  ```
- **Full k-point list**: `nkpts` lines, each containing 3 floating-point numbers giving the Cartesian coordinates of one k-point, in units of Bohr${}^{-1}$.
- **Mapping to irreducible k-points**: `nkpts` lines, each containing 1 integer.
  Suppose the integer on the n-th line is m.
  This means that the irreducible representative of the n-th k-point in the full k-point set is the m-th k-point in the full set.

(basis-files)=
## Basis files

The historical split basis files `basis_wfc_out`, `basis_aux_out`, and
`basis_aux_shrink_out` use the same format. Current ABACUS reader-v1 datasets
use `wfc_basis.txt`, `aux_basis.txt`, and `aux_basis_s.txt` for the same three
roles. They describe, respectively, the wave-function basis, the full
auxiliary basis, and the shrink auxiliary basis.

The first line contains three entries:

1. total number of atom types, `n_atom_types`
2. total number of basis functions in this basis
3. a string specifying the convention used for the Bloch-sum phase, basis
   ordering, and real spherical harmonics

Recognized producer presets are `aims`, `abacus`, `openmx`, `pyscf`, and
`fallback`. `fallback` means the convention is not known from this file.

For example:

```text
2        26    abacus
```

The next `n_atom_types` lines provide the size for each atom type. Each line
contains:

1. atom type index
2. number of basis functions for this atom type

For example:

```text
1        13
2        13
```

The remaining content gives the l-shell layout. There are `n_atom_types`
blocks, one for each atom type. In each block:

- the first line contains the atom type index and the number of radial functions
- the following lines list the angular momentum quantum number `l` for each
  radial function, one integer per line

For example:

```text
1       5
0
0
1
1
2
```

Each radial function with angular momentum `l` contributes `2*l + 1` basis
functions. In the example above, atom type `1` has `1 + 1 + 3 + 3 + 5 = 13`
basis functions.

(basis-out)=
## `basis_out` (deprecated)

`basis_out` is the legacy combined basis file. New datasets should write
[`basis_wfc_out`](#basis-files), [`basis_aux_out`](#basis-files), and, when
`use_shrink_abfs = t`, [`basis_aux_shrink_out`](#basis-files). The driver
still reads `basis_out` as a fallback when `basis_wfc_out` and
`basis_aux_out` are absent.

The first line contains four entries:

1. total number of atom types, `n_atom_types`
2. total number of wave-function basis functions
3. total number of auxiliary basis functions
4. the basis convention string

The next `n_atom_types` lines contain:

1. atom type index
2. number of wave-function basis functions for this atom type
3. number of auxiliary basis functions for this atom type

After that, `basis_out` stores the l-shell blocks for the wave-function basis,
then the same block structure for the auxiliary basis.

(bz-sampling-out)=
## `bz_sampling_out`

This file describes the Brillouin-zone sampling used in the calculation,
including the full k-point grid and its reduction to the irreducible set.

Its structure is as follows.
The first line contains three integers:

1. `nk1`
2. `nk2`
3. `nk3`

These specify the number of k-point divisions along the three reciprocal lattice directions.
For example:

```text
3   3   3
```

The second line contains two integers:

1. number of SCF k-points, the same value as in `band_out`
2. number of irreducible k-points for Coulomb matrices

Here "SCF k-points" means the k-point dimension used to store SCF
eigenvalues, occupation numbers, and Kohn-Sham eigenvectors in
[`band_out`](#band-out) and [`KS_eigenvector_xxx.txt`](#ks-eigenvector-scf).

For example:

```text
27     14
```

This means that the Kohn-Sham SCF data contains `27` k-points, of which `14` are used as Coulomb-matrix representatives.

The next `n_k_points` lines describe the SCF k-points. Each line contains ten fields:

1. k-point index in the SCF set (1-based)
2. k-point weight; the sum over the `n_k_points` rows must be `1`
3. fractional coordinate `k1`
4. fractional coordinate `k2`
5. fractional coordinate `k3`
6. Cartesian coordinate `kx`
7. Cartesian coordinate `ky`
8. Cartesian coordinate `kz`
9. index of the corresponding irreducible Coulomb k-point
10. index of its representative in the SCF k-point list

For example:

```text
2   0.37037037037E-01   0.00000000000E+00   0.00000000000E+00   0.33333333333E+00   0.00000000000E+00   0.00000000000E+00   0.35439508162E+00   2   2
```

This line indicates that SCF k-point `2`

- has weight `0.037037037037`
- has fractional coordinates `(0, 0, 1/3)`
- has Cartesian coordinates `(0, 0, 0.35439508162)`
- maps to irreducible Coulomb k-point `2`
- whose representative in the SCF k-point list is also point `2`

Older files may contain irreducible-k-point weight summaries after these rows.
LibRPA ignores those summaries because the row weights above are already authoritative.

### Optional explicit full-grid mapping

ABACUS reader-v1 may append the following block when the SCF list is reduced:

```text
full_kmap N_FULL
id kx ky kz SCF_representative
```

There are `N_FULL = nk1 * nk2 * nk3` rows. Both indices are 1-based;
`SCF_representative` indexes the saved SCF list, not the full grid. Cartesian
coordinates have the same Bohr$^{-1}$ units as the preceding rows. The current
extension requires one Coulomb representative per listed SCF point.

When present, this block is authoritative for the full-grid coordinates and
full-to-SCF assignment. LibRPA checks coverage, unique points modulo reciprocal
lattice vectors, the actual mesh offset, representative membership, and folded
weights before using it. It does not overwrite this mapping with an independently
inferred one. Structure-side symmetry operations still determine the AO/ABF
transformations; `full_kmap` supplies file indices, not symmetry matrices or
additional wave functions.

Files without this marker retain the original reading and inference path,
including historical FHI-aims and ABACUS files and their ignored trailing
summaries. No new block is required for those inputs. Malformed explicit
`full_kmap` blocks are rejected rather than silently falling back.

A few remarks

- If the SCF k-point count equals `nk1 * nk2 * nk3`, no spatial symmetry was used to shrink the SCF k-list.
- If that count is larger than the Coulomb irreducible count, time-reversal symmetry was used for Coulomb matrices.
- If the SCF k-point count is smaller than `nk1 * nk2 * nk3`, spatial symmetry was used and `stru_out` must provide symmetry operations.

(cs-data)=
## `Cs_data*`

These files contain the localized RI triple coefficients.
The same format rules apply to the files selected by `prefix_lri_coeff_shrink`,
whose default prefix is [`Cs_shrinked_data`](#cs-data).

LibRPA supports two reader versions:

- `version_lri_reader = 0`: legacy text or legacy binary files
- `version_lri_reader = 1`: binary v1 files with a block table and payload offsets
- `version_lri_reader = -1`: auto-detect from the first file matching the prefix

Do not mix legacy and v1 files under the same prefix. Files are discovered by
prefix only, so suffixes such as `.txt`, `.dat`, or no suffix are all accepted
by the v1 reader.
For full/shrink Cs reads, the two prefix families are treated as distinct; a
full prefix must not select shrink files and vice versa.

### Legacy text format

In plain text format, each file has a header with two integers:
total number of atoms and number of periodic unit cells.
Then till the end of file, the data is formatted as blocks of RI coefficient $C$ on each pair of atoms and unit cell
```
i_atom_1  i_atom_2  n_1  n_2  n_3  n_basis_1  n_basis_2  n_aux_basis_1
C(1, 1, 1)
...
C(n_aux_basis_1, n_basis_2, n_basis_1)
```
Here `C` is the RI coefficients between the atom `i_atom_1` and `i_atom_2` in unit cells separated by
lattice vector $\mathbf{R} = n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 + n_3 \mathbf{a}_3$.
The auxiliary basis is located on `i_atom_1`. The number of basis functions on `i_atom_1` and `i_atom_2` 
are `n_basis_1` and `n_basis_2`, respectively. The number of auxiliary functions is `n_aux_basis_1`.
The indices of `C` runs in the Fortran order, i. e. the first index runs the fastest.

Equivalently, the file order is

```text
for i in basis_1:
  for j in basis_2:
    for mu in aux_basis_1:
      C(mu, j, i)
```

and LibRPA stores the loaded block as a row-major matrix
`Cs(i * n_basis_2 + j, mu)`.

### Legacy binary format

In binary format, the data is organized similarly in the plain text format, except for an extra integer
is included in the header, which is the number of atom pairs and lattice vectors included in the file.
The coefficients are saved in double precision. To better illustrate the format of binary file, the
following Python snippet could be helpful
```python
import struct
import numpy as np

# ensure that "Cs_data_0" exists and was generated with binary output mode in DFT code
cfile_path = "Cs_data_0.txt"

with open(cfile_path, 'rb') as h:
    n_atoms, n_cells, n_apcell_file = struct.unpack('iii', h.read(12))
    for _ in range(n_apcell_file):
        a1, a2, r1, r2, r3, nb1, nb2, nbb1 = struct.unpack('i' * 8, h.read(4 * 8))
        apcell = (a1, a2, r1, r2, r3)
        array_size = nb1 * nb2 * nbb1
        array = np.array(struct.unpack('d' * array_size, h.read(8 * array_size)))
        array = np.reshape(array, (nb1, nb2, nbb1))
        apcells[apcell] = array
```

### Binary v1 format

The v1 LRI coefficient format is designed for random-access reading and
parallel scheduling. It begins with a fixed-size binary header followed by a
block table and dense double-precision payloads. All integer fields use native
32-bit or 64-bit binary representation as listed below, and all floating-point
fields are native double precision.

Header:

| Field | Type | Meaning |
|-------|------|---------|
| `marker` | `int32` | must be `-10267453` |
| `n_atoms` | `int32` | number of atoms |
| `n_cells` | `int32` | number of periodic unit cells represented by the dataset |
| `n_apcell_file` | `int64` | number of valid atom-pair/cell blocks in this file |
| `n_apcell_file_max` | `int64` | number of reserved block-table records |

The header is followed by `n_apcell_file_max` block-table records. The first
`n_apcell_file` records are valid; the remaining records, if any, must be
zero-filled padding.

Each block-table record has the following fields:

| Field | Type | Meaning |
|-------|------|---------|
| `i_atom_1` | `int32` | 1-based atom index for the auxiliary-basis center |
| `i_atom_2` | `int32` | 1-based atom index for the second orbital-basis center |
| `R1`, `R2`, `R3` | `int32` | unit-cell displacement between the two atoms |
| `max_abs` | `double` | maximum absolute coefficient in the block, used for threshold filtering |
| `offset` | `int64` | absolute byte offset of this block's payload from the start of the file |

The v1 record does not store `n_basis_1`, `n_basis_2`, or `n_aux_basis_1`.
Those dimensions are reconstructed from the wave-function and auxiliary basis
metadata that have already been loaded from the dataset, normally from
[`basis_out`](#basis-out):

```text
n_basis_1     = number of wave-function basis functions on i_atom_1
n_basis_2     = number of wave-function basis functions on i_atom_2
n_aux_basis_1 = number of auxiliary basis functions on i_atom_1
```

The payload for one record contains
`n_basis_1 * n_basis_2 * n_aux_basis_1` double-precision values in the same
logical order as the legacy format:

```text
for i in basis_1:
  for j in basis_2:
    for mu in aux_basis_1:
      C(mu, j, i)
```

When writing or inspecting v1 files from Python, this corresponds to:

```python
import struct
import numpy as np

marker = -10267453

with open("Cs_data_0.dat", "rb") as h:
    marker_read, n_atoms, n_cells = struct.unpack("iii", h.read(12))
    if marker_read != marker:
        raise ValueError("not an LRI coefficient v1 file")
    n_blocks, n_blocks_reserved = struct.unpack("qq", h.read(16))

    records = []
    for _ in range(n_blocks_reserved):
        ia1, ia2, r1, r2, r3 = struct.unpack("iiiii", h.read(20))
        max_abs, offset = struct.unpack("dq", h.read(16))
        records.append((ia1, ia2, (r1, r2, r3), max_abs, offset))

    ia1, ia2, R, max_abs, offset = records[0]
    # The dimensions come from basis_out. These are example values.
    n_basis_1, n_basis_2, n_aux_basis_1 = 5, 5, 18
    h.seek(offset)
    raw = h.read(8 * n_basis_1 * n_basis_2 * n_aux_basis_1)
    coeff = np.frombuffer(raw, dtype=np.float64).reshape(
        (n_basis_1, n_basis_2, n_aux_basis_1)
    )
```

Legacy Cs files can be converted one file at a time with:

```bash
c++ -std=c++17 -O2 -o convert_legacy_Cs.exe utilities/convert_legacy_Cs.cpp
./convert_legacy_Cs.exe Cs_data_0.txt Cs_data_0.dat --overwrite
```

Then set `version_lri_reader = 1` in `librpa.in`, or leave
`version_lri_reader = -1` to auto-detect the v1 marker.

(band-out)=
## `band_out`

This file contains band energies and occupation numbers from the mean-field starting-point calculation.
It has a 5-line header
```
n_k_points
n_spins
n_states
n_basis
e_fermi
```
The first 4 lines contain an integer in each. The 5th line is a float number, which is the Fermi energy
in Hartree unit.

The remaining lines consists of `n_k_points*n_spins` blocks of `n_states+1` lines, in the format of
```
i_k_point    i_spin
1           f_1        e_1_ha      e_1_ev
2           f_2        e_2_ha      e_2_ev
3           f_3        e_3_ha      e_3_ev
...
n           f_n        e_n_ha      e_n_ev
...
```
This block contains the energies and occupation numbers of states $\left|\psi_{n,k\sigma}\right\rangle$
`i_k_point` marks the index of SCF k-point $k$.
`i_spin` specify the spin channel $\sigma$.
In each of the following lines, the first integer species the index of state.
The 3 float numbers stand for the occupation number, the energy in Hartree unit and that in electronvolt
unit, respectively.
For spin-unpolarized calculation, `f_n` is a number from 0 to 2, otherwise it is from 0 to 1.

(ks-eigenvector-scf)=
## `KS_eigenvector*`
These files contain the wave functions (eigenvectors) from the starting-point calculation expanded by orbital basis.
LibRPA auto-detects the legacy text format and the binary v1 format from the file header.
Do not mix legacy and v1 files under the same `prefix_eigvecs_scf`.

### Legacy text format
Each file can be divided in blocks of `n_states*n_basis*n_spins+1` lines,
where `n_states`, `n_basis` and `n_spins` will be extracted from
[`band_out`](#band-out).
Each block stores the data for a particular k-point, $c^i_{n,k\sigma}$:
```
i_k_point
c(1,1,1)_real c(1,1,1)_imag
...
c(i,n,s)_real c(i,n,s)_imag
...
```
The first line contains single integer, the index of the SCF k-point of following data.
The remaining lines store the data with running index $i$, $n$, $\sigma$ in C-style row-major order,
i. e., spin index runs fastest, then state index and finally basis index.
Each line has two float numbers, which are the real and imaginary part of $c^i_{n,k\sigma}$.

### Binary v1 format

The binary v1 format is intended for parallel k-point reading. Values are
native-endian; integers are `int32` unless stated otherwise. The leading header stores:

```
int32 marker          = -12345679
int32 kind            = 28
int32 nkpoints_local
int32 nspins
int32 nstates
int32 nbasis_wfc
```

Only `kind = 28`, packed `complex<double>`, is currently implemented. Each
complex number is stored as two consecutive `double` values, real then
imaginary.

The header is followed by `nkpoints_local` block records:

```
int32 ik              # 1-based SCF k-point index
int64 payload_offset  # absolute byte offset in this file
```

Each payload block contains one k-point with
`nspins * nspinor * nstates * nbasis` complex numbers. For non-spinor data,
`nspinor = 1` and `nbasis = nbasis_wfc`. For spinor data, `nspinor = 2` and
`nbasis = nbasis_wfc / 2`.

The payload order is:

```
for ispin
  for ispinor
    for istate
      for ibasis
```

where `ibasis` is the fastest index.

## `velocity_matrix`

This file stores the PyATB velocity matrix used by the head/wing correction.
LibRPA auto-detects the legacy text format and the binary v1 format from the
file header.

For an ABACUS input preset, the driver discovers the file from
`prefix_velocity`: it prefers `<prefix_velocity>.txt` and accepts the legacy
extensionless `<prefix_velocity>` name as a fallback.

### Legacy text format

The legacy text file starts with:

```
nkpoints
nspins
nbands
naos
```

It then stores blocks ordered by spin, k-point, and Cartesian component:

```
ialpha ik ispin
v(1,1)_real v(1,1)_imag
...
v(i,j)_real v(i,j)_imag
```

All indices in the block header are 1-based.

### Binary v1 format

Values are native-endian. Integers are `int32` unless stated otherwise. The
leading header stores:

```
int32 marker          = -12345680
int32 kind            = 29
int32 nkpoints_local
int32 nspins
int32 nbands
int32 naos
int32 nalpha          = 3
```

Only `kind = 29`, packed `complex<double>`, is currently implemented. Each
complex number is stored as two consecutive `double` values, real then
imaginary.

The header is followed by `nkpoints_local` block records:

```
int32 ik              # 1-based source k-point index
int64 payload_offset  # absolute byte offset in this file
```

A parallel producer may split the source k-points across several files named
`velocity_matrix`, `velocity_matrix_1.dat`, `velocity_matrix_2.dat`, etc. In
that case each file uses its own `nkpoints_local`, while `ik` still refers to
the 1-based index in `k_path_info`.

Each payload block contains one source k-point with
`nspins * 3 * nbands * nbands` complex numbers. The payload order is:

```
for ispin
  for ialpha
    for iband
      for jband
```

where `jband` is the fastest index.

(coulomb-mat)=
## `coulomb_mat*`

These files contain the bare Coulomb matrices in the auxiliary basis.
The truncated Coulomb matrices used in GW are selected by `prefix_coul_cut`,
whose default prefix is [`coulomb_cut`](#coulomb-cut), and use the same formats.

LibRPA supports two reader versions:

- `version_coul_reader = 0`: legacy text or legacy binary rectangular matrix blocks
- `version_coul_reader = 1`: binary v1 files with atom-pair blocks
- `version_coul_reader = -1`: auto-detect from the first file matching the prefix

Do not mix legacy and v1 Coulomb files under the same prefix. The legacy reader
expects filenames that both start with the selected prefix and end in `.txt`.
The v1 reader discovers files by prefix only, so binary files such as
`coulomb_full_iq_1.dat` are accepted if `prefix_coul_full = coulomb_full_iq`.

### Legacy text format

A single header line contains an integer, the number of irreducible k-point at
which the Coulomb matrices are computed.
The remaining part of the file is organized in blocks
```
n_aux_basis    row_start    row_end    col_start    col_end
i_k_point      k_weight
v(row_start, col_start  )_real       v(row_start, col_start  )_imag
v(row_start, col_start+1)_real       v(row_start, col_start+1)_imag
...
v(row_end, col_end)_real             v(row_end, col_end)_imag
```
where
- integer `n_aux_basis` is the total number of auxiliary basis functions.
- integer `row_start`, `row_end`, `col_start` and `col_end` mark the submatrix of the full Coulomb matrix
  that this block contain.
- integer `i_k_point` is the index of k-point of the current Coulomb matrix, in the full k-point list.
- float number `k_weight` is the weight of the irreducible k-points.

After the block header, there should be `(row_end-row_start+1)` times `(col_end-col_start+1)` lines
for the actual matrix element data. Each line contains two float numbers, which are the real and imaginary
parts of the element. The data is ordered in C-style row major.

### Legacy binary format

The legacy binary Coulomb format stores the same rectangular blocks as the
legacy text format. It starts with two native `int32` values:

1. total number of irreducible q-points
2. number of q-point blocks stored in this file

Each q-point block then contains:

| Field | Type | Meaning |
|-------|------|---------|
| `n_aux_basis` | `int32` | total number of auxiliary basis functions |
| `row_start`, `row_end` | `int32` | 1-based inclusive row range |
| `col_start`, `col_end` | `int32` | 1-based inclusive column range |
| `i_q_point` | `int32` | 1-based q-point index in the full k-point list |
| `q_weight` | `double` | weight of the irreducible q-point |
| payload | `complex<double>[]` | dense row-major block values |

Here `complex<double>` is stored as two consecutive double values, real part
first and imaginary part second.

### Binary v1 atom-pair format

The v1 Coulomb format stores one q-point per file. Each file contains dense
atom-pair blocks in the auxiliary basis. Only upper-triangular atom pairs
`I <= J` are stored because the full matrix is Hermitian.

Header:

| Field | Type | Meaning |
|-------|------|---------|
| `marker` | `int32` | must be `-20129433` |
| `i_q_point` | `int32` | 1-based q-point index in the full k-point list |
| `n_aux_basis` | `int32` | total number of auxiliary basis functions |
| `value_flag` | `int32` | `0` for real double payloads, `1` for complex payloads |
| `n_atoms` | `int32` | number of atoms |
| `n_blocks` | `int32` | number of stored atom-pair blocks |

The header is followed by `n_atoms` `int32` values giving the number of
auxiliary basis functions on each atom. These counts must sum to
`n_aux_basis` and must match [`basis_out`](#basis-out).

Next comes a block table with `n_blocks` records:

| Field | Type | Meaning |
|-------|------|---------|
| `i_pair` | `int32` | zero-based upper-triangular atom-pair index |
| `offset` | `int64` | absolute byte offset of this atom-pair payload |

The atom-pair index enumerates upper-triangular pairs in this order:

```text
(0,0), (0,1), ..., (0,n_atoms-1), (1,1), (1,2), ...
```

For a pair `(I, J)` with `I <= J`, the corresponding index is:

```text
i_pair = I * n_atoms - I * (I - 1) / 2 + (J - I)
```

Each payload is a dense row-major matrix of shape
`(n_aux_basis_on_atom_I, n_aux_basis_on_atom_J)`.
If `value_flag = 1`, values are stored as `complex<double>` pairs
`real, imag`. If `value_flag = 0`, values are stored as real `double` values
and LibRPA sets the imaginary part to zero.

Unlike the legacy format, v1 Coulomb files do not store q-point weights.
Keep [`bz_sampling_out`](#bz-sampling-out) in the dataset so q-point weights and irreducible
q-point mapping are available separately.

A minimal Python reader for the v1 header and the first atom-pair block is:

```python
import struct
import numpy as np

marker = -20129433

with open("coulomb_full_iq_1.dat", "rb") as h:
    header = struct.unpack("iiiiii", h.read(24))
    marker_read, iq, naux, value_flag, n_atoms, n_blocks = header
    if marker_read != marker:
        raise ValueError("not a Coulomb v1 file")

    atom_naux = struct.unpack(f"{n_atoms}i", h.read(4 * n_atoms))
    table = []
    for _ in range(n_blocks):
        i_pair = struct.unpack("i", h.read(4))[0]
        offset = struct.unpack("q", h.read(8))[0]
        table.append((i_pair, offset))

    i_pair, offset = table[0]
    nrow, ncol = atom_naux[0], atom_naux[0]
    h.seek(offset)
    if value_flag == 1:
        block = np.frombuffer(h.read(16 * nrow * ncol), dtype=np.complex128)
    else:
        block = np.frombuffer(h.read(8 * nrow * ncol), dtype=np.float64)
    block = block.reshape((nrow, ncol))
```

Legacy Coulomb files can be converted to v1 with the MPI converter:

```bash
mpicxx -std=c++17 -O2 -o convert_legacy_coulomb_mat_mpi.exe \
  utilities/convert_legacy_coulomb_mat_mpi.cpp
mpirun -np 4 ./convert_legacy_coulomb_mat_mpi.exe /path/to/dataset \
  -i coulomb_mat -o coulomb_full_iq
```

Use the generated prefix in `librpa.in`:

```text
prefix_coul_full = coulomb_full_iq
version_coul_reader = 1
```

For truncated Coulomb data, convert the [`coulomb_cut`](#coulomb-cut) files with a separate
output prefix and set `prefix_coul_cut` accordingly.

(coulomb-cut)=
## `coulomb_cut*`

These files are the same as [`coulomb_mat*`](#coulomb-mat), but store the truncated Coulomb to
be used in the GW calculation.

(dielecfunc-out)=
## `dielecfunc_out`

The file `dielecfunc_out` stores the macroscopic dielectric function on the
imaginary-frequency grid used for dielectric-head correction.

Each data line contains three columns:

1. imaginary frequency
2. real part of the dielectric function
3. imaginary part of the dielectric function

LibRPA reads the first two columns for the imaginary-frequency dielectric
function; the third column is accepted for compatibility with exported data
that writes complex values.

(vxc-out)=
## `vxc_out`

The file `vxc_out` stores the exchange-correlation potential for electronic states on the SCF k-point grid.

The header consists of three lines:

```text
n_k_points
n_spins
n_states
```

The header is followed by n_k_points * n_spins * n_states data lines.
Each line contains two columns and corresponds to one state identified by the tuple `(i_k, i_spin, i_state)`.
The data are ordered such that `i_state` runs fastest, followed by `i_spin`, and then `i_k`. In other words, the lines are arranged as

```
# i_k       i_spin      i_state
    0            0            0
    0            0            1
    0            0            2
...
    0            0   n_states-1
    0            1            0
...
    1            0            0
...
```

The two columns contain the same exchange-correlation potential,
first in Hartree unit while the second in eV.

## Input files for band structure calculation

For band-structure calculations, LibRPA reads the following input files:

- [`band_kpath_info`](#band-kpath-info)
- [`band_KS_eigenvalue_k_{ik:05d}.txt`](#band-ks-eigenvalue)
- [`band_KS_eigenvector_k_{ik:05d}.txt`](#band-ks-eigenvector)
- [`band_vxc_k_{ik:05d}.txt`](#band-vxc)

Here `ik` is the 1-based index of the k-point along the band path, written with five digits.

(band-kpath-info)=
### `band_kpath_info`

The file `band_kpath_info` defines the k-point path used for the band-structure calculation.

The first line contains four integers:

1. number of basis functions
2. number of states
3. number of spin channels
4. number of k-points on the band path

For example:

```text
18    18     1    10
```

The remaining `n_kpath_points` lines each contain three floating-point numbers, giving the fractional coordinates of one k-point on the band path.

For example:

```text
0.500000000000000000E+00   0.500000000000000000E+00   0.500000000000000000E+00
```

Each such line represents one k-point in fractional reciprocal coordinates.

(band-ks-eigenvalue)=
### `band_KS_eigenvalue_k_{ik:05d}.txt`

For each k-point on the band path, the file `band_KS_eigenvalue_k_{ik:05d}.txt` stores the Kohn-Sham eigenvalues used by LibRPA.
Each line corresponds to one state and contains five columns:

1. spin index
2. state index
3. occupation number
4. Kohn-Sham eigenvalue in Hartree
5. Kohn-Sham eigenvalue in eV

For example:

```text
1       3   0.200000000000000000E+01  -0.658097773108510893E+02  -0.179077515426494506E+04
```

This line indicates that, at the selected k-point,

- the spin index is `1`
- the state index is `3`
- the occupation number is `2.0`
- the eigenvalue is given both in Hartree and in eV

The data are ordered such that state index `i_state` runs fastest and followed by spin index `i_spin`.

(band-vxc)=
### `band_vxc_k_{ik:05d}.txt`

For each k-point on the band path, the file `band_vxc_k_{ik:05d}.txt` stores the diagonal matrix elements of the exchange-correlation potential for the corresponding Kohn-Sham states.

Each line contains three columns:

1. spin index
2. state index
3. exchange-correlation potential in Hartree

For example:

```text
1       3  -0.562542321738239171E+01
```

This line gives the exchange-correlation potential for state `3` in spin channel `1` at the selected k-point.
The data are ordered such that state index `i_state` runs fastest and followed by spin index `i_spin`.

(band-ks-eigenvector)=
### `band_KS_eigenvector_k_{ik:05d}.txt`

For each k-point on the band path, the file `band_KS_eigenvector_k_{ik:05d}.txt` stores the Kohn-Sham eigenvectors at that k-point.
The file contains a complex array of shape `(n_spins, n_states, n_basis)`,
written in binary format using C-style ordering.

### Shrink transform v1 format

When `use_shrink_abfs = t`, `prefix_shrink_sinvS` selects the transform from
the compressed auxiliary basis back to the full auxiliary basis. The default
legacy prefix is `shrink_sinvS_`; ABACUS reader-v1 producer runs should use
`sinvS_`.

The ABACUS reader-v1 binary format starts with:

- `int32 marker = -30241621`
- `int32 nblocks`

Each block record stores:

- `int32 iq` (1-based irreducible q-point index)
- `int32 nrow_total`, `int32 ncol_total`
- `int32 begin_row`, `int32 end_row`, `int32 begin_col`, `int32 end_col`
- `double q_weight`
- `int64 payload_offset`

The payload is row-major `complex<double>` data for the rectangular block
described by the row and column range. Multiple files and blocks may contribute
to the same q-point.
