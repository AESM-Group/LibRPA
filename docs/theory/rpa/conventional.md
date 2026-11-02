# Conventional k-space RPA

LibRPA can compute the independent response directly from band transitions at imaginary frequencies.
This method avoids the imaginary-time Green functions and the time-to-frequency transform.
It uses the same Coulomb matrices, RPA energy routines, and output format as the space-time method.

## Conventions

For an initial state $(n,\mathbf{k})$ and a final state $(m,\mathbf{k}+\mathbf{q})$, the transition density is

$$
D^{I\mu}_{nm}(\mathbf{k},\mathbf{q}) =
\sum_{J\mathbf{R}ij} C^{I\mu}_{Ii,Jj}(\mathbf{R})
\left[
 c^*_{n\mathbf{k},Ii} c_{m,\mathbf{k}+\mathbf{q},Jj}
 e^{i(\mathbf{k}+\mathbf{q})\cdot\mathbf{R}}
 + c^*_{n\mathbf{k},Jj} c_{m,\mathbf{k}+\mathbf{q},Ii}
 e^{-i\mathbf{k}\cdot\mathbf{R}}
\right].
$$

Both terms are necessary, including for a same-atom coefficient block.
Wave vectors in this equation use physical reciprocal units.
The implementation includes the factor $2\pi$ for LibRPA's internal reciprocal coordinates.

The response follows the band sum

$$
\chi_{0,\mu\nu}(\mathbf{q},i\omega) =
\sum_{s\mathbf{k}nm}
\frac{w_{sn\mathbf{k}}-w_{sm,\mathbf{k}+\mathbf{q}}}
{\epsilon_{sn\mathbf{k}}-\epsilon_{sm,\mathbf{k}+\mathbf{q}}+i\omega}
D^\mu_{snm}(\mathbf{k},\mathbf{q})
D^{\nu*}_{snm}(\mathbf{k},\mathbf{q}).
$$

The stored occupation weights $w$ already include the spin multiplicity and $1/N_k$.
Eigenvalues and frequencies use Hartree units.
The q weights enter the existing RPA energy integral once.
The implementation checks cancellation of the odd-frequency part before it passes the Hermitian response to that integral.

MPI ranks own upper-triangular auxiliary atom blocks.
Each rank obtains the RI coefficients for the atoms in its blocks.
The code processes one q point and one initial band at a time and uses matrix products for the final-band sum.
It does not allocate a square matrix in the space of transitions or store all k/q transition densities.
This method is intended for direct frequency calculations and checks of the space-time approximation.
Its band-sum cost can exceed the space-time cost for large systems.

## Relation to the historical branch

The reference branch is `upstream/conventional_rpa`, commit `860b9b87dac34f85d075ec68ec8a70fc48faef4d`.
It contains two commits after the 0.3.0 ancestor.
The new implementation follows its RI-to-band transition-density approach within the current `Chi0` interface.

The historical response loop retains only equal k indices, and its energy routine uses the first response block.
It also has an explicit factor of 0.5 in the transition energies and its own determinant calculation.
These parts are not numerical references for the current full-mesh implementation.
The current code uses all connected k pairs and the mainline energy routines.

## Usage

### Supported inputs

- Supply every k point of the full, uniform SCF mesh and its eigenvectors. The input order can be arbitrary.
- Supply scalar or collinear-spin wave functions. Spinor wave functions and symmetry restoration are not supported by this route.
- Use the existing LibRI distribution for the RI coefficients. Each coefficient block must have one MPI owner.
- Keep the eigenvectors replicated across MPI ranks and disable auxiliary-basis shrinking.
- Use strictly positive frequencies. The static metallic limit at zero frequency requires a separate treatment.

The response includes the supplied fractional occupations and all retained band pairs at each positive frequency.
`n_bands_chi0` and `n_bands_exclude_chi0` apply to both ends of a transition.
Selected-band cRPA response and time-reversal-broken response are outside this implementation's supported scope.
The code rejects a nonzero odd-frequency response because the existing energy routines require Hermitian response matrices.

### Input

Use the following options with an existing RPA dataset:

```text
task = rpa
parallel_routing = libri
tfgrids_type = GL
nfreq = 48
use_scalapack_ecrpa = t
use_symmetry_rpa = f
use_kpara_scf_eigvec = f
use_shrink_abfs = f
```

Increase `nfreq` until the correlation energy converges to the required tolerance.
The value 48 is a starting point, not a convergence guarantee.
`GL` selects Gauss–Legendre quadrature on the positive imaginary-frequency axis.
Other frequency-only grids also use the direct response.
The default `minimax` grid includes time nodes and continues to use the space-time method.

The C and C++ APIs use the same selection through `LibrpaOptions::tfgrids_type`.
Set it to `LIBRPA_TFGRID_GAUSS_LEGENDRE` and use `LIBRPA_ROUTING_LIBRI`.
No new public API fields are required.
