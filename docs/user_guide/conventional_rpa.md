# Conventional k-space RPA

A frequency-only grid selects the conventional method for `task = rpa`. This
method computes the response by a direct sum over band transitions at each
frequency. Use it to check the frequency convergence of the space-time method,
or to calculate on a grid without time nodes. The equations and the cost are in
`docs/theory/rpa/conventional.md` in the source tree.

## Input

Use these options with an existing RPA dataset:

```ini
task = rpa
tfgrids_type = GL
nfreq = 48
```

The method works with all `parallel_routing` values, and with builds without
LibRI. With `atompair` and `libri`, MPI ranks share the atom pairs. With `rtau`,
they share the q points. In a build without LibRI, set
`use_scalapack_ecrpa = false` with `atompair`, because the ScaLAPACK energy
routine needs LibRI.

`GL` selects Gauss-Legendre quadrature on the positive imaginary-frequency axis.
The other frequency-only grids, such as `GC-I` and `GL-II`, also select this
method. The default `minimax` grid has time nodes and selects the space-time
method.

Increase `nfreq` until the correlation energy converges to the tolerance that
you need. For the H2O and Si test data, the change from 24 to 48 points is about
2.5e-7 Ha.

In the C and C++ APIs, set `tfgrids_type` to `LIBRPA_TFGRID_GAUSS_LEGENDRE`.

## Requirements

- Supply the eigenvectors of all k points of the full SCF mesh. The mesh must
  contain -k for each k. The order of the k points can be arbitrary.
- Use scalar or collinear-spin wave functions.
- Keep `use_kpara_scf_eigvec`, `use_symmetry_rpa` and `use_shrink_abfs` at their
  default value, `false`.
- Use grids with strictly positive frequencies.

`n_bands_chi0` and `n_bands_exclude_chi0` apply to both bands of a transition.
Do not set a cutoff inside a degenerate multiplet: the retained states then
depend on the eigenvector gauge, and LibRPA can stop with a non-Hermitian
response error. The same error occurs for time-reversal-broken input and for a
mesh without -k.

GW and cRPA need a grid with time nodes, and they stop at the start of the
calculation on a frequency-only grid.

Head and wing replacement (`replace_w_head`) has not been validated with this
method.
