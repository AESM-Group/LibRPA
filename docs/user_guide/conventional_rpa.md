# Conventional k-space RPA

The conventional method computes the response by a direct sum over band
transitions at each frequency. It is available for `task = rpa`. Use it to check
the space-time method, or to calculate on a grid without time nodes. The
equations and the cost are in `docs/theory/rpa/conventional.md` in the source
tree.

## Input

The option `chi0_method` selects the method of the response function:

| Value | Method |
|---|---|
| `auto` (default) | `space_time` for a grid with time nodes, `conventional` for a frequency-only grid |
| `space_time` | Space-time method. The grid must have time nodes. |
| `conventional` | Conventional method at the frequency nodes of the grid |

Only the `minimax` and `evenspaced_tf` grids have time nodes. The other grids,
such as `GL`, `GC-I` and `GL-II`, have frequency nodes only.

To use Gauss-Legendre quadrature on the positive imaginary-frequency axis:

```ini
task = rpa
tfgrids_type = GL
nfreq = 48
```

Increase `nfreq` until the correlation energy converges to the tolerance that
you need. For the H2O and Si test data, the change from 24 to 48 points is about
2.5e-7 Ha.

To compare the two methods with the same frequency quadrature, use a minimax
grid in two calculations, one with each method:

```ini
task = rpa
tfgrids_type = minimax
nfreq = 16
chi0_method = conventional
```

The difference between the two energies then comes only from the space-time
method, for example from the time grid and the time-to-frequency transform. For
the H2O and Si test data on a 16-point minimax grid, the two energies agree to
1e-9 Ha.

In the C and C++ APIs, set `chi0_method` to `LIBRPA_METHOD_CONVENTIONAL`, and
`tfgrids_type` to the grid, for example `LIBRPA_TFGRID_GAUSS_LEGENDRE`.

## Parallelization

The method works with all `parallel_routing` values, and with builds without
LibRI. With `atompair` and `libri`, MPI ranks share the atom pairs. With `rtau`,
they share the q points. In a build without LibRI, set
`use_scalapack_ecrpa = false` with `atompair`, because the ScaLAPACK energy
routine needs LibRI. Otherwise the calculation stops at the start.

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

GW and cRPA use the space-time method only. They stop at the start of the
calculation on a frequency-only grid or with `chi0_method = conventional`.

Head and wing replacement (`replace_w_head`) has not been validated with this
method.
