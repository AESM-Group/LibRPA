# Bare chi0 export

`output_chi0_rf = t` writes bare chi0 at the grid node of smallest |omega| to
`<output_dir>/chi0_lowest`. `output_chi0_static = t` writes the zero-frequency
response to `<output_dir>/chi0_static`. Both default to false and apply only to
RPA and single-shot G0W0; QSGW/cRPA ignore them with a warning.

Both exports use the full auxiliary basis. With `use_shrink_abfs = t`, the
lowest-frequency export rebuilds chi0 in that basis with the same grid, band
selection and thresholds. No screening or head/wing correction is applied.

The static export requires a minimax grid and a separate space-time build:

    chi0(q, i*0) = 2 * integral_0^infty chi0(q, i*tau) d tau

Its time nodes and weights are twice the original minimax nodes and weights.
Only frequency slot 0 is meaningful; the GW grid is unchanged. Check convergence
with `nfreq`: a finite quadrature is approximate, particularly near small gaps.
The additional build is labelled in the log and timing table.

## Real-space files

Without full-crystal symmetry restoration, `chi0_rf.info` contains:

    LIBRPA_CHI0_RF 1
    <omega in Ha> <n_aux> <period_x> <period_y> <period_z>
    grid_index <index> n_grid_nodes <N> source <lowest_grid_node|static_tau_integral>

`Chi0_Mu_<I>_Nu_<J>_iR_<r>_ifreq_0.mtx` contains an atom-pair block in
MatrixMarket coordinate complex general format (one-based matrix indices).
Atom and lattice-vector indices are zero-based; the comment gives R in lattice
units. `ifreq_0` is the single export slot, not the original grid index.
The transform is `chi0(R) = sum_q exp(-2 pi i q.R) chi0(q) / N_q`.

## Full-crystal symmetry files

This path writes representative q-points instead of real-space blocks:

- `chi0_q.info`: the same layout as above, with magic `LIBRPA_CHI0_Q_GRID 1`.
- `chi0_q_<id>.dat`: header `LIBRPA_CHI0_Q 1`, then `n_aux id id`, then
  `mu nu real imag` entries. Matrix and full-grid q indices are one-based.
- `rpa_delta_q_<target>.dat`: header `LIBRPA_DELTA 1`, then
  `n_aux target representative`, then the same matrix entry format.
- `rpa_symmetry_chi0.dat`: header `LIBRPA_SYMMETRY_ALL 1`, then `n_aux n_q`,
  then `BARE_CHI0 D_CHI0_DAGGER`. Each following line gives target id,
  representative id, one-based spatial operation index, time-reversal flag,
  target q, representative q (both fractional), and the Delta file name.

Delta restores `chi0(target) = Delta chi0(rep) Delta^dagger`, with `chi0(rep)`
conjugated when the time-reversal flag is 1. Write failures are propagated to
all MPI ranks.
