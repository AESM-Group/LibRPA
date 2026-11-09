# Analytic continuation methods

`anacon_method` selects `pade` (the unchanged default), `aaa`, or `ridge` for GW
quasiparticle energies, spectral functions, imaginary-grid resampling, and QSGW
correlation-potential matrices. cRPA static and real-axis interaction output
uses the same method and penalties on Delta U=U-v without a fermionic shift.
The standalone cRPA driver keeps its original all-node parameter count, independent
of the GW `n_params_anacon` setting. Library callers can set the count explicitly
with `librpa_continue_crpa_with_options`. All methods supply complex values and analytic
derivatives to the existing quasiparticle solvers.

| Parameter | Meaning | Default |
| --- | --- | --- |
| `anacon_method` | `pade`, `aaa`, or `ridge` | `pade` |
| `n_params_anacon` | Padé selected-node count, AAA support cap, or ridge coefficient count | `-1`: source point count |
| `n_params_anacon_resample` | Coefficient/support count for the first fit when resampling | negative: use `n_params_anacon` |
| `anacon_aaa_tolerance` | Maximum AAA source residual relative to the largest datum | `1e-13` |
| `anacon_ridge_lambda` | Ridge coefficient penalty | `1e-12` |
| `anacon_ridge_den_weight` | Relative denominator-coefficient penalty | `10` |

The same coefficient count does not imply the same complexity. Padé retains the
existing Thiele construction and barycentric fallback. AAA selects support nodes
greedily and computes barycentric weights with a complex Loewner SVD. A capped
AAA fit may stop before reaching tolerance, including with the default count
when supports are exhausted. The count is capped at the number of distinct input
nodes; the Loewner SVD keeps at least one unsampled node. The rank-local log
reports the first unmet-tolerance AAA fit per process, comparing its relative
residual against the relative tolerance and also giving the absolute residual.
Further AAA tolerance warnings are suppressed for that process lifetime; this
first-fit report is not a count or a worst-case summary of subsequent fits.
Source residuals include zero-weight
supports; evaluating a non-support input point returns the fitted value, not an
unconditional copy of the original datum.

Ridge uses every input node in a scaled, linearized rational least-squares fit.
It penalizes nonconstant numerator and denominator coefficients, with squared
power weights. It does not minimize the nonlinear value residual directly.
The historical normal-equation solver and small diagonal-shift retries are
retained. A zero penalty selects an unregularized rational coefficient fit,
not Thiele Padé. For `n_params_anacon = 8`, Padé selects eight nodes, AAA uses
at most eight supports, and ridge has four numerator coefficients (degree three)
and four nonconstant denominator coefficients (degree four).

Singular denominators cause an explicit failure; there is no
denominator clipping or silent change to another method. GW QP diagnostics leave
failed states as NaN, including when `override_qpe_solver_nan` is enabled: an
exception supplies no usable final iterate. A failed Hedin reference leaves all
states sharing that shift as NaN. Spectral output marks only affected frequencies
as NaN, or the whole state if construction fails. The rank-local log records the
state, reason and source samples. Invalid inputs remain errors; these handlers
do not suppress configuration errors. A QSGW static-matrix failure remains fatal rather than installing a partial
Hamiltonian update. cRPA retains its existing per-entry status bits and NaN
values for failed continuation entries; inspect those diagnostics before using
the returned tensor.

None of these methods constrains causal poles or guarantees accurate real-axis
spectra. A small imaginary-axis residual, a smooth band plot, and successful
QSGW iteration are separate from physical acceptance. The fit equations and derivatives are in
the source-only note `docs/theory/analytic_continuation.md`.

## Compatibility and reproducibility

Omitting `anacon_method` retains the original Padé method. AAA and ridge require
explicit selection. Existing reference inputs, reference data and tolerances
remain unchanged. Choosing ridge does not establish a physical accuracy ranking.

The C/C++ selector is `LibrpaAnalyContMethod`: `LIBRPA_ANACON_PADE`,
`LIBRPA_ANACON_AAA`, or `LIBRPA_ANACON_RIDGE`. Fortran exports the same constants.
Initialize options with `librpa_init_options()` (C), `Options` (C++), or
`opts%init()` (Fortran). The options structure grows: rebuild host programs and
bindings together with the library. An executable built against the previous
options layout is not binary-compatible with the new shared library; initializing
options does not repair a layout mismatch. Preserved old executables retain their
old defaults only when used with their matching old library.

For the original default method:

```ini
anacon_method = pade
```

For AAA:

```ini
anacon_method = aaa
anacon_aaa_tolerance = 1e-13
```

For an explicit ridge input:

```ini
anacon_method = ridge
anacon_ridge_lambda = 1e-12
anacon_ridge_den_weight = 10
```

When comparing methods, hold the imaginary-frequency input, basis, energy
reference, frequency grid, and QSGW update settings fixed. Resampling uses the
selected method in both fitting stages.

The standalone QSGW task keeps its existing static-update behavior: it fits
correlation matrices directly on the source frequency grid. `anacon_nfreq`
and `anacon_tfgrids_type` resample its entering-state GW quasiparticle
diagnostics, not its static Hamiltonian update. The low-level matrix builder
supports explicitly supplied resampling nodes; that capability does not imply
that the standalone task supplies them.
