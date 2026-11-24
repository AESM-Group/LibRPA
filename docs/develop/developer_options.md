# Developer options

These internal options are read from the standalone driver's input and stored
in the C++ `DevOptions` structure (`src/utils/dev_options.h`). They are intended
for development and prototype comparisons rather than the public C/Fortran
options interface.

## `use_mixed_type_ddla`

- Default: `false`.
- Prerequisites: a `LIBRPA_USE_DDLA=ON` build and `linear_algorithm = ddla`.
- Enable with `use_mixed_type_ddla = true` in the driver input.

The option selects eligible products with real and complex operands. Both
retain FP64 precision: `double` and `complex<double>`, with complex results.
It does not select a lower-precision linear solve.

The affected products are:

- Ordinary Gamma Wc: `S chi0`, `-(S chi0) S`, and `Sc (X-Sc)`, where
  `S=sqrt(V)`, `Sc=sqrt(Vc)` and the complex solve gives `epsilon X=Sc`.
- The complex Wc frequency-to-time cosine transform: real cosine coefficients
  multiply complex Wc batches through a local DDLA CPU GEMM.
- Eligible cRPA pair-density contractions: real coefficients are retained for
  the mixed CPU-handle path, with the conventional contraction used when those
  CPU resources are unavailable.

The DDLA handle and owning descriptors select the execution backend. Mixed
cosine transforms stay on the CPU; Gamma head/wing corrections and linear
solves retain complex FP64. LibDDLA manages the mixed-GEMM scratch internally.

Compare OFF/ON using identical inputs, resources and numerical references.
Use inclusive Wc/transform timings to account for setup and transfers;
`epsilon_compute_eps_pgemm_1/2`, `epsilon_multiply_coulwc_2` and
`wc_cosine_gemm` identify the eligible products in profiler output.
