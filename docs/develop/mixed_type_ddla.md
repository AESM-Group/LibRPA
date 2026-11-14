# Experimental mixed-domain DDLA products

The standalone driver accepts `use_mixed_type_ddla = true` (default `false`)
with a `LIBRPA_USE_DDLA=ON` build and `linear_algorithm = ddla`.
This is an internal C++ `DevOptions` prototype, not a public C/Fortran option.
It does not change the public options ABI.
On a one-rank owning grid, the mixed adapter calls local DDLA GEMM directly;
larger grids use distributed mixed PGEMM with the original descriptors.

The operation remains FP64: `double` multiplied by `complex<double>` produces
`complex<double>`. It does not enable FP32, TF32 or reduced-precision solves.

The first implementation covers:

- Ordinary Gamma Wc: `S chi0`, `-(S chi0) S`, and `Sc (X-Sc)`, where
  `S=sqrt(V)`, `Sc=sqrt(Vc)`, and the unchanged complex solve computes
  `epsilon X=Sc`. Gamma's real Coulomb-power producer establishes realness;
  no imaginary response component is discarded. The no-longer-used local
  eigenvector allocation stores both real factors. On GPU, its existing
  Coulomb buffer is reused for real products and complex solve/subtraction.
- The complex Wc frequency-to-time cosine transform: real coefficients replace
  complex copies and a local DDLA CPU GEMM acts on the host-resident batches.
  One CPU handle on `MPI_COMM_SELF` is reused across batches. No transform
  batch is uploaded to the GPU.

Non-Gamma Wc products, Gamma head/wing projections (including strict 2D),
complex LU/Cholesky solves, LibRI contractions and pair-density vertices retain
their existing implementation. The separate real-storage Wc transform already
uses real GEMM and is unchanged. The flag does not accelerate every DDLA call.

Execution logs identify eligible mixed paths. Profiler entries
`epsilon_pack_mixed_factors`, `epsilon_compute_eps_pgemm_1/2`,
`epsilon_multiply_coulwc_2` and `wc_cosine_gemm` expose factor packing and
products; use inclusive Wc/transform timings to account for setup and transfers.
Compare OFF/ON on identical resources with unchanged numerical references and
record peak memory. Fewer arithmetic operations alone do not establish a
speedup, particularly for small matrices or short cosine transforms.
