# Third-party dependencies

## LibRI

The bundled base is upstream `a0661cbb4b4f89ceeb23e040d49accb00943093f`.
The same headers support CPU and GPU tensor contractions. Enable GPU contractions
with `LIBRPA_USE_LIBRI_GPU=ON` and either `LIBRPA_USE_CUDA=ON` or
`LIBRPA_USE_HIP=ON`; both GPU backends use LibDDLA.

The following local patch is applied:

- Portable allocator cleanup in the k-space Hartree and CVC kernels.
  `RI/global/Memory.h` provides `RI::Global_Func::release_free_memory()`: it
  calls `malloc_trim(0)` on glibc and `malloc_zone_pressure_relief` on macOS.
  Other platforms perform no explicit cleanup.
