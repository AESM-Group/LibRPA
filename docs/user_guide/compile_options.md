# Compile Options

## Overview

| Option                                                      | Type              | Default    |
|-------------------------------------------------------------|-------------------|------------|
| [`LIBRPA_USE_LIBRI`](#librpa-use-libri)                     | bool              | `ON`       |
| [`LIBRPA_USE_LIBRI_GPU`](#librpa-use-libri-gpu)             | bool              | `OFF`      |
| [`LIBRPA_USE_CUDA`](#librpa-use-cuda)                       | bool              | `OFF`      |
| [`LIBRPA_USE_HIP`](#librpa-use-hip)                         | bool              | `OFF`      |
| [`LIBRPA_USE_CMAKE_INC`](#librpa-use-cmake-inc)             | bool              | `OFF`      |
| [`LIBRPA_USE_EXTERNAL_GREENX`](#librpa-use-external-greenx) | bool              | `OFF`      |
| [`LIBRPA_ENABLE_FORTRAN_BIND`](#librpa-enable-fortran-bind) | bool              | `OFF`      |
| [`LIBRPA_FORTRAN_DP`](#librpa-fortran-dp)                   | string or integer | `c_double` |
| [`LIBRPA_ENABLE_DRIVER`](#librpa-enable-driver)             | bool              | `ON`       |
| [`LIBRPA_MPI_THREAD_LEVEL`](#librpa-mpi-thread-level)       | string            | automatic  |
| [`LIBRPA_VERBOSE_OUTPUT`](#librpa-verbose-output)           | bool              | `ON`       |
| [`LIBRPA_ENABLE_TEST`](#librpa-enable-test)                 | bool              | `ON`       |
| [`LIBRPA_ENABLE_CPP_TEST`](#librpa-enable-cpp-test)         | bool              | `ON`       |
| [`LIBRPA_ENABLE_FORTRAN_TEST`](#librpa-enable-fortran-test) | bool              | `ON`       |
| [`LIBRI_INCLUDE_DIR`](#libri-include-dir)                   | string            | empty      |
| [`LIBCOMM_INCLUDE_DIR`](#libcomm-include-dir)               | string            | empty      |
| [`CEREAL_INCLUDE_DIR`](#cereal-include-dir)                 | string            | empty      |
| [`LIBDDLA_PATH`](#libddla-path)                             | string            | empty      |
| [`SCALAPACK_DIR`](#scalapack-dir)                           | string            | empty      |
| [`LIBRPA_USE_EXTERNAL_ELPA`](#librpa-use-external-elpa)     | bool              | `OFF`      |
| [`EXTERNAL_ELPA_DIR`](#external-elpa-dir)                   | string            | empty      |
| [`LIBRPA_USE_BUNDLED_ELPA`](#librpa-use-bundled-elpa)       | bool              | `OFF`      |
| [`LIBRPA_BUNDLED_ELPA_VERSION`](#librpa-bundled-elpa-version) | string          | `2026.02.001` |
| [`LIBRPA_BUNDLED_ELPA_KERNEL`](#librpa-bundled-elpa-kernel) | string            | empty      |
| [`LIBRPA_BUNDLED_ELPA_OPENMP`](#librpa-bundled-elpa-openmp) | bool              | `OFF`      |
| [`LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS`](#librpa-bundled-elpa-configure-args) | string | empty |
| [`LIBRPA_BUNDLED_ELPA_LIBS`](#librpa-bundled-elpa-libs)     | string            | empty      |

These options can be parsed on the CMake command line, for example:

```sh
cmake -DLIBRPA_USE_LIBRI=ON
```

(librpa-use-libri)=
## `LIBRPA_USE_LIBRI`

When enabled, LibRPA is compiled with [LibRI](https://github.com/abacusmodeling/LibRI)
for RI tensor contractions.

The *GW* and EXX functionalities require LibRPA to be compiled with LibRI, i.e. `-DLIBRPA_USE_LIBRI=ON`.
By contrast, the RPA correlation energy can also be computed without this option.
The analytic optical wing also requires LibRI tensor communication; the analytic
head and RPA calculations without an analytic wing remain available without LibRI.

(librpa-use-libri-gpu)=
## `LIBRPA_USE_LIBRI_GPU`

Enables GPU-accelerated LibRI tensor contractions. This option is meaningful
only when [`LIBRPA_USE_LIBRI`](#librpa-use-libri) and exactly one of
[`LIBRPA_USE_CUDA`](#librpa-use-cuda) or
[`LIBRPA_USE_HIP`](#librpa-use-hip) are enabled.

CPU and GPU builds use the same bundled headers under `thirdparty/LibRI`.
With this option enabled, LibRPA defines `__GPU_RI` and `__DDLA_RI` and selects
LibRI's GPU calculation mode for RPA, EXX, and GW tensor contractions. LibDDLA
uses the selected CUDA or HIP backend. With this option disabled, LibRI tensor
contractions remain on the CPU even if CUDA or HIP is enabled for other operations.

When an external `LIBRI_INCLUDE_DIR` is supplied, its headers must support
`__GPU_RI`, the DDLA backend, and `RI::LRI_Cal_Mode::GPU`.

Example:
```sh
cmake -B build \
      -DLIBRPA_USE_CUDA=ON \
      -DLIBRPA_USE_LIBRI_GPU=ON
```

(librpa-use-cuda)=
## `LIBRPA_USE_CUDA`

Enables the NVIDIA CUDA backend. CUDA and HIP backends are mutually exclusive.
The CUDA toolkit, NCCL, and LibDDLA must be discoverable by the build.

If `CMAKE_CUDA_ARCHITECTURES` is not specified, LibRPA uses `70;75;80`.
Set this standard CMake variable explicitly when building for other GPU
architectures. An external LibDDLA installation can be selected with
[`LIBDDLA_PATH`](#libddla-path); otherwise the bundled LibDDLA is built.

Example:
```sh
cmake -B build \
      -DLIBRPA_USE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80
```

(librpa-use-hip)=
## `LIBRPA_USE_HIP`

Enables the AMD HIP/ROCm backend. HIP and CUDA backends are mutually exclusive.
The HIP/ROCm libraries, RCCL, and LibDDLA must be discoverable by the build.

Set `CMAKE_HIP_ARCHITECTURES` for the target GPU when necessary. `ROCM_PATH`
can be used when the ROCm installation cannot be inferred from the
environment. An external LibDDLA installation can be selected with
[`LIBDDLA_PATH`](#libddla-path); otherwise the bundled LibDDLA is built.

Example:
```sh
cmake -B build -DLIBRPA_USE_HIP=ON
```

(librpa-use-external-elpa)=
## `LIBRPA_USE_EXTERNAL_ELPA`

When enabled, LibRPA is linked against an external
[ELPA](https://elpa.mpcdf.mpg.de/) installation.

ELPA support is intended for optimized linear algebra subroutines, such as
ELPA-provided dense eigensolver routines, in ELPA-backed implementations. This
option provides the build interface for those code paths.

Set [`EXTERNAL_ELPA_DIR`](#external-elpa-dir) to the ELPA installation prefix
so CMake can find the ELPA headers, Fortran module directory, and library.

Example:
```sh
cmake -DLIBRPA_USE_EXTERNAL_ELPA=ON -DEXTERNAL_ELPA_DIR=/path/to/elpa
```

(librpa-use-bundled-elpa)=
## `LIBRPA_USE_BUNDLED_ELPA`

When enabled, LibRPA builds and links against a bundled ELPA source release
under `thirdparty/ELPA`.

This option is mutually exclusive with
[`LIBRPA_USE_EXTERNAL_ELPA`](#librpa-use-external-elpa).

The bundled ELPA build is managed through CMake's `ExternalProject` mechanism.
After ELPA has been built in an existing build directory, changing compiler
flags or `CMAKE_BUILD_TYPE` may not automatically reconfigure and rebuild ELPA.
Use a fresh build directory, or clean the bundled ELPA sub-build, when those
settings need to be applied to ELPA itself.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON
```

(librpa-bundled-elpa-version)=
## `LIBRPA_BUNDLED_ELPA_VERSION`

Selects which bundled ELPA release is built when
[`LIBRPA_USE_BUNDLED_ELPA`](#librpa-use-bundled-elpa) is enabled.

Supported values are:

- `2026.02.001`

(librpa-bundled-elpa-kernel)=
## `LIBRPA_BUNDLED_ELPA_KERNEL`

Selects an x86 SIMD kernel family for the bundled ELPA build.

By default, this option is empty. In that case, LibRPA disables ELPA's
x86-specific SIMD kernels and lets ELPA build portable generic kernels. Set this
option on compatible x86 systems when an optimized kernel family is desired.

Supported values are:

- empty
- `SSE`
- `SSE_ASSEMBLY`
- `AVX`
- `AVX2`
- `AVX512`

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_KERNEL=AVX512
```

(librpa-bundled-elpa-openmp)=
## `LIBRPA_BUNDLED_ELPA_OPENMP`

Controls whether the bundled ELPA library is built with ELPA's own OpenMP
support.

The default is `OFF`. In that case, ELPA is built as an MPI-only static library,
even if LibRPA itself is compiled with OpenMP. Set this option to `ON` only when
the runtime process and thread layout is chosen with ELPA threading in mind.
For example, with MPI ranks, OpenMP regions in LibRPA, threaded BLAS, and
OpenMP-enabled ELPA all active at the same time, the total number of runnable
threads can exceed the available cores unless `OMP_NUM_THREADS`,
BLAS-specific thread controls, and the MPI rank count are coordinated.

When this option is `ON`, LibRPA also enables ELPA's runtime MPI threading
support checks and allows ELPA to limit its OpenMP thread count when the MPI
library does not provide the thread level ELPA needs.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_OPENMP=ON
```

(librpa-bundled-elpa-configure-args)=
## `LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS`

Additional arguments passed to the bundled ELPA `configure` script.

Arguments passed through this option are appended after LibRPA's defaults,
including [`LIBRPA_BUNDLED_ELPA_KERNEL`](#librpa-bundled-elpa-kernel), so they
can override the default kernel selection when a specific ELPA setup is needed.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS="--enable-store-build-config"
```

(librpa-bundled-elpa-libs)=
## `LIBRPA_BUNDLED_ELPA_LIBS`

Linker flags passed to the bundled ELPA `configure` script through its `LIBS`
environment variable.

By default, LibRPA forwards the detected LAPACK and ScaLAPACK libraries to the
bundled ELPA build. Set this option only when the autodetected flags are not
suitable for a particular compiler or math library setup.

When static math libraries are used, ELPA's libtool build may try to include
those static archives inside `libelpa.a`. LibRPA removes such nested archive
members after the bundled ELPA install step and links the math libraries
separately through CMake.

Example:
```sh
cmake -DLIBRPA_USE_BUNDLED_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_LIBS="-L/path/to/lib -lscalapack -llapack -lblas"
```

(librpa-use-cmake-inc)=
## `LIBRPA_USE_CMAKE_INC`

When enabled, the `cmake.inc` file is used to initialize compilers and other build options.

**Deprecated**. It is recommended to use standard CMake command-line options such as `-C` or `-D` to specify custom variables.

(librpa-use-external-greenx)=
## `LIBRPA_USE_EXTERNAL_GREENX`

Controls whether LibRPA uses the bundled GreenX library or an external one.

The minimax grids used by LibRPA are provided through the
[GreenX](https://nomad-coe.github.io/greenX/) library.

When this option is `OFF` (default), LibRPA builds and links against the bundled GreenX source distributed with LibRPA under `thirdparty/greenX`.

When this option is `ON`, LibRPA does not build the bundled GreenX copy.
Instead, it expects an external GreenX library to be provided by the parent or higher-level CMake project.
In particular, the CMake target `LibGXMiniMax` must already be defined and available for linking.

This option is mainly intended for developer workflows or project setups in which GreenX is managed outside LibRPA.

(librpa-enable-fortran-bind)=
## `LIBRPA_ENABLE_FORTRAN_BIND`

When enabled, the Fortran bindings of LibRPA are built.

(librpa-fortran-dp)=
## `LIBRPA_FORTRAN_DP`

Specifies the Fortran kind used for double-precision real and complex data in the Fortran bindings.

The default value is `c_double`, which is suitable when interoperability with C is desired.
This option may also be set to an integer kind value if needed by the calling code.

This option is meaningful only if `LIBRPA_ENABLE_FORTRAN_BIND=ON`.

(librpa-enable-driver)=
## `LIBRPA_ENABLE_DRIVER`

When enabled, the LibRPA driver executable is built.

(librpa-mpi-thread-level)=
## `LIBRPA_MPI_THREAD_LEVEL`

Selects the MPI thread-support level requested by the LibRPA driver and C++
tests. Supported values are `MPI_THREAD_SINGLE`, `MPI_THREAD_FUNNELED`,
`MPI_THREAD_SERIALIZED`, and `MPI_THREAD_MULTIPLE`.

When left empty, LibRPA selects the value from the enabled components: CPU
ELPA builds use `MPI_THREAD_MULTIPLE`, builds with bundled LibComm use
`MPI_THREAD_FUNNELED`, and other builds use `MPI_THREAD_MULTIPLE`. An explicit
value takes precedence over this automatic selection:

```sh
cmake -B build -DLIBRPA_MPI_THREAD_LEVEL=MPI_THREAD_FUNNELED
```

The override must remain compatible with the selected dependencies and their
threading requirements.

(librpa-verbose-output)=
## `LIBRPA_VERBOSE_OUTPUT`

Compiles additional per-timer timestamp and memory diagnostics into LibRPA.
The runtime output level still controls which diagnostic messages are emitted.
Disable this option when those detailed diagnostics are not needed.

(librpa-enable-test)=
## `LIBRPA_ENABLE_TEST`

When enabled, the unit tests of LibRPA are built.

After LibRPA has been compiled successfully, the tests can be run from the build directory with:
```sh
ctest
```
or equivalently
```sh
make test
```

```{note}
At present, the unit tests do not cover the entire code base.
Test coverage is still being expanded.
```

(librpa-enable-cpp-test)=
## `LIBRPA_ENABLE_CPP_TEST`

When enabled, the C++ unit tests are built.

This option is meaningful only if `LIBRPA_ENABLE_TEST=ON`.

(librpa-enable-fortran-test)=
## `LIBRPA_ENABLE_FORTRAN_TEST`

When enabled, the Fortran unit tests are built.

This option is meaningful only if both `LIBRPA_ENABLE_TEST=ON` and `LIBRPA_ENABLE_FORTRAN_BIND=ON`.

(libri-include-dir)=
## `LIBRI_INCLUDE_DIR`

Specifies the path to the LibRI include directory.

If this variable is empty, the internal LibRI copy is used.
Otherwise, CMake searches for `RI/ri/RI_Tools.h` under the specified directory.
An error is raised if the file cannot be found.

Example:
```sh
cmake -DLIBRI_INCLUDE_DIR=/path/to/LibRI/include
```

(libcomm-include-dir)=
## `LIBCOMM_INCLUDE_DIR`

Specifies the path to the LibComm include directory.

If this variable is empty, the internal LibComm copy is used.
Otherwise, CMake searches for `Comm/Comm_Tools.h` under the specified directory.
An error is raised if the file cannot be found.

Example:
```sh
cmake -DLIBCOMM_INCLUDE_DIR=/path/to/LibComm/include
```

(cereal-include-dir)=
## `CEREAL_INCLUDE_DIR`

Specifies the path to the cereal include directory.

If this variable is empty, the bundled cereal copy is used.
Otherwise, CMake searches for `cereal/cereal.hpp` under the specified directory.
An error is raised if the file cannot be found.

Example:
```sh
cmake -DCEREAL_INCLUDE_DIR=/path/to/cereal/include
```

(libddla-path)=
## `LIBDDLA_PATH`

Specifies the installation prefix of an external LibDDLA library for CUDA or
HIP builds. It can be set as either a CMake variable or an environment
variable.

CMake expects `include/ddla/ddla.h` and a `libddla` shared library under
`lib/` or `lib64/`. If this variable is empty, LibRPA builds the bundled
LibDDLA source.

Example:
```sh
cmake -B build \
      -DLIBRPA_USE_CUDA=ON \
      -DLIBDDLA_PATH=/path/to/libddla
```

(scalapack-dir)=
## `SCALAPACK_DIR`

`SCALAPACK_DIR` specifies the installation path of ScaLAPACK and is used to
locate the ScaLAPACK libraries when `MKLROOT` is not defined.

This variable can be provided in two ways:

- as a CMake option:

  ```bash
  cmake -DSCALAPACK_DIR=/path/to/scalapack
  ```

- or as an environment variable:

  ```bash
  export SCALAPACK_DIR=/path/to/scalapack
  cmake
  ```

This option is intended for environments where ScaLAPACK is provided as a
standalone installation rather than through Intel MKL.

(external-elpa-dir)=
## `EXTERNAL_ELPA_DIR`

`EXTERNAL_ELPA_DIR` specifies the installation prefix of an external ELPA
library. It is used when
[`LIBRPA_USE_EXTERNAL_ELPA`](#librpa-use-external-elpa) is enabled.

CMake searches below this prefix for:

- headers such as `include/elpa-*/elpa/elpa.h`
- Fortran modules such as `include/elpa-*/modules/elpa.mod`
- libraries such as `lib/libelpa.so` or `lib/libelpa_openmp.so`

This variable can be provided as a CMake option:

```bash
cmake -DLIBRPA_USE_EXTERNAL_ELPA=ON -DEXTERNAL_ELPA_DIR=/path/to/elpa
```

or as an environment variable:

```bash
export EXTERNAL_ELPA_DIR=/path/to/elpa
cmake -DLIBRPA_USE_EXTERNAL_ELPA=ON
```
