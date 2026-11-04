# Compile Options

(librpa-how-to-set-options)=
## How to set options

Run the examples below from the LibRPA source directory. Use `-DNAME=value` with an explicit build directory:

```sh
cmake -B build -DLIBRPA_USE_LIBRI=ON
```

CMake cache values persist in that build directory. Use a separate directory when changing compilers or MPI installations. To remove a path override, remove its cache entry and unset its environment variable. When a dependency path cache entry is empty, LibRPA imports a nonempty environment value into the cache and reports its origin. The saved path remains active on later configurations without that environment variable. A nonempty cache value takes precedence.

Header include paths, compiled-dependency source and installation paths, and `LIBRPA_THIRDPARTY_ARCHIVE_DIR` use `PATH` cache entries. On the first configuration, untyped relative `-DNAME=value` paths resolve against the directory where CMake is invoked. Supply absolute paths for environment values, typed `-DNAME:PATH=value` values and later cache updates. Any remaining relative value causes an error before use. Header checks and resolution use the same path.

`LIBRPA_THIRDPARTY_ARCHIVE_DIR` and `LIBRPA_THIRDPARTY_DOWNLOAD` also work with the standalone `cmake -P` scripts: `cmake/LibRPADownloadThirdparty.cmake` for archives and `cmake/LibRPAPrepareThirdparty.cmake` for prepared sources. The archive directory also accepts an environment variable. Separately, header include paths and the source and installation paths of compiled dependencies read environment variables of the same name.

For the build requirements and a complete build procedure, see {ref}`Installation <install-build-requirements>`.

## Compiler and build-tool settings

Use these standard CMake settings before the first configuration of a build directory:

| Setting | Purpose |
| --- | --- |
| `CMAKE_CXX_COMPILER` | C++ compiler or MPI C++ wrapper |
| `CMAKE_Fortran_COMPILER` | Fortran compiler or MPI Fortran wrapper, required even without the binding |
| `CMAKE_C_COMPILER` | C compiler or MPI C wrapper for managed ELPA |
| `CMAKE_CXX_FLAGS`, `CMAKE_Fortran_FLAGS`, `CMAKE_C_FLAGS` | Flags for the corresponding language |
| `CMAKE_PREFIX_PATH` | Additional package search prefixes; this does not select header-only dependencies |

LibRPA sets the C++ language standard to C++17. Keep compiler, MPI and math-library choices compatible. Use a new build directory to change compilers. For managed LibDDLA, see {ref}`the list of forwarded child-build settings <libddla-source-dir>`.

Use `cmake --build build --parallel N` to set build concurrency. This is a build command option, not a LibRPA configure option.

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
| [`LIBRI_INCLUDE_DIR`](#libri-include-dir)                   | path              | empty      |
| [`LIBCOMM_INCLUDE_DIR`](#libcomm-include-dir)               | path              | empty      |
| [`CEREAL_INCLUDE_DIR`](#cereal-include-dir)                 | path              | empty      |
| [`LIBDDLA_PATH`](#libddla-path)                             | path              | empty      |
| [`SCALAPACK_DIR`](#scalapack-dir)                           | string            | empty      |
| [`LIBRPA_USE_ELPA`](#librpa-use-elpa) | bool | `OFF` |
| [`LIBRPA_USE_EXTERNAL_ELPA`](#librpa-use-external-elpa) | bool (deprecated) | unset |
| [`EXTERNAL_ELPA_DIR`](#external-elpa-dir)                   | path              | empty      |
| [`LIBRPA_USE_BUNDLED_ELPA`](#librpa-use-bundled-elpa) | bool (deprecated) | unset |
| [`LIBRPA_BUNDLED_ELPA_VERSION`](#librpa-bundled-elpa-version) | string          | `2026.02.001` |
| [`LIBRPA_BUNDLED_ELPA_KERNEL`](#librpa-bundled-elpa-kernel) | string            | empty      |
| [`LIBRPA_BUNDLED_ELPA_OPENMP`](#librpa-bundled-elpa-openmp) | bool              | `OFF`      |
| [`LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS`](#librpa-bundled-elpa-configure-args) | string | empty |
| [`LIBRPA_BUNDLED_ELPA_LIBS`](#librpa-bundled-elpa-libs)     | string            | empty      |
| [`LIBRPA_THIRDPARTY_DOWNLOAD`](#librpa-thirdparty-download) | bool | `ON` |
| [`LIBRPA_THIRDPARTY_ARCHIVE_DIR`](#librpa-thirdparty-archive-dir) | path | empty |
| [`GREENX_SOURCE_DIR`](#greenx-source-dir) | path | empty |
| [`EXTERNAL_GREENX_DIR`](#external-greenx-dir) | path | empty |
| [`ELPA_SOURCE_DIR`](#elpa-source-dir) | path | empty |
| [`LIBDDLA_SOURCE_DIR`](#libddla-source-dir) | path | empty |
| [`BUILD_SHARED_LIBS`](#build-shared-libs) | bool | `OFF` in a standalone build |
| [`CMAKE_BUILD_TYPE`](#cmake-build-type) | string | empty |
| [`CMAKE_INSTALL_PREFIX`](#cmake-install-prefix) | path | source `install/` (standalone) |
| [`MKLROOT`](#mklroot) | path | environment value, if defined |
| [`LAPACK_LIBRARIES`](#lapack-libraries) | library list | detected |
| [`BLAS_LIBRARIES`](#blas-libraries) | library list | detected |

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

CPU and GPU builds use the same resolved LibRI headers.
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
[`LIBDDLA_PATH`](#libddla-path); otherwise `LIBDDLA_SOURCE_DIR` or managed LibDDLA sources are built.

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
[`LIBDDLA_PATH`](#libddla-path); otherwise `LIBDDLA_SOURCE_DIR` or managed LibDDLA sources are built.

Example:
```sh
cmake -B build -DLIBRPA_USE_HIP=ON
```

(librpa-use-elpa)=
## `LIBRPA_USE_ELPA`

Enable ELPA for distributed eigensolvers. The default is `OFF`. With `ON` and no supplied dependency, LibRPA downloads and verifies the pinned release, extracts and patches it under `thirdparty/src`, and builds it with LibRPA. Existing managed sources or cached archives can also be used with downloads disabled.

An existing `ELPA::ELPA` target, `ELPA_SOURCE_DIR`, or an installation prefix (`EXTERNAL_ELPA_DIR`, with `ELPA_ROOT` as a fallback) automatically sets this option to `ON`, including when it was set to `OFF`. Parent targets take precedence over paths. Choose either a source path or an installation prefix.

```sh
# Build the managed release
cmake -B build -DLIBRPA_USE_ELPA=ON

# Build a supplied source tree; ELPA is enabled automatically
cmake -B build -DELPA_SOURCE_DIR=/path/to/elpa-source

# Use an installed library; ELPA is enabled automatically
cmake -B build -DEXTERNAL_ELPA_DIR=/path/to/elpa
```

Source builds use upstream autotools and a private installation inside the build directory. Choose source and build paths without whitespace. Use a fresh build directory after changes to compiler flags, `CMAKE_BUILD_TYPE` or ELPA sources so the separate ELPA build applies them.

(librpa-use-external-elpa)=
## `LIBRPA_USE_EXTERNAL_ELPA` (deprecated)

Use `LIBRPA_USE_ELPA` and the dependency settings above for new configurations. Supplying this legacy option, with either `ON` or `OFF`, emits a CMake deprecation warning. With `ON`, it enables ELPA. If no target or path is supplied, it retains the legacy search for an installed library in standard search paths. An explicit source path selects a source build.

To migrate an installed-library configuration, set `EXTERNAL_ELPA_DIR` to its installation prefix and remove the old cache entry with `-ULIBRPA_USE_EXTERNAL_ELPA`.

(librpa-use-bundled-elpa)=
## `LIBRPA_USE_BUNDLED_ELPA` (deprecated)

Use `LIBRPA_USE_ELPA=ON` to build the managed release, or supply `ELPA_SOURCE_DIR` to build a source tree. Supplying this legacy option, with either `ON` or `OFF`, emits a CMake deprecation warning. With `ON`, it enables ELPA and uses the target and path selection described above. With `OFF`, neither legacy option disables an otherwise enabled ELPA configuration. When both legacy options are `ON` without a target or path, the installed-library search takes precedence.

Remove the old cache entry with `-ULIBRPA_USE_BUNDLED_ELPA` when migrating. Legacy options retained in the cache continue to emit warnings on later configurations.

(librpa-bundled-elpa-version)=
## `LIBRPA_BUNDLED_ELPA_VERSION`

Selects which managed ELPA release is built when
[`LIBRPA_USE_ELPA`](#librpa-use-elpa) selects a source build.

Supported value: `2026.02.001`, matching the manifest. Other values are rejected. This option does not select an arbitrary upstream release.

(librpa-bundled-elpa-kernel)=
## `LIBRPA_BUNDLED_ELPA_KERNEL`

Selects an x86 SIMD kernel family for the managed ELPA build.

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
cmake -B build -DLIBRPA_USE_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_KERNEL=AVX512
```

(librpa-bundled-elpa-openmp)=
## `LIBRPA_BUNDLED_ELPA_OPENMP`

Controls whether the managed ELPA library is built with ELPA's own OpenMP
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
cmake -B build -DLIBRPA_USE_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_OPENMP=ON
```

(librpa-bundled-elpa-configure-args)=
## `LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS`

Additional arguments passed to the managed ELPA `configure` script. Supply one quoted string with space-separated arguments. CMake parses this string with `separate_arguments(... UNIX_COMMAND ...)`.

Arguments passed through this option are appended after LibRPA's defaults,
including [`LIBRPA_BUNDLED_ELPA_KERNEL`](#librpa-bundled-elpa-kernel), so they
can override the default kernel selection when a specific ELPA setup is needed.

Example:
```sh
cmake -B build -DLIBRPA_USE_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS="--enable-store-build-config"
```

(librpa-bundled-elpa-libs)=
## `LIBRPA_BUNDLED_ELPA_LIBS`

Linker flags passed to the managed ELPA `configure` script through its `LIBS`
environment variable.

By default, LibRPA forwards the detected LAPACK and ScaLAPACK libraries to the
managed ELPA build. Set this option only when the autodetected flags are not
suitable for a particular compiler or math library setup.

When static math libraries are used, ELPA's libtool build may try to include
those static archives inside `libelpa.a`. LibRPA removes such nested archive
members after the managed ELPA install step and links the math libraries
separately through CMake.

Example:
```sh
cmake -B build -DLIBRPA_USE_ELPA=ON \
      -DLIBRPA_BUNDLED_ELPA_LIBS="-L/path/to/lib -lscalapack -llapack -lblas"
```

(librpa-use-cmake-inc)=
## `LIBRPA_USE_CMAKE_INC`

When enabled, the `cmake.inc` file is used to initialize compilers and other build options.

**Deprecated**. It is recommended to use standard CMake command-line options such as `-C` or `-D` to specify custom variables.

(librpa-use-external-greenx)=
## `LIBRPA_USE_EXTERNAL_GREENX`

Require an external greenX source. Default: `OFF`.

LibRPA first uses an existing `LibGXMiniMax` or `greenX::LibGXMiniMax` target, regardless of this option. Otherwise, it uses `GREENX_SOURCE_DIR`, then `EXTERNAL_GREENX_DIR`. Supplying both paths is an error.

With `ON`, configuration fails if none of these external sources is supplied. With `OFF`, LibRPA falls back to the managed source. GreenX is required even with `LIBRPA_USE_LIBRI=OFF`.

A source-built target must provide its link dependencies and `Fortran_MODULE_DIRECTORY`. Imported targets must expose compatible module directories through their link interface. See {ref}`Parent-project integration <thirdparty-parent-projects>`.

(greenx-source-dir)=
## `GREENX_SOURCE_DIR`

External greenX source directory containing `GX-TimeFrequency/CMakeLists.txt`. A nonempty CMake value takes precedence over the environment variable. A parent target takes precedence over this path.

LibRPA hashes the whole tree. Matching patched sources are `pinned`. Matching pristine sources become a patched copy in managed storage (`overlay`). Other sources produce an `unpinned` warning and are used directly. The supplied tree remains unchanged. LibRPA builds the selected tree with its own compilers.

`GREENX_SOURCE_DIR` and `EXTERNAL_GREENX_DIR` are mutually exclusive. An invalid explicit path stops configuration without a download fallback.

(external-greenx-dir)=
## `EXTERNAL_GREENX_DIR`

Prebuilt greenX installation prefix. A nonempty CMake value takes precedence over the environment variable. A parent target takes precedence over installation discovery.

LibRPA loads `greenXConfig.cmake` only from this prefix and uses `greenX::LibGXMiniMax`. Its link interface must provide `gx_minimax.mod` and `kinds.mod`. Missing modules stop configuration. If the module path identifies a different Fortran compiler or version, LibRPA warns.

Each configuration warns about compatibility with LibRPA's compilers and flags. Prefer the managed source (leave this variable empty), then a parent target, then `GREENX_SOURCE_DIR`. The installation must also provide compatible BLAS/LAPACK dependencies. Supply `BLAS_LIBRARIES` and `LAPACK_LIBRARIES` when necessary.

(elpa-source-dir)=
## `ELPA_SOURCE_DIR`

External ELPA source directory containing `configure` and `src/elpa_impl.F90`. A nonempty CMake value takes precedence over the environment variable.

Supplying this path automatically enables `LIBRPA_USE_ELPA`. Choose either this source path or an installation prefix (`EXTERNAL_ELPA_DIR` or `ELPA_ROOT`), including environment values. A parent `ELPA::ELPA` target takes precedence over the source path.

LibRPA hashes the whole tree. Matching patched sources are `pinned`. Matching pristine sources become a patched copy in managed storage (`overlay`). Other sources produce an `unpinned` warning. The out-of-source build preserves the user tree and overlay. An invalid path stops configuration without a download fallback.

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

MPI thread-support level requested by the driver and C++ tests. Accepted values are `MPI_THREAD_SINGLE`, `MPI_THREAD_FUNNELED`, `MPI_THREAD_SERIALIZED` and `MPI_THREAD_MULTIPLE`.

The default cache value is empty. In that case, LibRPA selects the level in this order:

| Condition | Selected level |
| --- | --- |
| `LIBRPA_USE_ELPA=ON`, with neither CUDA nor HIP | `MPI_THREAD_MULTIPLE` |
| Otherwise, LibComm status is `managed` or `pinned` | `MPI_THREAD_FUNNELED` |
| Otherwise | `MPI_THREAD_MULTIPLE` |

Modified managed LibComm headers have status `unpinned` and select `MPI_THREAD_MULTIPLE` automatically. A parent LibComm target has status `target`. LibRPA does not infer its pinned status from the target name or include path. An explicit value takes precedence over the automatic selection.

```sh
cmake -B build -DLIBRPA_MPI_THREAD_LEVEL=MPI_THREAD_FUNNELED
```

The chosen level must satisfy the dependencies' threading requirements.

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

Include directory containing `RI/ri/LRI.h`. Used only with `LIBRPA_USE_LIBRI=ON`.

An existing `LibRI::LibRI` target takes precedence. Otherwise, a nonempty CMake value takes precedence over the environment variable. An invalid explicit directory is an error. LibRPA does not search alternative system paths or download a replacement.

For an explicit path, LibRPA hashes the include tree on every configuration:

| Match | Action and status |
| --- | --- |
| Pinned patched headers | Use the directory directly (`pinned`) |
| Pinned pristine headers | Copy into managed storage, apply the LibRI patch and compare with the reference (`overlay` if unchanged) |
| Any other headers | Use directly with a warning (`unpinned`) |

An overlay never modifies the supplied directory. Local edits to an existing overlay remain in use with an `unpinned` warning. Creating the patched copy requires a writable `thirdparty/src/` directory in the LibRPA checkout. It can be created with downloads disabled.

If no target or path is supplied, LibRPA prepares managed headers. A required compile probe instantiates `RI::LRI<int,int,3,double>` with the selected LibComm and cereal headers. Failure stops configuration and reports all three sources. The separate fine-grained-lock probe remains optional. Hartree support depends on `RI/physics/Hartree.h` in the resolved include directories.

```sh
cmake -B build -DLIBRI_INCLUDE_DIR=/path/to/LibRI/include
```

(libcomm-include-dir)=
## `LIBCOMM_INCLUDE_DIR`

Include directory containing `Comm/Comm_Tools.h`. Used only with `LIBRPA_USE_LIBRI=ON`.

An existing `LibComm::LibComm` target takes precedence. Otherwise, a nonempty CMake value takes precedence over the environment variable. Invalid explicit paths cause an error without a search or download fallback.

Headers matching the pinned upstream tree are used directly (`pinned`). LibComm has no managed patch, so it does not need an overlay. Other headers are used with an `unpinned` warning. The required LibRI compile probe checks the selected combination.

If no target or path is supplied, LibRPA prepares managed headers. Unmodified managed and pinned LibComm influence the automatic [MPI thread-level selection](#librpa-mpi-thread-level).

```sh
cmake -B build -DLIBCOMM_INCLUDE_DIR=/path/to/LibComm/include
```

(cereal-include-dir)=
## `CEREAL_INCLUDE_DIR`

Include directory containing `cereal/cereal.hpp`. Used only with `LIBRPA_USE_LIBRI=ON`.

An existing `cereal::cereal` target takes precedence. Otherwise, a nonempty CMake value takes precedence over the environment variable. Invalid explicit paths cause an error without a search or download fallback.

Headers matching the pinned upstream tree are used directly (`pinned`). Cereal has no managed patch, so it does not need an overlay. Other headers are used with an `unpinned` warning. The required LibRI compile probe checks the selected combination.

If no target or path is supplied, LibRPA prepares managed headers. The manifest pins cereal to `d81e2f7df7b334fee057e53017388d02e555a836`, which includes the tuple-serialization fix used by this LibRI revision.

```sh
cmake -B build -DCEREAL_INCLUDE_DIR=/path/to/cereal/include
```

(libddla-path)=
## `LIBDDLA_PATH`

Installation prefix of LibDDLA. Used only with a CUDA or HIP backend. A nonempty CMake value takes precedence over the environment variable.

The resolution order is:

1. Existing `LibDDLA::ddla` target.
2. `LIBDDLA_SOURCE_DIR` source tree or `LIBDDLA_PATH` installation. Supplying both is an error.
3. Managed source from the manifest.

`LIBDDLA_PATH` and `LIBDDLA_SOURCE_DIR` are mutually exclusive. Each installation-mode configuration warns about compiler and flag compatibility. Prefer the managed source (leave both variables empty), then a parent target, then a source build.

The installation must contain `include/ddla/ddla.h` and a `ddla` library under `lib/` or `lib64/`. Missing headers or a missing library cause an error. Installed libraries are not checked against the manifest's source-tree hash. The installation must match the selected backend, compiler and MPI environment.

```sh
cmake -B build -DLIBRPA_USE_CUDA=ON -DLIBDDLA_PATH=/path/to/libddla
```

(scalapack-dir)=
## `SCALAPACK_DIR`

`SCALAPACK_DIR` specifies the installation path of ScaLAPACK and is used to
locate the ScaLAPACK libraries when `MKLROOT` is empty. The finder checks
`lib/` and `lib64/` for `scalapack` or `scalapack-openmpi`. CMake and
environment values are search hints; standard search paths remain available.

This variable can be provided in two ways:

- as a CMake option:

  ```bash
  cmake -B build -DSCALAPACK_DIR=/path/to/scalapack
  ```

- or as an environment variable:

  ```bash
  export SCALAPACK_DIR=/path/to/scalapack
  cmake -B build
  ```

This option is intended for environments where ScaLAPACK is provided as a
standalone installation rather than through Intel MKL.

(external-elpa-dir)=
## `EXTERNAL_ELPA_DIR`

ELPA installation prefix. Supplying this path automatically enables `LIBRPA_USE_ELPA`. A nonempty CMake value takes precedence over the same-named environment variable. An existing `ELPA::ELPA` target bypasses discovery.

The finder looks for ELPA headers and a library. It also adds the Fortran module directory when found. Typical locations are:

- `include/elpa-*/elpa/elpa.h`
- `include/elpa-*/modules/elpa.mod`
- `lib/libelpa.so` or `lib/libelpa_openmp.so`, also under `lib64/`

`ELPA_ROOT`, including its environment variable, is used when `EXTERNAL_ELPA_DIR` is absent. An explicit prefix restricts the search to that installation. Missing headers or libraries stop configuration. Supplying `ELPA_ROOT` also enables `LIBRPA_USE_ELPA`. Installation mode emits a compatibility warning on every configuration. Source builds use the compilers and settings selected for LibRPA.

```sh
cmake -B build -DEXTERNAL_ELPA_DIR=/path/to/elpa
```

(librpa-thirdparty-archive-dir)=
## `LIBRPA_THIRDPARTY_ARCHIVE_DIR`

Shared archive directory. The cache default is empty. Resolution uses a nonempty CMake value, then the environment variable, then `thirdparty/downloads/` in the LibRPA checkout. The common path rules above apply. The expanded default is not stored in the cache.

Different LibRPA checkouts can share these archives. Each checkout prepares its sources and patched copies in its own `thirdparty/src/` directory, including in parent projects. Both `thirdparty/downloads/` and `thirdparty/src/` are ignored by Git. Local archives work with `LIBRPA_THIRDPARTY_DOWNLOAD=OFF`. Downloads use temporary files inside the archive directory and rename them after success.

If a download is needed, the directory must be writable. Select a writable directory or prefetch the archives elsewhere. Corrupt archives remain in place. Check the full path in the error before deleting and replacing a shared archive.

```sh
export LIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads"
cmake -P cmake/LibRPADownloadThirdparty.cmake
```

(librpa-thirdparty-download)=
## `LIBRPA_THIRDPARTY_DOWNLOAD`

Allow downloads of missing archives. Default: `ON`.

With `OFF`, configuration can use an existing source tree or verify, extract and patch a local archive. Source-path overlays also remain available. The download script checks every selected archive, including when its sources already exist. The preparation script reuses existing sources. Missing inputs cause an error with instructions for downloading on a connected machine and manual archive placement.

The option controls downloads from the upstream URLs in `thirdparty/manifest.cmake`. Archive hash checks and source-change warnings apply in both modes.

```sh
cmake -B build \
  -DLIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads" \
  -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF
```

See {ref}`Offline installation <install-offline>` for preparation and transfer commands.

(libddla-source-dir)=
## `LIBDDLA_SOURCE_DIR`

LibDDLA source directory for CUDA or HIP builds. A nonempty CMake value takes precedence over the environment variable. An existing `LibDDLA::ddla` target takes precedence. Supplying both `LIBDDLA_SOURCE_DIR` and `LIBDDLA_PATH` is an error.

The directory must contain `include/ddla/ddla.h` and `CMakeLists.txt`. An invalid directory causes an error. LibRPA hashes the whole source tree. A match has status `pinned`, and a different hash produces an `unpinned` warning. Extra files, including Git metadata, can change this comparison.

LibRPA uses the supplied source directly, without patches or an overlay. It builds LibDDLA in a separate ExternalProject to isolate GPU language configuration. Each top-level build has its own `_deps/libddla-build` and `_deps/libddla-install` directories. Use a fresh build directory after changes to the LibDDLA sources or child toolchain.

The child receives `CMAKE_CXX_COMPILER`, `CMAKE_CXX_FLAGS`, `CMAKE_BUILD_TYPE`, `MPI_CXX_COMPILER` and `CMAKE_PREFIX_PATH`. Backend-specific variables are forwarded when set:

| Backend | Forwarded variables |
| --- | --- |
| CUDA | `CMAKE_CUDA_COMPILER`, `CMAKE_CUDA_FLAGS`, `CMAKE_CUDA_ARCHITECTURES`, `CMAKE_CUDA_SEPARABLE_COMPILATION`, `CUDAToolkit_ROOT` |
| HIP | `CMAKE_HIP_COMPILER`, `CMAKE_HIP_FLAGS`, `CMAKE_HIP_ARCHITECTURES`, `CMAKE_HIP_SEPARABLE_COMPILATION`, `CMAKE_HIP_COMPILER_ROCM_ROOT`, `ROCM_PATH` |

Other parent cache entries are not automatically forwarded. `ROCM_PATH` also accepts an environment fallback. LibRPA enables LibDDLA's collective communication backend, disables its tests, and selects GPU/CPU tunneling for HIP only.

Without a target, installation or source path, LibRPA prepares managed LibDDLA sources.

```sh
cmake -B build -DLIBRPA_USE_HIP=ON -DLIBDDLA_SOURCE_DIR=/path/to/LibDDLA
```

(build-shared-libs)=
## `BUILD_SHARED_LIBS`

Build shared LibRPA and Fortran-binding libraries when `ON`, or static libraries when `OFF`. The standalone default is `OFF` because LibRPA does not set this standard CMake variable. A parent project can supply its own value. Managed GreenX uses the same setting.

```sh
cmake -B build -DBUILD_SHARED_LIBS=ON
```

Managed ELPA remains a private static library. LibDDLA's child build provides a shared library independently of this option.

(cmake-build-type)=
## `CMAKE_BUILD_TYPE`

Configuration for a single-configuration generator, such as Unix Makefiles or Ninja. LibRPA does not set a default. Select `Release` for an optimized build, or `Debug` for debugging. A build type containing `Debug` also enables `LIBRPA_DEBUG`.

Use separate build directories for different compiler or child-library configurations. See the CMake generator's configuration mechanism for multi-configuration builds.

(cmake-install-prefix)=
## `CMAKE_INSTALL_PREFIX`

Destination prefix for installation. In a standalone build with no user-supplied prefix, LibRPA selects `install/` under its source directory. A parent project controls its own prefix. Library, executable and header subdirectories follow `GNUInstallDirs`.

```sh
cmake -B build -DCMAKE_INSTALL_PREFIX="$HOME/opt/librpa"
cmake --build build --target install
```

Build the selected configuration before installation. See {ref}`Installation <install-build>` for the installed files.

(mklroot)=
## `MKLROOT`

When nonempty, selects LibRPA's Intel MKL discovery path for BLAS, LAPACK and ScaLAPACK. A defined CMake value takes precedence over the environment variable, including an explicitly empty value.

Set `-DMKLROOT=` to select the separate LAPACK and ScaLAPACK discovery path when the shell has an MKL environment. Use math libraries compatible with the compiler and MPI toolchain.

(lapack-libraries)=
## `LAPACK_LIBRARIES`

Explicit LAPACK link libraries for the non-MKL discovery path. CMake detects them when no value is supplied. OpenBLAS can supply both LAPACK and BLAS. A semicolon-separated list supports multiple library files.

```sh
cmake -B build -DMKLROOT= \
  -DLAPACK_LIBRARIES=/path/to/openblas/lib/libopenblas.so \
  -DSCALAPACK_DIR=/path/to/scalapack
```

(blas-libraries)=
## `BLAS_LIBRARIES`

Explicit BLAS link libraries for CMake's math-library discovery. CMake detects them when no value is supplied. This setting does not override a nonempty `MKLROOT`. Supply a complete, compatible LAPACK selection as described in {ref}`Installation <install-build>`.
