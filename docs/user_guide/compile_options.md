# Compile Options

(librpa-how-to-set-options)=
## How to set options

You can set build options with `-DNAME=value` when configuring LibRPA with CMake.
For example, from the LibRPA source directory, run:

```sh
cmake -B build -DLIBRPA_USE_LIBRI=ON
```

CMake saves these settings in the `build` directory and reuses them on later configurations.
You can change an option by running CMake again with a new value.
When changing compilers or MPI installations, use a new build directory to avoid mixing incompatible build files.

The following standard CMake options let you choose compilers, compiler flags and package search paths.
Set them when you first configure the build directory:

| Setting | Purpose |
| --- | --- |
| `CMAKE_CXX_COMPILER` | C++ compiler or MPI C++ wrapper |
| `CMAKE_Fortran_COMPILER` | Fortran compiler or MPI Fortran wrapper, required even without the binding |
| `CMAKE_C_COMPILER` | C compiler or MPI C wrapper for ELPA source builds |
| `CMAKE_CXX_FLAGS`<br>`CMAKE_Fortran_FLAGS`<br>`CMAKE_C_FLAGS` | Flags for the corresponding language |
| `CMAKE_PREFIX_PATH` | Additional package search prefixes; header-only dependencies have separate include-path options |

LibRPA uses C++17 and needs compatible compilers, MPI and math libraries.
Once configuration is complete, you can choose the number of parallel build jobs with `cmake --build build --parallel N`.
For the build requirements and a complete build procedure, see {ref}`Installation <install-build-requirements>`.

## Overview

The table below summarizes LibRPA-specific build options.
It also includes selected standard CMake options for reference, but does not cover all of them.

| Option                                                      | Type              | Default    |
|-------------------------------------------------------------|-------------------|------------|
| [`LIBRPA_USE_LIBRI`](#librpa-use-libri)                     | bool              | `ON`       |
| [`LIBRPA_USE_LIBRI_GPU`](#librpa-use-libri-gpu)             | bool              | `OFF`      |
| [`LIBRPA_USE_CUDA`](#librpa-use-cuda)                       | bool              | `OFF`      |
| [`LIBRPA_USE_HIP`](#librpa-use-hip)                         | bool              | `OFF`      |
| [`LIBRPA_USE_CMAKE_INC`](#librpa-use-cmake-inc)             | bool              | `OFF`      |
| [`LIBRPA_USE_EXTERNAL_GREENX`](#librpa-use-external-greenx) | bool (deprecated, ignored) | unset |
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
| [`ELPA_PATH`](#elpa-path) | path | empty |
| [`EXTERNAL_ELPA_DIR`](#external-elpa-dir) | path (deprecated) | unset |
| [`LIBRPA_USE_BUNDLED_ELPA`](#librpa-use-bundled-elpa) | bool (deprecated) | unset |
| [`LIBRPA_BUNDLED_ELPA_VERSION`](#librpa-bundled-elpa-version) | string (deprecated, ignored) | unset |
| [`LIBRPA_ELPA_KERNEL`](#librpa-elpa-kernel) | string            | empty      |
| [`LIBRPA_ELPA_OPENMP`](#librpa-elpa-openmp) | bool              | `OFF`      |
| [`LIBRPA_ELPA_CONFIGURE_ARGS`](#librpa-elpa-configure-args) | string | empty |
| [`LIBRPA_ELPA_LIBS`](#librpa-elpa-libs)     | string            | empty      |
| [`LIBRPA_THIRDPARTY_DOWNLOAD`](#librpa-thirdparty-download) | bool | `ON` |
| [`LIBRPA_THIRDPARTY_ARCHIVE_DIR`](#librpa-thirdparty-archive-dir) | path | empty |
| [`GREENX_SOURCE_DIR`](#greenx-source-dir) | path | empty |
| [`GREENX_PATH`](#greenx-path) | path | empty |
| [`EXTERNAL_GREENX_DIR`](#external-greenx-dir) | path (deprecated) | unset |
| [`ELPA_SOURCE_DIR`](#elpa-source-dir) | path | empty |
| [`LIBDDLA_SOURCE_DIR`](#libddla-source-dir) | path | empty |
| [`BUILD_SHARED_LIBS`](#build-shared-libs) | bool | `OFF` in a standalone build |
| [`CMAKE_BUILD_TYPE`](#cmake-build-type) | string | empty |
| [`CMAKE_INSTALL_PREFIX`](#cmake-install-prefix) | path | source `install/` (standalone) |
| [`MKLROOT`](#mklroot) | path | environment value, if defined |
| [`LAPACK_LIBRARIES`](#lapack-libraries) | library list | detected |
| [`BLAS_LIBRARIES`](#blas-libraries) | library list | detected |

## Option details

(librpa-use-libri)=
### `LIBRPA_USE_LIBRI`

This option enables [LibRI](https://github.com/abacusmodeling/LibRI) for RI tensor contractions.

The *GW* and EXX calculations require `LIBRPA_USE_LIBRI=ON`, while the RPA correlation energy can also be computed without LibRI.
The analytic optical wing also requires LibRI tensor communication; the analytic
head and RPA calculations without an analytic wing remain available without LibRI.

(librpa-use-libri-gpu)=
### `LIBRPA_USE_LIBRI_GPU`

This option enables GPU-accelerated LibRI tensor contractions. It requires [`LIBRPA_USE_LIBRI`](#librpa-use-libri) and exactly one of
[`LIBRPA_USE_CUDA`](#librpa-use-cuda) or
[`LIBRPA_USE_HIP`](#librpa-use-hip) to be enabled.

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
### `LIBRPA_USE_CUDA`

This option enables the NVIDIA CUDA backend and requires the CUDA toolkit, NCCL and LibDDLA.
You can enable either CUDA or HIP in a build, but not both.

If `CMAKE_CUDA_ARCHITECTURES` is not specified, LibRPA uses `70;75;80`.
Set this standard CMake variable explicitly when building for other GPU
architectures. You can use an existing LibDDLA installation through
[`LIBDDLA_PATH`](#libddla-path), or build your own sources through [`LIBDDLA_SOURCE_DIR`](#libddla-source-dir).
Otherwise, CMake prepares and builds the managed LibDDLA sources.

Example:
```sh
cmake -B build \
      -DLIBRPA_USE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80
```

(librpa-use-hip)=
### `LIBRPA_USE_HIP`

This option enables the AMD HIP/ROCm backend and requires the HIP/ROCm libraries, RCCL and LibDDLA.
You can enable either HIP or CUDA in a build, but not both.

Set `CMAKE_HIP_ARCHITECTURES` for the target GPU when necessary. `ROCM_PATH`
can be used when the ROCm installation cannot be inferred from the
environment. You can use an existing LibDDLA installation through
[`LIBDDLA_PATH`](#libddla-path), or build your own sources through [`LIBDDLA_SOURCE_DIR`](#libddla-source-dir).
Otherwise, CMake prepares and builds the managed LibDDLA sources.

Example:
```sh
cmake -B build -DLIBRPA_USE_HIP=ON
```

(librpa-use-elpa)=
### `LIBRPA_USE_ELPA`

This option enables ELPA for distributed eigensolvers and is `OFF` by default.
When you set it to `ON` without supplying a dependency, LibRPA downloads the release listed in the manifest, verifies the archive, and extracts and patches the sources under `thirdparty/src` before building them.
You can also supply your own ELPA sources or installation, as shown below.
For offline builds, LibRPA can reuse existing managed sources or cached archives with downloads disabled.

Supplying an `ELPA::ELPA` target, `ELPA_SOURCE_DIR`, or an installation prefix (`ELPA_PATH`, with `ELPA_ROOT` as a fallback) automatically enables ELPA, even if this option was set to `OFF`.
A parent target takes precedence over paths. When supplying a path, choose either a source directory or an installation prefix.

```sh
# Build the managed release
cmake -B build -DLIBRPA_USE_ELPA=ON

# Build a supplied source tree; ELPA is enabled automatically
cmake -B build -DELPA_SOURCE_DIR=/path/to/elpa-source

# Use an installed library; ELPA is enabled automatically
cmake -B build -DELPA_PATH=/path/to/elpa
```

LibRPA builds ELPA sources with upstream autotools and keeps the resulting installation inside the build directory.
The source and build paths need to be free of whitespace.
After changing compiler flags, `CMAKE_BUILD_TYPE` or ELPA sources, use a fresh build directory so that the separate ELPA build picks up the changes.

(librpa-use-external-elpa)=
### `LIBRPA_USE_EXTERNAL_ELPA` (deprecated)

For new configurations, use `LIBRPA_USE_ELPA` and the dependency settings above.
CMake warns whenever this legacy option is defined, whether its value is `ON` or `OFF`.
Setting it to `ON` enables ELPA and, if no target or path is supplied, searches for an installed library in the standard locations.
You can still select a source build by supplying an explicit source path.

To migrate an installed-library configuration, set `ELPA_PATH` to its installation prefix and remove the old cache entry with `-ULIBRPA_USE_EXTERNAL_ELPA`.

(librpa-use-bundled-elpa)=
### `LIBRPA_USE_BUNDLED_ELPA` (deprecated)

For new configurations, use `LIBRPA_USE_ELPA=ON` to build the managed release, or supply your own sources through `ELPA_SOURCE_DIR`.
CMake warns whenever this legacy option is defined, whether its value is `ON` or `OFF`.
Setting it to `ON` enables ELPA and selects the dependency using the target and path rules described above.

Setting either legacy option to `OFF` does not disable ELPA if another setting enables it.
When both legacy options are `ON` and no target or path is supplied, LibRPA searches for an installed library.

Remove the old cache entry with `-ULIBRPA_USE_BUNDLED_ELPA` when migrating. Legacy options retained in the cache continue to emit warnings on later configurations.

(librpa-bundled-elpa-version)=
### `LIBRPA_BUNDLED_ELPA_VERSION` (deprecated)

LibRPA no longer uses this option and issues a deprecation warning if it is defined.
The manifest determines which ELPA release to download, while `ELPA_SOURCE_DIR` lets you choose the version through your supplied sources.
LibRPA reads `PACKAGE_VERSION` from the selected source tree's `configure` script to determine the header installation paths and version-specific configure arguments.

Remove the old cache entry with `-ULIBRPA_BUNDLED_ELPA_VERSION`. There is no replacement version option.

(librpa-elpa-kernel)=
### `LIBRPA_ELPA_KERNEL`

This option selects an x86 SIMD kernel family when LibRPA builds downloaded ELPA sources or a tree supplied through `ELPA_SOURCE_DIR`.
Installed libraries and parent ELPA targets keep their own kernel settings.

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
      -DLIBRPA_ELPA_KERNEL=AVX512
```

(librpa-elpa-openmp)=
### `LIBRPA_ELPA_OPENMP`

This option controls ELPA's own OpenMP support when LibRPA builds downloaded ELPA sources or a tree supplied through `ELPA_SOURCE_DIR`.
Installed libraries and parent ELPA targets keep their own threading settings.

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
      -DLIBRPA_ELPA_OPENMP=ON
```

(librpa-elpa-configure-args)=
### `LIBRPA_ELPA_CONFIGURE_ARGS`

You can use this option to pass additional arguments to the `configure` script when building downloaded ELPA sources or a tree supplied through `ELPA_SOURCE_DIR`.
It does not affect installed libraries or parent ELPA targets.
Put the arguments in a single quoted string, separated by spaces; CMake parses it with `separate_arguments(... UNIX_COMMAND ...)`.

Arguments passed through this option are appended after LibRPA's defaults,
including [`LIBRPA_ELPA_KERNEL`](#librpa-elpa-kernel), so they
can override the default kernel selection when a specific ELPA setup is needed.

Example:
```sh
cmake -B build -DLIBRPA_USE_ELPA=ON \
      -DLIBRPA_ELPA_CONFIGURE_ARGS="--enable-store-build-config"
```

(librpa-elpa-libs)=
### `LIBRPA_ELPA_LIBS`

You can use this option to pass linker flags to the ELPA `configure` script through its `LIBS` environment variable.
It applies to downloaded ELPA sources and trees supplied through `ELPA_SOURCE_DIR`; installed libraries and parent ELPA targets keep their own link settings.

By default, LibRPA forwards the detected LAPACK and ScaLAPACK libraries to the
ELPA source build. Set this option only when the autodetected flags are not
suitable for a particular compiler or math library setup.

When static math libraries are used, ELPA's libtool build may try to include
those static archives inside `libelpa.a`. LibRPA removes such nested archive
members after the ELPA source install step and links the math libraries
separately through CMake.

Example:
```sh
cmake -B build -DLIBRPA_USE_ELPA=ON \
      -DLIBRPA_ELPA_LIBS="-L/path/to/lib -lscalapack -llapack -lblas"
```

(librpa-bundled-elpa-kernel)=
(librpa-bundled-elpa-openmp)=
(librpa-bundled-elpa-configure-args)=
(librpa-bundled-elpa-libs)=
### Deprecated ELPA source-build option names

| Deprecated name | Replacement |
| --- | --- |
| `LIBRPA_BUNDLED_ELPA_KERNEL` | `LIBRPA_ELPA_KERNEL` |
| `LIBRPA_BUNDLED_ELPA_OPENMP` | `LIBRPA_ELPA_OPENMP` |
| `LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS` | `LIBRPA_ELPA_CONFIGURE_ARGS` |
| `LIBRPA_BUNDLED_ELPA_LIBS` | `LIBRPA_ELPA_LIBS` |

These options apply only when LibRPA builds ELPA from source.
CMake warns when an old name is defined and uses its value to initialize the replacement if the new name is unset.
A value supplied under the new name takes precedence, including `OFF` or an empty string.
After migrating, remove the old cache entries to stop the warnings.

(librpa-use-cmake-inc)=
### `LIBRPA_USE_CMAKE_INC`

This option initializes compilers and other build options from `cmake.inc`.
It is **deprecated**; for new configurations, use standard CMake command-line options such as `-C` or `-D`.

(librpa-use-external-greenx)=
### `LIBRPA_USE_EXTERNAL_GREENX` (deprecated)

GreenX is always enabled, so LibRPA no longer uses this option.
CMake issues a deprecation warning if it is defined, whether its value is `ON` or `OFF`.
To use your own GreenX copy, supply a parent target or choose either a source directory through `GREENX_SOURCE_DIR` or an installation prefix through `GREENX_PATH`.
Otherwise, LibRPA prepares managed sources.

Remove the old cache entry with `-ULIBRPA_USE_EXTERNAL_GREENX`. No replacement switch is needed. See {ref}`Parent-project integration <thirdparty-parent-projects>` for target requirements.

(greenx-source-dir)=
### `GREENX_SOURCE_DIR`

You can set this option to a greenX source directory containing `GX-TimeFrequency/CMakeLists.txt`.
A nonempty CMake value takes precedence over the environment variable of the same name, and a parent target takes precedence over this path.

LibRPA compares the whole source tree with the version recorded in the manifest.
It uses matching patched sources directly (`pinned`), or creates a patched copy of matching upstream sources in managed storage (`overlay`).
Other versions are used directly with an `unpinned` warning.
Your original sources remain unchanged, and LibRPA builds the selected tree with its own compilers.

`GREENX_SOURCE_DIR` and `GREENX_PATH` are mutually exclusive. An invalid explicit path stops configuration without a download fallback.

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(greenx-path)=
### `GREENX_PATH`

You can set this option to the installation prefix of a prebuilt greenX library.
A nonempty CMake value takes precedence over the environment variable of the same name.
If a parent target is available, LibRPA uses it instead of searching for an installation.

LibRPA loads `greenXConfig.cmake` from this prefix and uses the `greenX::LibGXMiniMax` target.
The target's link interface must provide `gx_minimax.mod` and `kinds.mod`; configuration stops if either module is missing.
LibRPA also warns if the module path identifies a different Fortran compiler or version.

Source builds are generally easier to match to LibRPA's toolchain.
The recommended starting point is to leave this option empty and use managed sources; a parent target or `GREENX_SOURCE_DIR` lets you build your own copy.
For a prebuilt installation, make sure its compilers, flags and BLAS/LAPACK dependencies are compatible with LibRPA. CMake displays a reminder on each configuration.
You can supply `BLAS_LIBRARIES` and `LAPACK_LIBRARIES` if needed.

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(external-greenx-dir)=
### `EXTERNAL_GREENX_DIR` (deprecated)

This is a legacy alias for `GREENX_PATH`. A nonempty `GREENX_PATH` CMake value takes precedence, followed by its environment variable, the legacy CMake value, and the legacy environment variable. Defining the old name in CMake or the environment emits a deprecation warning, including when its value is empty or the new name takes precedence.

To migrate, set `GREENX_PATH`, remove the old cache entry with `-UEXTERNAL_GREENX_DIR`, and unset the old environment variable. An environment value selected through the alias is saved under the legacy name in the cache for later configurations.

(elpa-source-dir)=
### `ELPA_SOURCE_DIR`

You can set this option to an ELPA source directory containing `configure` and `src/elpa_impl.F90`.
A nonempty CMake value takes precedence over the environment variable of the same name.

Supplying this path automatically enables `LIBRPA_USE_ELPA`. Choose either this source path or an installation prefix (`ELPA_PATH` or `ELPA_ROOT`), including environment values. A parent `ELPA::ELPA` target takes precedence over the source path.

LibRPA compares the whole source tree with the version recorded in the manifest.
It uses matching patched sources directly (`pinned`), or creates a patched copy of matching upstream sources in managed storage (`overlay`).
Other versions produce an `unpinned` warning. The build takes place outside the source tree, preserving both your sources and any patched copy.
If the supplied path is invalid, configuration stops rather than downloading a replacement.

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(librpa-enable-fortran-bind)=
### `LIBRPA_ENABLE_FORTRAN_BIND`

This option builds the Fortran bindings for LibRPA.

(librpa-fortran-dp)=
### `LIBRPA_FORTRAN_DP`

This option sets the Fortran kind for double-precision real and complex data in the Fortran bindings.

The default value is `c_double`, which is suitable when interoperability with C is desired.
This option may also be set to an integer kind value if needed by the calling code.

This option is meaningful only if `LIBRPA_ENABLE_FORTRAN_BIND=ON`.

(librpa-enable-driver)=
### `LIBRPA_ENABLE_DRIVER`

This option builds the LibRPA driver executable.

(librpa-mpi-thread-level)=
### `LIBRPA_MPI_THREAD_LEVEL`

This option sets the MPI thread-support level requested by the driver and C++ tests. Accepted values are `MPI_THREAD_SINGLE`, `MPI_THREAD_FUNNELED`, `MPI_THREAD_SERIALIZED` and `MPI_THREAD_MULTIPLE`.

The default cache value is empty. In that case, LibRPA selects the level in this order:

| Condition | Selected level |
| --- | --- |
| `LIBRPA_USE_ELPA=ON`, with neither CUDA nor HIP | `MPI_THREAD_MULTIPLE` |
| Otherwise, LibComm status is `managed`, `pinned` or `overlay` | `MPI_THREAD_FUNNELED` |
| Otherwise | `MPI_THREAD_MULTIPLE` |

Modified managed LibComm headers have status `unpinned` and select `MPI_THREAD_MULTIPLE` automatically. A parent LibComm target has status `target`. LibRPA does not infer its pinned status from the target name or include path. An explicit value takes precedence over the automatic selection.

```sh
cmake -B build -DLIBRPA_MPI_THREAD_LEVEL=MPI_THREAD_FUNNELED
```

The chosen level must satisfy the dependencies' threading requirements.

(librpa-verbose-output)=
### `LIBRPA_VERBOSE_OUTPUT`

This option adds per-timer timestamp and memory diagnostics to the build.
The runtime output level still controls which diagnostic messages are emitted.
Disable this option when those detailed diagnostics are not needed.

(librpa-enable-test)=
### `LIBRPA_ENABLE_TEST`

This option builds the LibRPA unit tests.

After building LibRPA, you can run the tests from the build directory with:
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
### `LIBRPA_ENABLE_CPP_TEST`

This option builds the C++ unit tests when `LIBRPA_ENABLE_TEST=ON`.

(librpa-enable-fortran-test)=
### `LIBRPA_ENABLE_FORTRAN_TEST`

This option builds the Fortran unit tests when both `LIBRPA_ENABLE_TEST=ON` and `LIBRPA_ENABLE_FORTRAN_BIND=ON`.

(libri-include-dir)=
### `LIBRI_INCLUDE_DIR`

You can use this option to select your own LibRI headers when `LIBRPA_USE_LIBRI=ON`.
Set it to the include directory containing `RI/ri/LRI.h`.

If a parent project supplies `LibRI::LibRI`, LibRPA uses that target.
Otherwise, it uses a nonempty CMake value first, then the environment variable of the same name. If you supply an invalid directory, configuration stops without searching system paths or downloading a replacement.

For an explicit path, LibRPA hashes the include tree on every configuration:

| Match | Action and status |
| --- | --- |
| Pinned patched headers | Use the directory directly (`pinned`) |
| Pinned pristine headers | Copy into managed storage, apply the LibRI patch and compare with the reference (`overlay` if unchanged) |
| Any other headers | Use directly with a warning (`unpinned`) |

LibRPA applies patches to a copy, keeping your supplied directory unchanged.
If you edit that copy, LibRPA continues to use it with an `unpinned` warning.
Creating the copy requires a writable `thirdparty/src/` directory in the LibRPA checkout and also works with downloads disabled.

If no target or path is supplied, LibRPA prepares managed headers.
During configuration, a compile check instantiates `RI::LRI<int,int,3,double>` with the selected LibComm and cereal headers to test that the three work together.
If this check fails, configuration stops and reports all three sources.
The separate check for fine-grained locks remains optional. Hartree support depends on `RI/physics/Hartree.h` being available in the selected include directories.

```sh
cmake -B build -DLIBRI_INCLUDE_DIR=/path/to/LibRI/include
```

(dependency-path-options)=
The header include paths (`LIBRI_INCLUDE_DIR`, `LIBCOMM_INCLUDE_DIR` and `CEREAL_INCLUDE_DIR`) and the source and installation paths for GreenX, ELPA and LibDDLA share the following CMake path handling.
When a path cache entry is empty, LibRPA imports a nonempty environment value, reports where it came from, and saves it for later configurations.
The saved path remains active even after you unset the environment variable. To clear an override, remove both its cache entry and its environment variable.

These options use CMake `PATH` cache entries. On the first configuration, an untyped relative value such as `-DNAME=path` is resolved against the directory where you run CMake.
For environment values, typed values such as `-DNAME:PATH=value`, and later cache updates, use absolute paths.
LibRPA rejects any remaining relative value before using it; header checks and dependency selection use the same resolved path.

(libcomm-include-dir)=
### `LIBCOMM_INCLUDE_DIR`

You can use this option to select your own LibComm headers when `LIBRPA_USE_LIBRI=ON`.
Set it to the include directory containing `Comm/Comm_Tools.h`.

If a parent project supplies `LibComm::LibComm`, LibRPA uses that target.
Otherwise, it uses a nonempty CMake value first, then the environment variable of the same name. If you supply an invalid directory, configuration stops without searching system paths or downloading a replacement.

Headers matching the pinned, patched tree are used directly (`pinned`). Headers matching the pinned upstream tree are copied and patched in managed storage (`overlay`). Other headers are used with an `unpinned` warning. The required LibRI compile probe checks the selected combination.

If no target or path is supplied, LibRPA prepares managed headers. Unmodified managed, pinned and overlay LibComm influence the automatic [MPI thread-level selection](#librpa-mpi-thread-level).

```sh
cmake -B build -DLIBCOMM_INCLUDE_DIR=/path/to/LibComm/include
```

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(cereal-include-dir)=
### `CEREAL_INCLUDE_DIR`

You can use this option to select your own cereal headers when `LIBRPA_USE_LIBRI=ON`.
Set it to the include directory containing `cereal/cereal.hpp`.

If a parent project supplies `cereal::cereal`, LibRPA uses that target.
Otherwise, it uses a nonempty CMake value first, then the environment variable of the same name. If you supply an invalid directory, configuration stops without searching system paths or downloading a replacement.

Headers matching the pinned upstream tree are used directly (`pinned`). Cereal has no managed patch, so it does not need an overlay. Other headers are used with an `unpinned` warning. The required LibRI compile probe checks the selected combination.

If no target or path is supplied, LibRPA prepares managed headers. The manifest pins cereal to `d81e2f7df7b334fee057e53017388d02e555a836`, which includes the tuple-serialization fix used by this LibRI revision.

```sh
cmake -B build -DCEREAL_INCLUDE_DIR=/path/to/cereal/include
```

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(libddla-path)=
### `LIBDDLA_PATH`

You can set this option to the installation prefix of LibDDLA for a CUDA or HIP build.
A nonempty CMake value takes precedence over the environment variable of the same name.

The resolution order is:

1. Existing `LibDDLA::ddla` target.
2. `LIBDDLA_SOURCE_DIR` source tree or `LIBDDLA_PATH` installation. Supplying both is an error.
3. Managed source from the manifest.

Choose either `LIBDDLA_PATH` or `LIBDDLA_SOURCE_DIR`.
Managed sources are the recommended starting point: leave both options empty to use them.
A parent target or a source build is another way to match LibDDLA to your toolchain.
For a prebuilt installation, CMake displays a reminder about compiler and flag compatibility on each configuration.

The installation must contain `include/ddla/ddla.h` and a `ddla` library under `lib/` or `lib64/`. Missing headers or a missing library cause an error. Installed libraries are not checked against the manifest's source-tree hash. The installation must match the selected backend, compiler and MPI environment.

```sh
cmake -B build -DLIBRPA_USE_CUDA=ON -DLIBDDLA_PATH=/path/to/libddla
```

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(scalapack-dir)=
### `SCALAPACK_DIR`

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

(elpa-path)=
### `ELPA_PATH`

You can set this option to the installation prefix of ELPA, which also enables `LIBRPA_USE_ELPA`.
A nonempty CMake value takes precedence over the environment variable of the same name.
If an `ELPA::ELPA` target is already available, LibRPA uses it instead of searching for an installation.

The finder looks for ELPA headers and a library. It also adds the Fortran module directory when found. Typical locations are:

- `include/elpa-*/elpa/elpa.h`
- `include/elpa-*/modules/elpa.mod`
- `lib/libelpa.so` or `lib/libelpa_openmp.so`, also under `lib64/`

`ELPA_ROOT`, including its environment variable, is used when `ELPA_PATH` is absent. An explicit prefix restricts the search to that installation. Missing headers or libraries stop configuration. Supplying `ELPA_ROOT` also enables `LIBRPA_USE_ELPA`. Installation mode emits a compatibility warning on every configuration. Source builds use the compilers and settings selected for LibRPA.

```sh
cmake -B build -DELPA_PATH=/path/to/elpa
```

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(external-elpa-dir)=
### `EXTERNAL_ELPA_DIR` (deprecated)

This is a legacy alias for `ELPA_PATH`. A nonempty `ELPA_PATH` CMake value takes precedence, followed by its environment variable, the legacy CMake value, and the legacy environment variable. `ELPA_ROOT` is the fallback when neither name supplies a path. Defining the old name in CMake or the environment emits a deprecation warning, including when its value is empty or the new name takes precedence.

A path supplied through this alias also enables `LIBRPA_USE_ELPA`. To migrate, set `ELPA_PATH`, remove the old cache entry with `-UEXTERNAL_ELPA_DIR`, and unset the old environment variable. An environment value selected through the alias is saved under the legacy name in the cache for later configurations.

(librpa-thirdparty-archive-dir)=
### `LIBRPA_THIRDPARTY_ARCHIVE_DIR`

This option selects the directory where LibRPA stores downloaded dependency archives.
You can set it through CMake or the environment variable of the same name; a nonempty CMake value takes precedence.
The cache entry is empty by default. If neither supplies a path, LibRPA uses `thirdparty/downloads/` in the checkout without saving that default path in the cache.

This option also works with the standalone archive download script, `cmake/LibRPADownloadThirdparty.cmake`, and source preparation script, `cmake/LibRPAPrepareThirdparty.cmake`.
During configuration, it follows the same {ref}`path and cache rules as dependency paths <dependency-path-options>`. With `cmake -P`, the selected path applies only to that script run.

Different LibRPA checkouts can share these archives, while each checkout keeps its prepared sources and patched copies in its own `thirdparty/src/` directory. This also applies when LibRPA is part of a parent project.
Both `thirdparty/downloads/` and `thirdparty/src/` are ignored by Git.
Local archives are available with `LIBRPA_THIRDPARTY_DOWNLOAD=OFF`. Downloads are saved to temporary files in the archive directory and renamed once complete.

For downloads, choose a writable directory or download the archives elsewhere beforehand.
If an existing archive fails verification, LibRPA leaves it in place and reports its full path.
Check that path before replacing the archive, since other checkouts may also use it.

```sh
export LIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads"
cmake -P cmake/LibRPADownloadThirdparty.cmake
```

(librpa-thirdparty-download)=
### `LIBRPA_THIRDPARTY_DOWNLOAD`

This option allows LibRPA to download missing dependency archives and is `ON` by default.

For offline builds, set this option to `OFF`. Configuration can then reuse existing source trees or verify, extract and patch local archives, including creating patched copies of supplied sources.
If a required source tree or archive is missing, CMake explains how to download the archive on a connected machine and where to place it.

This option also works with the standalone scripts `cmake/LibRPADownloadThirdparty.cmake` and `cmake/LibRPAPrepareThirdparty.cmake`.
The download script checks every selected archive, even when its sources already exist, while the preparation script reuses existing sources.

The option controls downloads from the upstream URLs in `thirdparty/manifest.cmake`. Archive hash checks and source-change warnings apply in both modes.

```sh
cmake -B build \
  -DLIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads" \
  -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF
```

See {ref}`Offline installation <install-offline>` for preparation and transfer commands.

(libddla-source-dir)=
### `LIBDDLA_SOURCE_DIR`

You can set this option to a LibDDLA source directory for a CUDA or HIP build.
A nonempty CMake value takes precedence over the environment variable of the same name, and an existing `LibDDLA::ddla` target takes precedence over the path.
Choose either a source directory through this option or an installation through `LIBDDLA_PATH`.

The directory must contain `include/ddla/ddla.h` and `CMakeLists.txt`; configuration stops if the directory is invalid.
LibRPA compares the hash of the whole source tree with the manifest and reports `pinned` for a match or an `unpinned` warning otherwise.
This comparison also includes extra files, such as Git metadata.

LibRPA builds the supplied sources directly, without patches or a separate source copy.
It uses CMake's ExternalProject mechanism to configure the GPU languages in a separate build, with `_deps/libddla-build` and `_deps/libddla-install` directories for each top-level build.
After changing the LibDDLA sources or its toolchain, use a fresh build directory so that the separate build picks up the changes.

The separate LibDDLA build receives `CMAKE_CXX_COMPILER`, `CMAKE_CXX_FLAGS`, `CMAKE_BUILD_TYPE`, `MPI_CXX_COMPILER` and `CMAKE_PREFIX_PATH`. Backend-specific variables are forwarded when set:

| Backend | Forwarded variables |
| --- | --- |
| CUDA | `CMAKE_CUDA_COMPILER`, `CMAKE_CUDA_FLAGS`, `CMAKE_CUDA_ARCHITECTURES`, `CMAKE_CUDA_SEPARABLE_COMPILATION`, `CUDAToolkit_ROOT` |
| HIP | `CMAKE_HIP_COMPILER`, `CMAKE_HIP_FLAGS`, `CMAKE_HIP_ARCHITECTURES`, `CMAKE_HIP_SEPARABLE_COMPILATION`, `CMAKE_HIP_COMPILER_ROCM_ROOT`, `ROCM_PATH` |

Other parent cache entries are not automatically forwarded. `ROCM_PATH` also accepts an environment fallback. LibRPA enables LibDDLA's collective communication backend, disables its tests, and selects GPU/CPU tunneling for HIP only.

Without a target, installation or source path, LibRPA prepares managed LibDDLA sources.

```sh
cmake -B build -DLIBRPA_USE_HIP=ON -DLIBDDLA_SOURCE_DIR=/path/to/LibDDLA
```

For path formats and cached environment values, see the {ref}`dependency path settings <dependency-path-options>`.

(build-shared-libs)=
### `BUILD_SHARED_LIBS`

This option selects shared LibRPA and Fortran-binding libraries when `ON`, or static libraries when `OFF`. The standalone default is `OFF` because LibRPA does not set this standard CMake variable. A parent project can supply its own value. Managed GreenX uses the same setting.

```sh
cmake -B build -DBUILD_SHARED_LIBS=ON
```

Managed ELPA remains a private static library. LibDDLA's child build provides a shared library independently of this option.

(cmake-build-type)=
### `CMAKE_BUILD_TYPE`

This option selects the build configuration for a single-configuration generator, such as Unix Makefiles or Ninja. LibRPA does not set a default. Select `Release` for an optimized build, or `Debug` for debugging. A build type containing `Debug` also enables `LIBRPA_DEBUG`.

Use separate build directories for different compiler or child-library configurations. See the CMake generator's configuration mechanism for multi-configuration builds.

(cmake-install-prefix)=
### `CMAKE_INSTALL_PREFIX`

This option sets the installation prefix for LibRPA. In a standalone build with no user-supplied prefix, LibRPA selects `install/` under its source directory. A parent project controls its own prefix. Library, executable and header subdirectories follow `GNUInstallDirs`.

```sh
cmake -B build -DCMAKE_INSTALL_PREFIX="$HOME/opt/librpa"
cmake --build build --target install
```

Build the selected configuration before installation. See {ref}`Installation <install-build>` for the installed files.

(mklroot)=
### `MKLROOT`

When this option is nonempty, LibRPA looks for BLAS, LAPACK and ScaLAPACK through Intel MKL. A defined CMake value takes precedence over the environment variable, including an explicitly empty value.

Set `-DMKLROOT=` to select the separate LAPACK and ScaLAPACK discovery path when the shell has an MKL environment. Use math libraries compatible with the compiler and MPI toolchain.

(lapack-libraries)=
### `LAPACK_LIBRARIES`

You can use this option to specify LAPACK libraries when building without MKL.
Otherwise, CMake detects the libraries from your environment.
OpenBLAS can supply both LAPACK and BLAS; for multiple library files, use a semicolon-separated list.

```sh
cmake -B build -DMKLROOT= \
  -DLAPACK_LIBRARIES=/path/to/openblas/lib/libopenblas.so \
  -DSCALAPACK_DIR=/path/to/scalapack
```

(blas-libraries)=
### `BLAS_LIBRARIES`

You can use this option to specify BLAS libraries, or leave it unset to let CMake detect them.
A nonempty `MKLROOT` still takes precedence.
Make sure the selected libraries also provide a complete, compatible LAPACK setup, as described in {ref}`Installation <install-build>`.
