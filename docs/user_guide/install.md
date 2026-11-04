# Installation

(install-build-requirements)=
## Overview of build procedure

### Requirements

The CPU build requires:

- A C++17 compiler with a working `std::filesystem` implementation.
- A Fortran compiler, including when the Fortran binding is disabled. GreenX contains Fortran code.
- MPI and OpenMP development libraries compatible with the selected compilers.
- BLAS, LAPACK and ScaLAPACK libraries compatible with that MPI installation.
- CMake 3.18 or newer, and a build tool supported by the selected CMake generator.
- Git or `patch` when source preparation needs to apply patches. Git has precedence when both tools are available.

### Procedure

1. Configure the build directory with CMake and select the compilers and libraries. During configuration, LibRPA prepares the third-party libraries that the selected options need (see the table below). By default, it downloads the versions listed in `thirdparty/manifest.cmake`, applies LibRPA's patches and checks the result against the hashes in the manifest. The first configuration thus needs network access. For a machine without network access, prepare the libraries on another machine, as described in {ref}`Offline installation <install-offline>`. To use your own copies instead, see {ref}`Choosing dependency sources <install-dependency-sources>`.
2. Build LibRPA and the compiled third-party libraries.
3. Install, if necessary.

{ref}`Download, build and install <install-build>` gives the commands for these steps.

### Third-party dependencies

| Component | When LibRPA needs it | Default source |
| --- | --- | --- |
| GreenX | Every build | Parent target, source path, installation, then managed source |
| LibRI, LibComm, cereal | `LIBRPA_USE_LIBRI=ON`, the default | Managed headers, unless the parent provides targets or the user supplies include paths |
| LibDDLA | `LIBRPA_USE_CUDA=ON` or `LIBRPA_USE_HIP=ON` | Managed source, unless a target, installation or local source tree is supplied |
| ELPA | Either ELPA option is enabled | Source path or managed source with `LIBRPA_USE_BUNDLED_ELPA=ON`, or a target/installation with `LIBRPA_USE_EXTERNAL_ELPA=ON` |

*GW* and EXX require LibRI. RPA correlation energies can also be computed with `LIBRPA_USE_LIBRI=OFF`. GreenX remains required in that configuration.

Managed ELPA also needs a C compiler, `make`, Bash and a C preprocessor. ELPA source builds require source and build directory paths without whitespace. Use a `LIBRPA_THIRDPARTY_DIR` without spaces, and check `ELPA_SOURCE_DIR` when supplied. CUDA and HIP builds need their GPU toolchains and communication libraries, in addition to LibDDLA. Source preparation does not install compilers, MPI, math libraries or GPU toolchains.

(install-build)=
## Download, build and install

Clone the repository and enter its directory:

```sh
git clone https://github.com/AESM-Group/LibRPA
cd LibRPA
```

The current source layout does not need submodule initialization. For historical checkouts before `28b7431`, including `v0.4.0-gw-benchmark`, use the instructions for that revision.

### Intel oneAPI example

Initialize the compiler, MPI and MKL environment before configuration. The example below uses the LLVM wrappers `mpiicpx` and `mpiifx`:

```sh
cmake -S . -B build \
  -DCMAKE_CXX_COMPILER=mpiicpx \
  -DCMAKE_Fortran_COMPILER=mpiifx \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLIBRPA_ENABLE_TEST=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/librpa"
cmake --build build --parallel 2
cmake --build build --target install
```

This example explicitly selects shared libraries and disables LibRPA test targets. Without these settings, `BUILD_SHARED_LIBS` is off in a standalone build and `LIBRPA_ENABLE_TEST` is on. The driver is enabled by default. Add `-DLIBRPA_ENABLE_FORTRAN_BIND=ON` to build the Fortran binding.

When `MKLROOT` is nonempty, LibRPA uses its Intel MKL finder. A CMake `MKLROOT` value takes precedence over the environment variable. The finder supplies the BLAS, LAPACK and ScaLAPACK dependencies.

### GCC and OpenBLAS example

Use C++ and Fortran wrappers from the same MPI installation. Replace the library paths below with the paths for your Linux installation:

```sh
cmake -S . -B build-gcc \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DCMAKE_Fortran_COMPILER=mpifort \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DLIBRPA_ENABLE_TEST=OFF \
  -DMKLROOT= \
  -DBLAS_LIBRARIES=/path/to/openblas/lib/libopenblas.so \
  -DLAPACK_LIBRARIES=/path/to/openblas/lib/libopenblas.so \
  -DSCALAPACK_DIR=/path/to/scalapack \
  -DCMAKE_INSTALL_PREFIX="$HOME/opt/librpa"
cmake --build build-gcc --parallel 2
cmake --build build-gcc --target install
```

An empty `MKLROOT` selects the LAPACK and ScaLAPACK discovery path. The explicit library paths remove ambiguity when several math libraries are installed. Platform-specific examples are available in [Build Examples](../examples/build/index).

Use a separate build directory when you change the compiler or MPI installation. Compiler flags and backend options for ELPA or LibDDLA can also require a fresh child build.

### Build products and installation layout

The build directory contains:

| Product | Location |
| --- | --- |
| Driver, when enabled | `build/chi0_main.exe` |
| Shared library | `build/src/librpa.so` on Linux, `build/src/librpa.dylib` on macOS |
| Static library | `build/src/librpa.a` when `BUILD_SHARED_LIBS=OFF` |
| Fortran binding, when enabled | `build/binding/fortran/`, including `librpa_f03.mod` |

Installation puts the driver under `bin/`, libraries under `lib/` or `lib64/`, and LibRPA public headers under `include/librpa/`. These paths follow CMake's `GNUInstallDirs` settings. The default standalone prefix is `install/` under the LibRPA source directory if no prefix was supplied.

The Fortran binding additionally installs its module and stub source. LibRI, LibComm and cereal headers are not installed. LibRPA does not currently install a CMake package configuration for `find_package(librpa)`. Parent projects can instead use the source-tree target `rpa_lib`.

The install target also runs install rules from managed GreenX. Managed ELPA stays private to its build directory. LibDDLA child builds install their library into the LibRPA installation. Shared-library users need the selected MPI, compiler and math runtimes at execution time.

### Optional tests

Enable LibRPA tests in the configured build:

```sh
cmake -S . -B build -DLIBRPA_ENABLE_TEST=ON -DENABLE_GREENX_CTEST=OFF
cmake --build build --parallel 2
(cd build && OMP_NUM_THREADS=1 ctest --output-on-failure)
```

Several tests launch multiple MPI ranks. Run them where the necessary CPU allocation is available. `ENABLE_GREENX_CTEST=OFF` disables GreenX's test registration, not LibRPA's tests.

(install-offline)=
## Offline installation

Use the same LibRPA manifest and patches on the preparation machine and the build machine. On a connected machine, run:

```sh
cmake -DLIBRPA_THIRDPARTY_DIR=/path/to/deps \
  -P cmake/LibRPAPrepareThirdparty.cmake
```

This prepares all six components without compiling them. To prepare only the default CPU dependencies, use:

```sh
cmake -DLIBRPA_THIRDPARTY_DIR=/path/to/deps \
  '-DCOMPONENTS=LibRI;LibComm;cereal;greenX' \
  -P cmake/LibRPAPrepareThirdparty.cmake
```

Add `ELPA` or `LibDDLA` to that list when the corresponding source build is needed. Component names are case-sensitive. `COMPONENTS` selects preparation-script components only. It does not change the build options.

Copy `/path/to/deps` to the offline machine, preserving symbolic links and file contents. For example:

```sh
rsync -a /path/to/deps/ build-host:/path/to/deps/
```

Configure from the LibRPA source directory on that machine:

```sh
cmake -S . -B build \
  -DLIBRPA_THIRDPARTY_DIR=/path/to/deps \
  -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF
cmake --build build --parallel 2
```

Add your compiler, math-library and installation options to this configuration command. Disabling downloads does not disable extraction, patch application or hash verification.

Alternatively, copy only the manifest's archives into `LIBRPA_THIRDPARTY_ARCHIVE_DIR` (default `/path/to/deps/downloads/`). Use the exact `ARCHIVE` filenames from `thirdparty/manifest.cmake`. CMake then prepares missing trees locally. This method also needs Git or `patch` for components with patches.

### Mirrors and shared storage

Use semicolon-separated URL prefixes to select mirrors:

```sh
cmake -S . -B build \
  -DLIBRPA_THIRDPARTY_DIR=/path/to/deps \
  '-DLIBRPA_THIRDPARTY_MIRRORS=https://mirror.example.org/librpa;file:///srv/librpa-archives'
```

Downloads stop after 60 seconds without transfer activity. There is no total transfer time limit. Failed attempts remove their temporary files.

Each mirror serves the archive filenames from the manifest. CMake tries the prefixes in order, then the upstream URL. Mirrors are used only when an archive is missing and downloads are enabled. Existing archives are not replaced by a change to the mirror setting.

`LIBRPA_THIRDPARTY_DIR` stores verified trees in `src/<Name>-<hash12>/`. Archives default to `downloads/` within that directory. Use a shared archive directory to avoid downloads across LibRPA checkouts:

```sh
export LIBRPA_THIRDPARTY_ARCHIVE_DIR=~/downloads/LibRPA_tarballs
cmake -P cmake/LibRPAPrepareThirdparty.cmake
```

A nonempty CMake value overrides this environment variable. When selected during configuration, the environment path is saved in the cache for later configurations. Standalone `cmake -P` preparation does not write a persistent cache. Local archives in the selected directory also work with downloads disabled. Build outputs remain in the build directory. Multiple build directories can share this source cache. A hash change selects a new directory and leaves earlier trees in place.

(install-dependency-sources)=
## Choosing dependency sources

For compiled dependencies (greenX, ELPA and LibDDLA), choose a source in this order of preference:

1. Managed sources. Do not set a target, source path or installation prefix. LibRPA uses the pinned version and patches, verifies the tree hash, and builds it with its own compilers and build settings. This is the recommended choice for standalone builds.
2. A parent project's CMake target. Use this when LibRPA is part of a larger project. The parent controls the toolchain and link dependencies.
3. An external source path. Use this to develop or test a dependency. LibRPA builds it with its own compilers and build settings.
4. A prebuilt installation prefix, only when the other choices are not possible. Prebuilt libraries can have incompatible compilers, flags, C++ ABIs, BLAS/LAPACK libraries or Fortran modules. Each configuration warns about these risks.

For ELPA, managed sources require `LIBRPA_USE_BUNDLED_ELPA=ON`, because ELPA is disabled by default.

When several sources are supplied, LibRPA selects the first available one in this order: parent target, source path, installation prefix, managed sources. Managed sources are used only when nothing else is supplied. This selection order is not a recommendation.

| Dependency | Parent target | Source path | Installation prefix |
| --- | --- | --- | --- |
| greenX | `LibGXMiniMax` or `greenX::LibGXMiniMax` | `GREENX_SOURCE_DIR` | `EXTERNAL_GREENX_DIR` |
| ELPA | `ELPA::ELPA` | `ELPA_SOURCE_DIR` | `EXTERNAL_ELPA_DIR` |
| LibDDLA | `LibDDLA::ddla` | `LIBDDLA_SOURCE_DIR` | `LIBDDLA_PATH` |

Do not supply both a source path and an installation prefix for the same dependency. Invalid explicit paths cause an error. For ELPA, enable `LIBRPA_USE_EXTERNAL_ELPA` for a target or installation, or `LIBRPA_USE_BUNDLED_ELPA` for a source build. ELPA is disabled by default. An ELPA path with the wrong mode enabled is an error. With both options OFF, either path produces a warning that names the required option.

Whole-tree source paths follow the same hash rules as header paths: `pinned`, `overlay` for a matching pristine tree with patches, or `unpinned`. Overlays preserve the original source tree and symbolic links. LibDDLA has no patches.

(install-local-dependencies)=
## Local dependencies and parent projects

For LibRI, LibComm and cereal, the selection order is: existing parent target, explicit include directory, then managed headers. A nonempty CMake path variable takes precedence over its environment variable. An environment path selected for an empty cache entry is saved there, with a status message. Later configurations keep the saved path even when the environment variable is absent.

```sh
cmake -S . -B build \
  -DLIBRI_INCLUDE_DIR=/path/to/LibRI/include \
  -DLIBCOMM_INCLUDE_DIR=/path/to/LibComm/include
```

In this example, cereal remains managed unless its target or path is also supplied. Invalid explicit paths cause an error without a download fallback. A supplied parent target takes precedence over the corresponding path.

| Header source | Result |
| --- | --- |
| Headers match the pinned, patched tree | Used directly, status `pinned` |
| LibRI headers match the pinned pristine tree | Copied, patched and verified in managed storage, status `overlay` |
| Other header contents | Used with a warning, status `unpinned` |
| Existing parent target | Used without patches or tree-hash checks, status `target` |

The user directory is never patched in place. LibComm and cereal currently have no patches, so their matching upstream headers are `pinned`, not `overlay`. A required compile probe checks the three selected header libraries together. Passing this probe does not establish complete numerical compatibility for another revision.

Parent projects can provide `LibRI::LibRI`, `LibComm::LibComm` and `cereal::cereal` before adding LibRPA. See {ref}`Parent-project integration <thirdparty-parent-projects>` for a complete example.

For GPU builds, LibDDLA resolves `LibDDLA::ddla`, then `LIBDDLA_SOURCE_DIR` or `LIBDDLA_PATH`, then managed sources. Supplying both paths is an error. Source builds use a separate CMake build. LibRPA does not patch a user-supplied LibDDLA source tree.

`LIBRPA_USE_EXTERNAL_GREENX=ON` requires a parent target, `GREENX_SOURCE_DIR` or `EXTERNAL_GREENX_DIR`. Parent greenX targets are also selected when this option is OFF.

See [Compile Options](compile_options.md) for defaults, accepted values and precedence rules.

## Troubleshooting

### A dependency is missing or a download fails

The error identifies the component, archive filename and upstream URL. Prepare the dependencies on a connected machine, or place the required archive in the effective archive directory. A local `file://` mirror still requires `LIBRPA_THIRDPARTY_DOWNLOAD=ON`. With downloads disabled, place the archive directly in that directory instead.

Downloads use CMake's built-in libcurl. Behind a proxy, set `https_proxy` (or `ALL_PROXY`) in the environment before preparation or configuration. If TLS verification fails because the system lacks the required CA certificates, pass a certificate bundle with `-DCMAKE_TLS_CAINFO=/path/to/ca-bundle.crt`. This also works with `cmake -P`. LibRPA does not disable TLS verification. Alternatively, prepare the archives on another machine.

### A managed tree fails verification

Every configuration verifies each selected managed tree, and there is no option to skip this check. Preserve any local edits before removing the exact directory named in the error. Reconfigure to recreate it from the cached archive. Do not change the manifest hash merely to accept a local edit. Use an explicit local path for dependency development.

If a newly extracted tree fails, check that the archive matches the manifest and the patches match the LibRPA checkout. Replace an incorrect or damaged cached archive before trying again. Tree verification covers files in the declared hash scope, not archive timestamps or compressed bytes.

Added files in that scope also change the hash. For example, a `.DS_Store` file that macOS Finder creates in a managed header directory stops configuration. Remove the added file, or remove the tree and reconfigure.

### Removing old dependency trees

A manifest update creates new `src/<Name>-<hash12>/` directories and keeps the old ones. Configuration uses and verifies only the trees that the current manifest selects. You can remove other `src/<Name>-<hash12>/` directories and archives that the current manifest does not name. Do not remove trees that another LibRPA checkout uses through the same `LIBRPA_THIRDPARTY_DIR`. `src/.tmp-*` directories and `downloads/*.tmp-*` files remain only after an interrupted preparation. Remove them when no configuration or preparation is running.

### An explicit path or the LibRI compile probe fails

The paths must contain `RI/ri/LRI.h`, `Comm/Comm_Tools.h` or `cereal/cereal.hpp`, as appropriate. They name include directories, not repository roots. To return to managed headers, remove the relevant CMake cache entry and unset the corresponding environment variable.

```sh
unset LIBRI_INCLUDE_DIR
cmake -S . -B build -U LIBRI_INCLUDE_DIR
```

Repeat this for other explicit paths as needed. A parent target still takes precedence. For probe failures, inspect the dependency paths and statuses in the CMake error. Check the compiler diagnostics in the build directory's `CMakeFiles/` logs.

### `std::filesystem` link errors with Intel compilers

Intel compilers on Linux use GCC's C++ standard library. An old system libstdc++ can cause unresolved `std::filesystem` symbols. Load the compatible GCC runtime for your Intel installation before configuring a fresh build directory. LibRPA probes both direct linking and linking with `stdc++fs`.

### ELPA `<complex.h>` macro conflict

Older external ELPA headers can define the macro `I` through `<complex.h>`. This conflicts with C++ variable names in LibRPA. Use ELPA headers with a C++ guard that includes `<complex>` for C++ and `<complex.h>` for C. Reconfigure after replacing an affected external installation. The managed ELPA release already has that guard.
