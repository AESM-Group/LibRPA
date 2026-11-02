# Installation

LibRPA prepares its source dependencies during CMake configuration. The repository contains their versions, URLs, patches and hashes, but not their source trees. A first configuration needs network access or a prepared dependency directory.

(install-build-requirements)=
## Build requirements

The CPU build requires:

- A C++17 compiler with a working `std::filesystem` implementation.
- A Fortran compiler, including when the Fortran binding is disabled. GreenX contains Fortran code.
- MPI and OpenMP development libraries compatible with the selected compilers.
- BLAS, LAPACK and ScaLAPACK libraries compatible with that MPI installation.
- CMake and a build tool supported by the selected CMake generator.
- Git or `patch` when source preparation needs to apply patches. Git has precedence when both tools are available.

### CMake version

LibRPA requires CMake **3.18 or newer**. Its own top-level build code uses `FindCUDAToolkit` (introduced in 3.17) and policy `CMP0104` (introduced in 3.18). These calls determine LibRPA's minimum version independently of third-party requirements.

Dependencies and compiler toolchains can have additional requirements. Their build configurations check those requirements separately. A subdirectory's minimum-version declaration checks the running CMake and sets policies in that directory. It does not raise the parent's policy version.

The standalone source-preparation script requires only CMake 3.13. It downloads, patches and verifies sources without configuring or compiling the dependency projects.

### Source dependencies

| Component | When LibRPA needs it | Default source |
| --- | --- | --- |
| GreenX | Every build | Managed source, unless `LIBRPA_USE_EXTERNAL_GREENX=ON` |
| LibRI, LibComm, cereal | `LIBRPA_USE_LIBRI=ON`, the default | Managed headers, unless the parent provides targets or the user supplies include paths |
| LibDDLA | `LIBRPA_USE_CUDA=ON` or `LIBRPA_USE_HIP=ON` | Managed source, unless a target, installation or local source tree is supplied |
| ELPA | Either ELPA option is enabled | Managed source with `LIBRPA_USE_BUNDLED_ELPA=ON`, or an external target/installation with `LIBRPA_USE_EXTERNAL_ELPA=ON` |

*GW* and EXX require LibRI. RPA correlation energies can also be computed with `LIBRPA_USE_LIBRI=OFF`. GreenX remains required in that configuration.

Managed ELPA also needs a C compiler, `make`, Bash and a C preprocessor. CUDA and HIP builds need their GPU toolchains and communication libraries, in addition to LibDDLA. Source preparation does not install compilers, MPI, math libraries or GPU toolchains.

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

Alternatively, copy only the manifest's archives into `/path/to/deps/downloads/`. Use the exact `ARCHIVE` filenames from `thirdparty/manifest.cmake`. CMake then prepares missing trees locally. This method also needs Git or `patch` for components with patches.

### Mirrors and shared storage

Use semicolon-separated URL prefixes to select mirrors:

```sh
cmake -S . -B build \
  -DLIBRPA_THIRDPARTY_DIR=/path/to/deps \
  '-DLIBRPA_THIRDPARTY_MIRRORS=https://mirror.example.org/librpa;file:///srv/librpa-archives'
```

Each mirror serves the archive filenames from the manifest. CMake tries the prefixes in order, then the upstream URL. Mirrors are used only when an archive is missing and downloads are enabled. Existing archives are not replaced by a change to the mirror setting.

`LIBRPA_THIRDPARTY_DIR` stores archives in `downloads/` and verified trees in `src/<Name>-<hash12>/`. Build outputs remain in the build directory. Multiple build directories can share this source cache. A hash change selects a new directory and leaves earlier trees in place.

(install-local-dependencies)=
## Local dependencies and parent projects

For LibRI, LibComm and cereal, the selection order is: existing parent target, explicit include directory, then managed headers. A nonempty CMake path variable takes precedence over its environment variable.

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

For GPU builds, LibDDLA resolves `LibDDLA::ddla`, `LIBDDLA_PATH`, `LIBDDLA_SOURCE_DIR`, then managed sources. Source builds use a separate CMake build. LibRPA does not patch a user-supplied LibDDLA source tree.

ELPA and GreenX have explicit external-mode switches. `LIBRPA_USE_EXTERNAL_ELPA=ON` selects `ELPA::ELPA` or installation discovery. `LIBRPA_USE_EXTERNAL_GREENX=ON` requires a parent-provided `LibGXMiniMax` target. Those flags do not specify a source directory to download or build.

See [Compile Options](compile_options.md) for defaults, accepted values and precedence rules.

## Troubleshooting

### A dependency is missing or a download fails

The error identifies the component, archive filename and upstream URL. Prepare the dependencies on a connected machine, or place the required archive in `downloads/`. A local `file://` mirror still requires `LIBRPA_THIRDPARTY_DOWNLOAD=ON`. With downloads disabled, place the archive directly in `downloads/` instead.

### A managed tree fails verification

Preserve any local edits before removing the exact directory named in the error. Reconfigure to recreate it from the cached archive. Do not change the manifest hash merely to accept a local edit. Use an explicit local path for dependency development.

If a newly extracted tree fails, check that the archive matches the manifest and the patches match the LibRPA checkout. Replace an incorrect or damaged cached archive before trying again. Tree verification covers files in the declared hash scope, not archive timestamps or compressed bytes.

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
