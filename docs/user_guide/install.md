# Installation

(install-build-requirements)=
## General requirements

To build LibRPA, you need:

- A C++17 compiler with a working `std::filesystem` implementation, and a Fortran compiler.
- MPI and OpenMP development libraries compatible with these compilers.
- BLAS, LAPACK and ScaLAPACK libraries compatible with that MPI installation.
- CMake 3.18 or newer, and a build tool such as `make` or Ninja.
- Git or `patch`.

GPU builds also need the CUDA toolkit and NCCL, or ROCm and RCCL.

## Third-party dependencies

LibRPA uses the third-party libraries below:

| Library | Needed when |
| --- | --- |
| GreenX | Always |
| LibRI, LibComm, cereal | `LIBRPA_USE_LIBRI=ON` (the default) |
| ELPA | `LIBRPA_USE_ELPA=ON` or a supplied ELPA target or path |
| LibDDLA | `LIBRPA_USE_CUDA=ON` or `LIBRPA_USE_HIP=ON` |

You do not have to install them beforehand: CMake downloads and prepares the ones that the selected options need.
In case your machine does not have network acess or you would like to use your own copies of these libraries,
please refer to {ref}`Third-party library details <install-thirdparty-details>`.

## Overview of build procedure

1. Get the LibRPA source code.
2. Configure the build with CMake. CMake also prepares the third-party libraries in this step when needed.
3. Build LibRPA on the target platform.
4. Install LibRPA, if necessary.

## Preparation: get the code

You can download a source archive from the [releases page](https://github.com/AESM-Group/LibRPA/releases) and extract it on your machine.
For the development version, clone the repository:

```sh
git clone https://github.com/AESM-Group/LibRPA
```

By default, CMake prepares the third-party libraries during configuration.
It downloads the versions that LibRPA uses, checks the archives and applies LibRPA's patches.
For an offline machine, download the archives on a connected machine and transfer them before the build.
See {ref}`Offline installation <install-offline>` for this procedure.
To experiment with a third-party component, you can also supply the path to your own copy.

(install-build)=
## Build

Once the compilers and libraries listed above are available in your environment, enter the LibRPA source directory.
You can then configure, build and install LibRPA with:

```sh
cmake -B build -DCMAKE_INSTALL_PREFIX="$HOME/opt/librpa"
cmake --build build -j 2
# Install public APIs and libraries to $HOME/opt/librpa
cmake --build build --target install
```

CMake searches your environment for the required components.
You can select a compiler toolchain, such as Intel oneAPI or GCC with OpenBLAS, through CMake built-in options like `-DCMAKE_CXX_COMPILER`.
LibRPA also provides its own options to select the features to build.

For examples with different toolchains and platforms, please refer to [Build Examples](../examples/build/index).
For all build options, see [Compile Options](compile_options.md).

## Build products

The build produces the following files. Installation paths are relative to `CMAKE_INSTALL_PREFIX` (`$HOME/opt/librpa` in the example above).

| Product | Location in the build or source tree | Installation directory |
| --- | --- | --- |
| Driver | `build/chi0_main.exe` | `bin/` |
| LibRPA library | `build/src/librpa.*` | `lib/` or `lib64/` |
| Public C and C++ headers | `include/librpa*.h` and `include/librpa.hpp` | `include/librpa/` |
| Optional Fortran binding library and module | `build/binding/fortran/librpa_f03.*` | Library: `lib/` or `lib64/`; module: `include/librpa/` |

Fortran binding is built when `-DLIBRPA_ENABLE_FORTRAN_BIND=ON`.
When you change the compiler or MPI installation, make sure you use a new build directory to avoid potential incompatibility issue.

## Run tests

To run the tests, configure with `-DLIBRPA_ENABLE_TEST=ON`, build, and run `ctest`:

```sh
cmake -B build -DLIBRPA_ENABLE_TEST=ON -DENABLE_GREENX_CTEST=OFF
cmake --build build -j 2
cd build && ctest --output-on-failure
```

This runs the LibRPA unit tests.
Some tests use several MPI processes (at most 4).

For regression tests and comparisons with reference results, see the [regression test guide](https://github.com/AESM-Group/LibRPA/blob/master/regression_tests/README.md#run-automatic-regression-test-and-analysis).

(install-thirdparty-details)=
## Third-party library details

You can use the sources that LibRPA prepares, supply your own copies, or share dependencies with a parent CMake project.

(install-offline)=
### Offline installation

You can download the dependency archives on a connected machine and upload them to the machine where you will build LibRPA.
Use the same LibRPA source version on both machines so that the archive names, hashes and patches match.
From the LibRPA source directory on the connected machine, run:

```sh
cmake -DLIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads" \
  -P cmake/LibRPADownloadThirdparty.cmake
```

This command saves the archives in `~/Downloads` and verifies their SHA-256 hashes. Extraction and patching take place on the build machine.
To download only the default CPU dependencies, use:

```sh
cmake -DLIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads" \
  '-DCOMPONENTS=LibRI;LibComm;cereal;greenX' \
  -P cmake/LibRPADownloadThirdparty.cmake
```

You can add `ELPA` or `LibDDLA` to this list for builds that use them.
`COMPONENTS` selects which archives the script downloads and verifies. Use the component names with the capitalization shown above.

Upload the downloaded archives to the build machine, keeping the `ARCHIVE` filenames listed in `thirdparty/manifest.cmake`.
You can choose the archive location independently on each machine:

| Archive location on the build machine | Configuration |
| --- | --- |
| `LibRPA/thirdparty/downloads/` | Use the default archive directory |
| `~/Downloads` or another directory | Set `LIBRPA_THIRDPARTY_ARCHIVE_DIR` to that directory |

For archives in `LibRPA/thirdparty/downloads/`, run from the LibRPA source directory:

```sh
cmake -B build -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF
cmake --build build --parallel 2
```

For archives in the build machine's `~/Downloads`, use:

```sh
cmake -B build \
  -DLIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads" \
  -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF
cmake --build build --parallel 2
```

Both configurations use local archives with downloads disabled.
CMake checks the archives, extracts them and applies patches in the build machine's `LibRPA/thirdparty/src/` directory.
Git or `patch` is needed on that machine to apply the patches.
You can add your compiler, math-library and installation options to either configuration command.

### Downloads and shared storage

CMake downloads each component from the upstream URL in `thirdparty/manifest.cmake` and reuses archives already in the cache.

Downloads can continue for as long as data arrives. An attempt stops after 60 seconds without transfer activity, and CMake removes its temporary file.

Each LibRPA checkout stores its prepared sources in `thirdparty/src/<Name>-<hash12>/` and uses `thirdparty/downloads/` as the default archive directory.
Both directories are ignored by Git. You can share downloaded archives across checkouts with:

```sh
export LIBRPA_THIRDPARTY_ARCHIVE_DIR="$HOME/Downloads"
cmake -P cmake/LibRPADownloadThirdparty.cmake
```

You can set `LIBRPA_THIRDPARTY_ARCHIVE_DIR` through either the environment or a CMake option. A nonempty CMake value takes precedence.
Configuration saves a selected environment path in the CMake cache for later use. With `cmake -P`, the path applies to that script run.
The archives in this directory are also available for offline builds.

Build directories for the same LibRPA checkout share its prepared sources, while each keeps its own build outputs.
A change to a component's reference hash selects a new source directory, alongside earlier versions.

(install-dependency-sources)=
### Choosing dependency sources

For compiled dependencies (greenX, ELPA and LibDDLA), the following options suit different build setups:

- **Managed sources** are the recommended starting point for standalone builds. CMake downloads the selected versions, checks the archives and applies LibRPA's patches. It builds these sources with the compilers and settings selected for LibRPA.
- **A parent project's CMake target** lets LibRPA share a dependency that the parent already builds. The parent supplies its build settings and link dependencies.
- **An external source path** lets you develop or test your own copy of a dependency. LibRPA builds it with the same compilers as the rest of the project.
- **A prebuilt installation** lets you reuse an installed library. Source builds are generally easier to match to LibRPA's toolchain. For a prebuilt library, choose one with compatible compilers, flags, C++ ABI, math libraries and Fortran modules. CMake displays a reminder about these requirements.

To use managed ELPA sources, set `LIBRPA_USE_ELPA=ON`.

CMake gives supplied dependencies precedence over managed sources. It first looks for a parent target, then an explicit path.
For a path, choose either a source directory or an installation prefix. The table below lists the targets and options for each dependency.

| Dependency | Parent target | Source path | Installation prefix |
| --- | --- | --- | --- |
| greenX | `LibGXMiniMax` or `greenX::LibGXMiniMax` | `GREENX_SOURCE_DIR` | `GREENX_PATH` |
| ELPA | `ELPA::ELPA` | `ELPA_SOURCE_DIR` | `ELPA_PATH` |
| LibDDLA | `LibDDLA::ddla` | `LIBDDLA_SOURCE_DIR` | `LIBDDLA_PATH` |

ELPA is optional and disabled by default. Set `LIBRPA_USE_ELPA=ON` to download and build the managed release. A parent `ELPA::ELPA` target, `ELPA_SOURCE_DIR`, or `ELPA_PATH` automatically enables ELPA. `ELPA_ROOT` also selects an installation when `ELPA_PATH` is absent. A parent target takes precedence over the supplied path.

For a source directory, CMake compares its contents with the version recorded in the manifest.
It uses a matching patched tree directly (`pinned`) or creates a patched copy of a matching upstream tree (`overlay`).
Other versions remain available with status `unpinned`. The copy preserves the original sources and symbolic links. LibDDLA uses the supplied source directly.

(install-local-dependencies)=
### Local dependencies and parent projects

For LibRI, LibComm and cereal, you can supply a parent target or an include directory. CMake uses managed headers for the remaining libraries.
For example, to use local LibRI and LibComm headers:

```sh
cmake -B build \
  -DLIBRI_INCLUDE_DIR=/path/to/LibRI/include \
  -DLIBCOMM_INCLUDE_DIR=/path/to/LibComm/include
```

In this example, CMake prepares cereal automatically.
A parent target takes precedence over an include path, and a nonempty CMake path option takes precedence over its environment variable.
When configuration selects a path from the environment, it saves that path in the cache for later configurations.
CMake checks explicit paths before it prepares the headers.

| Header source | Result |
| --- | --- |
| Headers match the pinned, patched tree | Used directly, status `pinned` |
| LibRI headers match the pinned pristine tree | Copied and patched in managed storage, status `overlay` if unchanged |
| Other header contents | Used with a warning, status `unpinned` |
| Existing parent target | Headers supplied by the parent project, status `target` |

LibRPA applies patches to a copy in managed storage, keeping your source directory unchanged.
LibComm and cereal currently use their upstream headers directly, with status `pinned` when they match the manifest.
During configuration, a compile check tests the three header libraries together. For other revisions, regression tests can help assess numerical compatibility.

Parent projects can provide `LibRI::LibRI`, `LibComm::LibComm` and `cereal::cereal` before adding LibRPA. See {ref}`Parent-project integration <thirdparty-parent-projects>` for a complete example.

For GPU builds, you can supply `LibDDLA::ddla`, a source directory through `LIBDDLA_SOURCE_DIR`, or an installation through `LIBDDLA_PATH`.
CMake uses the parent target first, then the selected path, and otherwise prepares managed sources. LibDDLA source builds use a separate CMake build.

GreenX is always enabled. Supply a parent target, `GREENX_SOURCE_DIR` or `GREENX_PATH` to select your own copy. Otherwise, CMake prepares managed sources.

See [Compile Options](compile_options.md) for defaults, accepted values and precedence rules.

### How LibRPA builds and installs the libraries

| Library | Default source | Build and installation |
| --- | --- | --- |
| GreenX | Managed source | Built with LibRPA. The LibRPA install target also installs GreenX. |
| LibRI, LibComm, cereal | Managed headers | Used during compilation and kept in the source cache. |
| ELPA | Managed source with `LIBRPA_USE_ELPA=ON` | Built in a separate sub-build. It stays in the build directory. |
| LibDDLA | Managed source | Built in a separate sub-build and installed with LibRPA. |

Managed sources and patched copies are stored in `thirdparty/src/<Name>-<hash12>/` in the LibRPA checkout.
Explicit component paths select your own sources through the options described above.
Build outputs stay in the build directory, so several builds of the same checkout can use the same prepared sources.

To build ELPA from downloaded sources or `ELPA_SOURCE_DIR`, you also need a C compiler, `make`, Bash and a C preprocessor. The `LIBRPA_ELPA_*` build options configure these source builds; installed libraries and parent targets keep their own build settings.
For managed ELPA, choose a LibRPA checkout and build directory without spaces. An explicit `ELPA_SOURCE_DIR` also needs a path without spaces.
For ELPA or LibDDLA, a fresh build directory is useful when you change compiler flags or backend options.

For a shared-library build, load the MPI, compiler and math-library runtimes in the environment where you run LibRPA.
A parent CMake project can add LibRPA with `add_subdirectory` and link to `rpa_lib`.
See {ref}`Parent-project integration <thirdparty-parent-projects>` for the setup.

## Troubleshooting

### A dependency is missing or a download fails

The CMake message gives the component name, archive filename and upstream URL.
You can use these to obtain the archive and place it in `LIBRPA_THIRDPARTY_ARCHIVE_DIR`.
The {ref}`offline preparation procedure <install-offline>` also prepares the dependencies for transfer from another machine.

Downloads use CMake's built-in libcurl.
If your network uses a proxy, set `https_proxy` or `ALL_PROXY` before preparation or configuration.
For a TLS certificate error, you can supply your network's CA certificate bundle with `-DCMAKE_TLS_CAINFO=/path/to/ca-bundle.crt`.
This option also works with `cmake -P`.

### A managed tree differs from the reference version

You can edit sources under `thirdparty/src/<Name>-<hash12>/` and use them in your next build.
During configuration, CMake compares them with the reference hash in the manifest. It reports differences with a warning and status `unpinned`, then continues with your sources.
For ELPA and LibDDLA, use a fresh build directory after source edits so their separate builds compile the changes.

The comparison covers `include/` for header-only dependencies and the whole tree for compiled dependencies.
Added and removed files also affect this comparison, including `.DS_Store` files.
For modified LibComm headers, the automatic MPI thread level is `MPI_THREAD_MULTIPLE`. You can select a level explicitly with `LIBRPA_MPI_THREAD_LEVEL`.

To restore the reference sources, first save any edits you want to keep.
Then remove the affected source directory and reconfigure. LibRPA prepares a new copy from the cached archive.

### An archive fails verification

LibRPA checks the SHA-256 of each archive before extraction. This check includes local archives and covers the compressed file.
If a downloaded archive differs from the manifest, CMake reports the mismatch and removes the temporary download.
For an archive already in the cache, the error message identifies the file to replace.
Obtain the archive specified in the manifest, replace that file, and run configuration again.
Normal configuration and the preparation script reuse prepared sources independently of their archives. The download script checks the archives so they are ready for transfer.

### Removing old dependency trees

When a component's reference hash changes, CMake prepares it in a new `thirdparty/src/<Name>-<hash12>/` directory and keeps the earlier version.
You can remove an older source directory once the builds of that checkout have finished using it.
Shared archives can be removed once the checkouts that use them have finished with those versions.
Save any local edits you want to keep before removing a source directory.

After an interrupted preparation, temporary directories can remain under `thirdparty/src/.tmp-*`, and temporary downloads can remain as `*.tmp-*` in the selected archive directory.
You can remove these once configuration and preparation processes have finished.

### An explicit path or the LibRI compile probe fails

Set each path to the include directory that contains the corresponding header:

| Option | Header relative to that directory |
| --- | --- |
| `LIBRI_INCLUDE_DIR` | `RI/ri/LRI.h` |
| `LIBCOMM_INCLUDE_DIR` | `Comm/Comm_Tools.h` |
| `CEREAL_INCLUDE_DIR` | `cereal/cereal.hpp` |

To return to managed headers, clear both the environment variable and the CMake cache entry. For example:

```sh
unset LIBRI_INCLUDE_DIR
cmake -B build -U LIBRI_INCLUDE_DIR
```

You can clear the other include paths in the same way. In a parent project, CMake continues to use any supplied dependency targets.

If the header compile check fails, the CMake message lists the selected paths and their statuses.
The logs under the build directory's `CMakeFiles/` contain the compiler diagnostics.

### `std::filesystem` link errors with Intel compilers

Intel compilers on Linux use GCC's C++ standard library. An old system libstdc++ can cause unresolved `std::filesystem` symbols. Load the compatible GCC runtime for your Intel installation before configuring a fresh build directory. LibRPA probes both direct linking and linking with `stdc++fs`.

### ELPA `<complex.h>` macro conflict

Older external ELPA headers can define the macro `I` through `<complex.h>`. This conflicts with C++ variable names in LibRPA. Use ELPA headers with a C++ guard that includes `<complex>` for C++ and `<complex.h>` for C. Reconfigure after replacing an affected external installation. The managed ELPA release already has that guard.
