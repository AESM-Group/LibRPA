# Maintaining third-party dependencies

The managed LibDDLA dependency is the unmodified 0.0.7 release at commit
`0a65e570d3d3bc70e4d53a7e992b2de60abc884c`. Its archive and extracted tree
are hash checked. The historical LibDDLA consumer backports are not applied.
External packages must expose the native mixed-GEMM interfaces checked by the
compile/link probe; caller-owned mixed-GEMM workspace is not required.

The tracked `thirdparty/manifest.cmake` declares each component's version, URL, archive filename, patches, license, archive SHA-256 and reference tree hashes.

`ARCHIVE_SHA256` covers the compressed archive bytes and is checked after download and before extraction. A mismatch stops preparation. Failed downloads are removed; existing cached archives are preserved. Regenerated upstream archives can have different bytes even when their extracted files match, so a hash update needs review.

Tree hashes identify source versions and detect local changes during configuration or source preparation. A mismatch produces a warning and status `unpinned`; the sources remain in use. Header-only libraries are hashed over `include/`; LibDDLA, ELPA and greenX are hashed over the whole extracted tree.

## Tree hash definition

`librpa_thirdparty_tree_sha256(out root)` recursively lists files relative to `root` with CMake `file(GLOB_RECURSE ... LIST_DIRECTORIES false)`, sorts the paths, hashes each file with SHA-256, and hashes the concatenation of lines:

```text
<lowercase file SHA256><two spaces><relative path><newline>
```

The hash includes file contents and relative paths, but excludes file modes, timestamps and directory symlinks. Expanding directory symlinks changes the file list. For user-supplied trees, `IGNORE_GIT` also excludes the root `.git` file or directory; nested `.git` entries remain included.

`TREE_SHA256` covers patched content. `PRISTINE_SHA256` covers unpatched content and defaults to `TREE_SHA256` when omitted. The CMake helper defines both hashes:

```cmake
cmake_minimum_required(VERSION 3.13)
include("/path/to/LibRPA/cmake/LibRPAThirdparty.cmake")
librpa_thirdparty_tree_sha256(hash "/path/to/tree/include")
message("${hash}")
```

Run this script with `cmake -P hash.cmake`. For whole-tree hashes, use the source root instead of `include/`. Add `IGNORE_GIT` as the third argument to match the check of a user-supplied tree.

## Local patch for upstream

Local third-party patches are intended for small, focused fixes and integration changes.
They are appropriate in either of these cases:

- We do not directly control the upstream project.
- We maintain the upstream project, but the effect on its other users and configurations still needs verification.

For dependencies that we maintain, the preferred approach is to make the change upstream and then update the pinned version in LibRPA.
A local patch can serve as a temporary solution while its wider effects are verified.
Large features, API changes and extensive implementation changes belong upstream where possible.

The patch header and its entry in `thirdparty/patches/README.md` record the upstream source, version, purpose, license, and upstream submission status.
They also explain why a local patch is needed.

When updating an upstream component (see below), we should remove any local patch whose changes are already included in the new version.

## Updating a component

1. Download the selected upstream archive into a scratch directory. Record its SHA-256 with `shasum -a 256 ARCHIVE`. Extract it and check the version and license, preserving symbolic links and original line endings.
2. Compare it with the currently selected tree. Keep only required portability and integration changes; do not carry unrelated formatting or generated files.
3. Generate any required `-p1` patches with `a/` and `b/` paths relative to the archive root, following the scope and documentation requirements above.
4. Compute the pristine tree hash with the CMake helper above, then apply the patches and compute the patched tree hash.
5. Update the declaration's full version, URL, archive name, patch list and hashes. Update the license list in `docs/develop/licensing.md`, then run `python3 utilities/update_readme_licensing_list.py`.
6. Prepare the updated component in a fresh LibRPA checkout with `cmake -DCOMPONENTS=NAME -P cmake/LibRPAPrepareThirdparty.cmake`, replacing `NAME` with the component name.
7. Run `python3 utilities/test_thirdparty.py`. It covers archive handling, source reuse, overlays, path selection and concurrent preparation.
8. Build configurations that use the updated component and run the relevant unit and numerical regression tests. If dependency selection changes, also test the affected parent targets and explicit paths.

Managed trees can be edited during development; configuration preserves local changes and reports them as `unpinned`. A separate checkout with an explicit source path allows Git tracking. Distributed updates belong in the manifest and patches.

## Dependency source preparation

`cmake/LibRPADownloadThirdparty.cmake` downloads and verifies archives, even when prepared sources already exist.

`cmake/LibRPAPrepareThirdparty.cmake` reuses existing sources or extracts and patches archives, downloading missing archives when needed. Both scripts accept `COMPONENTS` to select dependencies; the default is all manifest components. Commands for offline use are in {ref}`Offline installation <install-offline>`.

Managed sources and patched copies use `thirdparty/src/` in the LibRPA checkout. Archives default to `thirdparty/downloads/`, or the directory selected by {ref}`LIBRPA_THIRDPARTY_ARCHIVE_DIR <librpa-thirdparty-archive-dir>`. Downloads use temporary files before renaming. Existing archives are preserved if preparation fails.

For pristine supplied sources that need patches, LibRPA prepares or reuses a patched copy in `thirdparty/src/`. New copies preserve symbolic links and omit the root `.git` entry. LibRPA verifies the copy's hash before applying patches. The original tree remains unchanged.

Read-only pristine trees are supported: the copy becomes owner-writable and retains execute permissions. The checkout's `thirdparty/src/` directory must be writable.

Each preparation uses a separate temporary directory, then renames the prepared tree to `src/<Name>-<hash12>`. Concurrent configurations reuse a tree published by another process. The directory name comes from the manifest's tree hash; local edits do not rename it. CI installs greenX's Python package from a scratch copy of its `python/` directory.

(thirdparty-parent-projects)=
## Parent-project integration

A parent can let LibRPA prepare dependencies in `LibRPA/thirdparty/src/`. Set an archive directory before adding LibRPA to share downloads with other checkouts:

```cmake
cmake_minimum_required(VERSION 3.18)
project(HostCode LANGUAGES CXX Fortran)

set(LIBRPA_THIRDPARTY_ARCHIVE_DIR "$ENV{HOME}/Downloads"
    CACHE PATH "Shared dependency archives")
set(LIBRPA_ENABLE_DRIVER OFF CACHE BOOL "Build the LibRPA driver")
set(LIBRPA_ENABLE_TEST OFF CACHE BOOL "Build LibRPA tests")
set(ENABLE_GREENX_CTEST OFF CACHE BOOL "Register GreenX tests")
add_subdirectory(LibRPA)

add_executable(host main.cpp)
target_link_libraries(host PRIVATE rpa_lib)
```

Source-built greenX keeps the parent installation prefix, including the default prefix. It does not install into the managed source tree.

This example assumes that `LibRPA/` and `main.cpp` are in the parent source directory. Configure the parent with its compiler, MPI and math-library settings. LibRPA's `.gitignore` excludes `thirdparty/downloads/` and `thirdparty/src/`. An existing cache value takes precedence over these `set(... CACHE ...)` defaults.

### Header targets

To supply header dependencies, define their targets before `add_subdirectory(LibRPA)`. For example:

```cmake
add_library(LibRI::LibRI INTERFACE IMPORTED)
set_target_properties(LibRI::LibRI PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "/path/to/compatible/LibRI/include")
add_library(LibComm::LibComm INTERFACE IMPORTED)
set_target_properties(LibComm::LibComm PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "/path/to/LibComm/include")
add_library(cereal::cereal INTERFACE IMPORTED)
set_target_properties(cereal::cereal PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "/path/to/cereal/include")
```

Each target takes precedence over its path variable and managed source. Dependencies without a parent target use their paths or managed sources.

Header targets have status `target`. LibRPA does not hash or patch them, so the parent supplies compatible headers with any required fixes. A path variable allows LibRPA to apply an overlay to pristine pinned LibComm headers.

With `LIBRPA_USE_LIBRI=ON`, a configure-time probe compiles the selected header combination. It reads each target's direct `INTERFACE_INCLUDE_DIRECTORIES`, without transitive includes or target compile definitions. Absolute paths and simple `BUILD_INTERFACE` expressions are supported. `INSTALL_INTERFACE` entries are ignored; other generator expressions cause an error.

A failed probe stops configuration. Target-provided LibComm uses the automatic `MPI_THREAD_MULTIPLE` selection. See {ref}`Compile Options <librpa-mpi-thread-level>`.

### Compiled dependency targets

| Parent target | Selection requirement | Parent responsibilities |
| --- | --- | --- |
| `ddla::ddla` (or `LibDDLA::ddla`) | Enable CUDA or HIP | Supply a library, public include paths and link dependencies compatible with that backend |
| `ELPA::ELPA` | Detected automatically | Supply the library, headers and required link dependencies |
| `LibGXMiniMax` or `greenX::LibGXMiniMax` | Detected automatically | Supply link dependencies and compatible modules: `Fortran_MODULE_DIRECTORY` for source builds, interface include directories for imported targets |

Define these targets before adding LibRPA. An ELPA target automatically enables `LIBRPA_USE_ELPA`. GreenX targets are detected automatically, so LibRPA shares the parent target. Without a target or explicit path, it uses managed sources.

In a parent project, prefer a target when the parent already builds the dependency; otherwise let LibRPA use managed sources. See {ref}`Choosing dependency sources <install-dependency-sources>` for the general recommendation and the path rules.

Use `rpa_lib` to link a C or C++ consumer. With `LIBRPA_ENABLE_FORTRAN_BIND=ON`, the Fortran target is `rpa_f03_lib`, which also exposes its build-time module directory. LibRPA does not currently install a package configuration or target export for `find_package(librpa)`.

## Migration from vendored sources

Existing `LIBRI_INCLUDE_DIR`, `LIBCOMM_INCLUDE_DIR`, `CEREAL_INCLUDE_DIR` and `LIBDDLA_PATH` settings remain supported. An environment value is now saved in the cache when the cache entry is empty, so a selection persists after the variable is unset. See {ref}`Compile Options <librpa-how-to-set-options>` for how to clear a selection.

The old `thirdparty/LibRI`, `thirdparty/LibComm`, `thirdparty/cereal`, `thirdparty/LibDDLA` and `thirdparty/greenX` source paths no longer exist. Replace scripts that copy those directories with the preparation script or explicit external paths. Do not hard-code `thirdparty/src/<Name>-<hash12>` paths: a manifest update changes them. For offline builds, transfer the archives and prepare the sources in the destination checkout.

`LIBRPA_USE_ELPA=ON` builds managed ELPA sources when no dependency is supplied. An ELPA target, source path or installation prefix automatically enables it. See {ref}`Compile Options <librpa-use-elpa>` for the selection rules and deprecated switches. The tracked `thirdparty/ELPA` directory contains only LibRPA's build wrapper. A dependency cache contains sources and archives, not compiler-specific installed libraries. Each build directory creates its own compiled dependencies.
