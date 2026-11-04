# Maintaining third-party dependencies

The tracked `thirdparty/manifest.cmake` declares each component's version, URL, archive filename, patches, license, archive SHA-256 and reference tree hashes.

`ARCHIVE_SHA256` covers the compressed archive bytes. Downloads must match before they enter the cache. Cached archives must match before extraction. A failed download check removes the temporary file and stops preparation. A cached archive mismatch stops preparation and leaves that archive unchanged. Regenerated upstream archives can have different bytes even when their extracted files match. Review such a change before updating `ARCHIVE_SHA256`.

Tree hashes identify source versions and detect local changes. A difference produces a warning and status `unpinned`, and configuration continues with those sources. Header-only libraries are hashed over `include/`; LibDDLA, ELPA and greenX are hashed over the whole extracted tree.

## Tree hash definition

`librpa_thirdparty_tree_sha256(out root)` recursively lists files relative to `root` with CMake `file(GLOB_RECURSE ... LIST_DIRECTORIES false)`, sorts the paths, hashes each file with SHA-256, and hashes the concatenation of lines:

```text
<lowercase file SHA256><two spaces><relative path><newline>
```

Directory names, file modes and timestamps are not included. Directory symlinks are explicitly excluded because GLOB_RECURSE can list them even with `LIST_DIRECTORIES false`. Preserve upstream symbolic links when extracting or copying archives; expanding directory links changes the file list. The equivalent shell check for a tree without symbolic links is:

```sh
(cd ROOT && find . -type f | sed 's|^\./||' | LC_ALL=C sort | while IFS= read -r f; do
  printf '%s  %s\n' "$(shasum -a 256 "$f" | cut -d' ' -f1)" "$f"; done) | shasum -a 256
```

Use the CMake helper as the authoritative definition for all trees, including those containing links. `TREE_SHA256` covers patched content. `PRISTINE_SHA256` covers unpatched content and defaults to `TREE_SHA256` when no patch is needed.

## Updating a component

1. Download the selected upstream archive into a fresh scratch directory. Record its SHA-256 with `shasum -a 256 ARCHIVE`, then extract it without running its scripts. Check the version and license. Preserve symbolic links and original line endings.
2. Compare it with the currently selected tree. Keep only required portability and integration changes; do not carry unrelated formatting or generated files.
3. Generate `-p1` patches with `a/` and `b/` paths relative to the archive root. Record the source, reason, upstream submission status and license in the patch header and `thirdparty/patches/README.md`.
4. Apply the patches to a fresh upstream tree. Compute pristine and patched hashes with `cmake/LibRPAThirdparty.cmake`, and independently cross-check a header tree with the shell command above.
5. Update the declaration's full version, URL, archive name, patch list and hashes. Update the license list in `docs/develop/licensing.md`, then run `python3 utilities/update_readme_licensing_list.py`.
6. Prepare sources in a fresh LibRPA checkout. Test upstream downloads, cached archives with downloads disabled, repeated configuration, archive hash rejection, and warnings that preserve modified managed trees. For LibRI, greenX and ELPA, also check pristine-path overlays and verify that the original tree is unchanged.
7. Run `python3 utilities/test_thirdparty.py` to check helper failure paths, overlays, concurrent preparation and directory-symlink handling.
8. Build with and without LibRI, run unit and numerical regression tests, and run the QSGW, Hartree and cRPA verification scripts used in CI. Check parent targets and explicit paths. For ELPA and greenX, verify the source hashes again after building.

A minimal hash script is:

```cmake
cmake_minimum_required(VERSION 3.13)
include("/path/to/LibRPA/cmake/LibRPAThirdparty.cmake")
librpa_thirdparty_tree_sha256(hash "/path/to/tree/include")
message("${hash}")
```

Run it with `cmake -P hash.cmake`. You can edit a managed tree during development. Configuration warns about differences in the declared hash scope and preserves the changes. This comparison does not need Git tracking. A separate Git checkout with an explicit source path is useful when you want to track and commit dependency changes. To distribute an update, record it in the manifest and patches.

## Preparation and concurrent builds

`cmake/LibRPADownloadThirdparty.cmake` downloads and verifies selected archives for transfer or shared storage. It populates the archive directory even when prepared sources already exist.

`cmake/LibRPAPrepareThirdparty.cmake` extracts and patches sources in the local checkout, downloading missing archives when needed. Like normal configuration, it reuses existing sources directly. Both scripts accept `COMPONENTS` to select dependencies; the default is all components in the manifest. The user procedure is in {ref}`Offline installation <install-offline>`.

Managed sources and patched copies always use `thirdparty/src/` in the LibRPA checkout. Archives default to `thirdparty/downloads/` in the same checkout. You can select a separate archive directory with {ref}`LIBRPA_THIRDPARTY_ARCHIVE_DIR <librpa-thirdparty-archive-dir>`. Its derived default stays uncached. Downloads use temporary files in that directory before renaming, so other configurations cannot read partial downloads. Shared archives are preserved when extraction or hash verification fails.

Whole-tree path overlays use `file(COPY)` without `FOLLOW_SYMLINK_CHAIN`. This preserves directory symbolic links on macOS and Linux. The copied tree must match the original hash before patches are applied. The user tree remains unchanged. Keep Git metadata outside a tree that should match the manifest; `.git` is not excluded from the hash.

Read-only pristine trees can supply header and whole-tree overlays. The copy becomes owner-writable before patch application, and executable files keep their execute permissions. Overlays require a writable `thirdparty/src/` directory in the LibRPA checkout. The supplied tree keeps its contents and permissions.

Each preparation uses a separate temporary directory. After extraction and patch application, the prepared tree is renamed into the directory selected by the manifest tree hash. Concurrent configurations reuse a tree published by another process; no filesystem lock is required. The name identifies the reference version; local edits do not rename the directory. Reuse compares its contents with the reference hash and warns on differences. Normal configuration reuses existing trees and overlays, preserving local edits. Use the download script to populate an archive directory independently of existing sources. Build outputs must stay in build directories. In particular, copy greenX's Python directory to scratch space before installing it, because Python packaging writes build files into its source directory.

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

Each target independently takes precedence over its path variable and managed source. A parent may provide only some of these targets. LibRPA resolves the others through their paths or managed sources.

Header targets have status `target`. LibRPA does not hash or patch them. The parent must supply compatible headers, including required portability fixes. Use a path variable instead if pristine pinned LibRI headers need LibRPA's managed patch overlay.

The required configure-time compile probe reads each target's direct `INTERFACE_INCLUDE_DIRECTORIES`. Absolute paths and simple `BUILD_INTERFACE` expressions are supported. `INSTALL_INTERFACE` entries are ignored, and other generator expressions cause an error. The probe does not traverse transitive include directories or inherit arbitrary target compile definitions. Expose the required header paths directly on these targets.

The header combination must compile before configuration can continue. This check does not prove numerical compatibility. Target-provided LibComm also keeps the conservative automatic `MPI_THREAD_MULTIPLE` selection. See {ref}`Compile Options <librpa-mpi-thread-level>`.

### Compiled dependency targets

| Parent target | Selection requirement | Parent responsibilities |
| --- | --- | --- |
| `LibDDLA::ddla` | Enable CUDA or HIP | Supply a library, public include paths and link dependencies compatible with that backend |
| `ELPA::ELPA` | Detected automatically | Supply the library, headers and required link dependencies |
| `LibGXMiniMax` or `greenX::LibGXMiniMax` | Detected automatically | Supply link dependencies and compatible modules: `Fortran_MODULE_DIRECTORY` for source builds, interface include directories for imported targets |

Define these targets before adding LibRPA. An ELPA target automatically enables `LIBRPA_USE_ELPA`. GreenX targets are detected even with `LIBRPA_USE_EXTERNAL_GREENX=OFF`, so LibRPA does not create a second target. Setting this option to ON requires a target, source path or installation.

In a parent project, prefer a target when the parent already builds the dependency; otherwise let LibRPA use managed sources. See {ref}`Choosing dependency sources <install-dependency-sources>` for the general recommendation and the path rules.

Use `rpa_lib` to link a C or C++ consumer. With `LIBRPA_ENABLE_FORTRAN_BIND=ON`, the Fortran target is `rpa_f03_lib`, which also exposes its build-time module directory. LibRPA does not currently install a package configuration or target export for `find_package(librpa)`.

## Migration from vendored sources

Existing `LIBRI_INCLUDE_DIR`, `LIBCOMM_INCLUDE_DIR`, `CEREAL_INCLUDE_DIR` and `LIBDDLA_PATH` settings remain supported. An environment value is now saved in the cache when the cache entry is empty, so a selection persists after the variable is unset. See {ref}`Compile Options <librpa-how-to-set-options>` for how to clear a selection.

The old `thirdparty/LibRI`, `thirdparty/LibComm`, `thirdparty/cereal`, `thirdparty/LibDDLA` and `thirdparty/greenX` source paths no longer exist. Replace scripts that copy those directories with the preparation script or explicit external paths. Do not hard-code `thirdparty/src/<Name>-<hash12>` paths: a manifest update changes them. For offline builds, transfer the archives and prepare the sources in the destination checkout.

`LIBRPA_USE_ELPA=ON` builds managed ELPA sources when no dependency is supplied. An ELPA target, source path or installation prefix automatically enables it. See {ref}`Compile Options <librpa-use-elpa>` for the selection rules and deprecated switches. The tracked `thirdparty/ELPA` directory contains only LibRPA's build wrapper. A dependency cache contains sources and archives, not compiler-specific installed libraries. Each build directory creates its own compiled dependencies.
