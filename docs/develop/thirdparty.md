# Maintaining third-party dependencies

The tracked `thirdparty/manifest.cmake` declares the version, URL, archive filename, patches, license and expected tree hash of each component. Header-only libraries are hashed over `include/`; LibDDLA, ELPA and greenX are hashed over the whole extracted tree. Archives are not hashed because generated upstream archives can change byte representation without changing file contents.

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

1. Download the selected upstream archive into a fresh scratch directory and extract it without running its scripts. Check the version and license. Preserve symbolic links and original line endings.
2. Compare it with the currently selected tree. Keep only required portability and integration changes; do not carry unrelated formatting or generated files.
3. Generate `-p1` patches with `a/` and `b/` paths relative to the archive root. Record the source, reason, upstream submission status and license in the patch header and `thirdparty/patches/README.md`.
4. Apply the patches to a fresh upstream tree. Compute pristine and patched hashes with `cmake/LibRPAThirdparty.cmake`, and independently cross-check a header tree with the shell command above.
5. Update the declaration's full version, URL, archive name, patch list and hashes. Update the license list in `docs/develop/licensing.md`, then run `python3 utilities/update_readme_licensing_list.py`.
6. Prepare sources into an empty dependency directory. Test online download, local mirrors, cached archives with downloads disabled, repeated configuration, and rejection of modified managed trees. For LibRI, greenX and ELPA, also check pristine-path overlays and verify that the original tree is unchanged.
7. Run `python3 utilities/test_thirdparty.py` to check helper failure paths, overlays, concurrent preparation and directory-symlink handling.
8. Build with and without LibRI, run unit and numerical regression tests, and run the QSGW, Hartree and cRPA verification scripts used in CI. Check parent targets and explicit paths. For ELPA and greenX, verify the source hashes again after building.

A minimal hash script is:

```cmake
cmake_minimum_required(VERSION 3.13)
include("/path/to/LibRPA/cmake/LibRPAThirdparty.cmake")
librpa_thirdparty_tree_sha256(hash "/path/to/tree/include")
message("${hash}")
```

Run it with `cmake -P hash.cmake`. Do not edit a managed tree to update a dependency: configuration rejects changed contents. Work on a separate tree and use path mode while developing.

## Preparation and concurrent builds

Run `cmake -P cmake/LibRPAPrepareThirdparty.cmake` on a connected machine to prepare every component, or select `-DCOMPONENTS="LibRI;LibComm;cereal;greenX"`. The `LIBRPA_THIRDPARTY_DIR`, `LIBRPA_THIRDPARTY_ARCHIVE_DIR`, `LIBRPA_THIRDPARTY_DOWNLOAD` and `LIBRPA_THIRDPARTY_MIRRORS` settings also work in script mode. Copy the prepared directory to offline machines and disable downloads there.

The source directory and archive directory can be separate. `LIBRPA_THIRDPARTY_ARCHIVE_DIR` uses a nonempty CMake value, then its environment variable, then `${LIBRPA_THIRDPARTY_DIR}/downloads`. Its cache default stays empty. Downloads use temporary files in that directory before renaming, so other configurations cannot read partial downloads. Shared archives are preserved when extraction or hash verification fails.

Whole-tree path overlays use `file(COPY)` without `FOLLOW_SYMLINK_CHAIN`. This preserves directory symbolic links on macOS and Linux. The copied tree must match the original hash before patches are applied. The user tree remains unchanged. Keep Git metadata outside a tree that should match the manifest; `.git` is not excluded from the hash.

Each preparation uses a separate temporary directory. Only a verified tree is renamed into its content-addressed final location. Concurrent configurations verify a tree published by another process; no filesystem lock is required. Build outputs must stay in build directories. In particular, copy greenX's Python directory to scratch space before installing it, because Python packaging writes build files into its source directory.

(thirdparty-parent-projects)=
## Parent-project integration

A parent can let LibRPA prepare dependencies while keeping its checkout free of generated source trees. Set the cache directory before adding LibRPA:

```cmake
cmake_minimum_required(VERSION 3.18)
project(HostCode LANGUAGES CXX Fortran)

set(LIBRPA_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/librpa"
    CACHE PATH "Shared LibRPA source cache")
set(LIBRPA_ENABLE_DRIVER OFF CACHE BOOL "Build the LibRPA driver")
set(LIBRPA_ENABLE_TEST OFF CACHE BOOL "Build LibRPA tests")
set(ENABLE_GREENX_CTEST OFF CACHE BOOL "Register GreenX tests")
add_subdirectory(LibRPA)

add_executable(host main.cpp)
target_link_libraries(host PRIVATE rpa_lib)
```

This example assumes that `LibRPA/` and `main.cpp` are in the parent source directory. Configure the parent with its compiler, MPI and math-library settings. Ignore `dependencies/librpa/downloads/` and `dependencies/librpa/src/` in the parent's repository. An existing cache value takes precedence over these `set(... CACHE ...)` defaults.

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
| `ELPA::ELPA` | Set `LIBRPA_USE_EXTERNAL_ELPA=ON` | Supply the library, headers and required link dependencies |
| `LibGXMiniMax` or `greenX::LibGXMiniMax` | Detected automatically | Supply link dependencies and compatible modules: `Fortran_MODULE_DIRECTORY` for source builds, interface include directories for imported targets |

Define these targets before adding LibRPA. ELPA still requires `LIBRPA_USE_EXTERNAL_ELPA=ON`. GreenX targets are detected even with `LIBRPA_USE_EXTERNAL_GREENX=OFF`, so LibRPA does not create a second target. Setting this option to ON requires a target, source path or installation.

For compiled dependencies, managed sources are recommended unless the parent already provides the dependency. In a parent project, prefer a target; for dependency development, prefer a source path. Installations are supported with a warning about compiler and flag compatibility. Source and installation paths for the same enabled source build are mutually exclusive. ELPA source paths outside source-build mode are ignored with a warning.

Use `rpa_lib` to link a C or C++ consumer. With `LIBRPA_ENABLE_FORTRAN_BIND=ON`, the Fortran target is `rpa_f03_lib`, which also exposes its build-time module directory. LibRPA does not currently install a package configuration or target export for `find_package(librpa)`.

## Migration from vendored sources

Existing `LIBRI_INCLUDE_DIR`, `LIBCOMM_INCLUDE_DIR`, `CEREAL_INCLUDE_DIR` and `LIBDDLA_PATH` settings remain supported. Check both the environment and the CMake cache when changing dependency selection. An empty header-path cache value permits an environment fallback.

The old `thirdparty/LibRI`, `thirdparty/LibComm`, `thirdparty/cereal`, `thirdparty/LibDDLA` and `thirdparty/greenX` source paths no longer exist. Replace scripts that copy those directories with the preparation script or explicit external paths. Do not hard-code `src/<Name>-<hash12>` paths: a manifest update changes them. Preserve symbolic links when transferring a prepared cache.

`LIBRPA_USE_BUNDLED_ELPA` and its related option names remain supported, but now build supplied source paths or downloaded managed sources. The tracked `thirdparty/ELPA` directory contains only LibRPA's build wrapper. A dependency cache contains sources and archives, not compiler-specific installed libraries. Each build directory creates its own compiled dependencies.
