# Maintaining third-party dependencies

The tracked `thirdparty/manifest.cmake` declares the version, URL, archive filename, patches, license and expected tree hash of each component. Header-only libraries are hashed over `include/`; LibDDLA, ELPA and greenX are hashed over the whole extracted tree. Archives are not hashed because generated upstream archives can change byte representation without changing file contents.

## Tree hash definition

`librpa_thirdparty_tree_sha256(out root)` recursively lists files relative to `root` with CMake `file(GLOB_RECURSE ... LIST_DIRECTORIES false)`, sorts the paths, hashes each file with SHA-256, and hashes the concatenation of lines:

```text
<lowercase file SHA256><two spaces><relative path><newline>
```

Directory names, file modes and timestamps are not included. Preserve upstream symbolic links when extracting or copying archives; expanding directory links changes the file list. The equivalent shell check for a tree without symbolic links is:

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
6. Prepare sources into an empty dependency directory. Test online download, local mirrors, cached archives with downloads disabled, repeated configuration, and rejection of modified managed trees. For LibRI, also check pristine-path overlays and verify that the original include tree is unchanged.
7. Build with and without LibRI, run unit and numerical regression tests, and run the QSGW, Hartree and cRPA verification scripts used in CI. Check parent targets and explicit paths. For ELPA and greenX, verify the source hashes again after building.

A minimal hash script is:

```cmake
cmake_minimum_required(VERSION 3.13)
include("/path/to/LibRPA/cmake/LibRPAThirdparty.cmake")
librpa_thirdparty_tree_sha256(hash "/path/to/tree/include")
message("${hash}")
```

Run it with `cmake -P hash.cmake`. Do not edit a managed tree to update a dependency: configuration rejects changed contents. Work on a separate tree and use path mode while developing.

## Preparation and concurrent builds

Run `cmake -P cmake/LibRPAPrepareThirdparty.cmake` on a connected machine to prepare every component, or select `-DCOMPONENTS="LibRI;LibComm;cereal;greenX"`. The `LIBRPA_THIRDPARTY_DIR`, `LIBRPA_THIRDPARTY_DOWNLOAD` and `LIBRPA_THIRDPARTY_MIRRORS` settings also work in script mode. Copy the prepared directory to offline machines and disable downloads there.

Each preparation uses a separate temporary directory. Only a verified tree is renamed into its content-addressed final location. Concurrent configurations verify a tree published by another process; no filesystem lock is required. Build outputs must stay in build directories. In particular, copy greenX's Python directory to scratch space before installing it, because Python packaging writes build files into its source directory.
