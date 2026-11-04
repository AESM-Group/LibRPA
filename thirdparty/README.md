# Managed third-party dependencies

LibRPA stores dependency versions, upstream URLs, patch lists, tree hashes and licenses in `manifest.cmake`. Third-party source code is downloaded during configuration and is not tracked in this repository. The default build needs greenX and, with `LIBRPA_USE_LIBRI=ON`, LibRI, LibComm and cereal. GPU builds also need LibDDLA; `LIBRPA_USE_BUNDLED_ELPA=ON` builds ELPA from `ELPA_SOURCE_DIR` or managed sources.

## Resolution and configuration

Header libraries resolve in this order: an existing `LibRI::LibRI`, `LibComm::LibComm` or `cereal::cereal` target; an explicit include directory; managed sources. Nonempty CMake path variables take precedence over environment variables. Explicit paths must contain the expected header; an invalid path is an error and never triggers a download fallback.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LIBRPA_THIRDPARTY_DIR` | LibRPA source `thirdparty/` | Shared archive and source storage, independent of build directories |
| `LIBRPA_THIRDPARTY_ARCHIVE_DIR` | empty | CMake value, then environment, then `${LIBRPA_THIRDPARTY_DIR}/downloads` |
| `LIBRPA_THIRDPARTY_DOWNLOAD` | `ON` | Allow downloads of missing archives |
| `LIBRPA_THIRDPARTY_MIRRORS` | empty | Semicolon-separated URL prefixes, tried before upstream URLs; supports `https://` and `file://` |

Compiled dependencies select the first supplied source: parent target, source path, installation prefix, then managed sources. This selection order is not a recommendation. Prefer managed sources, then parent targets, then source paths. Use installation prefixes only when the other choices are not possible. ELPA source and installation paths are mutually exclusive in every mode. An ELPA path with the wrong mode enabled is an error. With both modes OFF, an unused path produces a warning. Prebuilt installations emit a compiler-compatibility warning on every configuration. GreenX recognizes parent targets automatically. ELPA keeps its external and source-build switches. LibDDLA source builds use a separate ExternalProject.

Sources are prepared under `src/<Name>-<TREE_SHA256 first 12 characters>/`. Archives default to `downloads/`, or use the separate `LIBRPA_THIRDPARTY_ARCHIVE_DIR`. Multiple checkouts can share that archive directory. Local archives work with downloads disabled. Downloads use a temporary file in the archive directory followed by a rename. Corrupt shared archives are not deleted automatically. The default `downloads/` and `src/` directories are ignored by Git. Every configuration verifies each selected managed tree; there is no option to skip this check. A mismatch stops configuration without deleting the modified tree. Preserve any edits, then delete that tree to prepare it again. Use path mode for dependency development.

Dependency include, source, installation and archive paths use `PATH` cache entries. Untyped relative `-D` values on the first configuration resolve against the CMake invocation directory. Use absolute paths for environment values and typed or later cache values. Remaining relative values are rejected.

When a dependency path cache entry is empty, a selected environment path is saved in the cache and reported. A nonempty cache value takes precedence. To clear it, unset the environment variable and run `cmake -U NAME -S . -B build`. Standalone `cmake -P` preparation does not persist a cache.

## Offline preparation

On a connected machine, from the LibRPA checkout:

```sh
cmake -DLIBRPA_THIRDPARTY_DIR=/path/to/deps -P cmake/LibRPAPrepareThirdparty.cmake
```

Copy `/path/to/deps` to the offline machine, preserving symbolic links (for example, with `rsync -a`). From the LibRPA checkout, configure with:

```sh
cmake -S . -B build -DLIBRPA_THIRDPARTY_DIR=/path/to/deps -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF
```

To share archives, set `export LIBRPA_THIRDPARTY_ARCHIVE_DIR=~/downloads/LibRPA_tarballs` before preparation or configuration. A nonempty CMake value takes precedence.

The preparation script defaults to all six components. To prepare a subset, add `-DCOMPONENTS="LibRI;LibComm;cereal;greenX"`. Alternatively, place the exact archive filenames listed in the manifest in the effective archive directory (default `/path/to/deps/downloads/`); configuration can extract, patch and verify these with downloads disabled. Mirrors serve those same filenames.

## Local paths and parent projects

Set `LIBRI_INCLUDE_DIR`, `LIBCOMM_INCLUDE_DIR` or `CEREAL_INCLUDE_DIR` to the directory containing `RI`, `Comm` or `cereal`, respectively. Matching patched headers are used directly (`pinned`). Pristine pinned LibRI headers are copied and patched in managed storage (`overlay`); the supplied directory is never modified. Other revisions are used with a warning (`unpinned`). A required compile probe checks the three libraries together. Parent targets have status `target` and are not patched; the parent must provide compatible headers.

Whole-tree paths (`GREENX_SOURCE_DIR`, `ELPA_SOURCE_DIR`, `LIBDDLA_SOURCE_DIR`) use the same hash rules. Matching pristine greenX or ELPA trees receive an overlay. `file(COPY)` preserves symbolic links, and the copied tree is verified before patching. The supplied source tree is unchanged.

Read-only pristine trees can supply header and whole-tree overlays. The copy becomes owner-writable before patch application, and executable files keep their execute permissions. Overlays require a writable `LIBRPA_THIRDPARTY_DIR/src` directory. The supplied tree keeps its contents and permissions.

For a complete example, see [Parent-project integration](../docs/develop/thirdparty.md#parent-project-integration). Set `LIBRPA_THIRDPARTY_DIR` to the parent's dependency directory before `add_subdirectory`, and ignore its `downloads/` and `src/` directories. The LibRPA source directory then needs no generated dependency files.

For compiler requirements, build/install commands and troubleshooting, see [Installation](../docs/user_guide/install.md). See [Compile Options](../docs/user_guide/compile_options.md) for defaults and all path rules. Of the four settings in the table above, only `LIBRPA_THIRDPARTY_ARCHIVE_DIR` reads an environment variable. Dependency include, source and installation paths also read environment variables.

## Updates

Update the manifest, regenerate any necessary patches, compute pristine and patched tree hashes, and run build and numerical regressions. See [the developer procedure](../docs/develop/thirdparty.md) for the exact hash definition and validation steps, and [patches/README.md](patches/README.md) for local modifications and licenses.
