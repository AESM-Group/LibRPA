# Managed third-party dependencies

LibRPA stores dependency versions, upstream URLs, patch lists, tree hashes and licenses in `manifest.cmake`. Third-party source code is downloaded during configuration and is not tracked in this repository. The default build needs greenX and, with `LIBRPA_USE_LIBRI=ON`, LibRI, LibComm and cereal. GPU builds also need LibDDLA; `LIBRPA_USE_BUNDLED_ELPA=ON` selects managed ELPA.

## Resolution and configuration

Header libraries resolve in this order: an existing `LibRI::LibRI`, `LibComm::LibComm` or `cereal::cereal` target; an explicit include directory; managed sources. Nonempty CMake path variables take precedence over environment variables. Explicit paths must contain the expected header; an invalid path is an error and never triggers a download fallback.

| Setting | Default | Purpose |
| --- | --- | --- |
| `LIBRPA_THIRDPARTY_DIR` | LibRPA source `thirdparty/` | Shared archive and source storage, independent of build directories |
| `LIBRPA_THIRDPARTY_DOWNLOAD` | `ON` | Allow downloads of missing archives |
| `LIBRPA_THIRDPARTY_MIRRORS` | empty | Semicolon-separated URL prefixes, tried before upstream URLs; supports `https://` and `file://` |

LibDDLA resolves an existing `LibDDLA::ddla` target, `LIBDDLA_PATH` installation, `LIBDDLA_SOURCE_DIR` source tree, then managed sources. Source builds use a separate ExternalProject. External ELPA and greenX retain their existing options.

Sources are prepared under `src/<Name>-<TREE_SHA256 first 12 characters>/`. Archives are cached under `downloads/`. Both directories are ignored by Git. Every configuration verifies each selected managed tree; there is no option to skip this check. A mismatch stops configuration without deleting the modified tree. Preserve any edits, then delete that tree to prepare it again. Use path mode for dependency development.

## Offline preparation

On a connected machine, from the LibRPA checkout:

```sh
cmake -DLIBRPA_THIRDPARTY_DIR=/path/to/deps -P cmake/LibRPAPrepareThirdparty.cmake
```

Copy `/path/to/deps` to the offline machine, preserving symbolic links (for example, with `rsync -a`). From the LibRPA checkout, configure with:

```sh
cmake -S . -B build -DLIBRPA_THIRDPARTY_DIR=/path/to/deps -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF
```

The preparation script defaults to all six components. To prepare a subset, add `-DCOMPONENTS="LibRI;LibComm;cereal;greenX"`. Alternatively, place the exact archive filenames listed in the manifest in `/path/to/deps/downloads/`; configuration can extract, patch and verify these with downloads disabled. Mirrors serve those same filenames.

## Local paths and parent projects

Set `LIBRI_INCLUDE_DIR`, `LIBCOMM_INCLUDE_DIR` or `CEREAL_INCLUDE_DIR` to the directory containing `RI`, `Comm` or `cereal`, respectively. Matching patched headers are used directly (`pinned`). Pristine pinned LibRI headers are copied and patched in managed storage (`overlay`); the supplied directory is never modified. Other revisions are used with a warning (`unpinned`). A required compile probe checks the three libraries together. Parent targets have status `target` and are not patched; the parent must provide compatible headers.

For a complete example, see [Parent-project integration](../docs/develop/thirdparty.md#parent-project-integration). Set `LIBRPA_THIRDPARTY_DIR` to the parent's dependency directory before `add_subdirectory`, and ignore its `downloads/` and `src/` directories. The LibRPA source directory then needs no generated dependency files.

For compiler requirements, build/install commands and troubleshooting, see [Installation](../docs/user_guide/install.md). See [Compile Options](../docs/user_guide/compile_options.md) for defaults and all path rules. The three cache settings above are not read from environment variables.

## Updates

Update the manifest, regenerate any necessary patches, compute pristine and patched tree hashes, and run build and numerical regressions. See [the developer procedure](../docs/develop/thirdparty.md) for the exact hash definition and validation steps, and [patches/README.md](patches/README.md) for local modifications and licenses.
