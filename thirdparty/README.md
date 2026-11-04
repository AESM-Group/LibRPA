# Managed third-party dependencies

LibRPA does not track third-party source code. This directory declares the dependencies and holds the files that LibRPA downloads and prepares:

| Path | Content |
| --- | --- |
| `manifest.cmake` | Version, upstream URL, archive filename, patches, license and tree hashes of each component |
| `patches/` | Local patches, applied after extraction; see [patches/README.md](patches/README.md) |
| `ELPA/` | LibRPA's build wrapper for ELPA source builds |
| `downloads/` | Default archive directory; generated and ignored by Git |
| `src/` | Verified, patched source trees in `src/<Name>-<hash12>/`; generated and ignored by Git |

A default configuration downloads, patches and verifies the required components here. `LIBRPA_THIRDPARTY_DIR` moves `downloads/` and `src/` to another directory, and `LIBRPA_THIRDPARTY_ARCHIVE_DIR` moves the archives only. Do not edit the trees in `src/`: configuration rejects changed contents.

## Documentation

- [Installation](../docs/user_guide/install.md): offline preparation, mirrors and shared archives, how to choose a dependency source, local paths, and troubleshooting.
- [Compile Options](../docs/user_guide/compile_options.md): each dependency option, with its default, precedence and path rules.
- [Maintaining third-party dependencies](../docs/develop/thirdparty.md): the tree hash definition, how to update a component, concurrent preparation, parent-project integration and migration from vendored sources.
- [patches/README.md](patches/README.md): what each patch changes.

## Offline preparation in short

On a connected machine, from the LibRPA checkout:

```sh
cmake -DLIBRPA_THIRDPARTY_DIR=/path/to/deps -P cmake/LibRPAPrepareThirdparty.cmake
```

Copy `/path/to/deps` to the offline machine with symbolic links preserved, for example with `rsync -a`. Then configure with `-DLIBRPA_THIRDPARTY_DIR=/path/to/deps -DLIBRPA_THIRDPARTY_DOWNLOAD=OFF`. See [Offline installation](../docs/user_guide/install.md#offline-installation) for component selection and archive-only transfer.
