# Managed third-party dependencies

LibRPA does not track third-party source code. This directory declares the dependencies and holds the files that LibRPA downloads and prepares:

| Path | Content |
| --- | --- |
| `manifest.cmake` | Version, upstream URL, archive filename, patches, license, archive SHA-256 and reference tree hashes of each component |
| `patches/` | Local patches, applied after extraction; see [patches/README.md](patches/README.md) |
| `ELPA/` | LibRPA's build wrapper for ELPA source builds |
| `downloads/` | Default archive directory; generated and ignored by Git |
| `src/` | Prepared, patched source trees in `src/<Name>-<hash12>/`; generated and ignored by Git |

A default configuration downloads the required archives, verifies their SHA-256 hashes, then extracts and patches the sources here.
Archives are downloaded under `LIBRPA_THIRDPARTY_ARCHIVE_DIR`, which defaults to `thirdparty/downloads/`.
Extracted sources and patched copies stay in `thirdparty/src/`.

You can edit the trees in `thirdparty/src/` directly after preparation.
Configuration warns when their contents differ from the reference tree and uses them with status `unpinned`.
It preserves your edits and does not download or apply patches again.
The warning uses the manifest's tree hash. It covers `include/` for header-only libraries and the full source tree for compiled libraries.
Please be careful to preserve local changes before deleting a source directory or switching dependency versions.

Relevant documentation includes:

- [Installation](../docs/user_guide/install.md): offline preparation, downloads and shared archives, how to choose a dependency source, local paths, and troubleshooting.
- [Compile Options](../docs/user_guide/compile_options.md): each dependency option, with its default, precedence and path rules.
- [Maintaining third-party dependencies](../docs/develop/thirdparty.md): the tree hash definition, how to update a component, concurrent preparation, parent-project integration and migration from vendored sources.
- [patches/README.md](patches/README.md): what each patch changes.

To download archives for an offline build, run the following command at the LibRPA root on a connected machine:

```sh
cmake -P cmake/LibRPADownloadThirdparty.cmake
```

Then copy the downloaded archives to `LibRPA/thirdparty/downloads/` on the offline machine and configure with `-DLIBRPA_THIRDPARTY_DOWNLOAD=OFF`.
If you upload them to another directory, set `-DLIBRPA_THIRDPARTY_ARCHIVE_DIR=/path/to/archives` on that machine.
Each LibRPA checkout prepares its own dependency sources in `thirdparty/src/`.
See [Offline installation](../docs/user_guide/install.md#offline-installation) for more details.

To extract and patch sources in the local checkout before configuration, run `cmake -P cmake/LibRPAPrepareThirdparty.cmake`. This script downloads missing archives when needed. Both scripts accept `-DCOMPONENTS="LibRI;LibComm;cereal;greenX"` before `-P` to select components.
