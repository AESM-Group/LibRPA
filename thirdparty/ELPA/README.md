# Managed ELPA build wrapper

This directory contains only LibRPA's CMake wrapper. With `LIBRPA_USE_ELPA=ON` and no supplied dependency, LibRPA prepares the release pinned in `../manifest.cmake`. Supplying `ELPA_SOURCE_DIR` automatically enables ELPA and selects that source tree. Matching pristine sources receive a patched overlay in managed storage. The wrapper builds the selected tree through upstream autotools in a separate build directory, preserving the source tree.

The supported `LIBRPA_BUNDLED_ELPA_VERSION` is `2026.02.001`. The wrapper exposes an `elpa` target. A parent `ELPA::ELPA` target or an installation prefix (`EXTERNAL_ELPA_DIR`, with `ELPA_ROOT` as a fallback) also enables ELPA and bypasses this wrapper.

ELPA is licensed under LGPL-3.0. License texts are in the prepared source tree's `LICENSE` and `COPYING/` paths. The macOS affinity patch is documented in `../patches/README.md`.
