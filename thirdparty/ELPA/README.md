# Managed ELPA build wrapper

This directory contains only LibRPA's CMake wrapper. With `LIBRPA_USE_BUNDLED_ELPA=ON`, the source release pinned in `../manifest.cmake` is downloaded, patched and verified in managed storage before the wrapper builds it through upstream autotools in a separate build directory.

The supported `LIBRPA_BUNDLED_ELPA_VERSION` is `2026.02.001`. The option names remain compatible with existing build scripts. The wrapper exposes an `elpa` target; external installations remain selectable with `LIBRPA_USE_EXTERNAL_ELPA`.

ELPA is licensed under LGPL-3.0. License texts are in the prepared source tree's `LICENSE` and `COPYING/` paths. The macOS affinity patch is documented in `../patches/README.md`.
