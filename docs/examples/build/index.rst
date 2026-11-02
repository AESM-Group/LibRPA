Build Examples
==============

This section provides example build scripts for LibRPA on several representative
platforms and software stacks. They are intended as practical references for
setting up a working build, rather than as exhaustive or universally applicable
recipes.

Depending on your local environment, you may need to adjust compiler commands,
library paths, MPI settings, and CMake options accordingly.

CMake prepares pinned source dependencies when their archives or verified trees
are absent. For an offline build, prepare the cache first as described in
:doc:`../../user_guide/install`. Pass ``LIBRPA_THIRDPARTY_DIR`` and
``LIBRPA_THIRDPARTY_DOWNLOAD=OFF`` in the script's CMake command.
Check explicit header paths in each script: they take precedence over managed
headers, and an invalid path stops configuration.

For all defaults and dependency selection rules, see
:doc:`../../user_guide/compile_options`.

The following examples are currently available:

.. toctree::
   :maxdepth: 1

   linux_gcc_openmpi_openblas
   linux_intel_classic_mkl
   linux_intel_llvm_mkl
   macos_gcc_mpich_openblas
