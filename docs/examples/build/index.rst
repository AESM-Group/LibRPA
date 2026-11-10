Build Examples
==============

This section provides example build scripts for LibRPA on several representative
platforms and software stacks. They are intended as practical references for
setting up a working build, rather than as exhaustive or universally applicable
recipes.

Depending on your local environment, you may need to adjust compiler commands,
library paths, MPI settings, and CMake options accordingly.
For all defaults and dependency selection rules, see
:doc:`../../user_guide/compile_options`.

The following examples are currently available:

.. toctree::
   :maxdepth: 1

   linux_gcc_openmpi_openblas
   linux_intel_classic_mkl
   linux_intel_llvm_mkl
   macos_gcc_mpich_openblas
