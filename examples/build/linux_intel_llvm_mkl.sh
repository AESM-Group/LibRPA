#!/bin/bash

# This script uses Intel LLVM-based compilers, Intel MPI and MKL to build LibRPA
# on Linux platform for develop and production use. Tested on Ubuntu and SUSE.

# Intel compilers icpx (C++) and ifx (Fortran) needs to be found under directories
# in environment variable PATH, as well as the C++ MPI wrapper mpiicpx.
# Note that Fortran MPI wrapper (mpiifx) is required if LIBRPA_ENABLE_FORTRAN_BIND is on.
# Intel MKL will be used as the working math library.

BUILDDIR="${BUILDDIR:=build_intel_llvm_mkl}"

# # Ensure environment variables are correctly set.
# # On PC, they can be set by sourcing setup script provided by the vendor.
# # As LLVM-based compilers are only provided in oneAPI toolchain, this can be done by:
# source /opt/intel/oneapi/setvars.sh

# # On HPC, usually environment modules are provided:
# module load intel/2023.1.0.x impi/2021.9 mkl/2023.1

# # Or you might set it manually, which is not recommended.

# Switch for LibRI
export USE_LIBRI="${USE_LIBRI:=ON}"
# Switch for Fortran binding
export ENABLE_FORTRAN_BIND="${ENABLE_FORTRAN_BIND:=OFF}"

# Optionally, one can specify the path of their own LibRI and LibComm libraries.
# If unset or empty, CMake prepares the pinned managed sources.
# See docs/user_guide/install.md for offline preparation and cache settings.
export LIBRI_INCLUDE="${LIBRI_INCLUDE:=}"
export LIBCOMM_INCLUDE="${LIBCOMM_INCLUDE:=}"

export CXX=mpiicpx

if [[ $ENABLE_FORTRAN_BIND == "ON" ]]; then
  export FC=mpiifx
else
  export FC=ifx
fi

if [[ $USE_LIBRI == "ON" ]]; then
  BUILDDIR="${BUILDDIR}_libri"
fi

cmake -B "$BUILDDIR" \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_FORTRAN_BIND=$ENABLE_FORTRAN_BIND \
  -DLIBRPA_USE_LIBRI=$USE_LIBRI \
  -DLIBRI_INCLUDE_DIR="$LIBRI_INCLUDE" \
  -DLIBCOMM_INCLUDE_DIR="$LIBCOMM_INCLUDE"

cd "$BUILDDIR" && make -j 4
