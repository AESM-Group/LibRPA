#!/bin/bash

# This script uses Intel classic compilers, Intel MPI and MKL to build LibRPA
# on Linux platform for develop and production use. Tested on Ubuntu, CentOS and SUSE.

# Intel compilers icpc (C++) and ifort (Fortran) needs to be found under directories
# in environment variable PATH, as well as the C++ MPI wrapper mpiicpc.
# Note that Fortran MPI wrapper (mpiifort) is required if LIBRPA_ENABLE_FORTRAN_BIND is on.
# Intel MKL will be used as the working math library.

BUILDDIR="${BUILDDIR:=build_intel_classic_mkl}"

# # Ensure environment variables are correctly set.
# # On PC, they can be set by sourcing setup script provided by the vendor.
# # For oneAPI:
# source /opt/intel/oneapi/setvars.sh
# # For old Intel versions
# source /opt/intel/compilers_and_libraries_2020.4.304/linux/bin/compilervars.sh intel64

# # On HPC, usually environment modules are provided.
# # Examples:
# #   MPCDF platforms, oneAPI 2023
# module load intel/2023.1.0.x impi/2021.9 mkl/2023.1

# #   MPCDF platforms, Intel 2020 update 4
# module load intel/19.1.3 impi/2019.9 mkl/2020.4

# # Or you might set it manually, which is not recommended.

# Switch for LibRI
export USE_LIBRI="${USE_LIBRI:=OFF}"
# Switch for Fortran binding
export ENABLE_FORTRAN_BIND="${ENABLE_FORTRAN_BIND:=OFF}"

# Optionally, one can specify the path of their own LibRI and LibComm libraries.
# If unset or empty, CMake prepares the pinned managed sources.
# See docs/user_guide/install.md for offline preparation and cache settings.
export LIBRI_INCLUDE="${LIBRI_INCLUDE:=}"
export LIBCOMM_INCLUDE="${LIBCOMM_INCLUDE:=}"

export CXX=mpiicpc

if [[ $ENABLE_FORTRAN_BIND == "ON" ]]; then
  export FC=mpiifort
else
  export FC=ifort
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
