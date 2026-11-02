#!/bin/bash

# This script uses GCC, OpenMPI, OpenBLAS and ScaLAPACK to build LibRPA on Linux.
# LibRI and Fortran binding can be switched on or off.
#
# Tested on Fedora 43.
#
# Compiler, MPI and math dependencies are installed with dnf.
# CMake prepares the managed source dependencies separately.
#

BUILDDIR="${BUILDDIR:=build_gcc_openmpi_scalapack}"

# Switch for LibRI
export USE_LIBRI="${USE_LIBRI:=ON}"
# Switch for Fortran binding
export ENABLE_FORTRAN_BIND="${ENABLE_FORTRAN_BIND:=ON}"

# Optionally, one can specify the path of their own LibRI and LibComm libraries.
# If unset or empty, CMake prepares the pinned managed sources.
# See docs/user_guide/install.md for offline preparation and cache settings.
export LIBRI_INCLUDE="${LIBRI_INCLUDE:=}"
export LIBCOMM_INCLUDE="${LIBCOMM_INCLUDE:=}"

# The following two variables need customize
export OPENBLAS_DIR="/usr/lib64"
export SCALAPACK_DIR="/usr/lib64/openmpi/lib"

# Compilers setup
export CXX=mpicxx
export CC=gcc
export OMPI_CC=gcc
export OMPI_CXX=g++

if [[ $ENABLE_FORTRAN_BIND == "ON" ]]; then
  export FC=mpifort
  export OMPI_FC=gfortran
else
  export FC=gfortran
fi

if [[ $USE_LIBRI == "ON" ]]; then
  BUILDDIR="${BUILDDIR}_libri"
fi

# Compiler flags
export FCFLAGS="-fPIC -g -fallow-argument-mismatch -ffree-line-length-none"
export CXXFLAGS="-fPIC -g -rdynamic -Wl,-lgfortran"

cmake -B "$BUILDDIR" \
  -DLIBRPA_USE_LIBRI=$USE_LIBRI \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_FORTRAN_BIND=$ENABLE_FORTRAN_BIND \
  -DLIBRI_INCLUDE_DIR="$LIBRI_INCLUDE" \
  -DLIBCOMM_INCLUDE_DIR="$LIBCOMM_INCLUDE" \
  -DBLAS_LIBRARIES="-L${OPENBLAS_DIR} -lopenblas" \
  -DLAPACK_LIBRARIES="-L${OPENBLAS_DIR} -lopenblas" \
  -DSCALAPACK_DIR="$SCALAPACK_DIR" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_Fortran_FLAGS="$FCFLAGS"

cmake --build "$BUILDDIR" -j 4
