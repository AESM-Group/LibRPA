#!/bin/bash

# This script uses GCC, MPICH, OpenBLAS and ScaLAPACK to build LibRPA
# on macOS for develop and test use. LibRI and Fortran binding can be switched on or off.
# Tested for Sequoia, Sonoma and Ventura.

# GCC compilers (g++, gfortran), MPICH and OpenBLAS are installed using homebrew.
# ScaLAPACK needs to be installed manually, because the homebrew ScaLAPACK is
# built on OpenMPI.

BUILDDIR="${BUILDDIR:=build_macos_gcc_mpich}"

# Switch for LibRI
export USE_LIBRI="${USE_LIBRI:=ON}"
# Switch for Fortran binding
export ENABLE_FORTRAN_BIND="${ENABLE_FORTRAN_BIND:=OFF}"

# Optionally, one can specify the path of their own LibRI and LibComm libraries.
# If unset or empty, CMake prepares the pinned managed sources.
# See docs/user_guide/install.md for offline preparation and cache settings.
export LIBRI_INCLUDE="${LIBRI_INCLUDE:=}"
export LIBCOMM_INCLUDE="${LIBCOMM_INCLUDE:=}"

# The following two variables need customize
export OPENBLAS_DIR="/opt/homebrew/Cellar/openblas/0.3.28/lib"
export SCALAPACK_DIR="/opt/packages/scalapack/2.2.0/gcc-14.2.0-mpich-4.2.2-openblas"

# Compilers setup
export CXX=mpicxx
export MPICH_CXX=g++-14
export CC=gcc-14

if [[ $ENABLE_FORTRAN_BIND == "ON" ]]; then
  export FC=mpifort
  export MPICH_FC=gfortran-14
else
  export FC=gfortran-14
fi

if [[ $USE_LIBRI == "ON" ]]; then
  BUILDDIR="${BUILDDIR}_libri"
fi

# Note: -Wl,-ld_classic may not be necessary for Sequoia.
export FCFLAGS="-Wl,-ld_classic"
export CXXFLAGS="-cxx=g++-14 -g -rdynamic -Wl,-lgfortran -Wl,-ld_classic"

cmake -B "$BUILDDIR" \
  -DLIBRPA_USE_LIBRI=$USE_LIBRI \
  -DLIBRPA_ENABLE_TEST=ON \
  -DLIBRPA_ENABLE_FORTRAN_BIND=$ENABLE_FORTRAN_BIND \
  -DLIBRI_INCLUDE_DIR="$LIBRI_INCLUDE" \
  -DLIBCOMM_INCLUDE_DIR="$LIBCOMM_INCLUDE" \
  -DBLAS_LIBRARIES="-L${OPENBLAS_DIR} -lblas" \
  -DLAPACK_LIBRARIES="-L${OPENBLAS_DIR} -llapack" \
  -DSCALAPACK_DIR="$SCALAPACK_DIR" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_Fortran_FLAGS="$FCFLAGS"

cmake --build "$BUILDDIR" -j 4
