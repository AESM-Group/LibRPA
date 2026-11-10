# - Find mkl
# Find the native MKL headers and libraries.
#
#  MKL_INCLUDE_DIRS - where to find mkl.h, etc.
#  MKL_LIBRARIES    - List of libraries when using mkl.
#  MKL_FOUND        - True if mkl found.
#
# The MKL threading layer must match the OpenMP runtime, which LibRPA links
# through OpenMP::OpenMP_CXX. GCC provides libgomp, which needs
# mkl_gnu_thread. The Intel and LLVM compilers provide libiomp5 or the
# compatible libomp, which need mkl_intel_thread.
#

find_path(MKL_INCLUDE_DIR mkl_dfti.h HINTS ${MKLROOT}/include)

find_library(MKL_INTEL NAMES mkl_intel_lp64 HINTS ${MKLROOT}/lib ${MKLROOT}/lib/intel64)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(_mkl_thread_name mkl_gnu_thread)
else()
  set(_mkl_thread_name mkl_intel_thread)
endif()
find_library(MKL_THREAD NAMES ${_mkl_thread_name} HINTS ${MKLROOT}/lib ${MKLROOT}/lib/intel64)
find_library(MKL_CORE NAMES mkl_core HINTS ${MKLROOT}/lib ${MKLROOT}/lib/intel64)
find_library(MKL_SCALAPACK NAMES mkl_scalapack_lp64 HINTS ${MKLROOT}/lib ${MKLROOT}/lib/intel64)
# Test the selected MPI headers, not the compiler wrapper's filename.
# Intel MPI and MPICH use the Intel-MPI BLACS ABI; Open MPI has its own.
include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
cmake_push_check_state(RESET)
set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
unset(_mkl_mpi_is_openmpi CACHE)
unset(_mkl_mpi_is_mpich CACHE)
check_cxx_source_compiles("#include <mpi.h>
#ifndef OMPI_MAJOR_VERSION
#error Not Open MPI
#endif
int main() { return 0; }" _mkl_mpi_is_openmpi)
check_cxx_source_compiles("#include <mpi.h>
#if !defined(MPICH_VERSION) && !defined(I_MPI_VERSION)
#error Not MPICH or Intel MPI
#endif
int main() { return 0; }" _mkl_mpi_is_mpich)
cmake_pop_check_state()
if(_mkl_mpi_is_openmpi)
  set(_mkl_blacs_name mkl_blacs_openmpi_lp64)
elseif(_mkl_mpi_is_mpich)
  set(_mkl_blacs_name mkl_blacs_intelmpi_lp64)
else()
  message(FATAL_ERROR "MKL BLACS: cannot identify the selected MPI as Open MPI, MPICH or Intel MPI. Use a supported MPI toolchain or supply separate LAPACK/ScaLAPACK libraries with -DMKLROOT=.")
endif()
# Re-select on reconfiguration, including when the MPI headers change.
unset(MKL_BLACS CACHE)
find_library(MKL_BLACS NAMES ${_mkl_blacs_name} HINTS ${MKLROOT}/lib ${MKLROOT}/lib/intel64)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE
# if all listed variables are TRUE

find_package_handle_standard_args(IntelMKL DEFAULT_MSG MKL_INTEL MKL_THREAD MKL_CORE MKL_SCALAPACK MKL_BLACS MKL_INCLUDE_DIR)

if(IntelMKL_FOUND)
  set(MKL_LIBRARIES ${MKL_INTEL} ${MKL_THREAD} ${MKL_CORE} ${MKL_SCALAPACK} ${MKL_BLACS})
  set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})

  if(NOT TARGET IntelMKL::INTEL)
    add_library(IntelMKL::INTEL UNKNOWN IMPORTED)
    set_target_properties(IntelMKL::INTEL PROPERTIES
      IMPORTED_LOCATION "${MKL_INTEL}"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
  endif()
  if(NOT TARGET IntelMKL::THREAD)
    add_library(IntelMKL::THREAD UNKNOWN IMPORTED)
    set_target_properties(IntelMKL::THREAD PROPERTIES
      IMPORTED_LOCATION "${MKL_THREAD}"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
  endif()
  if(NOT TARGET IntelMKL::CORE)
    add_library(IntelMKL::CORE UNKNOWN IMPORTED)
    set_target_properties(IntelMKL::CORE PROPERTIES
      IMPORTED_LOCATION "${MKL_CORE}"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
  endif()
  if(NOT TARGET IntelMKL::SCALAPACK)
    add_library(IntelMKL::SCALAPACK UNKNOWN IMPORTED)
    set_target_properties(IntelMKL::SCALAPACK PROPERTIES
      IMPORTED_LOCATION "${MKL_SCALAPACK}"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
  endif()
  if(NOT TARGET IntelMKL::BLACS)
    add_library(IntelMKL::BLACS UNKNOWN IMPORTED)
    set_target_properties(IntelMKL::BLACS PROPERTIES
      IMPORTED_LOCATION "${MKL_BLACS}"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
  endif()
  add_library(IntelMKL::MKL INTERFACE IMPORTED)
  set_property(TARGET IntelMKL::MKL PROPERTY
  INTERFACE_LINK_LIBRARIES
  "-Wl,--start-group"
  IntelMKL::INTEL IntelMKL::THREAD IntelMKL::CORE IntelMKL::SCALAPACK IntelMKL::BLACS
  "-Wl,--end-group"
  )
endif()

mark_as_advanced(MKL_INCLUDE_DIR MKL_INTEL MKL_THREAD MKL_CORE MKL_SCALAPACK MKL_BLACS)
