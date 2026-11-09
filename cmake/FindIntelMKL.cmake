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
find_library(MKL_BLACS_INTELMPI NAMES mkl_blacs_intelmpi_lp64 HINTS ${MKLROOT}/lib ${MKLROOT}/lib/intel64)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set MKL_FOUND to TRUE
# if all listed variables are TRUE

find_package_handle_standard_args(IntelMKL DEFAULT_MSG MKL_INTEL MKL_THREAD MKL_CORE MKL_SCALAPACK MKL_BLACS_INTELMPI MKL_INCLUDE_DIR)

if(IntelMKL_FOUND)
  set(MKL_LIBRARIES ${MKL_INTEL} ${MKL_THREAD} ${MKL_CORE} ${MKL_SCALAPACK} ${MKL_BLACS_INTELMPI})
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
  if(NOT TARGET IntelMKL::BLACS_INTELMPI)
    add_library(IntelMKL::BLACS_INTELMPI UNKNOWN IMPORTED)
    set_target_properties(IntelMKL::BLACS_INTELMPI PROPERTIES
      IMPORTED_LOCATION "${MKL_BLACS_INTELMPI}"
      INTERFACE_INCLUDE_DIRECTORIES "${MKL_INCLUDE_DIR}")
  endif()
  add_library(IntelMKL::MKL INTERFACE IMPORTED)
  set_property(TARGET IntelMKL::MKL PROPERTY
  INTERFACE_LINK_LIBRARIES
  "-Wl,--start-group"
  IntelMKL::INTEL IntelMKL::THREAD IntelMKL::CORE IntelMKL::SCALAPACK IntelMKL::BLACS_INTELMPI
  "-Wl,--end-group"
  )
endif()

mark_as_advanced(MKL_INCLUDE_DIR MKL_INTEL MKL_THREAD MKL_CORE MKL_SCALAPACK MKL_BLACS_INTELMPI)
