# These settings apply to managed and user-supplied ELPA source builds.
# Preserve explicit new values, including OFF and empty strings.
foreach(_suffix KERNEL OPENMP CONFIGURE_ARGS LIBS)
  set(_option "LIBRPA_ELPA_${_suffix}")
  set(_legacy "LIBRPA_BUNDLED_ELPA_${_suffix}")
  if(DEFINED ${_legacy})
    message(DEPRECATION "${_legacy} is deprecated. Use ${_option} instead, and remove ${_legacy} from the CMake cache.")
    if(NOT DEFINED ${_option})
      set(${_option} "${${_legacy}}")
    endif()
  endif()
endforeach()

if(DEFINED LIBRPA_BUNDLED_ELPA_VERSION)
  message(DEPRECATION
    "LIBRPA_BUNDLED_ELPA_VERSION is deprecated and ignored. The manifest selects "
    "the managed release; ELPA_SOURCE_DIR selects a supplied source tree. "
    "Remove LIBRPA_BUNDLED_ELPA_VERSION from the CMake cache.")
endif()

if(NOT DEFINED LIBRPA_ELPA_OPENMP)
  set(LIBRPA_ELPA_OPENMP OFF)
endif()
set(LIBRPA_ELPA_OPENMP "${LIBRPA_ELPA_OPENMP}" CACHE BOOL "Build ELPA sources with OpenMP support")
set(LIBRPA_ELPA_KERNEL "${LIBRPA_ELPA_KERNEL}" CACHE STRING "ELPA source build x86 SIMD kernel family")
set_property(CACHE LIBRPA_ELPA_KERNEL PROPERTY STRINGS "" "SSE" "SSE_ASSEMBLY" "AVX" "AVX2" "AVX512")
set(LIBRPA_ELPA_CONFIGURE_ARGS "${LIBRPA_ELPA_CONFIGURE_ARGS}" CACHE STRING "Extra configure arguments for ELPA source builds")
set(LIBRPA_ELPA_LIBS "${LIBRPA_ELPA_LIBS}" CACHE STRING "Libraries passed to ELPA source configure through LIBS")
