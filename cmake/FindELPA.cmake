# Find ELPA headers, Fortran modules and library from an installation prefix.
#
# Variables accepted as hints:
#   EXTERNAL_ELPA_DIR - ELPA installation prefix
#   ELPA_ROOT         - ELPA installation prefix
#
# Variables provided:
#   ELPA_FOUND        - True if ELPA was found
#   ELPA_INCLUDE_DIR  - Directory containing ELPA C headers
#   ELPA_MODULE_DIR   - Directory containing ELPA Fortran module files, if found
#   ELPA_INCLUDE_DIRS - Include directories for ELPA headers and modules
#   ELPA_LIBRARY      - ELPA library
#   ELPA_LIBRARIES    - ELPA libraries
#   ELPA::ELPA        - Imported target

set(_ELPA_ROOT_HINTS)
if(EXTERNAL_ELPA_DIR)
  list(APPEND _ELPA_ROOT_HINTS "${EXTERNAL_ELPA_DIR}")
elseif(DEFINED ENV{EXTERNAL_ELPA_DIR} AND NOT "$ENV{EXTERNAL_ELPA_DIR}" STREQUAL "")
  list(APPEND _ELPA_ROOT_HINTS "$ENV{EXTERNAL_ELPA_DIR}")
elseif(ELPA_ROOT)
  list(APPEND _ELPA_ROOT_HINTS "${ELPA_ROOT}")
elseif(DEFINED ENV{ELPA_ROOT} AND NOT "$ENV{ELPA_ROOT}" STREQUAL "")
  list(APPEND _ELPA_ROOT_HINTS "$ENV{ELPA_ROOT}")
endif()

set(_ELPA_FIND_OPTIONS)
if(_ELPA_ROOT_HINTS)
  list(REMOVE_DUPLICATES _ELPA_ROOT_HINTS)
  # An explicit prefix must not fall back to another installation or cached path.
  set(_ELPA_FIND_OPTIONS NO_DEFAULT_PATH)
  foreach(_variable ELPA_INCLUDE_DIR ELPA_MODULE_DIR ELPA_LIBRARY)
    unset(${_variable} CACHE)
    unset(${_variable})
  endforeach()
endif()

set(_ELPA_INCLUDE_HINTS)
foreach(_elpa_root IN LISTS _ELPA_ROOT_HINTS)
  list(APPEND _ELPA_INCLUDE_HINTS
    "${_elpa_root}"
    "${_elpa_root}/include"
  )
  file(GLOB _ELPA_VERSIONED_INCLUDE_DIRS
    LIST_DIRECTORIES true
    "${_elpa_root}/include/elpa*"
  )
  list(APPEND _ELPA_INCLUDE_HINTS ${_ELPA_VERSIONED_INCLUDE_DIRS})
endforeach()

message(STATUS "_ELPA_INCLUDE_HINTS: ${_ELPA_INCLUDE_HINTS}")

find_path(ELPA_INCLUDE_DIR
  ${_ELPA_FIND_OPTIONS}
  NAMES
    elpa/elpa.h
    elpa.h
  HINTS
    ${_ELPA_INCLUDE_HINTS}
  PATH_SUFFIXES
    include
)

set(_ELPA_MODULE_HINTS)
foreach(_elpa_include_dir IN LISTS _ELPA_INCLUDE_HINTS)
  list(APPEND _ELPA_MODULE_HINTS
    "${_elpa_include_dir}"
    "${_elpa_include_dir}/modules"
  )
endforeach()

if(ELPA_INCLUDE_DIR)
  list(APPEND _ELPA_MODULE_HINTS
    "${ELPA_INCLUDE_DIR}"
    "${ELPA_INCLUDE_DIR}/modules"
  )
endif()

find_path(ELPA_MODULE_DIR
  ${_ELPA_FIND_OPTIONS}
  NAMES
    elpa.mod
  HINTS
    ${_ELPA_MODULE_HINTS}
  PATH_SUFFIXES
    include
    modules
)

find_library(ELPA_LIBRARY
  ${_ELPA_FIND_OPTIONS}
  NAMES
    elpa_openmp
    elpa
  HINTS
    ${_ELPA_ROOT_HINTS}
  PATH_SUFFIXES
    lib
    lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ELPA
  FOUND_VAR
    ELPA_FOUND
  REQUIRED_VARS
    ELPA_INCLUDE_DIR
    ELPA_LIBRARY
)

if(ELPA_FOUND)
  set(ELPA_INCLUDE_DIRS "${ELPA_INCLUDE_DIR}")
  if(ELPA_MODULE_DIR)
    list(APPEND ELPA_INCLUDE_DIRS "${ELPA_MODULE_DIR}")
  endif()
  set(ELPA_LIBRARIES "${ELPA_LIBRARY}")
  set(_ELPA_INTERFACE_LINK_LIBRARIES)
  if(TARGET MPI::MPI_Fortran)
    list(APPEND _ELPA_INTERFACE_LINK_LIBRARIES MPI::MPI_Fortran)
  endif()
  if(ELPA_LIBRARY MATCHES "elpa_openmp" AND TARGET OpenMP::OpenMP_Fortran)
    list(APPEND _ELPA_INTERFACE_LINK_LIBRARIES OpenMP::OpenMP_Fortran)
  endif()

  if(NOT TARGET ELPA::ELPA)
    add_library(ELPA::ELPA UNKNOWN IMPORTED)
    set_target_properties(ELPA::ELPA PROPERTIES
      IMPORTED_LOCATION "${ELPA_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${ELPA_INCLUDE_DIRS}"
    )
  endif()
  if(_ELPA_INTERFACE_LINK_LIBRARIES)
    set_property(TARGET ELPA::ELPA PROPERTY
      INTERFACE_LINK_LIBRARIES "${_ELPA_INTERFACE_LINK_LIBRARIES}"
    )
    list(APPEND ELPA_LIBRARIES ${_ELPA_INTERFACE_LINK_LIBRARIES})
  endif()
endif()

mark_as_advanced(ELPA_INCLUDE_DIR ELPA_MODULE_DIR ELPA_LIBRARY)

unset(_ELPA_ROOT_HINTS)
unset(_ELPA_INCLUDE_HINTS)
unset(_ELPA_MODULE_HINTS)
unset(_ELPA_INTERFACE_LINK_LIBRARIES)
unset(_ELPA_VERSIONED_INCLUDE_DIRS)

unset(_ELPA_FIND_OPTIONS)
