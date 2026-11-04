# Autoconf rejects whitespace in source and build paths.
foreach(_elpa_path IN ITEMS "${_elpa_source_dir}" "${CMAKE_CURRENT_BINARY_DIR}")
  if(_elpa_path MATCHES "[ \t\r\n]")
    message(FATAL_ERROR "ELPA source builds do not support whitespace in paths: '${_elpa_path}'. Use a LibRPA checkout (or ELPA_SOURCE_DIR) and a CMake build directory without spaces or other whitespace.")
  endif()
endforeach()

if(NOT EXISTS "${_elpa_source_dir}/configure")
  message(FATAL_ERROR "ELPA source directory is incomplete: ${_elpa_source_dir}")
endif()

# The selected source determines the installed header layout and configure flags.
file(STRINGS "${_elpa_source_dir}/configure" _elpa_version_line
  REGEX [=[^PACKAGE_VERSION=['"]([^'"]+)['"]$]=] LIMIT_COUNT 1)
if(NOT _elpa_version_line)
  message(FATAL_ERROR "Cannot read PACKAGE_VERSION from ${_elpa_source_dir}/configure")
endif()
string(REGEX REPLACE [=[^PACKAGE_VERSION=['"]([^'"]+)['"]$]=] "\\1"
  _elpa_source_version "${_elpa_version_line}")
set(_elpa_binary_dir "${CMAKE_CURRENT_BINARY_DIR}/build-${_elpa_source_version}")
message(STATUS "ELPA source version: ${_elpa_source_version}")

include(ExternalProject)
if(TARGET elpa)
  message(FATAL_ERROR "Cannot build ELPA sources because target 'elpa' already exists")
endif()

# Find the tools needed by ELPA's autotools build.
find_program(_librpa_elpa_make_executable NAMES gmake make)
if(NOT _librpa_elpa_make_executable)
  message(FATAL_ERROR "ELPA source builds require a make executable")
endif()

find_program(_librpa_elpa_bash_executable NAMES bash)
if(NOT _librpa_elpa_bash_executable)
  message(FATAL_ERROR "ELPA source builds require bash for its Fortran dependency preprocessor wrapper")
endif()

find_program(_librpa_elpa_cpp_executable NAMES cpp-13 cpp)
if(NOT _librpa_elpa_cpp_executable)
  message(FATAL_ERROR "ELPA source builds require a C preprocessor executable named cpp")
endif()

# Tool discovery is internal; these paths are not user-facing build options.
foreach(_tool make bash cpp)
  set(_variable "_librpa_elpa_${_tool}_executable")
  set(${_variable} "${${_variable}}" CACHE INTERNAL "ELPA source build ${_tool} executable" FORCE)
endforeach()

# Keep the ELPA install private to this build tree and mirror the installed
# include/module layout expected by ELPA clients.
set(_elpa_prefix "${CMAKE_CURRENT_BINARY_DIR}/install")
set(_elpa_libdir "${_elpa_prefix}/${CMAKE_INSTALL_LIBDIR}")
set(_elpa_includedir "${_elpa_prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
set(_elpa_library_base elpa)
if(LIBRPA_ELPA_OPENMP)
  set(_elpa_library_base elpa_openmp)
endif()
set(_elpa_versioned_includedir "${_elpa_includedir}/${_elpa_library_base}-${_elpa_source_version}")
set(_elpa_moduledir "${_elpa_versioned_includedir}/modules")
set(_elpa_libtool_library "lib${_elpa_library_base}.la")
set(_elpa_library "${_elpa_libdir}/${CMAKE_STATIC_LIBRARY_PREFIX}${_elpa_library_base}${CMAKE_STATIC_LIBRARY_SUFFIX}")

file(MAKE_DIRECTORY
  "${_elpa_libdir}"
  "${_elpa_versioned_includedir}"
  "${_elpa_moduledir}"
)

# Prefer MPI compiler wrappers for ELPA so its configure script sees the same
# MPI setup that LibRPA uses.
set(_elpa_c_compiler "${CMAKE_C_COMPILER}")
if(MPI_C_COMPILER)
  set(_elpa_c_compiler "${MPI_C_COMPILER}")
endif()

set(_elpa_cxx_compiler "${CMAKE_CXX_COMPILER}")
if(MPI_CXX_COMPILER)
  set(_elpa_cxx_compiler "${MPI_CXX_COMPILER}")
endif()

set(_elpa_fortran_compiler "${CMAKE_Fortran_COMPILER}")
if(MPI_Fortran_COMPILER)
  set(_elpa_fortran_compiler "${MPI_Fortran_COMPILER}")
endif()

set(_elpa_c_flags "${CMAKE_C_FLAGS}")
set(_elpa_cxx_flags "${CMAKE_CXX_FLAGS}")
set(_elpa_fortran_flags "${CMAKE_Fortran_FLAGS}")

# Base ELPA configure flags: build a static, position-independent MPI library
# with only the pieces LibRPA needs.
set(_elpa_configure_args
  "--prefix=${_elpa_prefix}"
  "--libdir=${_elpa_libdir}"
  "--includedir=${_elpa_includedir}"
  "--with-mpi=yes"
  "--disable-shared"
  "--enable-static"
  "--with-pic"
  "--disable-dependency-tracking"
  "--disable-doxygen-doc"
  "--disable-c-tests"
  "--disable-python"
)

# Configure ELPA's own OpenMP support separately from LibRPA's OpenMP usage.
if(LIBRPA_ELPA_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS C CXX Fortran)
  list(APPEND _elpa_configure_args
    "--enable-openmp"
    "--enable-runtime-threading-support-checks"
    "--enable-allow-thread-limiting"
    "--without-threading-support-check-during-build"
  )
  string(APPEND _elpa_c_flags " ${OpenMP_C_FLAGS}")
  string(APPEND _elpa_cxx_flags " ${OpenMP_CXX_FLAGS}")
  string(APPEND _elpa_fortran_flags " ${OpenMP_Fortran_FLAGS}")
  message(STATUS "Use ELPA source OpenMP          : ON")
else()
  list(APPEND _elpa_configure_args "--disable-openmp")
  message(STATUS "Use ELPA source OpenMP          : OFF")
endif()

# Select x86-specific ELPA kernels only when explicitly requested. By default,
# use portable generic kernels, which also works on non-x86 platforms.
set(_elpa_supported_x86_kernels SSE SSE_ASSEMBLY AVX AVX2 AVX512)
string(STRIP "${LIBRPA_ELPA_KERNEL}" _elpa_x86_kernel)
string(TOUPPER "${_elpa_x86_kernel}" _elpa_x86_kernel)
string(REPLACE "-" "_" _elpa_x86_kernel "${_elpa_x86_kernel}")
if(_elpa_x86_kernel AND NOT _elpa_x86_kernel IN_LIST _elpa_supported_x86_kernels)
  message(FATAL_ERROR
    "Unsupported LIBRPA_ELPA_KERNEL='${LIBRPA_ELPA_KERNEL}'. "
    "Supported values: empty, ${_elpa_supported_x86_kernels}")
endif()

if(_elpa_source_version VERSION_LESS 2026.02.001)
  set(_elpa_disable_x86_kernel_args
    "--disable-sse"
    "--disable-sse-assembly"
    "--disable-avx"
    "--disable-avx2"
    "--disable-avx512"
  )
else()
  set(_elpa_disable_x86_kernel_args
    "--disable-sse-kernels"
    "--disable-sse-assembly-kernels"
    "--disable-avx-kernels"
    "--disable-avx2-kernels"
    "--disable-avx512-kernels"
  )
endif()
list(APPEND _elpa_configure_args ${_elpa_disable_x86_kernel_args})

if(_elpa_x86_kernel)
  string(TOLOWER "${_elpa_x86_kernel}" _elpa_x86_kernel_flag)
  string(REPLACE "_" "-" _elpa_x86_kernel_flag "${_elpa_x86_kernel_flag}")
  if(_elpa_source_version VERSION_LESS 2026.02.001)
    list(APPEND _elpa_configure_args "--enable-${_elpa_x86_kernel_flag}")
  else()
    list(APPEND _elpa_configure_args "--enable-${_elpa_x86_kernel_flag}-kernels")
  endif()
  message(STATUS "Use ELPA source x86 kernel     : ${_elpa_x86_kernel}")
else()
  message(STATUS "Use ELPA source x86 kernel     : none")
endif()

# Let advanced users pass additional configure arguments after LibRPA's
# defaults so those arguments can override the defaults when needed.
if(LIBRPA_ELPA_CONFIGURE_ARGS)
  separate_arguments(_elpa_extra_configure_args UNIX_COMMAND "${LIBRPA_ELPA_CONFIGURE_ARGS}")
  list(APPEND _elpa_configure_args ${_elpa_extra_configure_args})
endif()

# ELPA's Fortran dependency generator calls cpp with "-o /dev/stdout", which is
# not portable on macOS. This wrapper rewrites that output path to stdout.
set(_elpa_fortran_cpp_wrapper "${CMAKE_CURRENT_BINARY_DIR}/elpa-fortran-cpp")
set(_elpa_fortran_cpp_wrapper_in "${CMAKE_CURRENT_BINARY_DIR}/elpa-fortran-cpp.in")
file(WRITE "${_elpa_fortran_cpp_wrapper_in}" [=[
#!/usr/bin/env bash
set -euo pipefail

args=()
while (($#)); do
  if [[ "$1" == "-o" && "${2:-}" == "/dev/stdout" ]]; then
    args+=("-o" "-")
    shift 2
  else
    args+=("$1")
    shift
  fi
done

exec "@_librpa_elpa_cpp_executable@" -P -traditional -w "${args[@]}"
]=])
configure_file(
  "${_elpa_fortran_cpp_wrapper_in}"
  "${_elpa_fortran_cpp_wrapper}"
  @ONLY
)

# Avoid inheriting recursive make flags from LibRPA and force ELPA to use the
# wrapper above during its generated Fortran dependency scan.
set(_elpa_make_environment
  "MAKEFLAGS="
  "FORTRAN_CPP=${_librpa_elpa_bash_executable} ${_elpa_fortran_cpp_wrapper}"
)

# Drive the upstream ELPA autotools build inside CMake. The install command is
# intentionally limited to the library, headers/modules, and pkg-config file.
ExternalProject_Add(librpa_bundled_elpa
  SOURCE_DIR "${_elpa_source_dir}"
  BINARY_DIR "${_elpa_binary_dir}"
  INSTALL_DIR "${_elpa_prefix}"
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E env
      "CC=${_elpa_c_compiler}"
      "CXX=${_elpa_cxx_compiler}"
      "FC=${_elpa_fortran_compiler}"
      "CFLAGS=${_elpa_c_flags}"
      "CXXFLAGS=${_elpa_cxx_flags}"
      "FCFLAGS=${_elpa_fortran_flags}"
      "LIBS=${LIBRPA_ELPA_LIBS}"
      "${_elpa_source_dir}/configure" ${_elpa_configure_args}
  BUILD_COMMAND
    ${CMAKE_COMMAND} -E env ${_elpa_make_environment}
      "${_librpa_elpa_make_executable}"
      "ACLOCAL=true"
      "AUTOCONF=true"
      "AUTOHEADER=true"
      "AUTOMAKE=true"
      "${_elpa_libtool_library}"
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E env ${_elpa_make_environment}
      "${_librpa_elpa_make_executable}"
      "ACLOCAL=true"
      "AUTOCONF=true"
      "AUTOHEADER=true"
      "AUTOMAKE=true"
      install-libLTLIBRARIES
      install-nobase_elpa_includeHEADERS
      install-nobase_nodist_elpa_includeHEADERS
      install-pkgconfigDATA
  BUILD_BYPRODUCTS "${_elpa_library}"
)

# Some libtool configurations copy static dependency archives, such as
# libscalapack.a, into libelpa.a. Apple ld rejects those nested archives when
# linking shared LibRPA, so generate a small script to remove them after install.
set(_elpa_prune_nested_archives_script
  "${CMAKE_CURRENT_BINARY_DIR}/prune-elpa-nested-archives.cmake")
set(_elpa_prune_nested_archives_script_in
  "${CMAKE_CURRENT_BINARY_DIR}/prune-elpa-nested-archives.cmake.in")
file(WRITE "${_elpa_prune_nested_archives_script_in}" [=[
set(_elpa_library "@_elpa_library@")
set(_elpa_ar "@CMAKE_AR@")
set(_elpa_ranlib "@CMAKE_RANLIB@")

if(NOT EXISTS "${_elpa_library}")
  message(FATAL_ERROR "Managed ELPA library was not found: ${_elpa_library}")
endif()

execute_process(
  COMMAND "${_elpa_ar}" -t "${_elpa_library}"
  OUTPUT_VARIABLE _elpa_archive_members_output
  RESULT_VARIABLE _elpa_archive_list_result
  ERROR_VARIABLE _elpa_archive_list_error
)
if(NOT _elpa_archive_list_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to inspect managed ELPA archive '${_elpa_library}': "
    "${_elpa_archive_list_error}")
endif()

string(REGEX MATCHALL "[^\r\n]+" _elpa_archive_members
  "${_elpa_archive_members_output}")
set(_elpa_nested_archives)
foreach(_elpa_archive_member IN LISTS _elpa_archive_members)
  if(_elpa_archive_member MATCHES "\\.a$")
    list(APPEND _elpa_nested_archives "${_elpa_archive_member}")
  endif()
endforeach()

if(_elpa_nested_archives)
  message(STATUS
    "Removing nested archives from managed ELPA library: "
    "${_elpa_nested_archives}")
  execute_process(
    COMMAND "${_elpa_ar}" -d "${_elpa_library}" ${_elpa_nested_archives}
    RESULT_VARIABLE _elpa_archive_prune_result
    ERROR_VARIABLE _elpa_archive_prune_error
  )
  if(NOT _elpa_archive_prune_result EQUAL 0)
    message(FATAL_ERROR
      "Failed to prune managed ELPA archive '${_elpa_library}': "
      "${_elpa_archive_prune_error}")
  endif()

  if(_elpa_ranlib)
    execute_process(
      COMMAND "${_elpa_ranlib}" "${_elpa_library}"
      RESULT_VARIABLE _elpa_ranlib_result
      ERROR_VARIABLE _elpa_ranlib_error
    )
    if(NOT _elpa_ranlib_result EQUAL 0)
      message(FATAL_ERROR
        "Failed to index managed ELPA archive '${_elpa_library}': "
        "${_elpa_ranlib_error}")
    endif()
  endif()
endif()
]=])
configure_file(
  "${_elpa_prune_nested_archives_script_in}"
  "${_elpa_prune_nested_archives_script}"
  @ONLY
)

# Run the pruning script as a real build target so rpa_lib can depend on the
# cleaned archive instead of the raw ExternalProject output.
set(_elpa_prune_stamp
  "${CMAKE_CURRENT_BINARY_DIR}/librpa_bundled_elpa_pruned.stamp")
add_custom_command(
  OUTPUT "${_elpa_prune_stamp}"
  COMMAND "${CMAKE_COMMAND}" -P "${_elpa_prune_nested_archives_script}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${_elpa_prune_stamp}"
  DEPENDS librpa_bundled_elpa "${_elpa_library}"
  VERBATIM
)
add_custom_target(librpa_bundled_elpa_pruned DEPENDS "${_elpa_prune_stamp}")

# Present the managed install as a normal imported CMake target. This hides the
# ExternalProject details from the rest of LibRPA.
add_library(elpa STATIC IMPORTED GLOBAL)
add_dependencies(elpa librpa_bundled_elpa_pruned)
set_target_properties(elpa PROPERTIES
  IMPORTED_LOCATION "${_elpa_library}"
  INTERFACE_INCLUDE_DIRECTORIES "${_elpa_versioned_includedir};${_elpa_moduledir}"
)

# ELPA itself remains a static archive; its MPI and math dependencies are linked
# separately through this target interface.
set(_elpa_interface_link_libraries)
if(TARGET MPI::MPI_C)
  list(APPEND _elpa_interface_link_libraries MPI::MPI_C)
endif()
if(TARGET MPI::MPI_Fortran)
  list(APPEND _elpa_interface_link_libraries MPI::MPI_Fortran)
endif()
if(LIBRPA_ELPA_OPENMP)
  foreach(_elpa_openmp_target IN ITEMS
      OpenMP::OpenMP_C
      OpenMP::OpenMP_CXX
      OpenMP::OpenMP_Fortran)
    if(TARGET ${_elpa_openmp_target})
      list(APPEND _elpa_interface_link_libraries ${_elpa_openmp_target})
    endif()
  endforeach()
endif()
list(APPEND _elpa_interface_link_libraries ${math_libs})

if(_elpa_interface_link_libraries)
  set_property(TARGET elpa PROPERTY
    INTERFACE_LINK_LIBRARIES "${_elpa_interface_link_libraries}")
endif()

# Export the target and include paths to the parent directory using the same
# variables as external ELPA discovery.
set(ELPA_INCLUDE_DIRS
  "${_elpa_versioned_includedir}"
  "${_elpa_moduledir}"
  PARENT_SCOPE)
set(ELPA_LIBRARIES elpa PARENT_SCOPE)
set(LIBRPA_BUNDLED_ELPA_PROJECT_TARGET librpa_bundled_elpa_pruned PARENT_SCOPE)
