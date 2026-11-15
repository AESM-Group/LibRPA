include_guard(GLOBAL)

# Installed DDLA 0.0.6 packages can predate the APIs used by this source.
# Check declarations AND exported symbols instead of accepting the version alone.
function(librpa_check_ddla_capabilities ddla_target)
  if(LIBRPA_BUNDLED_DDLA_PROJECT_TARGET)
    # ExternalProject's installed library does not exist during configure.
    return()
  endif()
  get_target_property(_aliased_target "${ddla_target}" ALIASED_TARGET)
  if(_aliased_target)
    set(ddla_target "${_aliased_target}")
  endif()
  get_target_property(_imported "${ddla_target}" IMPORTED)
  if(NOT _imported)
    # A caller may add_subdirectory(LibDDLA) before LibRPA. Its source target
    # cannot link in a configure-time try_compile before it has been built.
    message(STATUS "LibDDLA capability check: source target; checked by the consumer build")
    return()
  endif()

  include(CheckCXXSourceCompiles)
  include(CMakePushCheckState)
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_LIBRARIES "${ddla_target}")
  # Function scope preserves the parent's cross-compile choice and flags.
  # A static try_compile would incorrectly accept a stale installed library.
  set(CMAKE_TRY_COMPILE_TARGET_TYPE EXECUTABLE)
  if(NOT CMAKE_CXX_STANDARD OR CMAKE_CXX_STANDARD LESS 17)
    set(CMAKE_CXX_STANDARD 17)
  endif()
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  file(READ "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/checks/ddla_capabilities.cpp" _probe)
  # Do not reuse success after the user changes an installation in place.
  unset(_librpa_ddla_capable CACHE)
  unset(_librpa_ddla_capable)
  check_cxx_source_compiles("${_probe}" _librpa_ddla_capable)
  cmake_pop_check_state()
  if(NOT _librpa_ddla_capable)
    message(FATAL_ERROR
      "The external LibDDLA target '${ddla_target}' lacks the required declarations or linkable APIs: "
      "diag_logdet/pdiag_logdet, gemm_mixed_workspace_bytes, explicit-workspace mixed GEMM, "
      "and mixed gemmVbatched/gemmVbatched2s. An older 0.0.6 installation is insufficient. "
      "Rebuild/install the LibDDLA revision required by this LibRPA source, select that prefix "
      "with LIBDDLA_PATH, or remove LIBDDLA_PATH and supplied DDLA targets to use the managed dependency. "
      "See the CMake configure error log for the failed compile/link probe.")
  endif()
endfunction()
