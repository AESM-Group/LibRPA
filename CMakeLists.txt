# main CMake file for LibRPA
cmake_minimum_required(VERSION 3.13)

project(librpa NONE)

# extract version number from version.h file
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/src/version.h version_file)
string(REGEX MATCH "LIBRPA_MAJOR_VERSION ([0-9]+)" librpa_major_version ${version_file})
set(librpa_major_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBRPA_MINOR_VERSION ([0-9]+)" librpa_minor_version ${version_file})
set(librpa_minor_version ${CMAKE_MATCH_1})
string(REGEX MATCH "LIBRPA_PATCH_VERSION ([0-9]+)" librpa_patch_version ${version_file})
set(librpa_patch_version ${CMAKE_MATCH_1})
set(version "${librpa_major_version}.${librpa_minor_version}.${librpa_patch_version}")

set(CMAKE_PROJECT_VERSION "${version}")

set(LIBRPA_DRIVER_NAME         "chi0_main.exe")

# Public options setup
# Convention:
# - `LIBRPA_USE_` for thirdparty/external components
# - `LIBRPA_ENABLE_` for internal components (binding, test, ...)
option(LIBRPA_USE_LIBRI            "Use LibRI for tensor contraction in LibRPA" ON)
option(LIBRPA_USE_LIBRI_GPU        "Use LibRI GPU for tensor contraction in LibRPA" OFF)
option(LIBRPA_USE_EXTERNAL_ELPA    "Use an external ELPA eigensolver library from an installation prefix" OFF)
option(LIBRPA_USE_BUNDLED_ELPA     "Build and use a bundled ELPA eigensolver library from thirdparty/ELPA" OFF)
option(LIBRPA_BUNDLED_ELPA_OPENMP  "Build bundled ELPA with OpenMP support" OFF)
option(LIBRPA_USE_CMAKE_INC        "Use cmake.inc for configure (deprecated)" OFF)
option(LIBRPA_USE_EXTERNAL_GREENX  "Use an external GreenX target provided by the parent project instead of the bundled copy" OFF)
option(LIBRPA_ENABLE_TEST          "Flag to build LibRPA tests" ON)
option(LIBRPA_ENABLE_DRIVER        "Flag to build LibRPA driver executable" ON)
option(LIBRPA_ENABLE_FORTRAN_BIND  "Flag to build LibRPA Fotran binding" OFF)
option(LIBRPA_VERBOSE_OUTPUT       "Flag to print verbose information in stdout and process output of LibRPA" ON)
option(LIBRPA_USE_CUDA             "Enable the CUDA backend" OFF)
option(LIBRPA_USE_HIP              "Enable the HIP backend" OFF)

# Comonent test switches, only used when LIBRPA_ENABLE_TEST is ON
option(LIBRPA_ENABLE_CPP_TEST      "Flag to build tests for C++ components" ON)
option(LIBRPA_ENABLE_FORTRAN_TEST  "Flag to build tests for Fortran bindings" ON)

# Customizable cmake macro
set(LIBRPA_FORTRAN_DP "c_double" CACHE STRING "Kind of double precision that the Fortran binding uses to interface the user code")
set(LIBRPA_MPI_THREAD_LEVEL "${LIBRPA_MPI_THREAD_LEVEL}" CACHE STRING "MPI thread level requested by LibRPA executables; empty selects automatically")
set_property(CACHE LIBRPA_MPI_THREAD_LEVEL PROPERTY STRINGS "" MPI_THREAD_SINGLE MPI_THREAD_FUNNELED MPI_THREAD_SERIALIZED MPI_THREAD_MULTIPLE)
set(EXTERNAL_ELPA_DIR "" CACHE PATH "External ELPA installation prefix")
set(LIBRPA_BUNDLED_ELPA_VERSION "2026.02.001" CACHE STRING "Bundled ELPA version")
set_property(CACHE LIBRPA_BUNDLED_ELPA_VERSION PROPERTY STRINGS "2026.02.001")
set(LIBRPA_BUNDLED_ELPA_KERNEL "" CACHE STRING "Bundled ELPA x86 SIMD kernel family")
set_property(CACHE LIBRPA_BUNDLED_ELPA_KERNEL PROPERTY STRINGS "" "SSE" "SSE_ASSEMBLY" "AVX" "AVX2" "AVX512")
set(LIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS "" CACHE STRING "Extra configure arguments for bundled ELPA")
set(LIBRPA_BUNDLED_ELPA_LIBS "" CACHE STRING "Libraries passed to bundled ELPA configure through LIBS")

# Deprecation stops
if (DEFINED USE_GREENX_MINIMAX)
  message(FATAL_ERROR "USE_GREENX_MINIMAX is deprecated: GreenX is now always used. Please remove USE_GREENX_MINIMAX")
endif()
if (DEFINED USE_LIBRI)
  message(FATAL_ERROR "USE_LIBRI is deprecated: superceded by LIBRPA_USE_LIBRI. Please remove USE_LIBRI")
endif()
if (DEFINED USE_GREENX_API)
  if (NOT USE_GREENX_API)
    message(FATAL_ERROR "USE_GREENX_API is deprecated: GreenX is now always used. Please remove USE_GREENX_API")
  endif()
endif()
# Hash number of the bundled greenX. Keep it consistent with thirdparty/greenX
set(_BUNDLED_GREENX_COMMIT_HASH  "c90d131")
set(_BUNDLED_LIBRI_COMMIT_HASH   "75902f8")
set(_BUNDLED_LIBRI_GPU_COMMIT_HASH "8f20f4c")
set(_BUNDLED_LIBCOMM_COMMIT_HASH "12457e7")
set(_BUNDLED_LIBDDLA_COMMIT_HASH "0d576d4")

# Add cmake load files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(math_libs "")
set(parallel_libs "")
set(_librpa_elpa_configure_libs "")
set(_loaded_cmake_inc OFF)
set(_LIBRPA_USE_BUNDLED_LIBCOMM OFF)
set(_LIBRPA_USE_BUNDLED_LIBRI_GPU OFF)

# TODO: maybe better way to customize environments?
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake.inc" AND LIBRPA_USE_CMAKE_INC)
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake.inc")
  list(APPEND math_libs ${LAPACK_LIBRARIES})
  list(APPEND _librpa_elpa_configure_libs ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES})
  set(_loaded_cmake_inc ON)
  message(STATUS "Using customize environemnt variables from ${CMAKE_CURRENT_SOURCE_DIR}/cmake.inc")
endif()

# C++ standard requirement
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# enable_language later than cmake.inc to take the variables therein into accout
# require Fortran for GreenX library and Fortran binding
if(LIBRPA_USE_BUNDLED_ELPA)
  # require C for bundled ELPA
  enable_language(CXX C Fortran)
else()
  enable_language(CXX Fortran)
endif()

# NOTE: Enabling HIP here may result in compiler inconsistency for the modules in test.
#       Switch off temporarily. Instead, HIP flags are appended manually.
# if(LIBRPA_USE_HIP)
#   find_package(hip REQUIRED)
#   enable_language(HIP)
# endif()

# bypass the deprecation warning of classic C++ of oneAPI
if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
  if(CMAKE_CXX_COMPILER MATCHES "mpiicpc" OR CMAKE_CXX_COMPILER MATCHES "icpc")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "2021.7.0.20221019")
      message(STATUS "Intel C++ Compiler Classic detected, adding -diag-disable=10441 flag")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=10441")
    endif()
  endif()
endif()

# Setup installation paths
include(GNUInstallDirs)
# Set to install/ when LibRPA is the root project and
# CMAKE_INSTALL_PREFIX was not manually specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND (CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR))
  set(default_install_prefix "${CMAKE_SOURCE_DIR}/install")  # default to install at the root project
  set(CMAKE_INSTALL_PREFIX ${default_install_prefix}
      CACHE STRING "The installation directory is not set, default to ${default_install_prefix}"
      FORCE)
  set(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT OFF)
endif()

# Get git commit number when LibRPA is the root project
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  include(GetGitRevisionDescription)
  get_git_head_revision(LIBRPA_GIT_REF LIBRPA_GIT_SHA1)
else()
  # Otherwise get the checkpoint recorded in the root project
  if (NOT DEFINED LIBRPA_GIT_REF)
    set(LIBRPA_GIT_REF "GIT-REF-UNKNOWN")
  endif()
  if (NOT DEFINED LIBRPA_GIT_SHA1)
    set(LIBRPA_GIT_SHA1 "GIT-SHA1-UNKNOWN")
  endif()
endif()

if(NOT _loaded_cmake_inc)
  if(DEFINED ENV{MKLROOT} AND NOT DEFINED MKLROOT)
    set(MKLROOT "$ENV{MKLROOT}")
  endif()
  if(MKLROOT)
    find_package(IntelMKL REQUIRED)
    # include_directories(${MKL_INCLUDE_DIRS} ${MKL_INCLUDE_DIRS}/fftw)
    include_directories(${MKL_INCLUDE_DIRS})
    list(APPEND math_libs IntelMKL::MKL)
    list(APPEND _librpa_elpa_configure_libs ${MKL_LIBRARIES})
  else()
    find_package(LAPACK REQUIRED)
    find_package(ScaLAPACK REQUIRED)
    list(APPEND math_libs
            ${LAPACK_LIBRARIES}
            ${ScaLAPACK_LIBRARIES}
            )
    list(APPEND _librpa_elpa_configure_libs
            ${LAPACK_LIBRARIES}
            ${ScaLAPACK_LIBRARIES}
            )
  endif()

  find_package(MPI REQUIRED)
  if (MPI_CXX_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_CXX_FLAGS}")
  endif()

  # message(INFO "${CMAKE_C_COMPILER_LOADED} ${CMAKE_CXX_COMPILER_LOADED} ${CMAKE_Fortran_COMPILER_LOADED}")
  find_package(OpenMP REQUIRED)
  if (OpenMP_CXX_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
  list(APPEND parallel_libs
          OpenMP::OpenMP_CXX
          MPI::MPI_CXX
          )
  include_directories(${MPI_CXX_INCLUDE_PATH})
endif()

# TODO: always switch on MPI, until test program for non-MPI case is implemented.
#       This will require a more comprehensive stub MPI library in LibRPA,
#       and is actually low in priority because LibRPA targets large-scale computation.
add_compile_definitions("LIBRPA_USE_MPI")

include(CheckCXXSourceCompiles)
set(_librpa_required_libraries_save "${CMAKE_REQUIRED_LIBRARIES}")
set(_librpa_filesystem_check_source [=[
#include <filesystem>

int main()
{
  return std::filesystem::exists(std::filesystem::path(".")) ? 0 : 1;
}
]=])
check_cxx_source_compiles(
  "${_librpa_filesystem_check_source}"
  LIBRPA_STD_FILESYSTEM_NO_EXTRA_LIB)
if(NOT LIBRPA_STD_FILESYSTEM_NO_EXTRA_LIB)
  set(CMAKE_REQUIRED_LIBRARIES "${_librpa_required_libraries_save};stdc++fs")
  check_cxx_source_compiles(
    "${_librpa_filesystem_check_source}"
    LIBRPA_STD_FILESYSTEM_STDCXXFS)
  if(LIBRPA_STD_FILESYSTEM_STDCXXFS)
    list(APPEND math_libs stdc++fs)
    message(STATUS "Linking std::filesystem with stdc++fs")
  else()
    message(FATAL_ERROR "std::filesystem is not linkable with the active C++ compiler")
  endif()
endif()
set(CMAKE_REQUIRED_LIBRARIES "${_librpa_required_libraries_save}")
unset(_librpa_required_libraries_save)
unset(_librpa_filesystem_check_source)

if(LIBRPA_USE_EXTERNAL_ELPA AND LIBRPA_USE_BUNDLED_ELPA)
  message(FATAL_ERROR "LIBRPA_USE_EXTERNAL_ELPA and LIBRPA_USE_BUNDLED_ELPA are mutually exclusive")
endif()

if(LIBRPA_USE_CUDA AND LIBRPA_USE_HIP)
  message(FATAL_ERROR "LIBRPA_USE_CUDA and LIBRPA_USE_HIP are mutually exclusive")
endif()

set(LIBRPA_ELPA_TARGET "")
if(LIBRPA_USE_EXTERNAL_ELPA)
  if(NOT EXTERNAL_ELPA_DIR AND DEFINED ENV{EXTERNAL_ELPA_DIR})
    set(EXTERNAL_ELPA_DIR "$ENV{EXTERNAL_ELPA_DIR}" CACHE PATH "External ELPA installation prefix" FORCE)
  endif()
  if(TARGET ELPA::ELPA)
    message(STATUS "Use external ELPA target        : ELPA::ELPA")
  else()
    find_package(ELPA REQUIRED)
    message(STATUS "Use external ELPA library       : ${ELPA_LIBRARY}")
    message(STATUS "Use external ELPA include dirs  : ${ELPA_INCLUDE_DIRS}")
  endif()
  add_compile_definitions("LIBRPA_USE_ELPA")
  set(LIBRPA_ELPA_TARGET ELPA::ELPA)
  list(APPEND math_libs ${LIBRPA_ELPA_TARGET})
elseif(LIBRPA_USE_BUNDLED_ELPA)
  if(NOT LIBRPA_BUNDLED_ELPA_LIBS)
    set(_librpa_elpa_configure_lib_flags "")
    foreach(_librpa_elpa_configure_lib IN LISTS _librpa_elpa_configure_libs)
      if(_librpa_elpa_configure_lib MATCHES "\\.framework$")
        get_filename_component(_librpa_elpa_framework_name "${_librpa_elpa_configure_lib}" NAME_WE)
        get_filename_component(_librpa_elpa_framework_dir "${_librpa_elpa_configure_lib}" DIRECTORY)
        list(APPEND _librpa_elpa_configure_lib_flags
          "-F${_librpa_elpa_framework_dir}"
          "-framework"
          "${_librpa_elpa_framework_name}"
        )
      else()
        list(APPEND _librpa_elpa_configure_lib_flags "${_librpa_elpa_configure_lib}")
      endif()
    endforeach()
    string(REPLACE ";" " " LIBRPA_BUNDLED_ELPA_LIBS "${_librpa_elpa_configure_lib_flags}")
  endif()
  message(STATUS "Use bundled ELPA version        : ${LIBRPA_BUNDLED_ELPA_VERSION}")
  add_compile_definitions("LIBRPA_USE_ELPA")
  add_subdirectory(thirdparty/ELPA)
  set(LIBRPA_ELPA_TARGET elpa)
endif()

# Switch on LibRI functionalities, which requires Cereal, LibComm and LibRI
# When USE_LIBRI is switched on but the include direcotires of the required dependenccies are not specified
# by command line or environment variables, the internal packages are used.
# If any of the include directory is specified but not found, a fatal error will be raised.
if(LIBRPA_USE_LIBRI)
  message(STATUS "Build with LibRI and dependencies")

  # Check Cereal
  if(NOT CEREAL_INCLUDE_DIR)
    set(CEREAL_INCLUDE_DIR $ENV{CEREAL_INCLUDE_DIR})
  endif()
  if("${CEREAL_INCLUDE_DIR}" STREQUAL "")
    message(STATUS "Use internal Cereal  : thirdparty/cereal-1.3.0")
    set(Cereal_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/cereal-1.3.0/include)
  else()
    find_package(Cereal)
    if(NOT Cereal_FOUND)
      message(FATAL_ERROR "User Cereal include directory set to ${Cereal_INCLUDE_DIR}, but not found")
    else()
      message(STATUS "Use user Cereal include directory  : ${Cereal_INCLUDE_DIR}")
    endif()
  endif()

  # Check LibComm
  if(NOT LIBCOMM_INCLUDE_DIR)
    set(LIBCOMM_INCLUDE_DIR $ENV{LIBCOMM_INCLUDE_DIR})
  endif()
  if("${LIBCOMM_INCLUDE_DIR}" STREQUAL "")
    message(STATUS "Use internal LibComm : thirdparty/LibComm (commit hash ${_BUNDLED_LIBCOMM_COMMIT_HASH})")
    set(LibComm_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/LibComm/include)
    set(_LIBRPA_USE_BUNDLED_LIBCOMM ON)
  else()
    find_package(LibComm)
    if(NOT LibComm_FOUND)
      message(FATAL_ERROR "User LibComm include directory set to ${LIBCOMM_INCLUDE_DIR}, but not found")
    else()
      message(STATUS "Use user LibComm include directory : ${LibComm_INCLUDE_DIR}")
    endif()
  endif()

  # Check LibRI
  if(NOT LIBRI_INCLUDE_DIR)
    set(LIBRI_INCLUDE_DIR $ENV{LIBRI_INCLUDE_DIR})
  endif()
  if("${LIBRI_INCLUDE_DIR}" STREQUAL "")
    if((LIBRPA_USE_CUDA OR LIBRPA_USE_HIP) AND LIBRPA_USE_LIBRI_GPU)
      message(STATUS "Use internal LibRI   : thirdparty/LibRI_GPU   (commit hash ${_BUNDLED_LIBRI_GPU_COMMIT_HASH})")
      set(LibRI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/LibRI_GPU/include)
      set(_LIBRPA_USE_BUNDLED_LIBRI_GPU ON)
    else()
      message(STATUS "Use internal LibRI   : thirdparty/LibRI   (commit hash ${_BUNDLED_LIBRI_COMMIT_HASH})")
      set(LibRI_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/LibRI/include)
    endif()
  else()
    find_package(LibRI)
    if(NOT LibRI_FOUND)
      message(FATAL_ERROR "User LibRI include directory set to ${LIBRI_INCLUDE_DIR}, but not found")
    else()
      message(STATUS "Use user LibRI include directory   : ${LibRI_INCLUDE_DIR}")
    endif()
  endif()

  add_compile_definitions("LIBRPA_USE_LIBRI")

  if(NOT (LIBRPA_USE_CUDA OR LIBRPA_USE_HIP))
    # Keep probe settings local and recheck when the selected headers change.
    function(check_libri_fine_grained_lock)
      set(CMAKE_REQUIRED_INCLUDES
        ${LibRI_INCLUDE_DIR} ${LibComm_INCLUDE_DIR} ${Cereal_INCLUDE_DIR})
      set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX OpenMP::OpenMP_CXX)
      set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
      unset(LIBRPA_LIBRI_HAS_FINE_GRAINED_LOCK CACHE)
      check_cxx_source_compiles([=[
#include <RI/ri/LRI.h>
#include <complex>
void set_mode(RI::LRI<int, int, 3, double>& real,
              RI::LRI<int, int, 3, std::complex<double>>& complex)
{
  real.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
  complex.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
}
]=] LIBRPA_LIBRI_HAS_FINE_GRAINED_LOCK)
    endfunction()
    check_libri_fine_grained_lock()
    if(LIBRPA_LIBRI_HAS_FINE_GRAINED_LOCK)
      add_compile_definitions(LIBRPA_LIBRI_HAS_FINE_GRAINED_LOCK)
      message(STATUS "LibRI CPU fine-grained locking enabled for RPA, EXX and GW")
    endif()
  endif()
endif()

if(NOT LIBRPA_MPI_THREAD_LEVEL)
  if((LIBRPA_USE_EXTERNAL_ELPA OR LIBRPA_USE_BUNDLED_ELPA) AND NOT (LIBRPA_USE_CUDA OR LIBRPA_USE_HIP))
    set(LIBRPA_MPI_THREAD_LEVEL MPI_THREAD_MULTIPLE)
  elseif(_LIBRPA_USE_BUNDLED_LIBCOMM)
    set(LIBRPA_MPI_THREAD_LEVEL MPI_THREAD_FUNNELED)
  else()
    set(LIBRPA_MPI_THREAD_LEVEL MPI_THREAD_MULTIPLE)
  endif()
endif()
set(_librpa_supported_mpi_thread_levels
  MPI_THREAD_SINGLE
  MPI_THREAD_FUNNELED
  MPI_THREAD_SERIALIZED
  MPI_THREAD_MULTIPLE)
list(FIND _librpa_supported_mpi_thread_levels "${LIBRPA_MPI_THREAD_LEVEL}" _librpa_mpi_thread_level_index)
if(_librpa_mpi_thread_level_index EQUAL -1)
  message(FATAL_ERROR
    "Invalid LIBRPA_MPI_THREAD_LEVEL=${LIBRPA_MPI_THREAD_LEVEL}; expected one of: ${_librpa_supported_mpi_thread_levels}")
endif()
unset(_librpa_mpi_thread_level_index)
unset(_librpa_supported_mpi_thread_levels)
message(STATUS "Executable MPI thread level     : ${LIBRPA_MPI_THREAD_LEVEL}")

if(LIBRPA_USE_CUDA)
  # find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)
  add_definitions(-DADD_)
  if(LIBRPA_USE_LIBRI_GPU)
    if(_LIBRPA_USE_BUNDLED_LIBRI_GPU)
      add_compile_definitions("__DDLA_RI")
      message(STATUS "Bundled LibRI GPU backend       : DDLA_RI")
    else()
      add_compile_definitions("__CUDA_RI")
    endif()
  endif()
  # include_directories(${MAGMA_ROOT}/include)
  # set(MAGMA_INCLUDE_DIR "${MAGMA_ROOT}/include")
  # link_directories(${MAGMA_ROOT}/lib)
  cmake_policy(SET CMP0104 NEW)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80" CACHE STRING "CUDA architectures" FORCE)
  endif()
  set(CMAKE_CUDA_STANDARD 17)
  set(CUDA_LIBRARIES "-lcudart -lcublas -lcusolver -lnccl -lcurand")
  list(APPEND math_libs
    ${CUDA_LIBRARIES}
    # CUDA::cudart
    # CUDA::cublas
    # CUDA::cusolver
    # CUDA::nccl
    # CUDA::curand
    # ${MAGMA_ROOT}/lib/libmagma.so
    )
  add_compile_definitions("LIBRPA_USE_CUDA")
  add_compile_definitions("DDLA_USE_CUDA")
endif()

# Check LibDDLA. LIBRPA_USE_CUDA/HIP select the public DDLA headers used by
# LibRPA, while a bundled LibDDLA enables its GPU language in a child build.
set(LIBRPA_DDLA_TARGET "")
set(LIBRPA_BUNDLED_DDLA_PROJECT_TARGET "")
if(LIBRPA_USE_CUDA OR LIBRPA_USE_HIP)
  if(NOT LIBDDLA_PATH)
    set(LIBDDLA_PATH $ENV{LIBDDLA_PATH})
  endif()
  set(DDLA_USE_CCL ON)
  if(NOT "${LIBDDLA_PATH}" STREQUAL "")
    set(DDLA_USE_GPU_CPU_TUNNEL "EXTERNAL DEFINED OFF(CUDA default), ON(HIP default)")
    message(STATUS "Use external LibDDLA from: ${LIBDDLA_PATH}")
    if(NOT EXISTS "${LIBDDLA_PATH}/include/ddla/ddla.h")
      message(FATAL_ERROR "LibDDLA headers not found in ${LIBDDLA_PATH}/include")
    endif()
    find_library(LIBDDLA_LIB ddla PATHS ${LIBDDLA_PATH}/lib ${LIBDDLA_PATH}/lib64 NO_DEFAULT_PATH)
    if(NOT LIBDDLA_LIB)
      message(FATAL_ERROR "libddla not found in ${LIBDDLA_PATH}/lib or ${LIBDDLA_PATH}/lib64")
    endif()
    add_library(LibDDLA::ddla SHARED IMPORTED GLOBAL)
    set_target_properties(LibDDLA::ddla PROPERTIES
      IMPORTED_LOCATION "${LIBDDLA_LIB}"
      INTERFACE_INCLUDE_DIRECTORIES "${LIBDDLA_PATH}/include"
    )
    set(LIBRPA_DDLA_TARGET LibDDLA::ddla)
  else()
    if(LIBRPA_USE_HIP)
      set(DDLA_USE_GPU_CPU_TUNNEL ON)
    else()
      set(DDLA_USE_GPU_CPU_TUNNEL OFF)
    endif()
    include(BuildBundledLibDDLA)
    set(LIBDDLA_PATH "${LIBRPA_BUNDLED_DDLA_INSTALL_DIR}")
    install(DIRECTORY "${LIBRPA_BUNDLED_DDLA_INSTALL_DIR}/lib/"
      DESTINATION ${CMAKE_INSTALL_LIBDIR} OPTIONAL)
  endif()
  add_compile_definitions("DDLA_USE_CCL")
endif()

if(LIBRPA_USE_HIP)
  add_compile_definitions("LIBRPA_USE_HIP")
  add_compile_definitions("DDLA_USE_HIP")
  add_compile_definitions("__HIP_PLATFORM_AMD__")
  if(LIBRPA_USE_LIBRI_GPU)
    if(_LIBRPA_USE_BUNDLED_LIBRI_GPU)
      add_compile_definitions("__DDLA_RI")
      message(STATUS "Bundled LibRI GPU backend       : DDLA_RI")
    else()
      add_compile_definitions("__HIP_RI")
    endif()
  endif()
  set(HIP_LIBRARIES "-lamdhip64 -lgalaxyhip -lhipblas -lhipsolver -lrccl -lhiprand")
  list(APPEND math_libs
      ${HIP_LIBRARIES}
  )
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  message(STATUS "Build type set to Debug, adding LIBRPA_DEBUG preprocessor directive")
  add_compile_definitions("LIBRPA_DEBUG")
endif()

if(LIBRPA_VERBOSE_OUTPUT)
  add_compile_definitions("LIBRPA_VERBOSE")
endif()

# Switch on GreenX
if(NOT LIBRPA_USE_EXTERNAL_GREENX)
  set(LIBRPA_GREENX_COMMIT_HASH "${_BUNDLED_GREENX_COMMIT_HASH}")
  message(STATUS "Generate MiniMax grids by using the bundled GreenX library (collected from commit ${LIBRPA_GREENX_COMMIT_HASH})")
  add_subdirectory(thirdparty/greenX)
else()
  set(LIBRPA_GREENX_COMMIT_HASH "(EXTERNAL)")
endif()

function(librpa_keep_cxx_includes_explicit)
  foreach(_include_dir IN LISTS ARGN)
    if(NOT "${_include_dir}" STREQUAL "")
      get_filename_component(_include_dir_abs "${_include_dir}" ABSOLUTE)
      list(REMOVE_ITEM CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "${_include_dir_abs}")
    endif()
  endforeach()
  set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES
      "${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES}" PARENT_SCOPE)
endfunction()

librpa_keep_cxx_includes_explicit(
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${Cereal_INCLUDE_DIR}"
  "${LibComm_INCLUDE_DIR}"
  "${LibRI_INCLUDE_DIR}"
)

if(LIBRPA_ENABLE_TEST)
  # FetchContent_Declare(
  #   googletest
  #   GIT_REPOSITORY https://github.com/google/googletest.git
  #   GIT_TAG release-1.12.1
  # )
  # FetchContent_MakeAvailable(googletest)
  # include(GoogleTest)
  message(STATUS "LibRPA test enabled")
  enable_testing()
endif()

if(LIBRPA_ENABLE_FORTRAN_BIND)
  message(STATUS "LibRPA Fortran binding enabled")
  add_subdirectory(binding/fortran)
endif()

if(LIBRPA_ENABLE_DRIVER)
  message(STATUS "LibRPA driver enabled")
  list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
endif()

add_subdirectory(src)
add_subdirectory(driver/reader)

if(LIBRPA_ENABLE_DRIVER)
  add_subdirectory(driver)
endif()

install(FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa_compute.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa_enums.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa_global.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa_handler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa_input.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa_options.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/file_reader.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/librpa_file_reader.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/librpa)

message(STATUS "CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_Fortran_FLAGS: ${CMAKE_Fortran_FLAGS}")
