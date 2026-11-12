include_guard(GLOBAL)
include(ExternalProject)

if(NOT LIBRPA_USE_DDLA)
  message(FATAL_ERROR "Managed LibDDLA requires LIBRPA_USE_DDLA=ON")
endif()
if(LIBRPA_USE_CUDA AND LIBRPA_USE_HIP)
  message(FATAL_ERROR "Managed LibDDLA cannot enable CUDA and HIP together")
endif()
if(NOT EXISTS "${_libddla_source_dir}/include/ddla/ddla.h")
  message(FATAL_ERROR "Managed LibDDLA source is incomplete: ${_libddla_source_dir}")
endif()

set(_libddla_binary_dir "${CMAKE_BINARY_DIR}/_deps/libddla-build")
set(_libddla_install_dir "${CMAKE_BINARY_DIR}/_deps/libddla-install")
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}" OR IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
  message(FATAL_ERROR "Managed LibDDLA requires relative CMAKE_INSTALL_LIBDIR and CMAKE_INSTALL_INCLUDEDIR; use LIBDDLA_PATH for an external installation")
endif()
set(_libddla_include_dir "${_libddla_install_dir}/${CMAKE_INSTALL_INCLUDEDIR}")
set(_libddla_lib_dir "${_libddla_install_dir}/${CMAKE_INSTALL_LIBDIR}")
set(_libddla_library
  "${_libddla_lib_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}ddla${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Imported-target include directories must exist while the parent is configured.
file(MAKE_DIRECTORY "${_libddla_include_dir}" "${_libddla_lib_dir}")

set(_libddla_cmake_args
  "-DCMAKE_INSTALL_PREFIX:PATH=${_libddla_install_dir}"
  "-DCMAKE_INSTALL_LIBDIR:STRING=${CMAKE_INSTALL_LIBDIR}"
  "-DCMAKE_INSTALL_INCLUDEDIR:STRING=${CMAKE_INSTALL_INCLUDEDIR}"
  "-DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}"
  "-DBUILD_TESTS:BOOL=OFF"
)

if(CMAKE_BUILD_TYPE)
  list(APPEND _libddla_cmake_args
    "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()
if(MPI_CXX_COMPILER)
  list(APPEND _libddla_cmake_args
    "-DMPI_CXX_COMPILER:FILEPATH=${MPI_CXX_COMPILER}")
endif()

if(CMAKE_PREFIX_PATH)
  string(REPLACE ";" "|" _libddla_prefix_path "${CMAKE_PREFIX_PATH}")
  list(APPEND _libddla_cmake_args
    "-DCMAKE_PREFIX_PATH:STRING=${_libddla_prefix_path}")
endif()

if(LIBRPA_USE_CUDA)
  list(APPEND _libddla_cmake_args
    "-DDDLA_USE_CUDA:BOOL=ON"
    "-DDDLA_USE_HIP:BOOL=OFF"
    "-DDDLA_USE_GPU_CPU_TUNNEL:BOOL=OFF"
    "-DDDLA_USE_CCL:BOOL=ON"
  )
  set(_libddla_backend CUDA)
  set(_libddla_forward_variables
    CMAKE_CUDA_COMPILER
    CMAKE_CUDA_FLAGS
    CMAKE_CUDA_ARCHITECTURES
    CMAKE_CUDA_SEPARABLE_COMPILATION
    CUDAToolkit_ROOT
  )
elseif(LIBRPA_USE_HIP)
  list(APPEND _libddla_cmake_args
    "-DDDLA_USE_CUDA:BOOL=OFF"
    "-DDDLA_USE_HIP:BOOL=ON"
    "-DDDLA_USE_GPU_CPU_TUNNEL:BOOL=ON"
    "-DDDLA_USE_CCL:BOOL=ON"
  )
  set(_libddla_backend HIP)
  set(_libddla_forward_variables
    CMAKE_HIP_COMPILER
    CMAKE_HIP_FLAGS
    CMAKE_HIP_ARCHITECTURES
    CMAKE_HIP_SEPARABLE_COMPILATION
    CMAKE_HIP_COMPILER_ROCM_ROOT
    ROCM_PATH
  )
else()
  list(APPEND _libddla_cmake_args
    "-DDDLA_USE_CUDA:BOOL=OFF"
    "-DDDLA_USE_HIP:BOOL=OFF"
    "-DDDLA_USE_CCL:BOOL=OFF"
    "-DDDLA_USE_GPU_CPU_TUNNEL:BOOL=OFF"
  )
  set(_libddla_backend CPU)
  set(_libddla_forward_variables "")
endif()

list(APPEND _libddla_forward_variables
  DDLA_CPU_BLAS_LIBRARY DDLA_LAPACK_LIBRARY
  BLA_VENDOR BLAS_LIBRARIES LAPACK_LIBRARIES
  DDLA_CUDA_MATH_LIBRARY_DIR
  CUBLAS_LIBRARY CUBLASLT_LIBRARY CURAND_LIBRARY CUSOLVER_LIBRARY CUDART_LIBRARY
)
foreach(_libddla_variable IN LISTS _libddla_forward_variables)
  if(DEFINED ${_libddla_variable}
      AND NOT "${${_libddla_variable}}" STREQUAL "")
    string(REPLACE ";" "|" _libddla_value "${${_libddla_variable}}")
    list(APPEND _libddla_cmake_args
      "-D${_libddla_variable}:STRING=${_libddla_value}")
  endif()
endforeach()

if(LIBRPA_USE_HIP
    AND (NOT DEFINED ROCM_PATH OR "${ROCM_PATH}" STREQUAL "")
    AND DEFINED ENV{ROCM_PATH}
    AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
  list(APPEND _libddla_cmake_args "-DROCM_PATH:PATH=$ENV{ROCM_PATH}")
endif()

ExternalProject_Add(librpa_bundled_libddla
  SOURCE_DIR "${_libddla_source_dir}"
  BINARY_DIR "${_libddla_binary_dir}"
  INSTALL_DIR "${_libddla_install_dir}"
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CMAKE_ARGS ${_libddla_cmake_args}
  LIST_SEPARATOR "|"
  BUILD_BYPRODUCTS "${_libddla_library}"
)

add_library(LibDDLA::ddla SHARED IMPORTED GLOBAL)
add_dependencies(LibDDLA::ddla librpa_bundled_libddla)
set_target_properties(LibDDLA::ddla PROPERTIES
  IMPORTED_LOCATION "${_libddla_library}"
  INTERFACE_INCLUDE_DIRECTORIES "${_libddla_include_dir}"
)

set(LIBRPA_DDLA_TARGET LibDDLA::ddla)
set(LIBRPA_BUNDLED_DDLA_PROJECT_TARGET librpa_bundled_libddla)
set(LIBRPA_BUNDLED_DDLA_INSTALL_DIR "${_libddla_install_dir}")

message(STATUS
  "Use managed LibDDLA             : ${_libddla_source_dir}")
message(STATUS
  "Managed LibDDLA backend         : ${_libddla_backend}")
message(STATUS
  "Managed LibDDLA build directory : ${_libddla_binary_dir}")
