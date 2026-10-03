# By default, if the compilation succeeds, you will have the executable at the "src" subdirectory in the build directory
# To put it in the build root directory
# set_target_properties(rpa_exe
#   PROPERTIES
#   ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
#   LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
#   RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

if (BUILD_SHARED_LIBS)
  add_library(rpa_lib SHARED)
else()
  add_library(rpa_lib STATIC)
endif()

add_dependencies(rpa_lib LibGXMiniMax)

set_target_properties(rpa_lib PROPERTIES
  OUTPUT_NAME rpa
  VERSION ${CMAKE_PROJECT_VERSION}
  SOVERSION ${CMAKE_PROJECT_VERSION}
)

target_link_libraries(rpa_lib PRIVATE LibGXMiniMax)
target_include_directories(rpa_lib PRIVATE
  $<TARGET_PROPERTY:LibGXMiniMax,Fortran_MODULE_DIRECTORY>
)

if(LIBRPA_ELPA_TARGET)
  # Public LibRPA headers include ELPA declarations, so consumers need the
  # selected ELPA target's include directories as part of the interface.
  target_link_libraries(rpa_lib PUBLIC ${LIBRPA_ELPA_TARGET})
endif()
if(LIBRPA_BUNDLED_ELPA_PROJECT_TARGET)
  # Bundled ELPA is built by ExternalProject_Add, so keep its build step
  # ordered before rpa_lib tries to use the installed library and modules.
  add_dependencies(rpa_lib ${LIBRPA_BUNDLED_ELPA_PROJECT_TARGET})
endif()

if(LIBRPA_DDLA_TARGET)
  # Internal headers used by the driver and tests include DDLA declarations,
  # so preserve the previous public include-path contract through the target.
  target_link_libraries(rpa_lib PUBLIC ${LIBRPA_DDLA_TARGET})
endif()
if(LIBRPA_BUNDLED_DDLA_PROJECT_TARGET)
  # The child CMake build installs LibDDLA and its public headers before any
  # LibRPA source that includes those headers is compiled.
  add_dependencies(rpa_lib ${LIBRPA_BUNDLED_DDLA_PROJECT_TARGET})
endif()

get_filename_component(LIBRPA_PUBLIC_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../include" ABSOLUTE)
target_include_directories(rpa_lib
  PUBLIC
    $<BUILD_INTERFACE:${LIBRPA_PUBLIC_INCLUDE_DIR}>
  )
target_include_directories(rpa_lib PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LIBRPA_PUBLIC_INCLUDE_DIR}
)

# Public C++ headers expose types whose declarations depend on these feature
# switches. Propagate the same definitions to consumers to keep their view of
# the LibRPA data model ABI-compatible with the library.
target_compile_definitions(rpa_lib PUBLIC LIBRPA_USE_MPI)
if(LIBRPA_USE_LIBRI)
  target_compile_definitions(rpa_lib PUBLIC LIBRPA_USE_LIBRI)
endif()
if(LIBRPA_USE_EXTERNAL_ELPA OR LIBRPA_USE_BUNDLED_ELPA)
  target_compile_definitions(rpa_lib PUBLIC LIBRPA_USE_ELPA)
endif()
if(LIBRPA_VERBOSE_OUTPUT)
  target_compile_definitions(rpa_lib PUBLIC LIBRPA_VERBOSE)
endif()
configure_file(utils/envs_cmake.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/envs_cmake.cpp)
configure_file(utils/envs_cmake.h ${CMAKE_CURRENT_BINARY_DIR}/envs_cmake.h)  # only to make the configured envs_cmake.cpp compile

list(APPEND lib_sources
        ${CMAKE_CURRENT_BINARY_DIR}/envs_cmake.cpp

        api/compute_exx.cpp
        api/compute_helper.cpp
        api/compute_g0w0.cpp
        api/compute_rpa.cpp
        api/dataset.cpp
        api/dataset_helper.cpp
        api/global.cpp
        api/handler.cpp
        api/input.cpp
        api/instance_manager.cpp
        api/librpa.cpp
        api/options.cpp
        core/analycont.cpp
        core/symmetry_context.cpp
        core/symmetry_spin_kernel.cpp
        core/atomic_basis.cpp
        core/chi0.cpp
        core/coulmat.cpp
        core/dielecmodel.cpp
        core/epsilon.cpp
        core/exx.cpp
        core/geometry.cpp
        core/gw.cpp
        core/meanfield.cpp
        core/meanfield_mpi.cpp
        core/minimax_gx.cpp
        core/pbc.cpp
        core/qpoint_view.cpp
        core/qpe_solver.cpp
        core/qsgw/convergence.cpp
        core/qsgw/correlation_potential.cpp
        core/qsgw/distributed_matrix.cpp
        core/qsgw/effective_hamiltonian.cpp
        core/qsgw/fixed_basis.cpp
        core/qsgw/hamiltonian_cut.cpp
        core/qsgw/hamiltonian_mixing.cpp
        core/qsgw/occupation.cpp
        core/qsgw/projection_target.cpp
        core/qsgw/vxc_projection.cpp
        core/ri.cpp
  core/sternheimer_rpa.cpp
  core/sternheimer_symmetry.cpp
        core/timefreq.cpp
        core/utils_atomic_basis_blacs.cpp
        core/utils_complexmatrix.cpp

        io/fs.cpp
        io/aux_basis_summary.cpp
        io/global_io.cpp
        io/input_elsi.cpp
        io/output_gw.cpp
        io/output_matrix.cpp

        interface/gx_minimax_wrp.f90

        math/complexmatrix.cpp
        math/fitting.cpp
        math/interpolate.cpp
        math/lebedev_laikov.cpp
        math/mathtools.cpp
        math/matrix.cpp
        math/matrix3.cpp
        math/rsh.cpp
        math/symmetry.cpp
        math/utils_matrix_mpi.cpp
        math/vector3_order.cpp
        math/wigner_rotation.cpp

        mpi/base_blacs.cpp
        mpi/base_mpi.cpp
        mpi/utils_blacs.cpp
        mpi/global_mpi.cpp
        mpi/two_level_parallel_context.cpp
        mpi/kpoint_blacs_parallel_context.cpp

        utils/constants.cpp
        utils/dev_options.cpp
        utils/profiler.cpp
        utils/utils_cmake.cpp
        utils/utils_mem.cpp
)

if(NOT LIBRPA_USE_EXTERNAL_ELPA AND NOT LIBRPA_USE_BUNDLED_ELPA)
  list(APPEND lib_sources utils/elpa_stub.cpp)
endif()

if(LIBRPA_USE_LIBRI)
  target_include_directories(rpa_lib
    PUBLIC
      $<BUILD_INTERFACE:${Cereal_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LibComm_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LibRI_INCLUDE_DIR}>
    )
  target_include_directories(rpa_lib PRIVATE
    ${Cereal_INCLUDE_DIR}
    ${LibComm_INCLUDE_DIR}
    ${LibRI_INCLUDE_DIR}
  )
endif()

if(LIBRPA_USE_CUDA OR LIBRPA_USE_HIP)
  list(APPEND lib_sources 
    gpu/device_connector.cpp
  )
endif()

if(LIBRPA_USE_EXTERNAL_ELPA OR LIBRPA_USE_BUNDLED_ELPA)
  list(APPEND lib_sources
    elpa/elpa_connector.cpp)
endif()

# source must be PRIVATE to avoid recompilation when linking driver to this library
target_sources(rpa_lib PRIVATE ${lib_sources})
target_link_libraries(rpa_lib PRIVATE ${math_libs} ${parallel_libs})

install(TARGETS rpa_lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(LIBRPA_ENABLE_TEST AND LIBRPA_ENABLE_CPP_TEST)
  add_subdirectory(test)
endif()
