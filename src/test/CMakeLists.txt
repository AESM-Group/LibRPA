# a simple macro to add source, include, link main library and run test
# see also
# - https://stackoverflow.com/questions/53868015/avoid-repetitive-cmake-code-with-multiple-targets
# - https://stackoverflow.com/questions/19278510/cmake-ctest-add-test-command-appends-extra-quotes-to-my-command-parameters
macro(config_test TARGET NUMPROC)
  add_executable(${TARGET} ${TARGET}.cpp)
  target_compile_definitions(${TARGET} PRIVATE "LIBRPA_MPI_THREAD_LEVEL=${LIBRPA_MPI_THREAD_LEVEL}")
  target_link_libraries(${TARGET} PRIVATE rpa_lib)
  if(LIBRPA_BUNDLED_ELPA_PROJECT_TARGET)
    # Tests include LibRPA headers directly, including base_blacs.h.
    add_dependencies(${TARGET} ${LIBRPA_BUNDLED_ELPA_PROJECT_TARGET})
  endif()
  # Math and MPI symbols will have been included in rpa_lib for static library case.
  # Only link them when shared library is built.
  if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${math_libs} ${parallel_libs})
  endif()
  # TODO: separate libri test targets
  if(LIBRPA_USE_LIBRI)
    target_include_directories(${TARGET}
      PRIVATE
        ${Cereal_INCLUDE_DIR}
        ${LibComm_INCLUDE_DIR}
        ${LibRI_INCLUDE_DIR}
      )
  endif()
  if(${NUMPROC} GREATER 1)
    # message("parallel test")
    add_test(
      NAME ${TARGET}
      COMMAND mpirun -np ${NUMPROC} $<TARGET_FILE:${TARGET}>
    )
  else()
    # message("serial test")
    add_test(
      NAME ${TARGET}
      COMMAND $<TARGET_FILE:${TARGET}>
    )
  endif()
endmacro()

# add serial tests
foreach(target
    test_analycont
    test_atom
    test_atomic_basis
    test_base_utility
    test_blas_lapack
    test_complexmatrix
    test_fitting
    test_io
    test_symmetry_context
    test_arbitrary_float
    test_geometry
    test_lebedev_laikov
    test_libri_utils
    # test_librpa
    test_matrix
    test_matrix_m
    test_meanfield
    test_pbc
    test_profiler
    test_qpe_solver
    test_rsh
    test_symmetry
    test_symmetry_spin_kernel
    test_timefreq
    test_vector3
    test_wigner_rotation
    )
  config_test(${target} 1)
endforeach()

# add 4-process tests
foreach(target
  test_atomic_basis_blacs
  test_base_mpi
  test_blacs
  test_dataset
  test_epsilon
  test_matrix_m_mpi
  test_meanfield_mpi
  test_kpoint_blacs_parallel_context
  test_rpa_headwing
  test_shared_memory_mpi
  )
  config_test(${target} 4)
endforeach()

target_compile_options(test_rpa_headwing PRIVATE -UNDEBUG)

# TODO(minye): make use of GTest
