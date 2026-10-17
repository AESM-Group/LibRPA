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
    test_si_density_symmetry
    test_task_helper
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
    test_rpa_headwing
    test_rpa_logdet
    test_sternheimer_rpa
    test_sternheimer_symmetry
    test_sternheimer_qsum
    test_symmetry
    test_symmetry_spin_kernel
    test_timefreq
    test_vector3
    test_wigner_rotation
    )
  config_test(${target} 1)
endforeach()

# These suites use standard assert() and must remain active in Release builds.
target_compile_options(test_sternheimer_rpa PRIVATE -UNDEBUG)
target_compile_options(test_sternheimer_symmetry PRIVATE -UNDEBUG)
target_sources(test_task_helper PRIVATE ../../driver/tasks/task_helper.cpp)
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
  test_shared_memory_mpi
  )
  config_test(${target} 4)
endforeach()

target_compile_options(test_rpa_headwing PRIVATE -UNDEBUG)
# The symmetry-context and k-BLACS symmetry-restore suites assert the k-star
# rotation conventions; keep them active in Release builds as well.
target_compile_options(test_symmetry_context PRIVATE -UNDEBUG)
target_compile_options(test_symmetry PRIVATE -UNDEBUG)
target_compile_options(test_meanfield PRIVATE -UNDEBUG)
target_compile_options(test_meanfield_mpi PRIVATE -UNDEBUG)
set_tests_properties(test_symmetry_context test_si_density_symmetry
  test_meanfield test_meanfield_mpi PROPERTIES LABELS "symmetry")

if(LIBRPA_USE_LIBRI)
  config_test(test_strict_2d_gw_wc 4)
  target_compile_options(test_strict_2d_gw_wc PRIVATE -UNDEBUG)
  add_test(NAME test_strict_2d_gw_wc_serial COMMAND $<TARGET_FILE:test_strict_2d_gw_wc>)
  set_tests_properties(test_strict_2d_gw_wc test_strict_2d_gw_wc_serial
    PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1;MKL_NUM_THREADS=1" TIMEOUT 60)
endif()

# The q-star Coulomb ownership receipt must be partitioned exactly once before
# the additive LibRI communication. Use square BLACS groups: two ranks form
# two 1x1 groups, while four ranks exercise the production 2x2 BLACS layout.
if(LIBRPA_USE_LIBRI)
  config_test(test_exx_coulomb_ownership 1)
  target_compile_options(test_exx_coulomb_ownership PRIVATE -UNDEBUG)
  config_test(test_sigc_matrix_output 1)
  target_compile_options(test_sigc_matrix_output PRIVATE -UNDEBUG)
  foreach(nproc 2 4)
    add_test(NAME test_exx_coulomb_ownership_mpi${nproc}
      COMMAND mpirun -np ${nproc} $<TARGET_FILE:test_exx_coulomb_ownership>)
    add_test(NAME test_sigc_matrix_output_mpi${nproc}
      COMMAND mpirun -np ${nproc} $<TARGET_FILE:test_sigc_matrix_output>)
  endforeach()
  set_tests_properties(test_exx_coulomb_ownership test_exx_coulomb_ownership_mpi2
    test_exx_coulomb_ownership_mpi4 PROPERTIES LABELS "symmetry;exx" TIMEOUT 120)
  set_tests_properties(test_sigc_matrix_output test_sigc_matrix_output_mpi2
    test_sigc_matrix_output_mpi4 PROPERTIES LABELS "gw;io" TIMEOUT 120
    ENVIRONMENT "OMP_NUM_THREADS=1")
endif()

add_test(NAME test_rpa_logdet_mpi
  COMMAND mpirun -np 4 $<TARGET_FILE:test_rpa_logdet>)
add_test(NAME test_rpa_headwing_mpi
  COMMAND mpirun -np 4 $<TARGET_FILE:test_rpa_headwing>)
set_tests_properties(test_rpa_headwing_mpi PROPERTIES TIMEOUT 120)
config_test(test_headwing_descriptor 2)
set_tests_properties(test_headwing_descriptor PROPERTIES TIMEOUT 30)

# TODO(minye): make use of GTest

target_compile_definitions(test_si_density_symmetry PRIVATE
  SI_DENSITY_REFERENCE="${CMAKE_CURRENT_SOURCE_DIR}/data/si_k444_density_reference.txt")
