add_executable(
  test_input_parser
  test_input_parser.cpp
  ../input_parser.cpp
)
add_test(NAME test_input_parser COMMAND $<TARGET_FILE:test_input_parser>)

add_executable(test_driver_params test_driver_params.cpp ../driver.cpp)
target_link_libraries(test_driver_params PRIVATE librpa_file_reader)
target_compile_options(test_driver_params PRIVATE -UNDEBUG)
add_test(NAME test_driver_params COMMAND $<TARGET_FILE:test_driver_params>)

add_executable(test_bz_mapping test_bz_mapping.cpp)
target_link_libraries(test_bz_mapping PRIVATE librpa_file_reader)
target_compile_options(test_bz_mapping PRIVATE -UNDEBUG)
add_test(NAME test_bz_mapping COMMAND $<TARGET_FILE:test_bz_mapping>)
add_test(NAME test_bz_mapping_mpi COMMAND mpirun -np 2 $<TARGET_FILE:test_bz_mapping>)

add_executable(
  test_velocity_reader
  test_velocity_reader.cpp
  ../driver.cpp
  ../read_data.cpp
  ../reader_2d_coulomb_head.cpp
)
target_link_libraries(test_velocity_reader PRIVATE librpa_file_reader)
target_compile_options(test_velocity_reader PRIVATE -UNDEBUG)
add_test(NAME test_velocity_reader COMMAND $<TARGET_FILE:test_velocity_reader>)

if(BUILD_SHARED_LIBS)
  foreach(target test_driver_params test_bz_mapping test_velocity_reader)
    target_link_libraries(${target} PRIVATE ${math_libs} ${parallel_libs})
  endforeach()
endif()

add_executable(test_stru_symops_parser test_stru_symops_parser.cpp ../stru_symops_parser.cpp)
target_link_libraries(test_stru_symops_parser PRIVATE rpa_lib)
add_test(NAME test_stru_symops_parser COMMAND $<TARGET_FILE:test_stru_symops_parser>)

add_executable(
  test_reader_2d_coulomb_head
  test_reader_2d_coulomb_head.cpp
  ../reader_2d_coulomb_head.cpp
)
# Strict 2D assertions must remain active in Release builds.
target_compile_options(test_reader_2d_coulomb_head PRIVATE -UNDEBUG)
add_test(NAME test_reader_2d_coulomb_head COMMAND $<TARGET_FILE:test_reader_2d_coulomb_head>)

add_executable(
  test_root_output_collective
  test_root_output_collective.cpp
  ../reader_sternheimer.cpp
)
target_link_libraries(test_root_output_collective PRIVATE rpa_lib)
target_compile_definitions(test_root_output_collective PRIVATE "LIBRPA_MPI_THREAD_LEVEL=${LIBRPA_MPI_THREAD_LEVEL}")
target_compile_options(test_root_output_collective PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_root_output_collective PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_root_output_collective
         COMMAND mpirun -np 2 $<TARGET_FILE:test_root_output_collective>)

add_executable(
  test_reader_sternheimer_qpoints
  test_reader_sternheimer_qpoints.cpp
  ../reader_sternheimer.cpp
  ../reader_sternheimer_qpoints.cpp
  ../rpa_qsum.cpp
)
target_link_libraries(test_reader_sternheimer_qpoints PRIVATE rpa_lib)
target_compile_options(test_reader_sternheimer_qpoints PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_reader_sternheimer_qpoints PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_reader_sternheimer_qpoints
         COMMAND $<TARGET_FILE:test_reader_sternheimer_qpoints>)

add_executable(
  test_reader_sternheimer_partial
  test_reader_sternheimer_partial.cpp
  ../reader_sternheimer.cpp
  ../reader_sternheimer_partial.cpp
)
target_link_libraries(test_reader_sternheimer_partial PRIVATE rpa_lib)
target_compile_options(test_reader_sternheimer_partial PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_reader_sternheimer_partial PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_reader_sternheimer_partial
         COMMAND $<TARGET_FILE:test_reader_sternheimer_partial>)

add_executable(
  test_sternheimer_partial_reconstruction
  test_sternheimer_partial_reconstruction.cpp
  ../sternheimer_partial_reconstruction.cpp
  ../rpa_qsum.cpp
)
target_link_libraries(test_sternheimer_partial_reconstruction PRIVATE rpa_lib)
target_compile_options(test_sternheimer_partial_reconstruction PRIVATE -UNDEBUG)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_sternheimer_partial_reconstruction PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_sternheimer_partial_reconstruction
         COMMAND $<TARGET_FILE:test_sternheimer_partial_reconstruction>)

add_executable(
  test_rpa_qsum
  test_rpa_qsum.cpp
  ../rpa_qsum.cpp
)
target_compile_options(test_rpa_qsum PRIVATE -UNDEBUG)
add_test(NAME test_rpa_qsum COMMAND $<TARGET_FILE:test_rpa_qsum>)

add_executable(test_gw_occupation_output
  test_gw_occupation_output.cpp
  ../tasks/task_helper.cpp)
target_link_libraries(test_gw_occupation_output PRIVATE rpa_lib)
if(BUILD_SHARED_LIBS)
  target_link_libraries(test_gw_occupation_output PRIVATE ${math_libs} ${parallel_libs})
endif()
add_test(NAME test_gw_occupation_output COMMAND $<TARGET_FILE:test_gw_occupation_output>)
