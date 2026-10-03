add_executable(
  test_input_parser
  test_input_parser.cpp
  ../input_parser.cpp
)
add_test(NAME test_input_parser COMMAND $<TARGET_FILE:test_input_parser>)

add_executable(test_driver_params test_driver_params.cpp ../driver.cpp)
target_link_libraries(test_driver_params PRIVATE librpa_file_reader)
add_test(NAME test_driver_params COMMAND $<TARGET_FILE:test_driver_params>)

add_executable(test_velocity_reader test_velocity_reader.cpp ../driver.cpp ../read_data.cpp)
target_link_libraries(test_velocity_reader PRIVATE librpa_file_reader)
add_test(NAME test_velocity_reader COMMAND $<TARGET_FILE:test_velocity_reader>)
add_executable(test_stru_symops_parser test_stru_symops_parser.cpp ../stru_symops_parser.cpp)
target_link_libraries(test_stru_symops_parser PRIVATE rpa_lib)
add_test(NAME test_stru_symops_parser COMMAND $<TARGET_FILE:test_stru_symops_parser>)

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
