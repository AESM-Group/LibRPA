add_executable(
  test_input_parser
  test_input_parser.cpp
  ../input_parser.cpp
)
add_test(NAME test_input_parser COMMAND $<TARGET_FILE:test_input_parser>)

add_executable(test_driver_params test_driver_params.cpp ../driver.cpp)
target_link_libraries(test_driver_params PRIVATE librpa_file_reader)
add_test(NAME test_driver_params COMMAND $<TARGET_FILE:test_driver_params>)

add_executable(
  test_velocity_reader
  test_velocity_reader.cpp
  ../driver.cpp
  ../read_data.cpp
  ../reader_2d_coulomb_head.cpp
)
target_link_libraries(test_velocity_reader PRIVATE librpa_file_reader)
add_test(NAME test_velocity_reader COMMAND $<TARGET_FILE:test_velocity_reader>)
add_executable(test_stru_symops_parser test_stru_symops_parser.cpp ../stru_symops_parser.cpp)
target_link_libraries(test_stru_symops_parser PRIVATE rpa_lib)
add_test(NAME test_stru_symops_parser COMMAND $<TARGET_FILE:test_stru_symops_parser>)

add_executable(
  test_reader_2d_coulomb_head
  test_reader_2d_coulomb_head.cpp
  ../reader_2d_coulomb_head.cpp
)
target_link_libraries(test_reader_2d_coulomb_head PRIVATE rpa_lib)
# Strict 2D assertions must remain active in Release builds.
target_compile_options(test_reader_2d_coulomb_head PRIVATE -UNDEBUG)
add_test(NAME test_reader_2d_coulomb_head COMMAND $<TARGET_FILE:test_reader_2d_coulomb_head>)
