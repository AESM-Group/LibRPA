# Invalid experimental configurations must fail before any dataset is read.
if(HAS_DDLA)
  set(policies scalapack)
  set(diagnostic "use_mixed_type_ddla requires linear_algorithm = ddla")
else()
  set(policies scalapack ddla)
  set(diagnostic "use_mixed_type_ddla requires a LIBRPA_USE_DDLA build")
endif()
foreach(policy IN LISTS policies)
  set(work "${TEST_WORK_DIR}/${policy}")
  file(MAKE_DIRECTORY "${work}")
  file(WRITE "${work}/librpa.in"
    "task = print_minimax\nlinear_algorithm = ${policy}\nuse_mixed_type_ddla = true\n")
  execute_process(COMMAND "${LIBRPA_DRIVER}"
    WORKING_DIRECTORY "${work}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  string(FIND "${output}${error}" "${diagnostic}" found)
  if("${status}" STREQUAL "0" OR found EQUAL -1)
    message(FATAL_ERROR "Missing mixed DDLA rejection for ${policy}: ${output}${error}")
  endif()
endforeach()
