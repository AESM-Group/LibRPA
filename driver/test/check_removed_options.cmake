# Reject removed switches before reading a dataset or starting a calculation.
foreach(keyword use_scalapack_ecrpa use_scalapack_gw_wc)
  foreach(value true false)
    set(work "${TEST_WORK_DIR}/${keyword}-${value}")
    file(MAKE_DIRECTORY "${work}")
    file(WRITE "${work}/librpa.in" "task = print_minimax\n${keyword} = ${value}\n")
    execute_process(COMMAND "${LIBRPA_DRIVER}"
      WORKING_DIRECTORY "${work}"
      RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if("${status}" STREQUAL "0")
      message(FATAL_ERROR "Removed option ${keyword}=${value} was accepted")
    endif()
    string(FIND "${output}${error}" "${keyword} was removed" found)
    if(found EQUAL -1)
      message(FATAL_ERROR "Missing migration diagnostic for ${keyword}: ${output}${error}")
    endif()
  endforeach()
endforeach()
