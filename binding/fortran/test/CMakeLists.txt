macro(config_test_f03 TARGET NUMPROC)
  add_executable(${TARGET} ${TARGET}.f90)
  add_dependencies(${TARGET} rpa_f03_lib rpa_lib)
  target_link_libraries(${TARGET} PRIVATE rpa_f03_lib rpa_lib)
  if(BUILD_SHARED_LIBS)
    target_link_libraries(${TARGET} PRIVATE ${math_libs} ${parallel_libs})
  endif()
  set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE Fortran)
  target_include_directories(${TARGET} PRIVATE
    $<TARGET_PROPERTY:rpa_f03_lib,Fortran_MODULE_DIRECTORY>
  )
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


if(LIBRPA_USE_LIBRI)
  config_test_f03(test_f03_binding_libri 1)
  config_test_f03(test_f03_binding_call_internal_test 2)
  config_test_f03(test_f03_binding_exx_bccHe 4)
  # This numerical fixture stores generated arrays on long free-form lines.
  target_compile_options(test_f03_binding_exx_bccHe PRIVATE
    $<$<Fortran_COMPILER_ID:GNU>:-ffree-line-length-none>)
endif()
