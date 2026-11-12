if (BUILD_SHARED_LIBS)
  add_library(rpa_f03_lib SHARED)
else()
  add_library(rpa_f03_lib STATIC)
endif()

set_target_properties(rpa_f03_lib PROPERTIES
  OUTPUT_NAME rpa_f03
  VERSION ${CMAKE_PROJECT_VERSION}
  SOVERSION ${CMAKE_PROJECT_VERSION}
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/librpa_f03.f90
  ${CMAKE_CURRENT_BINARY_DIR}/librpa_f03.f90
)

target_sources(rpa_f03_lib
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/librpa_f03.f90
)

set_target_properties(rpa_f03_lib PROPERTIES
  Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

find_package(MPI REQUIRED COMPONENTS Fortran)
target_link_libraries(rpa_f03_lib PRIVATE rpa_lib MPI::MPI_Fortran)

target_include_directories(rpa_f03_lib
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

install(TARGETS rpa_f03_lib
        ARCHIVE
        DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/librpa_f03.mod
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/librpa)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/librpa_f03_stubs.f90
        DESTINATION ${CMAKE_INSTALL_DATADIR})

if(LIBRPA_ENABLE_TEST AND LIBRPA_ENABLE_FORTRAN_TEST)
  add_subdirectory(test)
endif()
