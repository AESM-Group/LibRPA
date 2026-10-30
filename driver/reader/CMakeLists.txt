# Internal to the source tree: kept static so that installed executables do not
# depend on it at run time; PIC allows linking it into shared libraries.
add_library(librpa_file_reader STATIC
  ../stru_symops_parser.cpp
  file_reader.cpp
  localized_operator_io.cpp
  reader_nao_overlap.cpp
  read_data.cpp
  bz_full_kmap.cpp
  reader_band.cpp
  reader_structure.cpp
  reader_basis.cpp
  reader_lri.cpp
  reader_coulomb.cpp
  reader_eigenvec.cpp)
target_link_libraries(librpa_file_reader PUBLIC rpa_lib)
if(LIBRPA_BUNDLED_ELPA_PROJECT_TARGET)
  # Reader headers include base_blacs.h, which needs the generated ELPA headers.
  add_dependencies(librpa_file_reader ${LIBRPA_BUNDLED_ELPA_PROJECT_TARGET})
endif()
set_target_properties(librpa_file_reader PROPERTIES
  OUTPUT_NAME rpa_file_reader
  POSITION_INDEPENDENT_CODE ON)
