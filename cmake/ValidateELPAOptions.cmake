# Resolve ELPA before enabling languages: source builds also need C.
foreach(_legacy_option LIBRPA_USE_EXTERNAL_ELPA LIBRPA_USE_BUNDLED_ELPA)
  if(DEFINED ${_legacy_option})
    message(DEPRECATION
      "${_legacy_option} is deprecated. Use LIBRPA_USE_ELPA, ELPA_SOURCE_DIR "
      "or ELPA_PATH instead, and remove ${_legacy_option} from the CMake cache.")
    if(${_legacy_option})
      set(LIBRPA_USE_ELPA ON)
    endif()
  endif()
endforeach()

librpa_thirdparty_path(ELPA_SOURCE_DIR _elpa_user_source)
librpa_thirdparty_path(ELPA_PATH ELPA_PATH EXTERNAL_ELPA_DIR)
if(NOT ELPA_PATH)
  librpa_thirdparty_path(ELPA_ROOT ELPA_ROOT)
endif()
if(_elpa_user_source AND (ELPA_PATH OR ELPA_ROOT))
  message(FATAL_ERROR "ELPA: ELPA_SOURCE_DIR and an installation prefix (ELPA_PATH or ELPA_ROOT) are mutually exclusive")
endif()

set(_elpa_mode "")
if(TARGET ELPA::ELPA)
  set(_elpa_mode target)
elseif(_elpa_user_source)
  set(_elpa_mode source)
elseif(ELPA_PATH OR ELPA_ROOT)
  set(_elpa_mode installed)
elseif(LIBRPA_USE_EXTERNAL_ELPA)
  # Preserve the legacy option's search of standard installation paths.
  set(_elpa_mode installed)
elseif(LIBRPA_USE_ELPA)
  set(_elpa_mode source)
endif()

if(_elpa_mode)
  set(LIBRPA_USE_ELPA ON CACHE BOOL "Use ELPA for distributed eigensolvers" FORCE)
  set(LIBRPA_USE_ELPA ON)
endif()
