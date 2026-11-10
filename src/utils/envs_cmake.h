#pragma once

namespace librpa_int
{

namespace envs
{

//! The path of the source code directory
extern const char * source_dir;

//! Git hash of the source code
extern const char * git_hash;

//! Git reference of the source code
extern const char * git_ref;

//! CMake C++ compiler
extern const char * cxx_compiler;

//! CMake Fortran compiler
extern const char * fortran_compiler;

//! CMake C++ compiler flags
extern const char * cxx_compiler_flags;

//! CMake Fortran compiler flags
extern const char * fortran_compiler_flags;

//! Commit hash of linked GreenX
extern const char * greenx_commit_hash;

//! Build timestamp
extern const char * build_time;

// CMake Options
//! CMake build type
extern const char * cmake_build_type;

extern const char * use_libri;

extern const char * use_elpa;

extern const char * external_elpa_dir;

extern const char * libri_include_dir;

extern const char * libcomm_include_dir;

extern const char * cereal_include_dir;

extern const char * thirdparty_sources;

extern const char * ddla_use_ccl;

extern const char * ddla_use_gpu_cpu_tunnel;

extern const char * libddla_path;

extern const char * elpa_dir;

} /* end of namespace envs */
} /* end of namespace librpa_int */
