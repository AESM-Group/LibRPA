#include "utils_cmake.h"

#include "envs_cmake.h"
#include "../io/global_io.h"

#include <sstream>

namespace librpa_int
{

void print_cmake_info()
{
    global::lib_printf(cmake_info_storage().c_str());
}

const std::string& cmake_info_storage()
{
    static const std::string s = []{
        std::stringstream ss;
        ss << "LibRPA CMake build info:" << "\n"
        << "| Build type               : " << envs::cmake_build_type << "\n"
        << "| Build time               : " << envs::build_time << "\n"
        << "| Source code directory    : " << envs::source_dir << "\n"
        << "| Source code Git reference: " << envs::git_ref << "\n"
        << "| Source code Git hash     : " << envs::git_hash << "\n"
        << "| C++ compiler             : " << envs::cxx_compiler << "\n"
        << "| C++ compiler flags       : " << envs::cxx_compiler_flags << "\n"
        << "| Fortran compiler         : " << envs::fortran_compiler << "\n"
        << "| Fortran compiler flags   : " << envs::fortran_compiler_flags << "\n"
        << "| GreenX commit hash       : " << envs::greenx_commit_hash << "\n"
        << "| Use LibRI                : " << envs::use_libri << "\n"
        << "| LibRI include directory  : " << envs::libri_include_dir << "\n"
        << "| LibComm include directory: " << envs::libcomm_include_dir << "\n"
        << "| cereal include directory : " << envs::cereal_include_dir << "\n"
        << "| Third-party sources      : " << envs::thirdparty_sources << "\n"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        << "| DDLA_USE_CCL             : " << envs::ddla_use_ccl << "\n"
        << "| DDLA_USE_GPU_CPU_TUNNEL  : " << envs::ddla_use_gpu_cpu_tunnel << "\n"
        << "| LibDDLA path             : " << envs::libddla_path << "\n"
        << "| ELPA directory           : " << envs::elpa_dir << "\n"
#endif
        ;
        return ss.str();
    }();
    return s;
}

} /* end of namespace librpa_int */
