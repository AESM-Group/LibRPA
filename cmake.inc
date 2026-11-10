# Basic C++ compiler setup
# set(CMAKE_C_COMPILER "mpiicc" CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER "mpiicpx" CACHE STRING "" FORCE)
set(CMAKE_Fortran_COMPILER "ifx" CACHE STRING "" FORCE)

# -DLIBRPA_VERBOSE
set(CMAKE_CXX_FLAGS "-g -O2 -qopenmp -DLIBRPA_VERBOSE" CACHE STRING "" FORCE)
if(DEBUG)
  set(CMAKE_CXX_FLAGS "-g -O0 -Wall -qopenmp" CACHE STRING "" FORCE)
endif()

set(CEREAL_INCLUDE_DIR "/mnt/sg001/home/ks_iopcas_rxg/software/cereal-1.3.0/include")
set(LIBRI_INCLUDE_DIR "/mnt/sg001/home/ks_iopcas_rxg/ghj/app/LibRI/include")
set(LIBCOMM_INCLUDE_DIR "/mnt/sg001/home/ks_iopcas_rxg/ghj/app/LibComm/include")


# set(LIBRI_INCLUDE_DIR "/home/zhangmy/programs/LibRI-loop3-mygh/include")
# set(LIBRI_INCLUDE_DIR "/home/bhjia/LibRI-wait/include")


# set(LAPACK_LIBRARIES "-L$ENV{MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core")
set(LAPACK_LIBRARIES "-L$ENV{MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lmkl_blacs_intelmpi_lp64 -lpthread -lm -ldl")

### set(CMAKE_BUILD_TYPE "Debug")

# whether to use LibRI for tensor contraction
# set(USE_LIBRI ON)
