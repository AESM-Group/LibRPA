# Installation

## Dependencies

LibRPA depends on the following core software components:

- a C++ compiler and an MPI library
- a Fortran compiler (and MPI support if [`LIBRPA_ENABLE_FORTRAN_BIND`](<librpa-enable-fortran-bind>) is enabled)
- BLAS and LAPACK libraries
- a ScaLAPACK library
- the [GreenX](https://github.com/nomad-coe/greenX) library for minimax
  time-frequency grids

Optionally, LibRPA can also be linked with an external
[ELPA](https://elpa.mpcdf.mpg.de/) installation. This is intended for
ELPA-backed optimized linear algebra subroutines. To enable the build
interface, configure with
`-DLIBRPA_USE_EXTERNAL_ELPA=ON -DEXTERNAL_ELPA_DIR=/path/to/elpa`.
Alternatively, LibRPA can build a bundled ELPA source release with
`-DLIBRPA_USE_BUNDLED_ELPA=ON`.

For *GW*, the following packages are additionally required:

- [LibRI](https://github.com/abacusmodeling/LibRI) for tensor contractions
- [LibComm](https://github.com/abacusmodeling/LibComm), which is required by LibRI for communication of tensor data between processes
- [cereal](https://uscilab.github.io/cereal), which is required by LibRI for data serialization

LibRI support is enabled by default and can be disabled for an RPA-only build
with `-DLIBRPA_USE_LIBRI=OFF`.

CUDA and HIP builds additionally require
[LibDDLA](https://github.com/userchba/LibDDLA) for distributed GPU linear
algebra. A compatible LibDDLA revision is bundled under `thirdparty/LibDDLA`;
an external installation can also be supplied through `LIBDDLA_PATH`.

The required source dependencies are bundled under the `thirdparty/`
directory in current LibRPA releases.

## Download

You can obtain the LibRPA code by cloning the GitHub repository:

```bash
git clone https://github.com/AESM-Group/LibRPA
```

For commits before `28b7431` (including tag `v0.4.0-gw-benchmark` and older),
LibRI and LibComm are included as Git submodules.
In this case, initialize the submodules before compiling with LibRI:
```bash
cd LibRPA
git submodule update --init --recursive
```

The source tree is now ready for compilation.

## Compile

To compile LibRPA, you need working compiler and library toolchains for C++,
Fortran, MPI, BLAS/LAPACK, and ScaLAPACK.

The Intel compilers and Intel MPI together with MKL from Intel oneAPI are often
the most straightforward choice. Alternatively, LibRPA can also be built with
GCC/GFortran together with an open-source MPI implementation such as
[MPICH](https://www.mpich.org) and an open-source ScaLAPACK library [Netlib ScaLAPACK](https://www.netlib.org/scalapack).

LibRPA uses CMake as its build system.
Ensure that the compilers and required libraries can be found under directories specified
by relevant environment variables, and under the root directory of the source tree:
```bash
mkdir build
cd build
cmake ..
make -j 4
```
This searches for the paths of required dependencies, and builds the LibRPA library and the driver executable.

After a successful build, the driver executable `chi0_main.exe` and the shared
library `src/librpa.so` (`src/librpa.dylib` on macOS, or `src/librpa.a` if `BUILD_SHARED_LIBS` is disabled)
can be found in the **build directory**.

You can specify the compilers through environment variables when invoking
CMake. For example, to use the Intel classic C++ and Fortran compilers:

```bash
CXX=mpiicpc FC=mpiifort cmake ..
```

To help CMake find the correct BLAS/LAPACK and ScaLAPACK libraries at link
time, you may need to ensure that the corresponding library directories are
visible through `LIBRARY_PATH` or `LD_LIBRARY_PATH`. For example, when using
MKL:

```bash
export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:$LD_LIBRARY_PATH"
CXX=mpiicpc FC=mpiifort cmake ..
```

By default, LibRPA builds and links against the bundled GreenX source
distributed under thirdparty/greenX.
If you want to use an external GreenX instead, you should enable the CMake
option [`LIBRPA_USE_EXTERNAL_GREENX`](<librpa-use-external-greenx>):

```bash
cmake -DLIBRPA_USE_EXTERNAL_GREENX=ON ..
```

In this case, LibRPA does not build the bundled GreenX copy. Instead, the
parent or higher-level CMake project must provide the external GreenX target
`LibGXMiniMax`.

Several build scripts are provided on the [build examples](../examples/build/index)
page to help users build LibRPA on different platforms and with different toolchains.
You may use them as starting points and adapt them to your local environment.

For a complete list of compile options, please refer to the
[Compile Options](compile_options) page.

## Troubleshooting

### Migration from `cmake.inc`

LibRPA no longer loads `cmake.inc`. Setting `LIBRPA_USE_CMAKE_INC=ON`
stops configuration and prints migration instructions.

Move your compiler settings, flags, and library settings to an initial cache
file, using `set(... CACHE <type> <description>)` for each entry:

```cmake
set(CMAKE_CXX_COMPILER "mpicxx" CACHE FILEPATH "C++ compiler")
set(CMAKE_Fortran_COMPILER "mpifort" CACHE FILEPATH "Fortran compiler")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
set(SCALAPACK_DIR "/path/to/scalapack" CACHE PATH "ScaLAPACK installation")
```

Load this file with CMake's `-C` option:

```bash
cmake -B build -C initial-cache.cmake

# Remove the old option when reusing an existing build directory.
cmake -B build -C initial-cache.cmake -U LIBRPA_USE_CMAKE_INC
```

The `-U` argument clears the old option from the cache. You can leave an
existing `LIBRPA_USE_CMAKE_INC=OFF` entry in place.
Use a new build directory if you change compilers.

### `std::filesystem` link errors with Intel compilers

When building LibRPA with Intel compilers, the final link step may fail with errors similar to

```text
undefined reference to `std::filesystem::create_directories(...)'
undefined reference to `std::filesystem::status(...)'
undefined reference to `std::filesystem::__cxx11::path::_M_split_cmpts()'
```

This is usually not a LibRPA source-code issue. On Linux, Intel compilers use GCC’s C++ standard library, libstdc++.
If the compiler wrapper picks up an old system GCC/libstdc++, C++17 `std::filesystem` symbols may be unavailable or may require extra linking.

A recommended solution is to use a recent GCC version before configuring and building LibRPA.
On HPC, it usually amounts to loading a recent GCC module, for example

```bash
module load gcc/13.4.0
```

Then rerun CMake from a clean build directory.

### ELPA `<complex.h>` macro conflict

When compiling LibRPA with external ELPA, C++ compilation may fail after
including `elpa/elpa.h` with errors similar to

```text
error: expected identifier before '(' token
error: expected ']' before '(' token
```

This can happen when `elpa/elpa.h` includes the C header `<complex.h>`, which
defines the macro `I`. The macro can then conflict with LibRPA C++ code that
uses `I` as a normal variable name. Official ELPA releases up to `2025.01.x`
may be affected; `2025.06.001` and newer use a C++ guard so C++ code sees
`<complex>` instead.

The recommended solution is to use ELPA `2025.06.001` or newer. If you must use
an affected ELPA version, a local workaround is to patch the installed
`elpa/elpa.h` header by replacing

```c
#include <complex.h>
```

with

```c
#ifdef __cplusplus
#include <complex>
#else
#include <complex.h>
#endif
```

Then rerun CMake from a clean build directory.
