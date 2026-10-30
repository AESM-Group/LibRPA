#!/bin/bash
#SBATCH -p normal
#SBATCH -J build_cpuri_hip
#SBATCH --nodes=1
#SBATCH --gres=dcu:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/public/home/hbchen/app/LibRPA/260531/librpa_hip_cpuri.out
#SBATCH --error=/public/home/hbchen/app/LibRPA/260531/librpa_hip_cpuri.out
#SBATCH --open-mode=truncate

set -eo pipefail

export LC_ALL=C
export LANG=C
export LANGUAGE=C

ulimit -s unlimited
ulimit -c unlimited

unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
unset LIBRARY_PATH LD_LIBRARY_PATH LD_RUN_PATH
unset PKG_CONFIG_PATH CMAKE_PREFIX_PATH
unset LIBDDLA_PATH LIBRI_INCLUDE_DIR
unset MAGMA_ROOT MAGMA_DIR
unset ELPA_PATH ELPA_ROOT EXTERNAL_ELPA_DIR ELPA_DIR
module purge

module load compiler/rocm/dtk/25.04.3
export CPATH="${ROCM_PATH}/include/rocrand"
export LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${ROCM_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
module load compiler/devtoolset/9.3.1
module load mpi/hpcx/2.13.1/gcc-9.3.1-wangxh
module load compiler/cmake/3.24.1

tool_root=/public/home/hbchen/app/LibRPA/260212/toolchain
tool_install="${tool_root}/install"
tool_build="${tool_root}/build"
source "${tool_build}/setup_openblas_extern"
source "${tool_build}/setup_scalapack_extern"
source "${tool_build}/setup_cereal_extern"

repo_root=/public/home/hbchen/app/LibRPA/260531/LibRPA
work_root=/public/home/hbchen/app/LibRPA/260531
job_id="${SLURM_JOB_ID:?SLURM_JOB_ID is required}"
build_dir="${work_root}/build_hip_cpuri"
install_dir="${work_root}/librpa_hip_cpuri"

if [[ "${LIBRPA_REUSE_BUILD:-0}" != 1 ]]; then
    rm -rf "${build_dir}"
fi

scalapack_dir="${tool_install}/scalapack-2.2.2/lib"
cereal_include="${tool_install}/cereal-master/include"
elpa_configure_args="HIPCC=hipcc HIPCCFLAGS='-DROCBLAS_V3 -D__HIP_PLATFORM_AMD__ -g -O3 -std=c++17 --gpu-max-threads-per-block=1024' --enable-option-checking=fatal --enable-amd-gpu-kernels --enable-single-precision --enable-gpu-streams=amd --enable-gpu-ccl=rccl --enable-hipcub --disable-cpp-tests --with-rocsolver"
elpa_libs="-lscalapack -lopenblas -lamdhip64 -lgalaxyhip -lrocblas -lrocsolver -lrccl -lhipblas -lhipsolver -lhiprand -fPIC -Wno-return-type"

echo "SLURM_JOB_ID=${job_id}"
echo "REPO_ROOT=${repo_root}"
echo "BUILD_DIR=${build_dir}"
echo "INSTALL_DIR=${install_dir}"
echo "ROCM_PATH=${ROCM_PATH}"
echo "START_TIME=$(date --iso-8601=seconds)"

cmake -S "${repo_root}" -B "${build_dir}" \
    -DCMAKE_INSTALL_PREFIX="${install_dir}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DMPI_C_COMPILER=mpicc \
    -DMPI_CXX_COMPILER=mpicxx \
    -DMPI_Fortran_COMPILER=mpifort \
    -DCMAKE_Fortran_COMPILER=gfortran \
    -DSCALAPACK_DIR="${scalapack_dir}" \
    -DCEREAL_INCLUDE_DIR="${cereal_include}" \
    -DLIBRPA_USE_LIBRI=ON \
    -DLIBRPA_USE_LIBRI_GPU=OFF \
    -DLIBRPA_USE_HIP=ON \
    -DLIBRPA_USE_EXTERNAL_ELPA=OFF \
    -DLIBRPA_USE_BUNDLED_ELPA=ON \
    -DLIBRPA_BUNDLED_ELPA_VERSION=2026.02.001 \
    -DLIBRPA_BUNDLED_ELPA_OPENMP=OFF \
    -DLIBRPA_BUNDLED_ELPA_CONFIGURE_ARGS:STRING="${elpa_configure_args}" \
    -DLIBRPA_BUNDLED_ELPA_LIBS:STRING="${elpa_libs}" \
    -DBUILD_SHARED_LIBS=ON \
    -DLIBRPA_VERBOSE_OUTPUT=ON \
    -DCMAKE_C_FLAGS="-g -O3" \
    -DCMAKE_CXX_FLAGS="-DROCBLAS_V3 -D__HIP_PLATFORM_AMD__ -g -O2" \
    -DCMAKE_Fortran_FLAGS="-g -O2" \
    -DCMAKE_HIP_FLAGS="-g -O2 -fopenmp -fgpu-rdc -Wno-return-type -Wno-pass-failed" \
    -DROCM_PATH="${ROCM_PATH}" \
    -DCMAKE_PREFIX_PATH="${ROCM_PATH}"

cmake --build "${build_dir}" -j "${SLURM_CPUS_PER_TASK}"

grep -q '^LIBRPA_USE_LIBRI_GPU:BOOL=OFF$' "${build_dir}/CMakeCache.txt"
if grep -Eq -- "-D(__GPU_RI|__DDLA_RI|__HIP_RI|__CUDA_RI)" "${build_dir}/compile_commands.json"; then
    echo "Unexpected GPU LibRI definition in the HIP CPURI build" >&2
    grep -E -- "-D(__GPU_RI|__DDLA_RI|__HIP_RI|__CUDA_RI)" "${build_dir}/compile_commands.json" >&2
    exit 1
fi

grep -q '^LIBRPA_USE_EXTERNAL_ELPA:BOOL=OFF$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_USE_BUNDLED_ELPA:BOOL=ON$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_BUNDLED_ELPA_VERSION:STRING=2026.02.001$' "${build_dir}/CMakeCache.txt"
grep -q '^LIBRPA_BUNDLED_ELPA_OPENMP:BOOL=OFF$' "${build_dir}/CMakeCache.txt"

elpa_config=$(find "${build_dir}/thirdparty/ELPA" -name config.h -print -quit)
test -n "${elpa_config}"
for macro in \
    WITH_MPI \
    WITH_AMD_GPU_KERNEL \
    WITH_AMD_RCCL \
    WITH_AMD_ROCSOLVER \
    WITH_GPU_STREAMS \
    WITH_HIPCUB \
    WANT_SINGLE_PRECISION_REAL \
    WANT_SINGLE_PRECISION_COMPLEX; do
    grep -q "^#define ${macro} 1$" "${elpa_config}"
done

backup_dir=""
restore_previous_install()
{
    status=$?
    if [[ "${status}" -ne 0 && -n "${backup_dir}" && -d "${backup_dir}" ]]; then
        rm -rf "${install_dir}"
        mv "${backup_dir}" "${install_dir}"
        echo "Restored previous installation from ${backup_dir}" >&2
    fi
    exit "${status}"
}
trap restore_previous_install EXIT

if [[ -e "${install_dir}" ]]; then
    backup_dir="${install_dir}.pre-bundled-elpa-backup-$(date +%Y%m%d-%H%M%S)"
    mv "${install_dir}" "${backup_dir}"
fi

cmake --install "${build_dir}"

test -x "${install_dir}/bin/chi0_main.exe"
ddla_library=""
for library_dir in "${install_dir}/lib" "${install_dir}/lib64"; do
    if [[ -e "${library_dir}/libddla.so" ]]; then
        ddla_library="${library_dir}/libddla.so"
        break
    fi
done
test -n "${ddla_library}"

if find "${install_dir}" \( -name 'libelpa.so*' -o -name 'libelpa_openmp.so*' \) | grep -q .; then
    echo "Bundled static ELPA unexpectedly installed a shared ELPA library" >&2
    find "${install_dir}" \( -name 'libelpa.so*' -o -name 'libelpa_openmp.so*' \) >&2
    exit 1
fi

rpa_library=""
for library_dir in "${install_dir}/lib" "${install_dir}/lib64"; do
    if [[ -e "${library_dir}/librpa.so" ]]; then
        rpa_library="${library_dir}/librpa.so"
        break
    fi
done
test -n "${rpa_library}"

for binary in "${install_dir}/bin/chi0_main.exe" "${rpa_library}"; do
    if readelf -d "${binary}" | grep -q 'NEEDED.*libelpa'; then
        echo "Bundled static ELPA unexpectedly appears as a runtime dependency: ${binary}" >&2
        readelf -d "${binary}" >&2
        exit 1
    fi
done

export LD_LIBRARY_PATH="${install_dir}/lib:${install_dir}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for binary in "${install_dir}/bin/chi0_main.exe" "${rpa_library}"; do
    if ldd "${binary}" | grep -q "not found"; then
        echo "Unresolved runtime dependency: ${binary}" >&2
        ldd "${binary}" >&2
        exit 1
    fi
    if ldd "${binary}" | grep -qi magma; then
        echo "Bundled HIP CPURI binary unexpectedly depends on MAGMA: ${binary}" >&2
        ldd "${binary}" >&2
        exit 1
    fi
done

trap - EXIT

echo "BUNDLE_HIP_CPURI_BUILD_PASS"
echo "PREVIOUS_INSTALL_BACKUP=${backup_dir:-none}"
echo "END_TIME=$(date --iso-8601=seconds)"
