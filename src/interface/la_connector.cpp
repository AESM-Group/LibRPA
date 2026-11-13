#include "la_connector.h"

#include <cmath>
#include <type_traits>

#include "../utils/profiler.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla_vector.h>
#endif

namespace librpa_int
{
namespace LaConnector
{
namespace
{
template <typename T, bool Real>
matrix_m<std::complex<T>> power_hemat_la_impl(
    matrix_m<std::complex<T>>& A_local, const ArrayDesc& ad_A,
    matrix_m<std::complex<T>>& Z_local, const ArrayDesc& ad_Z, size_t& n_filtered, T* W,
    T power, const T& threshold, bool use_gpu, std::string method,
    std::conditional_t<Real, T, std::complex<T>>* d_A,
    std::conditional_t<Real, T, std::complex<T>>* d_Z,
    std::conditional_t<Real, T, std::complex<T>>* d_scaled)
{
    using Scalar = std::conditional_t<Real, T, std::complex<T>>;
    assert(A_local.is_col_major() && Z_local.is_col_major());
    const int n = ad_A.m();
    if (ad_A.n() != n || ad_Z.m() != n || ad_Z.n() != n ||
        ad_A.ictxt() != ad_Z.ictxt())
        throw std::invalid_argument("matrix power requires square A/Z on the same BLACS grid");
#ifndef LIBRPA_USE_DDLA
    if (method == "ddla") method = "scalapack";
#endif
#ifndef LIBRPA_USE_ELPA
    if (method == "elpa") method = "scalapack";
#endif
    if (method != "ddla" && method != "elpa") method = "scalapack";

    const char* label = Real ? "power_hemat_la_real" : "power_hemat_la";
    const auto weight = [&](T value, int index)
    {
        if (value < threshold) return T(0);
        if (value < 0 && std::fabs(power - int(power)) >= 1e-4)
            global::lib_printf(LIBRPA_VERBOSE_WARN,
                               "Warning! unfiltered negative eigenvalue with non-integer power: "
                               "# %d ev = %f , pow = %f\n", index, value, power);
        if (std::fabs(value) < 1e-10 && power < 0)
            global::lib_printf(LIBRPA_VERBOSE_WARN,
                               "Warning! unfiltered nearly-singular eigenvalue with negative "
                               "power: # %d ev = %f , pow = %f\n", index, value, power);
        return std::pow(value, power);
    };

    // Native PZHEEV's scalar quick return can yield an unnormalized vector.
    // Keep scalar ownership explicit, including different A/Z source ranks.
    if (n == 1)
    {
        global::profiler.start(label, LIBRPA_VERBOSE_DEBUG);
        W[0] = ad_A.is_src() ? A_local(0, 0).real() : T(0);
        MPI_Allreduce(MPI_IN_PLACE, W, 1, mpi_datatype<T>::value, MPI_SUM, ad_A.comm());
        n_filtered = W[0] < threshold ? 1 : 0;
        const T powered = weight(W[0], 0);
        auto scaled = Z_local.copy();
        scaled.zero_out();
        Z_local.zero_out();
        if (ad_A.is_src()) A_local(0, 0) = {powered, T(0)};
        if (ad_Z.is_src())
        {
            Z_local(0, 0) = {T(1), T(0)};
            scaled(0, 0) = {powered, T(0)};
        }
        global::profiler.stop(label);
        return scaled;
    }

    bool gpu = false;
    bool gpu_solver = false;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    const auto handle = ad_A.ddla_handle();
    gpu = method == "ddla" ? ddla::ddlaGetBackend(handle) == ddla::DdlaBackend::GPU : use_gpu;
    gpu_solver = gpu && method != "scalapack";
    if (gpu && (ddla::ddlaGetBackend(handle) != ddla::DdlaBackend::GPU ||
                ad_Z.ddla_handle() != handle))
        throw std::invalid_argument("GPU matrix power requires A/Z to share the caller-provided GPU handle");
    if (gpu && (d_A == nullptr || d_Z == nullptr || d_scaled == nullptr ||
                d_A == d_Z || d_A == d_scaled || d_Z == d_scaled))
        throw std::invalid_argument("GPU matrix power requires distinct caller-provided A/Z/scaled buffers");
#else
    (void)use_gpu;
    (void)d_A;
    (void)d_Z;
    (void)d_scaled;
#endif

    global::profiler.start(label, LIBRPA_VERBOSE_DEBUG);
    // Work in the caller's distribution; no descriptor conversion or reblocking.
    const auto local_capacity = [](const ArrayDesc& ad)
    {
        return std::max<size_t>(1, as_size(ad.lld()) * ad.n_loc());
    };
    const size_t a_capacity = local_capacity(ad_A), z_capacity = local_capacity(ad_Z);
    std::vector<Scalar> a_storage, z_storage, scaled_storage;
    Scalar *a, *z;
    if constexpr (Real)
    {
        a_storage.resize(a_capacity);
        z_storage.resize(z_capacity);
        a = a_storage.data();
        z = z_storage.data();
        for (size_t i = 0; i < A_local.size(); ++i) a[i] = -A_local.ptr()[i].real();
    }
    else
    {
        // Complex solves use the existing host A/Z allocations directly.
        a = A_local.ptr();
        z = Z_local.ptr();
        if (A_local.size() < a_capacity) { a_storage.resize(a_capacity); a = a_storage.data(); }
        if (Z_local.size() < z_capacity) { z_storage.resize(z_capacity); z = z_storage.data(); }
    }
    int info = 0;
    const std::string solve_label = std::string(label) + "_" + method;
    global::profiler.start(solve_label, LIBRPA_VERBOSE_DEBUG);
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (gpu_solver)
    {
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d_A, a, a_capacity * sizeof(Scalar)));
        if (method == "elpa")
        {
            ddla::vector<T> device_w(handle, as_size(n));
            ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
            pheev('V', 'U', n, d_A, 1, 1, ad_A, device_w.data(), d_Z, 1, 1, ad_Z, info, method);
            ddla::DDLA_CHECK(ddla::ddlaDeviceSynchronize(handle));
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, W, device_w.data(), as_size(n) * sizeof(T)));
            ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        }
        else
            pheev('V', 'U', n, d_A, 1, 1, ad_A, W, d_Z, 1, 1, ad_Z, info, method);
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, z, d_Z, z_capacity * sizeof(Scalar)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    }
    else
#endif
        pheev('V', 'U', n, a, 1, 1, ad_A, W, z, 1, 1, ad_Z, info, method);
    if (info != 0) throw std::runtime_error("matrix-power pheev failed: info=" + std::to_string(info));
    global::profiler.stop(solve_label);

    if constexpr (Real)
        for (int i = 0; i < n; ++i) W[i] = -W[i];
    if constexpr (Real)
        for (size_t i = 0; i < Z_local.size(); ++i) Z_local.ptr()[i] = {z[i], T(0)};
    // Complex eigenvalues are ascending; real eigenvalues are descending.
    n_filtered = as_size(n);
    for (int i = 0; i < n; ++i)
        if (W[Real ? n - 1 - i : i] >= threshold)
        {
            n_filtered = as_size(i);
            break;
        }
    std::vector<T> weights(as_size(n));
    for (int i = 0; i < n; ++i) weights[i] = weight(W[i], i);
    matrix_m<std::complex<T>> result;
    Scalar* scaled;
    if constexpr (Real)
    {
        scaled_storage.assign(z, z + z_capacity);
        scaled = scaled_storage.data();
    }
    else
    {
        result = gpu ? matrix_m<std::complex<T>>(ad_Z.m_loc(), ad_Z.n_loc(), MAJOR::COL)
                     : Z_local.copy();
        scaled = result.ptr();
        if (Z_local.size() < z_capacity)
        {
            scaled_storage.assign(z, z + z_capacity);
            scaled = scaled_storage.data();
        }
    }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (gpu)
    {
        // d_A becomes the powered matrix; d_scaled has Z's distribution.
        if (!gpu_solver)
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d_Z, z, z_capacity * sizeof(Scalar)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d_scaled, d_Z, z_capacity * sizeof(Scalar)));
        if (ad_Z.m_loc() > 0)
            for (int j = 0; j < n; ++j)
            {
                const int local = ad_Z.indx_g2l_c(j);
                if (local >= 0)
                    ddla::scal(handle, ad_Z.m_loc(), Scalar(weights[j]),
                               d_scaled + as_size(ad_Z.lld()) * local, 1);
            }
        LaConnector::pgemm('N', 'C', n, n, n, Scalar(1), d_Z, 1, 1, ad_Z,
                           d_scaled, 1, 1, ad_Z, Scalar(0), d_A, 1, 1, ad_A);
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, scaled, d_scaled, z_capacity * sizeof(Scalar)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, a, d_A, a_capacity * sizeof(Scalar)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    }
    else
#endif
    {
        if (ad_Z.m_loc() > 0)
            for (int j = 0; j < n; ++j)
            {
                const int local = ad_Z.indx_g2l_c(j);
                if (local >= 0)
                    LapackConnector::scal(ad_Z.m_loc(), weights[j],
                                         scaled + as_size(ad_Z.lld()) * local, 1);
            }
        LaConnector::pgemm('N', 'C', n, n, n, Scalar(1), z, 1, 1, ad_Z,
                           scaled, 1, 1, ad_Z, Scalar(0), a, 1, 1, ad_A);
    }
    if constexpr (Real)
    {
        for (size_t i = 0; i < A_local.size(); ++i) A_local.ptr()[i] = {a[i], T(0)};
        // Release real A/Z workspaces before allocating the complex return value.
        std::vector<Scalar>().swap(a_storage);
        std::vector<Scalar>().swap(z_storage);
        result = matrix_m<std::complex<T>>(ad_Z.m_loc(), ad_Z.n_loc(), MAJOR::COL);
        for (size_t i = 0; i < Z_local.size(); ++i) result.ptr()[i] = {scaled[i], T(0)};
    }
    global::profiler.stop(label);
    return result;
}
}  // namespace

template <typename T>
matrix_m<std::complex<T>> power_hemat_la(
    matrix_m<std::complex<T>>& A, const ArrayDesc& ad_A,
    matrix_m<std::complex<T>>& Z, const ArrayDesc& ad_Z, size_t& filtered, T* W,
    T power, const T& threshold, bool gpu, const std::string& method,
    std::complex<T>* d_A, std::complex<T>* d_Z, std::complex<T>* d_scaled)
{
    return power_hemat_la_impl<T, false>(A, ad_A, Z, ad_Z, filtered, W, power, threshold,
                                        gpu, method, d_A, d_Z, d_scaled);
}

template <typename T>
matrix_m<std::complex<T>> power_hemat_la_real(
    matrix_m<std::complex<T>>& A, const ArrayDesc& ad_A,
    matrix_m<std::complex<T>>& Z, const ArrayDesc& ad_Z, size_t& filtered, T* W,
    T power, const T& threshold, bool gpu, const std::string& method,
    T* d_A, T* d_Z, T* d_scaled)
{
    return power_hemat_la_impl<T, true>(A, ad_A, Z, ad_Z, filtered, W, power, threshold,
                                       gpu, method, d_A, d_Z, d_scaled);
}

#define INSTANTIATE_POWER(T) \
template matrix_m<std::complex<T>> power_hemat_la( \
    matrix_m<std::complex<T>>&, const ArrayDesc&, matrix_m<std::complex<T>>&, \
    const ArrayDesc&, size_t&, T*, T, const T&, bool, const std::string&, \
    std::complex<T>*, std::complex<T>*, std::complex<T>*); \
template matrix_m<std::complex<T>> power_hemat_la_real( \
    matrix_m<std::complex<T>>&, const ArrayDesc&, matrix_m<std::complex<T>>&, \
    const ArrayDesc&, size_t&, T*, T, const T&, bool, const std::string&, T*, T*, T*);

INSTANTIATE_POWER(float)
INSTANTIATE_POWER(double)
#undef INSTANTIATE_POWER
}  // namespace LaConnector
}  // namespace librpa_int
