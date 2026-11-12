#include <cassert>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "../gpu/la_connector.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla.h>
#endif
#include "elpa_connector.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/global/Tensor.h>
#else
#include "../utils/libri_stub.h"
#endif
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/profiler.h"
#include "../core/utils_atomic_basis_blacs.h"
#include <vector>

namespace librpa_int
{

namespace ElpaConnector
{

// FIXME: need review with ELPA and GPU/HIP
/*!
 * @brief Compute power of Hermitian matrix using BLACS
 *
 * @param  [in,out]  A_local       Process-local part of global matrix A to be powered
 * @param  [in]      ad_A          Array descriptor of A
 * @param  [out]     Z_local       Process-local part of the eigenvector matrix (Z) of A
 * @param  [in]      ad_Z          Array descriptor of Z
 * @param  [out]     n_filtered    Array descriptor of Z
 * @param  [out]     W             Eigenvalues of A, including those smaller than threshold
 * @param  [in]      power         Power to perform
 * @param  [in]      threshold     The threshold to filter the eigenvalues
 *
 * @retval           scale_Z       Eigenvectors scaled by the power of eigenvalues, using ad_Z
 */
template <typename T>
matrix_m<std::complex<T>> power_hemat_elpa(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, matrix_m<std::complex<T>> &Z_local,
    const ArrayDesc &ad_Z, size_t &n_filtered, T *W, T power, const T &threshold,
    bool use_gpu_replace_scalapack, std::complex<T>* d_A, std::complex<T>* d_Z, std::complex<T>* d_C)
{
    using global::ofs_myid;
    using global::profiler;

    profiler.start(__FUNCTION__);

    assert (A_local.is_col_major() && Z_local.is_col_major());
    const bool is_int_power = fabs(power - int(power)) < 1e-4;
    const size_t n = ad_A.m();

    // initialize descriptor of array Z for optimized block size
    if (ad_A.ictxt() != ad_Z.ictxt())
    {
        ofs_myid << "Warning(power_hemat_blacs): input contexts of A and Z are different!" << std::endl;
    }

    global::profiler.start("power_hemat_blacs_2");
    std::complex<T> *A, *Z;
    T* W_uni;

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    auto ddla_handle = ad_A.ddla_handle();
    T* d_W;
    if(use_gpu_replace_scalapack){
        ddla::DDLA_CHECK(ddla::ddlaMalloc(ddla_handle, (void**)&d_W, n * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, d_A, A_local.ptr(), A_local.size() * sizeof(std::complex<T>)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, d_Z, Z_local.ptr(), Z_local.size() * sizeof(std::complex<T>)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, d_W, W, n * sizeof(T)));
        
        A = d_A;
        Z = d_Z;
        W_uni = d_W;
    }else
#endif
    {
        A = A_local.ptr();
        Z = Z_local.ptr();
        W_uni = W;
    }
    int error;
    elpa_eigenvectors(ad_A.elpa_handle(), A, W_uni, Z, &error);
    if(error != ELPA_OK){
        throw std::runtime_error("elpa eigenvectors error\n");
    }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(use_gpu_replace_scalapack){
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, W, W_uni, n * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, Z_local.ptr(), Z, Z_local.size()*sizeof(std::complex<T>)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(ddla_handle));
    }
#endif

    global::profiler.stop("power_hemat_blacs_2");

    // check the number of non-singular eigenvalues,
    // using the fact that W is in ascending order
    n_filtered = n;
    for (size_t i = 0; i != n; i++)
        if (W[i] >= threshold)
        {
            n_filtered = i;
            break;
        }

    // filter and scale the eigenvalues, store in a temp array
    profiler.start("power_hemat_blacs_4");
    std::vector<T> W_temp(n);
    for (size_t i = 0; i < n_filtered; i++) W_temp[i] = 0.0;
    for (size_t i = n_filtered; i != n; i++)
    {
        if (W[i] < 0 && !is_int_power)
        {
            global::lib_printf(
                LIBRPA_VERBOSE_WARN,
                "Warning! unfiltered negative eigenvalue with non-integer power: # %d ev = %f , "
                "pow = %f\n",
                i, W[i], power);
        }
        if (fabs(W[i]) < 1e-10 && power < 0)
        {
            global::lib_printf(
                LIBRPA_VERBOSE_WARN,
                "Warning! unfiltered nearly-singular eigenvalue with negative power: # %d ev = %f "
                ", pow = %f\n",
                i, W[i], power);
        }
        W_temp[i] = std::pow(W[i], power);
    }
    profiler.stop("power_hemat_blacs_4");
    // debug print
    // for (int i = 0; i != n; i++)
    // {
    //     librpa_int::global::lib_printf("%d %f %f\n", i, W[i], W_temp[i]);
    // }

    profiler.start("power_hemat_blacs_5");
    // create scaled eigenvectors
    auto scaled = Z_local.copy();
    std::complex<T> *C;
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if(use_gpu_replace_scalapack){
        ddla::DDLA_CHECK(ddla::ddlaMalloc(ddla_handle, (void**)&d_C, A_local.size() * sizeof(std::complex<T>)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, d_A, d_Z, Z_local.size() * sizeof(std::complex<T>)));
        C = d_C;
    }else
#endif
    {
        Z = Z_local.ptr();
        A = scaled.ptr();
        C = A_local.ptr();
    }
    for (int i = 0; i != n; i++)
    {
        int j_loc = ad_Z.indx_g2l_c(i);
        if(j_loc>=0)
            LaConnector::scal(ad_Z.m_loc(), W_temp[i], A + ad_Z.lld() * j_loc, 1, ad_Z);
    }
    LaConnector::pgemm('N', 'C', n, n, n, {(T)1.0, (T)0.0}, Z, 1, 1, ad_Z, A, 1, 1, ad_Z, {(T)0.0, (T)0.0}, C, 1, 1, ad_A);
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if(use_gpu_replace_scalapack){
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, scaled.ptr(), A, scaled.size() * sizeof(std::complex<T>)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, A_local.ptr(), C, A_local.size() * sizeof(std::complex<T>)));
        ddla::DDLA_CHECK(ddla::ddlaFree(ddla_handle, d_C));
        ddla::DDLA_CHECK(ddla::ddlaFree(ddla_handle, d_W));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(ddla_handle));
    }
#endif
    // send back the scaled eigenvector matrix with descriptor using optimized block size to that
    profiler.stop("power_hemat_blacs_5");
    profiler.stop(__FUNCTION__);

    return scaled;
}


template <typename T>
matrix_m<std::complex<T>> power_hemat_elpa_real(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, matrix_m<std::complex<T>> &Z_local,
    const ArrayDesc &ad_Z, size_t &n_filtered, T *W, T power, const T &threshold, 
    bool use_gpu_replace_scalapack, T* d_A, T* d_Z, T* d_C)
{
    using global::ofs_myid;
    using global::profiler;

    profiler.start(__FUNCTION__);
    // Step 1: Extract real part of the complex matrix
    auto A_local_real = A_local.get_real();
    A_local_real *= -1.0;
    assert(A_local.is_col_major() && Z_local.is_col_major());
    const bool is_int_power = fabs(power - int(power)) < 1e-4;
    const int n = ad_A.m();

    // initialize descriptor of array Z for optimized block size
    if (ad_A.ictxt() != ad_Z.ictxt())
    {
        ofs_myid << "Warning(power_hemat_blacs): input contexts of A and Z are different!\n";
    }
    auto Z_local_real = init_local_mat<T>(ad_Z, MAJOR::COL);

    profiler.start("power_hemat_blacs_1");
    profiler.stop("power_hemat_blacs_1");
    
    profiler.start("power_hemat_blacs_2");
    T *A, *Z, *W_uni;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    auto ddla_handle = ad_A.ddla_handle();
    T *d_W;
    if(use_gpu_replace_scalapack)
    {
        profiler.start("power_hemat_blacs_2_device_prepare");
        ddla::DDLA_CHECK(ddla::ddlaMalloc(ddla_handle, (void**)&d_W, n * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, d_A, A_local_real.ptr(), A_local_real.size() * sizeof(T)));
        profiler.stop("power_hemat_blacs_2_device_prepare");
        A = d_A;
        Z = d_Z;
        W_uni = d_W;
    }else
#endif
    {
        A = A_local_real.ptr();
        Z = Z_local_real.ptr();
        W_uni = W;
    }
    int error;
    profiler.start("power_hemat_blacs_2_elpa_eigenvectors");
    elpa_eigenvectors(ad_A.elpa_handle(), A, W_uni, Z, &error);
    profiler.stop("power_hemat_blacs_2_elpa_eigenvectors");
    if(error != ELPA_OK){
        throw std::runtime_error("elpa eigenvectors error\n");
    }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(use_gpu_replace_scalapack){
        profiler.start("power_hemat_blacs_2_device_download");
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, W, W_uni, n * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, Z_local_real.ptr(), Z, Z_local_real.size()*sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(ddla_handle));
        profiler.stop("power_hemat_blacs_2_device_download");
    }
#endif
    for (int i = 0; i < n; i++)
    {
        W[i] *= -1.0;
    }
    // A_local_real.clear();

    // Convert real eigenvectors to complex (with zero imaginary part)
    for (size_t i = 0; i < Z_local.size(); i++)
        Z_local.ptr()[i] = {Z_local_real.ptr()[i], T(0)};
    profiler.stop("power_hemat_blacs_2");

    // Check number of non-singular eigenvalues
    n_filtered = n;
    for (int i = 0; i < n; i++)
    {
        if (W[n - 1 - i] >= threshold)
        {
            n_filtered = i;
            break;
        }
    }

    // Filter and scale eigenvalues
    profiler.start("power_hemat_blacs_3");
    std::vector<T> W_temp(n, T(0));
    for (int i = 0; i < n - n_filtered; i++)
    {
        if (W[i] < 0 && !is_int_power)
        {
            global::lib_printf(
                LIBRPA_VERBOSE_WARN,
                "Warning! unfiltered negative eigenvalue with non-integer power: # %d ev = %f , "
                "pow = %f\n",
                i, W[i], power);
        }
        if (fabs(W[i]) < 1e-10 && power < 0)
        {
            global::lib_printf(
                LIBRPA_VERBOSE_WARN,
                "Warning! unfiltered nearly-singular eigenvalue with negative power: # %d ev = %f "
                ", pow = %f\n",
                i, W[i], power);
        }
        W_temp[i] = std::pow(W[i], power);
    }
    profiler.stop("power_hemat_blacs_3");

    profiler.start("power_hemat_blacs_4");
    // Scale eigenvectors using complex matrix operations
    auto scaled_real = Z_local_real.copy();
    T* C;
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if(use_gpu_replace_scalapack)
    {
        ddla::DDLA_CHECK(ddla::ddlaMalloc(ddla_handle, (void**)&d_C, A_local_real.size() * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, d_A, d_Z, Z_local_real.size() * sizeof(T)));
        C = d_C;
    }else
#endif
    {
        Z = Z_local_real.ptr();
        A = scaled_real.ptr();
        C = A_local_real.ptr();
    }
    for (int i = 0; i < n; i++)
    {
        // Use ScaLAPACK scaling function with complex matrix
        int j_loc = ad_Z.indx_g2l_c(i);
        if(j_loc>=0)
            LaConnector::scal(ad_Z.m_loc(), W_temp[i], A + ad_Z.lld() * j_loc, 1, ad_Z);
    }

    // Compute Z * diag(W_temp) * Z^H
    LaConnector::pgemm('N', 'C', n, n, n, (T)1.0, Z, 1, 1, ad_Z, A, 1, 1, ad_Z, (T)0.0, C, 1, 1, ad_A);
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if(use_gpu_replace_scalapack){
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, scaled_real.ptr(), A, scaled_real.size() * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(ddla_handle, A_local_real.ptr(), C, A_local_real.size() * sizeof(T)));
        ddla::DDLA_CHECK(ddla::ddlaFree(ddla_handle, d_W));
        ddla::DDLA_CHECK(ddla::ddlaFree(ddla_handle, d_C));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(ddla_handle));
    }
#endif
    for (size_t i = 0; i < A_local.size(); i++)
        A_local.ptr()[i] = {A_local_real.ptr()[i], T(0)};
    auto scaled = scaled_real.to_complex();

    profiler.stop("power_hemat_blacs_4");
    profiler.stop(__FUNCTION__);

    return scaled;
}

template matrix_m<std::complex<double>> power_hemat_elpa_real<double>(
    matrix_m<std::complex<double>> &A_local, const ArrayDesc &ad_A,
    matrix_m<std::complex<double>> &Z_local, const ArrayDesc &ad_Z,
    size_t &n_filtered, double *W, double power, const double &threshold,
    bool use_gpu_replace_scalapack, double* d_A, double* d_Z, double* d_C);
template matrix_m<std::complex<float>> power_hemat_elpa_real<float>(
    matrix_m<std::complex<float>> &A_local, const ArrayDesc &ad_A,
    matrix_m<std::complex<float>> &Z_local, const ArrayDesc &ad_Z,
    size_t &n_filtered, float *W, float power, const float &threshold,
    bool use_gpu_replace_scalapack, float* d_A, float* d_Z, float* d_C);

template matrix_m<std::complex<double>> power_hemat_elpa<double>(
    matrix_m<std::complex<double>> &A_local, const ArrayDesc &ad_A, matrix_m<std::complex<double>> &Z_local,
     const ArrayDesc &ad_Z, size_t &n_filtered, double *W, double power, const double &threshold,
     bool use_gpu_replace_scalapack, std::complex<double>* d_A, std::complex<double>* d_Z, std::complex<double>* d_C);
template matrix_m<std::complex<float>> power_hemat_elpa<float>(
    matrix_m<std::complex<float>> &A_local, const ArrayDesc &ad_A, matrix_m<std::complex<float>> &Z_local,
    const ArrayDesc &ad_Z, size_t &n_filtered, float *W, float power, const float &threshold,
    bool use_gpu_replace_scalapack, std::complex<float>* d_A, std::complex<float>* d_Z, std::complex<float>* d_C);

} // namespace ElpaConnector


} // namespace librpa_int
