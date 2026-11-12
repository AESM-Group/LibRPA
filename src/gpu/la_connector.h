#ifndef LA_CONNECTOR_H
#define LA_CONNECTOR_H

#include <string>
#include <vector>

#ifdef LIBRPA_USE_DDLA
#include <ddla/ddla.h>
#endif
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include "device_connector.h"
#endif
#include "../math/matrix_m.h"
#include "../math/scalapack_connector.h"
#include "../mpi/base_blacs.h"
#ifdef LIBRPA_USE_ELPA
#include "../elpa/elpa_connector.h"
#endif

namespace librpa_int
{

namespace LaConnector
{

// A/Z are caller-owned pointers in the selected solver's memory space.
// W is host memory for DDLA/ScaLAPACK; ELPA uses its native A/W/Z placement.
// Handles and GPU buffers must be prepared by the caller. ScaLAPACK workspace
// is queried here. DDLA/ELPA currently operate on complete, source-zero arrays.
template <typename T, typename Real = typename to_real<T>::type>
inline void pheev(char jobz, char uplo, int n, T* A, int ia, int ja, const ArrayDesc& ad_A, Real* W,
                  T* Z, int iz, int jz, const ArrayDesc& ad_Z, int& info,
                  const std::string& method = "scalapack")
{
    info = 0;
#ifdef LIBRPA_USE_DDLA
    if (method == "ddla")
    {
        if (ia != 1 || ja != 1 || iz != 1 || jz != 1)
            throw std::invalid_argument("DDLA pheev requires unit submatrix offsets");
        const auto handle = ad_A.ddla_handle();
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        if (ddla::ddlaGetBackend(handle) == ddla::DdlaBackend::GPU)
            ddla::pheevd<ddla::DdlaBackend::GPU>(handle, jobz, uplo, n, A, ad_A.desc, W, Z,
                                                 ad_Z.desc, info);
        else
#endif
            ddla::pheevd<ddla::DdlaBackend::CPU>(handle, jobz, uplo, n, A, ad_A.desc, W, Z,
                                                 ad_Z.desc, info);
        return;
    }
#endif
#ifdef LIBRPA_USE_ELPA
    if (method == "elpa")
    {
        if (ia != 1 || ja != 1 || iz != 1 || jz != 1 || n != ad_A.m() || n != ad_A.n() ||
            n != ad_Z.m() || n != ad_Z.n() ||
            (uplo != 'U' && uplo != 'u'))
            throw std::invalid_argument(
                "ELPA pheev requires the complete upper-triangle matrix configured on its handle");
        if (jobz == 'V' || jobz == 'v')
            elpa_eigenvectors(ad_A.elpa_handle(), A, W, Z, &info);
        else if (jobz == 'N' || jobz == 'n')
            elpa_eigenvalues(ad_A.elpa_handle(), A, W, &info);
        else
            throw std::invalid_argument("ELPA pheev requires JOBZ=N or V");
        return;
    }
#endif

    T work_query{};
    Real rwork_query{}, w_query{};
    if constexpr (is_complex_t<T>::value)
        ScalapackConnector::pheev_f(jobz, uplo, n, A, ia, ja, ad_A.desc, &w_query, Z, iz, jz,
                                    ad_Z.desc, &work_query, -1, &rwork_query, -1, info);
    else
        ScalapackConnector::psyev_f(jobz, uplo, n, A, ia, ja, ad_A.desc, &w_query, Z, iz, jz,
                                    ad_Z.desc, &work_query, -1, info);
    if (info != 0) return;
    const int lwork = std::max(1, static_cast<int>(get_real(work_query)));
    std::vector<T> work(lwork);
    if constexpr (is_complex_t<T>::value)
    {
        const int lrwork = std::max(1, static_cast<int>(rwork_query));
        std::vector<Real> rwork(lrwork);
        ScalapackConnector::pheev_f(jobz, uplo, n, A, ia, ja, ad_A.desc, W, Z, iz, jz, ad_Z.desc,
                                    work.data(), lwork, rwork.data(), lrwork, info);
    }
    else
        ScalapackConnector::psyev_f(jobz, uplo, n, A, ia, ja, ad_A.desc, W, Z, iz, jz, ad_Z.desc,
                                    work.data(), lwork, info);
}

template <typename T>
inline void pgemm(
    const char& transa, const char& transb,
    const int & m, const int & n,const int & k,
    const T & alpha,
    const T* A, const int64_t& ia, const int64_t& ja, const ArrayDesc& array_descA,
    const T* B, const int64_t& ib, const int64_t& jb, const ArrayDesc& array_descB,
    const T & beta,
    T* C,const int64_t& ic,const int64_t& jc,const ArrayDesc& array_descC
)
{
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)A)){
        ddla::pgemm(
            array_descA.ddla_handle(),
            transa, transb,
            m, n, k,
            alpha,
            A, array_descA.desc,
            B, array_descB.desc,
            beta,
            C, array_descC.desc
        );
    }
    else
    #endif
    ScalapackConnector::pgemm_f(
        transa, transb,
        m, n, k,
        alpha,
        A, ia, ja, array_descA.desc,
        B, ib, jb, array_descB.desc,
        beta,
        C, ic, jc, array_descC.desc
    );

}

template <typename T1, typename T2>
inline void scal(
    const int& N,
    const T1& alpha,
    T2* X,
    const int& incX,
    const ArrayDesc &array_desc
){
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)X)){
        ddla::scal(array_desc.ddla_handle(), N, T2(alpha), X, incX);
    }else
    #endif
    {
        LapackConnector::scal(N, alpha, X, incX);
    }
}


template <typename T1, typename T2>
inline void pdam(const T1& num, T2* A, const ArrayDesc& array_desc)
{
    if(array_desc.m() != array_desc.n()){
        throw std::runtime_error("In LaConnector::pdam, only square matrix is supported!");
    }
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)A)){
        ddla::pdam(array_desc.ddla_handle(), num, A, array_desc.desc);
    }else
    #endif
    {
        #pragma omp parallel for
        for (int i = 0; i != array_desc.m(); i++)
        {
            const int ilo = array_desc.indx_g2l_r(i);
            if (ilo < 0) continue;
            const int jlo = array_desc.indx_g2l_c(i);
            if (jlo < 0) continue;
            A[ilo + array_desc.lld() * jlo] += num;
        }
    }
}

template <typename T>
inline void axpy(
    const int& N,
    const T& alpha,
    const T* X, const int& incX,
    T* Y, const int& incY,
    const BlacsCtxtHandler &blacs_h
){
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)X)){
        ddla::axpy(
            blacs_h.ddla_handle,
            N,
            alpha,
            X, incX,
            Y, incY
        );
    }else
    #endif
    {
        LapackConnector::axpy(
            N,
            alpha,
            X, incX,
            Y, incY
        );
    }
}

template <typename T>
inline void pgetrf_bpiv(
    const int& m, const int& n,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    int* ipiv, // host or device
    int& info // host
)
{
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(ia!=1 || ja!=1){
        throw std::runtime_error("In LaConnector::pgetrf_bpiv, only support ia=ja=1 for device implementation!");
    }
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        ddla::pgetrf_bpiv(array_descA.ddla_handle(), m, n, d_A, array_descA.desc, ipiv, info);
    }else
    #endif
    {
        ScalapackConnector::pgetrf_f(m, n, d_A, ia, ja, array_descA.desc, ipiv, info);
    }
}

template <typename T>
inline void pgesv(
    const int& n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const ArrayDesc& array_descB,
    int& info
)
{
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        if(ia!=1 || ja!=1 || ib!=1 || jb!=1){
            throw std::runtime_error("In LaConnector::pgesv, only support ia=ja=ib=jb=1 for device implementation!");
        }
        ddla::pgesv(
            array_descA.ddla_handle(), 'L', 'N',
            n, nrhs,
            d_A, array_descA.desc,
            d_B, array_descB.desc
        );
    }else
    #endif
    {
        std::vector<int> ipiv(array_descA.m_loc() + array_descA.mb());
        ScalapackConnector::pgesv_f(
            n, nrhs,
            d_A, ia, ja, array_descA.desc,
            ipiv.data(),
            d_B, ib, jb, array_descB.desc,
            info
        );
        if (info != 0){
            printf("Error in ScalapackConnector::pgesv_f, info = %d\n", info);
            throw std::runtime_error("info !=0\n");
        }
    }
}

template <typename T>
inline void pposv(
    const char& side, const char& uplo, const char& trans,
    const int & n, const int& nrhs,
    T* d_A, const int& ia, const int& ja, const ArrayDesc& array_descA,
    T* d_B, const int& ib, const int& jb, const ArrayDesc& array_descB,
    int& info, // host
    bool is_head = false, int location = -1
)
{
    assert(side == 'L');
    #if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if(DeviceConnector::check_device_ptr((void*)d_A)){
        if(ia!=1 || ja!=1 || ib!=1 || jb!=1){
            throw std::runtime_error("In LaConnector::pposv, only support ia=ja=ib=jb=1 for device implementation!");
        }
        ddla::pposv(
            array_descA.ddla_handle(),
            side, uplo, trans,
            n, nrhs,
            d_A, 1, 1, array_descA.desc,
            d_B, 1, 1, array_descB.desc,
            info,
            is_head, location
        );
    }else
    #endif
    {
        assert(trans == 'N' && side == 'L');
        ScalapackConnector::pposv_f(
            uplo, n, nrhs,
            d_A, ia, ja, array_descA.desc,
            d_B, ib, jb, array_descB.desc,
            info
        );
        if (info != 0){
            printf("the matrix is not positive definite info:%d\n", info);
            throw std::runtime_error("info !=0\n");
        }
    }
}

template <typename T>
inline matrix_m<std::complex<T>> power_hemat_la(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, 
    matrix_m<std::complex<T>> &Z_local, const ArrayDesc &ad_Z,
    size_t &n_filtered, T *W, T power, const T &threshold = -1.e5,
    bool use_gpu_replace_scalapack = false, bool use_elpa_sqrt_coulomb = false, std::complex<T>* d_A = nullptr, 
    std::complex<T>* d_Z = nullptr, std::complex<T>* d_power = nullptr)
{
    #if defined(LIBRPA_USE_ELPA)
    if(use_elpa_sqrt_coulomb){
        return ElpaConnector::power_hemat_elpa(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold,
            use_gpu_replace_scalapack, d_A, d_Z, d_power);
    }else
    #endif
    {
        return power_hemat_blacs(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold);
    }
}

template <typename T>
inline matrix_m<std::complex<T>> power_hemat_la_real(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, 
    matrix_m<std::complex<T>> &Z_local, const ArrayDesc &ad_Z,
    size_t &n_filtered, T *W, T power, const T &threshold = -1.e5,
    bool use_gpu_replace_scalapack = false, bool use_elpa_sqrt_coulomb = false,
    T* d_A = nullptr, T* d_Z = nullptr, T* d_power = nullptr)
{
    #if defined(LIBRPA_USE_ELPA)
    if(use_elpa_sqrt_coulomb){
        return ElpaConnector::power_hemat_elpa_real(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold,
            use_gpu_replace_scalapack, d_A, d_Z, d_power);
    }else
    #endif
    {
        return power_hemat_blacs_real(
            A_local, ad_A, Z_local, ad_Z,
            n_filtered, W, power, threshold);
    }
}

}  // namespace LaConnector

}  // namespace librpa_int

#endif // LA_CONNECTOR_H
