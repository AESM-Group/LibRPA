#ifndef LA_CONNECTOR_H
#define LA_CONNECTOR_H

#include <algorithm>
#include <string>
#include <vector>

#ifdef LIBRPA_USE_DDLA
#include <ddla/ddla.h>
#endif
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include "ddla/device_connector.h"
#endif
#include "../math/matrix_m.h"
#include "scalapack/scalapack_connector.h"
#include "../mpi/base_blacs.h"
#ifdef LIBRPA_USE_ELPA
#include <elpa/elpa.h>
#endif

namespace librpa_int
{

namespace LaConnector
{

// Matrix ownership belongs to the descriptor's grid, not MPI_COMM_WORLD.
// A one-rank k-point grid can exist inside a multi-rank calculation.
inline bool is_single_rank_grid(const ArrayDesc& desc)
{
    return desc.is_initialized() && desc.nprows() == 1 && desc.npcols() == 1 &&
           desc.myprow() == 0 && desc.mypcol() == 0;
}

inline bool is_same_single_rank_grid(const ArrayDesc& a, const ArrayDesc& b)
{
    return is_single_rank_grid(a) && is_single_rank_grid(b) && a.ictxt() == b.ictxt();
}

template <typename T>
inline T* local_submatrix(T* a, int64_t i, int64_t j, const ArrayDesc& desc)
{
    return a + (i - 1) + int64_t(desc.lld()) * (j - 1);
}

inline bool submatrix_fits(const ArrayDesc& desc, int64_t i, int64_t j,
                          int64_t rows, int64_t cols)
{
    return i >= 1 && j >= 1 && rows >= 0 && cols >= 0 &&
           i - 1 <= desc.m() && rows <= desc.m() - (i - 1) &&
           j - 1 <= desc.n() && cols <= desc.n() - (j - 1);
}

inline bool use_ddla_cpu(const ArrayDesc& desc)
{
#ifdef LIBRPA_USE_DDLA
    if (desc.use_ddla_linear())
    {
        if (desc.ddla_handle() == nullptr)
            throw std::invalid_argument("DDLA linear operations require a caller-provided handle");
        return ddla::ddlaGetBackend(desc.ddla_handle()) == ddla::DdlaBackend::CPU;
    }
#endif
    return false;
}

inline bool is_device_array(const void* ptr, const ArrayDesc& desc)
{
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    return desc.ddla_handle() != nullptr &&
           ddla::ddlaGetBackend(desc.ddla_handle()) == ddla::DdlaBackend::GPU &&
           DeviceConnector::check_device_ptr(const_cast<void*>(ptr));
#else
    return false;
#endif
}

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
        // Matrix powers need all eigenvectors; the handle selects CPU or GPU.
        ddla::pheevd(handle, jobz, uplo, n, n, A, ad_A.desc, W, Z, ad_Z.desc, info);
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

    if (is_single_rank_grid(ad_A) && submatrix_fits(ad_A, ia, ja, n, n) &&
        ((jobz == 'N' || jobz == 'n') ||
         (is_same_single_rank_grid(ad_A, ad_Z) && submatrix_fits(ad_Z, iz, jz, n, n))))
    {
        // LAPACK stores eigenvectors in A; the distributed interface returns Z.
        auto* a = local_submatrix(A, ia, ja, ad_A);
        T work_query{};
        if constexpr (is_complex_t<T>::value)
        {
            std::vector<Real> rwork(std::max(1, 3 * n - 2));
            LapackConnector::heev_f(jobz, uplo, n, a, ad_A.lld(), W,
                                    &work_query, -1, rwork.data(), info);
            if (info != 0) return;
            std::vector<T> work(std::max(1, static_cast<int>(get_real(work_query))));
            LapackConnector::heev_f(jobz, uplo, n, a, ad_A.lld(), W,
                                    work.data(), work.size(), rwork.data(), info);
        }
        else
        {
            LapackConnector::syev_f(jobz, uplo, n, a, ad_A.lld(), W, &work_query, -1, info);
            if (info != 0) return;
            std::vector<T> work(std::max(1, static_cast<int>(work_query)));
            LapackConnector::syev_f(jobz, uplo, n, a, ad_A.lld(), W,
                                    work.data(), work.size(), info);
        }
        if (info == 0 && (jobz == 'V' || jobz == 'v'))
        {
            auto* z = local_submatrix(Z, iz, jz, ad_Z);
            if (a != z || ad_A.lld() != ad_Z.lld())
                for (int j = 0; j < n; ++j)
                    std::copy_n(a + int64_t(j) * ad_A.lld(), n, z + int64_t(j) * ad_Z.lld());
        }
        return;
    }

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
inline void pgemm(const char& transa, const char& transb, const int& m, const int& n, const int& k,
                  const T& alpha, const T* A, const int64_t& ia, const int64_t& ja,
                  const ArrayDesc& array_descA, const T* B, const int64_t& ib, const int64_t& jb,
                  const ArrayDesc& array_descB, const T& beta, T* C, const int64_t& ic,
                  const int64_t& jc, const ArrayDesc& array_descC)
{
#ifdef LIBRPA_USE_DDLA
    // DDLA has no submatrix offsets. Host submatrices retain ScaLAPACK/LAPACK.
    const bool device = is_device_array(A, array_descA);
    if (device || (ia == 1 && ja == 1 && ib == 1 && jb == 1 && ic == 1 && jc == 1 &&
                   use_ddla_cpu(array_descA)))
    {
        if (ia != 1 || ja != 1 || ib != 1 || jb != 1 || ic != 1 || jc != 1)
            throw std::runtime_error("DDLA pgemm requires unit submatrix offsets");
        ddla::pgemm(array_descA.ddla_handle(), transa, transb, m, n, k, alpha, A,
                    array_descA.desc, B, array_descB.desc, beta, C,
                    array_descC.desc);
        return;
    }
#endif
    const bool normal_a = transa == 'N' || transa == 'n';
    const bool normal_b = transb == 'N' || transb == 'n';
    if (is_same_single_rank_grid(array_descA, array_descB) &&
        is_same_single_rank_grid(array_descA, array_descC) &&
        submatrix_fits(array_descA, ia, ja, normal_a ? m : k, normal_a ? k : m) &&
        submatrix_fits(array_descB, ib, jb, normal_b ? k : n, normal_b ? n : k) &&
        submatrix_fits(array_descC, ic, jc, m, n))
    {
        if (m == 0 || n == 0) return;
        LapackConnector::gemm_f(transa, transb, m, n, k, alpha,
            local_submatrix(A, ia, ja, array_descA), array_descA.lld(),
            local_submatrix(B, ib, jb, array_descB), array_descB.lld(), beta,
            local_submatrix(C, ic, jc, array_descC), array_descC.lld());
        return;
    }
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
    if(is_device_array(X, array_desc)){
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
    if(is_device_array(A, array_desc)){
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
    if(blacs_h.ddla_handle != nullptr && DeviceConnector::check_device_ptr((void*)X)){
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
#ifdef LIBRPA_USE_DDLA
    if (is_device_array(d_A, array_descA) ||
        (ia == 1 && ja == 1 && use_ddla_cpu(array_descA)))
    {
        if(ia!=1 || ja!=1){
            throw std::runtime_error("In LaConnector::pgetrf_bpiv, only support ia=ja=1 for device implementation!");
        }
        ddla::pgetrf_bpiv(array_descA.ddla_handle(), m, n, d_A, array_descA.desc, ipiv, info);
        return;
    }
#endif
    {
        if (is_single_rank_grid(array_descA) &&
            submatrix_fits(array_descA, ia, ja, m, n))
        {
            if (m == 0 || n == 0) { info = 0; return; }
            // ScaLAPACK stores pivots at parent rows with global row numbers.
            auto* pivots = ipiv + (ia - 1);
            LapackConnector::getrf_f(m, n, local_submatrix(d_A, ia, ja, array_descA),
                                     array_descA.lld(), pivots, info);
            if (info >= 0)
                for (int i = 0; i < std::min(m, n); ++i) pivots[i] += ia - 1;
            return;
        }
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
#ifdef LIBRPA_USE_DDLA
    if (is_device_array(d_A, array_descA) ||
        (ia == 1 && ja == 1 && ib == 1 && jb == 1 && use_ddla_cpu(array_descA)))
    {
        if(ia!=1 || ja!=1 || ib!=1 || jb!=1){
            throw std::runtime_error("In LaConnector::pgesv, only support ia=ja=ib=jb=1 for device implementation!");
        }
        ddla::pgesv(
            array_descA.ddla_handle(), 'L', 'N',
            n, nrhs,
            d_A, array_descA.desc,
            d_B, array_descB.desc
        );
        info = 0;
        return;
    }
#endif
    {
        std::vector<int> ipiv(array_descA.m_loc() + array_descA.mb());
        if (is_same_single_rank_grid(array_descA, array_descB) &&
            submatrix_fits(array_descA, ia, ja, n, n) &&
            submatrix_fits(array_descB, ib, jb, n, nrhs))
            LapackConnector::gesv_f(n, nrhs, local_submatrix(d_A, ia, ja, array_descA),
                array_descA.lld(), ipiv.data(),
                local_submatrix(d_B, ib, jb, array_descB), array_descB.lld(), info);
        else
        ScalapackConnector::pgesv_f(
            n, nrhs,
            d_A, ia, ja, array_descA.desc,
            ipiv.data(),
            d_B, ib, jb, array_descB.desc,
            info
        );
        if (info != 0){
            printf("Error in LaConnector::pgesv, info = %d\n", info);
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
    int& info // host
)
{
    assert(side == 'L');
#ifdef LIBRPA_USE_DDLA
    if (is_device_array(d_A, array_descA) ||
        (ia == 1 && ja == 1 && ib == 1 && jb == 1 && use_ddla_cpu(array_descA)))
    {
        if(ia!=1 || ja!=1 || ib!=1 || jb!=1){
            throw std::runtime_error("In LaConnector::pposv, only support ia=ja=ib=jb=1 for device implementation!");
        }
        ddla::pposv(
            array_descA.ddla_handle(),
            side, uplo, trans,
            n, nrhs,
            d_A, 1, 1, array_descA.desc,
            d_B, 1, 1, array_descB.desc,
            info
        );
        return;
    }
#endif
    {
        assert(trans == 'N' && side == 'L');
        if (is_same_single_rank_grid(array_descA, array_descB) &&
            submatrix_fits(array_descA, ia, ja, n, n) &&
            submatrix_fits(array_descB, ib, jb, n, nrhs))
            LapackConnector::posv_f(uplo, n, nrhs, local_submatrix(d_A, ia, ja, array_descA),
                array_descA.lld(),
                local_submatrix(d_B, ib, jb, array_descB), array_descB.lld(), info);
        else
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

// Uses the caller's block-cyclic distribution, with no reblocking. Handles and
// device selection belong to the caller. GPU A/Z descriptors share one DDLA
// handle. GPU buffers are distinct and sized
// for LLD*local_columns: d_A uses ad_A; d_Z and d_scaled use ad_Z.
// For n>1 GPU scratch holds the power in d_A, eigenvectors in d_Z and their
// weighted copy in d_scaled. The scalar shortcut updates host outputs only.
// Host output matrix storage is preserved. W always ends on host.
template <typename T>
matrix_m<std::complex<T>> power_hemat_la(
    matrix_m<std::complex<T>>& A, const ArrayDesc& ad_A,
    matrix_m<std::complex<T>>& Z, const ArrayDesc& ad_Z, size_t& filtered, T* W,
    T power, const T& threshold = -1.e5, bool gpu = false,
    const std::string& method = "scalapack", std::complex<T>* d_A = nullptr,
    std::complex<T>* d_Z = nullptr, std::complex<T>* d_scaled = nullptr);

// Diagonalizes -real(A) and negates eigenvalues, preserving descending order.
template <typename T>
matrix_m<std::complex<T>> power_hemat_la_real(
    matrix_m<std::complex<T>>& A, const ArrayDesc& ad_A,
    matrix_m<std::complex<T>>& Z, const ArrayDesc& ad_Z, size_t& filtered, T* W,
    T power, const T& threshold = -1.e5, bool gpu = false,
    const std::string& method = "scalapack", T* d_A = nullptr,
    T* d_Z = nullptr, T* d_scaled = nullptr);

}  // namespace LaConnector

}  // namespace librpa_int

#endif // LA_CONNECTOR_H
