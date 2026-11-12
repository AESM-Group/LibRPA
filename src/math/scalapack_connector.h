#ifndef SCALAPACK_CONNECTOR_H
#define SCALAPACK_CONNECTOR_H

#include <complex>
#include "../interface/blacs_scalapack.h"
#include "lapack_connector.h"

namespace librpa_int {

class ScalapackConnector
{
public:
    // indexing functions adapted from ScaLAPACK TOOLS

    inline static int indxg2p(const int &indxglob, const int &nb, const int &iproc,
                              const int &isrcproc, const int &nprocs)
    {
        return (isrcproc + indxglob / nb) % nprocs;
    }

    inline static int indxg2p(const int &indxglob, const int &nb, const int &isrcproc, const int &nprocs)
    {
        return (isrcproc + indxglob / nb) % nprocs;
    }

    inline static int indxg2l(const int &indxglob, const int &nb, const int &iproc,
                              const int &isrcproc, const int &nprocs)
    {
        return nb * (indxglob / (nb * nprocs)) + indxglob % nb;
    }

    inline static int indxg2l(const int &indxglob, const int &nb, const int &isrcproc, const int &nprocs)
    {
        return nb * (indxglob / (nb * nprocs)) + indxglob % nb;
    }

    inline static int indxl2g(const int &indxloc, const int &nb, const int &iproc,
                              const int &isrcproc, const int &nprocs)
    {
        return nprocs * nb * (indxloc / nb) + indxloc % nb +
               ((nprocs + iproc - isrcproc) % nprocs) * nb;
    }

    static void transpose_desc(int desc_T[9], const int desc[9])
    {
        desc_T[0] = desc[0];
        desc_T[1] = desc[1];
        desc_T[2] = desc[3];
        desc_T[3] = desc[2];
        desc_T[4] = desc[5];
        desc_T[5] = desc[4];
        desc_T[6] = desc[6];
        desc_T[7] = desc[7];
        desc_T[8] = desc[8];
    }

    static void blacs_gridinit(int &ictxt, const char order, const int nprow, const int npcol)
    {
        blacs_gridinit_(&ictxt, &order, &nprow, &npcol);
    }

    static void blacs_gridinfo(const int &ictxt, int &nprow, int &npcol, int &myprow, int &mypcol)
    {
        blacs_gridinfo_(&ictxt, &nprow, &npcol, &myprow, &mypcol);
    }

    static int numroc(const int n, const int nb, const int iproc, const int srcproc,
                      const int nprocs)
    {
        return numroc_(&n, &nb, &iproc, &srcproc, &nprocs);
    }

    static void descinit(int *desc, const int m, const int n, const int mb, const int nb,
                         const int irsrc, const int icsrc, const int ictxt, const int lld,
                         int &info)
    {
        descinit_(desc, &m, &n, &mb, &nb, &irsrc, &icsrc, &ictxt, &lld, &info);
        //		descinit_(desc, &n, &m, &nb, &mb, &irsrc, &icsrc, &ictxt, &lld, &info);
    }

    // C = a * A.? * B.? + b * C
    static void pgemm(const char transa, const char transb, const int M, const int N, const int K,
                      const double alpha, const double *A, const int IA, const int JA,
                      const int *DESCA, const double *B, const int IB, const int JB,
                      const int *DESCB, const double beta, double *C, const int IC, const int JC,
                      const int *DESCC)
    {
        int DESCA_T[9], DESCB_T[9], DESCC_T[9];
        transpose_desc(DESCA_T, DESCA);
        transpose_desc(DESCB_T, DESCB);
        transpose_desc(DESCC_T, DESCC);
        pdgemm_(&transb, &transa, &N, &M, &K, &alpha, B, &JB, &IB, DESCB_T, A, &JA, &IA, DESCA_T,
                &beta, C, &JC, &IC, DESCC_T);
        // pdgemm_(
        // 	&transa, &transb,
        // 	&M, &N, &K,
        // 	&alpha,
        // 	A, &JA, &IA, DESCA,
        // 	B, &JB, &IB, DESCB,
        // 	&beta,
        //	C, &JC, &IC, DESCC);
    }

    static inline void pdgetrf(int m, int n, matrix &a, int ia, int ja, int *desca, int *ipiv,
                               int *info)
    {
        double *aux = LapackConnector::transpose_matrix(a, n, m);
        pdgetrf_(&m, &n, aux, &ia, &ja, desca, ipiv, info);
        LapackConnector::transpose_matrix(aux, a, n, m);
        delete[] aux;
        return;
    }

    static inline void pscal_f(const int &N, const float &alpha, float *X, const int &IX,
                               const int &JX, const int *DESCX, const int &INCX)
    {
        psscal_(&N, &alpha, X, &IX, &JX, DESCX, &INCX);
    }

    static inline void pscal_f(const int &N, const double &alpha, double *X, const int &IX,
                               const int &JX, const int *DESCX, const int &INCX)
    {
        pdscal_(&N, &alpha, X, &IX, &JX, DESCX, &INCX);
    }

    static inline void pscal_f(const int &N, const std::complex<float> &alpha,
                               std::complex<float> *X, const int &IX, const int &JX,
                               const int *DESCX, const int &INCX)
    {
        pcscal_(&N, &alpha, X, &IX, &JX, DESCX, &INCX);
    }

    static inline void pscal_f(const int &N, const std::complex<double> &alpha,
                               std::complex<double> *X, const int &IX, const int &JX,
                               const int *DESCX, int &INCX)
    {
        pzscal_(&N, &alpha, X, &IX, &JX, DESCX, &INCX);
    }

    static inline void pscal_f(const int &N, const float &alpha, std::complex<float> *X,
                               const int &IX, const int &JX, const int *DESCX, const int &INCX)
    {
        pcsscal_(&N, &alpha, X, &IX, &JX, DESCX, &INCX);
    }

    static inline void pscal_f(const int &N, const double &alpha, std::complex<double> *X,
                               const int &IX, const int &JX, const int *DESCX, const int &INCX)
    {
        pzdscal_(&N, &alpha, X, &IX, &JX, DESCX, &INCX);
    }

    static inline float pdot_f(const int &N, const float *X, const int &IX, const int &JX,
                               const int *DESCX, const int &INCX, const float *Y, const int &IY,
                               const int &JY, const int *DESCY, const int &INCY)
    {
        float result = 0.0;
        psdot_(&N, &result, X, &IX, &JX, DESCX, &INCX, Y, &IY, &JY, DESCY, &INCY);
        return result;
    }

    static inline double pdot_f(const int &N, const double *X, const int &IX, const int &JX,
                                const int *DESCX, const int &INCX, const double *Y, const int &IY,
                                const int &JY, const int *DESCY, const int &INCY)
    {
        double result = 0.0;
        pddot_(&N, &result, X, &IX, &JX, DESCX, &INCX, Y, &IY, &JY, DESCY, &INCY);
        return result;
    }

    static inline std::complex<float> pdot_f(const int &N, const std::complex<float> *X,
                                             const int &IX, const int &JX, const int *DESCX,
                                             const int &INCX, const std::complex<float> *Y,
                                             const int &IY, const int &JY, const int *DESCY,
                                             const int &INCY)
    {
        std::complex<float> result = 0.0;
        pcdotu_(&N, &result, X, &IX, &JX, DESCX, &INCX, Y, &IY, &JY, DESCY, &INCY);
        return result;
    }

    static inline std::complex<float> pdotc_f(const int &N, const std::complex<float> *X,
                                              const int &IX, const int &JX, const int *DESCX,
                                              const int &INCX, const std::complex<float> *Y,
                                              const int &IY, const int &JY, const int *DESCY,
                                              const int &INCY)
    {
        std::complex<float> result = 0.0;
        pcdotc_(&N, &result, X, &IX, &JX, DESCX, &INCX, Y, &IY, &JY, DESCY, &INCY);
        return result;
    }

    static inline std::complex<double> pdot_f(const int &N, const std::complex<double> *X,
                                              const int &IX, const int &JX, const int *DESCX,
                                              const int &INCX, const std::complex<double> *Y,
                                              const int &IY, const int &JY, const int *DESCY,
                                              const int &INCY)
    {
        std::complex<double> result = 0.0;
        pzdotu_(&N, &result, X, &IX, &JX, DESCX, &INCX, Y, &IY, &JY, DESCY, &INCY);
        return result;
    }

    static inline std::complex<double> pdotc_f(const int &N, const std::complex<double> *X,
                                               const int &IX, const int &JX, const int *DESCX,
                                               const int &INCX, const std::complex<double> *Y,
                                               const int &IY, const int &JY, const int *DESCY,
                                               const int &INCY)
    {
        std::complex<double> result = 0.0;
        pzdotc_(&N, &result, X, &IX, &JX, DESCX, &INCX, Y, &IY, &JY, DESCY, &INCY);
        return result;
    }

    static inline void pgemv_f(const char &transa, const int &M, const int &N, const float &alpha,
                               const float *A, const int &IA, const int &JA, const int *DESCA,
                               const float *X, const int &IX, const int &JX, const int *DESCX,
                               const int &INCX, const float &beta, float *Y, const int &IY,
                               const int &JY, const int *DESCY, const int &INCY)
    {
        psgemv_(&transa, &M, &N, &alpha, A, &IA, &JA, DESCA, X, &IX, &JX, DESCX, &INCX, &beta, Y,
                &IY, &JY, DESCY, &INCY);
    }

    static inline void pgemv_f(const char &transa, const int &M, const int &N, const double &alpha,
                               const double *A, const int &IA, const int &JA, const int *DESCA,
                               const double *X, const int &IX, const int &JX, const int *DESCX,
                               const int &INCX, const double &beta, double *Y, const int &IY,
                               const int &JY, const int *DESCY, const int &INCY)
    {
        pdgemv_(&transa, &M, &N, &alpha, A, &IA, &JA, DESCA, X, &IX, &JX, DESCX, &INCX, &beta, Y,
                &IY, &JY, DESCY, &INCY);
    }

    static inline void pgemv_f(const char &transa, const int &M, const int &N,
                               const std::complex<float> &alpha, const std::complex<float> *A,
                               const int &IA, const int &JA, const int *DESCA,
                               const std::complex<float> *X, const int &IX, const int &JX,
                               const int *DESCX, const int &INCX, const std::complex<float> &beta,
                               std::complex<float> *Y, const int &IY, const int &JY,
                               const int *DESCY, const int &INCY)
    {
        pcgemv_(&transa, &M, &N, &alpha, A, &IA, &JA, DESCA, X, &IX, &JX, DESCX, &INCX, &beta, Y,
                &IY, &JY, DESCY, &INCY);
    }

    static inline void pgemv_f(const char &transa, const int &M, const int &N,
                               const std::complex<double> &alpha, const std::complex<double> *A,
                               const int &IA, const int &JA, const int *DESCA,
                               const std::complex<double> *X, const int &IX, const int &JX,
                               const int *DESCX, const int &INCX, const std::complex<double> &beta,
                               std::complex<double> *Y, const int &IY, const int &JY,
                               const int *DESCY, const int &INCY)
    {
        pzgemv_(&transa, &M, &N, &alpha, A, &IA, &JA, DESCA, X, &IX, &JX, DESCX, &INCX, &beta, Y,
                &IY, &JY, DESCY, &INCY);
    }

    static inline void pgemm_f(const char &transa, const char &transb, const int &M, const int &N,
                               const int &K, const float &alpha, const float *A, const int &IA,
                               const int &JA, const int *DESCA, const float *B, const int &IB,
                               const int &JB, const int *DESCB, const float &beta, float *C,
                               const int &IC, const int &JC, const int *DESCC)
    {
        psgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, &beta,
                C, &IC, &JC, DESCC);
    }

    static inline void pgemm_f(const char &transa, const char &transb, const int &M, const int &N,
                               const int &K, const double &alpha, const double *A, const int &IA,
                               const int &JA, const int *DESCA, const double *B, const int &IB,
                               const int &JB, const int *DESCB, const double &beta, double *C,
                               const int &IC, const int &JC, const int *DESCC)
    {
        pdgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, &beta,
                C, &IC, &JC, DESCC);
    }

    static inline void pgemm_f(const char &transa, const char &transb, const int &M, const int &N,
                               const int &K, const std::complex<float> &alpha,
                               const std::complex<float> *A, const int &IA, const int &JA,
                               const int *DESCA, const std::complex<float> *B, const int &IB,
                               const int &JB, const int *DESCB, const std::complex<float> &beta,
                               std::complex<float> *C, const int &IC, const int &JC,
                               const int *DESCC)
    {
        pcgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, &beta,
                C, &IC, &JC, DESCC);
    }

    static inline void pgemm_f(const char &transa, const char &transb, const int &M, const int &N,
                               const int &K, const std::complex<double> &alpha,
                               const std::complex<double> *A, const int &IA, const int &JA,
                               const int *DESCA, const std::complex<double> *B, const int &IB,
                               const int &JB, const int *DESCB, const std::complex<double> &beta,
                               std::complex<double> *C, const int &IC, const int &JC,
                               const int *DESCC)
    {
        pzgemm_(&transa, &transb, &M, &N, &K, &alpha, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, &beta,
                C, &IC, &JC, DESCC);
    }

    // Matrix redistribution for Fortran (column-major) matrices.
    static inline void pgemr2d_f(const int m, const int n, const float *a, const int ia,
                                 const int ja, const int *desca, float *b, const int ib,
                                 const int jb, const int *descb, const int ictxt)
    {
        psgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline void pgemr2d_f(const int m, const int n, const double *a, const int ia,
                                 const int ja, const int *desca, double *b, const int ib,
                                 const int jb, const int *descb, const int ictxt)
    {
        pdgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline void pgemr2d_f(const int m, const int n, const std::complex<float> *a,
                                 const int ia, const int ja, const int *desca,
                                 std::complex<float> *b, const int ib, const int jb,
                                 const int *descb, const int ictxt)
    {
        pcgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline void pgemr2d_f(const int m, const int n, const std::complex<double> *a,
                                 const int ia, const int ja, const int *desca,
                                 std::complex<double> *b, const int ib, const int jb,
                                 const int *descb, const int ictxt)
    {
        pzgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline void ptranu_f(const int m, const int n,
                                const std::complex<double> alpha,
                                const std::complex<double> *a,
                                const int ia, const int ja, const int *desca,
                                const std::complex<double> beta,
                                std::complex<double> *c,
                                const int ic, const int jc, const int *descc)
    {
        pztranu_(&m, &n, &alpha, a, &ia, &ja, desca,
                 &beta, c, &ic, &jc, descc);
    }

    static inline void ptranc_f(const int m, const int n,
                                const std::complex<double> alpha,
                                const std::complex<double> *a,
                                const int ia, const int ja, const int *desca,
                                const std::complex<double> beta,
                                std::complex<double> *c,
                                const int ic, const int jc, const int *descc)
    {
        pztranc_(&m, &n, &alpha, a, &ia, &ja, desca,
                 &beta, c, &ic, &jc, descc);
    }

    // Matrix redistribution for C (row-major) matrices.
    // For now it is the same as Fortran (column-major) case.
    // The user should take care of local leading dimension in desca and descb.
    // NOT WORKING, DO NOT USE for now. Use Fortran (column-major) matrices to communicate
    static inline
    void pgemr2d(const int m, const int n,
                 const float *a, const int ia, const int ja, const int *desca,
                 float *b, const int ib, const int jb, const int *descb,
                 const int ictxt)
    {
        psgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline
    void pgemr2d(const int m, const int n,
                 const double *a, const int ia, const int ja, const int *desca,
                 double *b, const int ib, const int jb, const int *descb,
                 const int ictxt)
    {
        pdgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline
    void pgemr2d(const int m, const int n,
                 const std::complex<float> *a, const int ia, const int ja, const int *desca,
                 std::complex<float> *b, const int ib, const int jb, const int *descb,
                 const int ictxt)
    {
        pcgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline
    void pgemr2d(const int m, const int n,
                 const std::complex<double> *a, const int ia, const int ja, const int *desca,
                 std::complex<double> *b, const int ib, const int jb, const int *descb,
                 const int ictxt)
    {
        // int desca_t[9], descb_t[9];
        // transpose_desc(desca_t, desca);
        // transpose_desc(descb_t, descb);
        // pzgemr2d_(&n, &m, a, &ja, &ia, desca_t, b, &jb, &ib, descb_t, &ictxt);
        pzgemr2d_(&m, &n, a, &ia, &ja, desca, b, &ib, &jb, descb, &ictxt);
    }

    static inline void psyev_f(const char &jobz, const char &uplo, const int &n, float *A,
                               const int &ia, const int &ja, const int *desca, float *W, float *Z,
                               const int &iz, const int &jz, const int *descz, float *work,
                               const int &lwork, int &info)
    {
        pssyev_(&jobz, &uplo, &n, A, &ia, &ja, desca, W, Z, &iz, &jz, descz, work, &lwork, &info);
    }
    static inline void psyev_f(const char &jobz, const char &uplo, const int &n, double *A,
                               const int &ia, const int &ja, const int *desca, double *W, double *Z,
                               const int &iz, const int &jz, const int *descz, double *work,
                               const int &lwork, int &info)
    {
        pdsyev_(&jobz, &uplo, &n, A, &ia, &ja, desca, W, Z, &iz, &jz, descz, work, &lwork, &info);
    }
    static inline void pheev_f(const char &jobz, const char &uplo, const int &n,
                               std::complex<float> *A, const int &ia, const int &ja,
                               const int *desca, float *W, std::complex<float> *Z, const int &iz,
                               const int &jz, const int *descz, std::complex<float> *work,
                               const int &lwork, float *rwork, const int &lrwork, int &info)
    {
        pcheev_(&jobz, &uplo, &n, A, &ia, &ja, desca, W, Z, &iz, &jz, descz, work, &lwork, rwork,
                &lrwork, &info);
    }

    static inline void pheev_f(const char &jobz, const char &uplo, const int &n,
                               std::complex<double> *A, const int &ia, const int &ja,
                               const int *desca, double *W, std::complex<double> *Z, const int &iz,
                               const int &jz, const int *descz, std::complex<double> *work,
                               const int &lwork, double *rwork, const int &lrwork, int &info)
    // const char, const char, const int, std::complex<double> *, int, int, const int [9], double *,
    // std::complex<double> *, int, int, const int [9], double *, int, std::complex<double> *, int,
    // int
    {
        pzheev_(&jobz, &uplo, &n, A, &ia, &ja, desca, W, Z, &iz, &jz, descz, work, &lwork, rwork,
                &lrwork, &info);
    }

    static inline void pgetrf_f(const int &m, const int &n, float *a, const int &ia, const int &ja,
                                const int *desca, int *ipiv, int &info)
    {
        psgetrf_(&m, &n, a, &ia, &ja, desca, ipiv, &info);
    }

    static inline void pgetrf_f(const int &m, const int &n, double *a, const int &ia, const int &ja,
                                const int *desca, int *ipiv, int &info)
    {
        pdgetrf_(&m, &n, a, &ia, &ja, desca, ipiv, &info);
    }

    static inline void pgetrf_f(const int &m, const int &n, std::complex<float> *a, const int &ia,
                                const int &ja, const int *desca, int *ipiv, int &info)
    {
        pcgetrf_(&m, &n, a, &ia, &ja, desca, ipiv, &info);
    }

    static inline void pgetrf_f(const int &m, const int &n, std::complex<double> *a, const int &ia,
                                const int &ja, const int *desca, int *ipiv, int &info)
    {
        pzgetrf_(&m, &n, a, &ia, &ja, desca, ipiv, &info);
    }

    static inline void pgesv_f(const int &n, const int &nrhs, std::complex<double> *a,
                               const int &ia, const int &ja, const int *desca, int *ipiv,
                               std::complex<double> *b, const int &ib, const int &jb,
                               const int *descb, int &info)
    {
        pzgesv_(&n, &nrhs, a, &ia, &ja, desca, ipiv, b, &ib, &jb, descb, &info);
    }

    static inline void pposv_f(const char &uplo, const int &n, const int &nrhs,
                               std::complex<double> *a, const int &ia, const int &ja,
                               const int *descA, std::complex<double> *b, const int &ib,
                               const int &jb, const int *descB, int &info)
    {
        pzposv_(&uplo, &n, &nrhs, a, &ia, &ja, descA, b, &ib, &jb, descB, &info);
    }

    static inline void pgetri_f(const int &n, float *a, const int &ia, const int &ja,
                                const int *desca, int *ipiv, float *work, const int &lwork,
                                int *iwork, const int &liwork, int &info)
    {
        psgetri_(&n, a, &ia, &ja, desca, ipiv, work, &lwork, iwork, &liwork, &info);
    }

    static inline void pgetri_f(const int &n, double *a, const int &ia, const int &ja,
                                const int *desca, int *ipiv, double *work, const int &lwork,
                                int *iwork, const int &liwork, int &info)
    {
        pdgetri_(&n, a, &ia, &ja, desca, ipiv, work, &lwork, iwork, &liwork, &info);
    }

    static inline void pgetri_f(const int &n, std::complex<float> *a, const int &ia, const int &ja,
                                const int *desca, int *ipiv, std::complex<float> *work,
                                const int &lwork, int *iwork, const int &liwork, int &info)
    {
        pcgetri_(&n, a, &ia, &ja, desca, ipiv, work, &lwork, iwork, &liwork, &info);
    }

    static inline void pgetri_f(const int &n, std::complex<double> *a, const int &ia, const int &ja,
                                const int *desca, int *ipiv, std::complex<double> *work,
                                const int &lwork, int *iwork, const int &liwork, int &info)
    {
        pzgetri_(&n, a, &ia, &ja, desca, ipiv, work, &lwork, iwork, &liwork, &info);
    }

    static inline void pgeadd_f(const char &transa, const int &m, const int &n,
                                const std::complex<double> &alpha, const std::complex<double> *a,
                                const int &ia, const int &ja, const int *desca,
                                const std::complex<double> &beta, std::complex<double> *c,
                                const int &ic, const int &jc, const int *descc)
    {
        pzgeadd_(&transa, &m, &n, &alpha, a, &ia, &ja, desca, &beta, c, &ic, &jc, descc);
    }
};

}
#endif
