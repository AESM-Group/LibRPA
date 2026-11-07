#pragma once
#include <complex>

#ifdef __cplusplus
extern "C" {
#endif
///////////////////////////
/* BLAS Fortran bindings */
///////////////////////////
    void sscal_(const int *N, const float *alpha, float *X, const int *incX);
    void dscal_(const int *N, const double *alpha, double *X, const int *incX);
    void cscal_(const int *N, const std::complex<float> *alpha, std::complex<float> *X, const int *incX);
    void zscal_(const int *N, const std::complex<double> *alpha, std::complex<double> *X, const int *incX);

    void saxpy_(const int *N, const float *alpha,
                const float *X, const int *incX, float *Y, const int *incY);
    void daxpy_(const int *N, const double *alpha,
                const double *X, const int *incX, double *Y, const int *incY);
    void caxpy_(const int *N, const std::complex<float> *alpha,
                const std::complex<float> *X, const int *incX, std::complex<float> *Y, const int *incY);
    void zaxpy_(const int *N, const std::complex<double> *alpha,
                const std::complex<double> *X, const int *incX, std::complex<double> *Y, const int *incY);
    
    void scopy_(long const *n, const float *X, int const *incX, float *Y, int const *incY);
    void dcopy_(long const *n, const double *X, int const *incX, double *Y, int const *incY);
    void ccopy_(long const *n, const std::complex<float> *X, int const *incX, std::complex<float> *Y, int const *incY); 
    void zcopy_(long const *n, const std::complex<double> *X, int const *incX, std::complex<double> *Y, int const *incY); 

    float sdot_(const int *N, const float *X, const int *incX, const float *Y,
                const int *incY);
    double ddot_(const int *N, const double *X, const int *incX, const double *Y,
                 const int *incY);
    void cdotc_(std::complex<float> *result, const int *n, const std::complex<float> *zx,
                const int *incx, const std::complex<float> *zy, const int *incy);
    void cdotu_(std::complex<float> *result, const int *n, const std::complex<float> *zx,
                const int *incx, const std::complex<float> *zy, const int *incy);
    void zdotc_(std::complex<double> *result, const int *n, const std::complex<double> *zx,
                const int *incx, const std::complex<double> *zy, const int *incy);
    void zdotu_(std::complex<double> *result, const int *n, const std::complex<double> *zx,
                const int *incx, const std::complex<double> *zy, const int *incy);
    
    void sgemv_(const char *transa, const int *m, const int *n, const float *alpha,
                const float *a, const int *lda, const float *x, const int *incx,
                const float *beta, float *y, const int *incy);
    void dgemv_(const char *transa, const int *m, const int *n, const double *alpha,
                const double *a, const int *lda, const double *x, const int *incx,
                const double *beta, double *y, const int *incy);
    void cgemv_(const char *transa, const int *m, const int *n,
                const std::complex<float> *alpha, const std::complex<float> *a,
                const int *lda, const std::complex<float> *x, const int *incx,
                const std::complex<float> *beta, std::complex<float> *y, const int *incy);
    void zgemv_(const char *transa, const int *m, const int *n,
                const std::complex<double> *alpha, const std::complex<double> *a,
                const int *lda, const std::complex<double> *x, const int *incx,
                const std::complex<double> *beta, std::complex<double> *y, const int *incy);
    
    void sgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const float *alpha, const float *a, const int *lda,
                const float *b, const int *ldb, const float *beta, float *c,
                const int *ldc);
    void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const double *alpha, const double *a, const int *lda,
                const double *b, const int *ldb, const double *beta, double *c,
                const int *ldc);
    void cgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const std::complex<float> *alpha, const std::complex<float> *a,
                const int *lda, const std::complex<float> *b, const int *ldb,
                const std::complex<float> *beta, std::complex<float> *c, const int *ldc);
    void zgemm_(const char *transa, const char *transb, const int *m, const int *n,
                const int *k, const std::complex<double> *alpha,
                const std::complex<double> *a, const int *lda, const std::complex<double> *b,
                const int *ldb, const std::complex<double> *beta, std::complex<double> *c,
                const int *ldc);
    void dzgemm_(const char *transa, const char *transb, const int *m, const int *n,
                 const int *k, const std::complex<double> *alpha, const double *a,
                 const int *lda, const std::complex<double> *b, const int *ldb,
                 const std::complex<double> *beta, std::complex<double> *c, const int *ldc);

    void chemm_(const char *side, const char *uplo, const int *m, const int *n, const std::complex<float> *alpha,
                const std::complex<float> *a, const int *lda,
                const std::complex<float> *b, const int *ldb,
                const std::complex<float> *beta,
                std::complex<float> *c, const int *ldc);
    void zhemm_(const char *side, const char *uplo, const int *m, const int *n, const std::complex<double> *alpha,
                const std::complex<double> *a, const int *lda,
                const std::complex<double> *b, const int *ldb,
                const std::complex<double> *beta,
                std::complex<double> *c, const int *ldc);

/////////////////////
/* LAPACK bindings */
/////////////////////
    int ilaenv_(int *ispec, const char *name, const char *opts,
                const int *n1, const int *n2, const int *n3, const int *n4, int name_len, int opts_len);
    void sgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
                const int *lda, float *wr, float *wi, float *vl, const int *ldvl,
                float *vr, const int *ldvr, float *work, const int *lwork, int *info);
    void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
                const int *lda, double *wr, double *wi, double *vl, const int *ldvl,
                double *vr, const int *ldvr, double *work, const int *lwork, int *info);
    void cgeev_(const char *jobvl, const char *jobvr, const int *n, float *a,
                const int *lda, std::complex<float> *wr, std::complex<float> *wi, std::complex<float> *vl, const int *ldvl,
                std::complex<float> *vr, const int *ldvr, std::complex<float> *work, const int *lwork, int *info);
    void zgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
                const int *lda, std::complex<double> *wr, std::complex<double> *wi, std::complex<double> *vl, const int *ldvl,
                std::complex<double> *vr, const int *ldvr, std::complex<double> *work, const int *lwork, int *info);

    void ssyev_(const char *jobz, const char *uplo, const int *n, float *a,
                const int *lda, float *w, float *work, const int *lwork,
                int *info);
    void dsyev_(const char *jobz, const char *uplo, const int *n, double *a,
                const int *lda, double *w, double *work, const int *lwork,
                int *info);
    void cheev_(const char *jobz, const char *uplo, const int *n,
                std::complex<float> *a, const int *lda, float *w,
                std::complex<float> *work, const int *lwork, float *rwork, int *info);
    void zheev_(const char *jobz, const char *uplo, const int *n,
                std::complex<double> *a, const int *lda, double *w,
                std::complex<double> *work, const int *lwork, double *rwork, int *info);

// computes the solution to a complex system of linear equations A * X = B
    void sgesv_(const int *n, const int *nrhs, float *A, const int *lda,
                const int *ipiv, float *B, const int* ldb, int* info);
    void dgesv_(const int *n, const int *nrhs, double *A, const int *lda,
                const int *ipiv, double *B, const int* ldb, int* info);
    void cgesv_(const int *n, const int *nrhs, std::complex<float> *A, const int *lda,
                const int *ipiv, std::complex<float> *B, const int* ldb, int* info);
    void zgesv_(const int *n, const int *nrhs, std::complex<double> *A, const int *lda,
                const int *ipiv, std::complex<double> *B, const int* ldb, int* info);

// computes all the eigenvalues, and optionally,
// the eigenvectors of a real generalized symmetric-definite eigenproblem
    void ssygv_(const int *itype, const char *jobz, const char *uplo, const int *n,
                float *a, const int *lda, float *b, const int *ldb,
                float *w, float *work, int *lwork, int *info);
    void dsygv_(const int *itype, const char *jobz, const char *uplo, const int *n,
                double *a, const int *lda, double *b, const int *ldb,
                double *w, double *work, int *lwork, int *info);

// computes all the eigenvalues, and optionally,
// the eigenvectors of a complex generalized Hermitian-definite eigenproblem
    void chegv_(const int *itype, const char *jobz, const char *uplo, const int *n,
                std::complex<float> *a, const int *lda, std::complex<float> *b, const int *ldb,
                float *w, std::complex<float> *work, int *lwork, float *rwork, int *info);
    void zhegv_(const int *itype, const char *jobz, const char *uplo, const int *n,
                std::complex<double> *a, const int *lda, std::complex<double> *b, const int *ldb,
                double *w, std::complex<double> *work, int *lwork, double *rwork, int *info);

// computes selected eigenvalues, and optionally, eigenvectors
// of a real generalized symmetric-definite eigenproblem
// A and B are assumed to be symmetric, stored in packed storage
    void sspgvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                 const int *n, float *ap,float *bp, const float *vl,
                 const int *vu, const int *il, const int *iu, const float *abstol,
                 const int *m, float *w,float *z, const int *ldz,
                 float *work, int *iwork, int *ifail, int *info);

// computes selected eigenvalues, and optionally, eigenvectors
// of a complex generalized Hermitian-definite eigenproblem
    void zhegvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                 const int *n, std::complex<double> *a, const int *lda, std::complex<double> *b,
                 const int *ldb, const double *vl, const double *vu, const int *il,
                 const int *iu, const double *abstol, const int *m, double *w,
                 std::complex<double> *z, const int *ldz, std::complex<double> *work, const int *lwork,
                 double *rwork, int *iwork, int *ifail, int *info);

// computes the singular value decomposition (SVD) of a real M-by-N matrix A,
// optionally computing the left and/or right singular vectors.
    void sgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, float *a,
                 const int *lda, float *s, float *u, const int *ldu, float *vt,
                 const int *ldvt, float *work, const int *lwork, int *info);
    void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, double *a,
                 const int *lda, double *s, double *u, const int *ldu, double *vt,
                 const int *ldvt, double *work, const int *lwork, int *info);
    void cgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, std::complex<float> *a,
                 const int *lda, std::complex<float> *s, std::complex<float> *u, const int *ldu, std::complex<float> *vt,
                 const int *ldvt, std::complex<float> *work, const int *lwork, int *info);
    void zgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
                 std::complex<double> *a, const int *lda, double *s, std::complex<double> *u,
                 const int *ldu, std::complex<double> *vt, const int *ldvt,
                 std::complex<double> *work, const int *lwork, double *rwork, int *info);

    // factorization
    void sgetrf_(const int *m, const int *n, float *A, const int *lda, int *ipiv, int *info);
    void dgetrf_(const int *m, const int *n, double *A, const int *lda, int *ipiv, int *info);
    void cgetrf_(const int *m, const int *n, std::complex<float> *A, const int *lda, int *ipiv, int *info);
    void zgetrf_(const int *m, const int *n, std::complex<double> *A, const int *lda, int *ipiv, int *info);
                           
// inversion from results of factorization
    void sgetri_(const int *n, float *A, const int *lda, int *ipiv, float *work, const int *lwork, int *info);
    void dgetri_(const int *n, double *A, const int *lda, int *ipiv, double *work, const int *lwork, int *info);
    void cgetri_(const int *n, std::complex<float> *A, const int *lda, int *ipiv, std::complex<float> *work, const int *lwork, int *info);
    void zgetri_(const int *n, std::complex<double> *A, const int *lda, int *ipiv, std::complex<double> *work, const int *lwork, int *info);

// computes the factorization of a real symmetric matrix A using
// the Bunch-Kaufman diagonal pivoting method
    void ssytrf_(const char *uplo, const int *n, float *a, const int *lda,
                 int *ipiv, float *work, int *lwork ,int *info);
    void dsytrf_(const char *uplo, const int *n, double *a, const int *lda,
                 int *ipiv, double *work, int *lwork ,int *info);

// computes the inverse of a real symmetric indefinite matrix
// A using the factorization A = U*D*U**T or A = L*D*L**T computed by ?SYTRF.
    void ssytri_(const char *uplo, const int *n, float *a, const int *lda,
                 int *ipiv, float *work, int *info);
    void dsytri_(const char *uplo, const int *n, double *a, const int *lda,
                 int *ipiv, double *work, int *info);

// compute op( A )*X = alpha*B, or X*op( A ) = alpha*B
    void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
                double *alpha, double *a, int *lda, double*b, int *ldb);
    void ztrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
                std::complex<double> *alpha, std::complex<double> *a, int *lda, std::complex<double>*b, int *ldb);

// computes all eigenvalues of a symmetric tridiagonal matrix
// using the Pal-Walker-Kahan variant of the QL or QR algorithm.
    void ssterf_(int *n, float *d, float *e, int *info);
    void dsterf_(int *n, double *d, double *e, int *info);

    void dstein_(int *n, double *d, double *e, int *m, double *w,
                 int *block, int *isplit, double *z, int *lda, double *work,
                 int *iwork, int *ifail, int *info);
    void zstein_(int *n, double *d, double *e, int *m, double *w,
                 int *block, int *isplit, std::complex<double> *z, int *lda, double *work,
                 int *iwork, int *ifail, int *info);

// computes Cholesky factorization of a symmetric/Hermitian positive definite matrix A
    void spotrf_(const char *uplo, const int *n, float *A, const int *lda, int *info);
    void dpotrf_(const char *uplo, const int *n, double *A, const int *lda, int *info);
    void cpotrf_(const char *uplo, const int *n, std::complex<float> *A, const int *lda, int *info);
    void zpotrf_(const char *uplo, const int *n, std::complex<double> *A, const int *lda, int *info);

// computes the inverse of a symmetric/Hermitian positive definite matrix A
// using the Cholesky factorization from ?potrf
    void spotri_(const char *uplo, const int *n, float *A, const int *lda, int *info);
    void dpotri_(const char *uplo, const int *n, double *A, const int *lda, int *info);
    void cpotri_(const char *uplo, const int *n, std::complex<float> *A, const int *lda, int *info);
    void zpotri_(const char *uplo, const int *n, std::complex<double> *A, const int *lda, int *info);

// computes the Cholesky factorization of a symmetric/Hermitian
// positive definite matrix A (unblocked algorithm)
    void dpotf2_(char *uplo, int *n, double *a, int *lda, int *info);
    void zpotf2_(char *uplo,int *n,std::complex<double> *a, int *lda, int *info);

// reduces a symmetric definite generalized eigenproblem to standard form,
// using the factorization results obtained from spotrf (unblocked algorithm).
    void dsygs2_(int *itype, char *uplo, int *n, double *a, int *lda, double *b, int *ldb, int *info);

// reduces a Hermitian definite generalized eigenproblem to standard form,
// using the factorization results obtained from cpotrf (unblocked algorithm).
    void zhegs2_(int *itype, char *uplo, int *n, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb, int *info);

// copies all or part of one two-dimensional array to another.
    void slacpy_(char *uplo, int *m, int *n, float *a, int *lda, float *b, int *ldb);
    void dlacpy_(char *uplo, int *m, int *n, float *a, int *lda, float *b, int *ldb);
    void clacpy_(char *uplo, int *m, int *n, std::complex<float> *a, int *lda, std::complex<float> *b, int *ldb);
    void zlacpy_(char *uplo, int *m, int *n, std::complex<double> *a, int *lda, std::complex<double> *b, int *ldb);

// generates an elementary reflector (Householder matrix).
    void slarfg_(int *n, float *alpha, float *x, int *incx, float *tau);
    void dlarfg_(int *n, double *alpha, double *x, int *incx, double *tau);
    void clarfg_(int *n, std::complex<float> *alpha, std::complex<float> *x, int *incx, std::complex<float> *tau);
    void zlarfg_(int *n, std::complex<double> *alpha, std::complex<double> *x, int *incx, std::complex<double> *tau);

// performs the rank 1 operation A := alpha*x*y**T + A,
// where x and y are m- and n-element vector, A is an m by n matrix.
    void sger_(int *m, int *n, float *alpha, float *x, int *incx, float *y, int *incy, float *a, int *lda);
    void dger_(int *m, int *n, double *alpha, double *x, int *incx, double *y, int *incy, double *a, int *lda);

// performs the rank 1 operation A := alpha*x*y**H + A,
// where x and y are m- and n-element vector, A is an m by n matrix.
    void cgerc_(int *m, int *n, std::complex<float> *alpha, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy, std::complex<float> *a, int *lda);
    void zgerc_(const int *m, const int *n, const std::complex<double> *alpha, const std::complex<double> *x, const int *incx, const std::complex<double> *y, const int *incy, std::complex<double> *a, const int *lda);

// performs the rank 1 operation A := alpha*x*y**T + A,
// where x and y are m- and n-element vector, A is an m by n matrix.
    void cgeru_(int *m, int *n, std::complex<float> *alpha, std::complex<float> *x, int *incx, std::complex<float> *y, int *incy, std::complex<float> *a, int *lda);
    void zgeru_(int *m, int *n, std::complex<double> *alpha, std::complex<double> *x, int *incx, std::complex<double> *y, int *incy, std::complex<double> *a, int *lda);

// performs one of the hermitian rank k operations
//    C := alpha*A*A**H + beta*C,
// or
//    C := alpha*A**H*A + beta*C,
// where  alpha and beta  are  real scalars,  C is an  n by n  hermitian
// matrix and  A  is an  n by k  matrix in the  first case and a  k by n
// matrix in the second case.
    void cherk_(const char *uplo, const char *trans, const int *n, const int *k, 
                const float *alpha, const std::complex<float> *A, const int *lda, 
                const float *beta, std::complex<float> *C, const int *ldc);
    void zherk_(const char *uplo, const char *trans, const int *n, const int *k, 
                const double *alpha, const std::complex<double> *A, const int *lda, 
                const double *beta, std::complex<double> *C, const int *ldc);
#ifdef __cplusplus
}
#endif
