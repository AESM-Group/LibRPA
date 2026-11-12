#pragma once

#include <complex>
#include "./mpi.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

////////////////////
/* BLACS bindings */
////////////////////
    void blacs_gridinit_(int *ictxt, const char *layout, const int *nprow, const int *npcol);
    void blacs_gridmap_(int *ictxt, const int *usermap, const int *ldup, const int *nprow0, const int *npcol0);
    void blacs_gridinfo_(const int *ictxt, int *nprow, int *npcol, int *myprow, int *mypcol);
    void blacs_setup_(int *pid, int *nprocs);
    void blacs_pinfo_(int *pid, int *nprocs);
    void blacs_get_(const int *ictxt, const int *what, int *val);
    void blacs_set_(const int *ictxt, const int *what, const int *val);
    void blacs_pcoord_(const int *ictxt, const int *pid, int *prow, int *pcol);
    int blacs_pnum_(const int *ictxt, const int *prow, const int *pcol);
    void blacs_gridexit_(int *ictxt);

    int Csys2blacs_handle(MPI_Comm SysCtxt);
    MPI_Comm Cblacs2sys_handle(int BlacsCtxt);
    void Cblacs_pinfo(int *myid, int *nprocs);
    void Cblacs_gridmap(int* ictxt, int *usermap, int ldumap, int nprow, int npcol);
    void Cblacs_gridinfo(int ictxt, int *nprow, int *npcol, int *prow, int *pcol);
    void Cblacs_gridinit(int *ictxt, char *layout, int nprow, int npcol);
    void Cblacs_gridexit(int ictxt);
    int  Cblacs_pnum(int ictxt, int prow, int pcol);
    void Cblacs_pcoord(int ictxt, int pnum, int *prow, int *pcol);
    void Cblacs_barrier(int ictxt, char *scope);
    void Csgsum2d(int ictxt, const char *scope, const char *top, int m, int n, float *a, int lda,
                  int rdest, int cdest);
    void Cdgsum2d(int ictxt, const char *scope, const char *top, int m, int n, double *a, int lda,
                  int rdest, int cdest);

////////////////////
/* PBLAS bindings */
////////////////////
// tools
    void descinit_(int *desc,
                   const int *m, const int *n, const int *mb, const int *nb,
                   const int *irsrc, const int *icsrc, const int *ictxt, const int *lld,
                   int *info);
    int numroc_(const int *n, const int *nb, const int *iproc, const int *srcproc, const int *nprocs);
    // void infog2l_();
    // void PB_Cinfog1l(int i, int j, int *desc, int nprow, int npcol,
    //                  int myrow, int mycol, int * ii, int * jj,
    //                  int *prow, int *pcol);
    // void PB_Cinfog2l(int i, int j, int *desc, int nprow, int npcol,
    //                  int myrow, int mycol, int * ii, int * jj,
    //                  int *prow, int *pcol);

    void pdpotrf_(char *uplo, int *n, double *a, int *ia, int *ja, int *desca, int *info);
    void pzpotrf_(char *uplo, int *n, std::complex<double> *a, int *ia, int *ja, int *desca, int *info);

    void pzposv_(
        const char *uplo, const int *n, const int *nrhs,   
        std::complex<double> *a, const int *ia, const int *ja, const int *descA,  
        std::complex<double> *b, const int *ib, const int *jb, const int *descB,  
        int *info
    );
    void pdtran_(int *m, int *n, double *alpha,
                 double *a, int *ia, int *ja, int *desca,
                 double *beta, 
                 double *c, int *ic, int *jc, int *descc);
    void pztranu_(const int *m, const int *n,
                  const std::complex<double> *alpha,
                  const std::complex<double> *a, const int *ia, const int *ja,
                  const int *desca, const std::complex<double> *beta,
                  std::complex<double> *c, const int *ic, const int *jc,
                  const int *descc);
    void pztranc_(const int *m, const int *n,
                  const std::complex<double> *alpha,
                  const std::complex<double> *a, const int *ia, const int *ja,
                  const int *desca, const std::complex<double> *beta,
                  std::complex<double> *c, const int *ic, const int *jc,
                  const int *descc);

    // scale
    // P?SCAL  multiplies  an  n  element  subvector  sub( X ) by the scalar alpha,
    // where
    //    sub( X ) denotes X(IX,JX:JX+N-1) if INCX = M_X,
    //                     X(IX:IX+N-1,JX) if INCX = 1 and INCX <> M_X.
    void psscal_(const int *N, const float *alpha, float *X, const int *IX,
                 const int *JX, const int *DESCX, const int *INCX);
    void pdscal_(const int *N, const double *alpha, double *X, const int *IX,
                 const int *JX, const int *DESCX, const int *INCX);
    void pcscal_(const int *N, const std::complex<float> *alpha,
                 std::complex<float> *X, const int *IX, const int *JX,
                 const int *DESCX, const int *INCX);
    void pzscal_(const int *N, const std::complex<double> *alpha,
                 std::complex<double> *X, const int *IX, const int *JX,
                 const int *DESCX, const int *INCX);
    // by real scalar
    void pcsscal_(const int *N, const float *alpha, std::complex<float> *X,
                  const int *IX, const int *JX, const int *DESCX, const int *INCX);
    void pzdscal_(const int *N, const double *alpha, std::complex<double> *X,
                  const int *IX, const int *JX, const int *DESCX, const int *INCX);

    // dot
    void psdot_(const int *N, float *DOT, const float *X, const int *IX,
                const int *JX, const int *DESCX, const int *INCX,
                const float *Y, const int *IY, const int *JY, const int *DESCY,
                const int *INCY);
    void pddot_(const int *N, double *DOT, const double *X, const int *IX,
                const int *JX, const int *DESCX, const int *INCX,
                const double *Y, const int *IY, const int *JY, const int *DESCY,
                const int *INCY);
    void pcdotc_(const int *N, std::complex<float> *DOT,
                 const std::complex<float> *X, const int *IX, const int *JX,
                 const int *DESCX, const int *INCX,
                 const std::complex<float> *Y, const int *IY, const int *JY,
                 const int *DESCY, const int *INCY);
    void pzdotc_(const int *N, std::complex<double> *DOT,
                 const std::complex<double> *X, const int *IX, const int *JX,
                 const int *DESCX, const int *INCX,
                 const std::complex<double> *Y, const int *IY, const int *JY,
                 const int *DESCY, const int *INCY);
    void pcdotu_(const int *N, std::complex<float> *DOT,
                 const std::complex<float> *X, const int *IX, const int *JX,
                 const int *DESCX, const int *INCX,
                 const std::complex<float> *Y, const int *IY, const int *JY,
                 const int *DESCY, const int *INCY);
    void pzdotu_(const int *N, std::complex<double> *DOT,
                 const std::complex<double> *X, const int *IX, const int *JX,
                 const int *DESCX, const int *INCX,
                 const std::complex<double> *Y, const int *IY, const int *JY,
                 const int *DESCY, const int *INCY);

    // matrix-vector
    void psgemv_(const char *transa, const int *M, const int *N,
                 const float *alpha, const float *A, const int *IA,
                 const int *JA, const int *DESCA, const float *X, const int *IX,
                 const int *JX, const int *DESCX, const int *INCX,
                 const float *beta, float *Y, const int *IY, const int *JY,
                 const int *DESCY, const int *INCY);
    void pdgemv_(const char *transa, const int *M, const int *N,
                 const double *alpha, const double *A, const int *IA,
                 const int *JA, const int *DESCA, const double *X,
                 const int *IX, const int *JX, const int *DESCX,
                 const int *INCX, const double *beta, double *Y, const int *IY,
                 const int *JY, const int *DESCY, const int *INCY);
    void pcgemv_(const char *transa, const int *M, const int *N,
                 const std::complex<float> *alpha, const std::complex<float> *A,
                 const int *IA, const int *JA, const int *DESCA,
                 const std::complex<float> *X, const int *IX, const int *JX,
                 const int *DESCX, const int *INCX,
                 const std::complex<float> *beta, std::complex<float> *Y,
                 const int *IY, const int *JY, const int *DESCY, const int *INCY);
    void pzgemv_(const char *transa, const int *M, const int *N,
                 const std::complex<double> *alpha,
                 const std::complex<double> *A, const int *IA, const int *JA,
                 const int *DESCA, const std::complex<double> *X, const int *IX,
                 const int *JX, const int *DESCX, const int *INCX,
                 const std::complex<double> *beta, std::complex<double> *Y,
                 const int *IY, const int *JY, const int *DESCY, const int *INCY);

    // matrix-matrix
    void psgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const float *alpha,
                 const float *A, const int *IA, const int *JA, const int *DESCA,
                 const float *B, const int *IB, const int *JB, const int *DESCB,
                 const float *beta,
                 float *C, const int *IC, const int *JC, const int *DESCC);
    void pcgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const std::complex<float> *alpha,
                 const std::complex<float> *A, const int *IA, const int *JA, const int *DESCA,
                 const std::complex<float> *B, const int *IB, const int *JB, const int *DESCB,
                 const std::complex<float> *beta,
                 std::complex<float> *C, const int *IC, const int *JC, const int *DESCC);
    void pdgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const double *alpha,
                 const double *A, const int *IA, const int *JA, const int *DESCA,
                 const double *B, const int *IB, const int *JB, const int *DESCB,
                 const double *beta,
                 double *C, const int *IC, const int *JC, const int *DESCC);
    void pzgemm_(const char *transa, const char *transb,
                 const int *M, const int *N, const int *K,
                 const std::complex<double> *alpha,
                 const std::complex<double> *A, const int *IA, const int *JA, const int *DESCA,
                 const std::complex<double> *B, const int *IB, const int *JB, const int *DESCB,
                 const std::complex<double> *beta,
                 std::complex<double> *C, const int *IC, const int *JC, const int *DESCC);
    void pdsymm_(const char *side, const char *uplo,
                 const int *m, const int *n, const double *alpha,
                 const double *a, const int *ia, const int *ja, const int *desca,
                 const double *b, const int *ib, const int *jb, const int *descb,
                 const double *beta,
                 double *c, const int *ic, const int *jc, const int *descc);
    void pdtrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                 const int *m, const int *n,
                 const double *alpha,
                 const double *a, const int *ia, const int *ja, const int *desca,
                 double *b, const int *ib, const int *jb, const int *descb);
    void pztrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
                 const int *m, const int *n,
                 const std::complex<double> *alpha, const std::complex<double> *a, const int *ia, const int *ja, const int *desca,
                 std::complex<double> *b, const int *ib, const int *jb, const int *descb);

// factoriazation
    void psgetrf_(const int *M, const int *N, 
                  float *A, const int *IA, const int *JA, const int *DESCA,
                  int *ipiv, int *info);
    void pdgetrf_(const int *M, const int *N, 
                  double *A, const int *IA, const int *JA, const int *DESCA,
                  int *ipiv, int *info);
    void pcgetrf_(const int *M, const int *N, 
                  std::complex<float> *A, const int *IA, const int *JA, const int *DESCA,
                  int *ipiv, int *info);
    void pzgetrf_(const int *M, const int *N, 
                  std::complex<double> *A, const int *IA, const int *JA, const int *DESCA,
                  int *ipiv, int *info);

    void pzgesv_(const int* n, const int* nrhs, std::complex<double>* a, const int* ia, const int* ja,   
                 const int* DESCA, int* ipiv, std::complex<double>* b, const int* ib, const int* jb,   
                 const int* DESCB, int* info);  

// Eigenvalues of symmetric and hermitian matrices
    void pssygvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                  const int *n, float *A, const int *ia, const int *ja, const int*desca,
                  float *B, const int *ib, const int *jb, const int*descb,
                  const float *vl, const float *vu, const int *il, const int *iu,
                  const float *abstol, int *m, int *nz, float *w, const float *orfac,
                  float *Z, const int *iz, const int *jz, const int *descz,
                  float *work, int *lwork, int *iwork, int *liwork, int *ifail, int *iclustr,
                  float *gap, int *info);
    void pdsygvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                  const int *n, double *A, const int *ia, const int *ja, const int*desca,
                  double *B, const int *ib, const int *jb, const int*descb,
                  const double *vl, const double *vu, const int *il, const int *iu,
                  const double *abstol, int *m, int *nz, double *w, const double *orfac,
                  double *Z, const int *iz, const int *jz, const int *descz,
                  double *work, int *lwork, int *iwork, int *liwork, int *ifail, int *iclustr,
                  double *gap, int *info);
    void pchegvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                  const int *n, std::complex<float> *A, const int *ia, const int *ja, const int *desca,
                  std::complex<float> *B, const int *ib, const int *jb, const int *descb,
                  const float *vl, const float *vu, const int *il, const int *iu,
                  const float *abstol, int *m, int *nz, float *w, const float *orfac,
                  std::complex<float> *Z, const int *iz, const int *jz, const int*descz,
                  std::complex<float> *work, int *lwork, float *rwork, int *lrwork, int *iwork, int*liwork, int *ifail, int *iclustr,
                  float *gap, int *info);
    void pzhegvx_(const int *itype, const char *jobz, const char *range, const char *uplo,
                  const int *n, std::complex<double> *A, const int *ia, const int *ja, const int *desca,
                  std::complex<double> *B, const int *ib, const int *jb, const int *descb,
                  const double *vl, const double *vu, const int *il, const int *iu,
                  const double *abstol, int *m, int *nz, double *w, const double *orfac,
                  std::complex<double> *Z, const int *iz, const int *jz, const int*descz,
                  std::complex<double> *work, int *lwork, double *rwork, int *lrwork, int *iwork, int*liwork, int *ifail, int *iclustr,
                  double *gap, int *info);

    void pssyev_(const char *jobz, const char *uplo,
                 const int *n, float *A, const int *ia, const int *ja, const int *desca,
                 float *W, float *Z, const int *iz, const int *jz, const int *descz,
                 float *work, const int *lwork, int *info);
    void pdsyev_(const char *jobz, const char *uplo,
                 const int *n, double *A, const int *ia, const int *ja, const int *desca,
                 double *W, double *Z, const int *iz, const int *jz, const int *descz,
                 double *work, const int *lwork, int *info);
    void pcheev_(const char *jobz, const char *uplo,
                 const int *n, std::complex<float> *A, const int *ia, const int *ja, const int *desca,
                 float *W, std::complex<float> *Z, const int *iz, const int *jz, const int *descz,
                 std::complex<float> *work, const int *lwork, float *rwork, const int *lrwork, int *info);
    void pzheev_(const char *jobz, const char *uplo,
                 const int *n, std::complex<double> *A, const int *ia, const int *ja, const int *desca,
                 double *W, std::complex<double> *Z, const int *iz, const int *jz, const int *descz,
                 std::complex<double> *work, const int *lwork, double *rwork, const int *lrwork, int *info);

// Matrix inversion
    void psgetri_(const int *n, 
                  float *A, const int *ia, const int *ja, const int *desca,
                  int *ipiv, float *work, const int *lwork, int *iwork, const int *liwork,
                  int *info);
    void pdgetri_(const int *n, 
                  double *A, const int *ia, const int *ja, const int *desca,
                  int *ipiv, double *work, const int *lwork, int *iwork, const int *liwork,
                  int *info);
    void pcgetri_(const int *n, 
                  std::complex<float> *A, const int *ia, const int *ja, const int *desca,
                  int *ipiv, std::complex<float> *work, const int *lwork, int *iwork, const int *liwork,
                  int *info);
    void pzgetri_(const int *n, 
                  std::complex<double> *A, const int *ia, const int *ja, const int *desca,
                  int *ipiv, std::complex<double> *work, const int *lwork, int *iwork, const int *liwork,
                  int *info);

// Matrix addition
    void psgeadd_(const char *transa, const int *m, const int *n,
                  float *alpha,
                  float *a, const int *ia, const int *ja, const int *desca,
                  float *beta,
                  float *c, const int *ic, const int *jc, const int *descc);
    void pdgeadd_(const char *transa, const int *m, const int *n,
                  double *alpha,
                  double *a, const int *ia, const int *ja, const int *desca,
                  double *beta,
                  double *c, const int *ic, const int *jc, const int *descc);
    void pcgeadd_(const char *transa, const int *m, const int *n,
                  const std::complex<float> *alpha,
                  const std::complex<float> *a, const int *ia, const int *ja, const int *desca,
                  const std::complex<float> *beta,
                  std::complex<float> *c, const int *ic, const int *jc, const int *descc);
    void pzgeadd_(const char *transa, const int *m, const int *n,
                  const std::complex<double> *alpha,
                  const std::complex<double> *a, const int *ia, const int *ja, const int *desca,
                  const std::complex<double> *beta,
                  std::complex<double> *c, const int *ic, const int *jc, const int *descc);

// submatrix redistribution
    void psgemr2d_(const int *m, const int *n,
                   const float *a, const int *ia, const int *ja, const int *desca,
                   float *b, const int *ib, const int *jb, const int *descb,
                   const int *ictxt);
    void pdgemr2d_(const int *m, const int *n,
                   const double *a, const int *ia, const int *ja, const int *desca,
                   double *b, const int *ib, const int *jb, const int *descb,
                   const int *ictxt);
    void pcgemr2d_(const int *m, const int *n,
                   const std::complex<float> *a, const int *ia, const int *ja, const int *desca,
                   std::complex<float>*b, const int *ib, const int *jb, const int *descb,
                   const int *ictxt);
    void pzgemr2d_(const int *m, const int *n,
                   const std::complex<double> *a, const int *ia, const int *ja, const int *desca,
                   std::complex<double> *b, const int *ib, const int *jb, const int *descb,
                   const int *ictxt);

    int pilaenvx_(const int *ictxt, const int *ispec, const char *name, const char *opts,
                  const int *n1, const int *n2, const int *n3, const int *n4);

    int pjlaenv_(const int *ictxt, const int *ispec, const char *name, const char *opts,
                 const int *n1, const int *n2, const int *n3, const int *n4);
#ifdef __cplusplus
}
#endif /* __cplusplus */
