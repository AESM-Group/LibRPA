#ifndef LAPACKCONNECTOR_HPP
#define LAPACKCONNECTOR_HPP

#include <vector>
#include <new>
#include <stdexcept>
#include <iostream>
#include <cassert>
#include <cstring>
#include "../../utils/base_utility.h"
#include "../../math/matrix.h"
#include "../../math/complexmatrix.h"
#include "blas_lapack.h"

namespace librpa_int {

// Class LapackConnector provide the connector to fortran lapack routine.
// The entire function in this class are static and inline function.
// Usage example:	LapackConnector::functionname(parameter list).
class LapackConnector
{
public:
    // Raw column-major solves, without the legacy matrix-container conversion.
    static inline void gesv_f(int n, int nrhs, float* a, int lda, int* ipiv,
                              float* b, int ldb, int& info)
    { sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); }
    static inline void gesv_f(int n, int nrhs, double* a, int lda, int* ipiv,
                              double* b, int ldb, int& info)
    { dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); }
    static inline void gesv_f(int n, int nrhs, std::complex<float>* a, int lda, int* ipiv,
                              std::complex<float>* b, int ldb, int& info)
    { cgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); }
    static inline void gesv_f(int n, int nrhs, std::complex<double>* a, int lda, int* ipiv,
                              std::complex<double>* b, int ldb, int& info)
    { zgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info); }

    static inline void posv_f(char uplo, int n, int nrhs, float* a, int lda,
                              float* b, int ldb, int& info)
    { sposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info); }
    static inline void posv_f(char uplo, int n, int nrhs, double* a, int lda,
                              double* b, int ldb, int& info)
    { dposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info); }
    static inline void posv_f(char uplo, int n, int nrhs, std::complex<float>* a, int lda,
                              std::complex<float>* b, int ldb, int& info)
    { cposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info); }
    static inline void posv_f(char uplo, int n, int nrhs, std::complex<double>* a, int lda,
                              std::complex<double>* b, int ldb, int& info)
    { zposv_(&uplo, &n, &nrhs, a, &lda, b, &ldb, &info); }

    // Transpose an row-major array to the fortran-form colum-major array.
    template <typename T>
    static inline T* transpose(const T* a, const int &n, const int &lda, bool conjugate = false)
    {
        T* a_fort = new T[lda*n];
        if (is_complex<T>() && conjugate)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < lda; j++)
                    a_fort[i*lda+j] = get_conj(a[j*n+i]);
        else
            for (int i = 0; i < n; i++)
                for (int j = 0; j < lda; j++)
                    a_fort[i*lda+j] = a[j*n+i];
        return a_fort;
    }

    template <typename T>
    static inline void transpose(const T* a, T* a_trans, const int &n, const int &lda, bool conjugate = false)
    {
        if (is_complex<T>() && conjugate)
            for (int i = 0; i < n; i++)
                for (int j = 0; j < lda; j++)
                    a_trans[j*n+i] = get_conj(a[i*lda+j]);
        else
            for (int i = 0; i < n; i++)
                for (int j = 0; j < lda; j++)
                    a_trans[j*n+i] = a[i*lda+j];
    }

    // Transpose the complex matrix to the fortran-form real-complex array.
    static inline
    complex<double>* transpose(const ComplexMatrix& a, const int n, const int lda)
    {
        complex<double>* aux = new complex<double>[lda*n];
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                aux[i*lda+j] = a(j,i);		// aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
        return aux;
    }
    // Transpose the  matrix to the fortran-form real array.
    static inline
    double * transpose_matrix(const matrix& a, const int n, const int lda)
    {
        double * aux = new double [lda*n];
		std::cout << " lda=" << lda << " n=" << n << std::endl;
        for (int i=0; i<n; i++)
        {
            for (int j=0; j<lda; j++)
            {
                aux[i*lda+j] = a(j,i);      // aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
        return aux;
    }

    // Transpose the fortran-form real-complex array to the complex matrix.
    static inline
    void transpose(const complex<double>* aux, ComplexMatrix& a, const int n, const int lda)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                a(j, i) = aux[i*lda+j];		// aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
    }
    
	// Transpose the fortran-form real array to the matrix.
    static inline
    void transpose_matrix(const double *aux, matrix &a, const int n, const int lda)
    {
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                a(j, i) = aux[i*lda+j];     // aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
            }
        }
    }
	
	// Peize Lin add 2015-12-27
	static inline
	char change_side(const char &side)
	{
		switch(side)
		{
			case 'L': return 'R';
			case 'R': return 'L';
			default: throw std::invalid_argument("Side must be 'L' or 'R'");
		}		
	}
	
	// Peize Lin add 2015-12-27	
	static inline
	char change_uplo(const char &uplo)
	{
		switch(uplo)
		{
			case 'U': return 'L';
			case 'L': return 'U';
			default: throw std::invalid_argument("uplo must be 'U' or 'L'");
		}		
	}
	
	// Peize Lin add 2019-04-14	
	static inline
	char change_trans_NT(const char &trans)
	{
		switch(trans)
		{
			case 'N': return 'T';
			case 'T': return 'N';
			default: throw std::invalid_argument("trans must be 'N' or 'T'");
		}
	}
	// Peize Lin add 2019-04-14
	static inline
	char change_trans_NC(const char &trans)
	{
		switch(trans)
		{
			case 'N': return 'C';
			case 'C': return 'N';
			default: throw std::invalid_argument("trans must be 'N' or 'C'");
		}
	}
	
public:

    static inline
    int ilaenv( int ispec, const char *name,const char *opts,const int n1,const int n2,
                const int n3,const int n4)
    {
        const int name_len = int(std::strlen(name));
        const int opts_len = int(std::strlen(opts));
        const int nb = ilaenv_(&ispec, name, opts, &n1, &n2, &n3, &n4, name_len, opts_len);
        return nb;
    }

    static inline
    void zgesv (const int n,const int nrhs,ComplexMatrix &A,const int lda,
                const int *ipiv,complex<double> *B,const int ldb,int* info)
    {
        complex<double> *aux = LapackConnector::transpose(A, n, lda);
        zgesv_(&n, &nrhs, aux, &lda, ipiv, B, &ldb, info);
        LapackConnector::transpose(aux, A, n, lda);
    }

    // wrap function of fortran lapack routine zhegv.
    static inline
    void zhegv(	const int itype,const char jobz,const char uplo,const int n,ComplexMatrix& a,
                const int lda,ComplexMatrix& b,const int ldb,double* w,complex<double>* work,
                int lwork,double* rwork,int info)
    {	// Transpose the complex matrix to the fortran-form real-complex array.
        complex<double>* aux = LapackConnector::transpose(a, n, lda);
        complex<double>* bux = LapackConnector::transpose(b, n, ldb);

        // call the fortran routine
        zhegv_(&itype, &jobz, &uplo, &n, aux, &lda, bux, &ldb, w, work, &lwork, rwork, &info);
        // Transpose the fortran-form real-complex array to the complex matrix.
        LapackConnector::transpose(aux, a, n, lda);
        LapackConnector::transpose(bux, b, n, ldb);
        // free the memory.
        delete[] aux;
        delete[] bux;
    }


	// calculate the selected eigenvalues.
	// mohan add 2010/03/21
	// static inline
	// void sspgvx(const int itype, const char jobz,const char range,const char uplo,
	// 			const int n,const matrix &ap,const matrix &bp,const double vl,
	// 			const int vu, const int il, const int iu,const double abstol,
	// 			const int m,double* w,matrix &z,const int ldz,
	// 			double *work,int* iwork,int* ifail,int* info)
	// {
	//     // Transpose the complex matrix to the fortran-form real*16 array.
	// 	double* aux = LapackConnector::transpose_matrix(ap, n, n);
	// 	double* bux = LapackConnector::transpose_matrix(bp, n, n);
	// 	double* zux = new double[n*iu];
	//
 //        // call the fortran routine
 //        sspgvx_(&itype, &jobz, &range, &uplo, 
	// 			&n, aux, bux, &vl,
 //                &vu, &il,&iu, &abstol, 
	// 			&m, w, zux, &ldz, 
	// 			work, iwork, ifail, info);
	//
 //        // Transpose the fortran-form real*16 array to the matrix
 //        LapackConnector::transpose_matrix(zux, z, iu, n);
	//
 //        // free the memory.
 //        delete[] aux;
 //        delete[] bux;
 //        delete[] zux;
	// }

	// calculate the eigenvalues and eigenfunctions of a real symmetric matrix.
    static inline
    void dsygv(	const int itype,const char jobz,const char uplo,const int n,matrix& a,
                const int lda,matrix& b,const int ldb,double* w,double* work,
                int lwork,int *info)
    {	// Transpose the complex matrix to the fortran-form real-complex array.
        double* aux = new double[lda*n];
        double* bux = new double[lda*n];
	    for (int i=0; i<n; i++)
        {
            for (int j=0; j<lda; j++)
            {
                aux[i*lda+j] = a(j,i);      // aux[i*lda+j] means aux[i][j] in semantic, not in syntax!
				bux[i*lda+j] = b(j,i);
            }
        }
        // call the fortran routine
        dsygv_(&itype, &jobz, &uplo, &n, aux, &lda, bux, &ldb, w, work, &lwork, info);
		for (int i=0; i<n; i++)
		{
			for(int j=0; j<lda; ++j)
			{
				a(j,i)=aux[i*lda+j];
				b(j,i)=bux[i*lda+j];
			}
		}
 
        // free the memory.
        delete[] aux;
        delete[] bux;
    }

	// wrap function of fortran lapack routine dsyev.
    static inline
	void dsyev(const char jobz,const char uplo,const int n,double *a,
			const int lda,double *w,double *work,const int lwork, int &info)
	{
		const char uplo_changed = change_uplo(uplo);
		dsyev_(&jobz, &uplo_changed, &n, a, &lda, w, work, &lwork, &info);
	}
    // static inline
	// void dsyev( const char jobz, const char uplo, matrix &a, double* w, int info )		// Peize Lin update 2017-10-17
	// {
		// assert(a.nr==a.nc);
		// const char uplo_changed = change_uplo(uplo);
		
		// double work_tmp;
		// constexpr int minus_one = -1;
		// dsyev_(&jobz, &uplo_changed, &a.nr, a.c, &a.nr, w, &work_tmp, &minus_one, &info);		// get best lwork
		
		// const int lwork = work_tmp;
		// vector<double> work(std::max(1,lwork));
		// dsyev_(&jobz, &uplo_changed, &a.nr, a.c, &a.nr, w, VECTOR_TO_PTR(work), &lwork, &info);		
	// }
    // wrap function of fortran lapack routine zheev.
    static inline
    void zheev( const char jobz,
                const char uplo,
                const int n,
                ComplexMatrix& a,
                const int lda,
                double* w,
                complex< double >* work,
                const int lwork,
                double* rwork,
                int *info	)
    {	// Transpose the complex matrix to the fortran-form real-complex array.
        complex<double> *aux = LapackConnector::transpose(a, n, lda);
        // call the fortran routine
        zheev_(&jobz, &uplo, &n, aux, &lda, w, work, &lwork, rwork, info);
        // Transpose the fortran-form real-complex array to the complex matrix.
        LapackConnector::transpose(aux, a, n, lda);
        // free the memory.
        delete[] aux;
    }
	
    // wrap function of fortran lapack routine dsytrf.
    static inline
    void dsytrf(char uplo, int n, matrix& a, int lda, int *ipiv,
                double *work, int lwork , int info)
    {
        double * aux = LapackConnector::transpose_matrix(a, n, lda) ;
        dsytrf_(&uplo, &n, aux, &lda, ipiv, work, &lwork, &info);
        LapackConnector::transpose_matrix(aux, a, n, lda);
    }

    // wrap function of fortran lapack routine dsytri.
    static inline
    void dsytri(char uplo, int n, matrix& a, int lda, int *iwork,double * work, int info)
    {
        double * aux = LapackConnector::transpose_matrix(a, n, lda) ;
        dsytri_(&uplo, &n, aux, &lda, iwork, work, &info);
        LapackConnector::transpose_matrix(aux, a, n, lda);
    }


    // wrap function of fortran lapack routine zheev.
    static inline
    void zhegvx( const int itype, const char jobz, const char range, const char uplo,
                 const int n, const ComplexMatrix& a, const int lda, const ComplexMatrix& b,
                 const int ldb, const double vl, const double vu, const int il, const int iu,
                 const double abstol, const int m, double* w, ComplexMatrix& z, const int ldz,
                 complex<double>* work, const int lwork, double* rwork, int* iwork,
                 int* ifail, int info)
    {
        // Transpose the complex matrix to the fortran-form real-complex array.
        complex<double>* aux = LapackConnector::transpose(a, n, lda);
        complex<double>* bux = LapackConnector::transpose(b, n, ldb);
        complex<double>* zux = new complex<double>[n*iu];// mohan modify 2009-08-02
        //for(int i=0; i<n*iu; i++) zux[i] = complex<double>(0.0,0.0);

        // call the fortran routine
        zhegvx_(&itype, &jobz, &range, &uplo, &n, aux, &lda, bux, &ldb, &vl,
                &vu, &il,&iu, &abstol, &m, w, zux, &ldz, work, &lwork, rwork, iwork, ifail, &info);

        // Transpose the fortran-form real-complex array to the complex matrix
        LapackConnector::transpose(zux, z, iu, n);	 // mohan modify 2009-08-02

        // free the memory.
        delete[] aux;
        delete[] bux;
        delete[] zux;

    }
    static inline
    void dgesvd(const char jobu,const char jobvt,const int m,const int n,
                matrix &a,const int lda,double *s,matrix &u,const int ldu,
                matrix &vt,const int ldvt,double *work,const int lwork,int info)
    {
        //Transpose the matrix to the fortran-form

        double *aux = LapackConnector::transpose_matrix(a, n, lda);
        double *uux = LapackConnector::transpose_matrix(u, m, ldu);
        double *vtux = LapackConnector::transpose_matrix(vt, n, ldvt);

        dgesvd_(&jobu, &jobvt , &m, &n, aux, &lda, s, uux, &ldu, vtux, &ldvt, work, &lwork, &info);

        LapackConnector::transpose_matrix(aux, a, n, lda);
        LapackConnector::transpose_matrix(uux, u, m, ldu);
        LapackConnector::transpose_matrix(vtux, vt, n, ldvt);

        delete[] aux;
        delete[] uux;
        delete[] vtux;

    }

    static inline
    void zpotrf(char uplo,int n,ComplexMatrix &a,const int lda,int* info)
    {
        complex<double> *aux = LapackConnector::transpose(a, n, lda);
        zpotrf_( &uplo, &n, aux, &lda, info);
        LapackConnector::transpose(aux, a, n, lda);
        delete[] aux;
		return;
    }

    static inline
    void zpotri(char uplo,int n,ComplexMatrix &a,const int lda,int* info)
    {
        complex<double> *aux = LapackConnector::transpose(a, n, lda);
        zpotri_( &uplo, &n, aux, &lda, info);
        LapackConnector::transpose(aux, a, n, lda);
        delete[] aux;
		return;
    }

 //    static inline
 //    void spotrf(char uplo,int n,matrix &a, int lda,int *info)
	// {
	// 	double *aux = LapackConnector::transpose_matrix(a, n, lda);
	// 	for(int i=0; i<4; i++)std::cout << "\n aux=" << aux[i]; 
	// 	spotrf_( &uplo, &n, aux, &lda, info);
	// 	for(int i=0; i<4; i++)std::cout << "\n aux=" << aux[i]; 
	// 	LapackConnector::transpose_matrix(aux, a, n, lda);
	// 	delete[] aux;
	// 	return;
	// }
	//
	// static inline
	// void spotri(char uplo,int n,matrix &a, int lda, int *info)
	// {
	// 	double *aux = LapackConnector::transpose_matrix(a, n, lda);
	// 	for(int i=0; i<4; i++)std::cout << "\n aux=" << aux[i]; 
	// 	spotri_( &uplo, &n, aux, &lda, info);
	// 	for(int i=0; i<4; i++)std::cout << "\n aux=" << aux[i]; 
	// 	LapackConnector::transpose_matrix(aux, a, n, lda);
	// 	delete[] aux;
	// 	return;
	// }

    static inline
    void zgetrf(int m, int n, ComplexMatrix &a, const int lda, int *ipiv, int *info)
    {
        complex<double> *aux = LapackConnector::transpose(a, n, lda);
        zgetrf_( &m, &n, aux, &lda, ipiv, info);
        LapackConnector::transpose(aux, a, n, lda);
        delete[] aux;
                return;
    }
    static inline
    void zgetri(int n, ComplexMatrix &a,  int lda, int *ipiv, complex<double> * work, int lwork, int *info)
    {
        complex<double> *aux = LapackConnector::transpose(a, n, lda);
        zgetri_( &n, aux, &lda, ipiv, work, &lwork, info);
        LapackConnector::transpose(aux, a, n, lda);
        delete[] aux;
        return;
    }

    static inline void getrf(const int &m, const int &n, float *A, const int &lda, int *ipiv, int &info)
    {
        float *a_fort = transpose(A, n, lda);
        sgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getrf(const int &m, const int &n, double *A, const int &lda, int *ipiv, int &info)
    {
        double *a_fort = transpose(A, n, lda);
        dgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getrf(const int &m, const int &n, std::complex<float> *A, const int &lda, int *ipiv, int &info)
    {
        std::complex<float> *a_fort = transpose(A, n, lda);
        cgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getrf(const int &m, const int &n, std::complex<double> *A, const int &lda, int *ipiv, int &info)
    {
        std::complex<double> *a_fort = transpose(A, n, lda);
        zgetrf_(&m, &n, a_fort, &lda, ipiv, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getrf_f(const int &m, const int &n, float *A, const int &lda, int *ipiv, int &info)
    {
        sgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    static inline void getrf_f(const int &m, const int &n, double *A, const int &lda, int *ipiv, int &info)
    {
        dgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    static inline void getrf_f(const int &m, const int &n, std::complex<float> *A, const int &lda, int *ipiv, int &info)
    {
        cgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    static inline void getrf_f(const int &m, const int &n, std::complex<double> *A, const int &lda, int *ipiv, int &info)
    {
        zgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    static inline void getri(const int &n, float *A, const int &lda, int *ipiv, float *work, const int &lwork, int &info)
    {
        float *a_fort = transpose(A, n, lda);
        sgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getri(const int &n, double *A, const int &lda, int *ipiv, double *work, const int &lwork, int &info)
    {
        double *a_fort = transpose(A, n, lda);
        dgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getri(const int &n, std::complex<float> *A, const int &lda, int *ipiv, std::complex<float> *work, const int &lwork, int &info)
    {
        std::complex<float> *a_fort = transpose(A, n, lda);
        cgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getri(const int &n, std::complex<double> *A, const int &lda, int *ipiv, std::complex<double> *work, const int &lwork, int &info)
    {
        std::complex<double> *a_fort = transpose(A, n, lda);
        zgetri_(&n, a_fort, &lda, ipiv, work, &lwork, &info);
        transpose(a_fort, A, n, lda);
        delete [] a_fort;
    }

    static inline void getri_f(const int &n, float *A, const int &lda, int *ipiv, float *work, const int &lwork, int &info)
    {
        sgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

    static inline void getri_f(const int &n, double *A, const int &lda, int *ipiv, double *work, const int &lwork, int &info)
    {
        dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

    static inline void getri_f(const int &n, std::complex<float> *A, const int &lda, int *ipiv, std::complex<float> *work, const int &lwork, int &info)
    {
        cgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

    static inline void getri_f(const int &n, std::complex<double> *A, const int &lda, int *ipiv, std::complex<double> *work, const int &lwork, int &info)
    {
        zgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    }

	// Peize Lin add 2016-07-09
	static inline
	void dpotrf( char uplo, const int n, matrix &a, const int lda, int *info )
	{
		const char uplo_changed = change_uplo(uplo);
		dpotrf_( &uplo_changed, &n, a.c, &lda, info );
	}	
	
	// Peize Lin add 2016-07-09
	static inline
	void dpotri( char uplo, const int n, matrix &a, const int lda, int *info )
	{
		const char uplo_changed = change_uplo(uplo);
		dpotri_( &uplo_changed, &n, a.c, &lda, info);		
	}	
	
	// Peize Lin add 2016-08-04
	// y=a*x+y
	static inline 
	void axpy( const int n, const float alpha, const float *X, const int incX, float *Y, const int incY)
	{
		saxpy_(&n, &alpha, X, &incX, Y, &incY);
	}	
	static inline 
	void axpy( const int n, const double alpha, const double *X, const int incX, double *Y, const int incY)
	{
		daxpy_(&n, &alpha, X, &incX, Y, &incY);
	}	
	static inline 
	void axpy( const int n, const complex<float> alpha, const complex<float> *X, const int incX, complex<float> *Y, const int incY)
	{
		caxpy_(&n, &alpha, X, &incX, Y, &incY);
	}	
	static inline 
	void axpy( const int n, const complex<double> alpha, const complex<double> *X, const int incX, complex<double> *Y, const int incY)
	{
		zaxpy_(&n, &alpha, X, &incX, Y, &incY);
	}	
	
	// Peize Lin add 2016-08-04
	// x=a*x
	static inline 
	void scal( const int n,  const float alpha, float *X, const int incX)
	{
		sscal_(&n, &alpha, X, &incX);
	}	
	static inline 
	void scal( const int n, const double alpha, double *X, const int incX)
	{
		dscal_(&n, &alpha, X, &incX);
	}	
	static inline 
	void scal( const int n, const complex<float> alpha, complex<float> *X, const int incX)
	{
		cscal_(&n, &alpha, X, &incX);
	}	
	static inline 
	void scal( const int n, const complex<double> alpha, complex<double> *X, const int incX)
	{
		zscal_(&n, &alpha, X, &incX);
	}	
	
	// Peize Lin add 2017-10-27
	// d=x*y
	static inline
	float dot( const int n, const float *X, const int incX, const float *Y, const int incY)
	{
		return sdot_(&n, X, &incX, Y, &incY);
	}
	static inline
	double dot( const int n, const double *X, const int incX, const double *Y, const int incY)
	{
		return ddot_(&n, X, &incX, Y, &incY);
	}

    // minyez add 2022-11-15
    static inline
    std::complex<float> dot( const int n, const std::complex<float> *X, const int incX, const std::complex<float> *Y, const int incY)
    {
        std::complex<float> result;
        cdotu_(&result, &n, X, &incX, Y, &incY);
        return result;
    }

    static inline
    std::complex<double> dot( const int n, const std::complex<double> *X, const int incX, const std::complex<double> *Y, const int incY)
    {
        std::complex<double> result;
        zdotu_(&result, &n, X, &incX, Y, &incY);
        return result;
    }

    // minyez add 2022-05-12
    // matrix-vector product
    // single-prec version
    static inline
    void gemv(const char transa, const int m, const int n, const float alpha, const float *a,
              const int lda, const float *x, const int incx, const float beta, float *y, const int incy)
    {
        char transa_f = change_trans_NT(transa);
        sgemv_(&transa_f, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }
    // double-prec version
    static inline
    void gemv(const char transa, const int m, const int n, const double alpha, const double *a,
              const int lda, const double *x, const int incx, const double beta, double *y, const int incy)
    {
        char transa_f = change_trans_NT(transa);
        dgemv_(&transa_f, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const float &alpha, const float *a, const int &lda,
                       const float *x, const int &incx, const float &beta, float *y, const int &incy)
    {
        sgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const double &alpha, const double *a, const int &lda,
                       const double *x, const int &incx, const double &beta, double *y, const int &incy)
    {
        dgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const std::complex<float> &alpha, const std::complex<float> *a, const int &lda,
                       const std::complex<float> *x, const int &incx, const std::complex<float> &beta, std::complex<float> *y, const int &incy)
    {
        cgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    inline void gemv_f(const char &transa, const int &m, const int &n,
                       const std::complex<double> &alpha, const std::complex<double> *a, const int &lda,
                       const std::complex<double> *x, const int &incx, const std::complex<double> &beta, std::complex<double> *y, const int &incy)
    {
        zgemv_(&transa, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
    }

    // Column-major A <- A + alpha * x * y^H. Vector strides follow BLAS conventions.
    static inline void gerc_f(const int m, const int n, const std::complex<double> alpha,
                              const std::complex<double> *x, const int incx,
                              const std::complex<double> *y, const int incy,
                              std::complex<double> *a, const int lda)
    {
        zgerc_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
    }

    // Peize Lin add 2017-10-27, fix bug trans 2019-01-17
	// C = a * A.? * B.? + b * C 
	static inline
	void gemm(const char transa, const char transb, const int m, const int n, const int k,
		const float alpha, const float *a, const int lda, const float *b, const int ldb, 
		const float beta, float *c, const int ldc)
	{
		sgemm_(&transb, &transa, &n, &m, &k,
			&alpha, b, &ldb, a, &lda, 
			&beta, c, &ldc);
	}
    
	static inline
	void gemm(const char transa, const char transb, const int m, const int n, const int k,
		const double alpha, const double *a, const int lda, const double *b, const int ldb, 
		const double beta, double *c, const int ldc)
	{
		dgemm_(&transb, &transa, &n, &m, &k,
			&alpha, b, &ldb, a, &lda, 
			&beta, c, &ldc);
	}

	static inline
    void gemm(const char transa, const char transb, const int m,
              const int n, const int k, const complex<float> alpha,
              const complex<float> *a, const int lda,
              const complex<float> *b, const int ldb,
              const complex<float> beta, complex<float> *c, const int ldc)
    {
        cgemm_(&transb, &transa, &n, &m, &k, &alpha, b, &ldb, a, &lda,
               &beta, c, &ldc);
    }

        static inline
    void gemm(const char transa, const char transb, const int m,
              const int n, const int k, const complex<double> alpha,
              const complex<double> *a, const int lda,
              const complex<double> *b, const int ldb,
              const complex<double> beta, complex<double> *c, const int ldc)
    {
		zgemm_(&transb, &transa, &n, &m, &k,
			&alpha, b, &ldb, a, &lda, 
			&beta, c, &ldc);
	}

    static inline
    void gemm_f(const char transa, const char transb,
                const int m, const int n, const int k,
                const float alpha, const float *a,
                const int lda, const float *b, const int ldb,
                const float beta, float *c, const int ldc)
    {
        sgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
               &beta, c, &ldc);
    }

    static inline void gemm_f(const char transa, const char transb,
                              const int m, const int n, const int k,
                              const complex<float> alpha,
                              const complex<float> *a, const int lda,
                              const complex<float> *b, const int ldb,
                              const complex<float> beta,
                              complex<float> *c, const int ldc)
    {
        cgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
               &ldc);
    }

    static inline void gemm_f(const char transa, const char transb,
                              const int m, const int n, const int k,
                              const double alpha, const double *a,
                              const int lda, const double *b, const int ldb,
                              const double beta, double *c, const int ldc)
    {
        dgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb,
               &beta, c, &ldc);
    }

    static inline void gemm_f(const char transa, const char transb,
                              const int m, const int n, const int k,
                              const complex<double> alpha,
                              const complex<double> *a, const int lda,
                              const complex<double> *b, const int ldb,
                              const complex<double> beta,
                              complex<double> *c, const int ldc)
    {
        zgemm_(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
               &ldc);
    }

    // eigenvector of symmetric matrix, row-major
    static inline
    void syev(const char jobz, const char uplo, const int n, float *a,
              const int lda, float *w, float *work, const int lwork, int &info)
    {
        const char uplo_changed = change_uplo(uplo);
        ssyev_(&jobz, &uplo_changed, &n, a, &lda, w, work, &lwork, &info);
    }

    static inline
    void syev(const char jobz, const char uplo, const int n, double *a,
              const int lda, double *w, double *work, const int lwork, int &info)
    {
        const char uplo_changed = change_uplo(uplo);
        dsyev_(&jobz, &uplo_changed, &n, a, &lda, w, work, &lwork, &info);
    }

    // eigenvector of symmetric matrix, column-major
    static inline
    void syev_f(const char jobz, const char uplo, const int n, float *a,
              const int lda, float *w, float *work, const int lwork, int &info)
    {
        ssyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
    }

    static inline
    void syev_f(const char jobz, const char uplo, const int n, double *a,
              const int lda, double *w, double *work, const int lwork, int &info)
    {
        dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, &info);
    }

    // eigenvector of hermitian matrix, row-major
    static inline
    void heev(const char &jobz, const char &uplo, const int &n,
              std::complex<float> *a, const int &lda, float *w,
              std::complex<float> *work, const int &lwork, float *rwork, int &info)
    {
        complex<float> *aux = LapackConnector::transpose(a, n, lda);
        // call the fortran routine
        cheev_(&jobz, &uplo, &n, aux, &lda, w, work, &lwork, rwork, &info);
        // Transpose the fortran-form real-complex array to the complex matrix.
        LapackConnector::transpose(aux, a, n, lda);
        // free the memory.
        delete[] aux;
    }

    static inline
    void heev(const char &jobz, const char &uplo, const int &n,
              std::complex<double> *a, const int &lda, double *w,
              std::complex<double> *work, const int &lwork, double *rwork, int &info)
    {
        complex<double> *aux = LapackConnector::transpose(a, n, lda);
        // call the fortran routine
        zheev_(&jobz, &uplo, &n, aux, &lda, w, work, &lwork, rwork, &info);
        // Transpose the fortran-form real-complex array to the complex matrix.
        LapackConnector::transpose(aux, a, n, lda);
        // free the memory.
        delete[] aux;
    }

    // eigenvector of hermitian matrix
    static inline
    void heev_f(const char &jobz, const char &uplo, const int &n,
                std::complex<float> *a, const int &lda, float *w,
                std::complex<float> *work, const int &lwork, float *rwork, int &info)
    {
        cheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    }

    static inline
    void heev_f(const char &jobz, const char &uplo, const int &n,
                std::complex<double> *a, const int &lda, double *w,
                std::complex<double> *work, const int &lwork, double *rwork, int &info)
    {
        zheev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info);
    }

        // Peize Lin add 2018-06-12
	// out = ||x||_2
	// static inline
	// float nrm2( const int n, const float *X, const int incX )
	// {
		// return snrm2_( &n, X, &incX );
	// }
	//static inline
	// double nrm2( const int n, const double *X, const int incX )
	// {
		// return dnrm2_( &n, X, &incX );
	// }
	// static inline
	// double nrm2( const int n, const complex<double> *X, const int incX )
	// {
		// return dznrm2_( &n, X, &incX );
	// }
	
	static inline
	void copy(const long n, const double *a, const int incx, double *b, const int incy)
	{
		dcopy_(&n, a, &incx, b, &incy);
	}
	static inline
	void copy(const long n, const complex<double> *a, const int incx, complex<double> *b, const int incy)
	{
		zcopy_(&n, a, &incx, b, &incy);
	}	

	// Peize Lin add 2019-04-14
	// if trans=='N':	C = a * A * A.H + b * C
	// if trans=='C':	C = a * A.H * A + b * C
	static inline
	void zherk(const char uplo, const char trans, const int n, const int k, 
		const double alpha, const complex<double> *A, const int lda, 
		const double beta, complex<double> *C, const int ldc)
	{
		const char uplo_changed = change_uplo(uplo);
		const char trans_changed = change_trans_NC(trans);
		zherk_(&uplo_changed, &trans_changed, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
	}
};

}
#endif  // LAPACKCONNECTOR_HPP
