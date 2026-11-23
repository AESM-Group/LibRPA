/* matrix.cpp file */
#include <ios>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <limits>

#include "matrix.h"
#include "lapack_connector.h"
#include "../io/global_io.h"

namespace librpa_int {

//*********************************************************
// The init() function is the main initialization routine.
// Sets up sizes and allocates memory for matrix class.
// All constructors call init()
// ********************************************************

//int matrix::mCount = 0;
const double matrix::EQUAL_THRES = 1e-9;

void matrixAlloc()
{
	//WARNING_QUIT("matrix","Allocation error for Matrix");
}

matrix::matrix( const int nrows, const int ncols, const bool flag_zero )
	:nr(nrows),
	 nc(ncols),
	 c(nullptr)
{
	if( nr && nc )
	{
		auto handler_old = std::set_new_handler(matrixAlloc);
		c = new double[nr*nc];
        size = nr * nc;
		std::set_new_handler(handler_old);
		if(flag_zero)	this->zero_out();
	}
}

matrix::matrix( const matrix &m_in )
	:nr(m_in.nr),
	 nc(m_in.nc),
	 c(nullptr)
{
	if( nr && nc )
	{
		auto handler_old = std::set_new_handler(matrixAlloc);
		c = new double[nr*nc];
		std::set_new_handler(handler_old);
		memcpy( c, m_in.c, nr*nc*sizeof(double) );
        size = m_in.size;
	}
}

// Peize Lin add 2016-08-05
matrix::matrix( matrix && m_in )
	:nr(m_in.nr),
	 nc(m_in.nc),
	 size(m_in.size)
{
	c = m_in.c;
	m_in.nr = m_in.nc = m_in.size = 0;
	m_in.c = nullptr;
}

// Peize Lin change 2018-07-02
matrix& matrix::operator=( const matrix & m_in )
{
    if(this == &m_in) return *this;
	this->create( m_in.nr, m_in.nc, false );
	memcpy( c, m_in.c, nr*nc*sizeof(double) );
	return *this;
}

// Peize Lin add 2016-08-05
matrix& matrix::operator=( matrix && m_in )
{
    if(this == &m_in) return *this;
	nr = m_in.nr;		nc = m_in.nc;
    size = m_in.size;
	if(c)	delete[] c;
	c = m_in.c;
	m_in.nr = m_in.nc = m_in.size = 0;
	m_in.c = nullptr;
	return *this;
}

/*
double & matrix::operator()(const int ir,const int ic)
{
	assert(ir>=0);	assert(ir<nr);	assert(ic>=0);	assert(ic<nc);
	return c[ir*nc+ic];
}

const double & matrix::operator()(const int ir,const int ic) const
{
	assert(ir>=0);	assert(ir<nr);	assert(ic>=0);	assert(ic<nc);
	return c[ir*nc+ic];
}
*/

//*************
//
// destructor
//
//*************
matrix::~matrix()
{
	if(c)					// Peize Lin add 2016-08-05
	{
		delete [] c;
		c = nullptr;
	}
}

//******************************
// reallocate memory for matrix
//******************************
// Peize Lin change 2018-07-29
// minyez change 2022-05-05
void matrix::create( const int nrow, const int ncol, const bool flag_zero )
{
    size = as_size(nrow) * as_size(ncol);
	if( size )
	{
		if(c)
		{
			if( size != as_size(nr*nc) )
			{
				delete[] c;
				auto handler_old = std::set_new_handler(matrixAlloc);			
				c = new double[size];
				std::set_new_handler(handler_old);
			}
		}
		else
		{
			auto handler_old = std::set_new_handler(matrixAlloc);
			c = new double[size];
			std::set_new_handler(handler_old);
		}			
			
		nr = nrow;
		nc = ncol;
		if(flag_zero)	zero_out();				// Peize Lin change 2018-03-12
	}
	else
	{
		if(c)	delete[] c;
		c = nullptr;
		nr = nrow;
		nc = ncol;
	}
}

/* Adding matrices, as a friend */
matrix operator+(const matrix &m1, const matrix &m2)
{
	// assert(m1.nr == m2.nr);
	// assert(m2.nc == m2.nc);

	matrix tm(m1);
	for (size_t i = 0; i < m1.size; i++) 
		tm.c[i] += m2.c[i];
	return tm;
}

/* Subtracting matrices, as a friend */
matrix operator-(const matrix &m1, const matrix &m2)
{
	// assert(m1.nr == m2.nr);
	// assert(m2.nc == m2.nc);

	matrix tm(m1);
	for(size_t i = 0; i < m1.size; i++) 
		tm.c[i] -= m2.c[i];
	return tm;
}

//***************************************
//
// Multiplying matrices, as a friend 
//
// *************************************
matrix operator*(const matrix &m1, const matrix &m2)
{
	// fixed bug 2010-01-26
	assert(m1.nc == m2.nr);
	
    // allocate the result and zero it out
    matrix mprod( m1.nr, m2.nc, false );

    // TODO: direct multiply may be faster in small size matrix?
    // do the multiply and return
   // for (int i = 0;i < m1.nr;i++)
       // for (int k = 0;k < m1.nc;k++)
           // for (int j = 0;j < m2.nc;j++)
               // mprod(i, j) += m1(i, k) * m2(k, j);
	
	//Peize Lin accelerate 2017-10-27
	LapackConnector::gemm(
		'N', 'N', 
		m1.nr, m2.nc, m1.nc,
		1, m1.c, m1.nc, m2.c, m2.nc, 
		0, mprod.c, mprod.nc);
	return mprod;
}

/* Scale a matrix */
matrix operator*(const double &s, const matrix &m)
{
	matrix sm(m);
	for (size_t i = 0; i < m.size; i++) 
		sm.c[i] *= s;
	return sm;
}

/* matrix * double */
matrix operator*(const matrix &m,const double &s)
{
	matrix sm(m);
	for (size_t i = 0; i < m.size; i++)
		sm.c[i] *= s;
	return sm;
}

/* Scale a matrix in place */
void matrix::operator*=(const double &s)
{
	for (size_t i = 0; i < size; i++) 
		c[i] *= s;
}

// minyez added 2022-05-06, shift all elements up by s
void matrix::operator+=(const double &s)
{
    for( size_t i = 0; i < size; ++i )
        c[i] += s;
}

// minyez added 2022-05-06, shift all elements down by s
void matrix::operator-=(const double &s)
{
    for( size_t i = 0; i < size; ++i )
        c[i] -= s;
}

/* Accumulate to a matrix in place */
void matrix::operator+=(const matrix & m)
{
	// assert( nr==m.nr );
	// assert( nc==m.nc );
	const double * const c_in = m.c;
	for( size_t i = 0; i < size; ++i ) 
		c[i] += c_in[i];
}

// minyez added 2022-05-06
matrix matrix::operator+(double s) const
{
	matrix newm = *this;
    newm += s;
    return newm;
}

// minyez added 2022-05-06
matrix matrix::operator-(double s) const
{
	matrix newm = *this;
    newm -= s;
    return newm;
}

/* decumulate to a matrix in place */
void matrix::operator-=(const matrix & m)
{
	assert( nr==m.nr );
	assert( nc==m.nc );
	const double * const c_in = m.c;
	for( size_t i = 0; i < size; ++i ) 
		c[i] -= c_in[i];
}

// minyez added 2022-05-06
bool matrix::operator==(const matrix & m)
{
    if ( size != m.size) return false;
    for ( size_t i = 0; i != size; i++)
        if ( std::abs( c[i] - m.c[i] ) > matrix::EQUAL_THRES ) return false;
    return true;
}

/* zero out the matrix */
void matrix::zero_out(void)
{
	for(size_t i = 0; i < size; i++)
		c[i] = 0.0;
}


matrix transpose(const matrix &m)
{
	matrix tm( m.nc, m.nr, false );
	for (int i = 0;i < m.nr;i++)
		for (int j = 0;j < m.nc;j++)
			tm(j, i) = m(i, j);
	return tm;
}

double matrix::trace_on(void) const
{
    assert(nr == nc);
    int inch = nc + 1;
    double tr = 0.0;
    for (size_t i = 0; i < size; i += inch)
    {
        tr += c[i];
    }
    return tr;
}

void matrix::get_extreme_eigen_values(double &ev_lower, double &ev_upper)const
{
    double *a = new double[nr];
    double *b = new double[nr];
    for (int i = 0; i < nr; ++i)
    {
        double sum = 0.0;
        for(int j = 0; j < nc; ++j)
        {
            sum += fabs(c[i * nc + j]);
        }
        sum -= fabs(c[i * nc + i]);
        a[i] = c[i * nc + i] - sum;
        b[i] = c[i * nc + i] + sum;
    }

    ev_lower = a[0];
    ev_upper = b[0];

    for (int i = 1; i < nr; ++i)
    {
        if (a[i] < ev_lower) ev_lower = a[i];
        if (b[i] > ev_upper) ev_upper = b[i];
    }
    delete[] a;
    delete[] b;
}

// Peize Lin add 2017-05-27
void matrix::reshape( const double nr_new, const double nc_new )
{
	assert( nr*nc == nr_new*nc_new );
	nr=nr_new;
	nc=nc_new;
}

double trace_on(const matrix &A, const matrix &B)
{
    double tr = 0.0;
    for (int i = 0; i < A.nr; ++i)
        for (int k = 0; k < A.nc; ++k)
            tr += A(i,k) * B(k, i);
    return tr;
}

double mdot(const matrix &A, const matrix &B)
{
    assert (A.nr == B.nr);
    assert (A.nc == B.nc);

    double sum = 0.0;
    for (size_t i = 0; i < A.size; ++i)
        sum += A.c[i] * B.c[i];
    return sum;
}

// Peize Lin add 2016-09-08
std::ostream & operator<<( std::ostream & os, const matrix & m )
{
	for( int ir=0; ir!=m.nr; ++ir )
	{
		for( int ic=0; ic!=m.nc; ++ic )
			os<<m(ir,ic)<<"\t";
		os<<std::endl;
	}	
	return os;
}

// Peize Lin add 2016-09-08
double matrix::max() const
{
	double value = std::numeric_limits<double>::min();
	for( size_t i=0; i<size; ++i )
		value = std::max( value, c[i] );
	return value;
}

// minyez add 2022-05-05
double matrix::max(int &ir, int &ic) const
{
    double value = std::numeric_limits<double>::min();
    int pos = 0;
    for( size_t i=0; i<size; ++i )
        if (value > c[i])
        {
            value = c[i];
            pos = i;
        }
    ir = pos / nc;
    ic = pos % nc;
    return value;
}

// Peize Lin add 2016-09-08
double matrix::min() const
{
	double value = std::numeric_limits<double>::max();
	for( size_t i=0; i<size; ++i )
		value = std::min( value, c[i] );
	return value;
}

// Peize Lin add 2018-07-02
double matrix::absmax() const
{
	double value = 0;
	for( size_t i=0; i<size; ++i )
		value = std::max( value, std::abs(c[i]) );
	return value;
}

// minyez add 2022-05-05
unsigned matrix::count_absle(double thres) const
{
    unsigned count = 0;
    for( size_t i=0; i<size; ++i )
        if (std::abs(c[i]) <= thres) count++;
    return count;
}

unsigned matrix::count_absge(double thres) const
{
    unsigned count = 0;
    for( size_t i=0; i<size; ++i )
        if (std::abs(c[i]) >= thres) count++;
    return count;
}

// minyez add 2022-05-05
matrix power_symat(matrix &mat, double power, double threshold)
{
    assert (mat.nr == mat.nc);
    matrix pmat = matrix(mat.nr, mat.nc);
    const char jobz = 'V';
    const char uplo = 'U';

    int nb = LapackConnector::ilaenv(1, "dsyev", "VU", mat.nc, 1, 1, 1);
    int lwork = mat.nc * (nb+2);
    int info = 0;
    std::vector<double> w(mat.nc);
    std::vector<double> work(lwork);
    LapackConnector::dsyev(jobz, uplo, mat.nc, mat.c, mat.nc,
                           w.data(), work.data(), lwork, info);
    bool is_int_power = fabs(power - int(power)) < 1e-8;
    for ( int i = 0; i != mat.nc; i++ )
    {
        if (w[i] < 0 && !is_int_power)
            librpa_int::global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning! negative eigenvalue with non-integer power: # %d ev = %f , pow = %f", i, w[i], power);
        if (fabs(w[i]) < 1e-10 && power < 0)
            librpa_int::global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning! nearly-zero eigenvalue with negative power: # %d ev = %f , pow = %f", i, w[i], power);
        if (w[i] < threshold)
            w[i] = 0;
        else
            w[i] = pow(w[i], power);
    }
    matrix evtrans = transpose(mat);
    for ( int i = 0; i != mat.nr; i++ )
        for ( int j = 0; j != mat.nc; j++ )
            evtrans.c[i*mat.nc+j] *= w[i];

    return mat * evtrans;

}

// minyez copied from Shi Rong's cal_periodic_chi0.cpp
void print_matrix(const char *desc, const matrix &mat)
{
    int nr = mat.nr;
    int nc = mat.nc;
    librpa_int::global::lib_printf("\n %s\n", desc);
    librpa_int::global::lib_printf("nr = %d, nc = %d\n", nr, nc);
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc; j++)
            librpa_int::global::lib_printf(" %9.5e", mat.c[i * nc + j]);
        librpa_int::global::lib_printf("\n");
    }
}

void print_matrix(const char *desc, const matrix &mat, std::ostream &os, bool use_scientific)
{
    using std::fixed;
    using std::scientific;
    using std::endl;
    using std::showpoint;
    using std::setw;
    using std::setprecision;
    using std::endl;

    int nr = mat.nr;
    int nc = mat.nc;
    int w = 22;
    int prec = 15;
    auto format = fixed;
    if (use_scientific) format = scientific;
    os << "#" << desc << endl;
    os << "# r " << nr << " " << nc << endl;
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc - 1; j++)
            os << setw(w) << showpoint << format << setprecision(prec) << mat.c[i * nc + j] << " ";
        os << setw(w) << showpoint << format << setprecision(prec) << mat.c[i * nc + nc - 1] << endl;
    }
}
// double matrix::norm() const
// {
	// return LapackConnector::nrm2(nr*nc,c,1);
// }

void print_matrix_mm(const matrix &mat, std::ostream &os, const std::string &comment, double threshold, bool row_first)
{
    using std::scientific;
    using std::showpoint;
    using std::setprecision;
    using std::endl;
    int nr = mat.nr;
    int nc = mat.nc;
    int prec = 15;
    size_t nnz = 0;
    auto format = scientific;
    os << "%%MatrixMarket matrix coordinate real general\n";
    os << "%" << endl;
    if (!comment.empty())
    {
        os << "% " << comment << "\n" << "%\n";
    }
    // count non-zero values first
    for (size_t i = 0; i < mat.size; i++)
    {
        auto v = mat.c[i];
        if ( fabs(v) > threshold )
            nnz++;
    }

    os << nr << " " << nc << " " << nnz << "\n";

    if (row_first)
    {
        for (int j = 0; j < nc; j++)
        {
            for (int i = 0; i < nr; i++)
            {
                auto v = mat.c[i*nc+j];
                if ( fabs(v) > threshold )
                    os << i + 1 << " " << j + 1 << " " << showpoint << format << setprecision(prec) << v << "\n";
            }
        }
    }
    else
    {
        for (int i = 0; i < nr; i++)
        {
            for (int j = 0; j < nc; j++)
            {
                auto v = mat.c[i*nc+j];
                if ( fabs(v) > threshold )
                    os << i + 1 << " " << j + 1 << " " << showpoint << format << setprecision(prec) << v << "\n";
            }
        }
    }
}

void print_matrix_mm(const matrix &mat, const string &fn, const std::string &comment, double threshold, bool row_first)
{
    ofstream fs;
    fs.open(fn);
    print_matrix_mm(mat, fs, comment, threshold, row_first);
    fs.close();
}

}
