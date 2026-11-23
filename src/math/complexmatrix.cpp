//==========================================================
// AUTHOR : Lixin He, Mohan Chen
// LAST UPDATE : 2009-03-23 modify "=" operator
//==========================================================

#include "complexmatrix.h"

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <ostream>

#include "../io/global_io.h"

#include "lapack_connector.h"

namespace librpa_int {

// constructor with sizes
ComplexMatrix::ComplexMatrix(const int nrows, const int ncols, const bool flag_zero)
    : nr(nrows), nc(ncols), size(nrows * ncols), c(nullptr)
{
    if (size)
    {
        c = new complex<double>[size];
        if (flag_zero) zero_out();
    }
}

// zero out the ComplexMatrix
void ComplexMatrix::zero_out(void)
{
    for (int i = 0; i < size; i++) c[i] = complex<double>(0.0, 0.0);
}

/*
void need_more_memory()
{
    cout << "\n Sorry to crash... but the running need more momory! Exit." << endl;
    exit(0);
}
*/

// Copy constructor
ComplexMatrix::ComplexMatrix(const ComplexMatrix &m1)
    : nr(m1.nr), nc(m1.nc), size(m1.size), c(nullptr)
{
    if (size)
    {
        c = new complex<double>[size];
        memcpy(c, m1.c, size * sizeof(complex<double>));
    }
}

// Peize Lin add 2016-08-05
ComplexMatrix::ComplexMatrix(ComplexMatrix &&m1) : nr(m1.nr), nc(m1.nc), size(m1.size), c(m1.c)
{
    m1.nr = m1.nc = m1.size = 0;
    m1.c = nullptr;
}

// Peize Lin add 2017-03-29
ComplexMatrix::ComplexMatrix(const matrix &m) : nr(m.nr), nc(m.nc), size(m.nr * m.nc), c(nullptr)
{
    if (size)
    {
        c = new complex<double>[size];
        for (int i = 0; i < size; ++i) c[i] = m.c[i];
    }
}

// deconstructor
ComplexMatrix::~ComplexMatrix()
{
    if (c)
    {
        delete[] c;
        c = nullptr;
    }
}

// reallocate memory for Complex Matrix
void ComplexMatrix::create(const int nr_in, const int nc_in, const bool flag_zero)
{
    if (nr_in && nc_in)
    {
        if (c)
        {
            const int size_in = nr_in * nc_in;
            if (size_in != nr * nc)
            {
                delete[] c;
                c = new complex<double>[size_in];
            }
        }
        else
        {
            c = new complex<double>[nr_in * nc_in];
        }

        nr = nr_in;
        nc = nc_in;
        size = nr * nc;
        if (flag_zero) zero_out();
    }
    else
    {
        if (c) delete[] c;
        c = nullptr;
        nr = nr_in;
        nc = nc_in;
        size = nr * nc;
    }
}

void ComplexMatrix::set_as_identity_matrix()
{
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            if (i == j)
                c[nc * i + j] = complex<double>(1.0, 0.0);
            else
                c[nc * i + j] = complex<double>(0.0, 0.0);
        }
    }
    return;
}

double ComplexMatrix::get_max_real() const
{
    // Not allocated
    if (size == 0)
        return 0.0;
    double rv = c[0].real();
    for (int i = 1; i != size; i++)
        if (c[i].real() > rv) rv = c[i].real();
    return rv;
}
double ComplexMatrix::get_max_real(int &ir, int &ic) const
{
    // Not allocated
    if (size == 0)
        return 0.0;
    double rv = c[0].real();
    int pos = 0;
    for (int i = 1; i != size; i++)
    {
        if (c[i].real() > rv)
        {
            rv = c[i].real();
            pos = i;
        }
    }
    ir = pos / nc;
    ic = pos % nc;
    return rv;
}

double ComplexMatrix::get_max_imag() const
{
    // Not allocated
    if (size == 0)
        return 0.0;
    double iv = c[0].imag();
    for (int i = 1; i != this->size; i++)
        if (c[i].imag() > iv) iv = c[i].imag();
    return iv;
}

double ComplexMatrix::get_max_abs() const
{
    // Not allocated
    if (size == 0)
        return 0.0;
    double iv = std::abs(this->c[0]);
    for (int i = 1; i != this->size; i++)
    {
        iv = std::max(iv, std::abs(this->c[i]));
    }
    return iv;
}

double ComplexMatrix::get_max_abs_imag() const
{
    // Not allocated
    if (size == 0)
        return 0.0;
    double iv = std::abs(this->c[0].imag());
    for (int i = 1; i != this->size; i++)
    {
        iv = std::max(iv, std::abs(this->c[i].imag()));
    }
    return iv;
}

double ComplexMatrix::get_max_abs_offdiag() const
{
    double maxabs = 0.;
    for (int i = 0; i != nr; i++)
        for (int j = 0; j != nc; j++)
        {
            if (i == j) continue;
            maxabs = std::max(std::abs(c[i * nr + j]), maxabs);
        }
    return maxabs;
}

bool ComplexMatrix::is_diagonal(const double &thres) const
{
    for (int i = 0; i != nr; i++)
        for (int j = 0; j != nc; j++)
        {
            if (i == j) continue;
            if (std::abs(c[i * nr + j]) > thres) return false;
        }
    return true;
}

// Adding matrices, as a friend
ComplexMatrix operator+(const ComplexMatrix &m1, const ComplexMatrix &m2)
{
    assert(m1.nr == m2.nr);
    assert(m2.nc == m2.nc);

    ComplexMatrix tm(m1);
    tm += m2;
    return tm;
}

ComplexMatrix operator-(const ComplexMatrix &m, const complex<double> &s)  // minyez add 2022-06-06
{
    ComplexMatrix tm(m);
    tm += s;
    return tm;
}

// Subtracting matrices, as a friend
ComplexMatrix operator-(const ComplexMatrix &m1, const ComplexMatrix &m2)
{
    assert(m1.nr == m2.nr);
    assert(m2.nc == m2.nc);

    ComplexMatrix tm(m1);
    tm -= m2;
    return tm;
}

ComplexMatrix operator-(const complex<double> &s, const ComplexMatrix &m)
{
    ComplexMatrix tm(m);
    tm *= -1;
    tm += s;
    return tm;
}

// Multiplying matrices, as a friend
// mprod = m1 * m2
ComplexMatrix operator*(const ComplexMatrix &m1, const ComplexMatrix &m2)
{
    assert(m1.nc == m2.nr);
    ComplexMatrix mprod(m1.nr, m2.nc);

    complex<double> z;
    //	for (int i = 0;i < m1.nr;i++)
    //	{
    //		for (int j = 0;j < m2.nc;j++)
    //		{
    //			z = complex<double>(0,0);
    //			for (int k = 0;k < m1.nc;k++)
    //			{
    //				z += m1(i, k) * m2(k, j);
    //			}
    //			mprod(i, j) = z;
    //		}
    //	}
    // Peize Lin accelerate 2017-10-27
    LapackConnector::gemm('N', 'N', m1.nr, m2.nc, m1.nc, 1, m1.c, m1.nc, m2.c, m2.nc, 0, mprod.c,
                          mprod.nc);
    return mprod;
}

// Scale a ComplexMatrix
ComplexMatrix operator*(const complex<double> &c, const ComplexMatrix &m)
{
    ComplexMatrix sm(m);
    for (int i = 0; i < m.size; i++) sm.c[i] *= c;
    return sm;
}

// ComplexMatrix scalar
ComplexMatrix operator*(const ComplexMatrix &m, const complex<double> &c)
{
    ComplexMatrix sm(m);
    for (int i = 0; i < m.size; i++) sm.c[i] *= c;
    return sm;
}

ComplexMatrix operator*(const double &r, const ComplexMatrix &m)
{
    ComplexMatrix sm(m);
    for (int i = 0; i < m.size; i++) sm.c[i] *= r;
    return sm;
}

ComplexMatrix operator*(const ComplexMatrix &m, const double &r)
{
    ComplexMatrix sm(m);
    for (int i = 0; i < m.size; i++) sm.c[i] *= r;
    return sm;
}

ComplexMatrix &ComplexMatrix::operator=(const ComplexMatrix &m)
{
    this->create(m.nr, m.nc, false);
    memcpy(c, m.c, size * sizeof(complex<double>));
    return *this;
}

// Peize Lin add 2016-08-05
ComplexMatrix &ComplexMatrix::operator=(ComplexMatrix &&m)
{
    nr = m.nr;
    nc = m.nc;
    size = m.size;
    if (c) delete[] c;
    c = m.c;
    m.nr = m.nc = m.size = 0;
    m.c = nullptr;
    return *this;
}

ComplexMatrix &ComplexMatrix::operator*=(const complex<double> &s)
{
    for (int i = 0; i < this->size; i++) c[i] *= s;
    return *this;
}

// Accumulate to a ComplexMatrix in place
ComplexMatrix &ComplexMatrix::operator+=(const ComplexMatrix &m)
{
    assert(this->nr == m.nr);
    assert(this->nc == m.nc);
    for (int i = 0; i < size; i++) this->c[i] += m.c[i];
    return *this;
}

ComplexMatrix &ComplexMatrix::operator+=(const complex<double> &s)
{
    for (int i = 0; i < size; i++) this->c[i] += s;
    return *this;
}

// decumulate to a ComplexMatrix in place
ComplexMatrix &ComplexMatrix::operator-=(const ComplexMatrix &m)
{
    for (int i = 0; i < size; i++) this->c[i] -= m.c[i];
    return *this;
}

// Peize Lin add 2017-03-29
matrix ComplexMatrix::real() const
{
    matrix m(nr, nc, false);
    for (int i = 0; i < this->size; ++i) m.c[i] = c[i].real();
    return m;
}

matrix ComplexMatrix::imag() const
{
    matrix m(nr, nc, false);
    for (int i = 0; i < this->size; ++i) m.c[i] = c[i].imag();
    return m;
}

// Returns trace of ComplexMatrix
complex<double> trace(const ComplexMatrix &m)
{
    complex<double> tr = complex<double>(0, 0);
    assert(m.nr == m.nc);
    for (int i = 0; i < m.nr; i++) tr += m(i, i);
    return tr;
}

// Do mout += s*min
void scale_accumulate(const complex<double> &s, const ComplexMatrix &min, ComplexMatrix &mout)
{
    assert(min.nr == mout.nr);
    assert(min.nc == mout.nc);
    for (int j = 0; j < min.size; j++)
    {
        mout.c[j] += s * min.c[j];
    }
    return;
}

// Do mout[i] += s*min[i]
void scale_accumulate(const int &nmat, const complex<double> &s, ComplexMatrix **min,
                      ComplexMatrix **mout)
{
    assert(nmat >= 0);
    for (int i = 0; i < nmat; i++)
    {
        scale_accumulate(s, *min[i], *mout[i]);
    }
    return;
}

// Do mout = s1*m1 + s2*m2
void scaled_sum(const complex<double> &s1, const ComplexMatrix &m1, const complex<double> &s2,
                const ComplexMatrix &m2, ComplexMatrix &mout)
{
    assert(m1.nr == m2.nr);
    assert(m1.nr == mout.nr);
    assert(m1.nc == m2.nc);
    assert(m1.nc == mout.nc);

    for (int i = 0; i < m1.size; i++)
    {
        mout.c[i] = s1 * m1.c[i] + s2 * m2.c[i];
    }
    return;
}

// Does mout[i] = s1*m1[i] + s2*m2[i]
void scaled_sum(const int &nmat, const complex<double> &s1, ComplexMatrix **m1,
                const complex<double> &s2, ComplexMatrix **m2, ComplexMatrix **mout)
{
    assert(nmat > 0);
    for (int i = 0; i < nmat; i++)
    {
        scaled_sum(s1, *m1[i], s2, *m2[i], *mout[i]);
    }
    return;
}

double abs2_row(const ComplexMatrix &m, const int ir)
{
    double r = 0.0;
    complex<double> z;
    for (int ic = 0; ic < m.nc; ic++)
    {
        z = m.c[m.nc * ir + ic];
        r += z.real() * z.real() + z.imag() * z.imag();
    }
    return r;
}

double abs2_column(const ComplexMatrix &m, const int ic)
{
    double r = 0.0;
    complex<double> z;
    for (int ir = 0; ir < m.nr; ir++)
    {
        z = m.c[m.nc * ir + ic];
        r += z.real() * z.real() + z.imag() * z.imag();
    }
    return r;
}

// returns absolute square magnitude of sum of all ComplexMatrix elements
double abs2(const ComplexMatrix &m)
{
    double r = 0.0;
    complex<double> z;

    for (int i = 0; i < m.size; i++)
    {
        z = m.c[i];
        r += z.real() * z.real() + z.imag() * z.imag();
    }
    return r;
}

// Same for an array of matrices
double abs2(const int nmat, ComplexMatrix **m)
{
    double r = 0.0;
    for (int i = 0; i < nmat; i++)
    {
        r += abs2(*m[i]);
    }
    return r;
}

ComplexMatrix transpose(const ComplexMatrix &m, const bool &conjugate)
{
    ComplexMatrix tm(m.nc, m.nr, false);
    if (conjugate)
        for (int i = 0; i < m.nr; i++)
            for (int j = 0; j < m.nc; j++) tm(j, i) = conj(m(i, j));
    else
        for (int i = 0; i < m.nr; i++)
            for (int j = 0; j < m.nc; j++) tm(j, i) = m(i, j);
    return tm;
}

ComplexMatrix conj(const ComplexMatrix &m)
{
    ComplexMatrix cm(m.nr, m.nc, false);
    for (int i = 0; i != m.size; ++i) cm.c[i] = conj(m.c[i]);
    return cm;
}

ComplexMatrix power_hemat(ComplexMatrix &cmat, double power, bool keep_ev, bool filter_original,
                          double threshold)
{
    assert(cmat.nr == cmat.nc);
    const char jobz = 'V';
    const char uplo = 'U';

    int nb = LapackConnector::ilaenv(1, "zheev", "VU", cmat.nc, -1, -1, -1);
    int lwork = cmat.nc * (nb + 1);
    int info = 0;
    std::vector<double> w(cmat.nc), wpow(cmat.nc), rwork(3 * cmat.nc - 2);
    std::vector<complex<double>> work(lwork);
    // make the final eigenvalues in the descending order
    cmat *= -1.0;
    LapackConnector::zheev(jobz, uplo, cmat.nc, cmat, cmat.nc, w.data(), work.data(), lwork,
                           rwork.data(), &info);
    bool is_int_power = fabs(power - int(power)) < 1e-8;
    for (int i = 0; i != cmat.nc; i++)
    {
        w[i] = -w[i];
        // lib_printf("%3d%22.12f\n", i+1, w[i]);
        if (w[i] < 0 && w[i] > threshold && !is_int_power)
	{
	    librpa_int::global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning! kept negative eigenvalue with non-integer power: # %d ev = %f , pow = %f\n", i, w[i], power);
	}
        if (fabs(w[i]) < 1e-10 && power < 0)
	{
            librpa_int::global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning! nearly-zero eigenvalue with negative power: # %d ev = %f , pow = %f\n", i, w[i], power);
	}
        if (w[i] < threshold)
        {
            wpow[i] = 0;
            if (filter_original) w[i] = 0;
        }
        else
        {
            wpow[i] = w[i];
        }
        wpow[i] = pow(wpow[i], power);
    }
    ComplexMatrix evconj = transpose(cmat, true);
    for (int i = 0; i != cmat.nr; i++)
        for (int j = 0; j != cmat.nc; j++) evconj.c[i * cmat.nc + j] *= wpow[i];
    auto pmat = cmat * evconj;

    // recover the original matrix here
    // if eigenvectors are requested, return now
    if (keep_ev) return pmat;
    evconj = transpose(cmat, true);
    // NOTE: may need a ComplexMatrix object to back up the matrix when the original matrix is
    // required to save a second matmul
    for (int i = 0; i != cmat.nr; i++)
        for (int j = 0; j != cmat.nc; j++) evconj.c[i * cmat.nc + j] *= w[i];
    cmat = cmat * evconj;
    return pmat;
}

void power_hemat_onsite(ComplexMatrix &cmat, double power, double threshold)
{
    assert(cmat.nr == cmat.nc);
    const char jobz = 'V';
    const char uplo = 'U';

    int nb = LapackConnector::ilaenv(1, "zheev", "VU", cmat.nc, -1, -1, -1);
    int lwork = cmat.nc * (nb + 1);
    int info = 0;
    double w[cmat.nc];
    double rwork[3 * cmat.nc - 2];
    complex<double> work[lwork];
    LapackConnector::zheev(jobz, uplo, cmat.nc, cmat, cmat.nc, w, work, lwork, rwork, &info);
    bool is_int_power = fabs(power - int(power)) < 1e-8;
    for (int i = 0; i != cmat.nc; i++)
    {
        if (w[i] < 0 && w[i] > threshold && !is_int_power)
            librpa_int::global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning! kept negative eigenvalue with non-integer power: # %d ev = %f , pow = %f\n", i, w[i], power);
        if (fabs(w[i]) < 1e-10 && power < 0)
            librpa_int::global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning! nearly-zero eigenvalue with negative power: # %d ev = %f , pow = %f\n", i, w[i], power);
        if (w[i] < threshold)
            w[i] = 0;
        else
            w[i] = pow(w[i], power);
    }
    ComplexMatrix evconj = transpose(cmat, true);
    for (int i = 0; i != cmat.nr; i++)
        for (int j = 0; j != cmat.nc; j++) evconj.c[i * cmat.nc + j] *= w[i];
    auto pmat = cmat * evconj;

    // parse back to the original matrix here
    for (int i = 0; i != cmat.nr; i++)
        for (int j = 0; j != cmat.nc; j++) cmat.c[i * cmat.nc + j] = pmat.c[i * cmat.nc + j];
}

void print_complex_matrix(const char *desc, const ComplexMatrix &mat)
{
    int nr = mat.nr;
    int nc = mat.nc;
    librpa_int::global::lib_printf("\n %s\n", desc);
    librpa_int::global::lib_printf("nr = %d, nc = %d\n", nr, nc);
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc; j++)
            librpa_int::global::lib_printf("%10.6e,%9.6e ", mat.c[i * nc + j].real(), mat.c[i * nc + j].imag());
        librpa_int::global::lib_printf("\n");
    }
}

std::ostream & operator<<( std::ostream & os, const ComplexMatrix & m)
{
    print_complex_matrix("", m, os, true);
    return os;
}

void print_complex_matrix(const char *desc, const ComplexMatrix &mat, std::ostream &os, bool use_scientific)
{
    using std::fixed;
    using std::scientific;
    using std::endl;
    using std::showpoint;
    using std::setw;
    using std::setprecision;

    int nr = mat.nr;
    int nc = mat.nc;
    int w = 22;
    int prec = 15;
    auto format = fixed;
    if (use_scientific)
        format = scientific;
    os << "#" << desc << endl;
    // c for complex
    os << "# c " << nr << " " << nc << endl;
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc - 1; j++)
            os << setw(w) << showpoint << format << setprecision(prec) << mat.c[i * nc + j].real() << " " << setw(w) << showpoint << format << setprecision(prec) << mat.c[i * nc + j].imag() << " ";
        os << setw(w) << showpoint << format << setprecision(prec) << mat.c[i * nc + nc - 1].real() << " " << setw(w) << showpoint << format << setprecision(prec) << mat.c[i * nc + nc - 1].imag() << endl;
    }
}

void print_complex_matrix_mm(const ComplexMatrix &mat, ofstream &fs, double threshold,
                             bool row_first)
{
    using std::fixed;
    using std::scientific;
    using std::endl;
    using std::showpoint;
    using std::setw;
    using std::setprecision;

    int nr = mat.nr;
    int nc = mat.nc;
    int prec = 15;
    size_t nnz = 0;
    auto format = scientific;
    fs << "%%MatrixMarket matrix coordinate complex general" << endl;
    fs << "%" << endl;
    // count non-zero values first
    for (int i = 0; i < mat.size; i++)
    {
        auto v = mat.c[i];
        if (fabs(v.real()) > threshold || fabs(v.imag()) > threshold) nnz++;
    }

    fs << nr << " " << nc << " " << nnz << endl;

    if (row_first)
    {
        for (int j = 0; j < nc; j++)
        {
            for (int i = 0; i < nr; i++)
            {
                auto v = mat.c[i * nc + j];
                if (fabs(v.real()) > threshold || fabs(v.imag()) > threshold)
                    fs << i + 1 << " " << j + 1 << " " << showpoint << format << setprecision(prec)
                       << v.real() << " " << showpoint << format << setprecision(prec) << v.imag()
                       << "\n";
            }
        }
    }
    else
    {
        for (int i = 0; i < nr; i++)
        {
            for (int j = 0; j < nc; j++)
            {
                auto v = mat.c[i * nc + j];
                if (fabs(v.real()) > threshold || fabs(v.imag()) > threshold)
                    fs << i + 1 << " " << j + 1 << " " << showpoint << format << setprecision(prec)
                       << v.real() << " " << showpoint << format << setprecision(prec) << v.imag()
                       << "\n";
            }
        }
    }
}

void print_real_matrix_mm(const ComplexMatrix &mat, ofstream &fs, double threshold, bool row_first)
{
    using std::setprecision;
    using std::scientific;
    using std::endl;
    using std::showpoint;

    int nr = mat.nr;
    int nc = mat.nc;
    int prec = 15;
    size_t nnz = 0;
    auto format = scientific;
    fs << "%%MatrixMarket matrix coordinate complex general" << endl;
    fs << "%" << endl;
    // count non-zero values first
    for (int i = 0; i < mat.size; i++)
    {
        auto v = mat.c[i];
        if (fabs(v.real()) > threshold || fabs(v.imag()) > threshold) nnz++;
    }

    fs << nr << " " << nc << " " << nnz << endl;

    if (row_first)
    {
        for (int j = 0; j < nc; j++)
        {
            for (int i = 0; i < nr; i++)
            {
                auto v = mat.c[i * nc + j];
                if (fabs(v.real()) > threshold || fabs(v.imag()) > threshold)
                    fs << i + 1 << " " << j + 1 << " " << showpoint << format << setprecision(prec)
                       << v.real() << " " << showpoint << format << setprecision(prec) << v.imag()
                       << "\n";
            }
        }
    }
    else
    {
        for (int i = 0; i < nr; i++)
        {
            for (int j = 0; j < nc; j++)
            {
                auto v = mat.c[i * nc + j];
                if (fabs(v.real()) > threshold || fabs(v.imag()) > threshold)
                    fs << i + 1 << " " << j + 1 << " " << showpoint << format << setprecision(prec)
                       << v.real() << " " << showpoint << format << setprecision(prec) << v.imag()
                       << "\n";
            }
        }
    }
}

void print_complex_matrix_file(const char *desc, const ComplexMatrix &mat, const string &fn,
                               bool use_scientific)
{
    ofstream fs;
    fs.open(fn);
    print_complex_matrix(desc, mat, fs, use_scientific);
    fs.close();
}

void print_complex_matrix_mm(const ComplexMatrix &mat, const string &fn, double threshold,
                             bool row_first)
{
    ofstream fs;
    fs.open(fn);
    print_complex_matrix_mm(mat, fs, threshold, row_first);
    fs.close();
}

void print_complex_real_matrix(const char *desc, const ComplexMatrix &mat)
{
    int nr = mat.nr;
    int nc = mat.nc;
    librpa_int::global::lib_printf("\n %s\n", desc);
    for (int i = 0; i < nr; i++)
    {
        for (int j = 0; j < nc; j++)
            librpa_int::global::lib_printf(" %10.6f", mat.c[i * nc + j].real());
        librpa_int::global::lib_printf("\n");
    }
}

}
