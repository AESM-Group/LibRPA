#include "../math/lapack_connector.h"
#include "../math/matrix_m.h"
#include "testutils.h"
#include "../math/complexmatrix.h"
#include <cassert>
#include <cstdlib>
#include <ctime>

using namespace librpa_int;

void test_gerc_rank_one_update()
{
    using Z = std::complex<double>;
    const auto assert_complex_close = [](Z actual, Z expected, double tolerance)
    { assert(std::abs(actual - expected) < tolerance); };
    const Z alpha(0.4, -0.7);
    const Z x[] = {{1, 2}, {99, 99}, {-3, 0.5}};
    const Z y[] = {{2, -1}, {99, 99}, {0.5, 3}, {99, 99}, {-1, -2}};
    std::vector<Z> a(9, Z(0.3, -0.2));
    const auto original = a;
    LapackConnector::gerc_f(2, 3, alpha, x, 2, y, 2, a.data(), 3);
    for (int j = 0; j < 3; ++j)
    {
        for (int i = 0; i < 2; ++i)
            assert_complex_close(
                a[i + 3 * j], original[i + 3 * j] + alpha * x[2 * i] * std::conj(y[2 * j]), 1e-14);
        assert(a[2 + 3 * j] == original[2 + 3 * j]);
    }
    // Exercise the real-vector rank-one update with both matrix layouts and a complex coefficient.
    const Z u[] = {0.6, 0.8};
    for (auto major : {MAJOR::ROW, MAJOR::COL})
    {
        matrix_m<Z> matrix(2, 2, major);
        matrix(0, 0) = 2.0;
        matrix(0, 1) = Z(0.3, 0.2);
        matrix(1, 0) = Z(-0.1, 0.5);
        matrix(1, 1) = 3.0;
        const std::vector<Z> before{matrix(0, 0), matrix(0, 1), matrix(1, 0), matrix(1, 1)};
        LapackConnector::gerc_f(2, 2, alpha, u, 1, u, 1, matrix.ptr(), 2);
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                assert_complex_close(matrix(i, j),
                                     before[2 * i + j] + alpha * u[i] * std::conj(u[j]), 1e-14);
    }
}

void test_blas_mv_mm_double(bool debug = false)
{
    const int RDIM = 6;
    const int IDIM = 2;
    const int JDIM = 3;
    assert( IDIM * JDIM == RDIM );
    const int CDIM = 2;
    double v1[RDIM] { -1, 2, 3, 1, 4, 6};
    double v2[RDIM] { 4, 0, 1, -1, 2, 3};
    double mat[RDIM][CDIM] {
        {1, -1}, {-1, 2}, {3, 0}, {0, -1}, {2, 3}, {0, 1}
    };
    int incr = 1;
    // vector dot
    assert(ddot_(&RDIM, v1, &incr, v2, &incr) == 24);
    assert(LapackConnector::dot(RDIM, v1, incr, v2, incr) == 24);

    // matrix-vector
    double y[CDIM] {};
    const char transn = 'N';
    const double alpha = 1.0, beta = 0.0;
    dgemv_(&transn, &CDIM, &RDIM, &alpha, *mat, &CDIM, v1, &incr, &beta, y, &incr);
    assert( y[0] == 14.0 && y[1] == 22.0 );
    LapackConnector::gemv('T', RDIM, CDIM, 1.0, *mat, CDIM, v1, 1, 0.5, y, 1);
    assert( y[0] == 21.0 && y[1] == 33.0 );

    srand(time(0));

    // matrix-matrix
    double mat2[CDIM][RDIM] = {};
    for ( int ic = 0; ic != CDIM; ic++ )
        for ( int ir = 0; ir != RDIM; ir++ )
            mat2[ic][ir] = double(rand()) / RAND_MAX;

    double mat_or[RDIM][RDIM] = {};
    // double mat_oc[CDIM][CDIM] = {};
    // const char transt = 'T';
    // mat[RDIM][CDIM] * mat2[CDIM][RIM] = m[RDIM][RDIM]
    dgemm_(&transn, &transn, &RDIM, &RDIM, &CDIM, &alpha, *mat2, &RDIM, *mat, &CDIM, &beta, *mat_or, &RDIM);
    assert( is_matmul_AB_equal_C(RDIM, RDIM, CDIM, *mat, *mat2, *mat_or, false, debug));
    LapackConnector::gemm(transn, transn, RDIM, RDIM, CDIM, 1.0, *mat, CDIM, *mat2, RDIM, 0.0, *mat_or, RDIM);
    assert( is_matmul_AB_equal_C(RDIM, RDIM, CDIM, *mat, *mat2, *mat_or, false, debug));

    // Check Cs like per-mu gemm
    const size_t n_mu = 3, n_i = 4, n_k = 12, n_j = 4;
    double Cs[n_mu][n_i*n_k];
    double G[n_k][n_j];
    double res[n_mu][n_i*n_j] = {};
    double ref[n_mu][n_i*n_j] = {};
    // initialize
    for ( int k = 0; k != n_k; k++ )
    {
        for ( int mu = 0; mu != n_mu; mu++ )
            for ( int i = 0; i != n_i; i++ )
            {
                Cs[mu][i*n_k+k] = double(rand()) / RAND_MAX;
            }
        for ( int j = 0; j != n_j; j++ )
            G[k][j] = double(rand()) / RAND_MAX;
    }
    // calculate reference in direct way
    for ( int mu = 0; mu != n_mu; mu++ )
        for ( int i = 0; i != n_i; i++ )
            for ( int k = 0; k != n_k; k++ )
                for ( int j = 0; j != n_j; j++ )
                {
                    ref[mu][i*n_j+j] += Cs[mu][i*n_k+k] * G[k][j];
                }
    for ( int mu = 0; mu != n_mu; mu++ )
        LapackConnector::gemm('N', 'N', n_i, n_j, n_k, 1.0,
                              Cs[mu], n_k,
                              *G, n_j,
                              1.0, res[mu], n_j);
    assert(is_mat_A_equal_B(n_mu, n_i*n_j, *res, *ref, false, debug));
}

void test_lapack_ev_complex(bool debug = false)
{
    ComplexMatrix a(2, 2);

    // Hermitian eigenvalue and eigenvectors
    a(0, 0) = cmplx(1., 0.);
    a(0, 1) = cmplx(0., -1.);
    a(1, 0) = cmplx(0., 1.);
    a(1, 1) = cmplx(1., 0.);
    int nb = LapackConnector::ilaenv(1, "zheev", "VU", 2, -1, -1, -1);
    int lwork = 2 * (nb+1);
    int info = 0;
    double w[2];
    double w_ref[2] = {0, 2};
    double rwork[3*a.nc-2];
    complex<double> work[lwork];
    ComplexMatrix ev_ref(2, 2);
    ev_ref(0, 0) = cmplx(0., -1.);
    ev_ref(1, 0) = cmplx(-1., 0.);
    ev_ref(1, 1) = cmplx(1., 0.);
    ev_ref(0, 1) = cmplx(0., -1.);
    ev_ref *= 1/sqrt(2);
    LapackConnector::zheev('V', 'U', 2, a, 2,
                           w, work, lwork, rwork, &info);
    assert(fequal_array(2, w, w_ref, debug) );
    assert(fequal_array(4, a.c, ev_ref.c, debug) );
}

int main (int argc, char *argv[])
{
    test_gerc_rank_one_update();
    test_blas_mv_mm_double(false);
    test_lapack_ev_complex(true);
    return 0;
}
