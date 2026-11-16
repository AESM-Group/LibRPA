#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

#include "../core/epsilon.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/constants.h"

using namespace librpa_int;

static std::string backend = "scalapack";
static void select_backend(ArrayDesc& desc, const BlacsCtxtHandler& blacs)
{
    desc.set_linear_algorithm(backend == "scalapack" ? "scalapack" : "ddla");
#ifdef LIBRPA_USE_DDLA
    desc.set_ddla_handle(backend == "gpu" ? blacs.ddla_handle : blacs.ddla_cpu_handle);
#endif
}

static bool check_logdet(const BlacsCtxtHandler& blacs,
                         const std::array<std::complex<double>, 9>& values,
                         const std::complex<double>& expected, const char* label,
                         const int block_size, const bool expect_failure = false)
{
    ArrayDesc desc(blacs);
    desc.init(3, 3, block_size, block_size, 0, 0);
    select_backend(desc, blacs);
    auto matrix = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const int row = desc.indx_g2l_r(i), col = desc.indx_g2l_c(j);
            if (row >= 0 && col >= 0) matrix(row, col) = values[3 * i + j];
        }
    std::vector<int> pivots(desc.m_loc() + desc.mb());
    int info = 0;
    std::complex<double> actual;
    try { actual = compute_pi_det_blacs_2d(matrix, desc, pivots.data(), info); }
    catch (const std::runtime_error& error)
    {
        if (blacs.myid == 0) std::cout << label << " backend=" << backend
            << " block_size=" << block_size << " diagnostic=" << error.what() << std::endl;
        return expect_failure && info > 0;
    }
    if (expect_failure) return false;
    const double real_error = std::abs(actual.real() - expected.real());
    const double phase_error = std::abs(std::remainder(actual.imag() - expected.imag(), TWO_PI));
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
        std::cout << label << " block_size=" << block_size << " actual=" << actual
                  << " expected=" << expected << " info=" << info << std::endl;
    return info == 0 && actual.imag() > -TWO_PI / 2 && actual.imag() <= TWO_PI / 2
           && std::isfinite(real_error) && std::isfinite(phase_error)
           && real_error < 1e-12 && phase_error < 1e-12;
}

static bool check_cross_column_pivot(const BlacsCtxtHandler& blacs)
{
    // The third pivot swaps rows in process column 1 on the 2x2 MPI grid.
    // det(M) = 0.5 * det(M[0:3,0:3]) = 2.817 and tr(I-M) = -2.5.
    constexpr int n = 4;
    const double m[n][n] = {{2., .3, 0., 0.}, {.5, 3., .2, 0.}, {.4, .6, 1., 0.}, {.7, .9, 4., .5}};
    ArrayDesc desc(blacs);
    desc.init(n, n, 2, 2, 0, 0);
    select_backend(desc, blacs);
    auto response = init_local_mat<std::complex<double>>(desc, MAJOR::COL);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int row = desc.indx_g2l_r(i), col = desc.indx_g2l_c(j);
            if (row >= 0 && col >= 0) response(row, col) = (i == j ? 1. : 0.) - m[i][j];
        }
    const auto actual = compute_rpa_response_trace_logdet_blacs_2d(response, desc);
    const std::complex<double> expected{std::log(2.817) - 2.5, 0.};
    if (blacs.myid == 0)
        std::cout << "cross_column_pivot actual=" << actual << " expected=" << expected
                  << std::endl;
    return std::isfinite(std::abs(actual)) && std::abs(actual - expected) < 1e-10;
}

static bool check_schur_solve(const BlacsCtxtHandler& blacs, int n, bool swap_rows = false)
{
    ArrayDesc ad(blacs), wd(blacs);
    ad.init(n, n, 2, 2, 0, 0);
    wd.init(n, 3, 2, 1, 0, 0);
    select_backend(ad, blacs); select_backend(wd, blacs);
    auto a = init_local_mat<std::complex<double>>(ad, MAJOR::COL);
    auto w = init_local_mat<std::complex<double>>(wd, MAJOR::COL);
    Matz head(3, 3, MAJOR::COL), schur;
    // W=A*X with an independently prescribed X. A is deliberately complex and
    // nonsymmetric; the Schur oracle is I-H-W^H X, with no reference inverse.
    const auto av = [swap_rows](int i, int j) {
        if (swap_rows && i < 2) i = 1 - i;
        return i == j ? std::complex<double>(2. + i, .2 * (i + 1))
                      : std::complex<double>(.11 * (1 + i), -.07 * (1 + j));
    };
    const auto xv = [](int i, int j) { return std::complex<double>(.1 * (i + 1) + .02 * j, -.04 * i + .07 * j); };
    const auto wv = [&](int i, int j) {
        std::complex<double> result{};
        for (int k = 0; k < n; ++k) result += av(i, k) * xv(k, j);
        return result;
    };
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
        {
            const int row = ad.indx_g2l_r(i), col = ad.indx_g2l_c(j);
            if (row >= 0 && col >= 0) a(row, col) = av(i, j);
        }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const int row = wd.indx_g2l_r(i), col = wd.indx_g2l_c(j);
            if (row >= 0 && col >= 0) w(row, col) = wv(i, j);
        }
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) head(i, j) = {.01 * (i + 1), -.02 * j};
    const auto actual = compute_rpa_body_schur(a, ad, w, wd, head, schur);
    std::complex<double> determinant = 1.;
    if (n == 1) determinant = av(0, 0);
    if (n == 3)
        determinant = av(0, 0) * (av(1, 1) * av(2, 2) - av(1, 2) * av(2, 1)) -
                      av(0, 1) * (av(1, 0) * av(2, 2) - av(1, 2) * av(2, 0)) +
                      av(0, 2) * (av(1, 0) * av(2, 1) - av(1, 1) * av(2, 0));
    double error = std::abs(actual - std::log(determinant));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            std::complex<double> expected = (i == j ? 1. : 0.) - head(i, j);
            for (int k = 0; k < n; ++k) expected -= std::conj(wv(k, i)) * xv(k, j);
            error = std::max(error, std::abs(schur(i, j) - expected));
        }
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, blacs.comm());
    if (blacs.myid == 0) std::cout << "Schur known-solution oracle backend=" << backend
        << " n=" << n << " swapped=" << swap_rows << " error=" << error << std::endl;
    return std::isfinite(error) && error < 2e-12;
}

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    bool passed = true;
    {
        BlacsCtxtHandler blacs(MPI_COMM_WORLD);
        blacs.init();
        blacs.set_square_grid();
#ifdef LIBRPA_USE_DDLA
        blacs.init_ddla_handle(ddla::DdlaBackend::CPU);
        const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        if (gpu) blacs.init_ddla_handle(ddla::DdlaBackend::GPU);
#endif
        const std::vector<std::string> backends = gpu ? std::vector<std::string>{"gpu"}
                                                     : std::vector<std::string>{"scalapack", "cpu"};
#else
        const std::vector<std::string> backends{"scalapack"};
#endif
        for (const auto& selected : backends)
        {
            backend = selected;
            if (selected == "scalapack")
            {
                for (int n : {0, 1, 3}) passed &= check_schur_solve(blacs, n);
                passed &= check_schur_solve(blacs, 3, true);
            }
            passed &= check_cross_column_pivot(blacs);
            const auto z = std::polar(1.0, TWO_PI / 6.0);
            const auto t = std::polar(1.0, TWO_PI * 0.4);
            for (const int block_size : {1, 2, 3})
            {
                // One row swap and z^3=-1 give det=+1, not a phase of pi.
                passed &= check_logdet(blacs, {0., z, 0., z, 0., 0., 0., 0., z},
                                       {0., 0.}, "positive_det_with_complex_pivots", block_size,
                                       selected != "scalapack" && block_size == 1);
                passed &= check_logdet(blacs, {-1., 0., 0., 0., 2., 0., 0., 0., 3.},
                                       {std::log(6.0), TWO_PI / 2.0}, "negative_determinant", block_size);
                passed &= check_logdet(blacs, {t, 0., 0., 0., t, 0., 0., 0., t},
                                       {0., TWO_PI * 0.2}, "complex_determinant", block_size);
                passed &= check_logdet(blacs, {2., 0., 0., 0., 3., 0., 0., 0., 5.},
                                       {std::log(30.0), 0.}, "positive_diagonal", block_size);
                passed &= check_logdet(blacs, {0., 0., 0., 0., 0., 0., 0., 0., 0.},
                                       {}, "singular_explicit_failure", block_size, true);
            }
        }
    }
    MPI_Finalize();
    return passed ? 0 : 1;
}
