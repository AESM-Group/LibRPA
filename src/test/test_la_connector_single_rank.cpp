#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#include "../interface/la_connector.h"

using namespace librpa_int;

#ifdef LIBRPA_TEST_LOCAL_LAPACK_DISPATCH
// GNU ld redirects these distributed entry points to traps. Numerical
// agreement alone cannot prove that the one-rank fallback was selected.
#define REJECT_DISTRIBUTED(symbol) \
    extern "C" void __wrap_##symbol() { \
        std::fprintf(stderr, "Unexpected distributed call: %s\n", #symbol); std::abort(); }
REJECT_DISTRIBUTED(psgemm_)
REJECT_DISTRIBUTED(pdgemm_)
REJECT_DISTRIBUTED(pcgemm_)
REJECT_DISTRIBUTED(pzgemm_)
REJECT_DISTRIBUTED(pssyev_)
REJECT_DISTRIBUTED(pdsyev_)
REJECT_DISTRIBUTED(pcheev_)
REJECT_DISTRIBUTED(pzheev_)
REJECT_DISTRIBUTED(pdgetrf_)
REJECT_DISTRIBUTED(pzgetrf_)
REJECT_DISTRIBUTED(pdgesv_)
REJECT_DISTRIBUTED(pzgesv_)
REJECT_DISTRIBUTED(pdposv_)
REJECT_DISTRIBUTED(pzposv_)
#undef REJECT_DISTRIBUTED
#endif

template <typename T>
T value(int i, int j)
{
    if constexpr (is_complex_t<T>::value) return T(0.3 * (i + 1), 0.2 * (j - i));
    else return T(0.3 * (i + j + 1));
}

template <typename T>
void near(T actual, T expected)
{
    using Real = typename to_real<T>::type;
    const double tolerance = sizeof(Real) == sizeof(float) ? 5e-5 : 1e-11;
    assert(std::isfinite(std::abs(actual - expected)));
    assert(std::abs(actual - expected) < tolerance * std::max(1.0, double(std::abs(expected))));
}

template <typename T>
void check_gemm_and_eigen(const BlacsCtxtHandler& grid)
{
    // Different leading dimensions and offsets expose accidental contiguous
    // copies, row-major calls and reuse of another operand's descriptor.
    ArrayDesc ad(grid), bd(grid), cd(grid);
    ad.init(8, 8, 2, 2, 0, 0);
    bd.init(9, 9, 3, 3, 0, 0);
    cd.init(10, 10, 2, 2, 0, 0);
    assert(LaConnector::is_single_rank_grid(ad));
    assert(LaConnector::submatrix_fits(ad, 2, 3, 4, 4));
    assert(!LaConnector::submatrix_fits(ad, 0, 1, 4, 4));
    assert(!LaConnector::submatrix_fits(ad, 2, 3, -1, 4));
    assert(!LaConnector::submatrix_fits(ad, 2, 3, 4, 7));
    std::vector<T> a(64, T(-123)), b(81, T(-123)), c(100, T(-123));
    const T alpha = value<T>(1, 2), beta = value<T>(2, 0);
    for (char trans : {'N', 'T', 'C'})
    {
        const bool nt = trans == 'N';
        for (int j = 0; j < (nt ? 4 : 3); ++j)
            for (int i = 0; i < (nt ? 3 : 4); ++i)
                a[1 + i + 8 * (2 + j)] = value<T>(i, j);
        for (int j = 0; j < 2; ++j)
            for (int i = 0; i < 4; ++i) b[2 + i + 9 * (1 + j)] = value<T>(i + 2, j);
        std::fill(c.begin(), c.end(), T(-123));
        LaConnector::pgemm(trans, 'N', 3, 2, 4, alpha, a.data(), 2, 3, ad,
                           b.data(), 3, 2, bd, beta, c.data(), 4, 2, cd);
        for (int j = 0; j < 10; ++j)
            for (int i = 0; i < 10; ++i)
            {
                T expected(-123);
                if (i >= 3 && i < 6 && j >= 1 && j < 3)
                {
                    T sum{};
                    for (int k = 0; k < 4; ++k)
                    {
                        T x = nt ? value<T>(i - 3, k) : value<T>(k, i - 3);
                        if (trans == 'C') x = get_conj(x);
                        sum += x * value<T>(k + 2, j - 1);
                    }
                    expected = alpha * sum + beta * T(-123);
                }
                near(c[i + 10 * j], expected);
            }
    }

    constexpr int n = 4;
    auto h = [](int i, int j) {
        return value<T>(i, 0) * get_conj(value<T>(j, 0)) + (i == j ? T(2) : T(0));
    };
    for (char uplo : {'U', 'L'})
        for (char jobz : {'V', 'N'})
        {
            std::fill(a.begin(), a.end(), T(-123));
            std::fill(c.begin(), c.end(), T(-123));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    a[1 + i + 8 * (2 + j)] =
                        ((uplo == 'U' && i <= j) || (uplo == 'L' && i >= j))
                        ? h(i, j) : T(std::numeric_limits<double>::quiet_NaN());
            typename to_real<T>::type w[n];
            int info = -999;
            LaConnector::pheev(jobz, uplo, n, a.data(), 2, 3, ad, w,
                               c.data(), 4, 2, cd, info);
            assert(info == 0);
            double norm = 0;
            for (int i = 0; i < n; ++i) norm += std::norm(value<T>(i, 0));
            for (int k = 0; k < n; ++k) near(T(w[k]), T(k == n - 1 ? 2 + norm : 2));
            if (jobz == 'N')
            {
                for (const auto x : c) near(x, T(-123));
                continue;
            }
            for (int k = 0; k < n; ++k)
                for (int i = 0; i < n; ++i)
                {
                    T az{}, dot{};
                    for (int j = 0; j < n; ++j)
                    {
                        az += h(i, j) * c[3 + j + 10 * (1 + k)];
                        dot += get_conj(c[3 + j + 10 * (1 + i)]) * c[3 + j + 10 * (1 + k)];
                    }
                    near(az, T(w[k]) * c[3 + i + 10 * (1 + k)]);
                    near(dot, i == k ? T(1) : T(0));
                }
        }
}

template <typename T>
void check_solves_at(const BlacsCtxtHandler& grid, int ia, int ja)
{
    ArrayDesc ad(grid), bd(grid);
    ad.init(7, 7, 2, 2, 0, 0);
    bd.init(9, 6, 3, 3, 0, 0);
    constexpr int n = 4, nrhs = 2;
    std::vector<T> a(49), b(54);
    const auto index_a = [ia, ja](int i, int j) { return ia - 1 + i + 7 * (ja - 1 + j); };
    for (bool cholesky : {false, true})
        for (char uplo : {'U', 'L'})
        {
            std::fill(a.begin(), a.end(), T(-123));
            std::fill(b.begin(), b.end(), T(-123));
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    a[index_a(i, j)] = value<T>(i, 0) * get_conj(value<T>(j, 0)) +
                                   (i == j ? T(2) : T(0));
            for (int j = 0; j < nrhs; ++j)
                for (int i = 0; i < n; ++i)
                {
                    T rhs{};
                    for (int k = 0; k < n; ++k) rhs += a[index_a(i, k)] * value<T>(k, j);
                    b[1 + i + 9 * (2 + j)] = rhs;
                }
            int info = -999;
            if (cholesky)
                LaConnector::pposv('L', uplo, 'N', n, nrhs, a.data(), ia, ja, ad,
                                   b.data(), 2, 3, bd, info);
            else
                LaConnector::pgesv(n, nrhs, a.data(), ia, ja, ad, b.data(), 2, 3, bd, info);
            assert(info == 0);
            for (int j = 0; j < 7; ++j)
                for (int i = 0; i < 7; ++i)
                    if (i < ia - 1 || i >= ia - 1 + n || j < ja - 1 || j >= ja - 1 + n)
                        near(a[i + 7 * j], T(-123));
            for (int j = 0; j < 6; ++j)
                for (int i = 0; i < 9; ++i)
                    near(b[i + 9 * j], i >= 1 && i < 5 && j >= 2 && j < 4
                         ? value<T>(i - 1, j - 2) : T(-123));
        }
    // Require a pivot and check the factorization, not only INFO.
    std::fill(a.begin(), a.end(), T(-123));
    a[index_a(0, 0)] = T(0); a[index_a(1, 0)] = T(2);
    a[index_a(0, 1)] = T(1); a[index_a(1, 1)] = T(3);
    std::vector<int> pivots(9, -123);
    int info = -999;
    LaConnector::pgetrf_bpiv(2, 2, a.data(), ia, ja, ad, pivots.data(), info);
    assert(info == 0 && pivots[ia - 1] == ia + 1 && pivots[ia] == ia + 1);
    for (int i = 0; i < static_cast<int>(pivots.size()); ++i)
        if (i != ia - 1 && i != ia) assert(pivots[i] == -123);
    near(a[index_a(0, 0)], T(2)); near(a[index_a(1, 0)], T(0));
    near(a[index_a(0, 1)], T(3)); near(a[index_a(1, 1)], T(1));
    for (int j = 0; j < 7; ++j)
        for (int i = 0; i < 7; ++i)
            if (i < ia - 1 || i >= ia + 1 || j < ja - 1 || j >= ja + 1)
                near(a[i + 7 * j], T(-123));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) a[index_a(i, j)] = T(0);
    LaConnector::pgetrf_bpiv(n, n, a.data(), ia, ja, ad, pivots.data(), info);
    assert(info == 1);
    for (const auto [rows, cols] : {std::pair<int, int>{0, 2}, {2, 0}})
    {
        const auto before = a;
        std::fill(pivots.begin(), pivots.end(), -123);
        info = -999;
        LaConnector::pgetrf_bpiv(rows, cols, a.data(), ia, ja, ad, pivots.data(), info);
        assert(info == 0 && a == before);
        for (int pivot : pivots) assert(pivot == -123);
    }
    for (bool cholesky : {false, true})
    {
        std::fill(a.begin(), a.end(), T(0));
        bool threw = false;
        try {
            if (cholesky)
                LaConnector::pposv('L', 'U', 'N', n, nrhs, a.data(), ia, ja, ad,
                                   b.data(), 2, 3, bd, info);
            else
                LaConnector::pgesv(n, nrhs, a.data(), ia, ja, ad, b.data(), 2, 3, bd, info);
        } catch (const std::runtime_error&) { threw = true; }
        assert(threw && info == 1);
    }
}

template <typename T>
void check_solves(const BlacsCtxtHandler& grid)
{
    for (const auto [ia, ja] : {std::pair<int, int>{1, 1}, {1, 3}, {2, 3},
                              {3, 1}, {3, 2}, {4, 4}})
        check_solves_at<T>(grid, ia, ja);
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    assert(provided >= LIBRPA_MPI_THREAD_LEVEL);
    {
        BlacsCtxtHandler world(MPI_COMM_WORLD);
        world.init(); world.set_square_grid();
        ArrayDesc desc(world); desc.init(8, 8, 2, 2, 0, 0);
        assert(LaConnector::is_single_rank_grid(desc) == (world.nprocs == 1));
        // Every world rank owns an independent local grid. This must not
        // accidentally select distributed work on MPI_COMM_WORLD.
        BlacsCtxtHandler local(MPI_COMM_SELF);
        local.init(); local.set_square_grid();
#ifdef LIBRPA_TEST_LOCAL_LAPACK_DISPATCH
        if (argc == 2 && std::string(argv[1]) == "--check-dispatch-trap")
        {
            ArrayDesc one(local); one.init(1, 1, 1, 1, 0, 0);
            double a = 1, b = 1, c = 0;
            ScalapackConnector::pgemm_f('N', 'N', 1, 1, 1, 1.0,
                &a, 1, 1, one.desc, &b, 1, 1, one.desc, 0.0, &c, 1, 1, one.desc);
            std::abort();
        }
#endif
        check_gemm_and_eigen<float>(local);
        check_gemm_and_eigen<double>(local);
        check_gemm_and_eigen<std::complex<float>>(local);
        check_gemm_and_eigen<std::complex<double>>(local);
        check_solves<double>(local);
        check_solves<std::complex<double>>(local);
    }
    MPI_Finalize();
}
