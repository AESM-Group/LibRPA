#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#include "../interface/la_connector.h"

using namespace librpa_int;

void check_mixed(const BlacsCtxtHandler& grid)
{
    using Z = std::complex<double>;
    const auto real_value = [](int i, int j) { return 0.2 + 0.1 * i - 0.03 * j; };
    const auto complex_value = [](int i, int j) { return Z(0.3 + 0.07 * i, -0.4 + 0.05 * j); };
    for (int m : {1, 7})
        for (bool real_a : {false, true})
            for (char ta : {'N', 'T', 'C'})
                for (char tb : {'N', 'T', 'C'})
                {
                    const int n = m == 1 ? 1 : 5, k = 3;
                    ArrayDesc da(grid), db(grid), dc(grid);
                    da.init(ta == 'N' ? m : k, ta == 'N' ? k : m, 2, 2, 0, 0);
                    db.init(tb == 'N' ? k : n, tb == 'N' ? n : k, 2, 2, 0, 0);
                    dc.init(m, n, 2, 2, 0, 0);
                    for (auto* d : {&da, &db, &dc})
                    {
                        d->set_ddla_handle(grid.ddla_cpu_handle);
                        d->set_linear_algorithm("ddla");
                    }
                    std::vector<double> ra(std::max(1, da.lld() * da.n_loc()));
                    std::vector<double> rb(std::max(1, db.lld() * db.n_loc()));
                    std::vector<Z> a(ra.size()), b(rb.size()), c(std::max(1, dc.lld() * dc.n_loc()));
                    const auto fill = [&](const ArrayDesc& d, auto& real, auto& complex) {
                        for (int j = 0; j < d.n_loc(); ++j)
                            for (int i = 0; i < d.m_loc(); ++i)
                            {
                                const int r = d.indx_l2g_r(i), s = d.indx_l2g_c(j);
                                real[i + d.lld() * j] = real_value(r, s);
                                complex[i + d.lld() * j] = complex_value(r, s);
                            }
                    };
                    fill(da, ra, a); fill(db, rb, b);
                    for (const Z beta : {Z(0), Z(0.2, -0.1)})
                    {
                        const Z alpha(0.7, -0.2), initial(-0.6, 0.9);
                        std::fill(c.begin(), c.end(), beta == Z(0)
                            ? Z(std::numeric_limits<double>::quiet_NaN(), 0) : initial);
                        if (real_a)
                            LaConnector::pgemm(ta, tb, m, n, k, alpha, ra.data(), 1, 1, da,
                                b.data(), 1, 1, db, beta, c.data(), 1, 1, dc);
                        else
                            LaConnector::pgemm(ta, tb, m, n, k, alpha, a.data(), 1, 1, da,
                                rb.data(), 1, 1, db, beta, c.data(), 1, 1, dc);
                        double error = 0;
                        for (int j = 0; j < dc.n_loc(); ++j)
                            for (int i = 0; i < dc.m_loc(); ++i)
                            {
                                const int r = dc.indx_l2g_r(i), s = dc.indx_l2g_c(j);
                                std::complex<long double> sum = 0;
                                for (int p = 0; p < k; ++p)
                                {
                                    const int ar = ta == 'N' ? r : p, ac = ta == 'N' ? p : r;
                                    const int br = tb == 'N' ? p : s, bc = tb == 'N' ? s : p;
                                    Z av = real_a ? Z(real_value(ar, ac)) : complex_value(ar, ac);
                                    Z bv = real_a ? complex_value(br, bc) : Z(real_value(br, bc));
                                    if (ta == 'C') av = std::conj(av);
                                    if (tb == 'C') bv = std::conj(bv);
                                    sum += std::complex<long double>(av) * std::complex<long double>(bv);
                                }
                                const Z expected = alpha * Z(sum) + beta * initial;
                                const double delta = std::abs(c[i + dc.lld() * j] - expected);
                                error = std::isfinite(delta) ? std::max(error, delta) : INFINITY;
                            }
                        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, grid.comm());
                        assert(error < 1e-12);
                    }
                    bool rejected = false;
                    try {
                        LaConnector::pgemm('N', 'N', 1, 1, 1, Z(1), ra.data(), 2, 1, da,
                            b.data(), 1, 1, db, Z(0), c.data(), 1, 1, dc);
                    } catch (const std::invalid_argument&) { rejected = true; }
                    assert(rejected);
                    // Alpha-zero/k-zero must not inspect absent inputs.
                    for (int inner : {0, k})
                    {
                        std::fill(c.begin(), c.end(), Z(1));
                        LaConnector::pgemm(ta, tb, m, n, inner, Z(0),
                            static_cast<const double*>(nullptr), 1, 1, da,
                            static_cast<const Z*>(nullptr), 1, 1, db,
                            Z(2, -1), c.data(), 1, 1, dc);
                        for (int j = 0; j < dc.n_loc(); ++j)
                            for (int i = 0; i < dc.m_loc(); ++i)
                                assert(c[i + dc.lld() * j] == Z(2, -1));
                    }
                }
}

template <typename T>
void check(const BlacsCtxtHandler& grid, bool ddla_linear)
{
    constexpr int n = 7;
    ArrayDesc ad(grid);
    ad.init(n, n, 2, 2, 0, 0);
    ad.set_ddla_handle(grid.ddla_cpu_handle);
    ad.set_linear_algorithm(ddla_linear ? "ddla" : "scalapack");
    assert(LaConnector::use_ddla_cpu(ad) == ddla_linear);
    const int count = std::max(1, ad.lld() * ad.n_loc());
    std::vector<T> original(count), identity(count), a(count), b(count), c(count);
    auto v = [](int i) {
        if constexpr (is_complex_t<T>::value) return T(1.0 + i, 0.2 * i);
        else return T(1.0 + i);
    };
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
        {
            const int row = ad.indx_l2g_r(i), col = ad.indx_l2g_c(j);
            const int k = i + ad.lld() * j;
            identity[k] = row == col ? T(1) : T(0);
            original[k] = v(row) * get_conj(v(col)) + T(4) * identity[k];
        }
    const auto verify = [&](const auto& actual, const auto& expected) {
        double error = 0;
        for (int j = 0; j < ad.n_loc(); ++j)
            for (int i = 0; i < ad.m_loc(); ++i)
            {
                const int k = i + ad.lld() * j;
                const double delta = std::abs(actual[k] - expected[k]);
                error = std::isfinite(delta) ? std::max(error, delta)
                                             : std::numeric_limits<double>::infinity();
            }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, grid.comm());
        assert(std::isfinite(error) && error < 1e-11);
    };
    LaConnector::pgemm('N', 'C', n, n, n, T(1), original.data(), 1, 1, ad,
                       identity.data(), 1, 1, ad, T(0), c.data(), 1, 1, ad);
    verify(c, original);
    int info = 12345;
    a = original; b = original;
    LaConnector::pgesv(n, n, a.data(), 1, 1, ad, b.data(), 1, 1, ad, info);
    assert(info == 0);
    verify(b, identity);
    a = original; b = original; info = 12345;
    LaConnector::pposv('L', 'L', 'N', n, n, a.data(), 1, 1, ad,
                       b.data(), 1, 1, ad, info);
    assert(info == 0);
    verify(b, identity);

    // A CPU handle used only by an eigensolver must not force DDLA linear ops.
    // Nonunit host offsets use local LAPACK on 1x1 grids, otherwise ScaLAPACK.
    a = original; b = original; c = identity; info = 12345;
    for (int j = 0; j < ad.n_loc(); ++j)
        if (ad.indx_l2g_c(j) == 0)
            for (int i = 0; i < ad.m_loc(); ++i)
                c[i + ad.lld() * j] = original[i + ad.lld() * j];
    LaConnector::pgesv(n, n - 1, a.data(), 1, 1, ad, b.data(), 1, 2, ad, info);
    assert(info == 0);
    verify(b, c);

    std::fill(a.begin(), a.end(), T(0));
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
            if (ad.indx_l2g_r(i) == ad.indx_l2g_c(j)) a[i + ad.lld() * j] = T(2);
    b = a;
    std::vector<int> pivots(ad.m_loc() + ad.mb());
    LaConnector::pgetrf_bpiv(n, n, a.data(), 1, 1, ad, pivots.data(), info);
    assert(info == 0);
    verify(a, b);
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    assert(provided >= LIBRPA_MPI_THREAD_LEVEL);
    {
        BlacsCtxtHandler grid(MPI_COMM_WORLD);
        grid.init(); grid.set_square_grid();
        grid.init_ddla_handle(ddla::DdlaBackend::CPU);
        for (bool ddla_linear : {false, true})
        {
            if (ddla_linear) check_mixed(grid);
            check<double>(grid, ddla_linear);
            check<std::complex<double>>(grid, ddla_linear);
        }
    }
    MPI_Finalize();
}
