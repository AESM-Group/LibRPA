#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#include "../interface/la_connector.h"

using namespace librpa_int;

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
            check<double>(grid, ddla_linear);
            check<std::complex<double>>(grid, ddla_linear);
        }
    }
    MPI_Finalize();
}
