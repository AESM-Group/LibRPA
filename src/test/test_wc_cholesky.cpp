#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <string>
#include <vector>
#include "librpa.h"

#include "../core/dielecmodel.h"
#include "../interface/la_connector.h"
#include "../math/utils_matrix_mpi.h"
#ifdef LIBRPA_USE_DDLA
#include <ddla/ddla_vector.h>
#endif

using namespace librpa_int;
using Z = std::complex<double>;

// Independent known-X systems exercise a failed, partially overwritten POTRF,
// off-diagonal complex entries, multiple RHS, singular LU and empty owners.
void check_system(const BlacsCtxtHandler& grid, int n, int mode, bool ddla_cpu, bool gpu)
{
    ArrayDesc ad(grid), bd(grid);
    ad.init(n, n, 2, 2, 0, 0);
    bd.init(n, 2, 2, 2, 0, 0);
#ifdef LIBRPA_USE_DDLA
    if (ddla_cpu || gpu)
        for (auto* d : {&ad, &bd})
        {
            d->set_linear_algorithm("ddla");
            d->set_ddla_handle(gpu ? grid.ddla_handle : grid.ddla_cpu_handle);
        }
#endif
    const auto entry = [=](int i, int j) {
        if (mode == 2) return Z(i == j && i != n - 1 ? 2. : 0.);
        if (mode == 7)
        {
            if (n == 1) return Z(-2.);
            if (i == 0 && j == 0) return Z{};
            if (i == 0 && j == 1) return Z(2., 1.);
            if (i == 1 && j == 0) return Z(2., -1.);
            return Z(i == j ? i + 2. : 0.);
        }
        const Z u(.2 + .1 * i, .03 * i), v(.2 + .1 * j, .03 * j);
        if ((mode == 1 || mode == 5) && i == j && i == n - 1) return Z(-2.);
        return u * std::conj(v) + Z(i == j ? n + 1. : 0.);
    };
    const auto x = [](int i, int j) { return Z(.2 + .13 * i + .07 * j, .04 * (i - j)); };
    const auto ac = std::max<std::size_t>(1, std::size_t(ad.lld()) * ad.n_loc());
    const auto bc = std::max<std::size_t>(1, std::size_t(bd.lld()) * bd.n_loc());
    std::vector<Z> a(ac), b(bc), backup(ac);
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
            a[i + ad.lld() * j] = entry(ad.indx_l2g_r(i), ad.indx_l2g_c(j));
    for (int j = 0; j < bd.n_loc(); ++j)
        for (int i = 0; i < bd.m_loc(); ++i)
            for (int k = 0; k < n; ++k)
                b[i + bd.lld() * j] += entry(bd.indx_l2g_r(i), k) * x(k, bd.indx_l2g_c(j));
    const bool gamma = mode == 1 || mode == 2 || mode == 7;
    const bool expect_error = mode == 2 || mode == 5;
    const bool expect_cholesky = mode == 0;
    bool threw = false, cholesky = false;
    std::string failure;
    const auto apply = [&](Z* ap, Z* bp, Z* save) {
        if (ad.m_loc() == 0 || ad.n_loc() == 0) ap = nullptr;
        if (bd.m_loc() == 0 || bd.n_loc() == 0) bp = nullptr;
        try { cholesky = solve_dielectric_system(ap, bp, ad, bd, gamma, save); }
        catch (const std::runtime_error& e)
        {
            threw = true;
            const std::string message(e.what());
            failure = message;
            assert(message.find(mode == 2 ? "LU factorization" : "Cholesky factorization") != std::string::npos);
        }
    };
#ifdef LIBRPA_USE_DDLA
    if (gpu)
    {
        const auto handle = grid.ddla_handle;
        ddla::vector<Z> da(handle, ac), db(handle, bc), saved(handle, ac);
        LaConnector::copy_local(da.data(), a.data(), ac, ad);
        LaConnector::copy_local(db.data(), b.data(), bc, bd);
        LaConnector::synchronize(ad);
        apply(da.data(), db.data(), saved.data());
        LaConnector::copy_local(b.data(), db.data(), bc, bd);
        LaConnector::synchronize(ad);
    }
    else
#endif
        apply(a.data(), b.data(), backup.data());
    if (threw != expect_error && grid.myid == 0)
        std::cerr << "Unexpected dielectric status n=" << n << " mode=" << mode
                  << " ddla_cpu=" << ddla_cpu << " gpu=" << gpu
                  << " threw=" << threw << " expected_error=" << expect_error
                  << " message=" << failure << std::endl;
    assert(threw == expect_error);
    if (expect_error) return;
    assert(cholesky == expect_cholesky);
    double error = 0.;
    for (int j = 0; j < bd.n_loc(); ++j)
        for (int i = 0; i < bd.m_loc(); ++i)
        {
            const auto delta = std::abs(b[i + bd.lld() * j] - x(bd.indx_l2g_r(i), bd.indx_l2g_c(j)));
            error = std::isfinite(delta) ? std::max(error, delta) : INFINITY;
        }
    double maximum = 0.;
    MPI_Allreduce(&error, &maximum, 1, MPI_DOUBLE, MPI_MAX, grid.comm());
    assert(maximum < 1.e-11);
}

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    assert(provided >= LIBRPA_MPI_THREAD_LEVEL);
    {
        bool gpu = false, ddla_cpu = false, rows = false;
        for (int i = 1; i < argc; ++i)
        {
            const std::string arg(argv[i]);
            gpu |= arg == "--gpu"; ddla_cpu |= arg == "--cpu-ddla"; rows |= arg == "--rows";
        }
        BlacsCtxtHandler grid(MPI_COMM_WORLD);
        grid.init();
        if (rows) grid.set_grid(grid.nprocs, 1); else grid.set_square_grid();
#ifdef LIBRPA_USE_DDLA
        if (gpu) grid.init_ddla_handle();
        if (ddla_cpu) grid.init_ddla_handle(ddla::DdlaBackend::CPU);
#else
        assert(!gpu && !ddla_cpu);
#endif
        for (int n : {1, 3, 7})
            for (int mode : {0, 1, 2, 5, 7})
            {
                check_system(grid, n, mode, ddla_cpu, gpu);
            }
        if (grid.myid == 0) std::cout << "WC_CHOLESKY_LU_FALLBACK_PASS" << std::endl;
    }
    MPI_Finalize();
}
