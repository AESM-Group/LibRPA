#include <cassert>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include <ddla/ddla_vector.h>

#include "../gpu/la_connector.h"

using namespace librpa_int;
using T = std::complex<double>;

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    assert(provided >= LIBRPA_MPI_THREAD_LEVEL);
    {
        BlacsCtxtHandler grid(MPI_COMM_WORLD);
        grid.init();
        grid.set_square_grid();
        grid.init_ddla_handle();
        ArrayDesc desc(grid);
        desc.init(7, 7, 2, 2, 0, 0);
        desc.set_ddla_handle(grid.ddla_handle);
        assert(desc.ddla_handle() == grid.ddla_handle);
        const auto handle = grid.ddla_handle;
        const int count = desc.lld() * desc.n_loc();
        std::vector<T> original(count), identity(count), diagonal(count), result(count);
        for (int j = 0; j < desc.n_loc(); ++j)
            for (int i = 0; i < desc.m_loc(); ++i)
            {
                const int row = desc.indx_l2g_r(i);
                const int col = desc.indx_l2g_c(j);
                const int k = i + desc.lld() * j;
                original[k] = T(1.0 + row + 2.0 * col, 0.2 * (row - col));
                identity[k] = row == col ? T(1.0) : T(0.0);
                diagonal[k] = 2.0 * identity[k];
            }
        ddla::vector<T> a(handle, count), b(handle, count), c(handle, count);
        auto upload = [&](auto& device, const auto& host) {
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, device.data(), host.data(), count * sizeof(T)));
        };
        auto check = [&](const std::vector<T>& expected, double scale) {
            ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, result.data(), c.data(), count * sizeof(T)));
            ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
            double error = 0.0;
            for (int k = 0; k < count; ++k)
                error = std::max(error, std::abs(result[k] - scale * expected[k]));
            double global_error = 0.0;
            MPI_Allreduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            assert(std::isfinite(global_error) && global_error < 1.e-12);
        };
        upload(a, original);
        upload(b, identity);
        LaConnector::pgemm('N', 'C', 7, 7, 7, T(1), a.data(), 1, 1, desc,
                           b.data(), 1, 1, desc, T(0), c.data(), 1, 1, desc);
        check(original, 1.0);
        LaConnector::scal(count, 2.0, c.data(), 1, desc);
        LaConnector::axpy(count, T(-1), a.data(), 1, c.data(), 1, grid);
        check(original, 1.0);
        upload(a, diagonal);
        upload(c, identity);
        int info = 0;
        LaConnector::pgesv(7, 7, a.data(), 1, 1, desc, c.data(), 1, 1, desc, info);
        check(identity, 0.5);
        upload(a, diagonal);
        upload(c, identity);
        if (grid.nprows == grid.npcols)
        {
            LaConnector::pposv('L', 'L', 'N', 7, 7, a.data(), 1, 1, desc,
                               c.data(), 1, 1, desc, info);
            assert(info == 0);
            check(identity, 0.5);
        }
        else
        {
            // DDLA's Cholesky factorization requires a square process grid.
            bool rejected_grid = false;
            try
            {
                LaConnector::pposv('L', 'L', 'N', 7, 7, a.data(), 1, 1, desc,
                                   c.data(), 1, 1, desc, info);
            }
            catch (const std::invalid_argument&) { rejected_grid = true; }
            assert(rejected_grid);
            check(identity, 1.0);
        }
        upload(c, identity);
        LaConnector::pdam(1.0, c.data(), desc);
        check(identity, 2.0);
        ddla::vector<int> pivots(handle, std::max(1, desc.m_loc()));
        upload(a, diagonal);
        LaConnector::pgetrf_bpiv(7, 7, a.data(), 1, 1, desc, pivots.data(), info);
        assert(info == 0);
        bool rejected = false;
        try
        {
            LaConnector::pgesv(7, 7, a.data(), 1, 1, desc,
                               c.data(), 1, 2, desc, info);
        }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
        rejected = false;
        try
        {
            LaConnector::pposv('L', 'L', 'N', 7, 7, a.data(), 1, 1, desc,
                               c.data(), 1, 2, desc, info);
        }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
        // Host ScaLAPACK retains its submatrix-offset support in GPU builds.
        for (const bool cholesky : {false, true})
        {
            auto host_a = diagonal;
            auto host_b = identity;
            if (cholesky)
                LaConnector::pposv('L', 'L', 'N', 7, 6, host_a.data(), 1, 1, desc,
                                   host_b.data(), 1, 2, desc, info);
            else
                LaConnector::pgesv(7, 6, host_a.data(), 1, 1, desc,
                                   host_b.data(), 1, 2, desc, info);
            assert(info == 0);
            double error = 0.0;
            for (int j = 0; j < desc.n_loc(); ++j)
                for (int i = 0; i < desc.m_loc(); ++i)
                {
                    const int k = i + desc.lld() * j;
                    const double scale = desc.indx_l2g_c(j) == 0 ? 1.0 : 0.5;
                    error = std::max(error, std::abs(host_b[k] - scale * identity[k]));
                }
            double global_error = 0.0;
            MPI_Allreduce(&error, &global_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            assert(std::isfinite(global_error) && global_error < 1.e-12);
        }
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        if (grid.myid == 0)
            std::puts("DDLA_CONNECTOR_PASS: distributed GEMM, scale, AXPY, LU, Cholesky grid contract and diagonal update");
    }
    MPI_Finalize();
}
