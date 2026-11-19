#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>
#include <ddla/ddla_vector.h>

#include "../core/dielecmodel.h"
#include "../core/epsilon.h"
#include "../utils/constants.h"
#include "../interface/la_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "mpi_test_config.h"
#include "headwing_rewrite_test_utils.h"

using namespace librpa_int;
using Z = std::complex<double>;

static void require(bool good, const char* message, MPI_Comm comm)
{
    int bad = !good;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) throw std::runtime_error(message);
}

static void run_case(BlacsCtxtHandler& grid, int n, int retained, int block,
                      int repeats, bool native, bool cholesky)
{
    const auto handle = grid.ddla_handle;
    ArrayDesc desc(grid), hd(grid), fd(grid);
    desc.init(n, n, block, block, 0, 0);
    hd.init(n, 1, block, block, 0, 0);
    fd.init(n, n - retained, block, block, 0, 0);
    for (auto* d : {&desc, &hd, &fd}) { d->set_linear_algorithm("ddla"); d->set_ddla_handle(handle); }
    auto eps = init_local_mat<Z>(desc, MAJOR::COL);
    auto sqrtv = init_local_mat<Z>(desc, MAJOR::COL);
    auto u = init_local_mat<Z>(desc, MAJOR::COL);
    auto host_scratch = init_local_mat<Z>(desc, MAJOR::COL); // models the existing caller scratch
    auto direction = init_local_mat<Z>(hd, MAJOR::COL);
    Matz head(3, 3, MAJOR::COL), wing(n, 3, MAJOR::COL);
    for (int a = 0; a < 3; ++a) head(a, a) = 1.1 + .1 * a;
    const auto v = [&](int i) { return i < retained ? Z(.1 + .02 * (i % 5), .03 * (i % 3)) / std::sqrt(double(n)) : Z(0); };
    const auto w = [&](int i, int a) { return Z(.025 + .003 * (i % 7) + .01 * a, .007 * (i % 3) - .005 * a) / std::sqrt(double(n)); };
    for (int i = 0; i < n; ++i) for (int a = 0; a < 3; ++a) wing(i, a) = w(i, a);
    for (int j = 0; j < desc.n_loc(); ++j) for (int i = 0; i < desc.m_loc(); ++i)
    {
        const int r = desc.indx_l2g_r(i), c = desc.indx_l2g_c(j);
        u(i, j) = r == c ? 1. : 0.;
        sqrtv(i, j) = r == c && r < retained ? 1. : 0.;
    }
    for (int j = 0; j < hd.n_loc(); ++j) for (int i = 0; i < hd.m_loc(); ++i)
        direction(i, j) = hd.indx_l2g_r(i) == 0 ? 1. : 0.;
    const size_t count = std::max<size_t>(1, size_t(desc.lld()) * desc.n_loc());
    ddla::vector<Z> d0(handle, count), d1(handle, count), d2(handle, count);
    double packing = 0.;
    if (native && retained < n)
    {
        const double begin = omp_get_wtime();
        ScalapackConnector::pgemr2d_f(n, n - retained, u.ptr(), 1, retained + 1,
            desc.desc, host_scratch.ptr(), 1, 1, fd.desc, grid.ictxt);
        packing = omp_get_wtime() - begin;
    }
    std::vector<double> qx{1., -1., 0., 0., 0., 0.}, qy{0., 0., 1., -1., 0., 0.},
        qz{0., 0., 0., 0., 1., -1.}, rho(6, 1. / 6.);
    // Independent rank-one body inverse B=I-v*v^H/(1+v^H*v).
    // This oracle forms neither a determinant nor a dense inverse/reference.
    double norm = 0.;
    std::array<Z, 3> vh_w{};
    for (int i = 1; i < retained; ++i)
    {
        norm += std::norm(v(i));
        for (int a = 0; a < 3; ++a) vh_w[a] += std::conj(v(i)) * w(i, a);
    }
    const auto a_value = [&](int i, int a) {
        return i > 0 && i < retained ? w(i, a) - v(i) * vh_w[a] / (1. + norm) : Z(0);
    };
    std::array<std::array<Z, 3>, 3> l{}, k{};
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
    {
        l[a][b] = head(a, b);
        for (int i = 1; i < retained; ++i) l[a][b] -= std::conj(w(i, a)) * a_value(i, b);
    }
    Z a0{};
    for (int p = 0; p < 6; ++p)
    {
        const std::array<double, 3> q{{qx[p], qy[p], qz[p]}};
        Z value{};
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) value += q[a] * l[a][b] * q[b];
        const auto weight = rho[p] / value;
        a0 += weight;
        for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) k[a][b] += weight * q[a] * q[b];
    }
    std::vector<double> times;
    double maximum_error = 0.;
    for (int repeat = 0; repeat < repeats + 1; ++repeat)
    {
        for (int j = 0; j < desc.n_loc(); ++j) for (int i = 0; i < desc.m_loc(); ++i)
        {
            const int r = desc.indx_l2g_r(i), c = desc.indx_l2g_c(j);
            eps(i, j) = (r == c ? 1. : 0.) + v(r) * std::conj(v(c));
        }
        if (eps.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d0.data(), eps.ptr(), eps.size() * sizeof(Z)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        const double begin = omp_get_wtime();
        // Include caller preparation and transfers before the common kernel,
        // then return the result to the outer device Coulomb products.
        if (eps.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, eps.ptr(), d0.data(), eps.size() * sizeof(Z)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        headwing_test::rewrite(eps, sqrtv, u, direction, hd, head, wing, qx, qy, qz, rho,
            desc, grid, retained, 0., cholesky, native);
        if (eps.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d0.data(), eps.ptr(), eps.size() * sizeof(Z)));
        if (native)
        {
            if (retained < n)
            {
                const size_t packed_count = size_t(fd.m_loc()) * fd.n_loc();
                if (packed_count) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d1.data(), host_scratch.ptr(), packed_count * sizeof(Z)));
                LaConnector::pgemm('N', 'C', n, n, n - retained, Z(1), d1.data(), 1, 1, fd,
                    d1.data(), 1, 1, fd, Z(1), d0.data(), 1, 1, desc);
            }
            LaConnector::pdam(-1., d0.data(), desc);
        }
        else
        {
            if (retained < n)
                ScalapackConnector::pgemm_f('N', 'C', n, n, n - retained, Z(1), u.ptr(), 1,
                    retained + 1, desc.desc, u.ptr(), 1, retained + 1, desc.desc,
                    Z(1), eps.ptr(), 1, 1, desc.desc);
            for (int i = 0; i < n; ++i)
            {
                const int r=desc.indx_g2l_r(i), c=desc.indx_g2l_c(i);
                if(r>=0&&c>=0) eps(r,c)-=1.;
            }
            if (eps.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d0.data(), eps.ptr(), eps.size() * sizeof(Z)));
        }
        // Sc=I is deliberate: it leaves filtered-channel mistakes observable.
        // Include the two production final Coulomb products in both timings.
        if (u.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d1.data(), u.ptr(), u.size() * sizeof(Z)));
        LaConnector::pgemm('N','N',n,n,n,Z(1),d1.data(),1,1,desc,d0.data(),1,1,desc,Z(0),d2.data(),1,1,desc);
        LaConnector::pgemm('N','N',n,n,n,Z(1),d2.data(),1,1,desc,d1.data(),1,1,desc,Z(0),d0.data(),1,1,desc);
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        double elapsed = omp_get_wtime() - begin;
        MPI_Allreduce(MPI_IN_PLACE, &elapsed, 1, MPI_DOUBLE, MPI_MAX, grid.comm());
        if (repeat > 0) times.push_back(elapsed);
        if (eps.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, eps.ptr(), d0.data(), eps.size() * sizeof(Z)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        double error = 0.;
        for (int j = 0; j < desc.n_loc(); ++j) for (int i = 0; i < desc.m_loc(); ++i)
        {
            const int r = desc.indx_l2g_r(i), c = desc.indx_l2g_c(j);
            Z expected = r == 0 && c == 0 ? a0 : Z(0);
            if (r > 0 && c > 0 && r < retained && c < retained)
                expected = (r == c ? 1. : 0.) - v(r) * std::conj(v(c)) / (1. + norm);
            for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b)
                expected += a_value(r, a) * k[a][b] * std::conj(a_value(c, b));
            if (r == c && r < retained) expected -= 1.;
            const double delta = std::abs(eps(i, j) - expected);
            error = std::isfinite(delta) ? std::max(error, delta) : INFINITY;
        }
        MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, grid.comm());
        maximum_error = std::max(maximum_error, error);
        require(std::isfinite(error) && error < 1e-10, "GPU headwing rank-one oracle", grid.comm());
    }
    std::sort(times.begin(), times.end());
    MPI_Allreduce(MPI_IN_PLACE, &packing, 1, MPI_DOUBLE, MPI_MAX, grid.comm());
    if (grid.myid == 0)
        std::cout << "HEADWING_STAGE n=" << n << " retained=" << retained << " block=" << block
                  << " native=" << native << " cholesky=" << cholesky
                  << " grid=" << grid.nprows << "x" << grid.npcols << " bytes_per_buffer=" << count * sizeof(Z)
                  << " pack_max=" << packing << " max_error=" << maximum_error
                  << " completed_median_max=" << times[times.size() / 2] << std::endl;
}

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    bool passed = true;
    try
    {
        BlacsCtxtHandler grid(MPI_COMM_WORLD);
        grid.init(); grid.set_square_grid(); grid.init_ddla_handle();
        if (argc > 1 && std::string(argv[1]) == "--cpu-ddla")
            grid.init_ddla_handle(ddla::DdlaBackend::CPU);
        const std::array<std::array<int,3>,8> shapes{{
            {{2,2,1}}, {{4,1,2}}, {{4,3,2}}, {{7,4,2}}, {{8,4,2}}, {{16,14,4}}, {{16,6,4}}, {{34,24,8}}}};
        for (bool native : {false, true}) for (bool chol : {false, true})
            for (const auto shape : shapes)
                run_case(grid, shape[0], shape[1], shape[2], 1, native, chol);
        if (grid.myid == 0) std::cout << "HEADWING_REWRITE_PASS" << std::endl;
    }
    catch (const std::exception& error) { std::cerr << error.what() << std::endl; passed = false; }
    MPI_Finalize();
    return passed ? 0 : 1;
}
