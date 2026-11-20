#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>

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

using ProbePhase = void (*)(const char*, int);
static ProbePhase get_probe_phase()
{
    const auto probe_phase = reinterpret_cast<void (*)(const char*, int)>(
        dlsym(RTLD_DEFAULT, "ddla_probe_phase"));
    if (std::getenv("DDLA_PROBE_DIR") && !probe_phase)
        throw std::runtime_error("DDLA probe phase hook not found through dlsym(RTLD_DEFAULT)");
    return probe_phase;
}

static void require(bool good, const char* message, MPI_Comm comm)
{
    int bad = !good;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) throw std::runtime_error(message);
}

static void run_case(BlacsCtxtHandler& grid, int n, int retained, int block,
                      int repeats, bool native)
{
    const auto probe_phase = get_probe_phase();
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
        if (probe_phase) probe_phase("headwing_native_pack", 1);
        const double begin = omp_get_wtime();
        ScalapackConnector::pgemr2d_f(n, n - retained, u.ptr(), 1, retained + 1,
            desc.desc, host_scratch.ptr(), 1, 1, fd.desc, grid.ictxt);
        packing = omp_get_wtime() - begin;
        if (probe_phase) probe_phase("headwing_native_pack", 0);
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
        const char* phase = native ? (repeat == 0 ? "headwing_native_warmup" : "headwing_native_measured")
                                   : (repeat == 0 ? "headwing_host_warmup" : "headwing_host_measured");
        if (probe_phase) probe_phase(phase, 1);
        const double begin = omp_get_wtime();
        // Include caller preparation and transfers before the common kernel,
        // then return the result to the outer device Coulomb products.
        if (eps.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, eps.ptr(), d0.data(), eps.size() * sizeof(Z)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        headwing_test::rewrite(eps, sqrtv, u, direction, hd, head, wing, qx, qy, qz, rho,
            desc, grid, retained, 0., native);
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
        if (probe_phase) probe_phase(phase, 0);
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
                  << " native=" << native
                  << " grid=" << grid.nprows << "x" << grid.npcols << " bytes_per_buffer=" << count * sizeof(Z)
                  << " pack_max=" << packing << " max_error=" << maximum_error
                  << " completed_median_max=" << times[times.size() / 2] << std::endl;
}

static void run_strict_projection(BlacsCtxtHandler& grid, int n, int retained,
                                   int block, int repeats, bool native)
{
    const auto probe_phase = get_probe_phase();
    const auto handle = grid.ddla_handle;
    ArrayDesc desc(grid);
    desc.init(n,n,block,block,0,0);
    desc.set_linear_algorithm("ddla"); desc.set_ddla_handle(handle);
    auto e=init_local_mat<Z>(desc,MAJOR::COL), u=e.copy(), host_scratch=e.copy();
    const bool dense_u=n<=8;
    const auto uv=[&](int i,int j) {
        return dense_u ? std::polar(1./std::sqrt(double(n)), TWO_PI*i*j/n)
                       : (i==j ? std::polar(1.,.031*i) : Z(0));
    };
    const auto ev=[&](int i,int j) { return Z((i==j?2.:0.)+.05/(1+i+j),.02*(i-2*j)/n); };
    const auto wc=[&](int i,int j) { return Z((i==j?.2:0.)+.03/(1+i+j),.01*(1+i+2*j)/n); };
    for(int j=0;j<desc.n_loc();++j) for(int i=0;i<desc.m_loc();++i)
        u(i,j)=uv(desc.indx_l2g_r(i),desc.indx_l2g_c(j));
    const size_t count=std::max<size_t>(1,size_t(desc.lld())*desc.n_loc());
    ddla::vector<Z> d0(handle,count),d1(handle,count),d2(handle,count);
    const auto rectangle=[&](bool to_host) {
        const int rows=ScalapackConnector::numroc(retained,block,desc.myprow(),0,desc.nprows());
        const int cols=ScalapackConnector::numroc(retained,block,desc.mypcol(),0,desc.npcols());
        const size_t pitch=size_t(desc.lld())*sizeof(Z);
        if(rows&&cols) ddla::DDLA_CHECK(ddla::ddlaMemcpy2D(handle,
            to_host?e.ptr():d0.data(),pitch,to_host?d0.data():e.ptr(),pitch,size_t(rows)*sizeof(Z),cols));
        if(to_host) ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    };
    std::vector<double> times;
    double maximum_error=0.;
    for(int repeat=0;repeat<=repeats;++repeat)
    {
        for(int j=0;j<desc.n_loc();++j) for(int i=0;i<desc.m_loc();++i)
            e(i,j)=ev(desc.indx_l2g_r(i),desc.indx_l2g_c(j));
        if(e.size()) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle,d0.data(),e.ptr(),e.size()*sizeof(Z)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        const char* phase=native?(repeat?"strict_native_measured":"strict_native_warmup")
                                 :(repeat?"strict_legacy_measured":"strict_legacy_warmup");
        if(probe_phase)probe_phase(phase,1);
        const double begin=omp_get_wtime();
        if(native)
        {
            if(u.size())ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle,d1.data(),u.ptr(),u.size()*sizeof(Z)));
            project_to_retained_coulomb_basis(d0.data(),d1.data(),d2.data(),retained,desc);
            rectangle(true);
        }
        else
        {
            if(e.size())ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle,e.ptr(),d0.data(),e.size()*sizeof(Z)));
            ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
            ScalapackConnector::pgemm_f('N','N',n,retained,n,Z(1),e.ptr(),1,1,desc.desc,
                u.ptr(),1,1,desc.desc,Z(0),host_scratch.ptr(),1,1,desc.desc);
            ScalapackConnector::pgemm_f('C','N',retained,retained,n,Z(1),u.ptr(),1,1,desc.desc,
                host_scratch.ptr(),1,1,desc.desc,Z(0),e.ptr(),1,1,desc.desc);
        }
        if(repeat==0)
        {
            double error=0.;
            for(int j=0;j<desc.n_loc();++j)for(int i=0;i<desc.m_loc();++i)
            {
                const int r=desc.indx_l2g_r(i),c=desc.indx_l2g_c(j);
                if(r>=retained||c>=retained)continue;
                Z expected{};
                if(dense_u)for(int k=0;k<n;++k)for(int l=0;l<n;++l)
                    expected+=std::conj(uv(k,r))*ev(k,l)*uv(l,c);
                else expected=std::conj(uv(r,r))*ev(r,c)*uv(c,c);
                const double delta=std::abs(e(i,j)-expected);
                error=std::isfinite(delta)?std::max(error,delta):INFINITY;
            }
            require(std::isfinite(error)&&error<1e-10,"strict forward complex projection oracle",grid.comm());
        }
        // A known complete retained Wc substitutes only the unchanged CPU
        // radial/body routine, isolating the production projection helpers.
        for(int j=0;j<desc.n_loc();++j)for(int i=0;i<desc.m_loc();++i)
        {
            const int r=desc.indx_l2g_r(i),c=desc.indx_l2g_c(j);
            e(i,j)=r<retained&&c<retained?wc(r,c):Z(0);
        }
        if(native)
        {
            rectangle(false);
            project_from_retained_coulomb_basis(d0.data(),d1.data(),d2.data(),retained,desc);
        }
        else
        {
            ScalapackConnector::pgemm_f('N','N',n,retained,retained,Z(1),u.ptr(),1,1,desc.desc,
                e.ptr(),1,1,desc.desc,Z(0),host_scratch.ptr(),1,1,desc.desc);
            ScalapackConnector::pgemm_f('N','C',n,n,retained,Z(1),host_scratch.ptr(),1,1,desc.desc,
                u.ptr(),1,1,desc.desc,Z(0),e.ptr(),1,1,desc.desc);
            if(e.size())ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle,d0.data(),e.ptr(),e.size()*sizeof(Z)));
        }
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        double elapsed=omp_get_wtime()-begin;
        if(probe_phase)probe_phase(phase,0);
        MPI_Allreduce(MPI_IN_PLACE,&elapsed,1,MPI_DOUBLE,MPI_MAX,grid.comm());
        if(repeat)times.push_back(elapsed);
        if(e.size())ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle,e.ptr(),d0.data(),e.size()*sizeof(Z)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        double error=0.;
        for(int j=0;j<desc.n_loc();++j)for(int i=0;i<desc.m_loc();++i)
        {
            const int r=desc.indx_l2g_r(i),c=desc.indx_l2g_c(j);
            Z expected{};
            if(dense_u)for(int k=0;k<retained;++k)for(int l=0;l<retained;++l)
                expected+=uv(r,k)*wc(k,l)*std::conj(uv(c,l));
            else if(r<retained&&c<retained)expected=uv(r,r)*wc(r,c)*std::conj(uv(c,c));
            const double delta=std::abs(e(i,j)-expected);
            error=std::isfinite(delta)?std::max(error,delta):INFINITY;
        }
        MPI_Allreduce(MPI_IN_PLACE,&error,1,MPI_DOUBLE,MPI_MAX,grid.comm());
        maximum_error=std::max(maximum_error,error);
        require(std::isfinite(error)&&error<1e-10,"strict backward complete-Wc projection oracle",grid.comm());
    }
    std::sort(times.begin(),times.end());
    if(grid.myid==0)std::cout<<"STRICT_PROJECTION_STAGE n="<<n<<" retained="<<retained
        <<" native="<<native<<" cpu_body=known_fixture max_error="<<maximum_error
        <<" completed_median_max="<<times[times.size()/2]<<std::endl;
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
        if (argc > 1 && std::string(argv[1]) == "--bench")
        {
            if (argc != 7) throw std::invalid_argument("--bench N retained block repeats native");
            const int n=std::stoi(argv[2]), r=std::stoi(argv[3]), b=std::stoi(argv[4]), count=std::stoi(argv[5]);
            if (n < 2 || r < 1 || r > n || b < 1 || count < 1) throw std::invalid_argument("invalid benchmark dimensions");
            run_case(grid, n, r, b, count, std::stoi(argv[6]) != 0);
        }
        else if(argc>1&&std::string(argv[1])=="--strict-bench")
        {
            if(argc!=7)throw std::invalid_argument("--strict-bench N retained block repeats native");
            const int n=std::stoi(argv[2]),r=std::stoi(argv[3]),b=std::stoi(argv[4]),count=std::stoi(argv[5]);
            if(n<2||r<0||r>n||b<1||count<1)throw std::invalid_argument("invalid strict benchmark dimensions");
            run_strict_projection(grid,n,r,b,count,std::stoi(argv[6])!=0);
        }
        else
        {
            const std::array<std::array<int,3>,8> shapes{{
                {{2,2,1}}, {{4,1,2}}, {{4,3,2}}, {{7,4,2}}, {{8,4,2}}, {{16,14,4}}, {{16,6,4}}, {{34,24,8}}}};
            for (bool native : {false, true})
                for (const auto shape : shapes)
                    run_case(grid, shape[0], shape[1], shape[2], 1, native);
            for(bool native:{false,true})for(int retained:{0,1,3,4})
                run_strict_projection(grid,4,retained,2,1,native);
        }
        if (grid.myid == 0) std::cout << "HEADWING_REWRITE_PASS" << std::endl;
    }
    catch (const std::exception& error) { std::cerr << error.what() << std::endl; passed = false; }
    MPI_Finalize();
    return passed ? 0 : 1;
}
