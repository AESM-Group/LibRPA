#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <ddla/ddla_vector.h>

#include "../interface/la_connector.h"

using namespace librpa_int;
using Z = std::complex<double>;

static void require(bool condition, const char* message, MPI_Comm comm)
{
    int bad = condition ? 0 : 1;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, comm);
    if (bad) throw std::runtime_error(message);
}

static void check_grid(MPI_Comm comm, bool gpu)
{
    BlacsCtxtHandler grid(comm);
    grid.init();
    grid.set_square_grid();
    grid.init_ddla_handle(gpu ? ddla::DdlaBackend::GPU : ddla::DdlaBackend::CPU);
    if (gpu) grid.init_ddla_handle(ddla::DdlaBackend::CPU);
    const auto handle = gpu ? grid.ddla_handle : grid.ddla_cpu_handle;
    ArrayDesc desc(grid);
    desc.init(1, 1, 2, 2, 0, 0);
    desc.set_ddla_handle(handle);
    desc.set_linear_algorithm("ddla");
    // Empty owners really supply nullptr, rather than max(1,...) allocations.
    const size_t count = size_t(desc.m_loc()) * desc.n_loc();
    ddla::vector<double> real_a(handle, count);
    ddla::vector<Z> a(handle, count), b(handle, count), c(handle, count);
    const double r = 2.;
    const Z av(2), bv(3, 1), initial(1, -1);
    auto upload = [&](auto& target, const auto& value)
    {
        if (count) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, target.data(), &value, sizeof(value)));
    };
    auto check = [&](Z expected)
    {
        Z actual = expected;
        if (count) ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, &actual, c.data(), sizeof(actual)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
        require(std::isfinite(std::abs(actual)) && std::abs(actual - expected) < 1e-12,
                "empty-owner GEMM value", comm);
    };
    upload(real_a, r); upload(a, av); upload(b, bv);
    LaConnector::pgemm('N', 'N', 1, 1, 1, Z(1), real_a.data(), 1, 1, desc,
                       b.data(), 1, 1, desc, Z(0), c.data(), 1, 1, desc);
    check(av * bv);
    LaConnector::pgemm('N', 'N', 1, 1, 1, Z(1), a.data(), 1, 1, desc,
                       b.data(), 1, 1, desc, Z(0), c.data(), 1, 1, desc);
    check(av * bv);
    for (bool mixed : {false, true})
    for (int k : {0, 1})
    {
        upload(c, initial);
        // k=0 alone, and alpha=0 alone, each permits absent input arrays.
        const Z alpha = k == 0 ? Z(1) : Z(0);
        if (mixed)
            LaConnector::pgemm('N', 'N', 1, 1, k, alpha,
                static_cast<const double*>(nullptr), 1, 1, desc,
                static_cast<const Z*>(nullptr), 1, 1, desc, Z(2), c.data(), 1, 1, desc);
        else
            LaConnector::pgemm('N', 'N', 1, 1, k, alpha,
                static_cast<const Z*>(nullptr), 1, 1, desc,
                static_cast<const Z*>(nullptr), 1, 1, desc, Z(2), c.data(), 1, 1, desc);
        check(Z(2) * initial);
    }
    bool rejected = false;
    try
    {
        LaConnector::pgemm('N', 'N', 1, 1, 1, Z(1), real_a.data(), 2, 1, desc,
                           b.data(), 1, 1, desc, Z(0), c.data(), 1, 1, desc);
    }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "mixed offset rejection", comm);
    if (gpu)
    {
        rejected = false;
        try
        {
            LaConnector::pgemm('N', 'N', 1, 1, 1, Z(1), a.data(), 2, 1, desc,
                               b.data(), 1, 1, desc, Z(0), c.data(), 1, 1, desc);
        }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "GPU homogeneous offset rejection", comm);

        // The caller selects a host backend for host arrays, even when the
        // owning grid also has a GPU handle.
        ArrayDesc host_desc = desc;
        host_desc.set_ddla_handle(grid.ddla_cpu_handle);
        Z host_c(0);
        for (const char* method : {"scalapack", "ddla"})
        {
            host_desc.set_linear_algorithm(method);
            host_c = Z(0);
            LaConnector::pgemm('N', 'N', 1, 1, 1, Z(1), count ? &av : nullptr, 1, 1, host_desc,
                               count ? &bv : nullptr, 1, 1, host_desc, Z(0), count ? &host_c : nullptr,
                               1, 1, host_desc);
            require(!count || std::abs(host_c - av * bv) < 1e-12, "explicit host GEMM backend", comm);
        }
        // Even without any used operands, the selected backend remains GPU.
        // It must not fall back to a host GEMM accepting nonunit offsets.
        ArrayDesc no_op_desc = desc;
        no_op_desc.init(3, 3, 1, 1, 0, 0);
        rejected = false;
        try
        {
            LaConnector::pgemm('N', 'N', 1, 1, 0, Z(0),
                static_cast<const Z*>(nullptr), 1, 1, no_op_desc,
                static_cast<const Z*>(nullptr), 1, 1, no_op_desc, Z(1),
                static_cast<Z*>(nullptr), 1, 2, no_op_desc);
        }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "GPU no-op GEMM must retain its selected backend", comm);
    }

    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
}

int main(int argc, char** argv)
{
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    bool passed = true;
    try
    {
        const bool gpu = argc > 1 && std::string(argv[1]) == "--gpu";
        check_grid(MPI_COMM_WORLD, gpu);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm subgroup;
        MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &subgroup);
        check_grid(subgroup, gpu);
        MPI_Comm_free(&subgroup);
        if (rank == 0) std::cout << "DDLA_EMPTY_CONNECTOR_PASS" << std::endl;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << std::endl;
        passed = false;
    }
    MPI_Finalize();
    return passed ? 0 : 1;
}
