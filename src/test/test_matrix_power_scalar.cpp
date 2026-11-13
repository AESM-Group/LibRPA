#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "../api/dataset.h"
#include "../api/dataset_helper.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/global_mpi.h"
#ifdef LIBRPA_USE_DDLA
#include <ddla/ddla_vector.h>
#endif

using namespace librpa_int;

namespace
{
void check_real_syev()
{
    Dataset ds{MPI_COMM_WORLD};
    ArrayDesc ad(ds.blacs_h.ictxt);
    constexpr int n = 7;
    ad.init(n, n, 2, 2, 0, 0);
    auto a = init_local_mat<double>(ad, MAJOR::COL);
    auto z = init_local_mat<double>(ad, MAJOR::COL);
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
            if (ad.indx_l2g_r(i) == ad.indx_l2g_c(j))
                a(i, j) = 2.0 + ad.indx_l2g_r(i);

    double query = 0, w[n]{};
    int info = 12345;
    ScalapackConnector::psyev_f('V', 'U', n, a.ptr(), 1, 1, ad.desc, w, z.ptr(), 1, 1,
                                ad.desc, &query, -1, info);
    if (info != 0 || !std::isfinite(query) || !(query > 0))
        throw std::runtime_error("real SYEV query did not return INFO/workspace");

    // Exercise both query and solve in the production real matrix-power path.
    auto powered = a.to_complex();
    auto vectors = init_local_mat<std::complex<double>>(ad, MAJOR::COL);
    size_t filtered = 0;
    LaConnector::power_hemat_la_real(powered, ad, vectors, ad, filtered, w, 0.5, 0.0);
    if (filtered != 0) throw std::runtime_error("positive real eigenvalues were filtered");
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
        {
            const int row = ad.indx_l2g_r(i), col = ad.indx_l2g_c(j);
            const double expected = row == col ? std::sqrt(2.0 + row) : 0.0;
            const double error = std::abs(powered(i, j) - expected);
            if (!std::isfinite(error) || error > 2e-12)
                throw std::runtime_error("real distributed matrix power mismatch");
        }
}

void close(std::complex<double> actual, std::complex<double> expected)
{
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(actual - expected) > 2e-14)
        throw std::runtime_error("scalar distributed matrix power mismatch");
}

void check(double eigenvalue, double power, double threshold, double expected)
{
    Dataset ds{MPI_COMM_WORLD};
    // Use the raw context constructor: the operation needs only the BLACS grid.
    ArrayDesc ad_a(ds.blacs_h.ictxt), ad_z(ds.blacs_h.ictxt);
    // Exercise redistribution from a nonzero owner and back to a different Z owner.
    ad_a.init(1, 1, 1, 1, ds.blacs_h.nprows - 1, ds.blacs_h.npcols - 1);
    ad_z.init(1, 1, 1, 1, 0, 0);
    auto a = init_local_mat<std::complex<double>>(ad_a, MAJOR::COL);
    auto z = init_local_mat<std::complex<double>>(ad_z, MAJOR::COL);
    if (ad_a.is_src()) a(0, 0) = eigenvalue;
    double w = 0;
    std::size_t filtered = 0;
    const auto scaled = LaConnector::power_hemat_la(a, ad_a, z, ad_z, filtered, &w, power, threshold);
    close(w, eigenvalue);
    if (filtered != std::size_t(eigenvalue < threshold))
        throw std::runtime_error("scalar distributed eigenvalue filtering mismatch");
    if (ad_a.is_src()) close(a(0, 0), expected);
    if (ad_z.is_src())
    {
        close(z(0, 0), 1);
        close(scaled(0, 0), expected);
    }
}
}  // namespace

// c I + v v^H has eigenvalues c and c + ||v||^2. Filtering c leaves
// (c + ||v||^2)^power v v^H / ||v||^2, independently of eigenvector phases.
void check_rank_one(const std::string& method, bool gpu, bool real, double power,
                    const std::string& linear)
{
    Dataset ds{MPI_COMM_WORLD};
    ArrayDesc ad(ds.blacs_h.ictxt);
    constexpr int n = 7;
    ad.init(n, n, 2, 2, 0, 0);
    ad.set_linear_algorithm(linear);
#ifdef LIBRPA_USE_DDLA
    if (method == "ddla" || linear == "ddla" || gpu)
    {
        LibrpaOptions opts;
        librpa_init_options(&opts);
        std::strncpy(opts.eigensolver_method, method.c_str(), LIBRPA_MAX_STRLEN);
        std::strncpy(opts.linear_algorithm, linear.c_str(), LIBRPA_MAX_STRLEN);
        opts.use_gpu_replace_scalapack = gpu ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
        initialize_ds_global_ddla(ds, opts);
        const auto handle = gpu ? ds.blacs_h.ddla_handle : ds.blacs_h.ddla_cpu_handle;
        initialize_ds_global_ddla(ds, opts);
        if (handle == nullptr ||
            handle != (gpu ? ds.blacs_h.ddla_handle : ds.blacs_h.ddla_cpu_handle))
            throw std::runtime_error("outer DDLA handle was not reused");
        ad.set_ddla_handle(handle);
    }
#endif
#ifdef LIBRPA_USE_ELPA
    if (method == "elpa") ad.set_elpa_handle(gpu);
#endif
    auto a = init_local_mat<std::complex<double>>(ad, MAJOR::COL);
    auto z = init_local_mat<std::complex<double>>(ad, MAJOR::COL);
    auto v = [real](int i) { return std::complex<double>(1.0 + i, real ? 0.0 : 0.2 * i); };
    double norm = 0;
    for (int i = 0; i < n; ++i) norm += std::norm(v(i));
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
        {
            int r = ad.indx_l2g_r(i), c = ad.indx_l2g_c(j);
            a(i, j) = v(r) * std::conj(v(c)) + (r == c ? 2.0 : 0.0);
        }
    double w[n];
    size_t filtered = 0;
    const auto a_ptr = a.ptr(), z_ptr = z.ptr();
    const auto compute_power = [&](std::complex<double>* d_A = nullptr,
                                   std::complex<double>* d_Z = nullptr,
                                   std::complex<double>* d_scaled = nullptr)
    {
        return real ? LaConnector::power_hemat_la_real(
                          a, ad, z, ad, filtered, w, power, 2.5, gpu, method,
                          reinterpret_cast<double*>(d_A), reinterpret_cast<double*>(d_Z),
                          reinterpret_cast<double*>(d_scaled))
                    : LaConnector::power_hemat_la(a, ad, z, ad, filtered, w, power, 2.5, gpu,
                                                 method, d_A, d_Z, d_scaled);
    };
    matrix_m<std::complex<double>> scaled;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (gpu)
    {
        const auto handle = ad.ddla_handle();
        ddla::DDLA_CHECK(ddla::ddlaSetDevice(handle));
        const size_t count = std::max<size_t>(1, as_size(ad.lld()) * ad.n_loc());
        ddla::vector<std::complex<double>> device_a(handle, count), device_z(handle, count),
            device_scaled(handle, count);
        bool rejected_missing_buffers = false;
        try { compute_power(); }
        catch (const std::invalid_argument&) { rejected_missing_buffers = true; }
        if (!rejected_missing_buffers)
            throw std::runtime_error("GPU matrix power accepted missing buffers");
        scaled = compute_power(device_a.data(), device_z.data(), device_scaled.data());
    }
    else
#endif
        scaled = compute_power();
    if (a.ptr() != a_ptr || z.ptr() != z_ptr)
        throw std::runtime_error("matrix-power output storage was replaced");
    if (filtered != n - 1) throw std::runtime_error("rank-one eigenvalue filtering mismatch");
    std::vector<std::complex<double>> dots(n), global_dots(n);
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
            dots[ad.indx_l2g_c(j)] += std::conj(v(ad.indx_l2g_r(i))) * z(i, j);
    MPI_Allreduce(dots.data(), global_dots.data(), n, MPI_C_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);
    double error = 0;
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
        {
            int r = ad.indx_l2g_r(i), c = ad.indx_l2g_c(j);
            auto expected = std::pow(2.0 + norm, power) * v(r) * std::conj(v(c)) / norm;
            const double weight = w[c] < 2.5 ? 0.0 : std::pow(w[c], power);
            for (double e :
                 {std::abs(a(i, j) - expected), std::abs(scaled(i, j) - weight * z(i, j)),
                  std::abs(2.0 * z(i, j) + v(r) * global_dots[c] - w[c] * z(i, j))})
                error = !std::isfinite(e) ? INFINITY : std::max(error, e);
        }
    double max_error = 0;
    MPI_Allreduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!std::isfinite(max_error) || max_error > 2e-10)
        throw std::runtime_error("analytic distributed matrix power mismatch: " +
                                 std::to_string(max_error));
}

template <typename Scalar>
void check_pheev(const std::string& method, bool gpu)
{
    Dataset ds{MPI_COMM_WORLD};
    ArrayDesc ad(ds.blacs_h.ictxt);
    constexpr int n = 7;
    ad.init(n, n, 2, 2, 0, 0);
#ifdef LIBRPA_USE_DDLA
    if (method == "ddla")
    {
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        ds.blacs_h.init_ddla_handle(gpu ? ddla::DdlaBackend::GPU : ddla::DdlaBackend::CPU);
#else
        ds.blacs_h.init_ddla_handle();
        if (gpu || ddla::ddlaGetBackend(ds.blacs_h.ddla_cpu_handle) != ddla::DdlaBackend::CPU)
            throw std::runtime_error("CPU-only DDLA build did not initialize a CPU handle");
#endif
        ad.set_ddla_handle(gpu ? ds.blacs_h.ddla_handle : ds.blacs_h.ddla_cpu_handle);
    }
#endif
#ifdef LIBRPA_USE_ELPA
    if (method == "elpa") ad.set_elpa_handle(false);
#endif
    auto a = init_local_mat<Scalar>(ad, MAJOR::COL);
    auto z = init_local_mat<Scalar>(ad, MAJOR::COL);
    auto v = [](int i) -> Scalar
    {
        if constexpr (is_complex_t<Scalar>::value)
            return Scalar(1.0 + i, 0.2 * i);
        else
            return 1.0 + i;
    };
    double norm = 0;
    for (int i = 0; i < n; ++i) norm += std::norm(std::complex<double>(v(i)));
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
        {
            int row = ad.indx_l2g_r(i), col = ad.indx_l2g_c(j);
            a(i, j) = v(row) * get_conj(v(col)) + (row == col ? 2.0 : 0.0);
        }
    double w[n]{};
    int info = 12345;
    if constexpr (!is_complex_t<Scalar>::value)
    {
        if (method == "scalapack")
        {
            double query = 0;
            ScalapackConnector::psyev_f('V', 'U', n, a.ptr(), 1, 1, ad.desc, w, z.ptr(), 1, 1,
                                        ad.desc, &query, -1, info);
            if (info != 0 || !(query > 0))
                throw std::runtime_error("real SYEV query did not return INFO/workspace");
        }
    }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (method == "ddla" && gpu)
    {
        const auto handle = ad.ddla_handle();
        ddla::DDLA_CHECK(ddla::ddlaSetDevice(handle));
        ddla::vector<Scalar> da(handle, std::max<size_t>(1, a.size())),
            dz(handle, std::max<size_t>(1, z.size()));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, da.data(), a.ptr(), a.size() * sizeof(Scalar)));
        LaConnector::pheev('V', 'U', n, da.data(), 1, 1, ad, w, dz.data(), 1, 1, ad, info, method);
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, z.ptr(), dz.data(), z.size() * sizeof(Scalar)));
        ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    }
    else
#endif
        LaConnector::pheev('V', 'U', n, a.ptr(), 1, 1, ad, w, z.ptr(), 1, 1, ad, info, method);
    if (info != 0) throw std::runtime_error("raw-pointer pheev failed");
    double error = 0;
    std::vector<std::complex<double>> dots(n), all_dots(n);
    std::vector<double> norms(n), all_norms(n);
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
        {
            int row = ad.indx_l2g_r(i), col = ad.indx_l2g_c(j);
            dots[col] += std::conj(std::complex<double>(v(row))) * std::complex<double>(z(i, j));
            norms[col] += std::norm(std::complex<double>(z(i, j)));
        }
    MPI_Allreduce(dots.data(), all_dots.data(), n, MPI_C_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(norms.data(), all_norms.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    auto accumulate = [&](double e) { error = !std::isfinite(e) ? INFINITY : std::max(error, e); };
    for (int j = 0; j < n; ++j)
    {
        accumulate(std::abs(w[j] - (j == n - 1 ? 2.0 + norm : 2.0)));
        accumulate(std::abs(all_norms[j] - 1.0));
    }
    for (int j = 0; j < ad.n_loc(); ++j)
        for (int i = 0; i < ad.m_loc(); ++i)
        {
            int row = ad.indx_l2g_r(i), col = ad.indx_l2g_c(j);
            accumulate(std::abs((2.0 - w[col]) * std::complex<double>(z(i, j)) +
                                std::complex<double>(v(row)) * all_dots[col]));
        }
    double max_error = 0;
    MPI_Allreduce(&error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (!std::isfinite(max_error) || max_error > 2e-10)
        throw std::runtime_error("raw-pointer pheev eigenpair mismatch");
}

int main(int argc, char** argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
#ifdef LIBRPA_USE_ELPA
    if (elpa_init(ELPA_API_VERSION) != ELPA_OK) throw std::runtime_error("elpa_init failed");
#endif
    int status = 0;
    try
    {
        const std::string method = argc > 1 ? argv[1] : "scalapack";
        const bool gpu = argc > 2 && std::string(argv[2]) == "gpu";
        const std::string linear = argc > 3 ? argv[3] : "scalapack";
        LibrpaOptions opts;
        librpa_init_options(&opts);
        std::strncpy(opts.eigensolver_method, method.c_str(), LIBRPA_MAX_STRLEN);
        check_pheev<double>(method, gpu);
        check_pheev<std::complex<double>>(method, gpu);
        check_real_syev();
        for (bool real : {false, true})
            for (double power : {0.5, -1.0}) check_rank_one(method, gpu, real, power, linear);
        check(2, 0.5, 0, std::sqrt(2.0));
        check(-0.4, 0.5, 0, 0);
        check(4, -1, 0, 0.25);
        check(-0.4, 2, -1, 0.16);
        if (global::mpi_comm_global_h.is_root()) std::cout << "scalar matrix powers passed\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        status = 1;
    }
    global::finalize_global_io();
#ifdef LIBRPA_USE_ELPA
    int elpa_error;
    elpa_uninit(&elpa_error);
#endif
    global::finalize_global_mpi();
    MPI_Finalize();
    return status;
}
