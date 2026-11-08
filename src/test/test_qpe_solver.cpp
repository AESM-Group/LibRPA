#ifdef NDEBUG
#undef NDEBUG
#endif

#include "../core/qpe_solver.h"
#include "../core/qsgw/qp_diagnostics.h"
#include "librpa.h"
#include <mpi.h>

#include <cassert>
#include <vector>

#include "testutils.h"

using namespace librpa_int;

static cplxdb single_pole_self_energy(const cplxdb &x, const cplxdb &x0,
                                      const cplxdb &strength = {-1.0, 0.0})
{
    return strength / (x - x0);
}

static cplxdb single_pole_self_energy_derivative(const cplxdb &x, const cplxdb &x0,
                                                 const cplxdb &strength = {-1.0, 0.0})
{
    return -strength / ((x - x0) * (x - x0));
}

static void initialize_single_pole_data(int n, const cplxdb &x0, std::vector<cplxdb> &xs,
                                        std::vector<cplxdb> &data,
                                        const cplxdb &strength = {-1.0, 0.0})
{
    xs.resize(n);
    data.resize(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, static_cast<double>(i+1)};
        data[i] = single_pole_self_energy(xs[i], x0, strength);
    }
}

static void initialize_linear_data(int n, const cplxdb &slope, const cplxdb &intercept,
                                   std::vector<cplxdb> &xs, std::vector<cplxdb> &data)
{
    xs.resize(n);
    data.resize(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, static_cast<double>(i+1)};
        data[i] = slope * xs[i] + intercept;
    }
}

void check_single_pole_self_energy(const bool use_adaptive_damp)
{
    constexpr int nfreq = 6;
    constexpr double e_mf = 0.5;
    constexpr double e_fermi = 0.2;
    constexpr double sigma_x = 0.3;
    constexpr double e_qp_ref = 0.75;
    const cplxdb x0 = {2.0, 2.0};
    const cplxdb strength = {-0.5, 0.0};
    const cplxdb sigc_ref = single_pole_self_energy(e_qp_ref - e_fermi, x0, strength);
    const double vxc = e_mf + sigma_x + sigc_ref.real() - e_qp_ref;

    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_single_pole_data(nfreq, x0, xs, data, strength);
    AnalyContPade pade(nfreq, xs, data);

    double e_qp = 0.0;
    cplxdb sigc;
    const int info = qpe_solver_pade_self_consistent(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 1.0e-3, 1.0e-8, 200, 0.5,
        use_adaptive_damp);

    assert(info == 0);
    assert(fequal(e_qp, e_qp_ref, 1.0e-8));
    assert(std::abs(sigc - sigc_ref) < 1.0e-8);
}

void test_quasi_newton_uses_pade_derivative()
{
    constexpr int nfreq = 4;
    constexpr double e_mf = 0.5;
    constexpr double e_fermi = 0.2;
    constexpr double sigma_x = 0.1;
    constexpr double e_qp_ref = 0.8;
    const cplxdb slope = {2.0, 0.0};
    const cplxdb intercept = {0.05, 0.0};
    const cplxdb sigc_ref = slope * cplxdb{e_qp_ref - e_fermi, 0.0} + intercept;
    const double vxc = e_mf + sigma_x + sigc_ref.real() - e_qp_ref;

    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_linear_data(nfreq, slope, intercept, xs, data);
    AnalyContPade pade(nfreq, xs, data);

    double e_qp = 0.0;
    cplxdb sigc;
    const int info = qpe_solver_pade_quasi_newton(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 0.0, 1.0e-10, 8, 1.0,
        false);

    assert(info == 0);
    assert(fequal(e_qp, e_qp_ref, 1.0e-10));
    assert(std::abs(sigc - sigc_ref) < 1.0e-10);
}

void test_nonadaptive_residual_mixing_keeps_legacy_entry()
{
    constexpr int nfreq = 4;
    constexpr double e_mf = 0.0;
    constexpr double e_fermi = 0.0;
    constexpr double vxc = 0.0;
    constexpr double sigma_x = 0.0;
    const cplxdb slope = {2.0, 0.0};
    const cplxdb intercept = {0.0, 0.0};

    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_linear_data(nfreq, slope, intercept, xs, data);
    AnalyContPade pade(nfreq, xs, data);

    double e_qp = 0.0;
    cplxdb sigc;
    const int info = qpe_solver_pade_self_consistent(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 1.0, 1.0e-12, 2, 0.1,
        false);

    assert(info != 0);
    assert(fequal(e_qp, 0.22, 1.0e-12));
    assert(fequal(sigc.real(), 0.22, 1.0e-12));

    const int legacy_info = qpe_solver_pade_self_consistent(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 1.0, 1.0e-12, 2, 0.1,
        false, true);

    assert(legacy_info != 0);
    assert(fequal(e_qp, 0.42, 1.0e-12));
    assert(fequal(sigc.real(), 0.42, 1.0e-12));
}

void test_perturbative_qp_weight()
{
    constexpr int nfreq = 6;
    constexpr double e_mf = 0.5;
    constexpr double e_fermi = 0.2;
    constexpr double vxc = 0.1;
    constexpr double sigma_x = 0.3;
    const cplxdb x0 = {2.0, 2.0};

    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_single_pole_data(nfreq, x0, xs, data);
    AnalyContPade pade(nfreq, xs, data);

    double e_qp = 0.0;
    cplxdb sigc;
    cplxdb sigc_deriv;
    double qp_weight = 0.0;
    const int info = qpe_solver_pade_perturbative(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, sigc_deriv, qp_weight);

    const cplxdb omega = e_mf - e_fermi;
    const cplxdb sigc_ref = single_pole_self_energy(omega, x0);
    const cplxdb sigc_deriv_ref = single_pole_self_energy_derivative(omega, x0);
    const double qp_weight_ref = 1.0 / (1.0 - sigc_deriv_ref.real());
    const double e_qp_ref = e_mf + qp_weight_ref * (sigma_x + sigc_ref.real() - vxc);

    assert(info == 0);
    assert(std::abs(sigc - sigc_ref) < 1.0e-10);
    assert(std::abs(sigc_deriv - sigc_deriv_ref) < 1.0e-10);
    assert(fequal(qp_weight, qp_weight_ref, 1.0e-10));
    assert(fequal(e_qp, e_qp_ref, 1.0e-10));
}

void test_adaptive_damp_residual_mixing_branch_hop_case()
{
    constexpr double ha2ev = 27.211386245988;
    constexpr double e_mf = 224.36314 / ha2ev;
    constexpr double e_fermi = 0.5857220000;
    constexpr double vxc = -14.27774 / ha2ev;
    constexpr double sigma_x = -3.38758 / ha2ev;

    std::vector<cplxdb> xs = {
        {0.0, 4.69249941085e-02},
        {0.0, 1.58112711805e-01},
        {0.0, 3.30595530175e-01},
        {0.0, 6.41469454789e-01},
        {0.0, 1.24723699203e+00},
        {0.0, 2.49729647300e+00},
        {0.0, 5.30291176215e+00},
        {0.0, 1.35255767277e+01},
    };
    std::vector<cplxdb> data = {
        {-1.37547254497e+00, -6.41621915982e-03},
        {-1.37287749437e+00, -4.96138067860e-02},
        {-1.36838585253e+00, -8.16399081261e-02},
        {-1.34663567190e+00, -1.74526824216e-01},
        {-1.28720864939e+00, -2.89219092788e-01},
        {-1.11524539502e+00, -4.89663540520e-01},
        {-7.60085551584e-01, -6.05643169055e-01},
        {-2.84673157366e-01, -5.08369524160e-01},
    };

    AnalyContPade pade(xs.size(), xs, data);
    double e_qp = 0.0;
    cplxdb sigc;
    const int info = qpe_solver_pade_self_consistent(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 1.0e-3, 1.0e-5, 200, 0.1, true);

    assert(info == 0);
    assert(std::isfinite(e_qp));
    assert(e_qp > 5.0 && e_qp < 6.5);
}

void test_adaptive_damp_recovers_from_min_damp_stall()
{
    constexpr double ha2ev = 27.211386245988;
    constexpr double e_mf = 63.40778 / ha2ev;
    constexpr double e_fermi = -0.1446680000;
    constexpr double vxc = -15.74190 / ha2ev;
    constexpr double sigma_x = -46.18735 / ha2ev;
    constexpr double e_qp_ref = -0.18386 / ha2ev;

    std::vector<cplxdb> xs = {
        {0.0, 1.23121555764e-02},
        {0.0, 4.67409978218e-02},
        {0.0, 1.23673684837e-01},
        {0.0, 3.39328839931e-01},
        {0.0, 1.05932755882e+00},
        {0.0, 4.20234445929e+00},
    };
    std::vector<cplxdb> data = {
        {-9.48990931382e-01,  3.19495393947e-03},
        {-9.45914515876e-01, -4.50290766015e-02},
        {-9.40595421040e-01, -6.89006000930e-02},
        {-8.92995931857e-01, -2.09702298175e-01},
        {-7.10480738080e-01, -3.59437112159e-01},
        {-2.78410350069e-01, -4.08374497370e-01},
    };

    AnalyContPade pade(xs.size(), xs, data);
    double e_qp = 0.0;
    cplxdb sigc;
    const int info = qpe_solver_pade_self_consistent(
        pade, e_mf, e_fermi, vxc, sigma_x, e_qp, sigc, 1.0e-3, 1.0e-6, 10000, 0.1,
        true);

    assert(info == 0);
    assert(std::abs(e_qp - e_qp_ref) < 1.0e-3);
}

void test_perturbative_zero_self_energy()
{
    const std::vector<cplxdb> xs = {{0.0, 1.0}, {0.0, 2.0}, {0.0, 4.0}, {0.0, 8.0}};
    const AnalyContPade pade(xs.size(), xs, std::vector<cplxdb>(xs.size(), 0.0));
    double energy, weight;
    cplxdb sigma, derivative;
    const int info =
        qpe_solver_pade_perturbative(pade, 0.5, 0.2, 0.1, 0.3, energy, sigma, derivative, weight);
    assert(info == 0);
    assert(std::abs(energy - 0.7) < 1.0e-14);
    assert(weight == 1.0);
    assert(sigma == cplxdb(0.0, 0.0));
    assert(derivative == cplxdb(0.0, 0.0));
}

void test_selected_method_qp_roots()
{
    const std::vector<cplxdb> xs{{0, 0.2}, {0, 0.5}, {0, 1}, {0, 2}, {0, 4}};
    std::vector<cplxdb> data;
    for (const auto z : xs) data.push_back(0.1 * z + 0.2);
    const double expected = 0.278 / 0.9;
    for (const auto method : {LIBRPA_ANACON_AAA, LIBRPA_ANACON_RIDGE})
    {
        const auto fit = make_analycont(method, 3, xs, data);
        double eqp = 0.0, weight = 0.0;
        cplxdb sigc, derivative;
        const int status =
            qpe_solver_pade_quasi_newton(*fit, 0.1, 0.02, 0.05, 0.03, eqp, sigc, 0.0, 1e-10, 100);
        assert(status == 0);
        assert(std::abs(eqp - expected) < 2e-7);
        assert(qpe_solver_pade_perturbative(*fit, 0.1, 0.02, 0.05, 0.03, eqp, sigc, derivative,
                                            weight) == 0);
        assert(std::abs(eqp - expected) < 2e-7);
        assert(std::abs(weight - 1.0 / 0.9) < 2e-7);
    }
}

// A real-axis pole in one rank's state must not suppress the next valid QP.
void test_qp_continuation_failure_is_local()
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MeanField mf(1, 1, 2, 2);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = 0.5;
    mf.get_eigenvals()[0](0, 1) = 0.1;
    TFGrids tfg(6);
    tfg.generate(LIBRPA_TFGRID_GAUSS_LEGENDRE);
    std::vector<cplxdb> sigma;
    for (const double freq : tfg.get_freq_nodes())
    {
        sigma.push_back(rank == 0 ? 1.0 / cplxdb{-0.5, freq} : cplxdb{0.2, 0});
        sigma.push_back({0.2, 0});
    }
    LibrpaOptions opts;
    librpa_init_options(&opts);
    opts.anacon_method = LIBRPA_ANACON_AAA;
    opts.n_params_anacon = 3;
    opts.option_qpe_solver = 2;
    const int ik = 0;
    const double vxc[2]{0, 0}, vexx[2]{0, 0};
    for (const auto override_nan : {LIBRPA_SWITCH_OFF, LIBRPA_SWITCH_ON})
    {
        opts.override_qpe_solver_nan = override_nan;
        for (const bool shared_shift : {false, true})
        {
            opts.use_hedin_shift = shared_shift ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
            opts.istate_ref_hedin_shift = 0;
            double re[2], im[2], energy[2];
            int info[2]{-1, -1};
            librpa_int::qsgw::solve_qsgw_qpe_diagonal(
                mf, tfg, opts, sigma, {0}, 1, &ik, 0, 2, vxc, vexx,
                re, im, energy, "pole control", info);
            if (rank == 0)
            {
                assert(std::isnan(energy[0]) && std::isnan(re[0]) && std::isnan(im[0]));
                assert(info[0] != 0);
            }
            else
            {
                assert(std::abs(energy[0] - 0.7) < 1e-12 && info[0] == 0);
            }
            if (rank == 0 && shared_shift)
            {
                assert(std::isnan(energy[1]) && info[1] != 0);
            }
            else
            {
                assert(std::abs(energy[1] - 0.3) < 1e-12 && info[1] == 0);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    test_qp_continuation_failure_is_local();
    test_selected_method_qp_roots();
    check_single_pole_self_energy(false);
    check_single_pole_self_energy(true);
    test_quasi_newton_uses_pade_derivative();
    test_nonadaptive_residual_mixing_keeps_legacy_entry();
    test_perturbative_qp_weight();
    test_adaptive_damp_residual_mixing_branch_hop_case();
    test_adaptive_damp_recovers_from_min_damp_stall();
    test_perturbative_zero_self_energy();
    MPI_Finalize();
}
