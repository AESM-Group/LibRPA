#include "qpe_solver.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
// #include <iomanip>

#include "../io/global_io.h"

namespace librpa_int
{

namespace
{

struct QpeFixedPointIteration
{
    int iter;
    double e_trial;
    double e_qp;
    cplxdb sigc;
    double diff;
    double damp;
};

struct QpeQuasiNewtonIteration
{
    int iter;
    double e_trial;
    double e_qp;
    cplxdb sigc;
    cplxdb sigc_deriv;
    double diff;
    double newton_step;
    double damp;
};

struct QpeEvaluation
{
    double energy;
    cplxdb sigc;
    cplxdb sigc_deriv;
    double residual;
    double abs_residual;
};

constexpr int n_qpe_iters_to_dump = 10;
constexpr int n_qpe_max_backtrack = 12;
constexpr double qpe_damp_min = 1.0e-4;
constexpr double qpe_damp_max = 1.0;
constexpr double qpe_damp_shrink = 0.5;
constexpr double qpe_damp_grow = 1.25;
constexpr double qpe_newton_denom_min = 1.0e-12;
// heuristic escape; use a bracketed root solve if more pathologies show up.
constexpr int qpe_min_damp_stall_iters_to_reset = 8;

double qpe_damp_upper_bound(const double damp_fac)
{
    if (!std::isfinite(damp_fac) || damp_fac <= 0.0)
    {
        return 0.0;
    }
    return std::clamp(damp_fac, qpe_damp_min, qpe_damp_max);
}

bool qpe_complex_is_finite(const cplxdb &x)
{
    return std::isfinite(x.real()) && std::isfinite(x.imag());
}

QpeEvaluation qpe_evaluate(const AnalyCont &pade, const double energy, const double e0,
                           const double e_fermi)
{
    QpeEvaluation result;
    result.energy = energy;
    result.sigc = pade.get(cplxdb{energy - e_fermi, 0.0});
    result.sigc_deriv = pade.get_derivative(cplxdb{energy - e_fermi, 0.0});
    result.residual = std::numeric_limits<double>::quiet_NaN();
    result.abs_residual = std::numeric_limits<double>::infinity();
    if (std::isfinite(energy) && qpe_complex_is_finite(result.sigc))
    {
        result.residual = e0 + result.sigc.real() - energy;
        result.abs_residual = std::abs(result.residual);
    }
    return result;
}

double qpe_quasi_newton_step(const QpeEvaluation &eval)
{
    if (!std::isfinite(eval.residual))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double denom = 1.0 - eval.sigc_deriv.real();
    if (qpe_complex_is_finite(eval.sigc_deriv) &&
        std::isfinite(denom) &&
        std::abs(denom) > qpe_newton_denom_min)
    {
        return eval.residual / denom;
    }

    return eval.residual;
}

} /* end of anonymous namespace */

int qpe_solver_pade_self_consistent(const AnalyCont &pade, const double &e_mf,
                                    const double &e_fermi, const double &vxc, const double &sigma_x,
                                    double &e_qp, cplxdb &sigc, const double diff_init,
                                    const double thres, const int n_iter_max, const double damp_fac,
                                    const bool use_adaptive_damp,
                                    const bool use_legacy_nonadaptive_update)
{
    int info = 0;
    int n_iter = 0;
    bool converged = false;

    // initial guess of e_qp as input mean-field energy
    const double e0 = e_mf - vxc + sigma_x;
    double e_qp_last = e_mf;
    double e_trial_last = e_mf;
    double e_trial_this;
    double e_qp_this = e_mf;
    cplxdb sigc_this, sigc_last;
    std::array<QpeFixedPointIteration, n_qpe_iters_to_dump> last_iters;
    int n_iters_recorded = 0;

    if (n_iter_max <= 0)
    {
        e_qp = e_qp_last;
        sigc = cplxdb{0.0, 0.0};
        return 1;
    }

    double diff = diff_init;
    const double damp_max = use_adaptive_damp ? qpe_damp_upper_bound(damp_fac) : damp_fac;
    const double damp_min = use_adaptive_damp ? std::min(qpe_damp_min, damp_max) : damp_max;
    double damp = damp_max;
    double abs_diff_last = std::numeric_limits<double>::infinity();
    double diff_last = std::numeric_limits<double>::quiet_NaN();
    int n_min_damp_stall = 0;
    bool force_damp_max_step = false;
    // NOTE: legacy update is kept only for non-adaptive comparison tests.
    const bool use_residual_mixing_update =
            use_adaptive_damp || !use_legacy_nonadaptive_update;

    // std::cout << "QPE: " << e_mf << " " << e_fermi << " " << vxc << " " << sigma_x << "\n";
    while (n_iter++ < n_iter_max)
    {
        double damp_this = force_damp_max_step ? damp_max : damp;
        double abs_diff_this = std::numeric_limits<double>::infinity();

        const int n_backtrack =
            use_adaptive_damp && !force_damp_max_step ? n_qpe_max_backtrack : 0;
        force_damp_max_step = false;
        for (int i_backtrack = 0; i_backtrack <= n_backtrack; ++i_backtrack)
        {
            const double e_update_base =
                use_residual_mixing_update ? e_trial_last : e_qp_last;
            e_trial_this = e_update_base + damp_this * diff;
            sigc_this = pade.get(static_cast<cplxdb>(e_trial_this - e_fermi));
            e_qp_this = e0 + sigc_this.real();
            abs_diff_this = std::abs(e_qp_this - e_trial_this);

            const bool residual_worse =
                use_adaptive_damp &&
                std::isfinite(abs_diff_last) &&
                (!std::isfinite(abs_diff_this) || abs_diff_this > abs_diff_last);
            if (!residual_worse || damp_this <= qpe_damp_min || i_backtrack == n_qpe_max_backtrack)
            {
                break;
            }
            damp_this = std::max(qpe_damp_min, qpe_damp_shrink * damp_this);
        }

        diff = e_qp_this - e_trial_this;
        damp = damp_this;
        e_qp_last = e_qp_this;
        e_trial_last = e_trial_this;
        sigc_last = sigc_this;
        last_iters[n_iters_recorded % n_qpe_iters_to_dump] =
                QpeFixedPointIteration{n_iter, e_trial_this, e_qp_this, sigc_last, diff, damp};
        ++n_iters_recorded;
        if (abs_diff_this < thres)
        {
            converged = true;
            // std::cout << "Iteration " << n_iter << ": "
            //         << "e_qp = " << e_qp_this << ", "
            //         << "sigc = " << sigc_last << ", "
            //         << "diff = " << diff << std::endl;
            break;
        }
        if (!std::isfinite(abs_diff_this))
        {
            break;
        }
        const bool sign_changed = std::isfinite(diff_last) && diff * diff_last < 0.0;
        if (use_adaptive_damp && sign_changed)
        {
            damp = std::max(damp_min, qpe_damp_shrink * damp);
            n_min_damp_stall = 0;
        }
        else if (use_adaptive_damp && abs_diff_this < abs_diff_last)
        {
            damp = std::min(damp_max, std::max(damp_min, qpe_damp_grow * damp));
            n_min_damp_stall = 0;
        }
        else if (use_adaptive_damp && damp <= damp_min && std::isfinite(abs_diff_last))
        {
            ++n_min_damp_stall;
            if (n_min_damp_stall >= qpe_min_damp_stall_iters_to_reset)
            {
                damp = damp_max;
                force_damp_max_step = true;
                n_min_damp_stall = 0;
            }
        }
        else
        {
            n_min_damp_stall = 0;
        }
        diff_last = diff;
        abs_diff_last = abs_diff_this;
    }
    // std::cout << "Finished QPE solve: " << n_iter << " " << std::scientific << diff << " " << std::abs(diff) << "\n";
    sigc = sigc_last;
    e_qp = e_qp_this;
    // std::cout << "Compare QPE solver: " << std::setw(16) << std::setprecision(8) << e_mf - vxc + sigma_x + sigc.real() << " " << e_qp << std::endl;

    if (!converged)
    {
        info = 1;
        auto &ofs_myid = global::ofs_myid;
        ofs_myid << "QPE solver did not converge after " << n_iter_max
                 << " iterations. Last " << std::min(n_iters_recorded, n_qpe_iters_to_dump)
                 << " iterations:" << std::endl;
        ofs_myid << "iter e_trial e_qp sigc_re sigc_im diff abs_diff damp" << std::endl;
        const int i_start = n_iters_recorded > n_qpe_iters_to_dump
                                    ? n_iters_recorded - n_qpe_iters_to_dump
                                    : 0;
        for (int i = i_start; i < n_iters_recorded; ++i)
        {
            const auto &iter = last_iters[i % n_qpe_iters_to_dump];
            ofs_myid << iter.iter << " " << iter.e_trial << " " << iter.e_qp << " "
                     << iter.sigc.real() << " " << iter.sigc.imag() << " " << iter.diff << " "
                     << std::abs(iter.diff) << " " << iter.damp << std::endl;
        }
    }

    return info;
}

int qpe_solver_pade_quasi_newton(const AnalyCont &pade, const double &e_mf, const double &e_fermi,
                                 const double &vxc, const double &sigma_x, double &e_qp,
                                 cplxdb &sigc, const double diff_init, const double thres,
                                 const int n_iter_max, const double damp_fac,
                                 const bool use_adaptive_damp)
{
    int info = 0;
    int n_iter = 0;
    bool converged = false;

    // The QPE residual is F(E) = e0 + Re Sigma_c(E - e_f) - E.
    const double e0 = e_mf - vxc + sigma_x;
    std::array<QpeQuasiNewtonIteration, n_qpe_iters_to_dump> last_iters;
    int n_iters_recorded = 0;

    if (n_iter_max <= 0)
    {
        e_qp = e_mf;
        sigc = cplxdb{0.0, 0.0};
        return 1;
    }

    const double damp_max = use_adaptive_damp ? qpe_damp_upper_bound(damp_fac) : damp_fac;
    const double damp_min = use_adaptive_damp ? std::min(qpe_damp_min, damp_max) : damp_max;
    if (!std::isfinite(damp_max) || damp_max <= 0.0)
    {
        e_qp = e_mf;
        sigc = cplxdb{0.0, 0.0};
        return 1;
    }

    double damp = damp_max;
    double e_current = e_mf;
    if (std::isfinite(diff_init) && diff_init != 0.0)
    {
        e_current += damp * diff_init;
    }
    QpeEvaluation current = qpe_evaluate(pade, e_current, e0, e_fermi);

    while (n_iter < n_iter_max)
    {
        if (current.abs_residual < thres)
        {
            converged = true;
            break;
        }
        if (!std::isfinite(current.abs_residual))
        {
            break;
        }

        ++n_iter;
        const double newton_step = qpe_quasi_newton_step(current);
        if (!std::isfinite(newton_step))
        {
            break;
        }

        double damp_this = damp;
        QpeEvaluation trial = current;

        const int n_backtrack = use_adaptive_damp ? n_qpe_max_backtrack : 0;
        for (int i_backtrack = 0; i_backtrack <= n_backtrack; ++i_backtrack)
        {
            const double e_trial = current.energy + damp_this * newton_step;
            trial = qpe_evaluate(pade, e_trial, e0, e_fermi);

            const bool residual_worse =
                use_adaptive_damp &&
                (!std::isfinite(trial.abs_residual) ||
                 trial.abs_residual > current.abs_residual);
            if (!residual_worse || damp_this <= qpe_damp_min || i_backtrack == n_qpe_max_backtrack)
            {
                break;
            }
            damp_this = std::max(qpe_damp_min, qpe_damp_shrink * damp_this);
        }

        damp = damp_this;
        last_iters[n_iters_recorded % n_qpe_iters_to_dump] =
                QpeQuasiNewtonIteration{n_iter, current.energy, trial.energy, trial.sigc,
                                        current.sigc_deriv, trial.residual, newton_step, damp};
        ++n_iters_recorded;
        if (trial.abs_residual < thres)
        {
            current = trial;
            converged = true;
            break;
        }
        if (!std::isfinite(trial.abs_residual))
        {
            current = trial;
            break;
        }

        const bool sign_changed = current.residual * trial.residual < 0.0;
        if (use_adaptive_damp && sign_changed)
        {
            damp = std::max(damp_min, qpe_damp_shrink * damp);
        }
        else if (use_adaptive_damp && trial.abs_residual < current.abs_residual)
        {
            damp = std::min(damp_max, std::max(damp_min, qpe_damp_grow * damp));
        }
        current = trial;
    }

    sigc = current.sigc;
    e_qp = current.energy;

    if (!converged)
    {
        info = 1;
        auto &ofs_myid = global::ofs_myid;
        ofs_myid << "QPE quasi-Newton solver did not converge after " << n_iter_max
                 << " iterations. Last " << std::min(n_iters_recorded, n_qpe_iters_to_dump)
                 << " iterations:" << std::endl;
        ofs_myid << "iter e_trial e_qp sigc_re sigc_im sigc_deriv_re diff abs_diff newton_step damp" << std::endl;
        const int i_start = n_iters_recorded > n_qpe_iters_to_dump
                                    ? n_iters_recorded - n_qpe_iters_to_dump
                                    : 0;
        for (int i = i_start; i < n_iters_recorded; ++i)
        {
            const auto &iter = last_iters[i % n_qpe_iters_to_dump];
            ofs_myid << iter.iter << " " << iter.e_trial << " " << iter.e_qp << " "
                     << iter.sigc.real() << " " << iter.sigc.imag() << " "
                     << iter.sigc_deriv.real() << " " << iter.diff << " "
                     << std::abs(iter.diff) << " " << iter.newton_step << " "
                     << iter.damp << std::endl;
        }
    }

    return info;
}

int qpe_solver_pade_perturbative(const AnalyCont &pade, const double &e_mf, const double &e_fermi,
                                 const double &vxc, const double &sigma_x, double &e_qp,
                                 cplxdb &sigc, cplxdb &sigc_deriv, double &qp_weight)
{
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();

    const cplxdb omega = e_mf - e_fermi;
    sigc = pade.get(omega);
    sigc_deriv = pade.get_derivative(omega);

    const double denom = 1.0 - sigc_deriv.real();
    if (!qpe_complex_is_finite(sigc) ||
        !qpe_complex_is_finite(sigc_deriv) ||
        !std::isfinite(denom) ||
        denom == 0.0)
    {
        e_qp = nan;
        qp_weight = nan;
        return 1;
    }

    qp_weight = 1.0 / denom;
    e_qp = e_mf + qp_weight * (sigma_x + sigc.real() - vxc);
    if (!std::isfinite(qp_weight) || !std::isfinite(e_qp))
    {
        e_qp = nan;
        qp_weight = nan;
        return 1;
    }
    return 0;
}

} /* end of namespace librpa_int */
