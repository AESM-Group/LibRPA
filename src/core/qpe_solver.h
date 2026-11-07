#pragma once

#include "analycont.h"

namespace librpa_int
{

/*!
 * \brief Solve quasi-particle equation self-consistently by computing self-energy on real frequency
 * through the selected analytic continuation.
 *
 * \param [in]     pade        AnalyCont object, constructed from correlation self-energy at
 * imaginary frequency
 * \param [in]     e_mf        Energy of the state of mean-field calculation
 * \param [in]     e_fermi     Fermi energy
 * \param [in]     vxc         Exchange-correlation potential in the mean-field calculation
 * \param [in]     sigma_x     Exchange self-energy
 * \param [out]    e_qp        Quasi-particle energy as the solution of QPE
 * \param [out]    sigc        Correlation self-energy of the quasi-particle
 * \param [in]     diff_init   Initial residual used for the first damped QPE update
 * \param [in]     thres       Convergence threshold for the QPE residual, in Hartree
 * \param [in]     n_iter_max  Maximum number of self-consistent QPE iterations; must be positive
 * \param [in]     damp_fac    Damping factor for QPE updates; used as the initial and maximum
 * factor when adaptive damping is enabled
 * \param [in]     use_adaptive_damp
 *                             Adapt the damping factor during the solve
 * \param [in]     use_legacy_nonadaptive_update
 *                             Recover the old fixed-point base update only when adaptive damping is
 * disabled
 * \retval         info        0 if QPE is solved successfully, non-zero otherwise
 */
int qpe_solver_pade_self_consistent(const AnalyCont &pade, const double &e_mf,
                                    const double &e_fermi, const double &vxc, const double &sigma_x,
                                    double &e_qp, cplxdb &sigc, const double diff_init = 1.0e-3,
                                    const double thres = 1.0e-5, const int n_iter_max = 10000,
                                    const double damp_fac = 0.1,
                                    const bool use_adaptive_damp = false,
                                    const bool use_legacy_nonadaptive_update = false);

/*!
 * \brief Solve quasi-particle equation with a damped quasi-Newton update.
 *
 * The derivative is the analytic derivative of the selected continuation:
 * dE = F(E) / (1 - Re dSigma_c/dE).
 *
 * \param [in]     pade        AnalyCont object, constructed from correlation self-energy at
 * imaginary frequency
 * \param [in]     e_mf        Energy of the state of mean-field calculation
 * \param [in]     e_fermi     Fermi energy
 * \param [in]     vxc         Exchange-correlation potential in the mean-field calculation
 * \param [in]     sigma_x     Exchange self-energy
 * \param [out]    e_qp        Quasi-particle energy as the solution of QPE
 * \param [out]    sigc        Correlation self-energy of the quasi-particle
 * \param [in]     diff_init   Small initial displacement from the mean-field energy;
 *                             set to zero to start exactly from e_mf
 * \param [in]     thres       Convergence threshold for the QPE residual, in Hartree
 * \param [in]     n_iter_max  Maximum number of quasi-Newton QPE iterations; must be positive
 * \param [in]     damp_fac    Damping factor for quasi-Newton QPE updates; used as the initial and
 * maximum factor when adaptive damping is enabled
 * \param [in]     use_adaptive_damp
 *                             Adapt the damping factor during the solve
 * \retval         info        0 if QPE is solved successfully, non-zero otherwise
 */
int qpe_solver_pade_quasi_newton(const AnalyCont &pade, const double &e_mf, const double &e_fermi,
                                 const double &vxc, const double &sigma_x, double &e_qp,
                                 cplxdb &sigc, const double diff_init = 0.0,
                                 const double thres = 1.0e-5, const int n_iter_max = 200,
                                 const double damp_fac = 1.0, const bool use_adaptive_damp = true);

/*!
 * \brief Solve quasi-particle equation perturbatively by linearizing self-energy at the mean-field
 * energy.
 *
 * \param [in]     pade        AnalyCont object, constructed from correlation self-energy at
 * imaginary frequency
 * \param [in]     e_mf        Energy of the state of mean-field calculation
 * \param [in]     e_fermi     Fermi energy
 * \param [in]     vxc         Exchange-correlation potential in the mean-field calculation
 * \param [in]     sigma_x     Exchange self-energy
 * \param [out]    e_qp        Perturbative quasi-particle energy
 * \param [out]    sigc        Correlation self-energy at e_mf - e_fermi
 * \param [out]    sigc_deriv  Derivative of correlation self-energy at e_mf - e_fermi
 * \param [out]    qp_weight   Quasi-particle weight, 1 / (1 - Re sigc_deriv)
 * \retval         info        0 if QPE is solved successfully, non-zero otherwise
 */
int qpe_solver_pade_perturbative(const AnalyCont &pade, const double &e_mf, const double &e_fermi,
                                 const double &vxc, const double &sigma_x, double &e_qp,
                                 cplxdb &sigc, cplxdb &sigc_deriv, double &qp_weight);
}
