/*!
 * @file      analycont.h
 * @brief     Utilities for analytic continuation
 * @author    Min-Ye Zhang
 * @date      2024-04-23
 */
#pragma once
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "../utils/base_utility.h"
#include "librpa_enums.h"

namespace librpa_int
{

class AnalyCont
{
protected:
    std::vector<cplxdb> source_xs;
    std::vector<cplxdb> source_data;

public:
    virtual ~AnalyCont() = default;
    virtual cplxdb get(const cplxdb &x) const = 0;
    virtual cplxdb get_derivative(const cplxdb &x) const = 0;

    /*!
     * @brief get the complex source points used by the analytic continuation
     */
    const std::vector<cplxdb> &get_source_xs() const { return source_xs; };

    /*!
     * @brief get the complex source values used by the analytic continuation
     */
    const std::vector<cplxdb> &get_source_data() const { return source_data; };
};

// Barycentric Padé interpolation of the supplied selected samples, with type
// [floor((N-1)/2) / floor(N/2)]. It can represent zero samples directly.
class AnalyContPadeBarycentric : public AnalyCont
{
private:
    std::vector<int> rational_supports;
    std::vector<std::complex<long double>> rational_weights;
    long double rational_x_scale = 1.0L;
    long double rational_y_scale = 1.0L;
    int rational_anchor = -1;

    bool initialize_rational();
    std::pair<cplxdb, cplxdb> evaluate_rational(const cplxdb &x) const;

public:
    AnalyContPadeBarycentric(const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data);

    cplxdb get(const cplxdb &x) const override;
    cplxdb get_derivative(const cplxdb &x) const override;
};

// Thiele reciprocal differences, with explicit delegation to the barycentric
// implementation when construction cannot produce a valid finite fraction.
class AnalyContPade: public AnalyCont
{
private:
    int n_pars;
    std::vector<cplxdb> par_y;
    std::optional<AnalyContPadeBarycentric> barycentric_fallback;

public:
    AnalyContPade(int n_pars_in,
                  const std::vector<cplxdb> &xs,
                  const std::vector<cplxdb> &data);

    /*!
     * @brief get the value of continued function at complex number
     *
     * @param [in]    x    complex argument of function
     *
     * @return    a complex double, the value of function at x
     */
    cplxdb get(const cplxdb &x) const override;

    /*!
     * @brief get the analytic derivative of continued function at complex number
     *
     * @param [in]    x    complex argument of function
     *
     * @return    a complex double, the derivative of function at x
     */
    cplxdb get_derivative(const cplxdb &x) const override;
};

class AnalyContAAA : public AnalyCont
{
private:
    double tolerance;
    double max_source_error = 0.0;
    bool tolerance_reached = false;
    std::vector<cplxdb> support_xs;
    std::vector<cplxdb> support_data;
    std::vector<cplxdb> weights;

public:
    AnalyContAAA(int n_support, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data,
                 double tolerance_in = 1.0e-13);

    cplxdb get(const cplxdb &x) const override;
    cplxdb get_derivative(const cplxdb &x) const override;
    int get_support_count() const { return static_cast<int>(support_xs.size()); }
    double get_max_source_error() const { return max_source_error; }
    bool reached_tolerance() const { return tolerance_reached; }
};

class AnalyContRidge : public AnalyCont
{
private:
    double x_scale = 1.0;
    double y_scale = 1.0;
    std::vector<cplxdb> numerator;
    std::vector<cplxdb> denominator;

public:
    AnalyContRidge(int n_params, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data,
                   double lambda = 1.0e-12, double denominator_weight = 10.0);
    cplxdb get(const cplxdb &x) const override;
    cplxdb get_derivative(const cplxdb &x) const override;
};

std::unique_ptr<AnalyCont> make_analycont(LibrpaAnalyContMethod method, int n_params,
                                          const std::vector<cplxdb> &xs,
                                          const std::vector<cplxdb> &data,
                                          double ridge_lambda = 1.0e-12,
                                          double ridge_denominator_weight = 10.0,
                                          double aaa_tolerance = 1.0e-13);

/*!
 * @brief get the spectral function from Pade approximant, based on diagonal approximation
 */
const std::vector<double> get_specfunc(const AnalyCont &ac, const std::vector<cplxdb> &omegas,
                                       const double &ref, const double &e_ks, const double &v_xc,
                                       const double &v_exx,
                                       const double &sigc_omega_imag_shift,
                                       const double &gf_omega_imag_shift);
}
