#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <limits>

#include "../core/analycont.h"
#include "../core/qsgw/correlation_potential.h"
#include "../io/stl_io_helper.h"
#include "../utils/constants.h"
#include "librpa_options.h"
#include "testutils.h"

using std::vector;
using std::cout;
using std::endl;
using namespace librpa_int;


static void initialize_1_over_x_minus_x0_data(int n, const cplxdb &x0, vector<cplxdb> &xs,
                                              vector<cplxdb> &data)
{
    xs.resize(n);
    data.resize(n);
    for (int i = 0; i < n; i++)
    {
        xs[i] = {0.0, static_cast<double>(i+1)};
        data[i] = -1.0 / (xs[i] - x0);
    }
}

void test_analycont_pade()
{
    int n = 6;
    // original function: y = -1 / (x - x0), x0 = {2.0, 2.0}
    const cplxdb x0 = {2.0, 2.0};
    std::vector<cplxdb> xs(n);
    std::vector<cplxdb> data(n);
    initialize_1_over_x_minus_x0_data(n, x0, xs, data);
    cout << "Input x: " << xs << endl;
    cout << "Input y: " << data << endl;

    librpa_int::AnalyContPade pade(n, xs, data);
    // std::cout << data << endl;
    const cplxdb test_x = {1.0, 1.0};
    const cplxdb ref = -1.0 / (test_x - x0);
    const cplxdb ref_deriv = 1.0 / ((test_x - x0) * (test_x - x0));
    cplxdb test_y = pade.get(test_x);
    cplxdb test_deriv = pade.get_derivative(test_x);
    cout << "Reference: "<< ref << endl;
    cout << "     Test: "<< test_y << endl;
    cout << "Reference derivative: "<< ref_deriv << endl;
    cout << "     Test derivative: "<< test_deriv << endl;
    assert(fequal(ref, test_y));
    assert(fequal(ref_deriv, test_deriv));
}

void test_get_spectfunc_acpade()
{
    int n = 6;
    // original function: y = -1 / (x - x0), x0 = {2.0, 2.0}
    const cplxdb x0 = {2.0, 2.0};
    std::vector<cplxdb> xs(n);
    std::vector<cplxdb> data(n);
    initialize_1_over_x_minus_x0_data(n, x0, xs, data);
    librpa_int::AnalyContPade pade(n, xs, data);

    vector<cplxdb> omegas(2);
    omegas[0] = {0.0, 0.0};
    omegas[1] = {2.0, 3.0};

    // assume e_ks = v_xc = v_exx = 0, and then use the function above, we have
    // A(w) = - 1 / PI * Im [ w + 1 / (w - 2 - 2i) ]^{-1}
    // and
    // A(w=0) = - 1 / PI * Im [-2 - 2i]
    // A(w=2+3i) = - 1 / PI * Im [2 + 2i]^{-1}
    vector<double> ref_sf(2);
    ref_sf[0] = 2.0 / PI;
    ref_sf[1] = 0.25 / PI;

    auto sf = librpa_int::get_specfunc(pade, omegas, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert(fequal_array(2, sf.data(), ref_sf.data(), false));

    // same results if we have v_xc cancels e_ks and v_exx
    sf = librpa_int::get_specfunc(pade, omegas, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0);
    assert(fequal_array(2, sf.data(), ref_sf.data(), false));

    // same results if we shift the omegas and e_ks by reference (2.0)
    omegas[0] = {2.0, -1.0};
    omegas[1] = {4.0, 2.0};
    sf = librpa_int::get_specfunc(pade, omegas, 2.0, 1.0 + 2.0, 2.0, 1.0, 1.0, 1.0);
    assert(fequal_array(2, sf.data(), ref_sf.data(), false));
}

void test_analycont_pade_source_data()
{
    constexpr int n = 8;
    constexpr int n_pars = 4;
    std::vector<cplxdb> xs(n);
    std::vector<cplxdb> data(n);
    initialize_1_over_x_minus_x0_data(n, {2.0, 2.0}, xs, data);

    librpa_int::AnalyContPade pade(n_pars, xs, data);
    const auto &source_xs = pade.get_source_xs();
    const auto &source_data = pade.get_source_data();
    assert(source_xs.size() == n_pars);
    assert(source_data.size() == n_pars);

    const int expected_indices[n_pars] = {0, 2, 4, 7};
    for (int i = 0; i != n_pars; ++i)
    {
        assert(source_xs[i] == xs[expected_indices[i]]);
        assert(source_data[i] == data[expected_indices[i]]);
    }
}

void test_analycont_pade_resample_imag_grid()
{
    constexpr int n = 6;
    const cplxdb x0 = {2.0, 2.0};
    std::vector<cplxdb> xs;
    std::vector<cplxdb> data;
    initialize_1_over_x_minus_x0_data(n, x0, xs, data);

    librpa_int::AnalyContPade original(n, xs, data);
    std::vector<cplxdb> xs_resampled = {
        {0.0, 0.5},
        {0.0, 1.5},
        {0.0, 3.0},
        {0.0, 6.0},
    };
    std::vector<cplxdb> data_resampled;
    data_resampled.reserve(xs_resampled.size());
    for (const auto &x: xs_resampled)
    {
        data_resampled.emplace_back(original.get(x));
    }

    librpa_int::AnalyContPade resampled(xs_resampled.size(), xs_resampled, data_resampled);
    const cplxdb test_x = {1.0, 1.0};
    const cplxdb ref = -1.0 / (test_x - x0);
    assert(fequal(ref, resampled.get(test_x)));
}

void test_analycont_pade_terminated_data()
{
    const vector<cplxdb> xs = {{0.0, 1.0}, {0.0, 2.0}, {0.0, 4.0}, {0.0, 8.0}};
    const cplxdb target{0.3, 0.2};
    for (const cplxdb constant : {cplxdb{0.0, 0.0}, cplxdb{2.0, 0.5}})
    {
        const vector<cplxdb> data(xs.size(), constant);
        const AnalyContPade pade(xs.size(), xs, data);
        assert(pade.get(target) == constant);
        assert(pade.get_derivative(target) == cplxdb(0.0, 0.0));
        assert(pade.get_source_xs() == xs);
        assert(pade.get_source_data() == data);
    }
    for (const double scale : {1.0, 1.0e-20})
    {
        vector<cplxdb> data;
        for (const auto x : xs) data.push_back(scale / x);
        const AnalyContPade pade(xs.size(), xs, data);
        assert(std::abs(pade.get(target) / scale - 1.0 / target) < 1.0e-13);
        assert(std::abs(pade.get_derivative(target) / scale + 1.0 / (target * target)) < 1.0e-13);
    }
}

void test_qsgw_pade_zero_offdiagonal()
{
    MeanField meanfield(1, 1, 2, 2);
    meanfield.get_eigenvals()[0](0, 0) = -0.4;
    meanfield.get_eigenvals()[0](0, 1) = 0.6;
    meanfield.get_efermi() = 0.1;
    const vector<double> frequencies{1.0, 2.0, 4.0, 8.0};
    std::map<double, Matz> sigma;
    for (const double frequency : frequencies)
    {
        Matz matrix(2, 2, MAJOR::ROW);
        matrix(0, 0) = 0.4 / (cplxdb{0.0, frequency} + 2.0);
        matrix(1, 1) = 0.3 / (cplxdb{0.0, frequency} + 2.0);
        matrix(0, 1) = matrix(1, 0) = 0.0;
        sigma.emplace(frequency, matrix);
    }
    for (const auto mode :
         {qsgw::CorrelationPotentialMode::ModeA, qsgw::CorrelationPotentialMode::ModeB})
    {
        qsgw::CorrelationPotentialSettings settings;
        settings.anacon_method = LIBRPA_ANACON_PADE;
        settings.mode = mode;
        const Matz potential =
            qsgw::build_qsgw_correlation_potential(meanfield, frequencies, sigma, 0, 0, settings);
        assert(std::abs(potential(0, 0) - 0.4 / 1.5) < 1.0e-13);
        assert(std::abs(potential(1, 1) - 0.3 / 2.5) < 1.0e-13);
        assert(potential(0, 1) == cplxdb(0.0, 0.0));
        assert(potential(1, 0) == cplxdb(0.0, 0.0));
    }
}

void test_analycont_pade_barycentric()
{
    const auto check = [](const vector<cplxdb> &xs, const vector<cplxdb> &unscaled_data,
                          const std::function<cplxdb(cplxdb)> &value,
                          const std::function<cplxdb(cplxdb)> &derivative)
    {
        for (const cplxdb scale :
             {cplxdb{1.0, 0.0}, cplxdb{1.0e-20, 0.0}, cplxdb{0.6, 0.8}, cplxdb{6.0e-21, 8.0e-21}})
        {
            vector<cplxdb> data;
            double data_scale = 0.0;
            for (const auto datum : unscaled_data)
            {
                data.push_back(scale * datum);
                data_scale = std::max(data_scale, std::abs(data.back()));
            }
            const AnalyContPadeBarycentric pade(xs, data);
            assert(pade.get_source_xs() == xs);
            assert(pade.get_source_data() == data);
            for (const auto x : xs)
            {
                assert(std::abs(pade.get(x) - scale * value(x)) <=
                       64.0 * std::numeric_limits<double>::epsilon() * data_scale);
                assert(std::abs(pade.get_derivative(x) / scale - derivative(x)) < 1.0e-11);
            }
            for (const cplxdb x :
                 {cplxdb{-0.7, 0.0}, cplxdb{0.0, 0.0}, cplxdb{0.31, 0.0}, cplxdb{1.2, 0.17}})
            {
                assert(std::abs(pade.get(x) / scale - value(x)) < 1.0e-11);
                assert(std::abs(pade.get_derivative(x) / scale - derivative(x)) < 1.0e-11);
            }
        }
    };
    const auto even_value = [](cplxdb z) { return 1.0 / (1.0 + z * z); };
    const auto even_derivative = [](cplxdb z) { return -2.0 * z / std::pow(1.0 + z * z, 2); };
    // Every ordering has a repeated value: no finite multiplicative Thiele form.
    check({-1.0, 1.0, -2.0, 2.0}, {0.5, 0.5, 0.2, 0.2}, even_value, even_derivative);
    const auto zero_value = [](cplxdb z) { return z / (1.0 + z * z); };
    const auto zero_derivative = [](cplxdb z) { return (1.0 - z * z) / std::pow(1.0 + z * z, 2); };
    check({0.0, 1.0, 2.0, 3.0}, {0.0, 0.5, 0.4, 0.3}, zero_value, zero_derivative);
    // Odd sample counts retain equal numerator and denominator degree.
    check(
        {0.0, 1.0, 2.0}, {0.0, 0.5, 2.0 / 3.0}, [](cplxdb z) { return z / (1.0 + z); },
        [](cplxdb z) { return 1.0 / std::pow(1.0 + z, 2); });

    // Complex offdiagonals of a matrix self-energy with PSD residues at real poles.
    // A zero sample and repeated samples are both compatible with matrix causality.
    const vector<cplxdb> xs{{0.0, 1.0}, {0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
    for (const cplxdb residue : {cplxdb{1.7, -0.6}, cplxdb{0.5, -1.5}})
    {
        const auto value = [=](cplxdb z) { return 1.0 / (z + 1.0) + residue / (z - 2.0); };
        const auto derivative = [=](cplxdb z)
        { return -1.0 / std::pow(z + 1.0, 2) - residue / std::pow(z - 2.0, 2); };
        vector<cplxdb> data;
        for (const auto x : xs) data.push_back(value(x));
        if (residue == cplxdb(1.7, -0.6))
            data[0] = data[1] = {-0.3, -0.6};
        else
            data[0] = 0.0;
        check(xs, data, value, derivative);
    }
}

void test_analycont_pade_barycentric_constants()
{
    const cplxdb target{0.3, 0.2};
    for (const int count : {1, 4})
        for (const cplxdb value : {cplxdb{0.0, 0.0}, cplxdb{2.0, 0.5}})
        {
            vector<cplxdb> xs;
            for (int i = 0; i < count; ++i) xs.emplace_back(0.0, i + 1.0);
            const AnalyContPadeBarycentric pade(xs, vector<cplxdb>(count, value));
            const AnalyCont &continuation = pade;
            assert(continuation.get(target) == value);
            assert(pade.get_derivative(target) == cplxdb(0.0, 0.0));
        }
}

void test_analycont_pade_barycentric_fallback()
{
    const auto check = [](const vector<cplxdb> &xs, const vector<cplxdb> &data)
    {
        const AnalyContPade fallback(xs.size(), xs, data);
        const AnalyContPadeBarycentric direct(xs, data);
        assert(fallback.get_source_xs() == direct.get_source_xs());
        assert(fallback.get_source_data() == direct.get_source_data());
        auto targets = xs;
        targets.emplace_back(0.31, 0.0);
        targets.emplace_back(1.2, 0.17);
        for (const auto target : targets)
        {
            assert(fallback.get(target) == direct.get(target));
            assert(fallback.get_derivative(target) == direct.get_derivative(target));
        }
    };
    check({-1.0, 1.0, -2.0, 2.0}, {0.5, 0.5, 0.2, 0.2});
    check({0.0, 1.0, 2.0, 3.0}, {0.0, 0.5, 0.4, 0.3});
}

void test_analycont_pade_removable_singularity()
{
    // This is the Thiele continuant evaluation, not barycentric construction.
    // An internal nested denominator can vanish at a regular numerator root.
    const vector<cplxdb> root_xs{0.0, 1.0, 2.0, 4.0};
    for (const double scale : {1.0, 1.0e-20})
    {
        vector<cplxdb> data;
        for (const auto x : root_xs) data.push_back(scale * (x - 3.0) / (x + 1.0));
        const AnalyContPade pade(root_xs.size(), root_xs, data);
        assert(std::abs(pade.get(3.0) / scale) < 64.0 * std::numeric_limits<double>::epsilon());
        assert(std::abs(pade.get_derivative(3.0) / scale - 0.25) < 1.0e-13);
    }
}

void test_analycont_pade_bi2se3_degenerate_pair()
{
    // Captured Gamma Sigma(12,13): samples 0 and 4 are bit-identical.
    // References: independent 90-decimal [7/8] polynomial interpolation.
    const vector<double> frequencies{
        0.0025993820787061869, 0.0082933952369329336, 0.01559849436121215, 0.026034639453473118,
        0.041908889091695815,  0.066926838652596854,  0.1072375165443057,  0.17327458156727799,
        0.28307174355137721,   0.46836247951091897,   0.78609830679002524, 1.3410801386313786,
        2.3340575185040118,    4.1840272211235652,    7.9800939324232596,  18.710213363255843};
    const vector<cplxdb> original{{9.0084190107475592e-15, -1.4285447824669006e-15},
                                  {9.0222967985553737e-15, -1.4341826337638253e-15},
                                  {9.0136231811754897e-15, -1.4333152720258369e-15},
                                  {9.0240315220313505e-15, -1.4307131868118717e-15},
                                  {9.0084190107475592e-15, -1.4285447824669006e-15},
                                  {9.010153734223536e-15, -1.4298458250738832e-15},
                                  {8.9719898177520463e-15, -1.4107638668381384e-15},
                                  {8.9355606247565333e-15, -1.4003555259822775e-15},
                                  {8.8627022387655074e-15, -1.3747683547116196e-15},
                                  {8.6996382320236876e-15, -1.3418086086680603e-15},
                                  {8.31799906730879e-15, -1.2654807757250808e-15},
                                  {7.5234957153114124e-15, -1.1058862159352145e-15},
                                  {6.0021432268797525e-15, -7.5113526509795747e-16},
                                  {3.9794556538907955e-15, -2.8449465006019636e-16},
                                  {2.0755966390062497e-15, 1.0061396160665481e-16},
                                  {7.4213638706632779e-16, 1.4311468676808659e-16}};
    vector<cplxdb> xs;
    for (const double frequency : frequencies) xs.emplace_back(0.0, frequency);
    for (const double scale : {1.0, 1.0e14, 1.0e-20})
    {
        vector<cplxdb> data;
        for (const auto value : original) data.push_back(scale * value);
        const AnalyContPadeBarycentric pade(xs, data);
        const AnalyContPade fallback(xs.size(), xs, data);
        for (int i = 0; i < static_cast<int>(xs.size()); ++i)
            assert(std::abs(pade.get(xs[i]) / scale - original[i]) < 1.3e-28);
        assert(std::abs(pade.get(0.0) / scale -
                        cplxdb(9.0136364054620232e-15, -1.4319544233203666e-15)) < 1.0e-27);
        assert(std::abs(pade.get_derivative(0.0) / scale -
                        cplxdb(6.0543052717614809e-16, 1.2025620912849089e-15)) < 1.0e-26);
        assert(fallback.get(0.0) == pade.get(0.0));
        assert(fallback.get_derivative(0.0) == pade.get_derivative(0.0));
    }
}

void test_analycont_aaa_constant_linear_and_rational()
{
    const std::vector<cplxdb> xs = {{-2.0, 0.2}, {-1.0, 0.5}, {0.0, 0.8}, {1.0, 1.1}, {2.0, 1.4}};
    std::vector<cplxdb> constant(xs.size(), {2.5, -0.75});
    AnalyContAAA aaa_constant(4, xs, constant);
    assert(fequal(aaa_constant.get({0.25, -0.4}), constant[0]));
    assert(fequal(aaa_constant.get_derivative({0.25, -0.4}), cplxdb{0.0, 0.0}));

    const cplxdb slope{1.2, -0.4};
    const cplxdb intercept{-0.3, 0.7};
    std::vector<cplxdb> linear;
    for (const auto &x : xs) linear.push_back(slope * x + intercept);
    AnalyContAAA aaa_linear(4, xs, linear);
    const cplxdb x_test{0.35, -0.15};
    assert(fequal(aaa_linear.get(x_test), slope * x_test + intercept, cplxdb{1.0e-11, 0.0}));
    assert(fequal(aaa_linear.get_derivative(x_test), slope, cplxdb{1.0e-10, 0.0}));

    const cplxdb pole{1.5, -0.8};
    std::vector<cplxdb> rational;
    for (const auto &x : xs) rational.push_back((x + cplxdb{0.4, 0.2}) / (x - pole));
    AnalyContAAA aaa_rational(4, xs, rational);
    const cplxdb rational_ref = (x_test + cplxdb{0.4, 0.2}) / (x_test - pole);
    const cplxdb rational_deriv = (-pole - cplxdb{0.4, 0.2}) / ((x_test - pole) * (x_test - pole));
    assert(fequal(aaa_rational.get(x_test), rational_ref, cplxdb{1.0e-10, 0.0}));
    assert(fequal(aaa_rational.get_derivative(x_test), rational_deriv, cplxdb{1.0e-9, 0.0}));
    const cplxdb support_derivative = aaa_rational.get_derivative(xs[0]);
    assert(std::isfinite(support_derivative.real()) && std::isfinite(support_derivative.imag()));
    assert(fequal(aaa_rational.get(xs[0]), rational[0]));
}

void test_aaa_zero_weight_support_residual()
{
    const std::vector<cplxdb> xs = {{0, 1}, {0, 2}, {0, 3}, {0, 4}};
    const std::vector<cplxdb> data = {{0, 0}, {1, 0}, {1, 0}, {1, 0}};
    AnalyContAAA fit(2, xs, data);
    // The zero-weight support at i is absent from the rational function,
    // which is the constant one, rather than an interpolant of all four data.
    assert(!fit.reached_tolerance());
    assert(std::abs(fit.get_max_source_error() - 1.0) < 1e-14);
    std::vector<cplxdb> queries = xs;
    queries.push_back(xs[0] + cplxdb{1e-7, 0});
    queries.push_back(xs[0] - cplxdb{1e-7, 0});
    for (const auto z : queries)
    {
        assert(std::abs(fit.get(z) - cplxdb{1, 0}) < 1e-14);
        assert(std::abs(fit.get_derivative(z)) < 1e-13);
    }
}

void test_ridge_value_derivative_and_historical_fit()
{
    std::vector<cplxdb> xs, data;
    for (int i = 1; i <= 16; ++i)
    {
        const cplxdb x{0.0, 0.5 * i};
        xs.push_back(x);
        data.push_back(0.2 / (x + 1.3) + 0.05 / (x - 2.1));
    }
    // Values from the unmodified 49adf9bb ridge kernel, with denominator
    // weight 10 and an inactive denominator floor. This checks the port's fit.
    AnalyContRidge ridge(8, xs, data, 1e-6, 10.0);
    const cplxdb z{-0.6, 0.1};
    const cplxdb reference{0.285633286226600724, 0.0220621556312846218};
    assert(std::abs(ridge.get(z) - reference) < 1e-11);
    assert(std::abs(ridge.get({0.3, 0.2}) - cplxdb{0.0895108357609937649, -0.0263183108993119974}) <
           1e-11);
    const double h = 1e-5;
    const auto finite_difference = (ridge.get(z + h) - ridge.get(z - h)) / (2 * h);
    assert(std::abs(ridge.get_derivative(z) - finite_difference) < 1e-8);

    data.clear();
    for (const auto x : xs) data.push_back((0.2 * x + 0.1) / (1.0 + 0.3 * x));
    AnalyContRidge exact(3, xs, data, 0.0);
    assert(std::abs(exact.get(z) - (0.2 * z + 0.1) / (1.0 + 0.3 * z)) < 1e-11);
    assert(std::abs(exact.get_derivative(z) - 0.17 / std::pow(1.0 + 0.3 * z, 2)) < 1e-11);
}

int main(int argc, char *argv[])
{
    LibrpaOptions defaults;
    librpa_init_options(&defaults);
    assert(defaults.anacon_method == LIBRPA_ANACON_PADE);
    assert(defaults.anacon_ridge_lambda == 1e-12);
    assert(defaults.anacon_ridge_den_weight == 10.0);
    const std::vector<cplxdb> xs{{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}};
    const std::vector<cplxdb> zeros(3, {0.0, 0.0});
    AnalyContPade zero(3, xs, zeros);
    assert(zero.get({1.0, 0.1}) == cplxdb(0.0, 0.0));
    assert(zero.get_derivative({1.0, 0.1}) == cplxdb(0.0, 0.0));
    assert(zero.get_source_data() == zeros);
    test_analycont_aaa_constant_linear_and_rational();
    test_aaa_zero_weight_support_residual();
    test_ridge_value_derivative_and_historical_fit();
    test_analycont_pade();
    test_get_spectfunc_acpade();
    test_analycont_pade_source_data();
    test_analycont_pade_resample_imag_grid();
    test_analycont_pade_terminated_data();
    test_qsgw_pade_zero_offdiagonal();
    test_analycont_pade_barycentric();
    test_analycont_pade_barycentric_constants();
    test_analycont_pade_barycentric_fallback();
    test_analycont_pade_removable_singularity();
    test_analycont_pade_bi2se3_degenerate_pair();
}
