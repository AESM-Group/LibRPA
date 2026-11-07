/*!
 * @author    Min-Ye Zhang
 * @date      2024-04-25
 */
#include "analycont.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "../interface/blas_lapack.h"
#include "../utils/constants.h"
// #include <iostream>

#include "../math/complexmatrix.h"
// #include "../io/stl_io_helper.h"

namespace librpa_int
{

namespace
{

bool finite_complex(const cplxdb &value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

int find_exact(const std::vector<cplxdb> &xs, const cplxdb &x)
{
    for (int i = 0; i != static_cast<int>(xs.size()); ++i)
    {
        if (xs[i] == x) return i;
    }
    return -1;
}

cplxdb barycentric_value(const cplxdb &x, const std::vector<cplxdb> &support_xs,
                         const std::vector<cplxdb> &support_data,
                         const std::vector<cplxdb> &weights)
{
    const int exact = find_exact(support_xs, x);
    if (exact >= 0 && weights[exact] != cplxdb{0.0, 0.0}) return support_data[exact];

    cplxdb numerator{0.0, 0.0};
    cplxdb denominator{0.0, 0.0};
    double denominator_scale = 0.0;
    for (int i = 0; i != static_cast<int>(support_xs.size()); ++i)
    {
        // A zero weight removes this term, including at its former support.
        if (weights[i] == cplxdb{0.0, 0.0}) continue;
        const cplxdb term = weights[i] / (x - support_xs[i]);
        numerator += term * support_data[i];
        denominator += term;
        denominator_scale += std::abs(term);
    }
    if (!std::isfinite(denominator_scale) ||
        std::abs(denominator) <= 100.0 * std::numeric_limits<double>::epsilon() * denominator_scale)
    {
        throw std::runtime_error("AAA barycentric denominator is numerically singular");
    }
    const cplxdb result = numerator / denominator;
    if (!finite_complex(result)) throw std::runtime_error("AAA produced non-finite value");
    return result;
}

void compute_aaa_weights(const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data,
                         const std::vector<int> &support_indices, std::vector<cplxdb> &weights)
{
    const int n_data = static_cast<int>(xs.size());
    const int n_support = static_cast<int>(support_indices.size());
    const int n_rows = n_data - n_support;
    if (n_rows <= 0) throw std::runtime_error("AAA has no unsampled points for SVD");

    std::vector<int> unsampled;
    unsampled.reserve(n_rows);
    for (int i = 0; i != n_data; ++i)
    {
        if (std::find(support_indices.begin(), support_indices.end(), i) == support_indices.end())
            unsampled.push_back(i);
    }

    // LAPACK uses column-major storage; all public LibRPA matrices are row-major.
    std::vector<cplxdb> matrix(n_rows * n_support);
    for (int row = 0; row != n_rows; ++row)
    {
        const int i = unsampled[row];
        for (int col = 0; col != n_support; ++col)
        {
            const int j = support_indices[col];
            const cplxdb dz = xs[i] - xs[j];
            if (dz == cplxdb{0.0, 0.0})
                throw std::runtime_error("AAA encountered duplicate support points");
            matrix[row + col * n_rows] = (data[i] - data[j]) / dz;
        }
    }

    const char jobu = 'N';
    const char jobvt = 'A';
    const int lda = std::max(1, n_rows);
    const int ldu = 1;
    const int ldvt = std::max(1, n_support);
    std::vector<double> singular_values(n_support);
    std::vector<cplxdb> vt(n_support * n_support);
    cplxdb work_query{0.0, 0.0};
    std::vector<double> rwork(std::max(1, 5 * std::min(n_rows, n_support)));
    int lwork = -1;
    int info = 0;
    cplxdb u_dummy{0.0, 0.0};
    zgesvd_(&jobu, &jobvt, &n_rows, &n_support, matrix.data(), &lda, singular_values.data(),
            &u_dummy, &ldu, vt.data(), &ldvt, &work_query, &lwork, rwork.data(), &info);
    if (info != 0 || !finite_complex(work_query) || work_query.real() < 1.0)
        throw std::runtime_error("AAA complex SVD workspace query failed");

    lwork = std::max(1, static_cast<int>(work_query.real()));
    std::vector<cplxdb> work(lwork);
    zgesvd_(&jobu, &jobvt, &n_rows, &n_support, matrix.data(), &lda, singular_values.data(),
            &u_dummy, &ldu, vt.data(), &ldvt, work.data(), &lwork, rwork.data(), &info);
    if (info != 0) throw std::runtime_error("AAA complex Loewner SVD failed");

    // With JOBVT='A', the last row is the smallest right singular vector,
    // including the exact null space when the Loewner matrix is wide.
    const int smallest_row = n_support - 1;
    weights.resize(n_support);
    double weight_scale = 0.0;
    for (int i = 0; i != n_support; ++i)
    {
        weights[i] = std::conj(vt[smallest_row + i * ldvt]);
        if (!finite_complex(weights[i]))
            throw std::runtime_error("AAA SVD returned non-finite weights");
        weight_scale = std::max(weight_scale, std::abs(weights[i]));
    }
    if (!std::isfinite(weight_scale) || weight_scale <= std::numeric_limits<double>::epsilon())
        throw std::runtime_error("AAA SVD returned machine-zero weights");
}

using cplxld = std::complex<long double>;

std::pair<cplxdb, cplxdb> evaluate_continuants(const std::vector<cplxdb> &coefficients,
                                               const std::vector<cplxdb> &nodes, const cplxdb &x)
{
    // Forward convergents remain regular at an internal nested-fraction pole.
    const cplxld z(x.real(), x.imag());
    cplxld previous_a = 0.0L, a(coefficients[0].real(), coefficients[0].imag());
    cplxld previous_b = 1.0L, b = 1.0L;
    cplxld previous_da = 0.0L, da = 0.0L, previous_db = 0.0L, db = 0.0L;
    for (int i = 1; i < static_cast<int>(coefficients.size()); ++i)
    {
        const cplxld coefficient(coefficients[i].real(), coefficients[i].imag());
        const cplxld term = coefficient * (z - cplxld(nodes[i - 1].real(), nodes[i - 1].imag()));
        const cplxld next_a = a + term * previous_a;
        const cplxld next_b = b + term * previous_b;
        const cplxld next_da = da + coefficient * previous_a + term * previous_da;
        const cplxld next_db = db + coefficient * previous_b + term * previous_db;
        previous_a = a;
        previous_b = b;
        previous_da = da;
        previous_db = db;
        a = next_a;
        b = next_b;
        da = next_da;
        db = next_db;
        const long double scale =
            std::max({std::abs(a), std::abs(b), std::abs(previous_a), std::abs(previous_b),
                      std::abs(da), std::abs(db), std::abs(previous_da), std::abs(previous_db)});
        if (scale > 0.0L)
            for (auto *value :
                 {&a, &b, &previous_a, &previous_b, &da, &db, &previous_da, &previous_db})
                *value /= scale;
    }
    const cplxld value = a / b;
    const cplxld derivative = (da - value * db) / b;
    return {{static_cast<double>(value.real()), static_cast<double>(value.imag())},
            {static_cast<double>(derivative.real()), static_cast<double>(derivative.imag())}};
}

// Complete pivoting avoids normal equations for this small homogeneous system.
std::vector<std::vector<cplxld>> rational_nullspace(std::vector<std::vector<cplxld>> matrix)
{
    const int rows = matrix.size();
    const int columns = matrix.front().size();
    std::vector<int> permutation(columns);
    std::vector<long double> scales(columns, 1.0L);
    for (auto &row : matrix)
    {
        long double scale = 0.0L;
        for (const auto value : row) scale = std::max(scale, std::abs(value));
        if (scale > 0.0L)
            for (auto &value : row) value /= scale;
    }
    for (int column = 0; column < columns; ++column)
    {
        permutation[column] = column;
        long double scale = 0.0L;
        for (const auto &row : matrix) scale = std::max(scale, std::abs(row[column]));
        if (scale > 0.0L)
        {
            scales[column] = scale;
            for (auto &row : matrix) row[column] /= scale;
        }
    }

    int rank = 0;
    for (int pivot = 0; pivot < rows; ++pivot)
    {
        int pivot_row = pivot, pivot_column = pivot;
        long double size = 0.0L;
        for (int row = pivot; row < rows; ++row)
            for (int column = pivot; column < columns; ++column)
                if (std::abs(matrix[row][column]) > size)
                {
                    size = std::abs(matrix[row][column]);
                    pivot_row = row;
                    pivot_column = column;
                }
        if (size <= 64.0L * std::numeric_limits<long double>::epsilon()) break;
        std::swap(matrix[pivot], matrix[pivot_row]);
        for (auto &row : matrix) std::swap(row[pivot], row[pivot_column]);
        std::swap(permutation[pivot], permutation[pivot_column]);
        std::swap(scales[pivot], scales[pivot_column]);
        for (int row = pivot + 1; row < rows; ++row)
        {
            const cplxld factor = matrix[row][pivot] / matrix[pivot][pivot];
            matrix[row][pivot] = 0.0L;
            for (int column = pivot + 1; column < columns; ++column)
                matrix[row][column] -= factor * matrix[pivot][column];
        }
        ++rank;
    }

    std::vector<std::vector<cplxld>> basis;
    for (int free_column = rank; free_column < columns; ++free_column)
    {
        std::vector<cplxld> vector(columns, 0.0L), weights(columns);
        vector[free_column] = 1.0L;
        for (int row = rank - 1; row >= 0; --row)
        {
            cplxld sum = 0.0L;
            for (int column = row + 1; column < columns; ++column)
                sum += matrix[row][column] * vector[column];
            vector[row] = -sum / matrix[row][row];
        }
        for (int column = 0; column < columns; ++column)
            weights[permutation[column]] = vector[column] / scales[column];
        basis.push_back(std::move(weights));
    }
    // A lower-degree interpolant can have several homogeneous representations.
    // Their sum can remove spurious support zeros of individual basis vectors.
    if (basis.size() > 1)
    {
        std::vector<cplxld> weights(columns, 0.0L);
        for (const auto &vector : basis)
            for (int column = 0; column < columns; ++column) weights[column] += vector[column];
        basis.push_back(std::move(weights));
    }
    return basis;
}

}  // namespace

AnalyContPade::AnalyContPade(int n_pars_in, const std::vector<cplxdb> &xs, const std::vector<cplxdb> &data)
    : n_pars(n_pars_in)
{
    const int n_data = data.size();

    // Use all data if requested number of parameters is invalid
    if (n_pars < 0) n_pars = n_data;
    if (xs.size() != data.size() || n_data == 0 || n_pars == 0)
        throw std::invalid_argument(
            "Padé requires nonempty matching data and a nonzero parameter count");

    if (n_data <= n_pars)
    {
        // Use all data points
        n_pars = n_data;
        source_xs = xs;
        source_data = data;
    }
    else
    {
        // Select the data points evenly, when number of parameters are fewer than data points
        source_xs.resize(n_pars);
        source_data.resize(n_pars);
        int step = n_pars == 1 ? 0 : n_data / (n_pars - 1);
        for (int ipar = 0; ipar < n_pars - 1; ipar++)
        {
            source_xs[ipar] = xs[ipar * step];
            source_data[ipar] = data[ipar * step];
        }
        source_xs[n_pars - 1] = xs[n_pars == 1 ? 0 : n_data - 1];
        source_data[n_pars - 1] = data[n_pars == 1 ? 0 : n_data - 1];
    }

    // Band exclusion can make the correlation self-energy identically zero.
    if (!source_data.empty() && std::all_of(source_data.begin(), source_data.end(),
                                          [](const cplxdb &value) { return value == cplxdb(0.0, 0.0); }))
    {
        n_pars = 1;
        par_y = {cplxdb(0.0, 0.0)};
        return;
    }

    // Calculate the continuation coefficients, using Thiel's reciprocal difference method
    ComplexMatrix g(n_pars, n_pars);
    double data_scale = 0.0;
    for (int i_par = 0; i_par < n_pars; i_par++)
    {
        if (!finite_complex(source_xs[i_par]) || !finite_complex(source_data[i_par]))
            throw std::invalid_argument("Padé source data must be finite");
        g(i_par, 0) = source_data[i_par];
        data_scale = std::max(data_scale, std::abs(source_data[i_par]));
    }
    par_y.resize(n_pars);
    par_y[0] = g(0, 0);
    const double epsilon = std::numeric_limits<double>::epsilon();
    const auto lower_order_fits = [&](int n_terms)
    {
        for (int i = 0; i != static_cast<int>(source_xs.size()); ++i)
        {
            cplxdb tmp{1.0, 0.0};
            for (int p = n_terms - 1; p > 0; --p)
                tmp = 1.0 + par_y[p] * (source_xs[i] - source_xs[p - 1]) / tmp;
            const cplxdb value = par_y[0] / tmp;
            if (!finite_complex(value) ||
                std::abs(value - source_data[i]) > 64.0 * epsilon * data_scale)
                return false;
        }
        return true;
    };
    for (int i_par = 1; i_par < n_pars; i_par++)
    {
        const cplxdb pivot = g(i_par - 1, i_par - 1);
        double column_scale = std::abs(pivot);
        double deviation = 0.0;
        bool zero_pivot = pivot == cplxdb{0.0, 0.0};
        for (int i = i_par; i < n_pars; ++i)
        {
            const cplxdb previous = g(i, i_par - 1);
            column_scale = std::max(column_scale, std::abs(previous));
            deviation = std::max(deviation, std::abs(previous - pivot));
            zero_pivot = zero_pivot || previous == cplxdb{0.0, 0.0};
        }
        // Terminate a lower-order fraction only after checking all selected data.
        if ((zero_pivot || deviation <= 32.0 * epsilon * column_scale) && lower_order_fits(i_par))
        {
            n_pars = i_par;
            par_y.resize(n_pars);
            return;
        }
        if (zero_pivot)
        {
            barycentric_fallback.emplace(source_xs, source_data);
            return;
        }
        for (int i = i_par; i < n_pars; i++)
        {
            const cplxdb denominator = (source_xs[i] - source_xs[i_par - 1]) * g(i, i_par - 1);
            if (denominator == cplxdb{0.0, 0.0} || !finite_complex(denominator))
            {
                barycentric_fallback.emplace(source_xs, source_data);
                return;
            }
            g(i, i_par) = (pivot - g(i, i_par - 1)) / denominator;
            if (!finite_complex(g(i, i_par)))
            {
                barycentric_fallback.emplace(source_xs, source_data);
                return;
            }
        }
        par_y[i_par] = g(i_par, i_par);
    }
}

AnalyContPadeBarycentric::AnalyContPadeBarycentric(const std::vector<cplxdb> &xs,
                                                   const std::vector<cplxdb> &data)
{
    if (xs.empty() || xs.size() != data.size())
        throw std::invalid_argument("Barycentric Padé requires nonempty matching data");
    for (std::size_t i = 0; i < xs.size(); ++i)
        if (!finite_complex(xs[i]) || !finite_complex(data[i]))
            throw std::invalid_argument("Barycentric Padé source data must be finite");
    source_xs = xs;
    source_data = data;
    if (std::all_of(data.begin(), data.end(),
                    [&](const cplxdb &value) { return value == data.front(); }))
        return;
    if (!initialize_rational())
        throw std::runtime_error("Barycentric Padé could not fit all selected samples");
}

bool AnalyContPadeBarycentric::initialize_rational()
{
    // The same type [(N-1)/2, N/2] interpolant in barycentric coordinates.
    // Unlike the multiplicative fraction, this representation permits zeros.
    const int count = source_xs.size();
    const int supports = count / 2 + 1;
    std::vector<cplxld> xs, values;
    rational_y_scale = 0.0L;
    for (int i = 0; i < count; ++i)
    {
        xs.emplace_back(source_xs[i].real(), source_xs[i].imag());
        values.emplace_back(source_data[i].real(), source_data[i].imag());
        rational_x_scale = std::max(rational_x_scale, std::abs(xs.back()));
        rational_y_scale = std::max(rational_y_scale, std::abs(values.back()));
    }
    if (count < 2 || rational_y_scale == 0.0L) return false;
    for (auto &x : xs) x /= rational_x_scale;
    for (auto &value : values) value /= rational_y_scale;
    for (int i = 0; i < count; ++i)
        for (int j = 0; j < i; ++j)
            if (xs[i] == xs[j]) return false;

    for (int shift = 0; shift < count; ++shift)
    {
        rational_supports.clear();
        std::vector<bool> is_support(count, false);
        for (int j = 0; j < supports; ++j)
        {
            const int index = (j * (count - 1) / (supports - 1) + shift) % count;
            rational_supports.push_back(index);
            is_support[index] = true;
        }
        std::vector<std::vector<cplxld>> equations;
        for (int i = 0; i < count; ++i)
        {
            if (is_support[i]) continue;
            std::vector<cplxld> row;
            for (const int j : rational_supports)
                row.push_back((values[i] - values[j]) / (xs[i] - xs[j]));
            equations.push_back(std::move(row));
        }
        if (count % 2 == 0)
        {
            // Cancel the leading numerator coefficient: degree <= N/2-1.
            std::vector<cplxld> row;
            for (const int j : rational_supports) row.push_back(values[j]);
            equations.push_back(std::move(row));
        }
        rational_anchor = -1;
        if (count % 2 == 0)
        {
            long double largest = 0.0L;
            for (int j = 0; j < supports; ++j)
                if (std::abs(values[rational_supports[j]]) > largest)
                {
                    largest = std::abs(values[rational_supports[j]]);
                    rational_anchor = j;
                }
            if (rational_anchor < 0) continue;
        }
        const auto candidates = rational_nullspace(std::move(equations));
        for (auto weights : candidates)
        {
            long double scale = 0.0L;
            for (const auto weight : weights) scale = std::max(scale, std::abs(weight));
            if (scale == 0.0L || !std::isfinite(scale)) continue;
            for (auto &weight : weights) weight /= scale;
            if (rational_anchor >= 0)
            {
                cplxld leading = 0.0L;
                for (int j = 0; j < supports; ++j)
                    if (j != rational_anchor) leading += weights[j] * values[rational_supports[j]];
                weights[rational_anchor] = -leading / values[rational_supports[rational_anchor]];
            }
            rational_weights = std::move(weights);
            bool fits = true;
            for (int i = 0; i < count; ++i)
            {
                const auto value = evaluate_rational(source_xs[i]).first;
                if (!finite_complex(value) ||
                    std::abs(cplxld(value.real(), value.imag()) / rational_y_scale - values[i]) >
                        64.0L * std::numeric_limits<double>::epsilon())
                {
                    fits = false;
                    break;
                }
            }
            if (fits) return true;
        }
    }
    rational_weights.clear();
    return false;
}

std::pair<cplxdb, cplxdb> AnalyContPadeBarycentric::evaluate_rational(const cplxdb &x) const
{
    if (rational_weights.empty()) return {source_data.front(), cplxdb{0.0, 0.0}};
    const cplxld z = cplxld(x.real(), x.imag()) / rational_x_scale;
    cplxld numerator = 0.0L, denominator = 0.0L;
    cplxld d_numerator = 0.0L, d_denominator = 0.0L;
    int hit = -1;
    cplxld anchor = 0.0L;
    if (rational_anchor >= 0)
    {
        const auto point = source_xs[rational_supports[rational_anchor]];
        anchor = cplxld(point.real(), point.imag()) / rational_x_scale;
        denominator = rational_weights[rational_anchor];
    }
    for (int j = 0; j < static_cast<int>(rational_supports.size()); ++j)
    {
        const int index = rational_supports[j];
        if (j == rational_anchor) continue;
        if (x == source_xs[index])
        {
            hit = j;
            continue;
        }
        const cplxld node =
            cplxld(source_xs[index].real(), source_xs[index].imag()) / rational_x_scale;
        const cplxld delta = z - node;
        const cplxld value =
            cplxld(source_data[index].real(), source_data[index].imag()) / rational_y_scale;
        const cplxld term = rational_weights[j] / delta;
        if (rational_anchor >= 0)
        {
            // Explicitly cancel the even-N leading numerator coefficient.
            const cplxld distance = node - anchor;
            numerator += term * value * distance;
            denominator += term * (z - anchor);
            d_numerator -= term * value * distance / delta;
            d_denominator -= term * distance / delta;
        }
        else
        {
            numerator += term * value;
            denominator += term;
            d_numerator -= term * value / delta;
            d_denominator -= term / delta;
        }
    }
    cplxld value, derivative;
    if (hit >= 0 && rational_weights[hit] != cplxld{0.0L, 0.0L})
    {
        const int index = rational_supports[hit];
        value = cplxld(source_data[index].real(), source_data[index].imag()) / rational_y_scale;
        if (rational_anchor >= 0)
        {
            denominator += rational_weights[hit];
            derivative = (numerator - value * denominator) / (rational_weights[hit] * (z - anchor));
        }
        else
            derivative = (numerator - value * denominator) / rational_weights[hit];
    }
    else
    {
        // Zero-weight supports must pass the actual ratio check as well.
        value = numerator / denominator;
        derivative =
            (d_numerator * denominator - numerator * d_denominator) / (denominator * denominator);
    }
    value *= rational_y_scale;
    derivative *= rational_y_scale / rational_x_scale;
    return {{static_cast<double>(value.real()), static_cast<double>(value.imag())},
            {static_cast<double>(derivative.real()), static_cast<double>(derivative.imag())}};
}

cplxdb AnalyContPadeBarycentric::get(const cplxdb &x) const { return evaluate_rational(x).first; }

cplxdb AnalyContPadeBarycentric::get_derivative(const cplxdb &x) const
{
    return evaluate_rational(x).second;
}

cplxdb AnalyContPade::get(const cplxdb &x) const
{
    if (barycentric_fallback) return barycentric_fallback->get(x);
    cplxdb tmp = {1.0, 0.0};

    for (int i_par = n_pars - 1; i_par > 0; i_par--)
    {
        tmp = 1.0 + par_y[i_par] * (x - source_xs[i_par-1]) / tmp;
    }
    const cplxdb value = par_y[0] / tmp;
    return finite_complex(value) ? value : evaluate_continuants(par_y, source_xs, x).first;
}

cplxdb
AnalyContPade::get_derivative(const cplxdb &x) const
{
    if (barycentric_fallback) return barycentric_fallback->get_derivative(x);
    if (n_pars <= 1)
    {
        return {0.0, 0.0};
    }

    std::vector<cplxdb> g(n_pars);
    std::vector<cplxdb> dg(n_pars, {0.0, 0.0});

    g[n_pars-1] = par_y[n_pars-1];
    for (int i_par = n_pars - 2; i_par >= 0; i_par--)
    {
        const cplxdb denominator = 1.0 + (x - source_xs[i_par]) * g[i_par+1];
        g[i_par] = par_y[i_par] / denominator;
        dg[i_par] = -par_y[i_par]
                    * (g[i_par+1] + (x - source_xs[i_par]) * dg[i_par+1])
                    / (denominator * denominator);
    }
    return finite_complex(dg[0]) ? dg[0] : evaluate_continuants(par_y, source_xs, x).second;
}

namespace
{
bool solve_linear_system(std::vector<std::vector<cplxdb>> a, std::vector<cplxdb> b,
                         std::vector<cplxdb> &x)
{
    const int n = static_cast<int>(b.size());
    x.assign(n, {0.0, 0.0});

    for (int k = 0; k < n; k++)
    {
        int pivot = k;
        double pivot_abs = std::abs(a[k][k]);
        for (int i = k + 1; i < n; i++)
        {
            const double cand = std::abs(a[i][k]);
            if (cand > pivot_abs)
            {
                pivot_abs = cand;
                pivot = i;
            }
        }

        if (pivot_abs < 1.0e-28)
        {
            return false;
        }

        if (pivot != k)
        {
            std::swap(a[pivot], a[k]);
            std::swap(b[pivot], b[k]);
        }

        for (int i = k + 1; i < n; i++)
        {
            const cplxdb factor = a[i][k] / a[k][k];
            a[i][k] = {0.0, 0.0};
            for (int j = k + 1; j < n; j++)
            {
                a[i][j] -= factor * a[k][j];
            }
            b[i] -= factor * b[k];
        }
    }

    for (int i = n - 1; i >= 0; i--)
    {
        cplxdb sum = b[i];
        for (int j = i + 1; j < n; j++)
        {
            sum -= a[i][j] * x[j];
        }
        if (std::abs(a[i][i]) < 1.0e-28)
        {
            return false;
        }
        x[i] = sum / a[i][i];
    }

    return true;
}

}  // namespace

AnalyContAAA::AnalyContAAA(int n_support, const std::vector<cplxdb> &xs,
                           const std::vector<cplxdb> &data, double tolerance_in)
    : tolerance(tolerance_in)
{
    if (xs.size() != data.size())
        throw std::invalid_argument("AAA source points and data have different sizes");
    if (xs.empty()) throw std::invalid_argument("AAA requires at least one source point");
    if (!std::isfinite(tolerance) || tolerance <= 0.0)
        throw std::invalid_argument("AAA tolerance must be positive and finite");

    for (int i = 0; i != static_cast<int>(xs.size()); ++i)
    {
        if (!finite_complex(xs[i]) || !finite_complex(data[i]))
            throw std::invalid_argument("AAA source data must be finite");
        const int duplicate = find_exact(source_xs, xs[i]);
        if (duplicate >= 0)
        {
            if (source_data[duplicate] != data[i])
                throw std::invalid_argument("AAA duplicate source points have different data");
            continue;
        }
        source_xs.push_back(xs[i]);
        source_data.push_back(data[i]);
    }

    const int n_data = static_cast<int>(source_xs.size());
    const int max_support = n_support < 0 ? n_data : n_support;
    if (max_support < 1) throw std::invalid_argument("AAA support count must be positive");
    if (n_data == 1 || max_support == 1)
    {
        support_xs = {source_xs[0]};
        support_data = {source_data[0]};
        weights = {{1.0, 0.0}};
        double scale = 0.0;
        for (const auto &value : source_data)
        {
            max_source_error = std::max(max_source_error, std::abs(value - support_data[0]));
            scale = std::max(scale, std::abs(value));
        }
        tolerance_reached = max_source_error <= tolerance * (scale == 0.0 ? 1.0 : scale);
        return;
    }

    int first = 0;
    double data_scale = 0.0;
    for (int i = 0; i != n_data; ++i)
    {
        if (std::abs(source_data[i]) > std::abs(source_data[first])) first = i;
        data_scale = std::max(data_scale, std::abs(source_data[i]));
    }
    // A zero data set has zero absolute scale; use one to keep the test safe.
    if (data_scale == 0.0) data_scale = 1.0;
    std::vector<int> support_indices{first};

    while (true)
    {
        support_xs.clear();
        support_data.clear();
        for (const int index : support_indices)
        {
            support_xs.push_back(source_xs[index]);
            support_data.push_back(source_data[index]);
        }
        if (support_indices.size() == 1) weights = {{1.0, 0.0}};

        double max_error = 0.0;
        double next_error = 0.0;
        int next = -1;
        for (int i = 0; i != n_data; ++i)
        {
            const double error =
                std::abs(source_data[i] -
                         barycentric_value(source_xs[i], support_xs, support_data, weights));
            if (!std::isfinite(error)) throw std::runtime_error("AAA produced non-finite residual");
            max_error = std::max(max_error, error);
            // Zero-weight supports need a residual check but cannot be added twice.
            if (error > next_error && std::find(support_indices.begin(), support_indices.end(),
                                                i) == support_indices.end())
            {
                next_error = error;
                next = i;
            }
        }
        const bool tolerance_met = max_error <= tolerance * data_scale;
        max_source_error = max_error;
        tolerance_reached = tolerance_met;
        const bool support_cap_reached = static_cast<int>(support_indices.size()) >= max_support;
        const bool support_exhausted = static_cast<int>(support_indices.size()) >= n_data - 1;
        if (tolerance_met || support_cap_reached) break;
        if (next < 0)
        {
            if (max_support >= n_data)
                throw std::runtime_error("AAA cannot reach tolerance with a zero-weight support");
            break;
        }
        if (support_exhausted)
        {
            if (max_support >= n_data)
                throw std::runtime_error("AAA did not reach tolerance before n_data-1 supports");
            break;
        }

        support_indices.push_back(next);
        compute_aaa_weights(source_xs, source_data, support_indices, weights);
    }
}

cplxdb AnalyContAAA::get(const cplxdb &x) const
{
    return barycentric_value(x, support_xs, support_data, weights);
}

cplxdb AnalyContAAA::get_derivative(const cplxdb &x) const
{
    const int support = find_exact(support_xs, x);
    if (support >= 0 && weights[support] != cplxdb{0.0, 0.0})
    {
        double weight_scale = 0.0;
        for (const cplxdb &weight : weights)
        {
            if (!finite_complex(weight))
                throw std::runtime_error("AAA support derivative has non-finite weights");
            weight_scale = std::max(weight_scale, std::abs(weight));
        }
        if (!std::isfinite(weight_scale) || weight_scale <= std::numeric_limits<double>::epsilon())
            throw std::runtime_error("AAA support derivative has machine-zero weights");
        const cplxdb support_weight = weights[support];
        if (std::abs(support_weight) <=
            100.0 * std::numeric_limits<double>::epsilon() * weight_scale)
            throw std::runtime_error("AAA support derivative weight is numerically zero");

        cplxdb derivative{0.0, 0.0};
        for (int i = 0; i != static_cast<int>(support_xs.size()); ++i)
        {
            if (i == support) continue;
            const cplxdb dz = support_xs[support] - support_xs[i];
            if (!finite_complex(dz) || dz == cplxdb{0.0, 0.0})
                throw std::runtime_error("AAA support derivative abscissa is singular");
            const cplxdb term =
                weights[i] * (support_data[i] - support_data[support]) / (support_weight * dz);
            if (!finite_complex(term))
                throw std::runtime_error("AAA support derivative is non-finite");
            derivative += term;
        }
        if (!finite_complex(derivative))
            throw std::runtime_error("AAA produced non-finite support derivative");
        return derivative;
    }

    if (!finite_complex(x)) throw std::runtime_error("AAA derivative argument is non-finite");
    cplxdb numerator{0.0, 0.0};
    cplxdb denominator{0.0, 0.0};
    cplxdb numerator_derivative{0.0, 0.0};
    cplxdb denominator_derivative{0.0, 0.0};
    double denominator_scale = 0.0;
    for (int i = 0; i != static_cast<int>(support_xs.size()); ++i)
    {
        if (weights[i] == cplxdb{0.0, 0.0}) continue;
        const cplxdb dz = x - support_xs[i];
        if (!finite_complex(dz) || dz == cplxdb{0.0, 0.0})
            throw std::runtime_error("AAA derivative abscissa is singular");
        const cplxdb term = weights[i] / dz;
        if (!finite_complex(term)) throw std::runtime_error("AAA derivative term is non-finite");
        numerator += term * support_data[i];
        denominator += term;
        numerator_derivative -= term * support_data[i] / dz;
        denominator_derivative -= term / dz;
        denominator_scale += std::abs(term);
    }
    if (!finite_complex(numerator) || !finite_complex(denominator) ||
        !finite_complex(numerator_derivative) || !finite_complex(denominator_derivative) ||
        !std::isfinite(denominator_scale) || denominator_scale <= 0.0 ||
        std::abs(denominator) <= 100.0 * std::numeric_limits<double>::epsilon() * denominator_scale)
        throw std::runtime_error("AAA barycentric derivative denominator is numerically singular");
    const cplxdb derivative =
        (numerator_derivative * denominator - numerator * denominator_derivative) /
        (denominator * denominator);
    if (!finite_complex(derivative)) throw std::runtime_error("AAA produced non-finite derivative");
    return derivative;
}

AnalyContRidge::AnalyContRidge(int n_pars_in, const std::vector<cplxdb> &xs,
                               const std::vector<cplxdb> &data, double lambda,
                               double denominator_weight)
{
    const int n_data = static_cast<int>(data.size());
    if (n_pars_in < 0) n_pars_in = n_data;
    if (n_pars_in == 0 || n_data == 0 || xs.size() != data.size())
        throw std::invalid_argument(
            "Ridge requires matching nonempty data and a nonzero parameter count");
    if (!std::isfinite(lambda) || lambda < 0.0 || !std::isfinite(denominator_weight) ||
        denominator_weight < 0.0)
        throw std::invalid_argument("Ridge penalties must be finite and nonnegative");
    for (int i = 0; i < n_data; ++i)
        if (!finite_complex(xs[i]) || !finite_complex(data[i]))
            throw std::invalid_argument("Ridge source data must be finite");
    source_xs = xs;
    source_data = data;

    const int n_unknowns = std::min(std::max(n_pars_in, 1), n_data);
    const int n_num = (n_unknowns + 1) / 2;
    const int n_den = n_unknowns - n_num;

    x_scale = 0.0;
    y_scale = 0.0;
    for (int i = 0; i < n_data; i++)
    {
        x_scale = std::max(x_scale, std::abs(xs[i]));
        y_scale = std::max(y_scale, std::abs(data[i]));
    }
    if (x_scale < 1.0e-30)
    {
        x_scale = 1.0;
    }
    if (y_scale < 1.0e-30)
    {
        y_scale = 1.0;
    }

    std::vector<std::vector<cplxdb>> normal(n_unknowns,
                                            std::vector<cplxdb>(n_unknowns, {0.0, 0.0}));
    std::vector<cplxdb> rhs(n_unknowns, {0.0, 0.0});

    for (int i = 0; i < n_data; i++)
    {
        const cplxdb x_scaled = xs[i] / x_scale;
        const cplxdb y_scaled = data[i] / y_scale;
        std::vector<cplxdb> row(n_unknowns, {0.0, 0.0});

        cplxdb x_power = {1.0, 0.0};
        for (int j = 0; j < n_num; j++)
        {
            row[j] = x_power;
            x_power *= x_scaled;
        }

        x_power = x_scaled;
        for (int j = 0; j < n_den; j++)
        {
            row[n_num + j] = -y_scaled * x_power;
            x_power *= x_scaled;
        }

        for (int j = 0; j < n_unknowns; j++)
        {
            rhs[j] += std::conj(row[j]) * y_scaled;
            for (int k = 0; k < n_unknowns; k++)
            {
                normal[j][k] += std::conj(row[j]) * row[k];
            }
        }
    }

    const double den_weight = denominator_weight;
    for (int j = 1; j < n_num; j++)
    {
        normal[j][j] += lambda * static_cast<double>(j * j);
    }
    for (int j = 0; j < n_den; j++)
    {
        const int power = j + 1;
        normal[n_num + j][n_num + j] += lambda * den_weight * static_cast<double>(power * power);
    }

    std::vector<cplxdb> solution;
    bool solved = solve_linear_system(normal, rhs, solution);
    for (int attempt = 0; !solved && attempt < 5; attempt++)
    {
        auto shifted = normal;
        const double shift = std::pow(10.0, -14 + attempt);
        for (int j = 0; j < n_unknowns; j++)
        {
            shifted[j][j] += shift;
        }
        solved = solve_linear_system(shifted, rhs, solution);
    }

    if (!solved)
    {
        throw std::runtime_error("Ridge normal equations are singular after diagonal shifts");
    }

    numerator.assign(solution.begin(), solution.begin() + n_num);
    denominator.assign(n_den + 1, {0.0, 0.0});
    denominator[0] = {1.0, 0.0};
    for (int j = 0; j < n_den; j++)
    {
        denominator[j + 1] = solution[n_num + j];
    }

    for (const auto &coeff : numerator)
    {
        if (!finite_complex(coeff))
            throw std::runtime_error("Ridge produced non-finite coefficients");
    }
    for (const auto &coeff : denominator)
    {
        if (!finite_complex(coeff))
            throw std::runtime_error("Ridge produced non-finite coefficients");
    }
}

namespace
{
std::pair<cplxdb, cplxdb> polynomial_value_derivative(const std::vector<cplxdb> &coefficients,
                                                      const cplxdb &x)
{
    cplxdb value{0.0, 0.0}, derivative{0.0, 0.0};
    for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it)
    {
        derivative = derivative * x + value;
        value = value * x + *it;
    }
    return {value, derivative};
}

void check_ridge_denominator(const cplxdb &denominator)
{
    if (!finite_complex(denominator) || std::abs(denominator) < 1.0e-12)
        throw std::runtime_error("Ridge denominator is numerically singular");
}
}  // namespace

cplxdb AnalyContRidge::get(const cplxdb &x) const
{
    const auto p = polynomial_value_derivative(numerator, x / x_scale);
    const auto q = polynomial_value_derivative(denominator, x / x_scale);
    check_ridge_denominator(q.first);
    const auto value = y_scale * p.first / q.first;
    if (!finite_complex(value)) throw std::runtime_error("Ridge produced a non-finite value");
    return value;
}

cplxdb AnalyContRidge::get_derivative(const cplxdb &x) const
{
    const auto p = polynomial_value_derivative(numerator, x / x_scale);
    const auto q = polynomial_value_derivative(denominator, x / x_scale);
    check_ridge_denominator(q.first);
    const auto derivative =
        (y_scale / x_scale) * (p.second * q.first - p.first * q.second) / (q.first * q.first);
    if (!finite_complex(derivative))
        throw std::runtime_error("Ridge produced a non-finite derivative");
    return derivative;
}

std::unique_ptr<AnalyCont> make_analycont(LibrpaAnalyContMethod method, int n_params,
                                          const std::vector<cplxdb> &xs,
                                          const std::vector<cplxdb> &data, double ridge_lambda,
                                          double ridge_denominator_weight, double aaa_tolerance)
{
    if (method == LIBRPA_ANACON_PADE) return std::make_unique<AnalyContPade>(n_params, xs, data);
    if (method == LIBRPA_ANACON_RIDGE)
        return std::make_unique<AnalyContRidge>(n_params, xs, data, ridge_lambda,
                                                ridge_denominator_weight);
    if (method == LIBRPA_ANACON_AAA)
        return std::make_unique<AnalyContAAA>(n_params, xs, data, aaa_tolerance);
    throw std::invalid_argument("Unknown analytic continuation method");
}

const std::vector<double> get_specfunc(const AnalyCont &ac, const std::vector<cplxdb> &omegas,
                                       const double &ref, const double &e_ks, const double &v_xc,
                                       const double &v_exx,
                                       const double &sigc_omega_imag_shift,
                                       const double &gf_omega_imag_shift)
{
    const int n_freq = omegas.size();
    std::vector<double> sf(n_freq, 0.0);
    for (int i = 0; i < n_freq; i++)
    {
        const auto omega_gf = omegas[i] + cplxdb(0.0, gf_omega_imag_shift);
        const auto omega_sigc = omegas[i] + cplxdb(0.0, sigc_omega_imag_shift);
        cplxdb sf_c = omega_gf - e_ks + v_xc - v_exx - ac.get(omega_sigc - ref);
        sf[i] = - ((1.0 / PI) / sf_c).imag();
    }
    return sf;
}

}
