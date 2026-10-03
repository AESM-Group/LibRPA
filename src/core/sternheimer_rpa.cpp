#include "sternheimer_rpa.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../io/global_io.h"
#include "../math/lapack_connector.h"
#include "../utils/constants.h"
#include "librpa_enums.h"

namespace librpa_int
{
namespace
{

ComplexMatrix hermitize(const ComplexMatrix &mat)
{
    if (mat.nr != mat.nc)
    {
        throw std::logic_error("Cannot hermitize a non-square matrix");
    }

    ComplexMatrix result(mat);
    for (int i = 0; i != mat.nr; ++i)
    {
        result(i, i) = {result(i, i).real(), 0.0};
        for (int j = i + 1; j != mat.nc; ++j)
        {
            const auto value = 0.5 * (mat(i, j) + std::conj(mat(j, i)));
            result(i, j) = value;
            result(j, i) = std::conj(value);
        }
    }
    return result;
}

ComplexMatrix filtered_inverse_sqrt_hemat(ComplexMatrix mat, const double threshold)
{
    const char jobz = 'V';
    const char uplo = 'U';
    const int n = mat.nc;
    const int nb = LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1);
    const int lwork = std::max(1, n * (nb + 1));
    int info = 0;
    std::vector<double> eigenvalues(n);
    std::vector<std::complex<double>> work(lwork);
    std::vector<double> rwork(std::max(1, 3 * n - 2));

    mat *= -1.0;
    LapackConnector::zheev(jobz, uplo, n, mat, n, eigenvalues.data(), work.data(), lwork,
                           rwork.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("ST-RPA Coulomb eigensolver failed with info=" +
                                 std::to_string(info));
    }

    std::vector<double> inverse_sqrt(n, 0.0);
    for (int i = 0; i != n; ++i)
    {
        eigenvalues[i] = -eigenvalues[i];
        if (eigenvalues[i] > threshold)
        {
            inverse_sqrt[i] = 1.0 / std::sqrt(eigenvalues[i]);
        }
        else if (eigenvalues[i] < -1.0e-12)
        {
            global::lib_printf(LIBRPA_VERBOSE_WARN,
                               "Warning! ST-RPA filters negative Coulomb eigenvalue: # %d ev = "
                               "%.12e threshold = %.12e\n",
                               i, eigenvalues[i], threshold);
        }
    }

    ComplexMatrix evconj = transpose(mat, true);
    for (int i = 0; i != mat.nr; ++i)
    {
        for (int j = 0; j != mat.nc; ++j)
        {
            evconj.c[i * mat.nc + j] *= inverse_sqrt[i];
        }
    }
    return mat * evconj;
}

struct CoulombEigenbasis
{
    ComplexMatrix eigenvectors;
    std::vector<double> eigenvalues;
    int active_size = 0;
};

CoulombEigenbasis diagonalize_coulomb(ComplexMatrix mat, const double threshold)
{
    const char jobz = 'V';
    const char uplo = 'U';
    const int n = mat.nc;
    const int nb = LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1);
    const int lwork = std::max(1, n * (nb + 1));
    int info = 0;
    std::vector<double> eigenvalues(n);
    std::vector<std::complex<double>> work(lwork);
    std::vector<double> rwork(std::max(1, 3 * n - 2));

    mat *= -1.0;
    LapackConnector::zheev(jobz, uplo, n, mat, n, eigenvalues.data(), work.data(), lwork,
                           rwork.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("ST-RPA Coulomb eigensolver failed with info=" +
                                 std::to_string(info));
    }

    int active_size = 0;
    for (int i = 0; i != n; ++i)
    {
        eigenvalues[i] = -eigenvalues[i];
        if (eigenvalues[i] > threshold)
        {
            ++active_size;
        }
        else if (eigenvalues[i] < -1.0e-12)
        {
            global::lib_printf(LIBRPA_VERBOSE_WARN,
                               "Warning! ST-RPA filters negative Coulomb eigenvalue: # %d ev = "
                               "%.12e threshold = %.12e\n",
                               i, eigenvalues[i], threshold);
        }
    }
    return {std::move(mat), std::move(eigenvalues), active_size};
}

ComplexMatrix project_response_to_coulomb_eigenbasis(const CoulombEigenbasis &basis,
                                                     const ComplexMatrix &response_m)
{
    const int active = basis.active_size;
    const auto transformed =
        transpose(basis.eigenvectors, true) * response_m * basis.eigenvectors;
    ComplexMatrix projected(active, active);
    for (int i = 0; i != active; ++i)
    {
        for (int j = 0; j != active; ++j)
        {
            projected(i, j) = transformed(i, j) /
                              std::sqrt(basis.eigenvalues[i] * basis.eigenvalues[j]);
        }
    }
    return hermitize(projected);
}

ComplexMatrix invert_general_matrix(ComplexMatrix matrix)
{
    if (matrix.nr != matrix.nc)
    {
        throw std::logic_error("Cannot invert a non-square matrix");
    }
    const int n = matrix.nr;
    std::vector<int> ipiv(std::max(1, n));
    int info = 0;
    LapackConnector::zgetrf(n, n, matrix, n, ipiv.data(), &info);
    if (info != 0)
    {
        throw std::runtime_error("ST-RPA head/wing body factorization failed with info=" +
                                 std::to_string(info));
    }
    const int nb = LapackConnector::ilaenv(1, "zgetri", "", n, -1, -1, -1);
    const int lwork = std::max(1, n * std::max(1, nb));
    std::vector<std::complex<double>> work(lwork);
    LapackConnector::zgetri(n, matrix, n, ipiv.data(), work.data(), lwork, &info);
    if (info != 0)
    {
        throw std::runtime_error("ST-RPA head/wing body inversion failed with info=" +
                                 std::to_string(info));
    }
    return matrix;
}

std::complex<double> directional_quadratic_form(const ComplexMatrix &matrix,
                                                const std::array<double, 3> &direction)
{
    std::complex<double> value = 0.0;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        for (int beta = 0; beta != 3; ++beta)
        {
            value += direction[alpha] * matrix(alpha, beta) * direction[beta];
        }
    }
    return value;
}

SternheimerRpaFrequencyResult make_frequency_result(const int ifreq, const double omega,
                                                    const double weight, const double qweight,
                                                    const std::complex<double> &trace_pi,
                                                    const std::complex<double> &logdet)
{
    SternheimerRpaFrequencyResult result;
    result.ifreq = ifreq;
    result.omega = omega;
    result.weight = weight;
    result.qweight = qweight;
    result.trace_pi = trace_pi;
    result.logdet = logdet;
    result.integrand = trace_pi + logdet;
    result.energy = result.integrand * weight * qweight / TWO_PI;
    return result;
}

}  // namespace

int sternheimer_headwing_frequency_index(const int response_ifreq, const int nfreq)
{
    if (nfreq <= 0 || response_ifreq <= 0 || response_ifreq > nfreq)
    {
        throw std::out_of_range("ST-RPA response frequency index is outside the one-based grid");
    }
    return response_ifreq - 1;
}

std::vector<double> sternheimer_frequency_grid_from_metadata(
    const std::vector<std::pair<int, double>> &metadata, const int expected_nfreq)
{
    if (expected_nfreq <= 0 || metadata.empty())
    {
        throw std::runtime_error("ST-RPA response frequency metadata are empty or invalid");
    }

    std::vector<double> frequencies(static_cast<std::size_t>(expected_nfreq), 0.0);
    std::vector<bool> present(static_cast<std::size_t>(expected_nfreq), false);
    const auto close = [](const double lhs, const double rhs) {
        return std::abs(lhs - rhs) <= 1e-10 * std::max({1.0, std::abs(lhs), std::abs(rhs)});
    };
    for (const auto &[ifreq, omega] : metadata)
    {
        if (ifreq <= 0 || ifreq > expected_nfreq || !std::isfinite(omega) || omega < 0.0)
        {
            throw std::runtime_error("ST-RPA response has invalid frequency metadata");
        }
        const auto index = static_cast<std::size_t>(ifreq - 1);
        if (present[index] && !close(frequencies[index], omega))
        {
            throw std::runtime_error("ST-RPA response frequencies disagree for ifreq=" +
                                     std::to_string(ifreq));
        }
        frequencies[index] = omega;
        present[index] = true;
    }

    for (int ifreq = 1; ifreq <= expected_nfreq; ++ifreq)
    {
        const auto index = static_cast<std::size_t>(ifreq - 1);
        if (!present[index])
        {
            throw std::runtime_error("ST-RPA response frequency grid is missing ifreq=" +
                                     std::to_string(ifreq));
        }
        if (index > 0 && frequencies[index] <= frequencies[index - 1])
        {
            throw std::runtime_error("ST-RPA response frequency grid is not strictly increasing");
        }
    }
    return frequencies;
}

ComplexMatrix compute_sternheimer_pi_from_m(const ComplexMatrix &coulomb,
                                            const ComplexMatrix &response_m,
                                            const double sqrt_coulomb_threshold)
{
    if (coulomb.nr != coulomb.nc || response_m.nr != response_m.nc || coulomb.nr != response_m.nr)
    {
        throw std::logic_error("ST-RPA Coulomb and response matrices must be square and same-size");
    }

    auto coulomb_herm = hermitize(coulomb);
    auto response_herm = hermitize(response_m);
    const auto inv_sqrt_coulomb = filtered_inverse_sqrt_hemat(coulomb_herm, sqrt_coulomb_threshold);
    auto pi = inv_sqrt_coulomb * response_herm * inv_sqrt_coulomb;
    return hermitize(pi);
}

std::complex<double> compute_rpa_trace_log_integrand(const ComplexMatrix &pi)
{
    if (pi.nr != pi.nc)
    {
        throw std::logic_error("RPA trace-log response matrix must be square");
    }

    ComplexMatrix identity_minus_pi(pi.nr, pi.nc);
    identity_minus_pi.set_as_identity_matrix();
    identity_minus_pi -= pi;

    std::complex<double> det_for_rpa(1.0, 0.0);
    std::vector<int> ipiv(pi.nr);
    int info_lu = 0;
    LapackConnector::zgetrf(pi.nr, pi.nc, identity_minus_pi, pi.nr, ipiv.data(), &info_lu);
    if (info_lu != 0)
    {
        throw std::runtime_error("RPA trace-log LU factorization failed with info=" +
                                 std::to_string(info_lu));
    }

    for (int ib = 0; ib != pi.nr; ++ib)
    {
        if (ipiv[ib] != ib + 1)
        {
            det_for_rpa = -det_for_rpa * identity_minus_pi(ib, ib);
        }
        else
        {
            det_for_rpa *= identity_minus_pi(ib, ib);
        }
    }

    return std::log(det_for_rpa) + trace(pi);
}

SternheimerRpaFrequencyResult compute_sternheimer_rpa_frequency(const ComplexMatrix &coulomb,
                                                                const ComplexMatrix &response_m,
                                                                const int ifreq, const double omega,
                                                                const double weight,
                                                                const double qweight,
                                                                const double sqrt_coulomb_threshold)
{
    const auto pi = compute_sternheimer_pi_from_m(coulomb, response_m, sqrt_coulomb_threshold);
    const auto integrand = compute_rpa_trace_log_integrand(pi);

    SternheimerRpaFrequencyResult result;
    result.ifreq = ifreq;
    result.omega = omega;
    result.weight = weight;
    result.qweight = qweight;
    result.trace_pi = trace(pi);
    result.logdet = integrand - result.trace_pi;
    result.integrand = integrand;
    result.energy = integrand * weight * qweight / TWO_PI;
    return result;
}

SternheimerRpaFrequencyResult compute_sternheimer_rpa_frequency_headwing(
    const ComplexMatrix &coulomb, const ComplexMatrix &response_m,
    const SternheimerRpaHeadwingInput &headwing, const int ifreq, const double omega,
    const double weight, const double qweight, const double sqrt_coulomb_threshold)
{
    if (coulomb.nr != coulomb.nc || response_m.nr != response_m.nc || coulomb.nr != response_m.nr)
    {
        throw std::logic_error("ST-RPA Coulomb and response matrices must be square and same-size");
    }
    if (headwing.head.nr != 3 || headwing.head.nc != 3)
    {
        throw std::logic_error("ST-RPA head/wing correction requires a 3x3 analytic head");
    }
    if (headwing.mode != "head_only" && headwing.mode != "qavg")
    {
        throw std::logic_error("ST-RPA head/wing mode must be qavg or head_only");
    }

    const auto basis = diagonalize_coulomb(hermitize(coulomb), sqrt_coulomb_threshold);
    if (basis.active_size < 1)
    {
        throw std::logic_error("ST-RPA head/wing Coulomb subspace is empty");
    }
    auto response = project_response_to_coulomb_eigenbasis(basis, hermitize(response_m));

    if (headwing.mode == "head_only")
    {
        std::complex<double> head_average = 0.0;
        for (int alpha = 0; alpha != 3; ++alpha) head_average += headwing.head(alpha, alpha);
        response(0, 0) = head_average / 3.0;
        const auto integrand = compute_rpa_trace_log_integrand(response);
        return make_frequency_result(ifreq, omega, weight, qweight, trace(response),
                                     integrand - trace(response));
    }

    if (headwing.body_start < 0)
    {
        throw std::logic_error("ST-RPA head/wing body start must be non-negative");
    }
    const int body_start = headwing.body_start;
    if (body_start >= basis.active_size)
    {
        throw std::logic_error("ST-RPA head/wing Coulomb subspace has no regular body channels");
    }
    if (headwing.wing_mu.nr != coulomb.nr || headwing.wing_mu.nc != 3)
    {
        throw std::logic_error("ST-RPA qavg requires an auxiliary-basis wing with three columns");
    }
    if (headwing.directions.empty())
    {
        throw std::logic_error("ST-RPA qavg requires angular quadrature points");
    }

    const int nbody = basis.active_size - body_start;
    ComplexMatrix body(nbody, nbody);
    for (int i = 0; i != nbody; ++i)
    {
        for (int j = 0; j != nbody; ++j)
        {
            body(i, j) = response(i + body_start, j + body_start);
        }
    }
    ComplexMatrix identity_minus_body(nbody, nbody);
    identity_minus_body.set_as_identity_matrix();
    identity_minus_body -= body;
    const auto body_inverse = invert_general_matrix(identity_minus_body);
    const auto body_integrand = compute_rpa_trace_log_integrand(body);
    const auto trace_body = trace(body);
    const auto logdet_body = body_integrand - trace_body;

    ComplexMatrix wing(nbody, 3);
    for (int ibody = 0; ibody != nbody; ++ibody)
    {
        const int ilambda = ibody + body_start;
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            std::complex<double> value = 0.0;
            for (int mu = 0; mu != coulomb.nr; ++mu)
            {
                value += std::conj(basis.eigenvectors(mu, ilambda)) * headwing.wing_mu(mu, alpha);
            }
            wing(ibody, alpha) = std::sqrt(basis.eigenvalues[ilambda]) * value;
        }
    }

    ComplexMatrix schur(3, 3);
    schur.set_as_identity_matrix();
    schur -= headwing.head;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        for (int beta = 0; beta != 3; ++beta)
        {
            std::complex<double> correction = 0.0;
            for (int i = 0; i != nbody; ++i)
            {
                for (int j = 0; j != nbody; ++j)
                {
                    correction += std::conj(wing(i, alpha)) * body_inverse(i, j) * wing(j, beta);
                }
            }
            schur(alpha, beta) -= correction;
        }
    }

    std::complex<double> averaged_trace = 0.0;
    std::complex<double> averaged_logdet = 0.0;
    for (const auto &point : headwing.directions)
    {
        if (!std::isfinite(point.weight) || point.weight < 0.0)
        {
            throw std::logic_error("ST-RPA qavg angular weights must be finite and non-negative");
        }
        averaged_trace += point.weight *
                          (trace_body + directional_quadratic_form(headwing.head, point.direction));
        averaged_logdet +=
            point.weight *
            (logdet_body + std::log(directional_quadratic_form(schur, point.direction)));
    }
    return make_frequency_result(ifreq, omega, weight, qweight, averaged_trace, averaged_logdet);
}

std::complex<double> sum_sternheimer_rpa_energies(
    const std::vector<SternheimerRpaFrequencyResult> &results)
{
    std::complex<double> total(0.0, 0.0);
    for (const auto &result : results)
    {
        total += result.energy;
    }
    return total;
}

}  // namespace librpa_int
