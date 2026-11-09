#include "correlation_potential.h"

#include <cmath>
#include <stdexcept>
#include <string>

#include "../analycont.h"

namespace librpa_int
{
namespace qsgw
{
namespace
{

bool finite_complex(const cplxdb value)
{
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

void validate_parameter_count(const int count,
                              const std::size_t data_size,
                              const char* label)
{
    if (count < -1 || count == 0 || (count == 1 && data_size > 1))
    {
        throw std::invalid_argument(
            std::string(label) +
            " must be -1 or at least 2 for a multi-point continuation");
    }
}

void validate_contract(
    const MeanField& meanfield,
    const std::vector<double>& source_frequencies,
    const std::map<double, Matz>& sigma,
    const int spin,
    const int kpoint,
    const CorrelationPotentialSettings& settings)
{
    if (!meanfield.initialized() || spin < 0 ||
        spin >= meanfield.get_n_spins() || kpoint < 0 ||
        kpoint >= meanfield.get_n_kpoints())
    {
        throw std::invalid_argument(
            "QSGW correlation potential received an invalid mean-field index");
    }
    if (!std::isfinite(meanfield.get_efermi()))
    {
        throw std::invalid_argument(
            "QSGW correlation potential Fermi energy is non-finite");
    }
    for (int band = 0; band < meanfield.get_n_bands(); ++band)
    {
        if (!std::isfinite(meanfield.get_eigenvals()[spin](kpoint, band)))
        {
            throw std::invalid_argument(
                "QSGW correlation potential eigenvalue is non-finite");
        }
    }

    if (source_frequencies.size() < 2 ||
        sigma.size() != source_frequencies.size())
    {
        throw std::invalid_argument(
            "QSGW correlation potential requires matching multi-point frequency data");
    }
    double previous = -1.0;
    for (const double frequency : source_frequencies)
    {
        if (!std::isfinite(frequency) || frequency <= 0.0 ||
            frequency <= previous)
        {
            throw std::invalid_argument(
                "QSGW imaginary frequencies must be finite, positive, and strictly increasing");
        }
        previous = frequency;
        const auto sigma_it = sigma.find(frequency);
        if (sigma_it == sigma.end())
        {
            throw std::invalid_argument(
                "QSGW self-energy map is missing a requested frequency");
        }
        const Matz& matrix = sigma_it->second;
        if (matrix.nr() != meanfield.get_n_bands() ||
            matrix.nc() != meanfield.get_n_bands())
        {
            throw std::invalid_argument(
                "QSGW self-energy matrix has an invalid shape");
        }
        for (int row = 0; row < matrix.nr(); ++row)
        {
            for (int column = 0; column < matrix.nc(); ++column)
            {
                if (!finite_complex(matrix(row, column)))
                {
                    throw std::invalid_argument(
                        "QSGW self-energy matrix contains non-finite data");
                }
            }
        }
    }

    validate_parameter_count(settings.n_params_anacon,
                             settings.resample_imagfreqs.empty()
                                 ? source_frequencies.size()
                                 : settings.resample_imagfreqs.size(),
                             "n_params_anacon");
    if (!settings.resample_imagfreqs.empty())
    {
        validate_parameter_count(settings.n_params_anacon_resample,
                                 source_frequencies.size(),
                                 "n_params_anacon_resample");
        double previous_imaginary = -1.0;
        for (const cplxdb frequency : settings.resample_imagfreqs)
        {
            if (!finite_complex(frequency) || frequency.real() != 0.0 ||
                frequency.imag() <= 0.0 ||
                frequency.imag() <= previous_imaginary)
            {
                throw std::invalid_argument(
                    "QSGW resampling points must be strictly increasing positive imaginary frequencies");
            }
            previous_imaginary = frequency.imag();
        }
    }

    if (settings.mode != CorrelationPotentialMode::ModeA &&
        settings.mode != CorrelationPotentialMode::ModeB)
    {
        throw std::invalid_argument(
            "QSGW correlation potential mode is invalid");
    }
}

std::unique_ptr<AnalyCont> fit_element(const std::vector<cplxdb>& source_points,
                                       const std::vector<const Matz*>& source_matrices,
                                       const int row, const int column,
                                       const CorrelationPotentialSettings& settings)
{
    std::vector<cplxdb> source_data;
    source_data.reserve(source_matrices.size());
    for (const Matz* matrix : source_matrices)
    {
        source_data.push_back((*matrix)(row, column));
    }

    if (settings.resample_imagfreqs.empty())
    {
        return make_analycont(settings.anacon_method, settings.n_params_anacon, source_points,
                              source_data, settings.anacon_ridge_lambda,
                              settings.anacon_ridge_den_weight, settings.anacon_aaa_tolerance);
    }

    const auto source_pade =
        make_analycont(settings.anacon_method, settings.n_params_anacon_resample, source_points,
                       source_data, settings.anacon_ridge_lambda, settings.anacon_ridge_den_weight,
                       settings.anacon_aaa_tolerance);
    std::vector<cplxdb> resampled_data;
    resampled_data.reserve(settings.resample_imagfreqs.size());
    for (const cplxdb frequency : settings.resample_imagfreqs)
    {
        const cplxdb value = source_pade->get(frequency);
        if (!finite_complex(value))
        {
            throw std::runtime_error(
                "QSGW first-stage analytic continuation produced non-finite data");
        }
        resampled_data.push_back(value);
    }
    return make_analycont(settings.anacon_method, settings.n_params_anacon,
                          settings.resample_imagfreqs, resampled_data, settings.anacon_ridge_lambda,
                          settings.anacon_ridge_den_weight, settings.anacon_aaa_tolerance);
}

cplxdb evaluate_element(const AnalyCont& pade, const double target_energy)
{
    const cplxdb result = pade.get(cplxdb(target_energy, 0.0));
    if (!finite_complex(result))
    {
        throw std::runtime_error("QSGW analytic continuation produced non-finite data");
    }
    return result;
}

}  // namespace

Matz build_qsgw_correlation_potential(
    const MeanField& meanfield,
    const std::vector<double>& source_frequencies,
    const std::map<double, Matz>& sigma_imaginary_axis,
    const int spin,
    const int kpoint,
    const CorrelationPotentialSettings& settings)
{
    validate_contract(meanfield, source_frequencies, sigma_imaginary_axis,
                      spin, kpoint, settings);
    const int dimension = meanfield.get_n_bands();
    const double fermi_energy = meanfield.get_efermi();

    std::vector<cplxdb> source_points;
    std::vector<const Matz*> source_matrices;
    source_points.reserve(source_frequencies.size());
    source_matrices.reserve(source_frequencies.size());
    for (const double frequency : source_frequencies)
    {
        source_points.emplace_back(0.0, frequency);
        source_matrices.push_back(&sigma_imaginary_axis.at(frequency));
    }

    Matz result(dimension, dimension, MAJOR::ROW);
    for (int row = 0; row < dimension; ++row)
    {
        const double row_energy = meanfield.get_eigenvals()[spin](kpoint, row) - fermi_energy;
        const auto diagonal_pade = fit_element(source_points, source_matrices, row, row, settings);
        const cplxdb diagonal = evaluate_element(*diagonal_pade, row_energy);
        result(row, row) = 0.5 * (diagonal + std::conj(diagonal));

        for (int column = row + 1; column < dimension; ++column)
        {
            // Fit both ordered elements: Sigma(i omega) need not be Hermitian.
            // Reuse each fit only at the energies required by the static potential.
            const auto forward_pade =
                fit_element(source_points, source_matrices, row, column, settings);
            const auto reverse_pade =
                fit_element(source_points, source_matrices, column, row, settings);
            if (settings.mode == CorrelationPotentialMode::ModeB)
            {
                result(row, column) = 0.5 * (evaluate_element(*forward_pade, 0.0) +
                                             std::conj(evaluate_element(*reverse_pade, 0.0)));
            }
            else
            {
                const double column_energy =
                    meanfield.get_eigenvals()[spin](kpoint, column) - fermi_energy;
                const cplxdb at_row_energy =
                    0.5 * (evaluate_element(*forward_pade, row_energy) +
                           std::conj(evaluate_element(*reverse_pade, row_energy)));
                const cplxdb at_column_energy =
                    0.5 * (evaluate_element(*forward_pade, column_energy) +
                           std::conj(evaluate_element(*reverse_pade, column_energy)));
                result(row, column) = 0.5 * (at_row_energy + at_column_energy);
            }
            result(column, row) = std::conj(result(row, column));
        }
    }
    return result;
}

}  // namespace qsgw
}  // namespace librpa_int
