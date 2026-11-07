#include "crpa_continuation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "../utils/error.h"
#include "analycont.h"

namespace librpa_int
{
std::vector<CrpaContinuedTensor> continue_crpa_interactions(
    const CrpaResult& result, const std::vector<std::complex<double>>& queries, int n_parameters,
    double tolerance, LibrpaAnalyContMethod method, double ridge_lambda, double ridge_den_weight,
    double aaa_tolerance)
{
    using Z = std::complex<double>;
    const auto finite = [](Z value)
    { return std::isfinite(value.real()) && std::isfinite(value.imag()); };
    if (queries.empty() || !std::isfinite(tolerance) || tolerance <= 0 || n_parameters == 0)
        throw LIBRPA_INVALID_ARGUMENT(
            "cRPA continuation requires queries, a positive tolerance and parameters");
    for (auto query : queries)
        if (!finite(query) || (query != Z{} && query.imag() <= 0))
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA continuation queries must be zero or in the upper half plane");
    for (double frequency : result.frequencies)
        if (!std::isfinite(frequency) || frequency <= 0)
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA continuation needs original positive imaginary nodes");
    const auto nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<CrpaContinuedTensor> output;
    for (const auto& tensor : result.tensors)
    {
        const int pairs = tensor.bare.nr;
        if (tensor.bare.nc != pairs || tensor.u.size() != result.frequencies.size())
            throw LIBRPA_INVALID_ARGUMENT("cRPA continuation has inconsistent tensor dimensions");
        CrpaContinuedTensor record{result.sites.at(tensor.site_index).atom_index,
                                   tensor.spin_left,
                                   tensor.spin_right,
                                   result.sites.at(tensor.site_index).n_orbitals,
                                   {},
                                   {},
                                   {}};
        for (std::size_t q = 0; q < queries.size(); ++q)
        {
            record.values.emplace_back(pairs, pairs);
            record.status.emplace_back(tensor.bare.size, 0);
            record.spread.emplace_back(tensor.bare.size, 0);
        }
        const int nodes = result.frequencies.size();
        std::vector<int> all(nodes), alternate;
        std::iota(all.begin(), all.end(), 0);
        for (int i = 0; i < nodes; i += 2) alternate.push_back(i);
        if (!alternate.empty() && alternate.back() != nodes - 1) alternate.push_back(nodes - 1);
        for (int entry = 0; entry < tensor.bare.size; ++entry)
        {
            int common_status = nodes < 4 ? 16 : 0;
            std::vector<Z> data;
            double scale = std::max(1.0, std::abs(tensor.bare.c[entry]));
            for (const auto& value : tensor.u)
            {
                if (value.nr != pairs || value.nc != pairs)
                    throw LIBRPA_INVALID_ARGUMENT(
                        "cRPA continuation frequency tensor has inconsistent shape");
                data.push_back(value.c[entry] - tensor.bare.c[entry]);
                scale = std::max(scale, std::abs(data.back()));
            }
            const auto fit = [&](const std::vector<int>& indices, int parameters)
            {
                std::vector<Z> xs, ys;
                for (int index : indices)
                {
                    xs.emplace_back(0, result.frequencies[index]);
                    ys.push_back(data[index]);
                }
                return make_analycont(method, parameters, xs, ys, ridge_lambda, ridge_den_weight,
                                      aaa_tolerance);
            };
            std::vector<Z> primary(queries.size(), Z{nan, nan});
            std::vector<double> spread(queries.size(), 0);
            bool constant = !data.empty() && finite(data.front());
            for (auto value : data)
                constant = constant && finite(value) &&
                           std::abs(value - data.front()) <=
                               std::min(64 * std::numeric_limits<double>::epsilon() * scale,
                                        0.1 * tolerance);
            if (constant)
                std::fill(primary.begin(), primary.end(), data.front() + tensor.bare.c[entry]);
            else
            {
                try
                {
                    const auto approximation = fit(all, n_parameters);
                    for (std::size_t q = 0; q < queries.size(); ++q)
                        primary[q] = approximation->get(queries[q]) + tensor.bare.c[entry];
                    double source_residual = 0;
                    for (int node : all)
                    {
                        const auto value = approximation->get(Z{0, result.frequencies[node]});
                        if (!finite(value))
                            throw LIBRPA_RUNTIME_ERROR("nonfinite source reconstruction");
                        source_residual = std::max(source_residual, std::abs(value - data[node]));
                    }
                    if (!std::isfinite(source_residual)) common_status |= 1;
                    record.max_source_residual =
                        std::max(record.max_source_residual, source_residual);
                    if (source_residual > tolerance) common_status |= 8;
                    if (nodes >= 4)
                        for (int variant = 0; variant < 2; ++variant)
                        {
                            try
                            {
                                const auto reduced =
                                    fit(variant == 0 ? all : alternate,
                                        variant == 0
                                            ? std::max(2, (n_parameters < 0
                                                               ? nodes
                                                               : std::min(nodes, n_parameters)) -
                                                              2)
                                            : -1);
                                for (std::size_t q = 0; q < queries.size(); ++q)
                                {
                                    const auto estimate =
                                        reduced->get(queries[q]) + tensor.bare.c[entry];
                                    if (!finite(estimate))
                                        throw LIBRPA_RUNTIME_ERROR("nonfinite subset estimate");
                                    spread[q] =
                                        std::max(spread[q], std::abs(estimate - primary[q]));
                                }
                                double holdout = 0;
                                for (int node : all)
                                {
                                    const auto estimate =
                                        reduced->get(Z{0, result.frequencies[node]});
                                    if (!finite(estimate))
                                        throw LIBRPA_RUNTIME_ERROR("nonfinite holdout estimate");
                                    holdout = std::max(holdout, std::abs(estimate - data[node]));
                                }
                                record.max_holdout_residual =
                                    std::max(record.max_holdout_residual, holdout);
                                if (holdout > tolerance) common_status |= 8;
                            }
                            catch (const std::exception&)
                            {
                                common_status |= 2;
                            }
                        }
                }
                catch (const std::exception&)
                {
                    common_status |= 1;
                }
            }
            for (std::size_t q = 0; q < queries.size(); ++q)
            {
                record.values[q].c[entry] = primary[q];
                record.spread[q][entry] = spread[q];
                record.status[q][entry] =
                    common_status | (spread[q] > tolerance ? 4 : 0) | (finite(primary[q]) ? 0 : 32);
            }
        }
        output.push_back(std::move(record));
    }
    return output;
}
}  // namespace librpa_int
