#include "orbital_weights.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include "../math/lapack_connector.h"
#include "../math/rsh.h"
#include "../utils/error.h"

namespace librpa_int
{
namespace
{
ComplexMatrix overlap_sqrt(const ComplexMatrix& s, double tolerance, double& minimum,
                           double& residual)
{
    const int n = s.nr;
    if (n <= 0 || s.nc != n) throw LIBRPA_INVALID_ARGUMENT("invalid orbital overlap shape");
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (!std::isfinite(s(i, j).real()) || !std::isfinite(s(i, j).imag()) ||
                std::abs(s(i, j) - std::conj(s(j, i))) > tolerance)
                throw LIBRPA_INVALID_ARGUMENT("orbital overlap is nonfinite or non-Hermitian");

    ComplexMatrix vectors(s);  // LAPACK overwrites its input; preserve S.
    std::vector<double> values(n), real_work(std::max(1, 3 * n - 2));
    const int work_size =
        std::max(1, n * (LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1) + 1));
    std::vector<std::complex<double>> work(work_size);
    int info = 0;
    LapackConnector::heev('V', 'U', n, vectors.c, n, values.data(), work.data(), work_size,
                          real_work.data(), info);
    if (info || !std::isfinite(values.front()) || values.front() <= 0)
        throw LIBRPA_INVALID_ARGUMENT("orbital overlap must be positive definite");
    minimum = std::min(minimum, values.front());
    auto scaled = vectors;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) scaled(i, j) *= std::sqrt(values[j]);
    auto root = scaled * transpose(vectors, true);
    residual = std::max(residual, (root * root - s).get_max_abs());
    if (residual > tolerance * std::max(1.0, s.get_max_abs()))
        throw LIBRPA_RUNTIME_ERROR("orbital overlap square-root residual exceeds tolerance");
    return root;
}
}  // namespace

OrbitalWeightResult analyze_orbital_weights(const AtomicBasis& basis,
                                            const BasisConvention& convention,
                                            const std::vector<int>& atom_types, const MeanField& mf,
                                            const std::vector<ComplexMatrix>& overlap,
                                            const std::vector<double>& kweights, double tolerance)
{
    if (!basis.has_l_shells() || !is_basis_convention_set(convention) || !mf.initialized() ||
        (mf.get_n_spinor() != 1 && mf.get_n_spinor() != 2) ||
        (mf.get_n_spinor() == 2 && mf.get_n_spins() != 1) ||
        basis.nb_total != static_cast<std::size_t>(mf.get_n_aos()) ||
        atom_types.size() != basis.n_atoms ||
        overlap.size() != static_cast<std::size_t>(mf.get_n_kpoints()) ||
        kweights.size() != overlap.size() || !std::isfinite(tolerance) || tolerance <= 0)
        throw LIBRPA_INVALID_ARGUMENT(
            "orbital analysis requires matching scalar/spinor coefficients, S, basis and k data");
    for (double weight : kweights)
        if (!std::isfinite(weight) || weight < 0)
            throw LIBRPA_INVALID_ARGUMENT(
                "orbital analysis k weights must be finite and nonnegative");
    if (std::abs(std::accumulate(kweights.begin(), kweights.end(), 0.0) - 1) > 1e-10)
        throw LIBRPA_INVALID_ARGUMENT("orbital analysis k weights must sum to one");

    OrbitalWeightResult result;
    result.n_spins = mf.get_n_spins();
    result.n_spinor = mf.get_n_spinor();
    result.n_kpoints = mf.get_n_kpoints();
    result.n_bands = mf.get_n_bands();
    result.n_aos = mf.get_n_aos();
    result.fermi = mf.get_efermi();
    result.highest_occupied = -std::numeric_limits<double>::infinity();
    result.min_overlap_eigenvalue = std::numeric_limits<double>::infinity();
    std::vector<int> ao_group(result.n_aos, -1);
    for (int atom = 0; atom < static_cast<int>(basis.n_atoms); ++atom)
    {
        std::map<std::pair<int, std::string>, int> groups;
        int offset = 0;
        for (int l : basis.get_l_shells(atom))
        {
            for (int m = -l; m <= l; ++m)
            {
                const std::string channel = l == 2 ? (m == 0 || m == 2 ? "eg" : "t2g")
                                                   : (l == 0   ? "s"
                                                      : l == 1 ? "p"
                                                               : "l" + std::to_string(l));
                const auto key = std::make_pair(l, channel);
                if (!groups.count(key))
                {
                    groups[key] = static_cast<int>(result.groups.size());
                    result.groups.push_back({atom, atom_types[atom], l, channel});
                }
                const int ao = static_cast<int>(
                    basis.get_global_index(atom, offset + rsh_m_to_index(l, m, convention.order)));
                ao_group.at(ao) = groups.at(key);
            }
            offset += 2 * l + 1;
        }
        if (offset != static_cast<int>(basis.get_atom_nb(atom)))
            throw LIBRPA_INVALID_ARGUMENT("orbital shell metadata does not cover the AO basis");
    }
    const std::size_t state_count =
        static_cast<std::size_t>(result.n_spins) * result.n_kpoints * result.n_bands;
    result.states.resize(state_count);
    const int ng = static_cast<int>(result.groups.size());
    for (int k = 0; k < result.n_kpoints; ++k)
    {
        const auto& s = overlap[k];
        if (s.nr != result.n_aos)
            throw LIBRPA_INVALID_ARGUMENT("orbital overlap AO dimension mismatch");
        auto root =
            overlap_sqrt(s, tolerance, result.min_overlap_eigenvalue, result.max_sqrt_residual);
        for (int spin = 0; spin < result.n_spins; ++spin)
        {
            std::vector<ComplexMatrix> coefficients, metric_coefficients, lowdin_coefficients;
            for (int component = 0; component < result.n_spinor; ++component)
            {
                const auto* wfc = mf.find_wfc(spin, component, k);
                if (!wfc || wfc->nr != result.n_bands || wfc->nc != result.n_aos)
                    throw LIBRPA_INVALID_ARGUMENT(
                        "orbital analysis requires replicated full eigenvector components");
                coefficients.push_back(transpose(*wfc, false));  // Stored C^T, not C-dagger.
                metric_coefficients.push_back(s * coefficients.back());
                lowdin_coefficients.push_back(root * coefficients.back());
            }
            for (int band = 0; band < result.n_bands; ++band)
            {
                auto& state =
                    result.states[(static_cast<std::size_t>(spin) * result.n_kpoints + k) *
                                      result.n_bands +
                                  band];
                state.energy = mf.get_eigenvals()[spin](k, band);
                state.occupation = mf.get_weight()[spin](k, band) * result.n_kpoints;
                state.kweight = kweights[k];
                if (!std::isfinite(state.energy) || !std::isfinite(state.occupation))
                    throw LIBRPA_INVALID_ARGUMENT("orbital state energy/occupation is nonfinite");
                if (state.occupation > 1e-8)
                    result.highest_occupied = std::max(result.highest_occupied, state.energy);
                state.norm = 0;
                state.mulliken_ao.resize(result.n_aos);
                state.lowdin_ao.resize(result.n_aos);
                state.mulliken.assign(ng, 0);
                state.lowdin.assign(ng, 0);
                for (int ao = 0; ao < result.n_aos; ++ao)
                {
                    double wm = 0, wl = 0;
                    for (int component = 0; component < result.n_spinor; ++component)
                    {
                        wm += std::real(std::conj(coefficients[component](ao, band)) *
                                        metric_coefficients[component](ao, band));
                        wl += std::norm(lowdin_coefficients[component](ao, band));
                    }
                    if (!std::isfinite(wm) || !std::isfinite(wl))
                        throw LIBRPA_INVALID_ARGUMENT("orbital weights are nonfinite");
                    state.norm += wm;
                    state.mulliken_ao[ao] = wm;
                    state.lowdin_ao[ao] = wl;
                    state.mulliken[ao_group[ao]] += wm;
                    state.lowdin[ao_group[ao]] += wl;
                }
                if (state.norm <= 0)
                    throw LIBRPA_INVALID_ARGUMENT("KS state has nonpositive S norm");
                result.max_norm_error = std::max(result.max_norm_error, std::abs(state.norm - 1));
                if (std::abs(state.norm - 1) > tolerance * std::sqrt(result.n_bands))
                    throw LIBRPA_INVALID_ARGUMENT(
                        "KS state S norm exceeds orbital-analysis tolerance");
                for (auto& weight : state.lowdin_ao) weight /= state.norm;
                for (auto& weight : state.lowdin) weight /= state.norm;
                const double sum = std::accumulate(state.lowdin.begin(), state.lowdin.end(), 0.0);
                result.max_sum_error = std::max(result.max_sum_error, std::abs(sum - 1));
                if (std::abs(sum - 1) > tolerance * std::sqrt(result.n_aos))
                    throw LIBRPA_RUNTIME_ERROR("Lowdin orbital weights do not sum to one");
            }
        }
    }
    return result;
}

void gaussian_orbital_pdos(const OrbitalWeightResult& result, const std::vector<double>& grid,
                           double sigma, std::vector<double>& mulliken, std::vector<double>& lowdin)
{
    if (grid.empty() || !std::isfinite(sigma) || sigma <= 0)
        throw LIBRPA_INVALID_ARGUMENT("orbital PDOS requires a grid and positive Gaussian sigma");
    const auto ng = result.groups.size();
    mulliken.assign(result.n_spins * grid.size() * ng, 0);
    lowdin.assign(mulliken.size(), 0);
    const double prefactor = 1 / (std::sqrt(2 * std::acos(-1.0)) * sigma);
    for (std::size_t e = 0; e < grid.size(); ++e)
    {
        if (!std::isfinite(grid[e])) throw LIBRPA_INVALID_ARGUMENT("nonfinite PDOS grid energy");
        for (int spin = 0; spin < result.n_spins; ++spin)
            for (int k = 0; k < result.n_kpoints; ++k)
                for (int band = 0; band < result.n_bands; ++band)
                {
                    const auto& state =
                        result.states[(static_cast<std::size_t>(spin) * result.n_kpoints + k) *
                                          result.n_bands +
                                      band];
                    const double distance = (grid[e] - state.energy) / sigma;
                    const double kernel =
                        state.kweight * prefactor * std::exp(-0.5 * distance * distance);
                    const auto offset = (spin * grid.size() + e) * ng;
                    for (std::size_t g = 0; g < ng; ++g)
                    {
                        mulliken[offset + g] += kernel * state.mulliken[g];
                        lowdin[offset + g] += kernel * state.lowdin[g];
                    }
                }
    }
}
}  // namespace librpa_int
