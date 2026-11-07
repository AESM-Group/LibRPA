#include "task.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <exception>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "../../io/fs.h"
#include "../../io/global_io.h"
#include "../../math/utils_matrix_mpi.h"
#include "../../math/utils_matrix_m_mpi.h"
#include "../../mpi/collective_stage.h"
#include "../../utils/constants.h"
#include "../../utils/profiler.h"
#include "qp_diagnostics.h"
#include "band_remap.h"
#include "../hartree.h"
#include "band_output.h"
#include "checkpoint.h"
#include "convergence.h"
#include "correlation_potential.h"
#include "distributed_matrix.h"
#include "effective_hamiltonian.h"
#include "fixed_basis.h"
#include "hamiltonian_cut.h"
#include "hamiltonian_mixing.h"
#include "iteration_trace.h"
#include "occupation.h"
#include "projection_target.h"

namespace librpa_int::qsgw
{
namespace
{
using SigmaMatrixMap = SpinKFrequencyMatrixMap;
template <class T>
bool enabled(T value)
{
    return value == LIBRPA_SWITCH_ON;
}
void configure_coulomb_head(TaskContext& ds, const LibrpaOptions& opts, const TaskInput& input)
{
    if (enabled(opts.use_2d_dielectric))
    {
        const auto normalization =
            strict_2d_coulomb_head_normalization(ds.pbc, input.auxiliary_monopole_norm_squared);
        const double scale = std::max(normalization.inplane_area_bohr2, input.inplane_area_bohr2);
        if (!(input.inplane_area_bohr2 > 0.0) ||
            std::abs(normalization.inplane_area_bohr2 - input.inplane_area_bohr2) > 1.0e-10 * scale)
            throw LIBRPA_RUNTIME_ERROR(
                "Strict 2D Coulomb metadata area is inconsistent with the input lattice");
        global::lib_printf_root(
            "Strict 2D Coulomb normalization computed from reader-v1 metadata: "
            "area=%20.12e Bohr^2, multipole_norm_squared=%20.12e, "
            "A_lambda=%20.12e, PW_to_auxiliary_scale=%20.12e\n",
            normalization.inplane_area_bohr2, normalization.auxiliary_monopole_norm_squared,
            normalization.auxiliary_head_coefficient, normalization.pw_to_auxiliary_scale);
        ds.p_headwing->configure_strict_2d_coulomb_head(true,
                                                        input.auxiliary_monopole_norm_squared);
    }
    else
        ds.p_headwing->configure_strict_2d_coulomb_head(false);
}
void broadcast_localized(LocalizedMatrix& matrices, const MpiCommHandler& comm, int dimension)
{
    int count = matrices.size();
    comm.bcast(&count, 1, 0);
    auto source = matrices.begin();
    for (int i = 0; i < count; ++i)
    {
        int r[3]{};
        Matz block(dimension, dimension, MAJOR::ROW);
        if (comm.is_root())
        {
            r[0] = source->first.x;
            r[1] = source->first.y;
            r[2] = source->first.z;
            for (int row = 0; row < dimension; ++row)
                for (int col = 0; col < dimension; ++col)
                    block(row, col) = source->second(row, col);
            ++source;
        }
        comm.bcast(r, 3, 0);
        comm.bcast(block.ptr(), block.size(), 0);
        if (!comm.is_root()) matrices[{r[0], r[1], r[2]}] = std::move(block);
    }
}
class ScopedSigmaMatrixRetention
{
public:
    explicit ScopedSigmaMatrixRetention(bool& flag) : flag_(flag), original_(flag_)
    {
        flag_ = true;
    }

    ~ScopedSigmaMatrixRetention() noexcept { flag_ = original_; }

    ScopedSigmaMatrixRetention(const ScopedSigmaMatrixRetention&) = delete;
    ScopedSigmaMatrixRetention& operator=(const ScopedSigmaMatrixRetention&) = delete;

private:
    bool& flag_;
    bool original_;
};

class ScopedLegacyHeadOccupations
{
public:
    explicit ScopedLegacyHeadOccupations(MeanField& meanfield)
        : meanfield_(meanfield), live_weights_(meanfield.get_weight())
    {
        auto& weights = meanfield_.get_weight();
        for (const auto& spin_weights : weights)
        {
            if (spin_weights.nr != meanfield_.get_n_kpoints() ||
                spin_weights.nc != meanfield_.get_n_bands())
            {
                throw std::invalid_argument("QSGW head occupation matrix has an invalid shape");
            }
        }
        for (auto& spin_weights : weights)
        {
            for (int kpoint = 1; kpoint < spin_weights.nr; ++kpoint)
            {
                for (int band = 0; band < spin_weights.nc; ++band)
                    spin_weights(kpoint, band) = spin_weights(0, band);
            }
        }
    }

    ~ScopedLegacyHeadOccupations() noexcept { meanfield_.get_weight().swap(live_weights_); }

    ScopedLegacyHeadOccupations(const ScopedLegacyHeadOccupations&) = delete;
    ScopedLegacyHeadOccupations& operator=(const ScopedLegacyHeadOccupations&) = delete;

private:
    MeanField& meanfield_;
    std::vector<librpa_int::matrix> live_weights_;
};

using librpa_int::collective_all_stage;

template <typename Function>
void collective_root_stage(const librpa_int::MpiCommHandler& communicator, const std::string& label,
                           Function&& function)
{
    std::exception_ptr root_error;
    int failed = 0;
    if (communicator.is_root())
    {
        try
        {
            function();
        }
        catch (const std::exception& error)
        {
            failed = 1;
            root_error = std::current_exception();
            librpa_int::global::lib_printf(LIBRPA_VERBOSE_CRITICAL, "%s failed: %s\n",
                                           label.c_str(), error.what());
        }
        catch (...)
        {
            failed = 1;
            root_error = std::current_exception();
        }
    }
    communicator.bcast(&failed, 1, 0);
    if (failed)
    {
        if (communicator.is_root()) std::rethrow_exception(root_error);
        throw LIBRPA_RUNTIME_ERROR(label + " failed on root");
    }
}

std::string resolve_input_path(const std::string& base, const std::string& path)
{
    return librpa_int::is_absolute_path(path) ? path : librpa_int::join_path(base, path);
}

void prepare_qsgw_symmetry_context(TaskContext& dataset, const LibrpaOptions& opts)
{
    const bool symmetry_reduced_scf_grid =
        dataset.pbc.kfrac_list.size() < dataset.pbc.kfrac_list_full.size();
    if (!symmetry_reduced_scf_grid) return;

    const bool all_symmetry_routes_enabled = enabled(opts.use_symmetry_exx) &&
                                             enabled(opts.use_symmetry_gw) &&
                                             enabled(opts.use_symmetry_rpa);
    if (all_symmetry_routes_enabled) dataset.initialize_symmetry(true);
}

void validate_qsgw_reference(const TaskContext& dataset, const bool compute_band,
                             const LibrpaOptions& opts)
{
    using namespace librpa_int::qsgw;
    validate_projection_target(dataset.mf, dataset.pbc.kfrac_list, dataset.mf.get_n_spins(),
                               dataset.mf.get_n_spinor(), dataset.mf.get_n_aos(), "grid");
    const bool symmetry_reduced_scf_grid =
        dataset.pbc.kfrac_list.size() < dataset.pbc.kfrac_list_full.size();
    if (symmetry_reduced_scf_grid)
    {
        const bool all_symmetry_routes_enabled = enabled(opts.use_symmetry_exx) &&
                                                 enabled(opts.use_symmetry_gw) &&
                                                 enabled(opts.use_symmetry_rpa);
        if (!all_symmetry_routes_enabled || !dataset.symmetry_context.available ||
            dataset.symmetry_context.kstars.size() != dataset.pbc.kfrac_list.size() ||
            dataset.symmetry_context.count_kstar_members() != dataset.pbc.kfrac_list_full.size())
        {
            throw std::invalid_argument(
                "QSGW symmetry-reduced SCF input requires complete EXX/GW/RPA k-star restoration");
        }
    }
    if (compute_band)
    {
        const int complete_dimension = dataset.mf.get_n_aos() * dataset.mf.get_n_spinor();
        if (dataset.mf.get_n_bands() != complete_dimension ||
            dataset.mf_band.get_n_bands() != complete_dimension)
        {
            throw std::invalid_argument(
                "QSGW fixed-basis band rotation requires complete square grid and band references");
        }
        validate_projection_target(dataset.mf_band, dataset.kfrac_band_list,
                                   dataset.mf.get_n_spins(), dataset.mf.get_n_spinor(),
                                   dataset.mf.get_n_aos(), "band");
    }
}

SpinKMatrixMap copy_exx_root(const librpa_int::Exx& exchange, const MeanField& reference)
{
    SpinKMatrixMap result;
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
        {
            const auto spin_it = exchange.exx_KS.find(spin);
            if (spin_it == exchange.exx_KS.end())
                throw std::runtime_error("QSGW EXX spin channel is missing");
            const auto k_it = spin_it->second.find(kpoint);
            if (k_it == spin_it->second.end() || k_it->second.nr() != reference.get_n_bands() ||
                k_it->second.nc() != reference.get_n_bands())
            {
                throw std::runtime_error("QSGW EXX fixed-basis matrix is missing or malformed");
            }
            result[spin][kpoint] = k_it->second.copy();
        }
    }
    return result;
}

SigmaMatrixMap collect_sigma_root(const librpa_int::G0W0& self_energy, const MeanField& reference,
                                  const std::vector<double>& frequencies,
                                  const librpa_int::MpiCommHandler& communicator)
{
    using librpa_int::collect_blacs_matrix_root;
    SigmaMatrixMap result;
    for (int spin = 0; spin < reference.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < reference.get_n_kpoints(); ++kpoint)
        {
            for (const double frequency : frequencies)
            {
                const Matz* local = nullptr;
                const auto spin_it = self_energy.sigc_is_ik_f_KS.find(spin);
                if (spin_it != self_energy.sigc_is_ik_f_KS.end())
                {
                    const auto k_it = spin_it->second.find(kpoint);
                    if (k_it != spin_it->second.end())
                    {
                        const auto f_it = k_it->second.find(frequency);
                        if (f_it != k_it->second.end()) local = &f_it->second;
                    }
                }
                int valid = local != nullptr && local->major() == librpa_int::MAJOR::COL &&
                                    local->nr() == self_energy.desc_sigc_is_ik_f_KS.m_loc() &&
                                    local->nc() == self_energy.desc_sigc_is_ik_f_KS.n_loc()
                                ? 1
                                : 0;
                int all_valid = 0;
                communicator.allreduce(&valid, &all_valid, 1, MPI_MIN);
                if (!all_valid)
                    throw std::runtime_error("QSGW distributed fixed-basis Sigma is incomplete");
                Matz full = collect_blacs_matrix_root(*local, self_energy.desc_sigc_is_ik_f_KS);
                if (communicator.is_root())
                {
                    if (full.nr() != reference.get_n_bands() ||
                        full.nc() != reference.get_n_bands())
                    {
                        throw std::runtime_error("QSGW collected Sigma has an invalid shape");
                    }
                    result[spin][kpoint][frequency] = std::move(full);
                }
            }
        }
    }
    return result;
}

SpinKMatrixMap build_correlation_map(const MeanField& live, const SigmaMatrixMap& sigma,
                                     const std::vector<double>& frequencies,
                                     const LibrpaOptions& options)
{
    using namespace librpa_int::qsgw;
    CorrelationPotentialSettings settings;
    settings.mode = CorrelationPotentialMode::ModeB;
    settings.anacon_method = options.anacon_method;
    settings.anacon_ridge_lambda = options.anacon_ridge_lambda;
    settings.anacon_ridge_den_weight = options.anacon_ridge_den_weight;
    settings.anacon_aaa_tolerance = options.anacon_aaa_tolerance;
    settings.n_params_anacon = options.n_params_anacon;
    settings.n_params_anacon_resample = options.n_params_anacon_resample < 0
                                            ? options.n_params_anacon
                                            : options.n_params_anacon_resample;

    SpinKMatrixMap result;
    for (int spin = 0; spin < live.get_n_spins(); ++spin)
    {
        for (int kpoint = 0; kpoint < live.get_n_kpoints(); ++kpoint)
        {
            result[spin][kpoint] = build_qsgw_correlation_potential(
                live, frequencies, sigma.at(spin).at(kpoint), spin, kpoint, settings);
        }
    }
    return result;
}

// The matrices are already available for QSGW. Rotate to the entering
// eigenframe and reuse G0W0 continuation/roots without a second GW build.
void write_qsgw_qp_state(std::ostream& output, int state_index, int update,
                         const std::string& channel, const MeanField& live,
                         const std::vector<Vector3_Order<double>>& kpoints,
                         const SpinKMatrixMap& rotation, const SpinKMatrixMap& entering_hamiltonian,
                         const SpinKMatrixMap& reference_hamiltonian, const SpinKMatrixMap& dft_vxc,
                         const SpinKMatrixMap* delta_hartree, const SpinKMatrixMap& exchange,
                         const SigmaMatrixMap& sigma, const librpa_int::TFGrids& tfg,
                         const std::vector<double>& kpoint_weights, const LibrpaOptions& options,
                         const TaskControls& controls,
                         const std::function<void(OutputKind, const std::string&, int, int)>& emit)
{
    const int low = std::max(0, controls.i_state_low);
    const int high = std::min(live.get_n_states(), controls.i_state_high);
    const int n = high - low, nk = live.get_n_kpoints(), ns = live.get_n_spins();
    if (n <= 0) return;
    const auto frequencies = tfg.get_freq_nodes();
    std::vector<int> iks(nk);
    std::iota(iks.begin(), iks.end(), 0);
    std::vector<double> potential(ns * nk * n), exx(potential.size());
    std::vector<cplxdb> diagonal;
    diagonal.reserve(potential.size() * frequencies.size());
    const auto static_subtraction = build_qp_static_subtraction(
        entering_hamiltonian, reference_hamiltonian, dft_vxc, delta_hartree);
    for (int spin = 0; spin < ns; ++spin)
        for (int kpoint = 0; kpoint < nk; ++kpoint)
        {
            const auto& u = rotation.at(spin).at(kpoint);
            const auto adjoint = transpose(u, true);
            const auto rotate = [&](const Matz& matrix)
            {
                // BLACS Sigma/EXX are column-major; the fixed-frame rotation
                // can be row-major. Normalize storage before the BLAS product.
                const Matz aligned(matrix.nr(), matrix.nc(), matrix.ptr(), matrix.major(),
                                   u.major());
                return adjoint * aligned * u;
            };
            const auto vstatic = rotate(static_subtraction.at(spin).at(kpoint));
            const auto sigx = rotate(exchange.at(spin).at(kpoint));
            for (int band = low; band < high; ++band)
            {
                const int i = (spin * nk + kpoint) * n + band - low;
                potential[i] = vstatic(band, band).real();
                exx[i] = sigx(band, band).real();
            }
            for (double frequency : frequencies)
            {
                const auto sigc = rotate(sigma.at(spin).at(kpoint).at(frequency));
                for (int band = low; band < high; ++band) diagonal.push_back(sigc(band, band));
            }
        }
    std::vector<double> real(potential.size()), imag(potential.size()), eqp(potential.size());
    std::vector<int> info(potential.size());
    solve_qsgw_qpe_diagonal(live, tfg, options, diagonal, iks, nk, iks.data(), low, n,
                                        potential.data(), exx.data(), real.data(), imag.data(),
                                        eqp.data(), "QSGW entering-state G0W0", info.data());
    for (int spin = 0; spin < ns; ++spin)
        for (int kpoint = 0; kpoint < nk; ++kpoint)
            for (int band = low; band < high; ++band)
            {
                const int i = (spin * nk + kpoint) * n + band - low;
                const auto& k = kpoints.at(kpoint);
                const double weighted_occupation = live.get_weight().at(spin)(kpoint, band);
                const double occupation = channel == "grid"
                                              ? occupation_number_from_kpoint_weight(
                                                    weighted_occupation, kpoint_weights.at(kpoint))
                                              : weighted_occupation * nk;
                output << state_index << ',' << update << ',' << channel << ',' << spin + 1 << ','
                       << kpoint + 1 << ',' << k.x << ',' << k.y << ',' << k.z << ',' << band + 1
                       << ',' << occupation << ','
                       << live.get_eigenvals().at(spin)(kpoint, band) * librpa_int::HA2EV << ','
                       << potential[i] * librpa_int::HA2EV << ',' << exx[i] * librpa_int::HA2EV
                       << ',' << real[i] * librpa_int::HA2EV << ',' << imag[i] * librpa_int::HA2EV
                       << ',' << eqp[i] * librpa_int::HA2EV << ',' << info[i] << '\n';
            }
    output.flush();
    if (!output) throw LIBRPA_RUNTIME_ERROR("Cannot write QSGW quasiparticle output");
    // Use the ordinary G0W0 writers, including their energy/occupation units
    // and selected-state layout. The iteration is encoded in the filename.
    std::vector<librpa_int::matrix> vstatic_diagonal(ns);
    std::vector<cplxdb> correlation(potential.size());
    for (int spin = 0; spin < ns; ++spin)
    {
        vstatic_diagonal[spin].create(nk, live.get_n_states());
        vstatic_diagonal[spin].zero_out();
        for (int kpoint = 0; kpoint < nk; ++kpoint)
            for (int band = low; band < high; ++band)
            {
                const int i = (spin * nk + kpoint) * n + band - low;
                vstatic_diagonal[spin](kpoint, band) = potential[i];
                correlation[i] = cplxdb(real[i], imag[i]);
            }
    }
    if (channel == "grid")
    {
        std::ostringstream table;
        write_energy_qp(table, live, kpoints, {}, vstatic_diagonal, exx, correlation, nk, low, n,
                        kpoint_weights);
        emit(OutputKind::EnergyQp, table.str(), update, 0);
    }
    if (channel == "band")
        for (int spin = 0; spin < ns; ++spin)
        {
            std::ostringstream table;
            write_gw_band_spin_table(table, live, kpoints, vstatic_diagonal, exx, correlation, spin,
                                     low, n);
            emit(OutputKind::GwBand, table.str(), update, spin);
        }
}

}  // namespace
Task::Task(std::shared_ptr<TaskContext> ds, const LibrpaOptions& opts, TaskControls controls)
    : dataset_(std::move(ds)), options_(opts), controls_(std::move(controls))
{
}
void Task::set_input(TaskInput input)
{
    if (ran_) throw LIBRPA_RUNTIME_ERROR("Cannot replace input after a QSGW run");
    input_set_ = false;
    input_ = std::move(input);
    const bool current_hr =
        enabled(options_.replace_w_head) &&
        (options_.option_dielect_func == 3 || options_.option_dielect_func == 4) &&
        controls_.qsgw_headwing_velocity == "current_hr";
    collective_all_stage(
        dataset_->comm_h, "QSGW native operator requirements",
        [&]
        {
            if (dataset_->comm_h.is_root() &&
                (controls_.qsgw_export_hamiltonian_for_pyatb || current_hr) && !input_.ao_operators)
                throw LIBRPA_INVALID_ARGUMENT(
                    "QSGW HR export/current-HR velocity requires native H/S/r operators on root");
        });
    // current-HR derivatives are evaluated collectively on every rank.
    if (current_hr)
    {
        auto& comm = dataset_->comm_h;
        int dim = comm.is_root() && input_.ao_operators ? input_.ao_operators->dimension : 0;
        comm.bcast(&dim, 1, 0);
        if (dim <= 0) throw LIBRPA_RUNTIME_ERROR("QSGW current-HR native operators are missing");
        if (!input_.ao_operators) input_.ao_operators.emplace();
        auto& operators = *input_.ao_operators;
        operators.dimension = dim;
        broadcast_localized(operators.overlap, comm, dim);
        for (auto& position : operators.position) broadcast_localized(position, comm, dim);
        int spins = operators.native_dft_hamiltonian.size();
        comm.bcast(&spins, 1, 0);
        operators.native_dft_hamiltonian.resize(spins);
        const int hamiltonian_dimension = dataset_->mf.get_n_aos() * dataset_->mf.get_n_spinor();
        for (auto& hamiltonian : operators.native_dft_hamiltonian)
            broadcast_localized(hamiltonian, comm, hamiltonian_dimension);
    }
    double normalization[2]{input_.inplane_area_bohr2, input_.auxiliary_monopole_norm_squared};
    dataset_->comm_h.bcast(normalization, 2, 0);
    input_.inplane_area_bohr2 = normalization[0];
    input_.auxiliary_monopole_norm_squared = normalization[1];
    if (controls_.qsgw_update_hartree)
    {
#ifndef LIBRPA_LIBRI_HAS_HARTREE
        throw LIBRPA_INVALID_ARGUMENT(
            "QSGW Hartree feedback requires LibRI Hartree support");
#else
        const auto& comm = dataset_->comm_h;
        const auto& aux =
            enabled(options_.use_shrink_abfs) ? dataset_->basis_aux_shrink : dataset_->basis_aux;
        collective_root_stage(
            comm, "QSGW Hartree input",
            [&]
            {
                if (!input_.hartree_coulomb_q0 || !input_.hartree_cs_raw ||
                    input_.hartree_cs_raw->data_libri.empty())
                    throw LIBRPA_INVALID_ARGUMENT(
                        "QSGW Hartree requires a dedicated G0=0 q0 metric and unwrapped Cs on "
                        "root");
                const auto& v = *input_.hartree_coulomb_q0;
                if (v.nr() != static_cast<int>(aux.nb_total) || v.nc() != v.nr())
                    throw LIBRPA_INVALID_ARGUMENT(
                        "QSGW Hartree metric does not match the selected auxiliary basis");
                for (const auto& row : input_.hartree_cs_raw->data_libri)
                    for (const auto& entry : row.second)
                    {
                        const auto& t = entry.second;
                        if (row.first < 0 || row.first >= static_cast<int>(aux.n_atoms) ||
                            entry.first.first < 0 ||
                            entry.first.first >= static_cast<int>(aux.n_atoms) ||
                            t.shape.size() != 3 || !t.data ||
                            t.shape[0] != aux.get_atom_nb(row.first) ||
                            t.shape[1] != dataset_->basis_wfc.get_atom_nb(row.first) ||
                            t.shape[2] != dataset_->basis_wfc.get_atom_nb(entry.first.first) ||
                            t.data->size() != t.shape[0] * t.shape[1] * t.shape[2])
                            throw LIBRPA_INVALID_ARGUMENT(
                                "QSGW Hartree raw Cs has an incompatible tensor");
                    }
            });
        const int naux = static_cast<int>(aux.nb_total);
        Matz metric(naux, naux, MAJOR::ROW);
        if (comm.is_root())
            for (int i = 0; i != naux; ++i)
                for (int j = 0; j != naux; ++j) metric(i, j) = (*input_.hartree_coulomb_q0)(i, j);
        comm.bcast(metric.ptr(), metric.size(), 0);
        input_.hartree_coulomb_q0 = std::move(metric);
        using RawBlock = std::tuple<int, libri_types<int, int>::TAC, RI::Tensor<double>>;
        std::vector<RawBlock> source;
        if (comm.is_root())
            for (const auto& i : input_.hartree_cs_raw->data_libri)
                for (const auto& jr : i.second) source.emplace_back(i.first, jr.first, jr.second);
        int count = static_cast<int>(source.size());
        comm.bcast(&count, 1, 0);
        if (!comm.is_root()) input_.hartree_cs_raw.emplace();
        input_.hartree_cs_raw->use_libri = true;
        for (int block = 0; block != count; ++block)
        {
            int header[8]{};
            if (comm.is_root())
            {
                const auto& key = std::get<1>(source[block]);
                const auto& tensor = std::get<2>(source[block]);
                header[0] = std::get<0>(source[block]);
                header[1] = key.first;
                for (int d = 0; d != 3; ++d) header[2 + d] = key.second[d];
                for (int d = 0; d != 3; ++d) header[5 + d] = static_cast<int>(tensor.shape[d]);
            }
            comm.bcast(header, 8, 0);
            RI::Tensor<double> tensor({static_cast<std::size_t>(header[5]),
                                       static_cast<std::size_t>(header[6]),
                                       static_cast<std::size_t>(header[7])});
            if (comm.is_root())
                std::copy(std::get<2>(source[block]).ptr(),
                          std::get<2>(source[block]).ptr() + tensor.data->size(), tensor.ptr());
            comm.bcast(tensor.ptr(), tensor.data->size(), 0);
            if (!comm.is_root())
                input_.hartree_cs_raw
                    ->data_libri[header[0]][{header[1], {header[2], header[3], header[4]}}] =
                    std::move(tensor);
        }
#endif
    }
    input_set_ = true;
}
void Task::set_output(std::function<void(const TaskOutput&)> sink) { sink_ = std::move(sink); }
void Task::set_extension(std::shared_ptr<TaskExtension> extension)
{
    if (ran_) throw LIBRPA_RUNTIME_ERROR("Cannot attach a companion after a QSGW run");
    extension_ = std::move(extension);
}
void Task::output_text(OutputKind kind, const std::string& text, int iteration, int spin,
                       bool append) const
{
    if (sink_ && !text.empty()) sink_({kind, iteration, spin, append, text, nullptr});
}
void Task::output_hamiltonian(OutputKind kind, const LocalizedMatrix& h, int iteration,
                              int spin) const
{
    if (sink_) sink_({kind, iteration, spin, false, {}, &h});
}
void Task::prepare()
{
    if (prepared_) return;
    const auto& opts = options_;
    const bool update_head = enabled(opts.replace_w_head) &&
                             (opts.option_dielect_func == 3 || opts.option_dielect_func == 4);
    if (enabled(opts.replace_w_head) && !update_head)
    {
        throw LIBRPA_RUNTIME_ERROR("QSGW analytic head/wing requires option_dielect_func = 3 or 4");
    }
    if (update_head && opts.option_dielect_func == 3 && !enabled(opts.use_scalapack_gw_wc))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "QSGW option_dielect_func = 3 requires use_scalapack_gw_wc = true "
            "to consume analytic wings");
    }
    if (enabled(opts.use_kpara_scf_eigvec))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "QSGW fixed-basis iteration requires replicated SCF wavefunctions; set "
            "use_kpara_scf_eigvec = false");
    }

    const auto& dataset = dataset_;
    auto& ibz_reference = ibz_reference_;
    auto& expanded_reference = expanded_reference_;
    if (dataset->pbc.kfrac_list.size() < dataset->pbc.kfrac_list_full.size() &&
        (controls_.expand_ibz || controls_.qsgw_export_hamiltonian_for_pyatb ||
         controls_.qsgw_update_hartree ||
         (update_head && controls_.qsgw_headwing_velocity == "current_hr")))
    {
        collective_all_stage(
            dataset->comm_h, "QSGW IBZ source expansion",
            [&]
            {
                if (!enabled(opts.use_symmetry_exx) || !enabled(opts.use_symmetry_gw) ||
                    !enabled(opts.use_symmetry_rpa))
                    throw LIBRPA_RUNTIME_ERROR("QSGW IBZ expansion requires EXX/GW/RPA symmetry");
                dataset->initialize_symmetry(true);
                expanded_reference = expand_symmetry_meanfield(
                    dataset->mf, dataset->basis_wfc, dataset->pbc, dataset->symmetry_context);
                ibz_reference = dataset->mf;
            });
        const auto& full = expanded_reference->meanfield;
        dataset->invalidate_compute_objects();
        dataset->mf = MeanField{};
        dataset->initialize_scf_dimension(full.get_n_spins(), full.get_n_kpoints(),
                                    full.get_n_bands(), full.get_n_aos(), full.get_n_spinor());
        dataset->mf = full;
        // Keep the producer's representative Coulomb grid and star weights.
        // Only the live KS grid becomes full; P0 and W may still use q stars.
        auto& pbc = dataset->pbc;
        pbc.klist = pbc.klist_full;
        pbc.kfrac_list = pbc.kfrac_list_full;
        pbc.weight_k.assign(full.get_n_kpoints(), 1.0 / full.get_n_kpoints());
        pbc.k_to_kfull.resize(full.get_n_kpoints());
        std::iota(pbc.k_to_kfull.begin(), pbc.k_to_kfull.end(), 0);
        pbc.kfull_to_k = pbc.k_to_kfull;
        pbc.kfull_to_k_relation.assign(full.get_n_kpoints(), KFullToKRelation::DIRECT);
        global::lib_printf_root(
            "QSGW: expanded %d paired IBZ source states to %d full-grid states; "
            "no eigenproblem was solved\n",
            ibz_reference->get_n_kpoints(), full.get_n_kpoints());
    }
    if (update_head && controls_.qsgw_headwing_velocity == "current_hr")
    {
        const auto& mf = dataset->mf;
        if ((mf.get_n_spinor() != 1 && mf.get_n_spinor() != 2) ||
            (mf.get_n_spinor() == 2 && mf.get_n_spins() != 1) ||
            mf.get_n_bands() != mf.get_n_aos() * mf.get_n_spinor())
            throw LIBRPA_RUNTIME_ERROR(
                "QSGW current-HR velocity requires a complete scalar or two-spinor AO basis; "
                "SOC has one physical spin channel");
        if (dataset->pbc.kfrac_list.size() != dataset->pbc.kfrac_list_full.size() ||
            mf.get_n_kpoints() != dataset->pbc.get_n_cells_bvk())
            throw LIBRPA_RUNTIME_ERROR(
                "QSGW current-HR velocity requires the complete full-BZ SCF mesh; "
                "IBZ velocity sewing is unsupported");
    }
    if (controls_.qsgw_update_hartree)
    {
        collective_all_stage(
            dataset->comm_h, "QSGW Hartree geometry and Fourier convention",
            [&]
            {
                if (enabled(opts.use_2d_dielectric))
                    throw LIBRPA_INVALID_ARGUMENT(
                        "QSGW Hartree requires a bare three-dimensional Coulomb metric");
                if (dataset->basis_convention.bloch_ratom != 0)
                    throw LIBRPA_INVALID_ARGUMENT(
                        "QSGW Hartree requires the native RI translation convention and "
                        "atom-position phase 0");
                const auto& pbc = dataset->pbc;
                const int nk = dataset->mf.get_n_kpoints();
                std::size_t mesh_size = 1;
                for (int period : pbc.period_array)
                {
                    if (period <= 0 || nk <= 0 || mesh_size > static_cast<std::size_t>(nk) / period)
                        throw LIBRPA_INVALID_ARGUMENT(
                            "QSGW Hartree requires a complete regular full-BZ SCF mesh");
                    mesh_size *= period;
                }
                if (mesh_size != static_cast<std::size_t>(nk) || pbc.klist.size() != mesh_size ||
                    pbc.kfrac_list.size() != mesh_size || pbc.kfrac_list_full.size() != mesh_size)
                    throw LIBRPA_INVALID_ARGUMENT(
                        "QSGW Hartree requires a complete regular full-BZ SCF mesh");
                const bool has_gamma =
                    std::any_of(pbc.klist.begin(), pbc.klist.end(),
                                [](const auto& k) { return k.norm() < 1.0e-10; });
                if (!has_gamma)
                    throw LIBRPA_INVALID_ARGUMENT(
                        "QSGW Hartree currently requires a Gamma-containing SCF mesh");
                std::set<std::array<int, 3>> residues;
                for (const auto& k : pbc.kfrac_list)
                {
                    const std::array<double, 3> fraction{k.x, k.y, k.z};
                    std::array<int, 3> residue{};
                    for (int d = 0; d != 3; ++d)
                    {
                        const double index = fraction[d] * pbc.period_array[d];
                        if (!std::isfinite(index) ||
                            std::abs(index) > std::numeric_limits<int>::max() ||
                            std::abs(index - std::round(index)) > 1.0e-8)
                            throw LIBRPA_INVALID_ARGUMENT(
                                "QSGW Hartree requires an unshifted regular SCF mesh");
                        const int integer = static_cast<int>(std::round(index));
                        residue[d] = integer % pbc.period_array[d];
                        if (residue[d] < 0) residue[d] += pbc.period_array[d];
                    }
                    if (!residues.insert(residue).second)
                        throw LIBRPA_INVALID_ARGUMENT(
                            "QSGW Hartree SCF mesh contains equivalent duplicate k points");
                }
            });
    }
    if (opts.parallel_routing != LIBRPA_ROUTING_LIBRI &&
        opts.parallel_routing != LIBRPA_ROUTING_AUTO)
    {
        throw LIBRPA_RUNTIME_ERROR("QSGW requires parallel_routing=libri or auto");
    }
    prepared_ = true;
}

void Task::run()
{
    if (ran_) throw LIBRPA_RUNTIME_ERROR("A QSGW task can run only once");
    prepare();
    if (!input_set_) throw LIBRPA_RUNTIME_ERROR("QSGW numerical input is missing");
    ran_ = true;
    using namespace librpa_int::global;
    const auto& dataset = dataset_;
    const auto& opts = options_;
    const bool compute_band = dataset->mf_band.get_n_kpoints() > 0;
    const bool update_head = enabled(opts.replace_w_head) &&
                             (opts.option_dielect_func == 3 || opts.option_dielect_func == 4);
    const bool update_wing = update_head && opts.option_dielect_func == 3;
    const auto emit = [&](OutputKind kind, const std::string& text, int iteration, int spin)
    { output_text(kind, text, iteration, spin); };
    profiler.start("qsgw", "QSGW fixed-basis self-consistent calculation");
    collective_root_stage(dataset->comm_h, "QSGW reference validation",
                          [&]
                          {
                              prepare_qsgw_symmetry_context(*dataset, opts);
                              validate_qsgw_reference(*dataset, compute_band, opts);
                          });
    const MeanField reference = dataset->mf;
    // Capture the DFT density before checkpoint restoration or any live update.
    // Each matrix owns separate storage; subsequent signed differences never mutate it.
    const auto hartree_reference_density =
        controls_.qsgw_update_hartree ? dataset->build_hartree_density() : std::vector<Matz>{};
    // Use the full SCF mesh as well as the loaded count: one IBZ representative
    // of a multi-k mesh must not enter the single-k supercell branch.
    const bool single_kpoint =
        reference.get_n_kpoints() == 1 && dataset->pbc.get_n_cells_bvk() == 1;
    const std::string effective_mixer = single_kpoint ? controls_.qsgw_mixer : "none";
    const double mixing_beta = effective_mixer == "linear" ? controls_.qsgw_mixing_beta : 1.0;
    if (dataset->comm_h.is_root())
        lib_printf(
            "QSGW mixing from SCF input: loaded_kpoints=%d mesh=%dx%dx%d requested=%s effective=%s "
            "beta=%.8f\n",
            reference.get_n_kpoints(), dataset->pbc.period.x, dataset->pbc.period.y,
            dataset->pbc.period.z, controls_.qsgw_mixer.c_str(), effective_mixer.c_str(),
            mixing_beta);
    const double electron_count = physical_electron_count(reference, dataset->pbc.weight_k);
    const OccupationResult initial_occupations =
        analyze_qsgw_occupations(reference, dataset->pbc.weight_k, electron_count);
    const SpinKMatrixMap reference_hamiltonian = build_reference_hamiltonian(reference);

    VelocityMatrix reference_velocity;
    const auto& ao_operators = input_.ao_operators;
    if (update_head && controls_.qsgw_headwing_velocity != "current_hr")
    {
        reference_velocity = dataset->velocity_matrix;
        if (controls_.interband_velocity)
        {
            prepare_interband_velocity(reference_velocity, reference);
            if (!dataset->p_headwing)
                throw LIBRPA_RUNTIME_ERROR("QSGW reference velocity frame is unavailable");
            align_distributed_velocity_to_reference_wfc(dataset->p_headwing->get_meanfield_df(),
                                                        reference, reference_velocity,
                                                        dataset->comm_h);
        }
    }

    std::optional<MeanField> band_reference;
    SpinKMatrixMap band_reference_hamiltonian;
    if (compute_band)
    {
        dataset->mf_band.get_efermi() = reference.get_efermi();
        band_reference = dataset->mf_band;
        band_reference_hamiltonian = build_reference_hamiltonian(*band_reference);
    }

    SpinKMatrixMap dft_vxc;
    SpinKMatrixMap dft_vxc_band;
    collective_root_stage(
        dataset->comm_h, "QSGW Vxc projection",
        [&]
        {
            const auto& source_reference = ibz_reference_ ? *ibz_reference_ : reference;
            for (int spin = 0; spin < source_reference.get_n_spins(); ++spin)
                for (int k = 0; k < source_reference.get_n_kpoints(); ++k)
                    dft_vxc[spin][k] = prepare_vxc_in_fixed_state_basis(
                        input_.vxc.at(spin).at(k), input_.vxc_basis, source_reference, spin, k);
            if (expanded_reference_)
            {
                const auto source_vxc = dft_vxc;
                dft_vxc.clear();
                for (int spin = 0; spin < reference.get_n_spins(); ++spin)
                    for (int k = 0; k < reference.get_n_kpoints(); ++k)
                    {
                        auto block =
                            source_vxc.at(spin).at(expanded_reference_->source_k[k]).copy();
                        if (expanded_reference_->antiunitary[k])
                            for (std::size_t i = 0; i < block.size(); ++i)
                                block.ptr()[i] = std::conj(block.ptr()[i]);
                        dft_vxc[spin][k] = std::move(block);
                    }
            }
            if (compute_band)
                for (int spin = 0; spin < band_reference->get_n_spins(); ++spin)
                    for (int k = 0; k < band_reference->get_n_kpoints(); ++k)
                        dft_vxc_band[spin][k] = prepare_vxc_in_fixed_state_basis(
                            input_.vxc_band.at(spin).at(k), input_.vxc_basis, *band_reference, spin,
                            k);
        });

    HamiltonianCutOptions cut_options;
    cut_options.unoccupied_keep = controls_.qsgw_unoccupied_keep;
    cut_options.mode = hamiltonian_cut_mode_from_int(controls_.qsgw_cut_mode);
    cut_options.shift_ha = controls_.qsgw_cut_shift_ha;
    const bool apply_cut = cut_options.mode != HamiltonianCutMode::Uncut;
    SpinKMatrixMap current_hamiltonian =
        apply_cut ? apply_hamiltonian_cut(reference_hamiltonian, reference_hamiltonian, dataset->mf,
                                          cut_options)
                  : reference_hamiltonian;
    SpinKMatrixMap current_band_hamiltonian =
        compute_band ? apply_hamiltonian_cut(band_reference_hamiltonian, band_reference_hamiltonian,
                                             dataset->mf_band, cut_options)
                     : band_reference_hamiltonian;

    const auto hartree_remap = controls_.qsgw_update_hartree
        ? build_qsgw_band_bvk_remap(dataset->atoms, dataset->pbc, opts.option_bvk_remap)
        : AtomPairBvKRemap<atom_t>{};
    const CheckpointContext checkpoint_context{
        reference,
        band_reference ? &*band_reference : nullptr,
        dataset->pbc.kfrac_list,
        dataset->kfrac_band_list,
        dataset->pbc.weight_k,
        dft_vxc,
        dft_vxc_band,
        controls_.qsgw_cut_mode,
        controls_.qsgw_unoccupied_keep,
        controls_.qsgw_cut_shift_ha,
        mixing_beta,
        opts.nfreq,
        update_head ? opts.option_dielect_func : 0,
        controls_.qsgw_headwing_velocity,
        ao_operators ? &*ao_operators : nullptr,
        &reference_velocity,
        &dataset->pbc.period,
        &dataset->pbc.latvec,
        controls_.qsgw_update_hartree ? &*input_.hartree_coulomb_q0 : nullptr,
        controls_.qsgw_update_hartree ? &*input_.hartree_cs_raw : nullptr,
        opts.option_bvk_remap, controls_.qsgw_update_hartree ? &hartree_remap : nullptr};
    int starting_iteration = 0;
    IterationSummary starting_summary;
    starting_summary.fermi_energy_ev = reference.get_efermi() * HA2EV;
    starting_summary.gap_ev = initial_occupations.gap * HA2EV;
    starting_summary.electron_count = initial_occupations.electron_count;
    starting_summary.beta = mixing_beta;
    if (controls_.qsgw_restart_file != "none")
    {
        collective_root_stage(
            dataset->comm_h, "QSGW checkpoint restore",
            [&]
            {
                starting_summary =
                    read_qsgw_checkpoint(controls_.qsgw_restart_file, checkpoint_context,
                                         dataset->mf, compute_band ? &dataset->mf_band : nullptr,
                                         current_hamiltonian, current_band_hamiltonian);
                starting_iteration = starting_summary.iteration;
                if (starting_iteration >= controls_.qsgw_max_iter)
                    throw std::runtime_error(
                        "qsgw_max_iter must exceed the checkpoint's completed iteration");
                lib_printf("QSGW restored completed iteration %d from %s\n", starting_iteration,
                           controls_.qsgw_restart_file.c_str());
            });
        dataset->comm_h.bcast(&starting_iteration, 1, 0);
        broadcast_checkpoint_meanfield(dataset->mf, dataset->comm_h);
        broadcast_spin_k_matrix_map(current_hamiltonian, 0, dataset->comm_h);
        if (compute_band)
        {
            broadcast_checkpoint_meanfield(dataset->mf_band, dataset->comm_h);
            broadcast_spin_k_matrix_map(current_band_hamiltonian, 0, dataset->comm_h);
            dataset->is_band_calc_done = true;
        }
    }

    std::ostringstream qp_output;
    SpinKMatrixMap qp_rotation, qp_band_rotation;
    if (controls_.qsgw_output_qp)
        collective_root_stage(dataset->comm_h, "QSGW QP frame recovery",
                              [&]
                              {
                                  qp_rotation = reference_basis_rotation(dataset->mf, reference);
                                  if (compute_band)
                                      qp_band_rotation = reference_basis_rotation(dataset->mf_band,
                                                                                  *band_reference);
                              });

    std::ostringstream trace;
    std::ostringstream eigenvalue_trace;
    std::ostringstream matrix_trace;
    SpinKMatrixMap delta_hartree, delta_hartree_band;
    std::map<int, Matz> delta_hartree_ao;
    double density_delta_frobenius = 0.0;
    std::vector<Vector3_Order<double>> hartree_kpoints = dataset->pbc.kfrac_list;
    if (compute_band)
        hartree_kpoints.insert(hartree_kpoints.end(), dataset->kfrac_band_list.begin(),
                               dataset->kfrac_band_list.end());
    const auto evaluate_hartree = [&]
    {
        if (!controls_.qsgw_update_hartree) return;
        auto density = dataset->build_hartree_density();
        double norm2 = 0.0;
        for (std::size_t k = 0; k != density.size(); ++k)
            for (std::size_t n = 0; n != density[k].size(); ++n)
            {
                density[k].ptr()[n] -= hartree_reference_density[k].ptr()[n];
                norm2 += std::norm(density[k].ptr()[n]);
            }
        density_delta_frobenius = std::sqrt(norm2);
        Cs_LRI unique_cs;
        unique_cs.use_libri = true;
        if (dataset->comm_h.is_root()) unique_cs = *input_.hartree_cs_raw;
        const auto& aux =
            enabled(opts.use_shrink_abfs) ? dataset->basis_aux_shrink : dataset->basis_aux;
        delta_hartree_ao =
            compute_hartree_ao_k(dataset->basis_wfc, aux, unique_cs, *input_.hartree_coulomb_q0,
                                 dataset->pbc, density, dataset->comm_h, &hartree_kpoints, hartree_remap);
        collective_root_stage(
            dataset->comm_h, "QSGW Delta Hartree projection",
            [&]
            {
                const auto project = [&](const MeanField& ref, int offset, SpinKMatrixMap& result)
                {
                    result.clear();
                    const int nao = ref.get_n_aos(), nspinor = ref.get_n_spinor();
                    for (int spin = 0; spin != ref.get_n_spins(); ++spin)
                        for (int k = 0; k != ref.get_n_kpoints(); ++k)
                        {
                            Matz lifted(nao * nspinor, nao * nspinor, MAJOR::ROW);
                            const auto& spatial = delta_hartree_ao.at(offset + k);
                            for (int i = 0; i != nao; ++i)
                                for (int j = 0; j != nao; ++j)
                                    for (int s = 0; s != nspinor; ++s)
                                        lifted(i * nspinor + s, j * nspinor + s) = spatial(i, j);
                            result[spin][k] = project_vxc_nao_to_fixed_basis(lifted, ref, spin, k);
                        }
                };
                project(reference, 0, delta_hartree);
                if (compute_band)
                    project(*band_reference, reference.get_n_kpoints(), delta_hartree_band);
            });
    };
    const auto report_hartree = [&](int state, const char* phase)
    {
        if (!controls_.qsgw_update_hartree) return;
        collective_root_stage(
            dataset->comm_h, "QSGW Delta Hartree output",
            [&]
            {
                std::ostringstream out;
                out << std::scientific << std::setprecision(17);
                const double electrons =
                    physical_electron_count(dataset->mf, dataset->pbc.weight_k);
                const auto write_channel =
                    [&](const char* channel, int offset, const SpinKMatrixMap& matrices)
                {
                    for (const auto& [spin, by_k] : matrices)
                        for (const auto& [k, matrix] : by_k)
                        {
                            const auto& ao = delta_hartree_ao.at(offset + k);
                            double am = 0, af = 0, km = 0, kf = 0, kd = 0;
                            for (std::size_t n = 0; n != ao.size(); ++n)
                            {
                                am = std::max(am, std::abs(ao.ptr()[n]));
                                af += std::norm(ao.ptr()[n]);
                            }
                            for (int i = 0; i != matrix.nr(); ++i)
                                for (int j = 0; j != matrix.nc(); ++j)
                                {
                                    km = std::max(km, std::abs(matrix(i, j)));
                                    kf += std::norm(matrix(i, j));
                                    if (i == j) kd = std::max(kd, std::abs(matrix(i, j)));
                                }
                            af = std::sqrt(af);
                            kf = std::sqrt(kf);
                            const auto row = [&, output_spin = spin + 1, output_k = k + 1](
                                                 const char* kind, int i, int j, cplxdb z)
                            {
                                out << state << ',' << phase << ',' << channel << ',' << output_spin
                                    << ',' << output_k << ',' << kind << ',' << i << ',' << j << ','
                                    << z.real() << ',' << z.imag() << ',' << z.real() * HA2EV << ','
                                    << z.imag() * HA2EV << ',' << am << ',' << af << ',' << km
                                    << ',' << kf << ',' << kd << ',' << am * HA2EV << ','
                                    << af * HA2EV << ',' << km * HA2EV << ',' << kf * HA2EV << ','
                                    << kd * HA2EV << ',' << electrons << ','
                                    << density_delta_frobenius << '\n';
                            };
                            row("summary", 0, 0, 0.0);
                            if (offset == 0)
                            {
                                for (int i = 0; i != ao.nr(); ++i)
                                    for (int j = 0; j != ao.nc(); ++j)
                                        row("ao", i + 1, j + 1, ao(i, j));
                                for (int i = 0; i != matrix.nr(); ++i)
                                    for (int j = 0; j != matrix.nc(); ++j)
                                        row("ks", i + 1, j + 1, matrix(i, j));
                            }
                            else
                                for (int i = 0; i != matrix.nr(); ++i)
                                    row("ks_diagonal", i + 1, i + 1, matrix(i, i));
                        }
                };
                write_channel("grid", 0, delta_hartree);
                if (compute_band)
                    write_channel("band", reference.get_n_kpoints(), delta_hartree_band);
                output_text(OutputKind::Hartree, out.str(), state, 0, true);
                if (controls_.qsgw_write_iteration_matrices)
                {
                    const std::string component = std::string("delta_vh_") + phase;
                    write_matrix_component_trace(matrix_trace, state, IterationChannel::Grid,
                                                 component, delta_hartree);
                    if (compute_band)
                        write_matrix_component_trace(matrix_trace, state, IterationChannel::Band,
                                                     component, delta_hartree_band);
                }
            });
    };
    collective_root_stage(
        dataset->comm_h, "QSGW trace setup",
        [&]
        {
            if (controls_.qsgw_update_hartree)
                output_text(
                    OutputKind::Hartree,
                    "state_index,phase,channel,spin,kpoint,kind,row,column,re_Ha,im_Ha,re_eV,im_eV,"
                    "ao_max_Ha,ao_frobenius_Ha,ks_max_Ha,ks_frobenius_Ha,ks_max_diagonal_Ha,"
                    "ao_max_eV,ao_frobenius_eV,ks_max_eV,ks_frobenius_eV,ks_max_diagonal_eV,"
                    "electrons,density_delta_frobenius\n");
            if (controls_.qsgw_output_qp)
            {
                qp_output << "state_index,gw_update,channel,spin,kpoint,kx,ky,kz,band,occupation,"
                             "e0_eV,vstatic_eV,sigma_x_eV,re_sigma_c_effective_eV,im_sigma_c_eV,"
                             "eqp_eV,solver_info\n"
                          << std::scientific << std::setprecision(17);
            }
            write_iteration_summary_header(trace);
            write_eigenvalue_trace_header(eigenvalue_trace);
            if (controls_.qsgw_write_iteration_matrices)
            {
                write_matrix_trace_header(matrix_trace);
                write_matrix_component_trace(matrix_trace, 0, IterationChannel::Grid, "h0",
                                             reference_hamiltonian);
                write_matrix_component_trace(matrix_trace, 0, IterationChannel::Grid, "vxc_dft",
                                             dft_vxc);
                write_wavefunction_trace(matrix_trace, 0, IterationChannel::Grid, reference);
                write_occupation_trace(matrix_trace, 0, IterationChannel::Grid, reference);
                if (compute_band)
                {
                    write_matrix_component_trace(matrix_trace, 0, IterationChannel::Band, "h0",
                                                 band_reference_hamiltonian);
                    write_matrix_component_trace(matrix_trace, 0, IterationChannel::Band, "vxc_dft",
                                                 dft_vxc_band);
                    write_wavefunction_trace(matrix_trace, 0, IterationChannel::Band,
                                             *band_reference);
                    write_occupation_trace(matrix_trace, 0, IterationChannel::Band,
                                           *band_reference);
                }
                if (starting_iteration > 0)
                {
                    write_matrix_component_trace(matrix_trace, starting_iteration,
                                                 IterationChannel::Grid, "mixed_h",
                                                 current_hamiltonian);
                    write_wavefunction_trace(matrix_trace, starting_iteration,
                                             IterationChannel::Grid, dataset->mf);
                    write_occupation_trace(matrix_trace, starting_iteration, IterationChannel::Grid,
                                           dataset->mf);
                    if (compute_band)
                    {
                        write_matrix_component_trace(matrix_trace, starting_iteration,
                                                     IterationChannel::Band, "mixed_h",
                                                     current_band_hamiltonian);
                        write_wavefunction_trace(matrix_trace, starting_iteration,
                                                 IterationChannel::Band, dataset->mf_band);
                        write_occupation_trace(matrix_trace, starting_iteration,
                                               IterationChannel::Band, dataset->mf_band);
                    }
                }
            }
            write_iteration_summary(trace, starting_summary);
            write_eigenvalue_trace(eigenvalue_trace, starting_iteration, IterationChannel::Grid,
                                   dataset->mf, dataset->pbc.kfrac_list);
            if (compute_band)
            {
                write_eigenvalue_trace(eigenvalue_trace, starting_iteration, IterationChannel::Band,
                                       dataset->mf_band, dataset->kfrac_band_list);
            }
        });
    const auto flush_traces = [&](bool append)
    {
        output_text(OutputKind::Iterations, trace.str(), 0, 0, append);
        output_text(OutputKind::Eigenvalues, eigenvalue_trace.str(), 0, 0, append);
        if (controls_.qsgw_write_iteration_matrices)
            output_text(OutputKind::Matrices, matrix_trace.str(), 0, 0, append);
        if (controls_.qsgw_output_qp)
            output_text(OutputKind::Quasiparticles, qp_output.str(), 0, 0, append);
        trace.str("");
        eigenvalue_trace.str("");
        matrix_trace.str("");
        qp_output.str("");
    };
    collective_root_stage(dataset->comm_h, "QSGW initial output", [&] { flush_traces(false); });

    bool converged = false;
    int completed_iterations = starting_iteration;
    const auto export_hamiltonian = [&](const int iteration, const SpinKMatrixMap& hamiltonian)
    {
        if (!controls_.qsgw_export_hamiltonian_for_pyatb) return;
        collective_root_stage(
            dataset->comm_h, "QSGW complete Hamiltonian HR export",
            [&]
            {
                const auto exported = reconstruct_qsgw_hamiltonian(
                    reference, hamiltonian, *ao_operators, dataset->pbc.kfrac_list,
                    dataset->pbc.period, dataset->pbc.latvec);
                for (int spin = 0; spin < reference.get_n_spins(); ++spin)
                {
                    output_hamiltonian(OutputKind::Hamiltonian, exported.hamiltonian.at(spin),
                                       iteration, spin);
                }
                lib_printf("QSGW HR iteration %d: full complex Ry, metric %.8e FT_Ha %.8e\n",
                           iteration, exported.reference_metric_residual,
                           exported.fourier_roundtrip_residual_ha);
            });
    };
    // Validate export inputs before the first GW build and retain the actual
    // entering state. Fresh iteration 0 is PBE; a restart keeps its saved index.
    export_hamiltonian(starting_iteration,
                       starting_iteration == 0 ? reference_hamiltonian : current_hamiltonian);
    if (extension_)
    {
        if (opts.read_sigc_mat_rf == LIBRPA_SWITCH_ON)
            throw LIBRPA_RUNTIME_ERROR("QSGW numerical companions require the live response");
        extension_->initialize(*dataset, opts, starting_iteration);
        if (starting_iteration > 0)
            extension_->read_checkpoint(*dataset, controls_.qsgw_restart_file, starting_iteration);
    }
    // The final response-only state also needs its velocity in the existing
    // numerical trace, so a frozen-state standalone calculation can reproduce U.
    const auto write_live_velocity_trace = [&](int update_index)
    {
        if (!controls_.qsgw_write_iteration_matrices ||
            controls_.qsgw_headwing_velocity == "reference")
            return;
        for (int a = 0; a < 3; ++a)
        {
            SpinKMatrixMap velocities;
            for (int spin = 0; spin < dataset->mf.get_n_spins(); ++spin)
                for (int ik = 0; ik < dataset->mf.get_n_kpoints(); ++ik)
                {
                    const auto& v = dataset->velocity_matrix[spin][ik][a];
                    Matz matrix(v.nr, v.nc, MAJOR::ROW);
                    for (int i = 0; i < v.nr; ++i)
                        for (int j = 0; j < v.nc; ++j) matrix(i, j) = v(i, j);
                    velocities[spin][ik] = std::move(matrix);
                }
            write_matrix_component_trace(matrix_trace, update_index, IterationChannel::Grid,
                                         "head_velocity_" + std::to_string(a), velocities);
        }
    };
    const auto prepare_optical_state =
        [&](int iteration, bool write_numerical_trace, bool rebuild_full_optical = true)
    {
        if (enabled(opts.use_symmetry_exx) || enabled(opts.use_symmetry_gw) ||
            enabled(opts.use_symmetry_rpa))
            collective_all_stage(dataset->comm_h, "QSGW live magnetic covariance",
                                 [&]
                                 {
                                     dataset->initialize_symmetry(true);
                                     validate_live_symmetry_state(
                                         dataset->mf, dataset->basis_wfc, dataset->pbc,
                                         dataset->symmetry_context, expanded_reference_.has_value());
                                 });
        if (update_head)
        {
            std::optional<LiveVelocityResult> live_velocity;
            collective_all_stage(
                dataset->comm_h, "QSGW live head/wing velocity",
                [&]
                {
                    if (controls_.qsgw_headwing_velocity == "current_hr")
                    {
                        auto velocity = compute_live_headwing_velocity(
                            reference, dataset->mf,
                            // The initial cut H is a residual reference; update 1
                            // still screens with the actual entering PBE eigenpairs.
                            starting_iteration == 0 && iteration == 1 ? reference_hamiltonian
                                                                      : current_hamiltonian,
                            *ao_operators, dataset->pbc.kfrac_list, dataset->pbc.period,
                            dataset->pbc.latvec);
                        dataset->velocity_matrix = std::move(velocity.velocity);
                        if (dataset->comm_h.is_root())
                            lib_printf(
                                "QSGW update %d current-HR velocity: metric %.8e live_metric %.8e "
                                "frame_Ha %.8e FT_Ha %.8e\n",
                                iteration, velocity.reference_metric_residual,
                                velocity.live_metric_residual,
                                velocity.hamiltonian_frame_residual_ha,
                                velocity.fourier_roundtrip_residual_ha);
                        live_velocity = std::move(velocity);
                    }
                    else
                    {
                        dataset->velocity_matrix = reference_velocity;
                        if (controls_.qsgw_headwing_velocity == "rotate_reference")
                            align_velocity_to_reference_wfc(reference, dataset->mf,
                                                            dataset->velocity_matrix);
                    }
                });
            dataset->p_headwing.reset();
            if (!rebuild_full_optical)
            {
                collective_root_stage(dataset->comm_h, "QSGW final cRPA velocity trace",
                                      [&] { write_live_velocity_trace(iteration); });
                return;
            }
            dataset->initialize_tfgrids(opts);
            // Legacy option-4 head reused the first k-point occupation row.
            // Preserve it only for the unchanged reference-entries map.
            if (controls_.qsgw_headwing_velocity == "reference")
            {
                ScopedLegacyHeadOccupations legacy_head_occupations(dataset->mf);
                dataset->initialize_headwing(opts, false);
            }
            else
            {
                dataset->initialize_headwing(opts, update_wing);
            }
            // The analytic object is replaced every update. Reapply the fixed producer
            // normalization while retaining its live eigenpairs, velocity and frequencies.
            collective_all_stage(dataset->comm_h, "QSGW head/wing Coulomb normalization",
                                 [&] { configure_coulomb_head(*dataset, opts, input_); });
            if (update_wing && (!dataset->p_headwing || !dataset->p_headwing->has_wing()))
                throw LIBRPA_RUNTIME_ERROR("QSGW analytic wing was not rebuilt for the live state");
            collective_root_stage(
                dataset->comm_h, "QSGW head/wing numerical trace",
                [&]
                {
                    if (!write_numerical_trace || !controls_.qsgw_write_iteration_matrices ||
                        controls_.qsgw_headwing_velocity == "reference")
                        return;
                    if (live_velocity)
                        for (int spin = 0; spin < dataset->mf.get_n_spins(); ++spin)
                            output_hamiltonian(OutputKind::HeadHamiltonian,
                                               live_velocity->hamiltonian.at(spin), iteration,
                                               spin);
                    write_live_velocity_trace(iteration);
                    const auto& frequencies = dataset->p_headwing->get_frequency_nodes();
                    for (const auto& [name, tensors] :
                         std::vector<std::pair<std::string, const std::vector<Matz>*>>{
                             {"head_tensor", &dataset->p_headwing->get_head_tensors()},
                             {"wing_auxiliary", &dataset->p_headwing->get_wing_auxiliary()}})
                    {
                        SpinKFrequencyMatrixMap matrices;
                        for (std::size_t i = 0; i < tensors->size(); ++i)
                            matrices[0][0][frequencies.at(i)] = tensors->at(i);
                        write_frequency_matrix_component_trace(
                            matrix_trace, iteration, IterationChannel::Grid, name, matrices);
                    }
                });
        }
    };
    evaluate_hartree();
    for (int iteration = starting_iteration + 1; iteration <= controls_.qsgw_max_iter; ++iteration)
    {
        completed_iterations = iteration;
        const EigenvalueSnapshot previous = eigenvalue_snapshot(dataset->mf);
        report_hartree(iteration - 1, "entering");
        dataset->invalidate_compute_objects();
        prepare_optical_state(iteration, true);
        const int entering_state = iteration - 1;
        ResponseObserver response;
        if (extension_ && extension_->wants_response(entering_state))
            response = [&](const Chi0QMap& p0)
            { extension_->evaluate_response(*dataset, opts, p0, entering_state, iteration); };
        dataset->build_sigma(opts, response);
        if (!dataset->p_exx || !dataset->p_g0w0)
            throw LIBRPA_RUNTIME_ERROR("QSGW failed to build live EXX/Sigma real-space objects");
        if (update_head && !dataset->p_headwing)
            throw LIBRPA_RUNTIME_ERROR(
                "QSGW head-only object was not rebuilt on the live frequency grid");

        {
            // The upstream projection path uses the output flag as its
            // full-matrix retention gate. Restore it before leaving this
            // scope so QSGW does not change the user-facing output setting.
            ScopedSigmaMatrixRetention retain_sigma_matrices(
                dataset->p_g0w0->output_sigc_ks_mat_kf);
            ScopedReferenceEigenvectors fixed_basis_projection(dataset->mf, reference);
            dataset->p_exx->build_KS_kgrid_blacs(
                dataset->blacs_h, opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON);
            dataset->p_g0w0->build_sigc_matrix_KS_kgrid_blacs(
                dataset->blacs_h, opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON);
        }
        const std::vector<double> frequencies = dataset->tfg.get_freq_nodes();
        const SigmaMatrixMap sigma =
            collect_sigma_root(*dataset->p_g0w0, reference, frequencies, dataset->comm_h);

        SpinKMatrixMap exchange;
        collective_root_stage(dataset->comm_h, "QSGW EXX collection",
                              [&] { exchange = copy_exx_root(*dataset->p_exx, reference); });

        SigmaMatrixMap sigma_band;
        SpinKMatrixMap exchange_band;
        if (compute_band)
        {
            dataset->p_exx->reset_kspace();
            dataset->p_g0w0->reset_kspace();
            const auto bvk_remap = build_qsgw_band_bvk_remap(dataset->atoms, dataset->pbc,
                                                                    opts.option_bvk_remap);
            {
                ScopedSigmaMatrixRetention retain_sigma_matrices(
                    dataset->p_g0w0->output_sigc_ks_mat_kf);
                dataset->p_exx->build_KS_band_blacs(
                    band_reference->get_eigenvectors(), dataset->kfrac_band_list, bvk_remap,
                    dataset->blacs_h, opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON);
                dataset->p_g0w0->build_sigc_matrix_KS_band_blacs(
                    band_reference->get_eigenvectors(), dataset->kfrac_band_list, bvk_remap,
                    dataset->blacs_h, opts.use_gpu_replace_scalapack == LIBRPA_SWITCH_ON);
            }
            sigma_band =
                collect_sigma_root(*dataset->p_g0w0, *band_reference, frequencies, dataset->comm_h);
            collective_root_stage(
                dataset->comm_h, "QSGW band EXX collection",
                [&] { exchange_band = copy_exx_root(*dataset->p_exx, *band_reference); });
        }

        if (controls_.qsgw_output_qp)
            collective_root_stage(
                dataset->comm_h, "QSGW entering-state QP equation",
                [&]
                {
                    write_qsgw_qp_state(
                        qp_output, entering_state, iteration, "grid", dataset->mf,
                        dataset->pbc.kfrac_list, qp_rotation,
                        entering_state == 0 ? reference_hamiltonian : current_hamiltonian,
                        reference_hamiltonian, dft_vxc,
                        controls_.qsgw_update_hartree ? &delta_hartree : nullptr, exchange, sigma,
                        dataset->tfg, dataset->pbc.weight_k, opts, controls_, emit);
                    if (compute_band)
                        write_qsgw_qp_state(
                            qp_output, entering_state, iteration, "band", dataset->mf_band,
                            dataset->kfrac_band_list, qp_band_rotation,
                            entering_state == 0 ? band_reference_hamiltonian
                                                : current_band_hamiltonian,
                            band_reference_hamiltonian, dft_vxc_band,
                            controls_.qsgw_update_hartree ? &delta_hartree_band : nullptr,
                            exchange_band, sigma_band, dataset->tfg, {}, opts, controls_, emit);
                    output_text(OutputKind::Quasiparticles, qp_output.str(), 0, 0, true);
                    qp_output.str("");
                });

        SpinKMatrixMap mixed_hamiltonian;
        SpinKMatrixMap mixed_band_hamiltonian;
        SpinKMatrixMap band_exchange_output;
        double residual_l2 = 0.0;
        double residual_max = 0.0;
        std::string matrix_rows;
        collective_root_stage(
            dataset->comm_h, "QSGW Hamiltonian update",
            [&]
            {
                const SpinKMatrixMap correlation =
                    build_correlation_map(dataset->mf, sigma, frequencies, opts);
                SpinKMatrixMap correlation_band;
                if (compute_band)
                {
                    correlation_band =
                        build_correlation_map(dataset->mf_band, sigma_band, frequencies, opts);
                }
                const SpinKMatrixMap raw_uncut = assemble_effective_hamiltonian(
                    reference_hamiltonian, dft_vxc, exchange, correlation,
                    controls_.qsgw_update_hartree ? &delta_hartree : nullptr);
                const SpinKMatrixMap raw =
                    apply_cut ? apply_hamiltonian_cut(raw_uncut, reference_hamiltonian, dataset->mf,
                                                      cut_options)
                              : raw_uncut;
                SpinKMatrixMap raw_band;
                if (compute_band)
                {
                    band_exchange_output = exchange_band;
                    const SpinKMatrixMap raw_band_uncut = assemble_effective_hamiltonian(
                        band_reference_hamiltonian, dft_vxc_band, exchange_band, correlation_band,
                        controls_.qsgw_update_hartree ? &delta_hartree_band : nullptr);
                    raw_band = apply_hamiltonian_cut(raw_band_uncut, band_reference_hamiltonian,
                                                     dataset->mf_band, cut_options);
                }
                const auto residual = measure_spin_k_hamiltonian_residual(raw, current_hamiltonian);
                residual_l2 = residual.l2;
                residual_max = residual.maximum;
                mixed_hamiltonian = mix_spin_k_hamiltonian(current_hamiltonian, raw, mixing_beta);
                if (compute_band)
                    mixed_band_hamiltonian =
                        mix_spin_k_hamiltonian(current_band_hamiltonian, raw_band, mixing_beta);
                if (apply_cut)
                {
                    mixed_hamiltonian = apply_hamiltonian_cut(
                        mixed_hamiltonian, reference_hamiltonian, dataset->mf, cut_options);
                    if (compute_band)
                        mixed_band_hamiltonian = apply_hamiltonian_cut(
                            mixed_band_hamiltonian, band_reference_hamiltonian, dataset->mf_band,
                            cut_options);
                }
                if (controls_.qsgw_write_iteration_matrices)
                {
                    std::ostringstream rows;
                    write_frequency_matrix_component_trace(rows, iteration, IterationChannel::Grid,
                                                           "sigma_c_iw", sigma);
                    write_matrix_component_trace(rows, iteration, IterationChannel::Grid, "exx",
                                                 exchange);
                    write_matrix_component_trace(rows, iteration, IterationChannel::Grid, "vc",
                                                 correlation);
                    write_matrix_component_trace(rows, iteration, IterationChannel::Grid, "raw_h",
                                                 raw);
                    write_matrix_component_trace(rows, iteration, IterationChannel::Grid, "mixed_h",
                                                 mixed_hamiltonian);
                    if (compute_band)
                    {
                        write_matrix_component_trace(rows, iteration, IterationChannel::Band, "exx",
                                                     exchange_band);
                        write_matrix_component_trace(rows, iteration, IterationChannel::Band, "vc",
                                                     correlation_band);
                        write_matrix_component_trace(rows, iteration, IterationChannel::Band,
                                                     "raw_h", raw_band);
                        write_matrix_component_trace(rows, iteration, IterationChannel::Band,
                                                     "mixed_h", mixed_band_hamiltonian);
                    }
                    matrix_rows = rows.str();
                }
            });

        broadcast_spin_k_matrix_map(mixed_hamiltonian, 0, dataset->comm_h);
        if (compute_band)
        {
            broadcast_spin_k_matrix_map(mixed_band_hamiltonian, 0, dataset->comm_h);
        }
        // The next velocity is evaluated on every rank. Its Hamiltonian must
        // advance with the live eigenpairs, not remain a root-only residual map.
        current_hamiltonian = mixed_hamiltonian;
        if (compute_band) current_band_hamiltonian = mixed_band_hamiltonian;
        // Export the exact mixed/post-cut operator about to be diagonalized.
        // AO reconstruction uses the immutable initial coefficients, so the
        // eigenpairs produced below are not needed for this output.
        export_hamiltonian(iteration, current_hamiltonian);
        const FixedBasisDiagonalizationResult diagonalization =
            diagonalize_in_reference_basis(dataset->mf, reference, mixed_hamiltonian);
        std::optional<FixedBasisDiagonalizationResult> band_diagonalization;
        if (compute_band)
        {
            band_diagonalization = diagonalize_in_reference_basis(dataset->mf_band, *band_reference,
                                                                  mixed_band_hamiltonian);
            dataset->is_band_calc_done = true;
        }
        const OccupationResult occupations =
            update_qsgw_occupations(dataset->mf, reference, dataset->pbc.weight_k, electron_count);
        if (controls_.qsgw_output_qp && dataset->comm_h.is_root())
        {
            qp_rotation = diagonalization.unitary;
            if (compute_band) qp_band_rotation = band_diagonalization->unitary;
        }
        if (compute_band) dataset->mf_band.get_efermi() = occupations.chemical_potential;
        // Evaluate the density-induced operator even on the last requested update.
        // This is a Hartree contraction only; it does not perform another GW build.
        evaluate_hartree();
        report_hartree(iteration, "updated");
        // Tracking is cheap and independent of the requested U-output interval.
        // The saved parent belongs to this updated state, also on restart.
        if (extension_) extension_->state_updated(*dataset, iteration);
        const double maximum_change_ev = max_eigenvalue_change(dataset->mf, previous) * HA2EV;

        int converged_flag = 0;
        collective_root_stage(
            dataset->comm_h, "QSGW reporting",
            [&]
            {
                converged_flag =
                    qsgw_iteration_converged(iteration, controls_.qsgw_min_iter, maximum_change_ev,
                                             controls_.qsgw_convergence_tolerance_ev)
                        ? 1
                        : 0;
                IterationSummary summary;
                summary.iteration = iteration;
                summary.maximum_eigenvalue_change_ev = maximum_change_ev;
                summary.residual_l2_ha = residual_l2;
                summary.residual_max_ha = residual_max;
                summary.fermi_energy_ev = occupations.chemical_potential * HA2EV;
                summary.gap_ev = occupations.gap * HA2EV;
                summary.electron_count = occupations.electron_count;
                summary.converged = converged_flag != 0;
                summary.beta = mixing_beta;
                if (controls_.qsgw_checkpoint_file != "none")
                {
                    const auto checkpoint_file =
                        resolve_input_path(opts.output_dir, controls_.qsgw_checkpoint_file);
                    write_qsgw_checkpoint(checkpoint_file, checkpoint_context, summary, dataset->mf,
                                          compute_band ? &dataset->mf_band : nullptr,
                                          current_hamiltonian, current_band_hamiltonian);
                    if (extension_)
                        extension_->write_checkpoint(*dataset, checkpoint_file, iteration);
                }
                write_iteration_summary(trace, summary);
                write_eigenvalue_trace(eigenvalue_trace, iteration, IterationChannel::Grid,
                                       dataset->mf, dataset->pbc.kfrac_list);
                if (compute_band)
                {
                    write_eigenvalue_trace(eigenvalue_trace, iteration, IterationChannel::Band,
                                           dataset->mf_band, dataset->kfrac_band_list);
                    for (int spin = 0; spin < dataset->mf_band.get_n_spins(); ++spin)
                    {
                        std::ostringstream ks_output, exx_output, qsgw_output;
                        write_qsgw_band_spin_tables(
                            ks_output, exx_output, qsgw_output, dataset->mf_band, *band_reference,
                            dataset->kfrac_band_list, band_reference_hamiltonian, dft_vxc_band,
                            band_exchange_output, spin, occupations.chemical_potential);
                        output_text(OutputKind::KsBand, ks_output.str(), iteration, spin);
                        output_text(OutputKind::ExxBand, exx_output.str(), iteration, spin);
                        output_text(OutputKind::QsgwBand, qsgw_output.str(), iteration, spin);
                    }
                }
                if (controls_.qsgw_write_iteration_matrices)
                {
                    matrix_trace << matrix_rows;
                    write_matrix_component_trace(matrix_trace, iteration, IterationChannel::Grid,
                                                 "rotation_u", diagonalization.unitary);
                    write_wavefunction_trace(matrix_trace, iteration, IterationChannel::Grid,
                                             dataset->mf);
                    write_occupation_trace(matrix_trace, iteration, IterationChannel::Grid,
                                           dataset->mf);
                    if (compute_band)
                    {
                        write_matrix_component_trace(matrix_trace, iteration,
                                                     IterationChannel::Band, "rotation_u",
                                                     band_diagonalization->unitary);
                        write_wavefunction_trace(matrix_trace, iteration, IterationChannel::Band,
                                                 dataset->mf_band);
                        write_occupation_trace(matrix_trace, iteration, IterationChannel::Band,
                                               dataset->mf_band);
                    }
                }
                flush_traces(true);
                lib_printf("QSGW iteration %d: max_delta=% .8e eV gap=% .8f eV mixer=%s%s\n",
                           iteration, maximum_change_ev, occupations.gap * HA2EV,
                           effective_mixer.c_str(), converged_flag ? ", converged" : "");
            });
        dataset->comm_h.bcast(&converged_flag, 1, 0);
        converged = converged_flag != 0;
        if (converged) break;
    }

    if (extension_)
    {
        prepare_optical_state(completed_iterations + 1, false, false);
        extension_->evaluate_final(*dataset, opts, completed_iterations);
        collective_root_stage(dataset->comm_h, "QSGW final numerical trace",
                              [&] { flush_traces(true); });
    }

    if (dataset->comm_h.is_root())
    {
        if (!converged)
            lib_printf(LIBRPA_VERBOSE_WARN, "QSGW reached qsgw_max_iter=%d without convergence\n",
                       controls_.qsgw_max_iter);
        lib_printf("QSGW completed iterations: %d\n", completed_iterations);
    }
    profiler.stop("qsgw");
}
}  // namespace librpa_int::qsgw
