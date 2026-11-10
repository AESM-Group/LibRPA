#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "../core/crpa_continuation.h"
#include "../core/crpa_model.h"
#include "../core/crpa_qsgw_extension.h"
#include "../core/crpa_task.h"
#include "../mpi/collective_stage.h"
#include "../utils/error.h"
#include "crpa_helper.h"
#include "dataset_helper.h"
#include "gw_response_helper.h"
#include "instance_manager.h"
#include "librpa_compute.h"
#include "orbital_helper.h"
#include "qsgw_helper.h"

struct LibrpaCrpaResult
{
    std::optional<librpa_int::CrpaTaskResult> owned;
    const librpa_int::CrpaTaskResult* borrowed = nullptr;

    const librpa_int::CrpaTaskResult& data() const { return owned ? *owned : *borrowed; }
};

struct LibrpaCrpaContinuedResult
{
    std::vector<librpa_int::CrpaContinuedTensor> tensors;
};

namespace
{
void collective_error(const librpa_int::MpiCommHandler& comm, const std::string& error)
{
    const int failed = !error.empty();
    int any_failed = 0;
    comm.allreduce(&failed, &any_failed, 1, MPI_MAX);
    if (any_failed)
        throw LIBRPA_RUNTIME_ERROR("cRPA input: " +
                                   (error.empty() ? "another MPI rank failed validation" : error));
}

}  // namespace

std::shared_ptr<librpa_int::CrpaTaskContext> librpa_int::api::make_crpa_task_context(
    const dataset_ptr_t& ds)
{
    return std::make_shared<CrpaTaskContext>(CrpaTaskContext{
        ds->mf, ds->pbc, ds->atoms, ds->basis_wfc, ds->basis_aux, ds->basis_aux_shrink,
        ds->basis_convention, ds->comm_h, ds->blacs_h, ds->desc_abf, ds->desc_abf_shrink,
        ds->atpairs_local, ds->cs_data, ds->sinvS, ds->vq, ds->p_chi0,
        [ds](const LibrpaOptions& opts)
        {
            initialize_ds_global_ddla(*ds, opts);
            initialize_ds_tfgrids(*ds, opts);
            initialize_ds_atpairs_local(*ds, LIBRPA_ROUTING_LIBRI);
            ds->redistribute_coulomb_blacs2ap();
            initialize_ds_chi0_qstar(*ds, opts);
        },
        [ds](const LibrpaOptions& opts, bool wing, const HeadwingBandSelection* excluded)
        { return make_analytic_headwing(*ds, opts, wing, excluded); }});
}

void librpa_init_crpa_input(LibrpaCrpaInput* input)
{
    if (!input) throw LIBRPA_INVALID_ARGUMENT("null cRPA input");
    *input = {};
    input->parent_orbitals = "d";
    input->output_orbitals = "d";
    input->gram_abs = input->gram_rel = input->s_rel = 1.0e-12;
    input->gram_cond_max = 1.0e12;
    input->s_abs = input->residual_tol = input->t_roundtrip_tol = 1.0e-10;
    input->s_cond_max = 1.0e10;
    input->tracking_min_capture = 0.99;
}

void librpa_get_crpa_kgrid(const LibrpaHandler* h, int n_kpoints, double* fractional_k)
{
    const auto ds = librpa_int::api::get_dataset_instance(h);
    if (!fractional_k || n_kpoints <= 0 ||
        ds->pbc.kfrac_list.size() != static_cast<std::size_t>(n_kpoints))
        throw LIBRPA_INVALID_ARGUMENT("cRPA k-grid query has inconsistent dimensions");
    for (int k = 0; k < n_kpoints; ++k)
    {
        fractional_k[3 * k] = ds->pbc.kfrac_list[k].x;
        fractional_k[3 * k + 1] = ds->pbc.kfrac_list[k].y;
        fractional_k[3 * k + 2] = ds->pbc.kfrac_list[k].z;
    }
}

librpa_int::CrpaTaskInput librpa_int::api::unpack_crpa_input(Dataset& dataset,
                                                             const LibrpaCrpaInput* input)
{
    using namespace librpa_int;
    auto* ds = &dataset;
    CrpaWindowOptions window;
    std::vector<std::string> species;
    std::vector<ComplexMatrix> overlap;
    std::string error;
    try
    {
        if (!input) throw LIBRPA_INVALID_ARGUMENT("missing cRPA input or options");
        if (input->n_species <= 0 || !input->species_labels || !input->correlated_species ||
            !input->parent_orbitals || !input->output_orbitals || !input->overlap_k_ri ||
            input->n_kpoints != ds->mf.get_n_kpoints() || input->n_aos != ds->mf.get_n_aos())
            throw LIBRPA_INVALID_ARGUMENT("incomplete cRPA species/window/overlap input");
        for (int type = 0; type < input->n_species; ++type)
        {
            if (!input->species_labels[type]) throw LIBRPA_INVALID_ARGUMENT("null species label");
            species.emplace_back(input->species_labels[type]);
        }
        window.correlated_species = input->correlated_species;
        window.ligand_species = input->ligand_species ? input->ligand_species : "";
        window.parent_orbitals = input->parent_orbitals;
        window.output_orbitals = input->output_orbitals;
        if (input->use_orbital_weights != LIBRPA_SWITCH_OFF &&
            input->use_orbital_weights != LIBRPA_SWITCH_ON)
            throw LIBRPA_INVALID_ARGUMENT("invalid cRPA orbital-weight switch");
        window.use_orbital_weights = input->use_orbital_weights == LIBRPA_SWITCH_ON;
        const auto copy_array = [](int count, const auto* source, auto& destination)
        {
            if (count < 0 || (count > 0 && !source))
                throw LIBRPA_INVALID_ARGUMENT("invalid cRPA selection array");
            if (count > 0) destination.assign(source, source + count);
        };
        copy_array(input->n_response_edges, input->response_windows_ha, window.response_windows_ha);
        copy_array(input->n_orbital_edges, input->orbital_windows_ha, window.orbital_windows_ha);
        copy_array(input->n_response_bands, input->response_bands, window.response_bands);
        copy_array(input->n_orbital_bands, input->orbital_bands, window.orbital_bands);
        window.gram_abs = input->gram_abs;
        window.gram_rel = input->gram_rel;
        window.gram_cond_max = input->gram_cond_max;
        window.s_abs = input->s_abs;
        window.s_rel = input->s_rel;
        window.s_cond_max = input->s_cond_max;
        window.residual_tol = input->residual_tol;
        window.t_roundtrip_tol = input->t_roundtrip_tol;
        window.tracking_min_capture = input->tracking_min_capture;
        if (input->n_previous_parent < 0 ||
            (input->n_previous_parent > 0 && !input->previous_parent_phi_ri))
            throw LIBRPA_INVALID_ARGUMENT("invalid cRPA previous-parent frame");
        if (input->n_previous_parent > 0)
        {
            const int rows = input->n_aos * ds->mf.get_n_spinor();
            const int columns = input->n_previous_parent;
            std::size_t offset = 0;
            window.previous_parent_spin_k.resize(ds->mf.get_n_spins());
            for (auto& spin : window.previous_parent_spin_k)
                for (int k = 0; k < input->n_kpoints; ++k)
                {
                    ComplexMatrix phi(rows, columns);
                    for (int i = 0; i < phi.size; ++i, offset += 2)
                        phi.c[i] = {input->previous_parent_phi_ri[offset],
                                    input->previous_parent_phi_ri[offset + 1]};
                    spin.push_back(std::move(phi));
                }
        }
        const std::size_t block_size = static_cast<std::size_t>(input->n_aos) * input->n_aos;
        for (int k = 0; k < input->n_kpoints; ++k)
        {
            overlap.emplace_back(input->n_aos, input->n_aos);
            for (std::size_t i = 0; i < block_size; ++i)
            {
                const auto offset = 2 * (static_cast<std::size_t>(k) * block_size + i);
                overlap.back().c[i] = {input->overlap_k_ri[offset],
                                       input->overlap_k_ri[offset + 1]};
            }
        }
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
    }
    collective_error(ds->comm_h, error);

    return {std::move(species), std::move(overlap), std::move(window),
            input->retain_raw_model == LIBRPA_SWITCH_ON};
}

LibrpaCrpaResult* librpa_int::api::own_crpa_result(CrpaTaskResult result)
{
    return new LibrpaCrpaResult{std::move(result), nullptr};
}

void librpa_init_qsgw_crpa_controls(LibrpaQsgwCrpaControls* controls)
{
    if (!controls) throw LIBRPA_INVALID_ARGUMENT("missing QSGW cRPA controls");
    *controls = {1, 0, 1, nullptr};
}

void librpa_set_qsgw_crpa(LibrpaQsgwTask* task, const LibrpaCrpaInput* input,
                          const LibrpaQsgwCrpaControls* controls, LibrpaCrpaOutputSink sink,
                          void* user_data)
{
    using namespace librpa_int;
    const auto dataset = api::qsgw_task_dataset(task);
    const auto numerical_input = api::unpack_crpa_input(*dataset, input);
    std::shared_ptr<CrpaQsgwExtension> extension;
    collective_all_stage(
        dataset->comm_h, "QSGW cRPA attachment",
        [&]
        {
            if (!task || !controls || !sink)
                throw LIBRPA_INVALID_ARGUMENT("missing QSGW cRPA task, controls or output sink");
            CrpaQsgwControls numerical_controls;
            numerical_controls.interval = controls->interval;
            numerical_controls.track_orbitals = controls->track_orbitals != 0;
            numerical_controls.output_analysis = controls->output_analysis != 0;
            numerical_controls.tracking_definition =
                controls->tracking_definition ? controls->tracking_definition : "";
            extension = std::make_shared<CrpaQsgwExtension>(
                api::make_crpa_task_context(dataset), numerical_input,
                std::move(numerical_controls),
                [dataset, sink, user_data](const CrpaTaskResult& result,
                                           const OrbitalWeightResult* analysis, int state,
                                           int gw_update, bool final)
                {
                    LibrpaCrpaResult result_view{std::nullopt, &result};
                    std::unique_ptr<LibrpaOrbitalResult, decltype(&librpa_delete_orbital_result)>
                        analysis_view(analysis ? api::borrow_orbital_result(*analysis) : nullptr,
                                      librpa_delete_orbital_result);
                    collective_all_stage(
                        dataset->comm_h, "QSGW cRPA output",
                        [&]
                        {
                            if (sink(&result_view, analysis_view.get(), state, gw_update, final,
                                     user_data) != 0)
                                throw LIBRPA_RUNTIME_ERROR("QSGW cRPA output sink failed");
                        });
                });
            api::qsgw_task_instance(task).set_extension(extension);
        });
}

LibrpaCrpaResult* librpa_prepare_crpa_model(const LibrpaHandler* h, const LibrpaOptions* options,
                                            const LibrpaCrpaInput* input)
{
    using namespace librpa_int;
    if (!options) throw LIBRPA_INVALID_ARGUMENT("missing cRPA model options");
    auto ds = api::get_dataset_instance(h);
    const auto numerical_input = api::unpack_crpa_input(*ds, input);
    return api::own_crpa_result(
        prepare_crpa_task(*api::make_crpa_task_context(ds), *options, numerical_input));
}

LibrpaCrpaResult* librpa_compute_crpa_u_window(const LibrpaHandler* h, const LibrpaOptions* options,
                                               const LibrpaCrpaInput* input)
{
    using namespace librpa_int;
    if (!options) throw LIBRPA_INVALID_ARGUMENT("missing cRPA options");
    auto ds = api::get_dataset_instance(h);
    const auto numerical_input = api::unpack_crpa_input(*ds, input);
    return api::own_crpa_result(
        compute_crpa_u_task(*api::make_crpa_task_context(ds), *options, numerical_input));
}

LibrpaCrpaResult* librpa_compute_crpa_window(const LibrpaHandler* h, const LibrpaOptions* options,
                                             const LibrpaCrpaInput* input)
{
    using namespace librpa_int;
    if (!options) throw LIBRPA_INVALID_ARGUMENT("missing cRPA options");
    auto ds = api::get_dataset_instance(h);
    const auto numerical_input = api::unpack_crpa_input(*ds, input);
    return api::own_crpa_result(
        compute_crpa_task(*api::make_crpa_task_context(ds), *options, numerical_input));
}

LibrpaCrpaResult* librpa_build_g0w0_sigma_crpa(LibrpaHandler* h, const LibrpaOptions* options,
                                               const LibrpaCrpaInput* input)
{
    if (!options || !input) throw LIBRPA_INVALID_ARGUMENT("missing shared GW/cRPA input");
    const auto dataset = librpa_int::api::get_dataset_instance(h);
    const auto context = librpa_int::api::make_crpa_task_context(dataset);
    auto model = librpa_int::prepare_crpa_task_model(
        *context, *options, librpa_int::api::unpack_crpa_input(*dataset, input),
        librpa_int::CrpaTaskMode::SharedGw);
    std::unique_ptr<LibrpaCrpaResult, decltype(&librpa_delete_crpa_result)> result(
        nullptr, librpa_delete_crpa_result);
    librpa_int::api::build_gw_sigma_for_task(
        *dataset, *options,
        [&](const librpa_int::Chi0QMap& response)
        {
            result.reset(librpa_int::api::own_crpa_result(
                librpa_int::evaluate_crpa_task_from_response(*context, *options, model, response)));
        });
    return result.release();
}

void librpa_write_crpa_tracking_checkpoint(const LibrpaCrpaResult* result, const char* filename,
                                           int iteration, const char* definition, int n_energies,
                                           const double* energies_ha)
{
    if (!result || !filename || !definition || n_energies < 0 || (n_energies > 0 && !energies_ha))
        throw LIBRPA_INVALID_ARGUMENT("missing cRPA tracking checkpoint input");
    std::vector<double> energies;
    if (n_energies) energies.assign(energies_ha, energies_ha + n_energies);
    librpa_int::write_crpa_tracking_checkpoint(
        filename, iteration, definition, energies,
        librpa_int::snapshot_crpa_tracking(result->data().model));
}

void librpa_delete_crpa_result(LibrpaCrpaResult* result) { delete result; }

void librpa_crpa_model_info(const LibrpaCrpaResult* result, LibrpaCrpaModelInfo* info)
{
    if (!result || !info) throw LIBRPA_INVALID_ARGUMENT("missing cRPA model info");
    const auto& model = result->data().model;
    const auto& state = model.spin_k.at(0).at(0);
    *info = {model.n_spins,
             model.n_spinor,
             static_cast<int>(model.spin_k.at(0).size()),
             model.n_bands,
             model.n_aos,
             static_cast<int>(result->data().atom_types.size()),
             state.parent_projector.nr,
             state.output_projector.nr,
             static_cast<int>(model.parent_sites.size()),
             static_cast<int>(model.output_sites.size()),
             model.fermi,
             result->data().lattice.data(),
             result->data().positions.data(),
             result->data().atom_types.data(),
             result->data().ao_counts.data(),
             result->data().k_fractional.data(),
             result->data().k_weights.data(),
             model.output_columns.data(),
             static_cast<int>(result->data().shell_l.size()),
             result->data().shell_offsets.data(),
             result->data().shell_l.data(),
             result->data().basis_convention.data()};
}

void librpa_crpa_model_k(const LibrpaCrpaResult* result, int spin, int kpoint,
                         LibrpaCrpaModelK* view)
{
    if (!result || !view || spin < 0 || kpoint < 0)
        throw LIBRPA_INVALID_ARGUMENT("missing cRPA model k view");
    const auto& model = result->data().model;
    const auto& state = model.spin_k.at(spin).at(kpoint);
    const auto packed = [](const librpa_int::ComplexMatrix& matrix)
    { return reinterpret_cast<const double*>(matrix.c); };
    *view = {state.energies.data(),
             state.occupations.data(),
             packed(state.coefficients),
             model.overlap.empty() ? nullptr : packed(model.overlap.at(kpoint)),
             packed(state.parent_phi),
             packed(state.output_phi),
             packed(state.parent_projector),
             packed(state.output_projector),
             packed(state.parent_h),
             packed(state.output_h),
             packed(state.parent_density),
             packed(state.output_density),
             state.parent_capture_min};
}

void librpa_crpa_model_local(const LibrpaCrpaResult* result, int parent, int spin, int site,
                             double* h_ha_ri, double* density_ri)
{
    using namespace librpa_int;
    if (!result || !h_ha_ri || !density_ri || spin < 0 || site < 0)
        throw LIBRPA_INVALID_ARGUMENT("missing local cRPA model buffers");
    const auto local = average_crpa_local_model(result->data().model, parent != 0, spin, site,
                                                result->data().k_weights);
    std::copy_n(reinterpret_cast<const double*>(local.first.c), 2 * local.first.size, h_ha_ri);
    std::copy_n(reinterpret_cast<const double*>(local.second.c), 2 * local.second.size, density_ri);
}

void librpa_crpa_model_hopping(const LibrpaCrpaResult* result, int parent, int spin,
                               const int* lattice_R, double* h_ha_ri)
{
    using namespace librpa_int;
    if (!result || !lattice_R || !h_ha_ri || spin < 0)
        throw LIBRPA_INVALID_ARGUMENT("missing cRPA hopping buffers");
    std::vector<Vector3_Order<double>> kpoints;
    for (std::size_t k = 0; k < result->data().k_fractional.size() / 3; ++k)
        kpoints.push_back({result->data().k_fractional[3 * k],
                           result->data().k_fractional[3 * k + 1],
                           result->data().k_fractional[3 * k + 2]});
    const auto h = fourier_crpa_model_hopping(result->data().model, parent != 0, spin, kpoints,
                                              result->data().k_weights,
                                              {lattice_R[0], lattice_R[1], lattice_R[2]});
    std::copy_n(reinterpret_cast<const double*>(h.c), 2 * h.size, h_ha_ri);
}

namespace
{
librpa_int::ComplexMatrix unpack_complex(int rows, int columns, const double* values)
{
    if (rows <= 0 || columns <= 0 || !values)
        throw LIBRPA_INVALID_ARGUMENT("missing complex numerical array");
    librpa_int::ComplexMatrix matrix(rows, columns);
    for (int i = 0; i < matrix.size; ++i) matrix.c[i] = {values[2 * i], values[2 * i + 1]};
    return matrix;
}
}  // namespace

void librpa_crpa_project_one_body(int n_orbitals, int n_bands, const double* projector_ri,
                                  const double* diagonal, double* operator_ri)
{
    if (!diagonal || !operator_ri)
        throw LIBRPA_INVALID_ARGUMENT("missing projected-operator buffers");
    const auto p = unpack_complex(n_orbitals, n_bands, projector_ri);
    const auto result = librpa_int::project_crpa_one_body(p, {diagonal, diagonal + n_bands});
    std::copy_n(reinterpret_cast<const double*>(result.c), 2 * result.size, operator_ri);
}

void librpa_crpa_project_orbitals(int n_aos, int n_spinor, int n_bands, int n_orbitals,
                                  const double* coefficients_ri, const double* overlap_ri,
                                  const double* phi_ri, double* projector_ri)
{
    using namespace librpa_int;
    if (!projector_ri) throw LIBRPA_INVALID_ARGUMENT("missing orbital-projector buffer");
    const auto c = unpack_complex(n_aos * n_spinor, n_bands, coefficients_ri);
    const auto s = unpack_complex(n_aos, n_aos, overlap_ri);
    const auto phi = unpack_complex(n_aos * n_spinor, n_orbitals, phi_ri);
    const auto p = project_crpa_orbitals(c, s, phi, n_spinor);
    std::copy_n(reinterpret_cast<const double*>(p.c), 2 * p.size, projector_ri);
}

void librpa_crpa_rotate_projector(int n_bands, int n_old, int n_new, const double* projector_ri,
                                  const double* rotation_ri, double* rotated_projector_ri)
{
    using namespace librpa_int;
    if (!rotated_projector_ri) throw LIBRPA_INVALID_ARGUMENT("missing rotated-projector buffer");
    const auto p = unpack_complex(n_old, n_bands, projector_ri);
    const auto rotation = unpack_complex(n_old, n_new, rotation_ri);
    const auto rotated = rotate_crpa_projector(p, rotation);
    std::copy_n(reinterpret_cast<const double*>(rotated.c), 2 * rotated.size, rotated_projector_ri);
}

void librpa_crpa_rotate_tensor(int n_old, int n_new, const double* tensor_ri,
                               const double* left_rotation_ri, const double* right_rotation_ri,
                               double* rotated_ri)
{
    if (!rotated_ri) throw LIBRPA_INVALID_ARGUMENT("missing rotated-interaction buffer");
    const auto left = unpack_complex(n_old, n_new, left_rotation_ri);
    const auto right = unpack_complex(n_old, n_new, right_rotation_ri);
    const auto input = unpack_complex(n_old * n_old, n_old * n_old, tensor_ri);
    const auto output = librpa_int::rotate_crpa_tensor(input, left, right);
    std::copy_n(reinterpret_cast<const double*>(output.c), 2 * output.size, rotated_ri);
}

void librpa_crpa_tensor_to_fermion(int n_orbitals, const double* density_ri, double* fermion_ri)
{
    if (!fermion_ri) throw LIBRPA_INVALID_ARGUMENT("missing fermionic-interaction buffer");
    const auto input = unpack_complex(n_orbitals * n_orbitals, n_orbitals * n_orbitals, density_ri);
    const auto fermion = librpa_int::crpa_tensor_to_fermion(input, n_orbitals);
    std::copy_n(reinterpret_cast<const double*>(fermion.c), 2 * fermion.size, fermion_ri);
}

void librpa_crpa_tensor_averages(int n_orbitals, const double* tensor_ri, double* averages_ha_ri)
{
    if (!averages_ha_ri) throw LIBRPA_INVALID_ARGUMENT("missing cRPA tensor-average buffer");
    const auto tensor = unpack_complex(n_orbitals * n_orbitals, n_orbitals * n_orbitals, tensor_ri);
    const auto averages = librpa_int::crpa_tensor_averages(tensor, n_orbitals);
    for (int channel = 0; channel < 3; ++channel)
    {
        averages_ha_ri[2 * channel] = averages[channel].real();
        averages_ha_ri[2 * channel + 1] = averages[channel].imag();
    }
}

LibrpaCrpaContinuedResult* librpa_continue_crpa(const LibrpaCrpaResult* result, int n_queries,
                                                const double* queries_ha_ri, int n_parameters,
                                                double tolerance_ha)
{
    LibrpaOptions options;
    librpa_init_options(&options);
    options.n_params_anacon = n_parameters;
    return librpa_continue_crpa_with_options(result, n_queries, queries_ha_ri, &options,
                                             tolerance_ha);
}

LibrpaCrpaContinuedResult* librpa_continue_crpa_with_options(const LibrpaCrpaResult* result,
                                                             int n_queries,
                                                             const double* queries_ha_ri,
                                                             const LibrpaOptions* options,
                                                             double tolerance_ha)
{
    if (!result || !options || n_queries <= 0 || !queries_ha_ri)
        throw LIBRPA_INVALID_ARGUMENT("missing cRPA continuation options or queries");
    std::vector<std::complex<double>> queries;
    for (int i = 0; i < n_queries; ++i)
        queries.emplace_back(queries_ha_ri[2 * i], queries_ha_ri[2 * i + 1]);
    auto output = std::make_unique<LibrpaCrpaContinuedResult>();
    output->tensors = librpa_int::continue_crpa_interactions(
        result->data().values, queries, options->n_params_anacon, tolerance_ha,
        options->anacon_method, options->anacon_ridge_lambda, options->anacon_ridge_den_weight,
        options->anacon_aaa_tolerance);
    return output.release();
}

void librpa_delete_crpa_continued_result(LibrpaCrpaContinuedResult* result) { delete result; }
int librpa_crpa_continued_size(const LibrpaCrpaContinuedResult* result)
{
    if (!result) throw LIBRPA_INVALID_ARGUMENT("missing continued cRPA result");
    return result->tensors.size();
}
void librpa_crpa_continued_tensor(const LibrpaCrpaContinuedResult* result, int tensor, int query,
                                  LibrpaCrpaContinuedTensor* view)
{
    if (!result || !view || tensor < 0 || query < 0)
        throw LIBRPA_INVALID_ARGUMENT("missing continued cRPA tensor view");
    const auto& record = result->tensors.at(tensor);
    *view = {record.atom_index,
             record.spin_left,
             record.spin_right,
             record.n_orbitals,
             reinterpret_cast<const double*>(record.values.at(query).c),
             record.status.at(query).data(),
             record.spread.at(query).data(),
             record.max_source_residual,
             record.max_holdout_residual};
}

void librpa_crpa_model_site(const LibrpaCrpaResult* result, int parent, int site, int* atom,
                            int* column_start, int* n_orbitals)
{
    if (!result || !atom || !column_start || !n_orbitals || site < 0)
        throw LIBRPA_INVALID_ARGUMENT("missing cRPA model site");
    const auto& group =
        (parent ? result->data().model.parent_sites : result->data().model.output_sites).at(site);
    *atom = group.atom_index;
    *column_start = group.orb_start;
    *n_orbitals = group.n_orbitals;
}

const char* librpa_crpa_model_label(const LibrpaCrpaResult* result, int parent, int site,
                                    int orbital)
{
    if (!result || site < 0 || orbital < 0)
        throw LIBRPA_INVALID_ARGUMENT("missing cRPA model label");
    const auto& group =
        (parent ? result->data().model.parent_sites : result->data().model.output_sites).at(site);
    return group.orbital_labels.at(orbital).c_str();
}

int librpa_crpa_result_size(const LibrpaCrpaResult* result)
{
    if (!result) throw LIBRPA_INVALID_ARGUMENT("null cRPA result");
    const auto count =
        result->data().values.tensors.size() * result->data().values.frequencies.size();
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("cRPA tensor count exceeds the public integer range");
    return static_cast<int>(count);
}

void librpa_crpa_result_tensor(const LibrpaCrpaResult* result, int index, LibrpaCrpaTensor* tensor)
{
    if (!tensor || index < 0 || index >= librpa_crpa_result_size(result))
        throw LIBRPA_INVALID_ARGUMENT("invalid cRPA tensor query");
    const auto nf = result->data().values.frequencies.size();
    const auto& record = result->data().values.tensors[index / nf];
    const auto& site = result->data().values.sites.at(record.site_index);
    tensor->atom_index = site.atom_index;
    tensor->spin_left = record.spin_left;
    tensor->spin_right = record.spin_right;
    tensor->n_orbitals = site.n_orbitals;
    tensor->frequency_ha = result->data().values.frequencies[index % nf];
    tensor->bare_ri = reinterpret_cast<const double*>(record.bare.c);
    tensor->partially_screened_ri = reinterpret_cast<const double*>(record.u.at(index % nf).c);
    tensor->fully_screened_ri =
        record.w.empty() ? nullptr : reinterpret_cast<const double*>(record.w.at(index % nf).c);
}

void librpa_crpa_result_bands(const LibrpaCrpaResult* result, int response_window, int spin,
                              int kpoint, int* count, const int** bands)
{
    if (!result || !count || !bands || spin < 0 || kpoint < 0)
        throw LIBRPA_INVALID_ARGUMENT("invalid cRPA window query");
    const auto& row =
        (response_window ? result->data().response_bands : result->data().orbital_bands)
            .at(spin)
            .at(kpoint);
    *count = static_cast<int>(row.size());
    *bands = row.data();
}

void librpa_crpa_result_weights(const LibrpaCrpaResult* result, int spin, int kpoint, int* count,
                                const double** weights)
{
    if (!result || !count || !weights || spin < 0 || kpoint < 0)
        throw LIBRPA_INVALID_ARGUMENT("invalid cRPA endpoint-weight query");
    const auto& row = result->data().response_weights.at(spin).at(kpoint);
    *count = static_cast<int>(row.size());
    *weights = row.data();
}

const char* librpa_crpa_result_orbital_label(const LibrpaCrpaResult* result, int atom, int orbital)
{
    if (!result || orbital < 0) throw LIBRPA_INVALID_ARGUMENT("invalid cRPA orbital query");
    for (const auto& site : result->data().values.sites)
        if (site.atom_index == atom) return site.orbital_labels.at(orbital).c_str();
    throw LIBRPA_INVALID_ARGUMENT("cRPA output does not contain the requested atom");
}
