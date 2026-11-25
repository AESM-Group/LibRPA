#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../mpi/collective_stage.h"
#include "instance_manager.h"
#include "dataset_helper.h"
#include "hartree_input.h"
#include "gw_response_helper.h"
#include "librpa_compute.h"
#include "options.h"
#include "qsgw_helper.h"

struct LibrpaQsgwTask
{
    std::unique_ptr<librpa_int::qsgw::Task> task;
    std::shared_ptr<librpa_int::Dataset> dataset;
};

namespace
{
using namespace librpa_int;
using namespace librpa_int::qsgw;
std::string copied(const char* text, const char* fallback) { return text ? text : fallback; }
Matz decode_matrix(const LibrpaQsgwMatrix& source)
{
    if (source.dimension <= 0 || !source.values_ri ||
        (source.row_major != 0 && source.row_major != 1))
        throw LIBRPA_INVALID_ARGUMENT("QSGW matrix dimensions/storage are invalid");
    Matz matrix(source.dimension, source.dimension, source.row_major ? MAJOR::ROW : MAJOR::COL);
    for (std::size_t i = 0; i < matrix.size(); ++i)
        matrix.ptr()[i] = {source.values_ri[2 * i], source.values_ri[2 * i + 1]};
    return matrix;
}
SpinKMatrixMap decode_matrices(int count, const LibrpaQsgwMatrix* source)
{
    if (count < 0 || (count && !source))
        throw LIBRPA_INVALID_ARGUMENT("QSGW matrix list is incomplete");
    SpinKMatrixMap result;
    for (int i = 0; i < count; ++i)
    {
        const auto& block = source[i];
        if (block.spin < 0 || block.kpoint < 0 ||
            !result[block.spin].emplace(block.kpoint, decode_matrix(block)).second)
            throw LIBRPA_INVALID_ARGUMENT("QSGW matrix repeats or has an invalid state index");
    }
    return result;
}
LocalizedMatrix decode_localized(const LibrpaLocalizedOperator& source, int dimension)
{
    if (source.dimension != dimension || source.n_blocks <= 0 || !source.blocks)
        throw LIBRPA_INVALID_ARGUMENT("QSGW native operator dimensions/blocks are incomplete");
    LocalizedMatrix result;
    for (int i = 0; i < source.n_blocks; ++i)
    {
        const auto& block = source.blocks[i];
        const LibrpaQsgwMatrix view{0, 0, dimension, block.row_major, block.values_ri};
        const Vector3_Order<int> r{block.translation[0], block.translation[1],
                                   block.translation[2]};
        if (!result.emplace(r, decode_matrix(view)).second)
            throw LIBRPA_INVALID_ARGUMENT("QSGW native operator repeats a lattice translation");
    }
    return result;
}
Cs_LRI decode_hartree_lri(const LibrpaQsgwInput& source, const AtomicBasis& basis_wfc)
{
#ifndef LIBRPA_LIBRI_HAS_HARTREE
    (void)source;
    (void)basis_wfc;
    throw LIBRPA_INVALID_ARGUMENT(
        "QSGW Hartree feedback requires LibRI Hartree support");
#else
    if (source.n_hartree_lri_blocks < 0 ||
        (source.n_hartree_lri_blocks && !source.hartree_lri_blocks))
        throw LIBRPA_INVALID_ARGUMENT("QSGW unwrapped Hartree LRI blocks are incomplete");
    Cs_LRI result;
    result.use_libri = true;
    for (int index = 0; index < source.n_hartree_lri_blocks; ++index)
    {
        const auto& block = source.hartree_lri_blocks[index];
        if (block.atom_i < 0 || block.atom_j < 0 ||
            static_cast<std::size_t>(block.atom_i) >= basis_wfc.n_atoms ||
            static_cast<std::size_t>(block.atom_j) >= basis_wfc.n_atoms || block.n_aos_i <= 0 ||
            block.n_aos_j <= 0 || block.n_aux_i <= 0 || !block.values ||
            static_cast<std::size_t>(block.n_aos_i) != basis_wfc[block.atom_i] ||
            static_cast<std::size_t>(block.n_aos_j) != basis_wfc[block.atom_j])
            throw LIBRPA_INVALID_ARGUMENT("QSGW unwrapped Hartree LRI dimensions are invalid");
        const auto ni = static_cast<std::size_t>(block.n_aos_i);
        const auto nj = static_cast<std::size_t>(block.n_aos_j);
        const auto nmu = static_cast<std::size_t>(block.n_aux_i);
        RI::Tensor<double> tensor({nmu, ni, nj});
        for (std::size_t i = 0; i < ni; ++i)
            for (std::size_t j = 0; j < nj; ++j)
                for (std::size_t mu = 0; mu < nmu; ++mu)
                {
                    const double value = block.values[(i * nj + j) * nmu + mu];
                    if (!std::isfinite(value))
                        throw LIBRPA_INVALID_ARGUMENT(
                            "QSGW unwrapped Hartree LRI value is nonfinite");
                    tensor(mu, i, j) = value;
                }
        const std::array<int, 3> r{block.translation[0], block.translation[1],
                                   block.translation[2]};
        if (!result.data_libri[block.atom_i]
                 .emplace(std::make_pair(block.atom_j, r), std::move(tensor))
                 .second)
            throw LIBRPA_INVALID_ARGUMENT("QSGW unwrapped Hartree LRI repeats a physical key");
    }
    return result;
#endif
}
}  // namespace

librpa_int::qsgw::Task& librpa_int::api::qsgw_task_instance(LibrpaQsgwTask* task)
{
    if (!task || !task->task) throw LIBRPA_INVALID_ARGUMENT("Missing QSGW task");
    return *task->task;
}

librpa_int::dataset_ptr_t librpa_int::api::qsgw_task_dataset(LibrpaQsgwTask* task)
{
    qsgw_task_instance(task);
    return task->dataset;
}

void librpa_init_qsgw_controls(LibrpaQsgwControls* controls)
{
    if (!controls) throw LIBRPA_INVALID_ARGUMENT("Missing QSGW controls");
    *controls = {1,
                 10,
                 10,
                 0,
                 0.2,
                 2.0 / 27.211386245988,
                 1.0e-4,
                 "none",
                 "reference",
                 "none",
                 "qsgw_checkpoint.bin",
                 0,
                 1,
                 1,
                 0,
                 2147483647,
                 0,
                 0,
                 1};
}
LibrpaQsgwTask* librpa_create_qsgw_task(LibrpaHandler* handler, const LibrpaOptions* options,
                                        const LibrpaQsgwControls* controls)
{
    using namespace librpa_int;
    using namespace librpa_int::qsgw;
    auto ds = api::get_dataset_instance(handler);
    // The native G0W0/cRPA route arms an expanded full-BZ eigenframe on the dataset.
    // That frame belongs to the task that built it: make_analytic_headwing selects it
    // on `headwing_full_bz_ready` alone, so a QSGW task would otherwise differentiate
    // its own live H/S/r on the previous task's mesh and velocity, and would also drop
    // the symmetric wing path. QSGW never arms a frame of its own, so drop any
    // inherited one here; the G0W0/cRPA prepare step arms its frame after this point.
    ds->invalidate_native_headwing_frame();
    TaskControls input;
    collective_all_stage(
        ds->comm_h, "QSGW task controls",
        [&]
        {
            if (!options || !controls)
                throw LIBRPA_INVALID_ARGUMENT("Missing QSGW options/controls");
            input.qsgw_min_iter = controls->min_iterations;
            input.qsgw_max_iter = controls->max_iterations;
            input.qsgw_unoccupied_keep = controls->unoccupied_keep;
            input.qsgw_cut_mode = controls->cut_mode;
            input.qsgw_mixing_beta = controls->mixing_beta;
            input.qsgw_cut_shift_ha = controls->cut_shift_ha;
            input.qsgw_convergence_tolerance_ev = controls->convergence_tolerance_ev;
            input.qsgw_mixer = copied(controls->mixer, "none");
            input.qsgw_headwing_velocity = copied(controls->velocity_mode, "reference");
            input.qsgw_restart_file = copied(controls->restart_file, "none");
            input.qsgw_checkpoint_file = copied(controls->checkpoint_file, "none");
            input.qsgw_write_iteration_matrices = controls->output_matrices != 0;
            input.qsgw_output_qp = controls->output_qp != 0;
            input.qsgw_export_hamiltonian_for_pyatb = controls->export_hamiltonian != 0;
            input.i_state_low = controls->state_low;
            input.i_state_high = controls->state_high;
            input.interband_velocity = controls->interband_velocity != 0;
            input.expand_ibz = controls->expand_ibz != 0;
            input.qsgw_update_hartree = controls->update_hartree != 0;
#ifndef LIBRPA_LIBRI_HAS_HARTREE
            if (input.qsgw_update_hartree)
                throw LIBRPA_INVALID_ARGUMENT(
                    "QSGW Hartree feedback requires LibRI Hartree support");
#endif
            if (input.qsgw_min_iter < 1 || input.qsgw_max_iter < input.qsgw_min_iter ||
                input.qsgw_unoccupied_keep < 0 ||
                (input.qsgw_cut_mode != 0 && input.qsgw_cut_mode != 1) ||
                !std::isfinite(input.qsgw_cut_shift_ha) ||
                !std::isfinite(input.qsgw_convergence_tolerance_ev) ||
                input.qsgw_convergence_tolerance_ev <= 0.0 ||
                (input.qsgw_mixer != "none" && input.qsgw_mixer != "linear") ||
                !(input.qsgw_mixing_beta > 0.0 && input.qsgw_mixing_beta <= 1.0) ||
                (input.qsgw_headwing_velocity != "reference" &&
                 input.qsgw_headwing_velocity != "rotate_reference" &&
                 input.qsgw_headwing_velocity != "current_hr"))
                throw LIBRPA_INVALID_ARGUMENT(
                    "Invalid QSGW iteration, mixing or velocity controls");
        });
    const auto normalized = resolve_linear_algebra_options(*options);
    auto result = std::make_unique<LibrpaQsgwTask>();
    result->dataset = ds;
    auto context = std::make_shared<TaskContext>(TaskContext{
        ds->atoms, ds->basis_wfc, ds->blacs_h, ds->comm_h, ds->is_band_calc_done, ds->kfrac_band_list,
        ds->mf, ds->mf_band, ds->p_exx, ds->p_g0w0, ds->p_headwing, ds->pbc,
        ds->symmetry_context, ds->tfg, ds->velocity_matrix,
        ds->basis_aux, ds->basis_aux_shrink, ds->basis_convention,
        [ds] { return build_hartree_density_k(*ds); },
        [ds] { ds->invalidate_compute_objects(); },
        [ds](int spins, int kpoints, int bands, int aos, int spinor)
        {
            ds->mf.set(spins, kpoints, bands, aos, spinor, 0, bands, 0, aos);
            ds->comm_h.barrier();
            KPointBlacsProcessShape shape;
            shape.favor_square_blacs_grid = true;
            auto& context = ds->scfk_blacs_ctxt;
            context.init(shape, ds->comm_h.comm, kpoints);
            ds->comm_h.barrier();
            const int ao_block = get_capped_blacs_block_size(aos, wfc_gemm_block_size_opt,
                                                             context.blacs_h);
            const int band_block = get_capped_blacs_block_size(bands, wfc_gemm_block_size_opt,
                                                               context.blacs_h);
            ds->desc_wfc_kb = context.create_array_desc(aos, bands, ao_block, band_block);
            ds->desc_wfc_kb_full = context.create_array_desc(aos, bands, aos, bands);
        },
        [ds](const LibrpaOptions& opts) { initialize_ds_tfgrids(*ds, opts); },
        [ds](const LibrpaOptions& opts, bool wing) { initialize_ds_headwing(*ds, opts, wing); },
        [ds](bool rotations)
        {
            reject_spinor_symmetry_speedup(*ds, "QSGW state");
            initialize_symmetry_context(*ds, rotations);
        },
        [ds](const LibrpaOptions& opts, const ResponseObserver& response)
        { api::build_gw_sigma_for_task(*ds, opts, response); }});
    result->task = std::make_unique<Task>(std::move(context), normalized, std::move(input));
    return result.release();
}
void librpa_prepare_qsgw_task(LibrpaQsgwTask* task)
{
    librpa_int::api::qsgw_task_instance(task).prepare();
}
void librpa_set_qsgw_input(LibrpaQsgwTask* task, const LibrpaQsgwInput* root_input)
{
    using namespace librpa_int;
    using namespace librpa_int::qsgw;
    auto& engine = api::qsgw_task_instance(task);
    TaskInput input;
    collective_all_stage(
        task->dataset->comm_h, "QSGW numeric input",
        [&]
        {
            if (!task->dataset->comm_h.is_root()) return;
            if (!root_input) throw LIBRPA_INVALID_ARGUMENT("Missing QSGW numeric input on root");
            input.vxc = decode_matrices(root_input->n_vxc, root_input->vxc);
            input.vxc_band = decode_matrices(root_input->n_vxc_band, root_input->vxc_band);
            input.vxc_basis = root_input->vxc_in_ao_basis ? VxcBasis::Nao : VxcBasis::State;
            input.inplane_area_bohr2 = root_input->inplane_area_bohr2;
            input.auxiliary_monopole_norm_squared = root_input->auxiliary_monopole_norm_squared;
            if (root_input->hartree_n_aux < 0 ||
                (root_input->hartree_n_aux > 0 && !root_input->hartree_coulomb_q0_ri) ||
                (root_input->hartree_n_aux == 0 && root_input->hartree_coulomb_q0_ri))
                throw LIBRPA_INVALID_ARGUMENT("QSGW Hartree auxiliary metric is incomplete");
            if (root_input->hartree_n_aux > 0)
                input.hartree_coulomb_q0 = decode_matrix(
                    {0, 0, root_input->hartree_n_aux, 1, root_input->hartree_coulomb_q0_ri});
            if (root_input->n_hartree_lri_blocks != 0 || root_input->hartree_lri_blocks)
                input.hartree_cs_raw = decode_hartree_lri(*root_input, task->dataset->basis_wfc);
            if (const auto* source = root_input->native_operators)
            {
                auto& operators = input.ao_operators.emplace();
                operators.dimension = source->overlap.dimension;
                operators.overlap = decode_localized(source->overlap, operators.dimension);
                for (int axis = 0; axis < 3; ++axis)
                    operators.position[axis] =
                        decode_localized(source->position[axis], operators.dimension);
                if (source->n_spins <= 0 || !source->hamiltonian)
                    throw LIBRPA_INVALID_ARGUMENT("QSGW native Hamiltonians are incomplete");
                for (int spin = 0; spin < source->n_spins; ++spin)
                    operators.native_dft_hamiltonian.push_back(decode_localized(
                        source->hamiltonian[spin],
                        task->dataset->mf.get_n_aos() * task->dataset->mf.get_n_spinor()));
            }
        });
    engine.set_input(std::move(input));
}
void librpa_set_qsgw_output_sink(LibrpaQsgwTask* task, LibrpaQsgwOutputSink sink, void* user_data)
{
    using namespace librpa_int;
    api::qsgw_task_instance(task).set_output(
        [sink, user_data](const qsgw::TaskOutput& event)
        {
            if (!sink) return;
            std::vector<LibrpaLocalizedBlock> blocks;
            LibrpaLocalizedOperator op{};
            if (event.hamiltonian)
            {
                for (const auto& [r, matrix] : *event.hamiltonian)
                    blocks.push_back({{r.x, r.y, r.z},
                                      reinterpret_cast<const double*>(matrix.ptr()),
                                      matrix.major() == MAJOR::ROW});
                op.dimension =
                    event.hamiltonian->empty() ? 0 : event.hamiltonian->begin()->second.nr();
                op.n_blocks = blocks.size();
                op.blocks = blocks.data();
            }
            const LibrpaQsgwOutput output{static_cast<LibrpaQsgwOutputKind>(event.kind),
                                          event.iteration,
                                          event.spin,
                                          event.append,
                                          event.text.c_str(),
                                          event.text.size(),
                                          event.hamiltonian ? &op : nullptr};
            if (sink(&output, user_data) != 0)
                throw LIBRPA_RUNTIME_ERROR("QSGW output sink failed");
        });
}
void librpa_run_qsgw_task(LibrpaQsgwTask* task) { librpa_int::api::qsgw_task_instance(task).run(); }
void librpa_delete_qsgw_task(LibrpaQsgwTask* task) { delete task; }
