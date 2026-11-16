#include <exception>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/collective_stage.h"
#include "../driver.h"
#include "../qsgw/vxc_io.h"
#include "../read_data.h"
#include "../reader/localized_operator_io.h"
#include "../reader_2d_coulomb_head.h"
#include "../task.h"
#include "crpa.h"
#include "librpa_input.h"
#include "librpa_compute.h"

namespace
{
using namespace librpa_int;
std::string input_path(const std::string& base, const std::string& filename)
{
    return is_absolute_path(filename) ? filename : join_path(base, filename);
}
template <class Function>
void root_stage(const std::string& label, Function&& operation)
{
    const auto& comm = global::mpi_comm_global_h;
    collective_all_stage(comm, label,
                         [&]
                         {
                             if (comm.is_root()) operation();
                         });
}
std::vector<LibrpaQsgwMatrix> matrix_views(const librpa::reader::VxcMatrices& matrices)
{
    std::vector<LibrpaQsgwMatrix> result;
    for (const auto& [spin, ks] : matrices)
        for (const auto& [k, matrix] : ks)
        {
            if (matrix.nr() != matrix.nc()) throw std::runtime_error("QSGW Vxc must be square");
            result.push_back({spin, k, matrix.nr(), matrix.major() == MAJOR::ROW,
                              reinterpret_cast<const double*>(matrix.ptr())});
        }
    return result;
}
struct NativeViews
{
    std::vector<std::vector<LibrpaLocalizedBlock>> blocks;
    std::vector<LibrpaLocalizedOperator> operators;
    std::vector<int> dimensions;
    std::vector<LibrpaLocalizedOperator> hamiltonians;
    LibrpaNativeVelocityInput input{};
    explicit NativeViews(const LocalizedAoOperators& source)
    {
        const auto append = [&](const LocalizedMatrix& matrices)
        {
            const int dimension = matrices.empty() ? 0 : matrices.begin()->second.nr();
            std::vector<LibrpaLocalizedBlock> list;
            for (const auto& [r, matrix] : matrices)
            {
                if (matrix.nr() != dimension || matrix.nc() != dimension)
                    throw std::runtime_error(
                        "Native QSGW operator blocks must be square and equal-sized");
                list.push_back({{r.x, r.y, r.z},
                                reinterpret_cast<const double*>(matrix.ptr()),
                                matrix.major() == MAJOR::ROW});
            }
            dimensions.push_back(dimension);
            blocks.push_back(std::move(list));
        };
        append(source.overlap);
        for (const auto& position : source.position) append(position);
        for (const auto& h : source.native_dft_hamiltonian) append(h);
        for (std::size_t i = 0; i < blocks.size(); ++i)
            operators.push_back(
                {dimensions[i], static_cast<int>(blocks[i].size()), blocks[i].data()});
        for (std::size_t i = 4; i < operators.size(); ++i) hamiltonians.push_back(operators[i]);
        input.n_spins = hamiltonians.size();
        input.hamiltonian = hamiltonians.data();
        input.overlap = operators[0];
        for (int axis = 0; axis < 3; ++axis) input.position[axis] = operators[axis + 1];
    }
};
struct OutputFiles
{
    std::string directory;
    const LocalizedAoOperators* native = nullptr;
    int nspins = 0;
    std::exception_ptr error;
    int write(const LibrpaQsgwOutput& event)
    {
        try
        {
            std::string filename;
            std::ostringstream name;
            switch (event.kind)
            {
                case LIBRPA_QSGW_ITERATIONS:
                    filename = "qsgw_iterations.dat";
                    break;
                case LIBRPA_QSGW_EIGENVALUES:
                    filename = "qsgw_eigenvalues.dat";
                    break;
                case LIBRPA_QSGW_HARTREE:
                    filename = "qsgw_hartree.csv";
                    break;
                case LIBRPA_QSGW_MATRICES:
                    filename = "qsgw_matrices.dat";
                    break;
                case LIBRPA_QSGW_QUASIPARTICLES:
                    filename = "qsgw_qp.csv";
                    break;
                case LIBRPA_QSGW_ENERGY_QP:
                    filename = "energy_qp_" + std::to_string(event.iteration);
                    break;
                case LIBRPA_QSGW_GW_BAND:
                    name << "GW_band_spin_";
                    break;
                case LIBRPA_QSGW_KS_BAND:
                    name << "KS_band_spin_";
                    break;
                case LIBRPA_QSGW_EXX_BAND:
                    name << "EXX_band_spin_";
                    break;
                case LIBRPA_QSGW_QSGW_BAND:
                    name << "QSGW_band_spin_";
                    break;
                case LIBRPA_QSGW_HAMILTONIAN:
                case LIBRPA_QSGW_HEAD_HAMILTONIAN:
                {
                    if (!event.hamiltonian)
                        throw std::runtime_error("Missing QSGW output operator");
                    LocalizedMatrix matrices;
                    const auto& op = *event.hamiltonian;
                    for (int i = 0; i < op.n_blocks; ++i)
                    {
                        const auto& b = op.blocks[i];
                        Matz matrix(op.dimension, op.dimension,
                                    b.row_major ? MAJOR::ROW : MAJOR::COL);
                        for (std::size_t j = 0; j < matrix.size(); ++j)
                            matrix.ptr()[j] = {b.values_ri[2 * j], b.values_ri[2 * j + 1]};
                        matrices[{b.translation[0], b.translation[1], b.translation[2]}] =
                            std::move(matrix);
                    }
                    if (event.kind == LIBRPA_QSGW_HAMILTONIAN)
                    {
                        if (!native) throw std::runtime_error("Missing QSGW output geometry");
                        name << "hrs" << event.spin + 1 << "_nao_qsgw_iter_" << std::setfill('0')
                             << std::setw(4) << event.iteration << ".csr";
                        librpa::reader::write_qsgw_hamiltonian_csr(
                            join_path(directory, name.str()), matrices, *native, event.iteration,
                            nspins, event.spin + 1);
                    }
                    else
                    {
                        name << "qsgw_head_HR_update" << event.iteration << "_spin" << event.spin
                             << ".eV.csr";
                        librpa::reader::write_headwing_hamiltonian_csr(
                            join_path(directory, name.str()), matrices);
                    }
                    return 0;
                }
            }
            if (filename.empty())
            {
                name << event.spin + 1 << '_' << event.iteration << ".dat";
                filename = name.str();
            }
            std::ofstream file(join_path(directory, filename),
                               event.append ? std::ios::app : std::ios::trunc);
            file.write(event.text, event.text_size);
            if (!file) throw std::runtime_error("Cannot write QSGW output " + filename);
            return 0;
        }
        catch (...)
        {
            error = std::current_exception();
            return 1;
        }
    }
};
}  // namespace

void driver::task_qsgw()
{
    using namespace librpa_int;
    using namespace global;
    if (driver_params.constants_choice == "aims" && driver_params.qsgw_vxc_basis != "state")
        throw LIBRPA_RUNTIME_ERROR("FHI-aims QSGW Vxc requires the KS-state basis");
    const auto* handler = h.get_c_handler();
    int ns, nk, nstates, naos, nspinor;
    librpa_get_meanfield_dimensions(handler, 0, &ns, &nk, &nstates, &naos, &nspinor);
    const int source_nk = nk;
    LibrpaQsgwControls controls;
    librpa_init_qsgw_controls(&controls);
    controls.min_iterations = driver_params.qsgw_min_iter;
    controls.max_iterations = driver_params.qsgw_max_iter;
    controls.unoccupied_keep = driver_params.qsgw_unoccupied_keep;
    controls.cut_mode = driver_params.qsgw_cut_mode;
    controls.mixing_beta = driver_params.qsgw_mixing_beta;
    controls.cut_shift_ha = driver_params.qsgw_cut_shift_ha;
    controls.convergence_tolerance_ev = driver_params.qsgw_convergence_tolerance_ev;
    controls.mixer = driver_params.qsgw_mixer.c_str();
    controls.velocity_mode = driver_params.qsgw_headwing_velocity.c_str();
    controls.restart_file = driver_params.qsgw_restart_file.c_str();
    controls.checkpoint_file = driver_params.qsgw_checkpoint_file.c_str();
    controls.output_matrices = driver_params.qsgw_write_iteration_matrices;
    controls.update_hartree = driver_params.qsgw_update_hartree;
    controls.output_qp = driver_params.qsgw_output_qp;
    controls.export_hamiltonian = driver_params.qsgw_export_hamiltonian_for_pyatb;
    controls.state_low = driver_params.i_state_low;
    controls.state_high = driver_params.i_state_high;
    controls.interband_velocity = driver_params.constants_choice == "aims";
    controls.expand_ibz = driver_params.crpa_enabled;
    std::optional<librpa::reader::HartreeCoulombQ0> hartree_metric;
    std::vector<librpa::reader::HartreeLriBlock> hartree_lri;
    if (controls.update_hartree)
        root_stage("QSGW dedicated Hartree metric input",
                   [&]
                   {
                       hartree_metric = read_hartree_coulomb_q0(
                           driver_params.input_dir, driver_params.prefix_coul_hartree,
                           driver_params.version_coul_reader, get_bool(opts.use_shrink_abfs));
                       hartree_lri = read_hartree_lri_blocks(driver_params.input_dir,
                                                             driver_params.prefix_lri_coeff_hartree,
                                                             get_bool(opts.use_shrink_abfs));
                   });
    auto task = h.create_qsgw_task(opts, controls);
    librpa_prepare_qsgw_task(task.get());
    librpa_get_meanfield_dimensions(handler, 0, &ns, &nk, &nstates, &naos, &nspinor);

    const auto band_path = input_path(driver_params.input_dir, driver_params.fn_band_kpath_info);
    int has_band = 0;
    root_stage("QSGW band-path discovery",
               [&]
               {
                   has_band =
                       !driver_params.fn_band_kpath_info.empty() && path_exists(band_path.c_str());
                   lib_printf("QSGW band k-path file %s %s; %s\n", band_path.c_str(),
                              has_band ? "found" : "not found",
                              has_band ? "include band-path updates" : "use SCF-grid updates only");
               });
    mpi_comm_global_h.bcast(&has_band, 1, 0);
    if (has_band)
    {
        collective_all_stage(mpi_comm_global_h, "QSGW band k-path input",
                             [&] { read_band_kpath_info(band_path); });
        mpi_comm_global_h.barrier();
        collective_all_stage(mpi_comm_global_h, "QSGW band mean-field input",
                             [&] { read_band_meanfield_data(driver_params.input_dir); });
        mpi_comm_global_h.barrier();
    }
    read_Vq_row(driver_params.input_dir, driver_params.prefix_coul_cut, opts.vq_threshold,
                local_atpair, true, driver_params.version_coul_reader,
                get_bool(opts.use_shrink_abfs));
    const bool update_head = get_bool(opts.replace_w_head) &&
                             (opts.option_dielect_func == 3 || opts.option_dielect_func == 4);
    if (update_head && driver_params.qsgw_headwing_velocity != "current_hr")
        read_headwing_input(driver_params.input_dir, false);

    std::optional<LocalizedAoOperators> native;
    std::optional<NativeViews> native_views;
    librpa::reader::VxcMatrices vxc, vxc_band;
    std::vector<LibrpaQsgwMatrix> vxc_views, band_views;
    std::vector<LibrpaQsgwLriBlock> hartree_lri_views;
    LibrpaQsgwInput input{};
    root_stage(
        "QSGW host numerical input",
        [&]
        {
            const bool aims = driver_params.constants_choice == "aims";
            vxc =
                librpa::reader::read_qsgw_vxc(driver_params.input_dir, driver_params.prefix_vxc_scf,
                                              ns, source_nk, nspinor, aims, false);
            if (has_band)
            {
                int bs, bk, bn, ba, bc;
                librpa_get_meanfield_dimensions(handler, 1, &bs, &bk, &bn, &ba, &bc);
                vxc_band = librpa::reader::read_qsgw_vxc(
                    driver_params.input_dir, driver_params.prefix_vxc_band, bs, bk, bc, aims, true);
            }
            vxc_views = matrix_views(vxc);
            band_views = matrix_views(vxc_band);
            input.n_vxc = vxc_views.size();
            input.vxc = vxc_views.data();
            input.n_vxc_band = band_views.size();
            input.vxc_band = band_views.data();
            input.vxc_in_ao_basis = driver_params.qsgw_vxc_basis == "nao";
            if (hartree_metric)
            {
                input.hartree_n_aux = hartree_metric->dimension;
                input.hartree_coulomb_q0_ri =
                    reinterpret_cast<const double*>(hartree_metric->values.data());
                for (const auto& block : hartree_lri)
                    hartree_lri_views.push_back(
                        {block.atom_i,
                         block.atom_j,
                         {block.translation[0], block.translation[1], block.translation[2]},
                         block.n_aos_i,
                         block.n_aos_j,
                         block.n_aux_i,
                         block.values.data()});
                input.n_hartree_lri_blocks = static_cast<int>(hartree_lri_views.size());
                input.hartree_lri_blocks = hartree_lri_views.data();
            }
            if (driver_params.qsgw_export_hamiltonian_for_pyatb ||
                (update_head && driver_params.qsgw_headwing_velocity == "current_hr"))
            {
                native = librpa::reader::read_headwing_ao_operators(
                    input_path(driver_params.input_dir, driver_params.qsgw_fn_sr),
                    input_path(driver_params.input_dir, driver_params.qsgw_fn_rr));
                for (int spin = 0; spin < ns; ++spin)
                    native->native_dft_hamiltonian.push_back(
                        librpa::reader::read_headwing_hamiltonian_csr(
                            input_path(driver_params.input_dir,
                                       "hrs" + std::to_string(spin + 1) + "_nao.csr"),
                            0.5));
                native_views.emplace(*native);
                input.native_operators = &native_views->input;
            }
            if (update_head && get_bool(opts.use_2d_dielectric))
            {
                const auto metadata = read_strict_2d_coulomb_head_metadata(
                    join_path(driver_params.input_dir, "librpa_2d_coulomb_head.txt"));
                input.inplane_area_bohr2 = metadata.inplane_area_bohr2;
                input.auxiliary_monopole_norm_squared = metadata.auxiliary_monopole_norm_squared;
            }
        });
    librpa_set_qsgw_input(task.get(), mpi_comm_global_h.is_root() ? &input : nullptr);
    OutputFiles outputs{opts.output_dir, native ? &*native : nullptr, ns, {}};
    librpa_set_qsgw_output_sink(
        task.get(), [](const LibrpaQsgwOutput* event, void* context)
        { return static_cast<OutputFiles*>(context)->write(*event); }, &outputs);
    std::optional<CrpaDriverInput> crpa_input;
    struct CrpaOutputFiles
    {
        CrpaDriverInput* input;
        std::string directory;
        std::exception_ptr* error;
    } crpa_outputs{nullptr, opts.output_dir, &outputs.error};
    if (driver_params.crpa_enabled)
    {
        crpa_input.emplace(read_crpa_driver_input());
        crpa_outputs.input = &*crpa_input;
        auto numerical_input = crpa_input->input();
        const auto definition = crpa_input->tracking_definition();
        LibrpaQsgwCrpaControls crpa_controls;
        librpa_init_qsgw_crpa_controls(&crpa_controls);
        crpa_controls.interval = driver_params.crpa_interval;
        crpa_controls.track_orbitals = driver_params.crpa_track_orbitals;
        crpa_controls.output_analysis = driver_params.crpa_output_analysis;
        crpa_controls.tracking_definition = definition.c_str();
        librpa_set_qsgw_crpa(
            task.get(), &numerical_input, &crpa_controls,
            [](const LibrpaCrpaResult* result, const LibrpaOrbitalResult* analysis, int state,
               int update, int final, void* context)
            {
                auto& output = *static_cast<CrpaOutputFiles*>(context);
                try
                {
                    std::ostringstream name;
                    name << "state_" << std::setfill('0') << std::setw(4) << state;
                    const auto directory =
                        join_path(join_path(output.directory, "crpa"), name.str());
                    write_crpa_state_outputs(result, *output.input, directory, state, update,
                                             final != 0, false, analysis);
                    return 0;
                }
                catch (...)
                {
                    *output.error = std::current_exception();
                    return 1;
                }
            },
            &crpa_outputs);
    }
    try
    {
        librpa_run_qsgw_task(task.get());
    }
    catch (...)
    {
        if (outputs.error) std::rethrow_exception(outputs.error);
        throw;
    }
}
