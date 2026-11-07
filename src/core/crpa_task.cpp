#include "crpa_task.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <memory>

#include "../utils/error.h"

namespace librpa_int
{
namespace
{
void collective_error(const MpiCommHandler& comm, const std::string& error)
{
    const int failed = !error.empty();
    int any_failed = 0;
    comm.allreduce(&failed, &any_failed, 1, MPI_MAX);
    if (any_failed)
        throw LIBRPA_RUNTIME_ERROR("cRPA input: " +
                                   (error.empty() ? "another MPI rank failed validation" : error));
}

CrpaContext crpa_context(CrpaTaskContext& ds, const LibrpaOptions& opts)
{
    const bool shrink = opts.use_shrink_abfs == LIBRPA_SWITCH_ON;
    // cRPA always screens and projects with full v. The independent GW
    // eps/Wc/EXX Coulomb flags do not enter this numerical context.
    return {*ds.p_chi0,   ds.cs_data,
            ds.basis_aux, ds.atpairs_local,
            ds.sinvS,     ds.vq,
            ds.blacs_h,   shrink ? ds.desc_abf_shrink : ds.desc_abf,
            shrink,       opts.sqrt_coulomb_threshold};
}

void initialize_crpa_response(CrpaTaskContext& ds, const LibrpaOptions& opts)
{
    ds.initialize_response(opts);
}

CrpaTaskResult own_crpa_task_result(CrpaResult values, CrpaPreparedModel& model,
                                    const CrpaTaskContext& dataset)
{
    CrpaTaskResult result;
    std::string error;
    try
    {
        result.values = std::move(values);
        result.model = librpa_int::snapshot_crpa_model(model, dataset.mf);
        for (const auto& row : dataset.pbc.latvec_array)
            for (double value : row) result.lattice.push_back(value);
        for (int atom = 0; atom < dataset.basis_wfc.n_atoms; ++atom)
        {
            const auto& position = dataset.atoms.coords.at(atom);
            result.positions.insert(result.positions.end(), {position.x, position.y, position.z});
            result.atom_types.push_back(dataset.atoms.types.at(atom));
            result.ao_counts.push_back(dataset.basis_wfc[atom]);
            const auto& shells = dataset.basis_wfc.get_l_shells(atom);
            result.shell_l.insert(result.shell_l.end(), shells.begin(), shells.end());
            result.shell_offsets.push_back(result.shell_l.size());
        }
        const auto& convention = dataset.basis_convention;
        result.basis_convention = {convention.bloch_phase, convention.bloch_ratom, convention.order,
                                   convention.coeff_m_negative, convention.coeff_m_positive};
        for (const auto& k : dataset.pbc.kfrac_list)
            result.k_fractional.insert(result.k_fractional.end(), {k.x, k.y, k.z});
        result.k_weights = dataset.pbc.weight_k;
        if (result.k_weights.empty())
            result.k_weights.assign(dataset.mf.get_n_kpoints(), 1.0 / dataset.mf.get_n_kpoints());
        result.response_bands = std::move(model.windows.response_bands);
        result.orbital_bands = std::move(model.windows.orbital_bands);
        result.response_weights = std::move(model.selection);
    }
    catch (const std::exception& failure)
    {
        error = "model snapshot: " + std::string(failure.what());
    }
    collective_error(dataset.comm_h, error);
    return result;
}
}  // namespace

CrpaPreparedModel prepare_crpa_task_model(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                          const CrpaTaskInput& input, CrpaTaskMode mode)
{
    std::string error;
    CrpaPreparedModel model;
    try
    {
        if (options.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON ||
            (options.parallel_routing != LIBRPA_ROUTING_LIBRI &&
             options.parallel_routing != LIBRPA_ROUTING_AUTO))
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA requires full-grid LIBRI and replicated eigenvectors");
        if (mode == CrpaTaskMode::SharedGw && options.read_sigc_mat_rf == LIBRPA_SWITCH_ON)
            throw LIBRPA_INVALID_ARGUMENT(
                "shared GW/cRPA requires a resident response and no SigC file-read shortcut");
        const bool optical = options.replace_w_head == LIBRPA_SWITCH_ON;
        if ((optical &&
             (options.option_dielect_func != 3 || options.use_2d_dielectric == LIBRPA_SWITCH_ON)) ||
            (!optical && options.option_dielect_func != 0) ||
            options.tfgrids_type != LIBRPA_TFGRID_MINIMAX || options.n_bands_chi0 >= 0 ||
            options.n_bands_exclude_chi0 != 0 ||
            (mode == CrpaTaskMode::Standalone && (options.use_fullcoul_eps != LIBRPA_SWITCH_ON ||
                                                  options.use_fullcoul_wc != LIBRPA_SWITCH_ON)))
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA requires native minimax, all loaded bands, full/full Coulomb and either "
                "no head correction or 3D analytic option 3");
        if (options.chi0_method == LIBRPA_METHOD_CONVENTIONAL)
            throw LIBRPA_INVALID_ARGUMENT("cRPA does not support chi0_method = conventional");
        const bool shrink = options.use_shrink_abfs == LIBRPA_SWITCH_ON;
        if (shrink && (options.use_shrink_chi != LIBRPA_SWITCH_ON ||
                       !dataset.basis_aux_shrink.initialized() || dataset.sinvS.empty()))
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA compressed basis requires use_shrink_chi and loaded auxiliary transforms");
        if (!shrink && options.use_shrink_chi == LIBRPA_SWITCH_ON)
            throw LIBRPA_INVALID_ARGUMENT("use_shrink_chi requires use_shrink_abfs for cRPA");
        if (!std::isfinite(options.sqrt_coulomb_threshold) || options.sqrt_coulomb_threshold < 0.0)
            throw LIBRPA_INVALID_ARGUMENT("cRPA Coulomb threshold must be finite and nonnegative");
        if (input.species_labels.empty() ||
            input.spatial_overlap.size() != static_cast<std::size_t>(dataset.mf.get_n_kpoints()))
            throw LIBRPA_INVALID_ARGUMENT("incomplete cRPA species/window/overlap input");
        for (const auto& overlap : input.spatial_overlap)
            if (overlap.nr != dataset.mf.get_n_aos() || overlap.nc != dataset.mf.get_n_aos())
                throw LIBRPA_INVALID_ARGUMENT("cRPA overlap dimensions differ from mean field");
        std::vector<int> atom_types(dataset.basis_wfc.n_atoms);
        for (std::size_t atom = 0; atom < atom_types.size(); ++atom)
            atom_types[atom] = dataset.atoms.types.at(atom);
        model = prepare_crpa_model(dataset.basis_wfc, dataset.basis_convention, atom_types,
                                   input.species_labels, input.spatial_overlap, dataset.pbc,
                                   dataset.mf, input.window);
        model.retain_raw_model = input.retain_raw_model;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
    }
    collective_error(dataset.comm_h, error);
    return model;
}

CrpaTaskResult snapshot_crpa_task_model(CrpaTaskContext& dataset, CrpaPreparedModel& model)
{
    return own_crpa_task_result({}, model, dataset);
}

CrpaTaskResult evaluate_crpa_task_from_response(CrpaTaskContext& ds, const LibrpaOptions& opts,
                                                CrpaPreparedModel& model,
                                                const Chi0QMap& full_response)
{
    auto context = crpa_context(ds, opts);
    std::unique_ptr<diele_func> constrained_headwing;
    if (opts.replace_w_head == LIBRPA_SWITCH_ON)
    {
        auto optical_options = opts;
        optical_options.use_symmetry_rpa = optical_options.use_symmetry_gw = LIBRPA_SWITCH_OFF;
        constrained_headwing = ds.make_headwing(optical_options, true, &model.selection);
        constrained_headwing->normalize_gamma_cell_weights = true;
        context.constrained_headwing = constrained_headwing.get();
    }
    std::vector<CorrelatedSubspace*> frames;
    for (const auto& frame : model.frames) frames.push_back(frame.get());
    return own_crpa_task_result(
        evaluate_crpa_onsite_from_response(context, model.selection, frames, full_response), model,
        ds);
}

CrpaTaskResult compute_crpa_task(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                 const CrpaTaskInput& input)
{
    auto* ds = &dataset;
    const auto& opts = options;
    auto model = prepare_crpa_task_model(dataset, opts, input);
    initialize_crpa_response(*ds, opts);
    auto context = crpa_context(*ds, opts);
    std::unique_ptr<diele_func> full_headwing, constrained_headwing;
    if (opts.replace_w_head == LIBRPA_SWITCH_ON)
    {
        // This first magnetic route reduces q-space screening only. Optical sums
        // retain every original full-BZ state and velocity for both response models.
        auto optical_options = opts;
        optical_options.use_symmetry_rpa = optical_options.use_symmetry_gw = LIBRPA_SWITCH_OFF;
        full_headwing = ds->make_headwing(optical_options, true, nullptr);
        constrained_headwing = ds->make_headwing(optical_options, true, &model.selection);
        full_headwing->normalize_gamma_cell_weights = true;
        constrained_headwing->normalize_gamma_cell_weights = true;
        context.full_headwing = full_headwing.get();
        context.constrained_headwing = constrained_headwing.get();
    }
    std::vector<CorrelatedSubspace*> frame_views;
    for (const auto& frame : model.frames) frame_views.push_back(frame.get());
    return own_crpa_task_result(compute_crpa_onsite(context, model.selection, frame_views), model,
                                *ds);
}

CrpaTaskResult compute_crpa_u_task(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                   const CrpaTaskInput& input)
{
    auto model = prepare_crpa_task_model(dataset, options, input, CrpaTaskMode::ResponseOnly);
    initialize_crpa_response(dataset, options);
    auto context = crpa_context(dataset, options);
    auto full_response = build_crpa_total_response(context);
    return evaluate_crpa_task_from_response(dataset, options, model, full_response);
}

CrpaTaskResult prepare_crpa_task(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                 const CrpaTaskInput& input)
{
    auto model = prepare_crpa_task_model(dataset, options, input, CrpaTaskMode::ResponseOnly);
    return snapshot_crpa_task_model(dataset, model);
}

CrpaTrackingState snapshot_crpa_tracking(const CrpaModelSnapshot& model)
{
    CrpaTrackingState state;
    state.n_aos = model.n_aos;
    state.n_spinor = model.n_spinor;
    state.parent_phi.resize(model.spin_k.size());
    for (std::size_t spin = 0; spin < model.spin_k.size(); ++spin)
        for (const auto& k : model.spin_k[spin])
        {
            state.parent_phi[spin].push_back(k.parent_phi);
            state.parent_capture.push_back(k.parent_capture_min);
        }
    return state;
}

std::vector<double> crpa_tracking_energies(const MeanField& meanfield)
{
    std::vector<double> energies;
    for (int spin = 0; spin < meanfield.get_n_spins(); ++spin)
        for (int k = 0; k < meanfield.get_n_kpoints(); ++k)
            for (int band = 0; band < meanfield.get_n_bands(); ++band)
                energies.push_back(meanfield.get_eigenvals()[spin](k, band));
    return energies;
}

void write_crpa_tracking_checkpoint(const std::string& filename, int iteration,
                                    const std::string& definition,
                                    const std::vector<double>& checkpoint_energies,
                                    const CrpaTrackingState& state)
{
    if (state.parent_phi.empty() || state.parent_phi.front().empty() ||
        state.parent_phi.front().front().nc <= 0)
        throw LIBRPA_INVALID_ARGUMENT("cRPA tracking checkpoint has no preceding parent frame");
    const int n_spins = state.parent_phi.size(), nk = state.parent_phi.front().size();
    const int n_parent = state.parent_phi.front().front().nc;
    if (state.parent_capture.size() != static_cast<std::size_t>(n_spins) * nk)
        throw LIBRPA_INVALID_ARGUMENT("cRPA tracking checkpoint capture dimensions differ");
    const std::string temporary = filename + ".tmp";
    std::ofstream out(temporary);
    out.exceptions(std::ios::badbit | std::ios::failbit);
    out << std::setprecision(17) << "LIBRPA_CRPA_TRACKING_1\n"
        << iteration << ' ' << n_spins << ' ' << state.n_spinor << ' ' << nk << ' ' << state.n_aos
        << ' ' << n_parent << '\n'
        << std::quoted(definition) << '\n';
    out << checkpoint_energies.size() << '\n';
    for (double energy : checkpoint_energies) out << energy << '\n';
    for (double capture : state.parent_capture) out << capture << '\n';
    for (const auto& spin : state.parent_phi)
    {
        if (spin.size() != static_cast<std::size_t>(nk))
            throw LIBRPA_INVALID_ARGUMENT("cRPA tracking checkpoint k dimensions differ");
        for (const auto& phi : spin)
        {
            if (phi.nr != state.n_aos * state.n_spinor || phi.nc != n_parent)
                throw LIBRPA_INVALID_ARGUMENT("cRPA tracking checkpoint orbital dimensions differ");
            for (int i = 0; i < phi.size; ++i)
                out << phi.c[i].real() << '\n' << phi.c[i].imag() << '\n';
        }
    }
    out.close();
    if (std::rename(temporary.c_str(), filename.c_str()) != 0)
        throw LIBRPA_RUNTIME_ERROR("cannot replace cRPA tracking checkpoint " + filename);
}

CrpaTrackingState read_crpa_tracking_checkpoint(const std::string& filename, int iteration,
                                                const std::string& definition,
                                                const MeanField& meanfield)
{
    std::ifstream in(filename);
    if (!in) throw LIBRPA_RUNTIME_ERROR("missing cRPA tracking checkpoint " + filename);
    in.exceptions(std::ios::badbit | std::ios::failbit);
    std::string magic, saved_definition;
    int saved_iteration, spins, spinor, nk, nao, parent;
    in >> magic >> saved_iteration >> spins >> spinor >> nk >> nao >> parent >>
        std::quoted(saved_definition);
    if (magic != "LIBRPA_CRPA_TRACKING_1" || saved_iteration != iteration ||
        spins != meanfield.get_n_spins() || spinor != meanfield.get_n_spinor() ||
        nk != meanfield.get_n_kpoints() || nao != meanfield.get_n_aos() || parent <= 0 ||
        parent > nao * spinor || saved_definition != definition)
        throw LIBRPA_INVALID_ARGUMENT("cRPA tracking checkpoint differs from the requested model");
    std::size_t count;
    in >> count;
    const auto energies = crpa_tracking_energies(meanfield);
    if (count != energies.size())
        throw LIBRPA_INVALID_ARGUMENT("cRPA tracking checkpoint state dimensions differ");
    for (const double expected : energies)
    {
        double value;
        in >> value;
        if (!std::isfinite(value) || value != expected)
            throw LIBRPA_INVALID_ARGUMENT(
                "cRPA tracking checkpoint belongs to a different QSGW state");
    }
    CrpaTrackingState state;
    state.n_aos = nao;
    state.n_spinor = spinor;
    state.parent_capture.resize(static_cast<std::size_t>(spins) * nk);
    for (double& capture : state.parent_capture) in >> capture;
    state.parent_phi.resize(spins);
    for (auto& spin : state.parent_phi)
        for (int k = 0; k < nk; ++k)
        {
            ComplexMatrix phi(nao * spinor, parent);
            for (int i = 0; i < phi.size; ++i)
            {
                double real, imaginary;
                in >> real >> imaginary;
                if (!std::isfinite(real) || !std::isfinite(imaginary))
                    throw LIBRPA_INVALID_ARGUMENT("nonfinite cRPA tracking checkpoint");
                phi.c[i] = {real, imaginary};
            }
            spin.push_back(std::move(phi));
        }
    return state;
}
}  // namespace librpa_int
