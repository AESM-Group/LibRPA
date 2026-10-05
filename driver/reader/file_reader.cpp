// Public API headers
#include "file_reader.hpp"

// Internal headers
#include "../../src/api/instance_manager.h"
#include "../../src/utils/error.h"
#include "read_data.h"
#include "reader_structure.h"
#include "reader_coulomb.h"
#include "reader_lri.h"
#include "reader_eigenvec.h"

// Standard headers
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace librpa
{
namespace
{

std::string normalize_input_dir(const std::string &path)
{
    if (path.empty())
    {
        throw LIBRPA_RUNTIME_ERROR("Input directory must not be empty");
    }
    return path.back() == '/' ? path : path + '/';
}

void require_input_file(const fs::path &path)
{
    if (!fs::is_regular_file(path))
    {
        throw LIBRPA_RUNTIME_ERROR("Required LibRPA input file is missing: " + path.string());
    }
}

template <typename Function>
void run_reader(const char *name, Function &&function)
{
    try
    {
        function();
    }
    catch (const std::exception &error)
    {
        throw LIBRPA_RUNTIME_ERROR(std::string("LibRPA file reader '") + name
                                   + "' failed: " + error.what());
    }
}

void read_scf_meanfield(reader::ReaderContext &ctx, const fs::path &path)
{
    librpa_int::MeanField meanfield;
    reader::read_scf_occ_eigenvalues(path.string(), meanfield, false);

    ctx.state.n_spins = meanfield.get_n_spins();
    ctx.state.n_kpoints = meanfield.get_n_kpoints();
    ctx.state.n_ibz_kpoints = ctx.state.n_kpoints;
    ctx.state.n_states = meanfield.get_n_states();
    ctx.state.n_basis_wfc = meanfield.get_n_aos();
    ctx.state.n_basis_ao = meanfield.get_n_aos();
    ctx.state.n_spinor = meanfield.get_n_spinor();
    ctx.state.iks_eigvec_this.resize(static_cast<std::size_t>(ctx.state.n_kpoints));
    for (int ik = 0; ik != ctx.state.n_kpoints; ++ik)
    {
        ctx.state.iks_eigvec_this[static_cast<std::size_t>(ik)] = ik;
    }

    const std::size_t block_size = static_cast<std::size_t>(ctx.state.n_kpoints)
                                   * static_cast<std::size_t>(ctx.state.n_states);
    std::vector<double> eigenvalues(static_cast<std::size_t>(ctx.state.n_spins) * block_size);
    std::vector<double> occupations(eigenvalues.size());
    for (int is = 0; is != ctx.state.n_spins; ++is)
    {
        const auto &eigenvalues_spin
            = meanfield.get_eigenvals()[static_cast<std::size_t>(is)];
        const auto &occupations_spin
            = meanfield.get_weight()[static_cast<std::size_t>(is)];
        for (std::size_t index = 0; index != block_size; ++index)
        {
            eigenvalues[static_cast<std::size_t>(is) * block_size + index]
                = eigenvalues_spin.c[index];
            // The file reader normalizes weights, while Handler performs this
            // normalization when setting the dataset. Restore the file values.
            occupations[static_cast<std::size_t>(is) * block_size + index]
                = occupations_spin.c[index] * ctx.state.n_kpoints;
        }
    }

    ctx.h.set_scf_dimension(ctx.state.n_spins, ctx.state.n_kpoints,
                                ctx.state.n_states, ctx.state.n_basis_ao,
                                ctx.state.n_spinor);
    ctx.h.set_wg_ekb_efermi(ctx.state.n_spins, ctx.state.n_kpoints,
                               ctx.state.n_states, occupations.data(),
                               eigenvalues.data(), meanfield.get_efermi());
}

} // namespace

std::shared_ptr<librpa_int::Dataset> read_dataset_from_files(
    MPI_Comm comm, const FileReaderOptions &opts)
{
    const std::string input_dir = normalize_input_dir(opts.input_dir);
    const fs::path input_path(input_dir);
    require_input_file(input_path / opts.fn_eigocc_scf);
    require_input_file(input_path / opts.fn_stru);

    Handler handler(comm);
    reader::ReaderState state;
    reader::ReaderParameters params;
    params.prefix_lri_coeff = opts.prefix_lri_coeff;
    params.prefix_lri_coeff_shrink = opts.prefix_lri_coeff_shrink;
    params.prefix_eigvecs_scf = opts.prefix_eigvecs_scf;
    Options runtime_options;
    runtime_options.parallel_routing = LIBRPA_ROUTING_LIBRI;
    runtime_options.vq_threshold = opts.coulomb_threshold;
    runtime_options.use_kpara_scf_eigvec = LIBRPA_SWITCH_OFF;
    reader::ReaderContext ctx{handler, state, params, runtime_options,
                              librpa_int::MpiCommHandler(comm, true)};

    read_scf_meanfield(ctx, input_path / opts.fn_eigocc_scf);
    reader::reader_structure(ctx, (input_path / opts.fn_stru).string());
    if (fs::is_regular_file(input_path / opts.fn_bz_sampling))
    {
        reader::read_bz_sampling(ctx, (input_path / opts.fn_bz_sampling).string());
    }
    else
    {
        reader::read_bz_sampling_from_stru(ctx, (input_path / opts.fn_stru).string());
    }
    reader::read_basis_wfc_aux(ctx, input_dir,
                       opts.fn_basis, opts.fn_basis_wfc, opts.fn_basis_aux);

    auto pds = librpa_int::api::get_dataset_instance(ctx.h.get_c_handler());
    const auto aux_sizes = pds->basis_aux.get_atom_nbs();
    if (aux_sizes.size() != pds->atoms.size())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "LibRPA auxiliary-basis atom partition is inconsistent: "
            + std::to_string(aux_sizes.size()) + " basis blocks for "
            + std::to_string(pds->atoms.size()) + " atoms");
    }

    if (opts.read_ri)
    {
        const auto all_pairs
            = librpa_int::generate_atom_pair_from_nat(ctx.state.n_atoms, false);
        run_reader(opts.prefix_coul_cut.c_str(), [&]
        {
            reader::read_Vq_row(ctx, input_dir, opts.prefix_coul_cut,
                        opts.coulomb_threshold, all_pairs, true,
                        -1, false);
        });
        if (opts.read_full_coulomb)
        {
            run_reader("coulomb_mat", [&]
            {
                reader::read_Vq_row(ctx, input_dir, "coulomb_mat",
                            opts.coulomb_threshold, all_pairs, false, -1, false);
            });
        }
        run_reader(opts.prefix_lri_coeff.c_str(), [&]
        {
            // Analytic head/wing construction sums partial C(q) across ranks.
            // Unique ownership prevents replicated coefficients being summed
            // once per MPI rank. Keep replicated loading as the default.
            if (opts.distribute_lri)
                reader::read_Cs_evenly_distribute(ctx, input_dir, opts.cs_threshold,
                    pds->comm_h.myid, pds->comm_h.nprocs, opts.prefix_lri_coeff, -1);
            else
                reader::read_Cs(ctx, input_dir, opts.cs_threshold, all_pairs,
                        opts.prefix_lri_coeff, -1);
        });
    }

    run_reader("coarse KS eigenvectors", [&]
    {
        if (reader::read_eigenvector(ctx, input_dir, pds->mf, false) != 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Failed to read coarse-grid KS eigenvectors");
        }
    });
    require_input_file(input_path / opts.fn_velocity);
    run_reader(opts.fn_velocity.c_str(), [&]
    {
        reader::read_velocity(ctx, (input_path / opts.fn_velocity).string(), pds->mf,
                      pds->velocity_matrix);
    });
    if (opts.read_band_data)
    {
        require_input_file(input_path / opts.fn_band_kpath_info);
        run_reader(opts.fn_band_kpath_info.c_str(), [&]
        {
            reader::read_band_kpath_info(ctx, (input_path / opts.fn_band_kpath_info).string());
        });
        run_reader("band meanfield", [&]
        {
            reader::read_band_meanfield_data(ctx, input_dir);
        });
    }
    else
    {
        pds->kfrac_band_list = pds->pbc.kfrac_list;
        pds->mf_band = pds->mf;
    }

    return pds;
}

} // namespace librpa
