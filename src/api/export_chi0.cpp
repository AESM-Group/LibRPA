#include "dataset_helper.h"
#include "export_chi0_symmetry.h"
#include "../core/chi0.h"
#include "../core/epsilon.h"
#include "../io/global_io.h"
#include "../utils/profiler.h"

#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace librpa_int
{

namespace
{

//! Index of the grid node with the smallest |omega|. The node is chosen by index,
//! so the export never depends on the ordering of the frequency grid.
int find_lowest_frequency_index(const TFGrids &chi0_tfg)
{
    const std::vector<double> chi0_freq_nodes = chi0_tfg.get_freq_nodes();
    if (chi0_freq_nodes.empty())
        throw LIBRPA_RUNTIME_ERROR("chi0 export requires a non-empty frequency grid");
    int lowest_index = 0;
    for (std::size_t ifreq = 1; ifreq < chi0_freq_nodes.size(); ++ifreq)
    {
        if (std::abs(chi0_freq_nodes[ifreq]) < std::abs(chi0_freq_nodes[lowest_index]))
            lowest_index = static_cast<int>(ifreq);
    }
    return lowest_index;
}

//! Write one real-space atom-pair block chi0_{I,J}(R) as a MatrixMarket file.
void write_chi0_export_rspace_block(const std::filesystem::path &chi0_block_path,
                                    const Matz &chi0_rspace_block,
                                    const Vector3_Order<int> &lattice_vector)
{
    std::ofstream chi0_block_file(chi0_block_path);
    chi0_block_file << "%%MatrixMarket matrix coordinate complex general\n";
    chi0_block_file << "%\n";
    chi0_block_file << "% bare chi0; R = ( " << lattice_vector.x << ' ' << lattice_vector.y
                    << ' ' << lattice_vector.z << " )\n";
    chi0_block_file << chi0_rspace_block.nr() << ' ' << chi0_rspace_block.nc() << ' '
                    << chi0_rspace_block.nr() * chi0_rspace_block.nc() << '\n';
    chi0_block_file << std::scientific << std::setprecision(17);
    for (std::size_t row = 0; row < chi0_rspace_block.nr(); ++row)
    {
        for (std::size_t col = 0; col < chi0_rspace_block.nc(); ++col)
        {
            chi0_block_file << row + 1 << ' ' << col + 1 << ' '
                            << chi0_rspace_block(row, col).real() << ' '
                            << chi0_rspace_block(row, col).imag() << '\n';
        }
    }
    chi0_block_file.close();
    if (!chi0_block_file)
        throw std::runtime_error("failed writing chi0 real-space matrix " +
                                 chi0_block_path.string());
}

//! Write chi0(R, i*omega) of the response object for the grid node chi0_grid_index.
void write_chi0_export_response(const Chi0 &chi0_export_object, const std::string &directory,
                                int chi0_grid_index, const std::string &chi0_source_tag)
{
    const double omega = chi0_export_object.tfg.get_freq_nodes()[chi0_grid_index];
    if (chi0_export_object.qpoint_view().restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL)
    {
        export_chi0_symmetry(chi0_export_object, directory, chi0_grid_index, chi0_source_tag);
        return;
    }

    const auto &chi0_aux_basis = chi0_export_object.atbasis_abf;
    const int n_aux = chi0_aux_basis.nb_total;
    auto chi0_export_pbc = chi0_export_object.pbc;
    chi0_export_pbc.map_irk_ks.clear();
    const auto &chi0_active_qpoints = chi0_export_object.active_qpoints();

    // Without rotation metadata only a full mesh or time-reversal pairs (q, -q) can be
    // restored. Do not apply conjugation to arbitrary space-group stars.
    for (const auto &q_full : chi0_export_object.pbc.klist_full)
    {
        auto q_representative = chi0_active_qpoints.end();
        for (auto q_iter = chi0_active_qpoints.begin(); q_iter != chi0_active_qpoints.end(); ++q_iter)
        {
            const auto chi0_q_difference = q_full - *q_iter;
            if (std::abs(chi0_q_difference.x) + std::abs(chi0_q_difference.y) +
                std::abs(chi0_q_difference.z) < 1.e-8)
            {
                q_representative = q_iter;
                break;
            }
        }
        if (q_representative == chi0_active_qpoints.end())
        {
            for (auto q_iter = chi0_active_qpoints.begin(); q_iter != chi0_active_qpoints.end(); ++q_iter)
            {
                // q_full = -q_rep modulo a reciprocal lattice vector
                const auto q_sum_frac = (q_full + *q_iter) * chi0_export_pbc.latvec.Transpose();
                const double q_sum_offset = std::abs(q_sum_frac.x - std::round(q_sum_frac.x)) +
                                            std::abs(q_sum_frac.y - std::round(q_sum_frac.y)) +
                                            std::abs(q_sum_frac.z - std::round(q_sum_frac.z));
                if (q_sum_offset < 1.e-8)
                {
                    q_representative = q_iter;
                    break;
                }
            }
        }
        if (q_representative == chi0_active_qpoints.end())
            throw std::runtime_error(
                "chi0 export requires full/inverse q coverage; disable general chi0 symmetry");
        chi0_export_pbc.map_irk_ks[*q_representative].push_back(q_full);
    }

    // Chi0 uses the same TFGrids nodes as map keys, including on distributed ranks.
    const auto &chi0_freq_q_map = chi0_export_object.get_chi0_q();
    const auto chi0_freq_entry = chi0_freq_q_map.find(omega);
    const int local_has_frequency = chi0_freq_entry != chi0_freq_q_map.end();
    int any_rank_has_frequency = 0;
    chi0_export_object.comm_h.allreduce(&local_has_frequency, &any_rank_has_frequency, 1, MPI_MAX);
    if (!any_rank_has_frequency)
        throw LIBRPA_RUNTIME_ERROR("chi0 export: no rank holds chi0 at frequency grid index " +
                                   std::to_string(chi0_grid_index));

    atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old chi0_q_atom_blocks;
    for (const auto &q_rep : chi0_active_qpoints)
    {
        // Assemble the full n_aux x n_aux chi0(q) from the distributed atom-pair blocks.
        Matz chi0_q_dense(n_aux, n_aux, MAJOR::ROW);
        std::vector<int> chi0_q_entry_counts(n_aux * n_aux, 0);
        if (chi0_freq_entry != chi0_freq_q_map.end())
        {
            const auto chi0_q_entry = chi0_freq_entry->second.find(q_rep);
            if (chi0_q_entry != chi0_freq_entry->second.end())
            {
                for (const auto &[atom_i, chi0_atom_columns] : chi0_q_entry->second)
                {
                    for (const auto &[atom_j, chi0_atom_block] : chi0_atom_columns)
                    {
                        for (int row = 0; row < chi0_atom_block.nr; ++row)
                        {
                            for (int col = 0; col < chi0_atom_block.nc; ++col)
                            {
                                const int mu = chi0_aux_basis.get_global_index(atom_i, row);
                                const int nu = chi0_aux_basis.get_global_index(atom_j, col);
                                chi0_q_dense(mu, nu) = chi0_atom_block(row, col);
                                chi0_q_entry_counts[mu * n_aux + nu] = 1;
                                // chi0(q) is Hermitian: chi0_{nu,mu}(q) = conj(chi0_{mu,nu}(q)).
                                // Diagonal atom blocks are stored complete and are written as stored.
                                if (atom_i != atom_j)
                                {
                                    chi0_q_dense(nu, mu) = std::conj(chi0_atom_block(row, col));
                                    chi0_q_entry_counts[nu * n_aux + mu] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, chi0_q_dense.ptr(), n_aux * n_aux, MPI_C_DOUBLE_COMPLEX,
                      MPI_SUM, chi0_export_object.comm_h.comm);
        MPI_Allreduce(MPI_IN_PLACE, chi0_q_entry_counts.data(), n_aux * n_aux, MPI_INT, MPI_SUM,
                      chi0_export_object.comm_h.comm);
        // Every rank holds the same counts, so every rank throws together here.
        for (int entry = 0; entry < n_aux * n_aux; ++entry)
        {
            if (!chi0_q_entry_counts[entry])
                throw std::runtime_error("missing chi0 atom block during real-space export");
            // Average entries that were provided by more than one rank or block.
            chi0_q_dense.ptr()[entry] /= static_cast<double>(chi0_q_entry_counts[entry]);
        }
        if (chi0_export_object.comm_h.is_root())
        {
            for (int atom_i = 0; atom_i < chi0_aux_basis.n_atoms; ++atom_i)
            {
                for (int atom_j = 0; atom_j < chi0_aux_basis.n_atoms; ++atom_j)
                {
                    Matz chi0_q_atom_block(chi0_aux_basis[atom_i], chi0_aux_basis[atom_j], MAJOR::ROW);
                    for (std::size_t row = 0; row < chi0_q_atom_block.nr(); ++row)
                    {
                        for (std::size_t col = 0; col < chi0_q_atom_block.nc(); ++col)
                        {
                            chi0_q_atom_block(row, col) =
                                chi0_q_dense(chi0_aux_basis.get_global_index(atom_i, row),
                                             chi0_aux_basis.get_global_index(atom_j, col));
                        }
                    }
                    chi0_q_atom_blocks[atom_i][atom_j][q_rep] = std::move(chi0_q_atom_block);
                }
            }
        }
    }

    // The LibRPA linear spatial Fourier transform applies equally to bare chi0:
    //   chi0(R, i*omega) = (1/N_q) sum_q exp(-2*pi*i q.R) chi0(q, i*omega).
    // These are unscreened response matrices, not epsilon, W or Wc.
    const auto chi0_rspace_atom_blocks =
        FT_Wc_q2R(chi0_export_object.comm_h, chi0_aux_basis, chi0_export_object.symmetry_context,
                  chi0_q_atom_blocks, chi0_export_object.tfg, chi0_export_pbc,
                  chi0_export_pbc.Rlist, true, "", false);

    // Only rank 0 writes. Its I/O error is shared with all ranks below, so a failed
    // write stops every rank instead of leaving the others waiting in a collective.
    std::string chi0_export_write_error;
    if (chi0_export_object.comm_h.is_root())
    {
        try
        {
            std::filesystem::create_directories(directory);
            for (const auto &[atom_i, chi0_rspace_columns] : chi0_rspace_atom_blocks)
            {
                for (const auto &[atom_j, chi0_rspace_cells] : chi0_rspace_columns)
                {
                    for (const auto &[lattice_vector, chi0_rspace_block] : chi0_rspace_cells)
                    {
                        // ifreq_0 is the single exported slot; the grid index of the node is
                        // recorded in chi0_rf.info.
                        const std::string chi0_block_name =
                            "Chi0_Mu_" + std::to_string(atom_i) + "_Nu_" + std::to_string(atom_j) +
                            "_iR_" + std::to_string(chi0_export_pbc.get_R_index(lattice_vector)) +
                            "_ifreq_0.mtx";
                        write_chi0_export_rspace_block(
                            std::filesystem::path(directory) / chi0_block_name, chi0_rspace_block,
                            lattice_vector);
                    }
                }
            }
            // Line 2 keeps the layout read by existing consumers; line 3 adds the provenance.
            std::ofstream chi0_info_file(std::filesystem::path(directory) / "chi0_rf.info");
            chi0_info_file << "LIBRPA_CHI0_RF 1\n";
            chi0_info_file << std::setprecision(17) << omega << ' ' << n_aux << ' '
                           << chi0_export_pbc.period.x << ' ' << chi0_export_pbc.period.y << ' '
                           << chi0_export_pbc.period.z << '\n';
            chi0_info_file << "grid_index " << chi0_grid_index << " n_grid_nodes "
                           << chi0_export_object.tfg.get_n_grids() << " source " << chi0_source_tag
                           << '\n';
            chi0_info_file.close();
            if (!chi0_info_file) throw std::runtime_error("failed writing chi0_rf.info");
        }
        catch (const std::exception &chi0_export_exception)
        {
            chi0_export_write_error = std::string("chi0 export: ") + chi0_export_exception.what();
        }
    }
    check_chi0_collective_error(chi0_export_object.comm_h, chi0_export_write_error);
    global::lib_printf_root(
        "Exported bare chi0(R,iw), grid index %d, omega = %.16e Ha (%s), to %s\n",
        chi0_grid_index, omega, chi0_source_tag.c_str(), directory.c_str());
}

//! Build a response object in the full auxiliary basis on the given grid.
std::unique_ptr<Chi0> build_chi0_export_object(Dataset &ds, const LibrpaOptions &opts,
                                               LibrpaParallelRouting routing,
                                               LibrpaMethod chi0_method,
                                               const TFGrids &chi0_export_tfg,
                                               const std::string &chi0_source_tag)
{
    const bool qstar_only = ds.p_chi0->qstar_only_symmetry;
    auto chi0_export_object =
        create_ds_chi0(ds, opts, qstar_only, ds.basis_aux, chi0_export_tfg);
    // Full auxiliary basis: C in the full basis, no shrink projector.
    std::map<Vector3_Order<double>, ComplexMatrix> chi0_export_empty_sinvS;
    global::lib_printf_root(
        "Bare chi0 export (%s): building a separate response in the full auxiliary basis; "
        "the grid printed next belongs to this export build\n",
        chi0_source_tag.c_str());
    global::profiler.start("chi0_export_build", "Build response for bare chi0 export");
    chi0_export_object->build(routing, chi0_method, ds.cs_data, ds.atpairs_local, ds.basis_aux,
                              chi0_export_empty_sinvS, ds.blacs_h);
    global::profiler.stop("chi0_export_build");
    return chi0_export_object;
}

} // namespace

void prepare_ds_chi0_export(Dataset &ds, const LibrpaOptions &opts)
{
    const bool export_rf = opts.output_chi0_rf == LIBRPA_SWITCH_ON;
    const bool export_static = opts.output_chi0_static == LIBRPA_SWITCH_ON;
    if (!export_rf && !export_static) return;

    if (export_static &&
        (ds.tfg.get_grid_type() != LIBRPA_TFGRID_MINIMAX || !ds.tfg.has_time_grids()))
        throw LIBRPA_RUNTIME_ERROR(
            "output_chi0_static requires tfgrids_type = minimax (it integrates over the "
            "minimax time quadrature)");

}

void export_ds_chi0(Dataset &ds, const LibrpaOptions &opts, LibrpaParallelRouting routing,
                    LibrpaMethod chi0_method)
{
    const bool export_rf = opts.output_chi0_rf == LIBRPA_SWITCH_ON;
    const bool export_static = opts.output_chi0_static == LIBRPA_SWITCH_ON;
    if (!export_rf && !export_static) return;
    global::profiler.start("chi0_export", "Export bare chi0");

    if (export_rf)
    {
        const std::string chi0_source_tag = "lowest_grid_node";
        if (opts.use_shrink_abfs == LIBRPA_SWITCH_ON)
        {
            // The main chi0 is in the shrunk basis; rebuild it in the full basis on the
            // same grid so that the exported matrix has the documented dimension.
            const auto chi0_lowest_object = build_chi0_export_object(
                ds, opts, routing, chi0_method, ds.tfg, chi0_source_tag);
            const int chi0_lowest_index = find_lowest_frequency_index(chi0_lowest_object->tfg);
            write_chi0_export_response(*chi0_lowest_object, std::string(opts.output_dir) + "/chi0_lowest",
                                       chi0_lowest_index, chi0_source_tag);
        }
        else
        {
            const Chi0 &chi0_main_object = *ds.p_chi0;
            const int chi0_lowest_index = find_lowest_frequency_index(chi0_main_object.tfg);
            write_chi0_export_response(chi0_main_object, std::string(opts.output_dir) + "/chi0_lowest",
                                       chi0_lowest_index, chi0_source_tag);
        }
    }

    if (export_static)
    {
        // A dedicated response build leaves the GW quadrature and chi0 intact.
        // omega = 0 is evaluated by imaginary-time integration,
        //   chi0(q, i*0) = 2 * integral_0^infty chi0(q, i*tau) d tau,
        // not by renaming the first minimax frequency or by extrapolation.
        const std::string chi0_source_tag = "static_tau_integral";
        TFGrids chi0_static_tfg;
        chi0_static_tfg.generate_static_export(ds.tfg);
        // The static grid carries time nodes, so the space-time method is requested explicitly.
        const auto chi0_static_object = build_chi0_export_object(
            ds, opts, routing, LIBRPA_METHOD_SPACE_TIME, chi0_static_tfg, chi0_source_tag);
        // generate_static_export puts omega = 0 at node 0.
        const int chi0_static_index = 0;
        write_chi0_export_response(*chi0_static_object, std::string(opts.output_dir) + "/chi0_static",
                                   chi0_static_index, chi0_source_tag);
    }

    global::profiler.stop("chi0_export");
}

} // namespace librpa_int
