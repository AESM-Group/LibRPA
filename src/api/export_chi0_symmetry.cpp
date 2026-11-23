#include "export_chi0_symmetry.h"
#include "../core/chi0.h"
#include "../io/global_io.h"
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>

namespace librpa_int {
namespace {
bool same_export_q(const Vector3_Order<double> &first, const Vector3_Order<double> &second) {
    const auto q_difference = first - second;
    return std::abs(q_difference.x-std::round(q_difference.x)) < 1.e-8
        && std::abs(q_difference.y-std::round(q_difference.y)) < 1.e-8
        && std::abs(q_difference.z-std::round(q_difference.z)) < 1.e-8;
}
const SymmetryKStarMember &export_star_member(const SymmetryKStar &star, const Vector3_Order<double> &q) {
    for (const auto &member : star.members) if (same_export_q(member.k_bz, q)) return member;
    throw std::runtime_error("chi0 export cannot locate star member");
}
int export_q_id(const Chi0 &chi, const Vector3_Order<double> &q_fractional) {
    for (std::size_t iq=0; iq<chi.pbc.klist_full.size(); ++iq)
        if (same_export_q(chi.pbc.klist_full[iq]*chi.pbc.latvec.Transpose(), q_fractional)) return iq+1;
    throw std::runtime_error("chi0 export target is outside the full q grid");
}
void write_export_matrix(const std::filesystem::path &path, const char *magic,
                         const ComplexMatrix &rpa_export_operator, int target_id, int representative_id) {
    std::ofstream output(path);
    output << magic << " 1\n" << rpa_export_operator.nr << ' ' << target_id << ' ' << representative_id
           << '\n' << std::scientific << std::setprecision(17);
    for (int mu=0; mu<rpa_export_operator.nr; ++mu) for (int nu=0; nu<rpa_export_operator.nc; ++nu)
        output << mu+1 << ' ' << nu+1 << ' ' << rpa_export_operator(mu,nu).real()
               << ' ' << rpa_export_operator(mu,nu).imag() << '\n';
    output.close();
    if (!output) throw std::runtime_error("cannot write chi0 symmetry matrix: " + path.string());
}

//! Delta(q_target) maps the representative chi0(q_rep) to chi0(q_target):
//!   chi0(q_target) = Delta chi0(q_rep) Delta^dagger   (conjugated chi0 if `conjugate`).
ComplexMatrix build_export_delta(const ComplexMatrix &rpa_target_transform,
                                 const ComplexMatrix &rpa_source_transform,
                                 bool target_time_reversal, bool source_time_reversal, int naux) {
    const bool conjugate = source_time_reversal != target_time_reversal;
    ComplexMatrix rpa_delta(naux, naux);
    rpa_delta.zero_out();
    // LibRPA applies T* X* T^T for time reversal, so its effective left operator is
    // conj(T). Representatives need not be the canonical star source. Compose the
    // target rotation with the inverse source rotation; antiunitary parity determines
    // whether the source adjoint and response are conjugated.
    for (int mu = 0; mu < naux; ++mu) {
        for (int k = 0; k < naux; ++k) {
            if (rpa_target_transform(mu, k) == std::complex<double>{}) continue;
            const auto rpa_target_factor = target_time_reversal
                ? std::conj(rpa_target_transform(mu, k)) : rpa_target_transform(mu, k);
            for (int nu = 0; nu < naux; ++nu) {
                const auto rpa_source_factor = source_time_reversal
                    ? std::conj(rpa_source_transform(nu, k)) : rpa_source_transform(nu, k);
                rpa_delta(mu, nu) += rpa_target_factor *
                    (conjugate ? rpa_source_factor : std::conj(rpa_source_factor));
            }
        }
    }
    return rpa_delta;
}
}

void export_chi0_symmetry(const Chi0 &chi, const std::string &directory, int chi0_grid_index,
                          const std::string &chi0_source_tag) {
    const auto &rpa_basis = chi.atbasis_abf;
    const int naux = rpa_basis.nb_total;
    const double omega = chi.tfg.get_freq_nodes()[chi0_grid_index];
    const auto &rpa_context = chi.symmetry_context;
    std::map<atom_t, std::size_t> rpa_atom_naux;
    for (int atom=0; atom<rpa_basis.n_atoms; ++atom) rpa_atom_naux[atom]=rpa_basis[atom];
    const auto rpa_aux_layouts = rpa_basis.build_species_basis_layouts(rpa_context.atom_to_type);
    if (!rpa_context.available || !symmetry_species_layouts_match_atom_counts(
            rpa_aux_layouts, rpa_context.atom_to_type, rpa_atom_naux))
        throw std::runtime_error("chi0 symmetry export requires complete auxiliary basis layouts");

    const auto &rpa_chi0_freq_q_map = chi.get_chi0_q();
    const auto rpa_frequency = rpa_chi0_freq_q_map.find(omega);
    const int rpa_local_has_frequency = rpa_frequency != rpa_chi0_freq_q_map.end();
    int rpa_any_rank_has_frequency = 0;
    chi.comm_h.allreduce(&rpa_local_has_frequency, &rpa_any_rank_has_frequency, 1, MPI_MAX);
    if (!rpa_any_rank_has_frequency)
        throw std::runtime_error("chi0 symmetry export: no rank holds chi0 at frequency grid index " +
                                 std::to_string(chi0_grid_index));

    // Only rank 0 writes. Its errors are collected and raised on every rank after the
    // loop, so the other ranks never wait in an Allreduce that rank 0 has left.
    std::string rpa_export_write_error;
    const std::filesystem::path export_dir(directory);
    std::ofstream rpa_manifest;
    if (chi.comm_h.is_root()) {
        try {
            std::filesystem::create_directories(export_dir);
            rpa_manifest.open(export_dir/"rpa_symmetry_chi0.dat");
            rpa_manifest << "LIBRPA_SYMMETRY_ALL 1\n" << naux << ' ' << chi.pbc.klist_full.size()
                         << "\nBARE_CHI0 D_CHI0_DAGGER\n" << std::scientific << std::setprecision(17);
            if (!rpa_manifest) throw std::runtime_error("cannot write rpa_symmetry_chi0.dat");
        } catch (const std::exception &rpa_export_exception) {
            rpa_export_write_error = std::string("chi0 symmetry export: ") + rpa_export_exception.what();
        }
    }

    std::set<int> rpa_exported_targets;
    for (const auto &q_rep : chi.active_qpoints()) {
        // Assemble the full naux x naux representative chi0(q_rep) from the atom-pair blocks.
        ComplexMatrix rpa_chi0_representative(naux,naux);
        rpa_chi0_representative.zero_out();
        std::vector<int> rpa_chi0_counts(naux*naux,0);
        if (rpa_frequency != rpa_chi0_freq_q_map.end()) {
            const auto rpa_q_entry = rpa_frequency->second.find(q_rep);
            if (rpa_q_entry != rpa_frequency->second.end()) {
                for (const auto &[atom_i,rpa_columns] : rpa_q_entry->second) {
                    for (const auto &[atom_j,rpa_atom_response] : rpa_columns) {
                        for (int row=0; row<rpa_atom_response.nr; ++row) {
                            for (int col=0; col<rpa_atom_response.nc; ++col) {
                                const int mu=rpa_basis.get_global_index(atom_i,row);
                                const int nu=rpa_basis.get_global_index(atom_j,col);
                                rpa_chi0_representative(mu,nu)=rpa_atom_response(row,col);
                                rpa_chi0_counts[mu*naux+nu]=1;
                                // chi0(q) is Hermitian: chi0_{nu,mu} = conj(chi0_{mu,nu}).
                                if (atom_i!=atom_j) {
                                    rpa_chi0_representative(nu,mu)=std::conj(rpa_atom_response(row,col));
                                    rpa_chi0_counts[nu*naux+mu]=1;
                                }
                            }
                        }
                    }
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,rpa_chi0_representative.c,naux*naux,MPI_C_DOUBLE_COMPLEX,MPI_SUM,chi.comm_h.comm);
        MPI_Allreduce(MPI_IN_PLACE,rpa_chi0_counts.data(),naux*naux,MPI_INT,MPI_SUM,chi.comm_h.comm);
        // Every rank holds the same counts, so every rank throws together here.
        for (int entry=0; entry<naux*naux; ++entry) {
            if (!rpa_chi0_counts[entry]) throw std::runtime_error("missing representative chi0 entries during export");
            rpa_chi0_representative.c[entry]/=rpa_chi0_counts[entry];
        }
        if (!chi.comm_h.is_root() || !rpa_export_write_error.empty()) continue;

        try {
            const Vector3_Order<double> q_rep_frac=q_rep*chi.pbc.latvec.Transpose();
            const int representative_id=export_q_id(chi,q_rep_frac);
            write_export_matrix(export_dir/("chi0_q_"+std::to_string(representative_id)+".dat"),
                                "LIBRPA_CHI0_Q",rpa_chi0_representative,representative_id,representative_id);
            const auto &rpa_star=find_symmetry_kstar_for_kpoint(rpa_context.kstars,q_rep_frac,"chi0 export");
            const auto &rpa_source=export_star_member(rpa_star,q_rep_frac);
            const auto rpa_source_transform=build_symmetry_kspace_operator_transform_matrix(
                rpa_context,rpa_aux_layouts,rpa_source,rpa_atom_naux,rpa_star.k_ibz,rpa_source.time_reversal,&q_rep_frac);
            for (const auto &q_target : chi.qpoint_view().members.at(q_rep)) {
                const Vector3_Order<double> q_target_frac=q_target*chi.pbc.latvec.Transpose();
                const int target_id=export_q_id(chi,q_target_frac);
                if (!rpa_exported_targets.insert(target_id).second) throw std::runtime_error("duplicate chi0 symmetry target");
                const auto &rpa_target=export_star_member(rpa_star,q_target_frac);
                const auto rpa_target_transform=build_symmetry_kspace_operator_transform_matrix(
                    rpa_context,rpa_aux_layouts,rpa_target,rpa_atom_naux,rpa_star.k_ibz,rpa_target.time_reversal,&q_target_frac);
                const bool conjugate = rpa_source.time_reversal != rpa_target.time_reversal;
                const ComplexMatrix rpa_delta = build_export_delta(
                    rpa_target_transform, rpa_source_transform,
                    rpa_target.time_reversal, rpa_source.time_reversal, naux);
                const auto delta_name="rpa_delta_q_"+std::to_string(target_id)+".dat";
                write_export_matrix(export_dir/delta_name,"LIBRPA_DELTA",rpa_delta,target_id,representative_id);
                rpa_manifest << target_id << ' ' << representative_id << ' ' << rpa_target.spatial_isym+1
                             << ' ' << int(conjugate) << ' ' << q_target_frac.x << ' ' << q_target_frac.y << ' ' << q_target_frac.z
                             << ' ' << q_rep_frac.x << ' ' << q_rep_frac.y << ' ' << q_rep_frac.z << ' ' << delta_name << '\n';
            }
        } catch (const std::exception &rpa_export_exception) {
            rpa_export_write_error = std::string("chi0 symmetry export: ") + rpa_export_exception.what();
        }
    }

    if (chi.comm_h.is_root() && rpa_export_write_error.empty()) {
        try {
            rpa_manifest.close();
            if (!rpa_manifest || rpa_exported_targets.size()!=chi.pbc.klist_full.size())
                throw std::runtime_error("incomplete chi0 symmetry manifest");
            // Line 2 keeps the layout read by existing consumers; line 3 adds the provenance.
            std::ofstream rpa_info(export_dir/"chi0_q.info");
            rpa_info << "LIBRPA_CHI0_Q_GRID 1\n" << std::setprecision(17) << omega << ' ' << naux << ' '
                     << chi.pbc.period.x << ' ' << chi.pbc.period.y << ' ' << chi.pbc.period.z << '\n';
            rpa_info << "grid_index " << chi0_grid_index << " n_grid_nodes " << chi.tfg.get_n_grids()
                     << " source " << chi0_source_tag << '\n';
            rpa_info.close();
            if (!rpa_info) throw std::runtime_error("cannot write chi0_q.info");
        } catch (const std::exception &rpa_export_exception) {
            rpa_export_write_error = std::string("chi0 symmetry export: ") + rpa_export_exception.what();
        }
    }
    check_chi0_collective_error(chi.comm_h, rpa_export_write_error);
    global::lib_printf_root(
        "Exported bare representative chi0(q,iw) and Delta, grid index %d, omega = %.16e Ha (%s), to %s\n",
        chi0_grid_index, omega, chi0_source_tag.c_str(), directory.c_str());
}
} // namespace librpa_int
