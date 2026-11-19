#include "export_chi0_symmetry.h"
#include "../core/chi0.h"
#include "../io/global_io.h"
#include <cmath>
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
    if (!output) throw std::runtime_error("cannot write chi0 symmetry matrix: " + path.string());
}
}

void export_chi0_symmetry(const Chi0 &chi, const std::string &directory, double omega) {
    const auto &rpa_basis = chi.atbasis_abf;
    const int naux = rpa_basis.nb_total;
    const auto &rpa_context = chi.symmetry_context;
    std::map<atom_t, std::size_t> rpa_atom_naux;
    for (int atom=0; atom<rpa_basis.n_atoms; ++atom) rpa_atom_naux[atom]=rpa_basis[atom];
    const auto rpa_aux_layouts = rpa_basis.build_species_basis_layouts(rpa_context.atom_to_type);
    if (!rpa_context.available || !symmetry_species_layouts_match_atom_counts(
            rpa_aux_layouts, rpa_context.atom_to_type, rpa_atom_naux))
        throw std::runtime_error("chi0 symmetry export requires complete auxiliary basis layouts");
    const std::filesystem::path export_dir(directory);
    std::ofstream rpa_manifest;
    if (chi.comm_h.is_root()) {
        std::filesystem::create_directories(export_dir);
        rpa_manifest.open(export_dir/"rpa_symmetry_chi0.dat");
        rpa_manifest << "LIBRPA_SYMMETRY_ALL 1\n" << naux << ' ' << chi.pbc.klist_full.size()
                     << "\nBARE_CHI0 D_CHI0_DAGGER\n" << std::scientific << std::setprecision(17);
    }
    std::set<int> rpa_exported_targets;
    for (const auto &q_rep : chi.active_qpoints()) {
        ComplexMatrix rpa_chi0_representative(naux,naux);
        rpa_chi0_representative.zero_out();
        std::vector<int> rpa_chi0_counts(naux*naux,0);
        const auto rpa_frequency = chi.get_chi0_q().find(omega);
        if (rpa_frequency != chi.get_chi0_q().end()) {
            const auto rpa_q_entry = rpa_frequency->second.find(q_rep);
            if (rpa_q_entry != rpa_frequency->second.end())
                for (const auto &[atom_i,rpa_columns] : rpa_q_entry->second)
                    for (const auto &[atom_j,rpa_atom_response] : rpa_columns)
                        for (int row=0; row<rpa_atom_response.nr; ++row)
                            for (int col=0; col<rpa_atom_response.nc; ++col) {
                                const int mu=rpa_basis.get_global_index(atom_i,row), nu=rpa_basis.get_global_index(atom_j,col);
                                rpa_chi0_representative(mu,nu)=rpa_atom_response(row,col);
                                rpa_chi0_counts[mu*naux+nu]=1;
                                if (atom_i!=atom_j) {
                                    rpa_chi0_representative(nu,mu)=std::conj(rpa_atom_response(row,col));
                                    rpa_chi0_counts[nu*naux+mu]=1;
                                }
                            }
        }
        MPI_Allreduce(MPI_IN_PLACE,rpa_chi0_representative.c,naux*naux,MPI_C_DOUBLE_COMPLEX,MPI_SUM,chi.comm_h.comm);
        MPI_Allreduce(MPI_IN_PLACE,rpa_chi0_counts.data(),naux*naux,MPI_INT,MPI_SUM,chi.comm_h.comm);
        for (int entry=0; entry<naux*naux; ++entry) {
            if (!rpa_chi0_counts[entry]) throw std::runtime_error("missing representative chi0 entries during export");
            rpa_chi0_representative.c[entry]/=rpa_chi0_counts[entry];
        }
        const Vector3_Order<double> q_rep_frac=q_rep*chi.pbc.latvec.Transpose();
        const int representative_id=export_q_id(chi,q_rep_frac);
        if (!chi.comm_h.is_root()) continue;
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
            ComplexMatrix rpa_delta(naux,naux);
            rpa_delta.zero_out();
            // LibRPA applies T* X* T^T for time reversal, so its effective
            // left operator is conj(T). Representatives need not be the canonical star source. Compose the
            // target rotation with the inverse source rotation; antiunitary parity
            // determines whether the source adjoint and response are conjugated.
            for (int mu=0; mu<naux; ++mu) for (int k=0; k<naux; ++k) {
                if (rpa_target_transform(mu,k)==std::complex<double>{}) continue;
                const auto rpa_target_factor = rpa_target.time_reversal
                    ? std::conj(rpa_target_transform(mu,k)) : rpa_target_transform(mu,k);
                for (int nu=0; nu<naux; ++nu) {
                    const auto rpa_source_factor = rpa_source.time_reversal
                        ? std::conj(rpa_source_transform(nu,k)) : rpa_source_transform(nu,k);
                    rpa_delta(mu,nu)+=rpa_target_factor*(conjugate ? rpa_source_factor : std::conj(rpa_source_factor));
                }
            }
            const auto delta_name="rpa_delta_q_"+std::to_string(target_id)+".dat";
            write_export_matrix(export_dir/delta_name,"LIBRPA_DELTA",rpa_delta,target_id,representative_id);
            rpa_manifest << target_id << ' ' << representative_id << ' ' << rpa_target.spatial_isym+1
                         << ' ' << int(conjugate) << ' ' << q_target_frac.x << ' ' << q_target_frac.y << ' ' << q_target_frac.z
                         << ' ' << q_rep_frac.x << ' ' << q_rep_frac.y << ' ' << q_rep_frac.z << ' ' << delta_name << '\n';
        }
    }
    if (!chi.comm_h.is_root()) return;
    if (!rpa_manifest || rpa_exported_targets.size()!=chi.pbc.klist_full.size())
        throw std::runtime_error("incomplete chi0 symmetry manifest");
    std::ofstream rpa_info(export_dir/"chi0_q.info");
    rpa_info << "LIBRPA_CHI0_Q_GRID 1\n" << std::setprecision(17) << omega << ' ' << naux << ' '
             << chi.pbc.period.x << ' ' << chi.pbc.period.y << ' ' << chi.pbc.period.z << '\n';
    if (!rpa_info) throw std::runtime_error("cannot write chi0_q.info");
    global::lib_printf_root("Exported bare representative chi0(q,iw) and Delta, omega = %.16e Ha, to %s\n",omega,directory.c_str());
}
} // namespace librpa_int
