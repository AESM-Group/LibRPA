#include "meanfield_mpi.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>

#include "../io/stl_io_helper.h"
#include "../math/lapack_connector.h"
#include "../math/matrix_m.h"
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../utils/constants.h"
#include "../utils/profiler.h"
#include "atomic_basis.h"
#include "dielecmodel.h"
#include "librpa_enums.h"
#include "symmetry_context.h"
#include "utils_atomic_basis_blacs.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include "../gpu/device_connector.h"
#include <ddla/ddla.h>
#endif

namespace librpa_int
{

SymmetryExpandedMeanField expand_symmetry_meanfield(const MeanField &source,
                                                    const AtomicBasis &basis,
                                                    const PeriodicBoundaryData &pbc,
                                                    const SymmetryContext &context)
{
    const int nk = pbc.get_n_cells_bvk(), loaded = source.get_n_kpoints();
    const int bands = source.get_n_bands(), naos = source.get_n_aos();
    const int spins = source.get_n_spins(), components = source.get_n_spinor();
    const auto layouts = basis.build_species_basis_layouts(context.atom_to_type);
    const auto atom_nw = basis.get_atom_nb_map();
    if (bands <= 0 || bands > naos * components || loaded != pbc.kfrac_list.size() ||
        nk != pbc.kfrac_list_full.size() || pbc.weight_k.size() != loaded ||
        !can_restore_symmetry_kstar_meanfield(context, layouts, source, pbc.kfrac_list, atom_nw))
        throw LIBRPA_RUNTIME_ERROR(
            "IBZ meanfield expansion requires complete source coefficients and k stars");
    validate_spin_operations_for_storage(context, spins, components);
    SymmetryExpandedMeanField result{MeanField(spins, nk, bands, naos, components),
                                     std::vector<int>(nk, -1), std::vector<bool>(nk)};
    result.meanfield.get_efermi() = source.get_efermi();
    for (int ik = 0; ik < loaded; ++ik)
    {
        const auto &star = find_symmetry_kstar_for_ibz_kpoint(context, pbc.kfrac_list[ik]);
        const double weight = pbc.weight_k[ik];
        if (!std::isfinite(weight) || weight <= 0 ||
            std::abs(weight * nk - star.members.size()) > 1e-8)
            throw LIBRPA_RUNTIME_ERROR(
                "IBZ expansion requires normalized star-multiplicity k weights");
        for (const auto &member : star.members)
        {
            const int target =
                fold_fractional_kpoint_to_targets(member.k_bz, pbc.kfrac_list_full, 1e-8)
                    .target_k_index;
            if (target < 0 || result.source_k[target] >= 0)
                throw LIBRPA_RUNTIME_ERROR(
                    "IBZ expansion has a missing or duplicate k-star member");
            result.source_k[target] = ik;
            result.antiunitary[target] = member.time_reversal;
            for (int spin = 0; spin < spins; ++spin)
            {
                for (int component = 0; component < components; ++component)
                {
                    const auto *c = source.find_wfc(spin, component, ik);
                    if (!c || c->nr != bands || c->nc != naos)
                        throw LIBRPA_RUNTIME_ERROR("IBZ expansion requires complete KS coefficients");
                }
                if (components == 2)
                {
                    const std::array<ComplexMatrix, 2> input{*source.find_wfc(spin, 0, ik),
                                                             *source.find_wfc(spin, 1, ik)};
                    auto rotated = rotate_headwing_spinor_wfc_to_kstar_member(
                        context, member, layouts, atom_nw, pbc.kfrac_list[ik], input,
                        &pbc.kfrac_list_full[target]);
                    for (int component = 0; component < 2; ++component)
                        result.meanfield.get_eigenvectors()[spin][component][target] =
                            std::move(rotated[component]);
                }
                else
                    result.meanfield.get_eigenvectors()[spin][0][target] =
                        rotate_headwing_wfc_to_kstar_member(
                            context, member, layouts, atom_nw, pbc.kfrac_list[ik],
                            *source.find_wfc(spin, 0, ik), &pbc.kfrac_list_full[target]);
                for (int band = 0; band < bands; ++band)
                {
                    result.meanfield.get_eigenvals()[spin](target, band) =
                        source.get_eigenvals()[spin](ik, band);
                    // Loaded body occupations contain the star weight. Expand
                    // to physical f/N_full exactly once, retaining source f.
                    result.meanfield.get_weight()[spin](target, band) =
                        source.get_weight()[spin](ik, band) / (weight * nk);
                }
            }
        }
    }
    if (std::find(result.source_k.begin(), result.source_k.end(), -1) != result.source_k.end())
        throw LIBRPA_RUNTIME_ERROR("IBZ expansion does not cover the complete full mesh");
    return result;
}

void validate_symmetry_band_weights(const MeanField &mf, const AtomicBasis &basis,
                                    const PeriodicBoundaryData &pbc,
                                    const SymmetryContext &symmetry,
                                    const std::vector<std::vector<std::vector<double>>> &weights)
{
    const int bands = mf.get_n_bands(), naos = mf.get_n_aos();
    const int components = mf.get_n_spinor();
    if (weights.empty() && mf.get_n_kpoints() < pbc.get_n_cells_bvk())
    {
        // Validate all operation routes and little-group constraints without
        // changing the live IBZ state, fixed reference, or public output grid.
        const auto expanded = expand_symmetry_meanfield(mf, basis, pbc, symmetry);
        auto full_pbc = pbc;
        full_pbc.klist = pbc.klist_full;
        full_pbc.kfrac_list = pbc.kfrac_list_full;
        full_pbc.weight_k.assign(pbc.get_n_cells_bvk(), 1.0 / pbc.get_n_cells_bvk());
        validate_symmetry_band_weights(expanded.meanfield, basis, full_pbc, symmetry);
        return;
    }
    const int dimension = naos * components;
    if (bands <= 0 || bands > dimension || mf.get_n_kpoints() != pbc.get_n_cells_bvk() ||
        !basis.has_l_shells() || !symmetry.available)
        throw LIBRPA_RUNTIME_ERROR(
            "magnetic endpoint weights require replicated full-grid KS coefficients");
    for (int spin = 0; spin < mf.get_n_spins(); ++spin)
        for (int k = 0; k < mf.get_n_kpoints(); ++k)
        {
            for (int component = 0; component < components; ++component)
            {
                const auto *c = mf.find_wfc(spin, component, k);
                if (!c || c->nr != bands || c->nc != naos)
                    throw LIBRPA_RUNTIME_ERROR(
                        "magnetic covariance requires complete KS coefficients");
                for (int i = 0; i < c->size; ++i)
                    if (!std::isfinite(c->c[i].real()) || !std::isfinite(c->c[i].imag()))
                        throw LIBRPA_RUNTIME_ERROR(
                            "magnetic covariance found non-finite KS coefficients");
            }
            for (int band = 0; band < bands; ++band)
                if (!std::isfinite(mf.get_eigenvals()[spin](k, band)) ||
                    !std::isfinite(mf.get_weight()[spin](k, band)))
                    throw LIBRPA_RUNTIME_ERROR("magnetic covariance found non-finite E/f");
        }
    validate_spin_operations_for_storage(symmetry, mf.get_n_spins(), components);
    const auto layouts = basis.build_species_basis_layouts(symmetry.atom_to_type);
    const auto atom_nw = basis.get_atom_nb_map();
    int lmax = 0;
    for (const auto &layout : layouts) lmax = std::max(lmax, layout.max_l);

    // Include little-group operations too: checking only the chosen k-star
    // routes would miss a target that breaks a stabilizer of a representative.
    auto context = symmetry;
    context.kspace_actions.clear();
    for (std::size_t operation = 0; operation < context.spin_operations.size(); ++operation)
        context.kspace_actions.push_back({operation, {operation}});

    constexpr double tolerance = 2e-7;
    const auto packed = [&](int spin, int k)
    {
        ComplexMatrix result(dimension, bands);
        for (int component = 0; component < components; ++component)
        {
            const auto *c = mf.find_wfc(spin, component, k);
            if (!c || c->nr != bands || c->nc != naos)
                throw LIBRPA_RUNTIME_ERROR(
                    "magnetic endpoint weights found an incomplete KS frame");
            for (int band = 0; band < bands; ++band)
                for (int ao = 0; ao < naos; ++ao)
                    result(component * naos + ao, band) = (*c)(band, ao);
        }
        return result;
    };
    for (int target = 0; target < mf.get_n_kpoints(); ++target)
    {
        std::vector<ComplexMatrix> inverse;
        for (int spin = 0; spin < mf.get_n_spins(); ++spin)
        {
            auto c = packed(spin, target);
            const auto adjoint = transpose(c, true);
            const bool rectangular = bands != dimension;
            if (rectangular) c = adjoint * c;
            const auto gram = c;
            std::vector<int> pivot(bands);
            std::vector<std::complex<double>> work(bands);
            int info = 0;
            LapackConnector::getrf(bands, bands, c.c, bands, pivot.data(), info);
            if (!info)
                LapackConnector::getri(bands, c.c, bands, pivot.data(), work.data(), bands, info);
            if (info)
                throw LIBRPA_RUNTIME_ERROR("magnetic endpoint weights found a singular KS frame");
            if (rectangular)
            {
                const auto infinity_norm = [](const ComplexMatrix &matrix)
                {
                    double result = 0;
                    for (int row = 0; row < matrix.nr; ++row)
                    {
                        double sum = 0;
                        for (int col = 0; col < matrix.nc; ++col)
                            sum += std::abs(matrix(row, col));
                        if (!std::isfinite(sum))
                            return std::numeric_limits<double>::infinity();
                        result = std::max(result, sum);
                    }
                    return result;
                };
                const double condition = infinity_norm(gram) * infinity_norm(c);
                if (!std::isfinite(condition) ||
                    condition * std::numeric_limits<double>::epsilon() > tolerance)
                    throw LIBRPA_RUNTIME_ERROR("magnetic covariance found an ill-conditioned KS frame");
                // A left inverse recovers the sewing only inside the retained
                // span. The reconstruction check below must also pass.
                c = c * adjoint;
            }
            inverse.push_back(std::move(c));
        }
        for (std::size_t operation = 0; operation < context.spin_operations.size(); ++operation)
        {
            const auto &op = context.spin_operations[operation];
            auto source_k = apply_space_group_rotation_to_kpoint(
                context.rspace_operations.at(op.spatial_id), pbc.kfrac_list[target]);
            if (op.antiunitary) source_k = source_k * -1.0;
            const int source =
                fold_fractional_kpoint_to_targets(source_k, pbc.kfrac_list, 1e-8).target_k_index;
            if (source < 0)
                throw LIBRPA_RUNTIME_ERROR(
                    "magnetic endpoint weights require a symmetry-closed k grid");
            auto member = build_symmetry_kspace_operation_member(
                context, static_cast<int>(op.spatial_id), op.antiunitary, pbc.kfrac_list[target],
                pbc.kfrac_list[source], lmax);
            member.action_id = operation;
            for (int spin = 0; spin < mf.get_n_spins(); ++spin)
            {
                ComplexMatrix generated(dimension, bands);
                if (components == 2)
                {
                    const std::array<ComplexMatrix, 2> source_c{*mf.find_wfc(spin, 0, source),
                                                                *mf.find_wfc(spin, 1, source)};
                    const auto rotated = rotate_headwing_spinor_wfc_to_kstar_member(
                        context, member, layouts, atom_nw, pbc.kfrac_list[source], source_c,
                        &pbc.kfrac_list[target]);
                    for (int component = 0; component < 2; ++component)
                        for (int ao = 0; ao < naos; ++ao)
                            for (int band = 0; band < bands; ++band)
                                generated(component * naos + ao, band) =
                                    rotated[component](band, ao);
                }
                else
                    generated =
                        transpose(rotate_headwing_wfc_to_kstar_member(
                                      context, member, layouts, atom_nw, pbc.kfrac_list[source],
                                      *mf.find_wfc(spin, 0, source), &pbc.kfrac_list[target]),
                                  false);
                const auto sewing = inverse[spin] * generated;
                const auto metric = transpose(sewing, true) * sewing;
                double residual = 0;
                if (bands != dimension)
                {
                    const auto reconstructed = packed(spin, target) * sewing;
                    double error = 0, scale = 0;
                    for (int i = 0; i < generated.size; ++i)
                    {
                        const double difference = std::abs(reconstructed.c[i] - generated.c[i]);
                        if (!std::isfinite(difference))
                            throw LIBRPA_RUNTIME_ERROR("magnetic covariance found non-finite sewing");
                        error = std::max(error, difference);
                        scale = std::max(scale, std::abs(generated.c[i]));
                    }
                    residual = scale > 0 ? error / scale
                                         : std::numeric_limits<double>::infinity();
                }
                for (int row = 0; row < bands; ++row)
                    for (int col = 0; col < bands; ++col)
                    {
                        residual = std::max(
                            residual,
                            std::abs(metric(row, col) - (row == col ? std::complex<double>(1.0)
                                                                    : std::complex<double>{})));
                        const double e_target = mf.get_eigenvals()[spin](target, row);
                        const double e_source = mf.get_eigenvals()[spin](source, col);
                        residual = std::max(
                            residual, std::abs(sewing(row, col)) * std::abs(e_target - e_source) /
                                          std::max({1.0, std::abs(e_target), std::abs(e_source)}));
                        residual =
                            std::max(residual, std::abs(sewing(row, col)) * mf.get_n_kpoints() *
                                                   std::abs(mf.get_weight()[spin](target, row) -
                                                            mf.get_weight()[spin](source, col)));
                        if (!weights.empty())
                            residual = std::max(residual, std::abs(sewing(row, col)) *
                                                              std::abs(weights[spin][target][row] -
                                                                       weights[spin][source][col]));
                    }
                if (!std::isfinite(residual) || residual > tolerance)
                {
                    std::ostringstream message;
                    message << "magnetic KS/endpoint-weight covariance failed: k=" << target
                            << ", source=" << source << ", spin=" << spin
                            << ", operation=" << operation << ", residual=" << residual
                            << "; use the full grid or a symmetry-compatible target/gauge";
                    throw LIBRPA_RUNTIME_ERROR(message.str());
                }
            }
        }
    }
}

void redistribute_meanfield_eigvecs_kblacs(
    MeanField &mf, const KPointBlacsParallelContext &kblacs_ctxt,
    const ArrayDesc &desc_wfc_src, const ArrayDesc &desc_wfc_dst,
    const std::string &label)
{
    global::profiler.start("redistribute_meanfield_eigvecs_kblacs_opt");
    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR(label + " k-point BLACS context is not initialized");
    if (!desc_wfc_src.is_initialized() || !desc_wfc_dst.is_initialized())
        throw LIBRPA_RUNTIME_ERROR(label + " wave-function descriptors are not initialized");
    if (desc_wfc_src.ictxt() != kblacs_ctxt.blacs_h.ictxt ||
        desc_wfc_dst.ictxt() != kblacs_ctxt.blacs_h.ictxt)
        throw LIBRPA_RUNTIME_ERROR(label + " wave-function descriptors use the wrong BLACS context");

    const int n_spins = mf.get_n_spins();
    const int n_spinor = mf.get_n_spinor();
    const int n_kpoints = mf.get_n_kpoints();
    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    if (n_spins <= 0 || n_spinor <= 0 || n_kpoints <= 0 || n_aos <= 0 ||
        n_states <= 0)
        throw LIBRPA_RUNTIME_ERROR(label + " mean-field dimensions are not initialized");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR(label + " k-point BLACS context has inconsistent dimensions");
    if (desc_wfc_src.m() != n_aos || desc_wfc_src.n() != n_states ||
        desc_wfc_dst.m() != n_aos || desc_wfc_dst.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR(label + " wave-function descriptors have inconsistent dimensions");

    using WfcMap = std::map<int, std::map<int, std::map<int, ComplexMatrix>>>;
    WfcMap wfc_dst;
    std::vector<cplxdb> dummy(1, C_ZERO);
    const std::size_t src_size = static_cast<std::size_t>(desc_wfc_src.m_loc()) *
                                 desc_wfc_src.n_loc();

    for (int ispin = 0; ispin != n_spins; ++ispin)
    {
        for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
        {
            for (const int ik : kblacs_ctxt.kpoints_local())
            {
                const auto *src = mf.find_wfc(ispin, ispinor, ik);
                const int bad_local =
                    (src_size > 0 && src == nullptr) ||
                    (src != nullptr &&
                     (src->nr != desc_wfc_src.n_loc() ||
                      src->nc != desc_wfc_src.m_loc() ||
                      static_cast<std::size_t>(src->size) != src_size));
                int bad = 0;
                MPI_Allreduce(&bad_local, &bad, 1, MPI_INT, MPI_MAX,
                              kblacs_ctxt.comm_blacs_h.comm);
                if (bad)
                    throw LIBRPA_RUNTIME_ERROR(
                        label + " source wave-function block is missing or inconsistent for k-point " +
                        std::to_string(ik));

                ComplexMatrix dst(desc_wfc_dst.n_loc(), desc_wfc_dst.m_loc(), false);
                ScalapackConnector::pgemr2d_f(
                    n_aos, n_states, src == nullptr ? dummy.data() : src->c, 1, 1,
                    desc_wfc_src.desc, dst.c == nullptr ? dummy.data() : dst.c, 1, 1,
                    desc_wfc_dst.desc, kblacs_ctxt.blacs_h.ictxt);
                wfc_dst[ispin][ispinor][ik] = std::move(dst);
            }
        }
    }

    mf.get_eigenvectors().swap(wfc_dst);
    global::ofs_myid << "Redistributed " << label
                     << " eigenvectors into permanent k-BLACS blocks "
                     << "(mb, nb) = (" << desc_wfc_dst.mb() << ", "
                     << desc_wfc_dst.nb() << ")" << std::endl;
    global::profiler.stop("redistribute_meanfield_eigvecs_kblacs_opt");
}

static void collect_Rs(const std::vector<Vector3_Order<int>> &Rs, std::vector<int> &n_Rs_all,
                       std::vector<int> &Rs_all, int &nR_max, const MpiCommHandler &comm_h)
{
    n_Rs_all.resize(comm_h.nprocs);
    for (int pid = 0; pid < comm_h.nprocs; pid++) n_Rs_all[pid] = 0;
    const int n_Rs_this = Rs.size();
    n_Rs_all[comm_h.myid] = n_Rs_this;
    // global::ofs_myid << global::myid_global << " " << global::size_global << std::endl;
    // comm_h.barrier();
    MPI_Allreduce(MPI_IN_PLACE, n_Rs_all.data(), comm_h.nprocs, mpi_datatype<int>::value, MPI_SUM, comm_h.comm);
    nR_max = *std::max_element(n_Rs_all.cbegin(), n_Rs_all.cend());
    Rs_all.resize(3 * nR_max * comm_h.nprocs);
    for (int iR = 0; iR < n_Rs_this; iR++)
    {
        Rs_all[comm_h.myid * nR_max * 3 + iR * 3] = Rs[iR].x;
        Rs_all[comm_h.myid * nR_max * 3 + iR * 3 + 1] = Rs[iR].y;
        Rs_all[comm_h.myid * nR_max * 3 + iR * 3 + 2] = Rs[iR].z;
    }
    MPI_Allreduce(MPI_IN_PLACE, Rs_all.data(), comm_h.nprocs * nR_max * 3, mpi_datatype<int>::value, MPI_SUM, comm_h.comm);
}

static void check_same_imagtimes(const std::vector<double> &imagtimes,
                                 const MpiCommHandler &comm_h)
{
    const int n_tau = imagtimes.size();
    int n_tau_min = 0, n_tau_max = 0;
    MPI_Allreduce(&n_tau, &n_tau_min, 1, mpi_datatype<int>::value, MPI_MIN, comm_h.comm);
    MPI_Allreduce(&n_tau, &n_tau_max, 1, mpi_datatype<int>::value, MPI_MAX, comm_h.comm);
    if (n_tau_min != n_tau_max)
        throw LIBRPA_RUNTIME_ERROR("imaginary-time lists differ in k-point BLACS context");
    if (n_tau == 0) return;

    std::vector<double> imagtimes_all(n_tau * comm_h.nprocs);
    MPI_Allgather(imagtimes.data(), n_tau, mpi_datatype<double>::value,
                  imagtimes_all.data(), n_tau, mpi_datatype<double>::value, comm_h.comm);
    for (int pid = 0; pid != comm_h.nprocs; ++pid)
    {
        for (int it = 0; it != n_tau; ++it)
        {
            if (imagtimes_all[pid * n_tau + it] != imagtimes[it])
                throw LIBRPA_RUNTIME_ERROR("imaginary-time lists differ in k-point BLACS context");
        }
    }
}

static std::vector<const SymmetryKAtomRotation*> build_rotations_by_from(
    const SymmetryKStarMember &member, const std::size_t n_atoms)
{
    std::vector<const SymmetryKAtomRotation*> rotations(n_atoms, nullptr);
    for (const auto &rotation : member.atom_rotations)
    {
        if (rotation.atom_from < 0 || rotation.atom_from >= static_cast<int>(n_atoms))
        {
            throw LIBRPA_RUNTIME_ERROR("k-star atom rotation source is out of range");
        }
        rotations[static_cast<std::size_t>(rotation.atom_from)] = &rotation;
    }
    for (const auto *rotation : rotations)
    {
        if (rotation == nullptr)
        {
            throw LIBRPA_RUNTIME_ERROR("k-star atom rotations do not cover every atom");
        }
    }
    return rotations;
}

static std::unordered_map<int, std::vector<atpair_t>> build_source_pair_requests(
    const std::unordered_map<int, std::set<atpair_t>> &target_pairs,
    const SymmetryKStarMember &member,
    const std::size_t n_atoms)
{
    const auto rotations = build_rotations_by_from(member, n_atoms);
    std::unordered_map<int, std::vector<atpair_t>> requests;
    for (const auto &[pid, pairs] : target_pairs)
    {
        std::set<atpair_t> source_pairs;
        for (const auto &pair : pairs)
        {
            const auto *rot_i = rotations.at(static_cast<std::size_t>(pair.first));
            const auto *rot_j = rotations.at(static_cast<std::size_t>(pair.second));
            source_pairs.insert({static_cast<atom_t>(rot_i->atom_to),
                                 static_cast<atom_t>(rot_j->atom_to)});
        }
        requests[pid] = std::vector<atpair_t>(source_pairs.cbegin(), source_pairs.cend());
    }
    return requests;
}

static ComplexMatrix matz_to_complex_matrix(const Matz &mat)
{
    ComplexMatrix block(mat.nr(), mat.nc());
    for (int i = 0; i != mat.nr(); ++i)
    {
        for (int j = 0; j != mat.nc(); ++j)
        {
            block(i, j) = mat(i, j);
        }
    }
    return block;
}

// The k-BLACS density/Green matrices are formed from row-major wavefunctions
// as C^T C*. Consequently their symmetry restore uses the wavefunction
// rotation convention directly: D_bz = M^H D_ibz M (unitary operations),
// rather than the AO-operator convention used by the Coulomb/GW path.
static symmetry_atom_block_matrix_map_t rotate_wfc_product_blocks(
    const SymmetryContext &ctx,
    const std::vector<SpeciesBasisLayout> &layouts,
    const SymmetryKStarMember &member,
    const symmetry_atom_block_matrix_map_t &source_blocks,
    const std::map<atom_t, size_t> &atom_nbasis,
    const Vector3_Order<double> &k_ibz,
    const std::set<atpair_t> *target_pairs,
    const Vector3_Order<double> *k_bz_target)
{
    const auto rotation = build_symmetry_kspace_rotation_matrix(
        ctx, layouts, member, atom_nbasis, k_ibz, member.time_reversal, k_bz_target);
    std::vector<int> offsets(atom_nbasis.size() + 1, 0);
    for (std::size_t atom = 0; atom != atom_nbasis.size(); ++atom)
        offsets[atom + 1] = offsets[atom] + static_cast<int>(atom_nbasis.at(static_cast<atom_t>(atom)));

    std::vector<const SymmetryKAtomRotation *> by_from(atom_nbasis.size(), nullptr);
    for (const auto &atom_rotation : member.atom_rotations)
    {
        if (atom_rotation.atom_from < 0
            || atom_rotation.atom_from >= static_cast<int>(atom_nbasis.size()))
            throw LIBRPA_RUNTIME_ERROR("k-star density restore has an invalid atom map");
        by_from[static_cast<std::size_t>(atom_rotation.atom_from)] = &atom_rotation;
    }

    const auto get_source_block = [&source_blocks](const atom_t atom_i, const atom_t atom_j) {
        const auto row_iter = source_blocks.find(atom_i);
        if (row_iter != source_blocks.end())
        {
            const auto col_iter = row_iter->second.find(atom_j);
            if (col_iter != row_iter->second.end()) return col_iter->second;
        }
        const auto col_iter = source_blocks.find(atom_j);
        if (col_iter != source_blocks.end())
        {
            const auto row_iter = col_iter->second.find(atom_i);
            if (row_iter != col_iter->second.end()) return transpose(row_iter->second, true);
        }
        throw LIBRPA_RUNTIME_ERROR("missing atom block while restoring a wavefunction product");
    };

    symmetry_atom_block_matrix_map_t rotated;
    for (std::size_t atom_i = 0; atom_i != atom_nbasis.size(); ++atom_i)
    {
        const auto *rotation_i = by_from[atom_i];
        if (rotation_i == nullptr) throw LIBRPA_RUNTIME_ERROR("incomplete k-star atom map");
        const auto source_i = static_cast<atom_t>(rotation_i->atom_to);
        const int row_i = offsets.at(static_cast<std::size_t>(source_i));
        const int col_i = offsets.at(atom_i);
        const int n_i = static_cast<int>(atom_nbasis.at(static_cast<atom_t>(atom_i)));
        const int n_source_i = static_cast<int>(atom_nbasis.at(source_i));
        ComplexMatrix M_i(n_source_i, n_i);
        for (int i = 0; i != n_source_i; ++i)
            for (int j = 0; j != n_i; ++j)
                M_i(i, j) = rotation(row_i + i, col_i + j);

        for (std::size_t atom_j = 0; atom_j != atom_nbasis.size(); ++atom_j)
        {
            const atpair_t pair{static_cast<atom_t>(atom_i), static_cast<atom_t>(atom_j)};
            if (target_pairs != nullptr && target_pairs->count(pair) == 0) continue;
            const auto *rotation_j = by_from[atom_j];
            if (rotation_j == nullptr) throw LIBRPA_RUNTIME_ERROR("incomplete k-star atom map");
            const auto source_j = static_cast<atom_t>(rotation_j->atom_to);
            const int row_j = offsets.at(static_cast<std::size_t>(source_j));
            const int col_j = offsets.at(atom_j);
            const int n_j = static_cast<int>(atom_nbasis.at(static_cast<atom_t>(atom_j)));
            const int n_source_j = static_cast<int>(atom_nbasis.at(source_j));
            ComplexMatrix M_j(n_source_j, n_j);
            for (int i = 0; i != n_source_j; ++i)
                for (int j = 0; j != n_j; ++j)
                    M_j(i, j) = rotation(row_j + i, col_j + j);

            const auto source = get_source_block(source_i, source_j);
            ComplexMatrix transformed;
            if (member.time_reversal)
                transformed = transpose(M_i, false) * conj(source) * conj(M_j);
            else
                transformed = transpose(M_i, true) * source * M_j;
            rotated[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                std::move(transformed);
        }
    }
    return rotated;
}

static Matz complex_matrix_to_colmajor_matz(const ComplexMatrix &block)
{
    Matz mat(block.nr, block.nc, MAJOR::COL);
    for (int i = 0; i != block.nr; ++i)
    {
        for (int j = 0; j != block.nc; ++j)
        {
            mat(i, j) = block(i, j);
        }
    }
    return mat;
}

struct WfcGemmWorkspace
{
    ArrayDesc desc_wfc_compute;
    ArrayDesc desc_out_opt;
    Matz out_opt;
};

static void check_wfc_gemm_context(const ArrayDesc &desc_wfc, const ArrayDesc &desc_out,
                                   const BlacsCtxtHandler &blacs_h)
{
    if (!blacs_h.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("active BLACS context is not initialized");
    if (!desc_wfc.is_initialized() || !desc_out.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm descriptors are not initialized");
    if (desc_wfc.ictxt() != blacs_h.ictxt || desc_out.ictxt() != blacs_h.ictxt)
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm descriptors must use the active k-point BLACS context");
}

static WfcGemmWorkspace create_wfc_gemm_workspace(const ArrayDesc &desc_wfc,
                                                  const ArrayDesc &desc_out,
                                                  const BlacsCtxtHandler &blacs_h)
{
    check_wfc_gemm_context(desc_wfc, desc_out, blacs_h);

    WfcGemmWorkspace workspace;
    // Keep an equivalent mutable descriptor for DDLA metadata. The matrix data
    // itself stays in MeanField and is consumed directly.
    workspace.desc_wfc_compute = ArrayDesc(blacs_h);
    workspace.desc_wfc_compute.init(
        desc_wfc.m(), desc_wfc.n(), desc_wfc.mb(), desc_wfc.nb(),
        desc_wfc.irsrc(), desc_wfc.icsrc());
    workspace.desc_out_opt = ArrayDesc(blacs_h);
    workspace.desc_out_opt.init(desc_out.m(), desc_out.n(),
                                desc_wfc.mb(), desc_wfc.mb(),
                                desc_out.irsrc(), desc_out.icsrc());
    workspace.out_opt.resize(workspace.desc_out_opt.m_loc(),
                             workspace.desc_out_opt.n_loc(), MAJOR::COL);
    return workspace;
}

static void pgemm_wfc_scaled_wfc_h(const int n_aos, const int n_cols,
                                   const cplxdb *wfc_bra_ptr,
                                   const cplxdb *scaled_wfc_ket_ptr,
                                   const ArrayDesc &desc_wfc,
                                   WfcGemmWorkspace &workspace,
                                   Matz &out, const ArrayDesc &desc_out,
                                   const BlacsCtxtHandler &blacs_h)
{
    global::profiler.start(__FUNCTION__, LIBRPA_VERBOSE_DEBUG);
    check_wfc_gemm_context(desc_wfc, desc_out, blacs_h);
    if (n_aos != desc_wfc.m() || n_aos != desc_out.m() || n_aos != desc_out.n())
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm matrix dimensions are inconsistent");
    if (n_cols < 0 || n_cols > desc_wfc.n())
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm column count is inconsistent with descriptor");
    if (!workspace.desc_wfc_compute.is_initialized() || !workspace.desc_out_opt.is_initialized() ||
        workspace.desc_wfc_compute.ictxt() != blacs_h.ictxt ||
        workspace.desc_out_opt.ictxt() != blacs_h.ictxt)
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm workspace is not initialized on the active k-point BLACS context");
    if (workspace.desc_wfc_compute.mb() != desc_wfc.mb() ||
        workspace.desc_wfc_compute.nb() != desc_wfc.nb() ||
        workspace.desc_wfc_compute.irsrc() != desc_wfc.irsrc() ||
        workspace.desc_wfc_compute.icsrc() != desc_wfc.icsrc())
        throw LIBRPA_RUNTIME_ERROR("WFC pgemm workspace layout differs from stored eigenvectors");

    std::vector<cplxdb> dummy(1, C_ZERO);
    const cplxdb *wfc_bra = wfc_bra_ptr == nullptr ? dummy.data() : wfc_bra_ptr;
    const cplxdb *scaled_wfc_ket =
        scaled_wfc_ket_ptr == nullptr ? dummy.data() : scaled_wfc_ket_ptr;
    cplxdb *out_opt = workspace.out_opt.ptr() == nullptr
                          ? dummy.data()
                          : workspace.out_opt.ptr();

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    auto &blacs_h_nc = const_cast<BlacsCtxtHandler &>(blacs_h);
    if (blacs_h_nc.ddla_handle == nullptr)
        blacs_h_nc.init_ddla_handle();
    workspace.desc_wfc_compute.set_ddla_handle(blacs_h_nc.ddla_handle);
    workspace.desc_out_opt.set_ddla_handle(blacs_h_nc.ddla_handle);

    auto handle = blacs_h_nc.ddla_handle;
    // LibRI GPU contractions can change the calling thread's current device.
    // Re-select the device owning this BLACS handle before using its stream.
    ddla::DDLA_CHECK(ddla::ddlaSetDevice(handle));
    const size_t size_ab = static_cast<size_t>(workspace.desc_wfc_compute.m_loc()) *
                           workspace.desc_wfc_compute.n_loc();
    const size_t size_c  = static_cast<size_t>(workspace.desc_out_opt.m_loc()) *
                           workspace.desc_out_opt.n_loc();
    const size_t alloc_size_ab = std::max<size_t>(size_ab, 1);
    const size_t alloc_size_c = std::max<size_t>(size_c, 1);
    cplxdb *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    ddla::DDLA_CHECK(ddla::ddlaMalloc(handle, (void**)&d_A, alloc_size_ab * sizeof(cplxdb)));
    ddla::DDLA_CHECK(ddla::ddlaMalloc(handle, (void**)&d_B, alloc_size_ab * sizeof(cplxdb)));
    ddla::DDLA_CHECK(ddla::ddlaMalloc(handle, (void**)&d_C, alloc_size_c * sizeof(cplxdb)));
    if (size_ab > 0)
    {
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d_A, wfc_bra, size_ab * sizeof(cplxdb)));
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, d_B, scaled_wfc_ket, size_ab * sizeof(cplxdb)));
    }
    ddla::pgemm(handle, 'N', 'C', n_aos, n_aos, n_cols, C_ONE,
                d_A, workspace.desc_wfc_compute.desc,
                d_B, workspace.desc_wfc_compute.desc,
                C_ZERO, d_C, workspace.desc_out_opt.desc);
    if (size_c > 0)
        ddla::DDLA_CHECK(ddla::ddlaMemcpy(handle, out_opt, d_C, size_c * sizeof(cplxdb)));
    ddla::DDLA_CHECK(ddla::ddlaSynchronize(handle));
    ddla::DDLA_CHECK(ddla::ddlaFree(handle, d_A));
    ddla::DDLA_CHECK(ddla::ddlaFree(handle, d_B));
    ddla::DDLA_CHECK(ddla::ddlaFree(handle, d_C));
#else
    global::profiler.start("pgemm_entry_wait", LIBRPA_VERBOSE_DEBUG);
    blacs_h.barrier();
    global::profiler.stop("pgemm_entry_wait");
    global::profiler.start("pgemm", LIBRPA_VERBOSE_DEBUG);
    ScalapackConnector::pgemm_f('N', 'C', n_aos, n_aos, n_cols, C_ONE,
                                wfc_bra, 1, 1,
                                workspace.desc_wfc_compute.desc,
                                scaled_wfc_ket, 1, 1,
                                workspace.desc_wfc_compute.desc,
                                C_ZERO, out_opt, 1, 1,
                                workspace.desc_out_opt.desc);
    global::profiler.stop("pgemm");
    global::profiler.start("pgemm_exit_wait", LIBRPA_VERBOSE_DEBUG);
    blacs_h.barrier();
    global::profiler.stop("pgemm_exit_wait");
#endif
    global::profiler.start("pgemr2d", LIBRPA_VERBOSE_DEBUG);
    ScalapackConnector::pgemr2d_f(n_aos, n_aos,
                                  out_opt, 1, 1,
                                  workspace.desc_out_opt.desc,
                                  out.ptr() == nullptr ? dummy.data() : out.ptr(),
                                  1, 1, desc_out.desc,
                                  blacs_h.ictxt);
    global::profiler.stop("pgemr2d");
    global::profiler.start("pgemr2d_exit_wait", LIBRPA_VERBOSE_DEBUG);
    blacs_h.barrier();
    global::profiler.stop("pgemr2d_exit_wait");
    global::profiler.stop(__FUNCTION__);
}

std::map<Vector3_Order<int>, ComplexMatrix> get_dmat_cplx_Rs_kpara(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf, const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<Vector3_Order<int>>& Rs, const MpiCommHandler &comm_h)
{
    global::profiler.start(__FUNCTION__);
    std::map<Vector3_Order<int>, ComplexMatrix> dmat_local;

    // Collect Rs requested by each process
    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, comm_h);
    // global::ofs_myid << "get_dmat_cplx_Rs_kpara nRs_all " << n_Rs_all << std::endl;

    const int n_aos = mf.get_n_aos();
    const size_t size = n_aos * n_aos;

    const auto iks_local = mf.get_iks_local();
    // global::ofs_myid << "iks_local " << iks_local << std::endl;
    const int nk_local = iks_local.size();
    // Check if there is duplicate k-point data
    int nk_local_sum;
    MPI_Allreduce(&nk_local, &nk_local_sum, 1, MPI_INT, MPI_SUM, comm_h.comm);
    if (nk_local_sum > mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("found duplicated k-point eigenvectors data");
    else if (nk_local_sum < mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("missing k-point eigenvectors data");

    Matz kmat(size, nk_local, MAJOR::COL);
    Matz transmat(nk_local, nR_max, MAJOR::COL);
    Matz rmat(size, nR_max, MAJOR::COL);

    // NOTE: Support dimension (number of basis) up to 46300 (Gamma-only) or 1930 (k 8x8x8) due to range of int of axpy.
    //       Need division into batches for larger system, particularly for memory consideration
    for (int ik_local = 0; ik_local < nk_local; ik_local++)
    {
        const auto dmat_k = mf.get_dmat_cplx(ispin, ispinor_bra, ispinor_ket, iks_local[ik_local]);
        memcpy(kmat.ptr() + size * ik_local, dmat_k.c, size * sizeof(Matz::type));
    }

    for (int pid = 0; pid < comm_h.nprocs; pid++)
    {
        const auto nR_this = n_Rs_all[pid];
        // NOTE: MPI_Reduce requires all processes use the same sendcount, but rmat is simply zero where nk_local == 0.
        const size_t count = size * nR_this;
        // global::ofs_myid << "get_dmat_cplx_Rs_kpara pid " << pid << " nR_this " << nR_this << " count " << count << " matsize " << rmat.size() << std::endl;

        if (nR_this < 1) continue;
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int iR = 0; iR < nR_this; iR++)
        {
            for (int ik = 0; ik < nk_local; ik++)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                const auto &kf = kfrac_list[iks_local[ik]];
                auto ang = - (kf.x * Rs_all[index] + kf.y * Rs_all[index+1] + kf.z * Rs_all[index+2]) * TWO_PI;
                transmat(ik, iR) = cplxdb{cos(ang), sin(ang)};
            }
        }
        // global::ofs_myid << "transmat pid" << std::endl;
        // global::ofs_myid << transmat << std::endl;
        rmat = 0.0;
        if (nk_local > 0)
        {
            LapackConnector::gemm_f('N', 'N', size, nR_this, nk_local, 1.0,
                                    kmat.ptr(), size, transmat.ptr(), nk_local, 0.0, rmat.ptr(), size);
        }
        // global::ofs_myid << rmat << std::endl;
        if (comm_h.myid == pid)
        {
            MPI_Reduce(MPI_IN_PLACE, rmat.ptr(), count, mpi_datatype<Matz::type>::value, MPI_SUM, pid, comm_h.comm);
            for (int iR = 0; iR < nR_this; iR++)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                ComplexMatrix m(n_aos, n_aos);
                memcpy(m.c, rmat.ptr() + size * iR, size * sizeof(Matz::type));
                Vector3_Order<int> R{Rs_all[index], Rs_all[index+1], Rs_all[index+2]};
                // global::ofs_myid << "iR " << iR << " R " << R << std::endl;
                dmat_local.emplace(R, std::move(m));
            }
        }
        else
        {
            MPI_Reduce(rmat.ptr(), rmat.ptr(), count, mpi_datatype<cplxdb>::value, MPI_SUM, pid, comm_h.comm);
        }
        comm_h.barrier(); // May try out non-blocking later
    }

    global::profiler.stop(__FUNCTION__);
    return dmat_local;
}

std::map<Vector3_Order<int>, ComplexMatrix> get_dmat_cplx_Rs_kpara(
    int ispin, const MeanField &mf, const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<Vector3_Order<int>>& Rs, const MpiCommHandler &comm_h)
{
    return get_dmat_cplx_Rs_kpara(ispin, 0, 0, mf, kfrac_list, Rs, comm_h);
}

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> get_gf_cplx_imagtimes_Rs_kpara(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const MpiCommHandler &comm_h, const std::vector<bool> &band_mask)
{
    std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> gf;

    // Collect Rs requested by each process
    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, comm_h);
    // global::ofs_myid << "get_gf_cplx_imagtimes_Rs_kpara nRs_all " << n_Rs_all << std::endl;

    const int n_aos = mf.get_n_aos();
    const size_t size = n_aos * n_aos;

    const auto iks_local = mf.get_iks_local();
    // global::ofs_myid << "iks_local " << iks_local << std::endl;
    // Check if there is duplicate k-point data
    const int nk_local = iks_local.size();
    int nk_local_sum;
    MPI_Allreduce(&nk_local, &nk_local_sum, 1, MPI_INT, MPI_SUM, comm_h.comm);
    if (nk_local_sum > mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("found duplicated k-point eigenvectors data");
    else if (nk_local_sum < mf.get_n_kpoints())
        throw LIBRPA_RUNTIME_ERROR("missing k-point eigenvectors data");

    Matz kmat(size, nk_local, MAJOR::COL);
    Matz transmat(nk_local, nR_max, MAJOR::COL);
    Matz rmat(size, nR_max, MAJOR::COL);

    for (auto tau: imagtimes)
    {
        std::map<Vector3_Order<int>, ComplexMatrix> gf_tau;
        // TODO: this part is the same as denstiy matrix calculation, so it may be extracted to a common function
        for (int ik_local = 0; ik_local < nk_local; ik_local++)
        {
            const auto dmat_k = mf.get_gf_cplx_imagtime(ispin, ispinor_bra, ispinor_ket, iks_local[ik_local], tau, band_mask);
            memcpy(kmat.ptr() + size * ik_local, dmat_k.c, size * sizeof(Matz::type));
        }

        for (int pid = 0; pid < comm_h.nprocs; pid++)
        {
            const auto nR_this = n_Rs_all[pid];
            // NOTE: MPI_Reduce requires all processes use the same sendcount, but rmat is simply zero where nk_local == 0.
            const size_t count = size * nR_this;
            // global::ofs_myid << "get_dmat_cplx_Rs_kpara pid " << pid << " nR_this " << nR_this << " count " << count << " matsize " << rmat.size() << std::endl;

            if (nR_this < 1) continue;
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (int iR = 0; iR <nR_this; iR++)
            {
                for (int ik = 0; ik < nk_local; ik++)
                {
                    const auto R_this = Rs_all.data() + pid * nR_max * 3 + iR * 3;
                    const auto &kf = kfrac_list[iks_local[ik]];
                    auto ang = - (kf.x * R_this[0] + kf.y * R_this[1] + kf.z * R_this[2]) * TWO_PI;
                    transmat(ik, iR) = cplxdb{cos(ang), sin(ang)};
                }
            }
            rmat = 0.0;
            if (nk_local > 0)
            {
                LapackConnector::gemm_f('N', 'N', size, nR_this, nk_local, 1.0,
                                        kmat.ptr(), size, transmat.ptr(), nk_local, 0.0, rmat.ptr(), size);
            }
            // global::ofs_myid << rmat << std::endl;
            if (comm_h.myid == pid)
            {
                MPI_Reduce(MPI_IN_PLACE, rmat.ptr(), count, mpi_datatype<cplxdb>::value, MPI_SUM, pid, comm_h.comm);
                for (int iR = 0; iR < nR_this; iR++)
                {
                    const auto R_this = Rs_all.data() + pid * nR_max * 3 + iR * 3;
                    ComplexMatrix m(n_aos, n_aos);
                    memcpy(m.c, rmat.ptr() + size * iR, size * sizeof(cplxdb));
                    Vector3_Order<int> R{R_this[0], R_this[1], R_this[2]};
                    gf_tau.emplace(R, std::move(m));
                }
            }
            else
            {
                MPI_Reduce(rmat.ptr(), rmat.ptr(), count, mpi_datatype<cplxdb>::value, MPI_SUM, pid, comm_h.comm);
            }
            comm_h.barrier(); // May try out non-blocking later
        }
        gf.emplace(tau, std::move(gf_tau));
    }

    return gf;
}

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> get_gf_cplx_imagtimes_Rs_kpara(
    int ispin, const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const MpiCommHandler &comm_h, const std::vector<bool> &band_mask)
{
    return get_gf_cplx_imagtimes_Rs_kpara(ispin, 0, 0, mf, kfrac_list, imagtimes, Rs, comm_h, band_mask);
}

std::map<Vector3_Order<int>, Matz> get_dmat_cplx_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm)
{
    global::profiler.start(__FUNCTION__);

    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("density-matrix descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and density-matrix descriptors must use the same BLACS context");

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<Vector3_Order<int>, Matz> dmat_Rs;
    for (const auto &R: Rs)
    {
        dmat_Rs.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
    }
    if (nR_max == 0)
    {
        global::profiler.stop(__FUNCTION__);
        return dmat_Rs;
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    const size_t dm_size_loc = static_cast<size_t>(desc_dm.m_loc()) * desc_dm.n_loc();
    if (dm_size_loc > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local density-matrix block is too large for MPI collectives");
    const int n_elem = static_cast<int>(dm_size_loc);

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz dmat_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");
    Matz kmat(nk_local, n_elem, MAJOR::COL);

    const double occ_thres = 1e-4 / n_kpoints;
    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    for (int ik_local = 0; ik_local != nk_local; ++ik_local)
    {
        const int ik = iks_local[ik_local];
        const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
        const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
        if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
            throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                      std::to_string(ik));
        if (wfc_size_loc > 0 && wfc_bra != nullptr &&
            static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
            throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
        if (wfc_size_loc > 0 && wfc_ket != nullptr &&
            static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
            throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

        int nocc = 0;
        std::vector<double> weights;
        weights.reserve(n_states);
        for (; nocc != n_states; ++nocc)
        {
            const double weight = mf.get_weight()[ispin](ik, nocc) * scale_spin;
            if (weight < occ_thres) break;
            weights.push_back(weight);
        }

        scaled_wfc_ket = C_ZERO;
        if (wfc_ket != nullptr && wfc_size_loc > 0)
            std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c, wfc_size_loc * sizeof(cplxdb));
        const int wfc_m_loc = desc_wfc.m_loc();
        std::vector<char> scale_wfc_col(desc_wfc.n_loc(), false);
        std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 1.0);
        for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
        {
            const int jglob = desc_wfc.indx_l2g_c(jloc);
            if (jglob < 0 || jglob >= nocc) continue;
            scale_wfc_col[jloc] = true;
            wfc_col_scale[jloc] = weights[jglob];
        }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
        for (size_t i = 0; i < wfc_size_loc; ++i)
        {
            const int jloc = static_cast<int>(i / wfc_m_loc);
            if (scale_wfc_col[jloc])
                scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
        }

        dmat_k = C_ZERO;
        if (nocc > 0)
        {
            const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
            pgemm_wfc_scaled_wfc_h(n_aos, nocc, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                   desc_wfc, wfc_gemm_workspace, dmat_k, desc_dm,
                                   kblacs_ctxt.blacs_h);
        }
#pragma omp parallel for schedule(static) if (n_elem > 4096)
        for (int i = 0; i != n_elem; ++i)
        {
            kmat.ptr()[ik_local + static_cast<size_t>(i) * nk_local] = dmat_k.ptr()[i];
        }
    }

    for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
    {
        const int nR_this = n_Rs_all[pid];
        if (nR_this < 1) continue;

        Matz transmat(nR_this, nk_local, MAJOR::COL);
        const size_t n_trans = static_cast<size_t>(nR_this) * nk_local;
#pragma omp parallel for schedule(static) if (n_trans > 4096)
        for (size_t i = 0; i < n_trans; ++i)
        {
            const int ik_local = static_cast<int>(i / nR_this);
            const int iR = static_cast<int>(i % nR_this);
            const auto &kf = kfrac_list[iks_local[ik_local]];
            const int index = pid * nR_max * 3 + iR * 3;
            const auto ang = - (kf.x * Rs_all[index] +
                                kf.y * Rs_all[index + 1] +
                                kf.z * Rs_all[index + 2]) * TWO_PI;
            transmat.ptr()[iR + static_cast<size_t>(ik_local) * nR_this] =
                cplxdb{std::cos(ang), std::sin(ang)};
        }

        Matz rmat(nR_this, n_elem, MAJOR::COL);
        if (nk_local > 0 && n_elem > 0)
        {
            LapackConnector::gemm_f('N', 'N', nR_this, n_elem, nk_local, C_ONE,
                                    transmat.ptr(), nR_this, kmat.ptr(), nk_local, C_ZERO,
                                    rmat.ptr(), nR_this);
        }

        const size_t count = static_cast<size_t>(nR_this) * n_elem;
        if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw LIBRPA_RUNTIME_ERROR("local density-matrix Fourier block is too large for MPI collectives");
        const int count_int = static_cast<int>(count);
        if (kblacs_ctxt.comm_kpoint_h.myid == pid)
        {
            kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
            for (int iR = 0; iR != nR_this; ++iR)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1], Rs_all[index + 2]};
                auto &m = dmat_Rs[R];
                m.resize(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                const auto *rmat_iR = rmat.ptr() + iR;
#pragma omp parallel for schedule(static) if (n_elem > 4096)
                for (int i = 0; i != n_elem; ++i)
                {
                    m.ptr()[i] = rmat_iR[static_cast<size_t>(i) * nR_this];
                }
            }
        }
        else
        {
            kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
        }
    }

    global::profiler.stop(__FUNCTION__);
    return dmat_Rs;
}

std::map<Vector3_Order<int>, Matz> get_dmat_cplx_Rs_kblacs_para(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm)
{
    return get_dmat_cplx_Rs_kblacs_para(ispin, 0, 0, mf, kfrac_list, Rs, kblacs_ctxt, desc_wfc, desc_dm);
}

std::map<Vector3_Order<int>, Matz> get_symmetry_restored_dmat_cplx_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const SymmetryContext &symmetry_context, const PeriodicBoundaryData &pbc,
    const AtomicBasis &atbasis_wfc)
{
    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("density-matrix descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and density-matrix descriptors must use the same BLACS context");

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    if (!can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw))
    {
        auto direct = get_dmat_cplx_Rs_kblacs_para(
            ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, Rs,
            kblacs_ctxt, desc_wfc, desc_dm);
        return direct;
    }

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<Vector3_Order<int>, Matz> dmat_Rs;
    for (const auto &R : Rs)
    {
        dmat_Rs.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
    }
    if (nR_max == 0)
    {
        return dmat_Rs;
    }

    const auto target_pair_distribution =
        get_balanced_ap_distribution_for_consec_descriptor(
            atbasis_wfc, atbasis_wfc, desc_dm, true);
    IndexScheduler target_sched;
    target_sched.init(target_pair_distribution, atbasis_wfc, atbasis_wfc, desc_dm, false);

    std::set<atpair_t> target_pairs_local_set(target_sched.atpairs.cbegin(),
                                              target_sched.atpairs.cend());
    std::map<atpair_t, std::size_t> target_pair_offsets;
    std::size_t block_size_total = 0;
    for (const auto &pair : target_sched.atpairs)
    {
        target_pair_offsets[pair] = block_size_total;
        block_size_total += atbasis_wfc.get_atom_nb(pair.first)
                          * atbasis_wfc.get_atom_nb(pair.second);
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    const size_t dm_size_loc = static_cast<size_t>(desc_dm.m_loc()) * desc_dm.n_loc();
    if (dm_size_loc > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local density-matrix block is too large for MPI collectives");

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz dmat_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    const double occ_thres = 1e-4 / n_kpoints;
    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    const auto member_kfrac_targets =
        build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc);

    for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
    {
        const int nR_this = n_Rs_all[pid];
        if (nR_this < 1) continue;

        const std::size_t count =
            static_cast<std::size_t>(nR_this) * block_size_total;
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        {
            throw LIBRPA_RUNTIME_ERROR(
                "local symmetry-restored density-matrix block is too large for MPI collectives");
        }
        std::vector<cplxdb> buffer(count, C_ZERO);

        for (int ik_local = 0; ik_local != nk_local; ++ik_local)
        {
            const int ik = iks_local[ik_local];
            const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
            const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
            if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
                throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                          std::to_string(ik));
            if (wfc_size_loc > 0 && wfc_bra != nullptr &&
                static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
            if (wfc_size_loc > 0 && wfc_ket != nullptr &&
                static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

            int nocc = 0;
            std::vector<double> weights;
            weights.reserve(n_states);
            for (; nocc != n_states; ++nocc)
            {
                const double weight = mf.get_weight()[ispin](ik, nocc) * scale_spin;
                if (weight < occ_thres) break;
                weights.push_back(weight);
            }

            scaled_wfc_ket = C_ZERO;
            if (wfc_ket != nullptr && wfc_size_loc > 0)
                std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c,
                            wfc_size_loc * sizeof(cplxdb));
            const int wfc_m_loc = desc_wfc.m_loc();
            std::vector<char> scale_wfc_col(desc_wfc.n_loc(), false);
            std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 1.0);
            for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
            {
                const int jglob = desc_wfc.indx_l2g_c(jloc);
                if (jglob < 0 || jglob >= nocc) continue;
                scale_wfc_col[jloc] = true;
                wfc_col_scale[jloc] = weights[jglob];
            }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
            for (size_t i = 0; i < wfc_size_loc; ++i)
            {
                const int jloc = static_cast<int>(i / wfc_m_loc);
                if (scale_wfc_col[jloc])
                    scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
            }

            dmat_k = C_ZERO;
            if (nocc > 0)
            {
                const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
                pgemm_wfc_scaled_wfc_h(n_aos, nocc, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                       desc_wfc, wfc_gemm_workspace, dmat_k, desc_dm,
                                       kblacs_ctxt.blacs_h);
            }

            const auto &k_ibz = kfrac_list[static_cast<std::size_t>(ik)];
            const auto &star = find_symmetry_kstar_for_ibz_kpoint(symmetry_context, k_ibz);
            const auto star_factor =
                1.0 / static_cast<double>(star.members.size());

            for (std::size_t imember = 0; imember != star.members.size(); ++imember)
            {
                const auto &member = star.members[imember];
                const Vector3_Order<double> *k_bz_target = nullptr;
                if (!member_kfrac_targets.empty()
                    && static_cast<std::size_t>(ik) < member_kfrac_targets.size()
                    && imember < member_kfrac_targets[static_cast<std::size_t>(ik)].size())
                {
                    k_bz_target = &member_kfrac_targets[static_cast<std::size_t>(ik)][imember];
                }
                const auto &k_for_phase = k_bz_target == nullptr ? member.k_bz : *k_bz_target;
                const auto source_pair_requests = build_source_pair_requests(
                    target_pair_distribution, member, atbasis_wfc.n_atoms);
                const auto source_pair_mats = get_ap_map_from_blacs_dist(
                    dmat_k, source_pair_requests, atbasis_wfc, atbasis_wfc, desc_dm);

                symmetry_atom_block_matrix_map_t source_blocks;
                for (const auto &[pair, mat] : source_pair_mats)
                {
                    source_blocks[pair.first][pair.second] = matz_to_complex_matrix(mat);
                }

                auto rotated_blocks = rotate_wfc_product_blocks(
                    symmetry_context, wfc_layouts, member, source_blocks, atom_nw,
                    k_ibz, &target_pairs_local_set, k_bz_target);

                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    const Vector3_Order<int> R{Rs_all[index],
                                               Rs_all[index + 1],
                                               Rs_all[index + 2]};
                    const std::size_t R_offset =
                        static_cast<std::size_t>(iR) * block_size_total;
                    for (const auto &atom_i_blocks : rotated_blocks)
                    {
                        const auto atom_i = atom_i_blocks.first;
                        for (const auto &atom_j_block : atom_i_blocks.second)
                        {
                            const atpair_t pair{atom_i, atom_j_block.first};
                            const auto offset_iter = target_pair_offsets.find(pair);
                            if (offset_iter == target_pair_offsets.end()) continue;
                            const auto angle = -(k_for_phase * R) * TWO_PI;
                            const cplxdb phase{std::cos(angle), std::sin(angle)};
                            const cplxdb factor = star_factor * phase;
                            const auto &block = atom_j_block.second;
                            const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                            const std::size_t block_offset = R_offset + offset_iter->second;
                            for (int i = 0; i != block.nr; ++i)
                            {
                                for (int j = 0; j != block.nc; ++j)
                                {
                                    buffer[block_offset + static_cast<std::size_t>(i) * n_J + j] +=
                                        factor * block(i, j);
                                }
                            }
                        }
                    }
                }
            }
        }

        if (kblacs_ctxt.comm_kpoint_h.myid == pid)
        {
            kblacs_ctxt.comm_kpoint_h.reduce(
                MPI_IN_PLACE, buffer.data(), static_cast<int>(count), pid, MPI_SUM);
            for (int iR = 0; iR != nR_this; ++iR)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                const Vector3_Order<int> R{Rs_all[index],
                                           Rs_all[index + 1],
                                           Rs_all[index + 2]};
                ap_p_map<Matz> dmat_ap;
                const std::size_t R_offset =
                    static_cast<std::size_t>(iR) * block_size_total;
                for (const auto &pair : target_sched.atpairs)
                {
                    const auto n_I = atbasis_wfc.get_atom_nb(pair.first);
                    const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                    Matz block(n_I, n_J, MAJOR::COL);
                    const std::size_t block_offset =
                        R_offset + target_pair_offsets.at(pair);
                    for (std::size_t i = 0; i != n_I; ++i)
                    {
                        for (std::size_t j = 0; j != n_J; ++j)
                        {
                            block(i, j) = buffer[block_offset + i * n_J + j];
                        }
                    }
                    dmat_ap[pair] = std::move(block);
                }
                auto &mat = dmat_Rs.at(R);
                mat = C_ZERO;
                fill_local_mat_from_ap_dist_scheduler(
                    mat, dmat_ap, target_sched, atbasis_wfc, atbasis_wfc, desc_dm);
            }
        }
        else
        {
            kblacs_ctxt.comm_kpoint_h.reduce(
                buffer.data(), buffer.data(), static_cast<int>(count), pid, MPI_SUM);
        }
    }

    return dmat_Rs;
}

std::map<double, std::map<Vector3_Order<int>, Matz>> get_gf_cplx_imagtimes_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const std::vector<bool> &band_mask)
{
    if (!band_mask.empty() && band_mask.size() != static_cast<std::size_t>(mf.get_n_bands()))
        throw LIBRPA_RUNTIME_ERROR("Green's-function band mask has inconsistent size");

    global::profiler.start(__FUNCTION__, LIBRPA_VERBOSE_DEBUG);

    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("Green's-function descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and Green's-function descriptors must use the same BLACS context");

    global::profiler.start("gf_check_same_imagtimes", LIBRPA_VERBOSE_DEBUG);
    check_same_imagtimes(imagtimes, kblacs_ctxt.comm_global_h);
    global::profiler.stop("gf_check_same_imagtimes");

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    global::profiler.start("gf_collect_Rs", LIBRPA_VERBOSE_DEBUG);
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);
    global::profiler.stop("gf_collect_Rs");

    std::map<double, std::map<Vector3_Order<int>, Matz>> gf;
    if (imagtimes.empty())
    {
        global::profiler.stop(__FUNCTION__);
        return gf;
    }
    if (nR_max == 0)
    {
        for (const auto tau: imagtimes) gf.emplace(tau, std::map<Vector3_Order<int>, Matz>{});
        global::profiler.stop(__FUNCTION__);
        return gf;
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    const size_t gf_size_loc = static_cast<size_t>(desc_dm.m_loc()) * desc_dm.n_loc();
    if (gf_size_loc > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local Green's-function block is too large for MPI collectives");
    const int n_elem = static_cast<int>(gf_size_loc);

    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    global::profiler.start("gf_check_kpoint_distribution", LIBRPA_VERBOSE_DEBUG);
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    global::profiler.stop("gf_check_kpoint_distribution");
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    global::profiler.start("gf_workspace_init", LIBRPA_VERBOSE_DEBUG);
    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz gf_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    Matz kmat(nk_local, n_elem, MAJOR::COL);
    global::profiler.stop("gf_workspace_init");

    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    for (const auto tau: imagtimes)
    {
        std::map<Vector3_Order<int>, Matz> gf_tau;
        for (const auto &R: Rs)
        {
            gf_tau.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
        }

        for (int ik_local = 0; ik_local != nk_local; ++ik_local)
        {
            const int ik = iks_local[ik_local];
            const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
            const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
            if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
                throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                          std::to_string(ik));
            if (wfc_size_loc > 0 && wfc_bra != nullptr &&
                static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
            if (wfc_size_loc > 0 && wfc_ket != nullptr &&
                static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

            global::profiler.start("gf_scale_wfc", LIBRPA_VERBOSE_DEBUG);
            std::vector<double> scales(n_states);
#pragma omp parallel for schedule(static) if (n_states > 64)
            for (int ib = 0; ib != n_states; ++ib)
            {
                const double wg_occ = mf.get_weight()[ispin](ik, ib) * scale_spin;
                double wg_empty = 1.0 / n_kpoints - wg_occ;
                if (wg_empty < 0.0) wg_empty = 0.0;
                const double prefac = tau > 0 ? wg_empty : wg_occ;
                double scale = -tau * (mf.get_eigenvals()[ispin](ik, ib) - mf.get_efermi());
                if (scale > 0.0) scale = 0.0;
                scales[ib] = !band_mask.empty() && !band_mask[ib] ? 0.0 : std::exp(scale) * prefac;
            }

            scaled_wfc_ket = C_ZERO;
            if (wfc_ket != nullptr && wfc_size_loc > 0)
                std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c, wfc_size_loc * sizeof(cplxdb));
            const int wfc_m_loc = desc_wfc.m_loc();
            std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 0.0);
            for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
            {
                const int jglob = desc_wfc.indx_l2g_c(jloc);
                if (jglob < 0 || jglob >= n_states) continue;
                wfc_col_scale[jloc] = scales[jglob];
            }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
            for (size_t i = 0; i < wfc_size_loc; ++i)
            {
                const int jloc = static_cast<int>(i / wfc_m_loc);
                scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
            }

            gf_k = C_ZERO;
            global::profiler.stop("gf_scale_wfc");
            const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
            pgemm_wfc_scaled_wfc_h(n_aos, n_states, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                   desc_wfc, wfc_gemm_workspace, gf_k, desc_dm,
                                   kblacs_ctxt.blacs_h);
            global::profiler.start("gf_pack_kmat", LIBRPA_VERBOSE_DEBUG);
#pragma omp parallel for schedule(static) if (n_elem > 4096)
            for (int i = 0; i != n_elem; ++i)
            {
                kmat.ptr()[ik_local + static_cast<size_t>(i) * nk_local] = gf_k.ptr()[i];
            }
            global::profiler.stop("gf_pack_kmat");
        }

        for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
        {
            const int nR_this = n_Rs_all[pid];
            if (nR_this < 1) continue;

            global::profiler.start("gf_k_to_R_local", LIBRPA_VERBOSE_DEBUG);
            Matz transmat(nR_this, nk_local, MAJOR::COL);
            const double tau_sign = tau > 0 ? 1.0 : -1.0;
            const size_t n_trans = static_cast<size_t>(nR_this) * nk_local;
#pragma omp parallel for schedule(static) if (n_trans > 4096)
            for (size_t i = 0; i < n_trans; ++i)
            {
                const int ik_local = static_cast<int>(i / nR_this);
                const int iR = static_cast<int>(i % nR_this);
                const auto &kf = kfrac_list[iks_local[ik_local]];
                const int index = pid * nR_max * 3 + iR * 3;
                const auto ang = - (kf.x * Rs_all[index] +
                                    kf.y * Rs_all[index + 1] +
                                    kf.z * Rs_all[index + 2]) * TWO_PI;
                transmat.ptr()[iR + static_cast<size_t>(ik_local) * nR_this] =
                    tau_sign * cplxdb{std::cos(ang), std::sin(ang)};
            }

            Matz rmat(nR_this, n_elem, MAJOR::COL);
            if (nk_local > 0 && n_elem > 0)
            {
                LapackConnector::gemm_f('N', 'N', nR_this, n_elem, nk_local, C_ONE,
                                        transmat.ptr(), nR_this, kmat.ptr(), nk_local, C_ZERO,
                                        rmat.ptr(), nR_this);
            }
            global::profiler.stop("gf_k_to_R_local");

            const size_t count = static_cast<size_t>(nR_this) * n_elem;
            if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
                throw LIBRPA_RUNTIME_ERROR("local Green's-function Fourier block is too large for MPI collectives");
            const int count_int = static_cast<int>(count);
            global::profiler.start("gf_k_to_R_reduce", LIBRPA_VERBOSE_DEBUG);
            if (kblacs_ctxt.comm_kpoint_h.myid == pid)
            {
                kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
            }
            else
            {
                kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, rmat.ptr(), count_int, pid, MPI_SUM);
            }
            global::profiler.stop("gf_k_to_R_reduce");
            if (kblacs_ctxt.comm_kpoint_h.myid == pid)
            {
                global::profiler.start("gf_store_R", LIBRPA_VERBOSE_DEBUG);
                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1], Rs_all[index + 2]};
                    auto &m = gf_tau[R];
                    m.resize(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                    const auto *rmat_iR = rmat.ptr() + iR;
#pragma omp parallel for schedule(static) if (n_elem > 4096)
                    for (int i = 0; i != n_elem; ++i)
                    {
                        m.ptr()[i] = rmat_iR[static_cast<size_t>(i) * nR_this];
                    }
                }
                global::profiler.stop("gf_store_R");
            }
        }
        gf.emplace(tau, std::move(gf_tau));
    }

    global::profiler.stop(__FUNCTION__);
    return gf;
}

std::map<double, std::map<Vector3_Order<int>, Matz>>
get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para(
    int ispin, int ispinor_bra, int ispinor_ket, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const SymmetryContext &symmetry_context, const PeriodicBoundaryData &pbc,
    const AtomicBasis &atbasis_wfc, const std::vector<bool> &band_mask)
{
    if (!band_mask.empty() && band_mask.size() != static_cast<std::size_t>(mf.get_n_bands()))
        throw LIBRPA_RUNTIME_ERROR("Green's-function band mask has inconsistent size");

    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("Green's-function descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and Green's-function descriptors must use the same BLACS context");

    check_same_imagtimes(imagtimes, kblacs_ctxt.comm_global_h);

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    if (!can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw))
    {
        return get_gf_cplx_imagtimes_Rs_kblacs_para(
            ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, imagtimes, Rs,
            kblacs_ctxt, desc_wfc, desc_dm, band_mask);
    }

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<double, std::map<Vector3_Order<int>, Matz>> gf;
    for (const auto tau : imagtimes)
    {
        auto &gf_tau = gf[tau];
        for (const auto &R : Rs)
        {
            gf_tau.emplace(R, Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL));
        }
    }
    if (imagtimes.empty() || nR_max == 0)
    {
        return gf;
    }

    const auto target_pair_distribution =
        get_balanced_ap_distribution_for_consec_descriptor(
            atbasis_wfc, atbasis_wfc, desc_dm, true);
    IndexScheduler target_sched;
    target_sched.init(target_pair_distribution, atbasis_wfc, atbasis_wfc, desc_dm, false);

    std::set<atpair_t> target_pairs_local_set(target_sched.atpairs.cbegin(),
                                              target_sched.atpairs.cend());
    std::map<atpair_t, std::size_t> target_pair_offsets;
    std::size_t block_size_total = 0;
    for (const auto &pair : target_sched.atpairs)
    {
        target_pair_offsets[pair] = block_size_total;
        block_size_total += atbasis_wfc.get_atom_nb(pair.first)
                          * atbasis_wfc.get_atom_nb(pair.second);
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    if (block_size_total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("local Green's-function atom-pair block is too large for MPI collectives");

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    Matz gf_k(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    const double full_k_count =
        static_cast<double>(symmetry_context.count_kstar_members());
    if (full_k_count <= 0.0)
        throw LIBRPA_RUNTIME_ERROR("k-star restore found zero full-k members");
    const auto member_kfrac_targets =
        build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc);

    for (const auto tau : imagtimes)
    {
        const double tau_sign = tau > 0.0 ? 1.0 : -1.0;
        for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
        {
            const int nR_this = n_Rs_all[pid];
            if (nR_this < 1) continue;

            const std::size_t count =
                static_cast<std::size_t>(nR_this) * block_size_total;
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "local symmetry-restored Green's-function block is too large for MPI collectives");
            }
            std::vector<cplxdb> buffer(count, C_ZERO);

            for (int ik_local = 0; ik_local != nk_local; ++ik_local)
            {
                const int ik = iks_local[ik_local];
                const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
                const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
                if ((wfc_bra == nullptr || wfc_ket == nullptr) && wfc_size_loc > 0)
                    throw LIBRPA_RUNTIME_ERROR("missing local wave-function block for k-point " +
                                              std::to_string(ik));
                if (wfc_size_loc > 0 && wfc_bra != nullptr &&
                    static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                    throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
                if (wfc_size_loc > 0 && wfc_ket != nullptr &&
                    static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                    throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

                const auto &k_ibz = kfrac_list[static_cast<std::size_t>(ik)];
                const auto &star = find_symmetry_kstar_for_ibz_kpoint(symmetry_context, k_ibz);
                const auto star_factor =
                    1.0 / static_cast<double>(star.members.size());
                const auto kpoint_weight =
                    static_cast<double>(star.members.size()) / full_k_count;

                std::vector<double> scales(n_states);
#pragma omp parallel for schedule(static) if (n_states > 64)
                for (int ib = 0; ib != n_states; ++ib)
                {
                    const double wg_occ = mf.get_weight()[ispin](ik, ib) * scale_spin;
                    const double prefac = tau > 0.0
                        ? std::max(0.0, kpoint_weight - wg_occ)
                        : wg_occ;
                    double scale = -tau * (mf.get_eigenvals()[ispin](ik, ib) - mf.get_efermi());
                    if (scale > 0.0) scale = 0.0;
                    scales[ib] = !band_mask.empty() && !band_mask[ib] ? 0.0 : std::exp(scale) * prefac;
                }

                scaled_wfc_ket = C_ZERO;
                if (wfc_ket != nullptr && wfc_size_loc > 0)
                    std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c,
                                wfc_size_loc * sizeof(cplxdb));
                const int wfc_m_loc = desc_wfc.m_loc();
                std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 0.0);
                for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
                {
                    const int jglob = desc_wfc.indx_l2g_c(jloc);
                    if (jglob < 0 || jglob >= n_states) continue;
                    wfc_col_scale[jloc] = scales[jglob];
                }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
                for (size_t i = 0; i < wfc_size_loc; ++i)
                {
                    const int jloc = static_cast<int>(i / wfc_m_loc);
                    scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
                }

                gf_k = C_ZERO;
                const cplxdb *wfc_bra_ptr = wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
                pgemm_wfc_scaled_wfc_h(n_aos, n_states, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                       desc_wfc, wfc_gemm_workspace, gf_k, desc_dm,
                                       kblacs_ctxt.blacs_h);

                for (std::size_t imember = 0; imember != star.members.size(); ++imember)
                {
                    const auto &member = star.members[imember];
                    const Vector3_Order<double> *k_bz_target = nullptr;
                    if (!member_kfrac_targets.empty()
                        && static_cast<std::size_t>(ik) < member_kfrac_targets.size()
                        && imember < member_kfrac_targets[static_cast<std::size_t>(ik)].size())
                    {
                        k_bz_target = &member_kfrac_targets[static_cast<std::size_t>(ik)][imember];
                    }
                    const auto &k_for_phase = k_bz_target == nullptr ? member.k_bz : *k_bz_target;
                    const auto source_pair_requests = build_source_pair_requests(
                        target_pair_distribution, member, atbasis_wfc.n_atoms);
                    const auto source_pair_mats = get_ap_map_from_blacs_dist(
                        gf_k, source_pair_requests, atbasis_wfc, atbasis_wfc, desc_dm);

                    symmetry_atom_block_matrix_map_t source_blocks;
                    for (const auto &[pair, mat] : source_pair_mats)
                    {
                        source_blocks[pair.first][pair.second] = matz_to_complex_matrix(mat);
                    }

                    auto rotated_blocks = rotate_wfc_product_blocks(
                        symmetry_context, wfc_layouts, member, source_blocks, atom_nw,
                        k_ibz, &target_pairs_local_set, k_bz_target);

                    for (int iR = 0; iR != nR_this; ++iR)
                    {
                        const int index = pid * nR_max * 3 + iR * 3;
                        const Vector3_Order<int> R{Rs_all[index],
                                                   Rs_all[index + 1],
                                                   Rs_all[index + 2]};
                        const auto angle = -(k_for_phase * R) * TWO_PI;
                        const cplxdb phase{std::cos(angle), std::sin(angle)};
                        const cplxdb factor = star_factor * tau_sign * phase;
                        const std::size_t R_offset =
                            static_cast<std::size_t>(iR) * block_size_total;
                        for (const auto &atom_i_blocks : rotated_blocks)
                        {
                            const auto atom_i = atom_i_blocks.first;
                            for (const auto &atom_j_block : atom_i_blocks.second)
                            {
                                const atpair_t pair{atom_i, atom_j_block.first};
                                const auto offset_iter = target_pair_offsets.find(pair);
                                if (offset_iter == target_pair_offsets.end()) continue;
                                const auto &block = atom_j_block.second;
                                const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                                const std::size_t block_offset = R_offset + offset_iter->second;
                                for (int i = 0; i != block.nr; ++i)
                                {
                                    for (int j = 0; j != block.nc; ++j)
                                    {
                                        buffer[block_offset + static_cast<std::size_t>(i) * n_J + j] +=
                                            factor * block(i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (kblacs_ctxt.comm_kpoint_h.myid == pid)
            {
                kblacs_ctxt.comm_kpoint_h.reduce(
                    MPI_IN_PLACE, buffer.data(), static_cast<int>(count), pid, MPI_SUM);
                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    const Vector3_Order<int> R{Rs_all[index],
                                               Rs_all[index + 1],
                                               Rs_all[index + 2]};
                    ap_p_map<Matz> gf_ap;
                    const std::size_t R_offset =
                        static_cast<std::size_t>(iR) * block_size_total;
                    for (const auto &pair : target_sched.atpairs)
                    {
                        const auto n_I = atbasis_wfc.get_atom_nb(pair.first);
                        const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                        Matz block(n_I, n_J, MAJOR::COL);
                        const std::size_t block_offset =
                            R_offset + target_pair_offsets.at(pair);
                        for (std::size_t i = 0; i != n_I; ++i)
                        {
                            for (std::size_t j = 0; j != n_J; ++j)
                            {
                                block(i, j) = buffer[block_offset + i * n_J + j];
                            }
                        }
                        gf_ap[pair] = std::move(block);
                    }
                    auto &mat = gf.at(tau).at(R);
                    mat = C_ZERO;
                    fill_local_mat_from_ap_dist_scheduler(
                        mat, gf_ap, target_sched, atbasis_wfc, atbasis_wfc, desc_dm);
                }
            }
            else
            {
                kblacs_ctxt.comm_kpoint_h.reduce(
                    buffer.data(), buffer.data(), static_cast<int>(count), pid, MPI_SUM);
            }
        }
    }

    return gf;
}

std::map<double, std::map<Vector3_Order<int>, Matz>> get_gf_cplx_imagtimes_Rs_kblacs_para(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const std::vector<bool> &band_mask)
{
    return get_gf_cplx_imagtimes_Rs_kblacs_para(ispin, 0, 0, mf, kfrac_list, imagtimes, Rs, kblacs_ctxt, desc_wfc, desc_dm, band_mask);
}

//! Fetch one atom-pair block from a block map, zero-filled when absent.
static ComplexMatrix get_block_or_zero(const symmetry_atom_block_matrix_map_t &blocks,
                                       const atom_t atom_i, const atom_t atom_j,
                                       const int n_i, const int n_j)
{
    const auto it_i = blocks.find(atom_i);
    if (it_i != blocks.end())
    {
        const auto it_j = it_i->second.find(atom_j);
        if (it_j != it_i->second.end())
        {
            return it_j->second;
        }
    }
    ComplexMatrix zero(n_i, n_j);
    zero.zero_out();
    return zero;
}

std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<Matz>>>
get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para_spinor(
    int ispin, const MeanField &mf,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs,
    const KPointBlacsParallelContext &kblacs_ctxt, const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const SymmetryContext &symmetry_context, const PeriodicBoundaryData &pbc,
    const AtomicBasis &atbasis_wfc)
{
    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (mf.get_n_spinor() != 2)
        throw LIBRPA_RUNTIME_ERROR("spinor k-BLACS Green's-function restore requires n_spinor == 2");
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("Green's-function descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR("wave-function and Green's-function descriptors must use the same BLACS context");

    check_same_imagtimes(imagtimes, kblacs_ctxt.comm_global_h);

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    if (!can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "spinor k-BLACS Green's-function restore requires a restorable k-star context");
    }

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<Matz>>> gf;
    for (const auto tau : imagtimes)
    {
        auto &gf_tau = gf[tau];
        for (const auto &R : Rs)
        {
            SpinorBlocks4<Matz> blocks;
            blocks.b00 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
            blocks.b01 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
            blocks.b10 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
            blocks.b11 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
            gf_tau.emplace(R, std::move(blocks));
        }
    }
    if (imagtimes.empty() || nR_max == 0)
    {
        return gf;
    }

    const auto target_pair_distribution =
        get_balanced_ap_distribution_for_consec_descriptor(
            atbasis_wfc, atbasis_wfc, desc_dm, true);
    IndexScheduler target_sched;
    target_sched.init(target_pair_distribution, atbasis_wfc, atbasis_wfc, desc_dm, false);

    std::set<atpair_t> target_pairs_local_set(target_sched.atpairs.cbegin(),
                                              target_sched.atpairs.cend());
    std::map<atpair_t, std::size_t> target_pair_offsets;
    std::size_t block_size_total = 0;
    for (const auto &pair : target_sched.atpairs)
    {
        target_pair_offsets[pair] = block_size_total;
        block_size_total += atbasis_wfc.get_atom_nb(pair.first)
                          * atbasis_wfc.get_atom_nb(pair.second);
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    if (block_size_total > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 4)
        throw LIBRPA_RUNTIME_ERROR("local Green's-function atom-pair block is too large for MPI collectives");

    std::vector<cplxdb> dummy(1, C_ZERO);
    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    // Zero stand-in for a missing (bra, ket) wfc channel: keeps the pgemm
    // collective intact while contributing a zero source block.
    Matz zero_wfc_stand_in(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    zero_wfc_stand_in = C_ZERO;
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm,
                                                        kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    const double full_k_count =
        static_cast<double>(symmetry_context.count_kstar_members());
    if (full_k_count <= 0.0)
        throw LIBRPA_RUNTIME_ERROR("k-star restore found zero full-k members");
    const auto member_kfrac_targets =
        build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc);

    // Identity orbital action for the per-atom-pair kernel call: the spatial
    // rotation is applied per channel beforehand, so the kernel only performs
    // the SU(2) mixing and the Theta remap.
    const auto identity_orbit = [](std::size_t spatial_id, const ComplexMatrix &block) {
        (void)spatial_id;
        return block;
    };

    for (const auto tau : imagtimes)
    {
        const double tau_sign = tau > 0.0 ? 1.0 : -1.0;
        for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
        {
            const int nR_this = n_Rs_all[pid];
            if (nR_this < 1) continue;

            const std::size_t count =
                static_cast<std::size_t>(nR_this) * block_size_total;
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 4)
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "local symmetry-restored Green's-function block is too large for MPI collectives");
            }
            // Four channels packed channel-major into one buffer: the reduce
            // rounds are unchanged from the scalar path.
            std::vector<cplxdb> buffer(4 * count, C_ZERO);

            for (int ik_local = 0; ik_local != nk_local; ++ik_local)
            {
                const int ik = iks_local[ik_local];
                const auto &k_ibz = kfrac_list[static_cast<std::size_t>(ik)];
                const auto &star = find_symmetry_kstar_for_ibz_kpoint(symmetry_context, k_ibz);
                const auto star_factor =
                    1.0 / static_cast<double>(star.members.size());
                const auto kpoint_weight =
                    static_cast<double>(star.members.size()) / full_k_count;

                std::vector<double> scales(n_states);
#pragma omp parallel for schedule(static) if (n_states > 64)
                for (int ib = 0; ib != n_states; ++ib)
                {
                    const double wg_occ = mf.get_weight()[ispin](ik, ib) * scale_spin;
                    const double prefac = tau > 0.0
                        ? std::max(0.0, kpoint_weight - wg_occ)
                        : wg_occ;
                    double scale = -tau * (mf.get_eigenvals()[ispin](ik, ib) - mf.get_efermi());
                    if (scale > 0.0) scale = 0.0;
                    scales[ib] = std::exp(scale) * prefac;
                }

                // Four source blocks G^{ab}(k_ibz, tau), zero-filled when a
                // (bra, ket) wfc channel is missing. pgemm is collective over
                // the k-group BLACS grid, so it is called unconditionally with
                // a zero stand-in block (the scalar path skips the collective
                // only because it can throw on missing wfc; spinor zero-fill
                // must not break the collective).
                Matz gf_k[4];
                for (int s = 0; s != 4; ++s)
                {
                    gf_k[s] = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                    gf_k[s] = C_ZERO;
                    const int ispinor_bra = s / 2;
                    const int ispinor_ket = s % 2;
                    const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
                    const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
                    if (wfc_size_loc > 0 && wfc_bra != nullptr &&
                        static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                        throw LIBRPA_RUNTIME_ERROR("wave-function bra block size is inconsistent with descriptor");
                    if (wfc_size_loc > 0 && wfc_ket != nullptr &&
                        static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                        throw LIBRPA_RUNTIME_ERROR("wave-function ket block size is inconsistent with descriptor");

                    scaled_wfc_ket = C_ZERO;
                    if (wfc_ket != nullptr && wfc_size_loc > 0)
                        std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c,
                                    wfc_size_loc * sizeof(cplxdb));
                    const int wfc_m_loc = desc_wfc.m_loc();
                    std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 0.0);
                    for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
                    {
                        const int jglob = desc_wfc.indx_l2g_c(jloc);
                        if (jglob < 0 || jglob >= n_states) continue;
                        wfc_col_scale[jloc] = scales[jglob];
                    }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
                    for (size_t i = 0; i < wfc_size_loc; ++i)
                    {
                        const int jloc = static_cast<int>(i / wfc_m_loc);
                        scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
                    }

                    const cplxdb *wfc_bra_ptr =
                        wfc_bra == nullptr ? zero_wfc_stand_in.ptr() : wfc_bra->c;
                    pgemm_wfc_scaled_wfc_h(n_aos, n_states, wfc_bra_ptr, scaled_wfc_ket.ptr(),
                                           desc_wfc, wfc_gemm_workspace, gf_k[s], desc_dm,
                                           kblacs_ctxt.blacs_h);
                }

                for (std::size_t imember = 0; imember != star.members.size(); ++imember)
                {
                    const auto &member = star.members[imember];
                    const auto &op = resolve_symmetry_kstar_member_spin_operation(
                        symmetry_context, member);
                    const Vector3_Order<double> *k_bz_target = nullptr;
                    if (!member_kfrac_targets.empty()
                        && static_cast<std::size_t>(ik) < member_kfrac_targets.size()
                        && imember < member_kfrac_targets[static_cast<std::size_t>(ik)].size())
                    {
                        k_bz_target = &member_kfrac_targets[static_cast<std::size_t>(ik)][imember];
                    }
                    const auto &k_for_phase = k_bz_target == nullptr ? member.k_bz : *k_bz_target;

                    // Spatial part only: time reversal is suppressed here and
                    // applied by the kernel remap; gauge phases are excluded
                    // and applied after the kernel as plain unitary factors.
                    SymmetryKStarMember spatial_member = member;
                    spatial_member.time_reversal = false;
                    const auto source_pair_requests = build_source_pair_requests(
                        target_pair_distribution, member, atbasis_wfc.n_atoms);
                    const auto gauge_phases = build_symmetry_kstar_member_target_gauge_phases(
                        symmetry_context, member, atom_nw.size(), k_bz_target);

                    symmetry_atom_block_matrix_map_t rotated[4];
                    for (int s = 0; s != 4; ++s)
                    {
                        const auto source_pair_mats = get_ap_map_from_blacs_dist(
                            gf_k[s], source_pair_requests, atbasis_wfc, atbasis_wfc, desc_dm);
                        symmetry_atom_block_matrix_map_t source_blocks;
                        for (const auto &[pair, mat] : source_pair_mats)
                        {
                            source_blocks[pair.first][pair.second] = matz_to_complex_matrix(mat);
                        }
                        rotated[s] = rotate_symmetry_kspace_operator_blocks(
                            symmetry_context, wfc_layouts, spatial_member, source_blocks,
                            atom_nw, k_ibz, false, &target_pairs_local_set, nullptr);
                    }

                    for (const auto &atom_i_blocks : rotated[0])
                    {
                        const auto atom_i = atom_i_blocks.first;
                        const auto n_I = atbasis_wfc.get_atom_nb(atom_i);
                        for (const auto &atom_j_block : atom_i_blocks.second)
                        {
                            const auto atom_j = atom_j_block.first;
                            const auto n_J = atbasis_wfc.get_atom_nb(atom_j);
                            const SpinorBlocks4<ComplexMatrix> pair_in{
                                get_block_or_zero(rotated[0], atom_i, atom_j, n_I, n_J),
                                get_block_or_zero(rotated[1], atom_i, atom_j, n_I, n_J),
                                get_block_or_zero(rotated[2], atom_i, atom_j, n_I, n_J),
                                get_block_or_zero(rotated[3], atom_i, atom_j, n_I, n_J),
                            };
                            auto pair_out = transform_spinor_bilinear(
                                op, pair_in, identity_orbit,
                                BilinearConvention::SourceToTarget_DXDdag);
                            const auto gauge_factor =
                                gauge_phases[atom_i] * std::conj(gauge_phases[atom_j]);
                            pair_out.b00 *= gauge_factor;
                            pair_out.b01 *= gauge_factor;
                            pair_out.b10 *= gauge_factor;
                            pair_out.b11 *= gauge_factor;

                            const atpair_t pair{atom_i, atom_j};
                            const auto offset_iter = target_pair_offsets.find(pair);
                            if (offset_iter == target_pair_offsets.end()) continue;
                            for (int iR = 0; iR != nR_this; ++iR)
                            {
                                const int index = pid * nR_max * 3 + iR * 3;
                                const Vector3_Order<int> R{Rs_all[index],
                                                           Rs_all[index + 1],
                                                           Rs_all[index + 2]};
                                const auto angle = -(k_for_phase * R) * TWO_PI;
                                const cplxdb phase{std::cos(angle), std::sin(angle)};
                                const cplxdb factor = star_factor * tau_sign * phase;
                                const ComplexMatrix *channel_blocks[4] = {
                                    &pair_out.b00, &pair_out.b01, &pair_out.b10, &pair_out.b11};
                                for (int s = 0; s != 4; ++s)
                                {
                                    const std::size_t block_offset =
                                        (static_cast<std::size_t>(s) * nR_this + iR)
                                            * block_size_total
                                        + offset_iter->second;
                                    const auto &block = *channel_blocks[s];
                                    for (int i = 0; i != block.nr; ++i)
                                    {
                                        for (int j = 0; j != block.nc; ++j)
                                        {
                                            buffer[block_offset + static_cast<std::size_t>(i) * n_J + j] +=
                                                factor * block(i, j);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            const int packed_count = static_cast<int>(4 * count);
            if (kblacs_ctxt.comm_kpoint_h.myid == pid)
            {
                kblacs_ctxt.comm_kpoint_h.reduce(
                    MPI_IN_PLACE, buffer.data(), packed_count, pid, MPI_SUM);
                for (int iR = 0; iR != nR_this; ++iR)
                {
                    const int index = pid * nR_max * 3 + iR * 3;
                    const Vector3_Order<int> R{Rs_all[index],
                                               Rs_all[index + 1],
                                               Rs_all[index + 2]};
                    auto &blocks = gf.at(tau).at(R);
                    Matz *channel_mats[4] = {&blocks.b00, &blocks.b01, &blocks.b10, &blocks.b11};
                    for (int s = 0; s != 4; ++s)
                    {
                        ap_p_map<Matz> gf_ap;
                        const std::size_t R_offset =
                            (static_cast<std::size_t>(s) * nR_this + iR) * block_size_total;
                        for (const auto &pair : target_sched.atpairs)
                        {
                            const auto n_I = atbasis_wfc.get_atom_nb(pair.first);
                            const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                            Matz block(n_I, n_J, MAJOR::COL);
                            const std::size_t block_offset =
                                R_offset + target_pair_offsets.at(pair);
                            for (std::size_t i = 0; i != n_I; ++i)
                            {
                                for (std::size_t j = 0; j != n_J; ++j)
                                {
                                    block(i, j) = buffer[block_offset + i * n_J + j];
                                }
                            }
                            gf_ap[pair] = std::move(block);
                        }
                        auto &mat = *channel_mats[s];
                        mat = C_ZERO;
                        fill_local_mat_from_ap_dist_scheduler(
                            mat, gf_ap, target_sched, atbasis_wfc, atbasis_wfc, desc_dm);
                    }
                }
            }
            else
            {
                kblacs_ctxt.comm_kpoint_h.reduce(
                    buffer.data(), buffer.data(), packed_count, pid, MPI_SUM);
            }
        }
    }

    return gf;
}

std::map<double, std::map<Vector3_Order<int>, Matz>>
extract_spinor_gf_block_kblacs(
    std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<Matz>>>&& gf_spinor,
    const int ispinor_bra, const int ispinor_ket)
{
    if (ispinor_bra < 0 || ispinor_bra > 1 || ispinor_ket < 0 || ispinor_ket > 1)
    {
        throw LIBRPA_RUNTIME_ERROR("spinor Green's-function block index out of range");
    }
    std::map<double, std::map<Vector3_Order<int>, Matz>> gf_block;
    for (auto &tau_entry : gf_spinor)
    {
        auto &gf_R = gf_block[tau_entry.first];
        for (auto &R_entry : tau_entry.second)
        {
            auto &blocks = R_entry.second;
            if (ispinor_bra == 0 && ispinor_ket == 0) gf_R[R_entry.first] = std::move(blocks.b00);
            else if (ispinor_bra == 0 && ispinor_ket == 1) gf_R[R_entry.first] = std::move(blocks.b01);
            else if (ispinor_bra == 1 && ispinor_ket == 0) gf_R[R_entry.first] = std::move(blocks.b10);
            else gf_R[R_entry.first] = std::move(blocks.b11);
        }
    }
    return gf_block;
}

std::map<Vector3_Order<int>, SpinorBlocks4<Matz>>
get_symmetry_restored_dmat_cplx_Rs_kblacs_para_spinor(
    int ispin, const MeanField &mf, const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<Vector3_Order<int>> &Rs, const KPointBlacsParallelContext &kblacs_ctxt,
    const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm, const SymmetryContext &symmetry_context,
    const PeriodicBoundaryData &pbc, const AtomicBasis &atbasis_wfc)
{
    global::profiler.start(__FUNCTION__);

    if (!kblacs_ctxt.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("KPointBlacsParallelContext is not initialized");

    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    const int n_kpoints = mf.get_n_kpoints();
    if (mf.get_n_spinor() != 2)
        throw LIBRPA_RUNTIME_ERROR("spinor k-BLACS density-matrix restore requires n_spinor == 2");
    if (static_cast<int>(kfrac_list.size()) != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point fractional coordinate list has inconsistent size");
    if (kblacs_ctxt.n_kpoints() != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of k-points");
    if (!desc_wfc.is_initialized() || !desc_dm.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("BLACS array descriptors are not initialized");
    if (desc_wfc.m() != n_aos || desc_wfc.n() != n_states)
        throw LIBRPA_RUNTIME_ERROR("wave-function descriptor must be n_aos x n_states");
    if (desc_dm.m() != n_aos || desc_dm.n() != n_aos)
        throw LIBRPA_RUNTIME_ERROR("density-matrix descriptor must be n_aos x n_aos");
    if (desc_wfc.ictxt() != desc_dm.ictxt())
        throw LIBRPA_RUNTIME_ERROR(
            "wave-function and density-matrix descriptors must use the same BLACS context");

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts =
        atbasis_wfc.has_l_shells()
            ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
            : std::vector<SpeciesBasisLayout>{};
    if (!can_restore_symmetry_kstar_meanfield(symmetry_context, wfc_layouts, mf, kfrac_list,
                                              atom_nw))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "spinor k-BLACS density-matrix restore requires a restorable k-star context");
    }

    std::vector<int> n_Rs_all, Rs_all;
    int nR_max;
    collect_Rs(Rs, n_Rs_all, Rs_all, nR_max, kblacs_ctxt.comm_kpoint_h);

    std::map<Vector3_Order<int>, SpinorBlocks4<Matz>> dmat_Rs;
    for (const auto &R : Rs)
    {
        SpinorBlocks4<Matz> blocks;
        blocks.b00 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
        blocks.b01 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
        blocks.b10 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
        blocks.b11 = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
        dmat_Rs.emplace(R, std::move(blocks));
    }
    if (nR_max == 0)
    {
        global::profiler.stop(__FUNCTION__);
        return dmat_Rs;
    }

    const auto target_pair_distribution =
        get_balanced_ap_distribution_for_consec_descriptor(atbasis_wfc, atbasis_wfc, desc_dm, true);
    IndexScheduler target_sched;
    target_sched.init(target_pair_distribution, atbasis_wfc, atbasis_wfc, desc_dm, false);

    std::set<atpair_t> target_pairs_local_set(target_sched.atpairs.cbegin(),
                                              target_sched.atpairs.cend());
    std::map<atpair_t, std::size_t> target_pair_offsets;
    std::size_t block_size_total = 0;
    for (const auto &pair : target_sched.atpairs)
    {
        target_pair_offsets[pair] = block_size_total;
        block_size_total +=
            atbasis_wfc.get_atom_nb(pair.first) * atbasis_wfc.get_atom_nb(pair.second);
    }

    const size_t wfc_size_loc = static_cast<size_t>(desc_wfc.m_loc()) * desc_wfc.n_loc();
    if (block_size_total > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 4)
        throw LIBRPA_RUNTIME_ERROR(
            "local symmetry-restored density-matrix block is too large for MPI collectives");

    Matz scaled_wfc_ket(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    // Zero stand-in for a missing (bra, ket) wfc channel: keeps the pgemm
    // collective intact while contributing a zero source block.
    Matz zero_wfc_stand_in(desc_wfc.m_loc(), desc_wfc.n_loc(), MAJOR::COL);
    zero_wfc_stand_in = C_ZERO;
    auto wfc_gemm_workspace = create_wfc_gemm_workspace(desc_wfc, desc_dm, kblacs_ctxt.blacs_h);
    const auto &iks_local = kblacs_ctxt.kpoints_local();
    const int nk_local = iks_local.size();
    int nk_sum = 0;
    MPI_Allreduce(&nk_local, &nk_sum, 1, mpi_datatype<int>::value, MPI_SUM,
                  kblacs_ctxt.comm_kpoint_h.comm);
    if (nk_sum != n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent k-point distribution");

    const double occ_thres = 1e-4 / n_kpoints;
    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();
    const auto member_kfrac_targets =
        build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc);

    // Identity orbital action for the per-atom-pair kernel call: the spatial
    // rotation is applied per channel beforehand, so the kernel only performs
    // the SU(2) mixing and the Theta remap.
    const auto identity_orbit = [](std::size_t spatial_id, const ComplexMatrix &block)
    {
        (void)spatial_id;
        return block;
    };

    for (int pid = 0; pid != kblacs_ctxt.comm_kpoint_h.nprocs; ++pid)
    {
        const int nR_this = n_Rs_all[pid];
        if (nR_this < 1) continue;

        const std::size_t count = static_cast<std::size_t>(nR_this) * block_size_total;
        if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / 4)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "local symmetry-restored density-matrix block is too large for MPI collectives");
        }
        // Four channels packed channel-major into one buffer: the reduce
        // rounds are unchanged from the scalar path.
        std::vector<cplxdb> buffer(4 * count, C_ZERO);

        for (int ik_local = 0; ik_local != nk_local; ++ik_local)
        {
            const int ik = iks_local[ik_local];
            const auto &k_ibz = kfrac_list[static_cast<std::size_t>(ik)];
            const auto &star = find_symmetry_kstar_for_ibz_kpoint(symmetry_context, k_ibz);
            const auto star_factor = 1.0 / static_cast<double>(star.members.size());

            int nocc = 0;
            std::vector<double> weights;
            weights.reserve(n_states);
            for (; nocc != n_states; ++nocc)
            {
                const double weight = mf.get_weight()[ispin](ik, nocc) * scale_spin;
                if (weight < occ_thres) break;
                weights.push_back(weight);
            }

            // Four source blocks D^{ab}(k_ibz), zero-filled when a (bra, ket)
            // wfc channel is missing. pgemm is collective over the k-group
            // BLACS grid, so it is called unconditionally with a zero
            // stand-in block; spinor zero-fill must not break the collective.
            Matz dmat_k[4];
            for (int s = 0; s != 4; ++s)
            {
                dmat_k[s] = Matz(desc_dm.m_loc(), desc_dm.n_loc(), MAJOR::COL);
                dmat_k[s] = C_ZERO;
                const int ispinor_bra = s / 2;
                const int ispinor_ket = s % 2;
                const auto *wfc_bra = mf.find_wfc(ispin, ispinor_bra, ik);
                const auto *wfc_ket = mf.find_wfc(ispin, ispinor_ket, ik);
                if (wfc_bra != nullptr && static_cast<size_t>(wfc_bra->size) != wfc_size_loc)
                    throw LIBRPA_RUNTIME_ERROR(
                        "wave-function bra block size is inconsistent with descriptor");
                if (wfc_ket != nullptr && static_cast<size_t>(wfc_ket->size) != wfc_size_loc)
                    throw LIBRPA_RUNTIME_ERROR(
                        "wave-function ket block size is inconsistent with descriptor");

                scaled_wfc_ket = C_ZERO;
                if (wfc_ket != nullptr && wfc_size_loc > 0)
                    std::memcpy(scaled_wfc_ket.ptr(), wfc_ket->c, wfc_size_loc * sizeof(cplxdb));
                const int wfc_m_loc = desc_wfc.m_loc();
                std::vector<char> scale_wfc_col(desc_wfc.n_loc(), false);
                std::vector<double> wfc_col_scale(desc_wfc.n_loc(), 1.0);
                for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
                {
                    const int jglob = desc_wfc.indx_l2g_c(jloc);
                    if (jglob < 0 || jglob >= nocc) continue;
                    scale_wfc_col[jloc] = true;
                    wfc_col_scale[jloc] = weights[jglob];
                }
#pragma omp parallel for schedule(static) if (wfc_size_loc > 4096)
                for (size_t i = 0; i < wfc_size_loc; ++i)
                {
                    const int jloc = static_cast<int>(i / wfc_m_loc);
                    if (scale_wfc_col[jloc]) scaled_wfc_ket.ptr()[i] *= wfc_col_scale[jloc];
                }

                if (nocc > 0)
                {
                    const cplxdb *wfc_bra_ptr =
                        wfc_bra == nullptr ? zero_wfc_stand_in.ptr() : wfc_bra->c;
                    pgemm_wfc_scaled_wfc_h(n_aos, nocc, wfc_bra_ptr, scaled_wfc_ket.ptr(), desc_wfc,
                                           wfc_gemm_workspace, dmat_k[s], desc_dm,
                                           kblacs_ctxt.blacs_h);
                }
            }

            for (std::size_t imember = 0; imember != star.members.size(); ++imember)
            {
                const auto &member = star.members[imember];
                const auto &op =
                    resolve_symmetry_kstar_member_spin_operation(symmetry_context, member);
                const Vector3_Order<double> *k_bz_target = nullptr;
                if (!member_kfrac_targets.empty() &&
                    static_cast<std::size_t>(ik) < member_kfrac_targets.size() &&
                    imember < member_kfrac_targets[static_cast<std::size_t>(ik)].size())
                {
                    k_bz_target = &member_kfrac_targets[static_cast<std::size_t>(ik)][imember];
                }
                const auto &k_for_phase = k_bz_target == nullptr ? member.k_bz : *k_bz_target;

                // Spatial part only: time reversal is suppressed here and
                // applied by the kernel remap; gauge phases are excluded and
                // applied after the kernel as plain unitary factors.
                SymmetryKStarMember spatial_member = member;
                spatial_member.time_reversal = false;
                const auto source_pair_requests = build_source_pair_requests(
                    target_pair_distribution, member, atbasis_wfc.n_atoms);
                const auto gauge_phases = build_symmetry_kstar_member_target_gauge_phases(
                    symmetry_context, member, atom_nw.size(), k_bz_target);

                symmetry_atom_block_matrix_map_t rotated[4];
                for (int s = 0; s != 4; ++s)
                {
                    const auto source_pair_mats = get_ap_map_from_blacs_dist(
                        dmat_k[s], source_pair_requests, atbasis_wfc, atbasis_wfc, desc_dm);
                    symmetry_atom_block_matrix_map_t source_blocks;
                    for (const auto &[pair, mat] : source_pair_mats)
                    {
                        source_blocks[pair.first][pair.second] = matz_to_complex_matrix(mat);
                    }
                    rotated[s] = rotate_symmetry_kspace_operator_blocks(
                        symmetry_context, wfc_layouts, spatial_member, source_blocks, atom_nw,
                        k_ibz, false, &target_pairs_local_set, nullptr);
                }

                for (const auto &atom_i_blocks : rotated[0])
                {
                    const auto atom_i = atom_i_blocks.first;
                    const auto n_I = atbasis_wfc.get_atom_nb(atom_i);
                    for (const auto &atom_j_block : atom_i_blocks.second)
                    {
                        const auto atom_j = atom_j_block.first;
                        const auto n_J = atbasis_wfc.get_atom_nb(atom_j);
                        // Union of the four channel block key sets, zero-filled
                        // per channel: a pair present in only some channels must
                        // not drop out of the joint kernel.
                        const SpinorBlocks4<ComplexMatrix> pair_in{
                            get_block_or_zero(rotated[0], atom_i, atom_j, n_I, n_J),
                            get_block_or_zero(rotated[1], atom_i, atom_j, n_I, n_J),
                            get_block_or_zero(rotated[2], atom_i, atom_j, n_I, n_J),
                            get_block_or_zero(rotated[3], atom_i, atom_j, n_I, n_J),
                        };
                        auto pair_out = transform_spinor_bilinear(
                            op, pair_in, identity_orbit, BilinearConvention::SourceToTarget_DXDdag);
                        const auto gauge_factor =
                            gauge_phases[atom_i] * std::conj(gauge_phases[atom_j]);
                        pair_out.b00 *= gauge_factor;
                        pair_out.b01 *= gauge_factor;
                        pair_out.b10 *= gauge_factor;
                        pair_out.b11 *= gauge_factor;

                        const atpair_t pair{atom_i, atom_j};
                        const auto offset_iter = target_pair_offsets.find(pair);
                        if (offset_iter == target_pair_offsets.end()) continue;
                        for (int iR = 0; iR != nR_this; ++iR)
                        {
                            const int index = pid * nR_max * 3 + iR * 3;
                            const Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1],
                                                       Rs_all[index + 2]};
                            const auto angle = -(k_for_phase * R) * TWO_PI;
                            const cplxdb phase{std::cos(angle), std::sin(angle)};
                            const cplxdb factor = star_factor * phase;
                            const ComplexMatrix *channel_blocks[4] = {&pair_out.b00, &pair_out.b01,
                                                                      &pair_out.b10, &pair_out.b11};
                            for (int s = 0; s != 4; ++s)
                            {
                                const std::size_t block_offset =
                                    (static_cast<std::size_t>(s) * nR_this + iR) *
                                        block_size_total +
                                    offset_iter->second;
                                const auto &block = *channel_blocks[s];
                                for (int i = 0; i != block.nr; ++i)
                                {
                                    for (int j = 0; j != block.nc; ++j)
                                    {
                                        buffer[block_offset + static_cast<std::size_t>(i) * n_J +
                                               j] += factor * block(i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        const int packed_count = static_cast<int>(4 * count);
        if (kblacs_ctxt.comm_kpoint_h.myid == pid)
        {
            kblacs_ctxt.comm_kpoint_h.reduce(MPI_IN_PLACE, buffer.data(), packed_count, pid,
                                             MPI_SUM);
            for (int iR = 0; iR != nR_this; ++iR)
            {
                const int index = pid * nR_max * 3 + iR * 3;
                const Vector3_Order<int> R{Rs_all[index], Rs_all[index + 1], Rs_all[index + 2]};
                auto &blocks = dmat_Rs.at(R);
                Matz *channel_mats[4] = {&blocks.b00, &blocks.b01, &blocks.b10, &blocks.b11};
                for (int s = 0; s != 4; ++s)
                {
                    ap_p_map<Matz> dmat_ap;
                    const std::size_t R_offset =
                        (static_cast<std::size_t>(s) * nR_this + iR) * block_size_total;
                    for (const auto &pair : target_sched.atpairs)
                    {
                        const auto n_I = atbasis_wfc.get_atom_nb(pair.first);
                        const auto n_J = atbasis_wfc.get_atom_nb(pair.second);
                        Matz block(n_I, n_J, MAJOR::COL);
                        const std::size_t block_offset = R_offset + target_pair_offsets.at(pair);
                        for (std::size_t i = 0; i != n_I; ++i)
                        {
                            for (std::size_t j = 0; j != n_J; ++j)
                            {
                                block(i, j) = buffer[block_offset + i * n_J + j];
                            }
                        }
                        dmat_ap[pair] = std::move(block);
                    }
                    auto &mat = *channel_mats[s];
                    mat = C_ZERO;
                    fill_local_mat_from_ap_dist_scheduler(mat, dmat_ap, target_sched, atbasis_wfc,
                                                          atbasis_wfc, desc_dm);
                }
            }
        }
        else
        {
            kblacs_ctxt.comm_kpoint_h.reduce(buffer.data(), buffer.data(), packed_count, pid,
                                             MPI_SUM);
        }
    }

    global::profiler.stop(__FUNCTION__);
    return dmat_Rs;
}
}
