#include "meanfield.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "symmetry_context.h"
#include "pbc.h"
#include "../utils/constants.h"
#include "../utils/error.h"
#include "../math/lapack_connector.h"

namespace librpa_int {

struct SymmetryKStarMeanFieldRestoreEntry
{
    const SymmetryKStar* star = nullptr;
    int ik_mf = -1;
    Vector3_Order<double> k_source{0.0, 0.0, 0.0};
    double star_factor = 1.0;
    double kpoint_weight = 0.0;
};

static void validate_symmetry_kstar_restore_metadata(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<atom_t, size_t>& atom_nw)
{
    if (!ctx.available || wfc_layouts.empty() || ctx.kstars.empty())
    {
        throw std::runtime_error("k-star restore requires AO shell layouts in the symmetry context");
    }
    if (mf.get_n_kpoints() != static_cast<int>(kfrac_list.size()))
    {
        throw std::runtime_error("k-star restore got inconsistent mean-field k-point counts");
    }
    if (ctx.count_kstar_members() == 0)
    {
        throw std::runtime_error("k-star restore found an empty full-k member list");
    }
    if (ctx.rspace_operations.empty())
    {
        throw std::runtime_error("k-star restore requires real-space symmetry operations");
    }
    for (const auto& atom_entry : atom_nw)
    {
        const auto atom = atom_entry.first;
        const auto type_iter = ctx.atom_to_type.find(atom);
        if (type_iter == ctx.atom_to_type.end())
        {
            throw std::runtime_error("k-star restore missing AO type metadata for an atom");
        }
        if (ctx.input_coord_frac.count(atom) == 0)
        {
            throw std::runtime_error("k-star restore missing fractional coordinates for an atom");
        }
        if (get_symmetry_species_layout(wfc_layouts, type_iter->second).n_ao
            != static_cast<int>(atom_entry.second))
        {
            throw std::runtime_error("k-star restore AO layout is inconsistent with atom_nw");
        }
    }
    for (const auto& type_entry : ctx.atom_to_type)
    {
        if (atom_nw.count(type_entry.first) == 0)
        {
            throw std::runtime_error("k-star restore atom_nw does not cover every atom");
        }
    }
}

static void validate_symmetry_kstar_meanfield_restore(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<atom_t, size_t>& atom_nw)
{
    validate_symmetry_kstar_restore_metadata(ctx, wfc_layouts, mf, kfrac_list, atom_nw);
    if (ctx.kstars.size() != kfrac_list.size())
    {
        throw std::runtime_error("k-star restore got inconsistent IBZ k-star counts");
    }
    for (const auto& kfrac : kfrac_list)
    {
        (void)librpa_int::find_symmetry_kstar_for_ibz_kpoint(ctx, kfrac);
    }
}

static void validate_symmetry_full_grid_kstar_meanfield_restore(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<atom_t, size_t>& atom_nw,
    const symmetry_kstar_representative_indices_t& representative_k_indices)
{
    validate_symmetry_kstar_restore_metadata(ctx, wfc_layouts, mf, kfrac_list, atom_nw);
    if (ctx.kstars.size() >= kfrac_list.size())
    {
        throw std::runtime_error("full-grid k-star restore does not reduce the mean-field k grid");
    }
    if (ctx.count_kstar_members() != kfrac_list.size())
    {
        throw std::runtime_error("full-grid k-star restore does not cover the loaded k grid");
    }
    if (representative_k_indices.size() != ctx.kstars.size())
    {
        throw std::runtime_error("full-grid k-star restore got inconsistent representative indices");
    }
    for (const auto ik_mf : representative_k_indices)
    {
        if (ik_mf < 0 || ik_mf >= mf.get_n_kpoints())
        {
            throw std::runtime_error("full-grid k-star restore representative index is out of range");
        }
    }
}

static void validate_symmetry_kstar_member_kfrac_targets(
    const std::vector<SymmetryKStarMeanFieldRestoreEntry>& restore_entries,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets)
{
    if (member_kfrac_targets == nullptr || member_kfrac_targets->empty())
    {
        return;
    }
    if (member_kfrac_targets->size() != restore_entries.size())
    {
        throw std::runtime_error(
            "k-star restore target k-point list has inconsistent IBZ size");
    }
    for (std::size_t ientry = 0; ientry != restore_entries.size(); ++ientry)
    {
        const auto* star = restore_entries[ientry].star;
        if (star == nullptr || (*member_kfrac_targets)[ientry].size() != star->members.size())
        {
            throw std::runtime_error(
                "k-star restore target k-point list has inconsistent star-member size");
        }
    }
}

static const Vector3_Order<double>& get_symmetry_kstar_member_kfrac_target(
    const librpa_int::SymmetryKStarMember& member,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets,
    const std::size_t ik_ibz,
    const std::size_t imember)
{
    if (member_kfrac_targets == nullptr || member_kfrac_targets->empty())
    {
        return member.k_bz;
    }
    return (*member_kfrac_targets)[ik_ibz][imember];
}

static double symmetry_kstar_geometric_weight(const SymmetryContext& ctx,
                                              const librpa_int::SymmetryKStar& star)
{
    const double full_count = static_cast<double>(ctx.count_kstar_members());
    if (full_count <= 0.0)
    {
        throw std::runtime_error("k-star restore found zero full-k members");
    }
    return static_cast<double>(star.members.size()) / full_count;
}

static std::vector<SymmetryKStarMeanFieldRestoreEntry> build_symmetry_kstar_restore_entries(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<atom_t, size_t>& atom_nw,
    const symmetry_kstar_representative_indices_t* representative_k_indices)
{
    std::vector<SymmetryKStarMeanFieldRestoreEntry> entries;
    const bool restore_from_full_grid =
        representative_k_indices != nullptr && !representative_k_indices->empty();

    if (restore_from_full_grid)
    {
        validate_symmetry_full_grid_kstar_meanfield_restore(
            ctx, wfc_layouts, mf, kfrac_list, atom_nw, *representative_k_indices);
        const double kpoint_weight = 1.0 / static_cast<double>(ctx.count_kstar_members());
        entries.reserve(ctx.kstars.size());
        for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
        {
            const auto& star = ctx.kstars[istar];
            if (star.members.empty())
            {
                throw std::runtime_error("k-star member list is empty");
            }
            const int ik_mf = (*representative_k_indices)[istar];
            entries.push_back({&star, ik_mf, kfrac_list[static_cast<std::size_t>(ik_mf)],
                               1.0, kpoint_weight});
        }
        return entries;
    }

    validate_symmetry_kstar_meanfield_restore(ctx, wfc_layouts, mf, kfrac_list, atom_nw);
    entries.reserve(kfrac_list.size());
    for (int ik_ibz = 0; ik_ibz != mf.get_n_kpoints(); ++ik_ibz)
    {
        const auto& k_ibz = kfrac_list[static_cast<std::size_t>(ik_ibz)];
        const auto& star = librpa_int::find_symmetry_kstar_for_ibz_kpoint(ctx, k_ibz);
        if (star.members.empty())
        {
            throw std::runtime_error("k-star member list is empty");
        }
        entries.push_back({&star,
                           ik_ibz,
                           k_ibz,
                           1.0 / static_cast<double>(star.members.size()),
                           symmetry_kstar_geometric_weight(ctx, star)});
    }
    return entries;
}

static ComplexMatrix build_gf_cplx_imagtime_with_prefactor(
    const MeanField& mf,
    const int ispin,
    const int ispinor_bra,
    const int ispinor_ket,
    const int ikpt,
    const double tau,
    const std::vector<double>& prefactors,
    const std::vector<bool> &band_mask)
{
    if (!band_mask.empty() && band_mask.size() != static_cast<std::size_t>(mf.get_n_bands()))
        throw LIBRPA_RUNTIME_ERROR("Green's-function band mask has inconsistent size");

    const int n_aos = mf.get_n_aos();
    const int n_bands = mf.get_n_bands();
    const auto wfc_bra = mf.find_wfc(ispin, ispinor_bra, ikpt);
    const auto wfc_ket = mf.find_wfc(ispin, ispinor_ket, ikpt);
    if (wfc_bra == nullptr)
        throw LIBRPA_RUNTIME_ERROR("wfc of ispinor_bra not found");
    if (wfc_ket == nullptr)
        throw LIBRPA_RUNTIME_ERROR("wfc of ispinor_ket not found");

    auto scaled_wfc_conj = conj(*wfc_ket);
    for (int ib = 0; ib != n_bands; ++ib)
    {
        const double energy_scale = -tau * (mf.get_eigenvals()[ispin](ikpt, ib) - mf.get_efermi());
        const double bounded_scale = energy_scale > 0.0 ? 0.0 : energy_scale;
        const double scale = !band_mask.empty() && !band_mask[ib] ? 0.0
            : std::exp(bounded_scale) * prefactors[static_cast<std::size_t>(ib)];
        LapackConnector::scal(n_aos, scale, scaled_wfc_conj.c + n_aos * ib, 1);
    }
    return transpose(*wfc_bra, false) * scaled_wfc_conj;
}

//! Zero-fill variant of build_gf_cplx_imagtime_with_prefactor: spinor source
//! blocks whose wfc channel is missing contribute a zero block.
static ComplexMatrix build_gf_cplx_imagtime_with_prefactor_zero_fill(
    const MeanField& mf,
    const int ispin,
    const int ispinor_bra,
    const int ispinor_ket,
    const int ikpt,
    const double tau,
    const std::vector<double>& prefactors,
    const std::vector<bool>& band_mask)
{
    if (mf.find_wfc(ispin, ispinor_bra, ikpt) == nullptr
        || mf.find_wfc(ispin, ispinor_ket, ikpt) == nullptr)
    {
        ComplexMatrix zero(mf.get_n_aos(), mf.get_n_aos());
        zero.zero_out();
        return zero;
    }
    return build_gf_cplx_imagtime_with_prefactor(
        mf, ispin, ispinor_bra, ispinor_ket, ikpt, tau, prefactors, band_mask);
}

void validate_kstar_band_cutoff_closure(
    const SymmetryContext& ctx,
    const MeanField& mf,
    const std::vector<bool>& band_mask,
    const double degen_tol)
{
    (void)ctx;
    if (band_mask.empty())
    {
        return;
    }
    if (band_mask.size() != static_cast<std::size_t>(mf.get_n_bands()))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Green's-function band mask size does not match the mean-field band count");
    }
    for (int ispin = 0; ispin != mf.get_n_spins(); ++ispin)
    {
        for (int ik = 0; ik != mf.get_n_kpoints(); ++ik)
        {
            for (int ib = 1; ib != mf.get_n_bands(); ++ib)
            {
                if (band_mask[static_cast<std::size_t>(ib - 1)]
                    == band_mask[static_cast<std::size_t>(ib)])
                {
                    continue;
                }
                const double gap = mf.get_eigenvals()[ispin](ik, ib)
                                 - mf.get_eigenvals()[ispin](ik, ib - 1);
                if (std::abs(gap) < degen_tol)
                {
                    throw LIBRPA_RUNTIME_ERROR(
                        "Green's-function band mask slices through a degenerate band "
                        "multiplet: ispin " + std::to_string(ispin)
                        + ", k-point " + std::to_string(ik)
                        + ", bands " + std::to_string(ib - 1) + "/"
                        + std::to_string(ib)
                        + " separated by " + std::to_string(gap)
                        + " Ha (< " + std::to_string(degen_tol)
                        + "). Symmetry restore is disabled for this truncation; "
                        "adjust the band mask or use the full band grid");
                }
            }
        }
    }
}

//! Multiply the four blocks by the plain unitary re-gauging phases
//! block(I, J) *= phase[I] * conj(phase[J]) (applied after the kernel, so
//! antiunitary members keep the scalar restore convention).
static void apply_spinor_member_gauge_phases(
    SpinorBlocks4<ComplexMatrix>& blocks,
    const std::map<atom_t, size_t>& atom_nw,
    const std::vector<std::complex<double>>& phases)
{
    std::vector<int> offsets;
    offsets.reserve(atom_nw.size() + 1);
    offsets.push_back(0);
    for (const auto& atom_entry : atom_nw)
    {
        offsets.push_back(offsets.back() + static_cast<int>(atom_entry.second));
    }
    const auto scale_block = [&offsets, &phases](ComplexMatrix& block) {
        for (std::size_t ai = 0; ai + 1 != offsets.size(); ++ai)
        {
            const std::complex<double> phase_i = phases[ai];
            for (int i = offsets[ai]; i != offsets[ai + 1]; ++i)
            {
                for (std::size_t aj = 0; aj + 1 != offsets.size(); ++aj)
                {
                    const auto factor = phase_i * std::conj(phases[aj]);
                    for (int j = offsets[aj]; j != offsets[aj + 1]; ++j)
                    {
                        block(i, j) *= factor;
                    }
                }
            }
        }
    };
    scale_block(blocks.b00);
    scale_block(blocks.b01);
    scale_block(blocks.b10);
    scale_block(blocks.b11);
}

bool can_restore_symmetry_kstar_meanfield(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::map<atom_t, size_t>& atom_nw)
{
    if (!ctx.available || wfc_layouts.empty() || ctx.kstars.empty())
    {
        return false;
    }
    if (mf.get_n_kpoints() != static_cast<int>(kfrac_list.size()))
    {
        return false;
    }
    if (ctx.kstars.size() != kfrac_list.size())
    {
        return false;
    }
    if (ctx.count_kstar_members() <= kfrac_list.size())
    {
        return false;
    }
    validate_symmetry_kstar_meanfield_restore(ctx, wfc_layouts, mf, kfrac_list, atom_nw);
    return true;
}

symmetry_kstar_member_kfrac_targets_t build_symmetry_kstar_member_kfrac_targets(
    const SymmetryContext& ctx,
    const PeriodicBoundaryData& pbc)
{
    if (!ctx.available || ctx.kstars.empty())
    {
        return {};
    }

    const auto& mapping = ctx.kstar_grid_mapping;
    symmetry_kstar_member_kfrac_targets_t targets(mapping.size());

    for (const auto& entry : mapping)
    {
        if (entry.iq_ibz < 0 || entry.iq_ibz >= static_cast<int>(targets.size()))
        {
            throw std::runtime_error("k-star restore mapping has an invalid IBZ index");
        }
        auto& star_targets = targets[static_cast<std::size_t>(entry.iq_ibz)];
        star_targets.reserve(entry.member_q_bz_keys.size());
        for (const auto& q_key : entry.member_q_bz_keys)
        {
            star_targets.emplace_back(pbc.latvec * q_key);
        }
    }

    return targets;
}

ComplexMatrix get_symmetry_restored_dmat_cplx_R(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const int ispin,
    const int ispinor_bra,
    const int ispinor_ket,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const Vector3_Order<int>& R,
    const std::map<atom_t, size_t>& atom_nw,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets,
    const symmetry_kstar_representative_indices_t* representative_k_indices)
{
    const auto restore_entries = build_symmetry_kstar_restore_entries(
        ctx, wfc_layouts, mf, kfrac_list, atom_nw, representative_k_indices);
    validate_symmetry_kstar_member_kfrac_targets(restore_entries, member_kfrac_targets);
    ComplexMatrix dmat_cplx(mf.get_n_aos(), mf.get_n_aos());

    for (std::size_t ientry = 0; ientry != restore_entries.size(); ++ientry)
    {
        const auto& entry = restore_entries[ientry];
        const auto& star = *entry.star;
        const auto dmat_ibz = mf.get_dmat_cplx(ispin, ispinor_bra, ispinor_ket, entry.ik_mf);

        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto& k_bz_target = get_symmetry_kstar_member_kfrac_target(
                member, member_kfrac_targets, ientry, imember);
            const auto dmat_member = librpa_int::rotate_symmetry_kspace_matrix(
                ctx, wfc_layouts, member, dmat_ibz, atom_nw, entry.k_source,
                member.time_reversal, &k_bz_target);
            const double angle = -(k_bz_target * R) * TWO_PI;
            const auto kphase = std::complex<double>(std::cos(angle), std::sin(angle));
            dmat_cplx += (entry.star_factor * kphase) * dmat_member;
        }
    }

    return dmat_cplx;
}

SpinorBlocks4<ComplexMatrix> get_symmetry_restored_dmat_cplx_R_spinor(
    const SymmetryContext& ctx, const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf, const int ispin, const std::vector<Vector3_Order<double>>& kfrac_list,
    const Vector3_Order<int>& R, const std::map<atom_t, size_t>& atom_nw,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets,
    const symmetry_kstar_representative_indices_t* representative_k_indices)
{
    if (mf.get_n_spinor() != 2)
    {
        throw LIBRPA_RUNTIME_ERROR("spinor k-star density-matrix restore requires n_spinor == 2");
    }
    const auto restore_entries = build_symmetry_kstar_restore_entries(
        ctx, wfc_layouts, mf, kfrac_list, atom_nw, representative_k_indices);
    validate_symmetry_kstar_member_kfrac_targets(restore_entries, member_kfrac_targets);

    const int n_aos = mf.get_n_aos();
    SpinorBlocks4<ComplexMatrix> dmat_R{ComplexMatrix(n_aos, n_aos), ComplexMatrix(n_aos, n_aos),
                                        ComplexMatrix(n_aos, n_aos), ComplexMatrix(n_aos, n_aos)};

    for (std::size_t ientry = 0; ientry != restore_entries.size(); ++ientry)
    {
        const auto& entry = restore_entries[ientry];
        const auto& star = *entry.star;
        // Four source blocks at the representative; get_dmat_cplx zero-fills
        // a missing (bra, ket) wfc channel.
        const SpinorBlocks4<ComplexMatrix> dmat_ibz{
            mf.get_dmat_cplx(ispin, 0, 0, entry.ik_mf),
            mf.get_dmat_cplx(ispin, 0, 1, entry.ik_mf),
            mf.get_dmat_cplx(ispin, 1, 0, entry.ik_mf),
            mf.get_dmat_cplx(ispin, 1, 1, entry.ik_mf),
        };

        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const auto& op = resolve_symmetry_kstar_member_spin_operation(ctx, member);
            const auto& k_bz_target = get_symmetry_kstar_member_kfrac_target(
                member, member_kfrac_targets, ientry, imember);

            // Orbital transform built once per member, shared by the four
            // source blocks. The gauge phases are excluded here and applied
            // after the kernel so that the antiunitary Theta remap conjugates
            // only the spatial-rotation part (same convention as the spinor
            // Green's-function restore, report section 6.7).
            const ComplexMatrix transform = build_symmetry_kspace_operator_transform_matrix(
                ctx, wfc_layouts, member, atom_nw, entry.k_source, false, nullptr);
            const ComplexMatrix transform_dag = transpose(transform, true);
            const auto orbit =
                [&transform, &transform_dag](std::size_t spatial_id, const ComplexMatrix& block)
            {
                (void)spatial_id;  // the transform is resolved per member, not per spatial_id
                return transform * block * transform_dag;
            };
            auto dmat_member = transform_spinor_bilinear(op, dmat_ibz, orbit,
                                                         BilinearConvention::SourceToTarget_DXDdag);

            // Plain unitary re-gauging to the target k-point basis.
            if (member_kfrac_targets != nullptr && !member_kfrac_targets->empty())
            {
                const auto phases = build_symmetry_kstar_member_target_gauge_phases(
                    ctx, member, atom_nw.size(), &k_bz_target);
                apply_spinor_member_gauge_phases(dmat_member, atom_nw, phases);
            }

            const double angle = -(k_bz_target * R) * TWO_PI;
            const auto kphase = std::complex<double>(std::cos(angle), std::sin(angle));
            const auto factor = entry.star_factor * kphase;
            dmat_R.b00 += factor * dmat_member.b00;
            dmat_R.b01 += factor * dmat_member.b01;
            dmat_R.b10 += factor * dmat_member.b10;
            dmat_R.b11 += factor * dmat_member.b11;
        }
    }

    return dmat_R;
}

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>>
get_symmetry_restored_gf_cplx_imagtimes_Rs(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const int ispin,
    const int ispinor_bra,
    const int ispinor_ket,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<double>& imagtimes,
    const std::vector<Vector3_Order<int>>& Rs,
    const std::map<atom_t, size_t>& atom_nw,
    const std::vector<bool> &band_mask,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets,
    const symmetry_kstar_representative_indices_t* representative_k_indices)
{
    validate_kstar_band_cutoff_closure(ctx, mf, band_mask);
    const auto restore_entries = build_symmetry_kstar_restore_entries(
        ctx, wfc_layouts, mf, kfrac_list, atom_nw, representative_k_indices);
    validate_symmetry_kstar_member_kfrac_targets(restore_entries, member_kfrac_targets);

    std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> gf_tau_R;
    for (const auto tau : imagtimes)
    {
        gf_tau_R[tau] = {};
    }
    if (Rs.empty())
    {
        return gf_tau_R;
    }

    const int n_bands = mf.get_n_bands();
    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();

    for (const auto tau : imagtimes)
    {
        const double tau_sign = tau > 0.0 ? 1.0 : -1.0;
        for (std::size_t ientry = 0; ientry != restore_entries.size(); ++ientry)
        {
            const auto& entry = restore_entries[ientry];
            const auto& star = *entry.star;

            std::vector<double> prefactors(static_cast<std::size_t>(n_bands), 0.0);
            for (int ib = 0; ib != n_bands; ++ib)
            {
                const double occ_weight = mf.get_weight()[ispin](entry.ik_mf, ib) * scale_spin;
                prefactors[static_cast<std::size_t>(ib)] =
                    tau > 0.0 ? std::max(0.0, entry.kpoint_weight - occ_weight) : occ_weight;
            }

            const auto gf_ibz = build_gf_cplx_imagtime_with_prefactor(
                mf, ispin, ispinor_bra, ispinor_ket, entry.ik_mf, tau, prefactors, band_mask);

            for (std::size_t imember = 0; imember != star.members.size(); ++imember)
            {
                const auto& member = star.members[imember];
                const auto& k_bz_target = get_symmetry_kstar_member_kfrac_target(
                    member, member_kfrac_targets, ientry, imember);
                const auto gf_member = librpa_int::rotate_symmetry_kspace_matrix(
                    ctx, wfc_layouts, member, gf_ibz, atom_nw, entry.k_source,
                    member.time_reversal, &k_bz_target);
                for (const auto& R : Rs)
                {
                    const double angle = -(k_bz_target * R) * TWO_PI;
                    const auto kphase = std::complex<double>(std::cos(angle), std::sin(angle));
                    if (gf_tau_R.at(tau).count(R) == 0)
                    {
                        gf_tau_R[tau][R].create(mf.get_n_aos(), mf.get_n_aos());
                    }
                    gf_tau_R[tau][R] += (entry.star_factor * tau_sign * kphase) * gf_member;
                }
            }
        }
    }

    return gf_tau_R;
}

std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<ComplexMatrix>>>
get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
    const SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const MeanField& mf,
    const int ispin,
    const std::vector<Vector3_Order<double>>& kfrac_list,
    const std::vector<double>& imagtimes,
    const std::vector<Vector3_Order<int>>& Rs,
    const std::map<atom_t, size_t>& atom_nw,
    const std::vector<bool>& band_mask,
    const symmetry_kstar_member_kfrac_targets_t* member_kfrac_targets,
    const symmetry_kstar_representative_indices_t* representative_k_indices)
{
    if (mf.get_n_spinor() != 2)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "spinor k-star Green's-function restore requires n_spinor == 2");
    }
    validate_kstar_band_cutoff_closure(ctx, mf, band_mask);
    const auto restore_entries = build_symmetry_kstar_restore_entries(
        ctx, wfc_layouts, mf, kfrac_list, atom_nw, representative_k_indices);
    validate_symmetry_kstar_member_kfrac_targets(restore_entries, member_kfrac_targets);

    std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<ComplexMatrix>>> gf_tau_R;
    for (const auto tau : imagtimes)
    {
        gf_tau_R[tau] = {};
    }
    if (Rs.empty())
    {
        return gf_tau_R;
    }

    const int n_aos = mf.get_n_aos();
    const int n_bands = mf.get_n_bands();
    const double scale_spin = 0.5 * mf.get_n_spins() * mf.get_n_spinor();

    for (const auto tau : imagtimes)
    {
        const double tau_sign = tau > 0.0 ? 1.0 : -1.0;
        for (std::size_t ientry = 0; ientry != restore_entries.size(); ++ientry)
        {
            const auto& entry = restore_entries[ientry];
            const auto& star = *entry.star;

            std::vector<double> prefactors(static_cast<std::size_t>(n_bands), 0.0);
            for (int ib = 0; ib != n_bands; ++ib)
            {
                const double occ_weight = mf.get_weight()[ispin](entry.ik_mf, ib) * scale_spin;
                prefactors[static_cast<std::size_t>(ib)] =
                    tau > 0.0 ? std::max(0.0, entry.kpoint_weight - occ_weight) : occ_weight;
            }

            // Four source blocks at the representative, zero-filled when a
            // (bra, ket) wfc channel is missing.
            const SpinorBlocks4<ComplexMatrix> gf_ibz{
                build_gf_cplx_imagtime_with_prefactor_zero_fill(
                    mf, ispin, 0, 0, entry.ik_mf, tau, prefactors, band_mask),
                build_gf_cplx_imagtime_with_prefactor_zero_fill(
                    mf, ispin, 0, 1, entry.ik_mf, tau, prefactors, band_mask),
                build_gf_cplx_imagtime_with_prefactor_zero_fill(
                    mf, ispin, 1, 0, entry.ik_mf, tau, prefactors, band_mask),
                build_gf_cplx_imagtime_with_prefactor_zero_fill(
                    mf, ispin, 1, 1, entry.ik_mf, tau, prefactors, band_mask),
            };

            for (std::size_t imember = 0; imember != star.members.size(); ++imember)
            {
                const auto& member = star.members[imember];
                const auto& op = resolve_symmetry_kstar_member_spin_operation(ctx, member);
                const auto& k_bz_target = get_symmetry_kstar_member_kfrac_target(
                    member, member_kfrac_targets, ientry, imember);

                // Orbital transform built once per member, shared by the four
                // source blocks. The gauge phases are excluded here and
                // applied after the kernel so that the antiunitary Theta remap
                // conjugates only the spatial-rotation part (report section
                // 6.7: spatial unitary first, Theta last).
                const ComplexMatrix transform = build_symmetry_kspace_operator_transform_matrix(
                    ctx, wfc_layouts, member, atom_nw, entry.k_source, false, nullptr);
                const ComplexMatrix transform_dag = transpose(transform, true);
                const auto orbit = [&transform, &transform_dag](std::size_t spatial_id,
                                                                const ComplexMatrix& block) {
                    (void)spatial_id;  // the transform is resolved per member, not per spatial_id
                    return transform * block * transform_dag;
                };
                auto gf_member = transform_spinor_bilinear(
                    op, gf_ibz, orbit, BilinearConvention::SourceToTarget_DXDdag);

                // Plain unitary re-gauging to the target k-point basis.
                if (member_kfrac_targets != nullptr && !member_kfrac_targets->empty())
                {
                    const auto phases = build_symmetry_kstar_member_target_gauge_phases(
                        ctx, member, atom_nw.size(), &k_bz_target);
                    apply_spinor_member_gauge_phases(gf_member, atom_nw, phases);
                }

                for (const auto& R : Rs)
                {
                    const double angle = -(k_bz_target * R) * TWO_PI;
                    const auto kphase = std::complex<double>(std::cos(angle), std::sin(angle));
                    const auto factor = entry.star_factor * tau_sign * kphase;
                    auto& gf_R = gf_tau_R[tau][R];
                    if (gf_R.b00.nr == 0)
                    {
                        gf_R.b00.create(n_aos, n_aos);
                        gf_R.b01.create(n_aos, n_aos);
                        gf_R.b10.create(n_aos, n_aos);
                        gf_R.b11.create(n_aos, n_aos);
                    }
                    gf_R.b00 += factor * gf_member.b00;
                    gf_R.b01 += factor * gf_member.b01;
                    gf_R.b10 += factor * gf_member.b10;
                    gf_R.b11 += factor * gf_member.b11;
                }
            }
        }
    }

    return gf_tau_R;
}

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>>
extract_spinor_gf_block(
    std::map<double, std::map<Vector3_Order<int>, SpinorBlocks4<ComplexMatrix>>>&& gf_spinor,
    const int ispinor_bra, const int ispinor_ket)
{
    if (ispinor_bra < 0 || ispinor_bra > 1 || ispinor_ket < 0 || ispinor_ket > 1)
    {
        throw LIBRPA_RUNTIME_ERROR("spinor Green's-function block index out of range");
    }
    std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> gf_block;
    for (auto& tau_entry : gf_spinor)
    {
        auto& gf_R = gf_block[tau_entry.first];
        for (auto& R_entry : tau_entry.second)
        {
            auto& blocks = R_entry.second;
            if (ispinor_bra == 0 && ispinor_ket == 0) gf_R[R_entry.first] = std::move(blocks.b00);
            else if (ispinor_bra == 0 && ispinor_ket == 1) gf_R[R_entry.first] = std::move(blocks.b01);
            else if (ispinor_bra == 1 && ispinor_ket == 0) gf_R[R_entry.first] = std::move(blocks.b10);
            else gf_R[R_entry.first] = std::move(blocks.b11);
        }
    }
    return gf_block;
}

void MeanField::resize(int ns, int nk, int nb, int nao, int nspinor, int st_ib, int nb_local, int st_iao, int nao_local)
{
    if (ns == 0 || nk == 0 || nb == 0 || nao == 0)
        throw LIBRPA_RUNTIME_ERROR("encounter zero dimension");
    if (n_spins != 0)
    {
        eskb.clear();
        wg.clear();
        wfc.clear();
    }

    n_spins = ns;
    n_kpoints = nk;
    n_states = nb;
    n_aos = nao;
    i_ao_start = st_iao;
    n_aos_local = nao_local;
    i_state_start = st_ib;
    n_states_local = nb_local;
    n_spinor = nspinor;

    eskb.resize(n_spins);
    wg.resize(n_spins);
    wfc.clear();

    for (int is = 0; is < n_spins; is++)
    {
        eskb[is].create(n_kpoints, n_states);
        wg[is].create(n_kpoints, n_states);
    }
}

std::vector<int> MeanField::get_iks_local() const
{
    std::vector<int> iks_local;
    auto it_spin = this->wfc.cbegin();  // assume same k-points in all spin channels
    if (it_spin != this->wfc.cend())
    {
        auto it_spinor = it_spin->second.cbegin();
        if (it_spinor != it_spin->second.cend())
        {
            for (const auto &[ik, _]: it_spinor->second)
            {
                iks_local.push_back(ik);
            }
        }
    }
    return iks_local;
}

MeanField::MeanField(int ns, int nk, int nb, int nao, int nspinor)
    : n_spins(ns),
      n_aos(nao),
      n_states(nb),
      n_kpoints(nk),
      n_spinor(nspinor),
      n_aos_local(nao),
      i_ao_start(0),
      n_states_local(nb),
      i_state_start(0),
      eskb(),
      wg(),
      wfc(),
      efermi(0)
{
    resize(ns, nk, nb, nao, nspinor, 0, nb, 0, nao);
}

MeanField::MeanField(int ns, int nk, int nb, int nao, int nspinor, int st_ib, int nb_local, int st_iao, int nao_local)
    : n_spins(ns),
      n_aos(nao),
      n_states(nb),
      n_kpoints(nk),
      n_spinor(nspinor),
      n_aos_local(nao_local),
      i_ao_start(st_iao),
      n_states_local(nb_local),
      i_state_start(st_ib),
      eskb(),
      wg(),
      wfc(),
      efermi(0)
{
    resize(ns, nk, nb, nao, nspinor, st_ib, nb_local, st_iao, nao_local);
}

void MeanField::set(int ns, int nk, int nb, int nao, int nspinor)
{
    if (n_spins != 0 || n_kpoints != 0 || n_states != 0 || n_aos != 0)
    {
        std::cout << n_spins << n_kpoints << n_states << n_aos << std::endl;
        throw LIBRPA_RUNTIME_ERROR("MeanField object already set");
    }
    resize(ns, nk, nb, nao, nspinor, 0, nb, 0, nao);
}

void MeanField::set(int ns, int nk, int nb, int nao, int nspinor, int st_ib, int nb_local, int st_iao, int nao_local)
{
    if (n_spins != 0 || n_kpoints != 0 || n_states != 0 || n_aos != 0)
    {
        std::cout << n_spins << n_kpoints << n_states << n_aos << std::endl;
        throw LIBRPA_RUNTIME_ERROR("MeanField object already set");
    }
    resize(ns, nk, nb, nao, nspinor, st_ib, nb_local, st_iao, nao_local);
}

ComplexMatrix *MeanField::find_wfc(int ispin, int ispinor, int ikpt) noexcept
{
    auto it_sp = wfc.find(ispin);
    if (it_sp == wfc.cend()) return nullptr;
    auto it_spinor = it_sp->second.find(ispinor);
    if (it_spinor == it_sp->second.cend()) return nullptr;
    auto it_k = it_spinor->second.find(ikpt);
    if (it_k == it_spinor->second.cend()) return nullptr;
    return &it_k->second;
}

const ComplexMatrix *MeanField::find_wfc(int ispin, int ispinor, int ikpt) const noexcept
{
    auto it_sp = wfc.find(ispin);
    if (it_sp == wfc.cend()) return nullptr;
    auto it_spinor = it_sp->second.find(ispinor);
    if (it_spinor == it_sp->second.cend()) return nullptr;
    auto it_k = it_spinor->second.find(ikpt);
    if (it_k == it_spinor->second.cend()) return nullptr;
    return &it_k->second;
}

// MeanField::MeanField(const MeanField &mf)
// {
//     resize(mf.n_spins, mf.n_kpoints, mf.n_states, mf.n_aos,
//            mf.i_state_start, mf.n_states_local, mf.i_ao_start, mf.n_aos_local);
//     // FIXME: copy data, not tested
//     eskb = mf.eskb;
//     wg = mf.wg;
//     wfc = mf.wfc;
//     efermi = mf.efermi;
// }

double MeanField::get_E_min_max(double &emin, double &emax) const
{
    // double lb = eskb[0](0, 0);
    // double ub = eskb[0](0, n_states - 1);
    double lb = std::numeric_limits<double>::max();
    double ub = std::numeric_limits<double>::min();
    for (int is = 0; is != n_spins; is++)
        for (int ik = 0; ik != n_kpoints; ik++)
        {
            lb = (lb > eskb[is](ik, 0)) ? eskb[is](ik, 0) : lb;
            ub = (ub < eskb[is](ik, n_states-1)) ? eskb[is](ik, n_states-1) : ub;
        }
    double gap = get_band_gap();
    emax = ub - lb;
    emin = gap;
    return emin;
}

double MeanField::get_band_gap() const
{
    constexpr double occupation_tol = 1e-8;
    double homo = -1e6, lumo = 1e6;
    double gap = lumo - homo;
    for (int is = 0; is != n_spins; is++)
    {
        //print_matrix("mf.eskb: ",this->eskb[is]);
        for (int ik = 0; ik != n_kpoints; ik++)
        {
            int homo_level = -1;
            for (int n = 0; n != n_states; n++)
            {
                if (wg[is](ik, n) > occupation_tol)
                {
                    homo_level = n;
                }
            }
            //cout<<"|is ik: "<<is<<" "<<ik<<"  homo_level: "<<homo_level<<"   eskb0: "<<eskb[is](ik, homo_level)<<"  eskb1: "<<eskb[is](ik, homo_level + 1)<<endl;
            if(homo_level != -1)
                homo = eskb[is](ik, homo_level) > homo ? eskb[is](ik, homo_level) : homo;
            if (homo_level + 1 < n_states)
                lumo = eskb[is](ik, homo_level + 1) < lumo ?  eskb[is](ik, homo_level + 1) : lumo;
            
            //cout<<"   homo: "<<homo<<"  lumo: "<<lumo<<endl;
        }
    }
    gap = lumo - homo;
    return gap;
}

std::pair<int, int> MeanField::find_highest_occupied_state(const int ispin, const int ikpt,
                                                           const double occupation_tol) const
{
    if (ispin < 0 || ispin >= n_spins)
        throw LIBRPA_RUNTIME_ERROR("spin index out of range: " + std::to_string(ispin));
    if (ikpt >= n_kpoints)
        throw LIBRPA_RUNTIME_ERROR("k-point index out of range: " + std::to_string(ikpt));

    const int ik_begin = ikpt >= 0 ? ikpt : 0;
    const int ik_end = ikpt >= 0 ? ikpt + 1 : n_kpoints;
    const double weight_tol = occupation_tol / n_kpoints;
    int ik_occ = -1;
    int state_occ = -1;
    double e_occ = -std::numeric_limits<double>::infinity();

    for (int ik = ik_begin; ik != ik_end; ++ik)
    {
        for (int i_state = 0; i_state != n_states; ++i_state)
        {
            if (wg[ispin](ik, i_state) <= weight_tol) continue;
            const double e_state = eskb[ispin](ik, i_state);
            if (ik_occ < 0 || e_state > e_occ)
            {
                ik_occ = ik;
                state_occ = i_state;
                e_occ = e_state;
            }
        }
    }

    return {ik_occ, state_occ};
}

int MeanField::get_max_state_below_energy(double energy) const
{
    int i_state_bound = -1;
    for (int i_state = 0; i_state != n_states; ++i_state)
    {
        bool all_below = true;
        for (int is = 0; is != n_spins && all_below; ++is)
        {
            for (int ik = 0; ik != n_kpoints; ++ik)
            {
                if (eskb[is](ik, i_state) >= energy)
                {
                    all_below = false;
                    break;
                }
            }
        }
        if (!all_below) break;
        i_state_bound = i_state;
    }
    return i_state_bound;
}

int MeanField::get_min_state_above_energy(double energy) const
{
    int i_state_bound = n_states;
    for (int i_state = n_states - 1; i_state >= 0; --i_state)
    {
        bool all_above = true;
        for (int is = 0; is != n_spins && all_above; ++is)
        {
            for (int ik = 0; ik != n_kpoints; ++ik)
            {
                if (eskb[is](ik, i_state) <= energy)
                {
                    all_above = false;
                    break;
                }
            }
        }
        if (!all_above) break;
        i_state_bound = i_state;
    }
    return i_state_bound;
}

// get_dmat_cplx can be used for both serial and k-porallel version
ComplexMatrix MeanField::get_dmat_cplx(int ispin, int ispinor_bra, int ispinor_ket, int ikpt) const
{
    assert(ispin < this->n_spins);
    assert(ikpt < this->n_kpoints);

    ComplexMatrix dmat_cplx(n_aos, n_aos);
    dmat_cplx.zero_out();

    const auto wfc_bra = find_wfc(ispin, ispinor_bra, ikpt);
    const auto wfc_ket = find_wfc(ispin, ispinor_ket, ikpt);

    if (wfc_bra == nullptr || wfc_ket == nullptr) return dmat_cplx;

    auto scaled_wfc_conj = conj(*wfc_ket);
    const double occ_thres = 1e-4 / n_kpoints;
    int nocc;
    for (nocc = 0; nocc != this->n_states; nocc++)
    {
        // Renormalize to single spin channel. Need to adpat SOC case (n_spins = 1 but remove 0.5)
        const auto weight = this->wg[ispin](ikpt, nocc) * 0.5 * n_spins * n_spinor;
        if (weight < occ_thres) break;
        LapackConnector::scal(this->n_aos, weight, scaled_wfc_conj.c + n_aos * nocc, 1);
    }
    LapackConnector::gemm('T', 'N', n_aos, n_aos, nocc, 1.0,
                          wfc_bra->c, n_aos, scaled_wfc_conj.c, n_aos,
                          0.0, dmat_cplx.c, n_aos);
    // auto dmat_cplx = transpose(wfc_sk, false) * scaled_wfc_conj;
    return dmat_cplx;
}

ComplexMatrix MeanField::get_dmat_cplx_R(int ispin, int ispinor_bra, int ispinor_ket,
                                         const std::vector<Vector3_Order<double>> &kfrac_list,
                                         const Vector3_Order<int> &R) const
{
    ComplexMatrix dmat_cplx(this->n_aos, this->n_aos);
    for (int ik = 0; ik != this->n_kpoints; ik++)
    {
        auto ang = - (kfrac_list[ik] * R) * TWO_PI;
        complex<double> kphase = complex<double>(cos(ang), sin(ang));
        // global::ofs_myid << "R " << R << " ik " << ik << " kfrac " << kfrac_list[ik] << " phase " << kphase << std::endl;
        dmat_cplx += kphase * this->get_dmat_cplx(ispin, ispinor_bra, ispinor_ket, ik);
    }
    return dmat_cplx;
}

std::map<Vector3_Order<int>, ComplexMatrix> MeanField::get_dmat_cplx_Rs(
    int ispin, int ispinor_bra, int ispinor_ket,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<Vector3_Order<int>> &Rs) const
{
    std::map<Vector3_Order<int>, ComplexMatrix> dmat_cplx_all;
    if (Rs.size() == 0) return dmat_cplx_all;
    for (const auto &R: Rs)
    {
        dmat_cplx_all[R] = ComplexMatrix(this->n_aos, this->n_aos);
    }
    for (int ik = 0; ik != this->n_kpoints; ik++)
    {
        const auto &kmat = this->get_dmat_cplx(ispin, ispinor_bra, ispinor_ket, ik);
        for (auto &[R, rmat]: dmat_cplx_all)
        {
            auto ang = - (kfrac_list[ik] * R) * TWO_PI;
            complex<double> kphase = complex<double>(cos(ang), sin(ang));
            rmat += kphase * kmat;
        }
    }
    return dmat_cplx_all;
}

ComplexMatrix MeanField::get_gf_cplx_imagtime(int ispin, int ispinor_bra, int ispinor_ket, int ikpt, double tau,
    const std::vector<bool> &band_mask) const
{
    if (!band_mask.empty() && band_mask.size() != static_cast<std::size_t>(n_states))
        throw LIBRPA_RUNTIME_ERROR("Green's-function band mask has inconsistent size");

    assert(ispin < this->n_spins);
    assert(ikpt < this->n_kpoints);

    const double scale_spin = 0.5 * n_spins * n_spinor;

    std::vector<double> wg_sk(wg[ispin].c + n_states * ikpt, wg[ispin].c + n_states * (ikpt + 1));
    std::vector<double> wg_empty_sk(wg_sk);
    for (int ib = 0; ib < n_states; ib++)
    {
        wg_sk[ib] *= scale_spin;
        wg_empty_sk[ib] = 1.0 / n_kpoints - wg_empty_sk[ib] * scale_spin;
        if (wg_empty_sk[ib] < 0.0) wg_empty_sk[ib] = 0.0;
    }
    const auto &prefac_occ = tau > 0 ? wg_empty_sk : wg_sk;

    std::vector<double> scale(eskb[ispin].c + n_states * ikpt, eskb[ispin].c + n_states * (ikpt + 1));
    for (int ib = 0; ib < n_states; ib++)
    {
        scale[ib] = -tau * (scale[ib] - efermi);
        if (scale[ib] > 0) scale[ib] = 0.0;
        scale[ib] = !band_mask.empty() && !band_mask[ib] ? 0.0
            : std::exp(scale[ib]) * prefac_occ[ib];
        if (tau <= 0) scale[ib] *= -1.0;
    }
    const auto wfc_bra = find_wfc(ispin, ispinor_bra, ikpt);
    const auto wfc_ket = find_wfc(ispin, ispinor_ket, ikpt);

    if (wfc_bra == nullptr)
        throw LIBRPA_RUNTIME_ERROR("wfc of ispinor_bra not found");
    if (wfc_ket == nullptr)
        throw LIBRPA_RUNTIME_ERROR("wfc of ispinor_ket not found");

    auto scaled_wfc_conj = conj(*wfc_ket);
    for (int ib = 0; ib < n_states; ib++)
        LapackConnector::scal(n_aos, scale[ib], scaled_wfc_conj.c + n_aos * ib, 1);
    return transpose(*wfc_bra, false) * scaled_wfc_conj;
}

std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> MeanField::get_gf_cplx_imagtimes_Rs(
    int ispin, int ispinor_bra, int ispinor_ket, const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const std::vector<bool> &band_mask) const
{
    if (!band_mask.empty() && band_mask.size() != static_cast<std::size_t>(n_states))
        throw LIBRPA_RUNTIME_ERROR("Green's-function band mask has inconsistent size");

    std::map<double, std::map<Vector3_Order<int>, ComplexMatrix>> gf_tau_R;
    const double scale_spin = 0.5 * n_spins * n_spinor;
    // NOTE: occupation must be copied here, not reference
    auto wg_empty = wg[ispin];
    // cout << "In get_gf_cplx_imagtimes_Rs ispin " << ispin << endl << wg_empty << endl;
    for (size_t i = 0; i != wg_empty.size; i++)
    {
        wg_empty.c[i] = 1.0 / n_kpoints - wg_empty.c[i] * scale_spin;
        if (wg_empty.c[i] < 0) wg_empty.c[i] = 0;
        // printf("%d %f\n", i, wg_empty.c[i]);
    }
    // cout << "wg_empty " << wg_empty << endl;
    const auto wg_occ = wg[ispin] * scale_spin;
    // cout << "wg_occ " << wg_occ << endl;
    for (const auto &tau : imagtimes)
    {
        gf_tau_R[tau] = {};
        // cout << "tau " << tau << endl;
        // Empty local R, cycle after initialize the tau container
        if (Rs.size() == 0) continue;
        const auto &prefac_occ = tau > 0 ? wg_empty : wg_occ;
        // cout << "prefac_occ " << prefac_occ << endl;
        const auto scale = -tau * (eskb[ispin] - efermi);
        for (size_t ie = 0; ie != scale.size; ie++)
        {
            if (scale.c[ie] > 0) scale.c[ie] = 0;
            scale.c[ie] = std::exp(scale.c[ie]) * prefac_occ.c[ie];
        }
        // ofs_myid cout << "tau " << tau << endl << scale << endl;
        for (int ik = 0; ik != n_kpoints; ik++)
        {
            const auto wfc_ket = find_wfc(ispin, ispinor_ket, ik);
            if (wfc_ket == nullptr)
                throw LIBRPA_RUNTIME_ERROR("wfc of ispinor_ket not found");
            auto scaled_wfc_conj = conj(*wfc_ket);
            for (int ib = 0; ib != n_states; ib++)
            {
                const auto scale_ib = !band_mask.empty() && !band_mask[ib] ? 0.0 : scale(ik, ib);
                LapackConnector::scal(n_aos, scale_ib, scaled_wfc_conj.c + n_aos * ib, 1);
            }
            const auto wfc_bra = find_wfc(ispin, ispinor_bra, ik);
            if (wfc_bra == nullptr)
                throw LIBRPA_RUNTIME_ERROR("wfc of ispinor_bra not found");
            const auto gf_k = transpose(*wfc_bra, false) * scaled_wfc_conj;
            for (const auto &R : Rs)
            {
                double ang = -kfrac_list[ik] * R * TWO_PI;
                auto kphase = std::complex<double>(cos(ang), sin(ang));
                auto phase = kphase * (tau > 0 ? 1.0 : -1.0);
                if (gf_tau_R.count(tau) == 0 || gf_tau_R.at(tau).count(R) == 0)
                {
                    gf_tau_R[tau][R].create(n_aos, n_aos);
                }
                gf_tau_R[tau][R] += gf_k * phase;
            }
        }
        // zmy debug, print
        // for (const auto &R: Rs)
        // {
        //     cout << "tau " << tau << " R " << R << endl;
        //     print_complex_matrix("", gf_tau_R[tau][R]);
        // }
    }
    return gf_tau_R;
}

std::map<double, std::map<Vector3_Order<int>, matrix>> MeanField::get_gf_real_imagtimes_Rs(
    int ispin, int ispinor_bra, int ispinor_ket,
    const std::vector<Vector3_Order<double>> &kfrac_list, std::vector<double> imagtimes,
    const std::vector<Vector3_Order<int>> &Rs, const std::vector<bool> &band_mask) const
{
    std::map<double, std::map<Vector3_Order<int>, matrix>> gf_tau_R;
    for (const auto &tau_gf_cplx_R :
         this->get_gf_cplx_imagtimes_Rs(ispin, ispinor_bra, ispinor_ket, kfrac_list, imagtimes, Rs, band_mask))
    {
        const auto &tau = tau_gf_cplx_R.first;
        gf_tau_R[tau] = {};
        for (const auto &R_gf_cplx: tau_gf_cplx_R.second)
        {
            const auto &R = R_gf_cplx.first;
            gf_tau_R[tau][R] = R_gf_cplx.second.real();
            // cout << "tau " << tau << " R " << R << endl;
            // print_matrix("", gf_tau_R[tau][R]);
        }
    }
    return gf_tau_R;
}

// void MeanField::allredue_wfc_isk()
// {
//     using librpa_int::global::mpi_comm_global_h;
// 
//     for(int is=0;is!=n_spins;is++)
//         for(int ik=0;ik!=n_kpoints;ik++)
//             {
//                 ComplexMatrix loc_wfc(n_states,n_aos);
//                 ComplexMatrix glo_wfc(n_states,n_aos);
//                 // if(mpi_comm_world_h.is_root())
//                 // {
//                 //     loc_wfc=wfc[is][ik];
//                 // }
//                 // mpi_comm_world_h.allreduce_ComplexMatrix(loc_wfc,glo_wfc);
//                 // if(!mpi_comm_world_h.is_root())
//                 // {
//                 //     wfc[is][ik]=glo_wfc;
//                 // }
//                 librpa_int::allreduce_ComplexMatrix(wfc[is][ik],glo_wfc,mpi_comm_global_h.comm);
//                 wfc[is][ik]=glo_wfc;
//             }
// }

}
