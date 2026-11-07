#include "qpoint_view.h"

#include <cstddef>
#include <utility>

#include "../math/symmetry.h"
#include "../utils/error.h"
#include "pbc.h"
#include "symmetry_context.h"

namespace librpa_int
{

static constexpr double kQPointViewCoordTol = 1e-5;

static SymmetryQPointView build_pbc_qpoint_view(const PeriodicBoundaryData& pbc)
{
    SymmetryQPointView view;
    view.representatives = pbc.klist_coul;
    bool restores_time_reversal = false;
    const double fallback_weight =
        pbc.get_n_cells_bvk() > 0 ? 1.0 / static_cast<double>(pbc.get_n_cells_bvk()) : 1.0;
    for (const auto& q : view.representatives)
    {
        const auto members_iter = pbc.map_irk_ks.find(q);
        if (members_iter != pbc.map_irk_ks.end())
        {
            view.members[q] = members_iter->second;
            for (const auto& member : members_iter->second)
            {
                if (member != q)
                {
                    restores_time_reversal = true;
                    break;
                }
            }
        }
        else
        {
            view.members[q] = {q};
        }
        const auto weight_iter = pbc.map_q_weight.find(q);
        view.weights[q] = weight_iter == pbc.map_q_weight.end()
                              ? fallback_weight * static_cast<double>(view.members[q].size())
                              : weight_iter->second;
    }
    view.restore_mode =
        restores_time_reversal ? SymmetryQPointRestoreMode::TIME_REVERSAL
                               : SymmetryQPointRestoreMode::NONE;
    return view;
}

SymmetryQPointView build_symmetry_qpoint_view(const SymmetryContext& ctx,
                                              const PeriodicBoundaryData& pbc,
                                              const bool use_symmetry)
{
    // Decided by exact list lengths; a list that matches neither count is
    // reported by classify_kpoint_input_form instead of being treated as the
    // nearer of the two.  When symmetry is off, only the full-grid form can
    // reproduce every member q-point on its own.
    const auto input_form = classify_kpoint_input_form(ctx, pbc);
    const bool symmetry_reduced_input = input_form == KPointInputForm::Reduced;
    if (!use_symmetry)
    {
        if (symmetry_reduced_input)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "K-point input lists one k-point per k-star, so the full q grid can only be "
                "recovered with symmetry; enable use_symmetry_rpa/use_symmetry_gw or supply "
                "every k-point of the Brillouin zone");
        }
        return build_pbc_qpoint_view(pbc);
    }
    const auto pbc_view = build_pbc_qpoint_view(pbc);

    if (!ctx.available || ctx.kstars.empty() || ctx.kstar_grid_mapping.empty())
    {
        if (symmetry_reduced_input)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "Symmetry-reduced k-point input requires a populated symmetry context");
        }
        return pbc_view;
    }
    const bool implicit_identity_group =
        ctx.rspace_operations.size() <= 1 && !ctx.has_explicit_spin_operations;
    if (!symmetry_reduced_input && (ctx.kspace_actions.size() <= 1 || implicit_identity_group))
    {
        return pbc_view;
    }
    const auto full_grid_representative_indices =
        build_symmetry_full_grid_kstar_representative_indices(ctx, pbc.kfrac_list);
    const auto full_grid_member_targets =
        build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, pbc.kfrac_list);
    const bool use_full_grid_kstars =
        full_grid_representative_indices.size() == ctx.kstars.size()
        && full_grid_member_targets.size() == ctx.kstars.size();
    if (!symmetry_reduced_input && !use_full_grid_kstars)
    {
        return pbc_view;
    }

    std::vector<const SymmetryKStarGridMappingEntry*> representative_entries(ctx.kstars.size(),
                                                                            nullptr);
    for (const auto& entry : ctx.kstar_grid_mapping)
    {
        if (entry.star_list_index < 0
            || entry.star_list_index >= static_cast<int>(ctx.kstars.size())
            || entry.iq_ibz < 0
            || entry.iq_ibz >= static_cast<int>(pbc.klist.size())
            || entry.iq_ibz >= static_cast<int>(pbc.kfrac_list.size()))
        {
            throw LIBRPA_RUNTIME_ERROR("Symmetry q-point view found an invalid k-star mapping");
        }
        auto& selected =
            representative_entries[static_cast<std::size_t>(entry.star_list_index)];
        const auto& star = ctx.kstars[static_cast<std::size_t>(entry.star_list_index)];
        if (full_grid_representative_indices.size() == ctx.kstars.size()
            && entry.iq_ibz == full_grid_representative_indices[
                                   static_cast<std::size_t>(entry.star_list_index)])
        {
            selected = &entry;
            continue;
        }
        const bool is_star_representative =
            same_fractional_kpoint(pbc.kfrac_list[static_cast<std::size_t>(entry.iq_ibz)],
                                   star.k_ibz, kQPointViewCoordTol);
        if (selected == nullptr || is_star_representative)
        {
            selected = &entry;
        }
    }

    SymmetryQPointView view;
    view.restore_mode = SymmetryQPointRestoreMode::FULL_CRYSTAL;
    const double inv_nk = 1.0 / static_cast<double>(pbc.get_n_cells_bvk());
    for (std::size_t istar = 0; istar != ctx.kstars.size(); ++istar)
    {
        const auto* entry = representative_entries[istar];
        if (entry == nullptr)
        {
            throw LIBRPA_RUNTIME_ERROR("Symmetry q-point view could not choose a k-star representative");
        }
        const auto& q_rep = pbc.klist[static_cast<std::size_t>(entry->iq_ibz)];
        std::vector<Vector3_Order<double>> members;
        if (use_full_grid_kstars)
        {
            members.reserve(full_grid_member_targets[istar].size());
            for (const auto& member_frac : full_grid_member_targets[istar])
            {
                members.push_back(Vector3_Order<double>{member_frac * pbc.G});
            }
        }
        else
        {
            members = entry->member_q_bz_keys;
        }
        if (members.empty())
        {
            members = {q_rep};
        }
        view.representatives.push_back(q_rep);
        view.members[q_rep] = std::move(members);
        view.weights[q_rep] =
            static_cast<double>(view.members.at(q_rep).size()) * inv_nk;
    }

    bool actually_reduced = view.representatives.size() != pbc.klist_coul.size();
    if (!actually_reduced)
    {
        for (const auto& q : view.representatives)
        {
            const auto& members = view.members.at(q);
            if (members.size() != 1 || members.front() != q)
            {
                actually_reduced = true;
                break;
            }
        }
    }
    if (!actually_reduced)
    {
        view.restore_mode = SymmetryQPointRestoreMode::NONE;
    }
    return view;
}

} // namespace librpa_int
