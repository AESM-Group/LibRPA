#include "chi0.h"

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <ctime>
#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <valarray>

#include "../io/global_io.h"
#include "../io/stl_io_helper.h"
#include "../math/complexmatrix.h"
#include "../math/lapack_connector.h"
#include "../math/matrix.h"
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_mpi.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../mpi/base_blacs.h"
#include "../mpi/two_level_parallel_context.h"
#include "../utils/base_utility.h"
#include "../utils/constants.h"
#include "../io/global_io.h"
#include "../utils/dev_options.h"
#include "../utils/error.h"
#include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "../utils/utils_mem.h"
#include "symmetry_context.h"
#include "atomic_basis.h"
#include "meanfield_mpi.h"
#include "pbc.h"
#include "ri.h"
#include "utils_atomic_basis_blacs.h"
#ifdef LIBRPA_USE_LIBRI
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
#include <RI/parallel/Parallel_LRI_Equally_Weighted.h>
#endif
#include <RI/physics/RPA.h>
#include <RI/physics/symmetry/Symmetry_Filter.h>
#endif
#include <array>
#include <map>


namespace librpa_int {

using std::map;
using std::pair;
using std::vector;

constexpr int SHRINK_SCALAPACK_BLOCK_CAP = 2048;

using Chi0CollectKey = std::pair<int, std::array<int, 3>>;
using Chi0BlockKey = Chi0CollectKey;

template <typename Tdata>
using Chi0CollectMap = std::map<int, std::map<Chi0BlockKey, RI::Tensor<Tdata>>>;

using Chi0CollectRequest = std::pair<std::set<int>, std::set<int>>;
using Chi0ExactCollectRequest = std::pair<std::set<int>, std::set<Chi0CollectKey>>;

using Chi0QBlockKey = std::pair<int, std::array<double, 3>>;
using Chi0QCollectMap = std::map<int, std::map<Chi0QBlockKey, RI::Tensor<std::complex<double>>>>;
using Chi0QCollectRequest = std::pair<std::set<int>, std::set<int>>;

static void create_chi0_q_blocks(
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q,
    const std::vector<double> &freqs,
    const std::vector<Vector3_Order<double>> &qlist,
    const std::vector<atpair_t> &atpairs,
    const AtomicBasis &atbasis_abf)
{
    global::profiler.start(__FUNCTION__);
    for (const auto freq : freqs)
    {
        for (const auto &q : qlist)
        {
            for (const auto atpair : atpairs)
            {
                const auto Mu = atpair.first;
                const auto Nu = atpair.second;
                auto &chi = chi0_q[freq][q][Mu][Nu];
                if (chi.size == 0)
                    chi.create(atbasis_abf[Mu], atbasis_abf[Nu]);
            }
        }
    }
    global::profiler.stop(__FUNCTION__);
}

static TwoLevelProcessShape resolve_chi0_q_uhap_process_shape(
    const int nprocs, const std::size_t nqpoints, const std::size_t nuhap)
{
    if (nprocs <= 1 || nqpoints == 0 || nuhap == 0)
        return TwoLevelProcessShape(1, std::max(1, nprocs));

    const int max_outer =
        static_cast<int>(std::min(nqpoints, static_cast<std::size_t>(nprocs)));
    for (int nouter = max_outer; nouter != 0; --nouter)
    {
        if (nprocs % nouter != 0)
            continue;
        const int ninner = nprocs / nouter;
        if (static_cast<std::size_t>(ninner) > nuhap)
            continue;
        return TwoLevelProcessShape(nouter, ninner);
    }
    return TwoLevelProcessShape(1, nprocs);
}

template <typename Tdata>
static Chi0CollectMap<Tdata> collect_chi0_map2_first(
    MPI_Comm comm, Chi0CollectMap<Tdata> &chi0s, const Chi0CollectRequest &request)
{
    global::profiler.start("chi0_collect_comm_map2_first", LIBRPA_VERBOSE_DEBUG);
    auto result = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        comm, chi0s, request.first, request.second);
    global::profiler.stop("chi0_collect_comm_map2_first");
    return result;
}

#ifdef LIBRPA_USE_LIBRI
template <typename Tdata>
static Chi0CollectMap<Tdata> collect_chi0_map2(
    MPI_Comm comm, Chi0CollectMap<Tdata> &chi0s, const Chi0ExactCollectRequest &request)
{
    global::profiler.start("chi0_collect_comm_map2", LIBRPA_VERBOSE_DEBUG);
    auto result = RI::Communicate_Tensors_Map_Judge::comm_map2(
        comm, chi0s, request.first, request.second);
    global::profiler.stop("chi0_collect_comm_map2");
    return result;
}
#endif

static Chi0QCollectRequest make_chi0_q_collect_request(
    const std::vector<Vector3_Order<double>> &qlist, const std::vector<atpair_t> &atpairs)
{
    Chi0QCollectRequest request;
    if (qlist.empty())
        return request;
    for (const auto &atpair : atpairs)
    {
        request.first.insert(as_int(atpair.first));
        request.second.insert(as_int(atpair.second));
    }
    return request;
}

static Chi0QCollectRequest make_padding_chi0_q_collect_request(
    const std::vector<Vector3_Order<double>> &qlist, const AtomicBasis &atbasis_abf)
{
    if (qlist.empty() || atbasis_abf.n_atoms == 0)
        return {};
    return {{0}, {0}};
}

static Chi0QCollectMap pack_chi0_q_for_comm_map2(
    const map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old> &chi0_wq,
    const std::vector<Vector3_Order<double>> &qlist,
    const std::vector<atpair_t> &atpairs,
    const AtomicBasis &atbasis_abf)
{
    Chi0QCollectMap chi0_libri;
    for (const auto &q : qlist)
    {
        const auto it_q = chi0_wq.find(q);
        if (it_q == chi0_wq.end())
            continue;
        const auto qa = q.to_array();
        for (const auto &atpair : atpairs)
        {
            const int I = as_int(atpair.first);
            const int J = as_int(atpair.second);
            const auto it_I = it_q->second.find(atpair.first);
            if (it_I == it_q->second.end())
                continue;
            const auto it_J = it_I->second.find(atpair.second);
            if (it_J == it_I->second.end())
                continue;
            RI::Tensor<std::complex<double>> tensor({atbasis_abf[atpair.first],
                                                     atbasis_abf[atpair.second]});
            const auto &chi0 = it_J->second;
            for (std::size_t ir = 0; ir != atbasis_abf[atpair.first]; ++ir)
            {
                for (std::size_t ic = 0; ic != atbasis_abf[atpair.second]; ++ic)
                    tensor(ir, ic) = chi0(ir, ic);
            }
            chi0_libri[I][{J, qa}] = std::move(tensor);
        }
    }
    return chi0_libri;
}

static Chi0QCollectMap collect_chi0_q_map2_first(
    MPI_Comm comm, Chi0QCollectMap &chi0_q, const Chi0QCollectRequest &request)
{
    return RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        comm, chi0_q, request.first, request.second);
}

static void unpack_chi0_q_from_comm_map2(
    const double freq,
    const Chi0QCollectMap &chi0_libri,
    const std::vector<Vector3_Order<double>> &qlist,
    const std::vector<atpair_t> &atpairs,
    const AtomicBasis &atbasis_abf,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q)
{
    for (const auto &q : qlist)
    {
        const auto qa = q.to_array();
        for (const auto &atpair : atpairs)
        {
            const int I = as_int(atpair.first);
            const int J = as_int(atpair.second);
            const auto it_I = chi0_libri.find(I);
            if (it_I == chi0_libri.end())
                throw LIBRPA_RUNTIME_ERROR("missing chi0_q atom in q/uhap redistribution");
            const auto it_Jq = it_I->second.find({J, qa});
            if (it_Jq == it_I->second.end())
                throw LIBRPA_RUNTIME_ERROR("missing chi0_q block in q/uhap redistribution");
            auto &chi0 = chi0_q[freq][q][atpair.first][atpair.second];
            if (chi0.size == 0)
                chi0.create(atbasis_abf[atpair.first], atbasis_abf[atpair.second]);
            const auto &tensor = it_Jq->second;
            for (std::size_t ir = 0; ir != atbasis_abf[atpair.first]; ++ir)
            {
                for (std::size_t ic = 0; ic != atbasis_abf[atpair.second]; ++ic)
                    chi0(ir, ic) = tensor(ir, ic);
            }
        }
    }
}

static void redistribute_chi0_q_to_atom_pair_layout(
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q_src,
    const std::vector<Vector3_Order<double>> &qlist_src,
    const std::vector<atpair_t> &atpairs_src,
    const std::vector<Vector3_Order<double>> &qlist_dst,
    const std::vector<atpair_t> &atpairs_dst,
    const AtomicBasis &atbasis_abf,
    MPI_Comm comm,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q_dst)
{
    auto request = make_chi0_q_collect_request(qlist_dst, atpairs_dst);
    const bool padding_request = request.first.empty() || request.second.empty();
    if (padding_request)
        request = make_padding_chi0_q_collect_request(qlist_dst, atbasis_abf);

    while (!chi0_q_src.empty())
    {
        auto it_freq = chi0_q_src.begin();
        const double freq = it_freq->first;
        auto packed = pack_chi0_q_for_comm_map2(
            it_freq->second, qlist_src, atpairs_src, atbasis_abf);
        const auto collected = collect_chi0_q_map2_first(comm, packed, request);
        if (!padding_request)
            unpack_chi0_q_from_comm_map2(
                freq, collected, qlist_dst, atpairs_dst, atbasis_abf, chi0_q_dst);
        chi0_q_src.erase(it_freq);
    }
}

static std::size_t estimate_chi0_block_bytes(
    const AtomicBasis &atbasis_abf, const int I, const int J, const std::size_t scalar_bytes)
{
    return atbasis_abf.get_pair_matrix_size(I, J) * scalar_bytes;
}

struct Chi0CollectPlan
{
    std::size_t n_atoms = 0;
    std::size_t n_R = 0;
    std::size_t max_bytes = 0;
    std::size_t total_bytes = 0;
    std::vector<std::size_t> pair_offsets;
    std::vector<std::size_t> chunk_offsets;
    std::map<std::array<int, 3>, std::size_t> R_index;

    std::size_t nchunks() const noexcept { return chunk_offsets.size(); }
};

static std::size_t upper_pair_index(const std::size_t n_atoms, const int I_in, const int J_in)
{
    const int I_norm = std::min(I_in, J_in);
    const int J_norm = std::max(I_in, J_in);
    if (I_norm < 0 || J_norm < 0 ||
        static_cast<std::size_t>(I_norm) >= n_atoms ||
        static_cast<std::size_t>(J_norm) >= n_atoms)
        return 0;
    const std::size_t I = static_cast<std::size_t>(I_norm);
    const std::size_t J = static_cast<std::size_t>(J_norm);
    return I * n_atoms - I * (I - 1) / 2 + (J - I);
}

static std::pair<int, int> upper_pair_from_index(const std::size_t n_atoms, const std::size_t pair_index)
{
    std::size_t offset = 0;
    for (std::size_t I = 0; I != n_atoms; ++I)
    {
        const std::size_t count = n_atoms - I;
        if (pair_index < offset + count)
            return {static_cast<int>(I), static_cast<int>(I + pair_index - offset)};
        offset += count;
    }
    return {0, 0};
}

template <typename Tdata>
static Chi0CollectPlan make_chi0_collect_plan_by_bytes(
    const std::size_t n_atoms,
    const std::vector<Vector3_Order<int>> &Rlist_gf,
    const AtomicBasis &atbasis_abf,
    const std::size_t max_bytes)
{
    Chi0CollectPlan plan;
    plan.n_atoms = n_atoms;
    plan.n_R = Rlist_gf.size();
    plan.max_bytes = max_bytes;
    if (max_bytes == 0 || n_atoms == 0 || Rlist_gf.empty())
        return plan;

    for (std::size_t iR = 0; iR != Rlist_gf.size(); ++iR)
        plan.R_index[{Rlist_gf[iR].x, Rlist_gf[iR].y, Rlist_gf[iR].z}] = iR;

    plan.chunk_offsets.push_back(0);
    std::size_t total = 0;
    std::size_t chunk_bytes = 0;
    for (std::size_t I = 0; I != n_atoms; ++I)
    {
        for (std::size_t J = I; J != n_atoms; ++J)
        {
            plan.pair_offsets.push_back(total);
            const auto block_bytes = estimate_chi0_block_bytes(
                atbasis_abf, static_cast<int>(I), static_cast<int>(J), sizeof(Tdata));
            for (std::size_t iR = 0; iR != Rlist_gf.size(); ++iR)
            {
                if (chunk_bytes > 0 && chunk_bytes + block_bytes > max_bytes)
                {
                    plan.chunk_offsets.push_back(total);
                    chunk_bytes = 0;
                }
                total += block_bytes;
                chunk_bytes += block_bytes;
                if (chunk_bytes >= max_bytes)
                {
                    plan.chunk_offsets.push_back(total);
                    chunk_bytes = 0;
                }
            }
        }
    }
    plan.pair_offsets.push_back(total);
    if (!plan.chunk_offsets.empty() && plan.chunk_offsets.back() == total)
        plan.chunk_offsets.pop_back();
    plan.total_bytes = total;
    return plan;
}

static std::size_t chunk_index_for_chi0_key(
    const Chi0CollectPlan &plan,
    const int I,
    const int J,
    const std::array<int, 3> &R)
{
    if (plan.nchunks() == 0)
        return 0;
    const auto it_R = plan.R_index.find(R);
    if (it_R == plan.R_index.end())
        return 0;
    const auto pair_idx = upper_pair_index(plan.n_atoms, I, J);
    if (pair_idx + 1 >= plan.pair_offsets.size())
        return 0;
    const auto pair_offset = plan.pair_offsets[pair_idx];
    const auto pair_bytes = plan.pair_offsets[pair_idx + 1] - pair_offset;
    const auto block_bytes = plan.n_R == 0 ? 0 : pair_bytes / plan.n_R;
    const auto byte_offset = pair_offset + it_R->second * block_bytes;
    auto it = std::upper_bound(plan.chunk_offsets.begin(), plan.chunk_offsets.end(), byte_offset);
    if (it == plan.chunk_offsets.begin())
        return 0;
    const auto idx = static_cast<std::size_t>((it - plan.chunk_offsets.begin()) - 1);
    return std::min(idx, plan.nchunks() - 1);
}

template <typename Tdata>
static std::vector<Chi0CollectRequest> make_local_chi0_request_chunks(
    const Chi0CollectPlan &plan,
    const std::vector<atpair_t> &atpairs_ABF,
    const std::vector<Vector3_Order<int>> &Rlist_gf)
{
    std::vector<Chi0CollectRequest> chunks(plan.nchunks());
    if (plan.nchunks() == 0)
        return chunks;
    for (const auto &atpair : atpairs_ABF)
    {
        const int I = static_cast<int>(atpair.first);
        const int J = static_cast<int>(atpair.second);
        for (const auto &Rvec : Rlist_gf)
        {
            const std::array<int, 3> R{Rvec.x, Rvec.y, Rvec.z};
            const auto ichunk = chunk_index_for_chi0_key(plan, I, J, R);
            chunks[ichunk].first.insert(I);
            chunks[ichunk].second.insert(J);
        }
    }
    return chunks;
}

static Chi0ExactCollectRequest make_chi0_exact_collect_request(
    const std::vector<atpair_t> &atpairs_ABF,
    const std::vector<Vector3_Order<int>> &Rlist_gf)
{
    Chi0ExactCollectRequest request;
    for (const auto &atpair : atpairs_ABF)
    {
        const int I = static_cast<int>(atpair.first);
        const int J = static_cast<int>(atpair.second);
        request.first.insert(I);
        for (const auto &Rvec : Rlist_gf)
        {
            const std::array<int, 3> R{Rvec.x, Rvec.y, Rvec.z};
            request.second.insert({J, R});
        }
    }
    return request;
}

static Chi0ExactCollectRequest make_padding_chi0_exact_collect_request(
    const std::vector<Vector3_Order<int>> &Rlist_gf, const AtomicBasis &atbasis_abf)
{
    if (Rlist_gf.empty() || atbasis_abf.n_atoms == 0)
        return {};
    const std::array<int, 3> R{Rlist_gf.front().x, Rlist_gf.front().y, Rlist_gf.front().z};
    return {{0}, {{0, R}}};
}

// An arbitrary band cutoff can split a little-group multiplet, so real-space
// symmetry restoration is guaranteed exact only in the complete AO space.
bool rspace_symmetry_has_complete_band_space(const MeanField &mf, const int n_bands)
{
    const int n_bands_used = n_bands < 0 ? mf.get_n_bands() : n_bands;
    return n_bands_used == mf.get_n_aos()
        && n_bands_used <= mf.get_n_bands();
}

#ifdef LIBRPA_USE_LIBRI
static std::array<int, 3> canonicalize_chi0_symmetry_R(
    const std::array<int, 3> &R,
    const std::array<int, 3> &period)
{
    const auto centered_mod = [](const int value, const int cell_period) {
        if (cell_period <= 0)
            return value;
        return (value % cell_period + 3 * cell_period / 2) % cell_period
            - cell_period / 2;
    };
    return {centered_mod(R[0], period[0]),
            centered_mod(R[1], period[1]),
            centered_mod(R[2], period[2])};
}

static std::map<std::pair<int, int>, std::set<std::array<int, 3>>>
convert_symmetry_irreducible_sector_to_libri_chi0(
    const symmetry_irreducible_sector_t &irreducible_sector,
    const std::array<int, 3> &period)
{
    std::map<std::pair<int, int>, std::set<std::array<int, 3>>> libri_sector;
    for (const auto &pair_Rs : irreducible_sector)
    {
        const std::pair<int, int> atom_pair{
            as_int(pair_Rs.first.first),
            as_int(pair_Rs.first.second)};
        for (const auto &R : pair_Rs.second)
            libri_sector[atom_pair].insert(canonicalize_chi0_symmetry_R(R, period));
    }
    return libri_sector;
}

static Chi0ExactCollectRequest make_chi0_symmetry_collect_request(
    const symmetry_rspace_sector_stars_t &sector_stars,
    const std::vector<atpair_t> &target_atpairs,
    std::vector<atpair_t> &irreducible_atpairs)
{
    const std::set<atpair_t> target_set(target_atpairs.begin(), target_atpairs.end());
    std::set<atpair_t> irreducible_set;
    Chi0ExactCollectRequest request;

    for (const auto &pair_star : sector_stars)
    {
        const auto &ir_pair = pair_star.first;
        for (const auto &R_star : pair_star.second)
        {
            const auto &ir_R = R_star.first;
            const bool needed =
                std::any_of(R_star.second.begin(), R_star.second.end(),
                            [&target_set](const SymmetryRSpaceRestoreMember &member) {
                                return target_set.count(member.full_atom_pair) != 0;
                            });
            if (!needed)
                continue;
            request.first.insert(as_int(ir_pair.first));
            request.second.insert({as_int(ir_pair.second), {ir_R.x, ir_R.y, ir_R.z}});
            irreducible_set.insert(ir_pair);
        }
    }

    irreducible_atpairs.assign(irreducible_set.begin(), irreducible_set.end());
    return request;
}

static bool can_use_chi0_rspace_symmetry(
    const SymmetryContext &symmetry_ctx,
    const AtomicBasis &abf_basis,
    const std::vector<Vector3_Order<int>> &Rlist_gf,
    const bool use_symmetry_context)
{
    if (!use_symmetry_context || !symmetry_ctx.available || Rlist_gf.empty())
        return false;
    if (!abf_basis.has_l_shells() || symmetry_ctx.rspace_operations.empty()
        || symmetry_ctx.irreducible_sector.empty()
        || symmetry_ctx.rspace_sector_stars.empty()
        || symmetry_ctx.rsh_rotations.empty())
    {
        return false;
    }
    if (symmetry_ctx.atom_to_type.size() != static_cast<std::size_t>(abf_basis.n_atoms)
        || symmetry_ctx.input_coord_frac.size() != static_cast<std::size_t>(abf_basis.n_atoms))
    {
        return false;
    }
    if (!symmetry_species_layouts_match_atom_counts(
            abf_basis.build_species_basis_layouts(symmetry_ctx.atom_to_type),
            symmetry_ctx.atom_to_type,
            abf_basis.get_atom_nb_map()))
    {
        return false;
    }
    const auto n_full_blocks =
        static_cast<std::size_t>(abf_basis.n_atoms) *
        static_cast<std::size_t>(abf_basis.n_atoms) *
        Rlist_gf.size();
    return symmetry_ctx.count_irreducible_blocks() < n_full_blocks;
}

template <typename TA, typename TC, typename Tdata>
class OutputOnlyFilter_Chi0_Symmetry : public RI::Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
    using TAC = std::pair<TA, TC>;

    OutputOnlyFilter_Chi0_Symmetry(
        const TC &period,
        const std::map<std::pair<TA, TA>, std::set<TC>> &irreducible_sector)
        : symmetry_(period, irreducible_sector)
    {
    }

    bool filter_for2(const RI::Label::ab_ab &label,
                     const TA &A1,
                     const TAC &A2) const override
    {
        if (label == RI::Label::ab_ab::a1b2_a2b1)
            return !this->symmetry_.in_irreducible_sector(A1, A2);
        return false;
    }

    bool filter_for32(const RI::Label::ab_ab &label,
                      const TA &A1,
                      const TAC &,
                      const TAC &A3) const override
    {
        if (label == RI::Label::ab_ab::a1b1_a2b2)
            return !this->symmetry_.in_irreducible_sector(A1, A3);
        return false;
    }

  private:
    RI::Symmetry_Filter<TA, TC, Tdata> symmetry_;
};

template <typename Tdata>
static RI::Tensor<Tdata> convert_complex_matrix_to_libri_tensor_chi0(
    const ComplexMatrix &matrix)
{
    RI::Tensor<Tdata> tensor(
        {static_cast<std::size_t>(matrix.nr), static_cast<std::size_t>(matrix.nc)});
    for (int row = 0; row != matrix.nr; ++row)
    {
        for (int col = 0; col != matrix.nc; ++col)
        {
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                tensor(row, col) = matrix(row, col);
            else
                tensor(row, col) = matrix(row, col).real();
        }
    }
    return tensor;
}

template <typename Tdata>
static Chi0CollectMap<Tdata> restore_symmetry_abf_rspace_tensor_map_chi0(
    const Chi0CollectMap<Tdata> &tensors_ir,
    const SymmetryContext &symmetry_ctx,
    const symmetry_rspace_sector_stars_t &sector_stars,
    const AtomicBasis &abf_basis,
    const std::array<int, 3> &period,
    const std::vector<atpair_t> &target_atpairs)
{
    Chi0CollectMap<Tdata> tensors_full;
    const std::set<atpair_t> target_set(target_atpairs.begin(), target_atpairs.end());
    if (target_set.empty())
        return tensors_full;

    const auto abf_layouts = abf_basis.build_species_basis_layouts(symmetry_ctx.atom_to_type);
    for (const auto &i_entry : tensors_ir)
    {
        const auto ir_I = static_cast<atom_t>(i_entry.first);
        for (const auto &jr_entry : i_entry.second)
        {
            const auto ir_J = static_cast<atom_t>(jr_entry.first.first);
            const auto ir_R_array =
                canonicalize_chi0_symmetry_R(jr_entry.first.second, period);
            const Vector3_Order<int> ir_R{
                ir_R_array[0], ir_R_array[1], ir_R_array[2]};
            const auto pair_iter = sector_stars.find({ir_I, ir_J});
            if (pair_iter == sector_stars.end() || pair_iter->second.count(ir_R) == 0)
            {
                std::ostringstream oss;
                oss << "Failed to match a symmetry-filtered chi0 block with the"
                    << " irreducible-sector restore map for I=" << ir_I
                    << " J=" << ir_J << " R=(" << ir_R.x << "," << ir_R.y
                    << "," << ir_R.z << ")";
                throw LIBRPA_RUNTIME_ERROR(oss.str());
            }

            const ComplexMatrix chi0_ir = convert_libri_tensor_to_complex_matrix(
                jr_entry.second, abf_basis.get_atom_nb(ir_I), abf_basis.get_atom_nb(ir_J));
            for (const auto &restore_member : pair_iter->second.at(ir_R))
            {
                if (target_set.count(restore_member.full_atom_pair) == 0)
                    continue;
                ComplexMatrix chi0_full = rotate_symmetry_rspace_block(
                    symmetry_ctx, abf_layouts, restore_member.isym, ir_I, ir_J, chi0_ir);
                // Charge-channel response under an antiunitary (g, U_s, eta=1)
                // operation: the spin trace cancels U_s and the orbital-rotated
                // block is complex conjugated (bosonic imaginary-time/frequency
                // convention, same as the dense Wc q-star TR handling).
                if (symmetry_rspace_restore_member_is_antiunitary(symmetry_ctx, restore_member))
                {
                    chi0_full = conj(chi0_full);
                }
                auto &target = tensors_full[as_int(restore_member.full_atom_pair.first)][{
                    as_int(restore_member.full_atom_pair.second),
                    {restore_member.full_R.x,
                     restore_member.full_R.y,
                     restore_member.full_R.z}}];
                if (!target.empty())
                {
                    throw LIBRPA_RUNTIME_ERROR(
                        "Duplicate full-sector chi0 block appears during symmetry restore");
                }
                target = convert_complex_matrix_to_libri_tensor_chi0<Tdata>(chi0_full);
            }
        }
    }
    return tensors_full;
}
#endif

template <typename Tdata>
static std::vector<Chi0CollectMap<Tdata>> split_chi0_map_by_collect_plan(
    const Chi0CollectPlan &plan,
    Chi0CollectMap<Tdata> &chi0s)
{
    std::vector<Chi0CollectMap<Tdata>> chunks(plan.nchunks());
    if (plan.nchunks() == 0)
        return chunks;
    for (auto &I_JRs : chi0s)
    {
        const int I = I_JRs.first;
        for (auto &JR_tensor : I_JRs.second)
        {
            const int J = JR_tensor.first.first;
            const auto &R = JR_tensor.first.second;
            const auto ichunk = chunk_index_for_chi0_key(plan, I, J, R);
            chunks[ichunk][I][JR_tensor.first] = std::move(JR_tensor.second);
        }
    }
    chi0s.clear();
    return chunks;
}

static Chi0CollectRequest padding_s0_s1_for_plan_chunk(
    const Chi0CollectPlan &plan, const std::size_t ichunk)
{
    if (plan.nchunks() == 0 || plan.n_R == 0)
        return {{}, {}};
    const auto offset = plan.chunk_offsets[std::min(ichunk, plan.nchunks() - 1)];
    auto it = std::upper_bound(plan.pair_offsets.begin(), plan.pair_offsets.end(), offset);
    if (it == plan.pair_offsets.begin())
        return {{0}, {0}};
    const auto pair_idx = static_cast<std::size_t>((it - plan.pair_offsets.begin()) - 1);
    const auto IJ = upper_pair_from_index(plan.n_atoms, pair_idx);
    return {{IJ.first}, {IJ.second}};
}

template <typename Tdata>
static Chi0CollectMap<Tdata> take_chi0_collect_s0_chunk(
    Chi0CollectMap<Tdata> &chi0s, const std::set<int> &s0_chunk)
{
    Chi0CollectMap<Tdata> selected;
    for (auto it_I = chi0s.begin(); it_I != chi0s.end(); )
    {
        if (s0_chunk.count(it_I->first) == 0)
        {
            ++it_I;
            continue;
        }
        selected[it_I->first] = std::move(it_I->second);
        it_I = chi0s.erase(it_I);
    }
    return selected;
}

template <typename Tdata>
static void accumulate_chi0_collect_map(
    Chi0CollectMap<Tdata> &dst, const Chi0CollectMap<Tdata> &src)
{
    for (const auto &IJRc : src)
    {
        const auto I = IJRc.first;
        for (const auto &JRc : IJRc.second)
        {
            const auto J = JRc.first.first;
            const auto R = JRc.first.second;
            const auto &chi0 = JRc.second;
            if (dst[I][{J, R}].empty())
            {
                dst[I][{J, R}] = RI::Tensor<Tdata>({chi0.shape[0], chi0.shape[1]});
            }
            dst[I][{J, R}] += chi0;
        }
    }
}

static void reduce_chi0_q_partial_to_q_owner(
    const map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q_partial,
    const std::vector<Vector3_Order<double>> &qlist_owner,
    const std::vector<atpair_t> &atpairs,
    const AtomicBasis &atbasis_abf,
    const int q_owner,
    MPI_Comm comm_qpoint,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q_work)
{
    int my_q_rank = 0;
    MPI_Comm_rank(comm_qpoint, &my_q_rank);
    const bool is_q_owner = my_q_rank == q_owner;
    const std::complex<double> one(1.0, 0.0);

    for (const auto &freq_q : chi0_q_partial)
    {
        const double freq = freq_q.first;
        for (const auto &q : qlist_owner)
        {
            for (const auto &atpair : atpairs)
            {
                const auto Mu = atpair.first;
                const auto Nu = atpair.second;
                const auto &chi0_partial = freq_q.second.at(q).at(Mu).at(Nu);
                if (chi0_partial.size == 0)
                    continue;
                std::vector<std::complex<double>> chi0_reduced;
                if (is_q_owner)
                    chi0_reduced.resize(chi0_partial.size);
                MPI_Reduce(chi0_partial.c,
                           is_q_owner ? chi0_reduced.data() : chi0_partial.c,
                           chi0_partial.size, MPI_DOUBLE_COMPLEX, MPI_SUM,
                           q_owner, comm_qpoint);
                if (!is_q_owner)
                    continue;
                auto &chi0 = chi0_q_work.at(freq).at(q).at(Mu).at(Nu);
                if (chi0.size == 0)
                    chi0.create(atbasis_abf[Mu], atbasis_abf[Nu]);
                LapackConnector::axpy(chi0.size, one, chi0_reduced.data(), 1, chi0.c, 1);
            }
        }
    }
}

Chi0::Chi0(const MeanField &mf_in, const AtomicBasis &atbasis_wfc_in,
           const AtomicBasis &atbasis_abf_in, const PeriodicBoundaryData &pbc_in,
           const SymmetryContext &symmetry_context_in,
           const TFGrids &tfg_in, const KPointBlacsParallelContext &kblacs_ctxt_in,
           const ArrayDesc &desc_wfc_in, bool is_mf_eigvec_k_distributed,
           const bool use_symmetry_context_in)
    : qpoint_view_(build_symmetry_qpoint_view(symmetry_context_in, pbc_in, use_symmetry_context_in)),
      mf(mf_in),
      desc_wfc(desc_wfc_in),
      atbasis_wfc(atbasis_wfc_in),
      atbasis_abf(atbasis_abf_in),
      pbc(pbc_in),
      symmetry_context(symmetry_context_in),
      use_symmetry_context(use_symmetry_context_in),
      tfg(tfg_in),
      comm_h(kblacs_ctxt_in.comm_global_h),
      kblacs_ctxt(kblacs_ctxt_in)
{
    comm_h.check_initialized();
    is_mf_eigvec_k_distributed_ = is_mf_eigvec_k_distributed;
    // Runtime options
    gf_threshold = 1e-9;
    libri_threshold_C = 0.0;
    libri_threshold_G = 0.0;
    libri_collect_s0_chunk = 0;
    libri_collect_max_bytes = 0;
    nbands_G = -1;
}

void Chi0::build(LibrpaParallelRouting routing,
                 const Cs_LRI &Cs,
                 const std::vector<atpair_t> &atpairs_ABF,
                 const AtomicBasis &abf_Cs,
                 std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
                 const BlacsCtxtHandler &blacs_ctxt_h)
{
    using librpa_int::global::lib_printf;

    if (nbands_G > mf.get_n_bands())
        throw LIBRPA_RUNTIME_ERROR("n_bands_chi0 exceeds the number of input bands");
    band_mask.clear();
    if (nbands_G >= 0 || n_bands_exclude > 0)
    {
        band_mask.assign(mf.get_n_bands(), false);
        const auto n_bands_used = nbands_G < 0 ? mf.get_n_bands() : nbands_G;
        std::fill_n(band_mask.begin(), n_bands_used, true);
        std::fill_n(band_mask.begin(), n_bands_exclude, false);
    }

    gf_save = gf_discard = 0;
    // reset chi0_q in case the method was called before
    chi0_q.clear();

    bool use_space_time = false;

    // If not using shrinking (sinvS is empty), abf_Cs must be the same as the initial atbasis_abf
    if (sinvS.size() == 0)
    {
        if (abf_Cs != atbasis_abf)
            throw LIBRPA_RUNTIME_ERROR("abf_Cs != atbasis_abf for non-shrinking chi0");
    }
    // Switch on space-time method when time grids and weights are available
    if (tfg.has_time_grids())
    {
        use_space_time = true;
    }

    if (comm_h.is_root())
        tfg.show();
    comm_h.barrier();

    const int natom = atbasis_abf.n_atoms;

    // use space-time method
    if (use_space_time)
    {
        for ( auto R: this->pbc.Rlist )
            Rlist_gf.push_back(R);

        if (routing == LIBRPA_ROUTING_LIBRI)
        {
            const auto atpairs_gf = generate_atom_pair_from_nat(natom, true);
            if (comm_h.is_root())
                global::lib_printf("Total count of GFs IJR: %zu\n",
                                               atpairs_gf.size() * Rlist_gf.size());
            this->IJRs_gf_local = librpa_int::dispatch_vector_prod(
                atpairs_gf, Rlist_gf, comm_h.myid, comm_h.nprocs, true, true);
            global::lib_printf("| Number of GFs IJR on Proc %4d: %zu\n", comm_h.myid,
                                           this->IJRs_gf_local.size());
        }
        else
        {
            for (auto tau : tfg.get_time_nodes())
            {
                for (auto R : Rlist_gf)
                {
                    build_gf_Rt(R, tau);
                    build_gf_Rt(R, -tau);
                }
            }
            if (comm_h.is_root())
            {
                // cout << " Green threshold: " << gf_R_threshold << endl;
                global::lib_printf("Finished construction of R-space Green's function\n");
                global::lib_printf("| Saved Green's function:     %zu\n", gf_save);
                global::lib_printf("| Discarded Green's function: %zu\n\n", gf_discard);
            }
        }
        build_chi0_q_space_time(routing, Cs, atpairs_ABF, abf_Cs, sinvS, blacs_ctxt_h);
    }
    else
    {
        // conventional method does not need to build Green's function explicitly
        build_chi0_q_conventional(Cs, atpairs_ABF);
    }

    // Free the intermediate Green's functions to lower memory load
    this->free_gf_Rt();
}

void Chi0::build_gf_Rt(Vector3_Order<int> R, double tau)
{
    global::profiler.start("cal_Green_func", "space-time Green's function");

    const auto nkpts = mf.get_n_kpoints();
    const auto nspins = mf.get_n_spins();
    const auto nbands = mf.get_n_bands();
    const auto naos = mf.get_n_aos();
    const int natom = atbasis_abf.n_atoms;

    const auto nsoc = 1; // TODO replace with meanfield member variable
    const bool use_soc = mf.get_n_spinor() > 1;

    assert(tau != 0);

    // temporary Green's function
    matrix gf_Rt_is_global(naos, naos);

    for (int is = 0; is != nspins; is++)
    {
        for (int isoc1 = 0; isoc1 != nsoc; isoc1++)
        {
            for (int isoc2 = 0; isoc2 != nsoc; isoc2++)
            {
                gf_Rt_is_global.zero_out();
                if (is_mf_eigvec_k_distributed_)
                {
                    const auto gf_tau_R = get_gf_cplx_imagtimes_Rs_kpara(is, isoc1, isoc2, this->mf, pbc.kfrac_list, {tau}, {R}, comm_h, band_mask);
                    gf_Rt_is_global += gf_tau_R.at(tau).at(R).real();
                }
                else
                {
                    auto wg = mf.get_weight()[is];
                    if (tau > 0)
                        for (std::size_t i = 0; i != wg.size; i++)
                        {
                            // wg.c[i] = 1.0 / nkpts *nspins - wg.c[i];
                            if (use_soc)
                                wg.c[i] = 1.0 / nkpts - wg.c[i];
                            else
                                wg.c[i] = 1.0 / nkpts - wg.c[i] / 2 * nspins;  //
                            if (wg.c[i] < 0) wg.c[i] = 0;
                        }
                    else
                    {
                        if (!use_soc) wg *= 0.5 * nspins;
                    }
                    matrix scale(nkpts, nbands);
                    // tau-energy phase
                    scale = -tau * (mf.get_eigenvals()[is] - mf.get_efermi());
                    /* print_matrix("-(e-ef)*tau", scale); */
                    for (std::size_t ie = 0; ie != scale.size; ie++)
                    {
                        // NOTE: enforce non-positive phase
                        if (scale.c[ie] > 0) scale.c[ie] = 0;
                        scale.c[ie] = std::exp(scale.c[ie]) * wg.c[ie];
                    }
                    /* print_matrix("exp(-dE*tau)", scale); */
                    for (int ik = 0; ik != nkpts; ik++)
                    {
                        double ang = - pbc.klist[ik] * (R * pbc.latvec) * TWO_PI;
                        complex<double> kphase = complex<double>(cos(ang), sin(ang));
                        const auto &ev1 = mf.get_eigenvectors().at(is).at(isoc1).at(ik);
                        const auto &ev2 = mf.get_eigenvectors().at(is).at(isoc2).at(ik);
                        auto scaled_wfc_conj = conj(ev2);
                        for (int ib = 0; ib != nbands; ib++)
                        {
                            const auto scale_ib = !band_mask.empty() && !band_mask[ib] ? 0.0 : scale(ik, ib);
                            LapackConnector::scal(naos, scale_ib, scaled_wfc_conj.c + naos * ib, 1);
                        }
                        gf_Rt_is_global += (kphase * transpose(ev1, false) * scaled_wfc_conj).real();
                    }
                    if (tau < 0) gf_Rt_is_global *= -1.;
                }
                omp_lock_t gf_lock;
                omp_init_lock(&gf_lock);
#pragma omp parallel for schedule(dynamic)
                for (int I = 0; I != natom; I++)
                {
                    const auto I_num = atbasis_wfc[I];
                    for (int J = 0; J != natom; J++)
                    {
                        const auto J_num = atbasis_wfc[J];
                        matrix tmp_green(I_num, J_num);
                        for (size_t i = 0; i != I_num; i++)
                        {
                            size_t i_glo = atbasis_wfc.get_global_index(I, i);
                            for (size_t j = 0; j != J_num; j++)
                            {
                                size_t j_glo = atbasis_wfc.get_global_index(J, j);
                                tmp_green(i, j) = gf_Rt_is_global(i_glo, j_glo);
                            }
                        }
                        if (tmp_green.absmax() > gf_threshold)
                        {
                            // cout<<" max_green_ele:  "<<tmp_green.absmax()<<endl;
                            omp_set_lock(&gf_lock);
                            gf_is_R_tau[is][isoc1][isoc2][I][J][R][tau] = std::move(tmp_green);
                            omp_unset_lock(&gf_lock);
                            gf_save++;
                        }
                        else
                        {
                            gf_discard++;
                        }
                    }
                }
                omp_destroy_lock(&gf_lock);
#pragma omp barrier
            }
        }
    }
    global::profiler.stop("cal_Green_func");
}


void Chi0::free_gf_Rt()
{
    this->gf_is_R_tau.clear();
}


void Chi0::build_chi0_q_space_time(const LibrpaParallelRouting routing,
                                   const Cs_LRI &Cs,
                                   const vector<atpair_t> &atpairs_ABF,
                                   const AtomicBasis &abf_shrink,
                                   std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
                                   const BlacsCtxtHandler &blacs_ctxt_h)
{
    const bool use_soc = mf.get_n_spinor() > 1;
   // int R_tau_size = Rlist_gf.size() * tfg_.size();
    if (routing == LIBRPA_ROUTING_LIBRI)
    {
        if (comm_h.is_root())
        {
            std::cout << "Use LibRI for chi0" << std::endl;
        }
        if (use_soc)
            build_chi0_q_space_time_LibRI_routing<std::complex<double>>(Cs, atpairs_ABF, abf_shrink,
                                                                        sinvS, blacs_ctxt_h);
        else
            build_chi0_q_space_time_LibRI_routing<double>(Cs, atpairs_ABF, abf_shrink, sinvS,
                                                          blacs_ctxt_h);
    }
    else if (routing == LIBRPA_ROUTING_RTAU)
    {
        // if (para_mpi.is_master())
        //     cout << "R_tau_routing" << endl;
        build_chi0_q_space_time_R_tau_routing(Cs, atpairs_ABF);
    }
    else
    {
        // if (para_mpi.is_master())
        //     cout << "atom_pair_routing" << endl;
        build_chi0_q_space_time_atom_pair_routing(Cs, atpairs_ABF);
    }
}

#ifdef LIBRPA_USE_LIBRI
template <typename Tdata>
static void build_gf_Rt_libri_serial(
    const MeanField &mf, const std::vector<bool> &band_mask,
    const AtomicBasis &atbasis_wfc,
    int ispin, int isoc1, int isoc2,
    const PeriodicBoundaryData &pbc,
    const SymmetryContext &symmetry_context,
    const bool use_symmetry_context,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    double tau,
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> &gf_libri)
{
    global::profiler.start("build_gf_Rt_libri_serial");

    const auto nkpts = mf.get_n_kpoints();
    const auto nspins = mf.get_n_spins();
    const auto nbands = mf.get_n_bands();
    const auto naos = mf.get_n_aos();
    const bool use_soc = mf.get_n_spinor() > 1;

    assert(kfrac_list.size() == as_size(nkpts));

    std::map<Vector3_Order<int>, std::vector<atpair_t>> map_R_IJs;
    for (const auto &IJR : IJRs)
    {
        const auto &R = IJR.second;
        map_R_IJs[R].push_back(IJR.first);
    }
    global::ofs_myid << "map_R_IJs " << map_R_IJs << std::endl;

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    const bool can_try_symmetry_kstar_restore =
        use_symmetry_context && !wfc_layouts.empty();
    const auto full_grid_kstar_representatives =
        can_try_symmetry_kstar_restore
            ? build_symmetry_full_grid_kstar_representative_indices(
                  symmetry_context, kfrac_list)
            : symmetry_kstar_representative_indices_t{};
    bool restore_symmetry_kstars_from_full_grid =
        !full_grid_kstar_representatives.empty();
    const bool restore_symmetry_kstars =
        can_try_symmetry_kstar_restore
        && !restore_symmetry_kstars_from_full_grid
        && can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw);
    if (restore_symmetry_kstars || restore_symmetry_kstars_from_full_grid)
    {
        auto member_kfrac_targets = restore_symmetry_kstars_from_full_grid
            ? build_symmetry_full_grid_kstar_member_kfrac_targets(symmetry_context, kfrac_list)
            : build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc);
        const bool use_spinor_restore = mf.get_n_spinor() == 2;
        const auto restore_gf = [&](const std::vector<Vector3_Order<int>> &Rs_sel) {
            if (use_spinor_restore)
            {
                return extract_spinor_gf_block(
                    get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
                        symmetry_context, wfc_layouts, mf, ispin, kfrac_list, {tau}, Rs_sel,
                        atom_nw, band_mask, &member_kfrac_targets,
                        restore_symmetry_kstars_from_full_grid ? &full_grid_kstar_representatives : nullptr),
                    isoc1, isoc2);
            }
            validate_kstar_band_cutoff_closure(symmetry_context, mf, band_mask);
            return get_symmetry_restored_gf_cplx_imagtimes_Rs(
                symmetry_context, wfc_layouts, mf, ispin, isoc1, isoc2, kfrac_list, {tau}, Rs_sel,
                atom_nw, band_mask, &member_kfrac_targets,
                restore_symmetry_kstars_from_full_grid ? &full_grid_kstar_representatives : nullptr);
        };
        std::vector<Vector3_Order<int>> Rs_this;
        Rs_this.reserve(map_R_IJs.size());
        for (const auto &R_IJs : map_R_IJs)
        {
            Rs_this.push_back(R_IJs.first);
        }
        // Full-grid wavefunctions can carry a gauge that is not reproduced exactly from k-star
        // metadata. Keep the representative route only when a cheap sample matches direct full-k.
        if (restore_symmetry_kstars_from_full_grid && !Rs_this.empty())
        {
            constexpr double restore_check_tol = 1e-6;
            const std::vector<Vector3_Order<int>> R_check{Rs_this.front()};
            const auto restored_check = restore_gf(R_check).at(tau).at(R_check.front());
            const auto direct_check =
                mf.get_gf_cplx_imagtimes_Rs(
                      ispin, isoc1, isoc2, kfrac_list, {tau}, R_check, band_mask).at(tau).at(R_check.front());
            const auto diff = restored_check - direct_check;
            if (diff.get_max_abs() > restore_check_tol)
            {
                restore_symmetry_kstars_from_full_grid = false;
                member_kfrac_targets.clear();
            }
        }
        if (restore_symmetry_kstars || restore_symmetry_kstars_from_full_grid)
        {
            const auto gf_cplx_R = restore_gf(Rs_this).at(tau);

            for (const auto &R_IJs : map_R_IJs)
            {
                const auto &R = R_IJs.first;
                const auto IJs = R_IJs.second;
                const std::array<int,3> Ra{R.x,R.y,R.z};
                const auto &gf_cplx = gf_cplx_R.at(R);
                omp_lock_t gf_lock;
                omp_init_lock(&gf_lock);
#pragma omp parallel for schedule(dynamic)
                for (const auto &IJ : IJs)
                {
                    const auto &I = IJ.first;
                    const auto &J = IJ.second;
                    const auto nI = atbasis_wfc[I];
                    const auto nJ = atbasis_wfc[J];
                    auto ptr = std::make_shared<std::valarray<Tdata>>(nI * nJ);
                    for (size_t i = 0; i != nI; i++)
                    {
                        size_t i_glo = atbasis_wfc.get_global_index(I, i);
                        for (size_t j = 0; j != nJ; j++)
                        {
                            size_t j_glo = atbasis_wfc.get_global_index(J, j);
                            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                                (*ptr)[i*nJ+j] = gf_cplx(i_glo, j_glo);
                            else
                                (*ptr)[i*nJ+j] = gf_cplx(i_glo, j_glo).real();
                        }
                    }
                    omp_set_lock(&gf_lock);
                    gf_libri[I][{J, Ra}] = RI::Tensor<Tdata>({nI, nJ}, ptr);
                    omp_unset_lock(&gf_lock);
                }
                omp_destroy_lock(&gf_lock);
            }

            global::profiler.stop("build_gf_Rt_libri_serial");
            return;
        }
    }

    auto wg = mf.get_weight()[ispin];
    if (tau > 0)
    {
        for (size_t i = 0; i != wg.size; i++)
        {
            if (use_soc)
                wg.c[i] = 1.0 / nkpts - wg.c[i];
            else
                wg.c[i] = 1.0 / nkpts - wg.c[i] / 2 * nspins;  //
            if (wg.c[i] < 0) wg.c[i] = 0;
        }
    }
    else
    {
        if (!use_soc) wg *= 0.5 * nspins;
    }
    auto scale = - tau * (mf.get_eigenvals()[ispin] - mf.get_efermi());
    for (size_t ie = 0; ie != scale.size; ie++)
    {
        // NOTE: enforce non-positive phase
        if (scale.c[ie] > 0) scale.c[ie] = 0;
        scale.c[ie] = std::exp(scale.c[ie]) * wg.c[ie];
    }

    for (const auto &R_IJs : map_R_IJs)
    {
        ComplexMatrix gf_cplx(naos, naos, true);
        // matrix gf_global(naos, naos, true);

        const auto R = R_IJs.first;
        const auto IJs = R_IJs.second;
        const std::array<int,3> Ra{R.x,R.y,R.z};
        // global::ofs_myid << "Chi0 Handling IJs: " << IJs << " - R " << Ra << std::endl;

        // Compute the full G(R, tau) matrix
#pragma omp parallel for schedule(dynamic)
        for (int ik = 0; ik != nkpts; ik++)
        {
            double ang = - (kfrac_list[ik] * R) * TWO_PI;
            complex<double> kphase = complex<double>(cos(ang), sin(ang));
            const auto &ev1 = mf.get_eigenvectors().at(ispin).at(isoc1).at(ik);
            const auto &ev2 = mf.get_eigenvectors().at(ispin).at(isoc2).at(ik);
            auto scaled_wfc_conj = conj(ev2);
            for (int ib = 0; ib != nbands; ib++)
            {
                const auto scale_ib = !band_mask.empty() && !band_mask[ib] ? 0.0 : scale(ik, ib);
                LapackConnector::scal(naos, scale_ib, scaled_wfc_conj.c + naos * ib, 1);
            }
            auto mat = (kphase * transpose(ev1, false) * scaled_wfc_conj);
#pragma omp critical
            {
                gf_cplx += mat;
            }
        }
        if (tau < 0) gf_cplx *= -1.;

        // Divide the full matrix to atom-pair blocks
        omp_lock_t gf_lock;
        omp_init_lock(&gf_lock);
#pragma omp parallel for schedule(dynamic)
        for (const auto &IJ : IJs)
        {
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto nI = atbasis_wfc[I];
            const auto nJ = atbasis_wfc[J];
            // 1D representation for row-major 2D array
            auto ptr = std::make_shared<std::valarray<Tdata>>(nI * nJ);
            for (size_t i = 0; i != nI; i++)
            {
                size_t i_glo = atbasis_wfc.get_global_index(I, i);
                for (size_t j = 0; j != nJ; j++)
                {
                    size_t j_glo = atbasis_wfc.get_global_index(J, j);
                    if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                        (*ptr)[i*nJ+j] = gf_cplx(i_glo, j_glo);
                    else
                        (*ptr)[i*nJ+j] = gf_cplx(i_glo, j_glo).real();
                }
            }
            omp_set_lock(&gf_lock);
            gf_libri[I][{J, Ra}] = RI::Tensor<Tdata>({nI, nJ}, ptr);
            omp_unset_lock(&gf_lock);
        }
#pragma omp barrier
        omp_destroy_lock(&gf_lock);
    }

    global::profiler.stop("build_gf_Rt_libri_serial");
}

template <typename Tdata>
static void build_gf_Rt_libri_kpara(
    const MeanField &mf, const std::vector<bool> &band_mask,
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs, 
    double tau,
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> &gf_libri)
{
    using namespace global;
    profiler.start("build_gf_Rt_libri_kpara");
    std::map<Vector3_Order<int>, std::vector<atpair_t>> map_R_IJs;
    const int n_basis = atbasis_wfc.nb_total;
    for (const auto &[IJ, R]: IJRs)
    {
        map_R_IJs[R].emplace_back(IJ);
    }
    std::vector<Vector3_Order<int>> Rs_this;
    for (const auto &[R, _]: map_R_IJs)
        Rs_this.emplace_back(R);
    ofs_myid << "map_R_IJs " << map_R_IJs << std::endl;
    const int n_Rs_this = as_int(map_R_IJs.size());
    int n_Rs_max = n_Rs_this;
    MPI_Allreduce(MPI_IN_PLACE, &n_Rs_max, 1, MPI_INT, MPI_MAX, comm_h.comm);
    // Compute the full G({R}, tau) matrices
    const auto gf_Rs_cplx = get_gf_cplx_imagtimes_Rs_kpara(ispin, ispinor_bra, ispinor_ket, mf,
                                                           kfrac_list, {tau}, Rs_this, comm_h, band_mask)
                                .at(tau);

    for (auto it = gf_Rs_cplx.cbegin(); it != gf_Rs_cplx.cend(); it++)
    {
        const auto &R = it->first;
        const auto &gf_cplx = it->second;
        const std::array<int,3> Ra{R.x,R.y,R.z};
        matrix gf_global;
        if constexpr (!std::is_same<Tdata, std::complex<double>>::value)
            gf_global = gf_cplx.real();
        const auto IJs = map_R_IJs.at(R);
        // global::ofs_myid << "Chi0 Handling IJs: " << IJs << " - R " << Ra << std::endl;
        // Divide the full matrix to atom-pair blocks
        omp_lock_t gf_lock;
        omp_init_lock(&gf_lock);
#pragma omp parallel for schedule(dynamic)
        for (const auto &IJ: IJs)
        {
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto nI = atbasis_wfc[I];
            const auto nJ = atbasis_wfc[J];
            // 1D representation for row-major 2D array
            auto ptr = std::make_shared<std::valarray<Tdata>>(nI * nJ);
            for (size_t i = 0; i != nI; i++)
            {
                const size_t i_glo = atbasis_wfc.get_global_index(I, i);
                for (size_t j = 0; j != nJ; j++)
                {
                    const size_t j_glo = atbasis_wfc.get_global_index(J, j);
                    const size_t index = j_glo + i_glo * n_basis;
                    if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                        (*ptr)[i*nJ+j] = gf_cplx.c[index];
                    else
                        (*ptr)[i*nJ+j] = gf_global.c[index];
                }
            }
            omp_set_lock(&gf_lock);
            gf_libri[I][{J, Ra}] = RI::Tensor<Tdata>({nI, nJ}, ptr);
            omp_unset_lock(&gf_lock);
        }
#pragma omp barrier
        omp_destroy_lock(&gf_lock);
    }
    profiler.stop("build_gf_Rt_libri_kpara");
}

template <typename Tdata>
static void build_gf_Rt_libri_kblacs_para(
    const MeanField &mf,
    const KPointBlacsParallelContext &kblacs_ctxt,
    const ArrayDesc &desc_wfc, const ArrayDesc &desc_gf,
    const IndexScheduler &sched,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const PeriodicBoundaryData &pbc,
    const SymmetryContext &symmetry_context,
    const bool use_symmetry_context,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<Vector3_Order<int>> &Rs,
    double tau,
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> &gf_libri,
    const std::vector<bool> &band_mask)
{
    global::profiler.start("build_gf_Rt_libri_kblacs_para", LIBRPA_VERBOSE_DEBUG);

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    const bool restore_symmetry_kstars =
        use_symmetry_context
        && can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw);
    if (global::should_output(LIBRPA_VERBOSE_DEBUG))
        global::ofs_myid << "Chi0 kBLACS GF symmetry restore: "
                         << (restore_symmetry_kstars ? "on" : "off") << std::endl;
    std::map<double, std::map<Vector3_Order<int>, Matz>> gf_imagtimes_Rs_cplx;
    if (restore_symmetry_kstars && mf.get_n_spinor() == 2)
    {
        gf_imagtimes_Rs_cplx = extract_spinor_gf_block_kblacs(
            get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para_spinor(
                ispin, mf, kfrac_list, {tau}, Rs, kblacs_ctxt, desc_wfc, desc_gf,
                symmetry_context, pbc, atbasis_wfc),
            ispinor_bra, ispinor_ket);
    }
    else if (restore_symmetry_kstars)
    {
        gf_imagtimes_Rs_cplx = get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para(
            ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, {tau}, Rs, kblacs_ctxt,
            desc_wfc, desc_gf, symmetry_context, pbc, atbasis_wfc);
    }
    else
    {
        gf_imagtimes_Rs_cplx = get_gf_cplx_imagtimes_Rs_kblacs_para(
            ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, {tau}, Rs, kblacs_ctxt,
            desc_wfc, desc_gf, band_mask);
    }
    auto &gf_Rs_cplx = gf_imagtimes_Rs_cplx.at(tau);

    for (auto &R_gf_cplx: gf_Rs_cplx)
    {
        const auto &R = R_gf_cplx.first;
        auto &mat_blacs = R_gf_cplx.second;
        auto pair_mat =
            get_ap_map_from_blacs_dist_scheduler(mat_blacs, sched, atbasis_wfc,
                                                 atbasis_wfc, desc_gf);
        for (auto &[pair, mat_ap]: pair_mat)
        {
            const auto &I = as_int(pair.first);
            const auto &J = as_int(pair.second);
            const auto &n_I = atbasis_wfc.get_atom_nb(I);
            const auto &n_J = atbasis_wfc.get_atom_nb(J);
            mat_ap.swap_to_row_major();
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
                gf_libri[I][{J, {R.x, R.y, R.z}}] =
                    RI::Tensor<Tdata>({n_I, n_J}, mat_ap.sptr());
            else
                gf_libri[I][{J, {R.x, R.y, R.z}}] =
                    RI::Tensor<Tdata>({n_I, n_J}, mat_ap.get_real().sptr());
        }
        mat_blacs.clear();
    }

    global::profiler.stop("build_gf_Rt_libri_kblacs_para");
}

// Perform both R-k Fourier transform and time-freq cosine transform of chi0
// Only for LibRI routing
template <typename Tdata>
static void chi_libri_ft_ct(
    const int &isp,
    const int &nspins,
    const int &it,
    const TFGrids &tfg,
    const AtomicBasis &atbasis_abf,
    const Matrix3 &latvec,
    const std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<Tdata>>> &chi0s_IJR,
    const vector<Vector3_Order<double>> &qlist, const vector<atpair_t> &atpairs_ABF,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q)
{
    const bool use_soc = std::is_same<Tdata, std::complex<double>>::value;
    const auto tau = tfg.get_time_nodes()[it];
    // a simple vector container for OpenMP parallel
    vector<pair<std::array<int, 4>, std::vector<std::array<int, 3>>>> ifreq_iq_mu_nu_to_Rs;
    map<int, vector<pair<int, std::array<int, 3>>>> Mu_NuRs;
    const int nfreq = as_int(tfg.get_n_grids());
    const int nqpts = as_int(qlist.size());
    for (int ifreq = 0; ifreq < nfreq; ++ifreq)
    {
        for (int iq = 0; iq < nqpts; iq++)
        {
            for (const auto &[Mu, Nu]: atpairs_ABF)
            {
                std::vector<std::array<int, 3>> Rs;
                if (chi0s_IJR.count(Mu) == 0) continue;
                const int Nu_i = as_int(Nu);
                for (const auto &[JR, chi0s]: chi0s_IJR.at(Mu))
                {
                    if (JR.first == Nu_i)
                    {
                        Rs.push_back(JR.second);
                    }
                }
                ifreq_iq_mu_nu_to_Rs.push_back({{ifreq, iq, as_int(Mu), as_int(Nu)}, Rs});
            }
        }
    }

    global::ofs_myid << "is: " << isp << " tau: " << tau << "  qifreq_atpair_all.size()" << ifreq_iq_mu_nu_to_Rs.size() << std::endl;
    // ofs_myid << "qifreq_atpair_all: " << ifreq_iq_mu_nu_to_Rs << endl;
    global::ofs_myid << "available chi0s_IJR: " << chi0s_IJR.size() << std::endl;
    // ofs_myid << "Keys:" << endl;
    // print_keys(ofs_myid, chi0s_IJR);
    // ofs_myid << endl;
#pragma omp parallel for schedule(dynamic)
    for (const auto &index_Rs : ifreq_iq_mu_nu_to_Rs)
    {
        // ofs_myid << index_Rs.first << endl;
        const auto &ifreq = index_Rs.first[0];
        const auto &iq = index_Rs.first[1];
        const auto &q = qlist[iq];
        const auto &Mu = index_Rs.first[2];
        const auto &Nu = index_Rs.first[3];
        const auto &n_mu = atbasis_abf.get_atom_nb(Mu);
        const auto &n_nu = atbasis_abf.get_atom_nb(Nu);
        const double freq = tfg.get_freq_nodes()[ifreq];
        const double trans = tfg.get_costrans_t2f()(ifreq, it);
        // ofs_myid << "Locating chi" << endl;
        const auto &chi = chi0_q[freq][q][static_cast<atom_t>(Mu)][static_cast<atom_t>(Nu)];
        // ofs_myid << n_mu << " " << n_nu << endl;
        ComplexMatrix cm_chi0(n_mu, n_nu);
        for (const auto &R: index_Rs.second)
        {
            const auto &chi_tensor = chi0s_IJR.at(Mu).at({Nu, R});
            Vector3_Order<int> Rint(R[0], R[1], R[2]);
            // profiler.start("chi0_libri_routing_ft_ct_1");
            // NOTE: ``if constexpr`` needs C++-17
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                LapackConnector::copy(cm_chi0.size, chi_tensor.ptr(), 1,
                                      reinterpret_cast<std::complex<double> *>(cm_chi0.c), 1);
            }
            else
            {
                LapackConnector::copy(cm_chi0.size, chi_tensor.ptr(), 1,
                                      reinterpret_cast<double *>(cm_chi0.c), 2);
            }
            // profiler.stop("chi0_libri_routing_ft_ct_1");

            const double arg = q * (Rint * latvec) * TWO_PI;
            const complex<double> kphase = complex<double>(cos(arg), sin(arg));
            if (use_soc)
                LapackConnector::axpy(cm_chi0.size, (trans * kphase), cm_chi0.c, 1, chi.c, 1);
            else
                LapackConnector::axpy(cm_chi0.size, 2.0 / nspins * (trans * kphase), cm_chi0.c, 1,
                                      chi.c, 1);
        }
    }
}

template <typename Tdata>
static void chi_libri_ct_accumulate_R(
    const int &isp,
    const int &nspins,
    const int &it,
    const TFGrids &tfg,
    const Chi0CollectMap<Tdata> &chi0s_IJR,
    const vector<atpair_t> &atpairs_ABF,
    map<double, Chi0CollectMap<Tdata>> &chi0_freq_R)
{
    const bool use_soc = std::is_same<Tdata, std::complex<double>>::value;
    const auto tau = tfg.get_time_nodes()[it];
    const auto freqs = tfg.get_freq_nodes();
    // Keep frequency keys even when the band mask leaves no transitions.
    for (const auto freq : freqs) chi0_freq_R.try_emplace(freq);

    struct CTTask
    {
        const RI::Tensor<Tdata> *src;
        std::vector<RI::Tensor<Tdata> *> dst_by_freq;
    };

    std::vector<CTTask> tasks;
    for (const auto &[Mu, Nu] : atpairs_ABF)
    {
        const auto it_Mu = chi0s_IJR.find(Mu);
        if (it_Mu == chi0s_IJR.end()) continue;
        const int Nu_i = as_int(Nu);
        for (const auto &[JR, chi_tensor] : it_Mu->second)
        {
            if (JR.first != Nu_i) continue;
            CTTask task;
            task.src = &chi_tensor;
            task.dst_by_freq.reserve(freqs.size());
            for (const auto freq : freqs)
            {
                auto &dst = chi0_freq_R[freq][as_int(Mu)][JR];
                if (dst.empty())
                    dst = RI::Tensor<Tdata>(chi_tensor.shape);
                task.dst_by_freq.push_back(&dst);
            }
            tasks.push_back(std::move(task));
        }
    }

    global::ofs_myid << "is: " << isp << " tau: " << tau
                     << " ct_R_tasks.size() " << tasks.size() << std::endl;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t itask = 0; itask < tasks.size(); ++itask)
    {
        const auto &task = tasks[itask];
        for (std::size_t ifreq = 0; ifreq != freqs.size(); ++ifreq)
        {
            const double trans = tfg.get_costrans_t2f()(as_int(ifreq), it);
            const Tdata scale = use_soc ? Tdata(trans) : Tdata(2.0 / nspins * trans);
            *task.dst_by_freq[ifreq]->data += scale * *task.src->data;
        }
    }
}

template <typename Tdata>
static void chi_libri_ft_Rq_from_freq_R(
    const double freq,
    const AtomicBasis &atbasis_abf,
    const Matrix3 &latvec,
    const Chi0CollectMap<Tdata> &chi0s_IJR,
    const vector<Vector3_Order<double>> &qlist,
    const vector<atpair_t> &atpairs_ABF,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q)
{
    struct FTTerm
    {
        Vector3_Order<int> R;
        const RI::Tensor<Tdata> *tensor;
    };
    struct FTTask
    {
        Vector3_Order<double> q;
        int Mu;
        int Nu;
        std::vector<FTTerm> terms;
        ComplexMatrix *out;
    };

    std::vector<FTTask> tasks;
    for (const auto &q : qlist)
    {
        chi0_q[freq].try_emplace(q);
        for (const auto &[Mu_atom, Nu_atom] : atpairs_ABF)
        {
            const int Mu = as_int(Mu_atom);
            const int Nu = as_int(Nu_atom);
            const auto it_Mu = chi0s_IJR.find(Mu);
            if (it_Mu == chi0s_IJR.end()) continue;

            FTTask task;
            task.q = q;
            task.Mu = Mu;
            task.Nu = Nu;
            for (const auto &[JR, chi_tensor] : it_Mu->second)
            {
                if (JR.first != Nu) continue;
                task.terms.push_back({Vector3_Order<int>{JR.second[0], JR.second[1], JR.second[2]},
                                      &chi_tensor});
            }
            if (task.terms.empty()) continue;

            auto &chi = chi0_q[freq][q][Mu_atom][Nu_atom];
            if (chi.size == 0)
                chi.create(atbasis_abf[Mu_atom], atbasis_abf[Nu_atom]);
            task.out = &chi;
            tasks.push_back(std::move(task));
        }
    }

    global::ofs_myid << "freq: " << freq
                     << " delayed_ft_Rq_tasks.size() " << tasks.size() << std::endl;

#pragma omp parallel for schedule(dynamic)
    for (std::size_t itask = 0; itask < tasks.size(); ++itask)
    {
        const FTTask &task = tasks[itask];
        const auto n_mu = atbasis_abf.get_atom_nb(task.Mu);
        const auto n_nu = atbasis_abf.get_atom_nb(task.Nu);
        ComplexMatrix cm_chi0(n_mu, n_nu);
        for (const auto &term : task.terms)
        {
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                LapackConnector::copy(cm_chi0.size, term.tensor->ptr(), 1,
                                      reinterpret_cast<std::complex<double> *>(cm_chi0.c), 1);
            }
            else
            {
                LapackConnector::copy(cm_chi0.size, term.tensor->ptr(), 1,
                                      reinterpret_cast<double *>(cm_chi0.c), 2);
            }

            const complex<double> kphase = is_gamma_point(task.q)
                ? complex<double>(1.0, 0.0)
                : complex<double>(
                      cos(task.q * (term.R * latvec) * TWO_PI),
                      sin(task.q * (term.R * latvec) * TWO_PI));
            LapackConnector::axpy(cm_chi0.size, kphase, cm_chi0.c, 1, task.out->c, 1);
        }
    }
}

template <typename Tdata>
static void chi_libri_ft_Rq(
    const int &isp, const int &nspins, const int &it, const TFGrids &tfg,
    const AtomicBasis &atbasis_abf,
    const Matrix3 &latvec,
    const std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<Tdata>>> &chi0s_IJR,
    const vector<Vector3_Order<double>> &qlist, const vector<atpair_t> &atpairs_ABF,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q)
{
    const bool use_soc = std::is_same<Tdata, std::complex<double>>::value;
    const auto tau = tfg.get_time_nodes()[it];
    // a simple vector container for OpenMP parallel
    vector<pair<std::array<int, 3>, std::vector<std::array<int, 3>>>> iq_mu_nu_to_Rs;
    map<int, vector<pair<int, std::array<int, 3>>>> Mu_NuRs;

    const int nqpts = as_int(qlist.size());
    for (int iq = 0; iq < nqpts; iq++)
    {
        for (const auto &atpair : atpairs_ABF)
        {
            const auto &Mu = atpair.first;
            const auto &Nu = atpair.second;
            std::vector<std::array<int, 3>> Rs;
            if (chi0s_IJR.count(Mu) == 0) continue;
            const int Nu_i = as_int(Nu);
            for (const auto &chi0s_JR : chi0s_IJR.at(Mu))
            {
                if (chi0s_JR.first.first == Nu_i)
                {
                    Rs.push_back(chi0s_JR.first.second);
                }
            }
            iq_mu_nu_to_Rs.push_back({{iq, static_cast<int>(Mu), static_cast<int>(Nu)}, Rs});
        }
    }

    global::ofs_myid << "is: " << isp << " tau: " << tau << " q_atpair_all.size()" << iq_mu_nu_to_Rs.size()
         << std::endl;
#pragma omp parallel for schedule(dynamic)
    for (const auto &index_Rs : iq_mu_nu_to_Rs)
    {
        const auto &iq = index_Rs.first[0];
        const auto &q = qlist[iq];
        const auto &Mu = index_Rs.first[1];
        const auto &Nu = index_Rs.first[2];
        const auto &n_mu = atbasis_abf.get_atom_nb(Mu);
        const auto &n_nu = atbasis_abf.get_atom_nb(Nu);
        auto &chi = chi0_q[tau][q][Mu][Nu];
        ComplexMatrix cm_chi0(n_mu, n_nu);
        for (const auto &R : index_Rs.second)
        {
            const auto &chi_tensor = chi0s_IJR.at(Mu).at({Nu, R});
            Vector3_Order<int> Rint(R[0], R[1], R[2]);
            // profiler.start("chi0_libri_routing_ft_ct_1");
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                LapackConnector::copy(cm_chi0.size, chi_tensor.ptr(), 1,
                                      reinterpret_cast<std::complex<double> *>(cm_chi0.c), 1);
            }
            else
            {
                LapackConnector::copy(cm_chi0.size, chi_tensor.ptr(), 1,
                                      reinterpret_cast<double *>(cm_chi0.c), 2);
            }
            // profiler.stop("chi0_libri_routing_ft_ct_1");

            const double arg = q * (Rint * latvec) * TWO_PI;
            const complex<double> kphase = complex<double>(cos(arg), sin(arg));
            if (use_soc)
                LapackConnector::axpy(cm_chi0.size, kphase, cm_chi0.c, 1, chi.c, 1);
            else
                LapackConnector::axpy(cm_chi0.size, 2.0 / nspins * kphase, cm_chi0.c, 1, chi.c, 1);
        }
    }
}

static void chi_libri_ft_tw(
    const int &isp, const int &nspins, const int &it, const TFGrids &tfg,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_tau_q,
    const vector<Vector3_Order<double>> &qlist, const vector<atpair_t> &atpairs_ABF,
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &chi0_q)
{
    const auto tau = tfg.get_time_nodes()[it];
    // a simple vector container for OpenMP parallel
    vector<std::array<int, 4>> ifreq_iq_mu_nu;
    map<int, vector<pair<int, std::array<int, 3>>>> Mu_NuRs;
    const int nfreq = as_int(tfg.get_n_grids());
    const int nqpts = as_int(qlist.size());
    for (int ifreq = 0; ifreq < nfreq; ++ifreq)
    {
        for (int iq = 0; iq < nqpts; iq++)
        {
            for (const auto &atpair : atpairs_ABF)
            {
                const auto &Mu = atpair.first;
                const auto &Nu = atpair.second;
                ifreq_iq_mu_nu.push_back({ifreq, iq, static_cast<int>(Mu), static_cast<int>(Nu)});
            }
        }
    }

#pragma omp parallel for schedule(dynamic)
    for (const auto &index : ifreq_iq_mu_nu)
    {
        const auto &ifreq = index[0];
        const auto &iq = index[1];
        const auto &q = qlist[iq];
        const auto &Mu = index[2];
        const auto &Nu = index[3];
        const double freq = tfg.get_freq_nodes()[ifreq];
        const double trans = tfg.get_costrans_t2f()(ifreq, it);
        auto &chi = chi0_q[freq][q][Mu][Nu];
        const auto &cm_chi0 = chi0_tau_q.at(tau).at(q).at(Mu).at(Nu);
        LapackConnector::axpy(cm_chi0.size, trans, cm_chi0.c, 1, chi.c, 1);
    }
}

static void shrink_abfs_chi0(
    map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old> &chi0_q_in,
    map<Vector3_Order<double>, ComplexMatrix> &sinvS, const vector<Vector3_Order<double>> &qlist,
    const AtomicBasis &abf_large, const AtomicBasis &abf_small,
    const BlacsCtxtHandler &blacs_ctxt_h)
{
    // before reset atom_mu: large abfs
    int all_mu = abf_large.nb_total;
    bool debug = false;

    const auto &comm_h = blacs_ctxt_h.comm_h();

    // after reset atom_mu: small abfs
    int all_mu_s = abf_small.nb_total;
    assert (abf_small.n_atoms == abf_large.n_atoms);
    int natom = abf_large.n_atoms;

    const complex<double> CONE{1.0, 0.0};
    ArrayDesc desc_nabf_nabf_ll(blacs_ctxt_h);
    ArrayDesc desc_nabf_nabf_sl(blacs_ctxt_h);
    ArrayDesc desc_nabf_nabf_ss(blacs_ctxt_h);
    desc_nabf_nabf_ll.init_square_blk_capped(all_mu, all_mu, SHRINK_SCALAPACK_BLOCK_CAP, 0, 0);
    desc_nabf_nabf_sl.init_square_blk_capped(all_mu_s, all_mu, SHRINK_SCALAPACK_BLOCK_CAP, 0, 0);
    desc_nabf_nabf_ss.init_square_blk_capped(all_mu_s, all_mu_s, SHRINK_SCALAPACK_BLOCK_CAP, 0, 0);
    const auto set_IJ_nabf_nabf =
        get_necessary_IJ_from_block_2D_sy('U', abf_large, desc_nabf_nabf_ll);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_ll, MAJOR::COL);
    auto chi0ss_block = init_local_mat<complex<double>>(desc_nabf_nabf_ss, MAJOR::COL);
    auto u_block = init_local_mat<complex<double>>(desc_nabf_nabf_sl, MAJOR::COL);
    auto u_chi0 = init_local_mat<complex<double>>(desc_nabf_nabf_sl, MAJOR::COL);
    // for 2D->IJ
    int I, iI;
    map<int, vector<int>> map_lor_v;
    map<int, vector<int>> map_loc_v;
    for (int i_lo = 0; i_lo != desc_nabf_nabf_ss.m_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf_ss.indx_l2g_r(i_lo);
        abf_small.get_local_index(i_glo, I, iI);
        map_lor_v[I].push_back(iI);
    }
    for (int i_lo = 0; i_lo != desc_nabf_nabf_ss.n_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf_ss.indx_l2g_c(i_lo);
        abf_small.get_local_index(i_glo, I, iI);
        map_loc_v[I].push_back(iI);
    }
    // IJ pair of shrinked chi0 to be returned
    std::pair<std::set<int>, std::set<int>> Iset_Jset_c;
    const auto atpair_local = dispatch_upper_triangular_tasks(
        natom, blacs_ctxt_h.myid, blacs_ctxt_h.nprows, blacs_ctxt_h.npcols,
        blacs_ctxt_h.myprow, blacs_ctxt_h.mypcol);
    for (const auto &ap : atpair_local)
    {
        Iset_Jset_c.first.insert(ap.first);
        Iset_Jset_c.second.insert(ap.second);
    }
    for (std::size_t iq = 0; iq < qlist.size(); iq++)
    {
        const auto &q = qlist[iq];
        std::array<double, 3> qa = {q.x, q.y, q.z};
        const auto &U = sinvS.at(q);
        // profiler.start("shrink_prepare_chi0_2d", "Prepare Chi0 2D block for shrink");
        chi0_block.zero_out();
        chi0ss_block.zero_out();
        u_block.zero_out();
        u_chi0.zero_out();
        {
            std::map<int,
                     std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
                chi0_libri;

            if (chi0_q_in.count(q) > 0)
            {
                const auto &chi0_wq = chi0_q_in.at(q);
                for (const auto &M_Nchi : chi0_wq)
                {
                    const auto &M = M_Nchi.first;
                    const auto n_mu = abf_large.get_atom_nb(M);
                    for (const auto &N_chi : M_Nchi.second)
                    {
                        const auto &N = N_chi.first;
                        const auto n_nu = abf_large.get_atom_nb(N);
                        const auto &chi = N_chi.second;
                        std::valarray<complex<double>> chi_va(chi.c, chi.size);
                        auto pchi = std::make_shared<std::valarray<complex<double>>>();
                        *pchi = chi_va;
                        chi0_libri[M][{N, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pchi);
                    }
                }
            }
            // wait for all mpi to calculate chi0_libri
            // then collect chi0_libri to chi0_block
            comm_h.barrier();
            // profiler.start("shrink_prepare_chi0_2d_comm_map2");
            const auto IJq_chi0 = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                comm_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
            // profiler.stop("shrink_prepare_chi0_2d_comm_map2");
            // profiler.start("shrink_prepare_chi0_2d_collect_block");
            collect_block_from_ALL_IJ_Tensor(chi0_block, desc_nabf_nabf_ll,
                                             abf_large, qa, true, CONE, IJq_chi0,
                                             MAJOR::ROW);
            // profiler.stop("shrink_prepare_chi0_2d_collect_block");
        }
        // profiler.stop("shrink_prepare_chi0_2d");
        for (int ir = 0; ir < U.nr; ir++)
        {
            const int ilo = desc_nabf_nabf_sl.indx_g2l_r(ir);
            if (ilo < 0) continue;
            for (int ic = 0; ic < U.nc; ic++)
            {
                const int jlo = desc_nabf_nabf_sl.indx_g2l_c(ic);
                if (jlo < 0) continue;
                u_block(ilo, jlo) = U(ir, ic);
            }
        }
        // Shape of u_block is N_small x N_large
        ScalapackConnector::pgemm_f('N', 'N', all_mu_s, all_mu, all_mu, 1.0, u_block.ptr(), 1, 1,
                                    desc_nabf_nabf_sl.desc, chi0_block.ptr(), 1, 1,
                                    desc_nabf_nabf_ll.desc, 0.0, u_chi0.ptr(), 1, 1,
                                    desc_nabf_nabf_sl.desc);
        ScalapackConnector::pgemm_f('N', 'C', all_mu_s, all_mu_s, all_mu, 1.0, u_chi0.ptr(), 1, 1,
                                    desc_nabf_nabf_sl.desc, u_block.ptr(), 1, 1,
                                    desc_nabf_nabf_sl.desc, 0.0, chi0ss_block.ptr(), 1, 1,
                                    desc_nabf_nabf_ss.desc);

        // shrinked_chi0 = U * large_chi0 * transpose(U, true);

        map<int, map<int, matrix_m<complex<double>>>> chi0s_MNmap;
        map_block_to_IJ_storage_new(chi0s_MNmap, abf_small, map_lor_v, map_loc_v,
                                    chi0ss_block, desc_nabf_nabf_ss, MAJOR::ROW);

        std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
            shrinked_chi0_libri;
        for (const auto &M_Nc : chi0s_MNmap)
        {
            const auto &M = M_Nc.first;
            const auto n_mu = abf_small[M];
            for (const auto &N_c : M_Nc.second)
            {
                const auto &N = N_c.first;
                const auto n_nu = abf_small[N];
                const auto &c = N_c.second;
                shrinked_chi0_libri[M][{N, qa}] =
                    RI::Tensor<complex<double>>({n_mu, n_nu}, c.sptr());
            }
        }
        const auto IJq_chi = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
            comm_h.comm, shrinked_chi0_libri, Iset_Jset_c.first, Iset_Jset_c.second);
        if (debug)
        {
            for (auto &IJqc : IJq_chi)
            {
                auto &I = IJqc.first;
                for (auto &Jqc : IJqc.second)
                {
                    auto &J = Jqc.first.first;
                    // auto &c = Jqc.second;
                    global::ofs_myid << "chi0ss I " << I << " J " << J << std::endl;
                }
            }
        }
        if (chi0_q_in.count(q) > 0)
        {
            for (auto &Ip : chi0_q_in[q])
            {
                auto I = as_int(Ip.first);
                for (auto &Jm : Ip.second)
                {
                    auto J = as_int(Jm.first);
                    // ofs_myid << "IJ=" << I << J << std::endl;
                    ComplexMatrix cm_chi0(abf_small[I], abf_small[J]);
                    const int nr = as_int(abf_small[I]);
                    const int nc = as_int(abf_small[J]);
                    for (int ir = 0; ir < nr; ir++)
                    {
                        for (int ic = 0; ic < nc; ic++)
                        {
                            cm_chi0(ir, ic) = IJq_chi.at(I).at({J, qa})(ir, ic);
                        }
                    }
                    Jm.second = cm_chi0;
                    Jm.second.nr = abf_small[I];
                    Jm.second.nc = abf_small[J];
                    Jm.second.size = abf_small[I] * abf_small[J];
                }
            }
        }
    }
}
#endif

template <typename Tdata>
void Chi0::build_chi0_q_space_time_LibRI_routing(const Cs_LRI &Cs,
                                                 const std::vector<atpair_t> &atpairs_ABF,
                                                 const AtomicBasis &abf_Cs,
                                                 std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
                                                 const BlacsCtxtHandler &blacs_ctxt_h)
{
    using global::profiler;
    using global::lib_printf;
#ifndef LIBRPA_USE_LIBRI
    std::cout << "LibRI routing requested, but the executable is not compiled with LibRI" << std::endl;
    std::cout << "Please recompiler libRPA with -DUSE_LIBRI and configure include path" << std::endl;
    global::mpi_comm_global_h.barrier();
    throw LIBRPA_RUNTIME_ERROR("compilation");
#else
    const bool use_shrink_chi = sinvS.size() > 0;
    const bool use_delayed_ft_shrink = use_shrink_chi && global::dev_opts.use_delayed_ft_shrink;
    global::profiler.start("LibRI_routing", "Loop over LibRI");
    const auto &qlist = this->active_qpoints();
    const std::vector<Vector3_Order<double>> qlist_all(qlist.begin(), qlist.end());
    const auto all_atpairs_ABF = generate_atom_pair_from_nat(atbasis_abf.n_atoms, false);
    const auto q_uhap_process_shape = resolve_chi0_q_uhap_process_shape(
        comm_h.nprocs, qlist.size(), all_atpairs_ABF.size());
    const bool chi0_rspace_symmetry_available =
        can_use_chi0_rspace_symmetry(
            this->symmetry_context, abf_Cs, Rlist_gf, this->use_symmetry_context);
    const bool chi0_band_space_complete =
        this->n_bands_exclude == 0
        && rspace_symmetry_has_complete_band_space(this->mf, this->nbands_G);
    const bool use_chi0_rspace_symmetry =
        chi0_rspace_symmetry_available && chi0_band_space_complete;
    if (chi0_rspace_symmetry_available && !chi0_band_space_complete
        && comm_h.is_root())
    {
        const int n_bands_used =
            std::max(0, (this->nbands_G < 0 ? this->mf.get_n_bands() : this->nbands_G)
                        - this->n_bands_exclude);
        global::lib_printf(
            "chi0 real-space irreducible-sector contraction disabled: "
            "%d response bands do not span the complete %d-state AO space; "
            "k-star and q-star symmetry remain active\n",
            n_bands_used, this->mf.get_n_aos());
    }
    const bool use_q_uhap_split =
        global::dev_opts.use_chi0_q_uhap_split && !use_shrink_chi &&
        !chi0_rspace_symmetry_available &&
        comm_h.nprocs > 1 && q_uhap_process_shape.nprocs_outer > 1;
    TwoLevelParallelContext q_uhap_ctxt;
    std::vector<Vector3_Order<double>> qlist_chi0(qlist.begin(), qlist.end());
    std::vector<Vector3_Order<int>> Rlist_chi0_collect(Rlist_gf.begin(), Rlist_gf.end());
    std::vector<atpair_t> atpairs_chi0(atpairs_ABF.begin(), atpairs_ABF.end());
    if (use_q_uhap_split)
    {
        q_uhap_ctxt.init(q_uhap_process_shape, comm_h.comm, TwoLevelRankLayout::CONTIGUOUS_INNER);
        qlist_chi0 = dispatch_vector(
            qlist_all, q_uhap_ctxt.outer_group_id(), q_uhap_process_shape.nprocs_outer, true);
        Rlist_chi0_collect = dispatch_vector(
            Rlist_gf, q_uhap_ctxt.outer_group_id(), q_uhap_process_shape.nprocs_outer, true);
        atpairs_chi0 = dispatch_vector(
            all_atpairs_ABF, q_uhap_ctxt.inner_rank(), q_uhap_process_shape.nprocs_inner, true);
        global::ofs_myid << "chi0 q/uhap split enabled: "
                         << q_uhap_process_shape.info("qpoint", "uhap")
                         << ", qpoint_group = " << q_uhap_ctxt.outer_group_id()
                         << ", uhap_rank = " << q_uhap_ctxt.inner_rank()
                         << ", local_qpoints = " << qlist_chi0.size()
                         << ", local_Rs = " << Rlist_chi0_collect.size()
                         << ", local_uhap = " << atpairs_chi0.size() << "\n";
        global::ofs_myid << "chi0 q/uhap local atom-pairs:";
        for (const auto &atpair : atpairs_chi0)
            global::ofs_myid << " (" << atpair.first << "," << atpair.second << ")";
        global::ofs_myid << "\n";
        global::ofs_myid << "chi0 q/uhap local q-points:";
        for (const auto &q : qlist_chi0)
            global::ofs_myid << " (" << q.x << "," << q.y << "," << q.z << ")";
        global::ofs_myid << "\n";
        global::ofs_myid << "chi0 q/uhap local R-vectors:";
        for (const auto &R : Rlist_chi0_collect)
            global::ofs_myid << " (" << R.x << "," << R.y << "," << R.z << ")";
        global::ofs_myid << "\n";
    }
    else if (use_chi0_rspace_symmetry && global::dev_opts.use_chi0_q_uhap_split
             && q_uhap_process_shape.nprocs_outer > 1)
    {
        global::ofs_myid << "chi0 q/uhap split disabled for symmetry chi0 path\n";
    }
    else if (use_shrink_chi && q_uhap_process_shape.nprocs_outer > 1)
    {
        global::ofs_myid << "chi0 q/uhap split disabled for shrink chi0 path\n";
    }

    map<int, std::array<double, 3>> atoms_pos;
    const int natoms = as_int(atbasis_wfc.n_atoms);
    for (int i = 0; i != natoms; i++)
    {
        atoms_pos.insert(pair<int, std::array<double, 3>>{i, {0, 0, 0}});
    }

    const auto estimate_chi0_q_mem_gb =
        [this](const std::vector<Vector3_Order<double>> &qpoints,
               const std::vector<atpair_t> &atpairs)
    {
        double mem_gb = 0.0;
        for (auto atpair : atpairs)
        {
            auto Mu = atpair.first;
            auto Nu = atpair.second;
            mem_gb += atbasis_abf[Mu] * atbasis_abf[Nu];
        }
        return mem_gb * tfg.get_n_grids() * qpoints.size() * 1.6e-8;
    };

    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> chi0_q_split;
    auto &chi0_q_work = use_q_uhap_split ? chi0_q_split : chi0_q;
    const auto freq_nodes = tfg.get_freq_nodes();
    const double chi0_q_work_mem_gb = estimate_chi0_q_mem_gb(qlist_chi0, atpairs_chi0);
    global::ofs_myid << "Estimated chi0_q work memory [GB]: "
                     << chi0_q_work_mem_gb << std::endl;
    global::ofs_myid << "chi0 delayed CT/FT enabled: "
                     << (!use_shrink_chi || use_delayed_ft_shrink)
                     << " (nq = " << qlist_all.size()
                     << ", nfreq = " << freq_nodes.size() << ")\n";
    if (use_shrink_chi)
        global::ofs_myid << "chi0 delayed FT shrink enabled: "
                         << use_delayed_ft_shrink << "\n";
    if (use_q_uhap_split)
    {
        global::ofs_myid << "Estimated chi0_q final-layout memory [GB]: "
                         << estimate_chi0_q_mem_gb(
                                qlist_all, atpairs_ABF)
                         << std::endl;
    }

    if (use_shrink_chi && !use_delayed_ft_shrink)
    {
        // Prepare relevant ComplexMatrix objects for the FT work layout.
        create_chi0_q_blocks(chi0_q_work, freq_nodes, qlist_chi0, atpairs_chi0, atbasis_abf);
    }
    const auto &lat_array = this->pbc.latvec_array;

    const auto &period_array = this->pbc.period_array;
    const auto atom_nw = atbasis_wfc.get_atom_nb_map<int>();

    RI::RPA<int, int, 3, Tdata> rpa;
#ifdef LIBRPA_LIBRI_HAS_FINE_GRAINED_LOCK
    rpa.lri.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
#endif
    global::profiler.start("chi0_libri_routing_set_parallel");
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
    rpa.lri.parallel =
        std::make_shared<RI::Parallel_LRI_Equally_Weighted<int, int, 3, Tdata>>(atom_nw);
#endif
    rpa.set_parallel(comm_h.comm, atoms_pos, lat_array, period_array);
    global::profiler.stop("chi0_libri_routing_set_parallel");
    const auto libri_chi0_irreducible_sector =
        use_chi0_rspace_symmetry ? convert_symmetry_irreducible_sector_to_libri_chi0(
                                       this->symmetry_context.irreducible_sector, period_array)
                                 : std::map<std::pair<int, int>, std::set<std::array<int, 3>>>{};
    if (use_chi0_rspace_symmetry)
    {
        if (comm_h.is_root())
            global::lib_printf(
                "Reducing chi0 real-space blocks with symmetry irreducible sectors\n");
        rpa.lri.filter_atom =
            std::make_shared<OutputOnlyFilter_Chi0_Symmetry<int, std::array<int, 3>, Tdata>>(
                rpa.lri.period, libri_chi0_irreducible_sector);
    }
    // rpa.chi0s is still produced on the global LibRI communicator.  The q/uhap
    // split narrows the requested atom pairs first; once Cs/Gs are scoped to the
    // two-level groups, this is the communicator boundary to switch.
    const MPI_Comm chi0_collect_comm = comm_h.comm;

    // local Rlist to collect after chi0s on each process
    std::vector<atpair_t> symmetry_irreducible_atpairs;
    const auto symmetry_exact_s0_s1 =
        use_chi0_rspace_symmetry
            ? make_chi0_symmetry_collect_request(
                  this->symmetry_context.rspace_sector_stars,
                  atpairs_chi0,
                  symmetry_irreducible_atpairs)
            : Chi0ExactCollectRequest{};
    const auto s0_s1 = get_s0_s1_for_comm_map2_first<atom_t, int>(atpairs_chi0);
    const auto exact_s0_s1 =
        use_chi0_rspace_symmetry
            ? symmetry_exact_s0_s1
            : make_chi0_exact_collect_request(atpairs_chi0, Rlist_chi0_collect);

    global::profiler.start("chi0_libri_routing_set_cs", "Set Cs");
    // if (Params::debug)
    //     ofs_myid << Cs_libri;
    // cout << "Setting Cs for rpa object" << endl;
    // librpa_int::utils::release_free_mem();
    // if(mpi_comm_global_h.is_root())
    // {
    //     printf("Begin set Cs !!! \n");
    //     librpa_int::utils::display_free_mem();
    //     // printf("chi0_freq_q size: %d,  freq: %f, q:( %f, %f, %f )\n",chi0_wq.size(),freq, q.x,q.y,q.z );
    // }

    // TODO: template Cs_LRI
    if constexpr (std::is_same<Tdata, std::complex<double>>::value)
    {
        std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<Tdata>>> data_libri;
        for (const auto &I_JR_C : Cs.data_libri)
        {
            const auto I = I_JR_C.first;
            for (const auto &JR_C : I_JR_C.second)
            {
                const auto J = JR_C.first.first;
                const auto R = JR_C.first.second;
                const auto &C = JR_C.second;
                auto JR = std::pair<int, std::array<int, 3>>(J, R);
                data_libri[I][JR] = RI::Global_Func::convert<Tdata>(C);
            }
        }
        rpa.set_Cs(data_libri, libri_threshold_C);
    }
    else
        rpa.set_Cs(Cs.data_libri, libri_threshold_C);

    // Cs_libri.clear();
    // librpa_int::utils::release_free_mem();
    //
    // if(mpi_comm_global_h.is_root())
    // {
    //     printf("After set Cs !!! \n");
    //     librpa_int::utils::display_free_mem();
    //     // printf("chi0_freq_q size: %d,  freq: %f, q:( %f, %f, %f )\n",chi0_wq.size(),freq, q.x,q.y,q.z );
    // }
    // cout << "Cs of rpa object set" << endl;
    global::profiler.stop("chi0_libri_routing_set_cs");

    ArrayDesc desc_gf;
    IndexScheduler sched_gf;
    std::vector<Vector3_Order<int>> Rs_gf;
    if (this->is_mf_eigvec_k_distributed_)
    {
        profiler.start("chi0_libri_routing_prepare_gf_index");
        const int n_basis_ao = mf.get_n_aos();
        desc_gf = kblacs_ctxt.create_array_desc(n_basis_ao, n_basis_ao);
        const auto map_atpairs_balanced =
            get_balanced_ap_distribution_for_consec_descriptor(atbasis_wfc, atbasis_wfc, desc_gf);
        sched_gf.init(map_atpairs_balanced, atbasis_wfc, atbasis_wfc, desc_gf, false);
        const auto iRs = dispatcher_balanced(0, Rlist_gf.size(), kblacs_ctxt.kpoints_local().size(),
                                             true, kblacs_ctxt.comm_kpoint_h.comm);
        Rs_gf.reserve(iRs.size());
        for (const auto &iR: iRs)
            Rs_gf.push_back(Rlist_gf[iR]);
        profiler.stop("chi0_libri_routing_prepare_gf_index");
    }

    const int n_soc = mf.get_n_spinor();
    const std::size_t collect_max_bytes = libri_collect_max_bytes > 0
        ? static_cast<std::size_t>(libri_collect_max_bytes)
        : 0;
    const auto byte_collect_plan = collect_max_bytes > 0
        ? make_chi0_collect_plan_by_bytes<Tdata>(
              atbasis_abf.n_atoms, Rlist_gf, atbasis_abf, collect_max_bytes)
        : Chi0CollectPlan{};
    // LibRI's map collectors gather a sparse nested map according to requested
    // atom or exact (atom, R) keys.  A full one-shot collect can briefly
    // duplicate every local chi0s tensor on every rank, so large cases
    // split the global (I,J,R) key space into chunks capped by an estimated
    // tensor byte count.
    const bool use_byte_collect_chunks =
        comm_h.nprocs > 1 && !use_chi0_rspace_symmetry && byte_collect_plan.nchunks() > 0;
    // Each rank only requests the atoms that can contribute to its local atom-pair work after
    // collection.  This keeps communication bounded by the final ownership instead of the full
    // tensor map.
    const auto local_request_chunks = use_byte_collect_chunks
        ? make_local_chi0_request_chunks<Tdata>(byte_collect_plan, atpairs_chi0, Rlist_gf)
        : std::vector<Chi0CollectRequest>{};
    const std::size_t s0_chunk = libri_collect_s0_chunk > 0
        ? static_cast<std::size_t>(libri_collect_s0_chunk)
        : 0;
    const bool use_s0_collect_chunks =
        comm_h.nprocs > 1 && !use_chi0_rspace_symmetry && !use_byte_collect_chunks &&
        s0_chunk > 0 && s0_chunk < atbasis_abf.n_atoms;
    if (use_byte_collect_chunks)
    {
        global::ofs_myid << "chi0_libri_routing_collect_Rs byte_chunks = "
                         << byte_collect_plan.nchunks()
                         << ", max_bytes = " << byte_collect_plan.max_bytes
                         << ", estimated_total_bytes = " << byte_collect_plan.total_bytes << "\n";
    }
    else if (use_s0_collect_chunks)
    {
        global::ofs_myid << "chi0_libri_routing_collect_Rs s0_chunk = "
                         << s0_chunk << ", global_s0_total = "
                         << atbasis_abf.n_atoms << "\n";
    }

    // omp_lock_t lock_chi0_fourier_cosine;
    // omp_init_lock(&lock_chi0_fourier_cosine);
    // int count_gf = 0;
    map<double, Chi0CollectMap<Tdata>> chi0_freq_R;
    for (size_t it = 0; it != tfg.size(); it++)
    {
        const double tau = tfg.get_time_nodes()[it];
        // cout << tau << " ";
        for (auto isp = 0; isp < this->mf.get_n_spins(); isp++)
        {
            std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> chi0s_IJR;
            std::clock_t cpu_clock_start_isp_tau = clock();
            double wtime_start_isp_tau = omp_get_wtime();
            for (auto is1 = 0; is1 < n_soc; is1++)
            {
                for (auto is2 = 0; is2 < n_soc; is2++)
                {
                    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> gf_po_libri;
                    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> gf_ne_libri;

                    // On-the-fly build of Green's function at specific spin channel and imaginary
                    // time
                    if (comm_h.is_root() && global::should_output())
                    {
                        if (nbands_G >= 0)
                            std::cout << "Green's Function sums over " << nbands_G
                                      << " states." << std::endl;
                        else
                            std::cout << "Green's Function sums over all states." << std::endl;
                    }
                    if (this->is_mf_eigvec_k_distributed_)
                    {
                        global::profiler.start("chi0_gf_pos_entry_wait", LIBRPA_VERBOSE_DEBUG);
                        comm_h.barrier();
                        global::profiler.stop("chi0_gf_pos_entry_wait");
                        build_gf_Rt_libri_kblacs_para(
                            this->mf, kblacs_ctxt, desc_wfc, desc_gf, sched_gf, this->atbasis_wfc,
                            isp, is1, is2, this->pbc, this->symmetry_context,
                            this->use_symmetry_context, this->pbc.kfrac_list, Rs_gf, tau,
                            gf_po_libri, this->band_mask);
                        global::profiler.start("chi0_gf_neg_entry_wait", LIBRPA_VERBOSE_DEBUG);
                        comm_h.barrier();
                        global::profiler.stop("chi0_gf_neg_entry_wait");
                        build_gf_Rt_libri_kblacs_para(
                            this->mf, kblacs_ctxt, desc_wfc, desc_gf, sched_gf, this->atbasis_wfc,
                            isp, is2, is1, this->pbc, this->symmetry_context,
                            this->use_symmetry_context, this->pbc.kfrac_list, Rs_gf, -tau,
                            gf_ne_libri, this->band_mask);
                    }
                    else
                    {
                        build_gf_Rt_libri_serial(this->mf, this->band_mask,
                                                 this->atbasis_wfc, isp, is1, is2,
                                                 this->pbc, this->symmetry_context,
                                                 this->use_symmetry_context,
                                                 this->pbc.kfrac_list, this->IJRs_gf_local, tau,
                                                 gf_po_libri);
                        build_gf_Rt_libri_serial(this->mf, this->band_mask,
                                                 this->atbasis_wfc, isp, is2, is1,
                                                 this->pbc, this->symmetry_context,
                                                 this->use_symmetry_context,
                                                 this->pbc.kfrac_list, this->IJRs_gf_local, -tau,
                                                 gf_ne_libri);
                    }
                    global::profiler.start("chi0_set_Gs");
                    rpa.set_Gs_pos(gf_po_libri, libri_threshold_G);
                    rpa.set_Gs_neg(gf_ne_libri, libri_threshold_G);
                    global::profiler.stop("chi0_set_Gs");
                    global::profiler.start("chi0_cal_entry_wait", LIBRPA_VERBOSE_DEBUG);
                    comm_h.barrier();
                    global::profiler.stop("chi0_cal_entry_wait");
                    global::ofs_myid << "rpa.cal_chi0s begin,    tau = " << tau << "\n";
                    global::profiler.start("chi0_libri_routing_cal_chi0s", "Call cal_chi0s");
                    rpa.cal_chi0s();
                    global::profiler.stop("chi0_libri_routing_cal_chi0s");
                    global::ofs_myid << "rpa.cal_chi0s finished, tau = " << tau << "\n";

                    global::profiler.start("chi0_libri_routing_free_gf");
                    rpa.free_Gs_neg();
                    rpa.free_Gs_pos();
                    global::profiler.stop("chi0_libri_routing_free_gf");

                    global::profiler.start("chi0_collect_entry_wait", LIBRPA_VERBOSE_DEBUG);
                    comm_h.barrier();
                    global::profiler.stop("chi0_collect_entry_wait");
                    // collect chi0 on selected atpairs and, for q/uhap split, selected R vectors
                    global::profiler.start("chi0_libri_routing_collect_Rs", "Collect R blocks");
                    if (comm_h.nprocs > 1)
                    {
                        if (use_chi0_rspace_symmetry)
                        {
                            auto request = exact_s0_s1;
                            const bool padding_request =
                                request.first.empty() || request.second.empty();
                            if (padding_request)
                                request = make_padding_chi0_exact_collect_request(
                                    Rlist_gf, abf_Cs);
                            global::ofs_myid << "chi0_libri_routing_collect_Rs symmetry exact "
                                             << "request_s0 = " << request.first.size()
                                             << ", request_s1 = " << request.second.size()
                                             << ", padding_request = " << padding_request << "\n";
                            const Chi0CollectMap<Tdata> tmp_chi0 =
                                collect_chi0_map2<Tdata>(
                                    chi0_collect_comm, rpa.chi0s, request);
                            if (!padding_request)
                                accumulate_chi0_collect_map(chi0s_IJR, tmp_chi0);
                            rpa.chi0s.clear();
                        }
                        else if (use_q_uhap_split)
                        {
                            auto request = exact_s0_s1;
                            const bool padding_request =
                                request.first.empty() || request.second.empty();
                            if (padding_request)
                                request = make_padding_chi0_exact_collect_request(
                                    Rlist_gf, atbasis_abf);
                            global::ofs_myid << "chi0_libri_routing_collect_Rs q/uhap exact "
                                             << "request_s0 = " << request.first.size()
                                             << ", request_s1 = " << request.second.size()
                                             << ", padding_request = " << padding_request << "\n";
                            const Chi0CollectMap<Tdata> tmp_chi0 =
                                collect_chi0_map2<Tdata>(
                                    chi0_collect_comm, rpa.chi0s, request);
                            if (!padding_request)
                                accumulate_chi0_collect_map(chi0s_IJR, tmp_chi0);
                            rpa.chi0s.clear();
                        }
                        else if (use_byte_collect_chunks)
                        {
                            // Move, not copy, the freshly computed rpa.chi0s blocks into the
                            // byte-budget chunks.  selected_chunks[ichunk] is cleared immediately
                            // after communication, so peak memory is roughly one communication
                            // chunk plus the accumulated locally owned result.
                            auto selected_chunks =
                                split_chi0_map_by_collect_plan<Tdata>(byte_collect_plan, rpa.chi0s);
                            const auto nchunks = byte_collect_plan.nchunks();
                            for (std::size_t ichunk = 0; ichunk != nchunks; ++ichunk)
                            {
                                auto chunk_s0_s1 = local_request_chunks[ichunk];
                                // Some chunks contain no atom pairs needed by this rank.  LibRI's
                                // collective still has to be entered by every rank with non-empty
                                // request sets, so use a tiny valid request as padding and discard
                                // the returned data on those ranks.
                                const bool padding_request =
                                    chunk_s0_s1.first.empty() || chunk_s0_s1.second.empty();
                                if (padding_request)
                                    chunk_s0_s1 = padding_s0_s1_for_plan_chunk(byte_collect_plan, ichunk);
                                // Defensive fallback for pathological empty plans; it preserves the
                                // old all-at-once request semantics rather than letting the collective
                                // see an empty atom set.
                                if (chunk_s0_s1.first.empty() || chunk_s0_s1.second.empty())
                                    chunk_s0_s1 = s0_s1;

                                const bool log_chunk =
                                    ichunk < 3 || ichunk + 1 == nchunks || (ichunk + 1) % 100 == 0;
                                if (log_chunk)
                                {
                                    global::ofs_myid << "chi0_libri_routing_collect_Rs byte_chunk "
                                                     << (ichunk + 1) << "/" << nchunks
                                                     << " selected_blocks = "
                                                     << get_num_keys(selected_chunks[ichunk])
                                                     << ", request_s0 = " << chunk_s0_s1.first.size()
                                                     << ", request_s1 = " << chunk_s0_s1.second.size()
                                                     << ", padding_request = " << padding_request << "\n";
                                }

                                const Chi0CollectMap<Tdata> tmp_chi0 =
                                    collect_chi0_map2_first<Tdata>(
                                        chi0_collect_comm, selected_chunks[ichunk], chunk_s0_s1);
                                selected_chunks[ichunk].clear();
                                // Padding requests only exist to keep all ranks synchronized in the
                                // collective.  Accumulating them would import blocks this rank does
                                // not own, so only real request chunks contribute to chi0s_IJR.
                                if (!padding_request)
                                    accumulate_chi0_collect_map(chi0s_IJR, tmp_chi0);
                            }
                        }
                        else if (use_s0_collect_chunks)
                        {
                            std::vector<int> s0_all(atbasis_abf.n_atoms);
                            for (std::size_t iat = 0; iat != atbasis_abf.n_atoms; ++iat)
                                s0_all[iat] = static_cast<int>(iat);
                            for (std::size_t begin = 0, ichunk = 0;
                                 begin != s0_all.size(); begin += s0_chunk, ++ichunk)
                            {
                                std::set<int> chunk_s0;
                                const auto end = std::min(begin + s0_chunk, s0_all.size());
                                for (auto i = begin; i != end; ++i)
                                    chunk_s0.insert(s0_all[i]);
                                auto selected = take_chi0_collect_s0_chunk<Tdata>(rpa.chi0s, chunk_s0);
                                global::ofs_myid << "chi0_libri_routing_collect_Rs s0_chunk "
                                                 << (ichunk + 1) << " selected_blocks = "
                                 << get_num_keys(selected)
                                 << ", request_s0 = " << chunk_s0.size()
                                 << ", request_s1 = " << s0_s1.second.size() << "\n";
                                auto chunk_s0_s1 = std::make_pair(chunk_s0, s0_s1.second);
                                if (chunk_s0_s1.second.empty())
                                    chunk_s0_s1.second.insert(0);
                                const Chi0CollectMap<Tdata> tmp_chi0 =
                                    collect_chi0_map2_first<Tdata>(
                                        chi0_collect_comm, selected, chunk_s0_s1);
                                selected.clear();
                                accumulate_chi0_collect_map(chi0s_IJR, tmp_chi0);
                            }
                            rpa.chi0s.clear();
                        }
                        else
                        {
                            const Chi0CollectMap<Tdata> tmp_chi0 =
                                collect_chi0_map2_first<Tdata>(chi0_collect_comm, rpa.chi0s, s0_s1);
                            accumulate_chi0_collect_map(chi0s_IJR, tmp_chi0);
                            rpa.chi0s.clear();
                        }
                    }
                    else
                    {
                        // Single MPI task, no need to perform communication.
                        accumulate_chi0_collect_map(chi0s_IJR, rpa.chi0s);
                        rpa.chi0s.clear();
                    }
                    global::profiler.stop("chi0_libri_routing_collect_Rs");
                }
            }

            std::clock_t cpu_clock_done_chi0s = clock();

            if (use_chi0_rspace_symmetry && use_shrink_chi)
            {
                profiler.start("chi0_libri_routing_symmetry_restore_R",
                               "Restore chi0 full real-space sector");
                chi0s_IJR = restore_symmetry_abf_rspace_tensor_map_chi0<Tdata>(
                    chi0s_IJR, this->symmetry_context,
                    this->symmetry_context.rspace_sector_stars,
                    abf_Cs, period_array, atpairs_chi0);
                profiler.stop("chi0_libri_routing_symmetry_restore_R");
            }

            // parse back to chi0
            if (use_shrink_chi && !use_delayed_ft_shrink)
            {
                map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>>
                    chi0_tau_q;
                for (auto q : qlist)
                {
                    for (auto atpair : atpairs_ABF)
                    {
                        auto Mu = atpair.first;
                        auto Nu = atpair.second;
                        chi0_tau_q[tau][q][Mu][Nu].create(abf_Cs[Mu], abf_Cs[Nu]);
                    }
                }
                profiler.start("chi0_libri_routing_ft_Rq");
                chi_libri_ft_Rq<Tdata>(isp, mf.get_n_spins(), it, tfg, abf_Cs, pbc.latvec, 
                                       chi0s_IJR, qlist,
                                       atpairs_ABF, chi0_tau_q);
                profiler.stop("chi0_libri_routing_ft_Rq");
                chi0s_IJR.clear();
                profiler.start("shrink_chi0_abfs", "Do shrink transformation");
                shrink_abfs_chi0(chi0_tau_q[tau], sinvS, qlist, abf_Cs, atbasis_abf, blacs_ctxt_h);
                profiler.stop("shrink_chi0_abfs");
                profiler.start("chi0_libri_routing_ft_tw");
                chi_libri_ft_tw(isp, mf.get_n_spins(), it, tfg, chi0_tau_q, qlist, atpairs_ABF,
                                chi0_q);
                profiler.stop("chi0_libri_routing_ft_tw");
                chi0_tau_q.clear();
            }
            else
            {
                profiler.start("chi0_libri_routing_ct_R", "Cosine transform to R-space");
                const auto &atpairs_ct =
                    use_chi0_rspace_symmetry && !use_shrink_chi
                        ? symmetry_irreducible_atpairs
                        : atpairs_chi0;
                chi_libri_ct_accumulate_R<Tdata>(
                    isp, mf.get_n_spins(), it, tfg, chi0s_IJR, atpairs_ct, chi0_freq_R);
                profiler.stop("chi0_libri_routing_ct_R");
                chi0s_IJR.clear();
            }

            std::clock_t cpu_clock_done_trans = clock();
            double wtime_end_isp_tau = omp_get_wtime();
            if (comm_h.myid == 0)
            {
                lib_printf(
                    "chi0s for time point %f, spin %d. CPU time: %f "
                    "(tensor), %f (trans). "
                    "Wall "
                    "time %f\n",
                    tau, isp,
                    cpu_time_from_clocks_diff(cpu_clock_start_isp_tau, cpu_clock_done_chi0s),
                    cpu_time_from_clocks_diff(cpu_clock_done_chi0s, cpu_clock_done_trans),
                    wtime_end_isp_tau - wtime_start_isp_tau);
            }
            // Release freed memory to OS, to resolve memory fragments in LibRI
            profiler.start("chi0_tau_malloc_trim", LIBRPA_VERBOSE_DEBUG);
            release_free_mem();
            profiler.stop("chi0_tau_malloc_trim");
        } // ispin
    } // itau

    if (!use_shrink_chi || use_delayed_ft_shrink)
    {
        profiler.start("chi0_libri_routing_delayed_ft_Rq", "Delayed Fourier transform");
        for (auto it_freq_R = chi0_freq_R.begin(); it_freq_R != chi0_freq_R.end(); )
        {
            const double freq = it_freq_R->first;
            const std::vector<double> freq_one{freq};
            if (use_chi0_rspace_symmetry && !use_shrink_chi)
            {
                profiler.start("chi0_libri_routing_symmetry_restore_R",
                               "Restore chi0 full real-space sector");
                it_freq_R->second = restore_symmetry_abf_rspace_tensor_map_chi0<Tdata>(
                    it_freq_R->second, this->symmetry_context,
                    this->symmetry_context.rspace_sector_stars,
                    abf_Cs, period_array, atpairs_chi0);
                profiler.stop("chi0_libri_routing_symmetry_restore_R");
            }
            if (use_delayed_ft_shrink)
            {
                map<double, map<Vector3_Order<double>,
                                atom_mapping<ComplexMatrix>::pair_t_old>> chi0_q_large;
                create_chi0_q_blocks(
                    chi0_q_large, freq_one, qlist_chi0, atpairs_chi0, abf_Cs);
                chi_libri_ft_Rq_from_freq_R<Tdata>(
                    freq, abf_Cs, pbc.latvec, it_freq_R->second, qlist_chi0,
                    atpairs_chi0, chi0_q_large);
                profiler.start("shrink_chi0_abfs", "Do shrink transformation");
                shrink_abfs_chi0(
                    chi0_q_large[freq], sinvS, qlist_chi0, abf_Cs, atbasis_abf, blacs_ctxt_h);
                profiler.stop("shrink_chi0_abfs");
                chi0_q[freq] = std::move(chi0_q_large[freq]);
            }
            else if (use_q_uhap_split)
            {
                create_chi0_q_blocks(
                    chi0_q_work, freq_one, qlist_chi0, atpairs_chi0, atbasis_abf);
                for (int q_owner = 0;
                     q_owner != q_uhap_process_shape.nprocs_outer; ++q_owner)
                {
                    const auto qlist_owner = dispatch_vector(
                        qlist_all, q_owner, q_uhap_process_shape.nprocs_outer, true);
                    map<double, map<Vector3_Order<double>,
                                     atom_mapping<ComplexMatrix>::pair_t_old>> chi0_q_partial;
                    create_chi0_q_blocks(
                        chi0_q_partial, freq_one, qlist_owner, atpairs_chi0, atbasis_abf);
                    chi_libri_ft_Rq_from_freq_R<Tdata>(
                        freq, atbasis_abf, pbc.latvec, it_freq_R->second, qlist_owner,
                        atpairs_chi0, chi0_q_partial);
                    reduce_chi0_q_partial_to_q_owner(
                        chi0_q_partial, qlist_owner, atpairs_chi0, atbasis_abf,
                        q_owner, q_uhap_ctxt.comm_outer_h.comm, chi0_q_work);
                }
            }
            else
            {
                chi_libri_ft_Rq_from_freq_R<Tdata>(
                    freq, atbasis_abf, pbc.latvec, it_freq_R->second, qlist_chi0,
                    atpairs_chi0, chi0_q_work);
            }
            it_freq_R = chi0_freq_R.erase(it_freq_R);
            release_free_mem();
        }
        profiler.stop("chi0_libri_routing_delayed_ft_Rq");
    }

    if (use_q_uhap_split)
    {
        profiler.start("chi0_q_uhap_redistribute", "Redistribute chi0_q to atom-pair layout");
        redistribute_chi0_q_to_atom_pair_layout(
            chi0_q_split, qlist_chi0, atpairs_chi0,
            qlist_all,
            atpairs_ABF, atbasis_abf, comm_h.comm, chi0_q);
        profiler.stop("chi0_q_uhap_redistribute");
        chi0_q_split.clear();
    }

    // if (use_shrink_chi)
    // {
    //     sinvS.clear();
    // }
    // omp_destroy_lock(&lock_chi0_fourier_cosine);

    if (comm_h.is_root()) lib_printf("\n");
    comm_h.barrier();
    global::profiler.stop("LibRI_routing");
#endif
}

void Chi0::build_chi0_q_space_time_R_tau_routing(const Cs_LRI &Cs,
                                                 const vector<atpair_t> &atpairs_ABF)
{
    using global::lib_printf;
    using global::profiler;

    assert(!Cs.use_libri);
    const auto &LRI_Cs = Cs.data_IJR;

    global::profiler.start("R_tau_routing", "Loop over R-tau");
    // taus and Rs to compute on MPI task
    // tend to calculate more Rs on one process
    vector<pair<int, int>> itauiRs_local =
        librpa_int::dispatcher(0, tfg.size(), 0, Rlist_gf.size(), comm_h.myid,
                           comm_h.nprocs, true, false);
    map<Vector3_Order<double>,int> qlist2myid;

    const auto &qlist = this->active_qpoints();
    auto loc_qlist = librpa_int::dispatch_vector(qlist , comm_h.myid, comm_h.nprocs, true);
    for(int id=0;id!=comm_h.nprocs;id++)
    {
        auto id_qlist = librpa_int::dispatch_vector(qlist, id, comm_h.nprocs, true);
        for(auto &id_q:id_qlist)
            qlist2myid.insert(std::make_pair(id_q, id));
    }

    const int n_soc = mf.get_n_spinor();

    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> chi0_q_tmp;
    for (auto freq : tfg.get_freq_nodes())
        for (auto q : qlist)
        {
            for (auto atpair : atpairs_ABF)
            {
                auto Mu = atpair.first;
                auto Nu = atpair.second;
                chi0_q_tmp[freq][q][Mu][Nu].create(atbasis_abf[Mu], atbasis_abf[Nu]);
            }
        }

    omp_lock_t chi0_lock;
    omp_init_lock(&chi0_lock);
    // double t_chi0_begin = omp_get_wtime();
    // double t_chi0_tot = 0;
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i != itauiRs_local.size(); i++)
    {
        auto itau = itauiRs_local[i].first;
        auto tau = tfg.get_time_nodes()[itau];
        auto iR = itauiRs_local[i].second;
        auto R = Rlist_gf[iR];
        double t_Rtau_begin = omp_get_wtime();
        for (auto atpair : atpairs_ABF)
        {
            atom_t Mu = atpair.first;
            atom_t Nu = atpair.second;
            for (int is = 0; is != mf.get_n_spins(); is++)
            {
                for (int isoc1 = 0; isoc1 != n_soc; isoc1++)
                {
                    for (int isoc2 = 0; isoc2 != n_soc; isoc2++)
                    {
                        // double chi0_ele_begin = omp_get_wtime();
                        matrix chi0_tau;
                        /* if (itau == 0) // debug first itau */
                        chi0_tau = 2.0 / mf.get_n_spins() *
                                   compute_chi0_s_munu_tau_R(LRI_Cs, is, isoc1, isoc2, Mu,
                                                             Nu, tau, R);
                        // print_matrix("", chi0_tau);
                        /* else continue; // debug first itau */
                        // double chi0_ele_t = omp_get_wtime() - chi0_ele_begin;
                        omp_set_lock(&chi0_lock);
                        for (auto q : qlist)
                        {
                            double arg = q * (R * pbc.latvec) * TWO_PI;
                            const complex<double> kphase = complex<double>(cos(arg), sin(arg));
                            const int nfreq = as_int(tfg.size());
                            for (int ifreq = 0; ifreq != nfreq; ifreq++)
                            {
                                double freq = tfg.get_freq_nodes()[ifreq];
                                double trans = tfg.get_costrans_t2f()(ifreq, itau);
                                const complex<double> weight = trans * kphase;
                                /* cout << weight << endl; */
                                // if(freq==tfg.get_freq_nodes()[10] && tau ==
                                // tfg.get_time_nodes()[10])
                                //     cout <<"freq:  "<<freq<<"   Mu Nu:"<<Mu<<", "<<Nu<<";
                                //     "<<complex<double>(cos(arg), sin(arg)) << " * " << trans <<"
                                //     chi0_tau: "<<chi0_tau(0,0)<<"
                                //     chi0_freq:"<<chi0_q_tmp[freq][q][Mu][Nu](0,0)<<endl;
                                chi0_q_tmp[freq][q][Mu][Nu] += ComplexMatrix(chi0_tau) * weight;
                            }
                        }
                        omp_unset_lock(&chi0_lock);
                    }
                }
            }
        }
        double t_Rtau_end = omp_get_wtime();
        double time_used = t_Rtau_end - t_Rtau_begin;
        // t_chi0_tot += time_used;
        lib_printf("CHI0 p_id: %3d, thread: %3d, R: ( %d,  %d,  %d ), tau: %f , TIME_USED: %f\n",
                   comm_h.myid, omp_get_thread_num(), R.x, R.y, R.z, tau, time_used);
    }

    const int nfreq = as_int(tfg.size());
    omp_destroy_lock(&chi0_lock);
    // Reduce to MPI
#pragma omp barrier
    for (int ifreq = 0; ifreq < nfreq; ifreq++)
    {
        double freq = tfg.get_freq_nodes()[ifreq];
        for (auto q : qlist)
        {
            int id_contain_q = qlist2myid[q];
            for (auto &[Mu, Nu]: atpairs_ABF)
            {
                ComplexMatrix tmp_chi0_recv(atbasis_abf[Mu], atbasis_abf[Nu]);
                tmp_chi0_recv.zero_out();

                comm_h.barrier();
                /* cout << "nr/nc chi0_q_tmp: " << chi0_q_tmp[ifreq][iq][Mu][Nu].nr << ", "<< chi0_q_tmp[ifreq][iq][Mu][Nu].nc << endl; */
                /* cout << "nr/nc chi0_q: " << chi0_q[ifreq][iq][Mu][Nu].nr << ", "<< chi0_q[ifreq][iq][Mu][Nu].nc << endl; */
                librpa_int::reduce_ComplexMatrix(chi0_q_tmp[freq][q][Mu][Nu], tmp_chi0_recv, id_contain_q, comm_h.comm);
                if(id_contain_q == comm_h.myid )
                    chi0_q[freq][q][Mu][Nu]=std::move(tmp_chi0_recv);
                /* if (librpa_int::comm_h_.myid==0 && Mu == 0 && Nu == 0 && ifreq == 0 && q == Vector3_Order<double>{0, 0, 0}) */
                /* if (librpa_int::comm_h_.myid==0 && Mu == 0 && Nu == 0 && ifreq == 0 ) */
                /* if (librpa_int::comm_h_.myid==0 && ifreq == 0 && q == Vector3_Order<double>{0, 0, 0}) */
                /* { */
                /*     cout <<  "freq: " << freq << ", q: " << q << endl; */
                /*     lib_printf("Mu %zu Nu %zu\n", Mu, Nu); */
                /*     print_complex_matrix("chi0_q ap, first freq first q",
                 * chi0_q[freq][q][Mu][Nu]); */
                /* } */
                chi0_q_tmp[freq][q][Mu].erase(Nu);
            }
        }
    }
    global::profiler.stop("R_tau_routing");
}

void Chi0::build_chi0_q_space_time_atom_pair_routing(const Cs_LRI &Cs,
                                                     const vector<atpair_t> &atpairs_ABF)
{
    using global::lib_printf;
    using global::profiler;

    profiler.start("atom_pair_routing", "Loop over atom pairs");
    //auto tot_pair = dispatch_vector(atpairs_ABF, librpa_int::mpi_comm_global_h.myid, para_mpi.get_size(), false);
    lib_printf("Number of atom pairs on Proc %4d: %zu\n", comm_h.myid, atpairs_ABF.size());
    comm_h.barrier();
    omp_lock_t chi0_lock;
    omp_init_lock(&chi0_lock);
    double t_chi0_begin = omp_get_wtime();

    assert(!Cs.use_libri);
    const auto &LRI_Cs = Cs.data_IJR;

    const auto &latvec = this->pbc.latvec;
    const auto &qlist = this->active_qpoints();
    const int nfreq = as_int(tfg.size());

    const auto n_spinor = mf.get_n_spinor();

#pragma omp parallel
    {
#pragma omp for schedule(dynamic)
        for (size_t atom_pair = 0; atom_pair != atpairs_ABF.size(); atom_pair++)
        {
            auto Mu = atpairs_ABF[atom_pair].first;
            auto Nu = atpairs_ABF[atom_pair].second;
            // const size_t mu_num = atom_mu[Mu];
            // const size_t nu_num = atom_mu[Nu];
            double task_begin = omp_get_wtime();
            map<double, map<Vector3_Order<double>, ComplexMatrix>> tmp_chi0_freq_k;
            for (auto freq : tfg.get_freq_nodes())
                for (auto q : qlist)
                {
                    tmp_chi0_freq_k[freq][q].create(atbasis_abf[Mu], atbasis_abf[Nu]);
                }

            for (int is = 0; is != mf.get_n_spins(); is++)
            {
                for (int isoc1 = 0; isoc1 != n_spinor; isoc1++)
                {
                    for (int isoc2 = 0; isoc2 != n_spinor; isoc2++)
                    {
                        for (auto &R : Rlist_gf)
                        {
                            for (int it = 0; it != nfreq; it++)
                            {
                                double tau = tfg.get_time_nodes()[it];
                                ComplexMatrix tmp_chi0_tau(
                                    2.0 / mf.get_n_spins() *
                                    ComplexMatrix(compute_chi0_s_munu_tau_R(
                                        LRI_Cs, is, isoc1, isoc2, Mu, Nu, tau, R)));
                                // print_complex_matrix("", tmp_chi0_tau);
                                for (auto &q : qlist)
                                {
                                    const double arg = (q * (R * latvec)) * TWO_PI;
                                    const complex<double> kphase =
                                        complex<double>(cos(arg), sin(arg));
                                    for (int ifreq = 0; ifreq != nfreq; ifreq++)
                                    {
                                        double freq = tfg.get_freq_nodes()[ifreq];
                                        double trans = tfg.get_costrans_t2f()(ifreq, it);
                                        const complex<double> weight = trans * kphase;
                                        /* const complex<double> cos_weight_kpashe = kphase *
                                         * tfg.get_costrans_t2f()[ifreq, it]; */
                                        //  cout<<"  tmp_chi0  nr nc:
                                        //  "<<tmp_chi0_freq_k[ifreq][ik_vec].nr<<"
                                        //  "<<tmp_chi0_freq_k[ifreq][ik_vec].nc<<"    chi0_tau:
                                        //  "<<tmp_chi0_tau.nr<<"  "<<tmp_chi0_tau.nr<<endl;
                                        tmp_chi0_freq_k[freq][q] += tmp_chi0_tau * weight;
                                    }
                                }
                            }
                            // vector<matrix>
                            // chi0_freq_tmp(tmp_cosine_tran(freq_grid.size(),chi0_tau_tmp));
                        }
                    }
                }
            }
            double task_end = omp_get_wtime();
            double time_used = task_end - task_begin;
            /* time_task_tot += time_used; */
            omp_set_lock(&chi0_lock);
            for (auto &freq : tfg.get_freq_nodes())
            {
                for (auto &q : qlist) chi0_q[freq][q][Mu][Nu] = std::move(tmp_chi0_freq_k[freq][q]);
            }
            double add_end = omp_get_wtime();
            double add_time = add_end - task_end;
            lib_printf("CHI0 p_id: %3d, thread: %3d, I: %zu, J: %zu, move time: %f  TIME_USED: %f\n", comm_h.myid, omp_get_thread_num(), Mu, Nu, add_time, time_used);
            omp_unset_lock(&chi0_lock);
        }
    }
    comm_h.barrier();
    double t_chi0_end= omp_get_wtime();
    global::profiler.stop("atom_pair_routing");
    if(comm_h.is_root())
        lib_printf("| total chi0 time: %f\n",t_chi0_end-t_chi0_begin);
}

matrix Chi0::compute_chi0_s_munu_tau_R(const atpair_R_mat_t &Cs_IJR,
                                       int spin_channel, int isoc1, int isoc2,
                                       atom_t Mu, atom_t Nu, double tau, Vector3_Order<int> R)
{
    /* lib_printf("     begin chi0  thread: %d,  I: %zu, J: %zu\n",omp_get_thread_num(), Mu, Nu); */

    assert(tau > 0);
    // Local RI requires
    const atom_t I_index = Mu;
    const atom_t J_index = Nu;

    const size_t i_num = atbasis_wfc[Mu];
    const size_t mu_num = atbasis_abf[Mu];

    const size_t j_num = atbasis_wfc[Nu];
    const size_t nu_num = atbasis_abf[Nu];

    int flag_G_IJRt = 0;
    int flag_G_IJRNt = 0;

    /* lib_printf("     check if already calculated\n"); */
    /* lib_printf("     size of Green_atom: %zu\n", Green_atom.size()); */
    const auto spin_it = gf_is_R_tau.find(spin_channel);
    if (spin_it == gf_is_R_tau.end()) return matrix(mu_num, nu_num);
    const auto soc1_it = spin_it->second.find(isoc1);
    if (soc1_it == spin_it->second.end()) return matrix(mu_num, nu_num);
    const auto soc2_it = soc1_it->second.find(isoc2);
    if (soc2_it == soc1_it->second.end()) return matrix(mu_num, nu_num);
    const auto &gf_R_tau = soc2_it->second;

    global::profiler.start("cal_chi0_element", "chi(tau,R,I,J)");

    if (gf_R_tau.count(I_index) && gf_R_tau.at(I_index).count(J_index))
        if (gf_R_tau.at(I_index).at(J_index).count(R))
        {
            if (gf_R_tau.at(I_index).at(J_index).at(R).count(tau)) flag_G_IJRt = 1;

            if (gf_R_tau.at(I_index).at(J_index).at(R).count(-tau)) flag_G_IJRNt = 1;
        }

    matrix X_R2(i_num, j_num * nu_num);
    matrix X_conj_R2(i_num, j_num * nu_num);

    const auto R_period = this->pbc.period;

    for (const auto &L_pair : Cs_IJR.at(J_index))
    {
        const auto L_index = L_pair.first;
        const size_t l_num = atbasis_wfc[L_index];
        /* librpa_int::global::lib_printf("     begin is loop\n"); */
        if (gf_R_tau.count(I_index) && gf_R_tau.at(I_index).count(L_index))
        {
            for (const auto &R2_index : L_pair.second)
            {
                const auto R2 = R2_index.first;
                const auto &Cs_mat2 = R2_index.second;

                Vector3_Order<int> R_temp_2(Vector3_Order<int>(R2 + R) % R_period);

                if (gf_R_tau.at(I_index).at(L_index).count(R_temp_2))
                {
                    assert(j_num * l_num == as_size((*Cs_mat2).nr));
                    /* librpa_int::global::lib_printf("          thread: %d, X_R2 IJL:   %zu,%zu,%zu  R:(  %d,%d,%d  )  tau:%f\n",omp_get_thread_num(),I_index,J_index,L_index,R.x,R.y,R.z,time_tau); */
                    matrix Cs2_reshape(reshape_Cs(j_num, l_num, nu_num, Cs_mat2));

                    if (gf_R_tau.at(I_index).at(L_index).at(R_temp_2).count(tau))
                    {
                        // cout<<"C";
                        // global::profiler.start("X");
                        X_R2 += gf_R_tau.at(I_index).at(L_index).at(R_temp_2).at(tau) * Cs2_reshape;
                        // global::profiler.stop("X");
                    }

                    if (gf_R_tau.at(I_index).at(L_index).at(R_temp_2).count(-tau))
                    {
                        // cout<<"D";
                        // global::profiler.start("X");
                        X_conj_R2 += gf_R_tau.at(I_index).at(L_index).at(R_temp_2).at(-tau) * Cs2_reshape;
                        // global::profiler.stop("X");
                    }
                }
            }
        }
    }
    matrix X_R2_rs(reshape_mat(i_num, j_num, nu_num, X_R2));
    matrix X_conj_R2_rs(reshape_mat(i_num, j_num, nu_num, X_conj_R2));

    matrix O_sum(mu_num, nu_num);
    for (const auto &K_pair : Cs_IJR.at(I_index))
    {
        const auto K_index = K_pair.first;
        const size_t k_num = atbasis_wfc[K_index];

        for (const auto &R1_index : K_pair.second)
        {
            const auto R1 = R1_index.first;
            const auto &Cs_mat1 = R1_index.second;
            // cout<<"R1:  begin  "<<R1<<endl;
            // matrix X_R2(i_num,j_num*nu_num);
            // matrix X_conj_R2(i_num,j_num*nu_num);
            matrix O(i_num, k_num * nu_num);
            matrix Z(k_num, i_num * nu_num);

            if (flag_G_IJRt || flag_G_IJRNt)
            {
                matrix N_R2(k_num, j_num * nu_num);
                matrix N_conj_R2(k_num, j_num * nu_num);
                for (const auto &L_pair : Cs_IJR.at(J_index))
                {
                    const auto L_index = L_pair.first;
                    const size_t l_num = atbasis_wfc[L_index];
                    if (gf_R_tau.count(K_index) && gf_R_tau.at(K_index).count(L_index))
                    {
                        for (const auto &R2_index : L_pair.second)
                        {
                            const auto R2 = R2_index.first;
                            const auto &Cs_mat2 = R2_index.second;
                            Vector3_Order<int> R_temp_1(Vector3_Order<int>(R + R2 - R1) % R_period);
                            // Vector3_Order<int> R_temp_2(R2+R);
                            if (gf_R_tau.at(K_index).at(L_index).count(R_temp_1))
                            {
                                assert(j_num * l_num == as_size((*Cs_mat2).nr));
                                // librpa_int::utils::lib_printf("          thread: %d, IJKL:
                                // %d,%d,%d,%d  R:(  %d,%d,%d  )
                                // tau:%f\n",omp_get_thread_num(),I_index,J_index,K_index,L_index,R.x,R.y,R.z,time_tau);
                                matrix Cs2_reshape(reshape_Cs(j_num, l_num, nu_num, Cs_mat2));
                                if (flag_G_IJRNt && gf_R_tau.at(K_index).at(L_index).at(R_temp_1).count(tau))
                                {
                                    // cout<<"A";
                                    // profiler.start("N");
                                    N_R2 += gf_R_tau.at(K_index).at(L_index).at(R_temp_1).at(tau) * Cs2_reshape;
                                    // profiler.stop("N");
                                }
                                if (flag_G_IJRt && gf_R_tau.at(K_index).at(L_index).at(R_temp_1).count(-tau))
                                {
                                    // cout<<"B";
                                    // profiler.start("N");
                                    N_conj_R2 += gf_R_tau.at(K_index).at(L_index).at(R_temp_1).at(-tau) * Cs2_reshape;
                                    // profiler.stop("N");
                                }
                            }
                        }
                    }
                }
                if (flag_G_IJRt)
                {
                    // profiler.start("O");
                    matrix N_conj_R2_rs(reshape_mat(k_num, j_num, nu_num, N_conj_R2));
                    O += gf_R_tau.at(I_index).at(J_index).at(R).at(tau) * N_conj_R2_rs;
                    // profiler.stop("O");
                }
                if (flag_G_IJRNt)
                {
                    // profiler.start("O");
                    matrix N_R2_rs(reshape_mat(k_num, j_num, nu_num, N_R2));
                    O += gf_R_tau.at(I_index).at(J_index).at(R).at(-tau) * N_R2_rs;
                    // profiler.stop("O");
                }
            }
            Vector3_Order<int> R_temp_3(Vector3_Order<int>(R - R1) % R_period);
            if (gf_R_tau.count(K_index) && gf_R_tau.at(K_index).count(J_index))
            {
                if (gf_R_tau.at(K_index).at(J_index).count(R_temp_3))
                {
                    if (gf_R_tau.at(K_index).at(J_index).at(R_temp_3).count(-tau))
                    {
                        // profiler.start("Z");
                        Z += gf_R_tau.at(K_index).at(J_index).at(R_temp_3).at(-tau) * X_R2_rs;
                        // profiler.stop("Z");
                    }

                    if (gf_R_tau.at(K_index).at(J_index).at(R_temp_3).count(tau))
                    {
                        // profiler.start("Z");
                        Z += gf_R_tau.at(K_index).at(J_index).at(R_temp_3).at(tau) * X_conj_R2_rs;
                        // profiler.stop("Z");
                    }
                }
            }

            matrix Z_rs(reshape_mat(k_num, i_num, nu_num, Z));

            O += Z_rs;
            matrix OZ(reshape_mat_21(i_num, k_num, nu_num, O));
            matrix Cs1_tran(transpose(*Cs_mat1));
            // profiler.start("O");
            O_sum += Cs1_tran * OZ;
            // profiler.stop("O");
            // cout<<"   K, R1:   "<<K_index<<"   "<<R1;
            // rt_m_max(O_sum);
        }
    }

    /* if ( librpa_int::mpi_comm_global_h.myid==0 && Mu == 1 && Nu == 1 && tau ==
     * tfg.get_time_nodes()[0]) */
    /* { */
    /*     cout << R << " Mu=" << Mu << " Nu=" << Nu << " tau:" << tau << endl; */
    /*     print_matrix("space-time chi0", O_sum); */
    /* } */
    // if (Params::debug)
    // {
    //     cout << R << " Mu=" << Mu << " Nu=" << Nu << " tau=" << tau << " R=" << R << endl;
    //     print_matrix("space-time chi0", O_sum);
    // }
    global::profiler.stop("cal_chi0_element");
    return O_sum;
}

void Chi0::build_chi0_q_conventional(const Cs_LRI &Cs,
                                     const vector<atpair_t> &atpair_ABF)
{
    // TODO: low implementation priority
    throw std::logic_error("Not implemented");
}

//! Compute the real-space independent reponse function in space-time method on a particular time
/*!
 * @param[in] gf_occ_ab_t: occupied Green's function at time tau in a particular spin alpha-beta
 * channel
 * @param[in] gf_unocc_ab_t: same as above, but for unoccupied Green's function
 * @param[in] LRI_Cs: LRI coefficients
 * @param[in] Rlist: the integer list of unit cell coordinates
 * @param[in] R_period: the periodicity of super cell
 * @param[in] iRs: indices of R to compute
 * @param[in] mu, nu: indices of atoms with ABF
 *
 * @retval mapping from unit cell index to matrix
 *
 * @warning ZMY: Since Green's function are currently stored in a real matrix,
 *          the complex conjugate in the equations are in fact not taken.
 *          But the result seems okay for the reference test cases.
 *          Definitely should be resolved in the future.
 */
static map<size_t, matrix> compute_chi0_munu_tau_LRI_saveN_noreshape(
    const AtomicBasis &basis_wfc, const AtomicBasis &basis_abf, 
    const MpiCommHandler &comm_h, const map<size_t, atom_mapping<matrix>::pair_t_old> &gf_occ_ab_t,
    const map<size_t, atom_mapping<matrix>::pair_t_old> &gf_unocc_ab_t,
    const atpair_R_mat_t &LRI_Cs, const vector<Vector3_Order<int>> &Rlist,
    const Vector3_Order<int> &R_period, const vector<int> iRs, atom_t Mu, atom_t Nu)
{
    map<size_t, matrix> chi0_tau;

    // Store N and N* so one does not need to recompute for each R of chi
    // the extra memory consumption scales as Cs/nR/natoms,
    // which is a small amount compared to Cs for either small (small natoms, large nR) or large
    // (small nR, large natoms) system N(tau) at R and atom K, at(iR, iK)
    map<pair<Vector3_Order<int>, atom_t>, matrix> N;
    // N*(-tau) at R and atom K, at(iR, iK)
    map<pair<Vector3_Order<int>, atom_t>, matrix> Ncnt;  // c -> conjugate, nt -> negative time

    const atom_t iI = Mu;
    const atom_t iJ = Nu;
    const size_t n_i = basis_wfc[iI];
    const size_t n_mu = basis_abf[Mu];
    const size_t n_j = basis_wfc[iJ];
    const size_t n_nu = basis_abf[Nu];

    // pre-compute N and N*
    // only need to calculate N on R-R1, with R in Green's function and R1 from Cs
    // TODO: may reuse the N code to compute N*, by parsing G(t) and G(-t) in same map as done by
    // Shi Rong
    // TODO: decouple N and N*, reduce memory usage
    std::cout << "Mu: " << Mu << ", Nu: " << Nu << std::endl;
    for (auto iR : iRs)
    {
        /* cout << "iR: " << iR << endl; */
        for (auto const &iK_R1_Cs : LRI_Cs.at(Mu))
        {
            auto iK = iK_R1_Cs.first;
            /* cout << "iK: " << iK << endl; */
            /* cout << "iR: " << iR << ", iK: " << iK << endl; */
            const size_t n_k = basis_wfc[iK];
            for ( auto const & R1_Cs: iK_R1_Cs.second )
            {
                auto const &R1 = R1_Cs.first;
                auto const &RmR1 = Vector3_Order<int>(Rlist[iR] - R1);
                /* cout << "iRmR1: " << iRmR1 << endl; */
                /* cout << "Testing N.count(iRmR1, iK): " << N.count({iRmR1, iK}) << endl; */
                /* cout << "Testing gf_occ_ab_t.count(iR): " << gf_occ_ab_t.count(iR) << endl; */
                // N part, do not recompute and should have the GF counterpart
                if (N.count({RmR1, iK}) == 0 && gf_occ_ab_t.count(iR))
                {
                    N[{RmR1, iK}].create(n_nu, n_j * n_k);
                    matrix &N_iRiK = N[{RmR1, iK}];
                    for (auto const &iL_R2_Cs : LRI_Cs.at(Nu))
                    {
                        auto const & iL = iL_R2_Cs.first;
                        const size_t n_l = basis_wfc[iL];
                        for (auto const & R2_Cs: iL_R2_Cs.second)
                        {
                            auto const &R2 = R2_Cs.first;
                            auto mRpR1mR2 = Vector3_Order<int>(-RmR1 - R2) % R_period;
                            int i_mRpR1mR2 = get_R_index(Rlist, mRpR1mR2);
                            if (gf_unocc_ab_t.count(i_mRpR1mR2))
                            {
                                if (gf_unocc_ab_t.at(i_mRpR1mR2).count(iL) == 0 ||
                                    gf_unocc_ab_t.at(i_mRpR1mR2).at(iL).count(iK) == 0)
                                    continue;
                                const matrix &gf_unocc = gf_unocc_ab_t.at(i_mRpR1mR2).at(iL).at(iK);
                                auto const &Cs_nu_jlR2 = R2_Cs.second;
                                assert(as_size(Cs_nu_jlR2->nr) == n_j * n_l && as_size(Cs_nu_jlR2->nc) == n_nu);
                                matrix tran_Cs_nu_jlR2 = transpose(*Cs_nu_jlR2);
                                assert(as_size(gf_unocc.nr) == n_l && as_size(gf_unocc.nc) == n_k);
                                for (size_t i_nu = 0; i_nu != n_nu; i_nu++)
                                    LapackConnector::gemm('N', 'N', n_j, n_k, n_l, 1.0,
                                                          tran_Cs_nu_jlR2.c + i_nu * n_j * n_l, n_l,
                                                          gf_unocc.c, gf_unocc.nc, 1.0,
                                                          N_iRiK.c + i_nu * n_j * n_k, n_k);
                            }
                        }
                    }
                }
                // N* part
                if (!Ncnt.count({RmR1, iK}) && gf_unocc_ab_t.count(iR))
                {
                    Ncnt[{RmR1, iK}].create(n_nu, n_j * n_k);
                    matrix &Ncnt_iRiK = Ncnt.at({RmR1, iK});
                    for (auto const &iL_R2_Cs : LRI_Cs.at(Nu))
                    {
                        auto const & iL = iL_R2_Cs.first;
                        const size_t n_l = basis_wfc[iL];
                        for (auto const & R2_Cs: iL_R2_Cs.second)
                        {
                            auto const &R2 = R2_Cs.first;
                            auto mRpR1mR2 = Vector3_Order<int>(-RmR1 - R2) % R_period;
                            int i_mRpR1mR2 = get_R_index(Rlist, mRpR1mR2);
                            if (gf_occ_ab_t.count(i_mRpR1mR2))
                            {
                                if (gf_occ_ab_t.at(i_mRpR1mR2).count(iL) == 0 ||
                                    gf_occ_ab_t.at(i_mRpR1mR2).at(iL).count(iK) == 0)
                                    continue;
                                const matrix &gf_occ = gf_occ_ab_t.at(i_mRpR1mR2).at(iL).at(iK);
                                auto const &Cs_nu_jlR2 = R2_Cs.second;
                                assert(as_size(Cs_nu_jlR2->nr) == n_j * n_l && as_size(Cs_nu_jlR2->nc) == n_nu);
                                matrix tran_Cs_nu_jlR2 = transpose(*Cs_nu_jlR2);
                                assert(as_size(gf_occ.nr) == n_l && as_size(gf_occ.nc) == n_k);
                                for (size_t i_nu = 0; i_nu != n_nu; i_nu++)
                                    LapackConnector::gemm('N', 'N', n_j, n_k, n_l, 1.0,
                                                          tran_Cs_nu_jlR2.c + i_nu * n_j * n_l, n_l,
                                                          gf_occ.c, n_k, 1.0,
                                                          Ncnt_iRiK.c + i_nu * n_j * n_k, n_k);
                            }
                        }
                    }
                }
            }
        }
    }
    std::cout << "Size N: " << N.size() << std::endl;
    /* cout << "Precalcualted N, size (iR1 x iK): " << N.size() << endl; */

    for (auto iR : iRs)
    {
        chi0_tau[iR].create(n_mu, n_nu);  // TODO: selectively create by G

        for (auto const &iK_R1_Cs : LRI_Cs.at(Mu))
        {
            auto const iK = iK_R1_Cs.first;
            const size_t n_k = basis_wfc[iK];
            for (auto const &R1_Cs: iK_R1_Cs.second)
            {
                const auto &Cs = R1_Cs.second;
                // a temporary matrix to store the sum of M(t) and M*(-t)
                matrix MpMc(n_nu, n_i * n_k);
                auto RmR1 = Vector3_Order<int>(Rlist[iR] - R1_Cs.first);
                // M(t)=G(t)N(t)
                if (gf_occ_ab_t.count(iR) && N.count({RmR1, iK}))
                {
                    if (gf_occ_ab_t.at(iR).count(iI) != 0 &&
                        gf_occ_ab_t.at(iR).at(iI).count(iJ) != 0)
                    {
                        /* cout << "Computing GN" << endl; */
                        const matrix &gf = gf_occ_ab_t.at(iR).at(iI).at(iJ);
                        const matrix &_N = N.at({RmR1, iK});
                        assert(as_size(_N.nr) == n_nu && as_size(_N.nc) == n_j * n_k);
                        for (size_t i_nu = 0; i_nu != n_nu; i_nu++)
                            LapackConnector::gemm('N', 'N', n_i, n_k, n_j, 1.0,
                                                  gf.c, n_j,
                                                  _N.c+i_nu*n_j*n_k, n_k,
                                                  1.0, MpMc.c+i_nu*n_i*n_k, n_k);
                    }
                }
                // M*(-t)=G*(-t)N*(-t)
                if (gf_unocc_ab_t.count(iR) && N.count({RmR1, iK}))
                {
                    if (gf_unocc_ab_t.at(iR).count(iI) != 0 &&
                        gf_unocc_ab_t.at(iR).at(iI).count(iJ) != 0)
                    {
                        /* cout << "Computing G*N*" << endl; */
                        const matrix &gf = gf_unocc_ab_t.at(iR).at(iI).at(iJ);
                        const matrix &_Ncnt = Ncnt.at({RmR1, iK});
                        assert(as_size(_Ncnt.nr) == n_nu && as_size(_Ncnt.nc) == n_j * n_k);
                        for ( size_t i_nu = 0; i_nu != n_nu; i_nu++ )
                            LapackConnector::gemm('N', 'N', n_i, n_k, n_j, 1.0,
                                                  gf.c, n_j,
                                                  _Ncnt.c+i_nu*n_j*n_k, n_k,
                                                  1.0, MpMc.c+i_nu*n_i*n_k, n_k);
                    }
                }
                LapackConnector::gemm('T', 'T', n_mu, n_nu, n_i * n_k, 1.0, Cs->c, n_mu, MpMc.c,
                                      n_i * n_k, 1.0, chi0_tau[iR].c, n_nu);
            }
        }
    }

    // clean up N to free memory, as they have finished their job
    N.clear();
    Ncnt.clear();

    for (auto iR : iRs)
    {
        matrix X(n_nu, n_j * n_i), Ynt(n_nu, n_i * n_j);
        matrix Xcnt(n_nu, n_j * n_i), Yc(n_nu, n_i * n_j);
        // X
        for (auto const &iL_R2_Cs : LRI_Cs.at(Nu))
        {
            auto const iL = iL_R2_Cs.first;
            auto const n_l = basis_wfc[iL];
            for (auto & R2_Cs: iL_R2_Cs.second)
            {
                auto const R2 = R2_Cs.first;
                auto const &Cs = R2_Cs.second;
                auto mR2mR = Vector3_Order<int>(-Rlist[iR] - R2) % R_period;
                auto i_mR2mR = get_R_index(Rlist, mR2mR);
                if (gf_unocc_ab_t.count(i_mR2mR) && gf_unocc_ab_t.at(i_mR2mR).count(iL) &&
                    gf_unocc_ab_t.at(i_mR2mR).at(iL).count(iI))
                {
                    const auto tran_Cs_nu_jl = transpose(*Cs);
                    auto const &gfu = gf_unocc_ab_t.at(i_mR2mR).at(iL).at(iI);
                    for (size_t i_nu = 0; i_nu < n_nu; i_nu++)
                        LapackConnector::gemm('N', 'N', n_j, n_i, n_l, 1.0,
                                              tran_Cs_nu_jl.c + i_nu * n_j * n_l, n_l, gfu.c, n_i,
                                              1.0, X.c + i_nu * n_j * n_i, n_i);
                }
                if (gf_occ_ab_t.count(i_mR2mR) && gf_occ_ab_t.at(i_mR2mR).count(iL) &&
                    gf_occ_ab_t.at(i_mR2mR).at(iL).count(iI))
                {
                    const auto tran_Cs_nu_jl = transpose(*Cs);
                    // gf shall take conjugate here
                    auto const &gf = gf_occ_ab_t.at(i_mR2mR).at(iL).at(iI);
                    for (size_t i_nu = 0; i_nu < n_nu; i_nu++)
                        LapackConnector::gemm('N', 'N', n_j, n_i, n_l, 1.0,
                                              tran_Cs_nu_jl.c + i_nu * n_j * n_l, n_l, gf.c, n_i,
                                              1.0, Xcnt.c + i_nu * n_j * n_i, n_i);
                }
            }
        }
        // Y
        for (auto const &iK_R1_Cs : LRI_Cs.at(Mu))
        {
            auto const iK = iK_R1_Cs.first;
            auto const n_k = basis_wfc[iK];
            for (auto & R1_Cs: iK_R1_Cs.second)
            {
                auto const R1 = R1_Cs.first;
                auto const Cs = R1_Cs.second;
                auto RmR1 = Vector3_Order<int>(Rlist[iR] - R1) % R_period;
                auto i_RmR1 = get_R_index(Rlist, RmR1);
                if (gf_occ_ab_t.count(i_RmR1) && gf_occ_ab_t.at(i_RmR1).count(iK) &&
                    gf_occ_ab_t.at(i_RmR1).at(iK).count(iJ))
                {
                    const matrix tran_Cs_mu_ik = transpose(*Cs);
                    auto const &gf = gf_occ_ab_t.at(i_RmR1).at(iK).at(iJ);
                    for (std::size_t i_mu = 0; i_mu != n_mu; i_mu++)
                        // transpose so that result stored in ji order
                        LapackConnector::gemm('T', 'T', n_j, n_i, n_k, 1.0, gf.c, n_j,
                                              tran_Cs_mu_ik.c + i_mu * n_i * n_k, n_k, 1.0,
                                              Ynt.c + i_mu * n_j * n_i, n_i);
                }
                if (gf_unocc_ab_t.count(i_RmR1) && gf_unocc_ab_t.at(i_RmR1).count(iK) &&
                    gf_unocc_ab_t.at(i_RmR1).at(iK).count(iJ))
                {
                    const matrix tran_Cs_mu_ik = transpose(*Cs);
                    auto const &gfu = gf_unocc_ab_t.at(i_RmR1).at(iK).at(iJ);
                    for ( size_t i_mu = 0; i_mu < n_mu; i_mu++ )
                        // transpose so that result stored in ji order
                        LapackConnector::gemm('T', 'T', n_j, n_i, n_k, 1.0, gfu.c, n_j,
                                              tran_Cs_mu_ik.c + i_mu * n_i * n_k, n_k, 1.0,
                                              Yc.c + i_mu * n_j * n_i, n_i);
                }
            }
        }
        LapackConnector::gemm('N', 'T', n_mu, n_nu, n_i * n_j, 1.0, Ynt.c, n_i * n_j, X.c,
                              n_i * n_j, 1.0, chi0_tau[iR].c, n_nu);
        LapackConnector::gemm('N', 'T', n_mu, n_nu, n_i * n_j, 1.0, Yc.c, n_i * n_j, Xcnt.c,
                              n_i * n_j, 1.0, chi0_tau[iR].c, n_nu);
    }
    /* if (librpa_int::mpi_comm_global_h.myid==0 && Mu == 0 && Nu == 0) */
    if (comm_h.myid==0 && Mu == 0 && Nu == 1)
    {
        /* for (auto iR_chi0_tau: chi0_tau) */
        /* { */
        /* if (iR_chi0_tau.first!=4) continue; */
        /* cout << Rlist[iR_chi0_tau.first] << " Mu=" << Mu << " Nu=" << Nu; */
        /* print_matrix("chi tauR at first tau and R", iR_chi0_tau.second); */
        /* } */
    }
    /* cout << "Done real-space chi0" << endl; */

    return chi0_tau;
}

void Chi0::free_chi0_q(const double freq, const Vector3_Order<double> q)
{
    auto &chi0_for_free = chi0_q.at(freq).at(q);
    chi0_for_free.clear();
    ap_n_map<ComplexMatrix>().swap(chi0_for_free);
}

static bool nearly_same_qpoint(const Vector3_Order<double> &lhs,
                               const Vector3_Order<double> &rhs,
                               const double tol = 1e-5)
{
    const auto same_component = [tol](const double lhs_component,
                                      const double rhs_component) {
        return std::abs((lhs_component - rhs_component) -
                        std::round(lhs_component - rhs_component)) < tol;
    };
    return same_component(lhs.x, rhs.x) && same_component(lhs.y, rhs.y) &&
           same_component(lhs.z, rhs.z);
}

template <typename QMap>
static typename QMap::iterator find_matching_qpoint(QMap &q_map,
                                                    const Vector3_Order<double> &q_target)
{
    const auto exact_iter = q_map.find(q_target);
    if (exact_iter != q_map.end())
    {
        return exact_iter;
    }

    return std::find_if(q_map.begin(), q_map.end(), [&q_target](const auto &entry) {
        return nearly_same_qpoint(entry.first, q_target);
    });
}

template <typename QMap>
static typename QMap::const_iterator find_matching_qpoint(
    const QMap &q_map,
    const Vector3_Order<double> &q_target)
{
    const auto exact_iter = q_map.find(q_target);
    if (exact_iter != q_map.end())
    {
        return exact_iter;
    }

    return std::find_if(q_map.begin(), q_map.end(), [&q_target](const auto &entry) {
        return nearly_same_qpoint(entry.first, q_target);
    });
}

void Chi0::unfold_abfs_Wc(
    map<Vector3_Order<double>, ComplexMatrix> &sinvS,
    map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old> &Wc,
    const vector<Vector3_Order<double>> &qlist,
    const AtomicBasis &abf_unfold,
    const BlacsCtxtHandler &blacs_ctxt_h)
{
    for (auto &[freq, Wc_q] : Wc)
    {
        (void)freq;
        unfold_abfs_Wc_q(sinvS, Wc_q, qlist, abf_unfold, blacs_ctxt_h);
    }
}

void Chi0::unfold_abfs_Wc_q(
    map<Vector3_Order<double>, ComplexMatrix> &sinvS,
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old &Wc_q,
    const vector<Vector3_Order<double>> &qlist,
    const AtomicBasis &abf_unfold,
    const BlacsCtxtHandler &blacs_ctxt_h)
{
    using global::profiler;
    using global::ofs_myid;

    const bool debug = false;
    const int all_mu = abf_unfold.nb_total;
    const int all_mu_s = this->atbasis_abf.nb_total;
    const int natom = abf_unfold.n_atoms;
    const auto comm_h = blacs_ctxt_h.comm_h();

    assert(natom == as_int(atbasis_abf.n_atoms));

    const complex<double> CONE{1.0, 0.0};
    ArrayDesc desc_nabf_nabf_ll(blacs_ctxt_h);
    ArrayDesc desc_nabf_nabf_ss(blacs_ctxt_h);
    ArrayDesc desc_nabf_nabf_sl(blacs_ctxt_h);
    desc_nabf_nabf_ll.init_square_blk_capped(all_mu, all_mu, SHRINK_SCALAPACK_BLOCK_CAP, 0, 0);
    desc_nabf_nabf_ss.init_square_blk_capped(all_mu_s, all_mu_s, SHRINK_SCALAPACK_BLOCK_CAP, 0, 0);
    desc_nabf_nabf_sl.init_square_blk_capped(all_mu_s, all_mu, SHRINK_SCALAPACK_BLOCK_CAP, 0, 0);
    const auto set_IJ_nabf_nabf = get_necessary_IJ_from_block_2D_sy(
        'U', this->atbasis_abf, desc_nabf_nabf_ss);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto Wc_block = init_local_mat<complex<double>>(desc_nabf_nabf_ss, MAJOR::COL);
    auto Wcll_block = init_local_mat<complex<double>>(desc_nabf_nabf_ll, MAJOR::COL);
    auto u_block = init_local_mat<complex<double>>(desc_nabf_nabf_sl, MAJOR::COL);
    auto Wc_u = init_local_mat<complex<double>>(desc_nabf_nabf_sl, MAJOR::COL);

    int I, iI;
    map<int, vector<int>> map_lor_v;
    map<int, vector<int>> map_loc_v;
    for (int i_lo = 0; i_lo != desc_nabf_nabf_ll.m_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf_ll.indx_l2g_r(i_lo);
        abf_unfold.get_local_index(i_glo, I, iI);
        map_lor_v[I].push_back(iI);
    }
    for (int i_lo = 0; i_lo != desc_nabf_nabf_ll.n_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf_ll.indx_l2g_c(i_lo);
        abf_unfold.get_local_index(i_glo, I, iI);
        map_loc_v[I].push_back(iI);
    }

    std::pair<std::set<int>, std::set<int>> Iset_Jset_c;
    const auto atpair_local = dispatch_upper_triangular_tasks(
        natom, blacs_ctxt_h.myid, blacs_ctxt_h.nprows, blacs_ctxt_h.npcols,
        blacs_ctxt_h.myprow, blacs_ctxt_h.mypcol);
    for (const auto &ap : atpair_local)
    {
        Iset_Jset_c.first.insert(ap.first);
        Iset_Jset_c.second.insert(ap.second);
    }

    for (std::size_t iq = 0; iq < qlist.size(); iq++)
    {
        const auto &q = qlist[iq];
        std::array<double, 3> qa = {q.x, q.y, q.z};
        const auto sinv_iter = find_matching_qpoint(sinvS, q);
        if (sinv_iter == sinvS.end())
        {
            throw LIBRPA_RUNTIME_ERROR("Cannot unfold Wc: missing shrink sinvS for q point");
        }
        const auto &U = sinv_iter->second;

        profiler.start("unfold_prepare_Wc_2d", "Prepare Wc 2D block for unfold");
        Wc_block.zero_out();
        Wcll_block.zero_out();
        u_block.zero_out();
        Wc_u.zero_out();
        {
            std::map<int, std::map<std::pair<int, std::array<double, 3>>,
                                   RI::Tensor<complex<double>>>>
                wc_libri;
            atom_mapping<ComplexMatrix>::pair_t_old Wc_IJ;
            for (const auto &IJqc : Wc_q)
            {
                const auto &atom_i = IJqc.first;
                for (const auto &Jqc : IJqc.second)
                {
                    const auto &atom_j = Jqc.first;
                    if (!Wc_IJ[atom_i].count(atom_j))
                    {
                        Wc_IJ[atom_i][atom_j].create(atbasis_abf[atom_i], atbasis_abf[atom_j]);
                    }
                    for (const auto &qc : Jqc.second)
                    {
                        if (nearly_same_qpoint(qc.first, q))
                        {
                            const auto &c = qc.second;
                            for (int ir = 0; ir < c.nr(); ir++)
                            {
                                for (int ic = 0; ic < c.nc(); ic++)
                                {
                                    Wc_IJ[atom_i][atom_j](ir, ic) = c(ir, ic);
                                }
                            }
                        }
                    }
                }
            }

            for (const auto &M_Nchi : Wc_IJ)
            {
                const auto &atom_i = M_Nchi.first;
                const auto n_mu = atbasis_abf.get_atom_nb(atom_i);
                for (const auto &N_chi : M_Nchi.second)
                {
                    const auto &atom_j = N_chi.first;
                    const auto n_nu = atbasis_abf.get_atom_nb(atom_j);
                    const auto &chi = N_chi.second;
                    std::valarray<complex<double>> chi_va(chi.c, chi.size);
                    auto pchi = std::make_shared<std::valarray<complex<double>>>();
                    *pchi = chi_va;
                    wc_libri[atom_i][{atom_j, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pchi);
                }
            }
            comm_h.barrier();
            profiler.start("unfold_prepare_Wc_2d_comm_map2");
            const auto IJq_wc = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                comm_h.comm, wc_libri, s0_s1.first, s0_s1.second);
            profiler.stop("unfold_prepare_Wc_2d_comm_map2");
            profiler.start("unfold_prepare_Wc_2d_collect_block");
            collect_block_from_ALL_IJ_Tensor(Wc_block, desc_nabf_nabf_ss,
                                             atbasis_abf, qa, true, CONE, IJq_wc,
                                             MAJOR::ROW);
            profiler.stop("unfold_prepare_Wc_2d_collect_block");
        }
        profiler.stop("unfold_prepare_Wc_2d");

        for (int ir = 0; ir < U.nr; ir++)
        {
            const int ilo = desc_nabf_nabf_sl.indx_g2l_r(ir);
            if (ilo < 0) continue;
            for (int ic = 0; ic < U.nc; ic++)
            {
                const int jlo = desc_nabf_nabf_sl.indx_g2l_c(ic);
                if (jlo < 0) continue;
                u_block(ilo, jlo) = U(ir, ic);
            }
        }

        ScalapackConnector::pgemm_f('N', 'N', all_mu_s, all_mu, all_mu_s, 1.0,
                                    Wc_block.ptr(), 1, 1, desc_nabf_nabf_ss.desc,
                                    u_block.ptr(), 1, 1, desc_nabf_nabf_sl.desc,
                                    0.0, Wc_u.ptr(), 1, 1, desc_nabf_nabf_sl.desc);
        ScalapackConnector::pgemm_f('C', 'N', all_mu, all_mu, all_mu_s, 1.0,
                                    u_block.ptr(), 1, 1, desc_nabf_nabf_sl.desc,
                                    Wc_u.ptr(), 1, 1, desc_nabf_nabf_sl.desc,
                                    0.0, Wcll_block.ptr(), 1, 1, desc_nabf_nabf_ll.desc);

        map<int, map<int, matrix_m<complex<double>>>> chi0s_MNmap;
        map_block_to_IJ_storage_new(chi0s_MNmap, abf_unfold, map_lor_v, map_loc_v,
                                    Wcll_block, desc_nabf_nabf_ll, MAJOR::ROW);

        std::map<int,
                 std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
            unfold_Wc_libri;
        for (const auto &M_Nc : chi0s_MNmap)
        {
            const auto &atom_i = M_Nc.first;
            const auto n_mu = abf_unfold[atom_i];
            for (const auto &N_c : M_Nc.second)
            {
                const auto &atom_j = N_c.first;
                const auto n_nu = abf_unfold[atom_j];
                const auto &c = N_c.second;
                unfold_Wc_libri[atom_i][{atom_j, qa}] =
                    RI::Tensor<complex<double>>({n_mu, n_nu}, c.sptr());
            }
        }
        const auto IJq_chi = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
            comm_h.comm, unfold_Wc_libri, Iset_Jset_c.first, Iset_Jset_c.second);
        if (debug)
        {
            for (auto &IJqc : IJq_chi)
            {
                auto &atom_i = IJqc.first;
                for (auto &Jqc : IJqc.second)
                {
                    auto &atom_j = Jqc.first.first;
                    ofs_myid << "Wcll I " << atom_i << " J " << atom_j << std::endl;
                }
            }
        }

        for (auto &IJqc : Wc_q)
        {
            const auto atom_i = IJqc.first;
            for (auto &Jqc : IJqc.second)
            {
                const auto atom_j = Jqc.first;
                for (auto &qc : Jqc.second)
                {
                    if (!nearly_same_qpoint(qc.first, q)) continue;
                    const auto I_iter = IJq_chi.find(atom_i);
                    if (I_iter == IJq_chi.end())
                    {
                        throw LIBRPA_RUNTIME_ERROR("Cannot unfold Wc: missing output atom block");
                    }
                    const auto block_iter = I_iter->second.find({atom_j, qa});
                    if (block_iter == I_iter->second.end())
                    {
                        throw LIBRPA_RUNTIME_ERROR("Cannot unfold Wc: missing output atom-pair block");
                    }

                    Matz matz_Wc(abf_unfold[atom_i], abf_unfold[atom_j]);
                    const int nr = as_int(abf_unfold[atom_i]);
                    const int nc = as_int(abf_unfold[atom_j]);
                    for (int ir = 0; ir < nr; ir++)
                    {
                        for (int ic = 0; ic < nc; ic++)
                        {
                            matz_Wc(ir, ic) = block_iter->second(ir, ic);
                        }
                    }
                    qc.second = matz_Wc.copy();
                }
            }
        }
    }
}

// template void Chi0::build_chi0_q_space_time_LibRI_routing<double>(
//     const Cs_LRI &, const Vector3_Order<int> &, const vector<atpair_t> &,
//     const vector<Vector3_Order<double>> &, std::map<Vector3_Order<double>, ComplexMatrix> &);
// template void Chi0::build_chi0_q_space_time_LibRI_routing<std::complex<double>>(
//     const Cs_LRI &, const Vector3_Order<int> &, const vector<atpair_t> &,
//     const vector<Vector3_Order<double>> &, std::map<Vector3_Order<double>, ComplexMatrix> &);
//
// template void chi_libri_ft_ct<double>(
//     const int &, const int &, const int &, const TFGrids &,
//     const AtomicBasis &atbasis_abf, const Matrix3 &latvec,
//     const std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> &,
//     const vector<Vector3_Order<double>> &, const vector<atpair_t> &,
//     map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &);
//
// template void chi_libri_ft_ct<std::complex<double>>(
//     const int &, const int &, const int &, const TFGrids &,
//     const AtomicBasis &atbasis_abf, const Matrix3 &latvec,
//     const std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<std::complex<double>>>> &,
//     const vector<Vector3_Order<double>> &, const vector<atpair_t> &,
//     map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &);
//
// template void chi_libri_ft_Rq<double>(
//     const int &, const int &, const int &, const TFGrids &,
//     const AtomicBasis &atbasis_abf, const Matrix3 &latvec,
//     const std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> &,
//     const vector<Vector3_Order<double>> &, const vector<atpair_t> &,
//     map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &);
//
// template void chi_libri_ft_Rq<std::complex<double>>(
//     const int &, const int &, const int &, const TFGrids &,
//     const AtomicBasis &atbasis_abf, const Matrix3 &latvec,
//     const std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<std::complex<double>>>> &,
//     const vector<Vector3_Order<double>> &, const vector<atpair_t> &,
//     map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> &);

}
