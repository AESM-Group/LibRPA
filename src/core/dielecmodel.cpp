#include "dielecmodel.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <utility>
#include <valarray>

#include "../math/fitting.h"
#include "../math/interpolate.h"
#include "../math/lebedev_laikov.h"
#include "../math/matrix_m.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../math/vec.h"
#include "../mpi/base_blacs.h"
#include "../utils/constants.h"
#include "../utils/error.h"
#include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "../utils/utils_mem.h"
#include "atomic_basis.h"
#include "meanfield_mpi.h"
#include "pbc.h"
#include "ri.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Tensor.h>
#else
#include "../utils/libri_stub.h"
#endif

namespace librpa_int
{

using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;

std::complex<double> compute_pi_det_blacs_2d(Matz &loc_piT, const ArrayDesc &arrdesc_pi, int *ipiv,
                                             int &info);

std::vector<HeadwingFourierTarget> build_headwing_full_bz_fourier_targets(
    const std::vector<Vector3_Order<double>> &kfrac_list)
{
    std::vector<HeadwingFourierTarget> targets;
    targets.reserve(kfrac_list.size());
    for (int ik = 0; ik != static_cast<int>(kfrac_list.size()); ++ik)
        targets.push_back({ik, ik, kfrac_list[ik]});
    return targets;
}

HeadwingSymmetryFourierTargets build_headwing_symmetry_fourier_targets(
    const SymmetryContext &ctx, const PeriodicBoundaryData &pbc,
    const std::vector<Vector3_Order<double>> &kfrac_ibz)
{
    HeadwingSymmetryFourierTargets result;
    result.target_ids_by_ibz_member.resize(kfrac_ibz.size());
    const auto member_targets = build_symmetry_kstar_member_kfrac_targets(ctx, pbc);
    if (!member_targets.empty() && member_targets.size() != kfrac_ibz.size())
        throw LIBRPA_RUNTIME_ERROR(
            "head/wing symmetry Fourier targets have inconsistent IBZ dimensions");

    for (int ik_ibz = 0; ik_ibz != static_cast<int>(kfrac_ibz.size()); ++ik_ibz)
    {
        const auto &star = find_symmetry_kstar_for_ibz_kpoint(ctx, kfrac_ibz[ik_ibz]);
        if (star.members.empty())
            throw LIBRPA_RUNTIME_ERROR("head/wing symmetry Fourier target has an empty k-star");
        if (!member_targets.empty() && member_targets[ik_ibz].size() != star.members.size())
            throw LIBRPA_RUNTIME_ERROR(
                "head/wing symmetry Fourier targets have inconsistent member dimensions");

        auto &target_ids = result.target_ids_by_ibz_member[ik_ibz];
        target_ids.reserve(star.members.size());
        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const int target_id = static_cast<int>(result.targets.size());
            const auto &k_bz = member_targets.empty() ? star.members[imember].k_bz
                                                       : member_targets[ik_ibz][imember];
            result.targets.push_back({target_id, ik_ibz, k_bz});
            target_ids.push_back(target_id);
        }
    }
    return result;
}

HeadwingCsIJKMap fourier_headwing_cs_to_ijk(
    const Cs_LRI &Cs_data, const AtomicBasis &atomic_basis_wfc,
    const AtomicBasis &atomic_basis_abf,
    const std::vector<HeadwingFourierTarget> &targets)
{
    struct CsRBlock
    {
        Vector3_Order<int> R;
        const RI::Tensor<double> *tensor;
    };
    struct CsPairBlocks
    {
        int I;
        int J;
        int n_mu;
        int n_I;
        int n_J;
        std::vector<CsRBlock> R_blocks;
    };
    struct FourierCsTask
    {
        std::size_t pair_index;
        std::size_t target_index;
    };

    std::vector<CsPairBlocks> pair_blocks;
    std::map<std::pair<int, int>, std::size_t> pair_index;
    for (const auto &[I, JR_Cs] : Cs_data.data_libri)
    {
        for (const auto &[JR, Cs] : JR_Cs)
        {
            const int J = JR.first;
            const auto &Ra = JR.second;
            const auto key = std::make_pair(I, J);
            auto [it, inserted] = pair_index.emplace(key, pair_blocks.size());
            if (inserted)
            {
                pair_blocks.push_back(
                    {I, J, static_cast<int>(atomic_basis_abf.get_atom_nb(I)),
                     static_cast<int>(atomic_basis_wfc.get_atom_nb(I)),
                     static_cast<int>(atomic_basis_wfc.get_atom_nb(J)), {}});
            }
            pair_blocks[it->second].R_blocks.push_back(
                {Vector3_Order<int>{Ra[0], Ra[1], Ra[2]}, &Cs});
        }
    }

    std::vector<FourierCsTask> fourier_tasks;
    fourier_tasks.reserve(pair_blocks.size() * targets.size());
    for (std::size_t ipair = 0; ipair != pair_blocks.size(); ++ipair)
        for (std::size_t itarget = 0; itarget != targets.size(); ++itarget)
            fourier_tasks.push_back({ipair, itarget});

    std::vector<std::shared_ptr<std::valarray<complex<double>>>> fourier_results(
        fourier_tasks.size());
    const auto n_tasks = static_cast<std::ptrdiff_t>(fourier_tasks.size());
#pragma omp parallel for schedule(dynamic)
    for (std::ptrdiff_t itask = 0; itask < n_tasks; ++itask)
    {
        const auto &task = fourier_tasks[static_cast<std::size_t>(itask)];
        const auto &blocks = pair_blocks[task.pair_index];
        const auto &target = targets[task.target_index];
        const std::size_t size = static_cast<std::size_t>(blocks.n_mu) *
                                 static_cast<std::size_t>(blocks.n_I) *
                                 static_cast<std::size_t>(blocks.n_J);
        auto data = std::make_shared<std::valarray<complex<double>>>(complex<double>{0.0, 0.0},
                                                                     size);
        for (const auto &R_block : blocks.R_blocks)
        {
            const auto ang = (target.kfrac * R_block.R) * TWO_PI;
            const complex<double> phase{std::cos(ang), std::sin(ang)};
            const auto &Cs = *R_block.tensor;
            for (int imu = 0; imu != blocks.n_mu; ++imu)
            {
                for (int i = 0; i != blocks.n_I; ++i)
                {
                    for (int j = 0; j != blocks.n_J; ++j)
                    {
                        const std::size_t idx =
                            (static_cast<std::size_t>(imu) * blocks.n_I + i) * blocks.n_J + j;
                        (*data)[idx] += phase * Cs(imu, i, j);
                    }
                }
            }
        }
        fourier_results[static_cast<std::size_t>(itask)] = std::move(data);
    }

    HeadwingCsIJKMap Cs_I_Jtarget;
    for (std::size_t itask = 0; itask != fourier_tasks.size(); ++itask)
    {
        const auto &task = fourier_tasks[itask];
        const auto &blocks = pair_blocks[task.pair_index];
        const auto &target = targets[task.target_index];
        Cs_I_Jtarget[static_cast<HeadwingIJKAtomKey>(blocks.I)]
                    [{static_cast<HeadwingIJKAtomKey>(blocks.J), target.target_id}] =
            RI::Tensor<complex<double>>(
                {static_cast<std::size_t>(blocks.n_mu),
                 static_cast<std::size_t>(blocks.n_I),
                 static_cast<std::size_t>(blocks.n_J)},
                fourier_results[itask]);
    }
    return Cs_I_Jtarget;
}

namespace
{

void collect_headwing_Cs_mu_from_ijk(
    Matz &C_nao_nao, const ArrayDesc &desc_nao_nao,
    const AtomicBasis &atomic_basis_wfc, const int Mu, const int mu_local,
    const int target_id, const HeadwingCsIJKMap &Cs_I_Jtarget)
{
    Matz tmp_loc(C_nao_nao.nr(), C_nao_nao.nc(), MAJOR::ROW);
    tmp_loc.zero_out();

#pragma omp parallel for schedule(static)
    for (int ilo = 0; ilo != desc_nao_nao.m_loc(); ++ilo)
    {
        int I_loc = -1, J_loc = -1, i_ab = -1, j_ab = -1;
        const int i_gl = desc_nao_nao.indx_l2g_r(ilo);
        atomic_basis_wfc.get_local_index(i_gl, I_loc, i_ab);
        if (I_loc != Mu) continue;
        const auto it_I = Cs_I_Jtarget.find(static_cast<HeadwingIJKAtomKey>(I_loc));
        if (it_I == Cs_I_Jtarget.end()) continue;
        for (int jlo = 0; jlo != desc_nao_nao.n_loc(); ++jlo)
        {
            const int j_gl = desc_nao_nao.indx_l2g_c(jlo);
            atomic_basis_wfc.get_local_index(j_gl, J_loc, j_ab);
            const auto it_Jtarget =
                it_I->second.find({static_cast<HeadwingIJKAtomKey>(J_loc), target_id});
            if (it_Jtarget == it_I->second.end()) continue;
            tmp_loc(ilo, jlo) = it_Jtarget->second(mu_local, i_ab, j_ab);
        }
    }

    if (C_nao_nao.is_col_major()) tmp_loc.swap_to_col_major();
    C_nao_nao = std::move(tmp_loc);
}

} // namespace

HeadwingCsIJKRequests build_headwing_cs_ijk_requests(
    const AtomicBasis &atomic_basis_wfc,
    const std::vector<HeadwingFourierTarget> &targets,
    const std::vector<int> &owner_iks_local, const ArrayDesc &desc_nao_nao)
{
    const auto necessary_IJ = get_necessary_IJ_from_block_2D(
        atomic_basis_wfc, atomic_basis_wfc, desc_nao_nao);
    const std::set<int> owners_local(owner_iks_local.begin(), owner_iks_local.end());
    HeadwingCsIJKRequests requests;
    auto &[request_I, request_Jtarget] = requests;
    for (const auto &IJ : necessary_IJ)
    {
        request_I.insert(static_cast<HeadwingIJKAtomKey>(IJ.first));
        for (const auto &target : targets)
        {
            if (owners_local.count(target.owner_ik) != 0)
                request_Jtarget.insert(
                    {static_cast<HeadwingIJKAtomKey>(IJ.second), target.target_id});
        }
    }
    return requests;
}

HeadwingCsIJKMap redistribute_headwing_cs_ijk(
    const Cs_LRI &Cs_data, const AtomicBasis &atomic_basis_wfc,
    const AtomicBasis &atomic_basis_abf,
    const std::vector<HeadwingFourierTarget> &targets,
    const std::vector<int> &owner_iks_local, const ArrayDesc &desc_nao_nao,
    const MpiCommHandler &comm_h)
{
    const auto requests = build_headwing_cs_ijk_requests(
        atomic_basis_wfc, targets, owner_iks_local, desc_nao_nao);

    global::profiler.start("headwing_Cs_fourier_world");
    auto Cs_I_Jtarget_tensor = fourier_headwing_cs_to_ijk(
        Cs_data, atomic_basis_wfc, atomic_basis_abf, targets);
    global::profiler.stop("headwing_Cs_fourier_world");

    global::profiler.start("headwing_Cs_ijk_redist");
    auto Cs_I_Jtarget = comm_map2(
        comm_h.comm, Cs_I_Jtarget_tensor, requests.first, requests.second);
    Cs_I_Jtarget_tensor.clear();
    global::profiler.stop("headwing_Cs_ijk_redist");
    return Cs_I_Jtarget;
}

const int DoubleHavriliakNegami::d_npar = 8;

const std::function<double(double, const std::vector<double> &)>
    DoubleHavriliakNegami::func_imfreq = [](double u, const std::vector<double> &pars)
{
    return 1.0 + (pars[0] - 1.0) / std::pow(1.0 + std::pow(u * pars[3], pars[1]), pars[2]) +
           (pars[4] - 1.0) / std::pow(1.0 + pow(u * pars[7], pars[5]), pars[6]);
};

const std::function<void(std::vector<double> &, double, const std::vector<double> &)>
    DoubleHavriliakNegami::grad_imfreq =
        [](std::vector<double> &grads, double u, const std::vector<double> &pars)
{
    using std::pow;
    using std::log;
    grads[0] = 1.0 / pow(1.0 + pow(u * pars[3], pars[1]), pars[2]);
    grads[1] = (pars[0] - 1.0) * (-pars[2]) / pow(1.0 + pow(u * pars[3], pars[1]), pars[2] + 1) *
               log(u * pars[3]) * pow(u * pars[3], pars[1]);
    grads[2] = (1.0 - pars[0]) * log(1.0 + pow(u * pars[3], pars[1])) /
               pow(1.0 + pow(u * pars[3], pars[1]), pars[2]);
    grads[3] = (pars[0] - 1.0) * (-pars[2]) / pow(1.0 + pow(u * pars[3], pars[1]), pars[2] + 1) *
               pars[1] / pars[3] * pow(u * pars[3], pars[1]);
    grads[4] = 1.0 / pow(1.0 + pow(u * pars[7], pars[5]), pars[6]);
    grads[5] = (pars[4] - 1.0) * (-pars[6]) / pow(1.0 + pow(u * pars[7], pars[5]), pars[6] + 1) *
               log(u * pars[7]) * pow(u * pars[7], pars[5]);
    grads[6] = (1.0 - pars[4]) * log(1.0 + pow(u * pars[7], pars[5])) /
               pow(1.0 + pow(u * pars[7], pars[5]), pars[6]);
    grads[7] = (pars[4] - 1.0) * (-pars[6]) / pow(1.0 + pow(u * pars[7], pars[5]), pars[6] + 1) *
               pars[5] / pars[7] * pow(u * pars[7], pars[5]);
};

double headwing_transition_weight(const double occupied_weight, const double unoccupied_weight,
                                  const int n_spin, const bool spin_orbit_coupled)
{
    const double occupation_difference = occupied_weight - unoccupied_weight;
    if (spin_orbit_coupled) return occupation_difference;
    return occupation_difference * static_cast<double>(n_spin) / 2.0;
}

double headwing_spin_prefactor(const int n_spin, const bool spin_orbit_coupled)
{
    if (spin_orbit_coupled) return 1.0;
    return 2.0 / static_cast<double>(n_spin);
}

std::array<std::array<std::complex<double>, 3>, 3> compute_wing_cartesian_gram(
    const ComplexMatrix &wing)
{
    if (wing.nc != 3)
        throw std::invalid_argument("wing Cartesian Gram requires exactly three columns");

    std::array<std::array<std::complex<double>, 3>, 3> gram{};
    for (int lambda = 0; lambda != wing.nr; ++lambda)
    {
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            for (int beta = 0; beta != 3; ++beta)
            {
                gram.at(alpha).at(beta) +=
                    std::conj(wing(lambda, alpha)) * wing(lambda, beta);
            }
        }
    }
    return gram;
}

void accumulate_wing_mu_for_pair(const std::vector<double> &omega,
                                 const std::array<std::complex<double>, 3> &velocity_unocc_occ,
                                 const std::complex<double> &c_mn, const double egap,
                                 const double factor1, const double factor2,
                                 std::complex<double> *wing_mu_for_mu,
                                 std::complex<double> *wing_mu_iomega0_for_mu)
{
    if (factor1 <= 1.e-8 && factor2 <= 1.e-8)
    {
        return;
    }

    for (std::size_t iomega = 0; iomega != omega.size(); ++iomega)
    {
        const double omega_ev = omega[iomega];
        const double denom = omega_ev * omega_ev + egap * egap;
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            auto &wing = wing_mu_for_mu[iomega * 3 + as_size(alpha)];
            if (factor1 > 1.e-8)
            {
                wing += factor1 * std::conj(c_mn * velocity_unocc_occ[alpha]) / denom;
            }
            if (factor2 > 1.e-8)
            {
                wing += factor2 * c_mn * velocity_unocc_occ[alpha] / denom;
            }
            if (iomega == 0 && wing_mu_iomega0_for_mu != nullptr)
            {
                auto &wing_iomega0 = wing_mu_iomega0_for_mu[alpha];
                if (factor1 > 1.e-8)
                {
                    wing_iomega0 += factor1 * std::conj(c_mn * velocity_unocc_occ[alpha]) / denom;
                }
                if (factor2 > 1.e-8)
                {
                    wing_iomega0 += factor2 * c_mn * velocity_unocc_occ[alpha] / denom;
                }
            }
        }
    }
}

static void print_wing_mu_k_contribution_gram(
    const char *route, const int ik, const Vector3_Order<double> &kfrac,
    const std::vector<std::complex<double>> &wing_mu_iomega0, const int n_abf,
    const SymmetryKStarMember *member = nullptr)
{
    if (static_cast<int>(wing_mu_iomega0.size()) != n_abf * 3)
        throw std::invalid_argument("Wing_mu k contribution has an invalid size");
    ComplexMatrix wing_mu_k(n_abf, 3);
    for (int mu = 0; mu != n_abf; ++mu)
        for (int alpha = 0; alpha != 3; ++alpha)
            wing_mu_k(mu, alpha) = wing_mu_iomega0[as_size(mu * 3 + alpha)];
    const auto gram = compute_wing_cartesian_gram(wing_mu_k);
    if (member == nullptr)
    {
        global::lib_printf(
            "Wing_mu k Gram (iomega=0): route=%s ik=%d k=(% .12f,% .12f,% .12f)\n",
            route, ik, kfrac.x, kfrac.y, kfrac.z);
    }
    else
    {
        global::lib_printf(
            "Wing_mu k Gram (iomega=0): route=%s ik=%d k=(% .12f,% .12f,% .12f) "
            "spatial_isym=%d time_reversal=%d\n",
            route, ik, kfrac.x, kfrac.y, kfrac.z, member->spatial_isym,
            member->time_reversal ? 1 : 0);
    }
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        global::lib_printf("(%15.8e,%15.8e) (%15.8e,%15.8e) (%15.8e,%15.8e)\n",
                           gram.at(alpha).at(0).real(), gram.at(alpha).at(0).imag(),
                           gram.at(alpha).at(1).real(), gram.at(alpha).at(1).imag(),
                           gram.at(alpha).at(2).real(), gram.at(alpha).at(2).imag());
    }
}

static void print_head_k_contribution(const char *route, const int ik,
                                      const Vector3_Order<double> &kfrac,
                                      const std::array<std::complex<double>, 9> &head_k)
{
    global::lib_printf("Head k matrix (iomega=0): route=%s ik=%d k=(% .12f,% .12f,% .12f)\n",
                       route, ik, kfrac.x, kfrac.y, kfrac.z);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        global::lib_printf("(%15.8e,%15.8e) (%15.8e,%15.8e) (%15.8e,%15.8e)\n",
                           head_k.at(as_size(alpha * 3)).real(),
                           head_k.at(as_size(alpha * 3)).imag(),
                           head_k.at(as_size(alpha * 3 + 1)).real(),
                           head_k.at(as_size(alpha * 3 + 1)).imag(),
                           head_k.at(as_size(alpha * 3 + 2)).real(),
                           head_k.at(as_size(alpha * 3 + 2)).imag());
    }
}

static bool is_wing_wfc_probe_kpoint(const Vector3_Order<double> &kfrac)
{
    return std::abs(kfrac.x - 0.625) < 1.0e-10 && std::abs(kfrac.y) < 1.0e-10 &&
           std::abs(kfrac.z) < 1.0e-10;
}

static void print_wing_wfc_probe(const int ik_ibz, const Vector3_Order<double> &k_ibz,
                                 const SymmetryKStarMember &member,
                                 const Vector3_Order<double> &k_bz,
                                 const ComplexMatrix &wfc_bz)
{
    global::lib_printf(
        "Wing WFC probe: ik_ibz=%d k_ibz=(% .12f,% .12f,% .12f) "
        "member_k=(% .12f,% .12f,% .12f) k_bz=(% .12f,% .12f,% .12f) "
        "spatial_isym=%d time_reversal=%d rows=band columns=AO\n",
        ik_ibz, k_ibz.x, k_ibz.y, k_ibz.z, member.k_bz.x, member.k_bz.y, member.k_bz.z,
        k_bz.x, k_bz.y, k_bz.z, member.spatial_isym, member.time_reversal ? 1 : 0);
    for (int iband = 0; iband != wfc_bz.nr; ++iband)
    {
        global::lib_printf("band=%d", iband);
        for (int iao = 0; iao != wfc_bz.nc; ++iao)
            global::lib_printf(" (% .15e,% .15e)", wfc_bz(iband, iao).real(),
                               wfc_bz(iband, iao).imag());
        global::lib_printf("\n");
    }
}

static void print_wing_cmnk_probe(const char *route, const int mu,
                                  const Vector3_Order<double> &kfrac,
                                  const ArrayDesc &desc_nband_nband,
                                  const matrix_m<std::complex<double>> &C_mnk)
{
    if (mu != 0 || !is_wing_wfc_probe_kpoint(kfrac) || !desc_nband_nband.is_src()) return;
    global::lib_printf(
        "Wing C_mnk probe: route=%s mu=%d k=(% .12f,% .12f,% .12f) rows=m columns=n\n",
        route, mu, kfrac.x, kfrac.y, kfrac.z);
    for (int m = 0; m != C_mnk.nr(); ++m)
    {
        global::lib_printf("m=%d", m);
        for (int n = 0; n != C_mnk.nc(); ++n)
            global::lib_printf(" (% .15e,% .15e)", C_mnk(m, n).real(), C_mnk(m, n).imag());
        global::lib_printf("\n");
    }
}

static void print_wing_cnao_probe(const char *route, const int mu,
                                  const Vector3_Order<double> &kfrac,
                                  const ArrayDesc &desc_nao_nao,
                                  const matrix_m<std::complex<double>> &C_nao_nao)
{
    if (mu != 0 || !is_wing_wfc_probe_kpoint(kfrac) || !desc_nao_nao.is_src()) return;
    global::lib_printf(
        "Wing C_nao probe: route=%s mu=%d k=(% .12f,% .12f,% .12f) rows=AO columns=AO\n",
        route, mu, kfrac.x, kfrac.y, kfrac.z);
    for (int row = 0; row != C_nao_nao.nr(); ++row)
    {
        global::lib_printf("ao=%d", row);
        for (int col = 0; col != C_nao_nao.nc(); ++col)
            global::lib_printf(" (% .15e,% .15e)", C_nao_nao(row, col).real(),
                               C_nao_nao(row, col).imag());
        global::lib_printf("\n");
    }
}

std::vector<int> headwing_local_kpoints(const int n_kpoints,
                                        const KPointBlacsParallelContext *kblacs_ctxt)
{
    if (kblacs_ctxt && kblacs_ctxt->is_initialized() && kblacs_ctxt->n_kpoints() == n_kpoints)
    {
        return kblacs_ctxt->kpoints_local();
    }

    std::vector<int> kpoints;
    kpoints.reserve(n_kpoints);
    for (int ik = 0; ik != n_kpoints; ++ik) kpoints.push_back(ik);
    return kpoints;
}

std::vector<int> headwing_local_kpoint_roots(const int n_kpoints,
                                             const KPointBlacsParallelContext *kblacs_ctxt)
{
    if (kblacs_ctxt && kblacs_ctxt->is_initialized() && kblacs_ctxt->n_kpoints() == n_kpoints)
    {
        if (kblacs_ctxt->comm_blacs_h.myid != 0) return {};
        return kblacs_ctxt->kpoints_local();
    }

    std::vector<int> kpoints;
    kpoints.reserve(n_kpoints);
    for (int ik = 0; ik != n_kpoints; ++ik) kpoints.push_back(ik);
    return kpoints;
}

static bool use_matching_kpoint_blacs(const int n_kpoints,
                                      const KPointBlacsParallelContext *kblacs_ctxt)
{
    return kblacs_ctxt && kblacs_ctxt->is_initialized() && kblacs_ctxt->n_kpoints() == n_kpoints;
}

ComplexMatrix rotate_headwing_wfc_to_kstar_member(
    const SymmetryContext &ctx,
    const SymmetryKStarMember &member,
    const std::vector<SpeciesBasisLayout> &wfc_layouts,
    const std::map<atom_t, size_t> &atom_nw,
    const Vector3_Order<double> &k_ibz,
    const ComplexMatrix &wfc_ibz,
    const Vector3_Order<double> *k_bz_target)
{
    const auto rotation = build_symmetry_kspace_rotation_matrix(
        ctx, wfc_layouts, member, atom_nw, k_ibz, member.time_reversal, k_bz_target);
    if (wfc_ibz.nc != rotation.nr || rotation.nr != rotation.nc)
    {
        throw std::runtime_error("headwing WFC rotation dimension mismatch");
    }

    // `rotation` is the coefficient-side matrix for the generated member.
    // The non-TR route is C_bz = C_ibz M. This was checked directly against
    // full-BZ PyATB WFCs for nondegenerate Si bands.
    if (member.time_reversal)
    {
        return conj(wfc_ibz) * rotation;
    }
    return wfc_ibz * rotation;
}

static bool can_use_headwing_single_member_stars(
    const SymmetryContext &ctx, const std::vector<SpeciesBasisLayout> &wfc_layouts,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::map<atom_t, size_t> &atom_nw)
{
    if (!ctx.available || wfc_layouts.empty() || ctx.kstars.size() != kfrac_list.size() ||
        ctx.count_kstar_members() != kfrac_list.size() ||
        !symmetry_species_layouts_match_atom_counts(wfc_layouts, ctx.atom_to_type, atom_nw))
        return false;

    for (const auto &kfrac : kfrac_list)
    {
        const auto &star = find_symmetry_kstar_for_ibz_kpoint(ctx, kfrac);
        if (star.members.size() != 1) return false;
        std::set<atom_t> atoms_covered;
        for (const auto &rotation : star.members.front().atom_rotations)
            atoms_covered.insert(rotation.atom_from);
        for (const auto &[atom, nw] : atom_nw)
        {
            (void)nw;
            if (atoms_covered.count(atom) == 0) return false;
        }
    }
    return true;
}

static void allreduce_head_matrices(std::vector<matrix_m<std::complex<double>>> &head,
                                    const MPI_Comm comm)
{
    for (auto &mat : head)
    {
        if (mat.size() == 0) continue;
        if (mat.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw LIBRPA_RUNTIME_ERROR("head matrix is too large for MPI_Allreduce");
        MPI_Allreduce(MPI_IN_PLACE, mat.ptr(), static_cast<int>(mat.size()), MPI_CXX_DOUBLE_COMPLEX,
                      MPI_SUM, comm);
    }
}

static void allreduce_head_check(
    std::vector<std::array<std::array<std::complex<double>, 3>, 3>> &head_check,
    const MPI_Comm comm)
{
    if (head_check.empty()) return;
    const auto n_elem = head_check.size() * 9;
    if (n_elem > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("head self-check buffer is too large for MPI_Allreduce");
    MPI_Allreduce(MPI_IN_PLACE, head_check.data(), static_cast<int>(n_elem), MPI_CXX_DOUBLE_COMPLEX,
                  MPI_SUM, comm);
}

std::array<ComplexMatrix, 3> rotate_headwing_velocity_to_kstar_member(
    const SymmetryContext &ctx, const SymmetryKStarMember &member,
    const std::array<ComplexMatrix, 3> &v_band_ibz, const int n_bands,
    const bool use_time_reversal)
{
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        if (v_band_ibz[alpha].nr != n_bands || v_band_ibz[alpha].nc != n_bands)
            throw std::runtime_error(
                "rotate_headwing_velocity: band velocity shape mismatch with eigenvectors");
    }

    std::array<ComplexMatrix, 3> v_band_rot;
    for (int alpha = 0; alpha < 3; ++alpha)
    {
        v_band_rot[alpha] = v_band_ibz[alpha];
        if (use_time_reversal)
        {
            for (int i = 0; i < n_bands; ++i)
                for (int j = 0; j < n_bands; ++j)
                    v_band_rot[alpha](i, j) = std::conj(v_band_ibz[alpha](i, j));
        }
    }

    const int spatial_isym = member.spatial_isym;
    if (spatial_isym < 0 || spatial_isym >= static_cast<int>(ctx.rspace_operations.size()))
        throw std::runtime_error("rotate_headwing_velocity: invalid symmetry index");
    const auto &operation = ctx.rspace_operations.at(spatial_isym);
    const Matrix3 cartesian_rotation =
        (preserves_lattice_metric(operation.rotation, ctx.lattice_vectors, 1e-6)
             ? fractional_rotation_to_cartesian(operation, ctx.lattice_vectors)
             : operation.rotation)
            .Inverse();
    const std::array<std::array<double, 3>, 3> cartesian_rotation_values{{
        {cartesian_rotation.e11, cartesian_rotation.e12, cartesian_rotation.e13},
        {cartesian_rotation.e21, cartesian_rotation.e22, cartesian_rotation.e23},
        {cartesian_rotation.e31, cartesian_rotation.e32, cartesian_rotation.e33},
    }};
    const double trs_sign = use_time_reversal ? -1.0 : 1.0;

    std::array<ComplexMatrix, 3> v_band_bz;
    for (int alpha = 0; alpha < 3; ++alpha) v_band_bz[alpha].create(n_bands, n_bands);
    for (int alpha_out = 0; alpha_out < 3; ++alpha_out)
    {
        for (int alpha_in = 0; alpha_in < 3; ++alpha_in)
        {
            const std::complex<double> coeff =
                trs_sign * cartesian_rotation_values.at(alpha_out).at(alpha_in);
            if (std::abs(coeff) < 1.e-15) continue;
            v_band_bz[alpha_out] += coeff * v_band_rot[alpha_in];
        }
    }
    return v_band_bz;
}

std::array<ComplexMatrix, 3> direct_full_bz_velocity_for_kstar_member(
    const velocity_matrix_t &velocity_full, const std::vector<std::vector<int>> &member_source_ik,
    const int ispin, const int ik_ibz, const std::size_t imember)
{
    if (ispin < 0 || ispin >= static_cast<int>(velocity_full.size()) || ik_ibz < 0 ||
        ik_ibz >= static_cast<int>(member_source_ik.size()) ||
        imember >= member_source_ik[ik_ibz].size())
    {
        throw std::runtime_error("direct_full_bz_velocity: invalid spin or k-star member index");
    }
    const int ik_source = member_source_ik[ik_ibz][imember];
    if (ik_source < 0 || ik_source >= static_cast<int>(velocity_full[ispin].size()) ||
        velocity_full[ispin][ik_source].size() != 3)
    {
        throw std::runtime_error("direct_full_bz_velocity: invalid PyATB full-BZ velocity entry");
    }
    return {velocity_full[ispin][ik_source][0], velocity_full[ispin][ik_source][1],
            velocity_full[ispin][ik_source][2]};
}

const ComplexMatrix &direct_full_bz_wfc_for_kstar_member(
    const MeanField &wfc_full, const std::vector<std::vector<int>> &member_source_ik,
    const int ispin, const int ispinor, const int ik_ibz, const std::size_t imember)
{
    if (ik_ibz < 0 || ik_ibz >= static_cast<int>(member_source_ik.size()) ||
        imember >= member_source_ik[ik_ibz].size())
    {
        throw std::runtime_error("direct_full_bz_wfc: invalid k-star member index");
    }
    const int ik_source = member_source_ik[ik_ibz][imember];
    const auto *wfc = wfc_full.find_wfc(ispin, ispinor, ik_source);
    if (wfc == nullptr || wfc->nr != wfc_full.get_n_states() ||
        wfc->nc != wfc_full.get_n_aos())
    {
        throw std::runtime_error("direct_full_bz_wfc: invalid PyATB full-BZ eigenvector entry");
    }
    return *wfc;
}

ComplexMatrix localize_direct_full_bz_wfc(const ComplexMatrix &wfc_full,
                                          const ArrayDesc &desc_wfc)
{
    if (!desc_wfc.is_initialized() || wfc_full.nr != desc_wfc.n() ||
        wfc_full.nc != desc_wfc.m())
    {
        throw std::runtime_error("direct full-BZ WFC descriptor is inconsistent");
    }

    ComplexMatrix wfc_local(desc_wfc.n_loc(), desc_wfc.m_loc(), false);
    for (int jloc = 0; jloc != desc_wfc.n_loc(); ++jloc)
    {
        const int iband = desc_wfc.indx_l2g_c(jloc);
        for (int iloc = 0; iloc != desc_wfc.m_loc(); ++iloc)
        {
            const int iao = desc_wfc.indx_l2g_r(iloc);
            wfc_local(jloc, iloc) = wfc_full(iband, iao);
        }
    }
    return wfc_local;
}

static std::vector<int> build_headwing_atom_offsets(const std::map<atom_t, size_t>& atom_nw)
{
    std::vector<int> offsets(atom_nw.size() + 1, 0);
    int running = 0;
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto iter = atom_nw.find(static_cast<atom_t>(atom));
        if (iter == atom_nw.end())
            throw std::runtime_error("headwing AO rotation atom_nw is not contiguous");
        offsets[atom] = running;
        running += static_cast<int>(iter->second);
    }
    offsets.back() = running;
    return offsets;
}

static Vector3_Order<int> headwing_equivalent_kpoint_shift(
    const Vector3_Order<double>& k_source, const Vector3_Order<double>& k_target)
{
    const Vector3_Order<double> k_shift{k_target.x - k_source.x,
                                        k_target.y - k_source.y,
                                        k_target.z - k_source.z};
    if (!nearly_integer_vector(k_shift, 1.e-5))
        throw std::runtime_error("headwing AO rotation got non-equivalent full-k targets");
    return round_to_integer_vector(k_shift);
}

static std::complex<double> headwing_reciprocal_gauge_phase(
    const SymmetryContext& ctx,
    const Vector3_Order<int>& k_shift,
    const atom_t atom)
{
    const auto coord_iter = ctx.input_coord_frac.find(atom);
    if (coord_iter == ctx.input_coord_frac.end())
        throw std::runtime_error("headwing AO rotation missing target-gauge coordinate");
    const Vector3_Order<double> tau{coord_iter->second};
    const double phase_arg =
        TWO_PI * (static_cast<double>(k_shift.x) * tau.x
                  + static_cast<double>(k_shift.y) * tau.y
                  + static_cast<double>(k_shift.z) * tau.z);
    return {std::cos(phase_arg), std::sin(phase_arg)};
}

struct HeadwingAoRotationBlocks
{
    std::vector<int> offsets;
    std::vector<ComplexMatrix> blocks;
};

static HeadwingAoRotationBlocks build_symmetry_ao_bloch_rotation_blocks(
    const SymmetryContext& ctx, const SymmetryKStarMember& member,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const std::map<atom_t, size_t>& atom_nw, const Vector3_Order<double>& k_ibz,
    const Vector3_Order<double>* k_bz_target)
{
    HeadwingAoRotationBlocks result;
    result.offsets = build_headwing_atom_offsets(atom_nw);

    std::vector<const SymmetryKAtomRotation*> rotations_by_from(atom_nw.size(), nullptr);
    std::vector<bool> visited_to(atom_nw.size(), false);
    for (const auto& atom_rotation : member.atom_rotations)
    {
        if (atom_rotation.atom_from < 0 || atom_rotation.atom_from >= static_cast<int>(atom_nw.size())
            || atom_rotation.atom_to < 0 || atom_rotation.atom_to >= static_cast<int>(atom_nw.size()))
            throw std::runtime_error("headwing AO rotation atom mapping is out of range");
        rotations_by_from[static_cast<std::size_t>(atom_rotation.atom_from)] = &atom_rotation;
        visited_to[static_cast<std::size_t>(atom_rotation.atom_to)] = true;
    }
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        if (rotations_by_from[atom] == nullptr)
            throw std::runtime_error("headwing AO rotations do not cover every atom");
        if (!visited_to[atom])
            throw std::runtime_error("headwing AO atom mapping is not a full permutation");
    }

    if (member.spatial_isym < 0
        || member.spatial_isym >= static_cast<int>(ctx.rspace_operations.size()))
        throw std::runtime_error("headwing AO rotation uses an invalid symmetry index");

    result.blocks.resize(atom_nw.size());
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto* atom_rotation = rotations_by_from[atom];
        const auto& layout =
            get_symmetry_species_layout(wfc_layouts, atom_rotation->atom_type);
        result.blocks[atom] =
            build_symmetry_rotation_matrix(layout, atom_rotation->bloch_rsh_rotations);
    }

    const bool apply_target_gauge = (k_bz_target != nullptr);
    std::vector<std::complex<double>> atom_target_phases(atom_nw.size(), {1.0, 0.0});
    if (apply_target_gauge)
    {
        const auto k_shift = headwing_equivalent_kpoint_shift(member.k_bz, *k_bz_target);
        for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
            atom_target_phases[atom] =
                headwing_reciprocal_gauge_phase(ctx, k_shift, static_cast<atom_t>(atom));
    }

    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        if (apply_target_gauge) result.blocks[atom] *= atom_target_phases[atom];
    }
    (void)k_ibz;
    return result;
}

static ComplexMatrix build_symmetry_ao_bloch_rotation_matrix_full(
    const SymmetryContext& ctx, const SymmetryKStarMember& member,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const std::map<atom_t, size_t>& atom_nw, const Vector3_Order<double>& k_ibz,
    const Vector3_Order<double>* k_bz_target)
{
    const auto rotation = build_symmetry_ao_bloch_rotation_blocks(
        ctx, member, wfc_layouts, atom_nw, k_ibz, k_bz_target);
    const int nao_total = rotation.offsets.back();
    ComplexMatrix M_full(nao_total, nao_total);
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const auto off = rotation.offsets[atom];
        const auto n_ao = static_cast<int>(atom_nw.at(static_cast<atom_t>(atom)));
        const auto &block = rotation.blocks[atom];
        for (int row = 0; row < n_ao; ++row)
            for (int col = 0; col < n_ao; ++col)
                M_full(off + row, off + col) = block(row, col);
    }
    return M_full;
}

static void fill_symmetry_ao_bloch_rotation_matrix_local(
    matrix_m<std::complex<double>> &M_local, const ArrayDesc &desc_M,
    const SymmetryContext& ctx, const SymmetryKStarMember& member,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const std::map<atom_t, size_t>& atom_nw, const Vector3_Order<double>& k_ibz,
    const Vector3_Order<double>* k_bz_target)
{
    const auto rotation = build_symmetry_ao_bloch_rotation_blocks(
        ctx, member, wfc_layouts, atom_nw, k_ibz, k_bz_target);
    if (desc_M.m() != rotation.offsets.back() || desc_M.n() != rotation.offsets.back())
        throw std::runtime_error("headwing AO rotation descriptor has inconsistent dimensions");
    M_local.zero_out();
    for (std::size_t atom = 0; atom < atom_nw.size(); ++atom)
    {
        const int off = rotation.offsets[atom];
        const int n_ao = static_cast<int>(atom_nw.at(static_cast<atom_t>(atom)));
        const auto &block = rotation.blocks[atom];
        for (int row = 0; row != n_ao; ++row)
        {
            const int iloc = desc_M.indx_g2l_r(off + row);
            if (iloc < 0) continue;
            for (int col = 0; col != n_ao; ++col)
            {
                const int jloc = desc_M.indx_g2l_c(off + col);
                if (jloc >= 0) M_local(iloc, jloc) = block(row, col);
            }
        }
    }
}

static std::vector<std::vector<ComplexMatrix>> rotate_headwing_wfc_symmetry_kblacs(
    const MeanField &mf, const int source_ik, const ArrayDesc &desc_wfc_src,
    const BlacsCtxtHandler &blacs_h, const SymmetryContext& ctx,
    const SymmetryKStarMember& member,
    const std::vector<SpeciesBasisLayout>& wfc_layouts,
    const std::map<atom_t, size_t>& atom_nw, const Vector3_Order<double>& k_ibz,
    const Vector3_Order<double>* k_bz_target)
{
    const int n_aos = mf.get_n_aos();
    const int n_states = mf.get_n_states();
    if (!desc_wfc_src.is_initialized() || desc_wfc_src.m() != n_aos ||
        desc_wfc_src.n() != n_states || desc_wfc_src.ictxt() != blacs_h.ictxt)
        throw std::runtime_error("symmetric headwing source wave-function descriptor is invalid");

    const int expected_block_ao =
        get_capped_blacs_block_size(n_aos, wfc_gemm_block_size_opt, blacs_h);
    const int expected_block_band =
        get_capped_blacs_block_size(n_states, wfc_gemm_block_size_opt, blacs_h);
    if (desc_wfc_src.mb() != expected_block_ao ||
        desc_wfc_src.nb() != expected_block_band)
        throw std::runtime_error(
            "symmetric headwing wave functions do not use the permanent capped layout");
    const int block_ao = desc_wfc_src.mb();
    ArrayDesc desc_M_opt(blacs_h);
    desc_M_opt.init(n_aos, n_aos, block_ao, block_ao, 0, 0);
    auto M_opt = init_local_mat<std::complex<double>>(desc_M_opt, MAJOR::COL);
    fill_symmetry_ao_bloch_rotation_matrix_local(
        M_opt, desc_M_opt, ctx, member, wfc_layouts, atom_nw, k_ibz, k_bz_target);

    std::vector<std::vector<ComplexMatrix>> result(mf.get_n_spins());
    std::vector<std::complex<double>> dummy(1, {0.0, 0.0});
    const std::size_t source_size =
        static_cast<std::size_t>(desc_wfc_src.m_loc()) * desc_wfc_src.n_loc();
    for (int ispin = 0; ispin != mf.get_n_spins(); ++ispin)
    {
        result[ispin].resize(mf.get_n_spinor());
        for (int ispinor = 0; ispinor != mf.get_n_spinor(); ++ispinor)
        {
            const auto *wfc_ibz = mf.find_wfc(ispin, ispinor, source_ik);
            const int bad_local =
                (source_size > 0 && wfc_ibz == nullptr) ||
                (wfc_ibz != nullptr &&
                 static_cast<std::size_t>(wfc_ibz->size) != source_size);
            int bad = 0;
            MPI_Allreduce(&bad_local, &bad, 1, MPI_INT, MPI_MAX, blacs_h.comm());
            if (bad)
                throw std::runtime_error(
                    "symmetric headwing local wave-function block is inconsistent");

            const auto *wfc_ibz_ptr =
                wfc_ibz == nullptr ? dummy.data() : wfc_ibz->c;
            auto &wfc_bz = result[ispin][ispinor];
            wfc_bz.create(desc_wfc_src.n_loc(), desc_wfc_src.m_loc(), false);
            ScalapackConnector::pgemm_f(
                'C', 'N', n_aos, n_states, n_aos, 1.0, M_opt.ptr(), 1, 1,
                desc_M_opt.desc, wfc_ibz_ptr, 1, 1, desc_wfc_src.desc, 0.0,
                wfc_bz.c == nullptr ? dummy.data() : wfc_bz.c, 1, 1,
                desc_wfc_src.desc);
        }
    }
    return result;
}

void initialize_velocity_matrix(velocity_matrix_t &velocity, const int n_spins,
                                const int n_kpoints, const int n_states)
{
    velocity.clear();
    velocity.resize(n_spins);
    for (int ispin = 0; ispin != n_spins; ++ispin)
    {
        velocity[ispin].resize(n_kpoints);
        for (int ik = 0; ik != n_kpoints; ++ik)
        {
            velocity[ispin][ik].resize(3);
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                velocity[ispin][ik][alpha].create(n_states, n_states);
            }
        }
    }
}

std::vector<int> map_kpoints_by_coordinates(
    const std::vector<Vector3_Order<double>> &target_kpoints,
    const std::vector<Vector3_Order<double>> &source_kpoints, const double tolerance)
{
    const auto periodic_abs_delta = [](const double lhs, const double rhs) {
        const double diff = lhs - rhs;
        return std::abs(diff - std::round(diff));
    };

    std::vector<int> target_to_source;
    target_to_source.reserve(target_kpoints.size());

    std::vector<char> used(source_kpoints.size(), 0);
    for (std::size_t itarget = 0; itarget != target_kpoints.size(); ++itarget)
    {
        int matched_source = -1;
        for (std::size_t isource = 0; isource != source_kpoints.size(); ++isource)
        {
            if (used[isource]) continue;
            const auto &lhs = target_kpoints[itarget];
            const auto &rhs = source_kpoints[isource];
            if (periodic_abs_delta(lhs.x, rhs.x) <= tolerance &&
                periodic_abs_delta(lhs.y, rhs.y) <= tolerance &&
                periodic_abs_delta(lhs.z, rhs.z) <= tolerance)
            {
                matched_source = static_cast<int>(isource);
                break;
            }
        }
        if (matched_source < 0)
        {
            std::ostringstream oss;
            oss << "Failed to map target k-point " << itarget + 1
                << " to the source k-point list";
            throw std::runtime_error(oss.str());
        }
        used[matched_source] = 1;
        target_to_source.emplace_back(matched_source);
    }

    return target_to_source;
}

std::vector<std::vector<int>> map_symmetry_kstar_members_to_source_kpoints(
    const SymmetryContext &ctx, const std::vector<Vector3_Order<double>> &ibz_kpoints,
    const std::vector<Vector3_Order<double>> &source_kpoints, const double tolerance)
{
    std::vector<std::size_t> member_offsets;
    member_offsets.reserve(ibz_kpoints.size() + 1);
    member_offsets.emplace_back(0);
    std::vector<Vector3_Order<double>> member_kpoints;
    for (const auto &k_ibz : ibz_kpoints)
    {
        const auto &star = find_symmetry_kstar_for_ibz_kpoint(ctx, k_ibz);
        if (star.members.empty())
            throw std::runtime_error("symmetry k-star has no members for full-BZ velocity");
        for (const auto &member : star.members) member_kpoints.emplace_back(member.k_bz);
        member_offsets.emplace_back(member_kpoints.size());
    }

    const auto member_source_indices =
        map_kpoints_by_coordinates(member_kpoints, source_kpoints, tolerance);
    std::vector<std::vector<int>> source_indices(ibz_kpoints.size());
    for (std::size_t ik = 0; ik != ibz_kpoints.size(); ++ik)
    {
        const auto begin = member_offsets[ik];
        const auto end = member_offsets[ik + 1];
        source_indices[ik].assign(member_source_indices.begin() + begin,
                                  member_source_indices.begin() + end);
    }
    return source_indices;
}

std::vector<double> interpolate_dielec_func(int option, const std::vector<double> &frequencies_in,
                                            const std::vector<double> &df_in,
                                            const std::vector<double> &frequencies_target)
{
    using librpa_int::DoubleHavriliakNegami;
    std::vector<double> df_target;

    switch (option)
    {
        case 0: /* No extrapolation, copy the input data to target */
        {
            assert(frequencies_in.size() == frequencies_target.size());
            df_target = df_in;
            break;
        }
        case 1: /* Use spline interpolation */
        {
            df_target = librpa_int::interp_cubic_spline(frequencies_in, df_in, frequencies_target);
            break;
        }
        case 2: /* Use dielectric model for fitting */
        {
            librpa_int::LevMarqFitting levmarq;
            // use double-dispersion Havriliak-Negami model
            // initialize the parameters as 1.0
            std::vector<double> pars(DoubleHavriliakNegami::d_npar, 1);
            pars[0] = pars[4] = df_in[0];
            df_target =
                levmarq.fit_eval(pars, frequencies_in, df_in, DoubleHavriliakNegami::func_imfreq,
                                 DoubleHavriliakNegami::grad_imfreq, frequencies_target);
            break;
        }
        default:
            throw LIBRPA_RUNTIME_ERROR("Unsupported value for option");
    }

    return df_target;
}

void diele_func::init(double coulomb_eigen_threshold, const librpa_int::atpair_k_cplx_mat_t &Vq)
{
    this->n_abf = atomic_basis_abf_.nb_total;
    this->nk = this->kfrac_band.size();
    if (static_cast<int>(velocity_.size()) != n_spin)
        throw LIBRPA_RUNTIME_ERROR(
            "head/wing velocity spin dimension is inconsistent with meanfield");
    for (int ispin = 0; ispin != n_spin; ++ispin)
    {
        if (static_cast<int>(velocity_[ispin].size()) != nk)
            throw LIBRPA_RUNTIME_ERROR(
                "head/wing velocity k-point dimension is inconsistent with k path");
        for (int ik = 0; ik != nk; ++ik)
        {
            if (velocity_[ispin][ik].size() != 3)
                throw LIBRPA_RUNTIME_ERROR(
                    "head/wing velocity must contain three Cartesian components");
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                const auto &vmat = velocity_[ispin][ik][alpha];
                if (vmat.nr != n_states || vmat.nc != n_states)
                    throw LIBRPA_RUNTIME_ERROR(
                        "head/wing velocity matrix size is inconsistent with bands");
            }
        }
    }
    if (has_direct_full_bz_headwing_inputs())
    {
        if (symmetry_context_ == nullptr ||
            static_cast<int>(direct_full_bz_velocity_.size()) != n_spin ||
            direct_full_bz_wfc_.get_n_spins() != n_spin ||
            direct_full_bz_wfc_.get_n_spinor() != meanfield_df.get_n_spinor() ||
            direct_full_bz_wfc_.get_n_states() != n_states ||
            direct_full_bz_wfc_.get_n_aos() != n_basis ||
            static_cast<int>(direct_full_bz_velocity_member_source_ik_.size()) != nk)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "direct full-BZ head/wing inputs are inconsistent with symmetric meanfield");
        }
        for (int ik_ibz = 0; ik_ibz != nk; ++ik_ibz)
        {
            const auto &star = librpa_int::find_symmetry_kstar_for_ibz_kpoint(
                *symmetry_context_, kfrac_band[ik_ibz]);
            if (direct_full_bz_velocity_member_source_ik_[ik_ibz].size() != star.members.size())
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "direct full-BZ velocity k-star mapping has an inconsistent member count");
            }
            for (const int ik_source : direct_full_bz_velocity_member_source_ik_[ik_ibz])
            {
                for (int ispin = 0; ispin != n_spin; ++ispin)
                {
                    if (ik_source < 0 ||
                        ik_source >= static_cast<int>(direct_full_bz_velocity_[ispin].size()) ||
                        direct_full_bz_velocity_[ispin][ik_source].size() != 3)
                    {
                        throw LIBRPA_RUNTIME_ERROR(
                            "direct full-BZ head/wing velocity has an invalid PyATB k point");
                    }
                    for (int alpha = 0; alpha != 3; ++alpha)
                    {
                        const auto &vmat = direct_full_bz_velocity_[ispin][ik_source][alpha];
                        if (vmat.nr != n_states || vmat.nc != n_states)
                        {
                            throw LIBRPA_RUNTIME_ERROR(
                                "direct full-BZ head/wing velocity matrix size is inconsistent "
                                "with bands");
                        }
                    }
                    for (int ispinor = 0; ispinor != meanfield_df.get_n_spinor(); ++ispinor)
                    {
                        const auto *wfc = direct_full_bz_wfc_.find_wfc(ispin, ispinor, ik_source);
                        if (wfc == nullptr || wfc->nr != n_states || wfc->nc != n_basis)
                        {
                            throw LIBRPA_RUNTIME_ERROR(
                                "direct full-BZ head/wing WFC has an invalid PyATB k point");
                        }
                    }
                }
            }
        }
    }
    int n_omega = this->omega.size();

    this->head.clear();
    this->head.resize(n_omega);
    for (int iomega = 0; iomega != n_omega; iomega++)
    {
        head[iomega].resize(3, 3, MAJOR::COL);
    }
};

void diele_func::init_wing(double coulomb_eigen_threshold, const atpair_k_cplx_mat_t &Vq)
{
    int n_omega = this->omega.size();
    this->n_abf = atomic_basis_abf_.nb_total;
    this->wing_mu.clear();
    this->wing_mu.resize(n_omega);
    this->wing.clear();
    this->n_nonsingular = n_abf;
    this->Lind.resize(3, 3, MAJOR::COL);
    for (int iomega = 0; iomega != n_omega; iomega++)
    {
        wing_mu[iomega].resize(n_abf, 3, MAJOR::COL);
    }
    get_Leb_points();
    if (use_2d_dielectric)
    {
        get_g_enclosing_gamma_2d();
        calculate_q_gamma_2d();
    }
    else
    {
        get_g_enclosing_gamma();
        calculate_q_gamma();
    }

    if (comm_h.is_root())
        std::cout << "* Success: initalize and calculate lebdev points and q_gamma." << std::endl;
}

// intraband term is not considered
void diele_func::cal_head()
{
    using global::profiler;

    profiler.start("cal_head");

    const bool can_try_sym =
        use_symmetry && symmetry_context_ != nullptr && atomic_basis_wfc_.has_l_shells();
    const auto wfc_layouts =
        can_try_sym ? atomic_basis_wfc_.build_species_basis_layouts(symmetry_context_->atom_to_type)
                    : std::vector<SpeciesBasisLayout>{};
    const bool can_sym =
        can_try_sym &&
        (librpa_int::can_restore_symmetry_kstar_meanfield(
             *symmetry_context_, wfc_layouts, meanfield_df, kfrac_band, atom_nw) ||
         can_use_headwing_single_member_stars(
             *symmetry_context_, wfc_layouts, kfrac_band, atom_nw));

    if (debug && use_symmetry && comm_h.is_root())
    {
        std::string reason;
        if (symmetry_context_ == nullptr)
            reason = "symmetry context is not set";
        else if (!atomic_basis_wfc_.has_l_shells())
            reason = "wave-function basis has no l-shell metadata";
        else if (wfc_layouts.empty())
            reason = "wave-function species layouts are empty";
        else if (!symmetry_context_->available)
            reason = "symmetry context is not available";
        else if (symmetry_context_->kstars.empty())
            reason = "k-star table is empty";
        else if (meanfield_df.get_n_kpoints() != static_cast<int>(kfrac_band.size()))
            reason = "meanfield and active k-list sizes differ";
        else if (symmetry_context_->kstars.size() != kfrac_band.size())
            reason = "k-star count does not match active k-list";
        else if (symmetry_context_->count_kstar_members() <= kfrac_band.size())
            reason = "active k-list is already full BZ";
        else
            reason = "all symmetry restore checks passed";
        std::cout << "Head symmetry restore for analytic head: "
                  << (can_sym ? "active" : "fallback") << " (" << reason << ")."
                  << std::endl;
    }

    if (can_sym)
        cal_head_symmetric();
    else
        cal_head_full_bz();

    // Common post-processing: apply dielectric unit and spin prefactor, add the
    // 1 contribution on the diagonal. Identical for both paths.
    const double dielectric_unit = cal_factor("head");
    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (int beta = 0; beta != 3; beta++)
        {
            for (size_t iomega = 0; iomega != this->omega.size(); iomega++)
            {
                this->head.at(iomega)(alpha, beta) *=
                    dielectric_unit * headwing_spin_prefactor(n_spin, use_soc);
                if (alpha == beta)
                {
                    this->head.at(iomega)(alpha, beta) += std::complex<double>(1.0, 0.0);
                }
            }
        }
    }
    global::ofs_myid << "* Success: calculate head term." << std::endl;
    profiler.stop("cal_head");
};

void diele_func::cal_head_full_bz()
{
    // Historical full-BZ summation. The k-grid must already cover the full BZ.
    // wg is indexed as wg(ik, ib) so that k-dependent occupations are honored.
    std::complex<double> tmp;
    const bool use_kblacs = use_matching_kpoint_blacs(nk, kblacs_ctxt_);
    const auto kpoints = headwing_local_kpoint_roots(nk, kblacs_ctxt_);
    std::vector<std::array<std::complex<double>, 9>> head_k_iomega0(as_size(nk));
    for (auto &head_k : head_k_iomega0) head_k.fill({0.0, 0.0});

    for (int ispin = 0; ispin != n_spin; ispin++)
    {
        auto &wg = this->meanfield_df.get_weight()[ispin];
        auto &eigenvalues = this->meanfield_df.get_eigenvals()[ispin];
        const auto &velocity = this->velocity_[ispin];
        for (const int ik : kpoints)
        {
            for (int iocc = 0; iocc != n_states; iocc++)
            {
                for (int iunocc = 0; iunocc != n_states; iunocc++)
                {
                    if (iocc < iunocc)
                    {
                        double egap =
                            (eigenvalues(ik, iocc) - eigenvalues(ik, iunocc));  // * HA2EV;
                        // NOTE: wg is matrix(n_kpoints, n_states); index with (ik, ib) not flat
                        // buffer, otherwise the k-dependence of the occupation is silently
                        // dropped. For an insulator all rows are equal so this is a numerical
                        // no-op, but it is required for correctness on metals / IBZ input.
                        const double factor = headwing_transition_weight(
                            wg(ik, iocc), wg(ik, iunocc), n_spin, use_soc);
                        if (factor > 1.e-8)
                        {
                            for (int alpha = 0; alpha != 3; alpha++)
                            {
                                for (int beta = 0; beta != 3; beta++)
                                {
                                    for (size_t iomega = 0; iomega != this->omega.size(); iomega++)
                                    {
                                        double omega_ev = this->omega[iomega];  // * HA2EV;
                                        tmp = 2.0 * factor * velocity[ik][alpha](iunocc, iocc) *
                                              velocity[ik][beta](iocc, iunocc) /
                                              (egap * egap + omega_ev * omega_ev) / egap;
                                        this->head.at(iomega)(alpha, beta) -= tmp;
                                        if (iomega == 0)
                                            head_k_iomega0.at(as_size(ik)).at(
                                                as_size(alpha * 3 + beta)) -= tmp;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (use_kblacs) allreduce_head_matrices(this->head, comm_h.comm);
    if (comm_h.is_root())
    {
        for (int ik = 0; ik != nk; ++ik)
            print_head_k_contribution("full_bz", ik, kfrac_band[ik],
                                      head_k_iomega0.at(as_size(ik)));
    }
}

void diele_func::cal_head_symmetric()
{
    // Symmetry-aware head: sum over the full BZ by looping IBZ k-points and, for
    // each, every member of the k-star. Full-BZ PyATB velocity is selected at
    // the member k point when available; otherwise the historical IBZ rotation
    // fallback is used. The eigenvalues are symmetry-invariant, so the IBZ gap
    // is reused directly.
    if (symmetry_context_ == nullptr)
        throw std::runtime_error("cal_head_symmetric: symmetry context is not set");
    const auto& ctx = *symmetry_context_;

    // Build the per-member BZ k-point targets so the rotated quantities land on
    // the same grid keys that the rest of the code expects (mirrors the
    // get_symmetry_restored_gf_cplx_imagtimes_Rs convention).
    const auto member_targets =
        librpa_int::build_symmetry_kstar_member_kfrac_targets(ctx, pbc_);

    // PyATB velocity inputs on an IBZ grid use the active-grid occupation normalization.
    // Expanding each representative to the full BZ therefore scales every member uniformly.
    const std::size_t n_kpoints_bz = ctx.count_kstar_members();
    const double bz_weight_scale = static_cast<double>(nk) / static_cast<double>(n_kpoints_bz);
    const bool use_kblacs = use_matching_kpoint_blacs(nk, kblacs_ctxt_);

    // Debug self-verification accumulator: independently accumulate the head
    // using the same per-BZ-k weight on the rotated velocities, then compare.
    std::vector<std::array<std::array<std::complex<double>, 3>, 3>> head_check(this->omega.size());
    if (debug)
        for (auto &h : head_check)
            for (auto &row : h) row.fill({0.0, 0.0});

    for (int ispin = 0; ispin != n_spin; ispin++)
    {
        auto &wg = this->meanfield_df.get_weight()[ispin];
        auto &eigenvalues = this->meanfield_df.get_eigenvals()[ispin];
        const auto &velocity = this->velocity_[ispin];

        for (int ik_ibz = 0; ik_ibz != nk; ik_ibz++)
        {
            const auto& k_ibz = kfrac_band[ik_ibz];
            const auto& star = librpa_int::find_symmetry_kstar_for_ibz_kpoint(ctx, k_ibz);
            if (star.members.empty())
                throw std::runtime_error("cal_head_symmetric: empty k-star");

            if (use_kblacs)
            {
                if (!kblacs_ctxt_->owns_kpoint(ik_ibz) || kblacs_ctxt_->comm_blacs_h.myid != 0)
                    continue;
            }
            else
            {
                const auto *C_ibz_ptr = meanfield_df.find_wfc(ispin, 0, ik_ibz);
                const int owner_rank = find_mpi_owner_rank(C_ibz_ptr != nullptr, comm_h.comm);
                if (owner_rank < 0)
                    throw std::runtime_error(
                        "cal_head_symmetric: no rank owns the IBZ eigenvectors");
                if (comm_h.myid != owner_rank) continue;
            }

            for (std::size_t imember = 0; imember != star.members.size(); ++imember)
            {
                const auto &member = star.members[imember];
                const auto &k_bz =
                    member_targets.empty() ? member.k_bz : member_targets[ik_ibz][imember];
                std::array<std::complex<double>, 9> head_k_iomega0{};

                (void)k_bz;
                const auto v_band_bz = has_direct_full_bz_headwing_inputs()
                                          ? direct_full_bz_velocity_for_kstar_member(
                                                direct_full_bz_velocity_,
                                                direct_full_bz_velocity_member_source_ik_, ispin,
                                                ik_ibz, imember)
                                          : rotate_headwing_velocity_to_kstar_member(
                                                ctx, member,
                                                {velocity[ik_ibz][0], velocity[ik_ibz][1],
                                                 velocity[ik_ibz][2]},
                                                n_states, member.time_reversal);

                // Sum over band pairs. Eigenvalues are symmetry-invariant, so the
                // IBZ gap Delta_cv applies to every star member unchanged.
                for (int iocc = 0; iocc != n_states; iocc++)
                {
                    for (int iunocc = 0; iunocc != n_states; iunocc++)
                    {
                        if (iocc >= iunocc) continue;
                        const double factor =
                            bz_weight_scale * headwing_transition_weight(wg(ik_ibz, iocc),
                                                                         wg(ik_ibz, iunocc), n_spin,
                                                                         use_soc);
                        if (factor <= 1.e-8) continue;

                        const double egap =
                            (eigenvalues(ik_ibz, iocc) - eigenvalues(ik_ibz, iunocc));

                        for (int alpha = 0; alpha != 3; ++alpha)
                        {
                            for (int beta = 0; beta != 3; ++beta)
                            {
                                for (size_t iomega = 0; iomega != this->omega.size(); iomega++)
                                {
                                    const double omega_ev = this->omega[iomega];
                                    const std::complex<double> tmp =
                                        2.0 * factor * v_band_bz[alpha](iunocc, iocc) *
                                        v_band_bz[beta](iocc, iunocc) /
                                        (egap * egap + omega_ev * omega_ev) / egap;
                                    this->head.at(iomega)(alpha, beta) -= tmp;
                                    if (iomega == 0)
                                        head_k_iomega0.at(as_size(alpha * 3 + beta)) -= tmp;
                                    if (debug) head_check[iomega][alpha][beta] -= tmp;
                                }
                            }
                        }
                    }
                }
                if (comm_h.is_root())
                    print_head_k_contribution("sym_restored", ik_ibz, k_bz, head_k_iomega0);
            }
        }
    }

    allreduce_head_matrices(this->head, comm_h.comm);
    if (debug) allreduce_head_check(head_check, comm_h.comm);

    // Self-verification: compare the symmetric-path head (this->head) against the
    // independent full-BZ-convention accumulation (head_check). The symmetric path
    // uses wg(ibz) and sums |star| members; the check uses wg(bz)=wg(ibz)/|star|.
    // Both traverse the identical {BZ k, v(k), Delta, wg(k)} set, so they must
    // agree to machine precision (the only difference is the order of summation).
    // NOTE: this->head has NOT yet been scaled by dielectric_unit/spin_prefactor
    // (that happens in the cal_head dispatcher after this function returns), and
    // head_check was accumulated with the same unscaled formula, so the raw sums
    // are directly comparable.
    if (debug && global::mpi_comm_global_h.is_root())
    {
        double max_abs_diff = 0.0;
        double max_abs_val = 0.0;
        for (size_t iw = 0; iw < this->omega.size(); ++iw)
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                {
                    const auto dv = this->head.at(iw)(a, b) - head_check[iw][a][b];
                    max_abs_diff = std::max(max_abs_diff, std::abs(dv));
                    max_abs_val = std::max(max_abs_val, std::abs(this->head.at(iw)(a, b)));
                }
        const double rel_diff = (max_abs_val > 0) ? max_abs_diff / max_abs_val : 0.0;
        std::cout << "[cal_head_symmetric self-check] head_sym vs head_fullbz_convention: "
                  << "max_abs_diff=" << max_abs_diff << ", max_abs_val=" << max_abs_val
                  << ", rel_diff=" << rel_diff << std::endl;
        if (rel_diff > 1e-10 && max_abs_val > 1e-12)
        {
            std::cerr << "WARNING: cal_head_symmetric self-check rel_diff=" << rel_diff
                      << " exceeds 1e-10! Symmetric and full-BZ-convention heads disagree."
                      << std::endl;
        }
        else
        {
            std::cout << "[cal_head_symmetric self-check] PASSED: symmetric head == full-BZ head "
                      << "(rel_diff < 1e-10)." << std::endl;
        }
    }
}

double diele_func::cal_factor(std::string name)
{
    using librpa_int::BOHR2ANG;
    using librpa_int::TWO_PI;

    double dielectric_unit;
    const auto &latvec = pbc_.latvec;
    double primitive_cell_volume;
    if (use_2d_dielectric)
    {
        // Bohr
        primitive_cell_volume = std::abs(latvec.e11 * latvec.e22 - latvec.e12 * latvec.e21) * 10;
    }
    else
    {                                                    //! Bohr to A
        primitive_cell_volume = std::abs(latvec.Det());  //* BOHR2ANG * BOHR2ANG * BOHR2ANG;}
    }
    // latvec.print();
    if (name == "head")
    {
        dielectric_unit = 2 * TWO_PI / primitive_cell_volume;
    }
    else if (name == "wing")
    {
        dielectric_unit = 2 * sqrt(2 * TWO_PI / primitive_cell_volume);  // bohr
    }
    else
        throw std::logic_error("Unsupported value for head/wing factor");
    return dielectric_unit;
};

void diele_func::set_0_wing()
{
    int n_lambda = this->n_nonsingular - 1;
    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (std::size_t iomega = 0; iomega != this->omega.size(); iomega++)
        {
            for (int mu = 0; mu != n_abf; mu++)
            {
                this->wing_mu.at(iomega)(mu, alpha) = 0.0;
            }
        }
    }
};

matrix_m<std::complex<double>> diele_func::get_rpa_chi0v_head(const int ifreq) const
{
    if (ifreq < 0 || static_cast<std::size_t>(ifreq) >= head.size())
    {
        throw std::runtime_error("RPA chi0*v head requested before dielectric head is available");
    }

    auto chi0v_head = head.at(ifreq).copy();
    chi0v_head *= -1.0;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        chi0v_head(alpha, alpha) += 1.0;
    }
    return chi0v_head;
}

matrix_m<std::complex<double>> diele_func::get_rpa_chi0v_wing(const int ifreq) const
{
    if (ifreq < 0 || static_cast<std::size_t>(ifreq) >= wing.size())
    {
        throw std::runtime_error("RPA chi0*v wing requested before dielectric wing is available");
    }

    auto chi0v_wing = wing.at(ifreq).copy();
    chi0v_wing *= -1.0;
    return chi0v_wing;
}

SternheimerRpaHeadwingInput diele_func::get_sternheimer_rpa_headwing_input(
    const int ifreq, const RpaHeadwingSettings &settings) const
{
    if (settings.rpa_headwing_mode != "qavg" && settings.rpa_headwing_mode != "head_only")
    {
        throw std::logic_error("ST-RPA head/wing mode must be qavg or head_only");
    }

    const auto chi0v_head = get_rpa_chi0v_head(ifreq);
    SternheimerRpaHeadwingInput result;
    result.mode = settings.rpa_headwing_mode;
    result.body_start = rpa_headwing_regular_body_start_channel(settings);
    result.head = ComplexMatrix(3, 3);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        for (int beta = 0; beta != 3; ++beta)
        {
            result.head(alpha, beta) = chi0v_head(alpha, beta);
        }
    }

    if (result.mode == "head_only")
    {
        return result;
    }
    if (ifreq < 0 || static_cast<std::size_t>(ifreq) >= wing_mu.size())
    {
        throw std::runtime_error("ST-RPA qavg requested before analytic wing is available");
    }
    if (qx_leb.size() != qy_leb.size() || qx_leb.size() != qz_leb.size() ||
        qx_leb.size() != qw_leb.size() || qx_leb.size() != q_gamma.size())
    {
        throw std::logic_error("ST-RPA head/wing angular quadrature data are inconsistent");
    }

    // cal_wing stores the dielectric-function convention. Negating here gives
    // the chi0*v convention returned by get_rpa_chi0v_wing after the usual
    // sqrt(V) auxiliary-to-Coulomb transformation.
    result.wing_mu = ComplexMatrix(n_abf, 3);
    for (int mu = 0; mu != n_abf; ++mu)
    {
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            result.wing_mu(mu, alpha) = -wing_mu.at(ifreq)(mu, alpha);
        }
    }

    const double volume = rpa_headwing_gamma_cell_volume(pbc_, settings.use_2d_dielectric);
    result.directions.reserve(qw_leb.size());
    for (std::size_t ileb = 0; ileb != qw_leb.size(); ++ileb)
    {
        const double radial_weight = settings.use_2d_dielectric
                                         ? std::pow(q_gamma[ileb], 2) / (2.0 * volume)
                                         : std::pow(q_gamma[ileb], 3) / (3.0 * volume);
        result.directions.push_back(
            {{{qx_leb[ileb], qy_leb[ileb], qz_leb[ileb]}}, qw_leb[ileb] * radial_weight});
    }
    return result;
}

void diele_func::cal_wing(const Cs_LRI &Cs_data, double coulomb_eigen_threshold,
                          const atpair_k_cplx_mat_t &Vq)
{
    const bool can_try_sym =
        use_symmetry && symmetry_context_ != nullptr && atomic_basis_wfc_.has_l_shells();
    const auto wfc_layouts =
        can_try_sym ? atomic_basis_wfc_.build_species_basis_layouts(symmetry_context_->atom_to_type)
                    : std::vector<SpeciesBasisLayout>{};
    const bool can_sym =
        can_try_sym &&
        (librpa_int::can_restore_symmetry_kstar_meanfield(
             *symmetry_context_, wfc_layouts, meanfield_df, kfrac_band, atom_nw) ||
         can_use_headwing_single_member_stars(
             *symmetry_context_, wfc_layouts, kfrac_band, atom_nw));

    if (debug && use_symmetry && comm_h.is_root())
    {
        std::string reason;
        if (symmetry_context_ == nullptr)
            reason = "symmetry context is not set";
        else if (!atomic_basis_wfc_.has_l_shells())
            reason = "wave-function basis has no l-shell metadata";
        else if (wfc_layouts.empty())
            reason = "wave-function species layouts are empty";
        else if (!symmetry_context_->available)
            reason = "symmetry context is not available";
        else if (symmetry_context_->kstars.empty())
            reason = "k-star table is empty";
        else if (meanfield_df.get_n_kpoints() != static_cast<int>(kfrac_band.size()))
            reason = "meanfield and active k-list sizes differ";
        else if (symmetry_context_->kstars.size() != kfrac_band.size())
            reason = "k-star count does not match active k-list";
        else if (symmetry_context_->count_kstar_members() <= kfrac_band.size())
            reason = "active k-list is already full BZ";
        else
            reason = "all symmetry restore checks passed";
        std::cout << "Wing symmetry restore for analytic wing: "
                  << (can_sym ? "active" : "fallback") << " (" << reason << ")."
                  << std::endl;
    }

    if (can_sym)
        cal_wing_symmetric(Cs_data, coulomb_eigen_threshold, Vq);
    else
        cal_wing_full_bz(Cs_data, coulomb_eigen_threshold, Vq);
}

void diele_func::cal_wing_full_bz(const Cs_LRI &Cs_data, double coulomb_eigen_threshold,
                                  const atpair_k_cplx_mat_t &Vq)
{
    using global::profiler;

    profiler.start("cal_wing_mu");
    init_wing(coulomb_eigen_threshold, Vq);
    std::vector<std::complex<double>> local_wing_mu;
    local_wing_mu.resize(this->omega.size() * 3 * n_abf, 0.0);
    std::vector<std::complex<double>> local_wing_mu_k_iomega0(
        as_size(nk) * as_size(n_abf) * 3, 0.0);
    const bool use_kblacs = use_kpara_eigvec_;
    if (use_kblacs &&
        (!use_matching_kpoint_blacs(nk, kblacs_ctxt_) || desc_wfc_kblacs_ == nullptr ||
         !desc_wfc_kblacs_->is_initialized()))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "k-local head/wing requires an initialized SCF k-BLACS context and "
            "wave-function descriptor");
    }
    const BlacsCtxtHandler &wing_blacs_h = use_kblacs ? kblacs_ctxt_->blacs_h : blacs_h;
    const auto kpoints_local = headwing_local_kpoints(nk, use_kblacs ? kblacs_ctxt_ : nullptr);

    ArrayDesc desc_nao_nao(wing_blacs_h);
    desc_nao_nao.init_1b1p(n_basis, n_basis, 0, 0);
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    HeadwingCsIJKMap Cs_IJ_k;
    if (use_kblacs)
    {
        profiler.start("headwing_transform_Cs2mnk_kblacs_para");
        const auto targets = build_headwing_full_bz_fourier_targets(kfrac_band);
        Cs_IJ_k = redistribute_headwing_cs_ijk(
            Cs_data, atomic_basis_wfc_, atomic_basis_abf_, targets, kpoints_local,
            desc_nao_nao, comm_h);
    }
    else
    {
        const auto set_IJ_nao_nao = get_necessary_IJ_from_block_2D(
            atomic_basis_wfc_, atomic_basis_wfc_, desc_nao_nao);
        const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nao_nao);
        Cs_IJ = comm_map2_first(comm_h.comm, Cs_data.data_libri, s0_s1.first, s0_s1.second);
    }

    // #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int mu = 0; mu < n_abf; ++mu)
    {
        for (const int ik : kpoints_local)
        {
            for (int isp = 0; isp != n_spin; isp++)
            {
                auto desc_C_mnk =
                    use_kblacs
                        ? transform_Cs2mnk_kblacs(ik, ik, mu, Cs_IJ_k, wing_blacs_h, nullptr,
                                                  isp)
                        : transform_Cs2mnk(ik, mu, Cs_IJ, isp);
                auto &desc_nband_nband = desc_C_mnk.first;
                auto &C_mnk = desc_C_mnk.second;
                print_wing_cmnk_probe("full_bz", mu, kfrac_band[ik], desc_nband_nband, C_mnk);
                const bool use_soc_wing = meanfield_df.get_n_spinor() > 1;
                const auto &eigenvalues = this->meanfield_df.get_eigenvals();
                const auto &wg = this->meanfield_df.get_weight()[isp];
                const auto &velocity = this->velocity_[isp][ik];
                auto *wing_mu_for_mu = local_wing_mu.data() + as_size(mu) * this->omega.size() * 3;
                auto *wing_mu_k_iomega0_for_mu =
                    local_wing_mu_k_iomega0.data() + as_size(ik * n_abf + mu) * 3;

                for (int iocc = 0; iocc != n_states; iocc++)
                {
                    const int loc_m = desc_nband_nband.indx_g2l_r(iocc);
                    if (loc_m < 0) continue;

                    for (int iunocc = iocc + 1; iunocc != n_states; iunocc++)
                    {
                        const int loc_n = desc_nband_nband.indx_g2l_c(iunocc);
                        if (loc_n < 0) continue;

                        const double egap =
                            eigenvalues[isp](ik, iunocc) - eigenvalues[isp](ik, iocc);
                        double factor1 = 0.0;
                        double factor2 = 0.0;
                        if (use_soc_wing)
                        {
                            factor1 = wg(ik, iocc) * (1.0 - wg(ik, iunocc) * nk);
                            factor2 = wg(ik, iunocc) * (1.0 - wg(ik, iocc) * nk);
                        }
                        else
                        {
                            factor1 = wg(ik, iocc) / 2 * n_spin *
                                      (1.0 - wg(ik, iunocc) / 2 * n_spin * nk);
                            factor2 = wg(ik, iunocc) / 2 * n_spin *
                                      (1.0 - wg(ik, iocc) / 2 * n_spin * nk);
                        }
                        if (factor1 <= 1.e-8 && factor2 <= 1.e-8) continue;

                        const auto c_mn = C_mnk(loc_m, loc_n);
                        const std::array<std::complex<double>, 3> velocity_unocc_occ{
                            velocity[0](iunocc, iocc), velocity[1](iunocc, iocc),
                            velocity[2](iunocc, iocc)};
                        accumulate_wing_mu_for_pair(this->omega, velocity_unocc_occ, c_mn, egap,
                                                    factor1, factor2, wing_mu_for_mu,
                                                    wing_mu_k_iomega0_for_mu);
                    }
                }
            }
            // profiler.stop("compute_wing");
        }
    }

    if (comm_h.is_root())
    {
        for (int ik = 0; ik != nk; ++ik)
        {
            const auto begin = local_wing_mu_k_iomega0.begin() + as_size(ik * n_abf) * 3;
            const auto end = begin + as_size(n_abf) * 3;
            print_wing_mu_k_contribution_gram("full_bz", ik, kfrac_band[ik],
                                              {begin, end}, n_abf);
        }
    }
    if (use_kblacs)
    {
        Cs_IJ_k.clear();
        profiler.stop("headwing_transform_Cs2mnk_kblacs_para");
    }

    profiler.start("Comm_wing");
    MPI_Allreduce(MPI_IN_PLACE, local_wing_mu.data(), static_cast<int>(local_wing_mu.size()),
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    profiler.stop("Comm_wing");
    double dielectric_unit = cal_factor("wing");

    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (int mu = 0; mu != n_abf; mu++)
        {
            for (std::size_t iomega = 0; iomega != this->omega.size(); iomega++)
            {
                const std::size_t index =
                    as_size(mu) * this->omega.size() * 3 + iomega * 3 + as_size(alpha);
                this->wing_mu.at(iomega)(mu, alpha) = local_wing_mu[index];
                this->wing_mu.at(iomega)(mu, alpha) *=
                    -dielectric_unit * headwing_spin_prefactor(n_spin, use_soc);
            }
        }
    }
    // transform_mu_to_lambda();
    if (comm_h.is_root()) std::cout << "* Success: calculate wing term." << std::endl;

    // this->wing_mu.clear();
    release_free_mem();
    profiler.stop("cal_wing_mu");
};

void diele_func::cal_wing_symmetric(const Cs_LRI &Cs_data, double coulomb_eigen_threshold,
                                    const atpair_k_cplx_mat_t &Vq)
{
    using global::profiler;
    profiler.start("cal_wing_mu");

    init_wing(coulomb_eigen_threshold, Vq);
    std::vector<std::complex<double>> local_wing_mu;
    local_wing_mu.resize(this->omega.size() * 3 * n_abf, 0.0);

    const bool use_kblacs = use_kpara_eigvec_;
    if (use_kblacs &&
        (!use_matching_kpoint_blacs(nk, kblacs_ctxt_) || desc_wfc_kblacs_ == nullptr ||
         !desc_wfc_kblacs_->is_initialized()))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "k-local symmetric head/wing requires an initialized SCF k-BLACS context and "
            "wave-function descriptor");
    }
    const BlacsCtxtHandler &wing_blacs_h = use_kblacs ? kblacs_ctxt_->blacs_h : blacs_h;
    const auto kpoints_local = headwing_local_kpoints(nk, use_kblacs ? kblacs_ctxt_ : nullptr);

    if (symmetry_context_ == nullptr)
        throw std::runtime_error("cal_wing_symmetric: symmetry context is not set");
    const auto &ctx = *symmetry_context_;
    const auto member_targets =
        librpa_int::build_symmetry_kstar_member_kfrac_targets(ctx, pbc_);
    const auto fourier_targets = build_headwing_symmetry_fourier_targets(ctx, pbc_, kfrac_band);

    ArrayDesc desc_nao_nao(wing_blacs_h);
    desc_nao_nao.init_1b1p(n_basis, n_basis, 0, 0);
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> Cs_IJ;
    HeadwingCsIJKMap Cs_IJ_k;
    if (use_kblacs)
    {
        profiler.start("headwing_transform_Cs2mnk_kblacs_para");
        profiler.start("headwing_transform_Cs2mnk_sym_kblacs_para");
        Cs_IJ_k = redistribute_headwing_cs_ijk(
            Cs_data, atomic_basis_wfc_, atomic_basis_abf_, fourier_targets.targets,
            kpoints_local, desc_nao_nao, comm_h);
    }
    else
    {
        const auto set_IJ_nao_nao = get_necessary_IJ_from_block_2D(
            atomic_basis_wfc_, atomic_basis_wfc_, desc_nao_nao);
        const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nao_nao);
        Cs_IJ = comm_map2_first(comm_h.comm, Cs_data.data_libri, s0_s1.first, s0_s1.second);
    }

    const int n_kpoints_ibz = nk;
    const int n_spinor = meanfield_df.get_n_spinor();
    const int n_kpoints_bz = static_cast<int>(ctx.count_kstar_members());
    const double bz_weight_scale_wing =
        static_cast<double>(n_kpoints_ibz) / static_cast<double>(n_kpoints_bz);
    const bool source_rank = wing_blacs_h.myprow == 0 && wing_blacs_h.mypcol == 0;
    if (!has_direct_full_bz_headwing_inputs())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "symmetry head/wing requires full-BZ PyATB velocity_matrix and KS eigenvectors");
    }

    for (const int ik_ibz : kpoints_local)
    {
        const auto &k_ibz = kfrac_band[ik_ibz];
        const auto &star = librpa_int::find_symmetry_kstar_for_ibz_kpoint(ctx, k_ibz);
        if (star.members.empty()) throw std::runtime_error("cal_wing_symmetric: empty k-star");

        for (std::size_t imember = 0; imember != star.members.size(); ++imember)
        {
            const auto &member = star.members[imember];
            const auto &k_bz =
                member_targets.empty() ? member.k_bz : member_targets[ik_ibz][imember];
            std::vector<std::complex<double>> member_wing_mu_iomega0(as_size(n_abf) * 3, 0.0);
            const int target_id =
                fourier_targets.target_ids_by_ibz_member.at(ik_ibz).at(imember);

            std::vector<std::array<ComplexMatrix, 3>> velocity_bz(n_spin);
            for (int ispin = 0; ispin != n_spin; ++ispin)
            {
                velocity_bz[ispin] = direct_full_bz_velocity_for_kstar_member(
                    direct_full_bz_velocity_, direct_full_bz_velocity_member_source_ik_, ispin,
                    ik_ibz, imember);
            }

            std::vector<std::vector<ComplexMatrix>> wfc_bz_storage(n_spin);
            std::vector<std::vector<const ComplexMatrix *>> wfc_bz_ptrs(n_spin);
            for (int ispin = 0; ispin != n_spin; ++ispin)
            {
                wfc_bz_storage[ispin].resize(n_spinor);
                wfc_bz_ptrs[ispin].resize(n_spinor, nullptr);
                for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
                {
                    if (!use_kblacs && !source_rank) continue;
                    const auto &wfc_full = direct_full_bz_wfc_for_kstar_member(
                        direct_full_bz_wfc_, direct_full_bz_velocity_member_source_ik_,
                        ispin, ispinor, ik_ibz, imember);
                    if (use_kblacs)
                    {
                        wfc_bz_storage[ispin][ispinor] =
                            localize_direct_full_bz_wfc(wfc_full, *desc_wfc_kblacs_);
                        wfc_bz_ptrs[ispin][ispinor] = &wfc_bz_storage[ispin][ispinor];
                    }
                    else
                    {
                        wfc_bz_ptrs[ispin][ispinor] = &wfc_full;
                    }
                }
            }

            for (int mu = 0; mu != n_abf; ++mu)
            {
                auto *wing_mu_for_mu = local_wing_mu.data() + as_size(mu) * this->omega.size() * 3;
                auto *member_wing_mu_iomega0_for_mu =
                    member_wing_mu_iomega0.data() + as_size(mu) * 3;

                for (int isp = 0; isp != n_spin; ++isp)
                {
                    auto desc_C_mnk =
                        use_kblacs
                            ? transform_Cs2mnk_kblacs(ik_ibz, target_id, mu, Cs_IJ_k,
                                                      wing_blacs_h, &wfc_bz_ptrs, isp)
                            : transform_Cs2mnk_kblacs(ik_ibz, mu, Cs_IJ, wing_blacs_h,
                                                      k_bz, &wfc_bz_ptrs, isp);
                    auto &desc_nband_nband = desc_C_mnk.first;
                    auto &C_mnk = desc_C_mnk.second;
                    print_wing_cmnk_probe("sym_restored", mu, k_bz, desc_nband_nband, C_mnk);
                    const bool use_soc_wing = meanfield_df.get_n_spinor() > 1;
                    const auto &eigenvalues = this->meanfield_df.get_eigenvals();
                    const auto &wg = this->meanfield_df.get_weight()[isp];
                    const auto &velocity = velocity_bz[isp];

                    for (int iocc = 0; iocc != n_states; iocc++)
                    {
                        const int loc_m = desc_nband_nband.indx_g2l_r(iocc);
                        if (loc_m < 0) continue;

                        for (int iunocc = iocc + 1; iunocc != n_states; iunocc++)
                        {
                            const int loc_n = desc_nband_nband.indx_g2l_c(iunocc);
                            if (loc_n < 0) continue;

                            const double egap =
                                eigenvalues[isp](ik_ibz, iunocc) - eigenvalues[isp](ik_ibz, iocc);
                            const double wg_occ = wg(ik_ibz, iocc) * bz_weight_scale_wing;
                            const double wg_unocc = wg(ik_ibz, iunocc) * bz_weight_scale_wing;
                            double factor1 = 0.0;
                            double factor2 = 0.0;
                            if (use_soc_wing)
                            {
                                factor1 = wg_occ * (1.0 - wg_unocc * n_kpoints_bz);
                                factor2 = wg_unocc * (1.0 - wg_occ * n_kpoints_bz);
                            }
                            else
                            {
                                factor1 = wg_occ / 2 * n_spin *
                                          (1.0 - wg_unocc / 2 * n_spin * n_kpoints_bz);
                                factor2 = wg_unocc / 2 * n_spin *
                                          (1.0 - wg_occ / 2 * n_spin * n_kpoints_bz);
                            }
                            if (factor1 <= 1.e-8 && factor2 <= 1.e-8) continue;

                            const auto c_mn = C_mnk(loc_m, loc_n);
                            const std::array<std::complex<double>, 3> velocity_unocc_occ{
                                velocity[0](iunocc, iocc), velocity[1](iunocc, iocc),
                                velocity[2](iunocc, iocc)};
                            accumulate_wing_mu_for_pair(this->omega, velocity_unocc_occ, c_mn, egap,
                                                        factor1, factor2, wing_mu_for_mu,
                                                        member_wing_mu_iomega0_for_mu);
                        }
                    }
                }
            }
            if (comm_h.is_root())
                print_wing_mu_k_contribution_gram("sym_restored", ik_ibz, k_bz,
                                                  member_wing_mu_iomega0, n_abf, &member);
        }
    }

    if (use_kblacs)
    {
        Cs_IJ_k.clear();
        profiler.stop("headwing_transform_Cs2mnk_sym_kblacs_para");
        profiler.stop("headwing_transform_Cs2mnk_kblacs_para");
    }

    profiler.start("Comm_wing");
    MPI_Allreduce(MPI_IN_PLACE, local_wing_mu.data(), static_cast<int>(local_wing_mu.size()),
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
    profiler.stop("Comm_wing");
    double dielectric_unit = cal_factor("wing");

    for (int alpha = 0; alpha != 3; alpha++)
    {
        for (int mu = 0; mu != n_abf; mu++)
        {
            for (std::size_t iomega = 0; iomega != this->omega.size(); iomega++)
            {
                const std::size_t index =
                    as_size(mu) * this->omega.size() * 3 + iomega * 3 + as_size(alpha);
                this->wing_mu.at(iomega)(mu, alpha) = local_wing_mu[index];
                this->wing_mu.at(iomega)(mu, alpha) *=
                    -dielectric_unit * headwing_spin_prefactor(n_spin, use_soc);
            }
        }
    }
    if (comm_h.is_root()) std::cout << "* Success: calculate wing term." << std::endl;
    release_free_mem();
    profiler.stop("cal_wing_mu");
}

std::pair<ArrayDesc, matrix_m<complex<double>>> diele_func::transform_Cs2mnk(
    const int ik, const int mu,
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> &Cs_IJ,
    const int spin_filter)
{
    using global::profiler;
    if (spin_filter < -1 || spin_filter >= n_spin)
        throw std::logic_error("transform_Cs2mnk: invalid spin_filter");

    const int n_soc = meanfield_df.get_n_spinor();
    const int Mu = atomic_basis_abf_.get_i_atom(mu);
    const int mu_local = atomic_basis_abf_.get_local_index(mu, Mu);
    const int n_ao_Mu = atomic_basis_wfc_.get_atom_nb(Mu);

    ArrayDesc desc_nband_nao(blacs_h);
    desc_nband_nao.init_1b1p(n_states, n_basis, 0, 0);
    ArrayDesc desc_nband_Mu(blacs_h);
    desc_nband_Mu.init_1b1p(n_states, n_ao_Mu, 0, 0);
    ArrayDesc desc_Mu_nband(blacs_h);
    desc_Mu_nband.init_1b1p(n_ao_Mu, n_states, 0, 0);
    ArrayDesc desc_nao_nao(blacs_h);
    desc_nao_nao.init_1b1p(n_basis, n_basis, 0, 0);
    ArrayDesc desc_nband_nband(blacs_h);
    desc_nband_nband.init_1b1p(n_states, n_states, 0, 0);

    auto C_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto C_Mu_nband = init_local_mat<complex<double>>(desc_Mu_nband, MAJOR::COL);
    auto C_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);

    const auto kfrac = kfrac_band[ik];
    const std::function<complex<double>(const int &, const std::pair<int, std::array<int, 3>> &)>
        fourier = [kfrac](const int &I, const std::pair<int, std::array<int, 3>> &J_Ra)
    {
        const auto &Ra = J_Ra.second;
        Vector3<double> R_IJ(Ra[0], Ra[1], Ra[2]);
        const auto ang = (kfrac * R_IJ) * TWO_PI;
        return complex<double>{std::cos(ang), std::sin(ang)};
    };

    C_nao_nao.zero_out();
    C_Mu_nband.zero_out();
    C_nband_nband.zero_out();
    collect_block_from_IJ_storage_tensor_transform_triple(C_nao_nao, desc_nao_nao,
                                                          atomic_basis_wfc_, atomic_basis_wfc_,
                                                          fourier, Cs_IJ, Mu, mu_local);
    print_wing_cnao_probe("full_bz", mu, kfrac, desc_nao_nao, C_nao_nao);
    // if (ik == 1 && mu == 4)
    // {
    //     for (const auto IJRc : Cs_IJ)
    //     {
    //         const auto I = IJRc.first;
    //         for (const auto J_R : IJRc.second)
    //         {
    //             const auto J = J_R.first.first;
    //             const auto R = J_R.first.second;
    //             const auto &tensor = J_R.second;
    //             ofs_myid << "Cs_IJ(" << I << "," << J << "," << R[0] << R[1] << R[2]
    //                      << "):" << tensor(mu_local, 1, 2) << std::endl;
    //         }
    //     }
    //     for (int i = 0; i < n_ao_Mu; i++)
    //     {
    //         const auto i_loc = desc_nao_nao.indx_g2l_r(i);
    //         const auto j_loc = desc_nao_nao.indx_g2l_c(10);
    //         if (i_loc >= 0 && j_loc >= 0)
    //             ofs_myid << "C_nao_nao(" << i << ",10): " << C_nao_nao(i_loc, j_loc) <<
    //             std::endl;
    //     }
    // }
    //  prepare wave function BLACS
    const int spin_begin = spin_filter < 0 ? 0 : spin_filter;
    const int spin_end = spin_filter < 0 ? n_spin : spin_filter + 1;
    for (int ispin = spin_begin; ispin != spin_end; ispin++)
    {
        for (int is1 = 0; is1 != n_soc; is1++)
        {
            for (int is2 = 0; is2 != n_soc; is2++)
            {
                const auto *wfc_isp1_k = meanfield_df.find_wfc(ispin, is1, ik);
                const auto *wfc_isp2_k = meanfield_df.find_wfc(ispin, is2, ik);
                const int bad_wfc_local =
                    wfc_isp1_k == nullptr || wfc_isp2_k == nullptr ||
                    wfc_isp1_k->nr != n_states || wfc_isp1_k->nc != n_basis ||
                    wfc_isp2_k->nr != n_states || wfc_isp2_k->nc != n_basis;
                int bad_wfc = 0;
                MPI_Allreduce(&bad_wfc_local, &bad_wfc, 1, MPI_INT, MPI_MAX, comm_h.comm);
                if (bad_wfc)
                    throw LIBRPA_RUNTIME_ERROR(
                        "world-BLACS head/wing rotation requires every rank to own every "
                        "wave function");

                ComplexMatrix wfc_Mu = ComplexMatrix(n_states, n_ao_Mu);
                // #pragma omp parallel for schedule collapse(2)
                for (int n = 0; n < n_states; n++)
                {
                    for (int i = 0; i < n_ao_Mu; i++)
                    {
                        const auto i_Mu = atomic_basis_wfc_.get_global_index(Mu, i);
                        wfc_Mu(n, i) = (*wfc_isp1_k)(n, i_Mu);
                    }
                }
                // blacs_ctxt_global_h.barrier();
                auto wfc1_block = get_local_mat(wfc_Mu.c, MAJOR::ROW, desc_nband_Mu, MAJOR::COL);
                auto wfc2_block =
                    get_local_mat(wfc_isp2_k->c, MAJOR::ROW, desc_nband_nao, MAJOR::COL);
                ScalapackConnector::pgemm_f(
                    'N', 'T', n_ao_Mu, n_states, n_basis, 1.0, C_nao_nao.ptr(),
                    1 + atomic_basis_wfc_.get_part_range()[Mu], 1, desc_nao_nao.desc,
                    wfc2_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0, C_Mu_nband.ptr(), 1, 1,
                    desc_Mu_nband.desc);
                ScalapackConnector::pgemm_f('N', 'N', n_states, n_states, n_ao_Mu, 1.0,
                                            conj(wfc1_block).ptr(), 1, 1, desc_nband_Mu.desc,
                                            C_Mu_nband.ptr(), 1, 1, desc_Mu_nband.desc, 1.0,
                                            C_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                ScalapackConnector::pgemm_f('C', 'T', n_states, n_states, n_ao_Mu, 1.0,
                                            C_Mu_nband.ptr(), 1, 1, desc_Mu_nband.desc,
                                            wfc1_block.ptr(), 1, 1, desc_nband_Mu.desc, 1.0,
                                            C_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
            }
        }
    }
    // profiler.stop("scalapack_multiply");
    return std::make_pair(desc_nband_nband, C_nband_nband);
};

std::pair<ArrayDesc, matrix_m<complex<double>>> diele_func::transform_Cs2mnk_kblacs(
    const int ik, const int mu,
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> &Cs_IJ,
    const BlacsCtxtHandler &wing_blacs_h, const Vector3_Order<double> &kfrac,
    const std::vector<std::vector<const ComplexMatrix *>> *wfc_override, const int spin_filter)
{
    if (spin_filter < -1 || spin_filter >= n_spin)
        throw std::logic_error("transform_Cs2mnk_kblacs: invalid spin_filter");

    const int Mu = atomic_basis_abf_.get_i_atom(mu);
    const int mu_local = atomic_basis_abf_.get_local_index(mu, Mu);
    ArrayDesc desc_wfc_src(wing_blacs_h);
    desc_wfc_src.init(n_basis, n_states, n_basis, n_states, 0, 0);
    ArrayDesc desc_nao_nao(wing_blacs_h);
    desc_nao_nao.init_1b1p(n_basis, n_basis, 0, 0);

    auto C_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    const std::function<complex<double>(const int &, const std::pair<int, std::array<int, 3>> &)>
        fourier = [kfrac](const int &I, const std::pair<int, std::array<int, 3>> &J_Ra)
    {
        const auto &Ra = J_Ra.second;
        Vector3<double> R_IJ(Ra[0], Ra[1], Ra[2]);
        const auto ang = (kfrac * R_IJ) * TWO_PI;
        return complex<double>{std::cos(ang), std::sin(ang)};
    };

    C_nao_nao.zero_out();
    collect_block_from_IJ_storage_tensor_transform_triple(C_nao_nao, desc_nao_nao,
                                                          atomic_basis_wfc_, atomic_basis_wfc_,
                                                          fourier, Cs_IJ, Mu, mu_local);
    print_wing_cnao_probe(wfc_override == nullptr ? "full_bz" : "sym_restored", mu, kfrac,
                          desc_nao_nao, C_nao_nao);
    return rotate_Cs_nao2mnk_kblacs(ik, mu, C_nao_nao, desc_nao_nao, wing_blacs_h,
                                    desc_wfc_src, wfc_override, spin_filter);
}

std::pair<ArrayDesc, matrix_m<complex<double>>> diele_func::transform_Cs2mnk_kblacs(
    const int source_ik, const int target_id, const int mu,
    const HeadwingCsIJKMap &Cs_IJ_k, const BlacsCtxtHandler &wing_blacs_h,
    const std::vector<std::vector<const ComplexMatrix *>> *wfc_override,
    const int spin_filter)
{
    if (spin_filter < -1 || spin_filter >= n_spin)
        throw std::logic_error("transform_Cs2mnk_kblacs: invalid spin_filter");
    if (desc_wfc_kblacs_ == nullptr || !desc_wfc_kblacs_->is_initialized())
        throw LIBRPA_RUNTIME_ERROR("missing k-BLACS wave-function descriptor for head/wing");
    if (source_ik < 0 || source_ik >= static_cast<int>(kfrac_band.size()))
        throw LIBRPA_RUNTIME_ERROR("head/wing source k-point index is out of range");
    if (desc_wfc_kblacs_->ictxt() != wing_blacs_h.ictxt)
        throw LIBRPA_RUNTIME_ERROR("head/wing source descriptor uses the wrong BLACS context");

    const int Mu = atomic_basis_abf_.get_i_atom(mu);
    const int mu_local = atomic_basis_abf_.get_local_index(mu, Mu);
    ArrayDesc desc_nao_nao(wing_blacs_h);
    desc_nao_nao.init_1b1p(n_basis, n_basis, 0, 0);
    auto C_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    collect_headwing_Cs_mu_from_ijk(C_nao_nao, desc_nao_nao, atomic_basis_wfc_, Mu,
                                    mu_local, target_id, Cs_IJ_k);
    return rotate_Cs_nao2mnk_kblacs(source_ik, mu, C_nao_nao, desc_nao_nao,
                                    wing_blacs_h, *desc_wfc_kblacs_, wfc_override,
                                    spin_filter);
}

std::pair<ArrayDesc, matrix_m<complex<double>>> diele_func::rotate_Cs_nao2mnk_kblacs(
    const int source_ik, const int mu, matrix_m<complex<double>> &C_nao_nao,
    const ArrayDesc &desc_nao_nao, const BlacsCtxtHandler &wing_blacs_h,
    const ArrayDesc &desc_wfc_src,
    const std::vector<std::vector<const ComplexMatrix *>> *wfc_override,
    const int spin_filter)
{
    if (spin_filter < -1 || spin_filter >= n_spin)
        throw std::logic_error("rotate_Cs_nao2mnk_kblacs: invalid spin_filter");
    if (desc_wfc_src.m() != n_basis || desc_wfc_src.n() != n_states ||
        desc_wfc_src.ictxt() != wing_blacs_h.ictxt ||
        desc_nao_nao.ictxt() != wing_blacs_h.ictxt)
        throw LIBRPA_RUNTIME_ERROR("head/wing rotation descriptors are inconsistent");

    const int n_soc = meanfield_df.get_n_spinor();
    const int Mu = atomic_basis_abf_.get_i_atom(mu);
    const int n_ao_Mu = atomic_basis_wfc_.get_atom_nb(Mu);
    const int expected_block_nao =
        get_capped_blacs_block_size(n_basis, wfc_gemm_block_size_opt, wing_blacs_h);
    const int block_Mu =
        get_capped_blacs_block_size(n_ao_Mu, wfc_gemm_block_size_opt, wing_blacs_h);
    const int expected_block_nband =
        get_capped_blacs_block_size(n_states, wfc_gemm_block_size_opt, wing_blacs_h);
    const bool wfc_is_permanent_opt =
        desc_wfc_src.mb() == expected_block_nao &&
        desc_wfc_src.nb() == expected_block_nband;
    const int block_nao =
        wfc_is_permanent_opt ? desc_wfc_src.mb() : expected_block_nao;
    const int block_nband =
        wfc_is_permanent_opt ? desc_wfc_src.nb() : expected_block_nband;

    ArrayDesc desc_Mu_nao_opt(wing_blacs_h);
    desc_Mu_nao_opt.init(n_ao_Mu, n_basis, block_Mu, block_nao, 0, 0);
    ArrayDesc desc_nao_nband_opt(wing_blacs_h);
    desc_nao_nband_opt.init(n_basis, n_states, block_nao, block_nband, 0, 0);
    ArrayDesc desc_Mu_nband_opt(wing_blacs_h);
    desc_Mu_nband_opt.init(n_ao_Mu, n_states, block_Mu, block_nband, 0, 0);
    ArrayDesc desc_nband_nband_opt(wing_blacs_h);
    desc_nband_nband_opt.init(n_states, n_states, block_nband, block_nband, 0, 0);

    auto C_Mu_nao_opt = init_local_mat<complex<double>>(desc_Mu_nao_opt, MAJOR::COL);
    auto C_Mu_nband_opt = init_local_mat<complex<double>>(desc_Mu_nband_opt, MAJOR::COL);
    auto C_nband_nband_opt = init_local_mat<complex<double>>(desc_nband_nband_opt, MAJOR::COL);
    auto wfc_nao_nband_opt = init_local_mat<complex<double>>(desc_nao_nband_opt, MAJOR::COL);
    auto wfc_Mu_nband_opt = init_local_mat<complex<double>>(desc_Mu_nband_opt, MAJOR::COL);
    C_Mu_nband_opt.zero_out();
    C_nband_nband_opt.zero_out();

    ScalapackConnector::pgemr2d_f(
        n_ao_Mu, n_basis, C_nao_nao.ptr(),
        1 + atomic_basis_wfc_.get_part_range()[Mu], 1, desc_nao_nao.desc,
        C_Mu_nao_opt.ptr(), 1, 1, desc_Mu_nao_opt.desc, wing_blacs_h.ictxt);

    const auto get_wfc = [&](const int ispin, const int ispinor) -> const ComplexMatrix *
    {
        if (wfc_override)
        {
            return (*wfc_override).at(ispin).at(ispinor);
        }
        return meanfield_df.find_wfc(ispin, ispinor, source_ik);
    };

    std::vector<complex<double>> dummy_wfc(1, {0.0, 0.0});
    const std::size_t wfc_size_local =
        static_cast<std::size_t>(desc_wfc_src.m_loc()) * desc_wfc_src.n_loc();
    const int spin_begin = spin_filter < 0 ? 0 : spin_filter;
    const int spin_end = spin_filter < 0 ? n_spin : spin_filter + 1;
    for (int ispin = spin_begin; ispin != spin_end; ispin++)
    {
        for (int is1 = 0; is1 != n_soc; is1++)
        {
            for (int is2 = 0; is2 != n_soc; is2++)
            {
                const ComplexMatrix *wfc_isp1_k = get_wfc(ispin, is1);
                const ComplexMatrix *wfc_isp2_k = get_wfc(ispin, is2);

                const int bad_source_local =
                    wfc_size_local > 0 &&
                    (wfc_isp1_k == nullptr || wfc_isp2_k == nullptr ||
                     static_cast<std::size_t>(wfc_isp1_k->size) != wfc_size_local ||
                     static_cast<std::size_t>(wfc_isp2_k->size) != wfc_size_local);
                int bad_source = 0;
                MPI_Allreduce(&bad_source_local, &bad_source, 1, MPI_INT, MPI_MAX,
                              wing_blacs_h.comm());
                if (bad_source)
                    throw LIBRPA_RUNTIME_ERROR(
                        "transform_Cs2mnk_kblacs: missing or inconsistent source eigenvector");

                const complex<double> *wfc2_src =
                    wfc_isp2_k == nullptr || wfc_isp2_k->c == nullptr
                        ? dummy_wfc.data()
                        : wfc_isp2_k->c;
                const complex<double> *wfc2_compute = wfc2_src;
                const int *desc_wfc_compute = desc_wfc_src.desc;
                if (!wfc_is_permanent_opt)
                {
                    wfc_nao_nband_opt.zero_out();
                    ScalapackConnector::pgemr2d_f(
                        n_basis, n_states, wfc2_src, 1, 1, desc_wfc_src.desc,
                        wfc_nao_nband_opt.ptr(), 1, 1, desc_nao_nband_opt.desc,
                        wing_blacs_h.ictxt);
                    wfc2_compute = wfc_nao_nband_opt.ptr();
                    desc_wfc_compute = desc_nao_nband_opt.desc;
                }

                C_Mu_nband_opt.zero_out();
                ScalapackConnector::pgemm_f(
                    'N', 'N', n_ao_Mu, n_states, n_basis, 1.0, C_Mu_nao_opt.ptr(), 1, 1,
                    desc_Mu_nao_opt.desc, wfc2_compute, 1, 1,
                    desc_wfc_compute, 0.0, C_Mu_nband_opt.ptr(), 1, 1,
                    desc_Mu_nband_opt.desc);

                const complex<double> *wfc1_src =
                    wfc_isp1_k == nullptr || wfc_isp1_k->c == nullptr
                        ? dummy_wfc.data()
                        : wfc_isp1_k->c;
                wfc_Mu_nband_opt.zero_out();
                ScalapackConnector::pgemr2d_f(n_ao_Mu, n_states, wfc1_src,
                                              1 + atomic_basis_wfc_.get_part_range()[Mu], 1,
                                              desc_wfc_src.desc, wfc_Mu_nband_opt.ptr(), 1, 1,
                                              desc_Mu_nband_opt.desc, wing_blacs_h.ictxt);

                ScalapackConnector::pgemm_f('C', 'N', n_states, n_states, n_ao_Mu, 1.0,
                                            wfc_Mu_nband_opt.ptr(), 1, 1,
                                            desc_Mu_nband_opt.desc, C_Mu_nband_opt.ptr(), 1, 1,
                                            desc_Mu_nband_opt.desc, 1.0,
                                            C_nband_nband_opt.ptr(), 1, 1,
                                            desc_nband_nband_opt.desc);
                ScalapackConnector::pgemm_f('C', 'N', n_states, n_states, n_ao_Mu, 1.0,
                                            C_Mu_nband_opt.ptr(), 1, 1,
                                            desc_Mu_nband_opt.desc, wfc_Mu_nband_opt.ptr(), 1, 1,
                                            desc_Mu_nband_opt.desc, 1.0,
                                            C_nband_nband_opt.ptr(), 1, 1,
                                            desc_nband_nband_opt.desc);
            }
        }
    }

    return std::make_pair(desc_nband_nband_opt, C_nband_nband_opt);
}

std::complex<double> diele_func::compute_wing(const int alpha, const int iomega, const int mu,
                                              const int ik, const int ispin,
                                              const ArrayDesc &desc_nband_nband,
                                              const matrix_m<complex<double>> &C_nband_nband)
{
    const auto &velocity = this->velocity_;
    auto &eigenvalues = this->meanfield_df.get_eigenvals();

    double omega_ev = this->omega[iomega];  // * HA2EV;
    std::complex<double> wing_term = 0.0;

    bool use_soc = meanfield_df.get_n_spinor() > 1;

    auto &wg = this->meanfield_df.get_weight()[ispin];
    for (int iocc = 0; iocc != n_states; iocc++)
    {
        for (int iunocc = 0; iunocc != n_states; iunocc++)
        {
            double egap =
                (eigenvalues[ispin](ik, iunocc) - eigenvalues[ispin](ik, iocc));  // * HA2EV;
            if (iocc < iunocc)
            {
                double factor1;
                double factor2;
                if (use_soc)
                {
                    // NOTE: wg is matrix(n_kpoints, n_states); use wg(ik, ib) so the
                    // k-dependence of the occupation is honored. The factor `* nk` recovers
                    // the occupation count (wg was divided by n_kpoints at read time).
                    factor1 = wg(ik, iocc) * (1.0 - wg(ik, iunocc) * nk);
                    factor2 = wg(ik, iunocc) * (1.0 - wg(ik, iocc) * nk);
                }
                else
                {
                    factor1 = wg(ik, iocc) / 2 * n_spin * (1.0 - wg(ik, iunocc) / 2 * n_spin * nk);
                    factor2 = wg(ik, iunocc) / 2 * n_spin * (1.0 - wg(ik, iocc) / 2 * n_spin * nk);
                }
                if (factor1 > 1.e-8)
                {
                    auto loc_m = desc_nband_nband.indx_g2l_r(iocc);
                    auto loc_n = desc_nband_nband.indx_g2l_c(iunocc);
                    if (loc_m < 0 || loc_n < 0) continue;
                    auto tmp = C_nband_nband(loc_m, loc_n);
                    wing_term += factor1 * conj(tmp * velocity[ispin][ik][alpha](iunocc, iocc)) /
                                 (omega_ev * omega_ev + egap * egap);
                }
                if (factor2 > 1.e-8)
                {
                    auto loc_m = desc_nband_nband.indx_g2l_r(iocc);
                    auto loc_n = desc_nband_nband.indx_g2l_c(iunocc);
                    if (loc_m < 0 || loc_n < 0) continue;
                    auto tmp = C_nband_nband(loc_m, loc_n);
                    // for metal
                    wing_term += factor2 * tmp * velocity[ispin][ik][alpha](iunocc, iocc) /
                                 (omega_ev * omega_ev + egap * egap);
                }

                /*if (Params::debug)
                {
                    if (alpha == 0 && iomega == 0 && mu == 0)
                    {
                        std::complex<double> test =
                conj(Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]] *
                                                         velocity[ispin][ik][alpha](iunocc,
                iocc)) / (omega_ev * omega_ev + egap * egap); if (iocc == 0 && iunocc == 10)
                        {
                            std::cout << "C,p: " << Ctri_mn[mu][iocc][iunocc][kfrac_band[ik]] <<
                ","
                                      << velocity[ispin][ik][alpha](iunocc, iocc) << std::endl;
                        }
                        test_tot += test;
                    }
                }*/
            }
        }
    }

    return wing_term;
};

void diele_func::wing_mu_to_lambda(matrix_m<std::complex<double>> &sqrtveig_blacs,
                                   ArrayDesc &desc_nabf_nabf_opt,
                                   const std::size_t n_nonsingular_in)
{
    using global::profiler;

    profiler.start("cal_wing");
    this->n_nonsingular = n_nonsingular_in;
    if (this->n_nonsingular < 2)
        throw std::logic_error("Head/wing Coulomb subspace has no regular channels");
    int n_lambda = this->n_nonsingular - 1;
    // Keep the contracted n_abf dimension aligned with sqrt(V)'s column
    // distribution.  A unit block is sufficient for the three Cartesian
    // columns and keeps all subsequent thin PGEMMs mutually aligned.
    ArrayDesc desc_wing_mu(blacs_h);
    desc_wing_mu.init(n_abf, 3, desc_nabf_nabf_opt.nb(), 1, 0, 0);
    ArrayDesc desc_wing(blacs_h);
    desc_wing.init_square_blk(n_nonsingular - 1, 3, 0, 0);
    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(n_nonsingular - 1, n_nonsingular - 1, 0, 0);
    // opt descriptor for wing
    ArrayDesc desc_wing_opt(blacs_h);
    desc_wing_opt.init(n_nonsingular - 1, 3, desc_body.mb(), desc_wing.nb(), 0, 0);
    const int n_omegas = this->omega.size();
    this->wing.clear();
    this->wing.resize(n_omegas);

    for (int iomega = 0; iomega != n_omegas; iomega++)
    {
        auto &wing_tmp = this->wing.at(iomega);
        wing_tmp = init_local_mat<complex<double>>(desc_wing_opt, MAJOR::COL);
        // TODO: reconstruct wing_mu
        auto wing_mu_tmp = init_local_mat<complex<double>>(desc_wing_mu, MAJOR::COL);
        for (int alpha = 0; alpha != 3; alpha++)
        {
            auto loc_alpha = desc_wing_mu.indx_g2l_c(alpha);
            if (loc_alpha < 0) continue;
            for (int mu = 0; mu != n_abf; mu++)
            {
                auto loc_mu = desc_wing_mu.indx_g2l_r(mu);
                if (loc_mu < 0) continue;
                wing_mu_tmp(loc_mu, loc_alpha) = this->wing_mu.at(iomega)(mu, alpha);
            }
        }
        // this->wing.at(iomega)(lambda, alpha) +=
        // conj(sqrtveig_blacs(mu, lambda)) * this->wing_mu.at(iomega)(mu, alpha);
        // drop the first column of sqrtveig_blacs, the largest eigenvalue
        ScalapackConnector::pgemm_f('C', 'N', n_lambda, 3, n_abf, 1.0, sqrtveig_blacs.ptr(), 1, 2,
                                    desc_nabf_nabf_opt.desc, wing_mu_tmp.ptr(), 1, 1,
                                    desc_wing_mu.desc, 0.0, wing_tmp.ptr(), 1, 1,
                                    desc_wing_opt.desc);
    }

    if (!this->wing.empty())
    {
        const auto &wing0 = this->wing.at(0);
        ComplexMatrix wing0_global(n_lambda, 3);
        for (int ilambda = 0; ilambda != wing0.nr(); ++ilambda)
        {
            const int lambda = desc_wing_opt.indx_l2g_r(ilambda);
            if (lambda < 0) continue;
            for (int ialpha = 0; ialpha != wing0.nc(); ++ialpha)
            {
                const int alpha = desc_wing_opt.indx_l2g_c(ialpha);
                if (alpha >= 0 && alpha < 3) wing0_global(lambda, alpha) = wing0(ilambda, ialpha);
            }
        }
        if (wing0_global.size > std::numeric_limits<int>::max())
            throw LIBRPA_RUNTIME_ERROR("wing Cartesian Gram buffer is too large for MPI_Allreduce");
        MPI_Allreduce(MPI_IN_PLACE, wing0_global.c, wing0_global.size, MPI_CXX_DOUBLE_COMPLEX,
                      MPI_SUM, comm_h.comm);
        const auto wing0_gram = compute_wing_cartesian_gram(wing0_global);

        double max_abs_real_local = 0.0;
        double max_abs_imag_local = 0.0;
        double max_abs_value_local = 0.0;
        for (int i = 0; i != wing0.nr(); ++i)
        {
            for (int j = 0; j != wing0.nc(); ++j)
            {
                const auto value = wing0(i, j);
                max_abs_real_local = std::max(max_abs_real_local, std::abs(value.real()));
                max_abs_imag_local = std::max(max_abs_imag_local, std::abs(value.imag()));
                max_abs_value_local = std::max(max_abs_value_local, std::abs(value));
            }
        }
        double max_abs_real = 0.0;
        double max_abs_imag = 0.0;
        double max_abs_value = 0.0;
        MPI_Allreduce(&max_abs_real_local, &max_abs_real, 1, MPI_DOUBLE, MPI_MAX, comm_h.comm);
        MPI_Allreduce(&max_abs_imag_local, &max_abs_imag, 1, MPI_DOUBLE, MPI_MAX, comm_h.comm);
        MPI_Allreduce(&max_abs_value_local, &max_abs_value, 1, MPI_DOUBLE, MPI_MAX, comm_h.comm);
        const double real_over_abs = max_abs_value > 0.0 ? max_abs_real / max_abs_value : 0.0;
        if (comm_h.is_root())
        {
            global::lib_printf(
                "Wing_lambda diagnostics (iomega=0): max_abs_real=%15.8e "
                "max_abs_imag=%15.8e max_abs_value=%15.8e real_over_abs=%15.8e\n",
                max_abs_real, max_abs_imag, max_abs_value, real_over_abs);
            global::lib_printf("Wing_lambda Gram (iomega=0, rows alpha, columns beta):\n");
            for (int alpha = 0; alpha != 3; ++alpha)
            {
                global::lib_printf("(%15.8e,%15.8e) (%15.8e,%15.8e) (%15.8e,%15.8e)\n",
                                   wing0_gram.at(alpha).at(0).real(),
                                   wing0_gram.at(alpha).at(0).imag(),
                                   wing0_gram.at(alpha).at(1).real(),
                                   wing0_gram.at(alpha).at(1).imag(),
                                   wing0_gram.at(alpha).at(2).real(),
                                   wing0_gram.at(alpha).at(2).imag());
            }
        }
        if (debug)
        {
            if (comm_h.is_root())
            {
                std::cout << "First wing_lambda rows at iomega=0 (lambda, x_re, x_im, y_re, "
                             "y_im, z_re, z_im):"
                          << std::endl;
            }
            const int n_sample = std::min(n_lambda, 8);
            for (int ilambda = 0; ilambda != n_sample; ++ilambda)
            {
                std::array<std::complex<double>, 3> row{};
                for (int alpha = 0; alpha != 3; ++alpha)
                {
                    const int loc_lambda = desc_wing_opt.indx_g2l_r(ilambda);
                    const int loc_alpha = desc_wing_opt.indx_g2l_c(alpha);
                    std::complex<double> value = 0.0;
                    if (loc_lambda >= 0 && loc_alpha >= 0)
                        value = wing0(loc_lambda, loc_alpha);
                    MPI_Allreduce(&value, &row[alpha], 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM,
                                  comm_h.comm);
                }
                if (comm_h.is_root())
                {
                    global::lib_printf(
                        "%4d %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e\n",
                        ilambda, row[0].real(), row[0].imag(), row[1].real(), row[1].imag(),
                        row[2].real(), row[2].imag());
                }
            }
        }
    }

    this->wing_mu.clear();
    profiler.stop("cal_wing");
};

// // real double diagonalization
// // Note complex diagonalization conserves symmetry much better
// void diele_func::get_Xv_real(double vq_threshold, const librpa_int::atpair_k_cplx_mat_t &Vq)
// {
//     using namespace librpa_int;
//     using RI::Tensor;
//     using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
//     using librpa_int::global::mpi_comm_global_h;
//
//     this->Coul_vector.clear();
//     this->Coul_value.clear();
//     const double CONE = 1.0;
//     std::array<double, 3> qa = {0.0, 0.0, 0.0};
//     Vector3_Order<double> q = {0.0, 0.0, 0.0};
//     size_t n_singular;
//     vec<double> eigenvalues(n_abf);
//
//     const auto &comm_h = mpi_comm_global_h;
//
//     comm_h.barrier();
//     BlacsCtxtHandler blacs_h(comm_h.comm);
//
//     ArrayDesc desc_nabf_nabf(blacs_h);
//     desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
//     const auto set_IJ_nabf_nabf =
//         get_necessary_IJ_from_block_2D_sy('U', atomic_basis_abf_, desc_nabf_nabf);
//     const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
//     auto coul_eigen_block = init_local_mat<double>(desc_nabf_nabf, MAJOR::COL);
//     auto coulwc_block = init_local_mat<double>(desc_nabf_nabf, MAJOR::COL);
//     coulwc_block.zero_out();
//     std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<double>>>
//         couleps_libri;
//     const int natom = atomic_basis_abf_.n_atoms;
//     const auto atpair_local = dispatch_upper_triangular_tasks(
//         natom, blacs_h.myid, blacs_h.nprows, blacs_h.npcols,
//         blacs_h.myprow, blacs_h.mypcol);
//     for (const auto &Mu_Nu : atpair_local)
//     {
//         const auto Mu = Mu_Nu.first;
//         const auto Nu = Mu_Nu.second;
//         // ofs_myid << "Mu " << Mu << " Nu " << Nu << endl;
//         if (Vq.count(Mu) == 0 || Vq.at(Mu).count(Nu) == 0 ||
//             Vq.at(Mu).at(Nu).count(q) == 0)
//             continue;
//         auto Vq_cpl = *(Vq.at(Mu).at(Nu).at(q));
//         // if (Vq.count(Mu) == 0 || Vq.at(Mu).count(Nu) == 0 || Vq.at(Mu).at(Nu).count(q) == 0)
//         //     continue;
//         // auto Vq_cpl = *(Vq.at(Mu).at(Nu).at(q));
//         const auto &Vq0 = std::make_shared<matrix>(Vq_cpl.real());
//         // const auto &Vq0 = Vq.at(Mu).at(Nu).at(q);
//         const auto n_mu = atomic_basis_abf_.get_atom_nb(Mu);
//         const auto n_nu = atomic_basis_abf_.get_atom_nb(Nu);
//         std::valarray<double> Vq_va(Vq0->c, Vq0->size);
//         auto pvq = std::make_shared<std::valarray<double>>();
//         *pvq = Vq_va;
//         couleps_libri[Mu][{Nu, qa}] = RI::Tensor<double>({n_mu, n_nu}, pvq);
//     }
//     const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
//         mpi_comm_global_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
//     collect_block_from_ALL_IJ_Tensor(coulwc_block, desc_nabf_nabf, atomic_basis_abf_, qa,
//                                      true, CONE, IJq_coul, MAJOR::ROW);
//     power_hemat_blacs_real(coulwc_block, desc_nabf_nabf, coul_eigen_block, desc_nabf_nabf,
//                            n_singular, eigenvalues.c, 1.0, vq_threshold);
//     this->n_nonsingular = n_abf - n_singular;
//     for (int iv = 1; iv != n_nonsingular; iv++)
//     {
//         // Here eigen solved by Scalapack is ascending order,
//         // however, what we want is descending order.
//         this->Coul_value.push_back(eigenvalues.c[iv] + 0.0);  // throw away the largest one
//         std::vector<std::complex<double>> newRow;
//
//         for (int jabf = 0; jabf != n_abf; jabf++)
//         {
//             newRow.push_back(coul_eigen_block(jabf, iv));
//         }
//         this->Coul_vector.push_back(newRow);
//     }
//
//     if (mpi_comm_global_h.is_root())
//     {
//         std::cout << "The largest/smallest eigenvalue of Coulomb matrix(non-singular): "
//                   << this->Coul_value.front() << ", " << this->Coul_value.back() << std::endl;
//         std::cout << "The 1st/2nd/3rd/-1th eigenvalue of Coulomb matrix(Full): " <<
//         eigenvalues.c[0]
//                   << ", " << eigenvalues.c[1] << ", " << eigenvalues.c[2] << ", "
//                   << eigenvalues.c[n_abf - 1] << std::endl;
//         std::cout << "Dim of eigenvectors: " << coul_eigen_block.nr() << ", "
//                   << coul_eigen_block.nc() << std::endl;
//     }
//     /*std::cout << "Coulomb vector: lambda=-1" << std::endl;
//     for (int j = 0; j != n_abf; j++)
//     {
//         std::cout << j << "," << coul_eigen_block(j, 0) << std::endl;
//     }
//     std::cout << "Coulomb vector: lambda=0" << std::endl;
//     for (int j = 0; j != n_abf; j++)
//     {
//         std::cout << j << "," << Coul_vector[0][j] << std::endl;
//     }
//     std::cout << "Coulomb vector: lambda=1" << std::endl;
//     for (int j = 0; j != n_abf; j++)
//     {
//         std::cout << j << "," << Coul_vector[1][j] << std::endl;
//     }*/
//     if (mpi_comm_global_h.is_root())
//         std::cout << "* Success: diagonalize Coulomb matrix in the ABFs repre." << std::endl;
// };

void diele_func::get_Xv_cpl(double coulomb_eigen_threshold,
                            const librpa_int::atpair_k_cplx_mat_t &Vq)
{
    using global::profiler;

    profiler.start("get_eigenvector_of_Coulomb_matrix");
    const complex<double> CONE{1.0, 0.0};
    std::array<double, 3> qa = {0.0, 0.0, 0.0};
    Vector3_Order<double> q = {0.0, 0.0, 0.0};
    size_t n_singular;
    librpa_int::vec<double> eigenvalues(n_abf);

    comm_h.barrier();

    ArrayDesc desc_nabf_nabf(blacs_h);
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    const auto set_IJ_nabf_nabf =
        get_necessary_IJ_from_block_2D_sy('U', atomic_basis_abf_, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto coul_eigen_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto coulwc_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    coulwc_block.zero_out();
    std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
        couleps_libri;
    const int natom = atomic_basis_abf_.n_atoms;
    const auto atpair_local = dispatch_upper_triangular_tasks(
        natom, blacs_h.myid, blacs_h.nprows, blacs_h.npcols, blacs_h.myprow, blacs_h.mypcol);
    for (const auto &Mu_Nu : atpair_local)
    {
        const auto Mu = Mu_Nu.first;
        const auto Nu = Mu_Nu.second;
        // ofs_myid << "Mu " << Mu << " Nu " << Nu << endl;
        if (Vq.count(Mu) == 0 || Vq.at(Mu).count(Nu) == 0 || Vq.at(Mu).at(Nu).count(q) == 0)
            continue;
        const auto &Vq0 = Vq.at(Mu).at(Nu).at(q);
        const auto n_mu = atomic_basis_abf_.get_atom_nb(Mu);
        const auto n_nu = atomic_basis_abf_.get_atom_nb(Nu);
        std::valarray<complex<double>> Vq_va(Vq0->c, Vq0->size);
        auto pvq = std::make_shared<std::valarray<complex<double>>>();
        *pvq = Vq_va;
        couleps_libri[Mu][{Nu, qa}] = RI::Tensor<complex<double>>({n_mu, n_nu}, pvq);
    }
    const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        comm_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
    collect_block_from_ALL_IJ_Tensor(coulwc_block, desc_nabf_nabf, atomic_basis_abf_, qa,
                                     true, CONE, IJq_coul, MAJOR::ROW);
    // Gamma Coulomb is real; keep this on the same eigensolver path used by epsilon.
    power_hemat_blacs_real(coulwc_block, desc_nabf_nabf, coul_eigen_block, desc_nabf_nabf,
                           n_singular, eigenvalues.c, 0.5, coulomb_eigen_threshold);
    this->n_nonsingular = n_abf - n_singular;

    // for (int iv = 1; iv != n_nonsingular; iv++)
    //{
    //  Here eigen solved by Scalapack is ascending order,
    //  however, what we want is descending order.
    //  this->Coul_value.push_back(eigenvalues.c[iv]);  // throw away the largest one
    //    std::vector<std::complex<double>> newRow;

    //    for (int jabf = 0; jabf != n_abf; jabf++)
    //   {
    //     newRow.push_back(coul_eigen_block(jabf, iv));
    //}
    // newRow.clear();
    //}
    if (comm_h.is_root())
    {
        std::cout << "n_singular: " << n_singular << std::endl;
        std::cout << "The largest/smallest eigenvalue of Coulomb matrix(non-singular): "
                  << eigenvalues.c[1] << ", " << eigenvalues.c[n_nonsingular - 1] << std::endl;
        std::cout << "The 1st/2nd/3rd/-1th eigenvalue of Coulomb matrix(Full): " << eigenvalues.c[0]
                  << ", " << eigenvalues.c[1] << ", " << eigenvalues.c[2] << ", "
                  << eigenvalues.c[n_abf - 1] << std::endl;
        // std::cout << "Dim of eigenvectors: " << coul_eigen_block.dataobj.nr() << ", "
        //           << coul_eigen_block.dataobj.nc() << std::endl;

        std::cout << "* Success: diagonalize Coulomb matrix in the ABFs repre.\n";
    }
    profiler.stop("get_eigenvector_of_Coulomb_matrix");
};

std::vector<double> diele_func::get_head_vec()
{
    const int n_omegas = this->omega.size();
    std::vector<double> head_vec(n_omegas, 0.0);
    for (int iomega = 0; iomega != n_omegas; iomega++)
    {
        std::complex<double> df = 0;
        for (int alpha = 0; alpha != 3; alpha++)
        {
            df += this->head.at(iomega)(alpha, alpha);
        }
        head_vec[iomega] = df.real() / 3.0;
    }
    return head_vec;
};

void diele_func::test_head()
{
    using global::lib_printf;

    const int n_omegas = this->omega.size();
    if (comm_h.is_root())
    {
        std::cout << "Freqency node & Head of dielectric function(Re, Im): " << std::endl;
        for (int iomega = 0; iomega != n_omegas; iomega++)
        {
            std::complex<double> df = 0;
            for (int alpha = 0; alpha != 3; alpha++)
            {
                df += this->head.at(iomega)(alpha, alpha);
            }
            lib_printf("%2d %15.8f %15.8f %15.8f\n", iomega, this->omega[iomega], df.real() / 3.0,
                       df.imag() / 3.0);
        }
        std::cout << "The first freqency head tensor: " << std::endl;
        for (int alpha = 0; alpha != 3; alpha++)
        {
            const auto &c0 = this->head.at(0)(alpha, 0);
            const auto &c1 = this->head.at(0)(alpha, 1);
            const auto &c2 = this->head.at(0)(alpha, 2);

            lib_printf("(%8.4f, %8.4f)  (%8.4f, %8.4f)  (%8.4f, %8.4f)\n", c0.real(), c0.imag(),
                       c1.real(), c1.imag(), c2.real(), c2.imag());
        }
        std::complex<double> trace = 0.0;
        for (int alpha = 0; alpha != 3; alpha++)
        {
            trace += this->head.at(0)(alpha, alpha);
        }
        const auto trace_over_3 = trace / 3.0;
        double max_abs_offdiag = 0.0;
        double max_abs_diag_delta = 0.0;
        double max_abs_imag = 0.0;
        for (int alpha = 0; alpha != 3; alpha++)
        {
            for (int beta = 0; beta != 3; beta++)
            {
                const auto value = this->head.at(0)(alpha, beta);
                max_abs_imag = std::max(max_abs_imag, std::abs(value.imag()));
                if (alpha == beta)
                    max_abs_diag_delta = std::max(max_abs_diag_delta, std::abs(value - trace_over_3));
                else
                    max_abs_offdiag = std::max(max_abs_offdiag, std::abs(value));
            }
        }
        lib_printf(
            "Head tensor diagnostics (iomega=0): trace_over_3=(%15.8f,%15.8f) "
            "max_abs_diag_delta=%15.8e max_abs_offdiag=%15.8e max_abs_imag=%15.8e\n",
            trace_over_3.real(), trace_over_3.imag(), max_abs_diag_delta, max_abs_offdiag,
            max_abs_imag);
    }
    // std::exit(0);
};

void diele_func::test_wing()
{
    using global::lib_printf;
    if (comm_h.is_root())
    {
        if (this->wing_mu.empty())
        {
            if (debug) std::cout << "Wing_mu diagnostics unavailable: wing_mu is empty." << std::endl;
            return;
        }
        double max_abs_real = 0.0;
        double max_abs_imag = 0.0;
        double max_abs_value = 0.0;
        for (int mu = 0; mu != n_abf; mu++)
        {
            for (int alpha = 0; alpha != 3; alpha++)
            {
                const auto value = this->wing_mu.at(0)(mu, alpha);
                max_abs_real = std::max(max_abs_real, std::abs(value.real()));
                max_abs_imag = std::max(max_abs_imag, std::abs(value.imag()));
                max_abs_value = std::max(max_abs_value, std::abs(value));
            }
        }
        const double real_over_abs = max_abs_value > 0.0 ? max_abs_real / max_abs_value : 0.0;
        lib_printf(
            "Wing_mu diagnostics (iomega=0): max_abs_real=%15.8e max_abs_imag=%15.8e "
            "max_abs_value=%15.8e real_over_abs=%15.8e\n",
            max_abs_real, max_abs_imag, max_abs_value, real_over_abs);
        ComplexMatrix wing_mu0(n_abf, 3);
        for (int mu = 0; mu != n_abf; ++mu)
            for (int alpha = 0; alpha != 3; ++alpha)
                wing_mu0(mu, alpha) = this->wing_mu.at(0)(mu, alpha);
        const auto wing_mu0_gram = compute_wing_cartesian_gram(wing_mu0);
        lib_printf("Wing_mu Gram (iomega=0, rows alpha, columns beta):\n");
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            lib_printf("(%15.8e,%15.8e) (%15.8e,%15.8e) (%15.8e,%15.8e)\n",
                       wing_mu0_gram.at(alpha).at(0).real(),
                       wing_mu0_gram.at(alpha).at(0).imag(),
                       wing_mu0_gram.at(alpha).at(1).real(),
                       wing_mu0_gram.at(alpha).at(1).imag(),
                       wing_mu0_gram.at(alpha).at(2).real(),
                       wing_mu0_gram.at(alpha).at(2).imag());
        }
        if (debug)
        {
            std::cout << "Index of abfs & wing(iomega=0, z) of dielectric function(Re, Im): "
                      << std::endl;
            for (int mu = 0; mu != n_abf; mu++)
            {
                const auto df = this->wing_mu.at(0)(mu, 2);
                lib_printf("%2d %15.8f %15.8f\n", mu, df.real(), df.imag());
            }
        }
    }
    // if (mpi_comm_global_h.myid == 1)
    // {
    //     std::cout << "id=1 Index of abfs & wing(iomega=0, z) of dielectric function(Re, Im): "
    //               << std::endl;
    //     for (int mu = 0; mu != n_abf; mu++)
    //     {
    //         std::complex<double> df = 0;
    //         // z direction
    //         df = this->wing_mu.at(0)(mu, 2);

    //         lib_printf("%2d %15.8f %15.8f\n", mu, df.real(), df.imag());
    //     }
    // }
    // std::exit(0);
};

void invert_headwing_body_with_identity_solve(
    matrix_m<std::complex<double>> &body, ArrayDesc &desc_body,
    const BlacsCtxtHandler &blacs_h, const bool use_cholesky, const bool use_device)
{
    using global::profiler;

    if (!desc_body.is_initialized() || desc_body.m() != desc_body.n() ||
        body.nr() != desc_body.m_loc() || body.nc() != desc_body.n_loc())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Head/wing body inverse expects a square distributed matrix matching its descriptor");
    }
    if (desc_body.ictxt() != blacs_h.ictxt)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Head/wing body inverse descriptor and BLACS context do not match");
    }

    const int n_body = desc_body.m();
    auto body_factor = body.copy();
    body.zero_out();
    for (int i = 0; i != n_body; ++i)
    {
        const int ilo = desc_body.indx_g2l_r(i);
        const int jlo = desc_body.indx_g2l_c(i);
        if (ilo >= 0 && jlo >= 0) body(ilo, jlo) = 1.0;
    }

    int info = 0;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (use_device)
    {
        if (blacs_h.ddla_handle == nullptr)
        {
            throw LIBRPA_RUNTIME_ERROR(
                "Head/wing body inverse requested DDLA before its BLACS handle was initialized");
        }
        desc_body.set_ddla_desc(blacs_h.ddla_handle);

        const std::size_t local_size = body.size();
        const std::size_t allocation_size = std::max<std::size_t>(local_size, 1);
        std::complex<double> *d_body_factor = nullptr;
        std::complex<double> *d_body_inverse = nullptr;
        ddla::DEVICE_CHECK(ddla::deviceMallocAsync(
            reinterpret_cast<void **>(&d_body_factor),
            allocation_size * sizeof(std::complex<double>), blacs_h.ddla_handle->stream));
        ddla::DEVICE_CHECK(ddla::deviceMallocAsync(
            reinterpret_cast<void **>(&d_body_inverse),
            allocation_size * sizeof(std::complex<double>), blacs_h.ddla_handle->stream));

        const auto release_device_buffers = [&]() {
            ddla::DEVICE_CHECK(
                ddla::deviceFreeAsync(d_body_factor, blacs_h.ddla_handle->stream));
            ddla::DEVICE_CHECK(
                ddla::deviceFreeAsync(d_body_inverse, blacs_h.ddla_handle->stream));
        };

        try
        {
            profiler.start("headwing_body_inverse_transfer");
            if (local_size > 0)
            {
                ddla::DEVICE_CHECK(deviceMemcpyAsync(
                    d_body_factor, body_factor.ptr(),
                    local_size * sizeof(std::complex<double>), ddla::deviceMemcpyHostToDevice,
                    blacs_h.ddla_handle->stream));
                ddla::DEVICE_CHECK(deviceMemcpyAsync(
                    d_body_inverse, body.ptr(), local_size * sizeof(std::complex<double>),
                    ddla::deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
            }
            ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(blacs_h.ddla_handle->stream));
            profiler.stop("headwing_body_inverse_transfer");

            profiler.start("headwing_body_inverse_trf_trs");
            if (use_cholesky)
            {
                LaConnector::pposv('L', 'L', 'N', n_body, n_body, d_body_factor, 1, 1,
                                   desc_body, d_body_inverse, 1, 1, desc_body, info);
            }
            else
            {
                LaConnector::pgesv(n_body, n_body, d_body_factor, 1, 1, desc_body,
                                   d_body_inverse, 1, 1, desc_body, info);
            }
            profiler.stop("headwing_body_inverse_trf_trs");

            if (info != 0)
            {
                std::ostringstream oss;
                oss << "Head/wing body "
                    << (use_cholesky ? "Cholesky" : "LU")
                    << " identity solve failed with info=" << info;
                throw LIBRPA_RUNTIME_ERROR(oss.str());
            }

            profiler.start("headwing_body_inverse_transfer");
            if (local_size > 0)
            {
                ddla::DEVICE_CHECK(deviceMemcpyAsync(
                    body.ptr(), d_body_inverse, local_size * sizeof(std::complex<double>),
                    ddla::deviceMemcpyDeviceToHost, blacs_h.ddla_handle->stream));
            }
            ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(blacs_h.ddla_handle->stream));
            profiler.stop("headwing_body_inverse_transfer");
        }
        catch (...)
        {
            release_device_buffers();
            ddla::DEVICE_CHECK(ddla::deviceStreamSynchronize(blacs_h.ddla_handle->stream));
            throw;
        }

        release_device_buffers();
        return;
    }
#else
    if (use_device)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Head/wing DDLA body inverse requested in a CPU-only build");
    }
#endif

    profiler.start("headwing_body_inverse_trf_trs");
    try
    {
        if (use_cholesky)
        {
            LaConnector::pposv('L', 'L', 'N', n_body, n_body, body_factor.ptr(), 1, 1,
                               desc_body, body.ptr(), 1, 1, desc_body, info);
        }
        else
        {
            LaConnector::pgesv(n_body, n_body, body_factor.ptr(), 1, 1, desc_body,
                               body.ptr(), 1, 1, desc_body, info);
        }
    }
    catch (const std::exception &error)
    {
        profiler.stop("headwing_body_inverse_trf_trs");
        throw LIBRPA_RUNTIME_ERROR(
            std::string("Head/wing body identity solve failed: ") + error.what());
    }
    profiler.stop("headwing_body_inverse_trf_trs");

    if (info != 0)
    {
        std::ostringstream oss;
        oss << "Head/wing body " << (use_cholesky ? "Cholesky" : "LU")
            << " identity solve failed with info=" << info;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
}

void diele_func::construct_rpa_trace_log_schur(const int ifreq, ArrayDesc &desc_body,
                                               const int wing_row_offset)
{
    using global::profiler;

    profiler.start("cal_rpa_chi0v_schur");
    if (ifreq < 0 || static_cast<std::size_t>(ifreq) >= head.size() ||
        static_cast<std::size_t>(ifreq) >= wing.size())
    {
        std::ostringstream oss;
        oss << "RPA chi0*v head/wing data unavailable for ifreq=" << ifreq
            << " (head_size=" << head.size() << ", wing_size=" << wing.size() << ")";
        throw std::runtime_error(oss.str());
    }

    const auto chi0v_head = get_rpa_chi0v_head(ifreq);
    const auto chi0v_wing = get_rpa_chi0v_wing(ifreq);
    const int n_body = desc_body.m();

    this->Lind.resize(3, 3, MAJOR::COL);
    this->bw.resize(n_body, 3, MAJOR::COL);
    this->wb.resize(3, n_body, MAJOR::COL);

    const auto desc_wing_opt =
        make_rpa_chi0v_wing_desc(desc_body, wing_row_offset, chi0v_wing.nr(), chi0v_wing.nc());

    ArrayDesc desc_lam_3(blacs_h);
    desc_lam_3.init_square_blk(n_body, 3, 0, 0);

    ArrayDesc desc_3_lam(blacs_h);
    desc_3_lam.init_square_blk(3, n_body, 0, 0);

    ArrayDesc desc_3_3(blacs_h);
    desc_3_3.init_square_blk(3, 3, 0, 0);

    auto lam_3 = init_local_mat<complex<double>>(desc_lam_3, MAJOR::COL);
    auto _3_lam = init_local_mat<complex<double>>(desc_3_lam, MAJOR::COL);
    auto schur_correction = init_local_mat<complex<double>>(desc_3_3, MAJOR::COL);

    ScalapackConnector::pgemm_f('N', 'N', n_body, 3, n_body, 1.0, body_inv.ptr(), 1, 1,
                                desc_body.desc, chi0v_wing.ptr(), 1 + wing_row_offset, 1,
                                desc_wing_opt.desc, 0.0, lam_3.ptr(), 1, 1, desc_lam_3.desc);
    ScalapackConnector::pgemm_f('C', 'N', 3, 3, n_body, 1.0, chi0v_wing.ptr(), 1 + wing_row_offset,
                                1, desc_wing_opt.desc, lam_3.ptr(), 1, 1, desc_lam_3.desc, 0.0,
                                schur_correction.ptr(), 1, 1, desc_3_3.desc);
    ScalapackConnector::pgemm_f('C', 'N', 3, n_body, n_body, 1.0, chi0v_wing.ptr(),
                                1 + wing_row_offset, 1, desc_wing_opt.desc, body_inv.ptr(), 1, 1,
                                desc_body.desc, 0.0, _3_lam.ptr(), 1, 1, desc_3_lam.desc);

    for (int i = 0; i != 3; ++i)
    {
        const int loc_i = desc_3_3.indx_g2l_r(i);
        for (int ilambda = 0; ilambda < n_body; ++ilambda)
        {
            int loc_ilambda = desc_lam_3.indx_g2l_r(ilambda);
            const int loc_ibw = desc_lam_3.indx_g2l_c(i);
            if (loc_ibw >= 0 && loc_ilambda >= 0)
                this->bw(ilambda, i) = lam_3(loc_ilambda, loc_ibw);

            loc_ilambda = desc_3_lam.indx_g2l_c(ilambda);
            const int loc_iwb = desc_3_lam.indx_g2l_r(i);
            if (loc_iwb >= 0 && loc_ilambda >= 0)
                this->wb(i, ilambda) = _3_lam(loc_iwb, loc_ilambda);

            MPI_Allreduce(MPI_IN_PLACE, &bw(ilambda, i), 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                          comm_h.comm);
            MPI_Allreduce(MPI_IN_PLACE, &wb(i, ilambda), 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                          comm_h.comm);
        }

        for (int j = 0; j != 3; ++j)
        {
            const int loc_j = desc_3_3.indx_g2l_c(j);
            std::complex<double> correction = 0.0;
            if (loc_i >= 0 && loc_j >= 0)
            {
                correction = schur_correction(loc_i, loc_j);
            }
            MPI_Allreduce(MPI_IN_PLACE, &correction, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, comm_h.comm);

            this->Lind(i, j) = -chi0v_head(i, j) - correction;
            if (i == j) this->Lind(i, j) += 1.0;
        }
    }

    profiler.stop("cal_rpa_chi0v_schur");
}

void diele_func::get_Leb_points()
{
    if (use_2d_dielectric)
    {
        const int n = 5000;
        qx_leb.clear();
        qy_leb.clear();
        qz_leb.clear();
        qw_leb.clear();

        qx_leb.resize(n);
        qy_leb.resize(n);
        qz_leb.resize(n);
        qw_leb.resize(n);
        for (int ileb = 0; ileb != n; ileb++)
        {
            double ang = TWO_PI * ileb / n;
            qx_leb[ileb] = std::cos(ang);
            qy_leb[ileb] = std::sin(ang);
            qz_leb[ileb] = 0.0;
            qw_leb[ileb] = TWO_PI / n;
        }
    }
    else
    {
        // TODO: check convergence issue and change 5810 to argument
        auto quad_points = lebedev_laikov::grid(5810);
        qx_leb = std::move(quad_points.x);
        qy_leb = std::move(quad_points.y);
        qz_leb = std::move(quad_points.z);
        qw_leb = std::move(quad_points.weights);
        const int n = qw_leb.size();
        for (int ileb = 0; ileb != n; ileb++)
        {
            qw_leb[ileb] *= 2 * TWO_PI;
        }
    }
};

void diele_func::get_g_enclosing_gamma()
{
    g_enclosing_gamma.clear();
    g_enclosing_gamma.resize(26);
    const auto &kv_nmp = pbc_.period_array;
    const auto &G = pbc_.G;
    int ik = 0;
    for (int a = -1; a != 2; a++)
    {
        for (int b = -1; b != 2; b++)
        {
            for (int c = -1; c != 2; c++)
            {
                if (a == 0 && b == 0 && c == 0) continue;
                g_enclosing_gamma.at(ik) = {
                    G.e11 * a / kv_nmp[0] + G.e12 * b / kv_nmp[1] + G.e13 * c / kv_nmp[2],
                    G.e21 * a / kv_nmp[0] + G.e22 * b / kv_nmp[1] + G.e23 * c / kv_nmp[2],
                    G.e31 * a / kv_nmp[0] + G.e32 * b / kv_nmp[1] + G.e33 * c / kv_nmp[2]};

                ik++;
            }
        }
    }
};

void diele_func::get_g_enclosing_gamma_2d()
{
    g_enclosing_gamma.clear();
    g_enclosing_gamma.resize(8);
    int ik = 0;
    const auto G = pbc_.G;
    for (int a = -1; a <= 1; a++)
    {
        for (int b = -1; b <= 1; b++)
        {
            if (a == 0 && b == 0) continue;
            g_enclosing_gamma.at(ik) = {G.e11 * a / pbc_.period.x + G.e12 * b / pbc_.period.y,
                                        G.e21 * a / pbc_.period.x + G.e22 * b / pbc_.period.y, 0.0};
            ik++;
        }
    }
};

void diele_func::calculate_q_gamma()
{
    q_gamma.clear();
    q_gamma.resize(qw_leb.size());
    const int n = qw_leb.size();
#pragma omp parallel for schedule(dynamic)
    for (int ileb = 0; ileb != n; ileb++)
    {
        double qmax = 1.0e10;
        Vector3_Order<double> q_quta = {qx_leb[ileb], qy_leb[ileb], qz_leb[ileb]};
        for (int ik = 0; ik != 26; ik++)
        {
            double denominator = q_quta * g_enclosing_gamma[ik];
            if (denominator > 1.0e-10)
            {
                double numerator = 0.5 * g_enclosing_gamma[ik] * g_enclosing_gamma[ik];
                double temp = numerator / denominator;
                qmax = std::min(qmax, temp);
            }
        }

        q_gamma[ileb] = qmax;
    }
};

void diele_func::calculate_q_gamma_2d()
{
    q_gamma.clear();
    q_gamma.resize(qw_leb.size());
    const int n = qw_leb.size();
#pragma omp parallel for schedule(dynamic)
    for (int ileb = 0; ileb != n; ileb++)
    {
        double qmax = 1.0e10;
        Vector3_Order<double> q_quta = {qx_leb[ileb], qy_leb[ileb], qz_leb[ileb]};
        if (use_2d_dielectric)
        {
            for (int ik = 0; ik != 8; ik++)
            {
                double denominator =
                    q_quta.x * g_enclosing_gamma[ik].x + q_quta.y * g_enclosing_gamma[ik].y;
                if (denominator > 1.0e-10)
                {
                    double numerator = 0.5 * g_enclosing_gamma[ik] * g_enclosing_gamma[ik];
                    double temp = numerator / denominator;
                    qmax = std::min(qmax, temp);
                }
            }
        }
        q_gamma[ileb] = qmax;
    }
};

int rpa_headwing_regular_body_start_channel(const RpaHeadwingSettings &settings)
{
    if (settings.rpa_headwing_body_start < -1)
    {
        throw std::logic_error("rpa_headwing_body_start must be at least -1");
    }
    if (settings.rpa_headwing_body_start == -1)
    {
        return 0;
    }
    if (settings.rpa_headwing_body_start > 0)
    {
        return settings.rpa_headwing_body_start;
    }
    return 1;
}

double rpa_headwing_reciprocal_cell_volume(const PeriodicBoundaryData &pbc,
                                           const bool use_2d_dielectric)
{
    if (use_2d_dielectric)
    {
        return std::abs(pbc.G.e11 * pbc.G.e22 - pbc.G.e12 * pbc.G.e21);
    }
    return std::abs(pbc.G.Det());
}

double rpa_headwing_gamma_cell_volume(const PeriodicBoundaryData &pbc,
                                      const bool use_2d_dielectric)
{
    const int n_full_bz = std::max(1, pbc.get_n_cells_bvk());
    return rpa_headwing_reciprocal_cell_volume(pbc, use_2d_dielectric)
           / static_cast<double>(n_full_bz);
}

ArrayDesc make_rpa_chi0v_wing_desc(const ArrayDesc &desc_body, const int wing_row_offset,
                                   const int wing_rows_loc, const int wing_cols_loc)
{
    const int n_body = desc_body.m();
    const int wing_rows = wing_row_offset + n_body;
    if (n_body <= 0 || wing_row_offset < 0)
    {
        std::ostringstream oss;
        oss << "RPA chi0*v Schur wing/body mismatch: n_body=" << n_body
            << ", wing_row_offset=" << wing_row_offset;
        throw std::logic_error(oss.str());
    }

    ArrayDesc desc_wing(desc_body.ictxt());
    desc_wing.init(wing_rows, 3, desc_body.mb(), 1, 0, 0);
    if (wing_rows_loc != desc_wing.m_loc() || wing_cols_loc != desc_wing.n_loc())
    {
        std::ostringstream oss;
        oss << "RPA chi0*v Schur wing local descriptor mismatch: global_wing_rows=" << wing_rows
            << ", local_wing=" << wing_rows_loc << "x" << wing_cols_loc
            << ", expected_local_wing=" << desc_wing.m_loc() << "x" << desc_wing.n_loc();
        throw std::logic_error(oss.str());
    }
    return desc_wing;
}

std::complex<double> compute_rpa_chi0v_headwing_trace_log_average(
    const matrix_m<std::complex<double>> &head, const matrix_m<std::complex<double>> &schur_l,
    const std::complex<double> &trace_body, const std::complex<double> &logdet_body,
    const std::vector<double> &qx, const std::vector<double> &qy, const std::vector<double> &qz,
    const std::vector<double> &weights, double *weight_sum_out,
    std::complex<double> *averaged_body_out, std::complex<double> *averaged_head_out,
    std::complex<double> *averaged_schur_log_out)
{
    if (head.nr() != 3 || head.nc() != 3 || schur_l.nr() != 3 || schur_l.nc() != 3)
    {
        throw std::logic_error(
            "RPA chi0*v head/wing trace-log average expects 3x3 head and Schur matrices");
    }
    if (qx.size() != qy.size() || qx.size() != qz.size() || qx.size() != weights.size())
    {
        throw std::logic_error("RPA head/wing trace-log average direction grids are inconsistent");
    }

    double weight_sum = 0.0;
    std::complex<double> averaged_head = 0.0;
    std::complex<double> averaged_schur_log = 0.0;
    for (std::size_t i = 0; i != weights.size(); ++i)
    {
        weight_sum += weights[i];
        const double nx = qx[i];
        const double ny = qy[i];
        const double nz = qz[i];
        const auto directional_head = nx * (nx * head(0, 0) + ny * head(0, 1) + nz * head(0, 2)) +
                                      ny * (nx * head(1, 0) + ny * head(1, 1) + nz * head(1, 2)) +
                                      nz * (nx * head(2, 0) + ny * head(2, 1) + nz * head(2, 2));
        const auto directional_schur =
            nx * (nx * schur_l(0, 0) + ny * schur_l(0, 1) + nz * schur_l(0, 2)) +
            ny * (nx * schur_l(1, 0) + ny * schur_l(1, 1) + nz * schur_l(1, 2)) +
            nz * (nx * schur_l(2, 0) + ny * schur_l(2, 1) + nz * schur_l(2, 2));
        averaged_head += weights[i] * directional_head;
        averaged_schur_log += weights[i] * std::log(directional_schur);
    }

    const auto averaged_body = weight_sum * (trace_body + logdet_body);
    if (weight_sum_out != nullptr) *weight_sum_out = weight_sum;
    if (averaged_body_out != nullptr) *averaged_body_out = averaged_body;
    if (averaged_head_out != nullptr) *averaged_head_out = averaged_head;
    if (averaged_schur_log_out != nullptr) *averaged_schur_log_out = averaged_schur_log;
    return averaged_body + averaged_head + averaged_schur_log;
}

void replace_rpa_response_headwing(matrix_m<std::complex<double>> &response_block,
                                   const matrix_m<std::complex<double>> &head,
                                   const matrix_m<std::complex<double>> &wing,
                                   const ArrayDesc &desc_response)
{
    if (head.nr() != 3 || head.nc() != 3)
    {
        throw std::logic_error("RPA head/wing replacement expects a 3x3 chi0*v head");
    }
    if (wing.nc() != 3)
    {
        throw std::logic_error("RPA head/wing replacement expects wing columns for x/y/z");
    }
    if (desc_response.m() < 1 || desc_response.n() < 1)
    {
        throw std::logic_error("RPA head/wing replacement expects a non-empty response matrix");
    }
    if (wing.nr() > desc_response.m() - 1)
    {
        throw std::logic_error(
            "RPA head/wing replacement wing size does not match the Coulomb body dimension");
    }

    std::complex<double> head_average = 0.0;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        head_average += head(alpha, alpha);
    }
    head_average /= 3.0;

    const int ilo_head = desc_response.indx_g2l_r(0);
    const int jlo_head = desc_response.indx_g2l_c(0);
    if (ilo_head >= 0 && jlo_head >= 0)
    {
        response_block(ilo_head, jlo_head) = head_average;
    }

    for (int ilambda = 0; ilambda != wing.nr(); ++ilambda)
    {
        std::complex<double> wing_average = 0.0;
        for (int alpha = 0; alpha != 3; ++alpha)
        {
            wing_average += wing(ilambda, alpha);
        }
        wing_average /= 3.0;

        const int global_body = ilambda + 1;
        const int ilo_body = desc_response.indx_g2l_r(global_body);
        const int jlo_body = desc_response.indx_g2l_c(global_body);
        if (ilo_body >= 0 && jlo_head >= 0)
        {
            response_block(ilo_body, jlo_head) = wing_average;
        }
        if (ilo_head >= 0 && jlo_body >= 0)
        {
            response_block(ilo_head, jlo_body) = std::conj(wing_average);
        }
    }
}

void replace_rpa_response_head_only(matrix_m<std::complex<double>> &response_block,
                                    const matrix_m<std::complex<double>> &head,
                                    const ArrayDesc &desc_response)
{
    if (head.nr() != 3 || head.nc() != 3)
    {
        throw std::logic_error("RPA head-only replacement expects a 3x3 chi0*v head");
    }
    if (desc_response.m() < 1 || desc_response.n() < 1)
    {
        throw std::logic_error("RPA head-only replacement expects a non-empty response matrix");
    }

    std::complex<double> head_average = 0.0;
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        head_average += head(alpha, alpha);
    }
    head_average /= 3.0;

    const int ilo_head = desc_response.indx_g2l_r(0);
    const int jlo_head = desc_response.indx_g2l_c(0);
    if (ilo_head >= 0 && jlo_head >= 0)
    {
        response_block(ilo_head, jlo_head) = head_average;
    }
}

void rewrite_eps_abf_space(
    matrix_m<std::complex<double>> &eps_block,
    const matrix_m<std::complex<double>> &sqrtv_block,
    const matrix_m<std::complex<double>> &coul_eigen_block,
    const matrix_m<std::complex<double>> &head,
    const matrix_m<std::complex<double>> &wing_mu,
    const std::vector<double> &qx, const std::vector<double> &qy,
    const std::vector<double> &qz, const std::vector<double> &rho,
    const ArrayDesc &desc_nabf_nabf_opt, const BlacsCtxtHandler &blacs_h,
    const std::size_t n_nonsingular,
    const double sqrt_coulomb_threshold, const bool use_cholesky, const bool use_device)
{
    using global::profiler;

    if (!desc_nabf_nabf_opt.is_initialized())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABF-space wing rewrite expects an initialized distributed descriptor");
    }
    const int n_abf = desc_nabf_nabf_opt.m();
    if (n_abf <= 0)
    {
        std::ostringstream oss;
        oss << "ABF-space wing rewrite expects a positive basis dimension, got n_abf=" << n_abf;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
    if (desc_nabf_nabf_opt.n() != n_abf)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABF-space wing rewrite expects a square distributed descriptor");
    }
    if (n_nonsingular == 0 || n_nonsingular > static_cast<std::size_t>(n_abf))
    {
        std::ostringstream oss;
        oss << "ABF-space wing rewrite expects 0 < n_nonsingular <= n_abf: "
               "sqrt_coulomb_threshold="
            << sqrt_coulomb_threshold << ", n_nonsingular=" << n_nonsingular
            << ", n_abf=" << n_abf;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
    if (desc_nabf_nabf_opt.ictxt() != blacs_h.ictxt)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABF-space wing rewrite descriptor and BLACS context do not match");
    }
    if (eps_block.nr() != desc_nabf_nabf_opt.m_loc() ||
        eps_block.nc() != desc_nabf_nabf_opt.n_loc())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABF-space wing rewrite E matrix does not match its distributed descriptor");
    }
    if (sqrtv_block.nr() != desc_nabf_nabf_opt.m_loc() ||
        sqrtv_block.nc() != desc_nabf_nabf_opt.n_loc() ||
        coul_eigen_block.nr() != desc_nabf_nabf_opt.m_loc() ||
        coul_eigen_block.nc() != desc_nabf_nabf_opt.n_loc())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABF-space wing rewrite sqrt(V)/eigenvector blocks do not match the descriptor");
    }
    if (head.nr() != 3 || head.nc() != 3)
    {
        throw LIBRPA_RUNTIME_ERROR("ABF-space wing rewrite expects a 3x3 head");
    }
    if (wing_mu.nr() != n_abf || wing_mu.nc() != 3)
    {
        throw LIBRPA_RUNTIME_ERROR("ABF-space wing rewrite expects an n_abf x 3 wing_mu");
    }
    if (qx.size() != qy.size() || qx.size() != qz.size() || qx.size() != rho.size())
    {
        throw LIBRPA_RUNTIME_ERROR(
            "ABF-space wing rewrite angular quadrature arrays have inconsistent lengths");
    }

    // Filtered Coulomb channels occupy the trailing eigenvector columns: both
    // eigensolvers redistribute the complete orthonormal set (eigenvalues
    // descending) before the filtering loop, so U U^H = I to O(eps) and
    // P + T + F = I with P = x1*x1^H, T the retained body projector and F the
    // filtered projector. Y = [x1 | X_s] therefore spans YY^H = P + F = I - T
    // with rank m = 1 + n_singular, and
    //     M = (I - YY^H) E (I - YY^H) + YY^H.
    // Expanding with G = E*Y and S = Y^H*G keeps every operand thin
    // (n_abf x m) and never materializes an n_abf x n_abf projector, so the cost
    // tracks n_singular instead of n_abf. Only E*Y is contracted, which is exact
    // because E is Hermitian here; see the note at the product below. The m == 1
    // case (no filtered channels) reduces term for term to the same expression
    // with Y = x1.
    // Above 2m > n_abf this low-rank form does more flops than forming T
    // explicitly; that is not the production regime (n_nonsingular >> n_singular)
    // and M = I still comes out by cancelling E against itself to O(|E|*eps_mach),
    // far below the 1e-10 tolerance the rewrite is validated at.
    profiler.start("epsilon_headwing_abf_build_M");
    const int n_sing = n_abf - static_cast<int>(n_nonsingular);
    const int m = 1 + n_sing;

    ArrayDesc desc_nabf_m(blacs_h);
    desc_nabf_m.init(n_abf, m, desc_nabf_nabf_opt.mb(), desc_nabf_nabf_opt.nb(), 0, 0);
    auto Y = init_local_mat<complex<double>>(desc_nabf_m, MAJOR::COL);
    if (desc_nabf_m.m_loc() == 0 || desc_nabf_m.n_loc() == 0) Y.resize(1, 1);

    // Y = [x1 | X_s]: the head column, then the trailing filtered block.
    ScalapackConnector::pgemr2d_f(n_abf, 1, coul_eigen_block.ptr(), 1, 1,
                                  desc_nabf_nabf_opt.desc, Y.ptr(), 1, 1,
                                  desc_nabf_m.desc, blacs_h.ictxt);
    if (n_sing > 0)
    {
        ScalapackConnector::pgemr2d_f(
            n_abf, n_sing, coul_eigen_block.ptr(), 1,
            static_cast<int>(n_nonsingular) + 1, desc_nabf_nabf_opt.desc,
            Y.ptr(), 1, 2, desc_nabf_m.desc, blacs_h.ictxt);
    }

    ArrayDesc desc_mm(blacs_h);
    desc_mm.init(m, m, desc_nabf_nabf_opt.mb(), desc_nabf_nabf_opt.nb(), 0, 0);
    auto G = init_local_mat<complex<double>>(desc_nabf_m, MAJOR::COL);
    if (desc_nabf_m.m_loc() == 0 || desc_nabf_m.n_loc() == 0) G.resize(1, 1);

    // G = E*Y. E is Hermitian on the imaginary-frequency axis (this path is only
    // reached from the option_dielect_func == 3 branch), so E^H*Y == E*Y and one
    // product suffices; the Cholesky inverse path taken below already reads only
    // the lower triangle of M via pposv, so building M would otherwise refuse an
    // assumption the inverse then makes. E is Hermitian to roundoff rather than
    // bitwise, so results are accurate to O(|E - E^H|) ~ eps_mach*|E|.
    ScalapackConnector::pgemm_f('N', 'N', n_abf, m, n_abf, 1.0,
                                eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                0.0, G.ptr(), 1, 1, desc_nabf_m.desc);

    // S = Y^H*G + I_m
    auto S = init_local_mat<complex<double>>(desc_mm, MAJOR::COL);
    if (desc_mm.m_loc() == 0 || desc_mm.n_loc() == 0) S.resize(1, 1);
    ScalapackConnector::pgemm_f('C', 'N', m, m, n_abf, 1.0,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                G.ptr(), 1, 1, desc_nabf_m.desc,
                                0.0, S.ptr(), 1, 1, desc_mm.desc);
    LaConnector::pdam(1.0, S.ptr(), desc_mm);

    // M = E - G*Y^H - Y*G^H + Y*(S)*Y^H, accumulated in place. The two rank-m
    // updates are exact conjugate transposes of each other by construction, so
    // the low-rank correction contributes no anti-Hermitian noise of its own.
    ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, m, -1.0,
                                G.ptr(), 1, 1, desc_nabf_m.desc,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                1.0, eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
    ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, m, -1.0,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                G.ptr(), 1, 1, desc_nabf_m.desc,
                                1.0, eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);

    // W = Y*S, then M += W*Y^H.
    auto W = init_local_mat<complex<double>>(desc_nabf_m, MAJOR::COL);
    if (desc_nabf_m.m_loc() == 0 || desc_nabf_m.n_loc() == 0) W.resize(1, 1);
    ScalapackConnector::pgemm_f('N', 'N', n_abf, m, m, 1.0,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                S.ptr(), 1, 1, desc_mm.desc,
                                0.0, W.ptr(), 1, 1, desc_nabf_m.desc);
    ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, m, 1.0,
                                W.ptr(), 1, 1, desc_nabf_m.desc,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                1.0, eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);

    profiler.stop("epsilon_headwing_abf_build_M");

    // Invert M in-place, then subtract the low-rank projector YY^H = I - T so
    // that D = M^-1 - YY^H vanishes on the filtered Coulomb channels.
    profiler.start("epsilon_headwing_abf_inverse");
    // ArrayDesc owns its ELPA handle, so copying it would duplicate ownership.
    // Recreate only the BLACS layout needed by the identity solve.
    ArrayDesc desc_invert(blacs_h);
    desc_invert.init(desc_nabf_nabf_opt.m(), desc_nabf_nabf_opt.n(),
                     desc_nabf_nabf_opt.mb(), desc_nabf_nabf_opt.nb(),
                     desc_nabf_nabf_opt.irsrc(), desc_nabf_nabf_opt.icsrc());
    invert_headwing_body_with_identity_solve(eps_block, desc_invert, blacs_h,
                                             use_cholesky, use_device);
    ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, m, -1.0,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                Y.ptr(), 1, 1, desc_nabf_m.desc,
                                1.0, eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
    profiler.stop("epsilon_headwing_abf_inverse");
    // eps_block now holds the inverse regular body embedded in ABF space.

    // Wing construction: Z = sqrt(V)*W_mu, A = D*Z.
    profiler.start("epsilon_headwing_abf_wing");

    // Keep the contracted n_abf dimension aligned with sqrt(V)'s column
    // distribution. A unit block is sufficient for the three Cartesian
    // columns and keeps all subsequent thin PGEMMs mutually aligned.
    ArrayDesc desc_wing_mu(blacs_h);
    desc_wing_mu.init(n_abf, 3, desc_nabf_nabf_opt.nb(), 1, 0, 0);
    auto wing_mu_dist = init_local_mat<complex<double>>(desc_wing_mu, MAJOR::COL);
    for (int alpha = 0; alpha != 3; ++alpha)
    {
        const int loc_alpha = desc_wing_mu.indx_g2l_c(alpha);
        if (loc_alpha < 0) continue;
        for (int mu = 0; mu != n_abf; ++mu)
        {
            const int loc_mu = desc_wing_mu.indx_g2l_r(mu);
            if (loc_mu < 0) continue;
            wing_mu_dist(loc_mu, loc_alpha) = wing_mu(mu, alpha);
        }
    }

    ArrayDesc desc_nabf_3(blacs_h);
    desc_nabf_3.init(n_abf, 3, desc_nabf_nabf_opt.mb(), desc_wing_mu.nb(), 0, 0);
    auto Z = init_local_mat<complex<double>>(desc_nabf_3, MAJOR::COL);
    // Z = sqrt(V) * W_mu
    ScalapackConnector::pgemm_f('N', 'N', n_abf, 3, n_abf, 1.0,
                                sqrtv_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                wing_mu_dist.ptr(), 1, 1, desc_wing_mu.desc,
                                0.0, Z.ptr(), 1, 1, desc_nabf_3.desc);

    // A = D * Z
    auto A = init_local_mat<complex<double>>(desc_nabf_3, MAJOR::COL);
    ScalapackConnector::pgemm_f('N', 'N', n_abf, 3, n_abf, 1.0,
                                eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                Z.ptr(), 1, 1, desc_nabf_3.desc,
                                0.0, A.ptr(), 1, 1, desc_nabf_3.desc);

    profiler.stop("epsilon_headwing_abf_wing");

    // L = H - Z^H*A and angular quadrature for a0, K.
    profiler.start("epsilon_headwing_abf_angular");

    ArrayDesc desc_3x3(blacs_h);
    desc_3x3.init(3, 3, desc_wing_mu.nb(), desc_wing_mu.nb(), 0, 0);
    auto zha_dist = init_local_mat<complex<double>>(desc_3x3, MAJOR::COL);
    ScalapackConnector::pgemm_f('C', 'N', 3, 3, n_abf, 1.0,
                                Z.ptr(), 1, 1, desc_nabf_3.desc,
                                A.ptr(), 1, 1, desc_nabf_3.desc,
                                0.0, zha_dist.ptr(), 1, 1, desc_3x3.desc);

    matrix_m<std::complex<double>> Lmat(3, 3, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        for (int j = 0; j != 3; ++j)
        {
            const int loc_i = desc_3x3.indx_g2l_r(i);
            const int loc_j = desc_3x3.indx_g2l_c(j);
            std::complex<double> zha = 0.0;
            if (loc_i >= 0 && loc_j >= 0) zha = zha_dist(loc_i, loc_j);
            MPI_Allreduce(MPI_IN_PLACE, &zha, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                          desc_nabf_nabf_opt.comm());
            Lmat(i, j) = head(i, j) - zha;
        }
    }

    const int nleb = static_cast<int>(qx.size());
    const auto L00 = Lmat(0, 0), L01 = Lmat(0, 1), L02 = Lmat(0, 2);
    const auto L10 = Lmat(1, 0), L11 = Lmat(1, 1), L12 = Lmat(1, 2);
    const auto L20 = Lmat(2, 0), L21 = Lmat(2, 1), L22 = Lmat(2, 2);

    std::complex<double> a0 = 0.0;
    std::array<std::array<std::complex<double>, 3>, 3> K{};
    for (int ileb = 0; ileb < nleb; ++ileb)
    {
        const double qxi = qx[ileb];
        const double qyi = qy[ileb];
        const double qzi = qz[ileb];
        const auto qLq = qxi * (qxi * L00 + qyi * L01 + qzi * L02) +
                         qyi * (qxi * L10 + qyi * L11 + qzi * L12) +
                         qzi * (qxi * L20 + qyi * L21 + qzi * L22);
        const auto w = rho[ileb] / qLq;
        a0 += w;
        const std::array<double, 3> qv{qxi, qyi, qzi};
        for (int alpha = 0; alpha < 3; ++alpha)
            for (int beta = 0; beta < 3; ++beta)
                K[alpha][beta] += w * qv[alpha] * qv[beta];
    }

    profiler.stop("epsilon_headwing_abf_angular");

    // Assemble eps_inv = a0*P + D + A*K*A^H  (eps_block holds D).
    profiler.start("epsilon_headwing_abf_assemble");

    ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, 1, a0,
                                coul_eigen_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                coul_eigen_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                1.0, eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);

    auto K_dist = init_local_mat<complex<double>>(desc_3x3, MAJOR::COL);
    for (int i = 0; i != 3; ++i)
    {
        const int loc_i = desc_3x3.indx_g2l_r(i);
        if (loc_i < 0) continue;
        for (int j = 0; j != 3; ++j)
        {
            const int loc_j = desc_3x3.indx_g2l_c(j);
            if (loc_j >= 0) K_dist(loc_i, loc_j) = K[i][j];
        }
    }
    auto AK = init_local_mat<complex<double>>(desc_nabf_3, MAJOR::COL);
    ScalapackConnector::pgemm_f('N', 'N', n_abf, 3, 3, 1.0,
                                A.ptr(), 1, 1, desc_nabf_3.desc,
                                K_dist.ptr(), 1, 1, desc_3x3.desc,
                                0.0, AK.ptr(), 1, 1, desc_nabf_3.desc);
    ScalapackConnector::pgemm_f('N', 'C', n_abf, n_abf, 3, 1.0,
                                AK.ptr(), 1, 1, desc_nabf_3.desc,
                                A.ptr(), 1, 1, desc_nabf_3.desc,
                                1.0, eps_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);

    profiler.stop("epsilon_headwing_abf_assemble");
}

void diele_func::rewrite_eps_abf_space(matrix_m<std::complex<double>> &eps_block, const int ifreq,
                                       const matrix_m<std::complex<double>> &sqrtv_block,
                                       const matrix_m<std::complex<double>> &coul_eigen_block,
                                       const ArrayDesc &desc_nabf_nabf_opt,
                                       const std::size_t n_nonsingular_in,
                                       const double sqrt_coulomb_threshold,
                                       const bool use_cholesky, const bool use_device)
{
    this->vol_gamma = rpa_headwing_gamma_cell_volume(pbc_, use_2d_dielectric);
    const int nleb = static_cast<int>(qw_leb.size());
    std::vector<double> rho(nleb);
    for (int ileb = 0; ileb < nleb; ++ileb)
    {
        rho[ileb] = use_2d_dielectric
                        ? qw_leb[ileb] * std::pow(q_gamma[ileb], 2) / (2.0 * vol_gamma)
                        : qw_leb[ileb] * std::pow(q_gamma[ileb], 3) / (3.0 * vol_gamma);
    }
    librpa_int::rewrite_eps_abf_space(eps_block, sqrtv_block, coul_eigen_block, head.at(ifreq),
                                      wing_mu.at(ifreq), qx_leb, qy_leb, qz_leb, rho,
                                      desc_nabf_nabf_opt, blacs_h, n_nonsingular_in,
                                      sqrt_coulomb_threshold, use_cholesky, use_device);
}

std::complex<double> diele_func::compute_rpa_trace_log_average(
    matrix_m<std::complex<double>> &response_block, const int ifreq, ArrayDesc &desc_response,
    const RpaHeadwingSettings &settings)
{
    if (ifreq < 0 || static_cast<std::size_t>(ifreq) >= head.size() ||
        static_cast<std::size_t>(ifreq) >= wing.size())
    {
        std::ostringstream oss;
        oss << "RPA chi0*v head/wing data unavailable for trace-log average for ifreq=" << ifreq
            << " (head_size=" << head.size() << ", wing_size=" << wing.size() << ")";
        throw std::runtime_error(oss.str());
    }
    if (desc_response.m() < 2 || desc_response.n() < 2)
    {
        throw std::logic_error("RPA head/wing trace-log average needs at least one body channel");
    }
    if (static_cast<std::size_t>(desc_response.m()) != n_nonsingular ||
        static_cast<std::size_t>(desc_response.n()) != n_nonsingular)
    {
        std::ostringstream oss;
        oss << "RPA head/wing trace-log subspace mismatch: response=" << desc_response.m() << "x"
            << desc_response.n() << ", headwing n_nonsingular=" << n_nonsingular;
        throw std::logic_error(oss.str());
    }

    const int body_start = rpa_headwing_regular_body_start_channel(settings);
    if (body_start == 0)
    {
        throw std::logic_error(
            "rpa_headwing_body_start=-1 is supported only by the Sternheimer RPA path");
    }
    const int wing_row_offset = body_start - 1;
    if (desc_response.m() <= body_start || wing_row_offset >= wing.at(ifreq).nr())
    {
        std::ostringstream oss;
        oss << "RPA head/wing regular-body split is inconsistent: response_size="
            << desc_response.m() << ", body_start=" << body_start
            << ", wing_rows=" << wing.at(ifreq).nr();
        throw std::logic_error(oss.str());
    }

    ArrayDesc desc_body(blacs_h);
    desc_body.init_square_blk(desc_response.m() - body_start, desc_response.n() - body_start, 0, 0);
    auto body = init_local_mat<std::complex<double>>(desc_body, MAJOR::COL);
    ScalapackConnector::pgemr2d_f(desc_response.m() - body_start, desc_response.n() - body_start,
                                  response_block.ptr(), body_start + 1, body_start + 1,
                                  desc_response.desc, body.ptr(), 1, 1, desc_body.desc,
                                  blacs_h.ictxt);

    std::complex<double> trace_body_loc = 0.0;
    for (int i = 0; i != desc_body.m(); ++i)
    {
        const int ilo = desc_body.indx_g2l_r(i);
        const int jlo = desc_body.indx_g2l_c(i);
        if (ilo >= 0 && jlo >= 0) trace_body_loc += body(ilo, jlo);
    }
    std::complex<double> trace_body = 0.0;
    MPI_Allreduce(&trace_body_loc, &trace_body, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, comm_h.comm);

    auto identity_minus_body = body.copy();
    identity_minus_body *= -1.0;
    for (int i = 0; i != desc_body.m(); ++i)
    {
        const int ilo = desc_body.indx_g2l_r(i);
        const int jlo = desc_body.indx_g2l_c(i);
        if (ilo >= 0 && jlo >= 0) identity_minus_body(ilo, jlo) += 1.0;
    }

    auto identity_minus_body_for_logdet = identity_minus_body.copy();
    int info = 0;
    std::vector<int> ipiv(std::max(1, desc_body.m_loc() * 10));
    std::complex<double> logdet_body =
        compute_pi_det_blacs_2d(identity_minus_body_for_logdet, desc_body, ipiv.data(), info);

    this->body_inv = identity_minus_body.copy();
    invert_scalapack(this->body_inv, desc_body);
    construct_rpa_trace_log_schur(ifreq, desc_body, wing_row_offset);

    this->vol_gamma = rpa_headwing_gamma_cell_volume(pbc_, settings.use_2d_dielectric);

    std::vector<double> weights(qw_leb.size());
    for (std::size_t ileb = 0; ileb != qw_leb.size(); ++ileb)
    {
        if (settings.use_2d_dielectric)
            weights[ileb] = qw_leb[ileb] * std::pow(q_gamma[ileb], 2) / (2.0 * vol_gamma);
        else
            weights[ileb] = qw_leb[ileb] * std::pow(q_gamma[ileb], 3) / (3.0 * vol_gamma);
    }

    double weight_sum = 0.0;
    std::complex<double> averaged_body = 0.0;
    std::complex<double> averaged_head = 0.0;
    std::complex<double> averaged_schur_log = 0.0;
    const auto result = compute_rpa_chi0v_headwing_trace_log_average(
        get_rpa_chi0v_head(ifreq), Lind, trace_body, logdet_body, qx_leb, qy_leb, qz_leb, weights,
        &weight_sum, &averaged_body, &averaged_head, &averaged_schur_log);

    if (debug && comm_h.is_root())
    {
        global::lib_printf(
            "RPA HW avg ifreq=%d trace_body=(%.12e,%.12e) logdet_body=(%.12e,%.12e) "
            "weight_sum=%.12e averaged_body=(%.12e,%.12e) "
            "averaged_head=(%.12e,%.12e) averaged_schur_log=(%.12e,%.12e) "
            "total=(%.12e,%.12e)\n",
            ifreq, trace_body.real(), trace_body.imag(), logdet_body.real(), logdet_body.imag(),
            weight_sum, averaged_body.real(), averaged_body.imag(), averaged_head.real(),
            averaged_head.imag(), averaged_schur_log.real(), averaged_schur_log.imag(),
            result.real(), result.imag());
    }

    this->Lind.clear();
    this->body_inv.clear();
    return result;
}

void diele_func::rewrite_rpa_response(matrix_m<std::complex<double>> &eps_minus_identity_block,
                                      const int ifreq, ArrayDesc &desc_nabf_nabf_opt)
{
    if (ifreq < 0 || static_cast<std::size_t>(ifreq) >= head.size() ||
        static_cast<std::size_t>(ifreq) >= wing.size())
    {
        std::ostringstream oss;
        oss << "RPA chi0*v head/wing data unavailable for response replacement for ifreq=" << ifreq
            << " (head_size=" << head.size() << ", wing_size=" << wing.size() << ")";
        throw std::runtime_error(oss.str());
    }

    replace_rpa_response_headwing(eps_minus_identity_block, get_rpa_chi0v_head(ifreq),
                                  get_rpa_chi0v_wing(ifreq), desc_nabf_nabf_opt);
}

}  // namespace librpa_int
