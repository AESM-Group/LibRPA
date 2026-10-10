#include "exx.h"

#include <algorithm>
#include <omp.h>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../io/fs.h"
#include "../io/global_io.h"
#include "../io/output_matrix.h"
#include "../io/stl_io_helper.h"
#include "../math/lapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../math/utils_matrix_mpi.h"
#include "../math/vector3_order.h"
#include "../mpi/global_mpi.h"
#include "../utils/base_utility.h"
#include "../utils/constants.h"
#include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "symmetry_context.h"
#include "../utils/utils_mem.h"
#include "atomic_basis.h"
#include "meanfield_mpi.h"
#include "geometry.h"
#include "librpa_enums.h"
#include "../gpu/la_connector.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla_connector.h>
using namespace ddla;
#endif
// #include "params.h"
#include "pbc.h"
#include "utils_atomic_basis_blacs.h"
#ifdef LIBRPA_USE_LIBRI
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
#include <RI/parallel/Parallel_LRI_Equally_Weighted.h>
#endif
#include <RI/physics/Exx.h>
#include <RI/physics/symmetry/Symmetry_Filter.h>
#include <RI/ri/Cell_Nearest.h>
#else
#include "../utils/libri_stub.h"
#endif

namespace librpa_int
{

static void add_phase_weighted_exx_ijk(const Vector3_Order<int>& R_bvk,
                                       const Vector3_Order<double>& kfrac_ik,
                                       const double weight,
                                       Matz exx_weighted,
                                       Matz& exx_ijk)
{
    const auto ang = (kfrac_ik * R_bvk) * TWO_PI;
    const cplxdb phase{weight * std::cos(ang), weight * std::sin(ang)};
    exx_weighted *= phase;
    exx_ijk += exx_weighted;
}

static Matz make_exx_ijk_complex_block(const Matd& exx)
{
    return exx.to_complex();
}

static Matz make_exx_ijk_complex_block(const Matz& exx)
{
    return exx.copy();
}

static bool use_symmetry_ibz_root_projection(
    const SymmetryContext& ctx,
    const PeriodicBoundaryData& pbc,
    const int n_target_kpoints,
    const int n_meanfield_kpoints)
{
    return ctx.available && !ctx.kstars.empty()
        && pbc.klist.size() < static_cast<std::size_t>(pbc.get_n_cells_bvk())
        && ctx.kstars.size() == static_cast<std::size_t>(n_meanfield_kpoints)
        && n_target_kpoints == n_meanfield_kpoints;
}

static std::map<std::pair<int, int>, std::set<std::array<int, 3>>>
convert_symmetry_irreducible_sector_to_libri(
    const librpa_int::symmetry_irreducible_sector_t& irreducible_sector,
    const std::array<int, 3>& period)
{
    auto canonicalize_r = [&period](const std::array<int, 3>& r) {
        auto centered_mod = [](const int value, const int cell_period) {
            if (cell_period <= 0)
            {
                return value;
            }
            return (value % cell_period + 3 * cell_period / 2) % cell_period
                - cell_period / 2;
        };
        return std::array<int, 3>{centered_mod(r[0], period[0]),
                                  centered_mod(r[1], period[1]),
                                  centered_mod(r[2], period[2])};
    };

    std::map<std::pair<int, int>, std::set<std::array<int, 3>>> libri_sector;
    for (const auto& pair_Rs : irreducible_sector)
    {
        const std::pair<int, int> atom_pair{
            static_cast<int>(pair_Rs.first.first),
            static_cast<int>(pair_Rs.first.second)};
        for (const auto& r : pair_Rs.second)
        {
            libri_sector[atom_pair].insert(canonicalize_r(r));
        }
    }
    return libri_sector;
}

static bool exx_coulomb_uses_symmetry_irreducible_sector_layout(
    const atpair_R_mat_t& coul_mat,
    const librpa_int::SymmetryContext& symmetry_ctx,
    const std::size_t n_R_blocks)
{
    if (coul_mat.empty())
    {
        return false;
    }
    const auto n_atoms = symmetry_ctx.atom_to_type.size();
    if (symmetry_ctx.count_irreducible_blocks() >= n_atoms * n_atoms * n_R_blocks)
    {
        return false;
    }
    for (const auto& i_entry : coul_mat)
    {
        const auto ir_I = static_cast<atom_t>(i_entry.first);
        for (const auto& j_entry : i_entry.second)
        {
            const auto ir_J = static_cast<atom_t>(j_entry.first);
            const auto sector_iter = symmetry_ctx.irreducible_sector.find({ir_I, ir_J});
            if (sector_iter == symmetry_ctx.irreducible_sector.end())
            {
                return false;
            }
            for (const auto& r_entry : j_entry.second)
            {
                const auto& R = r_entry.first;
                const librpa_int::symmetry_R_t r_array{R.x, R.y, R.z};
                if (sector_iter->second.count(r_array) == 0)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

#ifdef LIBRPA_USE_LIBRI
template <typename TA, typename TC, typename Tdata>
class OutputOnlyFilter_Atom_Symmetry : public RI::Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
    using TAC = std::pair<TA, TC>;

    OutputOnlyFilter_Atom_Symmetry(
        const TC& period,
        const std::map<std::pair<TA, TA>, std::set<TC>>& irreducible_sector)
        : symmetry(period, irreducible_sector)
    {
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TA& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a1b0_a2b1:
            case RI::Label::ab_ab::a1b1_a2b0:
            case RI::Label::ab_ab::a0b0_a2b1:
            case RI::Label::ab_ab::a0b1_a2b0:
            case RI::Label::ab_ab::a1b1_a2b2:
            case RI::Label::ab_ab::a0b1_a2b2:
            case RI::Label::ab_ab::a1b0_a2b2:
            case RI::Label::ab_ab::a0b0_a2b2:
                return !this->symmetry.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TA&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b2:
            case RI::Label::ab_ab::a0b1_a1b2:
            case RI::Label::ab_ab::a0b2_a1b0:
            case RI::Label::ab_ab::a0b2_a1b1:
                return !this->symmetry.in_irreducible_sector(A3, A1);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b1:
            case RI::Label::ab_ab::a0b1_a1b0:
                return !this->symmetry.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

  private:
    RI::Symmetry_Filter<TA, TC, Tdata> symmetry;
};
#endif

static void write_exx_nao_k_matrix(const Matz &mat, const ArrayDesc &desc,
                                   const std::string &output_dir, const std::string &source,
                                   const int ispin, const int ispinor_bra, const int ispinor_ket,
                                   const int n_spinor, const int ik)
{
    std::ostringstream ss;
    ss << path_as_directory(output_dir) << "ExxK_" << source << "_ispin_" << ispin;
    if (n_spinor > 1) ss << "_spinor_" << ispinor_bra << "_" << ispinor_ket;
    ss << "_ik_" << ik << ".bin";
    // The internal NAO matrix carries the opposite sign; KS rotation applies -1.
    write_matrix_binary_parallel(mat * (-1.0), desc, ss.str());
}

Exx::Exx(const MeanField &mf_in, const AtomicBasis &atbasis_wfc_in,
         const PeriodicBoundaryData &pbc_in, const SymmetryContext &symmetry_context_in,
         const KPointBlacsParallelContext &kblacs_ctxt_in,
         const KPointBlacsParallelContext &band_kblacs_ctxt_in,
         const ArrayDesc &desc_wfc_in, const ArrayDesc &desc_band_wfc_in,
         bool is_mf_eigvec_k_distributed,
         const bool use_symmetry_context_in)
    : mf(mf_in),
      desc_wfc(desc_wfc_in),
      desc_band_wfc(desc_band_wfc_in),
      atbasis_wfc(atbasis_wfc_in),
      pbc(pbc_in),
      symmetry_context(symmetry_context_in),
      use_symmetry_context(use_symmetry_context_in),
      comm_h(kblacs_ctxt_in.comm_global_h),
      kblacs_ctxt(kblacs_ctxt_in),
      band_kblacs_ctxt(band_kblacs_ctxt_in)
{
    is_mf_eigvec_k_distributed_ = is_mf_eigvec_k_distributed;
    is_rspace_built_ = false;
    is_kspace_built_ = false;
    is_rspace_redist_for_KS_ = false;
    is_rspace_redist_blacs_ = false;

    // Runtime options
    libri_threshold_C = 0.0;
    libri_threshold_V = 0.0;
    libri_threshold_D = 0.0;
};

static ComplexMatrix extract_dmat_cplx_R_IJblock(const ComplexMatrix& dmat_cplx, const AtomicBasis &ab_wfc, const atom_t& I, const atom_t& J)
{
    const auto I_num = ab_wfc.get_atom_nb(I);
    const auto J_num = ab_wfc.get_atom_nb(J);
    ComplexMatrix dmat_cplx_IJR(I_num, J_num);
    for (size_t i = 0; i != I_num; i++)
    {
        size_t i_glo = ab_wfc.get_global_index(I, i);
        for (size_t j = 0; j != J_num; j++)
        {
            size_t j_glo = ab_wfc.get_global_index(J, j);
            dmat_cplx_IJR(i, j) = dmat_cplx(i_glo, j_glo);
        }
    }
    return dmat_cplx_IJR;
}


static void warn_dmat_IJR_nonzero_imag(const ComplexMatrix& dmat_cplx, const int& ispin, const atom_t& I, const atom_t& J, const Vector3_Order<int> R)
{
    if (dmat_cplx.get_max_abs_imag() > 1e-2)
        global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning: complex-valued density matrix, spin %d IJR %zu %zu (%d, %d, %d)\n", ispin, I, J, R.x, R.y, R.z);
}


#ifdef LIBRPA_USE_LIBRI
static void build_dmat_libri_kserial(
    const MeanField &mf,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const PeriodicBoundaryData &pbc,
    const SymmetryContext &symmetry_context,
    const bool use_symmetry_context,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    const bool save_cplx,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> &dmat_libri,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> &dmat_libri_cplx)
{
    // for (const auto &[ik, mat]: mf.get_eigenvectors().at(ispin))
    // {
    //     global::ofs_myid << ik << std::endl;
    //     print_complex_matrix("eigenvector", mat, global::ofs_myid, true);
    // }
    global::profiler.start("exx_build_dmat_libri_kserial");
    std::map<Vector3_Order<int>, std::vector<atpair_t>> map_R_IJs;
    for (const auto &IJR: IJRs)
    {
        const auto &R = IJR.second;
        map_R_IJs[R].push_back(IJR.first);
    }
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
    auto member_kfrac_targets = restore_symmetry_kstars_from_full_grid
        ? build_symmetry_full_grid_kstar_member_kfrac_targets(symmetry_context, kfrac_list)
        : restore_symmetry_kstars
        ? build_symmetry_kstar_member_kfrac_targets(symmetry_context, pbc)
        : symmetry_kstar_member_kfrac_targets_t{};
    // Full-grid wavefunctions can carry a gauge that is not reproduced exactly from k-star
    // metadata. Keep the representative route only when a cheap sample matches direct full-k.
    if (restore_symmetry_kstars_from_full_grid && !map_R_IJs.empty())
    {
        constexpr double restore_check_tol = 1e-6;
        const auto& R_check = map_R_IJs.begin()->first;
        const auto restored_check = get_symmetry_restored_dmat_cplx_R(
            symmetry_context, wfc_layouts, mf, ispin, ispinor_bra, ispinor_ket, kfrac_list, R_check,
            atom_nw, &member_kfrac_targets, &full_grid_kstar_representatives);
        const auto direct_check =
            mf.get_dmat_cplx_R(ispin, ispinor_bra, ispinor_ket, kfrac_list, R_check);
        const auto diff = restored_check - direct_check;
        if (diff.get_max_abs() > restore_check_tol)
        {
            restore_symmetry_kstars_from_full_grid = false;
            member_kfrac_targets.clear();
        }
    }
    for (const auto &R_IJs: map_R_IJs)
    {
        const auto &R = R_IJs.first;
        const auto &IJs = R_IJs.second;
        std::array<int,3> Ra{R.x,R.y,R.z};
        const auto dmat_cplx = (restore_symmetry_kstars || restore_symmetry_kstars_from_full_grid)
            ? get_symmetry_restored_dmat_cplx_R(
                  symmetry_context, wfc_layouts, mf, ispin, ispinor_bra, ispinor_ket, kfrac_list, R, atom_nw,
                  &member_kfrac_targets,
                  restore_symmetry_kstars_from_full_grid ? &full_grid_kstar_representatives : nullptr)
            : mf.get_dmat_cplx_R(ispin, ispinor_bra, ispinor_ket, kfrac_list, R);
        // global::ofs_myid << R << std::endl;
        // print_complex_matrix("dmat_cplx[R]", dmat_cplx, global::ofs_myid, true);
        omp_lock_t dmat_lock;
        omp_init_lock(&dmat_lock);
#pragma omp parallel for schedule(dynamic)
        for (const auto &IJ: IJs)
        {
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto dmat_IJR = extract_dmat_cplx_R_IJblock(dmat_cplx, atbasis_wfc, I, J);
            if (save_cplx)
            {
                std::valarray<cplxdb> dmat_va(dmat_IJR.c, dmat_IJR.size);
                auto pdmat = std::make_shared<std::valarray<cplxdb>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri_cplx[I][{J, Ra}] = RI::Tensor<cplxdb>({size_t(dmat_IJR.nr), size_t(dmat_IJR.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
            else
            {
                warn_dmat_IJR_nonzero_imag(dmat_IJR, ispin, I, J, R);
                std::valarray<double> dmat_va(dmat_IJR.real().c, dmat_IJR.size);
                auto pdmat = std::make_shared<std::valarray<double>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(dmat_IJR.nr), size_t(dmat_IJR.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
        }
#pragma omp barrier
        omp_destroy_lock(&dmat_lock);
    }
    global::profiler.stop("exx_build_dmat_libri_kserial");
}

static void build_dmat_libri_kpara(
    const MeanField &mf,
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    const bool save_cplx,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> &dmat_libri,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> &dmat_libri_cplx)
{
    global::profiler.start("exx_build_dmat_libri_kpara");

    std::map<Vector3_Order<int>, std::vector<atpair_t>> map_R_IJs;
    for (const auto &[IJ, R]: IJRs)
    {
        map_R_IJs[R].emplace_back(IJ);
    }
    std::vector<Vector3_Order<int>> Rs_this;
    for (const auto &[R, _]: map_R_IJs)
        Rs_this.emplace_back(R);
    const int n_Rs_this = map_R_IJs.size();
    int n_Rs_max = n_Rs_this;
    MPI_Allreduce(MPI_IN_PLACE, &n_Rs_max, 1, MPI_INT, MPI_MAX, comm_h.comm);
    const auto dmat_Rs_cplx = get_dmat_cplx_Rs_kpara(ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, Rs_this, comm_h);
    // global::ofs_myid << kfrac_list << std::endl;
    for (const auto &R_dmat_cplx: dmat_Rs_cplx)
    {
        const auto &R = R_dmat_cplx.first;
        const auto &dmat_cplx = R_dmat_cplx.second;
        // global::ofs_myid << R << std::endl;
        // print_complex_matrix("test", dmat_cplx, global::ofs_myid, true);
        std::array<int,3> Ra{R.x,R.y,R.z};
        const auto &map_IJs = map_R_IJs.at(R);
        omp_lock_t dmat_lock;
        omp_init_lock(&dmat_lock);
#pragma omp parallel for schedule(dynamic)
        for (const auto &IJ: map_IJs)
        {
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto dm_block = extract_dmat_cplx_R_IJblock(dmat_cplx, atbasis_wfc, I, J);
            if (save_cplx)
            {
                std::valarray<cplxdb> dmat_va(dm_block.c, dm_block.size);
                auto pdmat = std::make_shared<std::valarray<cplxdb>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri_cplx[I][{J, Ra}] = RI::Tensor<cplxdb>({size_t(dm_block.nr), size_t(dm_block.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
            else
            {
                warn_dmat_IJR_nonzero_imag(dm_block, ispin, I, J, R);
                std::valarray<double> dmat_va(dm_block.real().c, dm_block.size);
                auto pdmat = std::make_shared<std::valarray<double>>();
                *pdmat = dmat_va;
                omp_set_lock(&dmat_lock);
                dmat_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(dm_block.nr), size_t(dm_block.nc)}, pdmat);
                omp_unset_lock(&dmat_lock);
            }
        }
#pragma omp barrier
        omp_destroy_lock(&dmat_lock);
    }
    global::profiler.stop("exx_build_dmat_libri_kpara");
}

static void build_dmat_libri_kblacs_para(
    const MeanField &mf,
    const KPointBlacsParallelContext &kblacs_ctxt,
    const ArrayDesc &desc_wfc, const ArrayDesc &desc_dm,
    const IndexScheduler &sched,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const PeriodicBoundaryData &pbc,
    const SymmetryContext &symmetry_context,
    const bool use_symmetry_context,
    const std::vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<Vector3_Order<int>> Rs,
    const bool save_cplx,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> &dmat_libri,
    std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> &dmat_libri_cplx)
{
    global::profiler.start("exx_build_dmat_libri_kblacs_para");

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    const bool restore_symmetry_kstars =
        use_symmetry_context
        && can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw);
    global::ofs_myid << "EXX kBLACS Dmat symmetry restore: "
                     << (restore_symmetry_kstars ? "on" : "off") << std::endl;
    auto dmat_Rs_cplx = restore_symmetry_kstars
        ? get_symmetry_restored_dmat_cplx_Rs_kblacs_para(
              ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, Rs, kblacs_ctxt,
              desc_wfc, desc_dm, symmetry_context, pbc, atbasis_wfc)
        : get_dmat_cplx_Rs_kblacs_para(
              ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, Rs, kblacs_ctxt,
              desc_wfc, desc_dm);
    // global::ofs_myid << kfrac_list << std::endl;
    for (auto &R_dmat_cplx: dmat_Rs_cplx)
    {
        auto &R = R_dmat_cplx.first;
        auto &mat_blacs = R_dmat_cplx.second;
        auto pair_mat = get_ap_map_from_blacs_dist_scheduler(mat_blacs, sched, atbasis_wfc, atbasis_wfc, desc_dm);
        for (auto &[pair, mat_ap]: pair_mat)
        {
            const auto &I = as_int(pair.first);
            const auto &J = as_int(pair.second);
            const auto &n_I = atbasis_wfc.get_atom_nb(I);
            const auto &n_J = atbasis_wfc.get_atom_nb(J);
            mat_ap.swap_to_row_major();
            if (save_cplx)
                dmat_libri_cplx[I][{J, {R.x, R.y, R.z}}] = RI::Tensor<cplxdb>({n_I, n_J}, mat_ap.sptr());
            else
                dmat_libri[I][{J, {R.x, R.y, R.z}}] = RI::Tensor<double>({n_I, n_J}, mat_ap.get_real().sptr());
        }
        mat_blacs.clear();
    }
    global::profiler.stop("exx_build_dmat_libri_kblacs_para");
}
#endif

void Exx::write_rspace_matrices_binary() const
{
#ifdef LIBRPA_USE_LIBRI
    using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
    const int n_atoms = as_int(atbasis_wfc.n_atoms);
    for (int isp = 0; isp != mf.get_n_spins(); ++isp)
        for (int bra = 0; bra != mf.get_n_spinor(); ++bra)
            for (int ket = 0; ket != mf.get_n_spinor(); ++ket)
            {
                auto write_channel = [&](const auto &blocks)
                {
                    for (int I = 0; I != n_atoms; ++I)
                    {
                        std::map<int, std::map<std::pair<int, std::array<int, 3>>,
                                               RI::Tensor<cplxdb>>> local;
                        const auto channel = find_nested_int_map_3(blocks, isp, bra, ket);
                        if (channel != nullptr && channel->count(I))
                            for (const auto &[J, R_blocks] : channel->at(I))
                                for (const auto &[R, mat] : R_blocks)
                                {
                                    auto block = make_exx_ijk_complex_block(mat);
                                    local[I][{as_int(J), {R.x, R.y, R.z}}] =
                                        RI::Tensor<cplxdb>({as_size(block.nr()), as_size(block.nc())},
                                                          block.sptr());
                                }
                        std::set<int> rows, columns;
                        if (comm_h.is_root())
                        {
                            rows.insert(I);
                            for (int J = 0; J != n_atoms; ++J)
                                columns.insert(J);
                        }
                        // Sum distributed contributions before writing rank-independent files.
                        const auto gathered = comm_map2_first(comm_h.comm, local, rows, columns);
                        int write_ok = 1;
                        for (const auto &[atom, JR_blocks] : gathered)
                            for (const auto &[JR, tensor] : JR_blocks)
                            {
                                const auto &R = JR.second;
                                std::ostringstream fn;
                                fn << path_as_directory(output_dir) << "ExxR_ispin_" << isp
                                   << "_spinor_" << bra << "_" << ket << "_I_" << atom
                                   << "_J_" << JR.first << "_R_" << R[0] << "_" << R[1]
                                   << "_" << R[2] << ".bin";
                                const Matz block(as_int(atbasis_wfc.get_atom_nb(atom)),
                                                 as_int(atbasis_wfc.get_atom_nb(JR.first)),
                                                 tensor.data, MAJOR::ROW);
                                const ap_p_map<std::map<Vector3_Order<int>, Matz>> output_blocks{
                                    {{as_atom(atom), as_atom(JR.first)},
                                     {{{R[0], R[1], R[2]}, block * (-1.0)}}}};
                                if (!librpa_int::write_rspace_matrices_binary(
                                        output_blocks, atbasis_wfc, pbc, fn.str()))
                                    write_ok = 0;
                            }
                        MPI_Allreduce(MPI_IN_PLACE, &write_ok, 1, MPI_INT, MPI_MIN, comm_h.comm);
                        if (!write_ok)
                            throw LIBRPA_RUNTIME_ERROR("failed to write real-space EXX matrix output");
                    }
                };
                if (mf.get_n_spinor() > 1)
                    write_channel(exx_IJR_cplx);
                else
                    write_channel(exx_IJR);
            }
#endif
}

void Exx::build(const LibrpaParallelRouting routing,
                const AtomicBasis &atbasis_abf, const Cs_LRI &Cs,
                const atpair_R_mat_t &coul_mat)
{
    using std::endl;
    using global::profiler;
    using global::ofs_myid;

    assert(routing == LIBRPA_ROUTING_LIBRI);

    if (this->is_rspace_built_)
    {
        return;
    }

    const auto &n_spins = this->mf.get_n_spins();
    const auto &n_spinor = this->mf.get_n_spinor();
    const auto &ab_wfc = this->atbasis_wfc;
    const int n_atoms = ab_wfc.n_atoms;
    const auto &Rlist = this->pbc.Rlist;

    // Use complex LibRI objects when 2-component wave function is used
    const bool use_complex_exx_r = n_spinor > 1 ? true : false;

#ifdef LIBRPA_USE_LIBRI
    if (comm_h.is_root())
    {
        global::lib_printf("Computing EXX orbital energy using LibRI\n");
    }
    comm_h.barrier();

    // Use either one
    RI::Exx<int, int, 3, double> exx_libri;
    RI::Exx<int, int, 3, cplxdb> exx_libri_cplx;
#ifdef LIBRPA_LIBRI_HAS_FINE_GRAINED_LOCK
    exx_libri.lri.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
    exx_libri_cplx.lri.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
#endif

    std::map<int,std::array<double,3>> atoms_pos;
    for (int i = 0; i < n_atoms; i++)
        atoms_pos.insert(std::pair<int, std::array<double, 3>>{i, {0, 0, 0}});
    const auto atom_nw = atbasis_wfc.get_atom_nb_map<int>();

    if (use_complex_exx_r)
    {
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
        exx_libri_cplx.lri.parallel =
            std::make_shared<RI::Parallel_LRI_Equally_Weighted<int, int, 3, cplxdb>>(atom_nw);
#endif
        exx_libri_cplx.set_parallel(comm_h.comm, atoms_pos, this->pbc.latvec_array,
                                    this->pbc.period_array);
    }
    else
    {
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
        exx_libri.lri.parallel =
            std::make_shared<RI::Parallel_LRI_Equally_Weighted<int, int, 3, double>>(atom_nw);
#endif
        exx_libri.set_parallel(comm_h.comm, atoms_pos, this->pbc.latvec_array,
                               this->pbc.period_array);
    }

    const auto &symmetry_ctx = this->symmetry_context;
    const auto wfc_layouts = atbasis_wfc.build_species_basis_layouts(symmetry_ctx.atom_to_type);
    const auto abf_layouts = atbasis_abf.build_species_basis_layouts(symmetry_ctx.atom_to_type);
    const auto n_full_rspace_blocks =
        static_cast<std::size_t>(n_atoms) * static_cast<std::size_t>(n_atoms) * Rlist.size();
    const bool symmetry_reduces_rspace =
        symmetry_ctx.count_irreducible_blocks() < n_full_rspace_blocks;
    const bool use_symmetry_exx =
        this->use_symmetry_context
        && symmetry_ctx.available
        && symmetry_species_layouts_match_atom_counts(
            wfc_layouts, symmetry_ctx.atom_to_type, atbasis_wfc.get_atom_nb_map())
        && !symmetry_ctx.irreducible_sector.empty()
        && !symmetry_ctx.rspace_sector_stars.empty()
        && !symmetry_ctx.rspace_operations.empty()
        && symmetry_ctx.atom_to_type.size() == static_cast<std::size_t>(n_atoms)
        && symmetry_ctx.input_coord_frac.size() == static_cast<std::size_t>(n_atoms)
        && symmetry_reduces_rspace;
    const auto libri_irreducible_sector =
        use_symmetry_exx
            ? convert_symmetry_irreducible_sector_to_libri(
                  symmetry_ctx.irreducible_sector, this->pbc.period_array)
            : std::map<std::pair<int, int>, std::set<std::array<int, 3>>>{};
    const auto& symmetry_sector_stars = symmetry_ctx.rspace_sector_stars;
    // Phase 7: under a general (g, U_s, eta) operation the four (bra, ket)
    // spin channels of the static EXX operator mix, exactly like the Phase 6
    // GW self-energy. When active, the per-channel LibRI Hs are stashed
    // unrestored and restored jointly with the shared four-block kernel.
    const bool defer_spinor_exx_restore =
        use_symmetry_exx && use_complex_exx_r && n_spinor == 2
        && !symmetry_ctx.spin_operations.empty();
    if (use_symmetry_exx)
    {
        global::lib_printf(
            "Reducing EXX real-space contractions with irreducible sectors\n");
        exx_libri.set_symmetry(false, {});
        exx_libri_cplx.set_symmetry(false, {});
        if (use_complex_exx_r)
        {
            exx_libri_cplx.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_Atom_Symmetry<int, std::array<int, 3>, cplxdb>>(
                    exx_libri_cplx.lri.period, libri_irreducible_sector);
        }
        else
        {
            exx_libri.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_Atom_Symmetry<int, std::array<int, 3>, double>>(
                    exx_libri.lri.period, libri_irreducible_sector);
        }
    }
    else
    {
        exx_libri.set_symmetry(false, {});
        exx_libri_cplx.set_symmetry(false, {});
    }

    // Initialize Cs libRI container on each process
    // Note: we use different treatment in different routings
    //     R-tau routing:
    //         Each process has a full Cs copy.
    //         Thus in each process we only pass a few to LibRI container.
    //     atom-pair routing:
    //         Cs is already distributed across all processes.
    //         Pass the all Cs to libRI container.

    profiler.start("build_real_space_exx_1", "Prepare C libRI object");
    ofs_myid << "Number of Cs keys: " << get_num_keys(Cs.data_libri) << endl;
    // print_keys(global::ofs_myid, Cs.data_libri);
    global::ofs_myid << "comm_h.comm " << comm_h.comm << " global::mpi_comm_global_h.comm " << global::mpi_comm_global_h.comm << std::endl;
    // global::ofs_myid << Cs.data_libri << std::endl;
    if (use_complex_exx_r)
    {
        std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<cplxdb>>> data_libri;
        for (const auto &I_JR_C : Cs.data_libri)
        {
            const auto I = I_JR_C.first;
            for (const auto &JR_C : I_JR_C.second)
            {
                const auto J = JR_C.first.first;
                const auto R = JR_C.first.second;
                const auto &C = JR_C.second;
                auto JR = std::pair<int, std::array<int, 3>>(J, R);
                data_libri[I][JR] = RI::Global_Func::convert<cplxdb>(C);
            }
        }
        exx_libri_cplx.set_Cs(data_libri, libri_threshold_C);
    }
    else
        exx_libri.set_Cs(Cs.data_libri, libri_threshold_C);
    // exx_libri.set_Cs({}, libri_threshold_C);
    profiler.stop("build_real_space_exx_1");
    ofs_myid << "Finished setup Cs for EXX" << endl;
    std::flush(global::ofs_myid);

    // initialize Coulomb matrix
    global::profiler.start("build_real_space_exx_2", "Prepare V libRI object");

    atpair_R_mat_t exx_coul_mat_restored;
    const atpair_R_mat_t* exx_coul_mat_ptr = &coul_mat;
    const bool use_input_exx_coulomb_restore =
        use_symmetry_exx
        && symmetry_species_layouts_match_atom_counts(
            abf_layouts, symmetry_ctx.atom_to_type, atbasis_abf.get_atom_nb_map())
        && exx_coulomb_uses_symmetry_irreducible_sector_layout(
            coul_mat, symmetry_ctx, Rlist.size());
    if (use_input_exx_coulomb_restore)
    {
        global::lib_printf(
            "Restoring EXX auxiliary Coulomb blocks from the irreducible sector to the full real-space sector before LibRI contraction\n");
        for (const auto& I_JRV : coul_mat)
        {
            const auto ir_I = I_JRV.first;
            for (const auto& J_RV : I_JRV.second)
            {
                const auto ir_J = J_RV.first;
                const auto sector_pair =
                    std::make_pair(static_cast<atom_t>(ir_I), static_cast<atom_t>(ir_J));
                const auto pair_iter = symmetry_sector_stars.find(sector_pair);
                if (pair_iter == symmetry_sector_stars.end())
                {
                    throw std::runtime_error(
                        "Failed to match an irreducible EXX Coulomb atom pair with the symmetry restore map");
                }
                for (const auto& R_V : J_RV.second)
                {
                    const auto& ir_R = R_V.first;
                    const auto star_iter = pair_iter->second.find(ir_R);
                    if (star_iter == pair_iter->second.end())
                    {
                        throw std::runtime_error(
                            "Failed to match an irreducible EXX Coulomb real-space block with the symmetry restore map");
                    }
                    const ComplexMatrix v_ir(*R_V.second);
                    for (const auto& restore_member : star_iter->second)
                    {
                        // Charge-channel Coulomb blocks: the spin trace cancels
                        // U_s, and the antiunitary conjugation is a no-op on
                        // this real storage (V(R) is real), so only the orbital
                        // rotation is needed even for eta = 1 members.
                        const ComplexMatrix v_full =
                            librpa_int::rotate_symmetry_rspace_block(
                                symmetry_ctx, abf_layouts, restore_member.isym,
                                static_cast<atom_t>(ir_I),
                                static_cast<atom_t>(ir_J), v_ir);
                        const matrix v_full_real = v_full.real();
                        auto& target_pair_map =
                            exx_coul_mat_restored[restore_member.full_atom_pair.first]
                                                 [restore_member.full_atom_pair.second];
                        if (target_pair_map.count(restore_member.full_R) != 0)
                        {
                            throw std::runtime_error(
                                "Duplicate full-sector EXX Coulomb block appears during symmetry restore");
                        }
                        target_pair_map[restore_member.full_R] =
                            std::make_shared<matrix>(v_full_real);
                    }
                }
            }
        }
        exx_coul_mat_ptr = &exx_coul_mat_restored;
    }
    const auto& exx_coul_mat = *exx_coul_mat_ptr;
    const bool use_replicated_symmetry_exx_coulomb =
        use_input_exx_coulomb_restore && comm_h.nprocs > 1;

    std::map<int, std::map<std::pair<int,std::array<int,3>>, RI::Tensor<double>>> V_libri;
    std::map<int, std::map<std::pair<int,std::array<int,3>>, RI::Tensor<cplxdb>>> V_libri_cplx;

    global::profiler.start("build_real_space_exx_2_1");
    if (routing == LIBRPA_ROUTING_RTAU)
    {
        // Full Coulomb case, have to re-distribute
        // TODO: remove as libri routing is enforced above
        for (auto IJR: dispatch_vector_prod(get_atom_pair(exx_coul_mat), Rlist, comm_h.myid, comm_h.nprocs, true, true))
        {
            const auto I = IJR.first.first;
            const auto J = IJR.first.second;
            const auto R = IJR.second;
            const auto& VIJR = exx_coul_mat.at(I).at(J).at(R);
            // debug
            // printf("I J R %zu %zu %d %d %d, max(V) %f\n", I, J, R.x, R.y, R.z, VIJR->max());
            std::array<int,3> Ra{R.x,R.y,R.z};
            if (use_complex_exx_r)
            {
                auto pv = std::make_shared<std::valarray<cplxdb>>(VIJR->size);
                for (size_t i = 0; i < VIJR->size; ++i)
                    (*pv)[i] = cplxdb(VIJR->c[i], 0.0);
                V_libri_cplx[I][{J, Ra}] = RI::Tensor<cplxdb>({size_t(VIJR->nr), size_t(VIJR->nc)}, pv);
            }
            else
            {
                auto pv = std::make_shared<std::valarray<double>>(VIJR->c, VIJR->size);
                V_libri[I][{J, Ra}] = RI::Tensor<double>({size_t(VIJR->nr), size_t(VIJR->nc)}, pv);
            }
        }
    }
    else
    {
        if (use_replicated_symmetry_exx_coulomb)
        {
            for (const auto& IJR : dispatch_vector_prod(
                     get_atom_pair(exx_coul_mat), Rlist, comm_h.myid, comm_h.nprocs, true, true))
            {
                const auto I = IJR.first.first;
                const auto J = IJR.first.second;
                const auto& R = IJR.second;
                if (exx_coul_mat.count(I) == 0 || exx_coul_mat.at(I).count(J) == 0
                    || exx_coul_mat.at(I).at(J).count(R) == 0)
                {
                    continue;
                }
                const auto& V = exx_coul_mat.at(I).at(J).at(R);
                std::array<int, 3> Ra{R.x, R.y, R.z};
                if (use_complex_exx_r)
                {
                    auto pv = std::make_shared<std::valarray<cplxdb>>(V->size);
                    for (size_t i = 0; i < V->size; ++i)
                        (*pv)[i] = cplxdb(V->c[i], 0.0);
                    V_libri_cplx[I][{J, Ra}] =
                        RI::Tensor<cplxdb>({size_t(V->nr), size_t(V->nc)}, pv);
                }
                else
                {
                    auto pv = std::make_shared<std::valarray<double>>(V->c, V->size);
                    V_libri[I][{J, Ra}] =
                        RI::Tensor<double>({size_t(V->nr), size_t(V->nc)}, pv);
                }
            }
        }
        else
        {
            for (const auto &I_JRV: exx_coul_mat)
            {
                const auto I = I_JRV.first;
                for (const auto &J_RV: I_JRV.second)
                {
                    const auto J = J_RV.first;
                    for (const auto &R_V: J_RV.second)
                    {
                        const auto &R = R_V.first;
                        const auto &V = R_V.second;
                        std::array<int,3> Ra{R.x,R.y,R.z};
                        if (use_complex_exx_r)
                        {
                            auto pv = std::make_shared<std::valarray<cplxdb>>(V->size);
                            for (size_t i = 0; i < V->size; ++i)
                                (*pv)[i] = cplxdb(V->c[i], 0.0);
                            V_libri_cplx[I][{J, Ra}] =
                                RI::Tensor<cplxdb>({size_t(V->nr), size_t(V->nc)}, pv);
                        }
                        else
                        {
                            auto pv = std::make_shared<std::valarray<double>>(V->c, V->size);
                            V_libri[I][{J, Ra}] =
                                RI::Tensor<double>({size_t(V->nr), size_t(V->nc)}, pv);
                        }
                    }
                }
            }
        }
    }
    global::profiler.stop("build_real_space_exx_2_1");

    // global::ofs_myid << V_libri << endl;
    global::profiler.start("build_real_space_exx_2_2");
    if (use_complex_exx_r)
    {
        global::ofs_myid << "Number of V keys: " << get_num_keys(V_libri_cplx) << endl;
        exx_libri_cplx.set_Vs(V_libri_cplx, libri_threshold_V);
        V_libri_cplx.clear();
    }
    else
    {
        global::ofs_myid << "Number of V keys: " << get_num_keys(V_libri) << endl;
        exx_libri.set_Vs(V_libri, libri_threshold_V);
        V_libri.clear();
    }
    // exx_libri.set_Vs({}, libri_threshold_V);
    release_free_mem();
    global::profiler.stop("build_real_space_exx_2_2");
    global::profiler.stop("build_real_space_exx_2");
    global::lib_printf("Task %4d: V setup for EXX\n", comm_h.myid);
    // cout << V_libri << endl;

    // initialize density matrix
    global::profiler.start("build_real_space_exx_prepare_index");
    std::vector<atpair_t> atpair_dmat;
    for (atom_t I = 0; I < as_atom(n_atoms); I++)
        for (atom_t J = 0; J < as_atom(n_atoms); J++)
            atpair_dmat.push_back({I, J});
    const auto dmat_IJRs_local = dispatch_vector_prod(atpair_dmat, Rlist, comm_h.myid, comm_h.nprocs, true, true);

    // For k-BLACS two-level parallelization
    const int n_basis_ao = mf.get_n_aos();
    const auto desc_dm = kblacs_ctxt.create_array_desc(n_basis_ao, n_basis_ao);
    IndexScheduler sched;
    const auto map_atpairs_balanced =
        get_balanced_ap_distribution_for_consec_descriptor(atbasis_wfc, atbasis_wfc, desc_dm);
    sched.init(map_atpairs_balanced, atbasis_wfc, atbasis_wfc, desc_dm, false);
    const auto iRs = dispatcher_balanced(0, Rlist.size(), kblacs_ctxt.kpoints_local().size(), true,
                                         kblacs_ctxt.comm_kpoint_h.comm);
    std::vector<Vector3_Order<int>> Rs;
    Rs.reserve(iRs.size());
    for (const auto &iR : iRs)
        Rs.push_back(Rlist[iR]);
    global::profiler.stop("build_real_space_exx_prepare_index");

    for (auto isp = 0; isp != n_spins; isp++)
    {
        // Irreducible-sector LibRI Hs stashed per (bra, ket) channel; only
        // populated when defer_spinor_exx_restore is true.
        std::array<
            std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>>, 4>
            exx_hs_ir_channels;
        for (auto ispn_bra = 0; ispn_bra != n_spinor; ispn_bra++)
        {
            for (auto ispn_ket = 0; ispn_ket != n_spinor; ispn_ket++)
            {
                global::profiler.start("build_real_space_exx_3", "Prepare DM libRI object");
                std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<double>>> dmat_libri;
                std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<cplxdb>>> dmat_libri_cplx;
                if (is_mf_eigvec_k_distributed_)
                {
                    // build_dmat_libri_kpara(mf, comm_h, atbasis_wfc, isp, ispn_bra, ispn_ket,
                    //                        this->pbc.kfrac_list, dmat_IJRs_local, use_complex_exx_r,
                    //                        dmat_libri, dmat_libri_cplx);
                    build_dmat_libri_kblacs_para(mf, kblacs_ctxt, desc_wfc, desc_dm, sched,
                                                 atbasis_wfc, isp, ispn_bra, ispn_ket,
                                                 this->pbc, this->symmetry_context,
                                                 this->use_symmetry_context,
                                                 this->pbc.kfrac_list, Rs, use_complex_exx_r,
                                                 dmat_libri, dmat_libri_cplx);
                }
                else
                {
                    build_dmat_libri_kserial(mf, atbasis_wfc, isp, ispn_bra, ispn_ket,
                                             this->pbc, this->symmetry_context,
                                             this->use_symmetry_context,
                                             this->pbc.kfrac_list, dmat_IJRs_local,
                                             use_complex_exx_r, dmat_libri, dmat_libri_cplx);
                }
                // global::ofs_myid << dmat_libri << std::endl;
                // print_keys(global::ofs_myid, dmat_libri);
                global::profiler.start("build_real_space_exx_libri_set_Ds");
                if (use_complex_exx_r)
                {
                    global::ofs_myid << "Number of Dmat keys: " << get_num_keys(dmat_libri_cplx) << "\n";
                    exx_libri_cplx.set_Ds(dmat_libri_cplx, libri_threshold_D);
                }
                else
                {
                    global::ofs_myid << "Number of Dmat keys: " << get_num_keys(dmat_libri) << "\n";
                    exx_libri.set_Ds(dmat_libri, libri_threshold_D);
                }
                // exx_libri.set_Ds({}, libri_threshold_D);
                global::profiler.stop("build_real_space_exx_libri_set_Ds");
                global::profiler.stop("build_real_space_exx_3");
                global::lib_printf("Task %4d: DM setup for EXX\n", comm_h.myid);

                global::profiler.start("build_real_space_exx_4", "Call libRI Hexx calculation");
                if (use_complex_exx_r)
                    exx_libri_cplx.cal_Hs();
                else
                    exx_libri.cal_Hs();
                global::profiler.stop("build_real_space_exx_4");
                release_free_mem();

                global::lib_printf(
                    "Task %4d: cal_Hs elapsed time: %f\n", comm_h.myid,
                    global::profiler.get_wall_time_last("build_real_space_exx_4"));
                // print_keys(global::ofs_myid, exx_libri.Hs);
                // ofs_myid << "exx_libri.Hs:\n" << exx_libri.Hs << endl;

                auto store_exx_block = [&](const atom_t full_I,
                                           const atom_t full_J,
                                           const Vector3_Order<int>& full_R,
                                           const ComplexMatrix& block)
                {
                    const auto n_full_I = ab_wfc.get_atom_nb(full_I);
                    const auto n_full_J = ab_wfc.get_atom_nb(full_J);
                    if (block.nr != as_int(n_full_I) || block.nc != as_int(n_full_J))
                    {
                        throw std::runtime_error(
                            "EXX real-space symmetry restore produced an AO block with an inconsistent dimension");
                    }
                    if (use_complex_exx_r)
                    {
                        this->exx_IJR_cplx[isp][ispn_bra][ispn_ket][full_I][full_J][full_R] =
                            Matz(n_full_I, n_full_J, block.c, MAJOR::ROW);
                    }
                    else
                    {
                        const auto block_real = block.real();
                        this->exx_IJR[isp][ispn_bra][ispn_ket][full_I][full_J][full_R] =
                            Matd(n_full_I, n_full_J, block_real.c, MAJOR::ROW);
                    }
                };

                auto copy_exx_blocks = [&](const auto &Hs)
                {
                    global::ofs_myid << "Number of exx_libri.Hs keys: "
                                     << get_num_keys(Hs) << std::endl;
                    if (use_symmetry_exx)
                    {
                        global::lib_printf(
                            "Restoring the symmetry-filtered LibRI EXX blocks to the full AO real-space sector\n");
                    }
                    for (const auto &I_JR_exx : Hs)
                    {
                        const auto &I = I_JR_exx.first;
                        const auto n_I = ab_wfc.get_atom_nb(I);
                        for (const auto &JR_exx : I_JR_exx.second)
                        {
                            const auto &J = JR_exx.first.first;
                            const auto n_J = ab_wfc.get_atom_nb(J);
                            const auto &Ra = JR_exx.first.second;
                            const auto R = Vector3_Order<int>{Ra[0], Ra[1], Ra[2]};
                            const auto block_ir =
                                convert_libri_tensor_to_complex_matrix(
                                    JR_exx.second, n_I, n_J);
                            if (use_symmetry_exx)
                            {
                                const auto sector_pair =
                                    std::make_pair(static_cast<atom_t>(I),
                                                   static_cast<atom_t>(J));
                                const auto pair_iter =
                                    symmetry_sector_stars.find(sector_pair);
                                if (pair_iter == symmetry_sector_stars.end()
                                    || pair_iter->second.count(R) == 0)
                                {
                                    std::ostringstream oss;
                                    oss << "Failed to match a symmetry-filtered EXX real-space block"
                                        << " with the irreducible-sector restore map for I="
                                        << I << " J=" << J << " R=(" << R.x << ","
                                        << R.y << "," << R.z << ")";
                                    throw std::runtime_error(oss.str());
                                }
                                for (const auto& restore_member : pair_iter->second.at(R))
                                {
                                    const auto block_full =
                                        librpa_int::rotate_symmetry_rspace_block(
                                            symmetry_ctx, wfc_layouts, restore_member.isym,
                                            static_cast<atom_t>(I),
                                            static_cast<atom_t>(J), block_ir);
                                    store_exx_block(
                                        restore_member.full_atom_pair.first,
                                        restore_member.full_atom_pair.second,
                                        restore_member.full_R,
                                        block_full);
                                }
                            }
                            else
                            {
                                store_exx_block(
                                    static_cast<atom_t>(I),
                                    static_cast<atom_t>(J),
                                    R,
                                    block_ir);
                            }
                        }
                    }
                };

                global::profiler.start("build_real_space_exx_5");
                if (use_complex_exx_r)
                {
                    if (defer_spinor_exx_restore)
                    {
                        // Stash the irreducible-sector Hs; the joint
                        // four-channel restore runs after the (bra, ket) loops.
                        exx_hs_ir_channels[static_cast<std::size_t>(ispn_bra * 2 + ispn_ket)] =
                            std::move(exx_libri_cplx.Hs);
                    }
                    else
                    {
                        copy_exx_blocks(exx_libri_cplx.Hs);
                    }
                }
                else
                    copy_exx_blocks(exx_libri.Hs);
                global::profiler.stop("build_real_space_exx_5");
            }
        }
        if (defer_spinor_exx_restore)
        {
            global::lib_printf(
                "Restoring the four spinor EXX channels jointly from the irreducible sector\n");
            std::array<symmetry_rspace_block_map_t, 4> blocks_ir;
            for (std::size_t channel = 0; channel != 4; ++channel)
            {
                for (const auto& I_JR : exx_hs_ir_channels[channel])
                {
                    const auto n_I = as_int(ab_wfc.get_atom_nb(I_JR.first));
                    for (const auto& JR : I_JR.second)
                    {
                        const auto n_J = as_int(ab_wfc.get_atom_nb(JR.first.first));
                        blocks_ir[channel][I_JR.first][JR.first] =
                            convert_libri_tensor_to_complex_matrix(JR.second, n_I, n_J);
                    }
                }
            }
            std::vector<int> atom_nb(as_size(n_atoms));
            for (int atom = 0; atom != n_atoms; ++atom)
            {
                atom_nb[atom] = as_int(ab_wfc.get_atom_nb(atom));
            }
            const auto blocks_full = restore_symmetry_spinor_rspace_blocks(
                blocks_ir, symmetry_ctx, symmetry_sector_stars, wfc_layouts, atom_nb);
            exx_hs_ir_channels = {};
            for (int bra = 0; bra != n_spinor; ++bra)
            {
                for (int ket = 0; ket != n_spinor; ++ket)
                {
                    const auto channel = static_cast<std::size_t>(bra * 2 + ket);
                    for (const auto& I_entry : blocks_full[channel])
                    {
                        const auto full_I = static_cast<atom_t>(I_entry.first);
                        const auto n_full_I = ab_wfc.get_atom_nb(full_I);
                        for (const auto& JR : I_entry.second)
                        {
                            const auto full_J = static_cast<atom_t>(JR.first.first);
                            const auto n_full_J = ab_wfc.get_atom_nb(full_J);
                            const auto& block = JR.second;
                            if (block.nr != as_int(n_full_I) || block.nc != as_int(n_full_J))
                            {
                                throw std::runtime_error(
                                    "EXX spinor symmetry restore produced an AO block with an inconsistent dimension");
                            }
                            const Vector3_Order<int> full_R{
                                JR.first.second[0], JR.first.second[1], JR.first.second[2]};
                            this->exx_IJR_cplx[isp][bra][ket][full_I][full_J][full_R] =
                                Matz(n_full_I, n_full_J, block.c, MAJOR::ROW);
                        }
                    }
                }
            }
        }
    }

    // debug, print the Hexx matrices
    // for (const auto& isp_IJkH: this->Hexx)
    // {
    //     const auto& isp = isp_IJkH.first;
    //     for (const auto& I_JkH: isp_IJkH.second)
    //     {
    //         const auto& I = I_JkH.first;
    //         for (const auto& J_kH: I_JkH.second)
    //         {
    //             const auto& J = J_kH.first;
    //             for (const auto& k_H: J_kH.second)
    //             {
    //                 cout << isp << " " << I << " " << J << " {" << k_H.first << "} whole size: " << k_H.second->size << endl;
    //                 print_complex_matrix("", *k_H.second);
    //             }
    //         }
    //     }
    // }

#else
    if (comm_h.is_root())
    {
        global::lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error: trying build EXX orbital energy with LibRI, but the program is not compiled against LibRI\n");
    }
    throw std::logic_error("compilation");
    comm_h.barrier();
#endif

    release_free_mem();
    if (output_exx_mat_r)
        write_rspace_matrices_binary();
    is_rspace_built_= true;
}

void Exx::build_KS(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                   const std::vector<Vector3_Order<double>> &kfrac_target,
                   const AtomPairBvKRemap<atom_t> &bvk_remap)
{
    throw LIBRPA_RUNTIME_ERROR("not implemented");
}

void Exx::build_KS_blacs(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                         const std::vector<Vector3_Order<double>> &kfrac_target,
                         const AtomPairBvKRemap<atom_t> &bvk_remap,
                         const BlacsCtxtHandler &blacs_ctxt_h,
                         bool use_gpu_replace_scalapack,
                         const std::string &source)
{
    using RI::Communicate_Tensors_Map_Judge::comm_map2;
    using RI::Communicate_Tensors_Map_Judge::comm_map2_first;

    // Ensure the communicator of BLACS context is the same as that parsed when constructed
    assert(blacs_ctxt_h.comm() == this->comm_h.comm);
    assert(this->is_rspace_built_);

    // Reset k-space matrices built from last call
    if (this->is_kspace_built_)
    {
        global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning: reset EXX k-space matrices\n");
        this->reset_kspace();
    }

    // NOTE: Here it assumes that wfc_target has the same spin, spinor and states dimensions
    const auto n_aos = this->mf.get_n_aos();
    const auto n_spins = this->mf.get_n_spins();
    const auto n_bands = this->mf.get_n_bands();
    const auto n_spinor = this->mf.get_n_spinor();
    const bool use_complex_exx_r = n_spinor > 1 ? true : false;
    const int n_target_kpoints = static_cast<int>(kfrac_target.size());
    const bool use_klocal_rotation = is_mf_eigvec_k_distributed_;
    const bool target_is_band_path = source.rfind("band_", 0) == 0;
    const KPointBlacsParallelContext &target_kblacs_ctxt =
        target_is_band_path ? band_kblacs_ctxt : kblacs_ctxt;
    if (use_klocal_rotation)
    {
        if (!target_kblacs_ctxt.is_initialized())
            throw LIBRPA_RUNTIME_ERROR("k-point BLACS context is not initialized for EXX rotation");
        if (target_kblacs_ctxt.n_kpoints() != n_target_kpoints)
            throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of target EXX k-points");
    }
    const BlacsCtxtHandler &rotation_blacs_h =
        use_klocal_rotation ? target_kblacs_ctxt.blacs_h : blacs_ctxt_h;
    const ArrayDesc &desc_wfc_target = target_is_band_path ? desc_band_wfc : desc_wfc;
    if (use_klocal_rotation &&
        (!desc_wfc_target.is_initialized() || desc_wfc_target.m() != n_aos ||
         desc_wfc_target.n() != n_bands ||
         desc_wfc_target.ictxt() != rotation_blacs_h.ictxt))
        throw LIBRPA_RUNTIME_ERROR("EXX source wave-function descriptor is inconsistent");

    // prepare scalapack array descriptors
    ArrayDesc desc_nao_nao(rotation_blacs_h);
    ArrayDesc desc_nband_nao(rotation_blacs_h);
    ArrayDesc desc_nband_nband(rotation_blacs_h);

    // For communication of eigenvectors and final KS matrices
    ArrayDesc desc_nao_nband_fb(rotation_blacs_h);  // emulate ComplexMatrix storage
    ArrayDesc desc_nao_nao_fb(rotation_blacs_h);
    ArrayDesc desc_nband_nband_fb(rotation_blacs_h);

    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);
    const bool use_root_dense_projection =
        this->use_symmetry_context
        && use_symmetry_ibz_root_projection(this->symmetry_context,
                                               this->pbc,
                                               n_target_kpoints,
                                               this->mf.get_n_kpoints());
    if (use_root_dense_projection)
    {
        desc_nao_nao_fb.init(n_aos, n_aos, n_aos, n_aos, 0, 0);
    }

    // local 2D-block submatrices
    auto Hexx_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto Hexx_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);
    auto temp_nband_nao = init_local_mat<complex<double>>(desc_nband_nao, MAJOR::COL);

    // opt block-size descriptors for better pgemm performance
    const int expected_block_nao =
        get_capped_blacs_block_size(n_aos, wfc_gemm_block_size_opt, rotation_blacs_h);
    const int expected_block_nband =
        get_capped_blacs_block_size(n_bands, wfc_gemm_block_size_opt, rotation_blacs_h);
    if (use_klocal_rotation &&
        (desc_wfc_target.mb() != expected_block_nao ||
         desc_wfc_target.nb() != expected_block_nband))
        throw LIBRPA_RUNTIME_ERROR(
            "EXX source wave functions do not use the permanent capped rectangular layout");
    const int block_nao =
        use_klocal_rotation ? desc_wfc_target.mb() : expected_block_nao;
    const int block_nband =
        use_klocal_rotation ? desc_wfc_target.nb() : expected_block_nband;
    ArrayDesc desc_nband_nao_opt(rotation_blacs_h), desc_nao_nao_opt(rotation_blacs_h);
    ArrayDesc desc_wfc_device(rotation_blacs_h), desc_nband_nband_opt(rotation_blacs_h);
    desc_nband_nao_opt.init(n_bands, n_aos, block_nband, block_nao, 0, 0);
    desc_nao_nao_opt.init(n_aos, n_aos, block_nao, block_nao, 0, 0);
    desc_nband_nband_opt.init(n_bands, n_bands, block_nband, block_nband, 0, 0);
    desc_wfc_device.init(n_aos, n_bands, block_nao, block_nband,
                         use_klocal_rotation ? desc_wfc_target.irsrc() : 0,
                         use_klocal_rotation ? desc_wfc_target.icsrc() : 0);
    desc_exx_KS.reset_handler(rotation_blacs_h);
    if (use_klocal_rotation)
        desc_exx_KS.init(n_bands, n_bands, block_nband, block_nband, 0, 0);
    else
        desc_exx_KS.init(n_bands, n_bands, n_bands, n_bands, 0, 0);

    auto Hexx_nao_nao_opt = init_local_mat<complex<double>>(desc_nao_nao_opt, MAJOR::COL);
    auto Hexx_nband_nband_opt = init_local_mat<complex<double>>(desc_nband_nband_opt, MAJOR::COL);
    auto temp_nband_nao_opt = init_local_mat<complex<double>>(desc_nband_nao_opt, MAJOR::COL);

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    std::complex<double> *d_wfc_bra = nullptr, *d_wfc_ket = nullptr,
                         *d_hexx_nao = nullptr, *d_temp = nullptr,
                         *d_hexx_nband = nullptr;
    size_t size_wfc = 0, size_hexx_nao = 0, size_temp = 0, size_hexx_nband = 0;
    if (use_gpu_replace_scalapack && use_klocal_rotation)
    {
        auto &rotation_blacs_h_nc = const_cast<BlacsCtxtHandler &>(rotation_blacs_h);
        if (rotation_blacs_h_nc.ddla_handle == nullptr)
            rotation_blacs_h_nc.init_ddla_handle();
        desc_wfc_device.set_ddla_desc(rotation_blacs_h.ddla_handle);
        desc_nao_nao_opt.set_ddla_desc(rotation_blacs_h.ddla_handle);
        desc_nband_nao_opt.set_ddla_desc(rotation_blacs_h.ddla_handle);
        desc_nband_nband_opt.set_ddla_desc(rotation_blacs_h.ddla_handle);

        size_wfc = static_cast<size_t>(desc_wfc_device.m_loc()) * desc_wfc_device.n_loc();
        size_hexx_nao = static_cast<size_t>(desc_nao_nao_opt.m_loc()) * desc_nao_nao_opt.n_loc();
        size_temp = static_cast<size_t>(desc_nband_nao_opt.m_loc()) * desc_nband_nao_opt.n_loc();
        size_hexx_nband = static_cast<size_t>(desc_nband_nband_opt.m_loc()) * desc_nband_nband_opt.n_loc();
        DEVICE_CHECK(deviceMallocAsync((void**)&d_wfc_bra, std::max<size_t>(size_wfc, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_wfc_ket, std::max<size_t>(size_wfc, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_hexx_nao, std::max<size_t>(size_hexx_nao, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_temp, std::max<size_t>(size_temp, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_hexx_nband, std::max<size_t>(size_hexx_nband, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
    }
#endif

    const auto set_IJ_nao_nao_rotation = get_necessary_IJ_from_block_2D(
        this->atbasis_wfc, this->atbasis_wfc, desc_nao_nao);

    if (!use_klocal_rotation && !is_rspace_redist_for_KS_)
    {
        global::profiler.start("exx_build_KS_blacs_redist");
        const auto Iset_Jset = convert_IJset_to_Iset_Jset(set_IJ_nao_nao_rotation);

        auto pack_real_exx_to_tensor = [&](const auto &exx_blocks)
        {
            std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> exx_tensor;
            for (const auto &[I, J_Rexx] : exx_blocks)
            {
                const auto n_I = this->atbasis_wfc.get_atom_nb(I);
                for (const auto &[J, R_exx] : J_Rexx)
                {
                    const auto n_J = this->atbasis_wfc.get_atom_nb(J);
                    for (const auto &[R, mat] : R_exx)
                    {
                        const std::array<int, 3> Ra{R.x, R.y, R.z};
                        exx_tensor[I][{J, Ra}] = RI::Tensor<double>({n_I, n_J}, mat.sptr());
                    }
                }
            }
            return exx_tensor;
        };

        auto pack_cplx_exx_to_tensor = [&](const auto &exx_blocks)
        {
            std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>> exx_tensor;

            for (const auto &[I, J_Rexx] : exx_blocks)
            {
                const auto n_I = this->atbasis_wfc.get_atom_nb(I);
                for (const auto &[J, R_exx] : J_Rexx)
                {
                    const auto n_J = this->atbasis_wfc.get_atom_nb(J);
                    for (const auto &[R, mat] : R_exx)
                    {
                        const std::array<int, 3> Ra{R.x, R.y, R.z};
                        exx_tensor[I][{J, Ra}] = RI::Tensor<cplxdb>({n_I, n_J}, mat.sptr());
                    }
                }
            }
            return exx_tensor;
        };

        for (int isp = 0; isp < n_spins; isp++)
        {
            for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
            {
                for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
                {
                    const auto exx = use_complex_exx_r
                                         ? nullptr
                                         : find_nested_int_map_3(exx_IJR, isp, ispn_bra, ispn_ket);
                    const auto exx_cplx = use_complex_exx_r
                                         ? find_nested_int_map_3(exx_IJR_cplx, isp, ispn_bra, ispn_ket)
                                         : nullptr;

                    if (use_complex_exx_r)
                    {
                        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>> exx_tensor_cplx;
                        if (exx_cplx != nullptr)
                            exx_tensor_cplx = pack_cplx_exx_to_tensor(*exx_cplx);
                        global::profiler.start("exx_build_KS_blacs_redist_comm_map2");
                        auto exx_redist = comm_map2_first(comm_h.comm, exx_tensor_cplx, Iset_Jset.first, Iset_Jset.second);
                        global::profiler.stop("exx_build_KS_blacs_redist_comm_map2");

                        std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matz>>> exx_is_new;
                        for (const auto &[I, JRmat]: exx_redist)
                        {
                            const int n_I = this->atbasis_wfc.get_atom_nb(I);
                            for (const auto &[JR, mat]: JRmat)
                            {
                                const atom_t J = JR.first;
                                const int n_J = this->atbasis_wfc.get_atom_nb(J);
                                const auto &Ra = JR.second;
                                const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};
                                exx_is_new[I][J][R] = Matz{n_I, n_J, mat.data, MAJOR::ROW};
                            }
                        }
                        if (exx_cplx != nullptr)
                            exx_cplx->swap(exx_is_new);
                        else if (!exx_is_new.empty())
                            exx_IJR_cplx[isp][ispn_bra][ispn_ket] = std::move(exx_is_new);
                    }
                    else
                    {
                        std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> exx_tensor;
                        if (exx != nullptr)
                            exx_tensor = pack_real_exx_to_tensor(*exx);
                        global::profiler.start("exx_build_KS_blacs_redist_comm_map2");
                        auto exx_redist = comm_map2_first(comm_h.comm, exx_tensor, Iset_Jset.first, Iset_Jset.second);
                        global::profiler.stop("exx_build_KS_blacs_redist_comm_map2");

                        std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matd>>> exx_is_new;
                        for (const auto &[I, JRmat]: exx_redist)
                        {
                            const int n_I = this->atbasis_wfc.get_atom_nb(I);
                            for (const auto &[JR, mat]: JRmat)
                            {
                                const atom_t J = JR.first;
                                const int n_J = this->atbasis_wfc.get_atom_nb(J);
                                const auto &Ra = JR.second;
                                const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};
                                exx_is_new[I][J][R] = Matd{n_I, n_J, mat.data, MAJOR::ROW};
                            }
                        }
                        if (exx != nullptr)
                            exx->swap(exx_is_new);
                        else if (!exx_is_new.empty())
                            exx_IJR[isp][ispn_bra][ispn_ket] = std::move(exx_is_new);
                    }
                }
            }
        }
        is_rspace_redist_for_KS_ = true;
        is_rspace_redist_blacs_ = true;
        global::profiler.stop("exx_build_KS_blacs_redist");
        global::lib_printf("Task %4d: tensor communicate elapsed time: %f\n", comm_h.myid,
                           global::profiler.get_wall_time_last("exx_build_KS_blacs_redist"));
    }
    else if (!use_klocal_rotation)
    {
        if (!is_rspace_redist_blacs_)
            throw LIBRPA_RUNTIME_ERROR("");
    }

    global::ofs_myid << "period:      " << this->pbc.period << std::endl;
    global::ofs_myid << "bvk_remap size: " << bvk_remap.size() << std::endl;
    global::ofs_myid << "latvec:      " << this->pbc.latvec << std::endl;

    auto shift_bvk = [&](const auto &map_orig, auto &map_shift)
    {
        auto add_mat = [](auto &R_mat_shift, const Vector3_Order<int> &R_bvk,
                          const auto &mat, const double weight)
        {
            auto mat_weighted = mat.copy();
            mat_weighted *= weight;
            auto [it, inserted] = R_mat_shift.emplace(R_bvk, mat_weighted);
            if (!inserted) it->second += mat_weighted;
        };

        for (const auto &[I, J_Rmat] : map_orig)
        {
            for (const auto &[J, R_mat] : J_Rmat)
            {
                const atpair_t IJ{I, J};
                auto &R_mat_shift = map_shift[I][J];

                for (const auto &[R, mat] : R_mat)
                {
                    const auto *R_bvks = bvk_remap.find_R_bvk(IJ, R);
                    if (R_bvks == nullptr || R_bvks->empty())
                    {
                        R_mat_shift[R] = mat;
                    }
                    else if (R_bvks->size() == 1)
                    {
                        R_mat_shift[R_bvks->front()] = mat;
                    }
                    else
                    {
                        const auto weight = 1.0 / static_cast<double>(R_bvks->size());
                        for (const auto &R_bvk: *R_bvks)
                        {
                            add_mat(R_mat_shift, R_bvk, mat, weight);
                        }
                    }
                }
            }
        }
    };

    using ExxIJKAtomKey = std::size_t;
    using ExxIJKKey = std::pair<ExxIJKAtomKey, int>; // (J, ik); J stays first for atom-pair routing.
    std::set<ExxIJKAtomKey> exx_ijk_s0;
    std::set<ExxIJKKey> exx_ijk_s1;
    if (use_klocal_rotation)
    {
        for (const auto &IJ: set_IJ_nao_nao_rotation)
        {
            exx_ijk_s0.insert(IJ.first);
            for (const int ik: target_kblacs_ctxt.kpoints_local())
                exx_ijk_s1.insert({IJ.second, ik});
        }
    }
    for (int isp = 0; isp < n_spins; isp++)
    {
        this->exx_KS[isp] = {};
        this->Eexx[isp] = {};
        for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
        {
            for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
            {
                std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matd>>> exx_is_local;
                std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matz>>> exx_is_local_cplx;
                // Convert each <I,<J, R>> pair to the configured BvK counterpart to speed up later
                // Fourier transform while keeping the accuracy in further band interpolation.
                // Reuse the cleared-up exx_I_JR_local object
                auto orig = find_nested_int_map_3(exx_IJR, isp, ispn_bra, ispn_ket);
                auto orig_cplx = find_nested_int_map_3(exx_IJR_cplx, isp, ispn_bra, ispn_ket);
                if (!use_klocal_rotation && use_complex_exx_r)
                {
                    if (orig_cplx != nullptr)
                    {
                        shift_bvk(*orig_cplx , exx_is_local_cplx);
                    }
                }
                else if (!use_klocal_rotation)
                {
                    if (orig != nullptr)
                    {
                        shift_bvk(*orig, exx_is_local);
                    }
                }

                if (use_klocal_rotation)
                {
                    global::profiler.start("exx_build_KS_rotate_kpara_klocal_blacs");
                    std::map<ExxIJKAtomKey, std::map<ExxIJKKey, Matz>> exx_I_Jik_mat;
                    global::profiler.start("exx_build_KS_fourier_world");
                    auto fourier_exx_to_ijk =
                        [&](const auto &exx_blocks)
                    {
                        using ExxBlocks = std::decay_t<decltype(exx_blocks)>;
                        using JExxMap = typename ExxBlocks::mapped_type;
                        using RExxMap = typename JExxMap::mapped_type;
                        struct FourierExxTask
                        {
                            ExxIJKAtomKey I;
                            ExxIJKAtomKey J;
                            int ik;
                            int n_I;
                            int n_J;
                            const RExxMap *R_exx;
                        };

                        std::vector<FourierExxTask> fourier_tasks;
                        for (const auto &[I_atom, J_Rexx] : exx_blocks)
                        {
                            const ExxIJKAtomKey I = I_atom;
                            const int n_I = static_cast<int>(
                                this->atbasis_wfc.get_atom_nb(static_cast<int>(I)));
                            for (const auto &[J_atom, R_exx] : J_Rexx)
                            {
                                if (R_exx.empty()) continue;
                                const ExxIJKAtomKey J = J_atom;
                                const int n_J = static_cast<int>(
                                    this->atbasis_wfc.get_atom_nb(static_cast<int>(J)));
                                for (int ik = 0; ik != n_target_kpoints; ++ik)
                                    fourier_tasks.push_back({I, J, ik, n_I, n_J, &R_exx});
                            }
                        }

                        std::vector<Matz> fourier_results(fourier_tasks.size());
                        const auto n_fourier_tasks =
                            static_cast<std::ptrdiff_t>(fourier_tasks.size());
                        #pragma omp parallel for schedule(dynamic)
                        for (std::ptrdiff_t itask = 0; itask < n_fourier_tasks; ++itask)
                        {
                            const auto &task = fourier_tasks[static_cast<std::size_t>(itask)];
                            const auto kfrac_ik = kfrac_target[task.ik];
                            Matz exx_ijk(task.n_I, task.n_J, MAJOR::ROW);
                            exx_ijk.zero_out();

                            for (const auto &[R, exx]: *task.R_exx)
                            {
                                auto exx_cplx = make_exx_ijk_complex_block(exx);
                                exx_cplx.swap_major(MAJOR::ROW);
                                const atpair_t IJ{task.I, task.J};
                                const auto *R_bvks = bvk_remap.find_R_bvk(IJ, R);
                                if (R_bvks == nullptr || R_bvks->empty())
                                {
                                    add_phase_weighted_exx_ijk(
                                        R, kfrac_ik, 1.0, std::move(exx_cplx), exx_ijk);
                                }
                                else if (R_bvks->size() == 1)
                                {
                                    add_phase_weighted_exx_ijk(
                                        R_bvks->front(), kfrac_ik, 1.0, std::move(exx_cplx),
                                        exx_ijk);
                                }
                                else
                                {
                                    const auto weight = 1.0 / static_cast<double>(R_bvks->size());
                                    for (const auto &R_bvk: *R_bvks)
                                        add_phase_weighted_exx_ijk(
                                            R_bvk, kfrac_ik, weight, exx_cplx.copy(), exx_ijk);
                                }
                            }
                            fourier_results[static_cast<std::size_t>(itask)] = std::move(exx_ijk);
                        }

                        for (std::size_t itask = 0; itask != fourier_tasks.size(); ++itask)
                        {
                            const auto &task = fourier_tasks[itask];
                            exx_I_Jik_mat[task.I][{task.J, task.ik}] =
                                std::move(fourier_results[itask]);
                        }
                    };
                    if (use_complex_exx_r)
                    {
                        if (orig_cplx != nullptr)
                            fourier_exx_to_ijk(*orig_cplx);
                    }
                    else
                    {
                        if (orig != nullptr)
                            fourier_exx_to_ijk(*orig);
                    }
                    global::profiler.stop("exx_build_KS_fourier_world");

                    global::profiler.start("exx_build_KS_ijk_redist");
                    std::map<ExxIJKAtomKey, std::map<ExxIJKKey, RI::Tensor<cplxdb>>> exx_I_Jik_tensor;
                    for (auto &[I, Jik_exx]: exx_I_Jik_mat)
                    {
                        const std::size_t n_I =
                            this->atbasis_wfc.get_atom_nb(static_cast<int>(I));
                        for (auto &[Jik, exx]: Jik_exx)
                        {
                            const std::size_t n_J =
                                this->atbasis_wfc.get_atom_nb(static_cast<int>(Jik.first));
                            exx_I_Jik_tensor[I][Jik] = RI::Tensor<cplxdb>({n_I, n_J}, exx.sptr());
                        }
                    }
                    auto exx_I_Jik =
                        comm_map2(comm_h.comm, exx_I_Jik_tensor, exx_ijk_s0, exx_ijk_s1);
                    exx_I_Jik_tensor.clear();
                    exx_I_Jik_mat.clear();
                    global::profiler.stop("exx_build_KS_ijk_redist");

                    std::vector<complex<double>> dummy(1, complex<double>{0.0, 0.0});
                    release_free_mem();
                    for (const int ik: target_kblacs_ctxt.kpoints_local())
                    {
                        if (ik < 0 || ik >= n_target_kpoints)
                            throw LIBRPA_RUNTIME_ERROR("k-point index out of range for k-local EXX rotation");
                        global::profiler.start("build_real_space_exx_6", "Hexx IJ/K -> 2D block");
                        collect_block_from_ordered_IJ_Tensor_sparse_zero_missing(
                            Hexx_nao_nao, desc_nao_nao, this->atbasis_wfc, ik,
                            complex<double>{1.0, 0.0}, exx_I_Jik);
                        global::profiler.stop("build_real_space_exx_6");

                        if (output_exx_mat_k)
                            write_exx_nao_k_matrix(Hexx_nao_nao, desc_nao_nao, output_dir,
                                                  source, isp, ispn_bra, ispn_ket, n_spinor, ik);

                        global::profiler.start("build_real_space_exx_7", "Rotate Hexx ij -> KS");
                        ScalapackConnector::pgemr2d_f(n_aos, n_aos,
                                                      Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                                      Hexx_nao_nao_opt.ptr(), 1, 1, desc_nao_nao_opt.desc,
                                                      desc_nao_nao.ictxt());
                        const auto *wfc_bra =
                            find_nested_int_map_3(wfc_target, isp, ispn_bra, ik);
                        const auto *wfc_ket =
                            find_nested_int_map_3(wfc_target, isp, ispn_ket, ik);
                        const std::size_t wfc_size_local =
                            static_cast<std::size_t>(desc_wfc_target.m_loc()) *
                            desc_wfc_target.n_loc();
                        const int bad_wfc_local =
                            (wfc_size_local > 0 && (wfc_bra == nullptr || wfc_ket == nullptr)) ||
                            (wfc_bra != nullptr &&
                             static_cast<std::size_t>(wfc_bra->size) != wfc_size_local) ||
                            (wfc_ket != nullptr &&
                             static_cast<std::size_t>(wfc_ket->size) != wfc_size_local);
                        int bad_wfc = 0;
                        MPI_Allreduce(&bad_wfc_local, &bad_wfc, 1, MPI_INT, MPI_MAX,
                                      rotation_blacs_h.comm());
                        if (bad_wfc)
                            throw LIBRPA_RUNTIME_ERROR(
                                "EXX local wave-function block is inconsistent with its descriptor");
                        const auto *wfc_bra_ptr =
                            wfc_bra == nullptr ? dummy.data() : wfc_bra->c;
                        const auto *wfc_ket_ptr =
                            wfc_ket == nullptr ? dummy.data() : wfc_ket->c;
                        release_free_mem();

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
                        if (use_gpu_replace_scalapack)
                        {
                            if (size_wfc > 0)
                            {
                                DEVICE_CHECK(deviceMemcpyAsync(d_wfc_bra, wfc_bra_ptr, size_wfc * sizeof(std::complex<double>),
                                                               deviceMemcpyHostToDevice, rotation_blacs_h.ddla_handle->stream));
                                DEVICE_CHECK(deviceMemcpyAsync(d_wfc_ket, wfc_ket_ptr, size_wfc * sizeof(std::complex<double>),
                                                               deviceMemcpyHostToDevice, rotation_blacs_h.ddla_handle->stream));
                            }
                            if (size_hexx_nao > 0)
                                DEVICE_CHECK(deviceMemcpyAsync(d_hexx_nao, Hexx_nao_nao_opt.ptr(), size_hexx_nao * sizeof(std::complex<double>),
                                                               deviceMemcpyHostToDevice, rotation_blacs_h.ddla_handle->stream));
                            LaConnector::pgemm(
                                'C', 'N', n_bands, n_aos, n_aos, std::complex<double>{1.0, 0.0},
                                d_wfc_bra, 1, 1, desc_wfc_device,
                                d_hexx_nao, 1, 1, desc_nao_nao_opt,
                                std::complex<double>{0.0, 0.0},
                                d_temp, 1, 1, desc_nband_nao_opt);
                            LaConnector::pgemm(
                                'N', 'N', n_bands, n_bands, n_aos, std::complex<double>{-1.0, 0.0},
                                d_temp, 1, 1, desc_nband_nao_opt,
                                d_wfc_ket, 1, 1, desc_wfc_device,
                                std::complex<double>{0.0, 0.0},
                                d_hexx_nband, 1, 1, desc_nband_nband_opt);
                            if (size_hexx_nband > 0)
                                DEVICE_CHECK(deviceMemcpyAsync(Hexx_nband_nband_opt.ptr(), d_hexx_nband,
                                                               size_hexx_nband * sizeof(std::complex<double>),
                                                               deviceMemcpyDeviceToHost, rotation_blacs_h.ddla_handle->stream));
                            DEVICE_CHECK(deviceStreamSynchronize(rotation_blacs_h.ddla_handle->stream));
                        }
                        else
#endif
                        {
                            ScalapackConnector::pgemm_f('C', 'N', n_bands, n_aos, n_aos, 1.0,
                                                        wfc_bra_ptr, 1, 1, desc_wfc_target.desc,
                                                        Hexx_nao_nao_opt.ptr(), 1, 1, desc_nao_nao_opt.desc,
                                                        0.0,
                                                        temp_nband_nao_opt.ptr(), 1, 1, desc_nband_nao_opt.desc);
                            ScalapackConnector::pgemm_f('N', 'N', n_bands, n_bands, n_aos, -1.0,
                                                        temp_nband_nao_opt.ptr(), 1, 1, desc_nband_nao_opt.desc,
                                                        wfc_ket_ptr, 1, 1, desc_wfc_target.desc,
                                                        0.0,
                                                        Hexx_nband_nband_opt.ptr(), 1, 1, desc_nband_nband_opt.desc);
                        }

                        std::vector<double> eexx_diag_send(n_bands, 0.0);
                        for (int ib = 0; ib != n_bands; ++ib)
                        {
                            const int ilo = desc_nband_nband_opt.indx_g2l_r(ib);
                            if (ilo < 0) continue;
                            const int jlo = desc_nband_nband_opt.indx_g2l_c(ib);
                            if (jlo < 0) continue;
                            eexx_diag_send[ib] =
                                Hexx_nband_nband_opt.ptr()[ilo + desc_nband_nband_opt.lld() * jlo].real();
                        }
                        std::vector<double> eexx_diag_recv(n_bands);
                        target_kblacs_ctxt.comm_blacs_h.reduce(
                            eexx_diag_send.data(), eexx_diag_recv.data(), n_bands, 0, MPI_SUM);
                        if (this->exx_KS.count(isp) == 0 || this->exx_KS.at(isp).count(ik) == 0)
                        {
                            this->exx_KS[isp][ik] = Hexx_nband_nband_opt.copy();
                            this->exx_KS[isp][ik] = C_ZERO;
                            if (target_kblacs_ctxt.comm_blacs_h.is_root())
                            {
                                for (int ib = 0; ib != n_bands; ib++)
                                    this->Eexx[isp][ik][ib] = 0.0;
                            }
                        }
                        this->exx_KS[isp][ik] += Hexx_nband_nband_opt;
                        if (target_kblacs_ctxt.comm_blacs_h.is_root())
                        {
                            for (int ib = 0; ib != n_bands; ib++)
                                this->Eexx[isp][ik][ib] += eexx_diag_recv[ib];
                        }
	                        global::profiler.stop("build_real_space_exx_7");
	                    }
	                    exx_I_Jik.clear();
	                    global::profiler.stop("exx_build_KS_rotate_kpara_klocal_blacs");
	                }
                else  // !is_mf_eigvec_k_distributed_
                {
                    // Everything will be collected to rank 0
                    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);
                    auto Hexx_nband_nband_fb =
                        init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);

                    // Each process have all KS eigenvectors, extract local blocks
                    for (size_t ik = 0; ik < kfrac_target.size(); ik++)
                    {
                        global::profiler.start("build_real_space_exx_6", "Hexx IJ -> 2D block");
                        Hexx_nao_nao.zero_out();
                        const auto& kfrac = kfrac_target[ik];
                        const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                            fourier = [kfrac](const atom_t I, const atom_t J, const Vector3_Order<int> &R)
                            {
                                const auto ang = (kfrac * R) * TWO_PI;
                                return complex<double>{std::cos(ang), std::sin(ang)};
                            };
                        if (use_complex_exx_r)
                            collect_block_from_IJ_storage_matrix_transform(Hexx_nao_nao, desc_nao_nao,
                                    this->atbasis_wfc, this->atbasis_wfc, fourier, exx_is_local_cplx);
                        else
                            collect_block_from_IJ_storage_matrix_transform(Hexx_nao_nao, desc_nao_nao,
                                    this->atbasis_wfc, this->atbasis_wfc, fourier, exx_is_local);
                        global::profiler.stop("build_real_space_exx_6");
                        // global::lib_printf("%s\n", str(Hexx_nao_nao).c_str());
                        if (output_exx_mat_k)
                            write_exx_nao_k_matrix(Hexx_nao_nao, desc_nao_nao, output_dir,
                                                  source, isp, ispn_bra, ispn_ket, n_spinor, ik);
                        if (use_root_dense_projection)
                        {
                            global::profiler.start(
                                "build_real_space_exx_7",
                                "Rotate Hexx ij -> KS with root dense IBZ projection");
                            auto Hexx_nao_nao_fb =
                                init_local_mat<complex<double>>(desc_nao_nao_fb, MAJOR::COL);
                            ScalapackConnector::pgemr2d_f(n_aos, n_aos, Hexx_nao_nao.ptr(), 1, 1,
                                                          desc_nao_nao.desc, Hexx_nao_nao_fb.ptr(),
                                                          1, 1, desc_nao_nao_fb.desc,
                                                          desc_nao_nao_fb.ictxt());

                            ComplexMatrix Hexx_nband_nband_dense;
                            if (comm_h.is_root())
                            {
                                ComplexMatrix Hexx_nao_nao_dense(n_aos, n_aos);
                                for (int iao = 0; iao != n_aos; ++iao)
                                {
                                    for (int jao = 0; jao != n_aos; ++jao)
                                    {
                                        Hexx_nao_nao_dense(iao, jao) = Hexx_nao_nao_fb(iao, jao);
                                    }
                                }
                                const auto &wfc_bra = wfc_target.at(isp).at(ispn_bra).at(ik);
                                const auto &wfc_ket = wfc_target.at(isp).at(ispn_ket).at(ik);
                                Hexx_nband_nband_dense =
                                    (-1.0) * (conj(wfc_bra) * Hexx_nao_nao_dense *
                                              transpose(wfc_ket, false));
                            }
                            broadcast_ComplexMatrix(Hexx_nband_nband_dense, 0, comm_h.comm);
                            global::profiler.stop("build_real_space_exx_7");

                            global::profiler.start("build_real_space_exx_8",
                                                   "Collect Eexx to root process");
                            if (this->exx_KS.count(isp) == 0 || this->exx_KS.at(isp).count(ik) == 0)
                            {
                                this->exx_KS[isp][ik] =
                                    Matz(n_bands, n_bands, Hexx_nband_nband_dense.c, MAJOR::ROW,
                                         MAJOR::COL);
                                if (comm_h.is_root())
                                {
                                    for (int ib = 0; ib != n_bands; ++ib)
                                    {
                                        this->Eexx[isp][ik][ib] = 0.0;
                                    }
                                }
                            }
                            else
                            {
                                this->exx_KS[isp][ik] +=
                                    Matz(n_bands, n_bands, Hexx_nband_nband_dense.c, MAJOR::ROW,
                                         MAJOR::COL);
                            }
                            if (comm_h.is_root())
                            {
                                for (int ib = 0; ib != n_bands; ++ib)
                                {
                                    this->Eexx[isp][ik][ib] +=
                                        Hexx_nband_nband_dense(ib, ib).real();
                                }
                            }
                            global::profiler.stop("build_real_space_exx_8");
                            continue;
                        }
                        const auto &wfc_bra = wfc_target.at(isp).at(ispn_bra).at(ik);
                        blacs_ctxt_h.barrier();
                        const auto wfc_bra_block = get_local_mat(wfc_bra.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                        Matz wfc_ket_block;
                        if (ispn_ket != ispn_bra)
                        {
                            const auto &wfc_ket = wfc_target.at(isp).at(ispn_ket).at(ik);
                            wfc_ket_block = get_local_mat(wfc_ket.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                        }
                        else
                        {
                            wfc_ket_block = wfc_bra_block;
                        }
                        // global::lib_printf("%s\n", str(wfc_block).c_str());
                        // global::lib_printf("%s\n", desc_nao_nao.info_desc().c_str());
                        // global::lib_printf("%s\n", desc_nband_nao.info_desc().c_str());
                        global::profiler.start("build_real_space_exx_7", "Rotate Hexx ij -> KS");
                        ScalapackConnector::pgemm_f('N', 'N', n_bands, n_aos, n_aos, 1.0,
                                                    wfc_bra_block.ptr(), 1, 1, desc_nband_nao.desc,
                                                    Hexx_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                                    0.0,
                                                    temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc);
                        ScalapackConnector::pgemm_f('N', 'C', n_bands, n_bands, n_aos, -1.0,
                                                    temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                    wfc_ket_block.ptr(), 1, 1, desc_nband_nao.desc,
                                                    0.0,
                                                    Hexx_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                        global::profiler.stop("build_real_space_exx_7");

                        // collect to master
                        global::profiler.start("build_real_space_exx_8", "Collect Eexx to root process");
                        if (global::should_output(LIBRPA_VERBOSE_DEBUG))
                            global::ofs_myid << "before pgemr2d_f" << std::endl;
                        ScalapackConnector::pgemr2d_f(n_bands, n_bands,
                                                    Hexx_nband_nband.ptr(), 1, 1, desc_nband_nband.desc,
                                                    Hexx_nband_nband_fb.ptr(), 1, 1, desc_nband_nband_fb.desc,
                                                    desc_nband_nband_fb.ictxt());
                        if (global::should_output(LIBRPA_VERBOSE_DEBUG))
                            global::ofs_myid << "after pgemr2d_f" << std::endl;
                        if (this->exx_KS.count(isp) == 0 || this->exx_KS.at(isp).count(ik) == 0)
                        {
                            this->exx_KS[isp][ik] = Hexx_nband_nband_fb.copy();
                            this->exx_KS[isp][ik] = C_ZERO;
                            if (blacs_ctxt_h.myid == 0)
                                for (int ib = 0; ib != n_bands; ib++)
                                    this->Eexx[isp][ik][ib] = 0.0;
                        }
                        this->exx_KS[isp][ik] += Hexx_nband_nband_fb;
                        // cout << "Hexx_nband_nband_fb isp " << isp  << " ik " << ik << endl << Hexx_nband_nband_fb;
                        if (blacs_ctxt_h.myid == 0)
                        {
                            for (int ib = 0; ib != n_bands; ib++)
                                this->Eexx[isp][ik][ib] += Hexx_nband_nband_fb(ib, ib).real();
                        }
                        if (global::should_output(LIBRPA_VERBOSE_DEBUG))
                            global::ofs_myid << "after Hexx_nband_nband_fb assign" << std::endl;
                        global::profiler.stop("build_real_space_exx_8");
                    }
                }
            }
        }
    }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (use_gpu_replace_scalapack && use_klocal_rotation)
    {
        DEVICE_CHECK(deviceFreeAsync(d_wfc_bra, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_wfc_ket, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_hexx_nao, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_temp, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_hexx_nband, rotation_blacs_h.ddla_handle->stream));
    }
#endif
    global::ofs_myid << "Done Exx::build_KS_blacs" << std::endl;
}

void Exx::build_KS_kgrid()
{
    if (is_mf_eigvec_k_distributed_)
        throw LIBRPA_RUNTIME_ERROR(
            "Exx::build_KS_kgrid cannot consume k-distributed eigenvectors; use build_KS_kgrid_blacs");
    this->build_KS(this->mf.get_eigenvectors(), this->pbc.kfrac_list, {});
}

// void Exx::build_KS0_kgrid()
// {
//     this->build_KS(this->mf_.get_eigenvectors0(), this->kfrac_list_);
// }

void Exx::build_KS_band(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_band,
                        const std::vector<Vector3_Order<double>> &kfrac_band,
                        const AtomPairBvKRemap<atom_t> &bvk_remap)
{
    if (is_mf_eigvec_k_distributed_)
        throw LIBRPA_RUNTIME_ERROR(
            "Exx::build_KS_band cannot consume k-distributed eigenvectors; use build_KS_band_blacs");
    this->build_KS(wfc_band, kfrac_band, bvk_remap);
}

void Exx::build_KS_kgrid_blacs(const BlacsCtxtHandler &blacs_ctxt_h,
                              bool use_gpu_replace_scalapack)
{
    this->build_KS_blacs(this->mf.get_eigenvectors(), this->pbc.kfrac_list, {}, blacs_ctxt_h,
                         use_gpu_replace_scalapack, "kgrid");
}

// void Exx::build_KS0_kgrid_blacs()
// {
//     this->build_KS(this->mf_.get_eigenvectors0(), this->kfrac_list_);
// }

void Exx::build_KS_band_blacs(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_band,
                              const std::vector<Vector3_Order<double>> &kfrac_band,
                              const AtomPairBvKRemap<atom_t> &bvk_remap,
                              const BlacsCtxtHandler &blacs_ctxt_h,
                              bool use_gpu_replace_scalapack,
                              const int band_index)
{
    this->build_KS_blacs(wfc_band, kfrac_band, bvk_remap, blacs_ctxt_h,
                         use_gpu_replace_scalapack, "band_" + std::to_string(band_index));
}

void Exx::write_exx_matrices_KS_binary(const std::string &output_dir,
                                       const std::string &source,
                                       const int istate_start,
                                       const int istate_end_option) const
{
    const int istate_end =
        istate_end_option < 0 ? desc_exx_KS.m() : istate_end_option;
    char fn[100];
    for (const auto &[ispin, exx_ik]: exx_KS)
    {
        for (const auto &[ik, exx]: exx_ik)
        {
            const Matz *exx_local = &exx;
            Matz exx_empty;
            if (exx.nr() != desc_exx_KS.m_loc() ||
                exx.nc() != desc_exx_KS.n_loc())
            {
                const bool root_dense_layout =
                    desc_exx_KS.mb() == desc_exx_KS.m() &&
                    desc_exx_KS.nb() == desc_exx_KS.n();
                if (!root_dense_layout || desc_exx_KS.is_src())
                    throw LIBRPA_RUNTIME_ERROR(
                        "EXX matrix local block does not match its descriptor");
                exx_empty =
                    init_local_mat<complex<double>>(desc_exx_KS, MAJOR::COL);
                exx_local = &exx_empty;
            }

            std::snprintf(fn, sizeof(fn),
                          "Exx_k_mn_%s_ispin_%d_ik_%d.bin",
                          source.c_str(), ispin, ik);
            write_matrix_binary_parallel(
                *exx_local, desc_exx_KS, path_as_directory(output_dir) + fn,
                istate_start, istate_end);
        }
    }
}

void Exx::reset_rspace()
{
    this->exx_IJR.clear();
    this->is_rspace_built_ = false;
}

void Exx::reset_kspace()
{
    this->exx_KS.clear();
    this->Eexx.clear();
    this->is_kspace_built_ = false;
}

} /* end of namespace librpa_int */
