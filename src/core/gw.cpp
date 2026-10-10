#include "gw.h"

// Public API headers
#include "librpa_enums.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "../io/fs.h"
#include "../io/global_io.h"
#include "../io/output_gw.h"
#include "../io/output_matrix.h"
#include "../io/stl_io_helper.h"
// #include "../math/utils_matrix_m.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../math/utils_matrix_mpi.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"
#include "../utils/dev_options.h"
#include "../utils/error.h"
#include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "../gpu/la_connector.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla_connector.h>
using namespace ddla;
#endif
#include "../utils/utils_mem.h"
#include "symmetry_context.h"
#include "atom.h"
#include "atomic_basis.h"
#include "chi0.h"
#include "epsilon.h"
#include "geometry.h"
#include "meanfield_mpi.h"
#include "pbc.h"
#include "ri.h"
#include "utils_atomic_basis_blacs.h"
#include "utils_complexmatrix.h"

#ifdef LIBRPA_USE_LIBRI
#include <RI/global/Tensor.h>
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
#include <RI/parallel/Parallel_LRI_Equally_Weighted.h>
#endif
#include <RI/physics/GW.h>
#include <RI/physics/symmetry/Symmetry_Filter.h>
using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
#endif

namespace librpa_int
{

using std::vector;

static int infer_target_n_bands(
    const MpiCommHandler &comm_h,
    const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
    const int fallback)
{
    int n_bands_local = 0;
    for (const auto &spin_wfc : wfc_target)
    {
        for (const auto &spinor_wfc : spin_wfc.second)
        {
            for (const auto &k_wfc : spinor_wfc.second)
            {
                const auto &wfc = k_wfc.second;
                if (wfc.nr <= 0) continue;
                if (n_bands_local != 0 && n_bands_local != wfc.nr)
                    throw LIBRPA_RUNTIME_ERROR("inconsistent target wave-function band counts");
                n_bands_local = wfc.nr;
            }
        }
    }

    int n_bands_max = 0;
    comm_h.allreduce(&n_bands_local, &n_bands_max, 1, MPI_MAX);
    if (n_bands_max == 0) return fallback;

    const int n_bands_min_send = n_bands_local == 0 ? n_bands_max : n_bands_local;
    int n_bands_min = 0;
    comm_h.allreduce(&n_bands_min_send, &n_bands_min, 1, MPI_MIN);
    if (n_bands_min != n_bands_max)
        throw LIBRPA_RUNTIME_ERROR("inconsistent target wave-function band counts");

    return n_bands_max;
}

static void add_phase_weighted_sigc_ijk(const Vector3_Order<int> &R_bvk,
                                        const Vector3_Order<double> &kfrac_ik,
                                        const double weight,
                                        const Matz &sigc,
                                        Matz &sigc_ijk)
{
    const auto ang = (kfrac_ik * R_bvk) * TWO_PI;
    const cplxdb phase{weight * std::cos(ang), weight * std::sin(ang)};
    auto sigc_weighted = sigc.copy();
    sigc_weighted *= phase;
    sigc_ijk += sigc_weighted;
}

static void add_weighted_sigc_R(std::map<Vector3_Order<int>, Matz> &R_sigc_shift,
                                const Vector3_Order<int> &R_bvk,
                                const Matz &sigc,
                                const double weight)
{
    auto sigc_weighted = sigc.copy();
    sigc_weighted *= weight;
    const auto it = R_sigc_shift.find(R_bvk);
    if (it == R_sigc_shift.end())
    {
        R_sigc_shift.emplace(R_bvk, std::move(sigc_weighted));
    }
    else
    {
        it->second += sigc_weighted;
    }
}

static std::vector<int> collect_target_iks(
    const MpiCommHandler &comm_h,
    const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
    const int n_kpts)
{
    std::vector<int> has_local(n_kpts, 0);
    std::vector<int> has_global(n_kpts, 0);

    for (const auto &spin_wfc : wfc_target)
    {
        for (const auto &spinor_wfc : spin_wfc.second)
        {
            for (const auto &k_wfc : spinor_wfc.second)
            {
                const int ik = k_wfc.first;
                if (ik >= 0 && ik < n_kpts) has_local[ik] = 1;
            }
        }
    }

    if (n_kpts > 0)
        comm_h.allreduce(has_local.data(), has_global.data(), n_kpts, MPI_MAX);

    std::vector<int> iks;
    for (int ik = 0; ik != n_kpts; ++ik)
    {
        if (has_global[ik]) iks.emplace_back(ik);
    }
    return iks;
}

static std::string make_sigc_ks_imagfreq_band_stem(const std::string &output_dir, const int index)
{
    std::ostringstream ss;
    ss << path_as_directory(output_dir) << "self_energy_omega_band_"
       << std::setfill('0') << std::setw(5) << index;
    return ss.str();
}

static std::vector<std::string> make_sigc_rf_filenames(
    const std::string &input_dir, const int ispin, const int ispinor_bra,
    const int ispinor_ket, const int n_spinor, const int iomega, const int myid)
{
    const auto dir = path_as_directory(input_dir);
    std::vector<std::string> names;

    std::ostringstream ss;
    ss << dir << "SigcRF_ispin_" << std::setfill('0') << std::setw(2) << ispin
       << "_s_" << std::setw(1) << ispinor_bra << std::setw(1) << ispinor_ket
       << "_iomega_" << std::setfill('0') << std::setw(3) << iomega
       << "_myid_" << std::setfill('0') << std::setw(5) << myid << ".dat";
    names.emplace_back(ss.str());

    ss.str("");
    ss.clear();
    // Fallback sigc saved files
    ss << dir << "SigcRF_ispin_" << std::setfill('0') << std::setw(5) << ispin;
    if (n_spinor > 1)
    {
        ss << "_spinor_" << ispinor_bra << "_" << ispinor_ket;
    }
    ss << "_iomega_" << std::setfill('0') << std::setw(5) << iomega
       << "_myid_" << std::setfill('0') << std::setw(5) << myid << ".dat";
    names.emplace_back(ss.str());

    return names;
}

static int parse_sigc_rf_myid(const std::string &file_path)
{
    const auto name = base_name(file_path);
    const auto tag = std::string{"_myid_"};
    const auto pos = name.rfind(tag);
    const auto suffix = std::string{".dat"};
    if (pos == std::string::npos || name.size() < suffix.size()
        || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
        return -1;

    const auto begin = pos + tag.size();
    const auto end = name.size() - suffix.size();
    if (begin >= end) return -1;

    int myid = 0;
    for (auto i = begin; i != end; ++i)
    {
        const char ch = name[i];
        if (ch < '0' || ch > '9') return -1;
        myid = myid * 10 + (ch - '0');
    }
    return myid;
}

static int discover_sigc_rf_nprocs_old(const std::string &input_dir)
{
    // Old SigcRF files have no manifest; largest _myid_ is the old rank count.
    int myid_max = -1;
    for (const auto &file_path: discover_files_with_prefix(input_dir, "SigcRF_"))
    {
        myid_max = std::max(myid_max, parse_sigc_rf_myid(file_path));
    }
    return myid_max + 1;
}

static int count_sigc_rf_files_for_rank(const int nfiles, const int nprocs, const int myid)
{
    const int count_base = nfiles / nprocs;
    const int count_extra = nfiles % nprocs;
    for (int iextra = 0; iextra != count_extra; ++iextra)
    {
        if (myid == iextra * nprocs / count_extra) return count_base + 1;
    }
    return count_base;
}

static std::pair<int, int> sigc_rf_file_range_for_rank(
    const int nfiles, const int nprocs, const int myid)
{
    int begin = 0;
    for (int rank = 0; rank != myid; ++rank)
    {
        begin += count_sigc_rf_files_for_rank(nfiles, nprocs, rank);
    }
    return {begin, begin + count_sigc_rf_files_for_rank(nfiles, nprocs, myid)};
}

static std::string find_sigc_rf_file(
    const std::string &input_dir, const int ispin, const int ispinor_bra,
    const int ispinor_ket, const int n_spinor, const int iomega, const int myid,
    bool *found)
{
    const auto candidates = make_sigc_rf_filenames(
        input_dir, ispin, ispinor_bra, ispinor_ket, n_spinor, iomega, myid);
    for (const auto &candidate: candidates)
    {
        if (file_exists(candidate))
        {
            require_readable_file(candidate);
            if (found != nullptr) *found = true;
            return candidate;
        }
    }
    if (found != nullptr) *found = false;
    return candidates.front();
}

static void read_exact(std::ifstream &ifs, void *dst, const std::streamsize n,
                       const std::string &fn)
{
    if (!ifs.read(static_cast<char *>(dst), n))
        throw LIBRPA_RUNTIME_ERROR("failed to read SigC checkpoint file: " + fn);
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
convert_symmetry_irreducible_sector_to_libri_gw(
    const symmetry_irreducible_sector_t& irreducible_sector,
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

#ifdef LIBRPA_USE_LIBRI
template <typename TA, typename TC, typename Tdata>
class OutputOnlyFilter_GW_Symmetry : public RI::Filter_Atom<TA, std::pair<TA, TC>>
{
  public:
    using TAC = std::pair<TA, TC>;

    OutputOnlyFilter_GW_Symmetry(
        const TC& period,
        const std::map<std::pair<TA, TA>, std::set<TC>>& irreducible_sector)
        : symmetry_(period, irreducible_sector)
    {
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TAC& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a1b1:
                return !this->symmetry_.in_irreducible_sector(A1, A3);
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
                return !this->symmetry_.in_irreducible_sector(A3, A1);
            default:
                return false;
        }
    }

    bool filter_for32(const RI::Label::ab_ab& label,
                      const TA& A1,
                      const TAC&,
                      const TAC& A3) const override
    {
        switch (label)
        {
            case RI::Label::ab_ab::a0b0_a2b1:
            case RI::Label::ab_ab::a0b0_a2b2:
                return !this->symmetry_.in_irreducible_sector(A1, A3);
            default:
                return false;
        }
    }

  private:
    RI::Symmetry_Filter<TA, TC, Tdata> symmetry_;
};

template <typename Tdata>
static RI::Tensor<Tdata> convert_complex_matrix_to_libri_tensor_gw(
    const ComplexMatrix& matrix)
{
    RI::Tensor<Tdata> tensor(
        {static_cast<std::size_t>(matrix.nr), static_cast<std::size_t>(matrix.nc)});
    for (int row = 0; row != matrix.nr; ++row)
    {
        for (int col = 0; col != matrix.nc; ++col)
        {
            if constexpr (std::is_same<Tdata, std::complex<double>>::value)
            {
                tensor(row, col) = matrix(row, col);
            }
            else
            {
                tensor(row, col) = matrix(row, col).real();
            }
        }
    }
    return tensor;
}

template <typename Tdata>
static std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>
restore_symmetry_ao_rspace_tensor_map_gw(
    const std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>>& tensors_ir,
    const SymmetryContext& symmetry_ctx,
    const symmetry_rspace_sector_stars_t& sector_stars,
    const AtomicBasis& atbasis_wfc)
{
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tdata>>> tensors_full;
    const auto wfc_layouts = atbasis_wfc.build_species_basis_layouts(symmetry_ctx.atom_to_type);
    for (const auto& i_entry : tensors_ir)
    {
        const auto ir_I = static_cast<atom_t>(i_entry.first);
        for (const auto& jr_entry : i_entry.second)
        {
            const auto ir_J = static_cast<atom_t>(jr_entry.first.first);
            const Vector3_Order<int> ir_R{
                jr_entry.first.second[0], jr_entry.first.second[1], jr_entry.first.second[2]};
            const auto pair_iter = sector_stars.find({ir_I, ir_J});
            if (pair_iter == sector_stars.end() || pair_iter->second.count(ir_R) == 0)
            {
                std::ostringstream oss;
                oss << "Failed to match a symmetry-filtered GW self-energy block with the"
                    << " irreducible-sector restore map for I=" << ir_I
                    << " J=" << ir_J << " R=(" << ir_R.x << "," << ir_R.y << ","
                    << ir_R.z << ")";
                throw std::runtime_error(oss.str());
            }

            const auto nao_I = atbasis_wfc.get_atom_nb(ir_I);
            const auto nao_J = atbasis_wfc.get_atom_nb(ir_J);
            const ComplexMatrix sigma_ir =
                convert_libri_tensor_to_complex_matrix(jr_entry.second, nao_I, nao_J);
            for (const auto& restore_member : pair_iter->second.at(ir_R))
            {
                const ComplexMatrix sigma_full = rotate_symmetry_rspace_block(
                    symmetry_ctx, wfc_layouts, restore_member.isym, ir_I, ir_J, sigma_ir);
                auto& target = tensors_full[restore_member.full_atom_pair.first][{
                    static_cast<int>(restore_member.full_atom_pair.second),
                    {restore_member.full_R.x, restore_member.full_R.y, restore_member.full_R.z}}];
                if (!target.empty())
                {
                    throw std::runtime_error(
                        "Duplicate full-sector GW self-energy block appears during symmetry restore");
                }
                target = convert_complex_matrix_to_libri_tensor_gw<Tdata>(sigma_full);
            }
        }
    }
    return tensors_full;
}

//! Four-channel ztensor map set of a spinor two-point AO operator, channel-outermost.
using gw_spinor_ztensor_maps_t = std::array<
    std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>>, 4>;

/*!
 * @brief Jointly restore the four spin channels of a spinor GW self-energy from
 * the symmetry irreducible sector (Phase 6).
 *
 * Thin LibRI-tensor wrapper around restore_symmetry_spinor_rspace_blocks:
 * converts each channel map to dense ComplexMatrix blocks, runs the
 * four-block (g, U_s, eta) kernel with key union and zero-fill, and converts
 * back. Called in the imaginary-time domain, before the tau -> i omega
 * transform, so antiunitary operations act as the plain Theta remap without
 * touching the fermionic frequency grid.
 */
static gw_spinor_ztensor_maps_t restore_symmetry_spinor_ao_rspace_tensor_map_gw(
    const gw_spinor_ztensor_maps_t& tensors_ir_channels,
    const SymmetryContext& symmetry_ctx,
    const symmetry_rspace_sector_stars_t& sector_stars,
    const AtomicBasis& atbasis_wfc)
{
    const auto wfc_layouts = atbasis_wfc.build_species_basis_layouts(symmetry_ctx.atom_to_type);
    std::vector<int> atom_nb(atbasis_wfc.n_atoms);
    for (std::size_t atom = 0; atom != atbasis_wfc.n_atoms; ++atom)
    {
        atom_nb[atom] = static_cast<int>(atbasis_wfc.get_atom_nb(static_cast<int>(atom)));
    }

    std::array<symmetry_rspace_block_map_t, 4> blocks_ir;
    for (std::size_t channel = 0; channel != 4; ++channel)
    {
        for (const auto& i_entry : tensors_ir_channels[channel])
        {
            const int nao_I = static_cast<int>(atbasis_wfc.get_atom_nb(i_entry.first));
            for (const auto& jr_entry : i_entry.second)
            {
                const int nao_J = static_cast<int>(atbasis_wfc.get_atom_nb(jr_entry.first.first));
                blocks_ir[channel][i_entry.first][jr_entry.first] =
                    convert_libri_tensor_to_complex_matrix(jr_entry.second, nao_I, nao_J);
            }
        }
    }

    auto blocks_full = restore_symmetry_spinor_rspace_blocks(
        blocks_ir, symmetry_ctx, sector_stars, wfc_layouts, atom_nb);

    gw_spinor_ztensor_maps_t tensors_full;
    for (std::size_t channel = 0; channel != 4; ++channel)
    {
        for (auto& i_entry : blocks_full[channel])
        {
            for (auto& jr_entry : i_entry.second)
            {
                tensors_full[channel][i_entry.first][jr_entry.first] =
                    convert_complex_matrix_to_libri_tensor_gw<cplxdb>(jr_entry.second);
            }
        }
    }
    return tensors_full;
}
#endif

template <typename T>
static bool is_effectively_zero_matrix(const matrix_m<T> &mat,
                                       const typename matrix_m<T>::real_t threshold = 1e-15)
{
    const auto data = mat.sptr();
    if (!data) return true;
    for (size_t i = 0; i != mat.size(); ++i)
    {
        if (std::abs((*data)[i]) > threshold) return false;
    }
    return true;
}

static void complete_hermitian_Wc_q_blocks(
    atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old &Wc_q)
{
    std::vector<atom_t> atoms_row;
    for (const auto &atom_i_pair : Wc_q) atoms_row.push_back(atom_i_pair.first);

    for (const auto atom_i : atoms_row)
    {
        std::vector<atom_t> atoms_col;
        for (const auto &atom_j_pair : Wc_q.at(atom_i))
            atoms_col.push_back(atom_j_pair.first);

        for (const auto atom_j : atoms_col)
        {
            for (const auto &q_block : Wc_q.at(atom_i).at(atom_j))
            {
                assert(q_block.second.major() == MAJOR::ROW);
                if (atom_i != atom_j)
                {
                    Wc_q[atom_j][atom_i][q_block.first] = q_block.second.get_transpose(true);
                }
                else
                {
                    auto hermitian_block = q_block.second;
                    hermitian_block =
                        (hermitian_block + hermitian_block.get_transpose(true)) * 0.5;
                    Wc_q[atom_i][atom_i][q_block.first] = hermitian_block;
                }
            }
        }
    }
}

static int wc_rf_checked_ifreq_end(const int start, const int end, const int n_freq)
{
    if (start < 0)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_start must be non-negative");
    if (start >= n_freq)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_start is outside the Wc frequency grid");
    if (end >= 0 && end <= start)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_end must be negative or greater than ifreq_output_wc_start");
    const int checked_end = end < 0 ? n_freq : end;
    if (checked_end > n_freq)
        throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_end is outside the Wc frequency grid");
    return checked_end;
}

static void write_wc_rf_atom_blocks(
    const atom_mapping<std::map<Vector3_Order<int>, Matz>>::pair_t_old &Wc_R,
    const PeriodicBoundaryData &pbc, const std::string &output_dir, const int ifreq,
    const double freq)
{
    for (const auto &[I, J_RWc] : Wc_R)
    {
        for (const auto &[J, R_Wc] : J_RWc)
        {
            for (const auto &[R, Wc] : R_Wc)
            {
                const auto iR = pbc.get_R_index(R);
                std::ostringstream ss;
                std::string info = "Wc at iR " + std::to_string(iR) + " ( " + std::to_string(R.x) +
                                   " " + std::to_string(R.y) + " " + std::to_string(R.z) +
                                   " ) and ifreq " + std::to_string(ifreq) + " ( " +
                                   std::to_string(freq) + " a.u. )";
                ss << path_as_directory(output_dir)
                   << "Wc_Mu_" << I << "_Nu_" << J
                   << "_iR_" << iR
                   << "_ifreq_" << ifreq << ".mtx";
                print_matrix_mm_file(Wc, ss.str(), info, 1e-10);
            }
        }
    }
}

static void write_wc_rf_full_matrix_from_atom_blocks(
    const atom_mapping<std::map<Vector3_Order<int>, Matz>>::pair_t_old &Wc_R,
    const AtomicBasis &atbasis_abf, const ArrayDesc &ad_Wc,
    const PeriodicBoundaryData &pbc, const std::string &output_dir, const int ifreq,
    const double freq)
{
    // Hermitian completion keeps the transposed atom block on the same rank
    // as its source. That rank need not own its BLACS matrix elements.
    // Elect one source per block (also handles replicated symmetry output),
    // then redistribute with the same communicator as the output descriptor.
    const int nat = as_int(atbasis_abf.n_atoms);
    std::vector<int> owners(nat * nat, ad_Wc.nprocs());
    for (const auto &[I, J_RWc] : Wc_R)
        for (const auto &[J, R_Wc] : J_RWc)
            if (!R_Wc.empty()) owners[I * nat + J] = ad_Wc.myid();
    MPI_Allreduce(MPI_IN_PLACE, owners.data(), as_int(owners.size()), MPI_INT,
                  MPI_MIN, ad_Wc.comm());
    std::map<int, std::set<atpair_t>> map_proc_IJs;
    for (int I = 0; I < nat; ++I)
        for (int J = 0; J < nat; ++J)
            if (owners[I * nat + J] < ad_Wc.nprocs())
                map_proc_IJs[owners[I * nat + J]].insert({I, J});
    IndexScheduler sched;
    sched.init(map_proc_IJs, atbasis_abf, atbasis_abf, ad_Wc, false);

    for (const auto &R : pbc.Rlist)
    {
        Matz Wc(ad_Wc.m_loc(), ad_Wc.n_loc(), MAJOR::COL);
        Wc.zero_out();
        ap_p_map<Matz> blocks;
        for (const auto &[I, J] : sched.atpairs)
        {
            const auto &R_Wc = Wc_R.at(I).at(J);
            const auto it = R_Wc.find(R);
            auto &block = blocks[{I, J}];
            if (it != R_Wc.end()) block = it->second.copy();
            else block = Matz(atbasis_abf[I], atbasis_abf[J], MAJOR::COL);
            block.swap_to_col_major();
        }
        fill_local_mat_from_ap_dist_scheduler(Wc, blocks, sched,
                                              atbasis_abf, atbasis_abf, ad_Wc);

        const auto iR = pbc.get_R_index(R);
        std::stringstream ss;
        std::string info = "Wc at iR " + std::to_string(iR) + " ( " + std::to_string(R.x) +
                           " " + std::to_string(R.y) + " " + std::to_string(R.z) +
                           " ) and ifreq " + std::to_string(ifreq) + " ( " +
                           std::to_string(freq) + " a.u. )";
        ss << path_as_directory(output_dir)
           << "Wc_iR_" << iR << "_ifreq_" << ifreq << ".mtx";
        print_matrix_mm_file_parallel(ss.str(), Wc, ad_Wc, info, 1e-10);
    }
}

static void write_sigc_nao_kf_matrix(const Matz &mat, const ArrayDesc &desc,
                                     const std::string &output_dir, const std::string &source,
                                     const int ispin, const int ispinor_bra,
                                     const int ispinor_ket, const int n_spinor,
                                     const int ik, const int ifreq)
{
    std::ostringstream ss;
    ss << path_as_directory(output_dir) << "SigcKF_" << source
       << "_ispin_" << ispin;
    if (n_spinor > 1)
    {
        ss << "_spinor_" << ispinor_bra << "_" << ispinor_ket;
    }
    ss << "_ik_" << ik << "_ifreq_" << ifreq << ".bin";
    write_matrix_binary_parallel(mat, desc, ss.str());
}

G0W0::G0W0(const MeanField &mf_in, const AtomicBasis &atbasis_wfc_in,
           const PeriodicBoundaryData &pbc_in,
           const SymmetryContext &symmetry_context_in,
           const TFGrids &tfg_in,
           const KPointBlacsParallelContext &kblacs_ctxt_in,
           const KPointBlacsParallelContext &band_kblacs_ctxt_in,
           const ArrayDesc &desc_wfc_in, const ArrayDesc &desc_band_wfc_in,
           bool is_eigvec_k_distributed,
           const bool use_symmetry_context_in)
    : mf(mf_in),
      desc_wfc(desc_wfc_in),
      desc_band_wfc(desc_band_wfc_in),
      atbasis_wfc(atbasis_wfc_in),
      pbc(pbc_in),
      symmetry_context(symmetry_context_in),
      use_symmetry_context(use_symmetry_context_in),
      qpoint_view(build_symmetry_qpoint_view(symmetry_context_in, pbc_in, use_symmetry_context_in)),
      tfg(tfg_in),
      comm_h(kblacs_ctxt_in.comm_global_h),
      kblacs_ctxt(kblacs_ctxt_in),
      band_kblacs_ctxt(band_kblacs_ctxt_in)
{
    comm_h.check_initialized();

    is_eigvec_k_distributed_ = is_eigvec_k_distributed;
    is_rspace_built_ = false;
    is_kspace_built_ = false;
    output_sigc_ks_kf_band_index_ = 0;
    sigc_kspace_source_.clear();

    // Public runtime options
    libri_threshold_C = 0.0;
    libri_threshold_Wc = 0.0;
    libri_threshold_G = 0.0;
    output_dir = "./";  // POSIX
    output_sigc_ks_mat_kf = false;
    output_sigc_ks_kf = false;
    istate_output_mat_start = 0;
    istate_output_mat_end = -1;
    output_sigc_mat_kf = false;
    output_sigc_mat_rt = false;
    output_sigc_mat_rf = false;
    output_wc_rf = false;
    output_wc_rf_atom_pair = false;
    ifreq_output_wc_start = 0;
    ifreq_output_wc_end = -1;
}

void G0W0::reset_rspace()
{
    sigc_is_f_IJ_R.clear();
    is_rspace_built_ = false;
}

void G0W0::reset_kspace()
{
    sigc_is_ik_f_KS.clear(); sigc_diag_is_ik_f_KS.clear(); is_kspace_built_ = false;
    sigc_kspace_source_.clear();
}

void G0W0::read_sigc(const std::string &input_dir)
{
    reset_rspace();
    reset_kspace();

    global::lib_printf_root("Reading real-space imaginary-frequency NAO sigma_c matrices from %s\n",
                            path_as_directory(input_dir).c_str());
    global::profiler.start("g0w0_read_sigc(R,iw) in NAO");

    const int nprocs_old = discover_sigc_rf_nprocs_old(input_dir);
    if (nprocs_old == 0)
        throw LIBRPA_RUNTIME_ERROR("no SigC checkpoint files found in: " + path_as_directory(input_dir));

    const auto myid_old_range = sigc_rf_file_range_for_rank(
        nprocs_old, global::size_global, global::myid_global);
    const int myid_old_begin = myid_old_range.first;
    const int myid_old_end = myid_old_range.second;
    global::lib_printf_root("Detected SigcRF files from %d MPI rank(s); current job has %d rank(s).\n",
                            nprocs_old, global::size_global);

    const int n_spinor = mf.get_n_spinor();
    int missing_file_local = 0;
    std::string missing_file;
    for (int ispin = 0; ispin != mf.get_n_spins(); ++ispin)
    {
        for (int ispinor_bra = 0; ispinor_bra != n_spinor; ++ispinor_bra)
        {
            for (int ispinor_ket = 0; ispinor_ket != n_spinor; ++ispinor_ket)
            {
                for (size_t iomega = 0; iomega != tfg.get_n_grids(); ++iomega)
                {
                    for (int myid_old = myid_old_begin; myid_old != myid_old_end; ++myid_old)
                    {
                        bool found = false;
                        const auto fn = find_sigc_rf_file(
                            input_dir, ispin, ispinor_bra, ispinor_ket,
                            n_spinor, as_int(iomega), myid_old, &found);
                        if (!found)
                        {
                            missing_file_local = 1;
                            if (missing_file.empty()) missing_file = fn;
                        }
                    }
                }
            }
        }
    }
    int missing_file_any = 0;
    comm_h.allreduce(&missing_file_local, &missing_file_any, 1, MPI_MAX);
    if (missing_file_any)
    {
        if (missing_file_local)
            global::ofs_myid << "Missing SigC checkpoint file: " << missing_file << std::endl;
        throw LIBRPA_RUNTIME_ERROR("missing SigC checkpoint file(s) in: "
                                  + path_as_directory(input_dir));
    }

    for (int ispin = 0; ispin != mf.get_n_spins(); ++ispin)
    {
        for (int ispinor_bra = 0; ispinor_bra != n_spinor; ++ispinor_bra)
        {
            for (int ispinor_ket = 0; ispinor_ket != n_spinor; ++ispinor_ket)
            {
                for (size_t iomega = 0; iomega != tfg.get_n_grids(); ++iomega)
                {
                    for (int myid_old = myid_old_begin; myid_old != myid_old_end; ++myid_old)
                    {
                        const auto fn = find_sigc_rf_file(
                            input_dir, ispin, ispinor_bra, ispinor_ket,
                            n_spinor, as_int(iomega), myid_old, nullptr);
                        std::ifstream ifs_sigmac_r(fn, std::ios::binary);
                        if (!ifs_sigmac_r)
                            throw LIBRPA_RUNTIME_ERROR("cannot open SigC checkpoint file: " + fn);

                        size_t n_IJR_myid = 0;
                        read_exact(ifs_sigmac_r, &n_IJR_myid, sizeof(n_IJR_myid), fn);

                        const auto omega = tfg.get_freq_nodes()[iomega];
                        for (size_t idx = 0; idx != n_IJR_myid; ++idx)
                        {
                            size_t dims[5];
                            read_exact(ifs_sigmac_r, dims, 5 * sizeof(size_t), fn);

                            if (dims[0] >= pbc.Rlist.size())
                                throw LIBRPA_RUNTIME_ERROR("SigC checkpoint R index is out of range: " + fn);
                            if (dims[1] >= atbasis_wfc.n_atoms || dims[2] >= atbasis_wfc.n_atoms)
                                throw LIBRPA_RUNTIME_ERROR("SigC checkpoint atom index is out of range: " + fn);

                            const int I = as_int(dims[1]);
                            const int J = as_int(dims[2]);
                            const auto n_I = atbasis_wfc.get_atom_nb(I);
                            const auto n_J = atbasis_wfc.get_atom_nb(J);
                            if (dims[3] != n_I || dims[4] != n_J)
                                throw LIBRPA_RUNTIME_ERROR("SigC checkpoint block size mismatch: " + fn);

                            Matz sigc(as_int(n_I), as_int(n_J), MAJOR::ROW);
                            read_exact(ifs_sigmac_r, sigc.ptr(),
                                       static_cast<std::streamsize>(n_I * n_J * sizeof(cplxdb)),
                                       fn);
                            sigc_is_f_IJ_R[ispin][ispinor_bra][ispinor_ket][omega][{I, J}]
                                          [pbc.Rlist[dims[0]]] = std::move(sigc);
                        }
                    }
                }
            }
        }
    }

    is_rspace_built_ = true;
    comm_h.barrier();
    global::lib_printf_root("Finished reading real-space imaginary-frequency NAO sigma_c matrices.\n");
    global::profiler.stop("g0w0_read_sigc(R,iw) in NAO");
}

void G0W0::collect_sigc_rf_output_shards()
{
#ifndef LIBRPA_USE_LIBRI
    throw LIBRPA_RUNTIME_ERROR("collecting SigC(R,iw) output shards requires LibRI");
#else
    std::set<int> all_atoms;
    std::vector<int> atoms;
    atoms.reserve(atbasis_wfc.n_atoms);
    for (std::size_t iatom = 0; iatom != atbasis_wfc.n_atoms; ++iatom)
    {
        all_atoms.insert(as_int(iatom));
        atoms.push_back(as_int(iatom));
    }

    // Disjoint (J,R) requests make comm_map2 sum each global block onto one output shard.
    std::set<std::pair<int, std::array<int, 3>>> local_JRs;
    for (const auto &[J, R] :
         dispatch_vector_prod(atoms, pbc.Rlist, comm_h.myid, comm_h.nprocs, true, false))
    {
        local_JRs.insert({J, {R.x, R.y, R.z}});
    }

    const int n_spinor = mf.get_n_spinor();
    for (int ispin = 0; ispin != mf.get_n_spins(); ++ispin)
    {
        for (int ispinor_bra = 0; ispinor_bra != n_spinor; ++ispinor_bra)
        {
            for (int ispinor_ket = 0; ispinor_ket != n_spinor; ++ispinor_ket)
            {
                for (const auto omega : tfg.get_freq_nodes())
                {
                    auto &sigc_IJ_R = sigc_is_f_IJ_R[ispin][ispinor_bra][ispinor_ket][omega];
                    std::map<int, std::map<std::pair<int, std::array<int, 3>>, Tensor<cplxdb>>>
                        sigc_I_JR_local;
                    for (const auto &[IJ, R_sigc] : sigc_IJ_R)
                    {
                        const int I = IJ.first;
                        const int J = IJ.second;
                        const auto n_I = atbasis_wfc.get_atom_nb(I);
                        const auto n_J = atbasis_wfc.get_atom_nb(J);
                        for (const auto &[R, sigc] : R_sigc)
                        {
                            sigc_I_JR_local[I][{J, {R.x, R.y, R.z}}] =
                                Tensor<cplxdb>({n_I, n_J}, sigc.sptr());
                        }
                    }

                    auto sigc_I_JR = comm_map2(comm_h.comm, sigc_I_JR_local, all_atoms, local_JRs);
                    ap_p_map<std::map<Vector3_Order<int>, Matz>> sigc_collected;
                    for (const auto &[I, JR_sigc] : sigc_I_JR)
                    {
                        const int n_I = as_int(atbasis_wfc.get_atom_nb(I));
                        for (const auto &[JR, sigc] : JR_sigc)
                        {
                            const int J = JR.first;
                            const int n_J = as_int(atbasis_wfc.get_atom_nb(J));
                            const auto &R = JR.second;
                            sigc_collected[{I, J}][{R[0], R[1], R[2]}] =
                                Matz{n_I, n_J, sigc.data, MAJOR::ROW};
                        }
                    }
                    sigc_IJ_R = std::move(sigc_collected);
                }
            }
        }
    }
#endif
}

void G0W0::write_sigc_rf_output_files() const
{
    const int n_spinor = mf.get_n_spinor();
    for (int ispin = 0; ispin != mf.get_n_spins(); ++ispin)
    {
        for (int ispinor_bra = 0; ispinor_bra != n_spinor; ++ispinor_bra)
        {
            for (int ispinor_ket = 0; ispinor_ket != n_spinor; ++ispinor_ket)
            {
                for (std::size_t iomega = 0; iomega != tfg.get_n_grids(); ++iomega)
                {
                    const auto fn =
                        make_sigc_rf_filenames(output_dir, ispin, ispinor_bra, ispinor_ket,
                                               n_spinor, as_int(iomega), global::myid_global)
                            .front();
                    const auto omega = tfg.get_freq_nodes()[iomega];
                    const auto &sigc_IJ_R =
                        sigc_is_f_IJ_R.at(ispin).at(ispinor_bra).at(ispinor_ket).at(omega);
                    int write_ok = write_rspace_matrices_binary(sigc_IJ_R, atbasis_wfc, pbc, fn);
                    MPI_Allreduce(MPI_IN_PLACE, &write_ok, 1, MPI_INT, MPI_MIN, comm_h.comm);
                    if (!write_ok)
                        throw LIBRPA_RUNTIME_ERROR("failed to write SigC output file: " + fn);
                }
            }
        }
    }
}

void G0W0::write_sigc_matrices_KS_binary(const std::string &output_dir,
                                         const std::string &source) const
{
    const int istate_end =
        istate_output_mat_end < 0 ? desc_sigc_is_ik_f_KS.m() : istate_output_mat_end;
    char fn[100];
    for (const auto &ispin_sigc: sigc_is_ik_f_KS)
    {
        const auto &ispin = ispin_sigc.first;
        for (const auto &ik_sigc: ispin_sigc.second)
        {
            const auto &ik = ik_sigc.first;
            for (const auto &freq_sigc: ik_sigc.second)
            {
                const auto ifreq = tfg.get_freq_index(freq_sigc.first);
                std::snprintf(fn, sizeof(fn), "Sigc_fk_mn_%s_ispin_%d_ik_%d_ifreq_%d.bin",
                              source.c_str(), ispin, ik, ifreq);
                write_matrix_binary_parallel(
                    freq_sigc.second, desc_sigc_is_ik_f_KS,
                    path_as_directory(output_dir) + fn,
                    istate_output_mat_start, istate_end);
            }
        }
    }
}

#ifdef LIBRPA_USE_LIBRI
template <typename Tdata>
static void build_gf_libri_kpara(
    const MeanField &mf,
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<double> &taus,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    std::map<double, std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<Tdata>>>> &tau_gf_libri)
{
    global::profiler.start("g0w0_build_gf_libri_kpara");
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
    comm_h.allreduce(MPI_IN_PLACE, &n_Rs_max, 1, MPI_MAX);
    auto gf_taus_Rs_cplx = get_gf_cplx_imagtimes_Rs_kpara(ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, taus, Rs_this, comm_h);
    // global::ofs_myid << "gf_taus_Rs_cplx " << gf_taus_Rs_cplx << std::endl;
    for (const auto &tau_gf_R_cplx: gf_taus_Rs_cplx)
    {
        const auto &tau = tau_gf_R_cplx.first;
        const auto &gf_R_cplx = tau_gf_R_cplx.second;
        for (const auto &R_gf_cplx: gf_R_cplx)
        {
            const auto &R = R_gf_cplx.first;
            const auto &gf_cplx = R_gf_cplx.second;
            std::array<int,3> Ra{R.x,R.y,R.z};
            const auto &map_IJs = map_R_IJs.at(R);
            omp_lock_t gf_lock;
            omp_init_lock(&gf_lock);
#pragma omp parallel for schedule(dynamic)
            for (const auto &IJ: map_IJs)
            {
                const auto &I = IJ.first;
                const auto &J = IJ.second;
                const auto gf_block = get_ap_block_from_global(gf_cplx, IJ, atbasis_wfc, atbasis_wfc);
                if constexpr (is_complex<Tdata>())
                {
                    auto p_gfmat = std::make_shared<std::valarray<Tdata>>(gf_block.c, gf_block.size);
                    omp_set_lock(&gf_lock);
                    tau_gf_libri[tau][I][{J, Ra}] = RI::Tensor<Tdata>({as_size(gf_block.nr), as_size(gf_block.nc)}, p_gfmat);
                    omp_unset_lock(&gf_lock);
                }
                else
                {
                    auto p_gfmat = std::make_shared<std::valarray<Tdata>>(gf_block.real().c, gf_block.size);
                    omp_set_lock(&gf_lock);
                    tau_gf_libri[tau][I][{J, Ra}] = RI::Tensor<Tdata>({as_size(gf_block.nr), as_size(gf_block.nc)}, p_gfmat);
                    omp_unset_lock(&gf_lock);
                }
            }
#pragma omp barrier
            omp_destroy_lock(&gf_lock);
        }
        // global::ofs_myid << R << std::endl;
        // print_complex_matrix("test", dmat_cplx, global::ofs_myid, true);
    }

    global::profiler.stop("g0w0_build_gf_libri_kpara");
}

template <typename Tdata>
static void build_gf_libri_kserial(
    const MeanField &mf,
    const AtomicBasis &atbasis_wfc,
    int ispin, int ispinor_bra, int ispinor_ket,
    const PeriodicBoundaryData &pbc,
    const SymmetryContext &symmetry_context,
    const bool use_symmetry_context,
    const vector<Vector3_Order<double>> &kfrac_list,
    const std::vector<double> &taus,
    const std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs,
    std::map<double, std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<Tdata>>>> &tau_gf_libri)
{
    global::profiler.start("g0w0_build_gf_libri_kserial");
    std::set<Vector3_Order<int>> Rs_local;
    for (const auto &IJR: IJRs)
    {
        Rs_local.insert(IJR.second);
    }
    const std::vector<Vector3_Order<int>> Rs_vec{Rs_local.cbegin(), Rs_local.cend()};
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
    if (restore_symmetry_kstars_from_full_grid && !taus.empty() && !Rs_vec.empty())
    {
        constexpr double restore_check_tol = 1e-6;
        const std::vector<double> tau_check{taus.front()};
        const std::vector<Vector3_Order<int>> R_check{Rs_vec.front()};
        const auto restored_check = get_symmetry_restored_gf_cplx_imagtimes_Rs(
            symmetry_context, wfc_layouts, mf, ispin, ispinor_bra, ispinor_ket, kfrac_list, tau_check,
            R_check, atom_nw, std::vector<bool>{}, &member_kfrac_targets,
            &full_grid_kstar_representatives).at(tau_check.front()).at(R_check.front());
        const auto direct_check =
            mf.get_gf_cplx_imagtimes_Rs(
                  ispin, ispinor_bra, ispinor_ket, kfrac_list, tau_check, R_check)
                .at(tau_check.front()).at(R_check.front());
        const auto diff = restored_check - direct_check;
        if (diff.get_max_abs() > restore_check_tol)
        {
            restore_symmetry_kstars_from_full_grid = false;
            member_kfrac_targets.clear();
        }
    }
    auto gf = (restore_symmetry_kstars || restore_symmetry_kstars_from_full_grid)
        ? get_symmetry_restored_gf_cplx_imagtimes_Rs(
              symmetry_context, wfc_layouts, mf, ispin, ispinor_bra, ispinor_ket, kfrac_list, taus, Rs_vec, atom_nw,
              std::vector<bool>{}, &member_kfrac_targets,
              restore_symmetry_kstars_from_full_grid ? &full_grid_kstar_representatives : nullptr)
        : mf.get_gf_cplx_imagtimes_Rs(ispin, ispinor_bra, ispinor_ket, kfrac_list, taus, Rs_vec);
    // global::ofs_myid << "gf " << gf << std::endl;
    tau_gf_libri.clear();
    // TODO: enable threading below
    for (auto t: taus)
    {
        tau_gf_libri[t] = {};
    }
    for (const auto &IJR: IJRs)
    {
        const auto &IJ = IJR.first;
        const auto &I = IJ.first;
        const auto &n_I = atbasis_wfc.get_atom_nb(I);
        const auto &J = IJ.second;
        const auto &n_J = atbasis_wfc.get_atom_nb(J);
        const auto &R = IJR.second;
        for (auto t: taus)
        {
            // skip when this process does not have any local GF data, i.e. Rs_local is empty
            if (gf.count(t) == 0 || gf.at(t).count(R) == 0) continue;
            const auto &gf_global = gf.at(t).at(R);
            std::shared_ptr<std::valarray<Tdata>> mat_ptr = std::make_shared<std::valarray<Tdata>>(n_I * n_J);
            if constexpr (is_complex<Tdata>())
            {
                for (size_t i = 0; i != n_I; i++)
                    for (size_t j = 0; j != n_J; j++)
                    {
                        (*mat_ptr)[i * n_J + j] = gf_global(atbasis_wfc.get_global_index(I, i),
                                                            atbasis_wfc.get_global_index(J, j));
                    }
            }
            else
            {
                for (size_t i = 0; i != n_I; i++)
                    for (size_t j = 0; j != n_J; j++)
                    {
                        (*mat_ptr)[i * n_J + j] = gf_global(atbasis_wfc.get_global_index(I, i),
                                                            atbasis_wfc.get_global_index(J, j)).real();
                    }
            }
            tau_gf_libri[t][as_int(I)][{as_int(J), {R.x, R.y, R.z}}] = RI::Tensor<Tdata>({n_I, n_J}, mat_ptr);
        }
    }
    gf.clear();
    global::profiler.stop("g0w0_build_gf_libri_kserial");
}

template <typename Tdata>
static void build_gf_libri_kblacs_para(
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
    const std::vector<double> &taus,
    const std::vector<Vector3_Order<int>> &Rs,
    std::map<double, std::map<int, std::map<std::pair<int,std::array<int,3>>,RI::Tensor<Tdata>>>> &tau_gf_libri)
{
    global::profiler.start("g0w0_build_gf_libri_kblacs_para", LIBRPA_VERBOSE_DEBUG);

    tau_gf_libri.clear();
    for (auto tau: taus)
        tau_gf_libri[tau] = {};

    const auto atom_nw = atbasis_wfc.get_atom_nb_map();
    const auto wfc_layouts = atbasis_wfc.has_l_shells()
        ? atbasis_wfc.build_species_basis_layouts(symmetry_context.atom_to_type)
        : std::vector<SpeciesBasisLayout>{};
    const bool restore_symmetry_kstars =
        use_symmetry_context
        && can_restore_symmetry_kstar_meanfield(
            symmetry_context, wfc_layouts, mf, kfrac_list, atom_nw);
    if (global::should_output(LIBRPA_VERBOSE_DEBUG))
        global::ofs_myid << "GW kBLACS GF symmetry restore: "
                         << (restore_symmetry_kstars ? "on" : "off") << std::endl;
    auto gf_taus_Rs_cplx = restore_symmetry_kstars
        ? get_symmetry_restored_gf_cplx_imagtimes_Rs_kblacs_para(
              ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, taus, Rs, kblacs_ctxt,
              desc_wfc, desc_gf, symmetry_context, pbc, atbasis_wfc)
        : get_gf_cplx_imagtimes_Rs_kblacs_para(
              ispin, ispinor_bra, ispinor_ket, mf, kfrac_list, taus, Rs, kblacs_ctxt,
              desc_wfc, desc_gf);

    for (auto &tau_gf_R_cplx: gf_taus_Rs_cplx)
    {
        const auto &tau = tau_gf_R_cplx.first;
        auto &gf_R_cplx = tau_gf_R_cplx.second;
        for (auto &R_gf_cplx: gf_R_cplx)
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
                if constexpr (is_complex<Tdata>())
                {
                    tau_gf_libri[tau][I][{J, {R.x, R.y, R.z}}] =
                        RI::Tensor<Tdata>({n_I, n_J}, mat_ap.sptr());
                }
                else
                {
                    tau_gf_libri[tau][I][{J, {R.x, R.y, R.z}}] =
                        RI::Tensor<Tdata>({n_I, n_J}, mat_ap.get_real().sptr());
                }
            }
            mat_blacs.clear();
        }
    }

    global::profiler.stop("g0w0_build_gf_libri_kblacs_para");
}

template <typename Tout, typename Tin>
static RI::Tensor<Tout> make_dense_wc_libri_tensor(matrix_m<Tin> &mat,
                                                    const std::size_t nr,
                                                    const std::size_t nc)
{
    if constexpr (std::is_same_v<Tout, Tin>)
    {
        return RI::Tensor<Tout>({nr, nc}, mat.sptr());
    }
    else
    {
        static_assert(std::is_same_v<Tout, double> && std::is_same_v<Tin, cplxdb>);
        auto real = mat.get_real();
        return RI::Tensor<Tout>({nr, nc}, real.sptr());
    }
}

template <typename Tout, typename Tin>
static void prepare_dense_wc_libri(
    std::map<double, std::map<Vector3_Order<int>, matrix_m<Tin>>> &Wc_tau_R_blacs,
    std::map<double,
             std::map<int,
                      std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<Tout>>>>
        &tau_Wc_libri,
    const AtomicBasis &atbasis_abf, const ArrayDesc &ad_Wc)
{
    global::profiler.start("g0w0_build_spacetime_get_ap",
                           "Compute balance atom-pairs distribution");
    const auto map_atpairs_balanced =
        get_balanced_ap_distribution_for_consec_descriptor(atbasis_abf, atbasis_abf, ad_Wc);
    const auto it_ap_myid = map_atpairs_balanced.find(ad_Wc.myid());
    if (it_ap_myid != map_atpairs_balanced.cend())
        global::ofs_myid << it_ap_myid->second << std::endl;
    else
        global::ofs_myid << "No atom pairs for Wc on this process" << std::endl;
    global::profiler.stop("g0w0_build_spacetime_get_ap");

    int wc_major_mask = 0;
    for (const auto &[tau, map_R_mat] : Wc_tau_R_blacs)
    {
        for (const auto &[R, mat_blacs] : map_R_mat)
        {
            if (mat_blacs.major() == MAJOR::ROW)
                wc_major_mask |= 1;
            else if (mat_blacs.major() == MAJOR::COL)
                wc_major_mask |= 2;
            else
                wc_major_mask |= 4;
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &wc_major_mask, 1, mpi_datatype<int>::value, MPI_BOR,
                  ad_Wc.comm());

    MAJOR wc_major = MAJOR::AUTO;
    if (wc_major_mask == 1)
        wc_major = MAJOR::ROW;
    else if (wc_major_mask == 2)
        wc_major = MAJOR::COL;
    else if (wc_major_mask == 0)
        throw LIBRPA_RUNTIME_ERROR("Wc(R,t) is empty");
    else
        throw LIBRPA_RUNTIME_ERROR("Inconsistent storage order among Wc(R,t) blocks");

    global::ofs_myid << "wc_major == MAJOR::ROW ? " << std::boolalpha
                     << (wc_major == MAJOR::ROW) << std::endl;

    IndexScheduler sched;
    sched.init(map_atpairs_balanced, atbasis_abf, atbasis_abf, ad_Wc,
               wc_major == MAJOR::ROW);
    for (auto &[tau, map_R_mat] : Wc_tau_R_blacs)
    {
        global::profiler.start("g0w0_build_spacetime_prep_Wc_all",
                               "Prepare LibRI Wc object");
        for (auto &[R, mat_blacs] : map_R_mat)
        {
            auto pair_mat = get_ap_map_from_blacs_dist_scheduler(
                mat_blacs, sched, atbasis_abf, atbasis_abf, ad_Wc);
            for (auto &[pair, mat_ap] : pair_mat)
            {
                const auto I = as_int(pair.first);
                const auto J = as_int(pair.second);
                const auto nabf_I = atbasis_abf.get_atom_nb(I);
                const auto nabf_J = atbasis_abf.get_atom_nb(J);
                if (mat_ap.is_col_major()) mat_ap.swap_to_row_major();
                tau_Wc_libri[tau][I][{J, {R.x, R.y, R.z}}] =
                    make_dense_wc_libri_tensor<Tout>(mat_ap, nabf_I, nabf_J);
            }
            mat_blacs.clear();
        }
        map_R_mat.clear();
        release_free_mem();
        global::profiler.stop("g0w0_build_spacetime_prep_Wc_all");
    }
}
#endif

void G0W0::build_spacetime(
    const LibrpaParallelRouting parallel_routing,
    const AtomicBasis& atbasis_abf,
    const Cs_LRI &LRI_Cs,
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const ArrayDesc &ad_Wc,
    std::map<double, atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old>
        *Wc_freq_q_atom_pair,
    std::map<Vector3_Order<double>, ComplexMatrix> *sinvS,
    const AtomicBasis *basis_aux_compressed,
    const AtomicBasis *basis_aux_unfold,
    const BlacsCtxtHandler *blacs_ctxt_h,
    const ArrayDesc *desc_wfc_in)
{
    using global::profiler;
    using global::lib_printf_root;
    using global::ofs_myid;

    if (parallel_routing != LIBRPA_ROUTING_LIBRI)
    {
        comm_h.barrier();
        throw LIBRPA_RUNTIME_ERROR("not implemented");
    }

    lib_printf_root("Calculating correlation self-energy by space-time method\n");
    if (!tfg.has_time_grids())
    {
        lib_printf_root("Parsed time-frequency object do not have time grids, exiting\n");
        comm_h.barrier();
        throw LIBRPA_RUNTIME_ERROR("input TFGrids object has no time grids");
    }
    comm_h.barrier();

    const bool use_complex_tensor = mf.get_n_spinor() > 1;
    if (use_complex_tensor)
        lib_printf_root("Using complex tensor\n");
    else
        lib_printf_root("Using real tensor\n");

    assert(ad_Wc.initialized());
    // assert(blacs_sigc_h.initialized());
    // blacs_sigc_h_ = blacs_sigc_h;

    const int natom = this->atbasis_wfc.n_atoms;

    std::ofstream ofs_sigmac_r;

#ifndef LIBRPA_USE_LIBRI
    if (comm_h.myid == 0)
    {
        std::cout << "LIBRA::G0W0::build_spacetime is only implemented on top of LibRI" << std::endl;
        std::cout << "Please recompiler LibRPA with -DUSE_LIBRI and configure include path" << std::endl;
    }
    comm_h.barrier();
    throw LIBRPA_RUNTIME_ERROR("compilation");
#else
    const bool use_atom_pair_Wc = Wc_freq_q_atom_pair != nullptr;

    typedef std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<double>>> dtensor_map;
    typedef std::map<int, std::map<std::pair<int, std::array<int, 3>>, RI::Tensor<cplxdb>>> ztensor_map;
    std::map<double, dtensor_map> tau_Wc_libri;
    std::map<double, ztensor_map> tau_Wc_libri_cplx;

    if (use_atom_pair_Wc)
    {
        if (sinvS == nullptr || basis_aux_compressed == nullptr || basis_aux_unfold == nullptr ||
            blacs_ctxt_h == nullptr || desc_wfc_in == nullptr)
            throw LIBRPA_RUNTIME_ERROR(
                "shrink Wc build_spacetime requires sinvS, compressed/full ABF bases, BLACS context, and WFC descriptor");

        Chi0 unfold_helper(mf, atbasis_wfc, *basis_aux_compressed, pbc, symmetry_context, tfg, kblacs_ctxt,
                           *desc_wfc_in, is_eigvec_k_distributed_, this->use_symmetry_context);

        if (output_wc_rf)
        {
            const int ifreq_end = wc_rf_checked_ifreq_end(
                ifreq_output_wc_start, ifreq_output_wc_end, tfg.get_n_grids());
            profiler.start("write_Wc_freq_R", "Export Wc(R,w) to file");
            for (int ifreq = ifreq_output_wc_start; ifreq != ifreq_end; ++ifreq)
            {
                const auto freq = tfg.get_freq_nodes()[ifreq];
                auto freq_iter = Wc_freq_q_atom_pair->find(freq);
                atom_mapping<std::map<Vector3_Order<double>, Matz>>::pair_t_old Wc_q;
                if (freq_iter != Wc_freq_q_atom_pair->end()) Wc_q = freq_iter->second;

                profiler.start("unfold_Wc_abfs", "Do shrink transformation");
                unfold_helper.unfold_abfs_Wc_q(*sinvS, Wc_q, pbc.klist_coul,
                                               *basis_aux_unfold, *blacs_ctxt_h);
                profiler.stop("unfold_Wc_abfs");

                profiler.start("construct_Wc_freq_lower_half", "Construct Lower Half of Wc(q,w)");
                complete_hermitian_Wc_q_blocks(Wc_q);
                profiler.stop("construct_Wc_freq_lower_half");

                auto Wc_R = FT_Wc_q2R(comm_h, *basis_aux_unfold, symmetry_context, Wc_q, tfg,
                                      pbc, pbc.Rlist, true, output_dir, this->use_symmetry_context);
                if (output_wc_rf_atom_pair)
                    write_wc_rf_atom_blocks(Wc_R, pbc, output_dir, ifreq, freq);
                else
                    write_wc_rf_full_matrix_from_atom_blocks(
                        Wc_R, atbasis_abf, ad_Wc, pbc, output_dir, ifreq, freq);
            }
            profiler.stop("write_Wc_freq_R");
        }

        profiler.start("g0w0_build_spacetime_wt_ft_wc", "Tranform Wc (q,w) -> (q,t)");
        auto Wc_tau_q = CT_Wc_freq2time_q(
            comm_h, *basis_aux_compressed, *Wc_freq_q_atom_pair, tfg, pbc.get_n_cells_bvk(),
            pbc.Rlist, pbc.klist_coul);
        Wc_freq_q_atom_pair->clear();
        release_free_mem();
        profiler.stop("g0w0_build_spacetime_wt_ft_wc");

        lib_printf_root("Time for Fourier transform of Wc in GW (seconds, Wall/CPU): %f %f\n",
                        profiler.get_wall_time_last("g0w0_build_spacetime_wt_ft_wc"),
                        profiler.get_cpu_time_last("g0w0_build_spacetime_wt_ft_wc"));

        const int n_tau = tfg.get_n_grids();
        profiler.start("g0w0_build_spacetime_ct_ft_wc", "Tranform Wc (q,t) -> (R,t)");
        for (auto itau = 0; itau != n_tau; ++itau)
        {
            const auto tau = tfg.get_time_nodes()[itau];
            auto &Wc_q = Wc_tau_q[tau];

            profiler.start("unfold_Wc_abfs", "Do shrink transformation");
            unfold_helper.unfold_abfs_Wc_q(*sinvS, Wc_q, pbc.klist_coul,
                                           *basis_aux_unfold, *blacs_ctxt_h);
            profiler.stop("unfold_Wc_abfs");

            profiler.start("construct_Wc_tau_lower_half", "Construct Lower Half of Wc(q,t)");
            complete_hermitian_Wc_q_blocks(Wc_q);
            profiler.stop("construct_Wc_tau_lower_half");

            profiler.start("g0w0_build_spacetime_Rq_ft_wc", "Tranform Wc (q,t) -> (R,t)");
            auto Wc_R = FT_Wc_q2R(comm_h, *basis_aux_unfold, symmetry_context, Wc_q, tfg,
                                  pbc, pbc.Rlist, false, output_dir, this->use_symmetry_context);
            profiler.stop("g0w0_build_spacetime_Rq_ft_wc");
            Wc_q.clear();

            profiler.start("g0w0_build_spacetime_prep_Wc_all", "Prepare LibRI Wc object");
            for (auto &[I, J_RWc] : Wc_R)
            {
                const auto n_I = atbasis_abf.get_atom_nb(I);
                for (auto &[J, R_Wc] : J_RWc)
                {
                    const auto n_J = atbasis_abf.get_atom_nb(J);
                    for (auto &[R, Wc_block] : R_Wc)
                    {
                        if (is_effectively_zero_matrix(Wc_block)) continue;
                        if (Wc_block.is_col_major()) Wc_block.swap_to_row_major();
                        if (use_complex_tensor)
                        {
                            tau_Wc_libri_cplx[tau][as_int(I)][{as_int(J), {R.x, R.y, R.z}}] =
                                RI::Tensor<cplxdb>({n_I, n_J}, Wc_block.sptr());
                        }
                        else
                        {
                            tau_Wc_libri[tau][as_int(I)][{as_int(J), {R.x, R.y, R.z}}] =
                                RI::Tensor<double>({n_I, n_J}, Wc_block.get_real().sptr());
                        }

                        if (I == J) continue;

                        const auto minus_R = (-R) % pbc.period;
                        const auto minus_R_iter = R_Wc.find(minus_R);
                        if (minus_R_iter == R_Wc.end()) continue;
                        if (is_effectively_zero_matrix(minus_R_iter->second)) continue;

                        if (use_complex_tensor)
                        {
                            const auto Wc_IJ_minus_R =
                                minus_R_iter->second.get_transpose(true);
                            tau_Wc_libri_cplx[tau][as_int(J)][{as_int(I), {R.x, R.y, R.z}}] =
                                RI::Tensor<cplxdb>({n_J, n_I}, Wc_IJ_minus_R.sptr());
                        }
                        else
                        {
                            const auto Wc_IJ_minus_R =
                                minus_R_iter->second.get_real().get_transpose();
                            tau_Wc_libri[tau][as_int(J)][{as_int(I), {R.x, R.y, R.z}}] =
                                RI::Tensor<double>({n_J, n_I}, Wc_IJ_minus_R.sptr());
                        }
                    }
                }
            }
            profiler.stop("g0w0_build_spacetime_prep_Wc_all");
        }
        Wc_tau_q.clear();
        profiler.stop("g0w0_build_spacetime_ct_ft_wc");
    }
    else // !use_atom_pair_Wc
    {
        // Transform from frequency/reciprocal to time/real-space
        profiler.start("g0w0_build_spacetime_ct_ft_wc", "Transform Wc (q,w) -> (R,t)");
        const bool use_real_dense_wc = global::dev_opts.use_real_dense_gw_wc
                                       && !use_complex_tensor && !output_wc_rf;
        if (global::dev_opts.use_real_dense_gw_wc && !use_real_dense_wc)
        {
            lib_printf_root(
                "Developer real dense Wc path disabled for %s\n",
                use_complex_tensor ? "complex/spinor tensors" : "output_wc_rf");
        }

        profiler.start("g0w0_build_spacetime_ct_ft_real_work", "Perform transformation");
        if (use_real_dense_wc)
        {
            lib_printf_root("Using developer real-storage dense Wc CT/FT path\n");
            auto Wc_tau_R_blacs = CT_FT_Wc_freq_q_real(
                comm_h, Wc_freq_q, pbc, tfg, &ad_Wc, &atbasis_abf,
                &qpoint_view, &symmetry_context);
            release_free_mem();
            profiler.stop("g0w0_build_spacetime_ct_ft_real_work");
            prepare_dense_wc_libri<double>(Wc_tau_R_blacs, tau_Wc_libri,
                                           atbasis_abf, ad_Wc);
        }
        else
        {
            auto Wc_tau_R_blacs = CT_FT_Wc_freq_q(
                comm_h, Wc_freq_q, pbc, tfg, true, output_wc_rf,
                ifreq_output_wc_start, ifreq_output_wc_end, output_wc_rf_atom_pair,
                output_dir, &ad_Wc, &atbasis_abf, &qpoint_view, &symmetry_context);
            release_free_mem();
            profiler.stop("g0w0_build_spacetime_ct_ft_real_work");
            if (use_complex_tensor)
                prepare_dense_wc_libri<cplxdb>(Wc_tau_R_blacs, tau_Wc_libri_cplx,
                                               atbasis_abf, ad_Wc);
            else
                prepare_dense_wc_libri<double>(Wc_tau_R_blacs, tau_Wc_libri,
                                                atbasis_abf, ad_Wc);
        }
        profiler.stop("g0w0_build_spacetime_ct_ft_wc");
    }

    lib_printf_root("Time for Fourier transform of Wc in GW (seconds, Wall/CPU): %f %f\n",
            profiler.get_wall_time_last("g0w0_build_spacetime_ct_ft_wc"),
            profiler.get_cpu_time_last("g0w0_build_spacetime_ct_ft_wc"));

    RI::GW<int, int, 3, double> gw_libri;
    RI::GW<int, int, 3, cplxdb> gw_libri_cplx;
#ifdef LIBRPA_LIBRI_HAS_FINE_GRAINED_LOCK
    gw_libri.lri.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
    gw_libri_cplx.lri.cal_mode = RI::LRI_Cal_Mode::CPU_fine_grained_lock;
#endif

    std::map<int,std::array<double,3>> atoms_pos;
    // Dummy atoms position
    for (int i = 0; i != natom; i++)
        atoms_pos.insert(std::pair<int, std::array<double, 3>>{i, {0, 0, 0}});
    const auto atom_nw = atbasis_wfc.get_atom_nb_map<int>();

    global::profiler.start("g0w0_build_spacetime_2", "Setup LibRI G0W0 object and C data");
    if (use_complex_tensor)
    {
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
        gw_libri_cplx.lri.parallel =
            std::make_shared<RI::Parallel_LRI_Equally_Weighted<int, int, 3, cplxdb>>(atom_nw);
#endif
        gw_libri_cplx.set_parallel(comm_h.comm, atoms_pos, pbc.latvec_array, pbc.period_array);
        ztensor_map data_libri;
        for (const auto &I_JR_C : LRI_Cs.data_libri)
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
        gw_libri_cplx.set_Cs(data_libri, this->libri_threshold_C);
    }
    else
    {
#if !defined(__DDLA_RI) && !defined(__CUDA_RI) && !defined(__HIP_RI)
        gw_libri.lri.parallel =
            std::make_shared<RI::Parallel_LRI_Equally_Weighted<int, int, 3, double>>(atom_nw);
#endif
        gw_libri.set_parallel(comm_h.comm, atoms_pos, pbc.latvec_array, pbc.period_array);
        gw_libri.set_Cs(LRI_Cs.data_libri, this->libri_threshold_C);
    }

    const auto &symmetry_ctx = this->symmetry_context;
    const auto wfc_layouts = atbasis_wfc.build_species_basis_layouts(symmetry_ctx.atom_to_type);
    const auto n_full_rspace_blocks =
        static_cast<std::size_t>(natom) * static_cast<std::size_t>(natom) * pbc.Rlist.size();
    const bool symmetry_reduces_rspace =
        symmetry_ctx.count_irreducible_blocks() < n_full_rspace_blocks;
    const bool sigc_rspace_symmetry_available =
        this->use_symmetry_context
        && symmetry_ctx.available
        && symmetry_species_layouts_match_atom_counts(
            wfc_layouts, symmetry_ctx.atom_to_type, atbasis_wfc.get_atom_nb_map())
        && !symmetry_ctx.irreducible_sector.empty()
        && !symmetry_ctx.rspace_sector_stars.empty()
        && !symmetry_ctx.rspace_operations.empty()
        && symmetry_ctx.atom_to_type.size() == static_cast<std::size_t>(natom)
        && symmetry_ctx.input_coord_frac.size() == static_cast<std::size_t>(natom)
        && symmetry_reduces_rspace;
    const bool sigc_band_space_complete =
        rspace_symmetry_has_complete_band_space(mf, -1);
    const bool use_input_sigc_symmetry =
        sigc_rspace_symmetry_available && sigc_band_space_complete;
    if (sigc_rspace_symmetry_available && !sigc_band_space_complete && comm_h.is_root())
    {
        global::lib_printf(
            "GW real-space self-energy irreducible-sector contraction disabled: "
            "%d response bands do not span the complete %d-state AO space; "
            "k-star and q-star symmetry remain active\n",
            mf.get_n_bands(), mf.get_n_aos());
    }
    const auto libri_sigc_irreducible_sector =
        use_input_sigc_symmetry
            ? convert_symmetry_irreducible_sector_to_libri_gw(
                  symmetry_ctx.irreducible_sector, this->pbc.period_array)
            : std::map<std::pair<int, int>, std::set<std::array<int, 3>>>{};
    const bool restore_input_sigc_output = use_input_sigc_symmetry;
    const auto& symmetry_sector_stars = symmetry_ctx.rspace_sector_stars;
    if (use_input_sigc_symmetry)
    {
        gw_libri.set_symmetry(false, {});
        gw_libri_cplx.set_symmetry(false, {});
        if (use_complex_tensor)
        {
            global::lib_printf(
                "Reducing GW real-space self-energy outputs with symmetry irreducible sectors\n");
            gw_libri_cplx.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_GW_Symmetry<int, std::array<int, 3>, cplxdb>>(
                    gw_libri_cplx.lri.period, libri_sigc_irreducible_sector);
        }
        else
        {
            global::lib_printf(
                "Reducing GW real-space self-energy outputs with symmetry irreducible sectors\n");
            gw_libri.lri.filter_atom =
                std::make_shared<OutputOnlyFilter_GW_Symmetry<int, std::array<int, 3>, double>>(
                    gw_libri.lri.period, libri_sigc_irreducible_sector);
        }
    }
    else
    {
        gw_libri.set_symmetry(false, {});
        gw_libri_cplx.set_symmetry(false, {});
    }
    profiler.stop("g0w0_build_spacetime_2");
    lib_printf_root("Time for LibRI G0W0 setup (seconds, Wall/CPU): %f %f\n",
            profiler.get_wall_time_last("g0w0_build_spacetime_2"),
            profiler.get_cpu_time_last("g0w0_build_spacetime_2"));

    const auto tot_atpair_ordered = generate_atom_pair_from_nat(natom, true);
    const auto &Rlist = this->pbc.Rlist;
    auto IJR_local_gf = dispatch_vector_prod(tot_atpair_ordered, Rlist, ad_Wc.myid(), ad_Wc.nprocs(), true, false);
    ofs_myid << "#IJR_local_gf: " << IJR_local_gf.size() << std::endl;

    ArrayDesc desc_gf;
    IndexScheduler sched_gf;
    std::vector<Vector3_Order<int>> Rs_gf;
    if (this->is_eigvec_k_distributed_)
    {
        profiler.start("g0w0_build_spacetime_prepare_gf_index");
        const int n_basis_ao = mf.get_n_aos();
        desc_gf = kblacs_ctxt.create_array_desc(n_basis_ao, n_basis_ao);
        const auto map_gf_atpairs_balanced =
            get_balanced_ap_distribution_for_consec_descriptor(atbasis_wfc, atbasis_wfc, desc_gf);
        sched_gf.init(map_gf_atpairs_balanced, atbasis_wfc, atbasis_wfc, desc_gf, false);
        const auto iRs = dispatcher_balanced(0, Rlist.size(), kblacs_ctxt.kpoints_local().size(),
                                             true, kblacs_ctxt.comm_kpoint_h.comm);
        Rs_gf.reserve(iRs.size());
        for (const auto &iR: iRs)
            Rs_gf.push_back(Rlist[iR]);
        profiler.stop("g0w0_build_spacetime_prepare_gf_index");
    }

    const int nfreq = tfg.get_n_grids();

    for (auto itau = 0; itau != nfreq; itau++)
    {
        // global::lib_printf("task %d itau %d start\n", mpi_comm_global_h.myid, itau);
        const auto tau = tfg.get_time_nodes()[itau];
        // global::lib_printf("task %d Wc_tau_R.count(tau) %zu\n", mpi_comm_global_h.myid, Wc_tau_R.count(tau));
        profiler.start("g0w0_setup_wc_entry_wait", LIBRPA_VERBOSE_DEBUG);
        comm_h.barrier();
        profiler.stop("g0w0_setup_wc_entry_wait");
        profiler.start("g0w0_build_spacetime_3", "Setup LibRI Wc");
        profiler.start("g0w0_setup_wc_set_Ws", LIBRPA_VERBOSE_DEBUG);
        size_t n_obj_wc_libri = 0;
        if (use_complex_tensor)
        {
            auto it = tau_Wc_libri_cplx.find(tau);
            const auto &Wc_libri = (it == tau_Wc_libri_cplx.end())? ztensor_map{} : it->second;
            for (const auto &w: Wc_libri)
            {
                n_obj_wc_libri += w.second.size();
            }
            gw_libri_cplx.set_Ws(Wc_libri, this->libri_threshold_Wc);
            if (it != tau_Wc_libri_cplx.end()) tau_Wc_libri_cplx.erase(it);
        }
        else
        {
            auto it = tau_Wc_libri.find(tau);
            const auto &Wc_libri = (it == tau_Wc_libri.end())? dtensor_map{} : it->second;
            for (const auto &w: Wc_libri)
            {
                n_obj_wc_libri += w.second.size();
            }
            gw_libri.set_Ws(Wc_libri, this->libri_threshold_Wc);
            if (it != tau_Wc_libri.end()) tau_Wc_libri.erase(it);
        }
        profiler.stop("g0w0_setup_wc_set_Ws");
        profiler.start("g0w0_setup_wc_malloc_trim", LIBRPA_VERBOSE_DEBUG);
        release_free_mem();
        profiler.stop("g0w0_setup_wc_malloc_trim");
        profiler.stop("g0w0_build_spacetime_3");

        const int n_spinor = mf.get_n_spinor();

        // Phase 6: under a general (g, U_s, eta) operation the four (bra, ket)
        // spin channels of Sigma_c mix, so the irreducible-sector restore must
        // see all four channels at the same tau. When active, the per-channel
        // Sigmas are stashed unrestored below and restored jointly in the tau
        // domain before the tau -> i omega accumulation.
        const bool defer_spinor_sigc_restore =
            restore_input_sigc_output && use_complex_tensor && n_spinor == 2
            && !symmetry_ctx.spin_operations.empty();

        const auto sigc_t2f_value_cplx =
            [](const cplxdb &sigc_cos, const cplxdb &sigc_sin, double t2f_cos,
               double t2f_sin) -> cplxdb
            {
                return sigc_cos * t2f_cos + sigc_sin * cplxdb(0.0, t2f_sin);
            };
        const auto sigc_t2f_value_real =
            [](double sigc_cos, double sigc_sin, double t2f_cos,
               double t2f_sin) -> cplxdb
            { return cplxdb{sigc_cos * t2f_cos, sigc_sin * t2f_sin}; };

        for (int ispin = 0; ispin != mf.get_n_spins(); ispin++)
        {
            // Irreducible-sector Sigmas stashed per (bra, ket) channel; only
            // populated when defer_spinor_sigc_restore is true.
            std::array<ztensor_map, 4> sigc_posi_tau_ir_channels;
            std::array<ztensor_map, 4> sigc_nega_tau_ir_channels;

            // Symmetrize Sigma_c(R, +/-tau), transform to (R, i omega) and
            // accumulate into sigc_is_f_IJ_R for one (bra, ket) channel.
            auto flush_sigc_channel = [&](int ispinor_bra, int ispinor_ket,
                                          const auto &sigc_posi_tau_in,
                                          const auto &sigc_nega_tau_in,
                                          auto block_scale_factor,
                                          auto make_freq_value, size_t elem_size)
            {
                size_t n_IJR_myid = 0; // for sigcmat output

                if (this->output_sigc_mat_rt)
                {
                    std::stringstream ss;
                    ss << path_as_directory(this->output_dir) << "SigcRT"
                        << "_ispin_" << std::setfill('0') << std::setw(5) << ispin;
                    if (n_spinor > 1)
                    {
                       ss << "_spinor_" << ispinor_bra << "_" << ispinor_ket;
                    }
                    ss << "_itau_" << std::setfill('0') << std::setw(5) << itau
                       << "_myid_" << std::setfill('0') << std::setw(5) << global::myid_global << ".dat";
                    ofs_sigmac_r.open(ss.str(), std::ios::out | std::ios::binary);
                    ofs_sigmac_r.write((char *) &n_IJR_myid, sizeof(size_t)); // placeholder
                }

                auto accumulate_sigc_tau_to_freq = [&](const auto &sigc_posi_tau_in_,
                                                       const auto &sigc_nega_tau_in_,
                                                       auto block_scale_factor_,
                                                       auto make_freq_value_, size_t elem_size_)
                {
                    for (const auto &[I, J_R_sigc_posi] : sigc_posi_tau_in_)
                    {
                        const auto n_I = this->atbasis_wfc.get_atom_nb(I);

                        auto it_nega_I = sigc_nega_tau_in_.find(I);
                        if (it_nega_I == sigc_nega_tau_in_.cend()) continue;

                        for (const auto &[JR, sigc_posi_block] : J_R_sigc_posi)
                        {
                            const auto J = JR.first;
                            const auto n_J = this->atbasis_wfc.get_atom_nb(J);
                            const auto &Ra = JR.second;

                            auto it_nega_JR = it_nega_I->second.find(JR);
                            if (it_nega_JR == it_nega_I->second.cend()) continue;

                            const auto &sigc_nega_block = it_nega_JR->second;

                            const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};

                            const auto it_R = std::find(Rlist.cbegin(), Rlist.cend(), R);
                            const auto iR = std::distance(Rlist.cbegin(), it_R);

                            const auto sigc_cos = block_scale_factor_ * (sigc_posi_block + sigc_nega_block);
                            const auto sigc_sin = block_scale_factor_ * (sigc_posi_block - sigc_nega_block);

                            ++n_IJR_myid;

                            if (this->output_sigc_mat_rt)
                            {
                                size_t dims[5];
                                dims[0] = as_size(iR);
                                dims[1] = as_size(I);
                                dims[2] = as_size(J);
                                dims[3] = as_size(n_I);
                                dims[4] = as_size(n_J);

                                ofs_sigmac_r.write(reinterpret_cast<const char *>(dims),
                                                   5 * sizeof(size_t));

                                ofs_sigmac_r.write(
                                    reinterpret_cast<const char *>(sigc_cos.ptr()),
                                    n_I * n_J * elem_size_);

                                ofs_sigmac_r.write(
                                    reinterpret_cast<const char *>(sigc_sin.ptr()),
                                    n_I * n_J * elem_size_);
                            }

                            for (size_t iomega = 0; iomega != tfg.get_n_grids(); ++iomega)
                            {
                                const auto omega = tfg.get_freq_nodes()[iomega];
                                const auto t2f_sin = tfg.get_sintrans_t2f()(iomega, itau);
                                const auto t2f_cos = tfg.get_costrans_t2f()(iomega, itau);

                                Matz sigc_temp(n_I, n_J, MAJOR::ROW);

                                for (size_t i = 0; i != static_cast<size_t>(n_I); ++i)
                                {
                                    for (size_t j = 0; j != static_cast<size_t>(n_J); ++j)
                                    {
                                        sigc_temp(i, j) = make_freq_value_(
                                            sigc_cos(i, j), sigc_sin(i, j), t2f_cos, t2f_sin);
                                    }
                                }

                                const atpair_t IJ{I, J};
                                auto &m_IJ =
                                    sigc_is_f_IJ_R[ispin][ispinor_bra][ispinor_ket][omega][IJ];

                                auto it = m_IJ.find(R);
                                if (it == m_IJ.cend())
                                {
                                    m_IJ.emplace(R, std::move(sigc_temp));
                                }
                                else
                                {
                                    it->second += sigc_temp;
                                }
                            }
                        }
                    }
                };

                // symmetrize and perform transformation
                global::profiler.start("g0w0_build_spacetime_6", "Transform Sigc (R,t) -> (R,w)");
                accumulate_sigc_tau_to_freq(sigc_posi_tau_in, sigc_nega_tau_in,
                                            block_scale_factor, make_freq_value, elem_size);
                global::profiler.stop("g0w0_build_spacetime_6");

                if (this->output_sigc_mat_rt)
                {
                    ofs_sigmac_r.seekp(0);
                    ofs_sigmac_r.write((char *) &n_IJR_myid, sizeof(size_t)); // overwrite
                    ofs_sigmac_r.close();
                }
            };

            for (int ispinor_bra = 0; ispinor_bra < n_spinor; ispinor_bra++)
            {
                for (int ispinor_ket = 0; ispinor_ket < n_spinor; ispinor_ket++)
                {
                    dtensor_map sigc_posi_tau, sigc_nega_tau;
                    ztensor_map sigc_posi_tau_cplx, sigc_nega_tau_cplx;

                    profiler.start("g0w0_gf_entry_wait", LIBRPA_VERBOSE_DEBUG);
                    comm_h.barrier();
                    profiler.stop("g0w0_gf_entry_wait");
                    profiler.start("g0w0_build_spacetime_4", "Compute G(R,t) and G(R,-t)",
                                   LIBRPA_VERBOSE_DEBUG);
                    // global::ofs_myid << "gf size " << gf.size() << endl;
                    // global::ofs_myid << "t " << gf[tau].size() << " ; -t " << gf[-tau].size() << endl;
                    std::map<double, dtensor_map> tau_gf_libri;
                    std::map<double, ztensor_map> tau_gf_libri_cplx;
                    const std::vector<double> taus{tau, -tau};
                    if (use_complex_tensor)
                    {
                        if (this->is_eigvec_k_distributed_)
                        {
                            build_gf_libri_kblacs_para(
                                mf, kblacs_ctxt, desc_wfc, desc_gf, sched_gf, atbasis_wfc,
                                ispin, ispinor_bra, ispinor_ket, this->pbc,
                                this->symmetry_context, this->use_symmetry_context,
                                this->pbc.kfrac_list, taus, Rs_gf, tau_gf_libri_cplx);
                        }
                        else
                        {
                            build_gf_libri_kserial(mf, atbasis_wfc, ispin, ispinor_bra, ispinor_ket, this->pbc,
                                                this->symmetry_context,
                                                this->use_symmetry_context,
                                                this->pbc.kfrac_list,
                                                taus, IJR_local_gf, tau_gf_libri_cplx);
                        }
                    }
                    else
                    {
                        if (this->is_eigvec_k_distributed_)
                            build_gf_libri_kblacs_para(
                                mf, kblacs_ctxt, desc_wfc, desc_gf, sched_gf, atbasis_wfc,
                                ispin, ispinor_bra, ispinor_ket, this->pbc,
                                this->symmetry_context, this->use_symmetry_context,
                                this->pbc.kfrac_list, taus, Rs_gf, tau_gf_libri);
                        else
                            build_gf_libri_kserial(mf, atbasis_wfc, ispin, ispinor_bra, ispinor_ket, this->pbc,
                                                this->symmetry_context,
                                                this->use_symmetry_context,
                                                this->pbc.kfrac_list,
                                                taus, IJR_local_gf, tau_gf_libri);
                    }
                    // if (itau == 0 && ispin == 0)  // debug
                    // {
                    //     global::ofs_myid << "tau_gf_libri itau=0 ispin=0" << std::endl;
                    //     global::ofs_myid << tau_gf_libri << std::endl;
                    // }
                    profiler.stop("g0w0_build_spacetime_4");

                    if (n_spinor > 1)
                    {
                        lib_printf_root("Time for Green's function, i_spin %d i_spinor_bra %d i_spinor_ket %d i_tau %d (seconds, Wall/CPU): %f %f\n",
                                ispin + 1, ispinor_bra + 1, ispinor_ket + 1, itau + 1,
                                profiler.get_wall_time_last("g0w0_build_spacetime_4"),
                                profiler.get_cpu_time_last("g0w0_build_spacetime_4"));
                    }
                    else
                    {
                        lib_printf_root("Time for Green's function, i_spin %d i_tau %d (seconds, Wall/CPU): %f %f\n",
                                ispin + 1, itau + 1,
                                profiler.get_wall_time_last("g0w0_build_spacetime_4"),
                                profiler.get_cpu_time_last("g0w0_build_spacetime_4"));
                    }

                    for (auto t: taus)
                    {
                        size_t n_obj_gf_libri = 0;
                        double wtime_g0w0_cal_sigc = omp_get_wtime();
                        if (use_complex_tensor)
                        {
                            const auto &gf_libri = tau_gf_libri_cplx.at(t);
                            for (const auto &gf: gf_libri)
                                n_obj_gf_libri += gf.second.size();

                            global::profiler.start("g0w0_set_Gs", LIBRPA_VERBOSE_DEBUG);
                            gw_libri_cplx.set_Gs(gf_libri, this->libri_threshold_G);
                            global::profiler.stop("g0w0_set_Gs");
                            global::profiler.start("g0w0_cal_sigc_entry_wait", LIBRPA_VERBOSE_DEBUG);
                            comm_h.barrier();
                            global::profiler.stop("g0w0_cal_sigc_entry_wait");
                            global::profiler.start("g0w0_build_spacetime_5", "Call libRI cal_Sigc");
                            gw_libri_cplx.cal_Sigmas();
                            if (restore_input_sigc_output && !defer_spinor_sigc_restore)
                            {
                                gw_libri_cplx.Sigmas =
                                    restore_symmetry_ao_rspace_tensor_map_gw(
                                        gw_libri_cplx.Sigmas, symmetry_ctx,
                                        symmetry_sector_stars, this->atbasis_wfc);
                            }
                            global::profiler.stop("g0w0_build_spacetime_5");
                            global::profiler.start("g0w0_cal_sigc_malloc_trim",
                                                   LIBRPA_VERBOSE_DEBUG);
                            release_free_mem();
                            global::profiler.stop("g0w0_cal_sigc_malloc_trim");
                            global::profiler.start("g0w0_build_spacetime_5_clean",
                                                   LIBRPA_VERBOSE_DEBUG);
                            gw_libri_cplx.free_Gs();
                            release_free_mem();
                            global::profiler.stop("g0w0_build_spacetime_5_clean");

                            // Check size of data
                            double mem_mb = get_tensor_map_bytes(gw_libri_cplx.Sigmas) * 1e-6;
                            global::ofs_myid << "Temporary Sigc_tau size for time " << t << " [MB]: " << mem_mb << std::endl;

                            if (defer_spinor_sigc_restore)
                            {
                                // Stash the irreducible-sector Sigmas; the joint
                                // four-channel restore runs after the (bra, ket)
                                // loops, still in the tau domain.
                                auto &sigc_tau_stash =
                                    (t > 0 ? sigc_posi_tau_ir_channels
                                           : sigc_nega_tau_ir_channels)[static_cast<std::size_t>(
                                        ispinor_bra * 2 + ispinor_ket)];
                                sigc_tau_stash = std::move(gw_libri_cplx.Sigmas);
                            }
                            else if (t > 0)
                                sigc_posi_tau_cplx = std::move(gw_libri_cplx.Sigmas);
                            else
                                sigc_nega_tau_cplx = std::move(gw_libri_cplx.Sigmas);
                            gw_libri_cplx.Sigmas.clear();
                        }
                        else
                        {
                            // ofs_myid << tau_gf_libri << std::endl;
                            const auto &gf_libri = tau_gf_libri.at(t);
                            for (const auto &gf: gf_libri)
                                n_obj_gf_libri += gf.second.size();
                            // if (t > 0)  // debug
                            // {
                            //     global::ofs_myid << "gf_libri posi ispin=0 itau " << itau << " " << get_num_keys(gf_libri)  << std::endl;
                            //     print_keys(global::ofs_myid, gf_libri);
                            //     // global::ofs_myid << gf_libri << std::endl;
                            //     for (const auto &[I, JR_gf]: gf_libri)
                            //     {
                            //         for (const auto &[JR, gf]: JR_gf)
                            //         {
                            //             std::stringstream ss;
                            //             Vector3_Order<int> R(JR.second[0], JR.second[1], JR.second[2]);
                            //             ss << "gf_libri_posi"
                            //                 << "_itau_" << std::setfill('0') << std::setw(5) << itau
                            //                 << "_I_" << std::setfill('0') << std::setw(5) << I
                            //                 << "_J_" << std::setfill('0') << std::setw(5) << JR.first
                            //                 << "_iR_" << std::setfill('0') << std::setw(5) << get_R_index(pbc.Rlist, R) << ".dat";
                            //             global::ofs_myid << "Writing GF to " << ss.str() << " " << tau << " " << I << " " << JR.second << " " << R << std::endl;
                            //             std::ofstream ofs(ss.str());
                            //             ofs << gf << std::endl;
                            //             ofs.close();
                            //         }
                            //     }
                            // }
                            // else
                            // {
                            //     global::ofs_myid << "gf_libri nega ispin=0 itau " << itau << " " << get_num_keys(gf_libri) << std::endl;
                            //     print_keys(global::ofs_myid, gf_libri);
                            //     // global::ofs_myid << gf_libri << std::endl;
                            //     for (const auto &[I, JR_gf]: gf_libri)
                            //     {
                            //         for (const auto &[JR, gf]: JR_gf)
                            //         {
                            //             std::stringstream ss;
                            //             Vector3_Order<int> R(JR.second[0], JR.second[1], JR.second[2]);
                            //             ss << "gf_libri_nega"
                            //                 << "_itau_" << std::setfill('0') << std::setw(5) << itau
                            //                 << "_I_" << std::setfill('0') << std::setw(5) << I
                            //                 << "_J_" << std::setfill('0') << std::setw(5) << JR.first
                            //                 << "_iR_" << std::setfill('0') << std::setw(5) << get_R_index(pbc.Rlist, R) << ".dat";
                            //             global::ofs_myid << "Writing GF to " << ss.str() << " " << tau << " " << I << " " << JR.second << " " << R << std::endl;
                            //             std::ofstream ofs(ss.str());
                            //             ofs << gf << std::endl;
                            //             ofs.close();
                            //         }
                            //     }
                            // }
	                            global::profiler.start("g0w0_set_Gs", LIBRPA_VERBOSE_DEBUG);
	                            gw_libri.set_Gs(gf_libri, this->libri_threshold_G);
	                            global::profiler.stop("g0w0_set_Gs");
	                            global::profiler.start("g0w0_cal_sigc_entry_wait",
	                                                   LIBRPA_VERBOSE_DEBUG);
	                            comm_h.barrier();
	                            global::profiler.stop("g0w0_cal_sigc_entry_wait");
	                            global::profiler.start("g0w0_build_spacetime_5", "Call libRI cal_Sigc");
	                            gw_libri.cal_Sigmas();
	                            if (restore_input_sigc_output)
	                            {
	                                gw_libri.Sigmas =
	                                    restore_symmetry_ao_rspace_tensor_map_gw(
	                                        gw_libri.Sigmas, symmetry_ctx,
	                                        symmetry_sector_stars, this->atbasis_wfc);
	                            }
	                            global::profiler.stop("g0w0_build_spacetime_5");
                            global::profiler.start("g0w0_build_spacetime_5_clean", LIBRPA_VERBOSE_DEBUG);
                            gw_libri.free_Gs();
                            release_free_mem();
                            global::profiler.stop("g0w0_build_spacetime_5_clean");

                            // Check size of data
                            double mem_mb = get_tensor_map_bytes(gw_libri.Sigmas) * 1e-6;
                            global::ofs_myid << "Temporary Sigc_tau size for time " << t << " [MB]: " << mem_mb << std::endl;

                            if (t > 0)
                                sigc_posi_tau = std::move(gw_libri.Sigmas);
                            else
                                sigc_nega_tau = std::move(gw_libri.Sigmas);
                            gw_libri.Sigmas.clear();
                            // if (itau == 0 && ispin == 0)
                            // {
                            //     if (t > 0)
                            //     {
                            //         global::ofs_myid << "sigc_posi_tau itau=0 ispin=0 " << get_num_keys(sigc_posi_tau) << std::endl;
                            //         print_keys(global::ofs_myid, sigc_posi_tau);
                            //         global::ofs_myid << sigc_posi_tau << std::endl;
                            //     }
                            //     if (t < 0)
                            //     {
                            //         global::ofs_myid << "sigc_nega_tau itau=0 ispin=0 " << get_num_keys(sigc_nega_tau) << std::endl;
                            //         print_keys(global::ofs_myid, sigc_nega_tau);
                            //         global::ofs_myid << sigc_nega_tau << std::endl;
                            //     }
                            // }
                        }
                        wtime_g0w0_cal_sigc = omp_get_wtime() - wtime_g0w0_cal_sigc;
                        if (n_spinor > 1)
                            global::lib_printf(
                                "Task %4d. libRI G0W0, spin %1d, bra %1d, ket %1d, time grid %12.6f. Wc size %zu, GF "
                                "size %zu. Wall time %f\n",
                                comm_h.myid, ispin, ispinor_bra, ispinor_ket, t, n_obj_wc_libri, n_obj_gf_libri,
                                wtime_g0w0_cal_sigc);
                        else
                            global::lib_printf(
                                "Task %4d. libRI G0W0, spin %1d, time grid %12.6f. Wc size %zu, GF "
                                "size %zu. Wall time %f\n",
                                comm_h.myid, ispin, t, n_obj_wc_libri, n_obj_gf_libri,
                                wtime_g0w0_cal_sigc);
                    }

                    if (!defer_spinor_sigc_restore)
                    {
                        if (use_complex_tensor)
                        {
                            flush_sigc_channel(ispinor_bra, ispinor_ket,
                                               sigc_posi_tau_cplx, sigc_nega_tau_cplx,
                                               cplxdb(0.5, 0.0), sigc_t2f_value_cplx,
                                               sizeof(cplxdb));
                            sigc_posi_tau_cplx.clear();
                            sigc_nega_tau_cplx.clear();
                        }
                        else
                        {
                            flush_sigc_channel(ispinor_bra, ispinor_ket,
                                               sigc_posi_tau, sigc_nega_tau,
                                               0.5, sigc_t2f_value_real, sizeof(double));
                            sigc_posi_tau.clear();
                            sigc_nega_tau.clear();
                        }
                    }
                }
            }
            if (defer_spinor_sigc_restore)
            {
                // Joint four-channel restore of the stashed irreducible-sector
                // Sigmas, still in the tau domain (Phase 6): the (g, U_s, eta)
                // kernel mixes the (bra, ket) channels, and the antiunitary
                // Theta remap acts on the tau blocks before the tau -> i omega
                // transform inside flush_sigc_channel.
                auto sigc_posi_tau_full_channels =
                    restore_symmetry_spinor_ao_rspace_tensor_map_gw(
                        sigc_posi_tau_ir_channels, symmetry_ctx,
                        symmetry_sector_stars, this->atbasis_wfc);
                auto sigc_nega_tau_full_channels =
                    restore_symmetry_spinor_ao_rspace_tensor_map_gw(
                        sigc_nega_tau_ir_channels, symmetry_ctx,
                        symmetry_sector_stars, this->atbasis_wfc);
                sigc_posi_tau_ir_channels = {};
                sigc_nega_tau_ir_channels = {};
                for (int ispinor_bra = 0; ispinor_bra != n_spinor; ispinor_bra++)
                {
                    for (int ispinor_ket = 0; ispinor_ket != n_spinor; ispinor_ket++)
                    {
                        const auto channel =
                            static_cast<std::size_t>(ispinor_bra * 2 + ispinor_ket);
                        flush_sigc_channel(ispinor_bra, ispinor_ket,
                                           sigc_posi_tau_full_channels[channel],
                                           sigc_nega_tau_full_channels[channel],
                                           cplxdb(0.5, 0.0), sigc_t2f_value_cplx,
                                           sizeof(cplxdb));
                        sigc_posi_tau_full_channels[channel].clear();
                        sigc_nega_tau_full_channels[channel].clear();
                    }
                }
            }
        }
        global::profiler.start("g0w0_build_spacetime_free_Ws");
        gw_libri.free_Ws();
        global::profiler.stop("g0w0_build_spacetime_free_Ws");
        // Release freed memory to OS, to resolve memory fragments in LibRI
        release_free_mem();
    }
    is_rspace_built_ = true;
#endif

    // Export real-space imaginary-frequency NAO sigma_c matrices
    if (is_rspace_built_ && this->output_sigc_mat_rf)
    {
        collect_sigc_rf_output_shards();
        write_sigc_rf_output_files();
    }
}

void G0W0::build_sigc_matrix_KS(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                                const std::vector<Vector3_Order<double>> &kfrac_target,
                                const AtomPairBvKRemap<atom_t> &bvk_remap)
{
    throw LIBRPA_RUNTIME_ERROR("Not implemented yet");
}

void G0W0::build_sigc_matrix_KS_blacs(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc_target,
                                      const std::vector<Vector3_Order<double>> &kfrac_target,
                                      const AtomPairBvKRemap<atom_t> &bvk_remap,
                                      const BlacsCtxtHandler &blacs_ctxt_h,
                                      const bool use_gpu_replace_scalapack,
                                      const std::string &source)
{
    assert(blacs_ctxt_h.comm() == this->comm_h.comm);
    assert(this->is_rspace_built_);

    if (this->is_kspace_built_)
    {
        global::lib_printf(LIBRPA_VERBOSE_WARN, "Warning: reset Sigmac_c k-space matrices\n");
        this->reset_kspace();
    }

    const int n_aos = mf.get_n_aos();
    const int n_bands = mf.get_n_bands();
    const int n_spins = mf.get_n_spins();
    const int n_spinor = mf.get_n_spinor();
    const int n_target_kpoints = static_cast<int>(kfrac_target.size());
    const bool use_klocal_rotation = is_eigvec_k_distributed_;
    const bool source_is_band_path = source.rfind("band_", 0) == 0;
    const KPointBlacsParallelContext &target_kblacs_ctxt =
        source_is_band_path ? band_kblacs_ctxt : kblacs_ctxt;
    const KPointBlacsParallelContext *rotation_kblacs_ctxt =
        use_klocal_rotation ? &target_kblacs_ctxt : nullptr;
    if (use_klocal_rotation)
    {
        if (rotation_kblacs_ctxt == nullptr || !rotation_kblacs_ctxt->is_initialized())
            throw LIBRPA_RUNTIME_ERROR("k-point BLACS context is not initialized");
        if (rotation_kblacs_ctxt->n_kpoints() != n_target_kpoints)
            throw LIBRPA_RUNTIME_ERROR("k-point BLACS context has inconsistent number of target k-points");
    }
    const BlacsCtxtHandler &rotation_blacs_h =
        use_klocal_rotation ? rotation_kblacs_ctxt->blacs_h : blacs_ctxt_h;
    const ArrayDesc &desc_wfc_target = source_is_band_path ? desc_band_wfc : desc_wfc;
    if (use_klocal_rotation &&
        (!desc_wfc_target.is_initialized() || desc_wfc_target.m() != n_aos ||
         desc_wfc_target.n() != n_bands ||
         desc_wfc_target.ictxt() != rotation_blacs_h.ictxt))
        throw LIBRPA_RUNTIME_ERROR("SigC source wave-function descriptor is inconsistent");

    global::profiler.start("g0w0_build_sigc_KS");

#ifndef LIBRPA_USE_LIBRI
    if (global::mpi_comm_global_h.myid == 0)
    {
        std::cout << "LIBRA::G0W0::build_sigc_matrix_KS is only implemented on top of LibRI" << std::endl;
        std::cout << "Please recompile LibRPA with -DLIBRPA_USE_LIBRI and optionally configure include path" << std::endl;
    }
    global::mpi_comm_global_h.barrier();
    throw LIBRPA_RUNTIME_ERROR("G0W0 needs compilation with LibRI");
#else
    ArrayDesc desc_nband_nao(rotation_blacs_h);
    desc_nband_nao.init_1b1p(n_bands, n_aos, 0, 0);
    ArrayDesc desc_nao_nao(rotation_blacs_h);
    desc_nao_nao.init_1b1p(n_aos, n_aos, 0, 0);
    ArrayDesc desc_nband_nband(rotation_blacs_h);
    desc_nband_nband.init_1b1p(n_bands, n_bands, 0, 0);

    const int expected_block_nao =
        get_capped_blacs_block_size(n_aos, wfc_gemm_block_size_opt, rotation_blacs_h);
    const int expected_block_nband =
        get_capped_blacs_block_size(n_bands, wfc_gemm_block_size_opt, rotation_blacs_h);
    if (use_klocal_rotation &&
        (desc_wfc_target.mb() != expected_block_nao ||
         desc_wfc_target.nb() != expected_block_nband))
        throw LIBRPA_RUNTIME_ERROR(
            "SigC source wave functions do not use the permanent capped rectangular layout");
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
    desc_sigc_is_ik_f_KS.reset_handler(rotation_blacs_h);
    desc_sigc_is_ik_f_KS.init(n_bands, n_bands, block_nband, block_nband, 0, 0);

    auto sigc_nao_nao_opt = init_local_mat<complex<double>>(desc_nao_nao_opt, MAJOR::COL);
    auto sigc_nband_nband_opt = init_local_mat<complex<double>>(desc_nband_nband_opt, MAJOR::COL);

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    std::complex<double> *d_wfc_bra = nullptr, *d_wfc_ket = nullptr,
                         *d_sigc_nao = nullptr, *d_temp = nullptr,
                         *d_sigc_nband = nullptr;
    size_t size_wfc = 0, size_sigc_nao = 0, size_temp = 0, size_sigc_nband = 0;
    if (use_gpu_replace_scalapack && use_klocal_rotation)
    {
        auto &rotation_blacs_h_nc = const_cast<BlacsCtxtHandler &>(rotation_blacs_h);
        if (rotation_blacs_h_nc.ddla_handle == nullptr)
            rotation_blacs_h_nc.init_ddla_handle();
        desc_wfc_device.set_ddla_desc(rotation_blacs_h.ddla_handle);
        desc_nao_nao_opt.set_ddla_desc(rotation_blacs_h.ddla_handle);
        desc_nband_nao_opt.set_ddla_desc(rotation_blacs_h.ddla_handle);
        desc_nband_nband_opt.set_ddla_desc(rotation_blacs_h.ddla_handle);

        size_wfc = static_cast<size_t>(desc_wfc_device.m_loc()) *
                   desc_wfc_device.n_loc();
        size_sigc_nao = static_cast<size_t>(desc_nao_nao_opt.m_loc()) *
                        desc_nao_nao_opt.n_loc();
        size_temp = static_cast<size_t>(desc_nband_nao_opt.m_loc()) *
                    desc_nband_nao_opt.n_loc();
        size_sigc_nband = static_cast<size_t>(desc_nband_nband_opt.m_loc()) *
                          desc_nband_nband_opt.n_loc();
        DEVICE_CHECK(deviceMallocAsync((void**)&d_wfc_bra, std::max<size_t>(size_wfc, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_wfc_ket, std::max<size_t>(size_wfc, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_sigc_nao, std::max<size_t>(size_sigc_nao, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_temp, std::max<size_t>(size_temp, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&d_sigc_nband, std::max<size_t>(size_sigc_nband, 1) * sizeof(std::complex<double>), rotation_blacs_h.ddla_handle->stream));
    }
#endif

    ArrayDesc desc_nao_nband_fb(rotation_blacs_h);  // For k-parallel
    ArrayDesc desc_nao_nao_fb(rotation_blacs_h);
    ArrayDesc desc_nband_nband_fb(rotation_blacs_h);
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
    const auto set_IJ_nao_nao_rotation = get_necessary_IJ_from_block_2D(
        this->atbasis_wfc, this->atbasis_wfc, desc_nao_nao);

    const SigcRspaceMap *sigc_rotation_source = &sigc_is_f_IJ_R;
    SigcRspaceMap sigc_is_f_IJ_R_redist;
    if (!use_klocal_rotation)
    {
        global::profiler.start("g0w0_build_sigc_KS_rspace_redist");
        const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nao_nao_rotation);
        const auto &sigc_redist_source = sigc_is_f_IJ_R;
        for (int isp = 0; isp < n_spins; isp++)
        {
            for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
            {
                for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
                {
                    const auto sigc_orig =
                        find_nested_int_map_3(sigc_redist_source, isp, ispn_bra, ispn_ket);
                    for (const auto& freq: this->tfg.get_freq_nodes())
                    {
                        std::map<int, std::map<std::pair<int, std::array<int, 3>>, Tensor<cplxdb>>> sigc_I_JR_local;
                        if (sigc_orig != nullptr)
                        {
                            auto it_sp_f = sigc_orig->find(freq);
                            if (it_sp_f != sigc_orig->cend())
                            {
                                for (const auto &IJ_R_sigc: it_sp_f->second)
                                {
                                    const auto IJ = IJ_R_sigc.first;
                                    const int I = IJ.first;
                                    const int J = IJ.second;
                                    const auto &n_I = this->atbasis_wfc.get_atom_nb(I);
                                    const auto &n_J = this->atbasis_wfc.get_atom_nb(J);
                                    for (const auto &[R, sigc]: IJ_R_sigc.second)
                                    {
                                        const std::array<int, 3> Ra{R.x, R.y, R.z};
                                        sigc_I_JR_local[I][{J, Ra}] = Tensor<complex<double>>({n_I, n_J}, sigc.sptr());
                                    }
                                }
                            }
                        }
                        // for certain spin and frequency
                        auto sigc_I_JR = comm_map2_first(comm_h.comm, sigc_I_JR_local, s0_s1.first, s0_s1.second);
                        // NOTE: sigc_I_JR may be empty for some process. This is a corner case 
                        //       of very small basis and large MPI tasks. However, it must be stored
                        //       to preserve the [spin][spinor][spinor][freq] key structure,
                        //       otherwise it will lead to map error in the subsequent rotation.
                        sigc_I_JR_local.clear();
                        ap_p_map<std::map<Vector3_Order<int>, Matz>> sigc_new;
                        for (const auto &[I, JRmat]: sigc_I_JR)
                        {
                            const int n_I = this->atbasis_wfc.get_atom_nb(I);
                            for (const auto &[JR, mat]: JRmat)
                            {
                                const atom_t J = JR.first;
                                atpair_t IJ{I, J};
                                const int n_J = this->atbasis_wfc.get_atom_nb(J);
                                const auto &Ra = JR.second;
                                const Vector3_Order<int> R{Ra[0], Ra[1], Ra[2]};
                                sigc_new[IJ][R] = Matz{n_I, n_J, mat.data, MAJOR::ROW};
                            }
                        }
                        auto sigc_dest =
                            find_nested_int_map_3(sigc_is_f_IJ_R_redist, isp, ispn_bra, ispn_ket);
                        if (sigc_dest == nullptr)
                            sigc_is_f_IJ_R_redist[isp][ispn_bra][ispn_ket][freq] = std::move(sigc_new);
                        else
                        {
                            auto it_sp_f = sigc_dest->find(freq);
                            if (it_sp_f == sigc_dest->cend())
                                sigc_dest->emplace(freq, std::move(sigc_new));
                            else
                                it_sp_f->second.swap(sigc_new);
                        }
                        sigc_I_JR.clear();
                    }
                }
            }
        }

        sigc_rotation_source = &sigc_is_f_IJ_R_redist;
        global::profiler.stop("g0w0_build_sigc_KS_rspace_redist");
    }

    sigc_is_ik_f_KS.clear(); sigc_diag_is_ik_f_KS.clear();
    auto store_sigc_local = [this](int isp, int ik, double freq, const Matz &sigc)
    {
        if (!this->output_sigc_ks_mat_kf) return;
        auto &mat_map = this->sigc_is_ik_f_KS[isp][ik];
        auto it = mat_map.find(freq);
        if (it == mat_map.end())
            mat_map[freq] = sigc.copy();
        else
            it->second += sigc;
    };
    using SigcIJKAtomKey = std::size_t;
    using SigcIJKKey = std::pair<SigcIJKAtomKey, int>; // (J, ik); J stays first for atom-pair routing.
    std::set<SigcIJKAtomKey> sigc_ijk_s0;
    std::set<SigcIJKKey> sigc_ijk_s1;
    if (use_klocal_rotation)
    {
        for (const auto &IJ: set_IJ_nao_nao_rotation)
        {
            sigc_ijk_s0.insert(IJ.first);
            for (const int ik: rotation_kblacs_ctxt->kpoints_local())
                sigc_ijk_s1.insert({IJ.second, ik});
        }
    }
    auto collect_sigc_nao_from_ijk =
        [this, &desc_nao_nao](Matz &sigc_nao_nao,
                              const std::map<int, std::map<int, Matz>> &sigc_ij)
    {
        sigc_nao_nao.zero_out();
        for (const auto &[I, J_sigc]: sigc_ij)
        {
            for (const auto &[J, sigc]: J_sigc)
            {
                collect_block_from_IJ_storage(
                    sigc_nao_nao, desc_nao_nao, this->atbasis_wfc, this->atbasis_wfc,
                    I, J, complex<double>{1.0, 0.0}, sigc.ptr(), sigc.major());
            }
        }
    };

    // local 2D-block submatrices
    auto sigc_nao_nao = init_local_mat<complex<double>>(desc_nao_nao, MAJOR::COL);
    auto sigc_nband_nband = init_local_mat<complex<double>>(desc_nband_nband, MAJOR::COL);

    for (int isp = 0; isp < n_spins; isp++)
    {
	// Initialize, make sure the map on every access has the isp key
        this->sigc_is_ik_f_KS[isp] = {}; this->sigc_diag_is_ik_f_KS[isp] = {};

        for (int ispn_bra = 0; ispn_bra < n_spinor; ispn_bra++)
        {
            for (int ispn_ket = 0; ispn_ket < n_spinor; ispn_ket++)
            {
                if (use_klocal_rotation)
                {
                    global::profiler.start("g0w0_build_sigc_KS_rotate_kpara_klocal_blacs");
                    auto temp_nband_nao_opt = init_local_mat<complex<double>>(desc_nband_nao_opt, MAJOR::COL);
                    std::vector<complex<double>> dummy(1, complex<double>{0.0, 0.0});
                    release_free_mem();
                    for (const auto& freq: this->tfg.get_freq_nodes())
                    {
                        std::map<SigcIJKAtomKey, std::map<SigcIJKKey, Matz>> sigc_I_Jik_mat;
                        global::profiler.start("g0w0_build_sigc_KS_fourier_world");
                        const auto sigc_orig =
                            find_nested_int_map_3(sigc_is_f_IJ_R, isp, ispn_bra, ispn_ket);
                        if (sigc_orig != nullptr)
                        {
                            auto it_sp_f = sigc_orig->find(freq);
                            if (it_sp_f != sigc_orig->cend())
                            {
                                using SigcFreqBlocks = std::decay_t<decltype(it_sp_f->second)>;
                                using RSigcMap = typename SigcFreqBlocks::mapped_type;
                                struct FourierSigcTask
                                {
                                    SigcIJKAtomKey I;
                                    SigcIJKAtomKey J;
                                    int ik;
                                    int n_I;
                                    int n_J;
                                    const RSigcMap *R_sigc;
                                };

                                std::vector<FourierSigcTask> fourier_tasks;
                                for (const auto &[IJ, R_sigc]: it_sp_f->second)
                                {
                                    if (R_sigc.empty()) continue;
                                    const SigcIJKAtomKey I = IJ.first;
                                    const SigcIJKAtomKey J = IJ.second;
                                    const int n_I = static_cast<int>(
                                        this->atbasis_wfc.get_atom_nb(static_cast<int>(I)));
                                    const int n_J = static_cast<int>(
                                        this->atbasis_wfc.get_atom_nb(static_cast<int>(J)));
                                    for (int ik = 0; ik != n_target_kpoints; ++ik)
                                        fourier_tasks.push_back({I, J, ik, n_I, n_J, &R_sigc});
                                }

                                std::vector<Matz> fourier_results(fourier_tasks.size());
                                const auto n_fourier_tasks =
                                    static_cast<std::ptrdiff_t>(fourier_tasks.size());
                                #pragma omp parallel for schedule(dynamic)
                                for (std::ptrdiff_t itask = 0; itask < n_fourier_tasks; ++itask)
                                {
                                    const auto &task = fourier_tasks[static_cast<std::size_t>(itask)];
                                    const auto kfrac_ik = kfrac_target[task.ik];
                                    Matz sigc_ijk(task.n_I, task.n_J, MAJOR::ROW);
                                    sigc_ijk.zero_out();

                                    for (const auto &[R, sigc]: *task.R_sigc)
                                    {
                                        const atpair_t IJ{task.I, task.J};
                                        const auto *R_bvks = bvk_remap.find_R_bvk(IJ, R);
                                        if (R_bvks == nullptr || R_bvks->empty())
                                        {
                                            add_phase_weighted_sigc_ijk(
                                                R, kfrac_ik, 1.0, sigc, sigc_ijk);
                                        }
                                        else if (R_bvks->size() == 1)
                                        {
                                            add_phase_weighted_sigc_ijk(
                                                R_bvks->front(), kfrac_ik, 1.0, sigc,
                                                sigc_ijk);
                                        }
                                        else
                                        {
                                            const auto weight = 1.0 / static_cast<double>(R_bvks->size());
                                            for (const auto &R_bvk: *R_bvks)
                                                add_phase_weighted_sigc_ijk(
                                                    R_bvk, kfrac_ik, weight, sigc, sigc_ijk);
                                        }
                                    }
                                    fourier_results[static_cast<std::size_t>(itask)] = std::move(sigc_ijk);
                                }

                                for (std::size_t itask = 0; itask != fourier_tasks.size(); ++itask)
                                {
                                    const auto &task = fourier_tasks[itask];
                                    sigc_I_Jik_mat[task.I][{task.J, task.ik}] =
                                        std::move(fourier_results[itask]);
                                }
                            }
                        }
                        global::profiler.stop("g0w0_build_sigc_KS_fourier_world");

                        global::profiler.start("g0w0_build_sigc_KS_ijk_redist");
                        std::map<SigcIJKAtomKey, std::map<SigcIJKKey, Tensor<cplxdb>>> sigc_I_Jik_tensor;
                        for (auto &[I, Jik_sigc]: sigc_I_Jik_mat)
                        {
                            const std::size_t n_I =
                                this->atbasis_wfc.get_atom_nb(static_cast<int>(I));
                            for (auto &[Jik, sigc]: Jik_sigc)
                            {
                                const std::size_t n_J =
                                    this->atbasis_wfc.get_atom_nb(static_cast<int>(Jik.first));
                                sigc_I_Jik_tensor[I][Jik] = Tensor<cplxdb>({n_I, n_J}, sigc.sptr());
                            }
                        }
                        auto sigc_I_Jik =
                            comm_map2(comm_h.comm, sigc_I_Jik_tensor, sigc_ijk_s0, sigc_ijk_s1);
                        sigc_I_Jik_tensor.clear();
                        sigc_I_Jik_mat.clear();
                        std::map<int, std::map<int, std::map<int, Matz>>> sigc_ijk_local;
                        for (const auto &[I, Jik_sigc]: sigc_I_Jik)
                        {
                            const int I_int = static_cast<int>(I);
                            const int n_I = static_cast<int>(this->atbasis_wfc.get_atom_nb(I_int));
                            for (const auto &[Jik, mat]: Jik_sigc)
                            {
                                const int J = static_cast<int>(Jik.first);
                                const int ik = Jik.second;
                                const int n_J = static_cast<int>(this->atbasis_wfc.get_atom_nb(J));
                                sigc_ijk_local[ik][I_int][J] = Matz{n_I, n_J, mat.data, MAJOR::ROW};
                            }
                        }
                        sigc_I_Jik.clear();
                        global::profiler.stop("g0w0_build_sigc_KS_ijk_redist");

                        for (const int ik: rotation_kblacs_ctxt->kpoints_local())
                        {
                            if (ik < 0 || ik >= n_target_kpoints)
                                throw LIBRPA_RUNTIME_ERROR("k-point index out of range for k-local SigC rotation");
                            const auto it_sigc_ik = sigc_ijk_local.find(ik);
                            if (it_sigc_ik == sigc_ijk_local.cend())
                                sigc_nao_nao.zero_out();
                            else
                                collect_sigc_nao_from_ijk(sigc_nao_nao, it_sigc_ik->second);
                            if (this->output_sigc_mat_kf)
                            {
                                const int ifreq = this->tfg.get_freq_index(freq);
                                write_sigc_nao_kf_matrix(
                                    sigc_nao_nao, desc_nao_nao, this->output_dir, source,
                                    isp, ispn_bra, ispn_ket, n_spinor, ik, ifreq);
                            }
                            ScalapackConnector::pgemr2d_f(n_aos, n_aos, sigc_nao_nao.ptr(), 1, 1,
                                                          desc_nao_nao.desc, sigc_nao_nao_opt.ptr(),
                                                          1, 1, desc_nao_nao_opt.desc, desc_nao_nao.ictxt());
                            const auto *wfc_bra =
                                find_nested_int_map_3(wfc_target, isp, ispn_bra, ik);
                            const auto *wfc_ket =
                                find_nested_int_map_3(wfc_target, isp, ispn_ket, ik);
                            const std::size_t wfc_size_local =
                                static_cast<std::size_t>(desc_wfc_target.m_loc()) *
                                desc_wfc_target.n_loc();
                            const int bad_wfc_local =
                                (wfc_size_local > 0 &&
                                 (wfc_bra == nullptr || wfc_ket == nullptr)) ||
                                (wfc_bra != nullptr &&
                                 static_cast<std::size_t>(wfc_bra->size) != wfc_size_local) ||
                                (wfc_ket != nullptr &&
                                 static_cast<std::size_t>(wfc_ket->size) != wfc_size_local);
                            int bad_wfc = 0;
                            MPI_Allreduce(&bad_wfc_local, &bad_wfc, 1, MPI_INT, MPI_MAX,
                                          rotation_blacs_h.comm());
                            if (bad_wfc)
                                throw LIBRPA_RUNTIME_ERROR(
                                    "SigC local wave-function block is inconsistent with its descriptor");
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
                                if (size_sigc_nao > 0)
                                    DEVICE_CHECK(deviceMemcpyAsync(d_sigc_nao, sigc_nao_nao_opt.ptr(), size_sigc_nao * sizeof(std::complex<double>),
                                                                   deviceMemcpyHostToDevice, rotation_blacs_h.ddla_handle->stream));
                                LaConnector::pgemm(
                                    'C', 'N', n_bands, n_aos, n_aos, std::complex<double>{1.0, 0.0},
                                    d_wfc_bra, 1, 1, desc_wfc_device,
                                    d_sigc_nao, 1, 1, desc_nao_nao_opt,
                                    std::complex<double>{0.0, 0.0},
                                    d_temp, 1, 1, desc_nband_nao_opt);
                                LaConnector::pgemm(
                                    'N', 'N', n_bands, n_bands, n_aos, std::complex<double>{1.0, 0.0},
                                    d_temp, 1, 1, desc_nband_nao_opt,
                                    d_wfc_ket, 1, 1, desc_wfc_device,
                                    std::complex<double>{0.0, 0.0},
                                    d_sigc_nband, 1, 1, desc_nband_nband_opt);
                                if (size_sigc_nband > 0)
                                    DEVICE_CHECK(deviceMemcpyAsync(sigc_nband_nband_opt.ptr(), d_sigc_nband,
                                                                   size_sigc_nband * sizeof(std::complex<double>),
                                                                   deviceMemcpyDeviceToHost, rotation_blacs_h.ddla_handle->stream));
                                DEVICE_CHECK(deviceStreamSynchronize(rotation_blacs_h.ddla_handle->stream));
                            }
                            else
#endif
                            {
                                ScalapackConnector::pgemm_f('C', 'N', n_bands, n_aos, n_aos, 1.0,
                                                            wfc_bra_ptr, 1, 1, desc_wfc_target.desc,
                                                            sigc_nao_nao_opt.ptr(), 1, 1, desc_nao_nao_opt.desc,
                                                            0.0,
                                                            temp_nband_nao_opt.ptr(), 1, 1, desc_nband_nao_opt.desc);
                                ScalapackConnector::pgemm_f('N', 'N', n_bands, n_bands, n_aos, 1.0,
                                                            temp_nband_nao_opt.ptr(), 1, 1, desc_nband_nao_opt.desc,
                                                            wfc_ket_ptr, 1, 1, desc_wfc_target.desc,
                                                            0.0,
                                                            sigc_nband_nband_opt.ptr(), 1, 1, desc_nband_nband_opt.desc);
                            }
                            if (this->output_sigc_ks_mat_kf)
                            {
                                store_sigc_local(isp, ik, freq, sigc_nband_nband_opt);
                            }
                            std::vector<cplxdb> diag_send(n_bands, cplxdb{0.0, 0.0});
                            for (int ib = 0; ib != n_bands; ++ib)
                            {
                                const int ilo = desc_nband_nband_opt.indx_g2l_r(ib);
                                if (ilo < 0) continue;
                                const int jlo = desc_nband_nband_opt.indx_g2l_c(ib);
                                if (jlo < 0) continue;
                                diag_send[ib] = sigc_nband_nband_opt.ptr()[ilo + desc_nband_nband_opt.lld() * jlo];
                            }
                            std::vector<cplxdb> diag_recv(n_bands);
                            rotation_kblacs_ctxt->comm_blacs_h.reduce(diag_send.data(), diag_recv.data(),
                                                                       n_bands, 0, MPI_SUM);
                            if (rotation_kblacs_ctxt->comm_blacs_h.is_root())
                            {
                                auto &diag_map = this->sigc_diag_is_ik_f_KS[isp][ik];
                                auto it = diag_map.find(freq);
                                if (it == diag_map.end())
                                {
                                    diag_map[freq] = std::move(diag_recv);
                                }
                                else
                                {
                                    for (int ib = 0; ib != n_bands; ++ib)
                                        it->second[ib] += diag_recv[ib];
                                }
                            }
                            release_free_mem();
                        }
                    }
                    global::profiler.stop("g0w0_build_sigc_KS_rotate_kpara_klocal_blacs");
                }
                else
                {
                    std::map<double, std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, Matz>>>> sigc_isp_local;
                    global::profiler.start("g0w0_build_sigc_KS_find_bvk");
                    for (const auto& freq: this->tfg.get_freq_nodes())
                    {
                        const auto &sigc_IJ_R =
                            sigc_rotation_source->at(isp).at(ispn_bra).at(ispn_ket).at(freq);
                        sigc_isp_local[freq] = {};

                        // Convert each <I,<J, R>> pair to the configured BvK counterpart
                        // to speed up later Fourier transform while keeping band interpolation accurate.
                        // A missing remap entry means that R is already the desired counterpart.
                        // NOTE: from local redistribution, sigc_rotation_source has all [spin][freq] keys,
                        // but each value (atom-pair map) can be empty.
                        for (const auto &[IJ, R_sigc]: sigc_IJ_R)
                        {
                            const auto &I = IJ.first;
                            const auto &J = IJ.second;
                            auto &R_sigc_shift = sigc_isp_local[freq][I][J];
                            for (auto &[R, sigc]: R_sigc)
                            {
                                const auto *R_bvks = bvk_remap.find_R_bvk(IJ, R);
                                if (R_bvks == nullptr || R_bvks->empty())
                                {
                                    R_sigc_shift[R] = sigc;
                                }
                                else if (R_bvks->size() == 1)
                                {
                                    R_sigc_shift[R_bvks->front()] = sigc;
                                }
                                else
                                {
                                    const auto weight = 1.0 / static_cast<double>(R_bvks->size());
                                    for (const auto &R_bvk: *R_bvks)
                                    {
                                        add_weighted_sigc_R(R_sigc_shift, R_bvk, sigc, weight);
                                    }
                                }
                            }
                        }
                    }
                    global::profiler.stop("g0w0_build_sigc_KS_find_bvk");

                    global::profiler.start("g0w0_build_sigc_KS_rotate_kserial");
                    desc_nband_nband_fb.init(n_bands, n_bands, n_bands, n_bands, 0, 0);
                    auto sigc_nband_nband_fb = init_local_mat<complex<double>>(desc_nband_nband_fb, MAJOR::COL);
                    for (const auto& freq: this->tfg.get_freq_nodes())
                    {
                        // Perform Fourier transform and rotate
                        for (size_t ik = 0; ik < kfrac_target.size(); ik++)
                        {
                            const auto kfrac = kfrac_target[ik];
                            // const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                            const std::function<complex<double>(const atom_t, const atom_t, const Vector3_Order<int> &)>
                                fourier = [kfrac](const atom_t I, const atom_t J, const Vector3_Order<int> &R)
                                {
                                    const auto ang = (kfrac * R) * TWO_PI;
                                    return complex<double>{std::cos(ang), std::sin(ang)};
                                };

                            sigc_nao_nao.zero_out();
                            sigc_nband_nband_fb.zero_out();
                            collect_block_from_IJ_storage_matrix_transform(
                                sigc_nao_nao, desc_nao_nao,
                                this->atbasis_wfc, this->atbasis_wfc,
                                fourier, sigc_isp_local.at(freq));
                            if (this->output_sigc_mat_kf)
                            {
                                const int ifreq = this->tfg.get_freq_index(freq);
                                write_sigc_nao_kf_matrix(
                                    sigc_nao_nao, desc_nao_nao, this->output_dir, source,
                                    isp, ispn_bra, ispn_ket, n_spinor, ik, ifreq);
                            }
                            if (use_root_dense_projection)
                            {
                                auto sigc_nao_nao_fb =
                                    init_local_mat<complex<double>>(desc_nao_nao_fb, MAJOR::COL);
                                ScalapackConnector::pgemr2d_f(
                                    n_aos, n_aos,
                                    sigc_nao_nao.ptr(), 1, 1, desc_nao_nao.desc,
                                    sigc_nao_nao_fb.ptr(), 1, 1, desc_nao_nao_fb.desc,
                                    desc_nao_nao_fb.ictxt());

                                ComplexMatrix sigc_nband_nband_dense;
                                if (comm_h.is_root())
                                {
                                    ComplexMatrix sigc_nao_nao_dense(n_aos, n_aos);
                                    for (int iao = 0; iao != n_aos; ++iao)
                                    {
                                        for (int jao = 0; jao != n_aos; ++jao)
                                        {
                                            sigc_nao_nao_dense(iao, jao) =
                                                sigc_nao_nao_fb(iao, jao);
                                        }
                                    }
                                    const auto &wfc_bra =
                                        wfc_target.at(isp).at(ispn_bra).at(ik);
                                    const auto &wfc_ket =
                                        wfc_target.at(isp).at(ispn_ket).at(ik);
                                    sigc_nband_nband_dense =
                                        conj(wfc_bra) * sigc_nao_nao_dense
                                        * transpose(wfc_ket, false);
                                }
                                broadcast_ComplexMatrix(
                                    sigc_nband_nband_dense, 0, comm_h.comm);
                                if (this->output_sigc_ks_mat_kf)
                                {
                                    const Matz sigc_dense(n_bands, n_bands,
                                                          sigc_nband_nband_dense.c,
                                                          MAJOR::ROW);
                                    const auto sigc_dense_opt =
                                        get_local_mat(sigc_dense, desc_sigc_is_ik_f_KS, MAJOR::COL);
                                    store_sigc_local(isp, ik, freq, sigc_dense_opt);
                                }
                                if (comm_h.is_root())
                                {
                                    std::vector<cplxdb> diag(n_bands);
                                    for (int ib = 0; ib != n_bands; ++ib)
                                        diag[ib] = sigc_nband_nband_dense(ib, ib);
                                    auto &diag_map = this->sigc_diag_is_ik_f_KS[isp][ik];
                                    auto it = diag_map.find(freq);
                                    if (it == diag_map.end())
                                        diag_map[freq] = std::move(diag);
                                    else
                                        for (int ib = 0; ib != n_bands; ++ib)
                                            it->second[ib] += diag[ib];
                                }
                                continue;
                            }
                            // prepare wave function BLACS
                            // FIXME: check if bra and ket are correct
                            const auto &wfc_bra = wfc_target.at(isp).at(ispn_bra).at(ik);
                            const auto &wfc_ket = wfc_target.at(isp).at(ispn_ket).at(ik);
                            blacs_ctxt_h.barrier();
                            const auto wfc_block = get_local_mat(wfc_bra.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL).conj();
                            auto temp_nband_nao = multiply_scalapack(wfc_block, desc_nband_nao, sigc_nao_nao, desc_nao_nao, desc_nband_nao);
                            if (ispn_bra == ispn_ket)
                                ScalapackConnector::pgemm_f('N', 'C', n_bands, n_bands, n_aos, 1.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                            wfc_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0,
                                                            sigc_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                            else
                            {
                                const auto wfc_ket_block = get_local_mat(wfc_ket.c, MAJOR::ROW, desc_nband_nao, MAJOR::COL);
                                ScalapackConnector::pgemm_f('N', 'T', n_bands, n_bands, n_aos, 1.0,
                                                            temp_nband_nao.ptr(), 1, 1, desc_nband_nao.desc,
                                                            wfc_ket_block.ptr(), 1, 1, desc_nband_nao.desc, 0.0,
                                                            sigc_nband_nband.ptr(), 1, 1, desc_nband_nband.desc);
                            }
                            if (this->output_sigc_ks_mat_kf)
                            {
                                ScalapackConnector::pgemr2d_f(n_bands, n_bands,
                                                              sigc_nband_nband.ptr(), 1, 1, desc_nband_nband.desc,
                                                              sigc_nband_nband_opt.ptr(), 1, 1, desc_nband_nband_opt.desc,
                                                              desc_nband_nband_opt.ictxt());
                                store_sigc_local(isp, ik, freq, sigc_nband_nband_opt);
                            }
                            // collect the full matrix to master
                            // TODO: would need a different strategy for large system
                            ScalapackConnector::pgemr2d_f(n_bands, n_bands,
                                                          sigc_nband_nband.ptr(), 1, 1, desc_nband_nband.desc,
                                                          sigc_nband_nband_fb.ptr(), 1, 1, desc_nband_nband_fb.desc,
                                                          desc_nband_nband_fb.ictxt());
                            // Only the matrix at the master process is meaningful; extract its diagonal
                            // instead of storing the full n_bands x n_bands matrix.
                            if (comm_h.is_root())
                            {
                                std::vector<cplxdb> diag(n_bands);
                                for (int ib = 0; ib != n_bands; ++ib)
                                    diag[ib] = sigc_nband_nband_fb(ib, ib);
                                auto &diag_map = this->sigc_diag_is_ik_f_KS[isp][ik];
                                auto it = diag_map.find(freq);
                                if (it == diag_map.end())
                                    diag_map[freq] = std::move(diag);
                                else
                                    for (int ib = 0; ib != n_bands; ++ib)
                                        it->second[ib] += diag[ib];
                            }
                        }
                    }
                    global::profiler.stop("g0w0_build_sigc_KS_rotate_kserial");
                }
            }
        }
    }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    if (use_gpu_replace_scalapack && use_klocal_rotation)
    {
        DEVICE_CHECK(deviceFreeAsync(d_wfc_bra, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_wfc_ket, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_sigc_nao, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_temp, rotation_blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(d_sigc_nband, rotation_blacs_h.ddla_handle->stream));
    }
#endif
#endif
    this->is_kspace_built_ = true;
    this->sigc_kspace_source_ = source;
    global::profiler.stop("g0w0_build_sigc_KS");
}

void G0W0::build_sigc_matrix_KS_kgrid(const Atoms &geometry)
{
    comm_h.barrier();
    global::ofs_myid << "build_sigc_matrix_KS_kgrid: constructing self-energy matrix for SCF k-grid" << std::endl;
    if (is_eigvec_k_distributed_)
        throw LIBRPA_RUNTIME_ERROR(
            "G0W0::build_sigc_matrix_KS_kgrid cannot consume k-distributed eigenvectors; use build_sigc_matrix_KS_kgrid_blacs");
    this->build_sigc_matrix_KS(this->mf.get_eigenvectors(), this->pbc.kfrac_list, {});
    if (this->output_sigc_ks_kf)
    {
        const auto fn = path_as_directory(this->output_dir) + "self_energy_omega.dat";
        write_self_energy_omega(fn.c_str(), *this, this->mf.get_n_kpoints(),
                                this->mf.get_n_bands());
    }
}

void G0W0::build_sigc_matrix_KS_band(const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc,
                                     const std::vector<Vector3_Order<double>> &kfrac_band,
                                     const AtomPairBvKRemap<atom_t> &bvk_remap,
                                     const std::vector<int> *output_iks)
{
    comm_h.barrier();
    if (comm_h.myid == 0)
    {
        global::lib_printf("build_sigc_matrix_KS_kgrid: constructing self-energy matrix for band k-path\n");
    }
    if (is_eigvec_k_distributed_)
        throw LIBRPA_RUNTIME_ERROR(
            "G0W0::build_sigc_matrix_KS_band cannot consume k-distributed eigenvectors; use build_sigc_matrix_KS_band_blacs");
    this->build_sigc_matrix_KS(wfc, kfrac_band, bvk_remap);
    if (this->output_sigc_ks_kf)
    {
        const int n_bands = infer_target_n_bands(comm_h, wfc, this->mf.get_n_bands());
        const auto iks = output_iks == nullptr
                             ? collect_target_iks(comm_h, wfc, static_cast<int>(kfrac_band.size()))
                             : *output_iks;
        const auto stem = make_sigc_ks_imagfreq_band_stem(
            this->output_dir, output_sigc_ks_kf_band_index_++);
        write_self_energy_omega((stem + ".dat").c_str(), *this, iks, n_bands);
        write_self_energy_omega_kpoints((stem + ".kidx").c_str(), *this, iks);
    }
}

void G0W0::build_sigc_matrix_KS_kgrid_blacs(const BlacsCtxtHandler &blacs_ctxt_h,
                                                   const bool use_gpu_replace_scalapack)
{
    comm_h.barrier();
    global::ofs_myid << "build_sigc_matrix_KS_kgrid: constructing self-energy matrix for SCF k-grid with BLACS" << std::endl;
    this->build_sigc_matrix_KS_blacs(this->mf.get_eigenvectors(), this->pbc.kfrac_list,
                                     {}, blacs_ctxt_h, use_gpu_replace_scalapack,
                                     "kgrid");
    if (this->output_sigc_ks_kf)
    {
        const auto fn = path_as_directory(this->output_dir) + "self_energy_omega.dat";
        write_self_energy_omega(fn.c_str(), *this, this->mf.get_n_kpoints(),
                                this->mf.get_n_bands());
    }
}

void G0W0::build_sigc_matrix_KS_band_blacs(
    const std::map<int, std::map<int, std::map<int, ComplexMatrix>>> &wfc,
    const std::vector<Vector3_Order<double>> &kfrac_band,
    const AtomPairBvKRemap<atom_t> &bvk_remap,
    const BlacsCtxtHandler &blacs_ctxt_h,
    const bool use_gpu_replace_scalapack,
    const std::vector<int> *output_iks)
{
    comm_h.barrier();
    if (comm_h.myid == 0)
    {
        global::lib_printf("build_sigc_matrix_KS_band: constructing self-energy matrix for band k-path with BLACS\n");
    }
    const int output_band_index = output_sigc_ks_kf_band_index_;
    this->build_sigc_matrix_KS_blacs(wfc, kfrac_band, bvk_remap, blacs_ctxt_h,
                                     use_gpu_replace_scalapack,
                                     "band_" + std::to_string(output_band_index));
    if (this->output_sigc_ks_kf)
    {
        const int n_bands = is_eigvec_k_distributed_
                                ? desc_band_wfc.n()
                                : infer_target_n_bands(comm_h, wfc,
                                                       this->mf.get_n_bands());
        const auto iks = output_iks == nullptr
                             ? collect_target_iks(comm_h, wfc, static_cast<int>(kfrac_band.size()))
                             : *output_iks;
        const auto stem = make_sigc_ks_imagfreq_band_stem(
            this->output_dir, output_band_index);
        write_self_energy_omega((stem + ".dat").c_str(), *this, iks, n_bands);
        write_self_energy_omega_kpoints((stem + ".kidx").c_str(), *this, iks);
    }
    if (this->output_sigc_ks_kf || this->output_sigc_mat_kf)
        ++output_sigc_ks_kf_band_index_;
}

} // namespace librpa_int
