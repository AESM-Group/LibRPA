#include "epsilon.h"
#include <math.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <valarray>

#include "../io/fs.h"
#include "../io/global_io.h"
#include "../io/stl_io_helper.h"
#include "../math/lapack_connector.h"
#include "../math/matrix_m.h"
#include "../math/scalapack_connector.h"
#include "../math/utils_matrix_m_mpi.h"
#include "../math/utils_matrix_mpi.h"
#include "../math/vector3_order.h"
#include "../utils/base_utility.h"
#include "../utils/constants.h"
#include "../utils/libri_utils.h"
#include "../utils/profiler.h"
#include "../utils/utils_mem.h"
#include "symmetry_context.h"
#include "atom.h"
#include "atomic_basis.h"
#include "librpa_enums.h"
#include "pbc.h"
#include "utils_atomic_basis_blacs.h"

#include "../gpu/la_connector.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla_connector.h>
using namespace ddla;
#endif

#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#include <RI/global/Tensor.h>

using RI::Tensor;
using RI::Communicate_Tensors_Map_Judge::comm_map2_first;
#endif

using std::map;
using std::vector;

namespace librpa_int {

bool strict_2d_complete_wc_requested(const bool replace_w_head, const int option_dielect_func,
                                     const bool use_2d_dielectric)
{
    return replace_w_head && option_dielect_func == 3 && use_2d_dielectric;
}

void validate_strict_2d_complete_wc_runtime(const bool strict_2d_requested,
                                            const bool headwing_data_available,
                                            const bool use_scalapack_gw_wc,
                                            const bool strict_2d_metadata_configured)
{
    if (!strict_2d_requested) return;
    if (!headwing_data_available)
        throw std::logic_error(
            "strict 2D complete-Wc was requested but analytic head/wing data are unavailable");
    if (!use_scalapack_gw_wc)
        throw std::logic_error("strict 2D complete-Wc currently requires the ScaLAPACK Wc path");
    if (!strict_2d_metadata_configured)
        throw std::logic_error(
            "strict 2D complete-Wc requires producer monopole metadata; "
            "use the standalone reader-v1 driver");
}

std::string strict_2d_finite_q_diagnostics_header()
{
    return "iq,qfrac_x,qfrac_y,qfrac_z,q_physical,qhat_x,qhat_y,q_weight,head_overlap,"
           "p_head_real,p_head_imag,p_head_over_q_real,p_head_over_q_imag,"
           "p_head_body_fro,p_body_head_fro,p_body_body_fro,"
           "analytic_p_over_q_real,analytic_p_over_q_imag,wc_head_real,wc_head_imag,"
           "wc_head_body_fro,wc_body_head_fro,wc_body_body_fro,"
           "alpha_wc_head_real,alpha_wc_head_imag,alpha_wc_head_body_fro,"
           "alpha_wc_body_head_fro,alpha_wc_body_body_fro,"
           "analytic_wc_limit_real,analytic_wc_limit_imag,q_wc_head_real,"
           "q_wc_head_imag,weighted_wc_head_real,weighted_wc_head_imag,"
           "weighted_wc_head_body_fro,weighted_wc_body_head_fro,"
           "weighted_wc_body_body_fro";
}

std::string strict_2d_gamma_wc_diagnostics_header()
{
    return "ifreq,frequency,q_weight,gamma_area,wc_head_real,wc_head_imag,"
           "wc_head_body_fro,wc_body_head_fro,wc_body_body_fro,"
           "alpha_wc_head_real,alpha_wc_head_imag,alpha_wc_head_body_fro,"
           "alpha_wc_body_head_fro,alpha_wc_body_body_fro,weighted_wc_head_real,"
           "weighted_wc_head_imag,weighted_wc_head_body_fro,weighted_wc_body_head_fro,"
           "weighted_wc_body_body_fro";
}

std::vector<Vector3_Order<double>> strict_2d_diagnostic_qpoint_order(
    const std::vector<Vector3_Order<double>> &qpoints, const bool diagnostics_enabled)
{
    auto ordered = qpoints;
    if (!diagnostics_enabled) return ordered;
    const auto gamma = std::find_if(ordered.begin(), ordered.end(),
                                    [](const auto &q) { return is_gamma_point(q); });
    if (gamma == ordered.end())
        throw std::logic_error("strict 2D diagnostics require a Gamma q point");
    std::rotate(ordered.begin(), gamma, std::next(gamma));
    return ordered;
}

bool use_strict_2d_complete_wc_gamma_route(const bool replace_w_head, const int option_dielect_func,
                                           const bool use_2d_dielectric, const bool gamma_point,
                                           const bool headwing_data_available)
{
    return strict_2d_complete_wc_requested(replace_w_head, option_dielect_func,
                                           use_2d_dielectric) &&
           gamma_point && headwing_data_available;
}

using abf_qspace_complex_block_map_t =
    atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old;
using abf_rspace_complex_block_map_t =
    atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old;
using abf_rspace_dense_block_map_t =
    std::map<atom_t, std::map<atom_t, std::map<Vector3_Order<int>, ComplexMatrix>>>;

static void dump_blacs_debug_matrix(const bool debug, const std::string &output_dir,
                                    const std::string &file_name,
                                    const matrix_m<std::complex<double>> &matrix_local,
                                    const ArrayDesc &matrix_desc, const std::string &comment = "",
                                    const double threshold = 1e-15)
{
    if (!debug) return;
    print_matrix_mm_file_parallel(path_as_directory(output_dir) + file_name, matrix_local,
                                  matrix_desc, comment, threshold);
}

static std::vector<std::complex<double>> collect_normalized_blacs_column(
    const matrix_m<std::complex<double>> &matrix_local, const ArrayDesc &matrix_desc,
    const int column, const MpiCommHandler &comm_h)
{
    std::vector<std::complex<double>> vector(matrix_desc.m(), 0.0);
    const int column_local = matrix_desc.indx_g2l_c(column);
    if (column_local >= 0)
    {
        for (int row_local = 0; row_local != matrix_desc.m_loc(); ++row_local)
        {
            const int row = matrix_desc.indx_l2g_r(row_local);
            if (row >= 0) vector[row] = matrix_local(row_local, column_local);
        }
    }
    if (vector.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("strict 2D diagnostic head vector is too large");
    MPI_Allreduce(MPI_IN_PLACE, vector.data(), static_cast<int>(vector.size()),
                  MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm_h.comm);
    double norm2 = 0.0;
    for (const auto &value : vector) norm2 += std::norm(value);
    if (!(norm2 > 0.0) || !std::isfinite(norm2))
        throw LIBRPA_RUNTIME_ERROR("strict 2D diagnostic head vector has invalid norm");
    const double inverse_norm = 1.0 / std::sqrt(norm2);
    for (auto &value : vector) value *= inverse_norm;
    return vector;
}

static void project_blacs_matrix_in_basis(const matrix_m<std::complex<double>> &matrix_local,
                                          const matrix_m<std::complex<double>> &basis_local,
                                          const ArrayDesc &matrix_desc, const int active_size,
                                          matrix_m<std::complex<double>> &work_local,
                                          matrix_m<std::complex<double>> &projected_local)
{
    if (matrix_desc.m() != matrix_desc.n() || active_size < 1 || active_size > matrix_desc.m() ||
        matrix_local.nr() != matrix_desc.m_loc() || matrix_local.nc() != matrix_desc.n_loc() ||
        basis_local.nr() != matrix_desc.m_loc() || basis_local.nc() != matrix_desc.n_loc() ||
        work_local.nr() != matrix_desc.m_loc() || work_local.nc() != matrix_desc.n_loc() ||
        projected_local.nr() != matrix_desc.m_loc() || projected_local.nc() != matrix_desc.n_loc())
        throw LIBRPA_RUNTIME_ERROR(
            "strict 2D diagnostic Coulomb-basis projection dimensions are inconsistent");

    work_local.zero_out();
    projected_local.zero_out();
    ScalapackConnector::pgemm_f('N', 'N', matrix_desc.m(), active_size, matrix_desc.m(), C_ONE,
                                matrix_local.ptr(), 1, 1, matrix_desc.desc, basis_local.ptr(), 1, 1,
                                matrix_desc.desc, C_ZERO, work_local.ptr(), 1, 1, matrix_desc.desc);
    ScalapackConnector::pgemm_f('C', 'N', active_size, active_size, matrix_desc.m(), C_ONE,
                                basis_local.ptr(), 1, 1, matrix_desc.desc, work_local.ptr(), 1, 1,
                                matrix_desc.desc, C_ZERO, projected_local.ptr(), 1, 1,
                                matrix_desc.desc);
}

static Strict2dBlockMetrics collect_strict_2d_block_metrics(
    const matrix_m<std::complex<double>> &matrix_local, const ArrayDesc &matrix_desc,
    const int active_size, const MpiCommHandler &comm_h)
{
    if (active_size < 1 || active_size > matrix_desc.m() || matrix_desc.m() != matrix_desc.n())
        throw LIBRPA_RUNTIME_ERROR("strict 2D diagnostic block size is invalid");

    Strict2dBlockMetricSums local;
    for (int row_local = 0; row_local != matrix_desc.m_loc(); ++row_local)
    {
        const int row = matrix_desc.indx_l2g_r(row_local);
        if (row < 0 || row >= active_size) continue;
        for (int column_local = 0; column_local != matrix_desc.n_loc(); ++column_local)
        {
            const int column = matrix_desc.indx_l2g_c(column_local);
            if (column < 0 || column >= active_size) continue;
            accumulate_strict_2d_block_metric(local, row, column,
                                              matrix_local(row_local, column_local));
        }
    }

    Strict2dBlockMetricSums global;
    MPI_Allreduce(&local.head, &global.head, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm_h.comm);
    const std::array<double, 3> local_squared = {local.head_body_squared, local.body_head_squared,
                                                 local.body_body_squared};
    std::array<double, 3> global_squared{};
    MPI_Allreduce(local_squared.data(), global_squared.data(),
                  static_cast<int>(global_squared.size()), MPI_DOUBLE, MPI_SUM, comm_h.comm);
    global.head_body_squared = global_squared[0];
    global.body_head_squared = global_squared[1];
    global.body_body_squared = global_squared[2];
    return finalize_strict_2d_block_metrics(global);
}

static double normalized_vector_overlap(const std::vector<std::complex<double>> &lhs,
                                        const std::vector<std::complex<double>> &rhs)
{
    if (lhs.size() != rhs.size())
        throw LIBRPA_RUNTIME_ERROR("strict 2D diagnostic head vectors have different sizes");
    std::complex<double> overlap = 0.0;
    for (std::size_t i = 0; i != lhs.size(); ++i) overlap += std::conj(lhs[i]) * rhs[i];
    return std::abs(overlap);
}

static bool are_equivalent_symmetry_qpoints(const Vector3_Order<double> &lhs,
                                            const Vector3_Order<double> &rhs,
                                            const double tol = 1e-5)
{
    const auto same_component = [tol](const double lhs_component, const double rhs_component) {
        return std::abs((lhs_component - rhs_component) - std::round(lhs_component - rhs_component))
               < tol;
    };
    return same_component(lhs.x, rhs.x) && same_component(lhs.y, rhs.y)
           && same_component(lhs.z, rhs.z);
}

bool strict_2d_qmember_diagnostic_keeps(const Vector3_Order<double>& q_member_frac,
                                        const Vector3_Order<double>& selected_q_frac,
                                        const bool diagnostics_enabled)
{
    return !diagnostics_enabled
           || are_equivalent_symmetry_qpoints(q_member_frac, selected_q_frac);
}

template <typename QMap>
static typename QMap::const_iterator find_matching_symmetry_qpoint(
    const QMap& q_map,
    const Vector3_Order<double>& q_target)
{
    const auto exact_iter = q_map.find(q_target);
    if (exact_iter != q_map.end())
    {
        return exact_iter;
    }

    return std::find_if(q_map.begin(), q_map.end(), [&q_target](const auto& entry) {
        return are_equivalent_symmetry_qpoints(entry.first, q_target);
    });
}

template <typename QMap>
static typename QMap::const_iterator find_matching_internal_qpoint(
    const QMap& q_map,
    const PeriodicBoundaryData& pbc,
    const Vector3_Order<double>& q_target)
{
    const auto exact_iter = q_map.find(q_target);
    if (exact_iter != q_map.end())
    {
        return exact_iter;
    }
    const Vector3_Order<double> q_target_frac{pbc.latvec * q_target};
    return std::find_if(q_map.begin(), q_map.end(), [&pbc, &q_target_frac](const auto& entry) {
        const Vector3_Order<double> q_current_frac{pbc.latvec * entry.first};
        return are_equivalent_symmetry_qpoints(q_current_frac, q_target_frac);
    });
}

template <typename QVector>
static typename QVector::const_iterator find_matching_symmetry_qpoint_in_sequence(
    const QVector& q_sequence,
    const Vector3_Order<double>& q_target)
{
    const auto exact_iter = std::find(q_sequence.begin(), q_sequence.end(), q_target);
    if (exact_iter != q_sequence.end())
    {
        return exact_iter;
    }

    return std::find_if(q_sequence.begin(), q_sequence.end(), [&q_target](const auto& q_current) {
        return are_equivalent_symmetry_qpoints(q_current, q_target);
    });
}

static std::map<atom_t, size_t> build_atom_nabf_map(const AtomicBasis& basis_abf)
{
    std::map<atom_t, size_t> atom_nabf;
    for (atom_t atom = 0; atom != static_cast<atom_t>(basis_abf.n_atoms); ++atom)
    {
        atom_nabf[atom] = basis_abf[static_cast<int>(atom)];
    }
    return atom_nabf;
}

static ComplexMatrix to_complex_matrix(const matrix_m<std::complex<double>>& mat)
{
    ComplexMatrix complex_mat(mat.nr(), mat.nc());
    for (int row = 0; row < mat.nr(); ++row)
    {
        for (int col = 0; col < mat.nc(); ++col)
        {
            complex_mat(row, col) = mat(row, col);
        }
    }
    return complex_mat;
}

static matrix_m<std::complex<double>> to_row_major_matrix_m(const ComplexMatrix& mat)
{
    return matrix_m<std::complex<double>>(mat.nr, mat.nc, mat.c, MAJOR::ROW);
}

static void add_scaled_complex_matrix(ComplexMatrix& matrix_dst,
                                      const ComplexMatrix& matrix_src,
                                      const std::complex<double>& scale)
{
    if (matrix_dst.nr != matrix_src.nr || matrix_dst.nc != matrix_src.nc)
    {
        throw std::runtime_error("Inconsistent W(R) symmetry block dimensions");
    }
    for (int i = 0; i < matrix_dst.size; ++i)
    {
        matrix_dst.c[i] += matrix_src.c[i] * scale;
    }
}

static librpa_int::symmetry_atom_block_matrix_map_t collect_symmetry_abf_ibz_blocks_for_q(
    const abf_qspace_complex_block_map_t& blocks_by_q,
    const Vector3_Order<double>& q_ibz_internal)
{
    librpa_int::symmetry_atom_block_matrix_map_t blocks_ibz;
    for (const auto& atom_i_pair : blocks_by_q)
    {
        const auto atom_i = atom_i_pair.first;
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            const auto atom_j = atom_j_pair.first;
            const auto q_iter = find_matching_symmetry_qpoint(atom_j_pair.second, q_ibz_internal);
            if (q_iter != atom_j_pair.second.end())
            {
                blocks_ibz[atom_i][atom_j] = to_complex_matrix(q_iter->second);
            }
        }
    }
    return blocks_ibz;
}

static librpa_int::symmetry_atom_block_matrix_map_t to_ordered_symmetry_blocks(
    const atom_mapping<ComplexMatrix>::pair_t_old& atom_blocks)
{
    librpa_int::symmetry_atom_block_matrix_map_t ordered_blocks;
    for (const auto& atom_i_pair : atom_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            ordered_blocks[atom_i_pair.first][atom_j_pair.first] = atom_j_pair.second;
        }
    }
    return ordered_blocks;
}

static atom_mapping<ComplexMatrix>::pair_t_old to_atom_mapping_blocks(
    const librpa_int::symmetry_atom_block_matrix_map_t& ordered_blocks)
{
    atom_mapping<ComplexMatrix>::pair_t_old atom_blocks;
    for (const auto& atom_i_pair : ordered_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            atom_blocks[atom_i_pair.first][atom_j_pair.first] = atom_j_pair.second;
        }
    }
    return atom_blocks;
}

static std::set<std::pair<atom_t, atom_t>> collect_symmetry_atom_pairs(
    const librpa_int::symmetry_atom_block_matrix_map_t& atom_blocks)
{
    std::set<std::pair<atom_t, atom_t>> atom_pairs;
    for (const auto& atom_i_pair : atom_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            atom_pairs.insert({atom_i_pair.first, atom_j_pair.first});
        }
    }
    return atom_pairs;
}

static std::set<std::pair<atom_t, atom_t>> collect_symmetry_atom_pairs(
    const atom_mapping<ComplexMatrix>::pair_t_old& atom_blocks)
{
    std::set<std::pair<atom_t, atom_t>> atom_pairs;
    for (const auto& atom_i_pair : atom_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            atom_pairs.insert({atom_i_pair.first, atom_j_pair.first});
        }
    }
    return atom_pairs;
}

static std::set<std::pair<atom_t, atom_t>> collect_all_upper_atom_pairs(
    const std::map<atom_t, size_t>& atom_nabf)
{
    std::set<std::pair<atom_t, atom_t>> atom_pairs;
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        for (std::size_t atom_j = atom_i; atom_j < atom_nabf.size(); ++atom_j)
        {
            atom_pairs.insert({static_cast<atom_t>(atom_i), static_cast<atom_t>(atom_j)});
        }
    }
    return atom_pairs;
}

static std::set<std::pair<atom_t, atom_t>> collect_local_target_atom_pairs_from_qspace(
    const abf_qspace_complex_block_map_t& blocks_by_q)
{
    std::set<std::pair<atom_t, atom_t>> target_atom_pairs;
    for (const auto& atom_i_pair : blocks_by_q)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            if (!atom_j_pair.second.empty())
            {
                target_atom_pairs.insert({atom_i_pair.first, atom_j_pair.first});
            }
        }
    }
    return target_atom_pairs;
}

static std::vector<int> build_symmetry_atom_offsets(const std::map<atom_t, size_t>& atom_nabf)
{
    std::vector<int> offsets(atom_nabf.size() + 1, 0);
    for (std::size_t atom = 0; atom < atom_nabf.size(); ++atom)
    {
        offsets[atom + 1] = offsets[atom]
                            + static_cast<int>(atom_nabf.at(static_cast<atom_t>(atom)));
    }
    return offsets;
}

static ComplexMatrix build_dense_symmetry_hermitian_matrix_from_local_blocks(
    const librpa_int::symmetry_atom_block_matrix_map_t& local_blocks,
    const std::map<atom_t, size_t>& atom_nabf)
{
    const auto offsets = build_symmetry_atom_offsets(atom_nabf);
    ComplexMatrix dense(offsets.back(), offsets.back());
    for (const auto& atom_i_pair : local_blocks)
    {
        const int row_offset = offsets.at(static_cast<std::size_t>(atom_i_pair.first));
        const int expected_nrows = static_cast<int>(atom_nabf.at(atom_i_pair.first));
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            const int col_offset = offsets.at(static_cast<std::size_t>(atom_j_pair.first));
            const int expected_ncols = static_cast<int>(atom_nabf.at(atom_j_pair.first));
            const auto& block = atom_j_pair.second;
            if (block.nr != expected_nrows || block.nc != expected_ncols)
            {
                std::ostringstream oss;
                oss << "Dense W(q) symmetry restore block dimension mismatch for atom pair ("
                    << atom_i_pair.first << "," << atom_j_pair.first << "): block=" << block.nr
                    << "x" << block.nc << ", expected=" << expected_nrows << "x"
                    << expected_ncols;
                throw std::runtime_error(oss.str());
            }
            for (int row = 0; row < block.nr; ++row)
            {
                for (int col = 0; col < block.nc; ++col)
                {
                    const auto value = block(row, col);
                    dense(row_offset + row, col_offset + col) = value;
                    if (atom_i_pair.first != atom_j_pair.first)
                    {
                        dense(col_offset + col, row_offset + row) = std::conj(value);
                    }
                }
            }
        }
    }
    return dense;
}

static librpa_int::symmetry_atom_block_matrix_map_t build_symmetry_blocks_from_dense_matrix(
    const ComplexMatrix& dense_matrix,
    const std::map<atom_t, size_t>& atom_nabf)
{
    const auto offsets = build_symmetry_atom_offsets(atom_nabf);
    librpa_int::symmetry_atom_block_matrix_map_t atom_blocks;
    for (std::size_t atom_i = 0; atom_i < atom_nabf.size(); ++atom_i)
    {
        const int row_offset = offsets.at(atom_i);
        const int nrows = static_cast<int>(atom_nabf.at(static_cast<atom_t>(atom_i)));
        for (std::size_t atom_j = atom_i; atom_j < atom_nabf.size(); ++atom_j)
        {
            const int col_offset = offsets.at(atom_j);
            const int ncols = static_cast<int>(atom_nabf.at(static_cast<atom_t>(atom_j)));
            ComplexMatrix block(nrows, ncols);
            for (int row = 0; row < nrows; ++row)
            {
                for (int col = 0; col < ncols; ++col)
                {
                    block(row, col) = dense_matrix(row_offset + row, col_offset + col);
                }
            }
            atom_blocks[static_cast<atom_t>(atom_i)][static_cast<atom_t>(atom_j)] =
                std::move(block);
        }
    }
    return atom_blocks;
}

static librpa_int::symmetry_atom_block_matrix_map_t gather_symmetry_ibz_blocks_collective(
    const MpiCommHandler& comm_h,
    const librpa_int::symmetry_atom_block_matrix_map_t& blocks_ibz_local,
    const std::map<atom_t, size_t>& atom_nabf)
{
    if (comm_h.nprocs <= 1)
    {
        return blocks_ibz_local;
    }

    const auto dense_ibz_local =
        build_dense_symmetry_hermitian_matrix_from_local_blocks(blocks_ibz_local, atom_nabf);
    ComplexMatrix dense_ibz_global(dense_ibz_local.nr, dense_ibz_local.nc);
    allreduce_ComplexMatrix(dense_ibz_local, dense_ibz_global, comm_h.comm);
    return build_symmetry_blocks_from_dense_matrix(dense_ibz_global, atom_nabf);
}

static abf_rspace_dense_block_map_t allocate_symmetry_full_wr_storage(
    const std::set<std::pair<atom_t, atom_t>>& target_atom_pairs,
    const std::vector<Vector3_Order<int>>& Rlist,
    const std::map<atom_t, size_t>& atom_nabf)
{
    abf_rspace_dense_block_map_t blocks_by_R_dense;
    for (const auto& atom_pair : target_atom_pairs)
    {
        const auto atom_i = atom_pair.first;
        const auto atom_j = atom_pair.second;
        const int n_i = static_cast<int>(atom_nabf.at(atom_i));
        const int n_j = static_cast<int>(atom_nabf.at(atom_j));
        for (const auto& R : Rlist)
        {
            blocks_by_R_dense[atom_i][atom_j][R] = ComplexMatrix(n_i, n_j);
        }
    }
    return blocks_by_R_dense;
}

static abf_rspace_complex_block_map_t convert_dense_rspace_blocks_to_row_major(
    const abf_rspace_dense_block_map_t& dense_blocks)
{
    abf_rspace_complex_block_map_t row_major_blocks;
    for (const auto& atom_i_pair : dense_blocks)
    {
        for (const auto& atom_j_pair : atom_i_pair.second)
        {
            for (const auto& R_block : atom_j_pair.second)
            {
                row_major_blocks[atom_i_pair.first][atom_j_pair.first][R_block.first] =
                    to_row_major_matrix_m(R_block.second);
            }
        }
    }
    return row_major_blocks;
}

static std::complex<double> build_ft_wq_phase(const PeriodicBoundaryData& pbc,
                                              const Vector3_Order<double>& q_internal,
                                              const Vector3_Order<int>& R)
{
    const auto q_frac = pbc.latvec * q_internal;
    const double ang = -(q_frac * R) * TWO_PI;
    return std::complex<double>(std::cos(ang), std::sin(ang))
           / static_cast<double>(pbc.get_n_cells_bvk());
}

static bool can_use_symmetry_qstar_wr_restore(
    const librpa_int::SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& abf_layouts,
    const std::map<atom_t, size_t>& atom_nabf,
    const PeriodicBoundaryData& pbc)
{
    if (!librpa_int::symmetry_species_layouts_match_atom_counts(
            abf_layouts, ctx.atom_to_type, atom_nabf))
    {
        return false;
    }
    return ctx.available
           && !ctx.kstars.empty()
           && ctx.kstars.size() == pbc.kfrac_list.size()
           && !ctx.kstar_grid_mapping.empty()
           && ctx.atom_to_type.size() == atom_nabf.size()
           && ctx.input_coord_frac.size() == atom_nabf.size()
           && pbc.klist.size() < static_cast<std::size_t>(pbc.get_n_cells_bvk())
           && !ctx.irreducible_sector.empty()
           && !ctx.rspace_sector_stars.empty()
           && !ctx.rspace_operations.empty();
}

static bool can_symmetrize_symmetry_chi0_ibz_blocks(
    const librpa_int::SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& abf_layouts,
    const std::map<atom_t, size_t>& atom_nabf,
    const PeriodicBoundaryData& pbc)
{
    if (!librpa_int::symmetry_species_layouts_match_atom_counts(
            abf_layouts, ctx.atom_to_type, atom_nabf))
    {
        return false;
    }
    return ctx.available
           && !ctx.kstars.empty()
           && ctx.kstars.size() == pbc.kfrac_list.size()
           && ctx.atom_to_type.size() == atom_nabf.size()
           && ctx.input_coord_frac.size() == atom_nabf.size();
}

static atom_mapping<ComplexMatrix>::pair_t_old symmetrize_symmetry_chi0_ibz_blocks_if_needed(
    const MpiCommHandler& comm_h,
    const librpa_int::SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& abf_layouts,
    const atom_mapping<ComplexMatrix>::pair_t_old& blocks_ibz,
    const Vector3_Order<double>& q_ibz_internal,
    const PeriodicBoundaryData& pbc,
    const std::map<atom_t, size_t>& atom_nabf)
{
    if (!can_symmetrize_symmetry_chi0_ibz_blocks(ctx, abf_layouts, atom_nabf, pbc))
    {
        return blocks_ibz;
    }

    const auto q_iter = find_matching_symmetry_qpoint_in_sequence(pbc.klist, q_ibz_internal);
    if (q_iter == pbc.klist.end())
    {
        return blocks_ibz;
    }
    const auto iq_ibz = static_cast<std::size_t>(std::distance(pbc.klist.cbegin(), q_iter));
    if (iq_ibz >= pbc.kfrac_list.size())
    {
        return blocks_ibz;
    }

    const auto& q_ibz_frac = pbc.kfrac_list.at(iq_ibz);

    auto blocks_for_symmetrization = to_ordered_symmetry_blocks(blocks_ibz);
    auto output_atom_pairs = collect_symmetry_atom_pairs(blocks_for_symmetrization);
    if (comm_h.nprocs > 1)
    {
        blocks_for_symmetrization =
            gather_symmetry_ibz_blocks_collective(
                comm_h, blocks_for_symmetrization, atom_nabf);
        output_atom_pairs = collect_all_upper_atom_pairs(atom_nabf);
    }

    if (output_atom_pairs.empty())
    {
        return blocks_ibz;
    }

    const auto symmetrized_blocks = librpa_int::symmetrize_symmetry_ibz_kspace_operator_blocks(
        ctx, abf_layouts, q_ibz_frac, blocks_for_symmetrization, atom_nabf, &output_atom_pairs);
    return to_atom_mapping_blocks(symmetrized_blocks);
}

static abf_rspace_complex_block_map_t accumulate_symmetry_full_wr_from_ibz_q(
    const MpiCommHandler& comm_h,
    const librpa_int::SymmetryContext& ctx,
    const std::vector<SpeciesBasisLayout>& abf_layouts,
    const abf_qspace_complex_block_map_t& Wc_q,
    const PeriodicBoundaryData& pbc,
    const std::vector<Vector3_Order<int>>& Rlist,
    const std::map<atom_t, size_t>& atom_nabf)
{
    const auto local_target_pairs = collect_local_target_atom_pairs_from_qspace(Wc_q);
    if (ctx.kstar_grid_mapping.empty())
    {
        return {};
    }
    auto blocks_by_R_full =
        allocate_symmetry_full_wr_storage(local_target_pairs, Rlist, atom_nabf);
    const auto full_grid_member_targets =
        build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, pbc.kfrac_list);
    const bool use_full_grid_member_targets =
        full_grid_member_targets.size() == ctx.kstars.size();
    const char* qmember_env = std::getenv("LIBRPA_STRICT2D_QMEMBER_DIAG");
    const bool qmember_diagnostic_enabled = qmember_env != nullptr && qmember_env[0] != '\0';
    Vector3_Order<double> selected_q_frac;
    if (qmember_diagnostic_enabled)
    {
        std::string value(qmember_env);
        std::replace(value.begin(), value.end(), ',', ' ');
        std::istringstream input(value);
        std::string extra;
        if (!(input >> selected_q_frac.x >> selected_q_frac.y >> selected_q_frac.z)
            || (input >> extra))
        {
            throw LIBRPA_RUNTIME_ERROR(
                "LIBRPA_STRICT2D_QMEMBER_DIAG requires three comma-separated reduced coordinates");
        }
    }
    std::size_t kept_qmembers = 0;

    for (const auto& star_mapping : ctx.kstar_grid_mapping)
    {
        const auto& star = ctx.kstars.at(static_cast<std::size_t>(star_mapping.star_list_index));

        const auto q_ibz_internal = pbc.klist.at(static_cast<std::size_t>(star_mapping.iq_ibz));
        const auto q_ibz_frac = pbc.kfrac_list.at(static_cast<std::size_t>(star_mapping.iq_ibz));
        const auto blocks_ibz_local =
            collect_symmetry_abf_ibz_blocks_for_q(Wc_q, q_ibz_internal);
        auto blocks_ibz =
            gather_symmetry_ibz_blocks_collective(comm_h, blocks_ibz_local, atom_nabf);
        if (local_target_pairs.empty() || blocks_ibz.empty())
        {
            continue;
        }
        const auto rotation_atom_pairs =
            librpa_int::build_symmetry_upper_atom_pair_closure(star, local_target_pairs);
        blocks_ibz = librpa_int::symmetrize_symmetry_ibz_kspace_operator_blocks(
            ctx, abf_layouts, q_ibz_frac, blocks_ibz, atom_nabf, &rotation_atom_pairs);
        if (star.members.size() != star_mapping.member_q_bz_keys.size())
        {
            throw std::runtime_error(
                "Symmetry q-star mapping is inconsistent with the loaded full-q keys");
        }

        for (std::size_t imember = 0; imember < star.members.size(); ++imember)
        {
            const auto& member = star.members[imember];
            const Vector3_Order<double> raw_q_bz_target_frac =
                use_full_grid_member_targets
                    ? full_grid_member_targets[static_cast<std::size_t>(
                          star_mapping.star_list_index)][imember]
                    : Vector3_Order<double>{pbc.latvec * star_mapping.member_q_bz_keys[imember]};
            const Vector3_Order<double> q_bz_target_frac =
                restrict_fractional_coordinate(raw_q_bz_target_frac);
            if (!strict_2d_qmember_diagnostic_keeps(
                    q_bz_target_frac, selected_q_frac, qmember_diagnostic_enabled))
            {
                continue;
            }
            ++kept_qmembers;
            librpa_int::symmetry_atom_block_matrix_map_t rotated_blocks;
            try
            {
                rotated_blocks = librpa_int::rotate_symmetry_kspace_operator_blocks(
                    ctx, abf_layouts, member, blocks_ibz, atom_nabf, star.k_ibz,
                    member.time_reversal, &local_target_pairs, &q_bz_target_frac);
            }
            catch (const std::exception& ex)
            {
                std::ostringstream oss;
                oss << "Symmetry irreducible-sector W(q)->W(R) accumulation failed for star="
                    << star.star_index << ", member=" << imember
                    << ", spatial_isym=" << member.spatial_isym
                    << ", time_reversal=" << member.time_reversal
                    << ": " << ex.what();
                throw std::runtime_error(oss.str());
            }

            const Vector3_Order<double> q_internal{q_bz_target_frac * pbc.G};
            for (const auto& atom_i_pair : rotated_blocks)
            {
                for (const auto& atom_j_pair : atom_i_pair.second)
                {
                    for (const auto& R : Rlist)
                    {
                        const auto phase = build_ft_wq_phase(pbc, q_internal, R);
                        add_scaled_complex_matrix(
                            blocks_by_R_full.at(atom_i_pair.first).at(atom_j_pair.first).at(R),
                            atom_j_pair.second, phase);
                    }
                }
            }
        }
    }

    if (qmember_diagnostic_enabled && kept_qmembers != 1)
    {
        std::ostringstream message;
        message << "LIBRPA_STRICT2D_QMEMBER_DIAG selected " << kept_qmembers
                << " full-BZ q members instead of one";
        throw LIBRPA_RUNTIME_ERROR(message.str());
    }
    if (qmember_diagnostic_enabled)
    {
        global::lib_printf_root(
            "Strict 2D diagnostic retained one full-BZ q member at reduced coordinates "
            "(% .12f, % .12f, % .12f) with the original 1/Nk Fourier weight.\n",
            selected_q_frac.x, selected_q_frac.y, selected_q_frac.z);
    }

    return convert_dense_rspace_blocks_to_row_major(blocks_by_R_full);
}
CorrEnergy compute_RPA_correlation_blacs_2d_gamma_only(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat,
                                                       const std::vector<atpair_t> &local_atpair,
                                                       const BlacsCtxtHandler &blacs_h, bool use_gpu_replace_scalapack)
{
    using librpa_int::ArrayDesc;
    using librpa_int::global::ofs_myid;
    using librpa_int::global::lib_printf;

    CorrEnergy corr;
    if (blacs_h.myid == 0)
    {
        if (use_gpu_replace_scalapack)
            lib_printf("Calculating EcRPA with BLACS GPU 2D gamma_only\n");
        else
            lib_printf("Calculating EcRPA with BLACS/ScaLAPACK 2D gamma_only\n");
    }

    // lib_printf("Calculating EcRPA with BLACS, pid:  %d\n", comm_h.myid);
    const auto &mf = chi0.mf;
    const double CONE = 1.0;
    const int n_abf = chi0.atbasis_abf.nb_total;
    const auto part_range = chi0.atbasis_abf.get_part_range();
    auto nbs_ = chi0.atbasis_abf.get_atom_nbs();
    // std::cout << "n_abf " << n_abf << std::endl;
    // std::cout << "n_atoms " << chi0.atbasis_abf.n_atoms << std::endl;
    // std::cout << "part_range " << part_range[0] << " " << part_range[1] << std::endl;
    // std::cout << "nbs_ " << nbs_[0] << " " << nbs_[1] << std::endl;

    const auto &comm_h = blacs_h.comm_h();
    comm_h.barrier();

    ArrayDesc desc_nabf_nabf(blacs_h);
    // use a square blocksize instead max block, otherwise heev and inversion will complain about illegal parameter
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    ArrayDesc desc_nabf_nabf_opt(blacs_h);
    const int nb_opt = std::min(128, desc_nabf_nabf.nb());
    desc_nabf_nabf_opt.init(n_abf, n_abf, nb_opt, nb_opt, 0, 0);
    const auto set_IJ_nabf_nabf = get_necessary_IJ_from_block_2D_sy('U', chi0.atbasis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto temp_block = init_local_mat<double>(desc_nabf_nabf, MAJOR::COL);
    auto chi0_block = init_local_mat<double>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_block = init_local_mat<double>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<double>(desc_nabf_nabf_opt, MAJOR::COL);
    std::vector<int> ipiv(desc_nabf_nabf_opt.m_loc() + desc_nabf_nabf_opt.mb());

    double* chi0_block_ptr = chi0_block.ptr();
    double* coul_block_ptr = coul_block.ptr();
    double* coul_chi0_block_ptr = coul_chi0_block.ptr();
    int* ipiv_ptr = ipiv.data();
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if(use_gpu_replace_scalapack)
    {
        desc_nabf_nabf_opt.set_ddla_desc(blacs_h.ddla_handle);
        DEVICE_CHECK(deviceMallocAsync((void**)&chi0_block_ptr, chi0_block.size() * sizeof(double), blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&coul_block_ptr, coul_block.size() * sizeof(double), blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&coul_chi0_block_ptr, coul_chi0_block.size() * sizeof(double), blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceMallocAsync((void**)&ipiv_ptr, ipiv.size() * sizeof(int), blacs_h.ddla_handle->stream));
    }
#endif
    const auto &qpts = chi0.active_qpoints();
    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    if(comm_h.is_root())
        lib_printf("Finish init RPA blacs 2d\n");
#ifdef LIBRPA_USE_LIBRI
    for (const auto &q: qpts)
    {
        coul_block.zero_out();

        int iq = std::distance(qpts.begin(), std::find(qpts.begin(), qpts.end(), q));
        std::array<double, 3> qa = {q.x, q.y, q.z};
        // collect the block elements of coulomb matrices
        {
            double vq_begin = omp_get_wtime();
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<double>>>
                coul_libri;

            for (const auto& Mu_coulmat: coulmat)
            {
                const auto Mu = Mu_coulmat.first;
                for(const auto& Nu_coulmat : Mu_coulmat.second){
                    const auto Nu = Nu_coulmat.first;
                    const auto &Vq = coulmat.at(Mu).at(Nu).at(q);
                    const auto n_mu = chi0.atbasis_abf.get_atom_nb(Mu);
                    const auto n_nu = chi0.atbasis_abf.get_atom_nb(Nu);
                    matrix tmp_vq_real=(*Vq).real();
                    std::valarray<double> Vq_va(tmp_vq_real.c, Vq->size);
                    auto pvq = std::make_shared<std::valarray<double>>();
                    *pvq = Vq_va;
                    coul_libri[Mu][{Nu, std::array<double, 3>{0,0,0}}] = Tensor<double>({n_mu, n_nu}, pvq);
                    // coulmat.at(Mu).at(Nu).at(q).reset();
                }
            }

            release_free_mem();

            // printf("Finish RPA blacs 2d  vq arr\n");
            double arr_end = omp_get_wtime();
            comm_h.barrier();
            double comm_begin = omp_get_wtime();
            //printf("Begin comm_map2_first  myid: %d\n",comm_h.myid);
            const auto IJq_coul = comm_map2_first(comm_h.comm, coul_libri, s0_s1.first, s0_s1.second);
            double comm_end = omp_get_wtime();
            comm_h.barrier();

            double block_begin = omp_get_wtime();

            collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, chi0.atbasis_abf,
                        qa,true, CONE, IJq_coul, MAJOR::ROW);
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                          coul_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                          blacs_h.ictxt);
            double block_end = omp_get_wtime();
            lib_printf("Vq Time  myid: %d  arr_time: %f  comm_time: %f   block_time: %f   pair_size: %d\n",comm_h.myid,arr_end-vq_begin, comm_end-comm_begin, block_end-block_begin,set_IJ_nabf_nabf.size());
            comm_h.barrier();
            double vq_end = omp_get_wtime();

            if(comm_h.myid == 0)
                lib_printf(" | Total vq time: %f  lri_coul: %f   comm_vq: %f   block_vq: %f\n",vq_end-vq_begin, comm_begin-vq_begin,block_begin-comm_begin,vq_end-block_begin);
        }

        double chi_arr_time = 0.0;
        double chi_comm_time = 0.0;
        double chi_2d_time = 0.0;
        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            // const auto ifreq = chi0.tfg.get_freq_index(freq);
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            double pi_freq_begin = omp_get_wtime();
            chi0_block.zero_out();
            {
                double chi_begin_arr = omp_get_wtime();
                std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<double>>>
                    chi0_libri;
                atom_mapping<ComplexMatrix>::pair_t_old chi0_wq;
                if (!chi0.get_chi0_q().empty()) chi0_wq = chi0.get_chi0_q().at(freq).at(q);

                for (const auto &M_Nchi : chi0_wq)
                {
                    const auto &M = M_Nchi.first;
                    const auto n_mu = chi0.atbasis_abf.get_atom_nb(M);
                    for (const auto &N_chi: M_Nchi.second)
                    {
                        const auto &N = N_chi.first;
                        const auto n_nu = chi0.atbasis_abf.get_atom_nb(N);
                        const auto &chi = N_chi.second.real();
                        std::valarray<double> chi_va(chi.c, chi.size);
                        auto pchi = std::make_shared<std::valarray<double>>();
                        *pchi = chi_va;
                        chi0_libri[M][{N, std::array<double, 3>{0, 0, 0}}] =
                            Tensor<double>({n_mu, n_nu}, pchi);
                    }
                }

                // if(comm_h.is_root())
                // {
                //     lib_printf("Begin to clean chi0 !!! \n");
                //     system("free -m");
                //     lib_printf("chi0_freq_q size: %d\n",chi0_wq.size());
                // }
                if (!chi0.get_chi0_q().empty()) chi0.free_chi0_q(freq, q);

                release_free_mem();

                // if(comm_h.is_root())
                // {
                //     lib_printf("After clean chi0 !!! \n");
                //     system("free -m");
                //     lib_printf("chi0_freq_q size: %d\n",chi0_wq.size());
                // }

                comm_h.barrier();
                double chi_end_arr = omp_get_wtime();
                // ofs_myid << "chi0_libri" << endl << chi0_libri;

                const auto IJq_chi0 = comm_map2_first(comm_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                // ofs_myid << "IJq_chi0" << endl << IJq_chi0;
                double chi_end_comm = omp_get_wtime();

                collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(temp_block, desc_nabf_nabf, chi0.atbasis_abf,
                        qa,true, CONE, IJq_chi0, MAJOR::ROW);
                ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                              chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                              blacs_h.ictxt);
                //printf("End collect block myid: %d ifreq: %d   TIME_USED: %f\n",comm_h.myid,ifreq,chi_end_comm-chi_end_arr);
                comm_h.barrier();
                double chi_end_2d = omp_get_wtime();

                chi_arr_time = (chi_end_arr - chi_begin_arr);
                chi_comm_time = (chi_end_comm - chi_end_arr);
                chi_2d_time = (chi_end_2d - chi_end_comm);
            }
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
            if (use_gpu_replace_scalapack)
            {
                DEVICE_CHECK(deviceMemcpyAsync(chi0_block_ptr, chi0_block.ptr(), chi0_block.size() * sizeof(double), deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
                DEVICE_CHECK(deviceMemcpyAsync(coul_block_ptr, coul_block.ptr(), coul_block.size() * sizeof(double), deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
            }
#endif
            double pi_begin = omp_get_wtime();
            LaConnector::pgemm(
                'N', 'N', n_abf, n_abf, n_abf, -1.0, coul_block_ptr, 1, 1,
                desc_nabf_nabf_opt, chi0_block_ptr, 1, 1, desc_nabf_nabf_opt,
                0.0, coul_chi0_block_ptr, 1, 1, desc_nabf_nabf_opt);
            double pi_end = omp_get_wtime();
            double trace_pi = 0.0;
            double trace_pi_loc = 0.0;
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
            if (use_gpu_replace_scalapack)
            {
                DEVICE_CHECK(deviceMemcpyAsync(coul_chi0_block.ptr(), coul_chi0_block_ptr, coul_chi0_block.size() * sizeof(double), deviceMemcpyDeviceToHost, blacs_h.ddla_handle->stream));
                DEVICE_CHECK(deviceStreamSynchronize(blacs_h.ddla_handle->stream));
            }
#endif
            for (int i = 0; i != n_abf; i++)
            {
                const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                if (ilo >= 0 && jlo >= 0) trace_pi_loc -= coul_chi0_block(ilo, jlo);
            }
            LaConnector::pdam(1.0, coul_chi0_block_ptr, desc_nabf_nabf_opt);
            int info = -1;
            double det_begin = omp_get_wtime();
            LaConnector::pgetrf_bpiv(n_abf, n_abf, coul_chi0_block_ptr, 1, 1, desc_nabf_nabf_opt, ipiv_ptr, info);
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
            if (use_gpu_replace_scalapack)
            {
                DEVICE_CHECK(deviceMemcpyAsync(coul_chi0_block.ptr(), coul_chi0_block_ptr, coul_chi0_block.size() * sizeof(double), deviceMemcpyDeviceToHost, blacs_h.ddla_handle->stream));
                DEVICE_CHECK(deviceStreamSynchronize(blacs_h.ddla_handle->stream));
            }
#endif
            assert(info == 0);
            double trf_end = omp_get_wtime();

            double ln_det_loc = 0.0;
            double ln_det = 0.0;

            for (int ig = 0; ig != n_abf; ig++)
            {
                int locr = desc_nabf_nabf_opt.indx_g2l_r(ig);
                int locc = desc_nabf_nabf_opt.indx_g2l_c(ig);
                if (locr >= 0 && locc >= 0)
                {
                    double tmp_ln_det;
                    if (coul_chi0_block(locr, locc) > 0)
                    {
                        tmp_ln_det = std::log(coul_chi0_block(locr, locc));
                    }
                    else
                    {
                        tmp_ln_det = std::log(-coul_chi0_block(locr, locc));
                    }
                    ln_det_loc += tmp_ln_det;
                }
            }

            MPI_Allreduce(&ln_det_loc, &ln_det, 1, MPI_DOUBLE, MPI_SUM, desc_nabf_nabf_opt.comm());
            //printf("End det  myid: %d ifreq: %d \n",comm_h.myid,ifreq);
            double det_end = omp_get_wtime();
            comm_h.barrier();
            MPI_Allreduce(&trace_pi_loc, &trace_pi, 1, MPI_DOUBLE, MPI_SUM, comm_h.comm);
            double pi_freq_end = omp_get_wtime();

            if(comm_h.myid==0)
            {
                lib_printf("| TIME of DET-freq-q:  %f,  q: ( %f, %f, %f)  TOT: %f  CHI_arr: %f  CHI_comm: %f, CHI_2d: %f, Pi: %f, Det: %f\n",freq, q.x,q.y,q.z,pi_freq_end-pi_freq_begin, chi_arr_time,chi_comm_time,chi_2d_time,pi_end-pi_begin,det_end-pi_end);
                complex<double> rpa_for_omega_q = complex<double>(trace_pi + ln_det);
                const auto qweight = chi0.q_weight(q);
                cRPA_q[q] += rpa_for_omega_q * freq_weight * qweight / TWO_PI;//!check
                tot_RPA_energy += rpa_for_omega_q * freq_weight * qweight / TWO_PI;
            }
        }
    }
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if(use_gpu_replace_scalapack){
        DEVICE_CHECK(deviceFreeAsync(chi0_block_ptr, blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(coul_block_ptr, blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(coul_chi0_block_ptr, blacs_h.ddla_handle->stream));
        DEVICE_CHECK(deviceFreeAsync(ipiv_ptr, blacs_h.ddla_handle->stream));
    }

#endif
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    if(comm_h.myid==0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
        }
    }
    comm_h.barrier();
    corr.value = tot_RPA_energy;

    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_RPA_correlation_blacs_2d(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat,
                                            const vector<atpair_t> &local_atpair,
                                            const BlacsCtxtHandler &blacs_h,
                                            const RpaHeadwingSettings &headwing_settings,
                                            diele_func *df_headwing)
{
    using librpa_int::global::ofs_myid;
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;

    profiler.start("compute_RPA_correlation_blacs_2d");

    const auto &comm_h = blacs_h.comm_h();
    lib_printf("Begin to compute_RPA_correlation_blacs_2d  myid: %d\n", comm_h.myid );
    release_free_mem();
    CorrEnergy corr;
    if (comm_h.myid == 0)
        lib_printf("Calculating EcRPA with BLACS/ScaLAPACK 2D\n");
    // lib_printf("Calculating EcRPA with BLACS, pid:  %d\n", comm_h.myid);
    // const auto & mf = chi0.mf;
    const int n_abf = chi0.atbasis_abf.nb_total;
    const auto part_range = chi0.atbasis_abf.get_part_range();

    comm_h.barrier();

    ArrayDesc desc_nabf_nabf(blacs_h);
    // use a square blocksize instead max block, otherwise heev and inversion will complain about illegal parameter
    desc_nabf_nabf.init_square_blk(n_abf, n_abf, 0, 0);
    ArrayDesc desc_nabf_nabf_opt(blacs_h);
    const int nb_opt = std::min(128, desc_nabf_nabf.nb());
    desc_nabf_nabf_opt.init(n_abf, n_abf, nb_opt, nb_opt, 0, 0);
    const auto set_IJ_nabf_nabf = get_necessary_IJ_from_block_2D_sy('U', chi0.atbasis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    auto temp_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_block = chi0_block.copy();
    auto coul_eigen_block = chi0_block.copy();
    auto coul_chi0_block = chi0_block.copy();
    ArrayDesc desc_headwing_response(blacs_h);
    matrix_m<std::complex<double>> headwing_response_block;
    // ofs_myid << "Iset Jset " << s0_s1 << endl;
    // ofs_myid << "atpair_unordered_local of myid " << blacs_h.myid << " " << atpair_unordered_local << endl;

    // NOTE: this may change later when q-points are parallelized
    vector<Vector3_Order<double>> qpts(chi0.active_qpoints());
    // const auto &chi0q = chi0.get_chi0_q();
    // for (const auto &qMuNuchi: chi0q.at(chi0.tfg.get_freq_nodes()[0]))
    //     qpts.push_back(qMuNuchi.first);
    ofs_myid << "compute_RPA_correlation_blacs_2d handling qpts: " << qpts.size() << std::endl;
    // return corr;

    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    if(comm_h.is_root()) lib_printf("Finish init RPA blacs 2d\n");
    comm_h.barrier();
    // ofs_myid << "atpair_unordered_local of myid " << blacs_ctxt_global_h.myid << " " <<
    // atpair_unordered_local << endl;

#ifdef LIBRPA_USE_LIBRI

    for (const auto &q : qpts)
    {
        coul_block.zero_out();

        // int iq = chi0.pbc.get_k_index_full(q);
        std::array<double, 3> qa = {q.x, q.y, q.z};
        // collect the block elements of coulomb matrices
        {
            double vq_begin = omp_get_wtime();
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
                coul_libri;
            coul_libri.clear();
            ofs_myid << "Initializing coul_libri Tensors" << std::endl;
            for (const auto &Mu_Nu: local_atpair)
            {
                const auto Mu = Mu_Nu.first;
                const auto Nu = Mu_Nu.second;
                // ofs_myid << "myid " << blacs_h.myid << "Mu " << Mu << " Nu " << Nu << endl;
                #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
                // printf("success before if coulmat.count:%d\n", mpi_comm_global_h.myid);
                #endif
                if (coulmat.count(Mu) == 0 ||
                    coulmat.at(Mu).count(Nu) == 0 ||
                    coulmat.at(Mu).at(Nu).count(q) == 0) continue;
                #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
                // printf("success after if coulmat.count:%d\n", mpi_comm_global_h.myid);
                #endif
                const auto &Vq = coulmat.at(Mu).at(Nu).at(q);
                const auto n_mu = chi0.atbasis_abf.get_atom_nb(Mu);
                const auto n_nu = chi0.atbasis_abf.get_atom_nb(Nu);
                ofs_myid << "- coul_libri Tensor Mu " << Mu << " Nu " << Nu
                         << " address " << Vq->c << " " << Vq->size << " shape " << n_mu << " x " << n_nu << std::endl;
                std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                auto pvq = std::make_shared<std::valarray<complex<double>>>();
                *pvq = Vq_va;
                coul_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
            }
            ofs_myid << "Done initializing coul_libri Tensors" << std::endl;
            //printf("Finish RPA blacs 2d  vq arr\n");
            double arr_end = omp_get_wtime();
            comm_h.barrier();
            double comm_begin = omp_get_wtime();
            //printf("Begin comm_map2_first  myid: %d\n",comm_h.myid);
            const auto IJq_coul = comm_map2_first(comm_h.comm, coul_libri, s0_s1.first, s0_s1.second);
            double comm_end = omp_get_wtime();
            comm_h.barrier();
            //printf("End vq comm_map2_first  myid: %d   TIME_USED: %f\n",comm_h.myid,comm_end-comm_begin);
            // ofs_myid << "IJq_coul" << endl << IJq_coul;
            //printf("Finish RPA blacs 2d  vq 2d\n");
            double block_begin = omp_get_wtime();
            // for (const auto &IJ: set_IJ_nabf_nabf)
            // {
            //     const auto &I = IJ.first;
            //     const auto &J = IJ.second;
            //     // cout << IJq_coul.at(I).at({J, qa});
            //     collect_block_from_IJ_storage_syhe(
            //         coul_block, desc_nabf_nabf, chi0.atbasis_abf, IJ.first,
            //         IJ.second, true, CONE, IJq_coul.at(I).at({J, qa}).ptr(), MAJOR::ROW);
            //     // lib_printf("myid %d I %d J %d nr %d nc %d\n%s",
            //     //        blacs_h.myid, I, J,
            //     //        coul_block.nr(), coul_block.nc(),
            //     //        str(coul_block).c_str());
            // }

            if (IJq_coul.size() > 0)
            {
                collect_block_from_ALL_IJ_Tensor(temp_block, desc_nabf_nabf, chi0.atbasis_abf,
                                                qa, true, C_ONE, IJq_coul, MAJOR::ROW);
                ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                              coul_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                              blacs_h.ictxt);
            }
            double block_end = omp_get_wtime();
            lib_printf("Vq Time  myid: %d  arr_time: %f  comm_time: %f   block_time: %f   pair_size: %d\n",comm_h.myid,arr_end-vq_begin, comm_end-comm_begin, block_end-block_begin,set_IJ_nabf_nabf.size());
            comm_h.barrier();
            double vq_end = omp_get_wtime();

            if(comm_h.myid == 0)
                lib_printf(" | Total vq time: %f  lri_coul: %f   comm_vq: %f   block_vq: %f\n",vq_end-vq_begin, comm_begin-vq_begin,block_begin-comm_begin,vq_end-block_begin);
        }


        //if(comm_h.is_root())
        //printf("Finish RPA blacs 2d  vq comm\n");
        // char fn[100];
        // sprintf(fn, "coul_iq_%d.mtx", iq);
        // print_matrix_mm_file_parallel(fn, coul_block, desc_nabf_nabf);
        // ofs_myid << str(coul_block);
        // lib_printf("coul_block\n%s", str(coul_block).c_str());
        const bool replace_gamma_headwing =
            headwing_settings.enabled && headwing_settings.option_dielect_func == 3 &&
            is_gamma_point(q);
        matrix_m<std::complex<double>> sqrtveig_blacs;
        int n_nonsingular_headwing = 0;
        if (replace_gamma_headwing)
        {
            if (df_headwing == nullptr)
                throw LIBRPA_RUNTIME_ERROR("RPA head/wing correction requested without headwing data");
            size_t n_singular = 0;
            vec<double> coul_eigenvalues(n_abf);
            sqrtveig_blacs = LaConnector::power_hemat_la_real(
                coul_block, desc_nabf_nabf_opt, coul_eigen_block, desc_nabf_nabf_opt,
                n_singular, coul_eigenvalues.c, 0.5,
                headwing_settings.sqrt_coulomb_threshold);
            n_nonsingular_headwing = n_abf - as_int(n_singular);
            if (headwing_settings.rpa_headwing_mode == "qavg")
                df_headwing->wing_mu_to_lambda(sqrtveig_blacs, desc_nabf_nabf,
                                               n_nonsingular_headwing);
            desc_headwing_response.init_square_blk(n_nonsingular_headwing, n_nonsingular_headwing, 0, 0);
            headwing_response_block =
                init_local_mat<std::complex<double>>(desc_headwing_response, MAJOR::COL);
        }

        double chi_arr_time = 0.0;
        double chi_comm_time = 0.0;
        double chi_2d_time = 0.0;
        for (const auto &freq: chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            double pi_freq_begin = omp_get_wtime();
            chi0_block.zero_out();
            {
                double chi_begin_arr = omp_get_wtime();
                std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>> chi0_libri;
                chi0_libri.clear();
                auto it_freq = chi0.get_chi0_q().find(freq);
                if (it_freq != chi0.get_chi0_q().cend())
                {
                    auto it_fq = it_freq->second.find(q);
                    if (it_fq != it_freq->second.cend())
                    {
                        const auto &chi0_wq = it_fq->second;
                        for (const auto &M_Nchi: chi0_wq)
                        {
                            const auto &M = M_Nchi.first;
                            const auto n_mu = chi0.atbasis_abf.get_atom_nb(M);
                            for (const auto &N_chi: M_Nchi.second)
                            {
                                const auto &N = N_chi.first;
                                const auto n_nu = chi0.atbasis_abf.get_atom_nb(N);
                                const auto &chi = N_chi.second;
                                std::valarray<complex<double>> chi_va(chi.c, chi.size);
                                auto pchi = std::make_shared<std::valarray<complex<double>>>();
                                *pchi = chi_va;
                                chi0_libri[M][{N,qa}] = Tensor<complex<double>>({n_mu, n_nu}, pchi);
                            }
                        }
                        if(comm_h.is_root())
                        {
                            lib_printf("Begin to clean chi0 !!! \n");
                            // display_free_mem();
                            lib_printf("chi0_freq_q size: %d,  freq: %f, q:( %f, %f, %f )\n",chi0_wq.size(),freq, q.x,q.y,q.z );
                        }
                        chi0.free_chi0_q(freq,q);
                    }
                }

                release_free_mem();
                // if(comm_h.is_root())
                // {
                //     lib_printf("After clean chi0 !!! \n");
                //     system("free -m");
                //     lib_printf("chi0_freq_q size: %d\n",chi0_wq.size());
                // }
                comm_h.barrier();
                double chi_end_arr = omp_get_wtime();
                // ofs_myid << "chi0_libri" << endl << chi0_libri;

                const auto IJq_chi0 =
                    comm_map2_first(comm_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                // ofs_myid << "IJq_chi0" << endl << IJq_chi0;
                double chi_end_comm = omp_get_wtime();
                if (IJq_chi0.size() > 0)
                {
                    collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(temp_block, desc_nabf_nabf, chi0.atbasis_abf,
                                                    qa, true, C_ONE, IJq_chi0, MAJOR::ROW);
                    ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                                  chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                                  blacs_h.ictxt);
                }
                comm_h.barrier();
                double chi_end_2d = omp_get_wtime();

                chi_arr_time = (chi_end_arr - chi_begin_arr);
                chi_comm_time = (chi_end_comm - chi_end_arr);
                chi_2d_time = (chi_end_2d - chi_end_comm);
                // char fnc[100];
                // sprintf(fnc, "chi_ifreq_%d_iq_%d.mtx", ifreq, iq);
                // if( ifreq== 0)
                //     print_matrix_mm_file_parallel(fnc, chi0_block, desc_nabf_nabf);
            }

            double pi_begin = omp_get_wtime();
            const bool average_gamma_headwing =
                replace_gamma_headwing && headwing_settings.option_dielect_func == 3 &&
                headwing_settings.rpa_headwing_mode == "qavg";
            const bool head_only_gamma =
                replace_gamma_headwing && headwing_settings.rpa_headwing_mode == "head_only";
            complex<double> rpa_for_omega_q = 0.0;
            bool rpa_for_omega_q_done = false;
            double headwing_proj_left_time = 0.0;
            double headwing_proj_right_time = 0.0;
            double headwing_trace_log_time = 0.0;
            if (replace_gamma_headwing)
            {
                headwing_response_block.zero_out();
                const double proj_left_begin = omp_get_wtime();
                ScalapackConnector::pgemm_f(
                    'N', 'N', n_abf, n_nonsingular_headwing, n_abf, C_ONE, chi0_block.ptr(), 1, 1,
                    desc_nabf_nabf_opt.desc, sqrtveig_blacs.ptr(), 1, 1, desc_nabf_nabf_opt.desc, C_ZERO,
                    coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
                const double proj_left_end = omp_get_wtime();
                ScalapackConnector::pgemm_f(
                    'C', 'N', n_nonsingular_headwing, n_nonsingular_headwing, n_abf, C_ONE,
                    sqrtveig_blacs.ptr(), 1, 1,
                    desc_nabf_nabf_opt.desc, coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                    C_ZERO, headwing_response_block.ptr(), 1, 1, desc_headwing_response.desc);
                const double proj_right_end = omp_get_wtime();
                headwing_proj_left_time = proj_left_end - proj_left_begin;
                headwing_proj_right_time = proj_right_end - proj_left_end;
                if (head_only_gamma)
                {
                    replace_rpa_response_head_only(headwing_response_block,
                                                   df_headwing->get_rpa_chi0v_head(ifreq),
                                                   desc_headwing_response);
                    const double trace_log_begin = omp_get_wtime();
                    rpa_for_omega_q = compute_rpa_response_trace_logdet_blacs_2d(
                        headwing_response_block, desc_headwing_response);
                    headwing_trace_log_time = omp_get_wtime() - trace_log_begin;
                    rpa_for_omega_q_done = true;
                }
                else if (!average_gamma_headwing)
                {
                    throw std::logic_error("Unsupported RPA headwing mode: "
                                           + headwing_settings.rpa_headwing_mode);
                }
            }
            else
            {
                ScalapackConnector::pgemm_f('N', 'N', n_abf, n_abf, n_abf, 1.0, coul_block.ptr(), 1, 1,
                                            desc_nabf_nabf_opt.desc, chi0_block.ptr(), 1, 1,
                                            desc_nabf_nabf_opt.desc, 0.0, coul_chi0_block.ptr(), 1, 1,
                                            desc_nabf_nabf_opt.desc);
            }
            // char fnp[100];
            // sprintf(fnp, "pi_ifreq_%d_iq_%d.mtx", ifreq, iq);
            double pi_end = omp_get_wtime();

            if (rpa_for_omega_q_done)
            {
                // Already evaluated in the reduced Coulomb-eigenvector subspace.
            }
            else if (average_gamma_headwing)
            {
                const double trace_log_begin = omp_get_wtime();
                rpa_for_omega_q = df_headwing->compute_rpa_trace_log_average(
                    headwing_response_block, ifreq, desc_headwing_response, headwing_settings);
                headwing_trace_log_time = omp_get_wtime() - trace_log_begin;
            }
            else
            {
                complex<double> trace_pi(0.0, 0.0);
                complex<double> trace_pi_loc(0.0, 0.0);
                for (int i = 0; i != n_abf; i++)
                {
                    const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                    const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                    if (ilo >= 0 && jlo >= 0) trace_pi_loc += coul_chi0_block(ilo, jlo);
                }

                coul_chi0_block *= -1.0;
                for (int i = 0; i != n_abf; i++)
                {
                    const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
                    const int jlo = desc_nabf_nabf_opt.indx_g2l_c(i);
                    if (ilo >= 0 && jlo >= 0) coul_chi0_block(ilo, jlo) += C_ONE;
                }
                // if( ifreq== 0 && comm_h.is_root() )
                //     print_whole_matrix("pi-2D-loc", coul_chi0_block);

                int *ipiv = new int[desc_nabf_nabf_opt.m_loc() * 10];
                int info;
                complex<double> ln_det =
                    compute_pi_det_blacs_2d(coul_chi0_block, desc_nabf_nabf_opt, ipiv, info);
                MPI_Allreduce(&trace_pi_loc,&trace_pi,1,MPI_DOUBLE_COMPLEX,MPI_SUM,comm_h.comm);
                delete[] ipiv;
                rpa_for_omega_q = trace_pi + ln_det;
            }
            double det_end = omp_get_wtime();
            comm_h.barrier();
            double pi_freq_end = omp_get_wtime();
            //double task_end = omp_get_wtime();
            // if(comm_h.is_root())
            //     lib_printf("| After det for freq:  %f,  q: ( %f, %f, %f)   TIME_LOCMAT: %f   TIME_DET: %f  TIME_CAL_Pi: %f, TIME_TRAN_LOC: %f\n",ifreq, q.x,q.y,q.z,task_mid-task_begin,task_end-task_mid,pi_time,loc_tran_time);
            //para_mpi.mpi_barrier();

            if(comm_h.myid==0)
            {
                lib_printf("| TIME of DET-freq-q:  %f,  q: ( %f, %f, %f)  TOT: %f  CHI_arr: %f  CHI_comm: %f, CHI_2d: %f, Pi: %f, Det: %f\n",freq, q.x,q.y,q.z,pi_freq_end-pi_freq_begin, chi_arr_time,chi_comm_time,chi_2d_time,pi_end-pi_begin,det_end-pi_end);
                if (replace_gamma_headwing)
                {
                    lib_printf("| TIME of HW-proj-freq-q: %f, q: ( %f, %f, %f)  left_chi0U: %f  right_Uchi0U: %f  trace_log_or_avg: %f\n",
                               freq, q.x, q.y, q.z, headwing_proj_left_time,
                               headwing_proj_right_time, headwing_trace_log_time);
                }
                //cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "      lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                const auto qweight = chi0.q_weight(q);
                cRPA_q[q] += rpa_for_omega_q * freq_weight * qweight / TWO_PI;//!check
                tot_RPA_energy += rpa_for_omega_q * freq_weight * qweight / TWO_PI;
            }
        }
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    if(comm_h.myid==0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
            // cout << q_crpa.first << q_crpa.second << endl;
        }
        // cout << "gx_num_" << chi0.tfg.size() << "  tot_RPA_energy:  " << setprecision(8)
        // <<tot_RPA_energy << endl;
    }
    comm_h.barrier();
    corr.value = tot_RPA_energy;

    corr.etype = CorrEnergy::type::RPA;
    profiler.stop("compute_RPA_correlation_blacs_2d");
    return corr;
}

double compute_pi_det_blacs_2d_gamma_only(matrix_m<double> &loc_piT, const ArrayDesc &arrdesc_pi, int *ipiv, int &info)
{
    const int range_all = arrdesc_pi.m();

    double det_begin = omp_get_wtime();

    ScalapackConnector::pgetrf_f(range_all, range_all, loc_piT.ptr(), 1, 1, arrdesc_pi.desc,
                                 ipiv, info);
    double trf_end = omp_get_wtime();

    double ln_det_loc = 0.0;
    double ln_det_all = 0.0;

    for (int ig = 0; ig != range_all; ig++)
    {
        int locr = arrdesc_pi.indx_g2l_r(ig);
        int locc = arrdesc_pi.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0)
        {
            double tmp_ln_det;
            if (loc_piT(locr, locc) > 0)
            {
                tmp_ln_det = std::log(loc_piT(locr, locc));
            }
            else
            {
                tmp_ln_det = std::log(-loc_piT(locr, locc));
            }
            ln_det_loc += tmp_ln_det;
        }
    }
    double ln_end = omp_get_wtime();

    MPI_Allreduce(&ln_det_loc,&ln_det_all,1,MPI_DOUBLE,MPI_SUM, arrdesc_pi.comm());
    double det_end = omp_get_wtime();
    return ln_det_all;
}

complex<double> compute_pi_det_blacs_2d(Matz &loc_piT, const ArrayDesc &arrdesc_pi, int *ipiv, int &info)
{
    int one = 1;
    const int range_all = arrdesc_pi.m();
    int DESCPI_T[9];
// if(out_pi)
// {
//     print_complex_real_matrix("first_pi",pi_freq_q.at(0).at(0));
//     print_complex_real_matrix("first_loc_piT_mat",loc_piT);
// }
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
    printf(
        "success before pzgetrf_ processid:%d,range_all: %d, loc_piT.nr(): %d, loc_piT.nc(): %d\n",
        arrdesc_pi.myid(), range_all, loc_piT.nr(), loc_piT.nc());
#endif
    double det_begin = omp_get_wtime();
    // ScalapackConnector::transpose_desc(DESCPI_T, arrdesc_pi.desc);
    pzgetrf_(&range_all, &range_all, loc_piT.ptr(), &one, &one, arrdesc_pi.desc, ipiv, &info);
    double trf_end = omp_get_wtime();
    // ScalapackConnector::pgetrf_f(range_all,range_all,loc_piT.c,one,one,DESCPI_T,ipiv, info);
    // printf("   after LU myid: %d\n",mpi_comm_global_h.myid);
    // printf("desc myid: %d,  m n: %d,%d,  mb nb: %d, %d,  loc_m_n: %d, %d, myp: %d,%d, npr,npc:
    // %d, %d\n",mpi_comm_global_h.myid, arrdesc_pi.m(),arrdesc_pi.n(),
    // arrdesc_pi.mb(),arrdesc_pi.nb(),
    // arrdesc_pi.m_loc(),arrdesc_pi.n_loc(),arrdesc_pi.myprow(),arrdesc_pi.mypcol(),arrdesc_pi.nprows(),arrdesc_pi.npcols());
    complex<double> ln_det_loc(0.0, 0.0);
    complex<double> ln_det_all(0.0, 0.0);
    // complex<double> det_loc(1.0,0.0);
    // complex<double> det_glo(0.0,0.0);
    // vector<complex<double>>  det_dig;
    // vector<complex<double>>  ln_det_dig;
    // vector<complex<double>>  det_dig_r;
    // vector<complex<double>>  det_dig_c;
    // printf(" myid: %d ig=25, locr,locc: %d,
    // %d)\n",mpi_comm_global_h.myid,arrdesc_pi.indx_g2l_r(25),arrdesc_pi.indx_g2l_c(25));
    for (int ig = 0; ig != range_all; ig++)
    {
        // int locr=para_mpi.localIndex(ig,row_nblk,para_mpi.nprow,para_mpi.myprow);
        // int locc=para_mpi.localIndex(ig,col_nblk,para_mpi.npcol,para_mpi.mypcol);
        int locr = arrdesc_pi.indx_g2l_r(ig);
        int locc = arrdesc_pi.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0)
        {
            // if(ipiv[locr]!=(ig+1))
            // 	det_loc=-1*det_loc * loc_piT(locc,locr);
            // else
            // 	det_loc=det_loc * loc_piT(locc,locr);
            // det_dig.push_back(loc_piT(locr,locc));
            // det_dig_r.push_back(locr);
            // det_dig_c.push_back(locc);
            complex<double> tmp_ln_det;
            if (loc_piT(locr, locc).real() > 0)
            {
                tmp_ln_det = std::log(loc_piT(locr, locc));
                // ln_det_dig.push_back(tmp_ln_det);
            }
            else
            {
                tmp_ln_det = std::log(-loc_piT(locr, locc));
                // ln_det_dig.push_back(tmp_ln_det);
            }
            ln_det_loc += tmp_ln_det;
        }
    }
    double ln_end = omp_get_wtime();
//     ComplexMatrix det_mm(loc_piT.nr(),loc_piT.nc());
//     for(int i=0;i!=loc_piT.nr();i++)
//         for(int j=0;j!=loc_piT.nc();j++)
//             det_mm(i,j)=loc_piT(i,j);
//    // sort(det_dig.rbegin(),det_dig.rend());
//     ComplexMatrix det_dig_mm(det_dig.size(),4);
//     for(int i=0;i!=det_dig.size();i++)
//     {
//         det_dig_mm(i,0) =det_dig_r[i];
//         det_dig_mm(i,1) =det_dig_c[i];
//         det_dig_mm(i,2)=det_dig[i];
//         det_dig_mm(i,3)=ln_det_dig[i];
//     }
//     char fn[100];
//     sprintf(fn, "det_dig_myid_%d.mtx", comm_h.myid);
//     print_complex_matrix_file("det_dig_loc", det_dig_mm, fn, false);

//     sprintf(fn, "det_mat_myid_%d.mtx", comm_h.myid);
//     print_complex_matrix_file("det_mat_loc", det_mm, fn, false);


    MPI_Allreduce(&ln_det_loc,&ln_det_all,1,MPI_DOUBLE_COMPLEX,MPI_SUM, arrdesc_pi.comm());
    double det_end = omp_get_wtime();
    // if(comm_h.myid == 0)
    //     lib_printf("    | Det time   trf: %f   ln: %f   allreduce: %f\n",trf_end-det_begin,ln_end-trf_end, det_end-ln_end);
    //MPI_Allreduce(&det_loc,&det_glo,1,MPI_DOUBLE_COMPLEX,MPI_PROD,comm_h.comm);
    //ln_det_all=std::log(det_glo);
    return ln_det_all;
}

cplxdb compute_rpa_response_trace_logdet_blacs_2d(
    const Matz &response, const ArrayDesc &response_desc)
{
    cplxdb trace_loc(0.0, 0.0);
    cplxdb trace(0.0, 0.0);
    for (int i = 0; i != response_desc.m(); ++i)
    {
        const int ilo = response_desc.indx_g2l_r(i);
        const int jlo = response_desc.indx_g2l_c(i);
        if (ilo >= 0 && jlo >= 0) trace_loc += response(ilo, jlo);
    }
    MPI_Allreduce(&trace_loc, &trace, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                  response_desc.comm());

    auto identity_minus_response = response.copy();
    identity_minus_response *= -1.0;
    for (int i = 0; i != response_desc.m(); ++i)
    {
        const int ilo = response_desc.indx_g2l_r(i);
        const int jlo = response_desc.indx_g2l_c(i);
        if (ilo >= 0 && jlo >= 0) identity_minus_response(ilo, jlo) += C_ONE;
    }

    int info = 0;
    std::vector<int> ipiv(std::max(1, response_desc.m_loc() * 10));
    const cplxdb ln_det = compute_pi_det_blacs_2d(
        identity_minus_response, response_desc, ipiv.data(), info);
    return trace + ln_det;
}

complex<double> compute_pi_det_blacs(ComplexMatrix &loc_piT, const ArrayDesc &arrdesc_pi, int *ipiv, int &info)
{
    // int range_all = atom_mu_part_range[natom-1]+atom_mu[natom-1];
    // int desc_pi[9];
    // int loc_row, loc_col, info;
    // int row_nblk=1;
    // int col_nblk=1;
    int one = 1;
    int range_all = arrdesc_pi.m();
    // para_mpi.set_blacs_mat(desc_pi,loc_row,loc_col,range_all,range_all,row_nblk,col_nblk);
    // int *ipiv = new int [loc_row*10];
    // ComplexMatrix loc_piT(loc_col,loc_row);

    // for(int i=0;i!=loc_row;i++)
    // {
    //     int global_row = para_mpi.globalIndex(i,row_nblk,para_mpi.nprow,para_mpi.myprow);
    //     int mu;
    //     int I=atom_mu_glo2loc(global_row,mu);
    //     for(int j=0;j!=loc_col;j++)
    //     {
    //         int global_col = para_mpi.globalIndex(j,col_nblk,para_mpi.npcol,para_mpi.mypcol);
    //         int nu;
    //         int J=atom_mu_glo2loc(global_col,nu);

    //         if( global_col == global_row)
    //         {
    //             loc_piT(j,i)=complex<double>(1.0,0.0) - pi_freq_q.at(I).at(J)(mu,nu);
    //         }
    //         else
    //         {
    //             loc_piT(j,i)=-1*  pi_freq_q.at(I).at(J)(mu,nu);
    //         }

    //     }
    // }
    int DESCPI_T[9];
    // if(out_pi)
    // {
    //     print_complex_real_matrix("first_pi",pi_freq_q.at(0).at(0));
    //     print_complex_real_matrix("first_loc_piT_mat",loc_piT);
    // }

    ScalapackConnector::transpose_desc(DESCPI_T, arrdesc_pi.desc);

   // para_mpi.mpi_barrier();
    //printf("   before LU Myid: %d        Available DOS memory = %ld bytes\n",comm_h.myid, memavail());
    //printf("   before LU myid: %d  range_all: %d,  loc_mat.size: %d\n",comm_h.myid,range_all,loc_piT.size);
    pzgetrf_(&range_all,&range_all,loc_piT.c,&one,&one,DESCPI_T,ipiv, &info);
    //printf("   after LU myid: %d\n",comm_h.myid);
    std::complex<double> ln_det_loc(0.0,0.0);
    std::complex<double> ln_det_all(0.0,0.0);
    for (int ig = 0; ig != range_all; ig++)
    {
        // int locr=para_mpi.localIndex(ig,row_nblk,para_mpi.nprow,para_mpi.myprow);
        // int locc=para_mpi.localIndex(ig,col_nblk,para_mpi.npcol,para_mpi.mypcol);
        int locr = arrdesc_pi.indx_g2l_r(ig);
        int locc = arrdesc_pi.indx_g2l_c(ig);
        if (locr >= 0 && locc >= 0)
        {
            // if(ipiv[locr]!=(ig+1))
            // 	det_loc=-1*det_loc * loc_piT(locc,locr);
            // else
            // 	det_loc=det_loc * loc_piT(locc,locr);
            if (loc_piT(locc, locr).real() > 0)
                ln_det_loc += std::log(loc_piT(locc, locr));
            else
                ln_det_loc += std::log(-loc_piT(locc, locr));
        }
    }
    MPI_Allreduce(&ln_det_loc,&ln_det_all,1,MPI_DOUBLE_COMPLEX,MPI_SUM,arrdesc_pi.comm());
    return ln_det_all;
}


CorrEnergy compute_RPA_correlation_blacs(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat,
                                         const vector<atpair_t> &local_atpair,
                                         const BlacsCtxtHandler &blacs_h)
{
    using librpa_int::global::ofs_myid;
    using librpa_int::global::lib_printf;

    CorrEnergy corr;
    const auto &comm_h = chi0.comm_h;
    if (comm_h.myid == 0) lib_printf("Calculating EcRPA with BLACS/ScaLAPACK row\n");

    const auto &mf = chi0.mf;
    const complex<double> CONE{1.0, 0.0};
    const int n_abf = chi0.atbasis_abf.nb_total;
    const auto &part_range = chi0.atbasis_abf.get_part_range();
    const int natom = chi0.atbasis_abf.n_atoms;

    comm_h.barrier();

    librpa_int::ArrayDesc arrdesc_pi(blacs_h);
    arrdesc_pi.init_square_blk(n_abf, n_abf, 0, 0);
    int loc_row = arrdesc_pi.m_loc(), loc_col = arrdesc_pi.n_loc(), info;

    // para_mpi.set_blacs_mat(desc_pi,loc_row,loc_col,N_all_mu,N_all_mu,row_nblk,col_nblk);
    int *ipiv = new int[loc_row * 10];
    // double vq_begin_m2t= omp_get_wtime();
    // std::map<int, std::map<std::pair<int, std::array<double, 3>>, Tensor<complex<double>>>>
    // vq_libri; for(auto &Ip:Vq)
    // {
    //     auto I=Ip.first;
    //     const auto n_mu = chi0.atbasis_abf.get_atom_nb(I);
    //     for(auto &Jp:Ip.second)
    //     {
    //         auto J=Jp.first;
    //         const auto n_nu = chi0.atbasis_abf.get_atom_nb(J);
    //         for(auto &qp:Jp.second)
    //         {
    //             auto q=qp.first;
    //             std::array<double, 3> qa = {q.x, q.y, q.z};
    //             const auto &vq_ptr=qp.second;
    //             std::valarray<complex<double>> Vq_va(vq_ptr->c, vq_ptr->size);
    //             auto pvq = std::make_shared<std::valarray<complex<double>>>();
    //             *pvq = Vq_va;
    //             vq_libri[I][{J, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
    //             if(I!=J)
    //             {
    //                 auto vqT=transpose(*vq_ptr, 1);
    //                 std::valarray<complex<double>> VqT_va(vqT.c, vqT.size);
    //                 auto pvqT = std::make_shared<std::valarray<complex<double>>>();
    //                 *pvqT = VqT_va;
    //                 vq_libri[J][{I, qa}] = Tensor<complex<double>>({n_nu, n_mu}, pvqT);
    //             }
    //         }
    //     }
    // }
    // double vq_end_m2t = omp_get_wtime();
    // set<int> loc_atp_IJ;
    // for(auto &atp:local_atpair)
    // {
    //     loc_atp_IJ.insert(atp.first);
    //     loc_atp_IJ.insert(atp.second);
    // }
    // set<int> all_atom_set;
    // for(int I=0;I!=natom;I++)
    //     all_atom_set.insert(I);
    // const auto IJq_coul = Communicate_Tensors_Map_Judge::comm_map2_first(comm_h.comm, vq_libri, all_atom_set, loc_atp_IJ);
    // atpair_k_cplx_mat_t Vq_loc;
    // double vq_end_comm = omp_get_wtime();
    // for(auto Ip:IJq_coul)
    // {
    //     auto I=Ip.first;
    //     auto n_mu=atom_mu[I];
    //     for(auto &Jqp:Ip.second)
    //     {
    //         auto J=Jqp.first.first;
    //         auto n_nu=atom_mu[J];
    //         auto qa=Jqp.first.second;
    //         Vector3_Order<double> q{qa[0],qa[1],qa[2]};
    //         shared_ptr<ComplexMatrix> vq_ptr = make_shared<ComplexMatrix>();
    //         vq_ptr->create(n_mu, n_nu);
    //         const auto length=sizeof(complex<double>)* n_mu *n_nu;
    //         memcpy((*vq_ptr).c, Jqp.second.ptr(),length);
    //         Vq_loc[I][J][q]=vq_ptr;
    //         //printf("| process %d, I: %d  J: %d\n",comm_h.myid, I,J );
    //     }
    // }
    // double vq_end_t2m = omp_get_wtime();
    // comm_h.barrier();
    // if(comm_h.is_root())
    //     lib_printf("| Vq_time %f, TIME_m2t: %f   TIME_comm: %f  TIME_t2m: %f\n",vq_end_t2m-vq_begin_m2t,vq_end_m2t-vq_begin_m2t,vq_end_comm-vq_end_m2t,vq_end_t2m-vq_end_comm);
    map<double, map<Vector3_Order<double>, ComplexMatrix>> pi_freq_q;
    complex<double> tot_RPA_energy(0.0, 0.0);
    map<Vector3_Order<double>, complex<double>> cRPA_q;
    for (const auto &freq_q_MuNuchi0 : chi0.get_chi0_q())
    {
        const auto freq = freq_q_MuNuchi0.first;
        const double freq_weight = chi0.tfg.find_freq_weight(freq);
        for (const auto &q_MuNuchi0 : freq_q_MuNuchi0.second)
        {
            double task_begin = omp_get_wtime();
            const auto q = q_MuNuchi0.first;
            auto &MuNuchi0 = q_MuNuchi0.second;

            // ComplexMatrix loc_piT(loc_col,loc_row);
            auto loc_piT = init_local_mat<complex<double>>(arrdesc_pi, MAJOR::COL);
            complex<double> trace_pi(0.0, 0.0);
            double vq_time = 0.0;
            double pi_time = 0.0;
            double loc_tran_time = 0.0;
            for (int Mu = 0; Mu != natom; Mu++)
            {
                double Mu_begin = omp_get_wtime();
                // lib_printf(" |process %d,  Mu:  %d\n",comm_h.myid,Mu);
                const int n_mu = chi0.atbasis_abf[Mu];
                atom_mapping<ComplexMatrix>::pair_t_old Vq_row = gather_vq_row_q(chi0.atbasis_abf, comm_h, Mu, coulmat, q);
                double Mu_after_vq = omp_get_wtime();
                // atom_mapping<ComplexMatrix>::pair_t_old Vq_row;
                // const auto IJq_coul = Communicate_Tensors_Map_Judge::comm_map2_first(comm_h.comm, vq_libri, {Mu}, loc_atp_atoms);
                // double Mu_vq_comm = omp_get_wtime();
                // for(auto Ip:IJq_coul)
                // {
                //     auto I=Ip.first;
                //     auto n_mu=atom_mu[I];
                //     for(auto &Jqp:Ip.second)
                //     {
                //         auto J=Jqp.first.first;
                //         auto n_nu=atom_mu[J];
                //         auto q=Jqp.first.second;
                //         Vq_row[I][J].create(n_mu,n_nu);
                //         const auto length=sizeof(complex<double>)* n_mu *n_nu;
                //         memcpy(Vq_row[I][J].c, Jqp.second.ptr(),length);
                //     }
                // }
                // double Mu_after_vq=omp_get_wtime();
                //printf("   |process %d, Mu: %d  vq_row.size: %d\n",para_mpi.get_myid(),Mu,Vq_row[Mu].size());
                //ComplexMatrix loc_pi_row=compute_Pi_freq_q_row(q,MuNuchi0,Vq_loc,Mu,q);
                ComplexMatrix loc_pi_row = compute_Pi_freq_q_row(chi0.atbasis_abf, q, MuNuchi0, Vq_row, local_atpair, Mu);
                //printf("   |process %d,   compute_pi\n",para_mpi.get_myid());
                ComplexMatrix glo_pi_row(n_mu, chi0.atbasis_abf.nb_total);
                comm_h.barrier();
                librpa_int::allreduce_ComplexMatrix(loc_pi_row,glo_pi_row,comm_h.comm);
                double Mu_after_pi_loc=omp_get_wtime();
                //cout<<"  glo_pi_rowT nr,nc: "<<glo_pi_row.nr<<" "<<glo_pi_row.nc<<endl;

                for (int i_mu = 0; i_mu != n_mu; i_mu++)
                    trace_pi += glo_pi_row(i_mu, part_range[Mu] + i_mu);
                //select glo_pi_rowT to pi_blacs
                for (int i = 0; i != loc_row; i++)
                {
                    // int global_row =
                    // para_mpi.globalIndex(i,row_nblk,para_mpi.nprow,para_mpi.myprow);
                    int global_row = arrdesc_pi.indx_l2g_r(i);
                    int mu_blacs, I_blacs;
                    chi0.atbasis_abf.get_local_index(global_row, I_blacs, mu_blacs);
                    if (I_blacs == Mu)
                        for (int j = 0; j != loc_col; j++)
                        {
                            // int global_col =
                            // para_mpi.globalIndex(j,col_nblk,para_mpi.npcol,para_mpi.mypcol);
                            int global_col = arrdesc_pi.indx_l2g_c(j);
                            int nu_blacs, J_blacs;
                            chi0.atbasis_abf.get_local_index(global_col, J_blacs, nu_blacs);
                            //cout<<" Mu: "<<Mu<<"  i,j: "<<i<<"  "<<j<<"    glo_row,col: "<<global_row<<"  "<<global_col<<"  J:"<<J_blacs<< "  index i,j: "<<atom_mu_part_range[J_blacs] + mu_blacs<<" "<<nu_blacs<<endl;
                            if( global_col == global_row)
                            {
                                loc_piT(i,j) = complex<double>(1.0,0.0) - glo_pi_row(mu_blacs, chi0.atbasis_abf.get_part_range()[J_blacs]+nu_blacs);
                            }
                            else
                            {
                                loc_piT(i,j) = -glo_pi_row(mu_blacs, part_range[J_blacs]+nu_blacs);
                            }
                        }
                }
                double Mu_after_loc_tran = omp_get_wtime();
                vq_time += (Mu_after_vq - Mu_begin);
                pi_time += (Mu_after_pi_loc - Mu_after_vq);
                loc_tran_time += (Mu_after_loc_tran - Mu_after_pi_loc);
            }
            // if(freq == chi0.tfg.get_freq_nodes()[0] && comm_h.is_root())
            //     print_complex_matrix(" loc_piT",loc_piT);
            double task_mid = omp_get_wtime();
            //printf("|process  %d, before det\n",comm_h.myid);
            std::complex<double> ln_det=compute_pi_det_blacs_2d(loc_piT, arrdesc_pi, ipiv, info);
            double task_end = omp_get_wtime();
            if(comm_h.is_root())
                lib_printf("| After det for freq:  %f,  q: ( %f, %f, %f)   TIME_Vq_COMM: %f   TIME_DET: %f  TIME_CAL_Pi: %f, TIME_TRAN_LOC: %f\n",freq, q.x,q.y,q.z,vq_time,task_end-task_mid,pi_time,loc_tran_time);
            //para_mpi.mpi_barrier();
            if(comm_h.myid==0)
            {
                std::complex<double> rpa_for_omega_q = trace_pi + ln_det;
                const auto kweight = chi0.q_weight(q);
                //cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "      lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * kweight / TWO_PI;//!check
                tot_RPA_energy += rpa_for_omega_q * freq_weight * kweight / TWO_PI;
            }
        }
    }

    if(comm_h.myid==0)
    {
        for (auto &q_crpa : cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
            // cout << q_crpa.first << q_crpa.second << endl;
        }
        // cout << "gx_num_" << chi0.tfg.size() << "  tot_RPA_energy:  " << setprecision(8)
        // <<tot_RPA_energy << endl;
    }
    comm_h.barrier();
    corr.value = tot_RPA_energy;
    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_RPA_correlation(LibrpaParallelRouting routing, const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    using global::ofs_myid;
    using global::lib_printf;

    CorrEnergy corr;
    const auto &comm_h = chi0.comm_h;
    if (comm_h.myid == 0)
        lib_printf("Calculating EcRPA without BLACS/ScaLAPACK\n");
    // lib_printf("Begin cal cRPA , pid:  %d\n", comm_h.myid);
    const auto & mf = chi0.mf;

    // freq, q
    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi_freq_q_Mu_Nu;
    if (routing == LIBRPA_ROUTING_ATOMPAIR || routing == LIBRPA_ROUTING_LIBRI)
        pi_freq_q_Mu_Nu = compute_Pi_q_MPI(chi0, coulmat);
    else
        pi_freq_q_Mu_Nu = compute_Pi_q(chi0, coulmat);
    lib_printf("Finish Pi freq on Proc %4d, size %zu\n", comm_h.myid, pi_freq_q_Mu_Nu.size());
    //comm_h.barrier();

    int range_all = chi0.atbasis_abf.nb_total;

#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before part_range processid:%d, atom_mu.size(): %zu\n",
//    mpi_comm_global_h.myid, atom_mu.size());
#endif
    const auto part_range = chi0.atbasis_abf.get_part_range();
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success after part_range processid:%d, atom_mu.size(): %zu\n",
//        mpi_comm_global_h.myid, atom_mu.size());
#endif

    // cout << "part_range:" << endl;
    // for (int I = 0; I != atom_mu.size(); I++)
    // {
    //     cout << part_range[I] << endl;
    // }
    // cout << "part_range over" << endl;

    // pi_freq_q contains all atoms
    map<double, map<Vector3_Order<double>, ComplexMatrix>> pi_freq_q;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("| process %d, qpts.size(): %zu,freq.size():%zu\n", mpi_comm_global_h.myid,
// chi0.klist.size(),chi0.tfg.get_freq_nodes().size());
#endif
    for (const auto &freq : chi0.tfg.get_freq_nodes())
    {
        // printf("| process %d, freq: %f\n", mpi_comm_global_h.myid, freq);
        map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old> freq_q_MuNupi;
        if (!chi0.get_chi0_q().empty()) freq_q_MuNupi = pi_freq_q_Mu_Nu.at(freq);
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
// printf("success before freq_q_MuNupi processid:%d, freq_q_MuNupi.size(): %zu\n",
//        mpi_comm_global_h.myid, freq_q_MuNupi.size());
#endif
        for (const auto &q : chi0.active_qpoints())
        {
            atom_mapping<ComplexMatrix>::pair_t_old q_MuNupi;
            if (!chi0.get_chi0_q().empty()) q_MuNupi = freq_q_MuNupi.at(q);
            const auto MuNupi = q_MuNupi;
            pi_freq_q[freq][q].create(range_all, range_all);

            ComplexMatrix pi_munu_tmp(range_all, range_all);
            pi_munu_tmp.zero_out();
            if (!chi0.get_chi0_q().empty())
            {
                for (const auto &Mu_Nupi : MuNupi)
                {
                    const auto Mu = Mu_Nupi.first;
                    const auto Nupi = Mu_Nupi.second;
                    const size_t n_mu = chi0.atbasis_abf[Mu];
                    for (const auto &Nu_pi : Nupi)
                    {
                        const auto Nu = Nu_pi.first;
                        const auto pimat = Nu_pi.second;
                        const size_t n_nu = chi0.atbasis_abf[Nu];

                        for (size_t mu = 0; mu != n_mu; ++mu)
                        {
                            for (size_t nu = 0; nu != n_nu; ++nu)
                            {
                                pi_munu_tmp(part_range[Mu] + mu, part_range[Nu] + nu) +=
                                    pimat(mu, nu);
                            }
                        }
                    }
                }
            }
            if (routing == LIBRPA_ROUTING_ATOMPAIR || routing == LIBRPA_ROUTING_LIBRI)
            {
                reduce_ComplexMatrix(pi_munu_tmp, pi_freq_q.at(freq).at(q), 0, comm_h.comm);
            }
            else
            {
                pi_freq_q.at(freq).at(q) = std::move(pi_munu_tmp);
            }
        }
    }
    // lib_printf("Finish Pi communicate %4d, size %zu\n", comm_h.myid, pi_freq_q_Mu_Nu.size());
    comm_h.barrier();
    // if (comm_h.myid == 0)
    {
        complex<double> tot_RPA_energy(0.0, 0.0);
        map<Vector3_Order<double>, complex<double>> cRPA_q;
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
        int num_iteration = 0;
#endif
        for (const auto &freq_qpi : pi_freq_q)
        {
            const auto freq = freq_qpi.first;
            const double freq_weight = chi0.tfg.find_freq_weight(freq);
            for (const auto &q_pi : freq_qpi.second)
            {
                const auto q = q_pi.first;
                const auto pimat = q_pi.second;
                std::complex<double> rpa_for_omega_q(0.0, 0.0);
                ComplexMatrix identity(range_all, range_all);
                ComplexMatrix identity_minus_pi(range_all, range_all);
                identity.set_as_identity_matrix();
                identity_minus_pi = identity - pi_freq_q[freq][q];
#ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
                // if(num_iteration==0)
                // if(mpi_comm_global_h.myid == 1)
                // {
                //     complex<double>* test_c= identity_minus_pi.c;
                //     for(int i=0;i<range_all;i++){
                //         for(int j=0;j<range_all;j++){
                //             printf("%f+%fi ",
                //                    test_c[i*range_all+j].real(), test_c[i*range_all+j].imag());
                //         }
                //         printf("\n");
                //     }
                // }
                num_iteration++;
#endif
                complex<double> det_for_rpa(1.0, 0.0);
                int info_LU = 0;
                int *ipiv = new int[range_all];
                LapackConnector::zgetrf(range_all, range_all, identity_minus_pi, range_all, ipiv,
                                        &info_LU);
                for (int ib = 0; ib != range_all; ib++)
                {
                    if (ipiv[ib] != (ib + 1))
                        det_for_rpa = -det_for_rpa * identity_minus_pi(ib, ib);
                    else
                        det_for_rpa = det_for_rpa * identity_minus_pi(ib, ib);
                }
                delete[] ipiv;

                complex<double> trace_pi;
                complex<double> ln_det;
                ln_det = std::log(det_for_rpa);
                trace_pi = trace(pi_freq_q.at(freq).at(q));
                // cout << "PI trace vector:" << endl;
                // cout << endl;
                rpa_for_omega_q = ln_det + trace_pi;
                const auto kweight = chi0.q_weight(q);
                // cout << " ifreq:" << freq << "      rpa_for_omega_k: " << rpa_for_omega_q << "      lnt_det: " << ln_det << "    trace_pi " << trace_pi << endl;
                cRPA_q[q] += rpa_for_omega_q * freq_weight * kweight / TWO_PI;
                tot_RPA_energy += rpa_for_omega_q * freq_weight * kweight / TWO_PI;
            }
        }
        // lib_printf("Finish EcRPA %4d, size %zu\n", comm_h.myid, pi_freq_q_Mu_Nu.size());
        comm_h.barrier();
        map<Vector3_Order<double>, complex<double>> global_cRPA_q;
        for (const auto &q : chi0.active_qpoints())
        {
            MPI_Reduce(&cRPA_q[q], &global_cRPA_q[q], 1,
                       MPI_DOUBLE_COMPLEX, MPI_SUM, 0, comm_h.comm);
        }

        for (auto &q_crpa : global_cRPA_q)
        {
            corr.qcontrib[q_crpa.first] = q_crpa.second;
        }
        complex<double> gather_tot_RPA_energy(0.0, 0.0);
        MPI_Reduce(&tot_RPA_energy,&gather_tot_RPA_energy,1,MPI_DOUBLE_COMPLEX,MPI_SUM,0,comm_h.comm);
        corr.value = gather_tot_RPA_energy;
    }
    corr.etype = CorrEnergy::type::RPA;
    return corr;
}

CorrEnergy compute_MP2_correlation(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    CorrEnergy corr;
    corr.etype = CorrEnergy::type::MP2;
    return corr;
}

map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q(
    const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    using librpa_int::global::ofs_myid;
    using librpa_int::global::lib_printf;

    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi;
    lib_printf("Begin compute_Pi_q , pid:  %d\n", chi0.comm_h.myid);
    for (auto const & freq_qJQchi0: chi0.get_chi0_q())
    {
        const double freq = freq_qJQchi0.first;
        for (auto &q_JQchi0 : freq_qJQchi0.second)
        {
            Vector3_Order<double> q = q_JQchi0.first;
            for (auto &JQchi0 : q_JQchi0.second)
            {
                const size_t J = JQchi0.first;
                const size_t J_mu = chi0.atbasis_abf[J];
                for (auto &Qchi0 : JQchi0.second)
                {
                    const size_t Q = Qchi0.first;
                    const size_t Q_mu = chi0.atbasis_abf[Q];
                    // auto &chi0_mat = Qchi0.second;
                    for (atom_t I = 0; I != chi0.atbasis_abf.n_atoms; I++)
                    {
                        //const size_t I = I_p.first;
                        const size_t I_mu = chi0.atbasis_abf[I];
                        pi[freq][q][I][Q].create(I_mu, Q_mu);
                        if (J != Q) pi[freq][q][I][J].create(I_mu, J_mu);
                    }
                }
            }
            // if(freq==chi0.tfg.get_freq_nodes()[0])
            //     for(auto &Ip:pi[freq][q])
            //         for(auto &Jp:Ip.second)
            //             lib_printf("  |process  %d, pi atpair: %d, %d \n",comm_h.myid,Ip.first,Jp.first);
        }
    }

    // ofstream fp;
    // std::stringstream ss;
    // ss<<"out_pi_rank_"<<comm_h.myid<<".txt";
    // fp.open(ss.str());
    for (auto &freq_p : chi0.get_chi0_q())
    {
        const double freq = freq_p.first;
        const auto chi0_freq = freq_p.second;
        for (auto &k_pair : chi0_freq)
        {
            Vector3_Order<double> ik_vec = k_pair.first;
            auto chi0_freq_k = k_pair.second;
            for (auto &J_p : chi0_freq_k)
            {
                const size_t J = J_p.first;
                for (auto &Q_p : J_p.second)
                {
                    const size_t Q = Q_p.first;
                    auto &chi0_mat = Q_p.second;
                    for (atom_t I = 0; I != chi0.atbasis_abf.n_atoms; I++)
                    {
                        //const size_t I = I_p.first;
                        //printf("cal_pi  pid: %d , IJQ:  %d  %d  %d\n", comm_h.myid, I, J, Q);
                        //  cout<<"         pi_IQ: "<<pi_k.at(freq).at(ik_vec).at(I).at(Q)(0,0)<<"   pi_IJ: "<<pi_k.at(freq).at(ik_vec).at(I).at(J)(0,0);
                        if (I <= J)
                        {
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", comm_h.myid, I, J, Q,1);
                            //      << "  Vq: " << (*Vq.at(I).at(J).at(ik_vec))(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(Q) += (*coulmat.at(I).at(J).at(ik_vec)) * chi0_mat;
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            // {
                            //     std:stringstream sm;
                            //     complex<double> trace_pi;
                            //     trace_pi = trace(pi.at(freq).at(ik_vec).at(I).at(Q));
                            //     sm << " IJQ: " << I << " " << J << " " << Q << "  ik_vec: " <<
                            //     ik_vec << "  trace_pi:  " << trace_pi << endl;
                            //     print_complex_matrix_file(sm.str().c_str(),
                            //     (*Vq.at(I).at(J).at(ik_vec)),fp,false);
                            //     print_complex_matrix_file("chi0:", chi0_mat,fp,false);
                            //     print_complex_matrix_file("pi_mat:",
                            //     pi.at(freq).at(ik_vec).at(I).at(Q),fp,false);
                            // }
                        }
                        else
                        {
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", comm_h.myid, I, J, Q,2);
                            //      << "  Vq: " << transpose(*Vq.at(J).at(I).at(ik_vec), 1)(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(Q) +=
                                transpose(*coulmat.at(J).at(I).at(ik_vec), 1) * chi0_mat;
                        }

                        if (J != Q)
                        {
                            ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
                            if (I <= Q)
                            {
                                // if (freq == chi0.tfg.get_freq_nodes()[0])
                                //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", comm_h.myid, I, J, Q,3);
                                //      << "  Vq: " << (*Vq.at(I).at(Q).at(ik_vec))(0, 0) << endl;
                                pi.at(freq).at(ik_vec).at(I).at(J) +=
                                    (*coulmat.at(I).at(Q).at(ik_vec)) * chi0_QJ;
                            }
                            else
                            {
                                // if (freq == chi0.tfg.get_freq_nodes()[0])
                                //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", comm_h.myid, I, J, Q,4);
                                //      << "  Vq: " << transpose(*Vq.at(J).at(I).at(ik_vec), 1)(0, 0) << endl;
                                pi.at(freq).at(ik_vec).at(I).at(J) +=
                                    transpose(*coulmat.at(Q).at(I).at(ik_vec), 1) * chi0_QJ;
                            }
                        }
                    }
                }
            }
        }
    }
    // fp.close();
    // print_complex_matrix("
    // first_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(0).at(0));
    /* print_complex_matrix("
     * last_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(natom-1).at(natom-1)); */
    return pi;
}

map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q_MPI(
    const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat)
{
    using librpa_int::global::ofs_myid;
    using librpa_int::global::lib_printf;

    map<double, map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> pi;
    lib_printf("Begin compute_Pi_q_MPI , pid:  %d\n", chi0.comm_h.myid);
    const auto &abf = chi0.atbasis_abf;
    for (auto const & freq_qJQchi0: chi0.get_chi0_q())
    {
        const double freq = freq_qJQchi0.first;
        for (auto &q_JQchi0 : freq_qJQchi0.second)
        {
            Vector3_Order<double> q = q_JQchi0.first;
            for (auto &JQchi0 : q_JQchi0.second)
            {
                const size_t J = JQchi0.first;
                const size_t J_mu = abf[J];
                for (auto &Qchi0 : JQchi0.second)
                {
                    const size_t Q = Qchi0.first;
                    const size_t Q_mu = abf[Q];
                    // auto &chi0_mat = Qchi0.second;
                    for (int I = 0; I != as_int(chi0.atbasis_abf.n_atoms); I++)
                    {
                        //const size_t I = I_p.first;
                        const size_t I_mu = abf[I];
                        pi[freq][q][I][Q].create(I_mu, Q_mu);
                        if (J != Q) pi[freq][q][I][J].create(I_mu, J_mu);
                    }
                }
            }
            // if(freq==chi0.tfg.get_freq_nodes()[0])
            //     for(auto &Ip:pi[freq][q])
            //         for(auto &Jp:Ip.second)
            //             lib_printf("  |process  %d, pi atpair: %d, %d \n",comm_h.myid,Ip.first,Jp.first);
        }
    }

    // ofstream fp;
    // std::stringstream ss;
    // ss<<"out_pi_rank_"<<comm_h.myid<<".txt";
    // fp.open(ss.str());
    const auto &comm_h = chi0.comm_h;
    #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
    // printf("success before irk_weight, pid: %d\n", mpi_comm_global_h.myid);
    #endif
    for (const auto &ik_vec : chi0.active_qpoints())
    {
        for (int I = 0; I != as_int(chi0.atbasis_abf.n_atoms); I++)
        {
            #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
            // printf("success before gather_vp_row_q irk_weight, pid: %d\n", mpi_comm_global_h.myid);
            #endif
            atom_mapping<ComplexMatrix>::pair_t_old Vq_row = gather_vq_row_q(chi0.atbasis_abf, comm_h, I, coulmat, ik_vec);
            #ifdef OPEN_TEST_FOR_LU_DECOMPOSITION
            // printf("success after gather_vp_row_q irk_weight, pid: %d\n", mpi_comm_global_h.myid);
            #endif

            for (auto &freq_p : chi0.get_chi0_q())
            {
                const double freq = freq_p.first;
                const auto chi0_freq = freq_p.second;

                auto chi0_freq_k = freq_p.second.at(ik_vec);

                for (auto &J_p : chi0_freq_k)
                {
                    const size_t J = J_p.first;
                    for (auto &Q_p : J_p.second)
                    {
                        const size_t Q = Q_p.first;
                        auto &chi0_mat = Q_p.second;

                        //const size_t I = I_p.first;
                        //printf("cal_pi  pid: %d , IJQ:  %d  %d  %d\n", comm_h.myid, I, J, Q);
                        //  cout<<"         pi_IQ: "<<pi_k.at(freq).at(ik_vec).at(I).at(Q)(0,0)<<"   pi_IJ: "<<pi_k.at(freq).at(ik_vec).at(I).at(J)(0,0);

                        // if (freq == chi0.tfg.get_freq_nodes()[0])
                        //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", comm_h.myid, I, J, Q,1);
                        //      << "  Vq: " << (*Vq.at(I).at(J).at(ik_vec))(0, 0) << endl;
                        pi.at(freq).at(ik_vec).at(I).at(Q) += Vq_row.at(I).at(J) * chi0_mat;
                        // if (freq == chi0.tfg.get_freq_nodes()[0])
                        // {
                        //     std:stringstream sm;
                        //     complex<double> trace_pi;
                        //     trace_pi = trace(pi.at(freq).at(ik_vec).at(I).at(Q));
                        //     sm << " IJQ: " << I << " " << J << " " << Q << "  ik_vec: " << ik_vec
                        //     << "  trace_pi:  " << trace_pi << endl;
                        //     print_complex_matrix_file(sm.str().c_str(),
                        //     Vq_row.at(I).at(J),fp,false); print_complex_matrix_file("chi0:",
                        //     chi0_mat,fp,false); print_complex_matrix_file("pi_mat:",
                        //     pi.at(freq).at(ik_vec).at(I).at(Q),fp,false);
                        // }

                        if (J != Q)
                        {
                            ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
                            // if (freq == chi0.tfg.get_freq_nodes()[0])
                            //     lib_printf("cal_pi  pid: %d , IJQ:  %d  %d  %d   type: %d \n", comm_h.myid, I, J,Q,3);
                            //      << "  Vq: " << (*Vq.at(I).at(Q).at(ik_vec))(0, 0) << endl;
                            pi.at(freq).at(ik_vec).at(I).at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
                        }
                    }
                }
            }
        }
    }
    //fp.close();
    // print_complex_matrix(" first_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(0).at(0));
    /* print_complex_matrix("  last_pi_mat:",pi.at(chi0.tfg.get_freq_nodes()[0]).at({0,0,0}).at(natom-1).at(natom-1)); */
    lib_printf("End compute_Pi_q_MPI , pid:  %d\n", chi0.comm_h.myid);
    return pi;
}

ComplexMatrix compute_Pi_freq_q_row(const AtomicBasis &atbasis_abf, const Vector3_Order<double> &ik_vec,
                                    const atom_mapping<ComplexMatrix>::pair_t_old &chi0_freq_q,
                                    const atom_mapping<ComplexMatrix>::pair_t_old &Vq_row,
                                    const vector<atpair_t> &local_atpair, const int &I)
{
    map<size_t, ComplexMatrix> pi;
    // lib_printf("Begin cal_pi_k , pid:  %d\n", para_mpi.get_myid());
    auto I_mu = atbasis_abf[I];
    const int natom = as_int(atbasis_abf.n_atoms);
    for (int J = 0; J != natom; J++) pi[J].create(I_mu, atbasis_abf[J]);
    const auto n_ap = local_atpair.size();

    omp_lock_t pi_lock;
    omp_init_lock(&pi_lock);
#pragma omp parallel for schedule(dynamic)
    for (size_t iap = 0; iap < n_ap; iap++)
    {
        const size_t J = local_atpair[iap].first;
        const size_t Q = local_atpair[iap].second;
        auto &chi0_mat = chi0_freq_q.at(J).at(Q);
        auto tmp_pi_mat = Vq_row.at(I).at(J) * chi0_mat;
        ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
        auto tmp_pi_mat2 = Vq_row.at(I).at(Q) * chi0_QJ;
        omp_set_lock(&pi_lock);
        pi.at(Q) += tmp_pi_mat;
        if (J != Q)
        {
            pi.at(J) += tmp_pi_mat2;
        }
        omp_unset_lock(&pi_lock);
    }
    omp_destroy_lock(&pi_lock);
    // for (auto &J_p : chi0_freq_q)
    // {
    //     const size_t J = J_p.first;
    //     for (auto &Q_p : J_p.second)
    //     {
    //         const size_t Q = Q_p.first;
    //         auto &chi0_mat = Q_p.second;
    //         pi.at(Q) += Vq_row.at(I).at(J) * chi0_mat;
    //         if (J != Q)
    //         {
    //             ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
    //             pi.at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
    //         }
    //     }
    // }
    //Pi_rowT
    // ComplexMatrix pi_row(N_all_mu,atom_mu[I]);
    // complex<double> *pi_row_ptr=pi_row.c;
    // for(auto &Jp:pi)
    // {
    //     auto J=Jp.first;
    //     auto J_mu=atom_mu[J];
    //     const auto length=sizeof(complex<double>)* I_mu *J_mu;
    //     memcpy(pi_row_ptr, pi.at(J).c,length);
    //     pi_row_ptr+=I_mu *J_mu;
    // }
    ComplexMatrix pi_row(atbasis_abf[I], atbasis_abf.nb_total);
    const auto atom_mu_part_range = atbasis_abf.get_part_range();
    for (int i = 0; i != pi_row.nr; i++)
        for (int J = 0; J != natom; J++)
            for (int j = 0; j != as_int(atbasis_abf[J]); j++)
                pi_row(i, atom_mu_part_range[J] + j) = pi.at(J)(i, j);
    return pi_row;
}

ComplexMatrix compute_Pi_freq_q_row_ri(const AtomicBasis &atbasis_abf, const Vector3_Order<double> &ik_vec, const atom_mapping<ComplexMatrix>::pair_t_old &chi0_freq_q, const atpair_k_cplx_mat_t &Vq_loc, const vector<atpair_t> &local_atpair, const int &I, const Vector3_Order<double> &q)
{
    map<size_t, ComplexMatrix> pi;
    // lib_printf("Begin cal_pi_k , pid:  %d\n", comm_h.myid);
    const int natom = atbasis_abf.n_atoms;
    auto I_mu = atbasis_abf[I];
    for (int J = 0; J != natom; J++) pi[J].create(I_mu, atbasis_abf[J]);
    const auto n_ap = local_atpair.size();

    omp_lock_t pi_lock;
    omp_init_lock(&pi_lock);
#pragma omp parallel for schedule(dynamic)
    for (size_t iap = 0; iap != n_ap; iap++)
    {
        const size_t J = local_atpair[iap].first;
        const size_t Q = local_atpair[iap].second;
        auto &chi0_mat= chi0_freq_q.at(J).at(Q);
        //printf("| IN cal Pi process %d, I: %d  J: %d  Q: %d\n",comm_h.myid, I,J,Q );
        auto tmp_pi_mat= *Vq_loc.at(I).at(J).at(q) * chi0_mat;
        ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
        auto tmp_pi_mat2= *Vq_loc.at(I).at(Q).at(q) * chi0_QJ;
        omp_set_lock(&pi_lock);
        pi.at(Q)+=tmp_pi_mat;
        if(J!=Q)
            {
                pi.at(J)+=tmp_pi_mat2;
            }
        omp_unset_lock(&pi_lock);
    }
    omp_destroy_lock(&pi_lock);
    // for (auto &J_p : chi0_freq_q)
    // {
    //     const size_t J = J_p.first;
    //     for (auto &Q_p : J_p.second)
    //     {
    //         const size_t Q = Q_p.first;
    //         auto &chi0_mat = Q_p.second;
    //         pi.at(Q) += Vq_row.at(I).at(J) * chi0_mat;
    //         if (J != Q)
    //         {
    //             ComplexMatrix chi0_QJ = transpose(chi0_mat, 1);
    //             pi.at(J) += Vq_row.at(I).at(Q) * chi0_QJ;
    //         }
    //     }
    // }
    //Pi_rowT
    // ComplexMatrix pi_row(N_all_mu,atom_mu[I]);
    // complex<double> *pi_row_ptr=pi_row.c;
    // for(auto &Jp:pi)
    // {
    //     auto J=Jp.first;
    //     auto J_mu=atom_mu[J];
    //     const auto length=sizeof(complex<double>)* I_mu *J_mu;
    //     memcpy(pi_row_ptr, pi.at(J).c,length);
    //     pi_row_ptr+=I_mu *J_mu;
    // }
    ComplexMatrix pi_row(atbasis_abf[I], atbasis_abf.nb_total);
    const auto atom_mu_part_range = atbasis_abf.get_part_range();
    for (int i = 0; i != pi_row.nr; i++)
        for (int J = 0; J != natom; J++)
            for (int j = 0; j != as_int(atbasis_abf[J]); j++)
                pi_row(i, atom_mu_part_range[J] + j) = pi.at(J)(i, j);
    return pi_row;
}

atom_mapping<ComplexMatrix>::pair_t_old gather_vq_row_q(const AtomicBasis &atbasis_abf, const MpiCommHandler &comm_h, const int &I, const atpair_k_cplx_mat_t &coulmat, const Vector3_Order<double> &ik_vec)
{
    auto I_mu = atbasis_abf[I];
    const int natom = atbasis_abf.n_atoms;
    atom_mapping<ComplexMatrix>::pair_t_old Vq_row;
    for (int J_tmp = 0; J_tmp != natom; J_tmp++)
    {
        auto J_mu = atbasis_abf[J_tmp];
        ComplexMatrix loc_vq(atbasis_abf[I], atbasis_abf[J_tmp]);
        Vq_row[I][J_tmp].create(atbasis_abf[I], atbasis_abf[J_tmp]);
        // const auto length=sizeof(complex<double>)* I_mu *J_mu;
        // complex<double> *loc_vq_ptr=loc_vq.c;
        if (I <= J_tmp)
        {
            if (coulmat.count(I))
                if (coulmat.at(I).count(J_tmp)) loc_vq = *coulmat.at(I).at(J_tmp).at(ik_vec);
        }
        else
        {
            if (coulmat.count(J_tmp))
                if (coulmat.at(J_tmp).count(I)) loc_vq = transpose(*coulmat.at(J_tmp).at(I).at(ik_vec), 1);
        }
        librpa_int::allreduce_ComplexMatrix(loc_vq,Vq_row[I][J_tmp], comm_h.comm);
    }
    return Vq_row;
}

std::map<double, std::map<Vector3_Order<double>, Matz>>
compute_Wc_freq_q(
    Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps, atpair_k_cplx_mat_t &coulmat_wc, double sqrt_coulomb_threshold,
    const vector<std::complex<double>> &epsmac_LF_imagfreq, bool debug, const char *output_dir)
{
    using std::cout;
    using std::endl;
    using global::lib_printf;

    // Object to return
    map<double, std::map<Vector3_Order<double>, Matz>> Wc_freq_q;
    const int range_all = chi0.atbasis_abf.nb_total;
    const auto &part_range = chi0.atbasis_abf.get_part_range();
    const auto &qpts_active = chi0.active_qpoints();

    const auto &comm_h = chi0.comm_h;
    const auto &abf = chi0.atbasis_abf;
    const auto validate_coulomb_block_shape = [&](const char *stage, const int mu,
                                                  const int nu,
                                                  const Vector3_Order<double> &q,
                                                  const auto &vq,
                                                  const int n_mu,
                                                  const int n_nu) {
        if (vq->nr == n_mu && vq->nc == n_nu) return;
        std::ostringstream errmsg;
        errmsg << "Coulomb block dimension mismatch while preparing " << stage
               << " for Wc at q=(" << q.x << ", " << q.y << ", " << q.z
               << "), atom pair (" << mu << ", " << nu << "): block shape is "
               << vq->nr << "x" << vq->nc << " but chi0 auxiliary basis expects "
               << n_mu << "x" << n_nu
               << ". Check use_shrink_abfs/use_shrink_chi and Coulomb prefixes; "
                  "legacy shrink Coulomb files cannot be used with full-chi Wc.";
        throw LIBRPA_RUNTIME_ERROR(errmsg.str());
    };

    if (comm_h.myid == 0)
    {
        cout << "Calculating Wc using LAPACK" << endl;
    }

    comm_h.barrier();
    // use q-points as the outmost loop, so that square root of Coulomb will not be recalculated at each frequency point
    std::vector<Vector3_Order<double>> qpts(qpts_active.begin(), qpts_active.end());

    for (const auto &q : qpts)
    {
        int iq = std::distance(qpts.begin(), std::find(qpts.begin(), qpts.end(), q));
        char fn[80];

        Matz Vq_all(range_all, range_all, MAJOR::COL);
        for (const auto &Mu_NuqVq: coulmat_eps)
        {
            auto Mu = Mu_NuqVq.first;
            int n_mu = abf[Mu];
            for ( auto &Nu_qVq: Mu_NuqVq.second )
            {
                auto Nu = Nu_qVq.first;
                if ( 0 == Nu_qVq.second.count(q) ) continue;
                int n_nu = abf[Nu];
                const auto &vq = Nu_qVq.second.at(q);
                validate_coulomb_block_shape("bare Coulomb", Mu, Nu, q, vq, n_mu, n_nu);
                for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                    for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                    {
                        Vq_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) =
                            (*vq)(i_mu, i_nu);
                        Vq_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) =
                            conj((*vq)(i_mu, i_nu));
                    }
            }
        }
        if (debug)
        {
            sprintf(fn, "Vq_all_q_%d.mtx", iq);
            print_matrix_mm_file(Vq_all,  + fn, "", 1e-15);
        }
        const auto sqrtVq_all = power_hemat(Vq_all, 0.5, true, false, sqrt_coulomb_threshold);
        // Vq_all is now eigenvectors of the original Coulomb matrix
        // only required for Gamma point
        const auto& Vq_eigen = Vq_all;
        if (debug)
        {
            sprintf(fn, "sqrtVq_all_q_%d.mtx", iq);
            print_matrix_mm_file(sqrtVq_all, path_as_directory(output_dir) + fn, "", 1e-15);
            // sprintf(fn, "rotated_sqrtVq_all_q_%d.mtx", iq);
            // print_complex_matrix_mm(Vq_all * sqrtVq_all * transpose(Vq_all, true), fn, 1e-15);
            // print_complex_matrix_mm(transpose(Vq_all, true) * sqrtVq_all * Vq_all, fn, 1e-15);
            sprintf(fn, "Vqeigenvec_q_%d.mtx", iq);
            print_matrix_mm_file(Vq_eigen, path_as_directory(output_dir) + fn, "", 1e-15);
        }

        // truncated (cutoff) Coulomb
        Matz Vqcut_all(range_all, range_all);
        for (auto &Mu_NuqVq : coulmat_wc)
        {
            auto Mu = Mu_NuqVq.first;
            int n_mu = abf[Mu];
            for (auto &Nu_qVq : Mu_NuqVq.second)
            {
                auto Nu = Nu_qVq.first;
                if (0 == Nu_qVq.second.count(q)) continue;
                int n_nu = abf[Nu];
                const auto &vq = Nu_qVq.second.at(q);
                validate_coulomb_block_shape("truncated Coulomb", Mu, Nu, q, vq, n_mu, n_nu);
                for (int i_mu = 0; i_mu != n_mu; i_mu++)
                    for (int i_nu = 0; i_nu != n_nu; i_nu++)
                    {
                        Vqcut_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) =
                            (*vq)(i_mu, i_nu);
                        Vqcut_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) =
                            conj((*vq)(i_mu, i_nu));
                    }
            }
        }
        auto sqrtVqcut_all = power_hemat(Vqcut_all, 0.5, false, true, sqrt_coulomb_threshold);
        // sprintf(fn, "sqrtVqcut_all_q_%d.mtx", iq);
        // print_complex_matrix_mm(sqrtVqcut_all, fn, 1e-15);
        // sprintf(fn, "Vqcut_all_filtered_q_%d.mtx", iq);
        // print_complex_matrix_mm(Vqcut_all, fn, 1e-15);
        // save the filtered truncated Coulomb back to the atom mapping object
        // TODO: revise the necessity
        for (auto &Mu_NuqVq : coulmat_wc)
        {
            auto Mu = Mu_NuqVq.first;
            int n_mu = abf[Mu];
            for ( auto &Nu_qVq: Mu_NuqVq.second )
            {
                auto Nu = Nu_qVq.first;
                if ( 0 == Nu_qVq.second.count(q) ) continue;
                int n_nu = abf[Nu];
                for ( int i_mu = 0; i_mu != n_mu; i_mu++ )
                    for ( int i_nu = 0; i_nu != n_nu; i_nu++ )
                        (*Nu_qVq.second.at(q))(i_mu, i_nu) = Vqcut_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu);
            }
        }

        Matz chi0fq_all(range_all, range_all, MAJOR::COL);
        for (const auto &freq_qMuNuchi: chi0.get_chi0_q())
        {
            auto freq = freq_qMuNuchi.first;
            auto ifreq = chi0.tfg.get_freq_index(freq);
            auto MuNuchi = freq_qMuNuchi.second.at(q);
            for (const auto &Mu_Nuchi : MuNuchi)
            {
                auto Mu = Mu_Nuchi.first;
                int n_mu = abf[Mu];
                for (auto &Nu_chi : Mu_Nuchi.second)
                {
                    auto Nu = Nu_chi.first;
                    int n_nu = abf[Nu];
                    for (int i_mu = 0; i_mu != n_mu; i_mu++)
                        for (int i_nu = 0; i_nu != n_nu; i_nu++)
                        {
                            chi0fq_all(part_range[Mu] + i_mu, part_range[Nu] + i_nu) =
                                Nu_chi.second(i_mu, i_nu);
                            chi0fq_all(part_range[Nu] + i_nu, part_range[Mu] + i_mu) =
                                conj(Nu_chi.second(i_mu, i_nu));
                        }
                }
            }
            if (debug)
            {
                sprintf(fn, "chi0fq_all_q_%d_freq_%d.mtx", iq, ifreq);
                print_matrix_mm_file(chi0fq_all, path_as_directory(output_dir) + fn, "", 1e-15);
            }

            auto eps_fq = - sqrtVq_all * chi0fq_all * sqrtVq_all;
            if (!epsmac_LF_imagfreq.empty() && is_gamma_point(q))
            {
                // rotate to Coulomb-diagonal basis
                // lib_printf("Largest off-diagonal = %f\n", eps_fq.get_max_abs_offdiag());
                // print_matrix("rotated eps_fq: ", eps_fq.real());
                // replacing the element corresponding to largest Coulomb eigenvalue with dielectric function
                eps_fq = transpose(Vq_eigen, true) * eps_fq * Vq_eigen;
                lib_printf("%22.12f %22.12f %22.12f %22.12f\n", freq, eps_fq(0, 0).real(), eps_fq(eps_fq.nr() - 1, eps_fq.nc() - 1).real(), epsmac_LF_imagfreq[ifreq].real());
                // eps_fq(eps_fq.nr - 1, eps_fq.nc - 1) = epsmac_LF_imagfreq[ifreq];
                eps_fq(0, 0) = 1.0 - epsmac_LF_imagfreq[ifreq];
                if (debug)
                {
                    sprintf(fn, "rotated_vsxvs_q_%d_freq_%d.mtx", iq, ifreq);
                    print_matrix_mm_file(eps_fq, path_as_directory(output_dir) + fn, "", 1e-10);
                }
                // rotate back to ABF
                eps_fq = Vq_eigen * eps_fq * transpose(Vq_eigen, true);
            }
            for (int i = 0; i < eps_fq.nr(); i++) eps_fq(i, i) += C_ONE;
            // eps_fq = identity - eps_fq;
            if (debug)
            {
                sprintf(fn, "eps_q_%d_freq_%d.mtx", iq, ifreq);
                print_matrix_mm_file(eps_fq, path_as_directory(output_dir) + fn, "", 1e-10);
            }

            // invert the epsilon matrix
            power_hemat_onsite(eps_fq, -1.0);
            for (int i = 0; i < eps_fq.nr(); i++) eps_fq(i, i) -= C_ONE;
            Wc_freq_q[freq][q] = sqrtVqcut_all * eps_fq * sqrtVqcut_all;
            // sprintf(fn, "inveps_q_%d_freq_%d.mtx", iq, ifreq);
            // print_complex_matrix_mm(eps_fq, fn, 1e-15);
            // sprintf(fn, "wc_q_%d_freq_%d.mtx", iq, ifreq);
            // print_complex_matrix_mm(wc_all, fn, 1e-15);
        }
    }

    return Wc_freq_q;
}

std::map<double, std::map<Vector3_Order<double>, Matz>> compute_Wc_freq_q_blacs(
    Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps, atpair_k_cplx_mat_t &coulmat_wc,
    const double sqrt_coulomb_threshold, const bool replace_w_head, int option_dielect_func,
    const vector<std::complex<double>> &epsmac_LF_imagfreq, diele_func *df_headwing,
    const BlacsCtxtHandler &blacs_h, const ArrayDesc &ad, const bool debug, const char *output_dir,
    bool use_cholesky_gw_wc, bool use_gpu_replace_scalapack, bool use_elpa_sqrt_coulomb,
    const bool output_2d_finite_q_diagnostics)
{
    using std::cout;
    using std::endl;
    using std::set;
    using std::pair;

    using global::ofs_myid;
    using global::lib_printf;
    using global::profiler;

    // Object to return
    map<double, std::map<Vector3_Order<double>, Matz>> Wc_freq_q;
    const int natom = chi0.atbasis_abf.n_atoms;
    const int n_abf = chi0.atbasis_abf.nb_total;
    const auto part_range = chi0.atbasis_abf.get_part_range();

    const auto &comm_h = blacs_h.comm_h();

    if (comm_h.myid == 0)
    {
        cout << "Calculating Wc using ScaLAPACK" << endl;
    }
    comm_h.barrier();

    global::profiler.start("compute_Wc_freq_q_blacs_init");
    const auto &desc_nabf_nabf = ad;
    assert(desc_nabf_nabf.initialized() && desc_nabf_nabf.m() == n_abf && desc_nabf_nabf.n() == n_abf);
    // Use a square blocksize instead max block, otherwise heev and inversion will complain about illegal parameter
    // Maximal blocksize ensure that atom indices related to the rows/columns of a local matrix is minimized.
    // This, however, is not optimal for matrix operations, and may lead to segment fault during
    // MPI operations with parallel linear algebra subroutine. Thus we define an optimal blocksize
    ArrayDesc desc_nabf_nabf_opt(blacs_h);
    const int nb_opt = std::min(128, desc_nabf_nabf.nb());
    desc_nabf_nabf_opt.init(n_abf, n_abf, nb_opt, nb_opt, 0, 0);
    // obtain the indices of atom-pair block necessary to build 2D block of a Hermitian/symmetric matrix
    const auto set_IJ_nabf_nabf = get_necessary_IJ_from_block_2D_sy('U', chi0.atbasis_abf, desc_nabf_nabf);
    const auto s0_s1 = get_s0_s1_for_comm_map2_first(set_IJ_nabf_nabf);
    // temp_block is used to collect data from IJ-pair data structure with comm_map2_first
    auto temp_block = init_local_mat<complex<double>>(desc_nabf_nabf, MAJOR::COL);
    // Below are the working arrays for matrix operations
    auto chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_eigen_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coul_chi0_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    auto coulwc_block = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    matrix_m<std::complex<double>> gamma_coulomb_basis;
    matrix_m<std::complex<double>> diagnostic_projection_work;
    matrix_m<std::complex<double>> diagnostic_projection;
    matrix_m<std::complex<double>> diagnostic_alpha_wc;
    if (output_2d_finite_q_diagnostics)
    {
        diagnostic_projection_work =
            init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
        diagnostic_projection = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
        diagnostic_alpha_wc = init_local_mat<complex<double>>(desc_nabf_nabf_opt, MAJOR::COL);
    }

    std::complex<double>* chi0_block_ptr = chi0_block.ptr();
    std::complex<double>* coul_block_ptr = coul_block.ptr();
    std::complex<double>* coul_chi0_block_ptr = coul_chi0_block.ptr();
    std::complex<double>* coul_eigen_block_ptr = coul_eigen_block.ptr();
    std::complex<double>* coulwc_block_ptr = coulwc_block.ptr();

#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if (use_gpu_replace_scalapack)
    {
        desc_nabf_nabf_opt.set_ddla_desc(blacs_h.ddla_handle); // set the descriptor for the device
        DEVICE_CHECK(deviceMallocAsync((void**)&chi0_block_ptr, chi0_block.size() * sizeof(std::complex<double>), blacs_h.ddla_handle->stream));
    }
#endif
#if defined(LIBRPA_USE_ELPA)
    if(use_elpa_sqrt_coulomb)
        desc_nabf_nabf_opt.set_elpa_handle(use_gpu_replace_scalapack);
#endif

    const double mem_blocks =
        (chi0_block.size() + coul_block.size() + coul_eigen_block.size() + coul_chi0_block.size() +
         coulwc_block.size() + diagnostic_projection_work.size() + diagnostic_projection.size() +
         diagnostic_alpha_wc.size()) *
        16.0e-6;
    ofs_myid << get_timestamp()
             << " Memory consumption of task-local blocks for screened Coulomb [MB]: " << mem_blocks
             << endl;

    const auto atpair_local = librpa_int::dispatch_upper_triangular_tasks(
        natom, blacs_h.myid, blacs_h.nprows, blacs_h.npcols,
        blacs_h.myprow, blacs_h.mypcol);
#ifdef LIBRPA_DEBUG
    ofs_myid << get_timestamp() << " atpair_local " << atpair_local << endl;
    ofs_myid << get_timestamp() << " s0_s1 " << s0_s1 << endl;
#endif

    // IJ pair of Wc to be returned
    pair<set<int>, set<int>> Iset_Jset_Wc;
    for (const auto &ap : atpair_local)
    {
        Iset_Jset_Wc.first.insert(ap.first);
        Iset_Jset_Wc.second.insert(ap.second);
    }

    // Prepare local basis indices for 2D->IJ map
    int I, iI;
    map<int, vector<int>> map_lor_v;
    map<int, vector<int>> map_loc_v;
    for (int i_lo = 0; i_lo != desc_nabf_nabf.m_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf.indx_l2g_r(i_lo);
        chi0.atbasis_abf.get_local_index(i_glo, I, iI);
        map_lor_v[I].push_back(iI);
    }
    for (int i_lo = 0; i_lo != desc_nabf_nabf.n_loc(); i_lo++)
    {
        int i_glo = desc_nabf_nabf.indx_l2g_c(i_lo);
        chi0.atbasis_abf.get_local_index(i_glo, I, iI);
        map_loc_v[I].push_back(iI);
    }

    const vector<Vector3_Order<double>> active_qpoints(chi0.active_qpoints().begin(),
                                                       chi0.active_qpoints().end());
    auto qpts = strict_2d_diagnostic_qpoint_order(active_qpoints, output_2d_finite_q_diagnostics);
    const auto &klist = chi0.pbc.klist;
    const auto &kfrac_list = chi0.pbc.kfrac_list;
    const auto atom_nabf = build_atom_nabf_map(chi0.atbasis_abf);
    const auto abf_layouts =
        chi0.atbasis_abf.build_species_basis_layouts(chi0.symmetry_context.atom_to_type);
    const bool use_symmetry_dense_chi0_collect =
        chi0.use_symmetry_context
        && comm_h.nprocs > 1
        && can_symmetrize_symmetry_chi0_ibz_blocks(
            chi0.symmetry_context, abf_layouts, atom_nabf, chi0.pbc);

    vec<double> eigenvalues(n_abf);
    global::profiler.stop("compute_Wc_freq_q_blacs_init");
    librpa_int::global::lib_printf_root("Time for Wc initialization (seconds, Wall/CPU): %f %f\n",
            global::profiler.get_wall_time_last("compute_Wc_freq_q_blacs_init"),
            global::profiler.get_cpu_time_last("compute_Wc_freq_q_blacs_init"));

    global::profiler.start("compute_Wc_freq_q_work");
#ifdef LIBRPA_USE_LIBRI
    const auto validate_coulomb_block_shape = [&](const char *stage, const int mu,
                                                  const int nu,
                                                  const Vector3_Order<double> &q,
                                                  const auto &vq,
                                                  const int n_mu,
                                                  const int n_nu) {
        if (vq->nr == n_mu && vq->nc == n_nu) return;
        std::ostringstream errmsg;
        errmsg << "Coulomb block dimension mismatch while preparing " << stage
               << " for Wc at q=(" << q.x << ", " << q.y << ", " << q.z
               << "), atom pair (" << mu << ", " << nu << "): block shape is "
               << vq->nr << "x" << vq->nc << " but chi0 auxiliary basis expects "
               << n_mu << "x" << n_nu
               << ". Check use_shrink_abfs/use_shrink_chi and Coulomb prefixes; "
                  "legacy shrink Coulomb files cannot be used with full-chi Wc.";
        throw LIBRPA_RUNTIME_ERROR(errmsg.str());
    };
    const bool strict_2d_requested =
        strict_2d_complete_wc_requested(replace_w_head, option_dielect_func,
                                        df_headwing != nullptr && df_headwing->use_2d_dielectric);
    if (output_2d_finite_q_diagnostics && !strict_2d_requested)
        throw LIBRPA_RUNTIME_ERROR("output_2d_finite_q_diagnostics requires strict 2D complete-Wc");
    constexpr double diagnostic_inverse_dielectric_alpha = 0.25;
    std::vector<std::complex<double>> gamma_head_vector;
    int gamma_n_nonsingular = 0;
    std::ofstream finite_q_diagnostics;
    std::ofstream gamma_wc_diagnostics;
    int diagnostic_output_ready = 1;
    if (output_2d_finite_q_diagnostics && comm_h.is_root())
    {
        finite_q_diagnostics.open(path_as_directory(output_dir) + "strict2d_finite_q_scaling.csv",
                                  std::ios::out | std::ios::trunc);
        gamma_wc_diagnostics.open(path_as_directory(output_dir) + "strict2d_gamma_wc_blocks.csv",
                                  std::ios::out | std::ios::trunc);
        diagnostic_output_ready = finite_q_diagnostics && gamma_wc_diagnostics ? 1 : 0;
        if (diagnostic_output_ready)
        {
            finite_q_diagnostics << strict_2d_finite_q_diagnostics_header() << '\n'
                                 << std::scientific << std::setprecision(16);
            gamma_wc_diagnostics << strict_2d_gamma_wc_diagnostics_header() << '\n'
                                 << std::scientific << std::setprecision(16);
        }
    }
    if (output_2d_finite_q_diagnostics)
    {
        MPI_Bcast(&diagnostic_output_ready, 1, MPI_INT, 0, comm_h.comm);
        if (!diagnostic_output_ready)
            throw LIBRPA_RUNTIME_ERROR("cannot open strict 2D diagnostics output");
        if (comm_h.is_root())
            std::cout << "Writing strict 2D fixed-Gamma-basis diagnostics under "
                      << path_as_directory(output_dir) << std::endl;
    }
    for (const auto &q : qpts)
    {
        const int iq = std::distance(qpts.cbegin(), std::find(qpts.cbegin(), qpts.cend(), q));
        const auto q_in_k = std::find(klist.cbegin(), klist.cend(), q);
        // q-point in fractional coordinates
        Vector3_Order<double> qf;
        if (q_in_k != klist.cend())
        {
            const int iq_in_k = std::distance(klist.cbegin(), q_in_k);
            qf = kfrac_list[iq_in_k];
        }
        else
        {
            qf = Vector3_Order<double>{chi0.pbc.latvec * q};
        }
        librpa_int::global::lib_printf_root("Computing Wc(q), %d / %d, q=(%f, %f, %f)\n", iq + 1,
                                            qpts.size(), qf.x, qf.y, qf.z);
        const bool strict_2d_complete_wc_gamma = use_strict_2d_complete_wc_gamma_route(
            replace_w_head, option_dielect_func,
            df_headwing != nullptr && df_headwing->use_2d_dielectric, is_gamma_point(q),
            !epsmac_LF_imagfreq.empty());
        const bool debug_output = global::should_output(LIBRPA_VERBOSE_DEBUG);
        const bool gamma_point = is_gamma_point(q);
        const bool gamma_full_headwing =
            gamma_point && !epsmac_LF_imagfreq.empty() && option_dielect_func == 3;
        const bool gamma_head_only =
            gamma_point && !epsmac_LF_imagfreq.empty() && option_dielect_func != 3;
        coul_block.zero_out();
        coulwc_block.zero_out();
        // lib_printf("coul_block\n%s", str(coul_block).c_str());

        // q-array for LibRI object
        std::array<double, 3> qa = {q.x, q.y, q.z};

        // collect the block elements of truncated coulomb matrices first
        // as we reuse coul_eigen_block to reduce memory usage
        global::profiler.start("epsilon_prepare_coulwc_sqrt", "Prepare sqrt of truncated Coulomb");
        {
            size_t n_singular_coulwc;
            // LibRI tensor for communication, release once done
            std::map<int, std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>> couleps_libri;
            global::profiler.start("epsilon_prepare_coulwc_sqrt_1", "Setup libRI object");

            for (const auto& Mu_coulmat: coulmat_wc)
            {
                const auto Mu = Mu_coulmat.first;
                for (const auto &Nu_coulmat : Mu_coulmat.second)
                {
                    const auto Nu = Nu_coulmat.first;
                    const auto &Vq = coulmat_wc.at(Mu).at(Nu).at(q);
                    const auto n_mu = chi0.atbasis_abf.get_atom_nb(Mu);
                    const auto n_nu = chi0.atbasis_abf.get_atom_nb(Nu);
                    validate_coulomb_block_shape("truncated Coulomb", Mu, Nu, q, Vq, n_mu, n_nu);
                    std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                    auto pvq = std::make_shared<std::valarray<complex<double>>>();
                    *pvq = Vq_va;
                    couleps_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
                    // coulmat_wc.at(Mu).at(Nu).at(q).reset();
                }
            }
            global::profiler.stop("epsilon_prepare_coulwc_sqrt_1");

            global::profiler.start("epsilon_prepare_coulwc_sqrt_2", "libRI Communicate");
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(comm_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
            global::profiler.stop("epsilon_prepare_coulwc_sqrt_2");

            global::profiler.start("epsilon_prepare_coulwc_sqrt_3", "Collect 2D-block from IJ");
            // for (const auto &IJ: set_IJ_nabf_nabf)
            // {
            //     const auto &I = IJ.first;
            //     const auto &J = IJ.second;
            //     collect_block_from_IJ_storage_syhe(
            //         coulwc_block, desc_nabf_nabf, chi0.atbasis_abf, IJ.first,
            //         IJ.second, true, CONE, IJq_coul.at(I).at({J, qa}).ptr(), MAJOR::ROW);
            // }
            collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(
                temp_block, desc_nabf_nabf, chi0.atbasis_abf, qa, true, C_ONE, IJq_coul,
                MAJOR::ROW);
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                          coulwc_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                          blacs_h.ictxt);
            global::profiler.stop("epsilon_prepare_coulwc_sqrt_3");
            global::profiler.start("epsilon_prepare_coulwc_sqrt_4", "Perform square root");
            if (is_gamma_point(q))
            {
                LaConnector::power_hemat_la_real(
                    coulwc_block, desc_nabf_nabf_opt, coul_eigen_block, desc_nabf_nabf_opt,
                    n_singular_coulwc, eigenvalues.c, 0.5, sqrt_coulomb_threshold,
                    use_gpu_replace_scalapack, use_elpa_sqrt_coulomb, (double*)chi0_block_ptr + chi0_block.size(), 
                (double*)chi0_block_ptr, (double*)coul_chi0_block_ptr);
            }
            else
            {
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
                if(use_gpu_replace_scalapack)
                    DEVICE_CHECK(deviceMallocAsync((void**)&coul_block_ptr, coul_block.size() * sizeof(std::complex<double>), blacs_h.ddla_handle->stream));
#endif
                LaConnector::power_hemat_la(
                    coulwc_block, desc_nabf_nabf_opt, coul_eigen_block, desc_nabf_nabf_opt,
                    n_singular_coulwc, eigenvalues.c, 0.5, sqrt_coulomb_threshold, use_gpu_replace_scalapack,
                    use_elpa_sqrt_coulomb, coul_block_ptr, chi0_block_ptr, coul_chi0_block_ptr);
            }
            global::profiler.stop("epsilon_prepare_coulwc_sqrt_4");
        }
        global::profiler.stop("epsilon_prepare_coulwc_sqrt");
        librpa_int::global::lib_printf_root("Time to prepare sqrt root of Coulomb for Wc(q) (seconds, Wall/CPU): %f %f\n",
                global::profiler.get_wall_time_last("epsilon_prepare_coulwc_sqrt"),
                global::profiler.get_cpu_time_last("epsilon_prepare_coulwc_sqrt"));

        global::profiler.start("epsilon_prepare_couleps_sqrt", "Prepare sqrt of bare Coulomb");
        // collect the block elements of coulomb matrices
        {
            // LibRI tensor for communication, release once done
            std::map<int,
                     std::map<std::pair<int, std::array<double, 3>>, RI::Tensor<complex<double>>>>
                couleps_libri;
            if (debug_output) ofs_myid << get_timestamp() << " Start build couleps_libri" << endl;

            for (const auto& Mu_coulmat: coulmat_eps)
            {
                const auto Mu = Mu_coulmat.first;
                for(const auto& Nu_coulmat : Mu_coulmat.second){
                    const auto Nu = Nu_coulmat.first;
                    const auto &Vq = coulmat_eps.at(Mu).at(Nu).at(q);
                    const auto n_mu = chi0.atbasis_abf.get_atom_nb(Mu);
                    const auto n_nu = chi0.atbasis_abf.get_atom_nb(Nu);
                    validate_coulomb_block_shape("bare Coulomb", Mu, Nu, q, Vq, n_mu, n_nu);
                    std::valarray<complex<double>> Vq_va(Vq->c, Vq->size);
                    auto pvq = std::make_shared<std::valarray<complex<double>>>();
                    *pvq = Vq_va;
                    couleps_libri[Mu][{Nu, qa}] = Tensor<complex<double>>({n_mu, n_nu}, pvq);
                }
            }

            if (debug_output) ofs_myid << get_timestamp() << " Done build couleps_libri" << endl;
            // ofs_myid << "Couleps_libri" << endl << couleps_libri;
            // if (couleps_libri.size() == 0)
            //     throw std::logic_error("data at q-point not found in coulmat_eps");

            // perform communication
            if (debug_output)
                ofs_myid << get_timestamp() << " Start collect couleps_libri, targets" << endl;
#ifdef LIBRPA_DEBUG
            ofs_myid << set_IJ_nabf_nabf << endl;
            ofs_myid << "Extended blocks" << endl;
            ofs_myid << "atom 1: " << s0_s1.first << endl;
            ofs_myid << "atom 2: " << s0_s1.second << endl;
#endif
            // ofs_myid << "Owned blocks\n";
            // print_keys(ofs_myid, couleps_libri);
            // comm_h.barrier();
            const auto IJq_coul = RI::Communicate_Tensors_Map_Judge::comm_map2_first(comm_h.comm, couleps_libri, s0_s1.first, s0_s1.second);
            if (debug_output)
                ofs_myid << get_timestamp() << " Done collect couleps_libri, collected blocks"
                         << endl;

            if (debug_output)
                ofs_myid << get_timestamp() << " Start construct couleps 2D block" << endl;
            collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(
                temp_block, desc_nabf_nabf, chi0.atbasis_abf, qa, true, C_ONE, IJq_coul,
                MAJOR::ROW);
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1, desc_nabf_nabf.desc,
                                          coul_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                          blacs_h.ictxt);
            if (debug_output)
                ofs_myid << get_timestamp() << " Done construct couleps 2D block" << endl;
        }
        // char fn[100];
        // sprintf(fn, "couleps_iq_%d.mtx", iq);
        // print_matrix_mm_file_parallel(fn, coul_block, desc_nabf_nabf);
        // ofs_myid << str(coul_block);
        // lib_printf("coul_block\n%s", str(coul_block).c_str());

        size_t n_singular;
        if (debug_output) ofs_myid << get_timestamp() << " Start power hemat couleps\n";
        matrix_m<std::complex<double>> sqrtveig_blacs;
        if (is_gamma_point(q))
        {
            // choice of power_hemat_blacs_real/power_hemat_blacs_desc
            // leads to sub-meV difference
            sqrtveig_blacs = LaConnector::power_hemat_la_real(
                coul_block, desc_nabf_nabf_opt, coul_eigen_block, desc_nabf_nabf_opt,
                n_singular, eigenvalues.c, 0.5, sqrt_coulomb_threshold,
                use_gpu_replace_scalapack, use_elpa_sqrt_coulomb, (double*)chi0_block_ptr + chi0_block.size(), 
                (double*)chi0_block_ptr, (double*)coul_chi0_block_ptr);
        }
        else
        {
            sqrtveig_blacs = LaConnector::power_hemat_la(
                coul_block, desc_nabf_nabf_opt, coul_eigen_block, desc_nabf_nabf_opt,
                n_singular, eigenvalues.c, 0.5, sqrt_coulomb_threshold, use_gpu_replace_scalapack,
                use_elpa_sqrt_coulomb, coul_block_ptr, chi0_block_ptr, coul_chi0_block_ptr);
        }
        double finite_q_head_overlap = std::numeric_limits<double>::quiet_NaN();
        if (output_2d_finite_q_diagnostics)
        {
            const auto current_head =
                collect_normalized_blacs_column(coul_eigen_block, desc_nabf_nabf_opt, 0, comm_h);
            if (is_gamma_point(q))
                gamma_head_vector = current_head;
            else
            {
                if (gamma_head_vector.empty())
                    throw LIBRPA_RUNTIME_ERROR(
                        "strict 2D finite-q diagnostics require Gamma to precede finite q");
                finite_q_head_overlap = normalized_vector_overlap(gamma_head_vector, current_head);
            }
        }
        if (debug_output) ofs_myid << get_timestamp() << " Done power hemat couleps\n";
        const size_t n_nonsingular = n_abf - n_singular;
        if (gamma_full_headwing && n_singular != 0)
        {
            librpa_int::global::lib_printf_root(
                "Gamma option-3 ABF-space head/wing Wc: using retained Coulomb "
                "subspace n_nonsingular=%zu of n_abf=%zu (n_filtered=%zu, "
                "sqrt_coulomb_threshold=%g)\n",
                n_nonsingular, n_abf, n_singular, sqrt_coulomb_threshold);
        }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        if (is_gamma_point(q) && use_gpu_replace_scalapack)
            DEVICE_CHECK(deviceMallocAsync((void**)&coul_block_ptr, coul_block.size() * sizeof(std::complex<double>), blacs_h.ddla_handle->stream));
#endif
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        std::complex<double> *gamma_head_eigen_device = nullptr;
        std::complex<double> *gamma_head_h_device = nullptr;
#endif
        ArrayDesc desc_1x1(blacs_h);
        std::complex<double> gamma_head_h_host{0.0, 0.0};
        if (gamma_head_only)
        {
            if (n_nonsingular == 0)
                throw LIBRPA_RUNTIME_ERROR(
                    "Gamma head correction requires a non-singular Coulomb eigenvector");
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
            if (use_gpu_replace_scalapack)
            {
                if (blacs_h.nprows != blacs_h.npcols)
                    throw LIBRPA_RUNTIME_ERROR(
                        "GPU Gamma head-only PGEMM requires a square process grid "
                        "(blacs_h.nprows == blacs_h.npcols)");
                const size_t eigen_local_size =
                    std::max<size_t>(1, coul_eigen_block.size());
                DEVICE_CHECK(deviceMallocAsync(
                    reinterpret_cast<void **>(&gamma_head_eigen_device),
                    eigen_local_size * sizeof(std::complex<double>),
                    blacs_h.ddla_handle->stream));
                if (coul_eigen_block.size() > 0)
                {
                    DEVICE_CHECK(deviceMemcpyAsync(
                        gamma_head_eigen_device, coul_eigen_block.ptr(),
                        coul_eigen_block.size() * sizeof(std::complex<double>),
                        deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
                }
                DEVICE_CHECK(deviceMallocAsync(
                    reinterpret_cast<void **>(&gamma_head_h_device),
                    sizeof(std::complex<double>),
                    blacs_h.ddla_handle->stream));
            }
#endif
            desc_1x1.init(1, 1, nb_opt, nb_opt, 0, 0);
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
            if (use_gpu_replace_scalapack)
                desc_1x1.set_ddla_desc(blacs_h.ddla_handle);
#endif
        }
        // The scaled Coulomb eigenvectors are no longer needed in the frequency
        // loop: the ABF-space path uses coul_block (sqrt(V)) directly.
        sqrtveig_blacs.clear();
        if (strict_2d_complete_wc_gamma)
        {
            const int n_coulomb = as_int(n_nonsingular);
            // cal_strict_2d_wc interprets channel 0 as the singular Coulomb
            // eigenvector and channels 1.. as the regular body. Express the
            // full-Ewald external legs in that same fixed Gamma basis first.
            ScalapackConnector::pgemm_f('C', 'N', n_coulomb, n_abf, n_abf, C_ONE,
                                        coul_eigen_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                        coulwc_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc, C_ZERO,
                                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
            ScalapackConnector::pgemm_f('N', 'N', n_coulomb, n_coulomb, n_abf, C_ONE,
                                        coul_chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                        coul_eigen_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                        C_ZERO, coulwc_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
            if (comm_h.is_root())
                std::cout << "Projected full-Ewald regular Wc legs into the fixed Gamma "
                             "Coulomb basis."
                          << std::endl;
        }
        if (output_2d_finite_q_diagnostics && is_gamma_point(q))
        {
            gamma_coulomb_basis = coul_eigen_block.copy();
            gamma_n_nonsingular = as_int(n_nonsingular);
        }
        Strict2dBlockMetrics alpha_wc_metrics;
        if (output_2d_finite_q_diagnostics)
        {
            if (gamma_n_nonsingular < 1 || gamma_coulomb_basis.size() == 0)
                throw LIBRPA_RUNTIME_ERROR(
                    "strict 2D alpha diagnostics require the fixed Gamma basis");
            diagnostic_alpha_wc.zero_out();
            const std::complex<double> alpha_response = diagnostic_inverse_dielectric_alpha - 1.0;
            if (is_gamma_point(q))
            {
                const int nbody = gamma_n_nonsingular - 1;
                if (nbody < 1)
                    throw LIBRPA_RUNTIME_ERROR(
                        "strict 2D alpha diagnostics require a regular Gamma body");
                ScalapackConnector::pgemm_f(
                    'N', 'N', nbody, nbody, nbody, alpha_response, coulwc_block.ptr(), 2, 2,
                    desc_nabf_nabf_opt.desc, coulwc_block.ptr(), 2, 2, desc_nabf_nabf_opt.desc,
                    C_ZERO, diagnostic_alpha_wc.ptr(), 2, 2, desc_nabf_nabf_opt.desc);
                const int head_row = desc_nabf_nabf_opt.indx_g2l_r(0);
                const int head_column = desc_nabf_nabf_opt.indx_g2l_c(0);
                if (head_row >= 0 && head_column >= 0)
                    diagnostic_alpha_wc(head_row, head_column) =
                        alpha_response * df_headwing->get_strict_2d_bare_coulomb_gamma_average();
                alpha_wc_metrics = collect_strict_2d_block_metrics(
                    diagnostic_alpha_wc, desc_nabf_nabf_opt, gamma_n_nonsingular, comm_h);
            }
            else
            {
                ScalapackConnector::pgemm_f(
                    'N', 'N', n_abf, n_abf, n_abf, alpha_response, coulwc_block.ptr(), 1, 1,
                    desc_nabf_nabf_opt.desc, coulwc_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                    C_ZERO, diagnostic_alpha_wc.ptr(), 1, 1, desc_nabf_nabf_opt.desc);
                project_blacs_matrix_in_basis(diagnostic_alpha_wc, gamma_coulomb_basis,
                                              desc_nabf_nabf_opt, gamma_n_nonsingular,
                                              diagnostic_projection_work, diagnostic_projection);
                alpha_wc_metrics = collect_strict_2d_block_metrics(
                    diagnostic_projection, desc_nabf_nabf_opt, gamma_n_nonsingular, comm_h);
            }
        }
        global::profiler.stop("epsilon_prepare_couleps_sqrt");
        librpa_int::global::lib_printf_root("Time to prepare sqrt root of Coulomb for Epsilon(q) (seconds, Wall/CPU): %f %f\n",
                global::profiler.get_wall_time_last("epsilon_prepare_couleps_sqrt"),
                global::profiler.get_cpu_time_last("epsilon_prepare_couleps_sqrt"));
        if (debug_output) ofs_myid << get_timestamp() << " Done couleps sqrt\n";
        std::flush(ofs_myid);
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        if(use_gpu_replace_scalapack)
            DEVICE_CHECK(deviceMallocAsync((void**)&coul_chi0_block_ptr, coul_chi0_block.size() * sizeof(std::complex<double>), blacs_h.ddla_handle->stream));
        if (gamma_head_only && use_gpu_replace_scalapack)
            DEVICE_CHECK(deviceStreamSynchronize(blacs_h.ddla_handle->stream));
#endif    
        for (const auto &freq : chi0.tfg.get_freq_nodes())
        {
            const auto ifreq = chi0.tfg.get_freq_index(freq);
            std::complex<double> finite_q_p_head = std::numeric_limits<double>::quiet_NaN();
            Strict2dBlockMetrics finite_q_p_metrics;
            global::profiler.start("epsilon_wc_work_q_omega");
            global::profiler.start("epsilon_prepare_chi0_2d", "Prepare Chi0 2D block");
            chi0_block.zero_out();
            {
                std::map<int, std::map<std::pair<int, std::array<double, 3>>,
                                       RI::Tensor<complex<double>>>>
                    chi0_libri;
                const bool has_local_chi0_q =
                    chi0.get_chi0_q().count(freq) > 0 && chi0.get_chi0_q().at(freq).count(q) > 0;
                atom_mapping<ComplexMatrix>::pair_t_old chi0_wq;
                if (has_local_chi0_q)
                {
                    chi0_wq = chi0.get_chi0_q().at(freq).at(q);
                    if (chi0.use_symmetry_context)
                    {
                        chi0_wq = symmetrize_symmetry_chi0_ibz_blocks_if_needed(
                            comm_h, chi0.symmetry_context, abf_layouts, chi0_wq, q,
                            chi0.pbc, atom_nabf);
                    }
                }
                else if (use_symmetry_dense_chi0_collect)
                {
                    chi0_wq = symmetrize_symmetry_chi0_ibz_blocks_if_needed(
                        comm_h, chi0.symmetry_context, abf_layouts, chi0_wq, q,
                        chi0.pbc, atom_nabf);
                }

                if (use_symmetry_dense_chi0_collect)
                {
                    const auto chi0_dense = build_dense_symmetry_hermitian_matrix_from_local_blocks(
                        to_ordered_symmetry_blocks(chi0_wq), atom_nabf);
                    temp_block.zero_out();
                    for (int i_lo = 0; i_lo != desc_nabf_nabf.m_loc(); ++i_lo)
                    {
                        const int i_glo = desc_nabf_nabf.indx_l2g_r(i_lo);
                        for (int j_lo = 0; j_lo != desc_nabf_nabf.n_loc(); ++j_lo)
                        {
                            const int j_glo = desc_nabf_nabf.indx_l2g_c(j_lo);
                            temp_block(i_lo, j_lo) = chi0_dense(i_glo, j_glo);
                        }
                    }
                    ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1,
                                                  desc_nabf_nabf.desc, chi0_block.ptr(), 1, 1,
                                                  desc_nabf_nabf_opt.desc, blacs_h.ictxt);
                }
                else
                {
                    for (const auto &M_Nchi : chi0_wq)
                    {
                        const auto &M = M_Nchi.first;
                        const auto n_mu = chi0.atbasis_abf.get_atom_nb(M);
                        for (const auto &N_chi : M_Nchi.second)
                        {
                            const auto &N = N_chi.first;
                            const auto n_nu = chi0.atbasis_abf.get_atom_nb(N);
                            const auto &chi = N_chi.second;
                            std::valarray<complex<double>> chi_va(chi.c, chi.size);
                            auto pchi = std::make_shared<std::valarray<complex<double>>>();
                            *pchi = chi_va;
                            chi0_libri[M][{N, qa}] =
                                RI::Tensor<complex<double>>({n_mu, n_nu}, pchi);
                        }
                    }
                    // ofs_myid << "chi0_libri" << endl << chi0_libri;
                    global::profiler.start("epsilon_prepare_chi0_2d_comm_map2",
                                           LIBRPA_VERBOSE_DEBUG);
                    const auto IJq_chi0 = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
                        comm_h.comm, chi0_libri, s0_s1.first, s0_s1.second);
                    global::profiler.stop("epsilon_prepare_chi0_2d_comm_map2");
                    // ofs_myid << "IJq_chi0" << endl << IJq_chi0;
                    // for (const auto &IJ: set_IJ_nabf_nabf)
                    // {
                    //     const auto &I = IJ.first;
                    //     const auto &J = IJ.second;
                    //     collect_block_from_IJ_storage_syhe(
                    //         chi0_block, desc_nabf_nabf, chi0.atbasis_abf, IJ.first,
                    //         IJ.second, true, CONE, IJq_chi0.at(I).at({J, qa}).ptr(), MAJOR::ROW);
                    // }
                    global::profiler.start("epsilon_prepare_chi0_2d_collect_block",
                                           LIBRPA_VERBOSE_DEBUG);
                    collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(
                        temp_block, desc_nabf_nabf, chi0.atbasis_abf, qa, true, C_ONE,
                        IJq_chi0, MAJOR::ROW);
                    ScalapackConnector::pgemr2d_f(n_abf, n_abf, temp_block.ptr(), 1, 1,
                                                  desc_nabf_nabf.desc, chi0_block.ptr(), 1, 1,
                                                  desc_nabf_nabf_opt.desc, blacs_h.ictxt);
                    global::profiler.stop("epsilon_prepare_chi0_2d_collect_block");
                }
                // Release the chi0 block for this frequency and q to reduce memory load,
                // as they will not be used again.
                if (has_local_chi0_q)
                {
                    chi0.free_chi0_q(freq, q);
                }
                std::ostringstream chi0_debug_name;
                chi0_debug_name << std::fixed << std::setprecision(10)
                                << "chi0_block_qx_" << q.x << "_qy_" << q.y << "_qz_"
                                << q.z << "_freq_" << ifreq << ".mtx";
                dump_blacs_debug_matrix(debug, output_dir, chi0_debug_name.str(), chi0_block,
                                        desc_nabf_nabf_opt, "");
            }
            global::profiler.stop("epsilon_prepare_chi0_2d");

#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
            if (use_gpu_replace_scalapack)
            {
                DEVICE_CHECK(deviceMemcpyAsync(chi0_block_ptr, chi0_block.ptr(), chi0_block.size() * sizeof(complex<double>), deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
            }
#endif
            profiler.start("epsilon_compute_eps", "Compute dielectric matrix");
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
            if (use_gpu_replace_scalapack)
            {
                DEVICE_CHECK(deviceMemcpyAsync(
                    coul_block_ptr, coul_block.ptr(),
                    coul_block.size() * sizeof(std::complex<double>),
                    deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
            }
#endif
            profiler.start("epsilon_compute_eps_pgemm_1", LIBRPA_VERBOSE_DEBUG);
            LaConnector::pgemm(
                'N', 'N', n_abf, n_abf, n_abf, {1.0, 0.0}, coul_block_ptr,
                1, 1, desc_nabf_nabf_opt, chi0_block_ptr, 1, 1,
                desc_nabf_nabf_opt, {0.0, 0.0}, coul_chi0_block_ptr, 1, 1,
                desc_nabf_nabf_opt);
            profiler.stop("epsilon_compute_eps_pgemm_1");
            profiler.start("epsilon_compute_eps_pgemm_2", LIBRPA_VERBOSE_DEBUG);
            LaConnector::pgemm(
                'N', 'N', n_abf, n_abf, n_abf, {-1.0, 0.0},
                coul_chi0_block_ptr, 1, 1, desc_nabf_nabf_opt, coul_block_ptr,
                1, 1, desc_nabf_nabf_opt, {0.0, 0.0}, chi0_block_ptr, 1, 1,
                desc_nabf_nabf_opt);
            profiler.stop("epsilon_compute_eps_pgemm_2");
            LaConnector::pdam(1.0, chi0_block_ptr, desc_nabf_nabf_opt);

            if (gamma_full_headwing)
            {
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                if (use_gpu_replace_scalapack)
                {
                    DEVICE_CHECK(deviceMemcpyAsync(
                        chi0_block.ptr(), chi0_block_ptr,
                        chi0_block.size() * sizeof(complex<double>),
                        deviceMemcpyDeviceToHost, blacs_h.ddla_handle->stream));
                    DEVICE_CHECK(deviceStreamSynchronize(blacs_h.ddla_handle->stream));
                }
#endif
                ofs_myid << get_timestamp() << "Perform the ABF-space head & wing overwrite"
                         << endl;
                if (df_headwing == nullptr)
                    throw LIBRPA_RUNTIME_ERROR("Head/wing dielectric function is not initialized");
                if (strict_2d_complete_wc_gamma)
                    df_headwing->rewrite_strict_2d_wc(
                        chi0_block, ifreq, desc_nabf_nabf_opt, coulwc_block);
                else
                    df_headwing->rewrite_eps_abf_space(
                        chi0_block, ifreq, coul_block, coul_eigen_block,
                        desc_nabf_nabf_opt, n_nonsingular, sqrt_coulomb_threshold,
                        use_cholesky_gw_wc, use_gpu_replace_scalapack);

#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                if (use_gpu_replace_scalapack)
                {
                    DEVICE_CHECK(deviceMemcpyAsync(
                        chi0_block_ptr, chi0_block.ptr(),
                        chi0_block.size() * sizeof(complex<double>),
                        deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
                }
#endif
                // The production epsilon matrix has already been formed above.  Copy it
                // back only for the optional fixed-Gamma-basis diagnostic projection.
                if (output_2d_finite_q_diagnostics && ifreq == 0)
                {
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                    if (use_gpu_replace_scalapack)
                    {
                        DEVICE_CHECK(deviceMemcpyAsync(chi0_block.ptr(), chi0_block_ptr,
                                                       chi0_block.size() * sizeof(complex<double>),
                                                       deviceMemcpyDeviceToHost,
                                                       blacs_h.ddla_handle->stream));
                        DEVICE_CHECK(deviceStreamSynchronize(blacs_h.ddla_handle->stream));
                    }
#endif
                    project_blacs_matrix_in_basis(
                        chi0_block, gamma_coulomb_basis, desc_nabf_nabf_opt, gamma_n_nonsingular,
                        diagnostic_projection_work, diagnostic_projection);
                    finite_q_p_metrics = collect_strict_2d_block_metrics(
                        diagnostic_projection, desc_nabf_nabf_opt, gamma_n_nonsingular, comm_h);
                    finite_q_p_metrics.head *= -1.0;
                    finite_q_p_head = finite_q_p_metrics.head;
                }
            }
            else if (gamma_head_only)
            {
                ofs_myid << get_timestamp()
                         << " Entering dielectric matrix head overwrite" << endl;
                global::profiler.start("epsilon_gamma_head_projection",
                                       LIBRPA_VERBOSE_DEBUG);
                std::complex<double> *eigen_ptr;
                #if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                if (use_gpu_replace_scalapack)
                    eigen_ptr = gamma_head_eigen_device;
                else
                #endif
                    eigen_ptr = coul_eigen_block.ptr();

                global::profiler.start("epsilon_gamma_head_projection_pgemm_y",
                                       LIBRPA_VERBOSE_DEBUG);
                LaConnector::pgemm(
                    'N', 'N', n_abf, 1, n_abf, {1.0, 0.0},
                    chi0_block_ptr, 1, 1, desc_nabf_nabf_opt,
                    eigen_ptr, 1, 1, desc_nabf_nabf_opt,
                    {0.0, 0.0}, coul_chi0_block_ptr, 1, 1,
                    desc_nabf_nabf_opt);
                global::profiler.stop("epsilon_gamma_head_projection_pgemm_y");

                global::profiler.start("epsilon_gamma_head_projection_pgemm_h",
                                       LIBRPA_VERBOSE_DEBUG);
                std::complex<double> *h_ptr;
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                if (use_gpu_replace_scalapack)
                    h_ptr = gamma_head_h_device;
                else
#endif
                    h_ptr = &gamma_head_h_host;
                LaConnector::pgemm(
                    'C', 'N', 1, 1, n_abf, {1.0, 0.0},
                    eigen_ptr, 1, 1, desc_nabf_nabf_opt,
                    coul_chi0_block_ptr, 1, 1, desc_nabf_nabf_opt,
                    {0.0, 0.0}, h_ptr, 1, 1, desc_1x1);
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                if (use_gpu_replace_scalapack)
                {
                    if (desc_1x1.is_src())
                    {
                        DEVICE_CHECK(deviceMemcpyAsync(
                            &gamma_head_h_host, gamma_head_h_device,
                            sizeof(std::complex<double>),
                            deviceMemcpyDeviceToHost,
                            blacs_h.ddla_handle->stream));
                    }
                    DEVICE_CHECK(deviceStreamSynchronize(
                        blacs_h.ddla_handle->stream));
                }
#endif
                const int h_root = blacs_h.get_pnum(0, 0);
                MPI_Bcast(&gamma_head_h_host, 1, MPI_CXX_DOUBLE_COMPLEX,
                          h_root, desc_1x1.comm());
                global::profiler.stop("epsilon_gamma_head_projection_pgemm_h");

                const std::complex<double> head_rank_one_coefficient =
                    epsmac_LF_imagfreq.at(as_size(ifreq)) - gamma_head_h_host;
                global::profiler.stop("epsilon_gamma_head_projection");

                global::profiler.start("epsilon_gamma_head_rank1_update",
                                       LIBRPA_VERBOSE_DEBUG);
                global::profiler.start("epsilon_gamma_head_rank1_update_pgemm",
                                       LIBRPA_VERBOSE_DEBUG);
                LaConnector::pgemm(
                    'N', 'C', n_abf, n_abf, 1, head_rank_one_coefficient,
                    eigen_ptr, 1, 1, desc_nabf_nabf_opt,
                    eigen_ptr, 1, 1, desc_nabf_nabf_opt,
                    {1.0, 0.0}, chi0_block_ptr, 1, 1,
                    desc_nabf_nabf_opt);
                global::profiler.stop("epsilon_gamma_head_rank1_update_pgemm");
                global::profiler.stop("epsilon_gamma_head_rank1_update");
            }
            profiler.stop("epsilon_compute_eps");
            // debug for Coulomb, epsilon^{-1} - 1 = -0.75
            // for (int i = 0; i != n_abf; i++)
            // {
            //     for (int j = 0; j != n_abf; j++)
            //     {
            //         const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
            //         if (ilo < 0) continue;
            //         const int jlo = desc_nabf_nabf_opt.indx_g2l_c(j);
            //         if (jlo < 0) continue;
            //         if (i == j)
            //             chi0_block(ilo, jlo) = -0.75;
            //         else
            //             chi0_block(ilo, jlo) = 0.0;
            //     }
            // }
            // debug for unfold shrink Wc
            // for (int i = 0; i != n_abf; i++)
            //{
            //     const int ilo = desc_nabf_nabf_opt.indx_g2l_r(i);
            //     if (ilo < 0) continue;
            //     for (int j = 0; j != n_abf; j++)
            //     {
            //         const int jlo = desc_nabf_nabf_opt.indx_g2l_c(j);
            //         if (jlo < 0) continue;
            //         if (i == j)
            //             chi0_block(ilo, jlo) = 1.0;
            //         else
            //             chi0_block(ilo, jlo) = 0.0;
            //     }
            // }
            // debug end
            std::ostringstream epsinv_debug_name;
            epsinv_debug_name << std::fixed << std::setprecision(10)
                              << "epsinv_minus_identity_qx_" << q.x << "_qy_" << q.y
                              << "_qz_" << q.z << "_freq_" << ifreq << ".mtx";
            dump_blacs_debug_matrix(debug, output_dir, epsinv_debug_name.str(), chi0_block,
                                    desc_nabf_nabf_opt, "", 1e-10);

            global::profiler.start("epsilon_to_wc");
            if (strict_2d_complete_wc_gamma)
            {
                if (ifreq == 0 && comm_h.is_root())
                    std::cout << "Skipping factorized q=0 Coulomb multiplication because the "
                                 "strict 2D branch already contains the complete Gamma-cell Wc "
                                 "average."
                              << std::endl;
            }
            else if (epsmac_LF_imagfreq.size() > 0 && is_gamma_point(q) && option_dielect_func == 3)
            {
                // Dielectric matrix is already inverted, only multiply by square root coulwc from both sides
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                if(use_gpu_replace_scalapack)
                {
                    coulwc_block_ptr = coul_block_ptr;
                    DEVICE_CHECK(deviceMemcpyAsync(coulwc_block_ptr, coulwc_block.ptr(), coulwc_block.size() * sizeof(complex<double>), deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
                }
#endif
                LaConnector::pdam(-1.0, chi0_block_ptr, desc_nabf_nabf_opt);
                global::profiler.start("epsilon_multiply_coulwc_1", "Multiply truncated Coulomb",
                                      LIBRPA_VERBOSE_DEBUG);
                LaConnector::pgemm('N', 'N', n_abf, n_abf, n_abf, {1.0, 0.0},
                        coulwc_block_ptr, 1, 1, desc_nabf_nabf_opt,
                        chi0_block_ptr, 1, 1, desc_nabf_nabf_opt, {0.0, 0.0},
                        coul_chi0_block_ptr, 1, 1, desc_nabf_nabf_opt);
                LaConnector::pgemm('N', 'N', n_abf, n_abf, n_abf, {1.0, 0.0},
                        coul_chi0_block_ptr, 1, 1, desc_nabf_nabf_opt,
                        coulwc_block_ptr, 1, 1, desc_nabf_nabf_opt, {0.0, 0.0},
                        chi0_block_ptr, 1, 1, desc_nabf_nabf_opt);
                global::profiler.stop("epsilon_multiply_coulwc_1");
            }
            else
            {
                // Solve epsilon * X = sqrt(Vc), then form sqrt(Vc) * (X - sqrt(Vc)).
                global::profiler.start("epsilon_solver_coulwc_1", "epsilon_solver_coulwc",
                                       LIBRPA_VERBOSE_DEBUG);
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
                if (use_gpu_replace_scalapack)
                {
                    coulwc_block_ptr = coul_block_ptr;
                    DEVICE_CHECK(deviceMemcpyAsync(coulwc_block_ptr, coulwc_block.ptr(), coulwc_block.size() * sizeof(std::complex<double>), deviceMemcpyHostToDevice, blacs_h.ddla_handle->stream));
                    DEVICE_CHECK(deviceMemcpyAsync(coul_chi0_block_ptr, coulwc_block_ptr, coulwc_block.size() * sizeof(std::complex<double>), deviceMemcpyDeviceToDevice, blacs_h.ddla_handle->stream));
                }
                else
#endif
                memcpy(coul_chi0_block_ptr, coulwc_block_ptr, coulwc_block.size() * sizeof(std::complex<double>));
                int info = 0;
                if (use_cholesky_gw_wc)
                {
                    LaConnector::pposv('L', 'L', 'N', n_abf, n_abf, chi0_block_ptr, 1, 1,
                                       desc_nabf_nabf_opt, coul_chi0_block_ptr, 1, 1,
                                       desc_nabf_nabf_opt, info);
                }
                else
                {
                    LaConnector::pgesv(n_abf, n_abf, chi0_block_ptr, 1, 1,
                                       desc_nabf_nabf_opt, coul_chi0_block_ptr, 1, 1,
                                       desc_nabf_nabf_opt, info);
                }
                if (info != 0)
                {
                    global::profiler.stop("epsilon_solver_coulwc_1");
                    std::ostringstream oss;
                    oss << "Dielectric " << (use_cholesky_gw_wc ? "Cholesky" : "LU")
                        << " solve failed with info=" << info;
                    throw LIBRPA_RUNTIME_ERROR(oss.str());
                }
                LaConnector::axpy(coulwc_block.size(), {-1.0, 0.0}, coulwc_block_ptr, 1, coul_chi0_block_ptr, 1, blacs_h);
                global::profiler.stop("epsilon_solver_coulwc_1");

                global::profiler.start("epsilon_multiply_coulwc_2",
                                       "Multiply truncated Coulomb", LIBRPA_VERBOSE_DEBUG);
                LaConnector::pgemm('N', 'N', n_abf, n_abf, n_abf, {1.0, 0.0}, coulwc_block_ptr, 1,
                                   1, desc_nabf_nabf_opt, coul_chi0_block_ptr, 1, 1,
                                   desc_nabf_nabf_opt, {0.0, 0.0}, chi0_block_ptr, 1, 1,
                                   desc_nabf_nabf_opt);
                global::profiler.stop("epsilon_multiply_coulwc_2");
            }
            global::profiler.stop("epsilon_to_wc");
#if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
            if (use_gpu_replace_scalapack)
            {
                DEVICE_CHECK(deviceMemcpyAsync(chi0_block.ptr(), chi0_block_ptr,
                                                chi0_block.size() * sizeof(complex<double>),
                                                deviceMemcpyDeviceToHost,
                                                blacs_h.ddla_handle->stream));
                DEVICE_CHECK(deviceStreamSynchronize(blacs_h.ddla_handle->stream));
            }
#endif
            Strict2dBlockMetrics wc_metrics;
            const bool collect_wc_metrics =
                output_2d_finite_q_diagnostics && (ifreq == 0 || is_gamma_point(q));
            if (collect_wc_metrics)
            {
                project_blacs_matrix_in_basis(chi0_block, gamma_coulomb_basis, desc_nabf_nabf_opt,
                                              gamma_n_nonsingular, diagnostic_projection_work,
                                              diagnostic_projection);
                wc_metrics = collect_strict_2d_block_metrics(
                    diagnostic_projection, desc_nabf_nabf_opt, gamma_n_nonsingular, comm_h);
            }
            if (output_2d_finite_q_diagnostics && is_gamma_point(q))
            {
                const double q_weight = chi0.q_weight(q);
                const double gamma_area = strict_2d_physical_gamma_cell_area(
                    rpa_headwing_gamma_cell_volume(chi0.pbc, true));
                if (comm_h.is_root())
                    gamma_wc_diagnostics
                        << ifreq << ',' << freq << ',' << q_weight << ',' << gamma_area << ','
                        << wc_metrics.head.real() << ',' << wc_metrics.head.imag() << ','
                        << wc_metrics.head_body_frobenius << ',' << wc_metrics.body_head_frobenius
                        << ',' << wc_metrics.body_body_frobenius << ','
                        << alpha_wc_metrics.head.real() << ',' << alpha_wc_metrics.head.imag()
                        << ',' << alpha_wc_metrics.head_body_frobenius << ','
                        << alpha_wc_metrics.body_head_frobenius << ','
                        << alpha_wc_metrics.body_body_frobenius << ','
                        << q_weight * wc_metrics.head.real() << ','
                        << q_weight * wc_metrics.head.imag() << ','
                        << q_weight * wc_metrics.head_body_frobenius << ','
                        << q_weight * wc_metrics.body_head_frobenius << ','
                        << q_weight * wc_metrics.body_body_frobenius << '\n';
            }
            if (output_2d_finite_q_diagnostics && ifreq == 0 && !is_gamma_point(q))
            {
                const double q_internal = std::hypot(q.x, q.y);
                const double q_physical = strict_2d_physical_q(q_internal);
                if (!(q_physical > 0.0) || !std::isfinite(q_physical))
                    throw LIBRPA_RUNTIME_ERROR(
                        "strict 2D finite-q diagnostic encountered invalid q");
                const double qhat_x = q.x / q_internal;
                const double qhat_y = q.y / q_internal;
                const auto reference =
                    df_headwing->get_strict_2d_finite_q_reference(ifreq, q.x, q.y);
                const auto wc_head = wc_metrics.head;
                const double q_weight = chi0.q_weight(q);
                const auto p_over_q = finite_q_p_head / q_physical;
                const auto analytic_p_over_q = -reference.epsilon_minus_identity_over_q;
                if (comm_h.is_root())
                    finite_q_diagnostics
                        << iq << ',' << qf.x << ',' << qf.y << ',' << qf.z << ',' << q_physical
                        << ',' << qhat_x << ',' << qhat_y << ',' << q_weight << ','
                        << finite_q_head_overlap << ',' << finite_q_p_head.real() << ','
                        << finite_q_p_head.imag() << ',' << p_over_q.real() << ','
                        << p_over_q.imag() << ',' << finite_q_p_metrics.head_body_frobenius << ','
                        << finite_q_p_metrics.body_head_frobenius << ','
                        << finite_q_p_metrics.body_body_frobenius << ',' << analytic_p_over_q.real()
                        << ',' << analytic_p_over_q.imag() << ',' << wc_head.real() << ','
                        << wc_head.imag() << ',' << wc_metrics.head_body_frobenius << ','
                        << wc_metrics.body_head_frobenius << ',' << wc_metrics.body_body_frobenius
                        << ',' << alpha_wc_metrics.head.real() << ','
                        << alpha_wc_metrics.head.imag() << ','
                        << alpha_wc_metrics.head_body_frobenius << ','
                        << alpha_wc_metrics.body_head_frobenius << ','
                        << alpha_wc_metrics.body_body_frobenius << ','
                        << reference.wc_head_limit.real() << ',' << reference.wc_head_limit.imag()
                        << ',' << (q_physical * wc_head).real() << ','
                        << (q_physical * wc_head).imag() << ',' << q_weight * wc_head.real() << ','
                        << q_weight * wc_head.imag() << ','
                        << q_weight * wc_metrics.head_body_frobenius << ','
                        << q_weight * wc_metrics.body_head_frobenius << ','
                        << q_weight * wc_metrics.body_body_frobenius << '\n';
            }
            // convert back to initial distribution
            ScalapackConnector::pgemr2d_f(n_abf, n_abf, chi0_block.ptr(), 1, 1, desc_nabf_nabf_opt.desc,
                                        temp_block.ptr(), 1, 1, desc_nabf_nabf.desc, blacs_h.ictxt);
            // lib_printf("chi0_block\n%s", str(chi0_block).c_str());
            global::profiler.stop("epsilon_wc_work_q_omega");
            // now temp_block contains the screened Coulomb interaction Wc (i.e. W-V)
            // under the desired array descriptor
            Wc_freq_q[freq][q] = temp_block.copy();

            librpa_int::global::lib_printf_root("Time for Wc(i_q=%d, i_omega=%d) (seconds, Wall/CPU): %f %f\n",
                    iq + 1, ifreq + 1,
                    global::profiler.get_wall_time_last("epsilon_wc_work_q_omega"),
                    global::profiler.get_cpu_time_last("epsilon_wc_work_q_omega"));
        }
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        if(use_gpu_replace_scalapack)
        {
            if (gamma_head_eigen_device != nullptr)
                DEVICE_CHECK(deviceFreeAsync(gamma_head_eigen_device,
                                             blacs_h.ddla_handle->stream));
            if (gamma_head_h_device != nullptr)
                DEVICE_CHECK(deviceFreeAsync(gamma_head_h_device,
                                             blacs_h.ddla_handle->stream));
            DEVICE_CHECK(deviceFreeAsync(coul_chi0_block_ptr, blacs_h.ddla_handle->stream));
            DEVICE_CHECK(deviceFreeAsync(coul_block_ptr, blacs_h.ddla_handle->stream));
        }

#endif
    }
#else
    throw std::logic_error("need compilation with LibRI");
#endif
    #if defined(LIBRPA_USE_HIP) || defined(LIBRPA_USE_CUDA)
    if (use_gpu_replace_scalapack)
    {
        DEVICE_CHECK(deviceFreeAsync(chi0_block_ptr, blacs_h.ddla_handle->stream));
    }
    #endif
    global::profiler.stop("compute_Wc_freq_q_work");
    librpa_int::global::lib_printf_root("Time for Wc computation (seconds, Wall/CPU): %f %f\n",
            global::profiler.get_wall_time_last("compute_Wc_freq_q_work"),
            global::profiler.get_cpu_time_last("compute_Wc_freq_q_work"));

    return Wc_freq_q;
}

void unfold_Wc_freq_q_blacs(
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
    const vector<Vector3_Order<double>> &qlist,
    const BlacsCtxtHandler &blacs_h,
    const ArrayDesc &desc_small,
    const ArrayDesc &desc_full)
{
    using global::profiler;

    const int n_small = desc_small.m();
    const int n_full = desc_full.m();
    if (desc_small.n() != n_small || desc_full.n() != n_full)
        throw LIBRPA_RUNTIME_ERROR("Cannot unfold Wc with non-square descriptors");

    ArrayDesc desc_sl(blacs_h);
    desc_sl.init(n_small, n_full, desc_small.mb(), desc_full.nb(), 0, 0);
    auto u_block = init_local_mat<complex<double>>(desc_sl, MAJOR::COL);
    auto Wc_u = init_local_mat<complex<double>>(desc_sl, MAJOR::COL);
    auto Wc_full_block = init_local_mat<complex<double>>(desc_full, MAJOR::COL);

    for (auto &[freq, q_Wc] : Wc_freq_q)
    {
        for (const auto &q : qlist)
        {
            const auto q_iter = q_Wc.find(q);
            if (q_iter == q_Wc.end()) continue;
            const auto sinvS_iter = sinvS.find(q);
            if (sinvS_iter == sinvS.end())
                throw LIBRPA_RUNTIME_ERROR("Cannot unfold Wc: missing shrink_sinvS q-point");
            const auto &U = sinvS_iter->second;
            if (U.nr != n_small || U.nc != n_full)
                throw LIBRPA_RUNTIME_ERROR("Cannot unfold Wc: shrink_sinvS dimensions do not match descriptors");

            u_block.zero_out();
            Wc_u.zero_out();
            Wc_full_block.zero_out();
            for (int ir = 0; ir != U.nr; ++ir)
            {
                const int ilo = desc_sl.indx_g2l_r(ir);
                if (ilo < 0) continue;
                for (int ic = 0; ic != U.nc; ++ic)
                {
                    const int jlo = desc_sl.indx_g2l_c(ic);
                    if (jlo < 0) continue;
                    u_block(ilo, jlo) = U(ir, ic);
                }
            }

            profiler.start("unfold_Wc_q_1");
            ScalapackConnector::pgemm_f('N', 'N', n_small, n_full, n_small,
                                        1.0, q_iter->second.ptr(), 1, 1,
                                        desc_small.desc, u_block.ptr(), 1, 1,
                                        desc_sl.desc, 0.0, Wc_u.ptr(), 1, 1,
                                        desc_sl.desc);
            profiler.stop("unfold_Wc_q_1");
            profiler.start("unfold_Wc_q_2");
            ScalapackConnector::pgemm_f('C', 'N', n_full, n_full, n_small,
                                        1.0, u_block.ptr(), 1, 1,
                                        desc_sl.desc, Wc_u.ptr(), 1, 1,
                                        desc_sl.desc, 0.0,
                                        Wc_full_block.ptr(), 1, 1,
                                        desc_full.desc);
            profiler.stop("unfold_Wc_q_2");
            q_iter->second = Wc_full_block.copy();
        }
    }
}

static void fill_blacs_local_from_dense(Matz& local_matrix,
                                        const ArrayDesc& desc,
                                        const ComplexMatrix& dense_matrix)
{
    if (dense_matrix.nr != desc.m() || dense_matrix.nc != desc.n())
    {
        throw LIBRPA_RUNTIME_ERROR("Dense symmetry rotation matrix dimension mismatch");
    }
    local_matrix.zero_out();
    for (int i_lo = 0; i_lo != desc.m_loc(); ++i_lo)
    {
        const int i_glo = desc.indx_l2g_r(i_lo);
        for (int j_lo = 0; j_lo != desc.n_loc(); ++j_lo)
        {
            const int j_glo = desc.indx_l2g_c(j_lo);
            local_matrix(i_lo, j_lo) = dense_matrix(i_glo, j_glo);
        }
    }
}

static Matz rotate_symmetry_blacs_wq(const Matz& Wq_rep,
                                     const ArrayDesc& desc,
                                     const ComplexMatrix& transform_dense,
                                     const bool use_time_reversal)
{
    if (desc.m() != desc.n())
    {
        throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry restore requires a square descriptor");
    }
    const int n = desc.m();
    auto transform = init_local_mat<cplxdb>(desc, MAJOR::COL);
    auto tmp = init_local_mat<cplxdb>(desc, MAJOR::COL);
    auto Wq_member = init_local_mat<cplxdb>(desc, MAJOR::COL);
    fill_blacs_local_from_dense(transform, desc, transform_dense);

    if (use_time_reversal)
    {
        auto transform_conj = transform.copy();
        transform_conj.conj();
        auto Wq_conj = Wq_rep.copy();
        Wq_conj.conj();
        ScalapackConnector::pgemm_f('N', 'N', n, n, n, 1.0,
                                    transform_conj.ptr(), 1, 1, desc.desc,
                                    Wq_conj.ptr(), 1, 1, desc.desc,
                                    0.0, tmp.ptr(), 1, 1, desc.desc);
        ScalapackConnector::pgemm_f('N', 'T', n, n, n, 1.0,
                                    tmp.ptr(), 1, 1, desc.desc,
                                    transform.ptr(), 1, 1, desc.desc,
                                    0.0, Wq_member.ptr(), 1, 1, desc.desc);
    }
    else
    {
        ScalapackConnector::pgemm_f('N', 'N', n, n, n, 1.0,
                                    transform.ptr(), 1, 1, desc.desc,
                                    Wq_rep.ptr(), 1, 1, desc.desc,
                                    0.0, tmp.ptr(), 1, 1, desc.desc);
        ScalapackConnector::pgemm_f('N', 'C', n, n, n, 1.0,
                                    tmp.ptr(), 1, 1, desc.desc,
                                    transform.ptr(), 1, 1, desc.desc,
                                    0.0, Wq_member.ptr(), 1, 1, desc.desc);
    }
    return Wq_member;
}

static Matz restore_symmetry_blacs_wq_to_star_source(const Matz& Wq_source,
                                                     const ArrayDesc& desc,
                                                     const ComplexMatrix& source_transform_dense,
                                                     const bool source_uses_time_reversal)
{
    if (desc.m() != desc.n())
    {
        throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry source restore requires a square descriptor");
    }
    const int n = desc.m();
    auto source_transform = init_local_mat<cplxdb>(desc, MAJOR::COL);
    auto tmp = init_local_mat<cplxdb>(desc, MAJOR::COL);
    auto Wq_star = init_local_mat<cplxdb>(desc, MAJOR::COL);
    fill_blacs_local_from_dense(source_transform, desc, source_transform_dense);

    auto Wq_effective = Wq_source.copy();
    if (source_uses_time_reversal)
    {
        Wq_effective.conj();
    }

    ScalapackConnector::pgemm_f('C', 'N', n, n, n, 1.0,
                                source_transform.ptr(), 1, 1, desc.desc,
                                Wq_effective.ptr(), 1, 1, desc.desc,
                                0.0, tmp.ptr(), 1, 1, desc.desc);
    ScalapackConnector::pgemm_f('N', 'N', n, n, n, 1.0,
                                tmp.ptr(), 1, 1, desc.desc,
                                source_transform.ptr(), 1, 1, desc.desc,
                                0.0, Wq_star.ptr(), 1, 1, desc.desc);
    return Wq_star;
}

static std::map<Vector3_Order<double>, Matz> restore_symmetry_dense_wq_map(
    const std::map<Vector3_Order<double>, Matz>& Wq_rep_map,
    const PeriodicBoundaryData& pbc,
    const SymmetryQPointView& qpoint_view,
    const SymmetryContext& symmetry_context,
    const AtomicBasis& atbasis_Wc,
    const ArrayDesc& ad_Wc)
{
    const auto atom_nabf = build_atom_nabf_map(atbasis_Wc);
    const auto abf_layouts =
        atbasis_Wc.build_species_basis_layouts(symmetry_context.atom_to_type);
    if (!symmetry_species_layouts_match_atom_counts(
            abf_layouts, symmetry_context.atom_to_type, atom_nabf))
    {
        throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry restore ABF layout mismatch");
    }

    std::map<Vector3_Order<double>, Matz> Wq_full_map;
    for (const auto& q_rep : qpoint_view.representatives)
    {
        const auto Wq_iter = find_matching_symmetry_qpoint(Wq_rep_map, q_rep);
        if (Wq_iter == Wq_rep_map.end())
        {
            throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry restore is missing a representative q");
        }
        const Vector3_Order<double> q_rep_frac =
            restrict_fractional_coordinate(Vector3_Order<double>{pbc.latvec * q_rep});
        const auto& star =
            find_symmetry_kstar_for_kpoint(symmetry_context.kstars, q_rep_frac,
                                           "dense Wc q-star restore");
        const auto& members = qpoint_view.members.at(q_rep);
        auto find_star_member_index = [](const SymmetryKStar& star,
                                         const Vector3_Order<double>& q_frac)
        {
            for (std::size_t imember = 0; imember != star.members.size(); ++imember)
            {
                if (same_fractional_kpoint(star.members[imember].k_bz, q_frac, 1e-5))
                {
                    return imember;
                }
            }
            return star.members.size();
        };
        const auto source_member_index = find_star_member_index(star, q_rep_frac);
        if (source_member_index == star.members.size())
        {
            throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry restore could not find the representative in its q-star");
        }
        const auto& source_member = star.members[source_member_index];
        const auto source_transform = build_symmetry_kspace_operator_transform_matrix(
            symmetry_context, abf_layouts, source_member, atom_nabf, star.k_ibz,
            source_member.time_reversal, &q_rep_frac);
        const auto Wq_star_source = restore_symmetry_blacs_wq_to_star_source(
            Wq_iter->second, ad_Wc, source_transform, source_member.time_reversal);

        for (std::size_t imember = 0; imember != members.size(); ++imember)
        {
            const auto& q_member = members[imember];
            const Vector3_Order<double> q_member_frac =
                restrict_fractional_coordinate(Vector3_Order<double>{pbc.latvec * q_member});
            const Vector3_Order<double> q_member_internal{q_member_frac * pbc.G};
            const auto star_member_index = find_star_member_index(star, q_member_frac);
            if (star_member_index == star.members.size())
            {
                throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry restore could not match a q-star member");
            }
            const auto& member = star.members[star_member_index];
            if (star_member_index == source_member_index)
            {
                Wq_full_map[q_member_internal] = Wq_iter->second.copy();
                continue;
            }

            const auto transform = build_symmetry_kspace_operator_transform_matrix(
                symmetry_context, abf_layouts, member, atom_nabf, star.k_ibz,
                member.time_reversal, &q_member_frac);
            Wq_full_map[q_member_internal] =
                rotate_symmetry_blacs_wq(Wq_star_source, ad_Wc, transform,
                                         member.time_reversal);
        }
    }
    return Wq_full_map;
}

using dense_wc_freq_r_complex_t = std::map<double, std::map<Vector3_Order<int>, Matz>>;
using dense_wc_freq_r_real_t = std::map<double, std::map<Vector3_Order<int>, Matd>>;

struct DenseWcRealResidual
{
    double max_abs_real = 0.0;
    double max_abs_imag = 0.0;
    double worst_freq = 0.0;
    Vector3_Order<int> worst_R{0, 0, 0};
    std::size_t worst_local_index = 0;
    bool has_nonfinite = false;
};

static void validate_dense_wc_real_residual(const MpiCommHandler &comm_h,
                                            const DenseWcRealResidual &local)
{
    constexpr double abs_tol = 1e-12;
    constexpr double rel_tol = 1e-10;

    const double local_maxima[2] = {local.max_abs_real, local.max_abs_imag};
    double global_maxima[2] = {0.0, 0.0};
    MPI_Allreduce(local_maxima, global_maxima, 2, MPI_DOUBLE, MPI_MAX, comm_h.comm);

    int nonfinite_local = local.has_nonfinite ? 1 : 0;
    int nonfinite_global = 0;
    MPI_Allreduce(&nonfinite_local, &nonfinite_global, 1, MPI_INT, MPI_MAX, comm_h.comm);

    struct
    {
        double value;
        int rank;
    } local_maxloc{local.max_abs_imag, comm_h.myid}, global_maxloc{0.0, 0};
    MPI_Allreduce(&local_maxloc, &global_maxloc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm_h.comm);

    double worst_freq = local.worst_freq;
    int worst_R[3] = {local.worst_R.x, local.worst_R.y, local.worst_R.z};
    unsigned long long worst_local_index = local.worst_local_index;
    MPI_Bcast(&worst_freq, 1, MPI_DOUBLE, global_maxloc.rank, comm_h.comm);
    MPI_Bcast(worst_R, 3, MPI_INT, global_maxloc.rank, comm_h.comm);
    MPI_Bcast(&worst_local_index, 1, MPI_UNSIGNED_LONG_LONG, global_maxloc.rank, comm_h.comm);

    const double allowed = abs_tol + rel_tol * global_maxima[0];
    global::lib_printf_root(
        "Dense Wc real projection residual: max|Re|=%15.8e max|Im|=%15.8e "
        "allowed=%15.8e worst_rank=%d freq=%15.8e R=(%d,%d,%d) local_index=%llu\n",
        global_maxima[0], global_maxima[1], allowed, global_maxloc.rank, worst_freq,
        worst_R[0], worst_R[1], worst_R[2], worst_local_index);

    if (nonfinite_global != 0)
        throw LIBRPA_RUNTIME_ERROR("Dense Wc real projection found a non-finite value");
    if (global_maxima[1] > allowed)
        global::lib_printf_root(LIBRPA_VERBOSE_WARN,
            "Warning: Dense Wc real projection discarding max|Im|=%15.8e, "
            "which exceeds the development tolerance %15.8e\n",
            global_maxima[1], allowed);
}

static void FT_Wc_freq_q_into(
    const MpiCommHandler &comm_h,
    map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc, bool remove_freq_q,
    const SymmetryQPointView *qpoint_view,
    const SymmetryContext *symmetry_context,
    const AtomicBasis *atbasis_Wc,
    const ArrayDesc *ad_Wc,
    dense_wc_freq_r_complex_t *Wc_freq_R_complex,
    dense_wc_freq_r_real_t *Wc_freq_R_real,
    DenseWcRealResidual *real_residual)
{
    using librpa_int::global::ofs_myid;
    using librpa_int::global::lib_printf;

    const bool use_real_output = Wc_freq_R_real != nullptr;
    if ((Wc_freq_R_complex == nullptr) == (Wc_freq_R_real == nullptr))
        throw LIBRPA_RUNTIME_ERROR("Dense Wc Fourier transform needs exactly one destination");
    if (use_real_output && real_residual == nullptr)
        throw LIBRPA_RUNTIME_ERROR("Dense real Wc Fourier transform needs residual storage");

    if (comm_h.is_root())
        lib_printf("Converting Wc q,w -> R,t\n");
    comm_h.barrier();
    const auto n_k_points = pbc.get_n_cells_bvk();
    const auto &Rlist = pbc.Rlist;

    // quick return if empty
    if (Wc_freq_q.size() == 0) return;
    // For single k-point (Gamma only), there is no need to transform: just remap and return
    if (n_k_points == 1)
    {
        ofs_myid << "Single k-point, remapping instead of explicit transform" << std::endl;
        for (auto it_freq = Wc_freq_q.begin(); it_freq != Wc_freq_q.end(); it_freq++)
        {
            const auto &freq = it_freq->first;
            auto &map_q_mat = it_freq->second;
            assert(map_q_mat.size() < 2);
            for (auto it_q = map_q_mat.begin(); it_q != map_q_mat.end(); )
            {
                const Vector3_Order<int> center{0, 0, 0};
                if (use_real_output)
                {
                    const auto &source = it_q->second;
                    Matd target(source.nr(), source.nc(), source.major());
                    for (std::size_t i = 0; i != source.size(); ++i)
                    {
                        const auto value = source.ptr()[i];
                        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                            real_residual->has_nonfinite = true;
                        else
                        {
                            real_residual->max_abs_real =
                                std::max(real_residual->max_abs_real, std::abs(value.real()));
                            const double abs_imag = std::abs(value.imag());
                            if (abs_imag > real_residual->max_abs_imag)
                            {
                                real_residual->max_abs_imag = abs_imag;
                                real_residual->worst_freq = freq;
                                real_residual->worst_R = center;
                                real_residual->worst_local_index = i;
                            }
                        }
                        target.ptr()[i] = value.real();
                    }
                    (*Wc_freq_R_real)[freq][center] = std::move(target);
                }
                else
                {
                    (*Wc_freq_R_complex)[freq][center] = it_q->second;
                }
                if (remove_freq_q)
                {
                    it_q = map_q_mat.erase(it_q);
                }
                else
                {
                    it_q++;
                }
            }
        }
        if (remove_freq_q) Wc_freq_q.clear();
        return;
    }

    // check major and size of the matrix to transform
    MAJOR major_orig = MAJOR::AUTO;
    int nr = 0, nc = 0;
    size_t size = 0;
    for (const auto &[freq, map_q_mat] : Wc_freq_q)
    {
        if (map_q_mat.cbegin() != map_q_mat.cend())
        {
            const auto &mat = map_q_mat.begin()->second;
            nr = mat.nr();
            nc = mat.nc();
            major_orig = mat.major();
            size = mat.size();
            break;
        }
    }
    if (major_orig == MAJOR::AUTO)
        throw LIBRPA_RUNTIME_ERROR("Dense Wc Fourier transform input has no matrices");

    const auto &latvec = pbc.latvec;
    const auto &klist_full = pbc.klist_full;
    if (static_cast<int>(klist_full.size()) != n_k_points)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "full-BZ k-point list size is inconsistent with the BvK grid");
    }
    const bool use_full_crystal_restore =
        qpoint_view != nullptr
        && qpoint_view->restore_mode == SymmetryQPointRestoreMode::FULL_CRYSTAL;
    if (use_full_crystal_restore
        && (symmetry_context == nullptr || atbasis_Wc == nullptr || ad_Wc == nullptr))
    {
        throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry restore is missing symmetry inputs");
    }
    // initialize conversion matrix.
    // major does not have to conform the original one
    Matz coeff_k2r(n_k_points, n_k_points, MAJOR::COL);
    for (int ik = 0; ik < n_k_points; ik++)
    {
        for (int ir = 0; ir < n_k_points; ir++)
        {
            const auto &R = Rlist[ir];
            const auto ang = - klist_full[ik] * (R * latvec) * TWO_PI;
            coeff_k2r(ik, ir) = complex<double>(cos(ang), sin(ang));
        }
    }
    coeff_k2r *= 1.0 / n_k_points;
    // if (librpa_int::global::myid_global == 0) cout << coeff_k2r << endl;

    // Divide into batches to limit the memory consumption of temporary matrices for Fourier transform
    // Maximal 1GB per process for HPC usage, about 500 * 500 elements with 216 k-points (6x6x6)
    const auto maxbytes_tmpmat = gbytes;
    // A valid BLACS rank can own no local matrix elements when the matrix is smaller than the
    // process grid. Keep the batch divisor positive; zero local data then gives zero batches.
    const auto size_batch_max = std::max<std::size_t>(
        1, std::min(maxbytes_tmpmat / sizeof(cplxdb) / n_k_points, size));
    const auto n_data_batches = ceil_div(size, size_batch_max);
    const auto n_r_batch_max = std::min(maxbytes_tmpmat / sizeof(cplxdb) / size_batch_max, as_size(n_k_points));
    const auto n_r_batches = ceil_div(as_size(n_k_points), n_r_batch_max);

    global::ofs_myid << "size_batch_max/n_r_batch_max " << size_batch_max << " " << n_r_batch_max << std::endl;
    global::ofs_myid << "n_data_batches/n_r_batches " << n_data_batches << " " << n_r_batches << std::endl;

    std::vector<cplxdb> kmat(size_batch_max * n_k_points);
    std::vector<cplxdb> rmat(size_batch_max * n_r_batch_max);

    for (auto it_freq = Wc_freq_q.begin(); it_freq != Wc_freq_q.end();)
    {
        const auto freq = it_freq->first;
        const bool use_full_crystal_restore_this_freq =
            use_full_crystal_restore
            && it_freq->second.size() == qpoint_view->representatives.size();
        std::map<Vector3_Order<double>, Matz> restored_map_q_mat;
        if (use_full_crystal_restore_this_freq)
        {
            restored_map_q_mat = restore_symmetry_dense_wq_map(
                it_freq->second, pbc, *qpoint_view, *symmetry_context, *atbasis_Wc, *ad_Wc);
            for (const auto& q_full : klist_full)
            {
                if (find_matching_internal_qpoint(restored_map_q_mat, pbc, q_full)
                    == restored_map_q_mat.end())
                {
                    throw LIBRPA_RUNTIME_ERROR("Dense Wc symmetry restore did not cover the full q grid");
                }
            }
        }
        const auto& map_q_mat =
            use_full_crystal_restore_this_freq ? restored_map_q_mat : it_freq->second;
        // initialize
        for (const auto &R: Rlist)
        {
            if (use_real_output)
                (*Wc_freq_R_real)[freq][R] = Matd(nr, nc, major_orig);
            else
                (*Wc_freq_R_complex)[freq][R] = Matz(nr, nc, major_orig);
        }
        for (size_t i_data_batch = 0; i_data_batch < n_data_batches; i_data_batch++)
        {
            const size_t displ_data = i_data_batch * size_batch_max;
            const size_t size_this_batch = std::min(size_batch_max, size - displ_data);
            for (size_t i_r_batch = 0; i_r_batch < n_r_batches; i_r_batch++)
            {
                const size_t displ_r = i_r_batch * n_r_batch_max;
                const size_t n_r_this_batch = std::min(n_r_batch_max, n_k_points - displ_r);
                // Copy raw data
                // for (const auto &[q, mat] : map_q_mat)
                // for (auto it = map_q_mat.begin(); it != map_q_mat.end(); it++)
                std::fill(kmat.begin(), kmat.end(), cplxdb{0.0, 0.0});
                if (use_full_crystal_restore_this_freq)
                {
                    #pragma omp parallel for schedule(dynamic)
                    for (int iq = 0; iq < n_k_points; iq++)
                    {
                        const auto& q = klist_full[static_cast<std::size_t>(iq)];
                        const auto it = find_matching_internal_qpoint(map_q_mat, pbc, q);
                        if (it == map_q_mat.end())
                        {
                            continue;
                        }
                        const auto& mat = it->second;
                        memcpy(kmat.data() + static_cast<std::size_t>(iq) * size_batch_max,
                               mat.ptr() + displ_data, size_this_batch * sizeof(cplxdb));
                    }
                }
                else
                {
                    #pragma omp parallel for schedule(dynamic)
                    for (size_t iq = 0; iq < pbc.klist_coul.size(); iq++)
                    {
                        const auto &q = pbc.klist_coul[iq];
                        auto it = map_q_mat.find(q);
                        if (it == map_q_mat.end()) continue;
                        const auto &mat = it->second;
                        for (const auto &q_fbz: pbc.map_irk_ks.at(q))
                        {
                            const auto iq = pbc.get_k_index_full(q_fbz);
                            if (q_fbz == q)
                            {
                                memcpy(kmat.data() + iq * size_batch_max,
                                       mat.ptr() + displ_data, size_this_batch * sizeof(cplxdb));
                            }
                            else // assume q_fbz = -q: mat(-q) = conjgate(mat(q))
                            {
                                Matz tmp(size_this_batch, 1, mat.ptr() + displ_data);
                                memcpy(kmat.data() + iq * size_batch_max,
                                       tmp.conj().ptr(),
                                       size_this_batch * sizeof(cplxdb));
                            }
                        }
                    }
                }
                // Transform
                LapackConnector::gemm_f('N', 'N', size_this_batch, n_r_this_batch, n_k_points,
                                        1.0, kmat.data(), size_batch_max, coeff_k2r.ptr() + n_k_points * displ_r, n_k_points,
                                        0.0, rmat.data(), size_batch_max);
                // Add to the mapping
                if (use_real_output)
                {
                    std::vector<double *> dest_ptrs(n_r_this_batch);
                    std::vector<double> batch_max_real(n_r_this_batch, 0.0);
                    std::vector<double> batch_max_imag(n_r_this_batch, 0.0);
                    std::vector<std::size_t> batch_worst_index(n_r_this_batch, 0);
                    std::vector<int> batch_has_nonfinite(n_r_this_batch, 0);
                    for (std::size_t ir_this = 0; ir_this != n_r_this_batch; ++ir_this)
                    {
                        const auto ir = displ_r + ir_this;
                        dest_ptrs[ir_this] =
                            Wc_freq_R_real->at(freq).at(Rlist[ir]).ptr() + displ_data;
                    }

                    #pragma omp parallel for schedule(static)
                    for (size_t ir_this = 0; ir_this < n_r_this_batch; ir_this++)
                    {
                        const auto *source = rmat.data() + size_batch_max * ir_this;
                        auto *target = dest_ptrs[ir_this];
                        for (std::size_t i = 0; i != size_this_batch; ++i)
                        {
                            const auto value = source[i];
                            target[i] = value.real();
                            if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
                            {
                                batch_has_nonfinite[ir_this] = 1;
                                continue;
                            }
                            batch_max_real[ir_this] =
                                std::max(batch_max_real[ir_this], std::abs(value.real()));
                            const double abs_imag = std::abs(value.imag());
                            if (abs_imag > batch_max_imag[ir_this])
                            {
                                batch_max_imag[ir_this] = abs_imag;
                                batch_worst_index[ir_this] = i;
                            }
                        }
                    }

                    for (std::size_t ir_this = 0; ir_this != n_r_this_batch; ++ir_this)
                    {
                        real_residual->max_abs_real =
                            std::max(real_residual->max_abs_real, batch_max_real[ir_this]);
                        real_residual->has_nonfinite =
                            real_residual->has_nonfinite || batch_has_nonfinite[ir_this] != 0;
                        if (batch_max_imag[ir_this] > real_residual->max_abs_imag)
                        {
                            const auto ir = displ_r + ir_this;
                            real_residual->max_abs_imag = batch_max_imag[ir_this];
                            real_residual->worst_freq = freq;
                            real_residual->worst_R = Rlist[ir];
                            real_residual->worst_local_index =
                                displ_data + batch_worst_index[ir_this];
                        }
                    }
                }
                else
                {
                    #pragma omp parallel for schedule(dynamic)
                    for (size_t ir_this = 0; ir_this < n_r_this_batch; ir_this++)
                    {
                        auto ir = displ_r + ir_this;
                        memcpy((*Wc_freq_R_complex)[freq][Rlist[ir]].ptr() + displ_data,
                               rmat.data() + size_batch_max * ir_this,
                               sizeof(cplxdb) * size_this_batch);
                    }
                }
            }
        }
        if (remove_freq_q)
        {
            // Remove data for this frequency
            it_freq = Wc_freq_q.erase(it_freq);
        }
        else
        {
            it_freq++;
        }
    }

    if (comm_h.is_root())
    {
        lib_printf("Done converting Wc(q,w) -> Wc(R,w)\n");
    }
    comm_h.barrier();
}

std::map<double, std::map<Vector3_Order<int>, Matz>> FT_Wc_freq_q(
    const MpiCommHandler &comm_h, map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc, bool remove_freq_q,
    const SymmetryQPointView *qpoint_view,
    const SymmetryContext *symmetry_context,
    const AtomicBasis *atbasis_Wc,
    const ArrayDesc *ad_Wc)
{
    dense_wc_freq_r_complex_t Wc_freq_R;
    FT_Wc_freq_q_into(comm_h, Wc_freq_q, pbc, remove_freq_q, qpoint_view,
                      symmetry_context, atbasis_Wc, ad_Wc, &Wc_freq_R, nullptr, nullptr);
    return Wc_freq_R;
}

static dense_wc_freq_r_real_t FT_Wc_freq_q_real(
    const MpiCommHandler &comm_h,
    map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc,
    const SymmetryQPointView *qpoint_view,
    const SymmetryContext *symmetry_context,
    const AtomicBasis *atbasis_Wc,
    const ArrayDesc *ad_Wc)
{
    dense_wc_freq_r_real_t Wc_freq_R;
    DenseWcRealResidual residual;
    FT_Wc_freq_q_into(comm_h, Wc_freq_q, pbc, true, qpoint_view, symmetry_context,
                      atbasis_Wc, ad_Wc, nullptr, &Wc_freq_R, &residual);
    validate_dense_wc_real_residual(comm_h, residual);
    return Wc_freq_R;
}

std::map<double, std::map<Vector3_Order<int>, Matz>> CT_FT_Wc_freq_q(
    const MpiCommHandler &comm_h,
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc, const TFGrids &tfg, bool remove_freq_q,
    bool output_wc_rf, int ifreq_output_wc_start, int ifreq_output_wc_end,
    bool output_wc_rf_atom_pair, const std::string &output_dir,
    const ArrayDesc *ad_Wc, const AtomicBasis *atbasis_Wc,
    const SymmetryQPointView *qpoint_view,
    const SymmetryContext *symmetry_context)
{
    using std::endl;

    std::map<double, std::map<Vector3_Order<int>, Matz>> Wc_tau_R;
    // quick return if empty
    if (Wc_freq_q.size() == 0) return Wc_tau_R;
    const int n_k_points = pbc.get_n_cells_bvk();
    const auto &Rlist = pbc.Rlist;

    if (!tfg.has_time_grids())
        throw LIBRPA_RUNTIME_ERROR("TFGrids object does not have time grids");
    const auto n_freq = tfg.get_n_grids();

    // check major and size of the matrix to transform
    MAJOR major_orig = MAJOR::AUTO;
    int nr, nc;
    size_t size;
    for (const auto &[freq, map_q_mat] : Wc_freq_q)
    {
        const auto &it = map_q_mat.cbegin();
        if (it != map_q_mat.cend())
        {
            const auto &mat = it->second;
            nr = mat.nr();
            nc = mat.nc();
            major_orig = mat.major();
            size = mat.size();
            break;
        }
    }
    assert(major_orig != MAJOR::AUTO);

    librpa_int::global::lib_printf_root("Converting Wc(q,w) -> W(R,t)\n");
    global::ofs_myid << "Converting Wc(q,w) -> W(R,t)" << std::endl;
    global::ofs_myid << "major_orig_row ? " << std::boolalpha << (major_orig == MAJOR::ROW) << std::endl;
    comm_h.barrier();

    // Perform Fourier transform first, then inverse cosine transform
    auto Wc_freq_R = FT_Wc_freq_q(comm_h, Wc_freq_q, pbc, remove_freq_q,
                                  qpoint_view, symmetry_context, atbasis_Wc, ad_Wc);
    if (output_wc_rf)
    {
        if (ad_Wc == nullptr)
            throw LIBRPA_RUNTIME_ERROR("output_wc_rf needs a Wc matrix descriptor");
        if (output_wc_rf_atom_pair && atbasis_Wc == nullptr)
            throw LIBRPA_RUNTIME_ERROR("output_wc_rf_atom_pair needs a Wc atomic basis");
        if (ifreq_output_wc_start < 0)
            throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_start must be non-negative");
        if (ifreq_output_wc_end >= 0 &&
            ifreq_output_wc_end <= ifreq_output_wc_start)
            throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_end must be negative or greater than ifreq_output_wc_start");
        if (ifreq_output_wc_start >= as_int(n_freq))
            throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_start is outside the Wc frequency grid");

        const int ifreq_end = ifreq_output_wc_end < 0 ? n_freq : ifreq_output_wc_end;
        if (ifreq_end > as_int(n_freq))
            throw LIBRPA_RUNTIME_ERROR("ifreq_output_wc_end is outside the Wc frequency grid");

        IndexScheduler sched;
        if (output_wc_rf_atom_pair)
        {
            const auto map_atpairs_balanced =
                get_balanced_ap_distribution_for_consec_descriptor(*atbasis_Wc, *atbasis_Wc, *ad_Wc);
            sched.init(map_atpairs_balanced, *atbasis_Wc, *atbasis_Wc, *ad_Wc,
                       major_orig == MAJOR::ROW);
        }

        global::profiler.start("write_Wc_freq_R", "Export Wc(R,w) to file");
        for (int ifreq = ifreq_output_wc_start; ifreq != ifreq_end; ++ifreq)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            auto freq_iter = Wc_freq_R.find(freq);
            if (freq_iter == Wc_freq_R.end()) continue;
            for (const auto &[R, Wc] : freq_iter->second)
            {
                const auto iR = pbc.get_R_index(R);
                if (output_wc_rf_atom_pair)
                {
                    const auto pair_mat = get_ap_map_from_blacs_dist_scheduler(
                        Wc, sched, *atbasis_Wc, *atbasis_Wc, *ad_Wc);
                    for (const auto &[IJ, Wc_block] : pair_mat)
                    {
                        std::ostringstream ss;
                        std::string info = "Wc at iR " + std::to_string(iR) + " ( " + std::to_string(R.x) +
                                           " " + std::to_string(R.y) + " " + std::to_string(R.z) +
                                           " ) and ifreq " + std::to_string(ifreq) + " ( " +
                                           std::to_string(freq) + " a.u. )";
                        ss << path_as_directory(output_dir)
                           << "Wc_Mu_" << IJ.first << "_Nu_" << IJ.second
                           << "_iR_" << iR << "_ifreq_" << ifreq << ".mtx";
                        print_matrix_mm_file(Wc_block, ss.str(), info, 1e-10);
                    }
                    continue;
                }

                std::stringstream ss;
                std::string info = "Wc at iR " + std::to_string(iR) + " ( " + std::to_string(R.x) +
                                   " " + std::to_string(R.y) + " " + std::to_string(R.z) +
                                   " ) and ifreq " + std::to_string(ifreq) + " ( " +
                                   std::to_string(freq) + " a.u. )";
                ss << path_as_directory(output_dir)
                   << "Wc_iR_" << iR << "_ifreq_" << ifreq << ".mtx";
                print_matrix_mm_file_parallel(ss.str(), Wc, *ad_Wc, info, 1e-10);
            }
        }
        global::profiler.stop("write_Wc_freq_R");
    }
    // if (Params::debug)
    // {
    //     const auto &Wc = Wc_freq_R.at(tfg.get_freq_nodes()[0]).at(Rlist[0]);
    //     std::stringstream ss;
    //     ss << Params::output_dir << "Wc_freq_R"
    //         << "_itau_" << std::setfill('0') << std::setw(5) << 0
    //         << "_iR_" << std::setfill('0') << std::setw(5) << 0 << ".csc";
    //     librpa_int::utils::write_matrix_elsi_csc_parallel(ss.str(), Wc, librpa_int::envs::array_desc_abf_global);
    // }
    // Switch to [R][freq] mapping to allow release the intermediate data on-the-fly during inverse cosine transformation
    std::map<Vector3_Order<int>, std::map<double, Matz>> Wc_R_freq;
    for (auto &[freq, Wc_R]: Wc_freq_R)
    {
        for (auto &[R, Wc]: Wc_R)
        {
            Wc_R_freq[R].emplace(freq, std::move(Wc));
        }
    }
    Wc_freq_R.clear();

    // initialize inverse consine tranform matrix.
    Matz coeff_f2t(n_freq, n_freq, MAJOR::COL);
    for (size_t itau = 0; itau < n_freq; itau++)
    {
        for (size_t ifreq = 0; ifreq < n_freq; ifreq++)
        {
            coeff_f2t(ifreq, itau) = {tfg.get_costrans_f2t()(itau, ifreq), 0.0};
        }
    }
    // global::ofs_myid << coeff_f2t << std::endl;

    // To balance performance and memory consumption, we divide basis x Rlist into batches as row indices.
    // Maximal 1GB per process for HPC usage ~ 4 R-vector with 16 frequency points for 1000x1000 matrix.
    const auto maxbytes_tmpmat = gbytes;
    size_t size_batch_max, n_data_batches;
    size_t n_r_batch_max, n_r_batches;
    const auto n_r_batches_with_whole_size = maxbytes_tmpmat / sizeof(cplxdb) / size / n_freq;
    if (n_r_batches_with_whole_size < 1)
    {
        // large basis case, the transform must be performed for each slice of the matrix at one R-vector.
        n_r_batch_max = 1;
        n_r_batches = n_k_points;
        size_batch_max = std::min(maxbytes_tmpmat / sizeof(cplxdb) / n_freq, size);
        n_data_batches = ceil_div(size, size_batch_max);
    }
    else
    {
        // Whole matrix for at least one R-vector can be transformed at once.
        size_batch_max = size;
        n_data_batches = 1;
        n_r_batch_max = std::min(maxbytes_tmpmat / sizeof(cplxdb) / n_freq / size, as_size(n_k_points));
        n_r_batches = ceil_div(as_size(n_k_points), n_r_batch_max);
    }

    const size_t row_max = size_batch_max * n_r_batch_max;
    std::vector<cplxdb> fmat(row_max * n_freq);
    std::vector<cplxdb> tmat(row_max * n_freq);

    global::ofs_myid << "size_batch_max/n_r_batch_max " << size_batch_max << " " << n_r_batch_max << endl;
    global::ofs_myid << "n_data_batches/n_r_batches " << n_data_batches << " " << n_r_batches << endl;
    global::ofs_myid << "row_max " << row_max << endl;

    // Loop over R-vector batches
    for (size_t i_r_batch = 0; i_r_batch < n_r_batches; i_r_batch++)
    {
        const size_t disp_r = i_r_batch * n_r_batch_max;
        const size_t n_r_this_batch = std::min(n_r_batch_max, as_size(n_k_points) - disp_r);

        // Initialize tau blocks for these R vectors
        for (size_t ir_this = 0; ir_this < n_r_this_batch; ir_this++)
        {
            const auto ir = disp_r + ir_this;
            const auto &R = Rlist[ir];
            for (const auto &tau: tfg.get_time_nodes())
            {
                Wc_tau_R[tau][R] = Matz(nr, nc, major_orig);
                Wc_tau_R[tau][R] = cplxdb{0.0, 0.0};
            }
        }

        // Loop over blocks of matrix to transform
        for (size_t i_data_batch = 0; i_data_batch < n_data_batches; i_data_batch++)
        {
            const size_t displ_data = i_data_batch * size_batch_max;
            const size_t size_this_batch = std::min(size_batch_max, size - displ_data);
            const size_t row_this = size_this_batch * n_r_this_batch;
            // Copy raw matrix to transform
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (size_t ir_this = 0; ir_this < n_r_this_batch; ir_this++)
            {
                for (size_t ifreq = 0; ifreq < n_freq; ifreq++)
                {
                    const auto freq = tfg.get_freq_nodes()[ifreq];
                    const auto &R = Rlist[disp_r + ir_this];
                    const auto &mat = Wc_R_freq.at(R).at(freq);
                    memcpy(fmat.data() + ifreq * row_max + ir_this * size_this_batch,
                           mat.ptr() + size_batch_max * i_data_batch, size_this_batch * sizeof(cplxdb));
                }
            }
            // Transform
            LapackConnector::gemm_f('N', 'N', row_this, n_freq, n_freq,
                                    C_ONE, fmat.data(), row_max, coeff_f2t.ptr(), n_freq,
                                    C_ZERO, tmat.data(), row_max);
            // librpa_int::global::ofs_myid << tmat << endl;

            // Copy back
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (size_t itau = 0; itau < n_freq; itau++)
            {
                for (size_t ir_this = 0; ir_this < n_r_this_batch; ir_this++)
                {
                    const auto tau = tfg.get_time_nodes()[itau];
                    const auto &R = Rlist[disp_r + ir_this];
                    memcpy(Wc_tau_R[tau][R].ptr() + displ_data,
                           tmat.data() + itau * row_max + ir_this * size_this_batch,
                           size_this_batch * sizeof(cplxdb));
                }
            }
        }

        // Data at these R-vector will not be used any more, release them
        for (size_t ir_this = 0; ir_this < n_r_this_batch; ir_this++)
        {
            const auto ir = disp_r + ir_this;
            const auto &R = Rlist[ir];
            Wc_R_freq.erase(R);
        }
    }

    librpa_int::global::lib_printf_root("Done converting Wc q,w -> R,t\n");
    comm_h.barrier();

    return Wc_tau_R;
}

std::map<double, std::map<Vector3_Order<int>, Matd>> CT_FT_Wc_freq_q_real(
    const MpiCommHandler &comm_h,
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc, const TFGrids &tfg,
    const ArrayDesc *ad_Wc, const AtomicBasis *atbasis_Wc,
    const SymmetryQPointView *qpoint_view,
    const SymmetryContext *symmetry_context)
{
    std::map<double, std::map<Vector3_Order<int>, Matd>> Wc_tau_R;
    if (Wc_freq_q.empty()) return Wc_tau_R;
    if (!tfg.has_time_grids())
        throw LIBRPA_RUNTIME_ERROR("TFGrids object does not have time grids");

    const auto n_freq = tfg.get_n_grids();
    const auto freq_nodes = tfg.get_freq_nodes();
    const auto time_nodes = tfg.get_time_nodes();
    const auto &Rlist = pbc.Rlist;
    const auto n_k_points = pbc.get_n_cells_bvk();

    global::lib_printf_root("Converting Wc(q,w) -> real W(R,t)\n");
    comm_h.barrier();
    auto Wc_freq_R = FT_Wc_freq_q_real(comm_h, Wc_freq_q, pbc, qpoint_view,
                                       symmetry_context, atbasis_Wc, ad_Wc);

    MAJOR major_orig = MAJOR::AUTO;
    int nr = 0, nc = 0;
    std::size_t size = 0;
    for (const auto &[freq, map_R_mat] : Wc_freq_R)
    {
        if (!map_R_mat.empty())
        {
            const auto &mat = map_R_mat.begin()->second;
            nr = mat.nr();
            nc = mat.nc();
            major_orig = mat.major();
            size = mat.size();
            break;
        }
    }
    if (major_orig == MAJOR::AUTO)
        throw LIBRPA_RUNTIME_ERROR("Dense real Wc cosine transform input has no matrices");

    std::map<Vector3_Order<int>, std::map<double, Matd>> Wc_R_freq;
    for (auto &[freq, Wc_R] : Wc_freq_R)
        for (auto &[R, Wc] : Wc_R)
            Wc_R_freq[R].emplace(freq, std::move(Wc));
    Wc_freq_R.clear();

    if (size == 0)
    {
        for (const auto &tau : time_nodes)
            for (const auto &R : Rlist)
                Wc_tau_R[tau][R] = Matd(nr, nc, major_orig);
        Wc_R_freq.clear();
        global::lib_printf_root("Done converting Wc q,w -> real R,t\n");
        comm_h.barrier();
        return Wc_tau_R;
    }

    Matd coeff_f2t(n_freq, n_freq, MAJOR::COL);
    for (std::size_t itau = 0; itau != n_freq; ++itau)
        for (std::size_t ifreq = 0; ifreq != n_freq; ++ifreq)
            coeff_f2t(ifreq, itau) = tfg.get_costrans_f2t()(itau, ifreq);

    const auto maxbytes_tmpmat = gbytes;
    std::size_t size_batch_max = 0, n_data_batches = 0;
    std::size_t n_r_batch_max = 0, n_r_batches = 0;
    const auto n_r_batches_with_whole_size =
        maxbytes_tmpmat / sizeof(double) / size / n_freq;
    if (n_r_batches_with_whole_size < 1)
    {
        n_r_batch_max = 1;
        n_r_batches = n_k_points;
        size_batch_max = std::max<std::size_t>(
            1, std::min(maxbytes_tmpmat / sizeof(double) / n_freq, size));
        n_data_batches = ceil_div(size, size_batch_max);
    }
    else
    {
        size_batch_max = size;
        n_data_batches = 1;
        n_r_batch_max = std::min(
            maxbytes_tmpmat / sizeof(double) / n_freq / size, as_size(n_k_points));
        n_r_batches = ceil_div(as_size(n_k_points), n_r_batch_max);
    }

    const std::size_t row_max = size_batch_max * n_r_batch_max;
    std::vector<double> fmat(row_max * n_freq);
    std::vector<double> tmat(row_max * n_freq);

    global::ofs_myid << "real size_batch_max/n_r_batch_max " << size_batch_max << " "
                     << n_r_batch_max << std::endl;
    global::ofs_myid << "real n_data_batches/n_r_batches " << n_data_batches << " "
                     << n_r_batches << std::endl;
    global::ofs_myid << "real row_max " << row_max << std::endl;

    for (std::size_t i_r_batch = 0; i_r_batch != n_r_batches; ++i_r_batch)
    {
        const std::size_t disp_r = i_r_batch * n_r_batch_max;
        const std::size_t n_r_this_batch =
            std::min(n_r_batch_max, as_size(n_k_points) - disp_r);

        for (std::size_t ir_this = 0; ir_this != n_r_this_batch; ++ir_this)
        {
            const auto &R = Rlist[disp_r + ir_this];
            for (const auto &tau : time_nodes)
                Wc_tau_R[tau][R] = Matd(nr, nc, major_orig);
        }

        for (std::size_t i_data_batch = 0; i_data_batch != n_data_batches; ++i_data_batch)
        {
            const std::size_t displ_data = i_data_batch * size_batch_max;
            const std::size_t size_this_batch =
                std::min(size_batch_max, size - displ_data);
            const std::size_t row_this = size_this_batch * n_r_this_batch;

            #pragma omp parallel for collapse(2) schedule(static)
            for (std::size_t ir_this = 0; ir_this < n_r_this_batch; ++ir_this)
            {
                for (std::size_t ifreq = 0; ifreq < n_freq; ++ifreq)
                {
                    const auto &R = Rlist[disp_r + ir_this];
                    const auto &mat = Wc_R_freq.at(R).at(freq_nodes[ifreq]);
                    memcpy(fmat.data() + ifreq * row_max + ir_this * size_this_batch,
                           mat.ptr() + displ_data, size_this_batch * sizeof(double));
                }
            }

            LapackConnector::gemm_f('N', 'N', row_this, n_freq, n_freq,
                                    1.0, fmat.data(), row_max, coeff_f2t.ptr(), n_freq,
                                    0.0, tmat.data(), row_max);

            #pragma omp parallel for collapse(2) schedule(static)
            for (std::size_t itau = 0; itau < n_freq; ++itau)
            {
                for (std::size_t ir_this = 0; ir_this < n_r_this_batch; ++ir_this)
                {
                    const auto &R = Rlist[disp_r + ir_this];
                    memcpy(Wc_tau_R.at(time_nodes[itau]).at(R).ptr() + displ_data,
                           tmat.data() + itau * row_max + ir_this * size_this_batch,
                           size_this_batch * sizeof(double));
                }
            }
        }

        for (std::size_t ir_this = 0; ir_this != n_r_this_batch; ++ir_this)
            Wc_R_freq.erase(Rlist[disp_r + ir_this]);
    }

    global::lib_printf_root("Done converting Wc q,w -> real R,t\n");
    comm_h.barrier();
    return Wc_tau_R;
}


map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
CT_FT_Wc_q2R_freq2time(
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_abf,
    map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        &Wc_freq_q,
    const TFGrids &tfg, const PeriodicBoundaryData &pbc, const vector<Vector3_Order<int>> &Rlist,
    const std::string &output_dir)
{
    using std::pair;
    using global::profiler;
    using global::lib_printf_coll;
    using global::lib_printf_root;
    using global::lib_printf;

    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;
    const int n_k_points = pbc.get_n_cells_bvk();

    map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_tau_R, Wc_freq_R;
    if (!tfg.has_time_grids()) throw LIBRPA_RUNTIME_ERROR("TFGrids object does not have time grids");
    const int ngrids = tfg.get_n_grids();
    std::set<std::pair<atom_t, atom_t>> atpairs_unique;

    global::lib_printf_root("Converting Wc(q,w) -> Wc(R,w) -> W(R,t)\n");

    profiler.start("construct_Wc_lower_half", "Construct Lower Half of Wc(q,w)");
    // NOTE: only upper half of Wc is built now
    //       here we recover the other half before transform to R space using the Hermitian property
    //       and Hermitize the diagonal blocks (due to numerical noise)
    for (int ifreq = 0; ifreq < ngrids; ifreq++)
    {
        const auto freq = tfg.get_freq_nodes()[ifreq];
        auto &Wc = Wc_freq_q.at(freq);
        vector<atom_t> iatoms_row;
        for (const auto &Mu_NuqWc : Wc) iatoms_row.push_back(Mu_NuqWc.first);
        for (auto iatom_row: iatoms_row)
        {
            vector<atom_t> iatoms_col;
            for (const auto &Nu_qWc : Wc.at(iatom_row))
            {
                iatoms_col.push_back(Nu_qWc.first);
            }
            for (auto iatom_col : iatoms_col)
            {
                atpairs_unique.insert({iatom_row, iatom_col});
                atpairs_unique.insert({iatom_col, iatom_row});
                for (const auto &q_Wc : Wc.at(iatom_row).at(iatom_col))
                {
                    assert(q_Wc.second.major() == major_Wc);
                    if(iatom_row != iatom_col)
                        Wc[iatom_col][iatom_row][q_Wc.first] = q_Wc.second.get_transpose(true);
                    else // Hermitize the diagonal blocks
                    {                        
                        auto Wc_mat = q_Wc.second;
                        Wc_mat = (Wc_mat + Wc_mat.get_transpose(true)) * 0.5;
                        Wc[iatom_row][iatom_row][q_Wc.first] = Wc_mat;
                    }
                }
            }
        }
    }
    comm_h.barrier();
    profiler.stop("construct_Wc_lower_half");

    profiler.start("Wc(q,w) -> Wc(R,w)", "Convert Wc(q,w) -> Wc(R,w)");
    vector<pair<pair<int, Vector3_Order<int>>, pair<atom_t, atom_t>>> ifreqtau_R_atpair_all;
    // allocate Wc(R,w) before hand
    for (auto R : Rlist)
    {
        for (int ifreq = 0; ifreq != ngrids; ifreq++)
        {
            auto tau = tfg.get_time_nodes()[ifreq];
            auto freq = tfg.get_freq_nodes()[ifreq];
            for (auto atpair_unique : atpairs_unique)
            {
                const auto Mu = atpair_unique.first;
                const int n_mu = atbasis_abf[Mu];
                const auto Nu = atpair_unique.second;
                const int n_nu = atbasis_abf[Nu];
                Wc_freq_R[freq][Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
                ifreqtau_R_atpair_all.push_back({{ifreq, R}, atpair_unique});// ifreq is same as itauR
            }
        }
    }

    lib_printf_coll("Task %4d: distributing %d {I, J, R, freq} on %d threads\n",
                    comm_h.myid, ifreqtau_R_atpair_all.size(),
                    omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto ifreqR_atpair : ifreqtau_R_atpair_all)
    {
        const auto ifreq = ifreqR_atpair.first.first;
        const auto freq = tfg.get_freq_nodes()[ifreq];
        const auto R = ifreqR_atpair.first.second;
        const auto Mu = ifreqR_atpair.second.first;
        const auto Nu = ifreqR_atpair.second.second;
        const int n_mu = atbasis_abf[Mu];
        const int n_nu = atbasis_abf[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> WfR_temp(n_mu, n_nu, major_Wc);

        if (Wc_freq_q.count(freq) == 0) continue;
        if (Wc_freq_q.at(freq).count(Mu) == 0) continue;
        if (Wc_freq_q.at(freq).at(Mu).count(Nu) == 0) continue;

        for (auto &Wc_q : Wc_freq_q.at(freq).at(Mu).at(Nu))
        {
            const auto q = Wc_q.first;
            const auto &Wc = Wc_q.second;
            for (auto q_bz : pbc.map_irk_ks.at(q))
            {
                const double ang = -q_bz * (R * pbc.latvec) * TWO_PI;
                const complex<double> weight =
                    complex<double>(cos(ang), sin(ang)) / double(n_k_points);
                if (q == q_bz)
                    WfR_temp += Wc * weight;
                else
                    WfR_temp += conj(Wc) * weight;
            }
        }
        // omp_set_lock(&lock_Wc);
        Wc_freq_R[freq][Mu][Nu][R] += WfR_temp;
        // omp_unset_lock(&lock_Wc);
    }
    comm_h.barrier();
    // HACK: Free up Wc_freq_q to save memory, especially for large Coulomb matrix case and many
    // minimax grids
    Wc_freq_q.clear();
    lib_printf_root("Done converting Wc(q,w) -> Wc(R,w)\n");

    profiler.stop("Wc(q,w) -> Wc(R,w)");

    profiler.start("Wc(R,w) -> Wc(R,t)", "Convert Wc(R,w) -> Wc(R,t)");
    if (comm_h.is_root())
    {
        lib_printf("Start converting Wc(R,w) -> Wc(R,t)\n");
    }
    // allocate Wc(R,t) before hand
    for (auto R : Rlist)
    {
        for (int itau = 0; itau != ngrids; itau++)
        {
            auto tau = tfg.get_time_nodes()[itau];
            auto freq = tfg.get_freq_nodes()[itau];
            for (auto atpair_unique : atpairs_unique)
            {
                const auto Mu = atpair_unique.first;
                const int n_mu = atbasis_abf[Mu];
                const auto Nu = atpair_unique.second;
                const int n_nu = atbasis_abf[Nu];
                Wc_tau_R[tau][Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
            }
        }
    }

    lib_printf_coll("Task %4d: distributing %d {I, J, R, tau} on %d threads\n",
        comm_h.myid, ifreqtau_R_atpair_all.size(),
        omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto itauR_atpair : ifreqtau_R_atpair_all)
    {
        const auto itau = itauR_atpair.first.first;
        const auto tau = tfg.get_time_nodes()[itau];
        const auto R = itauR_atpair.first.second;
        const auto Mu = itauR_atpair.second.first;
        const auto Nu = itauR_atpair.second.second;
        const int n_mu = atbasis_abf[Mu];
        const int n_nu = atbasis_abf[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> WtR_temp(n_mu, n_nu, major_Wc);

        for (int ifreq = 0; ifreq < ngrids; ifreq++)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto f2t = tfg.get_costrans_f2t()(itau, ifreq);
            // ofs_myid << "f2t cos eff for freq " << freq << " -> tau " << tau  << ": " << f2t <<
            // "\n";
            if (Wc_freq_R.count(freq) == 0) continue;
            if (Wc_freq_R.at(freq).count(Mu) == 0) continue;
            if (Wc_freq_R.at(freq).at(Mu).count(Nu) == 0) continue;
            if (Wc_freq_R.at(freq).at(Mu).at(Nu).count(R) == 0) continue;
            // cout << "freq: " << freq << "\n";

            const auto &Wc = Wc_freq_R.at(freq).at(Mu).at(Nu).at(R);
            WtR_temp += Wc * f2t;
        }
        // omp_set_lock(&lock_Wc);
        Wc_tau_R[tau][Mu][Nu][R] += WtR_temp;
        // omp_unset_lock(&lock_Wc);
    }
    // NOTE: Wc(R,w) will not be used any more, clean to free up memory
    Wc_freq_R.clear();
    release_free_mem();
    comm_h.barrier();
    lib_printf_root("Done converting Wc(R,w) -> Wc(R,t)\n");
    profiler.stop("Wc(R,w) -> Wc(R,t)");

    return Wc_tau_R;
}

map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
CT_Wc_freq2time_q(
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_abf,
    const map<double,
              atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        &Wc_freq_q,
    const TFGrids &tfg, const int &n_k_points, const vector<Vector3_Order<int>> &Rlist,
    const vector<Vector3_Order<double>> &qlist)
{
    using std::set;
    using std::pair;
    using global::lib_printf_root;
    using global::lib_printf_coll;
    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;

    map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_tau_q;
    if (!tfg.has_time_grids()) throw LIBRPA_RUNTIME_ERROR("TFGrids object does not have time grids");
    const int ngrids = tfg.get_n_grids();

    lib_printf_root("Converting Wc(q,w) -> W(q,t)\n");
    comm_h.barrier();

    set<pair<atom_t, atom_t>> atpairs_unique;
    for (const auto &freq_MuNuqWc : Wc_freq_q)
    {
        for (const auto &Mu_NuqWc : freq_MuNuqWc.second)
        {
            const auto Mu = Mu_NuqWc.first;
            for (const auto &Nu_qWc : Mu_NuqWc.second)
            {
                const auto Nu = Nu_qWc.first;
                atpairs_unique.insert({Mu, Nu});
                for (const auto &q_Wc : Nu_qWc.second)
                {
                    assert(q_Wc.second.major() == major_Wc);
                }
            }
        }
    }
    vector<pair<int, pair<atom_t, atom_t>>> itau_atpair_all;
    // allocate space before hand

    for (int itau = 0; itau != ngrids; itau++)
    {
        auto tau = tfg.get_time_nodes()[itau];
        for (auto atpair_unique : atpairs_unique)
        {
            const auto Mu = atpair_unique.first;
            const int n_mu = atbasis_abf[Mu];
            const auto Nu = atpair_unique.second;
            const int n_nu = atbasis_abf[Nu];
            for (auto q : qlist)
                Wc_tau_q[tau][Mu][Nu][q] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
            itau_atpair_all.push_back({itau, atpair_unique});
        }
    }

    lib_printf_coll("Task %4d: distributing %d {I, J, R, tau} on %d threads\n",
                    comm_h.myid, itau_atpair_all.size(),
                    omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto itau_atpair : itau_atpair_all)
    {
        const auto itau = itau_atpair.first;
        const auto tau = tfg.get_time_nodes()[itau];
        const auto Mu = itau_atpair.second.first;
        const auto Nu = itau_atpair.second.second;
        const int n_mu = atbasis_abf[Mu];
        const int n_nu = atbasis_abf[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> Wtq_temp(n_mu, n_nu, major_Wc);

        for (int ifreq = 0; ifreq < ngrids; ifreq++)
        {
            const auto freq = tfg.get_freq_nodes()[ifreq];
            const auto f2t = tfg.get_costrans_f2t()(itau, ifreq);
            // ofs_myid << "f2t cos eff for freq " << freq << " -> tau " << tau  << ": " << f2t <<
            // "\n";
            if (Wc_freq_q.count(freq) == 0) continue;
            if (Wc_freq_q.at(freq).count(Mu) == 0) continue;
            if (Wc_freq_q.at(freq).at(Mu).count(Nu) == 0) continue;
            // cout << "freq: " << freq << "\n";

            const auto &Wc_q_all = Wc_freq_q.at(freq).at(Mu).at(Nu);
            for (auto &Wc_q : Wc_q_all)
            {
                const auto q = Wc_q.first;
                const auto &Wc = Wc_q.second;
                const double weight = f2t;
                Wtq_temp = Wc * weight;
                // omp_set_lock(&lock_Wc);
                Wc_tau_q[tau][Mu][Nu][q] += Wtq_temp;
                // omp_unset_lock(&lock_Wc);
            }
        }
    }

    lib_printf_root("Done converting Wc q,w -> q,t\n");
    comm_h.barrier();

    return Wc_tau_q;
}

/// @brief Wc(q,w) -> Wc(R,w) or Wc(q,t) -> W(R,t)
atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old FT_Wc_q2R(
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_abf,
    const SymmetryContext &symmetry_context,
    const atom_mapping<std::map<Vector3_Order<double>, matrix_m<cplxdb>>>::pair_t_old
        &Wc_q,
    const TFGrids &, const PeriodicBoundaryData &pbc, const vector<Vector3_Order<int>> &Rlist, const bool,
    const std::string &,
    const bool use_symmetry_context)
{
    using global::lib_printf_root;
    using global::lib_printf_coll;
    using std::set;
    using std::pair;

    // major of Wc_freq_q input and Wc_tau_R output
    const MAJOR major_Wc = MAJOR::ROW;
    const auto n_k_points = pbc.get_n_cells_bvk();

    atom_mapping<std::map<Vector3_Order<int>, matrix_m<complex<double>>>>::pair_t_old Wc_R;

    lib_printf_root("Converting Wc(q) -> W(R)\n");
    comm_h.barrier();

    const auto atom_nabf = build_atom_nabf_map(atbasis_abf);
    const auto abf_layouts =
        atbasis_abf.build_species_basis_layouts(symmetry_context.atom_to_type);
    if (use_symmetry_context
        && can_use_symmetry_qstar_wr_restore(
            symmetry_context, abf_layouts, atom_nabf, pbc))
    {
        lib_printf_root(
            "GW symmetry accumulates full `W(R)` directly from IBZ q-stars\n");
        Wc_R = accumulate_symmetry_full_wr_from_ibz_q(
            comm_h, symmetry_context, abf_layouts, Wc_q, pbc, Rlist, atom_nabf);
        comm_h.barrier();
        lib_printf_root("Done converting Wc q -> R\n");
        return Wc_R;
    }

    set<pair<atom_t, atom_t>> atpairs_unique;
    for (const auto &MuNuqWc : Wc_q)
    {
        const auto Mu = MuNuqWc.first;
        for (const auto &Nu_qWc : MuNuqWc.second)
        {
            const auto Nu = Nu_qWc.first;
            atpairs_unique.insert({Mu, Nu});
            for (const auto &q_Wc : Nu_qWc.second)
            {
                assert(q_Wc.second.major() == major_Wc);
            }
        }
    }

    vector<pair<Vector3_Order<int>, pair<atom_t, atom_t>>> iR_atpair_all;
    // allocate space before hand
    for (auto R : Rlist)
    {
        for (auto atpair_unique : atpairs_unique)
        {
            const auto Mu = atpair_unique.first;
            const int n_mu = atbasis_abf[Mu];
            const auto Nu = atpair_unique.second;
            const int n_nu = atbasis_abf[Nu];
            Wc_R[Mu][Nu][R] = matrix_m<complex<double>>(n_mu, n_nu, major_Wc);
            iR_atpair_all.push_back({R, atpair_unique});
        }
    }

    lib_printf_coll("Task %4d: distributing %d {I, J, R} on %d threads\n",
                    comm_h.myid, iR_atpair_all.size(),
                    omp_get_max_threads());

#pragma omp parallel for schedule(dynamic)
    for (auto iR_atpair : iR_atpair_all)
    {
        const auto R = iR_atpair.first;
        const auto Mu = iR_atpair.second.first;
        const auto Nu = iR_atpair.second.second;
        const int n_mu = atbasis_abf[Mu];
        const int n_nu = atbasis_abf[Nu];

        // thread local temporary matrix
        matrix_m<complex<double>> WR_temp(n_mu, n_nu, major_Wc);

        if (Wc_q.count(Mu) == 0) continue;
        if (Wc_q.at(Mu).count(Nu) == 0) continue;

        const auto &Wc_q_all = Wc_q.at(Mu).at(Nu);
        for (auto &Wc_q : Wc_q_all)
        {
            const auto q = Wc_q.first;
            const auto &Wc = Wc_q.second;
            for (auto q_bz : pbc.map_irk_ks.at(q))
            {
                const double ang = -q_bz * (R * pbc.latvec) * TWO_PI;
                const complex<double> weight =
                    complex<double>(cos(ang), sin(ang)) / double(n_k_points);
                if (q == q_bz)
                    WR_temp += Wc * weight;
                else
                    WR_temp += conj(Wc) * weight;
            }
        }
        // omp_set_lock(&lock_Wc);
        Wc_R[Mu][Nu][R] += WR_temp;
        // omp_unset_lock(&lock_Wc);
    }
    comm_h.barrier();
    lib_printf_root("Done converting Wc q -> R\n");

    return Wc_R;
}

}  // namespace librpa_int
