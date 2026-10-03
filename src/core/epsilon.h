#pragma once
#include <complex>
#include <map>
#include <string>
#include <vector>

#include "../math/matrix_m.h"
#include "../mpi/base_blacs.h"
#include "atom.h"
#include "atomic_basis.h"
#include "chi0.h"
#include "dielecmodel.h"
#include "pbc.h"
#include "qpoint_view.h"
#include "ri.h"
#include "symmetry_context.h"

namespace librpa_int {

bool strict_2d_complete_wc_requested(bool replace_w_head, int option_dielect_func,
                                     bool use_2d_dielectric);
void validate_strict_2d_complete_wc_runtime(bool strict_2d_requested, bool headwing_data_available,
                                            bool use_scalapack_gw_wc,
                                            bool strict_2d_metadata_configured);
std::string strict_2d_finite_q_diagnostics_header();
std::string strict_2d_gamma_wc_diagnostics_header();
std::vector<Vector3_Order<double>> strict_2d_diagnostic_qpoint_order(
    const std::vector<Vector3_Order<double>> &qpoints, bool diagnostics_enabled);
bool strict_2d_qmember_diagnostic_keeps(const Vector3_Order<double>& q_member_frac,
                                        const Vector3_Order<double>& selected_q_frac,
                                        bool diagnostics_enabled);
bool use_strict_2d_complete_wc_gamma_route(bool replace_w_head, int option_dielect_func,
                                           bool use_2d_dielectric, bool gamma_point,
                                           bool headwing_data_available);

struct CorrEnergy
{
    enum type { RPA, MP2 };
    type etype;
    cplxdb value;
    std::map<Vector3_Order<double>, cplxdb> qcontrib;
};

CorrEnergy compute_RPA_correlation(LibrpaParallelRouting routing, const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat);

CorrEnergy compute_RPA_correlation_blacs(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat,
                                         const std::vector<atpair_t> &local_atpair,
                                         const BlacsCtxtHandler &blacs_h);
CorrEnergy compute_RPA_correlation_blacs_2d(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat,
                                            const std::vector<atpair_t> &local_atpair,
                                            const BlacsCtxtHandler &blacs_h,
                                            const RpaHeadwingSettings &headwing_settings = {},
                                            diele_func *df_headwing = nullptr);
CorrEnergy compute_RPA_correlation_blacs_2d_gamma_only(Chi0 &chi0, atpair_k_cplx_mat_t &coulmat,
                                                       const std::vector<atpair_t> &local_atpair,
                                                       const BlacsCtxtHandler &blacs_h, bool use_gpu_replace_scalapack = false);
CorrEnergy compute_MP2_correlation(const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat);

std::map<double, std::map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q(
    const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat);
std::map<double, std::map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>> compute_Pi_q_MPI(
    const Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat);

atom_mapping<ComplexMatrix>::pair_t_old gather_vq_row_q(const AtomicBasis &atbasis_abf,
                                                        const MpiCommHandler &comm_h, const int &I,
                                                        const atpair_k_cplx_mat_t &coulmat,
                                                        const Vector3_Order<double> &ik_vec);

std::map<double, std::map<Vector3_Order<double>, Matz>> compute_Wc_freq_q(
    Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps, atpair_k_cplx_mat_t &coulmat_wc,
    double sqrt_coulomb_threshold, const std::vector<cplxdb> &epsmac_LF_imagfreq,
    bool debug = false, const char *output_dir = ".");

std::map<double, std::map<Vector3_Order<double>, Matz>> compute_Wc_freq_q_blacs(
    Chi0 &chi0, const atpair_k_cplx_mat_t &coulmat_eps, atpair_k_cplx_mat_t &coulmat_wc,
    double sqrt_coulomb_threshold, const bool replace_w_head, int option_dielect_func,
    const std::vector<std::complex<double>> &epsmac_LF_imagfreq, diele_func *df_headwing,
    const BlacsCtxtHandler &blacs_h, const librpa_int::ArrayDesc &ad, bool debug = false,
    const char *output_dir = ".", bool use_cholesky_gw_wc = false,
    bool use_gpu_replace_scalapack = false, bool use_elpa_sqrt_coulomb = false,
    bool output_2d_finite_q_diagnostics = false);

void unfold_Wc_freq_q_blacs(
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
    const std::vector<Vector3_Order<double>> &qlist,
    const BlacsCtxtHandler &blacs_h,
    const librpa_int::ArrayDesc &desc_small,
    const librpa_int::ArrayDesc &desc_full);

//! Fourier transform screened Coulomb in q-space to R-space, but still in frequency domain
std::map<double, std::map<Vector3_Order<int>, Matz>> FT_Wc_freq_q(
    const MpiCommHandler &comm_h,
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc, bool remove_freq_q = true,
    const SymmetryQPointView *qpoint_view = nullptr,
    const SymmetryContext *symmetry_context = nullptr,
    const AtomicBasis *atbasis_Wc = nullptr,
    const ArrayDesc *ad_Wc = nullptr);

std::map<double, std::map<Vector3_Order<int>, Matz>> CT_FT_Wc_freq_q(
    const MpiCommHandler &comm_h,
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc, const TFGrids &tfg, bool remove_freq_q = true,
    bool output_wc_rf = false, int ifreq_output_wc_start = 0,
    int ifreq_output_wc_end = -1, bool output_wc_rf_atom_pair = false,
    const std::string &output_dir = ".", const ArrayDesc *ad_Wc = nullptr,
    const AtomicBasis *atbasis_Wc = nullptr,
    const SymmetryQPointView *qpoint_view = nullptr,
    const SymmetryContext *symmetry_context = nullptr);

//! Dense Wc(q,iw) -> real Wc(R,tau), consuming Wc_freq_q.
std::map<double, std::map<Vector3_Order<int>, Matd>> CT_FT_Wc_freq_q_real(
    const MpiCommHandler &comm_h,
    std::map<double, std::map<Vector3_Order<double>, Matz>> &Wc_freq_q,
    const PeriodicBoundaryData &pbc, const TFGrids &tfg,
    const ArrayDesc *ad_Wc = nullptr,
    const AtomicBasis *atbasis_Wc = nullptr,
    const SymmetryQPointView *qpoint_view = nullptr,
    const SymmetryContext *symmetry_context = nullptr);

// @brief Fourier transform screened Coulomb Wc(q,w) -> Wc(R,w) -> W(R,t)
// @details transform step by step to output Wc_freq_R, and return full atom-pair matrix
// @attention CT_FT_Wc_freq_q only return upper atom-pair, but final Wc_libri should be same
std::map<double, atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old>
CT_FT_Wc_q2R_freq2time(
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_abf,
    std::map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<cplxdb>>>::pair_t_old>
        &Wc_freq_q,  // upper atom-pair input
    const TFGrids &tfg, const PeriodicBoundaryData &pbc, const std::vector<Vector3_Order<int>> &Rlist,
    const std::string &output_dir);

/// @brief Wc(q,w) -> Wc(q,t)
std::map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old>
CT_Wc_freq2time_q(
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_abf,
    const std::map<double,
              atom_mapping<std::map<Vector3_Order<double>, matrix_m<cplxdb>>>::pair_t_old>
        &Wc_freq_q,
    const TFGrids &tfg, const int &n_kpoints, const std::vector<Vector3_Order<int>> &Rlist,
    const std::vector<Vector3_Order<double>> &qlist);

/// @brief Wc(q,w) -> Wc(R,w) or Wc(q,t) -> W(R,t)
atom_mapping<std::map<Vector3_Order<int>, matrix_m<std::complex<double>>>>::pair_t_old FT_Wc_q2R(
    const MpiCommHandler &comm_h,
    const AtomicBasis &atbasis_abf,
    const SymmetryContext &symmetry_context,
    const atom_mapping<std::map<Vector3_Order<double>, matrix_m<cplxdb>>>::pair_t_old
        &Wc_q,
    const TFGrids &tfg, const PeriodicBoundaryData &pbc, const std::vector<Vector3_Order<int>> &Rlist, const bool is_freq,
    const std::string &output_dir,
    bool use_symmetry_context = true);

ComplexMatrix compute_Pi_freq_q_row_ri(const AtomicBasis &atbasis_abf, const Vector3_Order<double> &ik_vec, const atom_mapping<ComplexMatrix>::pair_t_old &chi0_freq_q, const atpair_k_cplx_mat_t &Vq_loc, const std::vector<atpair_t> &local_atpair, const int &I, const Vector3_Order<double> &q);
ComplexMatrix compute_Pi_freq_q_row(const AtomicBasis &atbasis_abf, const Vector3_Order<double> &ik_vec,
                                    const atom_mapping<ComplexMatrix>::pair_t_old &chi0_freq_q,
                                    const atom_mapping<ComplexMatrix>::pair_t_old &Vq_row,
                                    const std::vector<atpair_t> &local_atpair, const int &I);

cplxdb compute_pi_det_blacs(ComplexMatrix &loc_piT, const librpa_int::ArrayDesc& arrdesc_pi, int *ipiv, int &info);

cplxdb compute_pi_det_blacs_2d(Matz &loc_piT, const librpa_int::ArrayDesc &arrdesc_pi, int *ipiv, int &info);
cplxdb compute_rpa_response_trace_logdet_blacs_2d(
    const Matz &response, const librpa_int::ArrayDesc &response_desc);
double compute_pi_det_blacs_2d_gamma_only(Matd &loc_piT, const librpa_int::ArrayDesc &arrdesc_pi, int *ipiv, int &info);

// void test_libcomm_for_system(const atpair_k_cplx_mat_t &coulmat);
}
