/*!
 @file chi0.h
 @brief Utlities to compute the independent response function
 */
#pragma once
#include <complex>
#include <cstddef>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../math/vector3_order.h"
#include "../math/matrix_m.h"
#include "../mpi/base_blacs.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "atom.h"
#include "atomic_basis.h"
#include "symmetry_context.h"
#include "meanfield.h"
#include "pbc.h"
#include "qpoint_view.h"
#include "ri.h"
#include "timefreq.h"

namespace librpa_int {

bool rspace_symmetry_has_complete_band_space(const MeanField &mf, int n_bands);

//! Throw on all ranks when the error message of any rank is not empty
void check_chi0_collective_error(const MpiCommHandler &comm, const std::string &error);

using Chi0QMap =
    std::map<double, std::map<Vector3_Order<double>, atom_mapping<ComplexMatrix>::pair_t_old>>;

//! Object to handle calculation of independent repsonse function (\f$\chi_0\f$)
class Chi0
{
public:
    //! Original-KS endpoint probabilities in [0,1], [spin][k][loaded band].
    using BandSelection = std::vector<std::vector<std::vector<double>>>;

private:
    BandSelection band_selection_;
    bool is_mf_eigvec_k_distributed_;
    std::size_t gf_save;
    std::size_t gf_discard;
    //! space-time Green's function in occupied space, [ispin][isoc1][isoc2][I][J][R][tau]
    /*!
     * @note: tau (index) less than zero correspond to occupied GF,
     *        and larger than zero correspond to unoccpued GF.
     * @note: May need to use ComplexMatrix for GF.
     */
    std::map<int,
        std::map<int, std::map<int, atom_mapping<std::map<Vector3_Order<int>, std::map<double, matrix>>>::pair_t_old>>>
        gf_is_R_tau;

    //! Band selection used by every response Green-function route.
    std::vector<bool> band_mask;

    //! R on which the space-time GF are created, used for atom-pair and rtau routings
    std::vector<Vector3_Order<int>> Rlist_gf;

    //! Indices of G_{IJ}(R) to build, local to process, used for LibRI routing
    std::vector<std::pair<atpair_t, Vector3_Order<int>>> IJRs_gf_local;

    //! chi0 data in frequency domain and reciprocal space, [omega][q]
    Chi0QMap chi0_q;
    SymmetryQPointView qpoint_view_;

    void build_gf_Rt(Vector3_Order<int> R, double tau);

    // Free the intermeidate Green's functions
    void free_gf_Rt();

    //! Internal procedure to compute chi0_q by space-time method
    /*
     * @todo add threshold parameter. Maybe in the class level?
     */
    void build_chi0_q_space_time(const LibrpaParallelRouting routing, const Cs_LRI &Cs,
                                 const std::vector<atpair_t> &atpairs_ABF,
                                 const AtomicBasis &abf_Cs,
                                 std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
                                 const BlacsCtxtHandler &blacs_ctxt_h);

    // NOTE: the following three methods could be converted to static functions in chi0.cpp
    void build_chi0_q_space_time_atom_pair_routing(const Cs_LRI &Cs,
                                                   const std::vector<atpair_t> &atpairs_ABF);
    void build_chi0_q_space_time_R_tau_routing(const Cs_LRI &Cs,
                                               const std::vector<atpair_t> &atpairs_ABF);
    template <typename Tdata>
    void build_chi0_q_space_time_LibRI_routing(const Cs_LRI &Cs,
                                               const std::vector<atpair_t> &atpairs_ABF,
                                               const AtomicBasis &abf_Cs,
                                               std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
                                               const BlacsCtxtHandler &blacs_ctxt_h);

    //! Internal procedure to compute chi0_q in the conventional method, i.e. in frequency domain and reciprocal space
    void build_chi0_q_conventional(LibrpaParallelRouting routing, const Cs_LRI &Cs,
                                   const std::vector<atpair_t> &atpairs_ABF);
    /*!
     * s_alpha and s_beta are the spin component of unoccupied Green's function, G_{alpha, beta}(tau)
     * correspondingly, occupied GF G_{beta, alpha}(-tau) will be used. itau must be positive.
     */
    matrix compute_chi0_s_munu_tau_R(const atpair_R_mat_t &Cs_IJR,
                                     int spin_channel, int isoc1, int isoc2,
                                     atom_t mu, atom_t nu, double tau, Vector3_Order<int> R);
    // copy some reshape method inside chi0, test performance
    /* matrix reshape_Cs(const size_t n1, const size_t n2, const size_t n3, const std::shared_ptr<matrix> &Cs); */
    /* matrix reshape_dim_Cs(const size_t n1, const size_t n2, const size_t n3, const std::shared_ptr<matrix> &Cs);//(n1*n2,n3) -> (n1,n2*n3) */
    /* matrix reshape_mat(const size_t n1, const size_t n2, const size_t n3, const matrix &mat); */
    /* matrix reshape_mat_21(const size_t n1, const size_t n2, const size_t n3, const matrix &mat); //(n1,n2*n3) -> (n1*n2,n3) */

public:
    const MeanField &mf;
    //! Array descriptor of wave functions saved in the MeanField object
    const ArrayDesc &desc_wfc;
    const AtomicBasis &atbasis_wfc;
    const AtomicBasis &atbasis_abf;
    const PeriodicBoundaryData &pbc;
    const SymmetryContext &symmetry_context;
    const bool use_symmetry_context;
    const TFGrids &tfg;
    const MpiCommHandler &comm_h;
    const KPointBlacsParallelContext &kblacs_ctxt;

    //! Reduce response q sampling only; retain original full-k GF and full real-space contractions.
    //! cRPA uses this route so the original endpoint mask is never lost in k-star restoration.
    bool qstar_only_symmetry = false;

    double libri_threshold_C;
    double libri_threshold_G;
    int n_bands_exclude = 0;
    double gf_threshold;
    int libri_collect_s0_chunk;
    long long libri_collect_max_bytes;

    int nbands_G;

    Chi0(const MeanField &mf_in, const AtomicBasis &atbasis_wfc_in,
         const AtomicBasis &atbasis_abf_in, const PeriodicBoundaryData &pbc_in,
         const SymmetryContext &symmetry_context_in,
         const TFGrids &tfg_in, const KPointBlacsParallelContext &kblacs_ctxt_in,
         const ArrayDesc &desc_wfc_in, bool is_mf_eigvec_k_distributed,
         bool use_symmetry_context_in = true);
    ~Chi0() {};
    //! Build the independent response function in q-omega domain for ABFs on the atom pairs atpair_ABF and q-vectors in qlist
    void build(LibrpaParallelRouting routing,
               const Cs_LRI &Cs,
               const std::vector<atpair_t> &atpair_ABF,
               const AtomicBasis &abf_Cs,
               std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
               const BlacsCtxtHandler &blacs_ctxt_h);
    const Chi0QMap &get_chi0_q() const { return chi0_q; }
    //! Collective setter. An empty outer table disables selection; an all-zero
    //! table builds zero response. Endpoint weights in [0,1] multiply both f and (1-f).
    void set_band_selection(const BandSelection &selection);
    const BandSelection &get_band_selection() const noexcept { return band_selection_; }
    //! Restore a previously validated selector without collectives or allocation.
    //! New masks must enter through the collective set_band_selection method.
    void swap_band_selection(BandSelection &selection) noexcept { band_selection_.swap(selection); }
    void clear_band_selection() { set_band_selection({}); }
    //! Transfer local map ownership without copying distributed blocks.
    Chi0QMap take_chi0_q() noexcept;
    void swap_chi0_q(Chi0QMap &other) noexcept;
    //! Collectively validate and replace resident Pd by full-Pd, preserving
    //! sparse zero blocks and the common atom-pair distribution of two builds.
    void replace_chi0_q_by_difference(const Chi0QMap &full);
    const SymmetryQPointView &qpoint_view() const { return qpoint_view_; }
    const std::vector<Vector3_Order<double>> &active_qpoints() const { return qpoint_view_.representatives; }
    double q_weight(const Vector3_Order<double> &q) const { return qpoint_view_.weights.at(q); }
    void free_chi0_q(const double freq, const Vector3_Order<double> q);

    void unfold_abfs_Wc(
        std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
        std::map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<std::complex<double>>>>::pair_t_old> &Wc,
        const std::vector<Vector3_Order<double>> &qlist, const AtomicBasis &abf_unfold,
        const BlacsCtxtHandler &blacs_ctxt_h);
    void unfold_abfs_Wc_q(
        std::map<Vector3_Order<double>, ComplexMatrix> &sinvS,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old &Wc_q,
        const std::vector<Vector3_Order<double>> &qlist, const AtomicBasis &abf_unfold,
        const BlacsCtxtHandler &blacs_ctxt_h);
};

}
