#pragma once

#include <string>

#include "chi0.h"
#include "correlated_subspace.h"
#include "dielecmodel.h"

namespace librpa_int
{
//! One site's ordered tensor U_ab,cd, with independently chosen spin frames.
struct CrpaOnsiteTensor
{
    int site_index;
    int spin_left;
    int spin_right;
    ComplexMatrix bare;
    std::vector<ComplexMatrix> u;
    std::vector<ComplexMatrix> w;
};

struct CrpaResult
{
    std::vector<SiteOrbitalGroup> sites;
    std::vector<double> frequencies;
    std::vector<CrpaOnsiteTensor> tensors;
};

//! Dataset-independent references to the existing response/screening machinery.
struct CrpaContext
{
    Chi0 &chi0;
    const Cs_LRI &coefficients;
    const AtomicBasis &parent_auxiliary_basis;
    const std::vector<atpair_t> &local_atom_pairs;
    std::map<Vector3_Order<double>, ComplexMatrix> &auxiliary_transform;
    const atpair_k_cplx_mat_t &coulomb;
    const BlacsCtxtHandler &blacs;
    const ArrayDesc &active_auxiliary_descriptor;
    bool transform_auxiliary_basis = false;
    double sqrt_coulomb_threshold = 0.0;
    diele_func *full_headwing = nullptr;
    diele_func *constrained_headwing = nullptr;
    //! Resolved by the owning task; manual contexts retain legacy defaults.
    bool use_gpu_replace_scalapack = false;
    std::string eigensolver_method = "scalapack";
    std::string linear_algorithm = "scalapack";
};

/*! Build P0 and Pd on the same original KS poles and positive minimax nodes.
 * Pr=P0-Pd, U=v+Wc[Pr], W=v+Wc[P0]. The bare term uses the same retained
 * Coulomb spectrum as the native Wc backend. Each frame corresponds to one
 * physical spin channel. No static node, continuation, or file I/O is added.
 * SOC uses one full spinor frame whose density vertex sums both components.
 * Results are returned on all ranks. Original full-grid scalar/spinor LIBRI is supported.
 * Optional magnetic q-star screening restores full kernels before orbital contraction.
 */
CrpaResult compute_crpa_onsite(CrpaContext &context, const Chi0::BandSelection &selection,
                               const std::vector<CorrelatedSubspace *> &frames);

//! Contract a full 2D-distributed kernel; useful for independent MPI tests.
ComplexMatrix contract_crpa_blacs(const ComplexMatrix &left_vertex,
                                  const ComplexMatrix &right_vertex, const Matz &kernel,
                                  const ArrayDesc &descriptor, int n_left, int n_right,
                                  const MpiCommHandler &comm);

}  // namespace librpa_int
