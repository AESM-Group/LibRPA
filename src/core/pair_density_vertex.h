#pragma once

/*!
 * @file pair_density_vertex.h
 * @brief Projector-consistent complex pair-density vertices for cRPA.
 *
 * For physical-position localized AO coefficients t(A), raw first-ABF-end
 * Cs[I][{J,R}](mu,i,j), and a finite Born-von Karman grid, this module builds
 *
 *   D^{I mu}_{ab}(A) = sum_{J,R,i,j} Cs^{I mu}_{Ii,Jj}(R)
 *       [conj(t^a_{Ii}(A)) t^b_{Jj}(A+R)
 *        + conj(t^a_{Jj}(A+R)) t^b_{Ii}(A)],
 *
 * including both AO orderings, not a generic factor of two on raw Cs,
 *
 * followed by the external-vector Fourier transform
 *
 *   D^{I mu}_{ab}(q) = sum_A exp(-i 2 pi q_frac.A) D^{I mu}_{ab}(A).
 *
 * The minus sign is the dual of LibRPA's plus-sign Fourier transform for the
 * translationally invariant response/Coulomb kernel.  Ordered pairs use
 * pair(a,b)=a*n_orbitals+b.  A screened interaction is contracted as
 * U_ab,cd = D_ba^dagger W D_cd.
 */

#include <map>
#include <string>
#include <vector>

#include "../math/complexmatrix.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "atomic_basis.h"
#include "correlated_subspace.h"
#include "pbc.h"
#include "ri.h"

namespace librpa_int
{

using PairDensityCsMap = std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>>;
using PairVertexAtomBlocks = std::map<int, ComplexMatrix>;

/*! Complex pair-density vertex for one correlated site. */
struct SitePairDensityVertex
{
    int site_index = -1;
    int atom_index = -1;
    int n_orbitals = 0;
    std::string label;
    std::vector<std::string> orbital_labels;

    /*! q (internal Cartesian reciprocal units) -> auxiliary atom ->
     *  D block of shape n_orbitals^2 x n_mu(atom). */
    std::map<Vector3_Order<double>, PairVertexAtomBlocks> q_blocks;
};

/*! Build one site vertex from explicit physical-position t(A) matrices.
 *
 * This is the production-linked pure kernel used by the independent unit
 * tests. T_R_site[iA] has shape (n_spinor*NAO) x n_orbitals and stores t(pbc.Rlist[iA]),
 * with one or two component-major spinor blocks. SOC sums the same-component
 * products before contraction; Cs and both atomic bases remain spatial.
 * not CorrelatedSubspace's negative-Fourier T_code(A). The wrapper below
 * converts t(A)=T_code(wrap_bvk(-A)) on untwisted BvK meshes. This pure kernel
 * independently accepts any complete full-q grid dual to pbc.Rlist.
 * A borrowed CPU grid permits the experimental mixed path when its explicit
 * scratch fits within the removed complex coefficient storage; otherwise the
 * conventional product is used.
 */
SitePairDensityVertex build_site_pair_density_vertex_from_T(
    int site_index, const SiteOrbitalGroup &site, const std::vector<ComplexMatrix> &T_R_site,
    const PairDensityCsMap &Cs_complete, const AtomicBasis &basis_wfc, const AtomicBasis &basis_abf,
    const PeriodicBoundaryData &pbc, const std::vector<Vector3_Order<double>> &qpoints,
    const BlacsCtxtHandler *mixed_cpu_grid = nullptr);

/*! Build one site vertex from an already materialized CorrelatedSubspace. */
SitePairDensityVertex build_site_pair_density_vertex(
    CorrelatedSubspace &subspace, int site_index, const PairDensityCsMap &Cs_complete,
    const AtomicBasis &basis_wfc, const AtomicBasis &basis_abf, const PeriodicBoundaryData &pbc,
    const std::vector<Vector3_Order<double>> &qpoints,
    const BlacsCtxtHandler *mixed_cpu_grid = nullptr);

/*! Express a site vertex in an active auxiliary basis without modifying inputs.
 *
 * sinvS[q] = L(q) has shape n_active x n_parent in global atomic-basis order.
 * D_active_rows(q) = D_parent_rows(q) * transpose(L(q)), NOT its adjoint.
 * Every parent q needs an L; extra L keys are ignored. Missing parent atom
 * blocks mean zero. Cross-atom mixing is allowed; every positive-width active
 * atom is retained, including all-zero blocks, so W ownership remains explicit.
 * Atoms with zero active ABFs have no output block. The atom count is unchanged.
 * No inverse, positivity, or time-reversal assumption is imposed on L.
 */
SitePairDensityVertex transform_site_pair_density_vertex_auxiliary_basis(
    const SitePairDensityVertex &parent, const AtomicBasis &parent_basis_abf,
    const AtomicBasis &active_basis_abf,
    const std::map<Vector3_Order<double>, ComplexMatrix> &sinvS);

/*! Contract a rectangular ordered-pair vertex between two distinct sites.
 *
 * The result has shape (n_left^2) x (n_right^2) and uses the same production
 * pair convention: U_ab,cd = sum conj(D_left[ba,mu]) W[mu,nu] D_right[cd,nu].
 * W is directed from the left auxiliary block to the right block; the
 * Hermitian reverse is obtained by conjugate-transposing the result.
 */
ComplexMatrix contract_rectangular_ordered_pair_vertex(const ComplexMatrix &D_left,
                                                       const ComplexMatrix &W,
                                                       const ComplexMatrix &D_right, int n_left,
                                                       int n_right);

/*! Production wrapper that redistributes the relevant distributed LibRI Cs
 * blocks to every rank and builds every configured site deterministically.
 */
class PairDensityVertex
{
public:
    PairDensityVertex(CorrelatedSubspace &subspace, const Cs_LRI &Cs, const AtomicBasis &basis_wfc,
                      const AtomicBasis &basis_abf, const PeriodicBoundaryData &pbc,
                      const std::vector<Vector3_Order<double>> &qpoints,
                      const MpiCommHandler &comm_h,
                      const BlacsCtxtHandler *mixed_cpu_grid = nullptr);

    const std::vector<SitePairDensityVertex> &sites() const { return sites_; }
    const SitePairDensityVertex &site(int site_index) const;

private:
    std::vector<SitePairDensityVertex> sites_;
};

}  // namespace librpa_int
