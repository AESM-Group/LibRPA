#include "pair_density_vertex.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <set>
#include <sstream>
#include <utility>
#include <valarray>

#include "../interface/blas_lapack/lapack_connector.h"
#include "../utils/constants.h"
#include "../utils/error.h"

#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#endif

namespace librpa_int
{
namespace
{

using Complex = std::complex<double>;

bool finite(const Complex &z) { return std::isfinite(z.real()) && std::isfinite(z.imag()); }

bool finite(const Vector3_Order<double> &v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

void validate_finite_matrix(const ComplexMatrix &m, const std::string &label)
{
    for (int i = 0; i != m.size; ++i)
    {
        if (!finite(m.c[i])) throw LIBRPA_RUNTIME_ERROR(label + " contains a non-finite value");
    }
}

void collective_stage_error(const MpiCommHandler &comm_h, const std::string &label,
                            const std::string &local_error)
{
    int local_failed = local_error.empty() ? 0 : 1;
    int global_failed = 0;
    MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX, comm_h.comm);
    if (global_failed != 0)
        throw LIBRPA_RUNTIME_ERROR(
            label + " failed on at least one MPI rank: " +
            (local_error.empty() ? "another rank reported an error" : local_error));
}

int wrap_component(const int value, const int period)
{
    if (period <= 0)
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex requires a positive BvK period");
    int wrapped = (value + period / 2) % period;
    if (wrapped < 0) wrapped += period;
    return wrapped - period / 2;
}

Vector3_Order<int> wrap_bvk(const Vector3_Order<int> &R, const Vector3_Order<int> &period)
{
    return {wrap_component(R.x, period.x), wrap_component(R.y, period.y),
            wrap_component(R.z, period.z)};
}

void validate_bvk_and_q_grid(const PeriodicBoundaryData &pbc,
                             const std::vector<Vector3_Order<double>> &qpoints)
{
    if (pbc.period.x <= 0 || pbc.period.y <= 0 || pbc.period.z <= 0)
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex requires positive BvK periods");
    const int ncells = pbc.get_n_cells_bvk();
    if (ncells <= 0 || static_cast<int>(pbc.Rlist.size()) != ncells)
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density vertex requires a complete BvK real-space grid");
    if (pbc.Rlist != construct_R_grid(pbc.period))
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density vertex R-list does not use the canonical BvK ordering");
    if (static_cast<int>(qpoints.size()) != ncells)
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density vertex requires a complete full-q grid dual to Rlist");
    for (const auto &q : qpoints)
        if (!finite(q))
            throw LIBRPA_INVALID_ARGUMENT("pair-density vertex q grid contains non-finite values");

    // A full (possibly uniformly shifted) q mesh is dual to the BvK R grid.
    // Check the finite Fourier matrix instead of assuming a particular q order.
    constexpr double dual_tol = 1.0e-9;
    for (std::size_t iq = 0; iq != qpoints.size(); ++iq)
    {
        const auto qf_i = pbc.latvec * qpoints[iq];
        for (std::size_t jq = 0; jq != qpoints.size(); ++jq)
        {
            const auto qf_j = pbc.latvec * qpoints[jq];
            Complex overlap{0.0, 0.0};
            for (const auto &R : pbc.Rlist)
            {
                const double arg = TWO_PI * ((qf_i.x - qf_j.x) * R.x + (qf_i.y - qf_j.y) * R.y +
                                             (qf_i.z - qf_j.z) * R.z);
                overlap += Complex(std::cos(arg), std::sin(arg));
            }
            overlap /= static_cast<double>(ncells);
            const Complex expected = iq == jq ? Complex{1.0, 0.0} : Complex{0.0, 0.0};
            if (std::abs(overlap - expected) > dual_tol)
                throw LIBRPA_INVALID_ARGUMENT(
                    "pair-density vertex q/R grids fail finite Fourier duality");
        }
    }
}

bool atom_block_nonzero(const ComplexMatrix &T, const AtomicBasis &basis, const int atom)
{
    const int begin = static_cast<int>(basis.get_part_range().at(atom));
    const int end = begin + static_cast<int>(basis.get_atom_nb(atom));
    const int nao = static_cast<int>(basis.nb_total);
    for (int s = 0; s < T.nr / nao; ++s)
        for (int i = begin; i != end; ++i)
            for (int a = 0; a != T.nc; ++a)
                if (T(s * nao + i, a) != Complex{0.0, 0.0}) return true;
    return false;
}

std::set<int> support_atoms(const std::vector<ComplexMatrix> &T_R, const AtomicBasis &basis)
{
    std::set<int> result;
    for (int atom = 0; atom != static_cast<int>(basis.n_atoms); ++atom)
    {
        for (const auto &T : T_R)
        {
            if (atom_block_nonzero(T, basis, atom))
            {
                result.insert(atom);
                break;
            }
        }
    }
    return result;
}

void validate_cs_tensor(const RI::Tensor<double> &Cs, const int n_mu, const int n_i, const int n_j,
                        const std::string &label)
{
    const std::size_t expected = static_cast<std::size_t>(n_mu) * n_i * n_j;
    if (Cs.data == nullptr || Cs.get_shape_all() != expected)
        throw LIBRPA_RUNTIME_ERROR(label + " has an inconsistent tensor size");
#ifdef LIBRPA_USE_LIBRI
    if (Cs.shape.size() != 3 || Cs.shape[0] != static_cast<std::size_t>(n_mu) ||
        Cs.shape[1] != static_cast<std::size_t>(n_i) ||
        Cs.shape[2] != static_cast<std::size_t>(n_j))
        throw LIBRPA_RUNTIME_ERROR(label + " must have shape (n_mu,n_i,n_j)");
#endif
    for (std::size_t i = 0; i != Cs.data->size(); ++i)
        if (!std::isfinite((*Cs.data)[i]))
            throw LIBRPA_RUNTIME_ERROR(label + " contains a non-finite value");
}

}  // namespace

SitePairDensityVertex build_site_pair_density_vertex_from_T(
    const int site_index, const SiteOrbitalGroup &site, const std::vector<ComplexMatrix> &T_R_site,
    const PairDensityCsMap &Cs_complete, const AtomicBasis &basis_wfc, const AtomicBasis &basis_abf,
    const PeriodicBoundaryData &pbc, const std::vector<Vector3_Order<double>> &qpoints)
{
    validate_bvk_and_q_grid(pbc, qpoints);
    if (!basis_wfc.initialized() || !basis_abf.initialized())
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density vertex requires initialized AO and auxiliary bases");
    if (basis_wfc.n_atoms != basis_abf.n_atoms)
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density vertex AO and auxiliary bases have different atom counts");
    if (site_index < 0 || site.n_orbitals <= 0 || site.label.empty())
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex received invalid site metadata");
    if (site.atom_index < 0 || site.atom_index >= static_cast<int>(basis_wfc.n_atoms))
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex site atom index is out of range");
    if (!site.orbital_labels.empty() &&
        static_cast<int>(site.orbital_labels.size()) != site.n_orbitals)
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex orbital-label count is inconsistent");
    if (T_R_site.size() != pbc.Rlist.size())
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density vertex T(R) count does not match the BvK R grid");

    const int nao = static_cast<int>(basis_wfc.nb_total);
    if (nao <= 0 || T_R_site.empty())
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex requires nonempty AO and T(R) data");
    const int n_spinor = T_R_site.front().nr / nao;
    if (n_spinor != 1 && n_spinor != 2)
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex requires one or two spinor components");
    for (const auto &T : T_R_site)
    {
        if (T.nr != n_spinor * nao || T.nc != site.n_orbitals)
            throw LIBRPA_INVALID_ARGUMENT(
                "pair-density vertex T(R) has an inconsistent matrix shape");
        validate_finite_matrix(T, "pair-density vertex T(R)");
    }
    const auto support = support_atoms(T_R_site, basis_wfc);
    if (support.empty())
        throw LIBRPA_RUNTIME_ERROR("pair-density vertex site has empty AO support");

    const int npair = site.n_orbitals * site.n_orbitals;
    std::vector<PairVertexAtomBlocks> D_by_R(pbc.Rlist.size());
    bool used_relevant_cs = false;

    for (const auto &[I, JR_Cs] : Cs_complete)
    {
        if (I < 0 || I >= static_cast<int>(basis_wfc.n_atoms))
            throw LIBRPA_RUNTIME_ERROR("pair-density vertex Cs first atom is out of range");
        if (support.count(I) == 0) continue;
        const int n_i = static_cast<int>(basis_wfc.get_atom_nb(I));
        const int n_mu = static_cast<int>(basis_abf.get_atom_nb(I));
        const int i_begin = static_cast<int>(basis_wfc.get_part_range().at(I));

        for (const auto &[JR, Cs] : JR_Cs)
        {
            const int J = JR.first;
            const auto R = Vector3_Order<int>{JR.second[0], JR.second[1], JR.second[2]};
            if (J < 0 || J >= static_cast<int>(basis_wfc.n_atoms))
                throw LIBRPA_RUNTIME_ERROR("pair-density vertex Cs second atom is out of range");
            if (support.count(J) == 0) continue;
            const int n_j = static_cast<int>(basis_wfc.get_atom_nb(J));
            const int j_begin = static_cast<int>(basis_wfc.get_part_range().at(J));
            std::ostringstream label;
            label << "pair-density vertex Cs[I=" << I << ",J=" << J << ",R=" << R.x << ',' << R.y
                  << ',' << R.z << ']';
            validate_cs_tensor(Cs, n_mu, n_i, n_j, label.str());
            used_relevant_cs = true;

            // Cs has the same (mu,ij) ordering for every translated orbital
            // product. Pack it once, and form the two AO orderings only once
            // per (ab,ij), rather than repeating them for every auxiliary mu.
            const int nij = n_i * n_j;
            ComplexMatrix coefficients(n_mu, nij);
            for (int mu = 0; mu != n_mu; ++mu)
                for (int ij = 0; ij != nij; ++ij)
                    coefficients(mu, ij) = (*Cs.data)[static_cast<std::size_t>(mu) * nij + ij];
            ComplexMatrix products(npair, nij);

            for (std::size_t iA = 0; iA != pbc.Rlist.size(); ++iA)
            {
                const auto B = wrap_bvk(pbc.Rlist[iA] + R, pbc.period);
                const int iB = pbc.get_R_index(B);
                if (iB < 0)
                    throw LIBRPA_RUNTIME_ERROR(
                        "pair-density vertex failed to wrap A+R onto the BvK grid");
                const auto &T_left = T_R_site[iA];
                const auto &T_right = T_R_site[static_cast<std::size_t>(iB)];
                if (!atom_block_nonzero(T_left, basis_wfc, I) ||
                    !atom_block_nonzero(T_right, basis_wfc, J))
                    continue;

                auto [it_D, inserted] = D_by_R[iA].emplace(I, ComplexMatrix(npair, n_mu));
                auto &D = it_D->second;
                (void)inserted;
                for (int a = 0; a != site.n_orbitals; ++a)
                    for (int b = 0; b != site.n_orbitals; ++b)
                    {
                        const int ab = a * site.n_orbitals + b;
                        for (int i = 0; i != n_i; ++i)
                            for (int j = 0; j != n_j; ++j)
                            {
                                Complex density{};
                                for (int s = 0; s < n_spinor; ++s)
                                    density += std::conj(T_left(s * nao + i_begin + i, a)) *
                                                   T_right(s * nao + j_begin + j, b) +
                                               std::conj(T_right(s * nao + j_begin + j, a)) *
                                                   T_left(s * nao + i_begin + i, b);
                                products(ab, i * n_j + j) = density;
                            }
                    }
                // Ordinary transpose: the conjugations are already in products.
                if (n_mu && nij)
                    LapackConnector::gemm('N', 'T', npair, n_mu, nij, Complex{1.0, 0.0}, products.c,
                                          nij, coefficients.c, nij, Complex{1.0, 0.0}, D.c, n_mu);
            }
        }
    }
    if (!used_relevant_cs)
        throw LIBRPA_RUNTIME_ERROR(
            "pair-density vertex found no Cs block connecting the site's AO support");

    SitePairDensityVertex result;
    result.site_index = site_index;
    result.atom_index = site.atom_index;
    result.n_orbitals = site.n_orbitals;
    result.label = site.label;
    result.orbital_labels = site.orbital_labels;

    double max_abs_D = 0.0;
    for (const auto &q : qpoints)
    {
        const auto qfrac = pbc.latvec * q;
        auto &D_q = result.q_blocks[q];
        for (std::size_t iA = 0; iA != pbc.Rlist.size(); ++iA)
        {
            const auto &A = pbc.Rlist[iA];
            const double arg = -TWO_PI * (qfrac.x * A.x + qfrac.y * A.y + qfrac.z * A.z);
            const Complex phase{std::cos(arg), std::sin(arg)};
            for (const auto &[I, D_A] : D_by_R[iA])
            {
                auto [it_D, inserted] = D_q.emplace(I, ComplexMatrix(D_A.nr, D_A.nc));
                auto &D = it_D->second;
                (void)inserted;
                for (int i = 0; i != D.size; ++i)
                {
                    D.c[i] += phase * D_A.c[i];
                    max_abs_D = std::max(max_abs_D, std::abs(D.c[i]));
                }
            }
        }
    }
    if (!(max_abs_D > 0.0) || !std::isfinite(max_abs_D))
        throw LIBRPA_RUNTIME_ERROR(
            "pair-density vertex is identically zero or non-finite on the full q grid");
    return result;
}

SitePairDensityVertex build_site_pair_density_vertex(
    CorrelatedSubspace &subspace, const int site_index, const PairDensityCsMap &Cs_complete,
    const AtomicBasis &basis_wfc, const AtomicBasis &basis_abf, const PeriodicBoundaryData &pbc,
    const std::vector<Vector3_Order<double>> &qpoints)
{
    const auto &sites = subspace.get_sites();
    if (site_index < 0 || site_index >= static_cast<int>(sites.size()))
        throw LIBRPA_INVALID_ARGUMENT("pair-density vertex site index is out of range");
    if (subspace.get_R_list() != pbc.Rlist)
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density vertex subspace and Dataset use different R grids");

    // T_code(R) uses exp(-ikR); physical AO positions require t(A)=T_code(-A).
    // Wrapping -A is valid without boundary phases only on an untwisted mesh.
    constexpr double mesh_tol = 1.0e-9;
    for (const auto &k : subspace.get_kfrac_list())
        for (const double component : {k.x * pbc.period.x, k.y * pbc.period.y, k.z * pbc.period.z})
            if (!std::isfinite(component) || std::abs(component - std::round(component)) > mesh_tol)
                throw LIBRPA_INVALID_ARGUMENT(
                    "pair-density vertex wrapper does not support twisted BvK k meshes");

    std::vector<ComplexMatrix> T_R_site;
    T_R_site.reserve(pbc.Rlist.size());
    for (const auto &A : pbc.Rlist)
    {
        const int iR = pbc.get_R_index(wrap_bvk({-A.x, -A.y, -A.z}, pbc.period));
        if (iR < 0 || static_cast<std::size_t>(iR) >= pbc.Rlist.size())
            throw LIBRPA_RUNTIME_ERROR("pair-density vertex failed to wrap -A onto the BvK grid");
        T_R_site.push_back(subspace.get_T_site(iR, site_index));
    }
    return build_site_pair_density_vertex_from_T(
        site_index, sites[static_cast<std::size_t>(site_index)], T_R_site, Cs_complete, basis_wfc,
        basis_abf, pbc, qpoints);
}

SitePairDensityVertex transform_site_pair_density_vertex_auxiliary_basis(
    const SitePairDensityVertex &parent, const AtomicBasis &parent_basis_abf,
    const AtomicBasis &active_basis_abf,
    const std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    const auto max_int = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (!parent_basis_abf.initialized() || !active_basis_abf.initialized() ||
        parent_basis_abf.n_atoms == 0 || parent_basis_abf.n_atoms != active_basis_abf.n_atoms ||
        parent_basis_abf.n_atoms > max_int || parent_basis_abf.nb_total == 0 ||
        active_basis_abf.nb_total == 0 || parent_basis_abf.nb_total > max_int ||
        active_basis_abf.nb_total > max_int)
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density auxiliary transform requires compatible initialized bases");
    if (parent.site_index < 0 || parent.atom_index < 0 ||
        static_cast<std::size_t>(parent.atom_index) >= parent_basis_abf.n_atoms ||
        parent.n_orbitals <= 0 || parent.label.empty() ||
        (!parent.orbital_labels.empty() &&
         parent.orbital_labels.size() != static_cast<std::size_t>(parent.n_orbitals)))
        throw LIBRPA_INVALID_ARGUMENT(
            "pair-density auxiliary transform received invalid site metadata");
    const auto norb = static_cast<std::size_t>(parent.n_orbitals);
    if (norb > max_int / norb)
        throw LIBRPA_INVALID_ARGUMENT("pair-density auxiliary transform pair count overflows");
    const auto npair_size = norb * norb;
    if (npair_size > max_int / parent_basis_abf.nb_total ||
        npair_size > max_int / active_basis_abf.nb_total ||
        active_basis_abf.nb_total > max_int / parent_basis_abf.nb_total)
        throw LIBRPA_INVALID_ARGUMENT("pair-density auxiliary transform matrix size overflows");
    if (parent.q_blocks.empty())
        throw LIBRPA_INVALID_ARGUMENT("pair-density auxiliary transform has no parent q grid");

    const int npair = static_cast<int>(npair_size);
    const int nparent = static_cast<int>(parent_basis_abf.nb_total);
    const int nactive = static_cast<int>(active_basis_abf.nb_total);
    SitePairDensityVertex result;
    result.site_index = parent.site_index;
    result.atom_index = parent.atom_index;
    result.n_orbitals = parent.n_orbitals;
    result.label = parent.label;
    result.orbital_labels = parent.orbital_labels;
    for (const auto &[q, blocks] : parent.q_blocks)
    {
        if (!finite(q))
            throw LIBRPA_INVALID_ARGUMENT("pair-density auxiliary transform has non-finite q");
        const auto it_L = sinvS.find(q);
        if (it_L == sinvS.end())
            throw LIBRPA_INVALID_ARGUMENT(
                "pair-density auxiliary transform is missing a required q");
        const auto &L = it_L->second;
        if (L.nr != nactive || L.nc != nparent || L.size != nactive * nparent || L.c == nullptr)
            throw LIBRPA_INVALID_ARGUMENT("pair-density auxiliary transform L has wrong shape");
        validate_finite_matrix(L, "pair-density auxiliary transform L");

        ComplexMatrix Dparent(npair, nparent);
        for (const auto &[I, D] : blocks)
        {
            if (I < 0 || static_cast<std::size_t>(I) >= parent_basis_abf.n_atoms)
                throw LIBRPA_INVALID_ARGUMENT(
                    "pair-density auxiliary transform atom is out of range");
            const int width = static_cast<int>(parent_basis_abf.get_atom_nb(I));
            if (D.nr != npair || D.nc != width || D.size != npair * width ||
                (D.size > 0 && D.c == nullptr))
                throw LIBRPA_INVALID_ARGUMENT("pair-density auxiliary transform D has wrong shape");
            validate_finite_matrix(D, "pair-density auxiliary transform D");
            const int begin = static_cast<int>(parent_basis_abf.get_part_range().at(I));
            for (int ab = 0; ab != npair; ++ab)
                for (int mu = 0; mu != width; ++mu) Dparent(ab, begin + mu) = D(ab, mu);
        }
        // Column coefficients obey d_active=L(q)d_parent; rows use transpose.
        const auto Dactive = Dparent * transpose(L, false);
        validate_finite_matrix(Dactive, "pair-density auxiliary transform result");
        auto &active_blocks = result.q_blocks[q];
        for (int I = 0; I != static_cast<int>(active_basis_abf.n_atoms); ++I)
        {
            const int width = static_cast<int>(active_basis_abf.get_atom_nb(I));
            if (width == 0) continue;
            const int begin = static_cast<int>(active_basis_abf.get_part_range().at(I));
            auto &D = active_blocks.emplace(I, ComplexMatrix(npair, width)).first->second;
            for (int ab = 0; ab != npair; ++ab)
                for (int mu = 0; mu != width; ++mu) D(ab, mu) = Dactive(ab, begin + mu);
        }
    }
    return result;
}

ComplexMatrix contract_rectangular_ordered_pair_vertex(const ComplexMatrix &D_left,
                                                       const ComplexMatrix &W,
                                                       const ComplexMatrix &D_right,
                                                       const int n_left, const int n_right)
{
    if (n_left <= 0 || n_right <= 0)
        throw LIBRPA_INVALID_ARGUMENT(
            "rectangular ordered-pair contraction requires positive orbital counts");
    const int left_pairs = n_left * n_left;
    const int right_pairs = n_right * n_right;
    if (D_left.nr != left_pairs || D_left.nc != W.nr || D_right.nr != right_pairs ||
        D_right.nc != W.nc)
        throw LIBRPA_INVALID_ARGUMENT(
            "rectangular ordered-pair contraction received inconsistent dimensions");
    validate_finite_matrix(D_left, "rectangular ordered-pair left D");
    validate_finite_matrix(W, "rectangular ordered-pair W");
    validate_finite_matrix(D_right, "rectangular ordered-pair right D");

    // Keep the ordered density convention: conjugate ba on the left,
    // and transpose (without conjugation) the right density vertex.
    // Factoring the auxiliary sums into GEMMs avoids O(n_left^2 n_right^2
    // n_aux^2) scalar work for every site, frequency, and spin pair.
    ComplexMatrix left(left_pairs, D_left.nc);
    for (int a = 0; a != n_left; ++a)
        for (int b = 0; b != n_left; ++b)
            for (int mu = 0; mu != D_left.nc; ++mu)
                left(a * n_left + b, mu) = std::conj(D_left(b * n_left + a, mu));
    ComplexMatrix result = (left * W) * transpose(D_right, false);
    validate_finite_matrix(result, "rectangular ordered-pair result");
    return result;
}

PairDensityVertex::PairDensityVertex(CorrelatedSubspace &subspace, const Cs_LRI &Cs,
                                     const AtomicBasis &basis_wfc, const AtomicBasis &basis_abf,
                                     const PeriodicBoundaryData &pbc,
                                     const std::vector<Vector3_Order<double>> &qpoints,
                                     const MpiCommHandler &comm_h)
{
    if (!comm_h.is_initialized())
        throw LIBRPA_INVALID_ARGUMENT("PairDensityVertex requires an initialized MPI communicator");

    std::string local_error;
    // Empty MPI owners have no representation to contradict LIBRI routing.
    // Only owners of an actual block receive the storage tag from input APIs.
    if (!Cs.data_IJR.empty() || (!Cs.data_libri.empty() && !Cs.use_libri))
        local_error = "projector pair-density vertex requires LIBRPA_ROUTING_LIBRI Cs storage";
    try
    {
        validate_bvk_and_q_grid(pbc, qpoints);
    }
    catch (const std::exception &error)
    {
        if (local_error.empty()) local_error = error.what();
    }
    catch (...)
    {
        if (local_error.empty())
            local_error = "unknown pair-density vertex grid validation failure";
    }
    collective_stage_error(comm_h, "PairDensityVertex initial validation", local_error);

#ifndef LIBRPA_USE_LIBRI
    (void)basis_wfc;
    (void)basis_abf;
    collective_stage_error(comm_h, "PairDensityVertex LibRI availability",
                           "projector pair-density vertex requires a build with LibRI enabled");
#else
    std::set<int> support;
    local_error.clear();
    try
    {
        if (subspace.get_R_list() != pbc.Rlist)
            local_error = "PairDensityVertex subspace and Dataset use different R grids";
        else if (subspace.get_sites().empty())
            local_error = "PairDensityVertex has no configured sites";
        else
        {
            for (std::size_t iR = 0; iR != pbc.Rlist.size(); ++iR)
            {
                const auto &T = subspace.get_T(static_cast<int>(iR));
                const int nao = static_cast<int>(basis_wfc.nb_total);
                if (nao <= 0 || (T.nr != nao && T.nr != 2 * nao))
                {
                    local_error = "PairDensityVertex subspace T(R) has the wrong AO dimension";
                    break;
                }
                const auto support_R = support_atoms({T}, basis_wfc);
                support.insert(support_R.begin(), support_R.end());
            }
            if (local_error.empty() && support.empty())
                local_error = "PairDensityVertex correlated subspace has empty AO support";
        }
    }
    catch (const std::exception &error)
    {
        local_error = error.what();
    }
    catch (...)
    {
        local_error = "unknown PairDensityVertex support validation failure";
    }
    collective_stage_error(comm_h, "PairDensityVertex support validation", local_error);

    // The LibRI collective receives the support set as a communication
    // parameter.  It must be identical on every rank; a local T(R) mismatch
    // would otherwise make ranks execute different collective schedules.
    std::vector<int> local_support_atoms(support.begin(), support.end());
    const int local_support_count =
        local_support_atoms.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max())
            ? static_cast<int>(local_support_atoms.size())
            : -1;
    std::vector<int> support_counts(static_cast<std::size_t>(comm_h.nprocs), 0);
    MPI_Allgather(&local_support_count, 1, MPI_INT, support_counts.data(), 1, MPI_INT, comm_h.comm);
    std::vector<int> support_displacements(support_counts.size(), 0);
    int total_support = 0;
    bool support_metadata_overflow = local_support_count < 0;
    for (std::size_t rank = 0; rank != support_counts.size(); ++rank)
    {
        if (support_counts[rank] < 0 ||
            total_support > std::numeric_limits<int>::max() - support_counts[rank])
        {
            support_metadata_overflow = true;
            break;
        }
        support_displacements[rank] = total_support;
        total_support += support_counts[rank];
    }
    local_error.clear();
    if (support_metadata_overflow)
        local_error = "PairDensityVertex support metadata exceeds MPI count capacity";
    collective_stage_error(comm_h, "PairDensityVertex support metadata", local_error);

    // Keep valid addresses for zero-count MPI arguments.  This matters for a
    // rank with an empty local support set even though the collective ignores
    // the corresponding buffer contents.
    int support_dummy = 0;
    std::vector<int> all_support_atoms(static_cast<std::size_t>(total_support), 0);
    MPI_Allgatherv(local_support_count == 0 ? &support_dummy : local_support_atoms.data(),
                   local_support_count, MPI_INT,
                   total_support == 0 ? &support_dummy : all_support_atoms.data(),
                   support_counts.data(), support_displacements.data(), MPI_INT, comm_h.comm);

    std::set<int> global_support(all_support_atoms.begin(), all_support_atoms.end());
    local_error.clear();
    for (const int atom : global_support)
    {
        if (atom < 0 || static_cast<std::size_t>(atom) >= basis_wfc.n_atoms)
        {
            local_error = "PairDensityVertex support contains an atom outside the AO basis";
            break;
        }
    }
    if (local_error.empty() && global_support != support)
        local_error = "PairDensityVertex correlated AO support differs between MPI ranks";
    collective_stage_error(comm_h, "PairDensityVertex support consensus", local_error);

    // The production LibRI reader distributes each Cs key uniquely.  Request
    // the complete relevant map on every rank.  A parallel scalar ownership
    // map detects accidental replicated keys before they could be summed.
    std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<int>>> local_owners;
    for (const auto &[I, JR_Cs] : Cs.data_libri)
    {
        if (support.count(I) == 0) continue;
        for (const auto &[JR, tensor] : JR_Cs)
        {
            (void)tensor;
            if (support.count(JR.first) == 0) continue;
            auto value = std::make_shared<std::valarray<int>>(1, 1);
            local_owners[I][JR] = RI::Tensor<int>({1UL}, value);
        }
    }

    // Both LibRI collectives are reached unconditionally after the synchronized
    // support checks.  Post-collective validation is deferred until the second
    // communication has completed on every rank.
    const auto Cs_complete = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        comm_h.comm, Cs.data_libri, support, support);
    const auto owners_complete = RI::Communicate_Tensors_Map_Judge::comm_map2_first(
        comm_h.comm, local_owners, support, support);
    local_error.clear();
    if (Cs_complete.empty())
        local_error = "PairDensityVertex found no global Cs block on the correlated AO support";
    if (local_error.empty())
    {
        for (const auto &[I, JR_Cs] : Cs_complete)
            for (const auto &[JR, tensor] : JR_Cs)
            {
                (void)tensor;
                const auto it_I = owners_complete.find(I);
                if (it_I == owners_complete.end() || it_I->second.count(JR) == 0 ||
                    it_I->second.at(JR)(0) != 1)
                {
                    local_error =
                        "PairDensityVertex requires exactly one MPI owner for every Cs key";
                    break;
                }
            }
    }
    collective_stage_error(comm_h, "PairDensityVertex LibRI result validation", local_error);

    const auto n_sites = subspace.get_sites().size();
    const int local_site_count =
        n_sites <= static_cast<std::size_t>(std::numeric_limits<int>::max())
            ? static_cast<int>(n_sites)
            : -1;
    int min_site_count = 0;
    int max_site_count = 0;
    MPI_Allreduce(&local_site_count, &min_site_count, 1, MPI_INT, MPI_MIN, comm_h.comm);
    MPI_Allreduce(&local_site_count, &max_site_count, 1, MPI_INT, MPI_MAX, comm_h.comm);
    local_error.clear();
    if (min_site_count != max_site_count || min_site_count < 0)
        local_error = "PairDensityVertex site count differs between MPI ranks";
    collective_stage_error(comm_h, "PairDensityVertex site-count validation", local_error);

    try
    {
        sites_.reserve(static_cast<std::size_t>(min_site_count));
    }
    catch (const std::exception &error)
    {
        local_error = error.what();
    }
    catch (...)
    {
        local_error = "unknown PairDensityVertex site allocation failure";
    }
    collective_stage_error(comm_h, "PairDensityVertex site allocation", local_error);

    for (int isite = 0; isite != min_site_count; ++isite)
    {
        SitePairDensityVertex site_result;
        local_error.clear();
        try
        {
            site_result = build_site_pair_density_vertex(subspace, isite, Cs_complete, basis_wfc,
                                                         basis_abf, pbc, qpoints);
        }
        catch (const std::exception &error)
        {
            local_error = error.what();
        }
        catch (...)
        {
            local_error = "unknown PairDensityVertex site build failure";
        }
        collective_stage_error(comm_h, "PairDensityVertex site build", local_error);
        sites_.push_back(std::move(site_result));
    }
#endif
}

const SitePairDensityVertex &PairDensityVertex::site(const int site_index) const
{
    if (site_index < 0 || site_index >= static_cast<int>(sites_.size()))
        throw LIBRPA_INVALID_ARGUMENT("PairDensityVertex site index is out of range");
    return sites_[static_cast<std::size_t>(site_index)];
}

}  // namespace librpa_int
