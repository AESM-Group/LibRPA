#include <algorithm>
#include <cmath>
#include <complex>
#include <set>
#include <string>

#include "../io/global_io.h"
#include "../utils/constants.h"
#include "../utils/error.h"
#include "chi0.h"

#ifdef LIBRPA_USE_LIBRI
#include <RI/comm/mix/Communicate_Tensors_Map_Judge.h>
#endif

namespace librpa_int
{
namespace
{
using Complex = std::complex<double>;

void check_collective(const MpiCommHandler &comm, const std::string &error)
{
    int failed = !error.empty(), any_failed = 0;
    comm.allreduce(&failed, &any_failed, 1, MPI_MAX);
    if (any_failed)
        throw LIBRPA_RUNTIME_ERROR("Conventional chi0: " +
                                   (error.empty() ? "another MPI rank rejected the input" : error));
}

// Match modulo a reciprocal lattice vector, retaining the input k ordering.
int find_k_plus_q(const PeriodicBoundaryData &pbc, int k, const Vector3_Order<double> &q)
{
    const auto target = pbc.latvec * (pbc.klist[k] + q);
    for (int j = 0; j < static_cast<int>(pbc.klist.size()); ++j)
    {
        const auto delta = target - pbc.latvec * pbc.klist[j];
        if (std::abs(delta.x - std::round(delta.x)) < 1.e-8 &&
            std::abs(delta.y - std::round(delta.y)) < 1.e-8 &&
            std::abs(delta.z - std::round(delta.z)) < 1.e-8)
            return j;
    }
    throw LIBRPA_RUNTIME_ERROR("q does not connect points of the loaded full k mesh");
}

#ifdef LIBRPA_USE_LIBRI
using CsMap = decltype(Cs_LRI::data_libri);
using FourierCs = std::map<int, std::map<int, ComplexMatrix>>;

FourierCs fourier_cs(const CsMap &cs, const AtomicBasis &ao, const AtomicBasis &aux,
                     const PeriodicBoundaryData &pbc, const Vector3_Order<double> &k)
{
    FourierCs result;
    for (const auto &[I, blocks] : cs)
        for (const auto &[JR, tensor] : blocks)
        {
            const int J = JR.first;
            auto &block = result[I][J];
            if (block.size == 0) block.create(aux[I], ao[I] * ao[J]);
            const Vector3_Order<int> R{JR.second[0], JR.second[1], JR.second[2]};
            const auto phase = std::polar(1.0, TWO_PI * (k * (R * pbc.latvec)));
            for (int i = 0; i < block.size; ++i) block.c[i] += phase * (*tensor.data)[i];
        }
    return result;
}

// For a fixed initial band, build all final-band transition densities:
// D_mu(nk,mkq) = sum C_mu(Ii,Jj,R) [conj(c_nk,Ii) c_mkq,Jj exp(ikq.R)
//                                 +conj(c_nk,Jj) c_mkq,Ii exp(-ik.R)].
// The two AO orderings must be retained even for I == J.
std::map<int, ComplexMatrix> transition_vertices(const FourierCs &cs_kq,
                                                 const FourierCs &cs_minus_k,
                                                 const std::set<int> &atoms, const AtomicBasis &ao,
                                                 const AtomicBasis &aux, const ComplexMatrix &left,
                                                 const ComplexMatrix &right_transpose, int band)
{
    std::map<int, ComplexMatrix> result;
    for (const int I : atoms)
    {
        ComplexMatrix half(aux[I], ao.nb_total);
        const auto found = cs_kq.find(I);
        if (found != cs_kq.end())
            for (const auto &[J, forward] : found->second)
            {
                const auto &reverse = cs_minus_k.at(I).at(J);
                const int ni = ao[I], nj = ao[J];
                const int i0 = ao.get_part_range()[I], j0 = ao.get_part_range()[J];
                for (int mu = 0; mu < half.nr; ++mu)
                    for (int i = 0; i < ni; ++i)
                        for (int j = 0; j < nj; ++j)
                        {
                            half(mu, j0 + j) +=
                                std::conj(left(band, i0 + i)) * forward(mu, i * nj + j);
                            half(mu, i0 + i) +=
                                std::conj(left(band, j0 + j)) * reverse(mu, i * nj + j);
                        }
            }
        result.emplace(I, half * right_transpose);
    }
    return result;
}
#endif
}  // namespace

void Chi0::build_chi0_q_conventional(const Cs_LRI &Cs, const std::vector<atpair_t> &pairs)
{
#ifndef LIBRPA_USE_LIBRI
    throw LIBRPA_RUNTIME_ERROR("Conventional chi0 requires a build with LibRI enabled");
#else
    const int nk = mf.get_n_kpoints(), nb = mf.get_n_bands();
    const auto frequencies = tfg.get_freq_nodes();
    const auto &qpoints = active_qpoints();
    const auto &energies = mf.get_eigenvals();
    const auto &weights = mf.get_weight();
    const auto &wfc = mf.get_eigenvectors();
    std::vector<std::vector<int>> kq_indices(qpoints.size(), std::vector<int>(nk));
    std::string error;
    try
    {
        if (is_mf_eigvec_k_distributed_ || mf.get_n_spinor() != 1 || use_symmetry_context)
            throw LIBRPA_RUNTIME_ERROR(
                "requires replicated scalar/collinear eigenvectors and use_symmetry_rpa=false");
        if (nk != pbc.get_n_cells_bvk() || static_cast<int>(pbc.klist.size()) != nk)
            throw LIBRPA_RUNTIME_ERROR("requires an unreduced full k mesh");
        if (!Cs.data_IJR.empty())
            throw LIBRPA_RUNTIME_ERROR("requires parallel_routing=libri and distributed LibRI Cs");
        for (const double omega : frequencies)
            if (!std::isfinite(omega) || omega <= 0)
                throw LIBRPA_RUNTIME_ERROR("requires strictly positive imaginary frequencies");
        for (std::size_t iq = 0; iq < qpoints.size(); ++iq)
            for (int k = 0; k < nk; ++k) kq_indices[iq][k] = find_k_plus_q(pbc, k, qpoints[iq]);
        for (int s = 0; s < mf.get_n_spins(); ++s)
            for (int k = 0; k < nk; ++k)
            {
                const auto &c = wfc.at(s).at(0).at(k);
                if (c.nr != nb || c.nc != static_cast<int>(atbasis_wfc.nb_total))
                    throw LIBRPA_RUNTIME_ERROR("missing full band/AO eigenvector matrix");
                for (int n = 0; n < nb; ++n)
                    if (!std::isfinite(energies[s](k, n)) || !std::isfinite(weights[s](k, n)))
                        throw LIBRPA_RUNTIME_ERROR("non-finite eigenvalue or occupation");
            }
        for (const auto &[I, blocks] : Cs.data_libri)
            for (const auto &[JR, tensor] : blocks)
            {
                const int J = JR.first;
                if (tensor.data == nullptr || tensor.shape.size() != 3 ||
                    tensor.shape[0] != atbasis_abf[I] || tensor.shape[1] != atbasis_wfc[I] ||
                    tensor.shape[2] != atbasis_wfc[J] ||
                    tensor.data->size() != atbasis_abf[I] * atbasis_wfc[I] * atbasis_wfc[J])
                    throw LIBRPA_RUNTIME_ERROR("inconsistent RI tensor dimensions");
            }
    }
    catch (const std::exception &e)
    {
        error = e.what();
    }
    check_collective(comm_h, error);

    std::set<int> needed_atoms, all_atoms;
    for (const auto &[I, J] : pairs)
    {
        needed_atoms.insert(I);
        needed_atoms.insert(J);
    }
    for (int I = 0; I < static_cast<int>(atbasis_wfc.n_atoms); ++I) all_atoms.insert(I);
    // Each rank requests the first-ABF atoms needed by its owned upper blocks.
    // Input Cs follow the existing LibRI reader's unique-key distribution.
    const auto cs = RI::Communicate_Tensors_Map_Judge::comm_map2_first(comm_h.comm, Cs.data_libri,
                                                                       needed_atoms, all_atoms);
    if (comm_h.is_root()) global::lib_printf("Build chi0 by direct k-space band summation\n");

    // Process one q and one initial band at a time. No transition-pair squared
    // matrix and no cache of vertices for every (k,q,n,m) is allocated.
    for (std::size_t iq = 0; iq < qpoints.size(); ++iq)
    {
        const auto &q = qpoints[iq];
        std::vector<atom_mapping<ComplexMatrix>::pair_t_old> odd(frequencies.size());
        for (std::size_t iw = 0; iw < frequencies.size(); ++iw)
        {
            auto &blocks = chi0_q[frequencies[iw]][q];
            for (const auto &[I, J] : pairs)
            {
                blocks[I][J].create(atbasis_abf[I], atbasis_abf[J]);
                odd[iw][I][J].create(atbasis_abf[I], atbasis_abf[J]);
            }
        }
        for (int k = 0; k < nk; ++k)
        {
            const int kq = kq_indices[iq][k];
            const auto forward = fourier_cs(cs, atbasis_wfc, atbasis_abf, pbc, pbc.klist[kq]);
            const auto reverse = fourier_cs(cs, atbasis_wfc, atbasis_abf, pbc, -pbc.klist[k]);
            for (int s = 0; s < mf.get_n_spins(); ++s)
            {
                const auto &left = wfc.at(s).at(0).at(k);
                const auto right = transpose(wfc.at(s).at(0).at(kq), false);
                for (int n = 0; n < nb; ++n)
                {
                    if (!band_mask.empty() && !band_mask[n]) continue;
                    const auto vertices = transition_vertices(
                        forward, reverse, needed_atoms, atbasis_wfc, atbasis_abf, left, right, n);
                    for (std::size_t iw = 0; iw < frequencies.size(); ++iw)
                    {
                        const double omega = frequencies[iw];
                        std::vector<Complex> factors(nb, 0.0);
                        for (int m = 0; m < nb; ++m)
                        {
                            if (!band_mask.empty() && !band_mask[m]) continue;
                            // Stored weights already include spin multiplicity and 1/Nk;
                            // eigenvalues and omega are both in Hartree.
                            const double dw = weights[s](k, n) - weights[s](kq, m);
                            factors[m] =
                                dw / Complex(energies[s](k, n) - energies[s](kq, m), omega);
                        }
                        for (const auto &[I, J] : pairs)
                        {
                            auto real_part = vertices.at(I), imag_part = vertices.at(I);
                            for (int mu = 0; mu < real_part.nr; ++mu)
                                for (int m = 0; m < nb; ++m)
                                {
                                    real_part(mu, m) *= factors[m].real();
                                    imag_part(mu, m) *= factors[m].imag();
                                }
                            const auto dagger = transpose(vertices.at(J), true);
                            chi0_q[omega][q][I][J] += real_part * dagger;
                            odd[iw][I][J] += imag_part * dagger;
                        }
                    }
                }
            }
        }
        // The existing energy kernels reconstruct the lower atom triangle by
        // adjoint. Reject a response with a nonzero odd-frequency component,
        // rather than silently discarding it for time-reversal-broken inputs.
        double residual = 0.0, scale = 1.0;
        error.clear();
        for (std::size_t iw = 0; iw < frequencies.size(); ++iw)
            for (const auto &[I, J] : pairs)
            {
                const auto &even = chi0_q[frequencies[iw]][q][I][J];
                const auto &part = odd[iw][I][J];
                for (int j = 0; j < part.size; ++j)
                {
                    if (!std::isfinite(std::abs(part.c[j])) || !std::isfinite(std::abs(even.c[j])))
                        error = "non-finite response matrix";
                    residual = std::max(residual, std::abs(part.c[j]));
                    scale = std::max(scale, std::abs(even.c[j]));
                }
            }
        if (error.empty() && residual > 1.e-9 * scale)
            error =
                "non-Hermitian imaginary-frequency response; time-reversal-broken inputs are "
                "unsupported";
        check_collective(comm_h, error);
    }
#endif
}
}  // namespace librpa_int
