#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <exception>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "../io/global_io.h"
#include "../mpi/base_mpi.h"
#include "../utils/constants.h"
#include "../utils/error.h"
#include "chi0.h"

namespace librpa_int
{
namespace
{
using Complex = std::complex<double>;

// Position of a fractional k point on the BvK mesh, modulo a reciprocal lattice
// vector, flattened in the order of the period.
int mesh_index(const PeriodicBoundaryData &pbc, const Vector3<double> &kfrac)
{
    const Vector3<double> n(kfrac.x * pbc.period.x, kfrac.y * pbc.period.y,
                            kfrac.z * pbc.period.z);
    if (!nearly_integer_vector(n, 1.e-5))
        throw LIBRPA_RUNTIME_ERROR("k or k + q is not on the BvK mesh");
    const auto i = round_to_integer_vector(n);
    const auto wrap = [](int a, int p) { return ((a % p) + p) % p; };
    return (wrap(i.x, pbc.period.x) * pbc.period.y + wrap(i.y, pbc.period.y)) * pbc.period.z +
           wrap(i.z, pbc.period.z);
}

// Index of k + q in the loaded k list for each k. set_kgrids_kvec has snapped
// kfrac_list to the mesh and checked that the loaded points cover it once.
std::vector<int> k_plus_q_indices(const PeriodicBoundaryData &pbc,
                                  const std::vector<int> &loaded_at_mesh,
                                  const Vector3_Order<double> &q)
{
    const auto qfrac = pbc.latvec * q;
    std::vector<int> result(pbc.kfrac_list.size());
    for (std::size_t k = 0; k < result.size(); ++k)
        result[k] = loaded_at_mesh[mesh_index(pbc, pbc.kfrac_list[k] + qfrac)];
    return result;
}

// I, J, R
using CsKey = std::array<int, 5>;
// A local block in either storage format of Cs_LRI
struct CsSource
{
    const double *data;
    bool mu_first;
};

// Cs_LRI keeps the blocks in data_libri or data_IJR, as use_libri selects.
std::map<CsKey, CsSource> local_cs_blocks(const Cs_LRI &Cs)
{
    std::map<CsKey, CsSource> blocks;
    if (Cs.use_libri)
        for (const auto &[I, blocks_JR] : Cs.data_libri)
            for (const auto &[JR, tensor] : blocks_JR)
                blocks[{I, JR.first, JR.second[0], JR.second[1], JR.second[2]}] = {
                    &(*tensor.data)[0], true};
    else
        for (const auto &[I, J_blocks] : Cs.data_IJR)
            for (const auto &[J, R_blocks] : J_blocks)
                for (const auto &[R, block] : R_blocks)
                    blocks[{static_cast<int>(I), static_cast<int>(J), R.x, R.y, R.z}] = {
                        block->c, false};
    return blocks;
}

// Copy to the (mu, i * nj + j) order of the Fourier-transformed blocks.
void append_cs_block(const CsSource &source, std::size_t n_mu, std::size_t n_ij,
                     std::vector<double> &out)
{
    const std::size_t begin = out.size();
    out.resize(begin + n_mu * n_ij);
    for (std::size_t mu = 0; mu < n_mu; ++mu)
        for (std::size_t ij = 0; ij < n_ij; ++ij)
            out[begin + mu * n_ij + ij] =
                source.mu_first ? source.data[mu * n_ij + ij] : source.data[ij * n_mu + mu];
}

struct CsBlock
{
    CsKey key;
    std::vector<double> data;
};

// Send a list of integers to each rank and receive one from each rank.
std::vector<std::vector<int>> exchange_lists(const MpiCommHandler &comm,
                                             const std::vector<std::vector<int>> &send)
{
    const int nprocs = comm.nprocs;
    std::vector<int> send_counts(nprocs), recv_counts(nprocs), send_displs(nprocs, 0),
        recv_displs(nprocs, 0);
    for (int rank = 0; rank < nprocs; ++rank) send_counts[rank] = static_cast<int>(send[rank].size());
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm.comm);
    std::partial_sum(send_counts.begin(), send_counts.end() - 1, send_displs.begin() + 1);
    std::partial_sum(recv_counts.begin(), recv_counts.end() - 1, recv_displs.begin() + 1);
    std::vector<int> send_flat, recv_flat(recv_displs.back() + recv_counts.back());
    for (const auto &list : send) send_flat.insert(send_flat.end(), list.begin(), list.end());
    MPI_Alltoallv(send_flat.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_flat.data(), recv_counts.data(), recv_displs.data(), MPI_INT, comm.comm);
    std::vector<std::vector<int>> recv(nprocs);
    for (int rank = 0; rank < nprocs; ++rank)
        recv[rank].assign(recv_flat.begin() + recv_displs[rank],
                          recv_flat.begin() + recv_displs[rank] + recv_counts[rank]);
    return recv;
}

// Give each rank all blocks whose first atom is in needed_atoms. The readers and
// the API can place one block on several ranks, so each key is sent by one
// owner, the lowest rank that holds it. The rank I % nprocs collects the keys
// of atom I and selects the owner, so no rank holds the keys of all atoms.
std::vector<CsBlock> collect_cs(const MpiCommHandler &comm, const Cs_LRI &Cs,
                                const std::set<int> &needed_atoms, const AtomicBasis &ao,
                                const AtomicBasis &aux)
{
    const int nprocs = comm.nprocs, myid = comm.myid, n_atoms = ao.n_atoms;
    const auto local = local_cs_blocks(Cs);

    std::vector<char> needed(n_atoms, 0), all_needed(n_atoms * nprocs);
    for (const int I : needed_atoms) needed[I] = 1;
    MPI_Allgather(needed.data(), n_atoms, MPI_CHAR, all_needed.data(), n_atoms, MPI_CHAR, comm.comm);

    // Register the local keys at the directory rank of their first atom.
    std::vector<std::vector<int>> to_directory(nprocs);
    for (const auto &[key, source] : local)
        to_directory[key[0] % nprocs].insert(to_directory[key[0] % nprocs].end(), key.begin(),
                                             key.end());
    const auto registered = exchange_lists(comm, to_directory);
    // Holders are in increasing rank order, because the ranks are visited in order.
    std::map<CsKey, std::vector<int>> holders;
    for (int rank = 0; rank < nprocs; ++rank)
        for (std::size_t i = 0; i < registered[rank].size(); i += 5)
            holders[{registered[rank][i], registered[rank][i + 1], registered[rank][i + 2],
                     registered[rank][i + 3], registered[rank][i + 4]}]
                .push_back(rank);

    // Tell each owner the key and the destination of each transfer.
    std::vector<std::vector<int>> to_owner(nprocs);
    for (const auto &[key, ranks] : holders)
        for (int dest = 0; dest < nprocs; ++dest)
            if (all_needed[dest * n_atoms + key[0]] &&
                !std::binary_search(ranks.begin(), ranks.end(), dest))
            {
                auto &order = to_owner[ranks.front()];
                order.insert(order.end(), key.begin(), key.end());
                order.push_back(dest);
            }
    const auto orders = exchange_lists(comm, to_owner);
    std::vector<std::vector<int>> send_keys(nprocs);
    for (const auto &list : orders)
        for (std::size_t i = 0; i < list.size(); i += 6)
            send_keys[list[i + 5]].insert(send_keys[list[i + 5]].end(), list.begin() + i,
                                          list.begin() + i + 5);
    // Each receiver gets the keys in the order of the payload.
    const auto recv_keys = exchange_lists(comm, send_keys);

    const auto block_size = [&](const int *key)
    { return static_cast<std::size_t>(aux[key[0]] * ao[key[0]] * ao[key[1]]); };

    std::vector<CsBlock> result;
    for (const auto &[key, source] : local)
        if (needed[key[0]])
        {
            std::vector<double> data;
            append_cs_block(source, aux[key[0]], ao[key[0]] * ao[key[1]], data);
            result.push_back({key, std::move(data)});
        }

    // Pairwise rounds keep one send buffer in memory at a time.
    const std::size_t max_chunk = 1u << 30;
    for (int shift = 1; shift < nprocs; ++shift)
    {
        const int dest = (myid + shift) % nprocs, src = (myid + nprocs - shift) % nprocs;
        std::vector<double> send;
        for (std::size_t i = 0; i < send_keys[dest].size(); i += 5)
        {
            const int *k = &send_keys[dest][i];
            append_cs_block(local.at({k[0], k[1], k[2], k[3], k[4]}), aux[k[0]], ao[k[0]] * ao[k[1]],
                            send);
        }
        std::size_t n_recv = 0;
        for (std::size_t i = 0; i < recv_keys[src].size(); i += 5)
            n_recv += block_size(&recv_keys[src][i]);
        std::vector<double> recv(n_recv);
        std::vector<MPI_Request> requests;
        const auto post = [&](double *buffer, std::size_t size, bool is_send)
        {
            for (std::size_t offset = 0; offset < size; offset += max_chunk)
            {
                const int count = static_cast<int>(std::min(max_chunk, size - offset));
                requests.emplace_back();
                if (is_send)
                    MPI_Isend(buffer + offset, count, MPI_DOUBLE, dest, 0, comm.comm,
                              &requests.back());
                else
                    MPI_Irecv(buffer + offset, count, MPI_DOUBLE, src, 0, comm.comm,
                              &requests.back());
            }
        };
        post(recv.data(), recv.size(), false);
        post(send.data(), send.size(), true);
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);

        std::size_t offset = 0;
        for (std::size_t i = 0; i < recv_keys[src].size(); i += 5)
        {
            const int *k = &recv_keys[src][i];
            const std::size_t size = block_size(k);
            result.push_back({{k[0], k[1], k[2], k[3], k[4]},
                              std::vector<double>(recv.begin() + offset,
                                                  recv.begin() + offset + size)});
            offset += size;
        }
    }
    return result;
}

using FourierCs = std::map<int, std::map<int, ComplexMatrix>>;

FourierCs fourier_cs(const std::vector<CsBlock> &cs, const AtomicBasis &ao, const AtomicBasis &aux,
                     const PeriodicBoundaryData &pbc, const Vector3_Order<double> &k)
{
    FourierCs result;
    for (const auto &[key, data] : cs)
    {
        const int I = key[0], J = key[1];
        auto &block = result[I][J];
        if (block.size == 0) block.create(aux[I], ao[I] * ao[J]);
        const Vector3_Order<int> R{key[2], key[3], key[4]};
        const auto phase = std::polar(1.0, TWO_PI * (k * (R * pbc.latvec)));
        for (int i = 0; i < block.size; ++i) block.c[i] += phase * data[i];
    }
    return result;
}

// For a fixed initial band, build the transition densities of the final bands
// in the columns of right_transpose:
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
            for (const auto &[J, cs_forward] : found->second)
            {
                const ComplexMatrix &forward = cs_forward;
                const auto &reverse = cs_minus_k.at(I).at(J);
                const int ni = ao[I], nj = ao[J];
                const int i0 = ao.get_part_range()[I], j0 = ao.get_part_range()[J];
#pragma omp parallel for schedule(static)
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
}  // namespace

void Chi0::build_chi0_q_conventional(LibrpaParallelRouting routing, const Cs_LRI &Cs,
                                     const std::vector<atpair_t> &pairs)
{
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
        for (const double omega : frequencies)
            if (!std::isfinite(omega) || omega <= 0)
                throw LIBRPA_RUNTIME_ERROR("requires strictly positive imaginary frequencies");
        std::vector<int> loaded_at_mesh(nk, -1);
        for (int k = 0; k < nk; ++k) loaded_at_mesh[mesh_index(pbc, pbc.kfrac_list[k])] = k;
        for (std::size_t iq = 0; iq < qpoints.size(); ++iq)
            kq_indices[iq] = k_plus_q_indices(pbc, loaded_at_mesh, qpoints[iq]);
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
        // set_lri_coeff with different routings can fill both containers.
        if (Cs.use_libri ? !Cs.data_IJR.empty() : !Cs.data_libri.empty())
            throw LIBRPA_RUNTIME_ERROR("RI coefficients are stored for another parallel routing");
        for (const auto &[I, blocks] : Cs.data_libri)
            for (const auto &[JR, tensor] : blocks)
            {
                const int J = JR.first;
                if (tensor.data == nullptr ||
                    tensor.data->size() != atbasis_abf[I] * atbasis_wfc[I] * atbasis_wfc[J])
                    throw LIBRPA_RUNTIME_ERROR("inconsistent RI tensor dimensions");
            }
        for (const auto &[I, J_blocks] : Cs.data_IJR)
            for (const auto &[J, R_blocks] : J_blocks)
                for (const auto &[R, block] : R_blocks)
                    if (block == nullptr ||
                        static_cast<std::size_t>(block->nr) != atbasis_wfc[I] * atbasis_wfc[J] ||
                        static_cast<std::size_t>(block->nc) != atbasis_abf[I])
                        throw LIBRPA_RUNTIME_ERROR("inconsistent RI tensor dimensions");
    }
    catch (const std::exception &e)
    {
        error = std::string("Conventional chi0: ") + e.what();
    }
    check_chi0_collective_error(comm_h, error);

    // The R-tau routing keeps each chi0(q) on one rank with all atom pairs. The
    // other routings distribute the atom pairs and compute all q on each rank.
    std::vector<int> local_iqs(qpoints.size());
    std::iota(local_iqs.begin(), local_iqs.end(), 0);
    if (routing == LIBRPA_ROUTING_RTAU)
        local_iqs = dispatch_vector(local_iqs, comm_h.myid, comm_h.nprocs, true);

    std::set<int> needed_atoms;
    if (!local_iqs.empty())
        for (const auto &[I, J] : pairs)
        {
            needed_atoms.insert(I);
            needed_atoms.insert(J);
        }
    const auto cs = collect_cs(comm_h, Cs, needed_atoms, atbasis_wfc, atbasis_abf);
    if (comm_h.is_root()) global::lib_printf("Build chi0 by direct k-space band summation\n");

    // Process one k, one q and one initial band at a time. Only pairs with
    // different weights enter the products, which removes the
    // unoccupied-unoccupied block.
    const std::size_t nw = frequencies.size();
    const std::size_t n_local_q = local_iqs.size();
    std::vector<double> odd_max(qpoints.size(), 0.0), even_max(qpoints.size(), 0.0);
    error.clear();
    // The Im part of the response is accumulated at the first frequency only.
    std::vector<atom_mapping<ComplexMatrix>::pair_t_old> odd(n_local_q);
    for (std::size_t i = 0; i < n_local_q; ++i)
    {
        for (const auto &[I, J] : pairs) odd[i][I][J].create(atbasis_abf[I], atbasis_abf[J]);
        // A rank without atom pairs must keep chi0_q empty for the energy routines.
        if (!pairs.empty())
            for (const double omega : frequencies)
            {
                auto &blocks = chi0_q[omega][qpoints[local_iqs[i]]];
                for (const auto &[I, J] : pairs)
                    blocks[I][J].create(atbasis_abf[I], atbasis_abf[J]);
            }
    }
    for (int k = 0; k < nk; ++k)
    {
        // The transforms are made only when a transition needs them. The one
        // at -k serves all q.
        std::optional<FourierCs> reverse;
        for (std::size_t i = 0; i < n_local_q; ++i)
        {
            const int iq = local_iqs[i];
            const auto &q = qpoints[iq];
            const int kq = kq_indices[iq][k];
            std::optional<FourierCs> forward;
            for (int s = 0; s < mf.get_n_spins(); ++s)
            {
                const auto &left = wfc.at(s).at(0).at(k);
                const auto &right = wfc.at(s).at(0).at(kq);
                for (int n = 0; n < nb; ++n)
                {
                    if (!band_mask.empty() && !band_mask[n]) continue;
                    std::vector<int> final_bands;
                    for (int m = 0; m < nb; ++m)
                        if ((band_mask.empty() || band_mask[m]) &&
                            weights[s](k, n) != weights[s](kq, m))
                            final_bands.push_back(m);
                    if (final_bands.empty()) continue;
                    if (!reverse)
                        reverse = fourier_cs(cs, atbasis_wfc, atbasis_abf, pbc, -pbc.klist[k]);
                    if (!forward)
                        forward = fourier_cs(cs, atbasis_wfc, atbasis_abf, pbc, pbc.klist[kq]);

                    const int n_final = static_cast<int>(final_bands.size());
                    ComplexMatrix right_transpose(right.nc, n_final);
                    for (int a = 0; a < right.nc; ++a)
                        for (int im = 0; im < n_final; ++im)
                            right_transpose(a, im) = right(final_bands[im], a);
                    const auto vertices =
                        transition_vertices(*forward, *reverse, needed_atoms, atbasis_wfc,
                                            atbasis_abf, left, right_transpose, n);

                    // Stored weights already include spin multiplicity and 1/Nk;
                    // eigenvalues and omega are both in Hartree.
                    std::vector<Complex> factors(nw * n_final);
                    for (std::size_t iw = 0; iw < nw; ++iw)
                        for (int im = 0; im < n_final; ++im)
                        {
                            const int m = final_bands[im];
                            factors[iw * n_final + im] =
                                (weights[s](k, n) - weights[s](kq, m)) /
                                Complex(energies[s](k, n) - energies[s](kq, m), frequencies[iw]);
                        }

                    for (const auto &[I, J] : pairs)
                    {
                        const auto &vertex = vertices.at(I);
                        const auto dagger = transpose(vertices.at(J), true);
                        ComplexMatrix scaled(vertex.nr, vertex.nc);
                        for (std::size_t iw = 0; iw < nw; ++iw)
                        {
                            for (int mu = 0; mu < vertex.nr; ++mu)
                                for (int im = 0; im < n_final; ++im)
                                    scaled(mu, im) =
                                        vertex(mu, im) * factors[iw * n_final + im].real();
                            chi0_q[frequencies[iw]][q][I][J] += scaled * dagger;
                        }
                        for (int mu = 0; mu < vertex.nr; ++mu)
                            for (int im = 0; im < n_final; ++im)
                                scaled(mu, im) = vertex(mu, im) * factors[im].imag();
                        odd[i][I][J] += scaled * dagger;
                    }
                }
            }
        }
    }
    for (std::size_t i = 0; i < n_local_q; ++i)
    {
        const int iq = local_iqs[i];
        const auto &q = qpoints[iq];
        for (const auto &[I, J] : pairs)
        {
            for (const double omega : frequencies)
            {
                const auto &even = chi0_q[omega][q][I][J];
                for (int j = 0; j < even.size; ++j)
                    if (!std::isfinite(std::abs(even.c[j])))
                        error = "Conventional chi0: non-finite response matrix";
            }
            const auto &even = chi0_q[frequencies[0]][q][I][J];
            const auto &part = odd[i][I][J];
            for (int j = 0; j < part.size; ++j)
            {
                if (!std::isfinite(std::abs(part.c[j])))
                    error = "Conventional chi0: non-finite response matrix";
                odd_max[iq] = std::max(odd_max[iq], std::abs(part.c[j]));
                even_max[iq] = std::max(even_max[iq], std::abs(even.c[j]));
            }
        }
    }
    check_chi0_collective_error(comm_h, error);

    // The existing energy kernels reconstruct the lower atom triangle by adjoint,
    // so the Im part must cancel. Compare it with the response of all atom blocks
    // at the same q, because a single block can be small.
    const int nq = static_cast<int>(qpoints.size());
    comm_h.allreduce(MPI_IN_PLACE, odd_max.data(), nq, MPI_MAX);
    comm_h.allreduce(MPI_IN_PLACE, even_max.data(), nq, MPI_MAX);
    for (int iq = 0; iq < nq; ++iq)
        if (odd_max[iq] > 1.e-9 * even_max[iq])
            check_chi0_collective_error(
                comm_h,
                "Conventional chi0: non-Hermitian imaginary-frequency response. Possible causes "
                "are a k mesh without -k, a band cutoff inside a degenerate multiplet, and "
                "time-reversal-broken input");
}
}  // namespace librpa_int
