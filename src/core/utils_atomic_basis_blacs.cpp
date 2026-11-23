#include "utils_atomic_basis_blacs.h"

#include <algorithm>
#include <cassert>
// #include <iostream>

#include "../math/scalapack_connector.h"
#include "../mpi/base_mpi.h"
#include "../utils/base_utility.h"
#include "../utils/error.h"
#include "../utils/profiler.h"
#include "librpa_enums.h"

// #include "../io/stl_io_helper.h"

namespace librpa_int
{

std::set<std::pair<int, int>> get_necessary_IJ_from_block_2D(const AtomicBasis &atbasis_row, const AtomicBasis &atbasis_col, const ArrayDesc& arrdesc)
{
    std::set<std::pair<int, int>> IJs;
    assert(as_size(arrdesc.m()) == atbasis_row.nb_total);
    assert(as_size(arrdesc.n()) == atbasis_col.nb_total);

    const auto mlo = arrdesc.m_loc();
    const auto nlo = arrdesc.n_loc();
    size_t glo;
    int I, J;
    for (int ilo = 0; ilo != mlo; ilo++)
        for (int jlo = 0; jlo != nlo; jlo++)
        {
            glo = arrdesc.indx_l2g_r(ilo);
            I = atbasis_row.get_i_atom(glo);
            glo = arrdesc.indx_l2g_c(jlo);
            J = atbasis_col.get_i_atom(glo);
            IJs.insert({I, J});
        }
    return IJs;
}

std::set<std::pair<int, int>> get_necessary_IJ_from_block_2D_sy(const char &uplo, const AtomicBasis &atbasis, const ArrayDesc& arrdesc)
{
    std::set<std::pair<int, int>> IJs;
    if (uplo != 'U' && uplo != 'L')
        throw LIBRPA_RUNTIME_ERROR("uplo should be U or L");
    assert(as_size(arrdesc.m()) == atbasis.nb_total);
    assert(as_size(arrdesc.n()) == atbasis.nb_total);

    const auto mlo = arrdesc.m_loc();
    const auto nlo = arrdesc.n_loc();
    size_t glo;
    int I, J;
    for (int ilo = 0; ilo != mlo; ilo++)
        for (int jlo = 0; jlo != nlo; jlo++)
        {
            glo = arrdesc.indx_l2g_r(ilo);
            I = atbasis.get_i_atom(glo);
            glo = arrdesc.indx_l2g_c(jlo);
            J = atbasis.get_i_atom(glo);
            if (uplo == 'L')
                I > J? IJs.insert({I, J}): IJs.insert({J, I});
            else
                I > J? IJs.insert({J, I}): IJs.insert({I, J});
        }
    return IJs;
}

void
IndexScheduler::init_impl_(const ap_p_map<int> &map_IJ_proc,
                           const AtomicBasis &atbasis_r, const AtomicBasis &atbasis_c,
                           const ArrayDesc &ad, const bool row_major) noexcept
{
    row_major_ = row_major;

    // process -> local 1D indices in atomic basis context to send when converting from AP->BLACS
    std::unordered_map<int, std::vector<size_t>> proc_ap_ipair;
    std::unordered_map<int, std::vector<size_t>> proc_ap_locid;
    // process -> local 1D indices in BLACS context to send when converting from BLACS->AP
    std::unordered_map<int, std::vector<size_t>> proc_blacs_locid;

    const auto m = as_size(ad.m());
    const auto n = as_size(ad.n());
    const auto myid = ad.myid();

    global::profiler.start("index_scheduler_ids_rc", LIBRPA_VERBOSE_DEBUG);
    // std::unordered_map<atom_t, std::vector<size_t>> row_ids_ap;
    // std::unordered_map<atom_t, std::vector<size_t>> col_ids_ap;
    // for (size_t I = 0; I < atbasis_r.n_atoms; I++) row_ids_ap[I] = atbasis_r.get_global_indices(I);
    // for (size_t I = 0; I < atbasis_c.n_atoms; I++) col_ids_ap[I] = atbasis_c.get_global_indices(I);
    const auto &row_procs_blacs = ad.g2p_r();
    const auto &col_procs_blacs = ad.g2p_c();
    const auto &ir_blacs = ad.g2l_r();
    const auto &ic_blacs = ad.g2l_c();
    const auto &m_loc = ad.m_loc();
    const auto &n_loc = ad.n_loc();
    global::profiler.stop("index_scheduler_ids_rc");

    global::profiler.start("index_scheduler_compute_map", LIBRPA_VERBOSE_DEBUG); // most intensive part, not paralleized
    // const auto n_avg = m * n / ad.nprocs();
    // proc_ap_locid.reserve(n_avg);
    // proc_ap_ipair.reserve(n_avg);
    // proc_blacs_locid.reserve(n_avg);

    size_t I, loi, J, loj;
    if (row_major)
    {
        for (size_t i = 0; i < m; i++)
        {
            atbasis_r.get_local_index(i, I, loi);
            int prow = row_procs_blacs[i];
            for (size_t j = 0; j < n; j++)
            {
                atbasis_c.get_local_index(j, J, loj);
                atpair_t IJ{I, J};
                auto it = map_IJ_proc.find(IJ);
                if (it == map_IJ_proc.end()) continue; // no global data for this pair
                const auto proc_ap = it->second; // process that requires in atom-pair context
                int pcol = col_procs_blacs[j];
                const auto proc_blacs = ad.get_pnum(prow, pcol); // process that requires in BLACS context
                if (myid != proc_ap && myid != proc_blacs) continue; // not related with me
                if (myid == proc_ap && myid == proc_blacs) continue; // already on me, no need to communicate
                if (proc_blacs == myid) // need in BLACS context, to obtain from elsewhere
                {
                    const auto &ir = ir_blacs[i];
                    const auto &ic = ic_blacs[j];
                    proc_blacs_locid[proc_ap].push_back(ir * n_loc + ic);
                }
                if (proc_ap == myid) // need in AP context, to obtain from elsewhere
                {
                    const size_t ipair = std::distance(atpairs.cbegin(), std::find(atpairs.cbegin(), atpairs.cend(), IJ));
                    proc_ap_ipair[proc_blacs].push_back(ipair);
                    const auto nJ = atbasis_c.get_atom_nb(J);
                    proc_ap_locid[proc_blacs].push_back(loi * nJ + loj);
                }
            }
        }
    }
    else
    {
        for (size_t j = 0; j < n; j++)
        {
            atbasis_c.get_local_index(j, J, loj);
            int pcol = col_procs_blacs[j];
            for (size_t i = 0; i < m; i++)
            {
                atbasis_r.get_local_index(i, I, loi);
                atpair_t IJ{I, J};
                auto it = map_IJ_proc.find(IJ);
                if (it == map_IJ_proc.end()) continue; // no global data for this pair
                const auto proc_ap = it->second; // process that requires in atom-pair context
                int prow = row_procs_blacs[i];
                auto proc_blacs = ad.get_pnum(prow, pcol); // process that requires in BLACS context
                if (myid != proc_ap && myid != proc_blacs) continue; // not related with me
                if (myid == proc_ap && myid == proc_blacs) continue; // already on me, no need to communicate
                if (proc_blacs == myid) // contained here in BLACS context, required by elsewhere in AP context
                {
                    const auto &ir = ir_blacs[i];
                    const auto &ic = ic_blacs[j];
                    proc_blacs_locid[proc_ap].push_back(ic * m_loc + ir);
                }
                if (proc_ap == myid) // contained here in AP context, required by elsewhere in BLACS context
                {
                    size_t ipair = std::distance(atpairs.cbegin(), std::find(atpairs.cbegin(), atpairs.cend(), IJ));
                    proc_ap_ipair[proc_blacs].push_back(ipair);
                    const auto nI = atbasis_c.get_atom_nb(I);
                    proc_ap_locid[proc_blacs].push_back(loj * nI + loi);
                }
            }
        }
    }
    global::profiler.stop("index_scheduler_compute_map");

    const auto nprocs = ad.nprocs();
    disp_ap.resize(nprocs, 0);
    counts_ap.resize(nprocs, 0);
    disp_blacs.resize(nprocs, 0);
    counts_blacs.resize(nprocs, 0);

    global::profiler.start("index_scheduler_flatten", LIBRPA_VERBOSE_DEBUG);
    // flatten the map and save
    total_count_ap = 0;
    for (int i = 0; i < nprocs; i++)
    {
        disp_ap[i] = total_count_ap;
        if (i == myid) continue;
        const auto it = proc_ap_ipair.find(i);
        if (it == proc_ap_ipair.cend()) continue;
        counts_ap[i] = it->second.size();
        total_count_ap += counts_ap[i];
    }
    ids_ap_ipair.reserve(total_count_ap);
    ids_ap_locid.reserve(total_count_ap);
    for (int i = 0; i < nprocs; i++)
    {
        if (i == myid) continue;
        const auto it = proc_ap_ipair.find(i);
        if (it == proc_ap_ipair.cend()) continue;
        const auto &ipairs = it->second;
        const auto &locids = proc_ap_locid.at(i);
        ids_ap_ipair.insert(ids_ap_ipair.cend(), ipairs.cbegin(), ipairs.cend());
        ids_ap_locid.insert(ids_ap_locid.cend(), locids.cbegin(), locids.cend());
    }

    total_count_blacs = 0;
    for (int i = 0; i < nprocs; i++)
    {
        disp_blacs[i] = total_count_blacs;
        if (i == myid) continue;
        const auto it = proc_blacs_locid.find(i);
        if (it == proc_blacs_locid.cend()) continue;
        counts_blacs[i] = it->second.size();
        total_count_blacs += counts_blacs[i];
    }
    ids_blacs_locid.reserve(total_count_blacs);
    for (int i = 0; i < nprocs; i++)
    {
        if (i == myid) continue;
        const auto it = proc_blacs_locid.find(i);
        if (it == proc_blacs_locid.cend()) continue;
        const auto &locids = it->second;
        ids_blacs_locid.insert(ids_blacs_locid.cend(), locids.cbegin(), locids.cend());
    }
    global::profiler.stop("index_scheduler_flatten");

    initialized_ = true;
}

void
IndexScheduler::init(const std::unordered_map<int, std::set<atpair_t>> &map_proc_IJs,
                     const AtomicBasis &atbasis_r, const AtomicBasis &atbasis_c,
                     const ArrayDesc &ad, const bool row_major) noexcept
{
    if (initialized_) reset();
    assert(ad.m() > 0 && atbasis_r.nb_total == as_size(ad.m()));
    assert(ad.n() > 0 && atbasis_c.nb_total == as_size(ad.n()));
    assert(atbasis_r.n_atoms == atbasis_c.n_atoms);

    global::profiler.start("index_scheduler_init_pairs_unordered_map", LIBRPA_VERBOSE_DEBUG);
    ap_p_map<int> map_IJ_proc;
    for (const auto &proc_IJs: map_proc_IJs)
    {
        const auto id = proc_IJs.first;
        const auto &IJs = proc_IJs.second;
        for (const auto &IJ: IJs)
        {
            assert(map_IJ_proc.count(IJ) == 0); // check atom-pair overlap
            map_IJ_proc[IJ] = id;
        }
    }

    const auto myid = ad.myid();
    if (map_proc_IJs.count(myid))
    {
        const auto row_fast = !row_major;
        const auto &IJs = map_proc_IJs.at(myid);
        atpairs = std::vector<atpair_t>(IJs.cbegin(), IJs.cend());
        std::sort(atpairs.begin(), atpairs.end(), FastLess<atpair_t>{row_fast});
    }
    global::profiler.stop("index_scheduler_init_pairs_unordered_map");
    this->init_impl_(map_IJ_proc, atbasis_r, atbasis_c, ad, row_major);
}

void
IndexScheduler::init(const std::map<int, std::set<atpair_t>> &map_proc_IJs,
                     const AtomicBasis &atbasis_r, const AtomicBasis &atbasis_c,
                     const ArrayDesc &ad, const bool row_major) noexcept
{
    if (initialized_) reset();
    assert(ad.m() > 0 && atbasis_r.nb_total == as_size(ad.m()));
    assert(ad.n() > 0 && atbasis_c.nb_total == as_size(ad.n()));
    assert(atbasis_r.n_atoms == atbasis_c.n_atoms);

    global::profiler.start("index_scheduler_init_pairs_map", LIBRPA_VERBOSE_DEBUG);
    ap_p_map<int> map_IJ_proc;
    for (const auto &proc_IJs: map_proc_IJs)
    {
        const auto id = proc_IJs.first;
        const auto &IJs = proc_IJs.second;
        for (const auto &IJ: IJs)
        {
            assert(map_IJ_proc.count(IJ) == 0); // check atom-pair overlap
            map_IJ_proc[IJ] = id;
        }
    }

    const auto myid = ad.myid();
    if (map_proc_IJs.count(myid))
    {
        const auto row_fast = !row_major;
        const auto &IJs = map_proc_IJs.at(myid);
        atpairs = std::vector<atpair_t>(IJs.cbegin(), IJs.cend());
        std::sort(atpairs.begin(), atpairs.end(), FastLess<atpair_t>{row_fast});
    }
    global::profiler.stop("index_scheduler_init_pairs_map");
    this->init_impl_(map_IJ_proc, atbasis_r, atbasis_c, ad, row_major);
}

void
IndexScheduler::reset()
{
    atpairs.clear();
    ids_ap_ipair.clear();
    ids_ap_locid.clear();
    ids_blacs_locid.clear();
    disp_ap.clear();
    counts_ap.clear();
    disp_blacs.clear();
    counts_blacs.clear();
    row_major_ = false;
    initialized_ = false;
}

std::unordered_map<int, std::set<atpair_t>>
get_balanced_ap_distribution_for_consec_descriptor(const AtomicBasis &atbasis_r,
                                                   const AtomicBasis &atbasis_c,
                                                   const ArrayDesc &ad, bool create_empty)
{
    assert(ad.is_col_consec() && ad.is_row_consec());
    assert(atbasis_r.n_atoms == atbasis_c.n_atoms);

    const auto n_atoms = atbasis_r.n_atoms;

    // locate the appropriate row with the center of the matrix
    std::vector<int> prows_at_center(n_atoms);
    for (size_t iat = 0; iat < n_atoms; iat++)
    {
        prows_at_center[iat] = ad.g2p_r()[(atbasis_r.get_part_range()[iat] + atbasis_r.get_part_range()[iat+1])/2];
    }

    std::vector<int> pcols_at_center(n_atoms);
    for (size_t iat = 0; iat < n_atoms; iat++)
    {
        pcols_at_center[iat] = ad.g2p_c()[(atbasis_c.get_part_range()[iat]+atbasis_c.get_part_range()[iat+1])/2];
    }

    const int myid = ad.myid();
    const int nprocs = ad.nprocs();
    const int myprow = ad.myprow();
    const int mypcol = ad.mypcol();
    std::vector<int> procs(n_atoms * n_atoms, 0);
    for (size_t iat_r = 0; iat_r < n_atoms; iat_r++)
    {
        if (myprow != prows_at_center[iat_r]) continue;
        for (size_t iat_c = 0; iat_c < n_atoms; iat_c++)
        {
            if (mypcol != pcols_at_center[iat_c]) continue;
            procs[iat_r + iat_c * n_atoms] = myid;
        }
    }

    MPI_Allreduce(MPI_IN_PLACE, procs.data(), n_atoms * n_atoms, mpi_datatype<int>::value, MPI_SUM, ad.comm());

    std::unordered_map<int, std::set<atpair_t>> map_proc_atpairs;
    if (create_empty)
    {
        for (int proc = 0; proc < nprocs; proc++)
        {
            map_proc_atpairs[proc] = {};
        }
    }
    for (size_t iat_r = 0; iat_r < n_atoms; iat_r++)
    {
        for (size_t iat_c = 0; iat_c < n_atoms; iat_c++)
        {
            const auto proc = procs[iat_r + iat_c * n_atoms];
            map_proc_atpairs[proc].insert({iat_r, iat_c});
        }
    }
    return map_proc_atpairs;
}


// indices computation backend for AP<->BLACS conversion of general matrix
static
std::pair<std::unordered_map<int, std::vector<std::pair<atpair_t, std::vector<std::pair<size_t, size_t>>>>>,
          std::unordered_map<int, std::vector<std::pair<size_t, size_t>>>>
_get_communicate_ids_list_ap_and_blacs(const int &myid,
                                       const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs,
                                       const AtomicBasis &atbasis_r,
                                       const AtomicBasis &atbasis_c,
                                       const ArrayDesc &ad, const bool row_fast,
                                       const bool return_local, const bool include_self = false)
{
    assert(ad.m() > 0 && atbasis_r.nb_total == as_size(ad.m()));
    assert(ad.n() > 0 && atbasis_c.nb_total == as_size(ad.n()));

    // Objects to return
    // process ID -> list of basis indices (AP)
    std::unordered_map<int, std::vector<std::pair<atpair_t, std::vector<std::pair<size_t, size_t>>>>> map_proc_id2d_ap;
    // process ID -> list of basis indices (BLACS)
    std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> map_proc_id2d_blacs;

    int myprow, mypcol;
    int dummy = 0;

    // basis indices availble at myid in the atom-pair distribution
    const std::vector<atpair_t> &IJs =
        map_proc_IJs.count(myid) ? map_proc_IJs.at(myid) : std::vector<atpair_t>();
    // if (myid == 0) std::cout << "IJs_avail " << IJs_avail << std::endl;
    global::profiler.start("get_2d_mat_indices_atpair", LIBRPA_VERBOSE_DEBUG);
    const auto ids_ap = get_2d_mat_indices_atpair(atbasis_r, atbasis_c, IJs, row_fast);
    global::profiler.stop("get_2d_mat_indices_atpair");

    // global basis indices required by BLACS
    global::profiler.start("get_2d_mat_indices_blacs", LIBRPA_VERBOSE_DEBUG);
    const auto ids_blacs = get_2d_mat_indices_blacs(ad, myid, row_fast);
    global::profiler.stop("get_2d_mat_indices_blacs");

    global::profiler.start("ap_and_blacs_map_proc_map_id2d_ap", LIBRPA_VERBOSE_DEBUG);
    std::unordered_map<int, atom_mapping<std::vector<std::pair<size_t, size_t>>>::pair_t> map_proc_map_id2d_ap;
    std::unordered_map<int, std::vector<atpair_t>> map_proc_IJs_save_order;
    for (const auto &id: ids_ap)
    {
        const auto &i_gl = as_int(id.first);
        const auto &j_gl = as_int(id.second);
        // filter out the required BLACS indices that already owns by myid
        myprow = ScalapackConnector::indxg2p(i_gl, ad.mb(), dummy, ad.irsrc(), ad.nprows());
        mypcol = ScalapackConnector::indxg2p(j_gl, ad.nb(), dummy, ad.icsrc(), ad.npcols());
        const auto target = Cblacs_pnum(ad.ictxt(), myprow, mypcol);
        // if (myid == 0) std::cout << "id/target " << id << " " << target << std::endl;
        if (myid != target || include_self)
        {
            if (return_local)
            {
                int I, J;
                int i_lo, j_lo;
                atbasis_r.get_local_index(i_gl, I, i_lo);
                atbasis_r.get_local_index(j_gl, J, j_lo);
                auto &IJs_save_order = map_proc_IJs_save_order[target];
                const atpair_t IJ = {static_cast<atom_t>(I), static_cast<atom_t>(J)};
                if (std::find(IJs_save_order.cbegin(), IJs_save_order.cend(), IJ) == IJs_save_order.cend())
                {
                    IJs_save_order.emplace_back(IJ);
                }
                const atpair_t ij_lo = {static_cast<atom_t>(i_lo), static_cast<atom_t>(j_lo)};
                map_proc_map_id2d_ap[target][IJ].emplace_back(ij_lo);
            }
            else
            {
                // NOTE: global index requested, save all to the (0,0) dummy pair index
                const atpair_t IJ{0, 0};
                auto &IJs_save_order = map_proc_IJs_save_order[target];
                if (std::find(IJs_save_order.cbegin(), IJs_save_order.cend(), IJ) == IJs_save_order.cend())
                {
                    IJs_save_order.emplace_back(IJ);
                }
                map_proc_map_id2d_ap[target][IJ].emplace_back(id);
            }
        }
    }
    for (const auto &target_map: map_proc_map_id2d_ap)
    {
        const auto &target = target_map.first;
        const auto &IJs_save_order = map_proc_IJs_save_order.at(target);
        for (const auto &IJ: IJs_save_order)
        {
            map_proc_id2d_ap[target].push_back({IJ, target_map.second.at(IJ)});
        }
    }
    global::profiler.stop("ap_and_blacs_map_proc_map_id2d_ap");

    global::profiler.start("reverse_map_proc_IJs_avail", LIBRPA_VERBOSE_DEBUG);
    // reverse map of map_proc_IJs_avail
    atom_mapping<int>::pair_t map_IJs_proc;
    for (const auto &proc_IJs: map_proc_IJs)
    {
        const auto &proc = proc_IJs.first;
        for (const auto &IJ: proc_IJs.second)
        {
            map_IJs_proc[IJ] = proc;
        }
    }
    global::profiler.stop("reverse_map_proc_IJs_avail");

    global::profiler.start("map_proc_id2d_blacs", LIBRPA_VERBOSE_DEBUG);
    Cblacs_pcoord(ad.ictxt(), myid, &myprow, &mypcol);
    for (const auto &id: ids_blacs)
    {
        const auto I = static_cast<atom_t>(atbasis_r.get_i_atom(id.first));
        const auto J = static_cast<atom_t>(atbasis_c.get_i_atom(id.second));
        const atpair_t IJ({I, J});
        if (map_IJs_proc.count(IJ))
        {
            const auto &source = map_IJs_proc[IJ];
            // exclude self if not requested
            if (!include_self && source == myid) continue;
            if (return_local)
            {
                // FIXME: now only works when myid == ad.myid
                const auto i_lo = as_size(ad.indx_g2l_r(as_int(id.first)));
                const auto j_lo = as_size(ad.indx_g2l_c(as_int(id.second)));
                map_proc_id2d_blacs[source].push_back({i_lo, j_lo});
            }
            else
            {
                map_proc_id2d_blacs[source].emplace_back(id);
            }
        }
    }
    global::profiler.stop("map_proc_id2d_blacs");

    return {map_proc_id2d_ap, map_proc_id2d_blacs};
}

// indices computation backend for AP->BLACS conversion of symmetric matrix
static
std::pair<std::unordered_map<int, std::vector<std::pair<atpair_t, std::vector<std::pair<size_t, size_t>>>>>,
          std::unordered_map<int, std::vector<std::pair<size_t, size_t>>>>
_get_communicate_ids_list_ap_to_blacs_sy(std::unordered_map<int, std::vector<char>> &map_lconj,
                                          const int &myid,
                                          const char &uplo,
                                          const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                          const AtomicBasis &atbasis,
                                          const ArrayDesc &ad, const bool row_fast,
                                          const bool return_local, const bool include_self = false)
{
    assert(ad.m() > 0 && ad.m() == ad.n() && atbasis.nb_total == as_size(ad.m()));
    assert(uplo == 'U' || uplo == 'u' || uplo == 'L' || uplo == 'l');

    // Objects to return
    // process ID -> list of basis indices (AP)
    std::unordered_map<int, std::vector<std::pair<atpair_t, std::vector<std::pair<size_t, size_t>>>>> map_proc_id2d_ap;
    // process ID -> list of basis indices (BLACS)
    std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> map_proc_id2d_blacs;

    const bool ap_upper_half = uplo == 'U' || uplo == 'u';
    // manually check the keys to ensure they are either all in upper half or all in lower half
    for (const auto &proc_IJs: map_proc_IJs_avail)
    {
        for (const auto &IJ: proc_IJs.second)
        {
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            assert((ap_upper_half && J >= I) || (!ap_upper_half && I >= J));
        }
    }

    int myprow, mypcol;
    int myprow_T, mypcol_T;
    int dummy = 0;

    // basis indices availble at myid in the atom-pair distribution
    const std::vector<atpair_t> &IJs =
        map_proc_IJs_avail.count(myid) ? map_proc_IJs_avail.at(myid) : std::vector<atpair_t>();
    const auto ids_ap = get_2d_mat_indices_atpair(atbasis, atbasis, IJs, row_fast);

    // global basis indices required by BLACS
    const auto ids_blacs = get_2d_mat_indices_blacs(ad, myid, row_fast);

    std::unordered_map<int, atom_mapping<std::vector<std::pair<size_t, size_t>>>::pair_t> map_proc_map_id2d_ap;
    std::unordered_map<int, std::vector<atpair_t>> map_proc_IJs_save_order;
    for (const auto &id: ids_ap)
    {
        const auto &i_gl = as_int(id.first);
        const auto &j_gl = as_int(id.second);
        // Only sending either half
        if (ap_upper_half && i_gl > j_gl) continue;
        if (!ap_upper_half && i_gl < j_gl) continue;
        // filter out the required BLACS indices that already owns by myid
        myprow = ScalapackConnector::indxg2p(i_gl, ad.mb(), dummy, ad.irsrc(), ad.nprows());
        mypcol = ScalapackConnector::indxg2p(j_gl, ad.nb(), dummy, ad.icsrc(), ad.npcols());
        myprow_T = ScalapackConnector::indxg2p(j_gl, ad.mb(), dummy, ad.irsrc(), ad.nprows());
        mypcol_T = ScalapackConnector::indxg2p(i_gl, ad.nb(), dummy, ad.icsrc(), ad.npcols());

        const auto target = Cblacs_pnum(ad.ictxt(), myprow, mypcol);
        const auto target_T = Cblacs_pnum(ad.ictxt(), myprow_T, mypcol_T);

        // if (myid == 0) std::cout << "id/target " << id << " " << target << std::endl;
        if (myid != target || include_self)
        {
            if (return_local)
            {
                int I, J;
                int i_lo, j_lo;
                atbasis.get_local_index(i_gl, I, i_lo);
                atbasis.get_local_index(j_gl, J, j_lo);
                auto &IJs_save_order = map_proc_IJs_save_order[target];
                const atpair_t IJ = {static_cast<atom_t>(I), static_cast<atom_t>(J)};
                if (std::find(IJs_save_order.cbegin(), IJs_save_order.cend(), IJ) == IJs_save_order.cend())
                {
                    IJs_save_order.emplace_back(IJ);
                }
                const atpair_t ij_lo = {static_cast<atom_t>(i_lo), static_cast<atom_t>(j_lo)};
                map_proc_map_id2d_ap[target][IJ].emplace_back(ij_lo);
            }
            else
            {
                // NOTE: global index requested, save all to the (0,0) dummy pair index
                const atpair_t IJ{0, 0};
                auto &IJs_save_order = map_proc_IJs_save_order[target];
                if (std::find(IJs_save_order.cbegin(), IJs_save_order.cend(), IJ) == IJs_save_order.cend())
                {
                    IJs_save_order.emplace_back(IJ);
                }
                map_proc_map_id2d_ap[target][IJ].emplace_back(id);
            }
        }

        // id_aps are only in either half, should complete the whole one except for the diagonals
        if (i_gl != j_gl && (myid != target_T || include_self))
        {
            if (return_local)
            {
                int I, J;
                int i_lo, j_lo;
                // Sending the same element, but may be conjugate on the receiver side
                atbasis.get_local_index(i_gl, I, i_lo);
                atbasis.get_local_index(j_gl, J, j_lo);
                auto &IJs_save_order = map_proc_IJs_save_order[target_T];
                const atpair_t IJ = {static_cast<atom_t>(I), static_cast<atom_t>(J)};
                if (std::find(IJs_save_order.cbegin(), IJs_save_order.cend(), IJ) == IJs_save_order.cend())
                {
                    IJs_save_order.emplace_back(IJ);
                }
                const atpair_t ij_lo = {static_cast<atom_t>(i_lo), static_cast<atom_t>(j_lo)};
                map_proc_map_id2d_ap[target_T][IJ].emplace_back(ij_lo);
            }
            else
            {
                // NOTE: global index requested, save all to the (0,0) dummy pair index
                const atpair_t IJ{0, 0};
                auto &IJs_save_order = map_proc_IJs_save_order[target_T];
                if (std::find(IJs_save_order.cbegin(), IJs_save_order.cend(), IJ) == IJs_save_order.cend())
                {
                    IJs_save_order.emplace_back(IJ);
                }
                map_proc_map_id2d_ap[target_T][IJ].emplace_back(id);
            }

        }
    }

    for (const auto &target_map: map_proc_map_id2d_ap)
    {
        const auto &target = target_map.first;
        const auto &IJs_save_order = map_proc_IJs_save_order.at(target);
        for (const auto &IJ: IJs_save_order)
        {
            map_proc_id2d_ap[target].push_back({IJ, target_map.second.at(IJ)});
        }
    }

    // reverse map of map_proc_IJs_avail
    atom_mapping<int>::pair_t map_IJs_proc;
    for (const auto &proc_IJs: map_proc_IJs_avail)
    {
        const auto &proc = proc_IJs.first;
        for (const auto &IJ: proc_IJs.second)
        {
            map_IJs_proc[IJ] = proc;
        }
    }

    Cblacs_pcoord(ad.ictxt(), myid, &myprow, &mypcol);
    for (const auto &id: ids_blacs)
    {
        const bool id_in_upper = id.first <= id.second;
        const auto I = static_cast<atom_t>(atbasis.get_i_atom(id.first));
        const auto J = static_cast<atom_t>(atbasis.get_i_atom(id.second));
        const atpair_t IJ({I, J});

        // transpose case
        const atpair_t id_T{id.second, id.first};
        const auto I_T = static_cast<atom_t>(atbasis.get_i_atom(id_T.first));
        const auto J_T = static_cast<atom_t>(atbasis.get_i_atom(id_T.second));
        const atpair_t IJ_T({I_T, J_T});

        if ((ap_upper_half && id_in_upper) || (!ap_upper_half && !id_in_upper))
        {
            if (map_IJs_proc.count(IJ))
            {
                const auto &source = map_IJs_proc[IJ];
                // exclude self
                if (!include_self && source == myid) continue;
                if (return_local)
                {
                    // FIXME: now only works when myid == ad.myid
                    const auto i_lo = as_size(ad.indx_g2l_r(as_int(id.first)));
                    const auto j_lo = as_size(ad.indx_g2l_c(as_int(id.second)));
                    map_proc_id2d_blacs[source].push_back({i_lo, j_lo});
                }
                else
                {
                    map_proc_id2d_blacs[source].emplace_back(id);
                }
                map_lconj[source].push_back('n');
            }
        }
        else
        {
            // id not available in atom-pair blocks, must transpose before looking for source
            if (map_IJs_proc.count(IJ_T))
            {
                const auto &source = map_IJs_proc[IJ_T];
                // exclude self
                if (!include_self && source == myid) continue;
                if (return_local)
                {
                    // FIXME: now only works when myid == ad.myid
                    const auto i_lo = as_size(ad.indx_g2l_r(as_int(id.first)));
                    const auto j_lo = as_size(ad.indx_g2l_c(as_int(id.second)));
                    map_proc_id2d_blacs[source].push_back({i_lo, j_lo});
                }
                else
                {
                    map_proc_id2d_blacs[source].emplace_back(id);
                }
                map_lconj[source].push_back('c');
            }
        }
    }

    return {map_proc_id2d_ap, map_proc_id2d_blacs};
}

/* ================================================================================= */
/* For general matrix redistribution from atom-pair (AP) to BLACS block-cyclic forms */
/* ================================================================================= */

std::pair<myid_ap_glid_map_t, myid_blacsid_map_t>
get_communicate_global_ids_list_ap_to_blacs(const int &myid,
                                            const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                            const AtomicBasis &atbasis_r,
                                            const AtomicBasis &atbasis_c,
                                            const ArrayDesc &ad,
                                            bool row_fast, bool row_major, bool include_self)
{
    // Objects to return
    myid_ap_glid_map_t map_proc_id1d_send;
    myid_blacsid_map_t map_proc_id1d_recv;

    const auto ids = _get_communicate_ids_list_ap_and_blacs(myid, map_proc_IJs_avail, atbasis_r,
                                                            atbasis_c, ad, row_fast, false, include_self);

    // process ID -> list of basis indices, sending from myid to others
    std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> map_proc_id2d_send;
    for (const auto &proc_id2d: ids.first)
    {
        const auto &proc = proc_id2d.first;
        // only up to 1 member for the case of global indices, which is for the the (0,0) dummy atom pair
        assert(proc_id2d.second.size() == 1);
        const auto &IJ_id2d = proc_id2d.second[0];
        assert(IJ_id2d.first.first == 0);
        assert(IJ_id2d.first.second == 0);
        map_proc_id2d_send[proc] = IJ_id2d.second;
    }

    // process ID -> list of basis indices, receiving by myid from others
    const auto &map_proc_id2d_recv = ids.second;

    // flatten 2D indices to 1D
    for (const auto &proc_id2d: map_proc_id2d_send)
    {
        map_proc_id1d_send[proc_id2d.first] = flatten_2d_indices(proc_id2d.second, atbasis_r.nb_total, atbasis_c.nb_total, row_major);
    }
    for (const auto &proc_id2d: map_proc_id2d_recv)
    {
        map_proc_id1d_recv[proc_id2d.first] = flatten_2d_indices(proc_id2d.second, atbasis_r.nb_total, atbasis_c.nb_total, row_major);
    }

    return {map_proc_id1d_send, map_proc_id1d_recv};
}

std::pair<myid_ap_loid_map_t, myid_blacsid_map_t>
get_communicate_local_ids_list_ap_to_blacs(const int &myid,
                                           const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                           const AtomicBasis &atbasis_r,
                                           const AtomicBasis &atbasis_c,
                                           const ArrayDesc &ad,
                                           bool row_fast, bool row_major, bool include_self)
{
    const auto ids = _get_communicate_ids_list_ap_and_blacs(myid, map_proc_IJs_avail, atbasis_r,
                                                            atbasis_c, ad, row_fast, true, include_self);
    // process ID -> list of basis indices, sending from myid to others
    const auto &map_proc_id2d_send = ids.first;
    // process ID -> list of basis indices, receiving by myid from others
    const auto &map_proc_id2d_recv = ids.second;

    // flattened 1D indices to return
    myid_ap_loid_map_t map_proc_id1d_send;
    myid_blacsid_map_t map_proc_id1d_recv;

    for (const auto &proc_IJ_id2d: map_proc_id2d_send)
    {
        const auto &proc = proc_IJ_id2d.first;
        for (const auto &IJ_id2d: proc_IJ_id2d.second)
        {
            auto &proc_vec = map_proc_id1d_send[proc];
            const auto &IJ = IJ_id2d.first;
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto id1d = flatten_2d_indices(IJ_id2d.second, atbasis_r.get_atom_nb(I), atbasis_c.get_atom_nb(J), row_major);
            const std::pair<atpair_t, std::vector<size_t>> v(IJ, std::move(id1d));
            proc_vec.emplace_back(v);
        }
    }
    for (const auto &proc_id2d: map_proc_id2d_recv)
    {
        map_proc_id1d_recv[proc_id2d.first] = flatten_2d_indices(proc_id2d.second, as_size(ad.m_loc()), as_size(ad.n_loc()), row_major);
    }
    return {map_proc_id1d_send, map_proc_id1d_recv};
}

/* ============================================================================================= */
/* For symmetric/Hermitian matrix redistribution from atom-pair (AP) to BLACS block-cyclic forms */
/* ============================================================================================= */

std::pair<myid_ap_glid_map_t, myid_blacs_idop_map_t>
get_communicate_global_ids_list_ap_to_blacs_sy(const int &myid,
                                               const char &uplo,
                                               const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                               const AtomicBasis &atbasis,
                                               const ArrayDesc &ad,
                                               bool row_fast, bool row_major, bool include_self)
{
    // Objects to return
    myid_ap_glid_map_t map_proc_id1d_send;
    myid_blacs_idop_map_t map_proc_id1d_lconj_recv;

    std::unordered_map<int, std::vector<char>> map_lconj;
    const auto ids = _get_communicate_ids_list_ap_to_blacs_sy(map_lconj, myid, uplo, map_proc_IJs_avail, atbasis,
                                                              ad, row_fast, false, include_self);

    // process ID -> list of basis indices, sending from myid to others
    std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> map_proc_id2d_send;
    for (const auto &proc_id2d: ids.first)
    {
        const auto &proc = proc_id2d.first;
        // only up to 1 member for the case of global indices, which is for the the (0,0) dummy atom pair
        assert(proc_id2d.second.size() == 1);
        const auto &IJ_id2d = proc_id2d.second[0];
        assert(IJ_id2d.first.first == 0);
        assert(IJ_id2d.first.second == 0);
        map_proc_id2d_send[proc] = IJ_id2d.second;
    }

    // process ID -> list of basis indices, receiving by myid from others
    const auto &map_proc_id2d_recv = ids.second;

    // flatten 2D indices to 1D
    for (const auto &proc_id2d: map_proc_id2d_send)
    {
        map_proc_id1d_send[proc_id2d.first] = flatten_2d_indices(proc_id2d.second, atbasis.nb_total, atbasis.nb_total, row_major);
    }
    for (const auto &proc_data: map_proc_id2d_recv)
    {
        const auto &proc = proc_data.first;
        const auto &id2d = proc_data.second;
        // indices
        map_proc_id1d_lconj_recv[proc].first = flatten_2d_indices(id2d, atbasis.nb_total, atbasis.nb_total, row_major);
        // flag for conjugate
        map_proc_id1d_lconj_recv[proc].second = std::move(map_lconj[proc]);
    }

    return {map_proc_id1d_send, map_proc_id1d_lconj_recv};
}

std::pair<myid_ap_loid_map_t, myid_blacs_idop_map_t>
get_communicate_local_ids_list_ap_to_blacs_sy(const int &myid,
                                              const char &uplo,
                                              const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                              const AtomicBasis &atbasis,
                                              const ArrayDesc &ad,
                                              bool row_fast, bool row_major, bool include_self)
{
    std::unordered_map<int, std::vector<char>> map_lconj;
    const auto ids = _get_communicate_ids_list_ap_to_blacs_sy(map_lconj, myid, uplo, map_proc_IJs_avail, atbasis,
                                                              ad, row_fast, true, include_self);
    // process ID -> list of basis indices, sending from myid to others
    const auto &map_proc_id2d_send = ids.first;
    // process ID -> list of basis indices, receiving by myid from others
    const auto &map_proc_id2d_recv = ids.second;

    // flattened 1D indices to return
    myid_ap_loid_map_t map_proc_id1d_send;
    myid_blacs_idop_map_t map_proc_id1d_lconj_recv;

    for (const auto &proc_IJ_id2d: map_proc_id2d_send)
    {
        const auto &proc = proc_IJ_id2d.first;
        for (const auto &IJ_id2d: proc_IJ_id2d.second)
        {
            auto &proc_vec = map_proc_id1d_send[proc];
            const auto &IJ = IJ_id2d.first;
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto id1d = flatten_2d_indices(IJ_id2d.second, atbasis.get_atom_nb(I), atbasis.get_atom_nb(J), row_major);
            const std::pair<atpair_t, std::vector<size_t>> v(IJ, std::move(id1d));
            proc_vec.emplace_back(v);
        }
    }
    for (const auto &proc_data: map_proc_id2d_recv)
    {
        const auto &proc = proc_data.first;
        const auto &id2d = proc_data.second;
        // indices
        map_proc_id1d_lconj_recv[proc].first = flatten_2d_indices(id2d, as_size(ad.m_loc()), as_size(ad.n_loc()), row_major);
        // flag for conjugate
        map_proc_id1d_lconj_recv[proc].second = std::move(map_lconj[proc]);
    }
    return {map_proc_id1d_send, map_proc_id1d_lconj_recv};
}

/* ================================================================================= */
/* For general matrix redistribution from BLACS block-cyclic to atom-pair (AP) forms */
/* ================================================================================= */

std::pair<myid_blacsid_map_t, myid_ap_glid_map_t>
get_communicate_global_ids_list_blacs_to_ap(const int &myid,
                                            const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_require,
                                            const AtomicBasis &atbasis_r,
                                            const AtomicBasis &atbasis_c,
                                            const ArrayDesc &ad,
                                            bool row_fast, bool row_major, bool include_self)
{
    // Objects to return
    myid_blacsid_map_t map_proc_id1d_send;
    myid_ap_glid_map_t map_proc_id1d_recv;

    const auto ids = _get_communicate_ids_list_ap_and_blacs(myid, map_proc_IJs_require, atbasis_r,
                                                           atbasis_c, ad, row_fast, false, include_self);

    std::unordered_map<int, std::vector<std::pair<size_t, size_t>>> map_proc_id2d_recv;
    for (const auto &proc_id2d: ids.first)
    {
        const auto &proc = proc_id2d.first;
        // only up to 1 member for the case of global indices, which is for the the (0,0) dummy atom pair
        assert(proc_id2d.second.size() == 1);
        const auto &IJ_id2d = proc_id2d.second[0];
        assert(IJ_id2d.first.first == 0);
        assert(IJ_id2d.first.second == 0);
        map_proc_id2d_recv[proc] = IJ_id2d.second;
    }

    // process ID -> list of basis indices, receiving by myid from others
    const auto &map_proc_id2d_send = ids.second;

    // flatten 2D indices to 1D
    for (const auto &proc_id2d: map_proc_id2d_send)
    {
        map_proc_id1d_send[proc_id2d.first] = flatten_2d_indices(proc_id2d.second, atbasis_r.nb_total, atbasis_c.nb_total, row_major);
    }
    for (const auto &proc_id2d: map_proc_id2d_recv)
    {
        map_proc_id1d_recv[proc_id2d.first] = flatten_2d_indices(proc_id2d.second, atbasis_r.nb_total, atbasis_c.nb_total, row_major);
    }

    return {map_proc_id1d_send, map_proc_id1d_recv};
}

std::pair<myid_blacsid_map_t, myid_ap_loid_map_t>
get_communicate_local_ids_list_blacs_to_ap(const int &myid,
                                           const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_require,
                                           const AtomicBasis &atbasis_r,
                                           const AtomicBasis &atbasis_c,
                                           const ArrayDesc &ad,
                                           bool row_fast, bool row_major, bool include_self)
{
    const auto ids = _get_communicate_ids_list_ap_and_blacs(myid, map_proc_IJs_require, atbasis_r,
                                                            atbasis_c, ad, row_fast, true, include_self);
    const auto &map_proc_id2d_send = ids.second;
    const auto &map_proc_id2d_recv = ids.first;

    myid_blacsid_map_t map_proc_id1d_send;
    myid_ap_loid_map_t map_proc_id1d_recv;

    for (const auto &proc_IJ_id2d: map_proc_id2d_recv)
    {
        const auto &proc = proc_IJ_id2d.first;
        for (const auto &IJ_id2d: proc_IJ_id2d.second)
        {
            auto &proc_vec = map_proc_id1d_recv[proc];
            const auto &IJ = IJ_id2d.first;
            const auto &I = IJ.first;
            const auto &J = IJ.second;
            const auto id1d = flatten_2d_indices(IJ_id2d.second, atbasis_r.get_atom_nb(I), atbasis_c.get_atom_nb(J), row_major);
            const std::pair<atpair_t, std::vector<size_t>> v(IJ, std::move(id1d));
            proc_vec.emplace_back(v);
        }
    }
    for (const auto &proc_id2d: map_proc_id2d_send)
    {
        map_proc_id1d_send[proc_id2d.first] = flatten_2d_indices(proc_id2d.second, as_size(ad.m_loc()), as_size(ad.n_loc()), row_major);
    }
    return {map_proc_id1d_send, map_proc_id1d_recv};
}

} /* end of namespace librpa_int */
