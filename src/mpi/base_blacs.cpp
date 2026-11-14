#include "base_blacs.h"

#include <algorithm>
#include <string>

#include "../interface/scalapack/blacs_scalapack.h"
#include "../io/global_io.h"
#include "../utils/base_utility.h"
#include "../utils/error.h"

namespace librpa_int
{

void CTXT_barrier(int ictxt, CTXT_SCOPE scope)
{
    char scope_ch;
    switch (scope)
    {
        case (CTXT_SCOPE::R): scope_ch = 'R';
        case (CTXT_SCOPE::C): scope_ch = 'C';
        case (CTXT_SCOPE::A): scope_ch = 'A';
    }
    Cblacs_barrier(ictxt, &scope_ch);
}

int get_capped_blacs_block_size(const int dimension, const int max_block_size,
                                const BlacsCtxtHandler &blacs_h)
{
    if (!blacs_h.is_initialized() || blacs_h.nprows <= 0 || blacs_h.npcols <= 0)
        throw LIBRPA_RUNTIME_ERROR("BLACS process grid is not initialized");
    if (dimension <= 0 || max_block_size <= 0)
        throw LIBRPA_RUNTIME_ERROR("BLACS block-size inputs must be positive");

    const int rows_per_process = 1 + (dimension - 1) / blacs_h.nprows;
    const int cols_per_process = 1 + (dimension - 1) / blacs_h.npcols;
    return std::max(1, std::min({max_block_size, rows_per_process, cols_per_process}));
}

BlacsCtxtHandler::BlacsCtxtHandler()
    : mpi_comm_h(),
      layout_ch(),
      initialized_(false),
      pgrid_set_(false),
      comm_set_(false),
      ictxt(-1),
      layout(),
      myid(0),
      nprocs(0),
      nprows(0),
      npcols(0),
      myprow(0),
      mypcol(0)
#ifdef LIBRPA_USE_DDLA
      , ddla_handle(nullptr), ddla_cpu_handle(nullptr)
#endif
{
}

BlacsCtxtHandler::BlacsCtxtHandler(MPI_Comm comm_in)
    : mpi_comm_h(comm_in),
      layout_ch(),
      initialized_(false),
      pgrid_set_(false),
      comm_set_(true),
      ictxt(-1),
      layout(),
      myid(0),
      nprocs(0),
      nprows(0),
      npcols(0),
      myprow(0),
      mypcol(0)
#ifdef LIBRPA_USE_DDLA
    , ddla_handle(nullptr), ddla_cpu_handle(nullptr)
#endif
{
}

void BlacsCtxtHandler::init()
{
    assert(this->comm_set_);
    this->mpi_comm_h.init();
    this->ictxt = Csys2blacs_handle(this->mpi_comm_h.comm);
    this->myid = mpi_comm_h.myid;
    this->nprocs = mpi_comm_h.nprocs;
    this->initialized_ = true;
}

#ifdef LIBRPA_USE_DDLA
void BlacsCtxtHandler::init_ddla_handle(ddla::DdlaBackend backend)
{
    assert(this->initialized_);
    assert(this->pgrid_set_);
    auto &handle = backend == ddla::DdlaBackend::CPU ? ddla_cpu_handle : ddla_handle;
    if (handle != nullptr) return;
    ddla::DDLA_CHECK(ddla::ddlaInit(handle, backend));
    try
    {
#ifdef LIBRPA_USE_HIP
        // Preserve the previous HIP host-staging policy through the public API.
        if (backend == ddla::DdlaBackend::GPU)
            ddla::DDLA_CHECK(ddla::ddlaSetCommunication(
                handle, ddla::DdlaCommunication::HostStagedMPI));
#endif
        ddla::DDLA_CHECK(ddla::ddlaSet(handle, mpi_comm_h.comm, nprows, npcols, layout_ch));
    }
    catch (...)
    {
        ddla::ddlaDestroy(handle);
        handle = nullptr;
        throw;
    }
}

void BlacsCtxtHandler::destroy_ddla_handles()
{
    for (auto *handle : {&ddla_handle, &ddla_cpu_handle})
    {
        if (*handle == nullptr) continue;
        ddla::DDLA_CHECK(ddla::ddlaSetDevice(*handle));
        ddla::DDLA_CHECK(ddla::ddlaDestroy(*handle));
        *handle = nullptr;
    }
}
#endif

void BlacsCtxtHandler::reset_comm()
{
    this->mpi_comm_h.reset_comm();
    if (this->pgrid_set_) this->exit();
    this->myid = 0;
    this->nprocs = 0;
    this->comm_set_ = false;
    this->initialized_ = false;
    this->pgrid_set_ = false;
#ifdef LIBRPA_USE_DDLA
    destroy_ddla_handles();
#endif
}

void BlacsCtxtHandler::reset_comm(MPI_Comm comm_in, bool init_on_reset)
{
    this->mpi_comm_h.reset_comm(comm_in, false);
    if (this->pgrid_set_) exit();
    this->myid = mpi_comm_h.myid;
    this->nprocs = mpi_comm_h.nprocs;
    this->initialized_ = false;
    this->comm_set_ = true;
    this->pgrid_set_ = false;
    if (init_on_reset) this->init();
#ifdef LIBRPA_USE_DDLA
    destroy_ddla_handles();
#endif
}

void BlacsCtxtHandler::set_grid(const int &nprows_in, const int &npcols_in,
                                CTXT_LAYOUT layout_in)
{
    assert(this->initialized_);
    // if the grid has been set, exit it first
    if (pgrid_set_) exit();
    if (nprocs != nprows_in * npcols_in)
    {
        throw LIBRPA_RUNTIME_ERROR("nprocs != nprows * npcols :" + std::to_string(nprocs) + " != " +
                                   std::to_string(nprows_in) + " + " + std::to_string(npcols_in));
    }
    layout = layout_in;
    if (layout == CTXT_LAYOUT::C)
        layout_ch = 'C';
    else
        layout_ch = 'R';
    Cblacs_gridinit(&ictxt, &layout_ch, nprows_in, npcols_in);
    Cblacs_gridinfo(ictxt, &nprows, &npcols, &myprow, &mypcol);
    pgrid_set_ = true;
}

void BlacsCtxtHandler::set_square_grid(bool more_rows, CTXT_LAYOUT layout_in)
{
    int nroc;
    layout = layout_in;
    for (nroc = int(sqrt(double(nprocs))); nroc >= 2; --nroc)
    {
        if ((nprocs) % nroc == 0) break;
    }
    more_rows ? nprows = nprocs / (npcols = nroc)
              : npcols = nprocs / (nprows = nroc);
    set_grid(nprows, npcols, layout_in);
}

void BlacsCtxtHandler::set_horizontal_grid()
{
    set_grid(1, nprocs, CTXT_LAYOUT::R);
}

void BlacsCtxtHandler::set_vertical_grid()
{
    set_grid(nprocs, 1, CTXT_LAYOUT::C);
}

void BlacsCtxtHandler::exit()
{
    if (pgrid_set_)
    {
        Cblacs_gridexit(ictxt);
        if (mpi_comm_h.comm == MPI_COMM_NULL)
            ictxt = -1;
        else
            ictxt = Csys2blacs_handle(mpi_comm_h.comm);
        pgrid_set_ = false;
    }
#ifdef LIBRPA_USE_DDLA
    destroy_ddla_handles();
#endif
}

std::string BlacsCtxtHandler::info() const
{
    std::string info;
    info = std::string("BlacsCtxtHandler: ")
         + "ICTXT " + std::to_string(ictxt) + " "
         + "PSIZE " + std::to_string(nprocs) + " "
         + "PID " + std::to_string(myid) + " "
         + "PGRID (" + std::to_string(nprows) + "," + std::to_string(npcols) + ") "
         + "PCOOD (" + std::to_string(myprow) + "," + std::to_string(mypcol) +")";
    return info;
}

int BlacsCtxtHandler::get_pnum(int prow, int pcol) const
{
    return Cblacs_pnum(ictxt, prow, pcol);
}

void BlacsCtxtHandler::get_pcoord(int pid, int &prow, int &pcol) const
{
    Cblacs_pcoord(ictxt, pid, &prow, &pcol);
}

pcoord_t BlacsCtxtHandler::get_pcoord(int pid) const
{
    int prow, pcol;
    Cblacs_pcoord(ictxt, pid, &prow, &pcol);
    return {prow, pcol};
}

void BlacsCtxtHandler::barrier(CTXT_SCOPE scope) const
{
    CTXT_barrier(ictxt, scope);
}

std::vector<int> get_proc_indices_blacs(const int &m, const int &n, const int &mb, const int &nb,
                                        const int &irsrc, const int &icsrc,
                                        const int &ictxt, bool row_fast)
{
    std::vector<int> procids;
    procids.reserve(as_size(m * n));
    // myprow/mypcol are not used here, declared for indxg2p interface
    int nprows, npcols, myprow, mypcol;
    Cblacs_gridinfo(ictxt, &nprows, &npcols, &myprow, &mypcol);

    if (row_fast)
    {
        for (auto j = 0; j < n; j++)
        {
            int pcol = ScalapackConnector::indxg2p(j, nb, mypcol, icsrc, npcols);
            for (auto i = 0; i < m; i++)
            {
                int prow = ScalapackConnector::indxg2p(i, mb, myprow, irsrc, nprows);
                int rank = Cblacs_pnum(ictxt, prow, pcol);
                procids.push_back(rank);
            }
        }
    }
    else
    {
        for (auto i = 0; i < m; i++)
        {
            int prow = ScalapackConnector::indxg2p(i, mb, myprow, irsrc, nprows);
            for (auto j = 0; j < n; j++)
            {
                int pcol = ScalapackConnector::indxg2p(j, nb, mypcol, icsrc, npcols);
                int rank = Cblacs_pnum(ictxt, prow, pcol);
                procids.push_back(rank);
            }
        }
    }

    return procids;
}

void ArrayDesc::set_blacs_params_(MPI_Comm comm, int ictxt, int nprocs, int myid, int nprows,
                                  int myprow, int npcols, int mypcol)
{
    assert(myid < nprocs && myprow < nprows && mypcol < npcols);
    comm_ = comm;
    ictxt_ = ictxt;
    nprocs_ = nprocs;
    myid_ = myid;
    nprows_ = nprows;
    myprow_ = myprow;
    npcols_ = npcols;
    mypcol_ = mypcol;
}

int ArrayDesc::set_desc_indices_(const int &m, const int &n, const int &mb, const int &nb,
                                  const int &irsrc, const int &icsrc)
{
    int info = 0;
    m_local_ = ScalapackConnector::numroc(m, mb, myprow_, irsrc, nprows_);
    // leading dimension
    // NOTE: use number of rows as leading dimension by default, i.e. for column-major
    if (row_leading_)
        lld_ = std::max(m_local_, 1);
    else
        lld_ = std::max(n_local_, 1);
    n_local_ = ScalapackConnector::numroc(n, nb, mypcol_, icsrc, npcols_);
    if (m_local_ < 1 || n_local_ < 1)
    {
        empty_local_mat_ = true;
    }
    is_loc_consecutive_r_ = mb * nprows_ >= m;
    is_loc_consecutive_c_ = nb * npcols_ >= n;

    ScalapackConnector::descinit(this->desc, m, n, mb, nb, irsrc, icsrc, ictxt_, lld_, info);
    if (info)
    {
        librpa_int::global::lib_printf(
            "ERROR DESCINIT! PROC %d (%d,%d) PARAMS: DESC %d %d %d %d %d %d %d %d\n",
            myid_, myprow_, mypcol_, m, n, mb, nb, irsrc, icsrc, ictxt_, m_local_);
    }
    // else
    //     librpa_int::global::lib_printf("SUCCE DESCINIT! PROC %d (%d,%d) PARAMS: DESC %d %d %d %d %d %d %d %d\n", myid_, myprow_, mypcol_, m, n, mb, nb, irsrc, icsrc, ictxt_, m_local_);
    // Safety check (really necessary?)
    assert(ictxt_ == desc[1]);
    assert(m == desc[2]);
    assert(n == desc[3]);
    assert(mb == desc[4]);
    assert(nb == desc[5]);
    assert(irsrc == desc[6]);
    assert(icsrc == desc[7]);
    assert(lld_ == desc[8]);
    m_ = desc[2];
    n_ = desc[3];
    mb_ = desc[4];
    nb_ = desc[5];
    irsrc_ = desc[6];
    icsrc_ = desc[7];
    this->build_index_();
    initialized_ = true;
    return info;
}

// void ArrayDesc::switch_to_row_leading() noexcept
// {
//     if (row_leading_) return;
//     if (initialized_)
//         desc[8] = lld_ = std::max(m_local_, 1);
//     row_leading_ = true;
// }

// void ArrayDesc::switch_to_col_leading() noexcept
// {
//     if (!row_leading_) return;  // already column leading
//     if (initialized_)
//         desc[8] = lld_ = std::max(n_local_, 1);
//     row_leading_ = false;
// }

void ArrayDesc::build_index_()
{
    g2l_r_.resize(m_, -1);
    g2p_r_.resize(m_, -1);
    for (int i = 0; i < m_; i++)
    {
        int row = ScalapackConnector::indxg2p(i, mb_, myprow_, irsrc_, nprows_);
        g2p_r_[i] = row;
        if (myprow_ == row)
        {
            g2l_r_[i] = ScalapackConnector::indxg2l(i, mb_, myprow_, irsrc_, nprows_);
        }
    }
    g2l_c_.resize(n_, -1);
    g2p_c_.resize(n_, -1);
    for (int j = 0; j < n_; j++)
    {
        int col = ScalapackConnector::indxg2p(j, nb_, mypcol_, icsrc_, npcols_);
        g2p_c_[j] = col;
        if (mypcol_ == col)
        {
            g2l_c_[j] = ScalapackConnector::indxg2l(j, nb_, mypcol_, icsrc_, npcols_);
        }
    }
    if (m_local_ > 0)
    {
        l2g_r_.resize(m_local_, 0);
        for (int i = 0; i < m_local_; i++)
        {
            l2g_r_[i] = ScalapackConnector::indxl2g(i, mb_, myprow_, irsrc_, nprows_);
        }
    }
    if (n_local_ > 0)
    {
        l2g_c_.resize(n_local_, 0);
        for (int j = 0; j < n_local_; j++)
        {
            l2g_c_[j] = ScalapackConnector::indxl2g(j, nb_, mypcol_, icsrc_, npcols_);
        }
    }
}

void ArrayDesc::clear_index_()
{
    g2l_r_.clear();
    g2l_c_.clear();
    l2g_r_.clear();
    l2g_c_.clear();
    g2p_r_.clear();
    g2p_c_.clear();
}

ArrayDesc::ArrayDesc()
    : ictxt_(0), nprocs_(0), myid_(0),
      nprows_(0), myprow_(0), npcols_(0), mypcol_(0),
      m_(0), n_(0), mb_(0), nb_(0), irsrc_(0), icsrc_(0),
      lld_(0), m_local_(0), n_local_(0),
      row_leading_(true),
      g2l_r_(), g2l_c_(), l2g_r_(), l2g_c_(),
      is_loc_consecutive_r_(false), is_loc_consecutive_c_(false),
      empty_local_mat_(false), initialized_(false)
#ifdef LIBRPA_USE_DDLA
    , ddla_handle_(nullptr)
#endif

#ifdef LIBRPA_USE_ELPA
    , elpa_handle_(nullptr)
#endif

{}

ArrayDesc::ArrayDesc(const BlacsCtxtHandler &blacs_h)
    : ictxt_(0), nprocs_(0), myid_(0),
      nprows_(0), myprow_(0), npcols_(0), mypcol_(0),
      m_(0), n_(0), mb_(0), nb_(0), irsrc_(0), icsrc_(0),
      lld_(0), m_local_(0), n_local_(0),
      row_leading_(true),
      g2l_r_(), g2l_c_(), l2g_r_(), l2g_c_(),
      is_loc_consecutive_r_(false), is_loc_consecutive_c_(false),
      empty_local_mat_(false), initialized_(false)
#ifdef LIBRPA_USE_DDLA
    , ddla_handle_(nullptr)
#endif

#ifdef LIBRPA_USE_ELPA
    , elpa_handle_(nullptr)
#endif
{
    this->reset_handler(blacs_h);
}

ArrayDesc::ArrayDesc(const int &ictxt)
    : ictxt_(0), nprocs_(0), myid_(0),
      nprows_(0), myprow_(0), npcols_(0), mypcol_(0),
      m_(0), n_(0), mb_(0), nb_(0), irsrc_(0), icsrc_(0),
      lld_(0), m_local_(0), n_local_(0),
      row_leading_(true),
      g2l_r_(), g2l_c_(), l2g_r_(), l2g_c_(),
      is_loc_consecutive_r_(false), is_loc_consecutive_c_(false),
      empty_local_mat_(false), initialized_(false)
#ifdef LIBRPA_USE_DDLA
    , ddla_handle_(nullptr)
#endif

#ifdef LIBRPA_USE_ELPA
    , elpa_handle_(nullptr)
#endif
{
    // TODO: how to check if ictxt is a valid context?
    int nprocs, myid, nprows, npcols, myprow, mypcol;
    MPI_Comm comm = Cblacs2sys_handle(ictxt);
    Cblacs_gridinfo(ictxt, &nprows, &npcols, &myprow, &mypcol);
    myid = Cblacs_pnum(ictxt, myprow, mypcol);
    nprocs = nprows * npcols;
    set_blacs_params_(comm, ictxt, nprocs, myid,
                      nprows, myprow, npcols,
                      mypcol);
}

void ArrayDesc::reset_handler()
{
    // BLACS
    comm_ = MPI_COMM_NULL;
    ictxt_ = -1;
    nprocs_ = 1;
    myid_ = 0;
    nprows_ = 1;
    myprow_ = 0;
    npcols_ = 1;
    mypcol_ = 0;
    // Array specific
    desc[1] = ictxt_;
    desc[2] = m_ = 0;
    desc[3] = n_ = 0;
    desc[4] = mb_ = 0;
    desc[5] = nb_ = 0;
    desc[6] = irsrc_ = 0;
    desc[7] = icsrc_ = 0;
    desc[8] = lld_ = 0;
    m_local_ = 0;
    n_local_ = 0;
    // row_leading_ = true;  // no need to change it
    is_loc_consecutive_r_ = false;
    is_loc_consecutive_c_ = false;
    empty_local_mat_ = false;
    this->clear_index_();
    initialized_ = false;
}

void ArrayDesc::reset_handler(const BlacsCtxtHandler &blacs_h)
{
    assert(blacs_h.is_initialized());
    // clean up current indices if the descriptor is already initialized
    if (initialized_)
    {
        clear_index_();
        initialized_ = false;
    }
    set_blacs_params_(blacs_h.get_comm(), blacs_h.ictxt, blacs_h.nprocs, blacs_h.myid,
                      blacs_h.nprows, blacs_h.myprow, blacs_h.npcols,
                      blacs_h.mypcol);
}

int ArrayDesc::init(const int &m, const int &n, const int &mb, const int &nb,
                    const int &irsrc, const int &icsrc)
{
    return set_desc_indices_(m, n, mb, nb, irsrc, icsrc);
}

int ArrayDesc::init_1b1p(const int &m, const int &n,
                          const int &irsrc, const int &icsrc)
{
    int mb = 1, nb = 1;
    mb = std::ceil(double(m)/nprows_);
    nb = std::ceil(double(n)/npcols_);
    return set_desc_indices_(m, n, mb, nb, irsrc, icsrc);
}

int ArrayDesc::init_square_blk(const int &m, const int &n,
                                    const int &irsrc, const int &icsrc)
{
    int mb = 1, nb = 1, minblk = 1;
    mb = std::ceil(double(m)/nprows_);
    nb = std::ceil(double(n)/npcols_);
    minblk = std::min(mb, nb);
    return set_desc_indices_(m, n, minblk, minblk, irsrc, icsrc);
}

int ArrayDesc::init_square_blk_capped(const int &m, const int &n, const int &maxblk,
                                      const int &irsrc, const int &icsrc)
{
    int mb = 1, nb = 1, minblk = 1;
    mb = std::ceil(double(m)/nprows_);
    nb = std::ceil(double(n)/npcols_);
    minblk = std::min(mb, nb);
    if (maxblk > 0)
        minblk = std::min(minblk, maxblk);
    minblk = std::max(minblk, 1);
    return set_desc_indices_(m, n, minblk, minblk, irsrc, icsrc);
}

std::string ArrayDesc::info() const
{
    std::string info;
    info = std::string("ArrayDesc: ")
         + "ICTXT " + std::to_string(ictxt_) + " "
         + "ID " + std::to_string(myid_) + " "
         + "PCOOR (" + std::to_string(myprow_) + "," + std::to_string(mypcol_) + ") "
         + "GSIZE (" + std::to_string(m_) + "," + std::to_string(n_) + ") "
         + "LSIZE (" + std::to_string(m_local_) + "," + std::to_string(n_local_) + ") "
         + "DUMMY? " + std::string(empty_local_mat_? "T" : "F");
    return info;
}

std::string ArrayDesc::info_desc() const
{
    char s[100];
    sprintf(s, "DESC %d %d %d %d %d %d %d %d %d",
            desc[0], desc[1], desc[2],
            desc[3], desc[4], desc[5],
            desc[6], desc[7], desc[8]);
    return std::string(s);
}

void ArrayDesc::barrier(CTXT_SCOPE scope) const
{
    CTXT_barrier(ictxt_, scope);
}

int get_globalIndex(int localIndex, int nblk, int nprocs, int myproc)
{
    int iblock, gIndex;
    iblock = localIndex / nblk;
    gIndex = (iblock * nprocs + myproc) * nblk + localIndex % nblk;
    return gIndex;
}

int get_localIndex(int globalIndex, int nblk, int nprocs, int myproc)
{
    int inproc = int((globalIndex % (nblk * nprocs)) / nblk);
    if (myproc == inproc)
    {
        return int(globalIndex / (nblk * nprocs)) * nblk + globalIndex % nblk;
    }
    else
    {
        return -1;
    }
}

std::vector<int> get_proc_indices_blacs(const ArrayDesc &ad, bool row_fast)
{
    assert(ad.initialized());
    return get_proc_indices_blacs(ad.m(), ad.n(), ad.mb(), ad.nb(), ad.irsrc(), ad.icsrc(),
                                  ad.ictxt(), row_fast);
}

std::vector<std::pair<size_t, size_t>>
get_2d_mat_indices_blacs(const int &m, const int &n,
                         const int &mb, const int &nb,
                         const int &irsrc, const int &icsrc,
                         const int &nprows, const int &npcols,
                         const int &myprow, const int &mypcol, bool row_fast)
{
    std::vector<std::pair<size_t, size_t>> indices;

    std::vector<size_t> row_ids;
    std::vector<size_t> col_ids;

    for (auto i = 0; i < m; i++)
    {
        if (myprow == ScalapackConnector::indxg2p(i, mb, myprow, irsrc, nprows))
        {
            row_ids.push_back(as_size(i));
        }
    }
    for (auto i = 0; i < n; i++)
    {
        if (mypcol == ScalapackConnector::indxg2p(i, nb, mypcol, icsrc, npcols))
        {
            col_ids.push_back(as_size(i));
        }
    }

    if (row_fast)
    {
        for (const auto &c: col_ids)
        {
            for (const auto &r: row_ids)
            {
                indices.push_back({r, c});
            }
        }
    }
    else
    {
        for (const auto &r: row_ids)
        {
            for (const auto &c: col_ids)
            {
                indices.push_back({r, c});
            }
        }
    }

    return indices;
}

std::vector<std::pair<size_t, size_t>> get_2d_mat_indices_blacs(const ArrayDesc &ad, const int &myid, bool row_fast)
{
    assert(ad.initialized());

    int myprow, mypcol;
    Cblacs_pcoord(ad.ictxt(), myid, &myprow, &mypcol);

    return get_2d_mat_indices_blacs(ad.m(), ad.n(), ad.mb(), ad.nb(), ad.irsrc(), ad.icsrc(),
                                    ad.nprows(), ad.npcols(), myprow, mypcol, row_fast);
}

std::vector<size_t>
get_1d_mat_indices_blacs(const int &m, const int &n,
                         const int &mb, const int &nb,
                         const int &irsrc, const int &icsrc,
                         const int &nprows, const int &npcols,
                         const int &myprow, const int &mypcol,
                         bool row_fast, bool row_major)
{
    const auto indices_2d = get_2d_mat_indices_blacs(m, n, mb, nb, irsrc, icsrc, nprows, npcols, myprow, mypcol, row_fast);
    return flatten_2d_indices(indices_2d, as_size(m), as_size(n),
                              row_major);
}

std::vector<size_t>
get_1d_mat_indices_blacs(const ArrayDesc &ad, const int &myid, bool row_fast, bool row_major)
{
    const auto indices_2d = get_2d_mat_indices_blacs(ad, myid, row_fast);
    return flatten_2d_indices(indices_2d, as_size(ad.m()), as_size(ad.n()),
                              row_major);
}

} /* end of namespace librpa_int */
