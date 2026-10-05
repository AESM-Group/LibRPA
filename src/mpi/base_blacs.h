#pragma once

#include <string>
#include <vector>

#include "base_mpi.h"
#include "../math/scalapack_connector.h"

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla.h>
#include <ddla/ddla_stream.h>
#endif

#ifdef LIBRPA_USE_ELPA
#include <elpa/elpa.h>
// Some ELPA headers expose the C imaginary-unit macro, but LibRPA uses I
// as an atom index in C++ interfaces.
#ifdef I
#undef I
#endif
#endif

namespace librpa_int
{

enum class CTXT_LAYOUT {R, C};
enum class CTXT_SCOPE {R, C, A};

void CTXT_barrier(int ictxt, CTXT_SCOPE scope = CTXT_SCOPE::A);

typedef std::pair<int, int> pcoord_t;

class BlacsCtxtHandler
{
private:
    MpiCommHandler mpi_comm_h;
    char layout_ch;
    bool initialized_;
    bool pgrid_set_;
    bool comm_set_;
public:
    int ictxt;
    CTXT_LAYOUT layout;
    int myid;
    int nprocs;
    int nprows;
    int npcols;
    int myprow;
    int mypcol;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    ddla::DdlaHandle_t ddla_handle;
    void init_ddla_handle();
#endif
    BlacsCtxtHandler();
    BlacsCtxtHandler(MPI_Comm comm_in);
    ~BlacsCtxtHandler() { this->exit(); }

    // Disable copy
    BlacsCtxtHandler(const BlacsCtxtHandler &) = delete;
    BlacsCtxtHandler operator=(const BlacsCtxtHandler &) = delete;

    void init();
    void reset_comm();
    void reset_comm(MPI_Comm comm_in, bool init_on_reset = true);
    void set_grid(const int &nprows_in, const int &npcols_in, CTXT_LAYOUT layout_in = CTXT_LAYOUT::R);
    void set_square_grid(bool more_rows = true, CTXT_LAYOUT layout_in = CTXT_LAYOUT::R);
    void set_horizontal_grid();
    void set_vertical_grid();
    std::string info() const;
    const MpiCommHandler &comm_h() const { return mpi_comm_h; }
    MPI_Comm comm() const { return mpi_comm_h.comm; }
    int get_pnum(int prow, int pcol) const;
    void get_pcoord(int pid, int &prow, int &pcol) const;
    pcoord_t get_pcoord(int pid) const;
    void barrier(CTXT_SCOPE scope = CTXT_SCOPE::A) const;
    //! call gridexit to reset process grid
    void exit();
    inline bool initialized() const { return initialized_; }  // To deprecate
    inline bool is_initialized() const { return initialized_; }
    const MPI_Comm &get_comm() const { return this->mpi_comm_h.comm; };
};

int get_capped_blacs_block_size(const int dimension, const int max_block_size,
                                const BlacsCtxtHandler &blacs_h);

/*!
 * @brief Get indices of process to which the distributed elements of global matrix belong
 *
 * @param  [in]  m, n            Number of rows and columns of the global matrix
 * @param  [in]  mb, nb          Block size along row and column direction.
 * @param  [in]  irsrc, icsrc    Source process along row and column.
 * @param  [in]  ictxt           BLACS context, must be initialized
 * @param  [in]  row_fast        Flag to set the row basis index goes faster.
 *
 * @retval       indices         List of process indices (size m * n)
 */
std::vector<int> get_proc_indices_blacs(const int &m, const int &n, const int &mb, const int &nb,
                                        const int &irsrc, const int &icsrc,
                                        const int &ictxt, bool row_fast);

class ArrayDesc
{
private:
    // BLACS parameters obtained upon construction
    MPI_Comm comm_;
    int ictxt_;
    int nprocs_;
    int myid_;
    int nprows_;
    int myprow_;
    int npcols_;
    int mypcol_;
    void set_blacs_params_(MPI_Comm comm, int ictxt, int nprocs, int myid, int nprows, int myprow, int npcols, int mypcol);
    int set_desc_indices_(const int &m, const int &n, const int &mb, const int &nb,
                          const int &irsrc, const int &icsrc);

    // Array dimensions
    int m_;
    int n_;
    int mb_;
    int nb_;
    int irsrc_;
    int icsrc_;
    int lld_;
    int m_local_;
    int n_local_;
    bool row_leading_;  // For now it is always True:
                        // the leading dimension is the number of local rows,
                        // for column-major matrices.

    // Precomputed indices
    std::vector<int> g2l_r_;
    std::vector<int> g2l_c_;
    std::vector<int> l2g_r_;
    std::vector<int> l2g_c_;
    // Process location of index
    std::vector<int> g2p_r_;
    std::vector<int> g2p_c_;
    void build_index_();
    void clear_index_();

    //! flag to indicate whether the local row indices correspond to consecutive global indices
    bool is_loc_consecutive_r_;
    //! flag to indicate whether the local column indices correspond to consecutive global indices
    bool is_loc_consecutive_c_;

    //! flag to indicate that the current process should contain no data of local matrix, but for scalapack routines, it will generate a dummy matrix of size 1, nrows = ncols = 1
    bool empty_local_mat_ = false;

    //! flag for initialization
    bool initialized_ = false;

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    ddla::DdlaDesc ddla_desc_;
#endif

#ifdef LIBRPA_USE_ELPA
    elpa_t elpa_handle_;
#endif

public:
    int desc[9];
    ArrayDesc();
    ArrayDesc(const BlacsCtxtHandler &blacs_ctxt_h);
    ArrayDesc(const int &ictxt);
    //! Initialize the array descriptor
    int init(const int &m, const int &n,
             const int &mb, const int &nb,
             const int &irsrc, const int &icsrc);
    //! Initialize the array descriptor such that each process has exactly one block, with continuous local indices
    int init_1b1p(const int &m, const int &n,
                  const int &irsrc, const int &icsrc);
    int init_square_blk(const int &m, const int &n,
                        const int &irsrc, const int &icsrc);
    //! Initialize square blocks as in init_square_blk, optionally capped by maxblk when maxblk > 0
    int init_square_blk_capped(const int &m, const int &n, const int &maxblk,
                               const int &irsrc, const int &icsrc);
    inline int indx_g2l_r(int gindx) const noexcept { return (gindx < m_ && gindx > -1)? g2l_r_[gindx]: -1; };
    inline int indx_g2l_c(int gindx) const noexcept { return (gindx < n_ && gindx > -1)? g2l_c_[gindx]: -1; };
    inline int indx_l2g_r(int lindx) const noexcept { return (lindx < m_local_ && lindx > -1)? l2g_r_[lindx]: -1; };
    inline int indx_l2g_c(int lindx) const noexcept { return (lindx < n_local_ && lindx > -1)? l2g_c_[lindx]: -1; };
    inline const std::vector<int> &g2l_r() const noexcept { return g2l_r_; }
    inline const std::vector<int> &g2l_c() const noexcept { return g2l_c_; }
    inline const std::vector<int> &g2p_r() const noexcept { return g2p_r_; }
    inline const std::vector<int> &g2p_c() const noexcept { return g2p_c_; }
    inline const std::vector<int> &l2g_r() const noexcept { return l2g_r_; }
    inline const std::vector<int> &l2g_c() const noexcept { return l2g_c_; }
    inline int myid() const noexcept { return myid_; }
    inline int ictxt() const noexcept { return ictxt_; }
    inline MPI_Comm comm() const noexcept { return comm_; }
    inline int m() const noexcept { return m_; }
    inline int n() const noexcept { return n_; }
    inline int mb() const noexcept { return mb_; }
    inline int nb() const noexcept { return nb_; }
    inline int lld() const noexcept { return lld_; }
    inline int irsrc() const noexcept { return irsrc_; }
    inline int icsrc() const noexcept { return icsrc_; }
    inline int m_loc() const noexcept { return m_local_; }
    inline int n_loc() const noexcept { return n_local_; }
    inline int myprow() const noexcept { return myprow_; }
    inline int mypcol() const noexcept { return mypcol_; }
    inline int nprocs() const noexcept { return nprocs_; }
    inline int nprows() const noexcept { return nprows_; }
    inline int npcols() const noexcept { return npcols_; }
    inline void get_pcoord(int pid, int &prow, int &pcol) const { Cblacs_pcoord(ictxt_, myid_, &prow, &pcol); };
    inline pcoord_t get_pcoord(int pid) const { int prow, pcol; Cblacs_pcoord(ictxt_, myid_, &prow, &pcol); return {prow, pcol}; };
    inline int get_pnum(int prow, int pcol) const { return Cblacs_pnum(ictxt_, prow, pcol); };
    inline int get_pnum(const pcoord_t &pc) const { return Cblacs_pnum(ictxt_, pc.first, pc.second); };
    void reset_handler();
    void reset_handler(const BlacsCtxtHandler &blacs_h);
    std::string info() const;
    std::string info_desc() const;
    // inline bool is_row_leading() const noexcept { return row_leading_; }
    // inline bool is_col_leading() const noexcept { return !row_leading_; }
    // void switch_to_row_leading() noexcept;
    // void switch_to_col_leading() noexcept;
    bool is_src() const noexcept { return myprow_ == irsrc_ && mypcol_ == icsrc_; }
    bool is_row_consec() const noexcept { return is_loc_consecutive_r_; }
    bool is_col_consec() const noexcept { return is_loc_consecutive_c_; }
    bool initialized() const noexcept { return this->initialized_; };  // to deprecate
    bool is_initialized() const noexcept { return this->initialized_; };
    void barrier(CTXT_SCOPE scope = CTXT_SCOPE::A) const;

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
    const ddla::DdlaDesc& ddla_desc() const noexcept { return ddla_desc_; }
    void set_ddla_desc(const ddla::DdlaHandle_t& ddla_handle){
        ddla_desc_.set_ddla_handle(ddla_handle);
        ddla_desc_.init(this->m_, this->n_, this->mb_, this->nb_, this->irsrc_, this->icsrc_);
    }
#endif
#ifdef LIBRPA_USE_ELPA
    void set_elpa_handle(bool use_gpu_replace_scalapack = true){
        int error;
        elpa_handle_ = elpa_allocate(&error);
        if(error != ELPA_OK){
            throw std::runtime_error("elpa allocate failure\n");
        }
        elpa_set(elpa_handle_, "na", m_, &error);
        elpa_set(elpa_handle_, "nev", m_, &error);
        elpa_set(elpa_handle_, "local_nrows", m_local_, &error);
        elpa_set(elpa_handle_, "local_ncols", n_local_, &error);
        elpa_set(elpa_handle_, "nblk", mb_, &error);
        elpa_set(elpa_handle_, "mpi_comm_parent", MPI_Comm_c2f(this->comm_), &error);
        elpa_set(elpa_handle_, "process_row", myprow_, &error);
        elpa_set(elpa_handle_, "process_col", mypcol_, &error);

        elpa_setup(elpa_handle_);

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        if(use_gpu_replace_scalapack)
        {
            elpa_set(elpa_handle_, "use_gpu_id", ddla_desc_.ddla_handle()->local_device, &error);
#ifdef LIBRPA_USE_HIP
            elpa_set(elpa_handle_, "amd-gpu", 1, &error);
#endif
#ifdef LIBRPA_USE_CUDA
            elpa_set(elpa_handle_, "nvidia-gpu", 1, &error);
#endif
            elpa_set(elpa_handle_, "use_ccl", 0, &error);
            // 初始化 GPU
            elpa_setup_gpu(elpa_handle_);
        }else{
            elpa_set(elpa_handle_, "gpu_tridiag",                   0, &error);
            elpa_set(elpa_handle_, "gpu_solve_tridi",               0, &error);
            elpa_set(elpa_handle_, "gpu_trans_ev",                  0, &error);
            elpa_set(elpa_handle_, "gpu_bandred",                   0, &error);
            elpa_set(elpa_handle_, "gpu_trans_ev_tridi_to_band",    0, &error);
            elpa_set(elpa_handle_, "gpu_trans_ev_band_to_full",     0, &error);
            elpa_set(elpa_handle_, "gpu_hermitian_multiply",        0, &error);
            elpa_set(elpa_handle_, "gpu_pxgemm_multiply",           0, &error);
            elpa_set(elpa_handle_, "gpu_invert_trm",                0, &error);
        }
#endif

    }
    const elpa_t& elpa_handle() const noexcept { return elpa_handle_; }
    ~ArrayDesc(){
        if(elpa_handle_ != nullptr){
            int error;
            elpa_deallocate(elpa_handle_, &error);
            if(error != ELPA_OK){
                std::cerr << "Error: elpa deallocate failure in ArrayDesc destructor" << std::endl;
            }
        }
    }
#endif
};


int get_globalIndex(int localIndex, int nblk, int nprocs, int myproc);
int get_localIndex(int globalIndex, int nblk, int nprocs, int myproc);

/*!
 * @brief Get indices of process to which the distributed elements of global matrix belong
 *
 * @param  [in]  ad              ArrayDesc object, must be initialized beforehand.
 * @param  [in]  row_fast        Flag to set the row basis index goes faster.
 *
 * @warning      Memory intensive for large system
 *
 * @retval       indices         List of process indices (size m * n)
 */
std::vector<int> get_proc_indices_blacs(const ArrayDesc &ad, bool row_fast);

/*!
 * @brief Get 2D indices of elements of submatrix in BLACS 2D block-cyclic format
 *
 * @param  [in]  m, n            Size of rows and columns.
 * @param  [in]  mb, nb          Block size along row and column direction.
 * @param  [in]  irsrc, icsrc    Source process along row and column.
 * @param  [in]  nprows, npcols  Number of processes along row and column,
 *                               i.e. the shape of proccess grid.
 * @param  [in]  myprow, mypcol  Coordinate of current process in a process grid.
 * @param  [in]  row_fast        Flag to set the row basis index faster.
 *
 * @retval       indices
 */
std::vector<std::pair<size_t, size_t>>
get_2d_mat_indices_blacs(const int &m, const int &n,
                         const int &mb, const int &nb,
                         const int &irsrc, const int &icsrc,
                         const int &nprows, const int &npcols,
                         const int &myprow, const int &mypcol, bool row_fast);

/*!
 * @brief Get 2D indices of elements of submatrix in BLACS 2D block-cyclic format
 *        from the ArrayDesc object.
 *
 * @param  [in]  ad         ArrayDesc object, must be initialized before parsing
 * @param  [in]  row_fast   Flag to set the row basis index faster.
 *
 * @retval indices
 */
std::vector<std::pair<size_t, size_t>>
get_2d_mat_indices_blacs(const ArrayDesc &ad, const int &myid, bool row_fast);

/*!
 * @brief Get 1D indices of elements of submatrix in BLACS 2D block-cyclic format
 *
 * @param  [in]  m, n            Size of rows and columns.
 * @param  [in]  mb, nb          Block size along row and column direction.
 * @param  [in]  irsrc, icsrc    Source process along row and column.
 * @param  [in]  nprows, npcols  Number of processes along row and column,
 *                               i.e. the shape of proccess grid.
 * @param  [in]  myprow, mypcol  Coordinate of current process in a process grid.
 * @param  [in]  row_fast        Flag to set the row basis index faster.
 * @param  [in]  row_major       Flag to compute the 1D indices in row-major (C-style)
 *
 * @retval       indices
 */
std::vector<size_t>
get_1d_mat_indices_blacs(const int &m, const int &n,
                         const int &mb, const int &nb,
                         const int &irsrc, const int &icsrc,
                         const int &nprows, const int &npcols,
                         const int &myprow, const int &mypcol,
                         bool row_fast, bool row_major);

/*!
 * @brief Get 1D indices of elements of submatrix in BLACS 2D block-cyclic format
 *        from the ArrayDesc object.
 *
 * @param  [in]  ad         ArrayDesc object, must be initialized before parsing
 * @param  [in]  row_fast   Flag to set the row basis index faster.
 * @param  [in]  row_major  Flag to compute the 1D indices in row-major (C-style)
 *
 * @retval indices
 */
std::vector<size_t>
get_1d_mat_indices_blacs(const ArrayDesc &ad, const int &myid, bool row_fast, bool row_major);

} /* end of namespace librpa_int */
