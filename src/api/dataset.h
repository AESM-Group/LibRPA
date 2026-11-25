#pragma once

#include <map>
#include <memory>
#include <unordered_map>

#include "../core/atom.h"
#include "../core/atomic_basis.h"
#include "../core/chi0.h"
#include "../core/dielecmodel.h"
#include "../core/exx.h"
#include "../core/geometry.h"
#include "../core/gw.h"
#include "../core/symmetry_context.h"
#include "../core/meanfield.h"
#include "../core/pbc.h"
#include "../core/ri.h"
#include "../core/timefreq.h"
#include "../math/matrix_m.h"
#include "../math/symmetry.h"
#include "../mpi/base_blacs.h"
#include "../mpi/base_mpi.h"
#include "../mpi/kpoint_blacs_parallel_context.h"

namespace librpa_int
{

/*!
 * Core object to hold runtime environment, input and output data
 */
class Dataset
{
private:
    bool input_blacs_matloc_row_major_;
    bool comm_blacs_coul_initialized_;
    bool coul_blacs2ap_redistributed_;
    bool eigvecs_kpara_redistributed_;
    bool eigvecs_kpara_2d_redistributed_;
    bool band_eigvecs_kpara_2d_redistributed_;
public:
    /* Member variables */
    // Environment control
    // Global MPI communicators and BLACS context handlers
    MpiCommHandler comm_h;
    BlacsCtxtHandler blacs_h;
    KPointBlacsParallelContext scfk_blacs_ctxt;
    KPointBlacsParallelContext bandk_blacs_ctxt;
    // Communicators for Coulomb matrices 2D input and re-distribution
    MpiCommHandler comm_coul_h;
    MpiCommHandler comm_coul_inter_q_h;
    MpiCommHandler comm_coul_intra_q_h;
    BlacsCtxtHandler blacs_coul_intra_q_h;
    ArrayDesc desc_coul_intra_q;
    //! Array descriptor for matrices of wave-function basis (using dataset.blacs_h)
    ArrayDesc desc_wfc;
    //! Permanent capped rectangular compute layout, N_ao x N_states, on scfk_blacs_ctxt.blacs_h.
    ArrayDesc desc_wfc_kb;
    //! Dense-root compatibility layout used only at API/input redistribution boundaries.
    ArrayDesc desc_wfc_kb_full;
    //! Band-path counterparts using bandk_blacs_ctxt.blacs_h.
    ArrayDesc desc_band_wfc_kb;
    ArrayDesc desc_band_wfc_kb_full;
    //! Array descriptor for matrices of auxiliary basis set size (using blacs_h)
    ArrayDesc desc_abf;
    //! Array descriptor for compressed auxiliary basis set size (using blacs_h)
    ArrayDesc desc_abf_shrink;
    //! Atom pairs on current process for atomic-basis matrix data
    std::vector<atpair_t> atpairs_local;
    //! Distribution of unique atom-pairs on all processes for atomic-basis matrix data
    std::unordered_map<int, std::set<atpair_t>> atpairs_unique_all;
    //! Unit cell vectors on current process
    std::vector<Vector3_Order<int>> Rs_local;

    // Physical system.
    //! Handling object for basis functions for wave function expansion.
    AtomicBasis basis_wfc;
    //! Handling object for auxiliary basis functions for RI
    AtomicBasis basis_aux;
    //! Handling object for shrinked auxiliary basis functions for RI
    AtomicBasis basis_aux_shrink;
    //! Basis ordering and real-spherical-harmonic convention.
    BasisConvention basis_convention;
    SpaceGroupSymOps spg_symops;
    //! Explicit spin-space operation table set by librpa_set_symmetry_spin_operations;
    //! spatial_id indexes spg_symops. Consumed only when spg_spin_ops_explicit is set.
    std::vector<SymmetrySpinOperation> spg_spin_ops;
    //! Whether spg_spin_ops should replace the default grey-group expansion.
    bool spg_spin_ops_explicit = false;
    //! Whether the explicit table is the unitary block of a grey group (LibRPA
    //! appends the antiunitary copies internally). Defaults to true to match the
    //! legacy behavior of librpa_set_symmetry_operations.
    bool spg_grey_group = true;
    //! Symmetry context built from structure and full k-point grid.
    SymmetryContext symmetry_context;
    //! Atomic structure
    Atoms atoms;
    //! Periodic boundary setting
    PeriodicBoundaryData pbc;
    //! Fractional coordiantes of k-points for band calculation
    std::vector<Vector3_Order<double>> kfrac_band_list;
    // Reserved for a future head/wing path with a k grid different from the
    // SCF mean field. The active path currently uses pbc.kfrac_list directly.
    // std::vector<Vector3_Order<double>> kfrac_headwing_list;

    //! Overlap matrix between normal and shrinked ABFs
    std::map<Vector3_Order<double>, ComplexMatrix> sinvS;

    // Input data.
    //! Mean-field starting point
    MeanField mf;
    //! Velocity/momentum matrix input.
    //! Currently consumed by analytic head/wing construction.
    //! File readers may convert producer-specific momentum/velocity files before
    //! storing them here; public API setters store values as provided.
    velocity_matrix_t velocity_matrix;
    //! Full-BZ eigenframe for analytic head/wing when the SCF grid is
    //! symmetry-reduced. Native velocity expands the IBZ states first and then
    //! differentiates H/S/r in that frame, so head/wing run on a plain full-BZ
    //! mesh instead of the k-star reconstruction path.
    MeanField headwing_full_bz_mf;
    PeriodicBoundaryData headwing_full_bz_pbc;
    velocity_matrix_t headwing_full_bz_velocity;
    bool headwing_full_bz_ready = false;
    //! Mean-field object for band calculation
    MeanField mf_band;
    //! Monotonic id for distinguishing successive band-path input datasets.
    int band_data_id;
    //! Whether band-path input data has been provided through set_band_* APIs.
    bool is_band_data_set;
    //! Whether band-path EXX/GW k-space matrices have been computed for the set band data.
    bool is_band_calc_done;
    //! Time-frequency grids
    TFGrids tfg;
    //! Real-space RI coefficient tensors (local RI)
    Cs_LRI cs_data;
    //! Real-space RI coefficient tensors (local RI) of shrinked auxiliary basis
    Cs_LRI cs_data_shrink;
    // atom-pair distribution of Coulomb matrices
    atpair_k_cplx_mat_t vq;
    atpair_k_cplx_mat_t vq_cut;
    // Local index boundaries and 2D distribution of Coulomb matrices
    // Indices are just for data parsing and blacs context initialization.
    // Further handling should rely on comm_coul_*/blacs_coul_* communicators.
    // Lower bounds are included, and upper bounds are excluded.
    // Bare and cut Coulombs are enforced to use the same layout.
    int vq_lbrow = -1, vq_ubrow = -1;
    int vq_lbcol = -1, vq_ubcol = -1;
    std::map<Vector3_Order<double>, Matz> vq_block_loc;
    std::map<Vector3_Order<double>, Matz> vq_cut_block_loc;
    // Macroscopic dielectric functions at imaginary frequencies
    // Only used for head correction of GW calculation
    std::vector<double> epsmacs_imagfreq;
    std::vector<double> omegas_imagfreq;

    std::unique_ptr<diele_func> p_headwing;

    // Output data, held by computation objects
    // All computation data objects should be contained here as pointers
    // and are created only when requested.
    std::unique_ptr<Exx> p_exx;
    std::unique_ptr<Chi0> p_chi0;
    std::unique_ptr<G0W0> p_g0w0;

    /* Constructors and destructors */
    Dataset(MPI_Comm comm, const bool input_blacs_matloc_row_major = true);
    ~Dataset() { free(); }
    void free();

    void initialize_comm_blacs_coul();
    void redistribute_coulomb_blacs2ap();
    void finalize_comm_blacs_coul();

    void redistribute_eigvecs_kpara();
    void redistribute_band_eigvecs_kpara();
    void redistribute_eigvecs_kpara_2d();
    void redistribute_band_eigvecs_kpara_2d();
    void initialize_band_kblacs_wfc_layout();
    void mark_eigvecs_kpara_2d_ready()
    {
        eigvecs_kpara_redistributed_ = true;
        eigvecs_kpara_2d_redistributed_ = true;
    }
    void mark_band_eigvecs_kpara_2d_ready()
    {
        band_eigvecs_kpara_2d_redistributed_ = true;
    }
    bool eigvecs_kpara_2d_ready() const { return eigvecs_kpara_2d_redistributed_; }
    bool band_eigvecs_kpara_2d_ready() const
    {
        return band_eigvecs_kpara_2d_redistributed_;
    }
    void reset_band_eigvecs_kpara_state();

    //! Drop the full-BZ eigenframe and velocity that the native head/wing route
    //! built, together with the optical object it feeds. Call this when the inputs
    //! the frame was differentiated from change, and when a task that never builds
    //! such a frame starts on a handler that may carry one.
    void invalidate_native_headwing_frame();

    void invalidate_compute_objects();

    /* Disable copy */
    Dataset(const Dataset &) = delete;
    Dataset &operator=(const Dataset &) = delete;
};

typedef std::shared_ptr<Dataset> dataset_ptr_t;

}
