#pragma once
#include <array>
#include <cstddef>
#include <complex>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../math/matrix3.h"
#include "../math/matrix_m.h"
#include "../mpi/base_blacs.h"
#include "../mpi/kpoint_blacs_parallel_context.h"
#include "atomic_basis.h"
#include "symmetry_context.h"
#include "meanfield.h"
#include "pbc.h"
#include "ri.h"

namespace librpa_int
{

//! double-dispersion Havriliak-Negami model
struct DoubleHavriliakNegami
{
    static const int d_npar;
    static const std::function<double(double, const std::vector<double> &)> func_imfreq;
    static const std::function<void(std::vector<double> &, double, const std::vector<double> &)>
        grad_imfreq;
};

std::vector<double> interpolate_dielec_func(int option, const std::vector<double> &frequencies_in,
                                            const std::vector<double> &df_in,
                                            const std::vector<double> &frequencies_target);

struct RpaHeadwingSettings
{
    bool enabled = false;
    int option_dielect_func = 0;
    bool use_2d_dielectric = false;
    bool use_soc = false;
    int rpa_headwing_body_start = 0;
    std::string rpa_headwing_mode = "qavg";
    double sqrt_coulomb_threshold = 0.0;
};

//! Velocity/momentum matrix, indexed as [spin][k][cartesian].
using velocity_matrix_t = std::vector<std::vector<std::vector<ComplexMatrix>>>;

void initialize_velocity_matrix(velocity_matrix_t &velocity, int n_spins, int n_kpoints,
                                int n_states);

std::vector<int> map_kpoints_by_coordinates(
    const std::vector<Vector3_Order<double>> &target_kpoints,
    const std::vector<Vector3_Order<double>> &source_kpoints, double tolerance = 1.0e-5);
std::vector<std::vector<int>> map_symmetry_kstar_members_to_source_kpoints(
    const SymmetryContext &ctx, const std::vector<Vector3_Order<double>> &ibz_kpoints,
    const std::vector<Vector3_Order<double>> &source_kpoints, double tolerance = 1.0e-5);

double headwing_transition_weight(double occupied_weight, double unoccupied_weight, int n_spin,
                                  bool spin_orbit_coupled);
double headwing_spin_prefactor(int n_spin, bool spin_orbit_coupled);
std::array<std::array<std::complex<double>, 3>, 3> compute_wing_cartesian_gram(
    const ComplexMatrix &wing);
void accumulate_wing_mu_for_pair(const std::vector<double> &omega,
                                 const std::array<std::complex<double>, 3> &velocity_unocc_occ,
                                 const std::complex<double> &c_mn, double egap, double factor1,
                                 double factor2, std::complex<double> *wing_mu_for_mu,
                                 std::complex<double> *wing_mu_iomega0_for_mu = nullptr);
std::vector<int> headwing_local_kpoints(int n_kpoints,
                                        const KPointBlacsParallelContext *kblacs_ctxt);
ComplexMatrix rotate_headwing_wfc_to_kstar_member(
    const SymmetryContext &ctx,
    const SymmetryKStarMember &member,
    const std::vector<SpeciesBasisLayout> &wfc_layouts,
    const std::map<atom_t, size_t> &atom_nw,
    const Vector3_Order<double> &k_ibz,
    const ComplexMatrix &wfc_ibz,
    const Vector3_Order<double> *k_bz_target = nullptr);
std::array<ComplexMatrix, 3> rotate_headwing_velocity_to_kstar_member(
    const SymmetryContext &ctx,
    const SymmetryKStarMember &member,
    const std::array<ComplexMatrix, 3> &velocity_ibz,
    int n_bands,
    bool use_time_reversal);
std::array<ComplexMatrix, 3> direct_full_bz_velocity_for_kstar_member(
    const velocity_matrix_t &velocity_full,
    const std::vector<std::vector<int>> &member_source_ik,
    int ispin,
    int ik_ibz,
    std::size_t imember);
const ComplexMatrix &direct_full_bz_wfc_for_kstar_member(
    const MeanField &wfc_full,
    const std::vector<std::vector<int>> &member_source_ik,
    int ispin,
    int ispinor,
    int ik_ibz,
    std::size_t imember);
ComplexMatrix localize_direct_full_bz_wfc(const ComplexMatrix &wfc_full,
                                          const ArrayDesc &desc_wfc);

using HeadwingIJKAtomKey = std::size_t;
using HeadwingIJKKey = std::pair<HeadwingIJKAtomKey, int>;
using HeadwingCsIJKMap =
    std::map<HeadwingIJKAtomKey,
             std::map<HeadwingIJKKey, RI::Tensor<std::complex<double>>>>;
using HeadwingCsIJKRequests =
    std::pair<std::set<HeadwingIJKAtomKey>, std::set<HeadwingIJKKey>>;

struct HeadwingFourierTarget
{
    int target_id = -1;
    int owner_ik = -1;
    Vector3_Order<double> kfrac{0.0, 0.0, 0.0};
};

struct HeadwingSymmetryFourierTargets
{
    std::vector<HeadwingFourierTarget> targets;
    std::vector<std::vector<int>> target_ids_by_ibz_member;
};

std::vector<HeadwingFourierTarget> build_headwing_full_bz_fourier_targets(
    const std::vector<Vector3_Order<double>> &kfrac_list);
HeadwingSymmetryFourierTargets build_headwing_symmetry_fourier_targets(
    const SymmetryContext &ctx, const PeriodicBoundaryData &pbc,
    const std::vector<Vector3_Order<double>> &kfrac_ibz);
HeadwingCsIJKMap fourier_headwing_cs_to_ijk(
    const Cs_LRI &Cs_data, const AtomicBasis &atomic_basis_wfc,
    const AtomicBasis &atomic_basis_abf,
    const std::vector<HeadwingFourierTarget> &targets);
HeadwingCsIJKRequests build_headwing_cs_ijk_requests(
    const AtomicBasis &atomic_basis_wfc,
    const std::vector<HeadwingFourierTarget> &targets,
    const std::vector<int> &owner_iks_local, const ArrayDesc &desc_nao_nao);
HeadwingCsIJKMap redistribute_headwing_cs_ijk(
    const Cs_LRI &Cs_data, const AtomicBasis &atomic_basis_wfc,
    const AtomicBasis &atomic_basis_abf,
    const std::vector<HeadwingFourierTarget> &targets,
    const std::vector<int> &owner_iks_local, const ArrayDesc &desc_nao_nao,
    const MpiCommHandler &comm_h);

void invert_headwing_body_with_identity_solve(
    matrix_m<std::complex<double>> &body, ArrayDesc &desc_body,
    const BlacsCtxtHandler &blacs_h, bool use_cholesky, bool use_device);

// ABF-space averaged inverse dielectric rewrite for Gamma option-3 GW Wc,
// factored out of diele_func so it can be unit tested directly.
// eps_block holds E = I - sqrt(V)*chi0*sqrt(V) (n_abf x n_abf, distributed) on
// entry and is overwritten with the averaged inverse dielectric matrix.
// sqrtv_block is the matrix square root sqrt(V); coul_eigen_block holds the
// Coulomb eigenvectors U with x1 = U[:,0]. head is the 3x3 dielectric head H
// (it already includes the identity contribution) and wing_mu the replicated
// n_abf x 3 ABF wing. qx/qy/qz are unit angular directions and rho the
// per-point quadrature weights (already including the Gamma-cell volume and
// q_gamma factors). When the Coulomb matrix has filtered eigenchannels, the
// rewrite is restricted to its first n_nonsingular eigenvectors; the filtered
// channels must be the trailing columns of coul_eigen_block, whose complete
// column set is required to be orthonormal (U U^H = I to O(eps)) -- both
// bundled eigensolvers redistribute the full set with descending eigenvalues
// before filtering, so this holds. The filtered projector is applied through
// its low-rank factor Y = [x1 | trailing filtered columns] (n_abf x m with
// m = 1 + n_singular), so no n_abf x n_abf projector is ever formed.
// eps_block must be Hermitian: the routine contracts it only as E*Y rather than
// forming both E*Y and E^H*Y, so the result is accurate to O(|E - E^H|). The
// Cholesky branch of the in-place inverse (pposv with uplo='L') already requires
// this, and E is Hermitian on the imaginary-frequency axis that reaches here.
// Collective reductions run on the descriptor communicator. The read-only
// matrix inputs are const; only eps_block is modified.
void rewrite_eps_abf_space(
    matrix_m<std::complex<double>> &eps_block,
    const matrix_m<std::complex<double>> &sqrtv_block,
    const matrix_m<std::complex<double>> &coul_eigen_block,
    const matrix_m<std::complex<double>> &head,
    const matrix_m<std::complex<double>> &wing_mu,
    const std::vector<double> &qx, const std::vector<double> &qy,
    const std::vector<double> &qz, const std::vector<double> &rho,
    const ArrayDesc &desc_nabf_nabf_opt, const BlacsCtxtHandler &blacs_h,
    std::size_t n_nonsingular,
    double sqrt_coulomb_threshold, bool use_cholesky, bool use_device);

// All calculation in unit: Bohr and Ha.
struct Strict2dFiniteQReference;
class diele_func
{
private:
    // ( omega, alpha * beta  )
    std::vector<matrix_m<std::complex<double>>> head;
    // ( omega, mu:n_abfs * alpha )
    std::vector<matrix_m<std::complex<double>>> wing_mu;
    // ( omega, lambda:n_lambda * alpha)
    std::vector<matrix_m<std::complex<double>>> wing;
    // std::vector<std::vector<std::vector<std::complex<double>>>> wing;

    // ( i:n_lambda, j:n_lambda )
    matrix_m<std::complex<double>> body_inv;
    // ( i:3, j:3 )
    matrix_m<std::complex<double>> Lind;
    std::vector<matrix_m<std::complex<double>>> strict_2d_lind_by_freq;
    // ( i:n_lambda, j:3 )
    matrix_m<std::complex<double>> bw;
    // ( i:3, j:n_lambda )
    matrix_m<std::complex<double>> wb;
    // ( i:n_lambda, j:n_lambda )
    matrix_m<std::complex<double>> chi0;
    // ( lambda: n_nonsingular-1, mu: n_abfs)
    // std::vector<std::vector<std::complex<double>>> Coul_vector;
    // ( lambda: n_nonsingular-1 )
    // std::vector<std::complex<double>> Coul_value;
    // ( mu: n_abfs, m: n_bands, n: n_bands, k )
    // std::vector<std::vector<std::vector<std::map<Vector3_Order<double>, std::complex<double>>>>>
    //    Ctri_mn;
    // ( mu: n_abfs@I, i: i atom basis, j: j atom basis, k, I atom, J atom, q cell  )
    // Ctri_ij.data_libri[I][{J, k_array}](mu, i, j)
    // Cs_LRI_clx Ctri_ij;
    // ( mu: n_abfs@I, i: i atom basis, j: j atom basis, k, I atom, J atom, R cell  )
    // Ctri_ij.data_libri[I][{J, R}](mu, i, j)
    // used for reduce all mpi Cs_data to Cs_IJR
    // Cs_LRI Cs_IJR;

    MeanField meanfield_df;
    std::vector<double> omega;
    std::vector<Vector3_Order<double>> kfrac_band;
    int n_basis, n_states, n_spin, n_abf, nk;
    const PeriodicBoundaryData &pbc_;
    const AtomicBasis &atomic_basis_wfc_;
    const AtomicBasis &atomic_basis_abf_;
    const velocity_matrix_t &velocity_;
    velocity_matrix_t direct_full_bz_velocity_;
    MeanField direct_full_bz_wfc_;
    std::vector<std::vector<int>> direct_full_bz_velocity_member_source_ik_;
    const MpiCommHandler &comm_h;
    const BlacsCtxtHandler &blacs_h;
    const SymmetryContext *symmetry_context_;
    const KPointBlacsParallelContext *kblacs_ctxt_;
    const ArrayDesc *desc_wfc_kblacs_;
    bool use_kpara_eigvec_;
    size_t n_nonsingular;
    // Lebedev-Laikov angular grid; qw has absorbed 4Pi.
    std::vector<double> qx_leb, qy_leb, qz_leb, qw_leb;
    // gamma reciprocal lattice vector, (27-1)*3
    std::vector<Vector3_Order<double>> g_enclosing_gamma;
    std::vector<double> q_gamma;
    double vol_gamma = 0.0;
    double strict_2d_pw_to_auxiliary_scale_ = 0.0;

public:
    bool use_2d_dielectric = false;
    bool use_soc = false;
    bool debug = false;

    void configure_strict_2d_coulomb_head(bool enabled);
    void configure_strict_2d_coulomb_head(bool enabled,
                                          double auxiliary_monopole_norm_squared);
    bool strict_2d_coulomb_head_configured() const noexcept;
    double get_strict_2d_pw_to_auxiliary_scale() const;

    // Symmetry-aware head/wing switches. When use_symmetry is true and the
    // input symmetry context can restore the BZ from the IBZ k-grid, cal_head
    // and cal_wing sum over k-star members instead of the bare IBZ grid.
    bool use_symmetry = false;
    std::map<atom_t, size_t> atom_nw;
    std::map<atom_t, std::array<double, 3>> coord_frac;
    void set_symmetry_context(const SymmetryContext &ctx) { symmetry_context_ = &ctx; }
    void set_direct_full_bz_headwing_inputs(
        velocity_matrix_t velocity_full,
        MeanField wfc_full,
        std::vector<std::vector<int>> member_source_ik)
    {
        direct_full_bz_velocity_ = std::move(velocity_full);
        direct_full_bz_wfc_ = std::move(wfc_full);
        direct_full_bz_velocity_member_source_ik_ = std::move(member_source_ik);
    }
    bool has_direct_full_bz_headwing_inputs() const
    {
        return !direct_full_bz_velocity_.empty() && direct_full_bz_wfc_.initialized();
    }

    // Public access to the head/wing mean-field copy for driver-side validation
    // of producer eigenvector coverage. The driver must not expand or broadcast
    // eigenvectors here; k-parallel consumers use their kBLACS owner groups.
    MeanField &get_meanfield_df() { return meanfield_df; }

public:
    diele_func(const MeanField &mf, const velocity_matrix_t &velocity,
               const std::vector<Vector3_Order<double>> &kfrac,
               const AtomicBasis &atomic_basis_wfc,
               const AtomicBasis &atomic_basis_abf,
               const std::vector<double> &frequencies_target, const int nbasis, const int nstates,
               const int nspin, const int nabf, const PeriodicBoundaryData &pbc,
               const MpiCommHandler &comm_h_in,
               const BlacsCtxtHandler &blacs_h_in,
               bool use_kpara_eigvec = false,
               const KPointBlacsParallelContext *kblacs_ctxt_in = nullptr,
               const ArrayDesc *desc_wfc_kblacs_in = nullptr)
        : meanfield_df(mf),
          omega(frequencies_target),
          kfrac_band(kfrac),
          n_basis(nbasis),
          n_states(nstates),
          n_spin(nspin),
          n_abf(nabf),
          nk(0),
          pbc_(pbc),
          atomic_basis_wfc_(atomic_basis_wfc),
          atomic_basis_abf_(atomic_basis_abf),
          velocity_(velocity),
          comm_h(comm_h_in),
          blacs_h(blacs_h_in),
          symmetry_context_(nullptr),
          kblacs_ctxt_(nullptr),
          desc_wfc_kblacs_(desc_wfc_kblacs_in),
          use_kpara_eigvec_(use_kpara_eigvec),
          n_nonsingular(0)
    {
        if (kblacs_ctxt_in && kblacs_ctxt_in->is_initialized())
        {
            if (kblacs_ctxt_in->n_kpoints() != static_cast<int>(kfrac.size()))
            {
                throw std::runtime_error(
                    "head/wing k-point list is inconsistent with the SCF k-point parallel "
                    "context");
            }
            kblacs_ctxt_ = kblacs_ctxt_in;
        }
        if (use_kpara_eigvec_)
        {
            if (kblacs_ctxt_ == nullptr || desc_wfc_kblacs_ == nullptr ||
                !desc_wfc_kblacs_->is_initialized())
            {
                throw std::runtime_error(
                    "k-parallel head/wing requires an initialized k-BLACS context and "
                    "wave-function descriptor");
            }
            if (desc_wfc_kblacs_->m() != n_basis || desc_wfc_kblacs_->n() != n_states ||
                desc_wfc_kblacs_->ictxt() != kblacs_ctxt_->blacs_h.ictxt)
            {
                throw std::runtime_error(
                    "k-parallel head/wing wave-function descriptor is inconsistent");
            }
        }
    };
    ~diele_func(){};
    void init(double coulomb_eigen_threshold, const atpair_k_cplx_mat_t &Vq);
    void init_wing(double coulomb_eigen_threshold, const atpair_k_cplx_mat_t &Vq);

    void cal_head();

private:
    // Full-BZ head summation (the historical path, now also used as the fallback
    // when symmetry restoration is unavailable). Expects wg indexed as wg(ik, ib).
    void cal_head_full_bz();
    // Symmetry-aware head: IBZ outer loop + k-star member inner loop. Full-BZ
    // PyATB velocity is selected directly when available; otherwise the
    // historical velocity-restoration fallback is used.
    void cal_head_symmetric();

public:
    double cal_factor(std::string name);
    void test_head();
    std::vector<double> get_head_vec();
    bool has_wing() const { return !wing.empty() || !wing_mu.empty(); }

    void cal_wing(const Cs_LRI &Cs_data, double coulomb_eigen_threshold,
                  const atpair_k_cplx_mat_t &Vq);  // atpair_k_cplx_mat_t &Vq, Cs_LRI &Cs_data

private:
    // Symmetry-aware wing: each kBLACS owner group loops over its local IBZ
    // k-points and contributes each k-star member to wing_mu. Full-BZ PyATB
    // velocity is selected directly when available; the IBZ mean-field is not
    // expanded or broadcast globally.
    void cal_wing_symmetric(const Cs_LRI &Cs_data, double coulomb_eigen_threshold,
                            const atpair_k_cplx_mat_t &Vq);
    // The historical full-BZ wing summation, also used as the fallback when
    // symmetry restoration is unavailable.
    void cal_wing_full_bz(const Cs_LRI &Cs_data, double coulomb_eigen_threshold,
                          const atpair_k_cplx_mat_t &Vq);

public:
    // tranform Cs_ij(R) to Cs_ij(k)
    // void FT_R2k(const librpa_int::Cs_LRI &Cs_data);
    // void Cs_ij2mn();
    // diagonalize Vq(q=0)
    void get_Xv(double coulomb_eigen_threshold,
                const librpa_int::atpair_k_cplx_mat_t &Vq);  // diagonalize Vq(q=0)
    std::complex<double> compute_wing(const int alpha, const int iomega, const int mu, const int ik,
                                      const int ispin, const ArrayDesc &desc_nband_nband,
                                      const matrix_m<complex<double>> &C_nband_nband);
    // std::complex<double> compute_Cs_ij2mn(int mu, int m, int n, int ik);
    // std::complex<double> compute_Cijk(const librpa_int::Cs_LRI &Cs_data, int mu, int I, int i,
    // int J, int j, int ik); transform wing from ABF to Coulomb representation
    void wing_mu_to_lambda(matrix_m<std::complex<double>> &sqrtveig_blacs,
                           ArrayDesc &desc_nabf_nabf_opt, std::size_t n_nonsingular_in);
    // tranform Cs_ij(R) to Cs_ij(k)
    // diagonalize real Vq_cut(q=0)
    // void get_Xv_real(double vq_threshold, const librpa_int::atpair_k_cplx_mat_t &Vq);
    // diagonalize complex Vq_cut(q=0)
    void get_Xv_cpl(double coulomb_eigen_threshold, const atpair_k_cplx_mat_t &Vq);
    std::pair<ArrayDesc, matrix_m<complex<double>>> transform_Cs2mnk(
        const int ik, const int mu,
        std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> &Cs_IJ,
        int spin_filter = -1);
    std::pair<ArrayDesc, matrix_m<complex<double>>> transform_Cs2mnk_kblacs(
        int ik, int mu,
        std::map<int, std::map<libri_types<int, int>::TAC, RI::Tensor<double>>> &Cs_IJ,
        const BlacsCtxtHandler &wing_blacs_h, const Vector3_Order<double> &kfrac,
        const std::vector<std::vector<const ComplexMatrix *>> *wfc_override = nullptr,
        int spin_filter = -1);
    std::pair<ArrayDesc, matrix_m<complex<double>>> transform_Cs2mnk_kblacs(
        int source_ik, int target_id, int mu, const HeadwingCsIJKMap &Cs_IJ_k,
        const BlacsCtxtHandler &wing_blacs_h,
        const std::vector<std::vector<const ComplexMatrix *>> *wfc_override = nullptr,
        int spin_filter = -1);
    // void FT_R2k();
    // std::complex<double> compute_Cijk(Cs_LRI &Cs_in, int mu, int I, int i, int J, int j, int
    // ik); void Cs_ij2mn(); std::complex<double> compute_Cs_ij2mn(int mu, int m, int n, int
    // ik);
    //  diagonalize real Vq_cut(q=0)
    //  void get_Xv_real();
    //  diagonalize complex Vq_cut(q=0)
    void test_wing();
    // set wing=0 for debug
    void set_0_wing();

    matrix_m<std::complex<double>> get_rpa_chi0v_head(const int ifreq) const;
    matrix_m<std::complex<double>> get_rpa_chi0v_wing(const int ifreq) const;

    void construct_rpa_trace_log_schur(const int ifreq, ArrayDesc &desc_body,
                                       int wing_row_offset = 0);
    ArrayDesc get_body_inv(matrix_m<std::complex<double>> &chi0_block,
                           ArrayDesc &desc_nabf_nabf_opt);
    void construct_L(const int ifreq, ArrayDesc &desc_body);

    // Lebedev-Laikov quadrature
    void get_Leb_points();
    void get_g_enclosing_gamma();
    void get_g_enclosing_gamma_2d();
    void calculate_q_gamma();
    void calculate_q_gamma_2d();
    // Complete-basis ABF-space wing rewrite for Gamma option-3 GW Wc.
    // Requires n_nonsingular_in == n_abf (sqrt_coulomb_threshold disabled).
    // eps_block holds E = I - sqrt(V)*chi0*sqrt(V) on entry and the averaged
    // inverse dielectric matrix on exit. Delegates to the free
    // rewrite_eps_abf_space helper using this object's head, wing_mu and
    // angular quadrature data.
    void rewrite_eps_abf_space(matrix_m<std::complex<double>> &eps_block, const int ifreq,
                               const matrix_m<std::complex<double>> &sqrtv_block,
                               const matrix_m<std::complex<double>> &coul_eigen_block,
                               const ArrayDesc &desc_nabf_nabf_opt,
                               std::size_t n_nonsingular_in,
                               double sqrt_coulomb_threshold,
                               bool use_cholesky, bool use_device);
    void cal_eps(const int ifreq, ArrayDesc &desc_nabf_nabf_opt, ArrayDesc &desc_body);
    void cal_strict_2d_wc(const int ifreq, ArrayDesc &desc_nabf_nabf_opt, ArrayDesc &desc_body,
                          const matrix_m<std::complex<double>> &regular_coulomb_basis);
    void assign_chi0(matrix_m<std::complex<double>> &chi0_block,
                     ArrayDesc &desc_nabf_nabf_opt);
    void rewrite_strict_2d_wc(matrix_m<std::complex<double>> &chi0_block, const int ifreq,
                              ArrayDesc &desc_nabf_nabf_opt,
                              const matrix_m<std::complex<double>> &regular_coulomb_basis);
    Strict2dFiniteQReference get_strict_2d_finite_q_reference(int ifreq, double qx,
                                                              double qy) const;
    double get_strict_2d_bare_coulomb_gamma_average() const;
    std::complex<double> compute_rpa_trace_log_average(
        matrix_m<std::complex<double>> &response_block, const int ifreq, ArrayDesc &desc_response,
        const RpaHeadwingSettings &settings);
    void rewrite_rpa_response(matrix_m<std::complex<double>> &eps_minus_identity_block,
                              const int ifreq, ArrayDesc &desc_nabf_nabf_opt);

private:
    std::pair<ArrayDesc, matrix_m<complex<double>>> rotate_Cs_nao2mnk_kblacs(
        int source_ik, int mu, matrix_m<complex<double>> &C_nao_nao,
        const ArrayDesc &desc_nao_nao, const BlacsCtxtHandler &wing_blacs_h,
        const ArrayDesc &desc_wfc_src,
        const std::vector<std::vector<const ComplexMatrix *>> *wfc_override,
        int spin_filter);
};

int rpa_headwing_regular_body_start_channel(const RpaHeadwingSettings &settings);

double rpa_headwing_reciprocal_cell_volume(const PeriodicBoundaryData &pbc, bool use_2d_dielectric);
double rpa_headwing_gamma_cell_volume(const PeriodicBoundaryData &pbc, bool use_2d_dielectric);

double strict_2d_head_prefactor(double inplane_cell_area);
double strict_2d_wing_prefactor(double inplane_cell_area);
double strict_2d_physical_q(double internal_q);
double strict_2d_physical_gamma_cell_area(double internal_gamma_cell_area);
struct Strict2dFiniteQReference
{
    std::complex<double> epsilon_minus_identity_over_q;
    std::complex<double> schur_a;
    std::complex<double> wc_head_limit;
};
struct Strict2dBlockMetricSums
{
    std::complex<double> head = 0.0;
    double head_body_squared = 0.0;
    double body_head_squared = 0.0;
    double body_body_squared = 0.0;
};
struct Strict2dBlockMetrics
{
    std::complex<double> head = 0.0;
    double head_body_frobenius = 0.0;
    double body_head_frobenius = 0.0;
    double body_body_frobenius = 0.0;
};
void accumulate_strict_2d_block_metric(Strict2dBlockMetricSums &sums, int row, int column,
                                       const std::complex<double> &value);
Strict2dBlockMetrics finalize_strict_2d_block_metrics(const Strict2dBlockMetricSums &sums);
Strict2dFiniteQReference strict_2d_finite_q_reference(const matrix_m<std::complex<double>> &head,
                                                      const matrix_m<std::complex<double>> &lind,
                                                      double qx, double qy);
std::complex<double> strict_2d_radial_i0(const std::complex<double> &a, double qmax);
std::complex<double> strict_2d_radial_i1(const std::complex<double> &a, double qmax);
std::complex<double> strict_2d_schur_coefficient(const matrix_m<std::complex<double>> &lind,
                                                 double qx, double qy);
void validate_strict_2d_screening_denominator(const std::complex<double> &a, double qmax);
void validate_strict_2d_gw_coulomb_choices(bool strict_2d_headwing_active, bool use_fullcoul_eps,
                                           bool use_fullcoul_wc);
matrix_m<std::complex<double>> strict_2d_wc_blocks_at_q(
    const matrix_m<std::complex<double>> &body_inv,
    const matrix_m<std::complex<double>> &bw_direction,
    const matrix_m<std::complex<double>> &wb_direction, const std::complex<double> &schur_a,
    const matrix_m<std::complex<double>> &regular_body_sqrt, double q);
matrix_m<std::complex<double>> strict_2d_average_wc_coulomb_basis(
    const matrix_m<std::complex<double>> &body_inv, const matrix_m<std::complex<double>> &bw_cart,
    const matrix_m<std::complex<double>> &wb_cart, const matrix_m<std::complex<double>> &lind,
    const matrix_m<std::complex<double>> &regular_body_sqrt, const std::vector<double> &qx,
    const std::vector<double> &qy, const std::vector<double> &weights,
    const std::vector<double> &qmax, double gamma_area);
matrix_m<std::complex<double>> strict_2d_alpha_wc_average_coulomb_basis(
    double inverse_dielectric_alpha, const matrix_m<std::complex<double>> &regular_body_sqrt,
    const std::vector<double> &weights, const std::vector<double> &qmax, double gamma_area);
struct Strict2dCoulombHeadNormalization
{
    double inplane_area_bohr2 = 0.0;
    double auxiliary_monopole_norm_squared = 0.0;
    double auxiliary_head_coefficient = 0.0;
    double pw_to_auxiliary_scale = 0.0;
};

double strict_2d_inplane_cell_area(const PeriodicBoundaryData &pbc);
Strict2dCoulombHeadNormalization strict_2d_coulomb_head_normalization(
    const PeriodicBoundaryData &pbc, double auxiliary_monopole_norm_squared);
matrix_m<std::complex<double>> strict_2d_transform_pw_wc_to_auxiliary_basis(
    const matrix_m<std::complex<double>> &pw_wc, double pw_to_auxiliary_scale);
matrix_m<std::complex<double>> strict_2d_project_operator_to_coulomb_basis(
    const matrix_m<std::complex<double>> &operator_matrix,
    const matrix_m<std::complex<double>> &coulomb_eigenvectors);
double strict_2d_bare_coulomb_gamma_average(const std::vector<double> &weights,
                                            const std::vector<double> &qmax, double gamma_area);

ArrayDesc make_rpa_chi0v_wing_desc(const ArrayDesc &desc_body, const int wing_row_offset,
                                   const int wing_rows_loc, const int wing_cols_loc);

std::complex<double> compute_rpa_chi0v_headwing_trace_log_average(
    const matrix_m<std::complex<double>> &head, const matrix_m<std::complex<double>> &schur_l,
    const std::complex<double> &trace_body, const std::complex<double> &logdet_body,
    const std::vector<double> &qx, const std::vector<double> &qy, const std::vector<double> &qz,
    const std::vector<double> &weights, double *weight_sum_out = nullptr,
    std::complex<double> *averaged_body_out = nullptr,
    std::complex<double> *averaged_head_out = nullptr,
    std::complex<double> *averaged_schur_log_out = nullptr);

void replace_rpa_response_headwing(matrix_m<std::complex<double>> &response_block,
                                   const matrix_m<std::complex<double>> &head,
                                   const matrix_m<std::complex<double>> &wing,
                                   const ArrayDesc &desc_response);
void replace_rpa_response_head_only(matrix_m<std::complex<double>> &response_block,
                                    const matrix_m<std::complex<double>> &head,
                                    const ArrayDesc &desc_response);

}  // namespace librpa_int
