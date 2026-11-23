/*!
 * @file correlated_subspace.cpp
 * @brief Metric polar orthonormalization of local output orbitals.
 *
 * See correlated_subspace.h for the theoretical contract and conventions.
 *
 * Key storage convention (audited against production):
 *   MeanField::wfc is bands×NAO with wfc(ib, iao) = C_{iao, ib}, i.e.
 *   wfc = C^T (plain transpose, no conjugation).  The AO Green kernel is
 *   built in src/core/meanfield.cpp as transpose(wfc, false) * conj(wfc),
 *   which is C * C†.  Consequently C† = conj(wfc) and
 *   Q = C† S Φ = conj(wfc) * S * Φ.
 */

#include "correlated_subspace.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <set>
#include <sstream>

#include "../math/lapack_connector.h"

namespace librpa_int
{

ComplexMatrix project_trial_orbitals_to_band_window(const ComplexMatrix& wfc,
                                                    const ComplexMatrix& S,
                                                    const ComplexMatrix& trials,
                                                    const std::vector<int>& bands)
{
    const int nao = wfc.nc;
    if (nao <= 0 || wfc.nr <= 0 || !wfc.c || S.nr != nao || S.nc != nao || !S.c ||
        trials.nr != nao || trials.nc <= 0 || !trials.c ||
        bands.size() < static_cast<std::size_t>(trials.nc))
        throw LIBRPA_INVALID_ARGUMENT("invalid trial/window projection dimensions");
    std::set<int> unique;
    ComplexMatrix cw(nao, bands.size());
    for (std::size_t b = 0; b < bands.size(); ++b)
    {
        const int band = bands[b];
        if (band < 0 || band >= wfc.nr || !unique.insert(band).second)
            throw LIBRPA_INVALID_ARGUMENT("invalid or duplicate window band index");
        for (int i = 0; i < nao; ++i) cw(i, b) = wfc(band, i);
    }
    return cw * (transpose(cw, true) * (S * trials));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void CorrelatedSubspace::check_matrix(const ComplexMatrix& m, int expected_nr, int expected_nc,
                                      const std::string& name) const
{
    if (m.nr != expected_nr || m.nc != expected_nc)
    {
        std::ostringstream oss;
        oss << name << " has wrong shape (" << m.nr << "x" << m.nc << "), expected (" << expected_nr
            << "x" << expected_nc << ")";
        throw LIBRPA_INVALID_ARGUMENT(oss.str());
    }
    for (int i = 0; i < m.size; ++i)
    {
        if (!std::isfinite(m.c[i].real()) || !std::isfinite(m.c[i].imag()))
        {
            throw LIBRPA_INVALID_ARGUMENT(name + " contains non-finite values");
        }
    }
}

double CorrelatedSubspace::frobenius_norm(const ComplexMatrix& m) const
{
    double s = 0.0;
    for (int i = 0; i < m.size; ++i) s += std::norm(m.c[i]);
    return std::sqrt(s);
}

std::complex<double> CorrelatedSubspace::fourier_phase(const Vector3_Order<double>& kfrac,
                                                       const Vector3_Order<int>& R) const
{
    const double dot = kfrac.x * R.x + kfrac.y * R.y + kfrac.z * R.z;
    const double arg = dot * TWO_PI;
    return {std::cos(arg), std::sin(arg)};
}

// ---------------------------------------------------------------------------
// LAPACK diagonalization (row-major ComplexMatrix, following fixed_basis.cpp)
// ---------------------------------------------------------------------------

void CorrelatedSubspace::diagonalize_hermitian(ComplexMatrix& mat,
                                               std::vector<double>& eigenvalues) const
{
    const int n = mat.nr;
    assert(mat.nr == mat.nc);

    const int block_size = LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1);
    const int work_size = std::max(1, n * (block_size + 1));
    const int real_work_size = std::max(1, 3 * n - 2);

    eigenvalues.assign(n, 0.0);
    std::vector<complex<double>> work(work_size);
    std::vector<double> real_work(real_work_size);
    int info = 0;

    // ComplexMatrix is always row-major; use heev (row-major variant).
    LapackConnector::heev('V', 'U', n, mat.c, n, eigenvalues.data(), work.data(), work_size,
                          real_work.data(), info);

    if (info != 0)
    {
        std::ostringstream oss;
        oss << "zheev failed with info=" << info << " while diagonalizing " << n << "x" << n
            << " Hermitian matrix";
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
}

// ---------------------------------------------------------------------------
// Site group validation
// ---------------------------------------------------------------------------

void CorrelatedSubspace::validate_site_groups(const std::vector<SiteOrbitalGroup>& sites,
                                              int n_corr)
{
    if (sites.empty()) throw LIBRPA_INVALID_ARGUMENT("no correlated sites provided");

    // Check each site is valid.
    for (int i = 0; i < static_cast<int>(sites.size()); ++i)
    {
        const auto& s = sites[i];
        if (s.n_orbitals <= 0)
            throw LIBRPA_INVALID_ARGUMENT("site " + std::to_string(i) +
                                          " has non-positive orbital count");
        if (s.orb_start < 0)
            throw LIBRPA_INVALID_ARGUMENT("site " + std::to_string(i) + " has negative orb_start");
        if (s.orb_start + s.n_orbitals > n_corr)
            throw LIBRPA_INVALID_ARGUMENT("site " + std::to_string(i) + " exceeds n_corr bounds");
        if (s.label.empty())
            throw LIBRPA_INVALID_ARGUMENT("site " + std::to_string(i) + " has empty label");

        // If orbital_labels provided, validate count, non-emptiness, no duplicates.
        if (!s.orbital_labels.empty())
        {
            if (static_cast<int>(s.orbital_labels.size()) != s.n_orbitals)
                throw LIBRPA_INVALID_ARGUMENT(
                    "site " + std::to_string(i) + " orbital_labels count (" +
                    std::to_string(s.orbital_labels.size()) + ") != n_orbitals (" +
                    std::to_string(s.n_orbitals) + ")");
            std::set<std::string> seen;
            for (const auto& lbl : s.orbital_labels)
            {
                if (lbl.empty())
                    throw LIBRPA_INVALID_ARGUMENT("site " + std::to_string(i) +
                                                  " has empty orbital label");
                if (!seen.insert(lbl).second)
                    throw LIBRPA_INVALID_ARGUMENT("site " + std::to_string(i) +
                                                  " has duplicate orbital label: " + lbl);
            }
        }
    }

    // Check no overlaps: collect all covered columns.
    std::vector<bool> covered(n_corr, false);
    for (const auto& s : sites)
    {
        for (int j = s.orb_start; j < s.orb_start + s.n_orbitals; ++j)
        {
            if (covered[j])
            {
                throw LIBRPA_INVALID_ARGUMENT("column " + std::to_string(j) +
                                              " is claimed by more than one site");
            }
            covered[j] = true;
        }
    }

    // Check no gaps.
    for (int j = 0; j < n_corr; ++j)
    {
        if (!covered[j])
        {
            throw LIBRPA_INVALID_ARGUMENT("column " + std::to_string(j) +
                                          " is not assigned to any site (gap in site groups)");
        }
    }

    // Check no duplicate site labels.
    for (int i = 0; i < static_cast<int>(sites.size()); ++i)
        for (int j = i + 1; j < static_cast<int>(sites.size()); ++j)
            if (sites[i].label == sites[j].label)
                throw LIBRPA_INVALID_ARGUMENT("duplicate site label: " + sites[i].label);

    // If atom_index >= 0, check no duplicate atom indices across sites.
    std::set<int> atom_indices;
    for (int i = 0; i < static_cast<int>(sites.size()); ++i)
    {
        if (sites[i].atom_index >= 0)
        {
            if (!atom_indices.insert(sites[i].atom_index).second)
                throw LIBRPA_INVALID_ARGUMENT("duplicate atom_index " +
                                              std::to_string(sites[i].atom_index) + " in site " +
                                              std::to_string(i));
        }
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

CorrelatedSubspace::CorrelatedSubspace(const std::vector<SiteOrbitalGroup>& sites,
                                       const std::vector<Vector3_Order<double>>& kfrac_list,
                                       const std::vector<Vector3_Order<int>>& R_list, int nao,
                                       int nbands, int n_spins, double abs_threshold,
                                       double rel_threshold, double cond_max,
                                       double s_abs_threshold, double s_rel_threshold,
                                       double s_cond_max, double residual_tol,
                                       double t_roundtrip_tol)
    : sites_(sites),
      kfrac_list_(kfrac_list),
      R_list_(R_list),
      nao_(nao),
      nbands_(nbands),
      n_spins_(n_spins),
      n_kpoints_(static_cast<int>(kfrac_list.size())),
      abs_threshold_(abs_threshold),
      rel_threshold_(rel_threshold),
      cond_max_(cond_max),
      s_abs_threshold_(s_abs_threshold),
      s_rel_threshold_(s_rel_threshold),
      s_cond_max_(s_cond_max),
      residual_tol_(residual_tol),
      t_roundtrip_tol_(t_roundtrip_tol),
      T_valid_(false)
{
    if (nao_ <= 0) throw LIBRPA_INVALID_ARGUMENT("nao must be positive");
    if (nbands_ <= 0) throw LIBRPA_INVALID_ARGUMENT("nbands must be positive");
    if (n_spins <= 0 || n_spins > 2)
        throw LIBRPA_INVALID_ARGUMENT("n_spins must be 1 or 2 for scalar");
    if (n_kpoints_ <= 0) throw LIBRPA_INVALID_ARGUMENT("kfrac_list must not be empty");
    if (R_list_.empty()) throw LIBRPA_INVALID_ARGUMENT("R_list must not be empty");
    if (abs_threshold_ <= 0.0) throw LIBRPA_INVALID_ARGUMENT("abs_threshold must be positive");
    if (rel_threshold_ <= 0.0) throw LIBRPA_INVALID_ARGUMENT("rel_threshold must be positive");
    if (cond_max_ <= 1.0) throw LIBRPA_INVALID_ARGUMENT("cond_max must be > 1");
    if (s_abs_threshold_ <= 0.0) throw LIBRPA_INVALID_ARGUMENT("s_abs_threshold must be positive");
    if (s_rel_threshold_ <= 0.0) throw LIBRPA_INVALID_ARGUMENT("s_rel_threshold must be positive");
    if (s_cond_max_ <= 1.0) throw LIBRPA_INVALID_ARGUMENT("s_cond_max must be > 1");
    if (residual_tol_ <= 0.0) throw LIBRPA_INVALID_ARGUMENT("residual_tol must be positive");
    if (t_roundtrip_tol_ <= 0.0) throw LIBRPA_INVALID_ARGUMENT("t_roundtrip_tol must be positive");

    // Compute n_corr from site groups.
    n_corr_ = 0;
    for (const auto& s : sites_) n_corr_ = std::max(n_corr_, s.orb_start + s.n_orbitals);

    if (n_corr_ <= 0) throw LIBRPA_INVALID_ARGUMENT("n_corr must be positive");
    if (n_corr_ > nao_)
        throw LIBRPA_INVALID_ARGUMENT("n_corr (" + std::to_string(n_corr_) +
                                      ") must not exceed nao (" + std::to_string(nao_) + ")");
    if (n_corr_ > nbands_)
        throw LIBRPA_INVALID_ARGUMENT("n_corr must not exceed nbands for a valid projection");

    validate_site_groups(sites_, n_corr_);

    // Allocate per-k storage.
    S_k_.resize(n_kpoints_);
    W_k_.resize(n_kpoints_);
    phi_k_.resize(n_kpoints_);
    phi_valid_.assign(n_kpoints_, false);
    kpt_diag_.resize(n_kpoints_);
    kpt_diag_valid_.assign(n_kpoints_, false);

    // Allocate per-(spin,k) storage.
    spin_kpt_diag_.resize(n_spins_);
    for (int is = 0; is < n_spins_; ++is)
    {
        spin_kpt_diag_[is].resize(n_kpoints_);
        for (int ik = 0; ik < n_kpoints_; ++ik) spin_kpt_diag_[is][ik].computed = false;
    }

    reset_diagnostics();
    diag_.n_kpoints = n_kpoints_;
    diag_.n_spins = n_spins_;
    diag_.nao = nao_;
    diag_.nbands = nbands_;
    diag_.n_corr = n_corr_;
    diag_.n_sites = static_cast<int>(sites_.size());
    diag_.t_computed = false;
}

// ---------------------------------------------------------------------------
// Cache invalidation
// ---------------------------------------------------------------------------

void CorrelatedSubspace::invalidate_k(int ik)
{
    phi_valid_[ik] = false;
    phi_k_[ik] = ComplexMatrix();  // release storage
    kpt_diag_valid_[ik] = false;
    // Invalidate all spins for this k.
    for (int is = 0; is < n_spins_; ++is)
    {
        spin_kpt_diag_[is][ik].computed = false;
    }
    // Invalidate T.
    T_valid_ = false;
    T_R_.clear();
    // Immediately recompute so get_diagnostics() never returns stale values.
    recompute_aggregate();
}

void CorrelatedSubspace::reset_diagnostics()
{
    diag_.max_gram_condition = 0.0;
    diag_.min_gram_lambda = std::numeric_limits<double>::max();
    diag_.max_gram_lambda = 0.0;
    diag_.max_phi_sphi_residual = 0.0;
    diag_.max_csc_residual = 0.0;
    diag_.max_q_completeness_residual = 0.0;
    diag_.max_t_roundtrip_residual = 0.0;
    diag_.min_S_lambda = std::numeric_limits<double>::max();
    diag_.max_S_lambda = 0.0;
    diag_.max_S_condition = 0.0;
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void CorrelatedSubspace::set_S_k(int ik, const ComplexMatrix& S)
{
    if (ik < 0 || ik >= n_kpoints_) throw LIBRPA_INVALID_ARGUMENT("k index out of range");
    check_matrix(S, nao_, nao_, "S(k)");
    S_k_[ik] = S;
    invalidate_k(ik);
}

void CorrelatedSubspace::set_W_k(int ik, const ComplexMatrix& W)
{
    if (ik < 0 || ik >= n_kpoints_) throw LIBRPA_INVALID_ARGUMENT("k index out of range");
    check_matrix(W, nao_, n_corr_, "W(k)");
    W_k_[ik] = W;
    invalidate_k(ik);
}

// ---------------------------------------------------------------------------
// S validation (fail-closed, full-matrix scale-aware)
// ---------------------------------------------------------------------------

void CorrelatedSubspace::check_S_k(int ik)
{
    if (S_k_[ik].nr != nao_ || S_k_[ik].nc != nao_)
    {
        throw LIBRPA_RUNTIME_ERROR("S(k) not set for k=" + std::to_string(ik));
    }

    const auto& S = S_k_[ik];

    // Full-matrix scale: scan all elements for maximum absolute value.
    double max_abs = 0.0;
    for (int i = 0; i < S.size; ++i) max_abs = std::max(max_abs, std::abs(S.c[i]));
    const double scale = std::max(max_abs, 1.0);

    // Hermiticity check with abs + rel*scale tolerance.
    const double hermit_tol = 1.0e-10 + 1.0e-10 * scale;
    for (int i = 0; i < nao_; ++i)
    {
        // Diagonal must be real (imaginary part ~ 0).
        if (std::abs(S(i, i).imag()) > hermit_tol)
        {
            throw LIBRPA_RUNTIME_ERROR("S(k=" + std::to_string(ik) + ") diagonal (" +
                                       std::to_string(i) + "," + std::to_string(i) +
                                       ") has non-zero imaginary part " +
                                       std::to_string(S(i, i).imag()));
        }
        for (int j = i + 1; j < nao_; ++j)
        {
            auto diff = S(i, j) - std::conj(S(j, i));
            if (std::abs(diff) > hermit_tol)
            {
                throw LIBRPA_RUNTIME_ERROR("S(k=" + std::to_string(ik) + ") is not Hermitian at (" +
                                           std::to_string(i) + "," + std::to_string(j) +
                                           "), |diff|=" + std::to_string(std::abs(diff)) +
                                           " > tol=" + std::to_string(hermit_tol));
            }
        }
    }

    // Positive-definiteness via eigenvalues.
    ComplexMatrix S_copy = S;
    std::vector<double> evals;
    diagonalize_hermitian(S_copy, evals);
    double slmin = *std::min_element(evals.begin(), evals.end());
    double slmax = *std::max_element(evals.begin(), evals.end());

    if (!std::isfinite(slmin) || !std::isfinite(slmax))
        throw LIBRPA_RUNTIME_ERROR("S(k=" + std::to_string(ik) + ") has non-finite eigenvalues");

    // Absolute threshold.
    if (slmin <= s_abs_threshold_)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "S(k=" + std::to_string(ik) +
            ") is not positive-definite (min eigenvalue=" + std::to_string(slmin) +
            " <= s_abs_threshold " + std::to_string(s_abs_threshold_) + ")");
    }
    // Relative threshold.
    if (slmin <= s_rel_threshold_ * slmax)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "S(k=" + std::to_string(ik) +
            ") is not positive-definite (min eigenvalue=" + std::to_string(slmin) +
            " <= s_rel_threshold * max (" + std::to_string(s_rel_threshold_ * slmax) + ")");
    }
    // Condition number.
    double s_cond = slmax / slmin;
    if (s_cond > s_cond_max_)
    {
        throw LIBRPA_RUNTIME_ERROR("S(k=" + std::to_string(ik) +
                                   ") is ill-conditioned: condition " + std::to_string(s_cond) +
                                   " > s_cond_max " + std::to_string(s_cond_max_));
    }

    // Record S diagnostics into per-k diagnostics.
    if (kpt_diag_valid_[ik])
    {
        kpt_diag_[ik].S_lambda_min = slmin;
        kpt_diag_[ik].S_lambda_max = slmax;
        kpt_diag_[ik].S_condition = s_cond;
    }
    else
    {
        // Pre-populate S diagnostics before build_phi fills the rest.
        SubspaceKptDiagnostics d;
        d.S_lambda_min = slmin;
        d.S_lambda_max = slmax;
        d.S_condition = s_cond;
        d.gram_lambda_min = 0.0;
        d.gram_lambda_max = 0.0;
        d.gram_condition = 0.0;
        d.phi_sphi_residual = 0.0;
        d.band_basis_complete = (nbands_ == nao_);
        d.rank = 0;
        kpt_diag_[ik] = d;
        kpt_diag_valid_[ik] = true;
    }
}

// ---------------------------------------------------------------------------
// Build Φ(k)
// ---------------------------------------------------------------------------

ComplexMatrix CorrelatedSubspace::build_phi(int ik)
{
    if (ik < 0 || ik >= n_kpoints_) throw LIBRPA_INVALID_ARGUMENT("k index out of range");

    // Fail-closed S checks (Hermiticity + positive-definiteness) are part of
    // the build, not an external precondition.
    check_S_k(ik);

    if (W_k_[ik].nr != nao_ || W_k_[ik].nc != n_corr_)
        throw LIBRPA_RUNTIME_ERROR("W(k) not set for k=" + std::to_string(ik));

    // Cache-valid fast path: if S/W unchanged since last build, return cached Φ.
    if (phi_valid_[ik]) return phi_k_[ik];

    const auto& S = S_k_[ik];
    const auto& W = W_k_[ik];

    // O = W† S W  (Nd × Nd)
    ComplexMatrix Wd = transpose(W, true);  // Nd × NAO
    ComplexMatrix SW = S * W;               // NAO × Nd
    ComplexMatrix O = Wd * SW;              // Nd × Nd

    // Diagonalize O.
    std::vector<double> evals;
    diagonalize_hermitian(O, evals);

    const int nd = n_corr_;
    double lmin = evals[0];
    double lmax = evals[0];
    for (int i = 0; i < nd; ++i)
    {
        lmin = std::min(lmin, evals[i]);
        lmax = std::max(lmax, evals[i]);
    }

    // Check rank and conditioning: fail closed.
    if (!std::isfinite(lmin) || !std::isfinite(lmax))
        throw LIBRPA_RUNTIME_ERROR("O(k=" + std::to_string(ik) + ") has non-finite eigenvalues");
    if (lmin <= abs_threshold_)
    {
        std::ostringstream oss;
        oss << "O(k=" << ik << ") is rank-deficient: min eigenvalue " << lmin
            << " <= abs_threshold " << abs_threshold_;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
    if (lmin <= rel_threshold_ * lmax)
    {
        std::ostringstream oss;
        oss << "O(k=" << ik << ") is rank-deficient: min eigenvalue " << lmin
            << " <= rel_threshold * max (" << rel_threshold_ * lmax << ")";
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
    double cond = lmax / lmin;
    if (cond > cond_max_)
    {
        std::ostringstream oss;
        oss << "O(k=" << ik << ") is ill-conditioned: condition " << cond << " > cond_max "
            << cond_max_ << " (λmin=" << lmin << ", λmax=" << lmax << ")";
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }

    // Construct O^{-1/2} = U diag(λ^{-1/2}) U†.
    ComplexMatrix Ud = transpose(O, true);  // U†
    for (int i = 0; i < nd; ++i)
    {
        const double inv_sqrt = 1.0 / std::sqrt(evals[i]);
        for (int j = 0; j < nd; ++j) Ud.c[i * nd + j] *= inv_sqrt;
    }
    ComplexMatrix O_inv_sqrt = O * Ud;  // nd × nd

    // Φ = W O^{-1/2}  (NAO × Nd)
    ComplexMatrix phi = W * O_inv_sqrt;

    // Diagnostics: ||Φ†SΦ - I||  (must pass fail-closed gate)
    ComplexMatrix PhiS = S * phi;                  // NAO × Nd
    ComplexMatrix Phi_dag = transpose(phi, true);  // Nd × NAO
    ComplexMatrix PhiSPhi = Phi_dag * PhiS;        // Nd × Nd
    ComplexMatrix I_nd(nd, nd);
    I_nd.set_as_identity_matrix();
    ComplexMatrix resid = PhiSPhi - I_nd;
    double phi_sphi_resid = frobenius_norm(resid);

    // Fail-closed: Φ†SΦ residual must be within tolerance * sqrt(nd).
    const double phi_tol = residual_tol_ * std::sqrt(static_cast<double>(nd));
    if (phi_sphi_resid > phi_tol)
    {
        std::ostringstream oss;
        oss << "Φ†SΦ residual " << phi_sphi_resid << " > tolerance " << phi_tol << " for k=" << ik;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }

    // Store per-k diagnostics (preserve S diagnostics from check_S_k).
    SubspaceKptDiagnostics d = kpt_diag_[ik];
    d.gram_lambda_min = lmin;
    d.gram_lambda_max = lmax;
    d.gram_condition = cond;
    d.phi_sphi_residual = phi_sphi_resid;
    d.band_basis_complete = (nbands_ == nao_);
    d.rank = nd;

    phi_k_[ik] = phi;
    phi_valid_[ik] = true;
    kpt_diag_[ik] = d;
    kpt_diag_valid_[ik] = true;

    // Invalidate T since Φ changed.
    T_valid_ = false;
    T_R_.clear();

    recompute_aggregate();

    return phi;
}

// ---------------------------------------------------------------------------
// Validate KS orthogonality and local-orbital capture
// ---------------------------------------------------------------------------

SpinKBuildResult CorrelatedSubspace::build_spin_k(int ispin, int ik, const ComplexMatrix& wfc)
{
    if (ispin < 0 || ispin >= n_spins_) throw LIBRPA_INVALID_ARGUMENT("spin index out of range");
    if (ik < 0 || ik >= n_kpoints_) throw LIBRPA_INVALID_ARGUMENT("k index out of range");
    check_matrix(wfc, nbands_, nao_, "wfc");

    // Use cache-valid fast path: if S/W unchanged since last build_phi(ik),
    // returns cached Φ.  build_phi performs its own fail-closed S checks.
    ComplexMatrix phi = build_phi(ik);

    const auto& S = S_k_[ik];

    // C† = conj(wfc) because wfc = C^T (production storage convention).
    // Q = C† S Φ = conj(wfc) * S * Φ  (bands × Nd)
    ComplexMatrix wfc_conj = conj(wfc);  // bands × NAO
    ComplexMatrix SPhi = S * phi;        // NAO × Nd
    ComplexMatrix Q = wfc_conj * SPhi;   // bands × Nd

    // Check the supplied KS window before evaluating normalized character.
    // Orthogonality is required for truncated windows too; completeness of
    // the supplied band space is a separate Q†Q condition below.
    const ComplexMatrix C_col = transpose(wfc, false);
    const ComplexMatrix SC = S * C_col;
    const ComplexMatrix CtSC = wfc_conj * SC;
    ComplexMatrix I_nb(nbands_, nbands_);
    I_nb.set_as_identity_matrix();
    const double csc_resid = frobenius_norm(CtSC - I_nb);
    const double csc_tol = residual_tol_ * std::sqrt(static_cast<double>(nbands_));
    if (!std::isfinite(csc_resid) || csc_resid > csc_tol)
    {
        std::ostringstream oss;
        oss.precision(17);
        oss << "C†SC residual " << csc_resid << " > tolerance " << csc_tol << " for (spin=" << ispin
            << ", k=" << ik << ")";
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }

    // Q†Q eigenspectrum (always computed — valuable diagnostic for truncated bands)
    // NOTE: diagonalize_hermitian overwrites its input with eigenvectors, so we
    // must compute the completeness residual from the original QtQ first.
    ComplexMatrix QtQ = transpose(Q, true) * Q;  // Nd × Nd (Hermitian)
    ComplexMatrix I_nd(n_corr_, n_corr_);
    I_nd.set_as_identity_matrix();
    ComplexMatrix qresid = QtQ - I_nd;
    double q_completeness_resid = frobenius_norm(qresid);

    // Now diagonalize a copy for the eigenspectrum.
    std::vector<double> qqt_evals;
    diagonalize_hermitian(QtQ, qqt_evals);
    double qqt_lmin = *std::min_element(qqt_evals.begin(), qqt_evals.end());
    double qqt_lmax = *std::max_element(qqt_evals.begin(), qqt_evals.end());
    double qqt_trace = 0.0;
    for (double v : qqt_evals) qqt_trace += v;
    const double qqt_captured_fraction = qqt_trace / static_cast<double>(n_corr_);

    // Additional completeness gate for a complete supplied band basis.
    const double q_tol = residual_tol_ * std::sqrt(static_cast<double>(n_corr_));

    if (nbands_ == nao_)
    {
        if (q_completeness_resid > q_tol)
        {
            std::ostringstream oss;
            oss << "Q†Q completeness residual " << q_completeness_resid << " > tolerance " << q_tol
                << " for (spin=" << ispin << ", k=" << ik << ")";
            throw LIBRPA_RUNTIME_ERROR(oss.str());
        }
    }

    // Store per-(spin,k) diagnostics.
    SpinKptDiagnostics sd;
    sd.csc_residual = csc_resid;
    sd.q_completeness_residual = q_completeness_resid;
    sd.qqt_lambda_min = qqt_lmin;
    sd.qqt_lambda_max = qqt_lmax;
    sd.qqt_captured_fraction = qqt_captured_fraction;
    sd.computed = true;
    spin_kpt_diag_[ispin][ik] = sd;

    recompute_aggregate();

    SpinKBuildResult result;
    result.Q = Q;
    result.kpt_diag = kpt_diag_[ik];
    result.spin_kpt_diag = sd;
    return result;
}

void CorrelatedSubspace::verify_fourier_duality() const
{
    const int nk = n_kpoints_;
    const int nR = static_cast<int>(R_list_.size());

    if (nR != nk)
    {
        std::ostringstream oss;
        oss << "R_list size (" << nR << ") != kfrac_list size (" << nk
            << "); Fourier pair must be dual";
        throw LIBRPA_INVALID_ARGUMENT(oss.str());
    }

    // Check for duplicate R vectors.
    for (int i = 0; i < nR; ++i)
        for (int j = i + 1; j < nR; ++j)
        {
            if (R_list_[i].x == R_list_[j].x && R_list_[i].y == R_list_[j].y &&
                R_list_[i].z == R_list_[j].z)
            {
                throw LIBRPA_INVALID_ARGUMENT("duplicate R vector at index " + std::to_string(i) +
                                              " and " + std::to_string(j));
            }
        }

    // Check for duplicate k vectors.
    for (int i = 0; i < nk; ++i)
        for (int j = i + 1; j < nk; ++j)
        {
            const auto tol = 1e-12;
            if (std::abs(kfrac_list_[i].x - kfrac_list_[j].x) < tol &&
                std::abs(kfrac_list_[i].y - kfrac_list_[j].y) < tol &&
                std::abs(kfrac_list_[i].z - kfrac_list_[j].z) < tol)
            {
                throw LIBRPA_INVALID_ARGUMENT("duplicate kfrac vector at index " +
                                              std::to_string(i) + " and " + std::to_string(j));
            }
        }

    // Verify closure: (1/Nk) sum_R exp(i 2pi (k_i - k_j) . R) = delta_{ij}
    const double closure_tol = 1.0e-10;
    for (int ik1 = 0; ik1 < nk; ++ik1)
    {
        for (int ik2 = 0; ik2 < nk; ++ik2)
        {
            Vector3_Order<double> dk{kfrac_list_[ik1].x - kfrac_list_[ik2].x,
                                     kfrac_list_[ik1].y - kfrac_list_[ik2].y,
                                     kfrac_list_[ik1].z - kfrac_list_[ik2].z};
            std::complex<double> sum(0.0, 0.0);
            for (int iR = 0; iR < nR; ++iR)
            {
                const double dot =
                    dk.x * R_list_[iR].x + dk.y * R_list_[iR].y + dk.z * R_list_[iR].z;
                const double arg = dot * TWO_PI;
                sum += std::complex<double>(std::cos(arg), std::sin(arg));
            }
            sum /= static_cast<double>(nk);
            const double expected = (ik1 == ik2) ? 1.0 : 0.0;
            if (std::abs(sum - std::complex<double>(expected, 0.0)) > closure_tol)
            {
                std::ostringstream oss;
                oss << "Fourier closure failed: (1/Nk) sum_R exp(i2pi (k" << ik1 << "-k" << ik2
                    << ").R) = " << sum << " expected " << expected;
                throw LIBRPA_INVALID_ARGUMENT(oss.str());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Real-space T(R) via Fourier
// ---------------------------------------------------------------------------

void CorrelatedSubspace::compute_T_R()
{
    if (T_valid_) return;

    // Fail-closed: verify k/R duality before proceeding.
    verify_fourier_duality();

    // Build Φ(k) for all k (with fail-closed S checks).
    for (int ik = 0; ik < n_kpoints_; ++ik) build_phi(ik);

    const int n_R = static_cast<int>(R_list_.size());
    T_R_.resize(n_R);

    // T(R) = (1/Nk) Σ_k exp(-i2π k·R) Φ(k)
    for (int iR = 0; iR < n_R; ++iR)
    {
        T_R_[iR] = ComplexMatrix(nao_, n_corr_);
        T_R_[iR].zero_out();

        for (int ik = 0; ik < n_kpoints_; ++ik)
        {
            const auto phase = fourier_phase(kfrac_list_[ik], R_list_[iR]);
            const auto phase_conj = std::conj(phase);
            for (int i = 0; i < nao_; ++i)
                for (int j = 0; j < n_corr_; ++j) T_R_[iR](i, j) += phase_conj * phi_k_[ik](i, j);
        }

        const double inv_nk = 1.0 / static_cast<double>(n_kpoints_);
        for (int idx = 0; idx < T_R_[iR].size; ++idx) T_R_[iR].c[idx] *= inv_nk;
    }

    T_valid_ = true;
    diag_.t_computed = true;

    // Fail-closed: round-trip residual must be within tolerance.
    double resid = compute_t_roundtrip_residual();
    diag_.max_t_roundtrip_residual = resid;
    if (resid > t_roundtrip_tol_)
    {
        std::ostringstream oss;
        oss << "T(R) round-trip residual " << resid << " > tolerance " << t_roundtrip_tol_;
        throw LIBRPA_RUNTIME_ERROR(oss.str());
    }
}

const ComplexMatrix& CorrelatedSubspace::get_T(int iR) const
{
    if (T_R_.empty() || !T_valid_)
        throw LIBRPA_RUNTIME_ERROR("T(R) not computed; call compute_T_R() first");
    if (iR < 0 || iR >= static_cast<int>(T_R_.size()))
        throw LIBRPA_INVALID_ARGUMENT("R index out of range");
    return T_R_[iR];
}

ComplexMatrix CorrelatedSubspace::get_T_site(int iR, int site_index) const
{
    if (site_index < 0 || site_index >= static_cast<int>(sites_.size()))
        throw LIBRPA_INVALID_ARGUMENT("site index out of range");
    const auto& T = get_T(iR);
    const auto& site = sites_[site_index];
    ComplexMatrix result(nao_, site.n_orbitals);
    for (int i = 0; i < nao_; ++i)
        for (int j = 0; j < site.n_orbitals; ++j) result(i, j) = T(i, site.orb_start + j);
    return result;
}

double CorrelatedSubspace::compute_t_roundtrip_residual() const
{
    if (T_R_.empty() || !T_valid_) throw LIBRPA_RUNTIME_ERROR("T(R) not computed");

    double max_resid = 0.0;
    const int n_R = static_cast<int>(R_list_.size());

    for (int ik = 0; ik < n_kpoints_; ++ik)
    {
        // Φ_reconstructed(k) = Σ_R exp(+i2π k·R) T(R)
        ComplexMatrix phi_recon(nao_, n_corr_);
        phi_recon.zero_out();

        for (int iR = 0; iR < n_R; ++iR)
        {
            const auto phase = fourier_phase(kfrac_list_[ik], R_list_[iR]);
            for (int i = 0; i < nao_; ++i)
                for (int j = 0; j < n_corr_; ++j) phi_recon(i, j) += phase * T_R_[iR](i, j);
        }

        // Residual = ||Φ_recon - Φ||
        ComplexMatrix diff = phi_recon - phi_k_[ik];
        double resid = frobenius_norm(diff);
        max_resid = std::max(max_resid, resid);
    }

    return max_resid;
}

// ---------------------------------------------------------------------------
// Aggregate diagnostics
// ---------------------------------------------------------------------------

void CorrelatedSubspace::recompute_aggregate()
{
    reset_diagnostics();
    for (int ik = 0; ik < n_kpoints_; ++ik)
    {
        if (!kpt_diag_valid_[ik]) continue;
        const auto& d = kpt_diag_[ik];
        diag_.max_gram_condition = std::max(diag_.max_gram_condition, d.gram_condition);
        diag_.min_gram_lambda = std::min(diag_.min_gram_lambda, d.gram_lambda_min);
        diag_.max_gram_lambda = std::max(diag_.max_gram_lambda, d.gram_lambda_max);
        diag_.max_phi_sphi_residual = std::max(diag_.max_phi_sphi_residual, d.phi_sphi_residual);
        diag_.min_S_lambda = std::min(diag_.min_S_lambda, d.S_lambda_min);
        diag_.max_S_lambda = std::max(diag_.max_S_lambda, d.S_lambda_max);
        diag_.max_S_condition = std::max(diag_.max_S_condition, d.S_condition);
    }
    // Aggregate per-(spin,k) band/Q diagnostics.
    for (int is = 0; is < n_spins_; ++is)
    {
        for (int ik = 0; ik < n_kpoints_; ++ik)
        {
            if (!spin_kpt_diag_[is][ik].computed) continue;
            const auto& sd = spin_kpt_diag_[is][ik];
            if (sd.csc_residual >= 0.0)
                diag_.max_csc_residual = std::max(diag_.max_csc_residual, sd.csc_residual);
            if (sd.q_completeness_residual >= 0.0)
                diag_.max_q_completeness_residual =
                    std::max(diag_.max_q_completeness_residual, sd.q_completeness_residual);
        }
    }
    diag_.t_computed = T_valid_;
}

}  // namespace librpa_int
