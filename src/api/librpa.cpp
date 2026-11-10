// Public API headers
#include "librpa.hpp"
#include "librpa_global.h"
#include "librpa_input.h"
#include "librpa_compute.h"

#include <cstddef>
#include <complex>
#include <limits>
#include <utility>

// Internal headers
#include "../utils/error.h"
#include "../mpi/collective_stage.h"
#include "instance_manager.h"

namespace librpa
{

QsgwTask Handler::create_qsgw_task(const Options& opts, const LibrpaQsgwControls& controls) const
{
    return QsgwTask(::librpa_create_qsgw_task(h_, &opts, &controls), ::librpa_delete_qsgw_task);
}

CrpaContinuedResult continue_crpa(const LibrpaCrpaResult* result,
                                  const std::vector<std::complex<double>>& queries,
                                  int n_parameters, double tolerance_ha)
{
    return CrpaContinuedResult(
        ::librpa_continue_crpa(result, queries.size(),
                               reinterpret_cast<const double*>(queries.data()), n_parameters,
                               tolerance_ha),
        ::librpa_delete_crpa_continued_result);
}

CrpaContinuedResult continue_crpa(const LibrpaCrpaResult* result,
                                  const std::vector<std::complex<double>>& queries,
                                  const Options& options, double tolerance_ha)
{
    return CrpaContinuedResult(
        ::librpa_continue_crpa_with_options(result, queries.size(),
                                            reinterpret_cast<const double*>(queries.data()),
                                            &options, tolerance_ha),
        ::librpa_delete_crpa_continued_result);
}

OrbitalResult Handler::analyze_orbitals(int nk, int nao, const std::vector<double>& overlap,
                                        double tolerance) const
{
    const bool valid_size =
        nk > 0 && nao > 0 && overlap.size() == 2 * static_cast<std::size_t>(nk) * nao * nao;
    return OrbitalResult(
        ::librpa_analyze_orbitals(h_, nk, nao, valid_size ? overlap.data() : nullptr, tolerance),
        ::librpa_delete_orbital_result);
}

CrpaResult Handler::build_g0w0_sigma_crpa(const Options& opts, const LibrpaCrpaInput& input)
{
    return CrpaResult(::librpa_build_g0w0_sigma_crpa(h_, &opts, &input),
                      ::librpa_delete_crpa_result);
}

CrpaResult Handler::compute_crpa_window(const Options& opts, const LibrpaCrpaInput& input) const
{
    return CrpaResult(::librpa_compute_crpa_window(h_, &opts, &input), ::librpa_delete_crpa_result);
}

CrpaResult Handler::compute_crpa_u_window(const Options& opts, const LibrpaCrpaInput& input) const
{
    return CrpaResult(::librpa_compute_crpa_u_window(h_, &opts, &input),
                      ::librpa_delete_crpa_result);
}

CrpaResult Handler::prepare_crpa_model(const Options& opts, const LibrpaCrpaInput& input) const
{
    return CrpaResult(::librpa_prepare_crpa_model(h_, &opts, &input), librpa_delete_crpa_result);
}

std::array<std::complex<double>, 3> crpa_tensor_averages(
    int n_orbitals, const std::vector<std::complex<double>>& tensor)
{
    if (n_orbitals <= 0 || tensor.size() != static_cast<std::size_t>(n_orbitals) * n_orbitals *
                                                n_orbitals * n_orbitals)
        throw LIBRPA_INVALID_ARGUMENT("incomplete cRPA tensor for orbital averages");
    std::array<std::complex<double>, 3> result;
    ::librpa_crpa_tensor_averages(n_orbitals, reinterpret_cast<const double*>(tensor.data()),
                                  reinterpret_cast<double*>(result.data()));
    return result;
}


/* Global environment functions */
const char* get_build_info(void) { return ::librpa_get_build_info(); }

int get_major_version(void) { return ::librpa_get_major_version(); }

int get_minor_version(void) { return ::librpa_get_minor_version(); }

int get_patch_version(void) { return ::librpa_get_patch_version(); }

void init_global(LibrpaSwitch switch_redirect_stdout, const char *redirect_path,
                 LibrpaSwitch switch_process_output)
{
    ::librpa_init_global(switch_redirect_stdout, redirect_path, switch_process_output);
}

void set_output_level(Verbose output_level) { ::librpa_set_output_level(output_level); }

Verbose get_output_level(void) { return ::librpa_get_output_level(); }

void finalize_global(void) { ::librpa_finalize_global(); }

void test(void) { ::librpa_test(); }

void print_profile(void) { ::librpa_print_profile(); }

/* Constructor and destructor definition for the C++ handler object */
Handler::Handler(MPI_Comm comm) : h_(nullptr) { this->init(comm); }

void Handler::init(MPI_Comm comm)
{
    if (h_) throw LIBRPA_RUNTIME_ERROR("double creating handler");  // safe guard
    h_ = ::librpa_create_handler(comm);
}

void Handler::free()
{
    if (h_) ::librpa_destroy_handler(h_);
    h_ = nullptr;
}

Handler::~Handler() { free(); }

namespace
{

void flatten_l_shells(const std::vector<std::vector<int>> &l_shells,
                      std::vector<int> &nshells,
                      std::vector<int> &flat_l_shells)
{
    nshells.clear();
    flat_l_shells.clear();
    nshells.reserve(l_shells.size());
    for (const auto &atom_l_shells : l_shells)
    {
        nshells.push_back(static_cast<int>(atom_l_shells.size()));
        flat_l_shells.insert(flat_l_shells.end(), atom_l_shells.begin(), atom_l_shells.end());
    }
}

void basis_l_shell_ptrs(const std::size_t natoms,
                        const std::vector<std::vector<int>> &l_shells,
                        std::vector<int> &nshells,
                        std::vector<int> &flat_l_shells,
                        const int *&nshells_ptr,
                        const int *&l_shells_ptr)
{
    nshells_ptr = nullptr;
    l_shells_ptr = nullptr;
    if (l_shells.empty())
    {
        return;
    }
    if (l_shells.size() != natoms)
    {
        throw LIBRPA_RUNTIME_ERROR("l-shell metadata must have one entry per atom");
    }

    flatten_l_shells(l_shells, nshells, flat_l_shells);
    nshells_ptr = nshells.data();
    l_shells_ptr = flat_l_shells.empty() ? nullptr : flat_l_shells.data();
}

} // namespace

// The following macros are helper to implement the methods of the C++ handler class
// that wraps the corresponding `librpa_` prefixed C-API functions.
#define LIBRPA_C_NAME(name) librpa_##name
#define LIBRPA_UNPAREN(x) LIBRPA_UNPAREN_ x
#define LIBRPA_UNPAREN_(...) __VA_ARGS__

#define LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(cpp_name, params, args)  \
    void Handler::cpp_name(LIBRPA_UNPAREN(params))                 \
    {                                                              \
        ::LIBRPA_C_NAME(cpp_name)(this->h_, LIBRPA_UNPAREN(args)); \
    }

#define LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID_WOPT(cpp_name, params, args)        \
    void Handler::cpp_name(const librpa::Options& opts, LIBRPA_UNPAREN(params)) \
    {                                                                         \
        ::LIBRPA_C_NAME(cpp_name)(this->h_, &opts, LIBRPA_UNPAREN(args));      \
    }

#define LIBRPA_CPP_H_METHOD_DEF_WRAP_RET(ret, cpp_name, params, pre, args, post)   \
    ret Handler::cpp_name(LIBRPA_UNPAREN(params))                                  \
    {                                                                              \
        pre auto retc = ::LIBRPA_C_NAME(cpp_name)(this->h_, LIBRPA_UNPAREN(args)); \
        post return retc;                                                          \
    }

#define LIBRPA_CPP_H_METHOD_DEF_WRAP_RET_WOPT(ret, cpp_name, params, pre, args, post)     \
    ret Handler::cpp_name(const librpa::Options& opts, LIBRPA_UNPAREN(params))            \
    {                                                                                     \
        pre auto retc = ::LIBRPA_C_NAME(cpp_name)(this->h_, &opts, LIBRPA_UNPAREN(args)); \
        post return retc;                                                                 \
    }

/* Input (set) functions */

// LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
//     set_dimension,
//     (int nspins, int nkpts, int nstates, int nbasis),
//     (nspins, nkpts, nstates, nbasis)
// )
// // Equivalent, write it explicitly to tell the language server that librpa_input.h is needed
void Handler::set_scf_dimension(int nspins, int nkpts, int nstates, int nbasis, int nspinor)
{
    ::librpa_set_scf_dimension(this->h_, nspins, nkpts, nstates, nbasis, nspinor);
}

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wg_ekb_efermi,
    (int nspins, int nkpts, int nstates, const double* wg, const double* ekb, double efermi),
    (nspins, nkpts, nstates, wg, ekb, efermi)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc,
    (int ispin, int ik, int nstates_local, int nbasis_local, const double* wfc_real, const double* wfc_imag),
    (ispin, ik, nstates_local, nbasis_local, wfc_real, wfc_imag)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc_spinor,
    (int ik, int nstates_local, int nbasis_local,
     const double* wfc_up_real, const double* wfc_up_imag,
     const double* wfc_dn_real, const double* wfc_dn_imag),
    (ik, nstates_local, nbasis_local, wfc_up_real, wfc_up_imag, wfc_dn_real, wfc_dn_imag)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc_packed,
    (int ispin, int ik, int nstates_local, int nbasis_local, const std::complex<double> *wfc),
    (ispin, ik, nstates_local, nbasis_local, reinterpret_cast<const double*>(wfc))
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc_spinor_packed,
    (int ik, int nstates_local, int nbasis_local,
     const std::complex<double> *wfc_up, const std::complex<double> *wfc_dn),
    (ik, nstates_local, nbasis_local,
     reinterpret_cast<const double*>(wfc_up), reinterpret_cast<const double*>(wfc_dn))
)

void Handler::set_ao_basis_wfc(const std::vector<size_t> &nbs_wfc,
                               const std::vector<std::vector<int>> &l_shells)
{
    std::vector<int> nshells;
    std::vector<int> flat_l_shells;
    const int *nshells_ptr = nullptr;
    const int *l_shells_ptr = nullptr;
    basis_l_shell_ptrs(nbs_wfc.size(), l_shells, nshells, flat_l_shells,
                       nshells_ptr, l_shells_ptr);
    ::librpa_set_ao_basis_wfc(this->h_, static_cast<int>(nbs_wfc.size()), nbs_wfc.data(),
                              nshells_ptr, l_shells_ptr);
}

void Handler::set_ao_basis_aux(const std::vector<size_t> &nbs_aux,
                               const std::vector<std::vector<int>> &l_shells)
{
    std::vector<int> nshells;
    std::vector<int> flat_l_shells;
    const int *nshells_ptr = nullptr;
    const int *l_shells_ptr = nullptr;
    basis_l_shell_ptrs(nbs_aux.size(), l_shells, nshells, flat_l_shells,
                       nshells_ptr, l_shells_ptr);
    ::librpa_set_ao_basis_aux(this->h_, static_cast<int>(nbs_aux.size()), nbs_aux.data(),
                              nshells_ptr, l_shells_ptr);
}

void Handler::set_ao_basis_aux_shrink(const std::vector<size_t> &nbs_aux_shrink,
                                      const std::vector<std::vector<int>> &l_shells)
{
    std::vector<int> nshells;
    std::vector<int> flat_l_shells;
    const int *nshells_ptr = nullptr;
    const int *l_shells_ptr = nullptr;
    basis_l_shell_ptrs(nbs_aux_shrink.size(), l_shells, nshells, flat_l_shells,
                       nshells_ptr, l_shells_ptr);
    ::librpa_set_ao_basis_aux_shrink(this->h_, static_cast<int>(nbs_aux_shrink.size()),
                                     nbs_aux_shrink.data(), nshells_ptr, l_shells_ptr);
}

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_basis_convention,
    (int bloch_phase, int bloch_ratom, LibrpaAngularOrder order, LibrpaRshCoeff nega_m, LibrpaRshCoeff posi_m),
    (bloch_phase, bloch_ratom, order, nega_m, posi_m)
)

void Handler::set_symmetry_operations(const int n_symops, const int row_conv,
                                      const int* rotmats, const double* trans)
{
    ::librpa_set_symmetry_operations(this->h_, n_symops, row_conv, rotmats, trans);
}

void Handler::set_symmetry_spin_operations(const int n_ops, const int row_conv,
                                           const int* rotmats, const double* trans,
                                           const int* antiunitary, const double* spin_u,
                                           const int spin_action_source, const int grey_group)
{
    ::librpa_set_symmetry_spin_operations(this->h_, n_ops, row_conv, rotmats, trans,
                                          antiunitary, spin_u, spin_action_source,
                                          grey_group);
}

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_latvec_and_G,
    (const double lat_mat[9], const double G_mat[9]),
    (lat_mat, G_mat)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_atoms,
    (const std::vector<int> &types, const std::vector<double> &pos_cart),
    (types.size(), types.data(), pos_cart.data())
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_kgrids_kvec,
    (int nk1, int nk2, int nk3, int nkpts, const double* kvecs, const double* kweights),
    (nk1, nk2, nk3, nkpts, kvecs, kweights)
)

void Handler::set_kgrids_kvec(const int nk1, const int nk2, const int nk3,
                              const std::vector<double> &kvecs,
                              const std::vector<double> &kweights)
{
    const auto n_kpoints = kvecs.size() / 3;
    if (kvecs.size() != 3 * n_kpoints)
    {
        throw LIBRPA_RUNTIME_ERROR("invalid k-point vector buffer size");
    }
    if (!kweights.empty() && kweights.size() != n_kpoints)
    {
        throw LIBRPA_RUNTIME_ERROR("k-point weight count does not match loaded k-point count");
    }
    if (n_kpoints > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        throw LIBRPA_RUNTIME_ERROR("loaded k-point count exceeds C API integer range");
    }

    ::librpa_set_kgrids_kvec(this->h_, nk1, nk2, nk3, static_cast<int>(n_kpoints),
                             kvecs.data(), kweights.empty() ? nullptr : kweights.data());
}

void Handler::set_irreducible_kgrids_kvec(
    const int nk1, const int nk2, const int nk3,
    const std::vector<double> &kvecs_scf, const std::vector<double> &kvecs_full,
    const std::vector<int> &full_to_scf)
{
    if (kvecs_scf.size() % 3 != 0 || kvecs_full.size() != 3 * full_to_scf.size() ||
        kvecs_scf.size() / 3 > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        full_to_scf.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_INVALID_ARGUMENT("Invalid explicit full-grid mapping buffer sizes");
    ::librpa_set_irreducible_kgrids_kvec(
        this->h_, nk1, nk2, nk3, static_cast<int>(kvecs_scf.size() / 3), kvecs_scf.data(),
        static_cast<int>(full_to_scf.size()), kvecs_full.data(), full_to_scf.data());
}

void Handler::set_kq_mapping(const std::vector<int> &map_q_ks)
{
    ::librpa_set_kq_mapping(this->h_, static_cast<int>(map_q_ks.size()), map_q_ks.data());
}

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_lri_coeff,
    (LibrpaParallelRouting routing, int I, int J, int nbasis_i, int nbasis_j, int naux_mu, const int R[3], const double* Cs_in, int shrink_aux),
    (routing, I, J, nbasis_i, nbasis_j, naux_mu, R, Cs_in, shrink_aux)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_bare_coulomb_k_atom_pair,
    (int ik, int I, int J, int naux_mu, int naux_nu, const double* Vq_real_in, const double* Vq_imag_in, double vq_threshold),
    (ik, I, J, naux_mu, naux_nu, Vq_real_in, Vq_imag_in, vq_threshold)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_bare_coulomb_k_atom_pair_packed,
    (int ik, int I, int J, int naux_mu, int naux_nu, const std::complex<double> *Vq, double vq_threshold),
    (ik, I, J, naux_mu, naux_nu, reinterpret_cast<const double*>(Vq), vq_threshold)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_cut_coulomb_k_atom_pair,
    (int ik, int I, int J, int naux_mu, int naux_nu, const double* Vq_real_in, const double* Vq_imag_in, double vq_threshold),
    (ik, I, J, naux_mu, naux_nu, Vq_real_in, Vq_imag_in, vq_threshold)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_cut_coulomb_k_atom_pair_packed,
    (int ik, int I, int J, int naux_mu, int naux_nu, const std::complex<double> *Vq, double vq_threshold),
    (ik, I, J, naux_mu, naux_nu, reinterpret_cast<const double*>(Vq), vq_threshold)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_bare_coulomb_k_2d_block,
    (int ik, int mu_begin, int mu_end, int nu_begin, int nu_end, const double* Vq_real_in, const double* Vq_imag_in),
    (ik, mu_begin, mu_end, nu_begin, nu_end, Vq_real_in, Vq_imag_in)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_bare_coulomb_k_2d_block_packed,
    (int ik, int mu_begin, int mu_end, int nu_begin, int nu_end, const std::complex<double> *Vq),
    (ik, mu_begin, mu_end, nu_begin, nu_end, reinterpret_cast<const double*>(Vq))
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_cut_coulomb_k_2d_block,
    (int ik, int mu_begin, int mu_end, int nu_begin, int nu_end, const double* Vq_real_in, const double* Vq_imag_in),
    (ik, mu_begin, mu_end, nu_begin, nu_end, Vq_real_in, Vq_imag_in)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_aux_cut_coulomb_k_2d_block_packed,
    (int ik, int mu_begin, int mu_end, int nu_begin, int nu_end, const std::complex<double> *Vq),
    (ik, mu_begin, mu_end, nu_begin, nu_end, reinterpret_cast<const double*>(Vq))
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_dielect_func_imagfreq,
    (const std::vector<double> &omegas_imag, const std::vector<double> &dielect_func),
    (omegas_imag.size(), omegas_imag.data(), dielect_func.data())
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_velocity_matrix,
    (int n_spins, int n_kpts, int n_states, const double *velocity_real, const double *velocity_imag),
    (n_spins, n_kpts, n_states, velocity_real, velocity_imag)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_velocity_matrix_packed,
    (int n_spins, int n_kpts, int n_states, const std::complex<double> *velocity),
    (n_spins, n_kpts, n_states, reinterpret_cast<const double*>(velocity))
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_band_kvec,
    (int n_kpts_band, const double* kfrac_band),
    (n_kpts_band, kfrac_band)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_band_occ_eigval,
    (int n_spins, int n_kpts_band, int n_states, const double* occ, const double* eig),
    (n_spins, n_kpts_band, n_states, occ, eig)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc_band,
    (int ispin, int ik_band, int nstates_local, int nbasis_local, const double* wfc_real, const double* wfc_imag),
    (ispin, ik_band, nstates_local, nbasis_local, wfc_real, wfc_imag)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc_band_spinor,
    (int ik, int nstates_local, int nbasis_local,
     const double* wfc_up_real, const double* wfc_up_imag,
     const double* wfc_dn_real, const double* wfc_dn_imag),
    (ik, nstates_local, nbasis_local, wfc_up_real, wfc_up_imag, wfc_dn_real, wfc_dn_imag)
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc_band_packed,
    (int ispin, int ik_band, int nstates_local, int nbasis_local, const std::complex<double> *wfc),
    (ispin, ik_band, nstates_local, nbasis_local, reinterpret_cast<const double*>(wfc))
)

LIBRPA_CPP_H_METHOD_DEF_WRAP_VOID(
    set_wfc_band_spinor_packed,
    (int ik_band, int nstates_local, int nbasis_local,
     const std::complex<double> *wfc_up, const std::complex<double> *wfc_dn),
    (ik_band, nstates_local, nbasis_local,
     reinterpret_cast<const double*>(wfc_up), reinterpret_cast<const double*>(wfc_dn))
)

void Handler::reset_band_data()
{
    ::librpa_reset_band_data(this->h_);
}

void Handler::prepare_native_velocity(const Options& opts,
                                      const LibrpaNativeVelocityInput* root_input,
                                      bool prepare_optical)
{
    ::librpa_prepare_native_velocity(h_, &opts, root_input, prepare_optical ? 1 : 0);
}

OpticalResult Handler::get_optical_result() const
{
    return OpticalResult(::librpa_get_optical_result(h_), ::librpa_delete_optical_result);
}

/* Compute (build/get) functions */
void Handler::get_imaginary_frequency_grids(const Options& opts, std::vector<double> &omegas, std::vector<double> &weights)
{
    omegas.resize(opts.nfreq);
    weights.resize(opts.nfreq);
    ::librpa_get_imaginary_frequency_grids(this->h_, &opts, omegas.data(), weights.data());
}

LIBRPA_CPP_H_METHOD_DEF_WRAP_RET_WOPT(
    double, get_rpa_correlation_energy,
    (std::vector<std::complex<double>> &rpa_corr_ibzk_contrib),
    using namespace std::complex_literals; const int nq = rpa_corr_ibzk_contrib.size(); std::vector<double> re(nq); std::vector<double> im(nq);,
    (nq, re.data(), im.data()),
    {for (int iq = 0; iq < nq; iq++) { rpa_corr_ibzk_contrib[iq] = re[iq] + im[iq] * 1.0i;}}
)

void Handler::build_exx(const Options &opts)
{
    ::librpa_build_exx(this->h_, &opts);
}

std::vector<std::complex<double>> Handler::compute_hartree_ao(
    const Options& opts, const std::vector<int>& iks_this, const int n_aos)
{
    const auto dataset = librpa_int::api::get_dataset_instance(h_);
    std::vector<double> real, imag;
    std::vector<std::complex<double>> result;
    librpa_int::collective_all_stage(dataset->comm_h, "Hartree C++ output storage", [&]() {
        if (n_aos <= 0 || n_aos != static_cast<int>(dataset->basis_wfc.nb_total) ||
            iks_this.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Invalid Hartree AO dimensions or k-point count");
        const auto n = static_cast<std::size_t>(n_aos);
        if (n > std::numeric_limits<std::size_t>::max() / n ||
            (!iks_this.empty() && n * n > std::numeric_limits<std::size_t>::max() / iks_this.size()))
            throw std::invalid_argument("Hartree AO output size overflows");
        const std::size_t size = iks_this.size() * n * n;
        real.resize(size);
        imag.resize(size);
        result.resize(size);
    });
    ::librpa_compute_hartree_ao(h_, &opts, static_cast<int>(iks_this.size()), iks_this.data(),
                               n_aos, real.data(), imag.data());
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = {real[i], imag[i]};
    return result;
}

std::vector<double> Handler::get_exx_pot_kgrid(const Options& opts, const int n_spins,
                                               const std::vector<int> &iks_this, int i_state_low,
                                               int i_state_high)
{
    const int n_kpts_this = iks_this.size();
    const int n_states_calc = i_state_high - i_state_low;
    std::vector<double> vexx(n_spins * n_kpts_this * n_states_calc);
    ::librpa_get_exx_pot_kgrid(this->h_, &opts, n_spins, n_kpts_this, iks_this.data(),
                                i_state_low, i_state_high, vexx.data());
    return vexx;
}

std::vector<double> Handler::get_exx_pot_band_k(const Options& opts, const int n_spins,
                                               const std::vector<int> &iks_band_this, int i_state_low,
                                               int i_state_high)
{
    const int n_kpts_band_this = iks_band_this.size();
    const int n_states_calc = i_state_high - i_state_low;
    std::vector<double> vexx_band(n_spins * n_kpts_band_this * n_states_calc);
    ::librpa_get_exx_pot_band_k(this->h_, &opts, n_spins, n_kpts_band_this, iks_band_this.data(),
                                i_state_low, i_state_high, vexx_band.data());
    return vexx_band;
}

void Handler::build_g0w0_sigma(const Options &opts)
{
    ::librpa_build_g0w0_sigma(this->h_, &opts);
}

std::vector<std::complex<double>> Handler::get_g0w0_sigc_kgrid(
    const Options& opts, const int n_spins, const std::vector<int>& iks_local, int i_state_low,
    int i_state_high, const std::vector<double>& vxc, const std::vector<double>& vexx)
{
    return get_g0w0_qpe_kgrid(opts, n_spins, iks_local, i_state_low, i_state_high,
                              vxc, vexx).sigc;
}

G0W0QpeResult Handler::get_g0w0_qpe_kgrid(
    const Options& opts, const int n_spins, const std::vector<int>& iks_local, int i_state_low,
    int i_state_high, const std::vector<double>& vxc, const std::vector<double>& vexx)
{
    const int n_kpoints_local = iks_local.size();
    const int n_states_calc = i_state_high - i_state_low;
    const size_t n = n_spins * n_kpoints_local * n_states_calc;
    std::vector<double> sigc_re(n);
    std::vector<double> sigc_im(n);
    std::vector<double> eqp(n);
    ::librpa_get_g0w0_qpe_kgrid(this->h_, &opts, n_spins, n_kpoints_local, iks_local.data(),
                                i_state_low, i_state_high, vxc.data(), vexx.data(),
                                sigc_re.data(), sigc_im.data(), eqp.data());
    std::vector<std::complex<double>> sigc(n);
    for (size_t i = 0; i < n; i++)
        sigc[i] = std::complex<double>{sigc_re[i], sigc_im[i]};
    return G0W0QpeResult{std::move(sigc), std::move(eqp)};
}

std::vector<double> Handler::get_g0w0_spectral_function_kgrid(
    const Options& opts, const int n_spins, const std::vector<int>& iks_local,
    int i_state_low, int i_state_high, const std::vector<double>& omegas,
    const std::vector<double>& vxc, const std::vector<double>& vexx)
{
    const int n_kpoints_local = iks_local.size();
    const int n_states_calc = i_state_high > i_state_low ? i_state_high - i_state_low : 0;
    const int n_omegas = omegas.size();
    const size_t n = static_cast<size_t>(n_spins) * n_kpoints_local * n_states_calc
                     * n_omegas;
    std::vector<double> spectral_function(n);
    ::librpa_get_g0w0_spectral_function_kgrid(
        this->h_, &opts, n_spins, n_kpoints_local, iks_local.data(), i_state_low,
        i_state_high, n_omegas, omegas.data(), vxc.data(), vexx.data(),
        spectral_function.data(), nullptr);
    return spectral_function;
}

G0W0SpectralFunctionResult Handler::get_g0w0_spectral_function_with_sigc_kgrid(
    const Options& opts, const int n_spins, const std::vector<int>& iks_local,
    int i_state_low, int i_state_high, const std::vector<double>& omegas,
    const std::vector<double>& vxc, const std::vector<double>& vexx)
{
    const int n_kpoints_local = iks_local.size();
    const int n_states_calc = i_state_high > i_state_low ? i_state_high - i_state_low : 0;
    const int n_omegas = omegas.size();
    const size_t n = static_cast<size_t>(n_spins) * n_kpoints_local * n_states_calc
                     * n_omegas;
    std::vector<std::complex<double>> sigc(n);
    std::vector<double> spectral_function(n);
    ::librpa_get_g0w0_spectral_function_kgrid(
        this->h_, &opts, n_spins, n_kpoints_local, iks_local.data(), i_state_low,
        i_state_high, n_omegas, omegas.data(), vxc.data(), vexx.data(),
        spectral_function.data(), reinterpret_cast<double *>(sigc.data()));
    return G0W0SpectralFunctionResult{std::move(sigc), std::move(spectral_function)};
}

std::vector<std::complex<double>> Handler::get_g0w0_sigc_band_k(
    const Options& opts, const int n_spins, const std::vector<int>& iks_band_this, int i_state_low,
    int i_state_high, const std::vector<double>& vxc_band, const std::vector<double>& vexx_band)
{
    return get_g0w0_qpe_band_k(opts, n_spins, iks_band_this, i_state_low, i_state_high,
                               vxc_band, vexx_band).sigc;
}

G0W0QpeResult Handler::get_g0w0_qpe_band_k(
    const Options& opts, const int n_spins, const std::vector<int>& iks_band_this, int i_state_low,
    int i_state_high, const std::vector<double>& vxc_band, const std::vector<double>& vexx_band)
{
    const int n_kpts_band_this = iks_band_this.size();
    const int n_states_calc = i_state_high - i_state_low;
    const size_t n = n_spins * n_kpts_band_this * n_states_calc;
    std::vector<double> sigc_band_re(n);
    std::vector<double> sigc_band_im(n);
    std::vector<double> eqp_band(n);
    ::librpa_get_g0w0_qpe_band_k(this->h_, &opts, n_spins, n_kpts_band_this,
                                 iks_band_this.data(), i_state_low, i_state_high,
                                 vxc_band.data(), vexx_band.data(), sigc_band_re.data(),
                                 sigc_band_im.data(), eqp_band.data());
    std::vector<std::complex<double>> sigc_band(n);
    for (size_t i = 0; i < n; i++)
        sigc_band[i] = std::complex<double>{sigc_band_re[i], sigc_band_im[i]};
    return G0W0QpeResult{std::move(sigc_band), std::move(eqp_band)};
}

std::vector<double> Handler::get_g0w0_spectral_function_band_k(
    const Options& opts, const int n_spins, const std::vector<int>& iks_band_this,
    int i_state_low, int i_state_high, const std::vector<double>& omegas,
    const std::vector<double>& vxc_band, const std::vector<double>& vexx_band)
{
    const int n_kpts_band_this = iks_band_this.size();
    const int n_states_calc = i_state_high > i_state_low ? i_state_high - i_state_low : 0;
    const int n_omegas = omegas.size();
    const size_t n = static_cast<size_t>(n_spins) * n_kpts_band_this * n_states_calc
                     * n_omegas;
    std::vector<double> spectral_function_band(n);
    ::librpa_get_g0w0_spectral_function_band_k(
        this->h_, &opts, n_spins, n_kpts_band_this, iks_band_this.data(), i_state_low,
        i_state_high, n_omegas, omegas.data(), vxc_band.data(), vexx_band.data(),
        spectral_function_band.data(), nullptr);
    return spectral_function_band;
}

G0W0SpectralFunctionResult Handler::get_g0w0_spectral_function_with_sigc_band_k(
    const Options& opts, const int n_spins, const std::vector<int>& iks_band_this,
    int i_state_low, int i_state_high, const std::vector<double>& omegas,
    const std::vector<double>& vxc_band, const std::vector<double>& vexx_band)
{
    const int n_kpts_band_this = iks_band_this.size();
    const int n_states_calc = i_state_high > i_state_low ? i_state_high - i_state_low : 0;
    const int n_omegas = omegas.size();
    const size_t n = static_cast<size_t>(n_spins) * n_kpts_band_this * n_states_calc
                     * n_omegas;
    std::vector<std::complex<double>> sigc_band(n);
    std::vector<double> spectral_function_band(n);
    ::librpa_get_g0w0_spectral_function_band_k(
        this->h_, &opts, n_spins, n_kpts_band_this, iks_band_this.data(), i_state_low,
        i_state_high, n_omegas, omegas.data(), vxc_band.data(), vexx_band.data(),
        spectral_function_band.data(), reinterpret_cast<double *>(sigc_band.data()));
    return G0W0SpectralFunctionResult{std::move(sigc_band),
                                      std::move(spectral_function_band)};
}

}
