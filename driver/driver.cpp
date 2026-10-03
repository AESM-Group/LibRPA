#include "driver.h"

#include <ios>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <sstream>
#include <string>
#include "librpa_enums.h"
#include "../src/io/global_io.h"

namespace driver
{

DriverParams::DriverParams():
    task("unset"),
    constants_choice("internal"),
    input_preset("fhi-aims"),
    input_dir("./"),
    output_level(LIBRPA_VERBOSE_INFO),
    use_spinor_wfc(false),
    prefix_lri_coeff("Cs_data"),
    prefix_lri_coeff_shrink("Cs_shrinked_data"),
    prefix_shrink_sinvS("shrink_sinvS_"),
    prefix_coul_full("coulomb_mat"),
    prefix_coul_cut("coulomb_cut"),
    prefix_eigvecs_scf("KS_eigenvector"),
    fn_stru("stru_out"),
    fn_bz_sampling("bz_sampling_out"),
    fn_basis("basis_out"),
    fn_basis_wfc("basis_wfc_out"),
    fn_basis_aux("basis_aux_out"),
    fn_basis_aux_shrink("basis_aux_shrink_out"),
    fn_eigocc_scf("band_out"),
    fn_dielfunc("dielecfunc_out"),
    fn_vxc_scf("vxc_out"),
    prefix_velocity("mommat_ks_kpt_"),
    fn_band_kpath_info("band_kpath_info"),
    version_coul_reader(-1),
    version_lri_reader(-1),
    cs_threshold(1e-6),
    output_energy_qp(false),
    i_state_low(0),
    i_state_high(default_i_state_high),
    output_hamgnn(false),
    use_pyatb(false),
    output_gw_spec_func(false),
    sf_omega_start(0.0),
    sf_omega_end(1.0),
    sf_omega_step(0.1),
    sf_state_start(-1),
    sf_state_end(-1),
    prefix_vxc_scf(""),
    prefix_vxc_band(""),
    qsgw_vxc_basis("state"),
    qsgw_mixer("none"),
    qsgw_mixing_beta(0.2),
    qsgw_min_iter(1),
    qsgw_max_iter(10),
    qsgw_band0_unoccupied_keep(10),
    qsgw_band0_cut_mode(0),
    qsgw_band0_cut_shift_ha(20.0),
    qsgw_write_iteration_matrices(false),
    qsgw_convergence_tolerance_ev(1.0e-4)
{
}

void DriverParams::apply_input_preset()
{
    if (input_preset == "aims")
        input_preset = "fhi-aims";

    if (input_preset != "fhi-aims" && input_preset != "abacus" &&
        input_preset != "abacus-legacy")
        throw std::invalid_argument("Unsupported input_preset: " + input_preset
                                    + "; expected fhi-aims (alias aims), abacus, or abacus-legacy");

    // The FHI-aims preset preserves the historical LibRPA driver defaults.
    fn_stru = "stru_out";
    fn_bz_sampling = "bz_sampling_out";
    fn_basis = "basis_out";
    fn_basis_wfc = "basis_wfc_out";
    fn_basis_aux = "basis_aux_out";
    fn_basis_aux_shrink = "basis_aux_shrink_out";
    fn_eigocc_scf = "band_out";
    fn_dielfunc = "dielecfunc_out";
    fn_vxc_scf = "vxc_out";
    prefix_velocity = "mommat_ks_kpt_";
    fn_band_kpath_info = "band_kpath_info";

    if (input_preset == "abacus")
    {
        fn_stru = "stru_out.txt";
        fn_eigocc_scf = "band_out.txt";
        fn_vxc_scf = "vxc_out.txt";
    }
    if (input_preset == "abacus" || input_preset == "abacus-legacy")
        prefix_velocity = "velocity_matrix";
}

std::string DriverParams::format()
{
    std::stringstream ss;

#define normal_pair(name) {#name, name}
    const std::vector<std::pair<std::string, std::string>> str_params
        {
            normal_pair(task),
            normal_pair(constants_choice),
            normal_pair(input_preset),
            normal_pair(input_dir),
            normal_pair(prefix_lri_coeff),
            normal_pair(prefix_lri_coeff_shrink),
            normal_pair(prefix_shrink_sinvS),
            normal_pair(prefix_coul_full),
            normal_pair(prefix_coul_cut),
            normal_pair(prefix_eigvecs_scf),
            normal_pair(fn_stru),
            normal_pair(fn_bz_sampling),
            normal_pair(fn_basis),
            normal_pair(fn_basis_wfc),
            normal_pair(fn_basis_aux),
            normal_pair(fn_basis_aux_shrink),
            normal_pair(fn_eigocc_scf),
            normal_pair(fn_dielfunc),
            normal_pair(fn_vxc_scf),
            normal_pair(prefix_velocity),
            normal_pair(fn_band_kpath_info),
        };
    for (const auto &[k, v]: str_params)
        ss << k << " = " << v << std::endl;
    ss << "output_level = " << get_verbose_string(output_level) << std::endl;

    const std::vector<std::pair<std::string, bool>> bool_params
        {
            normal_pair(use_spinor_wfc),
            normal_pair(output_energy_qp),
            normal_pair(output_gw_spec_func),
            normal_pair(output_hamgnn),
            normal_pair(use_pyatb),
        };
    for (const auto &[k, v]: bool_params)
        ss << k << " = " << std::boolalpha << v << std::endl;
#undef normal_pair

    ss << "version_coul_reader = " << version_coul_reader << std::endl;
    ss << "version_lri_reader = " << version_lri_reader << std::endl;
    ss << "cs_R_threshold = " << cs_threshold << std::endl;
    ss << "i_state_low = " << i_state_low << std::endl;
    ss << "i_state_high = " << i_state_high << std::endl;
    std::string normalized_task = task;
    std::transform(normalized_task.begin(), normalized_task.end(),
                   normalized_task.begin(), [](unsigned char c) { return std::tolower(c); });
    if (normalized_task == "qsgw" || normalized_task == "qsgw_band")
    {
        const bool aims_input = constants_choice == "aims";
        ss << "prefix_vxc_scf = " << (prefix_vxc_scf.empty()
            ? (aims_input ? "xc_matr" : "vxc") : prefix_vxc_scf) << std::endl;
        ss << "prefix_vxc_band = " << (prefix_vxc_band.empty()
            ? (aims_input ? "band_vxc_mat" : "band_vxc") : prefix_vxc_band) << std::endl;
        ss << "qsgw_vxc_basis = " << qsgw_vxc_basis << std::endl;
        ss << "qsgw_mixer = " << qsgw_mixer << std::endl;
        ss << "qsgw_mixing_beta = " << qsgw_mixing_beta << std::endl;
        ss << "qsgw_min_iter = " << qsgw_min_iter << std::endl;
        ss << "qsgw_max_iter = " << qsgw_max_iter << std::endl;
        ss << "qsgw_band0_unoccupied_keep = "
           << qsgw_band0_unoccupied_keep << std::endl;
        ss << "qsgw_band0_cut_mode = " << qsgw_band0_cut_mode
           << std::endl;
        ss << "qsgw_band0_cut_shift_ha = "
           << qsgw_band0_cut_shift_ha << std::endl;
        ss << "qsgw_convergence_tolerance_ev = "
           << qsgw_convergence_tolerance_ev << std::endl;
        ss << "qsgw_write_iteration_matrices = " << std::boolalpha
           << qsgw_write_iteration_matrices << std::endl;
    }
    if (output_gw_spec_func)
    {
        ss << "sf_omega_start = " << sf_omega_start << std::endl;
        ss << "sf_omega_end   = " << sf_omega_end << std::endl;
        ss << "sf_omega_step  = " << sf_omega_step << std::endl;
        ss << "sf_state_start = " << sf_state_start << std::endl;
        ss << "sf_state_end   = " << sf_state_end << std::endl;
    }
    return ss.str();
}

DriverParams driver_params;

const std::string input_filename = "librpa.in";

// Preserve the driver metadata names while sharing their storage with readers.
librpa::reader::ReaderState reader_state;
std::vector<int> &atom_types = reader_state.atom_types;
size_t &n_atoms = reader_state.n_atoms;
int &n_spins = reader_state.n_spins;
int &n_kpoints = reader_state.n_kpoints;
int &n_ibz_kpoints = reader_state.n_ibz_kpoints;
int &n_kpoints_band = reader_state.n_kpoints_band;
int &n_states = reader_state.n_states;
int &n_basis_wfc = reader_state.n_basis_wfc;
int &n_basis_ao = reader_state.n_basis_ao;
int &n_spinor = reader_state.n_spinor;
std::vector<size_t> &nbs_wfc = reader_state.nbs_wfc;
std::vector<size_t> &nbs_aux = reader_state.nbs_aux;
std::vector<size_t> &nbs_aux_shrink = reader_state.nbs_aux_shrink;
std::vector<int> &iks_eigvec_this = reader_state.iks_eigvec_this;
std::vector<int> &iks_band_eigvec_this = reader_state.iks_band_eigvec_this;
std::vector<librpa_int::Vector3_Order<double>> &ibz_kpoints = reader_state.ibz_kpoints;
std::vector<librpa_int::Vector3_Order<double>> &kfrac_band = reader_state.kfrac_band;
bool &is_basis_convention_read = reader_state.is_basis_convention_read;
std::string &basis_convention_label = reader_state.basis_convention_label;

// Used to read Coulomb matrix data.
// Should be consistent with the internal `atpairs_local` of the Dataset object
std::vector<std::pair<size_t, size_t>> local_atpair;

librpa::Handler h;

librpa::Options opts;

#define normal_pair(name) {#name, opts.name}
#define bool_pair(name) {#name, get_bool(opts.name)}

std::string format_runtime_options(const librpa::Options &opts) noexcept
{
    using std::endl;
    using std::boolalpha;

    std::stringstream ss;
    const std::vector<std::pair<std::string, double>> double_params
        {
            {"gf_R_threshold", opts.gf_threshold},
            normal_pair(vq_threshold),
            normal_pair(minimax_emin),
            normal_pair(minimax_emax),
            normal_pair(minimax_regulation),
            normal_pair(sqrt_coulomb_threshold),
            normal_pair(libri_chi0_threshold_C),
            normal_pair(libri_chi0_threshold_G),
            normal_pair(libri_exx_threshold_C),
            normal_pair(libri_exx_threshold_D),
            normal_pair(libri_exx_threshold_V),
            normal_pair(libri_g0w0_threshold_C),
            normal_pair(libri_g0w0_threshold_G),
            normal_pair(libri_g0w0_threshold_Wc),
            normal_pair(qpe_solver_thres),
            normal_pair(qpe_solver_damp_factor),
            normal_pair(sf_gf_omega_shift),
            normal_pair(sf_sigc_omega_shift),
        };

    const std::vector<std::pair<std::string, int>> int_params
        {
            normal_pair(nfreq),
            normal_pair(n_bands_chi0),
            normal_pair(n_bands_sigc),
            normal_pair(n_bands_exclude_chi0),
            normal_pair(n_bands_exclude_sigc),
            normal_pair(n_params_anacon),
            normal_pair(n_params_anacon_resample),
            normal_pair(anacon_nfreq),
            normal_pair(option_qpe_solver),
            normal_pair(option_dielect_func),
            normal_pair(option_bvk_remap),
            normal_pair(rpa_headwing_body_start),
            normal_pair(qpe_solver_n_iter_max),
            normal_pair(libri_chi0_collect_s0_chunk),
            normal_pair(istate_ref_hedin_shift),
            normal_pair(ifreq_output_wc_start),
            normal_pair(ifreq_output_wc_end),
            normal_pair(istate_output_mat_start),
            normal_pair(istate_output_mat_end),
        };

    const std::vector<std::pair<std::string, std::string>> str_params
        {
            {"output_dir", opts.output_dir},
            {"restart_from_dir", opts.restart_from_dir[0] ? opts.restart_from_dir : "<output_dir>"},
            {"tfgrids_type", get_tfgrid_string(opts.tfgrids_type)},
            {"anacon_tfgrids_type", get_tfgrid_string(opts.anacon_tfgrids_type)},
            {"parallel_routing", get_routing_string(opts.parallel_routing)},
            {"rpa_headwing_mode", opts.rpa_headwing_mode},
        };

    const std::vector<std::pair<std::string, bool>> bool_params
        {
            {"debug", librpa_int::global::should_output(LIBRPA_VERBOSE_DEBUG)},
            bool_pair(replace_w_head),
            bool_pair(use_fullcoul_eps),
            bool_pair(use_fullcoul_exx),
            bool_pair(use_fullcoul_wc),
            bool_pair(use_symmetry_exx),
            bool_pair(use_symmetry_gw),
            bool_pair(use_symmetry_rpa),
            bool_pair(output_abacus_gw_gf),
            bool_pair(use_shrink_abfs),
            bool_pair(use_shrink_chi),
            bool_pair(use_scalapack_ecrpa),
            bool_pair(use_scalapack_gw_wc),
            bool_pair(use_cholesky_gw_wc),
            bool_pair(use_gpu_replace_scalapack),
            bool_pair(use_elpa_sqrt_coulomb),
            bool_pair(use_kpara_scf_eigvec),
            bool_pair(read_sigc_mat_rf),
            bool_pair(output_gw_sigc_ks_kf),
            bool_pair(output_gw_sigc_ks_mat_kf),
            bool_pair(output_exx_ks_mat_k),
            bool_pair(output_exx_mat_k),
            bool_pair(output_gw_sigc_mat_kf),
            bool_pair(output_gw_sigc_mat_rf),
            bool_pair(output_gw_sigc_mat_rt),
            bool_pair(output_wc_rf),
            bool_pair(output_wc_rf_atom_pair),
            bool_pair(output_2d_finite_q_diagnostics),
            bool_pair(use_qpe_adaptive_damp),
            bool_pair(use_qpe_legacy_update),
            bool_pair(override_qpe_solver_nan),
            bool_pair(use_hedin_shift),
        };

    for (const auto &[k, v] : str_params) ss << k << " = " << v << endl;

    for (const auto &[k, v] : int_params) ss << k << " = " << v << endl;

    ss << "libri_chi0_collect_max_bytes = "
       << opts.libri_chi0_collect_max_bytes << endl;

    for (const auto &[k, v] : double_params) ss << k << " = " << v << endl;

    for (const auto &[k, v] : bool_params) ss << k << " = " << boolalpha << v << endl;

    return ss.str();
}

#undef normal_pair
#undef bool_pair

LibrpaTimeFreqGrid get_tfgrid_type(const std::string& grid_str)
{
    if (grid_str == "unset")
        return LIBRPA_TFGRID_UNSET;
    if (grid_str == "GL")
        return LIBRPA_TFGRID_GAUSS_LEGENDRE;
    if (grid_str == "split-GL")
        return LIBRPA_TFGRID_SPLIT_GAUSS_LEGENDRE;
    if (grid_str == "GC-I")
        return LIBRPA_TFGRID_GAUSS_CHEBYSHEV_I;
    if (grid_str == "GL-II")
        return LIBRPA_TFGRID_GAUSS_CHEBYSHEV_II;
    if (grid_str == "minimax")
        return LIBRPA_TFGRID_MINIMAX;
    if (grid_str == "evenspaced")
        return LIBRPA_TFGRID_EVEN_SPACED;
    if (grid_str == "evenspaced_tf")
        return LIBRPA_TFGRID_EVEN_SPACED_TF;
    throw std::runtime_error("Unknown time-frequency grid string: " + grid_str);
}

std::string get_tfgrid_string(const LibrpaTimeFreqGrid& grid_type) noexcept
{
    if (grid_type == LIBRPA_TFGRID_GAUSS_LEGENDRE)
        return "GL";
    if (grid_type == LIBRPA_TFGRID_SPLIT_GAUSS_LEGENDRE)
        return "split-GL";
    if (grid_type == LIBRPA_TFGRID_GAUSS_CHEBYSHEV_I)
        return "GC-I";
    if (grid_type == LIBRPA_TFGRID_GAUSS_CHEBYSHEV_II)
        return "GL-II";
    if (grid_type == LIBRPA_TFGRID_MINIMAX)
        return "minimax";
    if (grid_type == LIBRPA_TFGRID_EVEN_SPACED)
        return "evenspaced";
    if (grid_type == LIBRPA_TFGRID_EVEN_SPACED_TF)
        return "evenspaced_tf";
    return "unset";
}

LibrpaParallelRouting get_parallel_routing(const std::string& routing_str_low)
{
    if (routing_str_low == "auto") return LIBRPA_ROUTING_AUTO;
    if (routing_str_low == "rtau") return LIBRPA_ROUTING_RTAU;
    if (routing_str_low == "atompair") return LIBRPA_ROUTING_ATOMPAIR;
    if (routing_str_low == "libri") return LIBRPA_ROUTING_LIBRI;
    throw std::runtime_error("Unknown parallel routing string: " + routing_str_low);
}

std::string get_routing_string(LibrpaParallelRouting routing)
{
    if (routing == LIBRPA_ROUTING_AUTO) return "auto";
    if (routing == LIBRPA_ROUTING_ATOMPAIR) return "atompair";
    if (routing == LIBRPA_ROUTING_RTAU) return "rtau";
    if (routing == LIBRPA_ROUTING_LIBRI) return "libri";
    return "unset";
}

std::string get_verbose_string(LibrpaVerbose verbose)
{
    if (verbose == LIBRPA_VERBOSE_DEBUG) return "debug";
    if (verbose == LIBRPA_VERBOSE_WARN) return "warn";
    if (verbose == LIBRPA_VERBOSE_INFO) return "info";
    if (verbose == LIBRPA_VERBOSE_CRITICAL) return "critical";
    return "silent";
}

LibrpaVerbose get_verbose(const std::string& verbose_str_low)
{
    if (verbose_str_low == "debug") return LIBRPA_VERBOSE_DEBUG;
    if (verbose_str_low == "warn") return LIBRPA_VERBOSE_WARN;
    if (verbose_str_low == "warning") return LIBRPA_VERBOSE_WARN;
    if (verbose_str_low == "info") return LIBRPA_VERBOSE_INFO;
    if (verbose_str_low == "critical") return LIBRPA_VERBOSE_CRITICAL;
    if (verbose_str_low == "silent") return LIBRPA_VERBOSE_SILENT;
    throw std::runtime_error("Unknown verbose level string: " + verbose_str_low);
}

}
