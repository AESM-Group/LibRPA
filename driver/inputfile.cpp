#include "inputfile.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <regex>
#include <sstream>
#include <limits>
#include <algorithm>
#include <cctype>
#include <cmath>

#include "driver.h"

#include <librpa_enums.h>

#include "../src/io/fs.h"
#include "../src/utils/constants.h"
#include "../src/utils/dev_options.h"

InputParser InputFile::load(const std::string &fn, bool error_if_fail_open)
{
    if (error_if_fail_open)
    {
        librpa_int::require_readable_file(fn);
    }
    std::ifstream t(fn);
    std::string params;
    if (t.is_open())
    {
        filename = fn;
        std::stringstream buffer;
        buffer << t.rdbuf();
        orig_content = buffer.str();
        // trim comment
        std::regex r(InputParser::COMMENTS_IDEN + "(.*?)\n");
        // extra \n to ensure comment in the last line is trimed
        params = std::regex_replace(orig_content + "\n", r, "\n");
    }
    else
    {
        const std::string errmsg = "Error! fail to open file " + fn;
        std::cout << errmsg << std::endl;
        if (error_if_fail_open) throw std::runtime_error(errmsg);
        std::cout << "Default parameters will be used" << std::endl;
    }
    return InputParser(params);
}

static std::string check_dirpath(const std::string &dirpath)
{
    if (dirpath.find(":") != std::string::npos)
    {
        throw std::runtime_error("input_dir contains invalid character (:) for POSIX path");
    }

    if (dirpath.back() != '/')
    {
        return dirpath + '/';
    }

    return std::string(dirpath);
}

#define _parse_int(obj, name) parser.parse_int(#name, obj.name, flag)
#define _parse_double(obj, name) parser.parse_double(#name, obj.name, flag)
#define _parse_bool(obj, name) parser.parse_bool(#name, obj.name, flag)
#define _parse_string(obj, name) parser.parse_string(#name, obj.name, flag);
#define _parse_switch(obj, name) parser.parse_bool(#name, btmp, flag); if (flag == 0) obj.name = get_switch(btmp);
#define _parse_string_post(obj, name, post) parser.parse_string(#name, stmp, flag); if (flag == 0) obj.name = post(stmp);

static bool get_last_assigned_value(const InputParser &parser,
                                    const std::string &name,
                                    std::string &value)
{
    const std::string space = "[ \\r\\f\\t]*";
    const std::regex assignment(
        "(^|[\\r\\n])" + space + name + space + "=" +
            "([^\\r\\n#!]*)",
        std::regex_constants::ECMAScript | std::regex_constants::icase);
    const std::string &params = parser.get_params();
    bool found = false;
    for (std::sregex_iterator it(params.begin(), params.end(), assignment), end;
         it != end; ++it)
    {
        value = (*it)[2].str();
        found = true;
    }
    if (!found)
        return false;

    const auto first = std::find_if_not(
        value.begin(), value.end(),
        [](const unsigned char ch) { return std::isspace(ch); });
    const auto last = std::find_if_not(
        value.rbegin(), value.rend(),
        [](const unsigned char ch) { return std::isspace(ch); }).base();
    value = first < last ? std::string(first, last) : std::string{};
    return true;
}

static void parse_qsgw_string(const InputParser &parser,
                              const std::string &name,
                              std::string &value)
{
    std::string token;
    if (!get_last_assigned_value(parser, name, token))
        return;
    if (token.empty())
        throw std::runtime_error(name + " must not be empty");
    value = token;
}

static void parse_qsgw_int(const InputParser &parser,
                           const std::string &name,
                           int &value)
{
    std::string token;
    if (!get_last_assigned_value(parser, name, token))
        return;
    try
    {
        std::size_t consumed = 0;
        const int parsed = std::stoi(token, &consumed);
        if (consumed != token.size())
            throw std::invalid_argument("trailing input");
        value = parsed;
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(name + " must be a valid integer");
    }
}

static void parse_qsgw_double(const InputParser &parser,
                              const std::string &name,
                              double &value)
{
    std::string token;
    if (!get_last_assigned_value(parser, name, token))
        return;
    std::replace(token.begin(), token.end(), 'd', 'e');
    std::replace(token.begin(), token.end(), 'D', 'E');
    try
    {
        std::size_t consumed = 0;
        const double parsed = std::stod(token, &consumed);
        if (consumed != token.size())
            throw std::invalid_argument("trailing input");
        value = parsed;
    }
    catch (const std::exception &)
    {
        throw std::runtime_error(name +
                                 " must be a valid floating-point value");
    }
}

static void parse_qsgw_bool(const InputParser &parser,
                            const std::string &name,
                            bool &value)
{
    std::string token;
    if (!get_last_assigned_value(parser, name, token))
        return;
    std::transform(token.begin(), token.end(), token.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    if (token == "true" || token == "t" || token == ".t.")
        value = true;
    else if (token == "false" || token == "f" || token == ".f.")
        value = false;
    else
        throw std::runtime_error(name + " must be true or false");
}

static void validate_input_parameters()
{
    const auto &params = driver::driver_params;
    std::string task = params.task;
    std::transform(task.begin(), task.end(), task.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    if (params.output_gw_spec_func)
    {
        if (params.sf_omega_step <= 0.0)
            throw std::runtime_error("sf_omega_step must be positive");
        if (params.sf_omega_end < params.sf_omega_start)
            throw std::runtime_error("sf_omega_end must be no smaller than sf_omega_start");
        if (params.sf_state_start >= 0 && params.sf_state_end >= 0
            && params.sf_state_end <= params.sf_state_start)
            throw std::runtime_error("sf_state_end must be greater than sf_state_start");
    }
    if (task == "qsgw" || task == "qsgw_band")
    {
        if (driver::opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON)
            throw std::runtime_error(
                "QSGW fixed-basis iteration requires replicated SCF wavefunctions; set use_kpara_scf_eigvec = false");
        if (params.qsgw_vxc_basis != "state" && params.qsgw_vxc_basis != "nao")
            throw std::runtime_error("qsgw_vxc_basis must be state or nao");
        if (params.qsgw_mixer != "none" && params.qsgw_mixer != "linear")
            throw std::runtime_error("qsgw_mixer must be none or linear");
        if (!(params.qsgw_mixing_beta > 0.0 &&
              params.qsgw_mixing_beta <= 1.0) ||
            !std::isfinite(params.qsgw_mixing_beta))
            throw std::runtime_error("qsgw_mixing_beta must be in (0, 1]");
        if (params.qsgw_min_iter < 1 ||
            params.qsgw_max_iter < params.qsgw_min_iter)
            throw std::runtime_error(
                "qsgw iteration bounds must satisfy 1 <= min <= max");
        if (params.qsgw_band0_unoccupied_keep < 0)
            throw std::runtime_error(
                "qsgw_band0_unoccupied_keep must be non-negative");
        if (params.qsgw_band0_cut_mode < 0 ||
            params.qsgw_band0_cut_mode > 2)
            throw std::runtime_error(
                "qsgw_band0_cut_mode must be 0, 1, or 2");
        if (!std::isfinite(params.qsgw_band0_cut_shift_ha))
            throw std::runtime_error(
                "qsgw_band0_cut_shift_ha must be finite");
        if (!(params.qsgw_convergence_tolerance_ev > 0.0) ||
            !std::isfinite(params.qsgw_convergence_tolerance_ev))
            throw std::runtime_error(
                "qsgw_convergence_tolerance_ev must be finite and positive");
        if (params.use_pyatb)
            throw std::runtime_error(
                "QSGW independent PyATB head updates are unsupported; use the same-grid velocity input");
        if (driver::opts.replace_w_head == LIBRPA_SWITCH_ON &&
            driver::opts.option_dielect_func != 4)
            throw std::runtime_error(
                "QSGW supports only analytic head-only mode; set option_dielect_func = 4");
        // QSGW inherits the upstream EXX, GW, and RPA symmetry switches.
    }
}

void parse_inputfile_to_params(const std::string &fn)
{
    using namespace driver;
    using librpa_int::global::dev_opts;

    InputFile inputf;
    int flag;
    auto parser = inputf.load(fn, true);

    std::string stmp;
    bool btmp;

    // TODO: invalid parameters checker

    // driver parameters
    parser.parse_string("task", driver_params.task, "unset", flag);
    parser.parse_string("constants_choice", driver_params.constants_choice, "internal", flag);
    if (driver_params.constants_choice == "aims")
    {
        librpa_int::set_aims_constants();
    }
    parser.parse_string("input_preset", driver_params.input_preset, "fhi-aims", flag);
    driver_params.apply_input_preset();
    _parse_string_post(driver_params, input_dir, check_dirpath);
    _parse_double(driver_params, cs_threshold);
    _parse_bool(driver_params, output_energy_qp);
    _parse_int(driver_params, i_state_low);
    _parse_int(driver_params, i_state_high);
    _parse_bool(driver_params, output_hamgnn);
    _parse_bool(driver_params, use_pyatb);
    _parse_bool(driver_params, output_gw_spec_func);
    _parse_bool(driver_params, use_spinor_wfc);
    parser.parse_bool("use_soc", driver_params.use_spinor_wfc, flag);  // backward-compatible
    if (driver_params.use_spinor_wfc)
    {
        driver::n_spinor = 2;
    }
    _parse_string(driver_params, prefix_lri_coeff);
    _parse_string(driver_params, prefix_lri_coeff_shrink);
    _parse_string(driver_params, prefix_shrink_sinvS);
    _parse_string(driver_params, prefix_coul_full);
    _parse_string(driver_params, prefix_sternheimer_chi0);
    _parse_string(driver_params, fn_sternheimer_qpoints);
    _parse_string(driver_params, fn_sternheimer_partial_manifest);
    _parse_string(driver_params, fn_sternheimer_symmetry_routes);
    _parse_string(driver_params, fn_sternheimer_qstar_routes);
    _parse_string(driver_params, prefix_sternheimer_reconstructed);
    _parse_string(driver_params, prefix_sternheimer_kresolved);
    _parse_string(driver_params, prefix_sternheimer_symmetry_diagnostic);
    _parse_int(driver_params, sternheimer_iq);
    _parse_double(driver_params, sternheimer_qweight);
    _parse_bool(driver_params, use_rpa_gamma);
    _parse_bool(driver_params, sternheimer_matrix_only);
    _parse_string(driver_params, prefix_coul_cut);
    _parse_string(driver_params, prefix_eigvecs_scf);
    _parse_string(driver_params, fn_stru);
    _parse_string(driver_params, fn_basis);
    _parse_string(driver_params, fn_basis_wfc);
    _parse_string(driver_params, fn_basis_aux);
    _parse_string(driver_params, fn_basis_aux_shrink);
    parser.parse_string("fn_basis_shrink", driver_params.fn_basis_aux_shrink, flag);
    _parse_string(driver_params, fn_bz_sampling);
    _parse_string(driver_params, fn_eigocc_scf);
    _parse_string(driver_params, fn_dielfunc);
    _parse_string(driver_params, fn_vxc_scf);
    _parse_string(driver_params, prefix_velocity);
    _parse_string(driver_params, fn_band_kpath_info);
    std::string task_normalized = driver_params.task;
    std::transform(task_normalized.begin(), task_normalized.end(),
                   task_normalized.begin(), [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    if (task_normalized == "qsgw" || task_normalized == "qsgw_band")
    {
        for (const std::string removed :
             {"qsgw_update_hartree", "qsgw_hartree_coulomb", "qsgw_hartree_normalization",
              "qsgw_export_hamiltonian_for_pyatb", "qsgw_hr_export_full_mp_rgrid"})
        {
            std::string ignored;
            if (get_last_assigned_value(parser, removed, ignored))
            {
                throw std::runtime_error(
                    removed + " is not supported by the head-only QSGW workflow");
            }
        }
        parse_qsgw_string(parser, "prefix_vxc_scf", driver_params.prefix_vxc_scf);
        parse_qsgw_string(parser, "prefix_vxc_band", driver_params.prefix_vxc_band);
        parse_qsgw_string(parser, "qsgw_vxc_basis", driver_params.qsgw_vxc_basis);
        parse_qsgw_string(parser, "qsgw_mixer", driver_params.qsgw_mixer);
        std::transform(driver_params.qsgw_mixer.begin(),
                       driver_params.qsgw_mixer.end(),
                       driver_params.qsgw_mixer.begin(),
                       [](const unsigned char ch) {
                           return static_cast<char>(std::tolower(ch));
                       });
        parse_qsgw_double(parser, "qsgw_mixing_beta",
                          driver_params.qsgw_mixing_beta);
        parse_qsgw_int(parser, "qsgw_min_iter", driver_params.qsgw_min_iter);
        parse_qsgw_int(parser, "qsgw_max_iter", driver_params.qsgw_max_iter);
        parse_qsgw_int(parser, "qsgw_band0_unoccupied_keep",
                       driver_params.qsgw_band0_unoccupied_keep);
        parse_qsgw_int(parser, "qsgw_band0_cut_mode",
                       driver_params.qsgw_band0_cut_mode);
        parse_qsgw_double(parser, "qsgw_band0_cut_shift_ha",
                          driver_params.qsgw_band0_cut_shift_ha);
        parse_qsgw_bool(parser, "qsgw_write_iteration_matrices",
                        driver_params.qsgw_write_iteration_matrices);
        parse_qsgw_double(parser, "qsgw_convergence_tolerance_ev",
                          driver_params.qsgw_convergence_tolerance_ev);
    }
    _parse_int(driver_params, version_coul_reader);
    _parse_int(driver_params, version_lri_reader);

    // TODO: implement a function to read multiple double values in one line
    if (driver_params.output_gw_spec_func)
    {
        _parse_double(driver_params, sf_omega_start);
        _parse_double(driver_params, sf_omega_end);
        _parse_double(driver_params, sf_omega_step);
        _parse_int(driver_params, sf_state_start);
        _parse_int(driver_params, sf_state_end);
    }

    // Development options, see src/utils/dev_options.h
    // Only for prototyping, testing and debugging in C++
    _parse_bool(dev_opts, use_chi0_q_uhap_split);
    _parse_bool(dev_opts, use_delayed_ft_shrink);
    _parse_bool(dev_opts, use_real_dense_gw_wc);

    // general runtime parameters
    parser.parse_string("output_dir", stmp, flag);
    if (flag == 0)
        opts.set_output_dir(stmp.c_str());
    parser.parse_string("restart_from_dir", stmp, flag);
    if (flag == 0)
        opts.set_restart_from_dir(stmp.c_str());
    _parse_string_post(opts, parallel_routing, get_parallel_routing);

    parser.parse_bool("debug", btmp, false, flag);  // backward-compatible
    if (btmp) driver_params.output_level = LIBRPA_VERBOSE_DEBUG;

    parser.parse_string("output_level", stmp, "info", flag);
    if (flag == 0) driver_params.output_level = get_verbose(stmp);
    librpa::set_output_level(driver_params.output_level);

    _parse_double(opts, vq_threshold);

    _parse_switch(opts, use_kpara_scf_eigvec);

    _parse_string_post(opts, tfgrids_type, get_tfgrid_type);
    if (flag != 0) // backward compatible
    {
        parser.parse_string("tfgrid_type", stmp, flag);
        if (flag == 0) opts.tfgrids_type = get_tfgrid_type(stmp);
    }
    if (opts.tfgrids_type == LIBRPA_TFGRID_UNSET)
        opts.tfgrids_type = LIBRPA_TFGRID_MINIMAX;
    _parse_int(opts, nfreq);
    _parse_double(opts, tfgrids_freq_min);
    _parse_double(opts, tfgrids_freq_interval);
    _parse_double(opts, tfgrids_freq_max);
    _parse_double(opts, tfgrids_time_min);
    _parse_double(opts, tfgrids_time_interval);

    _parse_double(opts, minimax_emin);
    _parse_double(opts, minimax_emax);
    _parse_double(opts, minimax_regulation);

    _parse_switch(opts, use_fullcoul_eps);
    _parse_switch(opts, use_fullcoul_exx);
    _parse_switch(opts, use_fullcoul_wc);
    _parse_switch(opts, use_symmetry_exx);
    _parse_switch(opts, use_symmetry_gw);
    parser.parse_bool("use_symmetry_rpa", btmp, flag);
    opts.use_symmetry_rpa = flag == 0 ? get_switch(btmp) : opts.use_symmetry_gw;
    _parse_switch(opts, output_abacus_gw_gf);

    _parse_int(opts, n_bands_chi0);
    _parse_int(opts, n_bands_sigc);
    _parse_int(opts, n_bands_exclude_chi0);
    _parse_int(opts, n_bands_exclude_sigc);
    _parse_int(opts, option_bvk_remap);

    // chi0 related
    _parse_switch(opts, use_shrink_abfs);
    _parse_switch(opts, use_shrink_chi);

    // RPA specific
    parser.parse_double("gf_R_threshold", opts.gf_threshold, flag); // backward compatible
    _parse_double(opts, gf_threshold);
    _parse_int(opts, libri_chi0_collect_s0_chunk);
    if (opts.libri_chi0_collect_s0_chunk < 0)
        throw std::runtime_error("libri_chi0_collect_s0_chunk must be non-negative");
    {
        double bytes_tmp = 0.0;
        parser.parse_double("libri_chi0_collect_max_bytes", bytes_tmp, flag);
        if (flag == 0)
        {
            if (bytes_tmp < 0.0 ||
                bytes_tmp > static_cast<double>(std::numeric_limits<long long>::max()))
            {
                throw std::runtime_error("libri_chi0_collect_max_bytes must be a non-negative integer byte count");
            }
            opts.libri_chi0_collect_max_bytes = static_cast<long long>(bytes_tmp);
        }
    }
    if (opts.libri_chi0_collect_max_bytes < 0)
        throw std::runtime_error("libri_chi0_collect_max_bytes must be non-negative");
    _parse_double(opts, libri_chi0_threshold_C);
    _parse_double(opts, libri_chi0_threshold_G);
    _parse_switch(opts, use_scalapack_ecrpa);

    // EXX specific
    _parse_double(opts, libri_exx_threshold_C);
    _parse_double(opts, libri_exx_threshold_D);
    _parse_double(opts, libri_exx_threshold_V);

    // GW specific
    _parse_int(opts, n_params_anacon);
    _parse_int(opts, n_params_anacon_resample);
    if (opts.n_params_anacon == 0)
        throw std::runtime_error("n_params_anacon must not be zero");
    if (opts.n_params_anacon_resample == 0)
        throw std::runtime_error("n_params_anacon_resample must not be zero");
    _parse_string_post(opts, anacon_tfgrids_type, get_tfgrid_type);
    _parse_int(opts, anacon_nfreq);
    if (opts.anacon_nfreq == 0)
        throw std::runtime_error("anacon_nfreq must not be zero");
    _parse_double(opts, sqrt_coulomb_threshold);
    _parse_switch(opts, use_scalapack_gw_wc);
    _parse_switch(opts, read_sigc_mat_rf);
    if (flag != 0)  // backward-compatible
    {
        parser.parse_bool("read_sigc", btmp, flag);
        if (flag == 0) opts.read_sigc_mat_rf = get_switch(btmp);
    }
    _parse_switch(opts, use_cholesky_gw_wc);
    _parse_switch(opts, use_gpu_replace_scalapack);
    _parse_switch(opts, use_elpa_sqrt_coulomb);
    _parse_switch(opts, use_hedin_shift);
    _parse_int(opts, istate_ref_hedin_shift);
    _parse_double(opts, libri_g0w0_threshold_C);
    _parse_double(opts, libri_g0w0_threshold_G);
    _parse_double(opts, libri_g0w0_threshold_Wc);
    _parse_switch(opts, replace_w_head);
    _parse_int(opts, option_dielect_func);
    _parse_switch(opts, use_2d_dielectric);
    _parse_switch(opts, output_gw_sigc_ks_kf);
    if (flag != 0)  // backward-compatible
    {
        parser.parse_bool("output_gw_sigc_ks_if", btmp, flag);
        if (flag == 0) opts.output_gw_sigc_ks_kf = get_switch(btmp);
    }
    _parse_int(opts, rpa_headwing_body_start);
    if (opts.rpa_headwing_body_start < -1)
        throw std::runtime_error("rpa_headwing_body_start must be at least -1");
    parser.parse_string("rpa_headwing_mode", stmp, "qavg", flag);
    if (flag == 0 || flag == 1)
    {
        if (stmp != "qavg" && stmp != "head_only")
            throw std::runtime_error("rpa_headwing_mode must be qavg or head_only");
        std::strncpy(opts.rpa_headwing_mode, stmp.c_str(), LIBRPA_MAX_STRLEN);
        opts.rpa_headwing_mode[LIBRPA_MAX_STRLEN - 1] = '\0';
    }
    _parse_switch(opts, output_gw_sigc_ks_mat_kf);
    if (flag != 0)  // backward-compatible
    {
        parser.parse_bool("output_gw_sigc_mat", btmp, flag);
        if (flag == 0) opts.output_gw_sigc_ks_mat_kf = get_switch(btmp);
    }
    _parse_switch(opts, output_exx_ks_mat_k);
    _parse_switch(opts, output_exx_mat_k);
    _parse_int(opts, istate_output_mat_start);
    _parse_int(opts, istate_output_mat_end);
    if (opts.istate_output_mat_start < 0)
        throw std::runtime_error("istate_output_mat_start must be non-negative");
    if (opts.istate_output_mat_end >= 0 &&
        opts.istate_output_mat_end <= opts.istate_output_mat_start)
        throw std::runtime_error(
            "istate_output_mat_end must be negative or greater than "
            "istate_output_mat_start");
    _parse_switch(opts, output_gw_sigc_mat_kf);
    _parse_switch(opts, output_gw_sigc_mat_rt);
    _parse_switch(opts, output_gw_sigc_mat_rf);
    _parse_switch(opts, output_wc_rf);
    _parse_switch(opts, output_wc_rf_atom_pair);
    _parse_int(opts, ifreq_output_wc_start);
    _parse_int(opts, ifreq_output_wc_end);
    {  // backward-compatible
        int option_output_Wc_Rf_mat = 0;
        parser.parse_int("option_output_Wc_Rf_mat", option_output_Wc_Rf_mat, flag);
        if (flag == 0)
        {
            if (option_output_Wc_Rf_mat < 0 || option_output_Wc_Rf_mat > 2)
                throw std::runtime_error("option_output_Wc_Rf_mat must be 0, 1, or 2");
            opts.output_wc_rf = get_switch(option_output_Wc_Rf_mat > 0);
            opts.ifreq_output_wc_start = 0;
            opts.ifreq_output_wc_end = option_output_Wc_Rf_mat == 1 ? 1 : -1;
        }
    }
    if (opts.ifreq_output_wc_start < 0)
        throw std::runtime_error("ifreq_output_wc_start must be non-negative");
    if (opts.ifreq_output_wc_end >= 0 &&
        opts.ifreq_output_wc_end <= opts.ifreq_output_wc_start)
        throw std::runtime_error("ifreq_output_wc_end must be negative or greater than ifreq_output_wc_start");

    // QPE solver
    _parse_int(opts, option_qpe_solver);
    _parse_int(opts, qpe_solver_n_iter_max);
    _parse_double(opts, qpe_solver_thres);
    _parse_double(opts, qpe_solver_damp_factor);
    _parse_switch(opts, use_qpe_adaptive_damp);
    _parse_switch(opts, use_qpe_legacy_update);
    _parse_switch(opts, override_qpe_solver_nan);

    // Spectral function
    _parse_double(opts, sf_gf_omega_shift);
    _parse_double(opts, sf_sigc_omega_shift);

    validate_input_parameters();
}

#undef _parse_int
#undef _parse_double
#undef _parse_switch
#undef _parse_string_post
