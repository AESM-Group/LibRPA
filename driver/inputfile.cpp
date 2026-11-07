#include "inputfile.h"

#include <librpa_enums.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "../src/io/fs.h"
#include "../src/utils/constants.h"
#include "../src/utils/dev_options.h"
#include "driver.h"

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

static void parse_crpa_parameters(const InputParser &parser, driver::DriverParams &params)
{
    for (auto selection :
         {std::make_tuple("crpa_response_windows_ha", "crpa_response_bands",
                          &params.crpa_response_windows_ha, &params.crpa_response_bands),
          std::make_tuple("crpa_orbital_windows_ha", "crpa_orbital_bands",
                          &params.crpa_orbital_windows_ha, &params.crpa_orbital_bands)})
    {
        std::string windows, bands;
        const bool assigned_windows =
            get_last_assigned_value(parser, std::get<0>(selection), windows);
        const bool assigned_bands = get_last_assigned_value(parser, std::get<1>(selection), bands);
        if (assigned_windows && !assigned_bands) std::get<3>(selection)->clear();
        if (assigned_bands && !assigned_windows) std::get<2>(selection)->clear();
    }
    parse_qsgw_bool(parser, "crpa_output_analysis", params.crpa_output_analysis);
    parse_qsgw_bool(parser, "crpa_output_model", params.crpa_output_model);
    parse_qsgw_bool(parser, "crpa_output_raw", params.crpa_output_raw);
    parse_qsgw_bool(parser, "crpa_track_orbitals", params.crpa_track_orbitals);
    parse_qsgw_double(parser, "crpa_tracking_min_capture", params.crpa_tracking_min_capture);
    parse_qsgw_bool(parser, "crpa_output_static", params.crpa_output_static);
    parse_qsgw_double(parser, "crpa_fit_tol_ev", params.crpa_fit_tol_ev);
    parse_qsgw_double(parser, "crpa_eta_ev", params.crpa_eta_ev);
    parse_qsgw_bool(parser, "crpa_analysis_only", params.crpa_analysis_only);
    get_last_assigned_value(parser, "crpa_pdos_reference", params.crpa_pdos_reference);
    parse_qsgw_double(parser, "crpa_pdos_min_ev", params.crpa_pdos_min_ev);
    parse_qsgw_double(parser, "crpa_pdos_max_ev", params.crpa_pdos_max_ev);
    parse_qsgw_double(parser, "crpa_pdos_step_ev", params.crpa_pdos_step_ev);
    parse_qsgw_double(parser, "crpa_pdos_sigma_ev", params.crpa_pdos_sigma_ev);
    for (auto item : std::initializer_list<std::pair<const char *, std::string *>>{
             {"crpa_overlap_file", &params.crpa_overlap_file},
             {"crpa_species_labels", &params.crpa_species_labels},
             {"crpa_correlated_species", &params.crpa_correlated_species},
             {"crpa_ligand_species", &params.crpa_ligand_species},
             {"crpa_parent_orbitals", &params.crpa_parent_orbitals},
             {"crpa_output_orbitals", &params.crpa_output_orbitals},
             {"crpa_response_mode", &params.crpa_response_mode},
             {"crpa_output_format", &params.crpa_output_format}})
        get_last_assigned_value(parser, item.first, *item.second);
    for (auto item : std::initializer_list<std::pair<const char *, std::vector<double> *>>{
             {"crpa_response_windows_ha", &params.crpa_response_windows_ha},
             {"crpa_orbital_windows_ha", &params.crpa_orbital_windows_ha},
             {"crpa_real_frequencies_ev", &params.crpa_real_frequencies_ev}})
    {
        std::string text;
        if (!get_last_assigned_value(parser, item.first, text)) continue;
        std::replace(text.begin(), text.end(), ',', ' ');
        std::replace(text.begin(), text.end(), 'D', 'E');
        std::replace(text.begin(), text.end(), 'd', 'e');
        std::istringstream stream(text);
        item.second->clear();
        double value;
        while (stream >> value) item.second->push_back(value);
        if (!stream.eof())
            throw std::runtime_error(std::string(item.first) + " contains an invalid number");
    }
    for (auto item : std::initializer_list<std::pair<const char *, std::vector<int> *>>{
             {"crpa_response_bands", &params.crpa_response_bands},
             {"crpa_orbital_bands", &params.crpa_orbital_bands}})
    {
        std::string text;
        if (!get_last_assigned_value(parser, item.first, text)) continue;
        std::replace(text.begin(), text.end(), ',', ' ');
        std::istringstream stream(text);
        item.second->clear();
        int value;
        while (stream >> value) item.second->push_back(value);
        if (!stream.eof())
            throw std::runtime_error(std::string(item.first) + " contains an invalid integer");
    }
    std::string tolerance;
    if (get_last_assigned_value(parser, "crpa_residual_tol", tolerance))
    {
        std::size_t consumed = 0;
        params.crpa_residual_tol = std::stod(tolerance, &consumed);
        if (consumed != tolerance.size())
            throw std::runtime_error("crpa_residual_tol must be a valid number");
    }
}

static void validate_input_parameters()
{
    const auto &params = driver::driver_params;
    std::string task = params.task;
    std::transform(task.begin(), task.end(), task.begin(),
                   [](const unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    if (task == "hartree")
    {
        if (params.prefix_coul_hartree.empty())
            throw std::runtime_error(
                "task=hartree requires explicit prefix_coul_hartree for bare 3D G=0-zero Coulomb input");
        if (!(params.hartree_reference_scale > 0.0) ||
            !std::isfinite(params.hartree_reference_scale))
            throw std::runtime_error("hartree_reference_scale must be finite and positive");
        if (!std::isfinite(params.hartree_check_tolerance) || params.hartree_check_tolerance == 0.0)
            throw std::runtime_error(
                "hartree_check_tolerance must be finite and positive, or negative for report only");
        if (params.hartree_check_tolerance > 0.0 && params.hartree_fn_reference.empty())
            throw std::runtime_error(
                "A positive hartree_check_tolerance requires hartree_fn_reference");
    }
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
    if (task == "g0w0" || task == "g0w0_band" || task == "crpa")
    {
        if (params.g0w0_headwing_velocity != "file" && params.g0w0_headwing_velocity != "native")
            throw std::runtime_error("g0w0_headwing_velocity must be file or native");
        if (params.g0w0_headwing_velocity == "native")
        {
            if (driver::opts.replace_w_head != LIBRPA_SWITCH_ON ||
                (driver::opts.option_dielect_func != 3 && driver::opts.option_dielect_func != 4))
                throw std::runtime_error(
                    "G0W0 native velocity requires replace_w_head and option_dielect_func = 3 or "
                    "4");
            if (driver::opts.option_dielect_func == 3 &&
                driver::opts.use_scalapack_gw_wc != LIBRPA_SWITCH_ON)
                throw std::runtime_error(
                    "G0W0 native velocity with option_dielect_func = 3 requires "
                    "use_scalapack_gw_wc = true to consume analytic wings");
            if (params.g0w0_fn_hr.empty() || params.g0w0_fn_sr.empty() || params.g0w0_fn_rr.empty())
                throw std::runtime_error(
                    "G0W0 native velocity requires g0w0_fn_hr, g0w0_fn_sr, and g0w0_fn_rr");
            if (params.g0w0_hr_unit != "Ha" && params.g0w0_hr_unit != "Ry" &&
                params.g0w0_hr_unit != "eV")
                throw std::runtime_error("g0w0_hr_unit must be Ha, Ry, or eV");
            if (params.g0w0_rr_unit != "Bohr" && params.g0w0_rr_unit != "Angstrom")
                throw std::runtime_error("g0w0_rr_unit must be Bohr or Angstrom");
            if (driver::opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON)
                throw std::runtime_error(
                    "G0W0 native velocity requires undistributed AO/band wavefunction blocks; "
                    "set use_kpara_scf_eigvec = false");
            const bool use_symmetry = driver::opts.use_symmetry_rpa == LIBRPA_SWITCH_ON ||
                                      driver::opts.use_symmetry_gw == LIBRPA_SWITCH_ON ||
                                      driver::opts.use_symmetry_exx == LIBRPA_SWITCH_ON;
            if (use_symmetry && !params.use_spinor_wfc)
                throw std::runtime_error(
                    "Scalar G0W0 native velocity requires a full-BZ input without symmetry "
                    "reduction");
            if (params.use_pyatb)
                throw std::runtime_error("G0W0 native velocity requires use_pyatb = false");
        }
    }
    if (params.crpa_enabled && task != "qsgw" && task != "g0w0" && task != "g0w0_band")
        throw std::runtime_error(
            "crpa_enabled requires task=g0w0 or task=qsgw; use task=crpa for standalone screening");
    if (params.crpa_interval < 1) throw std::runtime_error("crpa_interval must be positive");
    if (params.crpa_enabled && params.crpa_analysis_only)
        throw std::runtime_error(
            "Coupled GW cRPA requires screening, so crpa_analysis_only must be false");
    if (task == "crpa" || params.crpa_enabled)
    {
        if (params.crpa_response_mode != "bands" && params.crpa_response_mode != "orbital_weights")
            throw std::runtime_error("crpa_response_mode must be bands or orbital_weights");
        if ((params.crpa_track_orbitals && params.crpa_response_mode != "orbital_weights") ||
            !std::isfinite(params.crpa_tracking_min_capture) ||
            params.crpa_tracking_min_capture <= 0 || params.crpa_tracking_min_capture > 1)
            throw std::runtime_error(
                "cRPA tracking requires orbital_weights and 0 < tracking_min_capture <= 1");
        if (params.crpa_response_mode == "orbital_weights" &&
            (!params.crpa_response_windows_ha.empty() || !params.crpa_response_bands.empty()))
            throw std::runtime_error(
                "orbital_weights derives A from output orbitals; remove A windows/bands");
        if (params.crpa_species_labels.empty() ||
            (!params.crpa_analysis_only && params.crpa_correlated_species.empty()) ||
            params.crpa_overlap_file.empty())
            throw std::runtime_error(
                "crpa requires species labels, overlap and, for screening, correlated species");
        if (!std::isfinite(params.crpa_residual_tol) || params.crpa_residual_tol <= 0.0)
            throw std::runtime_error("crpa_residual_tol must be finite and positive");
        if (params.crpa_analysis_only && !params.crpa_output_analysis)
            throw std::runtime_error("crpa_analysis_only requires crpa_output_analysis");
        if (driver::opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON)
            throw std::runtime_error("crpa requires full-grid replicated KS states");
        if (params.crpa_output_analysis)
        {
            if (params.crpa_pdos_reference != "vbm" && params.crpa_pdos_reference != "fermi")
                throw std::runtime_error("crpa_pdos_reference must be vbm or fermi");
            if (!std::isfinite(params.crpa_pdos_min_ev) ||
                !std::isfinite(params.crpa_pdos_max_ev) ||
                !std::isfinite(params.crpa_pdos_step_ev) ||
                !std::isfinite(params.crpa_pdos_sigma_ev) || params.crpa_pdos_step_ev <= 0 ||
                params.crpa_pdos_sigma_ev <= 0 ||
                params.crpa_pdos_max_ev < params.crpa_pdos_min_ev ||
                (params.crpa_pdos_max_ev - params.crpa_pdos_min_ev) / params.crpa_pdos_step_ev >
                    std::numeric_limits<int>::max() - 1)
                throw std::runtime_error("invalid crpa PDOS grid or Gaussian width");
        }
        if (params.crpa_output_format != "npz" && params.crpa_output_format != "dat")
            throw std::runtime_error("crpa_output_format must be npz or dat");
        for (const auto selection :
             {std::make_pair(&params.crpa_response_windows_ha, &params.crpa_response_bands),
              std::make_pair(&params.crpa_orbital_windows_ha, &params.crpa_orbital_bands)})
        {
            if (params.crpa_analysis_only) continue;
            if (selection.first == &params.crpa_response_windows_ha &&
                params.crpa_response_mode == "orbital_weights")
                continue;
            const auto *edges = selection.first;
            const auto *bands = selection.second;
            if (edges->empty() == bands->empty())
                throw std::runtime_error(
                    "each cRPA A/B selection requires either energy windows or explicit bands");
            if (edges->size() % 2 != 0)
                throw std::runtime_error(
                    "cRPA A/B energy windows require lower/upper pairs in Hartree");
            for (std::size_t i = 0; i < edges->size(); i += 2)
                if (!std::isfinite((*edges)[i]) || !std::isfinite((*edges)[i + 1]) ||
                    (*edges)[i] > (*edges)[i + 1])
                    throw std::runtime_error("invalid cRPA energy-window bounds");
        }
    }
    if (task == "qsgw")
    {
        if (params.qsgw_update_hartree && params.prefix_coul_hartree.empty())
            throw std::runtime_error(
                "qsgw_update_hartree requires a dedicated prefix_coul_hartree for bare 3D G=0-zero Coulomb input");
        if (params.qsgw_update_hartree && params.prefix_lri_coeff_hartree.empty())
            throw std::runtime_error(
                "qsgw_update_hartree requires prefix_lri_coeff_hartree with unwrapped physical translations");
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
        if (params.qsgw_unoccupied_keep < 0)
            throw std::runtime_error("qsgw_unoccupied_keep must be non-negative");
        if (params.qsgw_cut_mode == 2)
            throw std::runtime_error(
                "qsgw_cut_mode = 2 was renumbered to 1; use 0 (uncut) or 1 (shifted KS diagonal)");
        if (params.qsgw_cut_mode != 0 && params.qsgw_cut_mode != 1)
            throw std::runtime_error("qsgw_cut_mode must be 0 or 1");
        if (!std::isfinite(params.qsgw_cut_shift_ha))
            throw std::runtime_error("qsgw_cut_shift_ha must be finite");
        if (!(params.qsgw_convergence_tolerance_ev > 0.0) ||
            !std::isfinite(params.qsgw_convergence_tolerance_ev))
            throw std::runtime_error(
                "qsgw_convergence_tolerance_ev must be finite and positive");
        if (params.use_pyatb)
            throw std::runtime_error(
                "QSGW independent PyATB head updates are unsupported; use the same-grid velocity input");
        if (driver::opts.replace_w_head == LIBRPA_SWITCH_ON &&
            driver::opts.option_dielect_func != 3 && driver::opts.option_dielect_func != 4)
            throw std::runtime_error(
                "QSGW analytic head/wing requires option_dielect_func = 3 or 4");
        if (params.qsgw_headwing_velocity != "reference" &&
            params.qsgw_headwing_velocity != "rotate_reference" &&
            params.qsgw_headwing_velocity != "current_hr")
            throw std::runtime_error("Unsupported qsgw_headwing_velocity convention");
        if (driver::opts.replace_w_head == LIBRPA_SWITCH_ON &&
            driver::opts.option_dielect_func == 3 && params.qsgw_headwing_velocity == "reference")
            throw std::runtime_error("QSGW wings require rotate_reference or current_hr velocity");
        if (params.qsgw_headwing_velocity == "current_hr" &&
            (params.qsgw_fn_sr.empty() || params.qsgw_fn_rr.empty()))
            throw std::runtime_error("QSGW current_hr velocity requires qsgw_fn_sr and qsgw_fn_rr");
        if (params.qsgw_export_hamiltonian_for_pyatb &&
            (params.qsgw_fn_sr.empty() || params.qsgw_fn_rr.empty()))
            throw std::runtime_error("QSGW HR export requires qsgw_fn_sr and qsgw_fn_rr");
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
    _parse_string(driver_params, prefix_lri_coeff_hartree);
    _parse_string(driver_params, prefix_shrink_sinvS);
    _parse_string(driver_params, prefix_coul_full);
    _parse_string(driver_params, prefix_coul_hartree);
    _parse_string(driver_params, hartree_fn_reference);
    _parse_double(driver_params, hartree_reference_scale);
    _parse_double(driver_params, hartree_check_tolerance);
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
    parse_qsgw_bool(parser, "crpa_enabled", driver_params.crpa_enabled);
    parse_qsgw_int(parser, "crpa_interval", driver_params.crpa_interval);
    parse_qsgw_string(parser, "crpa_model_file", driver_params.crpa_model_file);
    if (task_normalized == "crpa" || driver_params.crpa_enabled)
    {
        if (driver_params.crpa_model_file != "none")
        {
            std::filesystem::path model_path(driver_params.crpa_model_file);
            if (model_path.is_relative())
                model_path = std::filesystem::path(fn).parent_path() / model_path;
            InputFile model_file;
            const auto model_parser = model_file.load(model_path.string(), true);
            parse_crpa_parameters(model_parser, driver_params);
        }
        parse_crpa_parameters(parser, driver_params);
    }
    if (task_normalized == "g0w0" || task_normalized == "g0w0_band" || task_normalized == "crpa")
    {
        parse_qsgw_string(parser, "g0w0_headwing_velocity", driver_params.g0w0_headwing_velocity);
        parse_qsgw_string(parser, "g0w0_fn_hr", driver_params.g0w0_fn_hr);
        parse_qsgw_string(parser, "g0w0_fn_hr_spin1", driver_params.g0w0_fn_hr_spin1);
        parse_qsgw_string(parser, "g0w0_fn_sr", driver_params.g0w0_fn_sr);
        parse_qsgw_string(parser, "g0w0_fn_rr", driver_params.g0w0_fn_rr);
        parse_qsgw_string(parser, "g0w0_hr_unit", driver_params.g0w0_hr_unit);
        parse_qsgw_string(parser, "g0w0_rr_unit", driver_params.g0w0_rr_unit);
    }
    if (task_normalized == "qsgw")
    {
        std::string removed_system_type;
        if (get_last_assigned_value(parser, "qsgw_system_type", removed_system_type))
            throw std::runtime_error(
                "qsgw_system_type was removed; QSGW selects mixing automatically from the SCF "
                "k-point input");
        for (const std::string removed :
             {"qsgw_hartree_coulomb", "qsgw_hartree_normalization",
              "qsgw_hr_export_full_mp_rgrid"})
        {
            std::string ignored;
            if (get_last_assigned_value(parser, removed, ignored))
            {
                throw std::runtime_error(removed +
                                         " is not supported by the current QSGW workflow");
            }
        }
        for (const std::string renamed :
             {"qsgw_band0_unoccupied_keep", "qsgw_band0_cut_mode", "qsgw_band0_cut_shift_ha"})
        {
            std::string ignored;
            if (get_last_assigned_value(parser, renamed, ignored))
                throw std::runtime_error(renamed +
                                         " was renamed; remove band0_ from the parameter name");
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
        parse_qsgw_int(parser, "qsgw_unoccupied_keep", driver_params.qsgw_unoccupied_keep);
        parse_qsgw_int(parser, "qsgw_cut_mode", driver_params.qsgw_cut_mode);
        driver_params.qsgw_cut_shift_ha = 2.0 / librpa_int::HA2EV;
        parse_qsgw_double(parser, "qsgw_cut_shift_ha", driver_params.qsgw_cut_shift_ha);
        parse_qsgw_bool(parser, "qsgw_write_iteration_matrices",
                        driver_params.qsgw_write_iteration_matrices);
        parse_qsgw_bool(parser, "qsgw_update_hartree", driver_params.qsgw_update_hartree);
        parse_qsgw_bool(parser, "qsgw_output_qp", driver_params.qsgw_output_qp);
        parse_qsgw_bool(parser, "qsgw_export_hamiltonian_for_pyatb",
                        driver_params.qsgw_export_hamiltonian_for_pyatb);
        parse_qsgw_string(parser, "qsgw_headwing_velocity", driver_params.qsgw_headwing_velocity);
        parse_qsgw_string(parser, "qsgw_fn_sr", driver_params.qsgw_fn_sr);
        parse_qsgw_string(parser, "qsgw_fn_rr", driver_params.qsgw_fn_rr);
        parse_qsgw_string(parser, "qsgw_restart_file", driver_params.qsgw_restart_file);
        parse_qsgw_string(parser, "qsgw_checkpoint_file", driver_params.qsgw_checkpoint_file);
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
    _parse_string_post(opts, chi0_method, get_method);
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
    _parse_string_post(opts, anacon_method, get_anacon_method);
    _parse_double(opts, anacon_aaa_tolerance);
    if (!std::isfinite(opts.anacon_aaa_tolerance) || opts.anacon_aaa_tolerance <= 0.0)
        throw std::runtime_error("AAA tolerance must be finite and positive");
    _parse_double(opts, anacon_ridge_lambda);
    _parse_double(opts, anacon_ridge_den_weight);
    if (!std::isfinite(opts.anacon_ridge_lambda) || opts.anacon_ridge_lambda < 0.0 ||
        !std::isfinite(opts.anacon_ridge_den_weight) || opts.anacon_ridge_den_weight < 0.0)
        throw std::runtime_error("Ridge penalties must be finite and nonnegative");
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
    _parse_switch(opts, output_exx_mat_r);
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

    // Option 6 selects native G0W0/cRPA velocity or per-update QSGW current-HR
    // velocity. Each uses the existing numerical option-3 head/wing method.
    requested_option_dielect_func = opts.option_dielect_func;
    if (opts.option_dielect_func == 6)
    {
        const bool is_g0w0 = task_normalized == "g0w0" || task_normalized == "g0w0_band";
        const bool is_qsgw = task_normalized == "qsgw";
        const bool is_crpa = task_normalized == "crpa";
        if (!is_g0w0 && !is_qsgw && !is_crpa)
            throw std::runtime_error(
                "option_dielect_func = 6 is only supported for g0w0, g0w0_band, qsgw, or crpa");
        if (opts.replace_w_head != LIBRPA_SWITCH_ON)
            throw std::runtime_error("option_dielect_func = 6 requires replace_w_head = true");
        if (opts.use_scalapack_gw_wc != LIBRPA_SWITCH_ON)
            throw std::runtime_error(
                "option_dielect_func = 6 requires use_scalapack_gw_wc = true to consume analytic "
                "wings");
        const std::string velocity_key =
            is_qsgw ? "qsgw_headwing_velocity" : "g0w0_headwing_velocity";
        const std::string native_mode = is_qsgw ? "current_hr" : "native";
        std::string &velocity_mode =
            is_qsgw ? driver_params.qsgw_headwing_velocity : driver_params.g0w0_headwing_velocity;
        std::string assigned_velocity;
        if (get_last_assigned_value(parser, velocity_key, assigned_velocity) &&
            velocity_mode != native_mode)
            throw std::runtime_error("option_dielect_func = 6 selects " + velocity_key + " = " +
                                     native_mode + " and conflicts with " + velocity_key + " = " +
                                     velocity_mode);
        velocity_mode = native_mode;
        opts.option_dielect_func = 3;
    }

    validate_input_parameters();
}

#undef _parse_int
#undef _parse_double
#undef _parse_switch
#undef _parse_string_post
