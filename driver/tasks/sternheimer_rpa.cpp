#include "../../src/core/sternheimer_rpa.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../src/api/dataset_helper.h"
#include "../../src/api/instance_manager.h"
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../driver.h"
#include "../read_data.h"
#include "../reader_sternheimer.h"
#include "../reader_sternheimer_partial.h"
#include "../reader_sternheimer_qpoints.h"
#include "../rpa_qsum.h"
#include "../sternheimer_partial_reconstruction.h"
#include "../task.h"

void driver::task_sternheimer_rpa()
{
    using librpa_int::compute_sternheimer_rpa_frequency;
    using librpa_int::global::lib_printf;
    using librpa_int::global::lib_printf_root;
    using librpa_int::global::mpi_comm_global_h;

    struct QResult
    {
        struct SymmetryAudit
        {
            int ifreq = 0;
            int full_k_count = 0;
            int representative_k_count = 0;
            int little_group_order = 0;
            int qstar_size = 0;
            double hermiticity_relative_residual = 0.0;
            double max_integrand_difference = 0.0;
            std::complex<double> response_trace = {0.0, 0.0};
        };

        SternheimerQPoint point;
        std::vector<librpa_int::SternheimerRpaFrequencyResult> frequencies;
        std::vector<SymmetryAudit> symmetry_audits;
        std::complex<double> energy = {0.0, 0.0};
    };

    const bool partial_mode = !driver_params.fn_sternheimer_partial_manifest.empty();
    const bool write_reconstructed =
        partial_mode && !driver_params.prefix_sternheimer_reconstructed.empty();
    const bool write_kresolved =
        partial_mode && !driver_params.prefix_sternheimer_kresolved.empty();
    const bool write_symmetry_diagnostic =
        partial_mode && !driver_params.prefix_sternheimer_symmetry_diagnostic.empty();
    const bool matrix_only = driver_params.sternheimer_matrix_only;
    const bool replace_gamma_headwing =
        driver::get_bool(driver::opts.replace_w_head) &&
        (driver::opts.option_dielect_func == 3 || driver::opts.option_dielect_func == 4);
    // Analytic head/wing is the Gamma replacement, so it must keep Gamma in the q sum
    // even when the ordinary direct Gamma route is disabled.
    const bool include_gamma_in_rpa = driver_params.use_rpa_gamma || replace_gamma_headwing;
    const std::string headwing_mode(driver::opts.rpa_headwing_mode);
    if (replace_gamma_headwing && headwing_mode == "qavg" && driver::opts.option_dielect_func != 3)
    {
        throw std::runtime_error("Sternheimer RPA qavg head/wing requires option_dielect_func=3");
    }
    if (matrix_only && replace_gamma_headwing)
    {
        throw std::runtime_error(
            "sternheimer_matrix_only cannot apply an RPA energy head/wing correction");
    }
    validate_sternheimer_partial_task_contract(driver_params.fn_sternheimer_qpoints,
                                               driver_params.fn_sternheimer_partial_manifest,
                                               driver_params.fn_sternheimer_symmetry_routes);
    if (!driver_params.fn_sternheimer_qstar_routes.empty() &&
        driver_params.fn_sternheimer_symmetry_routes.empty())
    {
        throw std::runtime_error(
            "fn_sternheimer_qstar_routes requires fn_sternheimer_symmetry_routes");
    }
    if (!matrix_only && !driver_params.fn_sternheimer_symmetry_routes.empty() &&
        driver_params.fn_sternheimer_qstar_routes.empty())
    {
        throw std::runtime_error(
            "Discrete fixed-q Sternheimer routes require fn_sternheimer_qstar_routes in energy "
            "mode");
    }
    if (matrix_only && (!partial_mode || driver_params.fn_sternheimer_symmetry_routes.empty() ||
                        !write_reconstructed || !write_kresolved))
    {
        throw std::runtime_error(
            "sternheimer_matrix_only requires partial responses, explicit symmetry routes, "
            "prefix_sternheimer_reconstructed, and prefix_sternheimer_kresolved");
    }

    std::string manifest_path;
    std::string partial_manifest_path;
    std::string route_manifest_path;
    std::string qstar_route_manifest_path;
    std::vector<SternheimerQPoint> qpoints;
    if (driver_params.fn_sternheimer_qpoints.empty())
    {
        if (driver_params.sternheimer_iq <= 0)
        {
            throw std::runtime_error("sternheimer_iq must be a positive one-based q index");
        }
        if (!std::isfinite(driver_params.sternheimer_qweight) ||
            driver_params.sternheimer_qweight <= 0.0)
        {
            throw std::runtime_error("sternheimer_qweight must be positive");
        }
        qpoints.push_back(
            {driver_params.sternheimer_iq, {0.0, 0.0, 0.0}, driver_params.sternheimer_qweight});
    }
    else
    {
        manifest_path = librpa_int::is_absolute_path(driver_params.fn_sternheimer_qpoints)
                            ? driver_params.fn_sternheimer_qpoints
                            : librpa_int::join_path(driver_params.input_dir,
                                                    driver_params.fn_sternheimer_qpoints);
        qpoints = read_sternheimer_qpoint_manifest(manifest_path);
    }
    if (!include_gamma_in_rpa && manifest_path.empty())
    {
        throw std::runtime_error(
            "use_rpa_gamma=false requires fn_sternheimer_qpoints with the full q mesh");
    }
    if (partial_mode)
    {
        if (!matrix_only)
        {
            validate_sternheimer_gamma_contract(qpoints, include_gamma_in_rpa);
        }
        partial_manifest_path =
            librpa_int::is_absolute_path(driver_params.fn_sternheimer_partial_manifest)
                ? driver_params.fn_sternheimer_partial_manifest
                : librpa_int::join_path(driver_params.input_dir,
                                        driver_params.fn_sternheimer_partial_manifest);
    }
    else
    {
        validate_sternheimer_qpoint_input_files(
            qpoints, driver_params.input_dir, driver_params.prefix_coul_full,
            driver_params.prefix_sternheimer_chi0, driver::opts.nfreq, include_gamma_in_rpa);
    }

    SternheimerPartialResponseGroups partial_groups;
    if (partial_mode)
    {
        const auto records = read_sternheimer_partial_manifest(partial_manifest_path);
        partial_groups = read_sternheimer_partial_response_groups(records);
    }

    std::map<int, std::vector<SternheimerChi0V1Matrix>> response_cache;
    std::vector<double> headwing_frequencies;
    if (replace_gamma_headwing)
    {
        std::vector<std::pair<int, double>> frequency_metadata;
        if (partial_mode)
        {
            frequency_metadata.reserve(partial_groups.size());
            for (const auto &[key, group] : partial_groups)
            {
                frequency_metadata.emplace_back(group.ifreq, group.omega);
            }
        }
        else
        {
            const auto gamma =
                std::find_if(qpoints.cbegin(), qpoints.cend(),
                             [](const auto &point) { return is_rpa_gamma_point(point.q); });
            if (gamma == qpoints.cend())
            {
                throw std::runtime_error(
                    "Sternheimer RPA head/wing correction requires a Gamma response");
            }
            auto responses = read_sternheimer_chi0_v1_matrices(
                driver_params.input_dir, driver_params.prefix_sternheimer_chi0, gamma->iq);
            frequency_metadata.reserve(responses.size());
            for (const auto &response : responses)
            {
                frequency_metadata.emplace_back(response.ifreq, response.omega);
            }
            response_cache.emplace(gamma->iq, std::move(responses));
        }
        headwing_frequencies = librpa_int::sternheimer_frequency_grid_from_metadata(
            frequency_metadata, driver::opts.nfreq);
    }

    auto pds = librpa_int::api::get_dataset_instance(driver::h);
    if (replace_gamma_headwing)
    {
        read_headwing_input(driver_params.input_dir, headwing_mode == "qavg",
                            &headwing_frequencies);
    }
    auto headwing_for_frequency = [&](const int ifreq)
    {
        if (!replace_gamma_headwing || pds->p_headwing == nullptr)
        {
            throw std::logic_error("Sternheimer RPA analytic head/wing data are unavailable");
        }
        librpa_int::RpaHeadwingSettings settings;
        settings.enabled = true;
        settings.option_dielect_func = driver::opts.option_dielect_func;
        settings.use_2d_dielectric = driver::get_bool(driver::opts.use_2d_dielectric);
        settings.rpa_headwing_body_start = driver::opts.rpa_headwing_body_start;
        settings.rpa_headwing_mode = headwing_mode;
        settings.sqrt_coulomb_threshold = driver::opts.sqrt_coulomb_threshold;
        const int headwing_index =
            librpa_int::sternheimer_headwing_frequency_index(ifreq, driver::opts.nfreq);
        return pds->p_headwing->get_sternheimer_rpa_headwing_input(headwing_index, settings);
    };

    std::vector<QResult> qresults;
    qresults.reserve(qpoints.size());
    if (partial_mode)
    {
        const auto &groups = partial_groups;
        std::vector<SternheimerFixedQRouteRecord> fixed_q_routes;
        std::vector<SternheimerQStarRouteRecord> qstar_routes;
        if (!driver_params.fn_sternheimer_symmetry_routes.empty())
        {
            route_manifest_path =
                librpa_int::is_absolute_path(driver_params.fn_sternheimer_symmetry_routes)
                    ? driver_params.fn_sternheimer_symmetry_routes
                    : librpa_int::join_path(driver_params.input_dir,
                                            driver_params.fn_sternheimer_symmetry_routes);
            fixed_q_routes = read_sternheimer_fixed_q_route_manifest(route_manifest_path);
        }
        if (!driver_params.fn_sternheimer_qstar_routes.empty())
        {
            qstar_route_manifest_path =
                librpa_int::is_absolute_path(driver_params.fn_sternheimer_qstar_routes)
                    ? driver_params.fn_sternheimer_qstar_routes
                    : librpa_int::join_path(driver_params.input_dir,
                                            driver_params.fn_sternheimer_qstar_routes);
            qstar_routes = read_sternheimer_qstar_route_manifest(qstar_route_manifest_path);
        }
        const auto full_kpoint_manifest_path = librpa_int::join_path(
            librpa_int::parent_path(partial_manifest_path), "v1_sternheimer_full_kpoints.dat");
        const auto full_kpoint_records =
            read_sternheimer_full_kpoint_manifest(full_kpoint_manifest_path);
        librpa_int::initialize_symmetry_context(*pds, true);
        const auto &symmetry = pds->symmetry_context;
        const auto layouts = pds->basis_aux.build_species_basis_layouts(symmetry.atom_to_type);
        const auto atom_nabf = pds->basis_aux.get_atom_nb_map();
        if (full_kpoint_records.size() != pds->pbc.kfrac_list_full.size())
        {
            throw std::runtime_error(
                "Sternheimer full-k-point manifest count does not match the BvK k grid");
        }
        std::vector<librpa_int::Vector3_Order<double>> full_kpoints;
        full_kpoints.reserve(full_kpoint_records.size());
        for (const auto &record : full_kpoint_records)
        {
            full_kpoints.push_back({record.k[0], record.k[1], record.k[2]});
        }
        for (const auto &grid_kpoint : pds->pbc.kfrac_list_full)
        {
            if (std::none_of(full_kpoints.cbegin(), full_kpoints.cend(),
                             [&grid_kpoint](const auto &manifest_kpoint) {
                                 return librpa_int::same_fractional_kpoint(manifest_kpoint,
                                                                           grid_kpoint, 1.0e-8);
                             }))
            {
                throw std::runtime_error(
                    "Sternheimer full-k-point manifest does not cover the BvK k grid");
            }
        }
        // Sternheimer q routes are indexed on the full BvK q grid.  Keep that
        // grid separate from the producer's full-k manifest used for response
        // reconstruction, even when the two grids currently have the same mesh.
        const auto &full_qpoints = pds->pbc.kfrac_list_full;
        if (full_qpoints.empty())
        {
            throw std::runtime_error("Sternheimer reconstruction requires a non-empty full q grid");
        }
        const int lmax = pds->basis_aux.get_max_l();
        std::vector<librpa_int::SternheimerQStarResponse> coulomb_full_q;
        if (!matrix_only)
        {
            std::vector<librpa_int::ComplexMatrix> coulomb_ibz;
            coulomb_ibz.reserve(pds->pbc.kfrac_list.size());
            for (std::size_t index = 0; index != pds->pbc.kfrac_list.size(); ++index)
            {
                coulomb_ibz.push_back(read_coulomb_v1_full_matrix(driver_params.input_dir,
                                                                  driver_params.prefix_coul_full,
                                                                  static_cast<int>(index + 1)));
            }
            coulomb_full_q = reconstruct_sternheimer_full_q_matrices_from_ibz(
                symmetry, layouts, atom_nabf, pds->pbc.kfrac_list, coulomb_ibz, lmax);
        }
        const auto reconstructed = reconstruct_sternheimer_partial_responses(
            symmetry, layouts, atom_nabf, full_kpoints, full_qpoints, qpoints, groups,
            driver::opts.nfreq,
            include_gamma_in_rpa, lmax, fixed_q_routes.empty() ? nullptr : &fixed_q_routes,
            matrix_only, qstar_routes.empty() ? nullptr : &qstar_routes);

        if (write_symmetry_diagnostic && mpi_comm_global_h.is_root())
        {
            for (const auto &point : qpoints)
            {
                if (!include_gamma_in_rpa && is_rpa_gamma_point(point.q))
                {
                    continue;
                }
                std::vector<SternheimerFixedQRouteRecord> point_routes;
                std::copy_if(fixed_q_routes.cbegin(), fixed_q_routes.cend(),
                             std::back_inserter(point_routes),
                             [&point](const auto &route) { return route.iq == point.iq; });
                const auto diagnostics = build_sternheimer_fixed_q_symmetry_diagnostics(
                    symmetry, layouts, atom_nabf, full_kpoints,
                    {point.q[0], point.q[1], point.q[2]}, lmax,
                    point_routes.empty() ? nullptr : &point_routes);
                write_sternheimer_fixed_q_symmetry_diagnostics(
                    driver_params.prefix_sternheimer_symmetry_diagnostic + "iq_" +
                        std::to_string(point.iq) + ".dat",
                    point.iq, diagnostics);
            }
        }

        if (matrix_only)
        {
            if (mpi_comm_global_h.is_root())
            {
                for (const auto &response : reconstructed)
                {
                    const auto &metadata = groups.at({response.iq, response.ifreq});
                    SternheimerChi0V1Matrix aggregate;
                    aggregate.iq = response.iq;
                    aggregate.ifreq = response.ifreq;
                    aggregate.omega = response.omega;
                    aggregate.weight = response.weight;
                    aggregate.atom_naux = metadata.atom_naux;
                    aggregate.matrix = response.matrix;
                    write_sternheimer_chi0_v1_matrix_file(
                        driver_params.prefix_sternheimer_reconstructed +
                            std::to_string(aggregate.iq) + "_ifreq_" +
                            std::to_string(aggregate.ifreq) + ".dat",
                        aggregate);

                    for (const auto &member : response.kresolved_responses)
                    {
                        SternheimerChi0V1Matrix output = aggregate;
                        output.matrix = member.matrix;
                        write_sternheimer_chi0_v1_matrix_file(
                            driver_params.prefix_sternheimer_kresolved + std::to_string(output.iq) +
                                "_ik_" + std::to_string(member.ik_full) + "_ifreq_" +
                                std::to_string(output.ifreq) + ".dat",
                            output);
                    }
                    lib_printf(
                        "| fixed-q matrix-only iq = %d, ifreq = %d, representatives = %d, full k = "
                        "%d\n",
                        response.iq, response.ifreq, response.representative_k_count,
                        response.full_k_count);
                }
                lib_printf(
                    "Sternheimer fixed-q matrix-only reconstruction completed; no RPA energy was "
                    "evaluated.\n");
            }
            mpi_comm_global_h.barrier();
            return;
        }

        for (const auto &point : qpoints)
        {
            if (!include_gamma_in_rpa && is_rpa_gamma_point(point.q))
            {
                continue;
            }

            const librpa_int::Vector3_Order<double> q{point.q[0], point.q[1], point.q[2]};
            const auto coulomb_iter =
                std::find_if(coulomb_full_q.cbegin(), coulomb_full_q.cend(),
                             [&q](const auto &member)
                             { return librpa_int::same_fractional_kpoint(member.q, q, 1.0e-8); });
            if (coulomb_iter == coulomb_full_q.cend())
            {
                throw std::runtime_error(
                    "Cannot map the Sternheimer response q point to an IBZ Coulomb matrix");
            }
            const auto &coulomb = coulomb_iter->matrix;
            std::vector<SternheimerQStarRouteRecord> point_qstar_routes;
            std::copy_if(
                qstar_routes.cbegin(), qstar_routes.cend(), std::back_inserter(point_qstar_routes),
                [&point](const auto &route) { return route.representative_iq == point.iq; });
            const auto coulomb_qstar = qstar_routes.empty()
                                           ? librpa_int::reconstruct_sternheimer_qstar_responses(
                                                 symmetry, layouts, atom_nabf, q, coulomb, lmax)
                                           : build_sternheimer_qstar_responses_from_routes(
                                                 symmetry, layouts, atom_nabf, full_qpoints,
                                                 point.iq, q, coulomb, point_qstar_routes, lmax);

            QResult qresult;
            qresult.point = point;
            qresult.frequencies.reserve(static_cast<std::size_t>(driver::opts.nfreq));
            qresult.symmetry_audits.reserve(static_cast<std::size_t>(driver::opts.nfreq));
            for (const auto &response : reconstructed)
            {
                if (response.iq != point.iq)
                {
                    continue;
                }
                const bool gamma_headwing = replace_gamma_headwing && is_rpa_gamma_point(point.q);
                const auto headwing = gamma_headwing ? headwing_for_frequency(response.ifreq)
                                                     : librpa_int::SternheimerRpaHeadwingInput{};
                const auto audit = compute_sternheimer_qstar_rpa_frequency(
                    response, coulomb_qstar, driver::opts.sqrt_coulomb_threshold,
                    gamma_headwing ? &headwing : nullptr);
                if (write_reconstructed && mpi_comm_global_h.is_root())
                {
                    const auto &metadata = groups.at({point.iq, response.ifreq});
                    for (const auto &member : response.qstar_responses)
                    {
                        const auto folded = librpa_int::fold_fractional_kpoint_to_targets(
                            member.q, full_qpoints, 1.0e-8);
                        SternheimerChi0V1Matrix output;
                        output.iq = folded.target_k_index + 1;
                        output.ifreq = response.ifreq;
                        output.omega = response.omega;
                        output.weight = response.weight;
                        output.atom_naux = metadata.atom_naux;
                        output.matrix = member.matrix;
                        const std::string path = driver_params.prefix_sternheimer_reconstructed +
                                                 std::to_string(output.iq) + "_ifreq_" +
                                                 std::to_string(output.ifreq) + ".dat";
                        write_sternheimer_chi0_v1_matrix_file(path, output);
                    }
                }
                if (write_kresolved && mpi_comm_global_h.is_root())
                {
                    const auto &metadata = groups.at({point.iq, response.ifreq});
                    for (const auto &member : response.kresolved_responses)
                    {
                        SternheimerChi0V1Matrix output;
                        output.iq = point.iq;
                        output.ifreq = response.ifreq;
                        output.omega = response.omega;
                        output.weight = response.weight;
                        output.atom_naux = metadata.atom_naux;
                        output.matrix = member.matrix;
                        const std::string path = driver_params.prefix_sternheimer_kresolved +
                                                 std::to_string(output.iq) + "_ik_" +
                                                 std::to_string(member.ik_full) + "_ifreq_" +
                                                 std::to_string(output.ifreq) + ".dat";
                        write_sternheimer_chi0_v1_matrix_file(path, output);
                    }
                }
                double matrix_scale = 0.0;
                double hermiticity_residual = 0.0;
                std::complex<double> response_trace(0.0, 0.0);
                for (int row = 0; row != response.matrix.nr; ++row)
                {
                    response_trace += response.matrix(row, row);
                    for (int column = 0; column != response.matrix.nc; ++column)
                    {
                        matrix_scale =
                            std::max(matrix_scale, std::abs(response.matrix(row, column)));
                        hermiticity_residual =
                            std::max(hermiticity_residual,
                                     std::abs(response.matrix(row, column) -
                                              std::conj(response.matrix(column, row))));
                    }
                }
                qresult.frequencies.push_back(audit.result);
                qresult.symmetry_audits.push_back(
                    {response.ifreq, response.full_k_count, response.representative_k_count,
                     response.little_group_order, audit.qstar_size,
                     hermiticity_residual / std::max(1.0, matrix_scale),
                     audit.max_integrand_difference, response_trace});
            }
            if (qresult.frequencies.size() != static_cast<std::size_t>(driver::opts.nfreq))
            {
                throw std::runtime_error(
                    "Sternheimer partial task did not reconstruct every frequency for iq=" +
                    std::to_string(point.iq));
            }
            qresult.energy = librpa_int::sum_sternheimer_rpa_energies(qresult.frequencies);
            qresults.push_back(std::move(qresult));
        }
    }
    else
    {
        for (const auto &point : qpoints)
        {
            if (!include_gamma_in_rpa && is_rpa_gamma_point(point.q))
            {
                continue;
            }
            auto coulomb = read_coulomb_v1_full_matrix(driver_params.input_dir,
                                                       driver_params.prefix_coul_full, point.iq);
            std::vector<SternheimerChi0V1Matrix> responses;
            const auto cached = response_cache.find(point.iq);
            if (cached != response_cache.end())
            {
                responses = std::move(cached->second);
                response_cache.erase(cached);
            }
            else
            {
                responses = read_sternheimer_chi0_v1_matrices(
                    driver_params.input_dir, driver_params.prefix_sternheimer_chi0, point.iq);
            }
            QResult qresult;
            qresult.point = point;
            qresult.frequencies.reserve(responses.size());
            for (const auto &response : responses)
            {
                if (replace_gamma_headwing && is_rpa_gamma_point(point.q))
                {
                    const auto headwing = headwing_for_frequency(response.ifreq);
                    qresult.frequencies.push_back(
                        librpa_int::compute_sternheimer_rpa_frequency_headwing(
                            coulomb, response.matrix, headwing, response.ifreq, response.omega,
                            response.weight, point.weight, driver::opts.sqrt_coulomb_threshold));
                }
                else
                {
                    qresult.frequencies.push_back(compute_sternheimer_rpa_frequency(
                        coulomb, response.matrix, response.ifreq, response.omega, response.weight,
                        point.weight, driver::opts.sqrt_coulomb_threshold));
                }
            }
            qresult.energy = librpa_int::sum_sternheimer_rpa_energies(qresult.frequencies);
            qresults.push_back(std::move(qresult));
        }
    }
    std::vector<librpa_int::SternheimerRpaFrequencyResult> all_frequencies;
    for (const auto &qresult : qresults)
    {
        all_frequencies.insert(all_frequencies.end(), qresult.frequencies.begin(),
                               qresult.frequencies.end());
    }
    const auto total_energy = librpa_int::sum_sternheimer_rpa_energies(all_frequencies);

    mpi_comm_global_h.barrier();
    if (mpi_comm_global_h.is_root())
    {
        lib_printf("Sternheimer RPA correlation energy (Hartree)\n");
        lib_printf("| q mode = %s\n", partial_mode
                                          ? "partial-symmetry"
                                          : (manifest_path.empty() ? "single" : "manifest"));
        if (!manifest_path.empty())
        {
            lib_printf("| q manifest = %s\n", manifest_path.c_str());
        }
        if (partial_mode)
        {
            lib_printf("| partial-response manifest = %s\n", partial_manifest_path.c_str());
            if (!route_manifest_path.empty())
            {
                lib_printf("| fixed-q route manifest = %s\n", route_manifest_path.c_str());
            }
            if (write_reconstructed)
            {
                lib_printf("| reconstructed Sternheimer prefix = %s\n",
                           driver_params.prefix_sternheimer_reconstructed.c_str());
            }
            if (write_kresolved)
            {
                lib_printf("| k-resolved Sternheimer prefix = %s\n",
                           driver_params.prefix_sternheimer_kresolved.c_str());
            }
            if (write_symmetry_diagnostic)
            {
                lib_printf("| fixed-q symmetry diagnostic prefix = %s\n",
                           driver_params.prefix_sternheimer_symmetry_diagnostic.c_str());
            }
        }
        lib_printf("| Coulomb prefix = %s\n", driver_params.prefix_coul_full.c_str());
        if (!partial_mode)
        {
            lib_printf("| Sternheimer chi0 prefix = %s\n",
                       driver_params.prefix_sternheimer_chi0.c_str());
        }
        lib_printf("| use_rpa_gamma = %s\n", driver_params.use_rpa_gamma ? "true" : "false");
        lib_printf("| analytic head/wing = %s\n", replace_gamma_headwing ? "on" : "off");
        if (replace_gamma_headwing)
        {
            lib_printf("| RPA head/wing mode = %s\n", headwing_mode.c_str());
        }
        if (!include_gamma_in_rpa)
        {
            for (const auto &point : qpoints)
            {
                if (is_rpa_gamma_point(point.q))
                {
                    lib_printf("| excluded Gamma iq = %d, q weight = %.16e\n", point.iq,
                               point.weight);
                }
            }
        }
        for (const auto &qresult : qresults)
        {
            lib_printf("| iq = %d\n", qresult.point.iq);
            if (!manifest_path.empty())
            {
                lib_printf("| q = %.16e %.16e %.16e\n", qresult.point.q[0], qresult.point.q[1],
                           qresult.point.q[2]);
            }
            const double q_weight = qresult.frequencies.empty()
                                        ? qresult.point.weight
                                        : qresult.frequencies.front().qweight;
            lib_printf("| q weight = %.16e\n", q_weight);
            if (partial_mode)
            {
                lib_printf("| q manifest weight = %.16e\n", qresult.point.weight);
            }
            lib_printf(
                "| ifreq omega_Ha weight_Ha integrand_real integrand_imag Ec_real Ec_imag\n");
            for (const auto &result : qresult.frequencies)
            {
                lib_printf("| %5d %20.12e %20.12e %20.12e %20.12e %20.12e %20.12e\n", result.ifreq,
                           result.omega, result.weight, result.integrand.real(),
                           result.integrand.imag(), result.energy.real(), result.energy.imag());
            }
            for (const auto &audit : qresult.symmetry_audits)
            {
                lib_printf(
                    "| symmetry audit ifreq=%d full_k=%d representative_k=%d "
                    "little_group=%d qstar=%d hermiticity_relative_residual=%.12e "
                    "response_trace_real=%.12e response_trace_imag=%.12e "
                    "max_integrand_difference=%.12e\n",
                    audit.ifreq, audit.full_k_count, audit.representative_k_count,
                    audit.little_group_order, audit.qstar_size, audit.hermiticity_relative_residual,
                    audit.response_trace.real(), audit.response_trace.imag(),
                    audit.max_integrand_difference);
            }
            lib_printf("| q Sternheimer EcRPA: %20.12e %20.12e\n", qresult.energy.real(),
                       qresult.energy.imag());
        }
        if (!include_gamma_in_rpa)
        {
            lib_printf("| Total Sternheimer EcRPA excluding q=0: %20.12e %20.12e\n",
                       total_energy.real(), total_energy.imag());
        }
        lib_printf("| Total Sternheimer EcRPA: %20.12e %20.12e\n", total_energy.real(),
                   total_energy.imag());
        if (std::abs(total_energy.imag()) > 1.0e-6)
        {
            lib_printf_root(LIBRPA_VERBOSE_WARN,
                            "Warning: sizable imaginary part in Sternheimer EcRPA = %.12e\n",
                            total_energy.imag());
        }
    }
    mpi_comm_global_h.barrier();
}
