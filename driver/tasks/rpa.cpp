#include <array>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "../../src/io/global_io.h"
#include "../../src/mpi/global_mpi.h"
#include "../driver.h"
#include "../read_data.h"
#include "../rpa_qsum.h"
#include "../task.h"
#include "librpa.hpp"

// #include "../../src/io/stl_io_helper.h"

void driver::task_rpa()
{
    using namespace librpa_int;
    using librpa_int::global::lib_printf;
    using librpa_int::global::mpi_comm_global_h;

    // Using public API.
    // std::vector<double> temp_corr(2);
    // std::vector<double> temp_corr_irk(2 * n_irk_points);
    // get_rpa_correlation_energy(temp_corr.data(), temp_corr_irk.data());
    // std::complex<double> corr(temp_corr[0], temp_corr[1]);
    // auto dp = reinterpret_cast<std::complex<double>*>(temp_corr_irk.data());
    // std::vector<std::complex<double>> corr_irk(dp, dp + n_irk_points);

    // Using the internal work function, avoid copying between vector and double*
    double corr = 0.0;
    std::vector<std::complex<double>> corr_irk(n_ibz_kpoints);

    const bool compute_headwing =
        driver::get_bool(driver::opts.replace_w_head) &&
        (driver::opts.option_dielect_func == 3 || driver::opts.option_dielect_func == 4);
    const bool head_only_mode =
        compute_headwing && std::string(driver::opts.rpa_headwing_mode) == "head_only";
    if (compute_headwing)
    {
        read_headwing_input(driver_params.input_dir,
                            driver::opts.option_dielect_func == 3 && !head_only_mode);
    }

    corr = h.get_rpa_correlation_energy(driver::opts, corr_irk);

    RpaQTotals qtotals;
    const bool have_q_coordinates = ibz_kpoints.size() == corr_irk.size();
    if (have_q_coordinates)
    {
        std::vector<RpaQContribution> contributions;
        contributions.reserve(corr_irk.size());
        for (std::size_t iq = 0; iq < corr_irk.size(); ++iq)
        {
            const auto &q = ibz_kpoints[iq];
            contributions.push_back({{q.x, q.y, q.z}, corr_irk[iq]});
        }
        qtotals = sum_rpa_q_contributions(contributions);
    }
    else if (!driver_params.use_rpa_gamma)
    {
        throw std::runtime_error(
            "use_rpa_gamma=false requires q coordinates for every RPA contribution");
    }

    const double corr_excluding_gamma = corr - qtotals.gamma.real();
    qtotals.including_gamma = {corr, 0.0};
    qtotals.excluding_gamma = {corr_excluding_gamma, 0.0};
    const double selected_corr = select_rpa_q_total(qtotals, driver_params.use_rpa_gamma).real();

    mpi_comm_global_h.barrier();
    constexpr auto result_output_level = LIBRPA_VERBOSE_CRITICAL;
    if (mpi_comm_global_h.is_root()
        && librpa_int::global::should_output(result_output_level))
    {
        lib_printf(result_output_level, "RPA correlation energy (Hartree)\n");
        lib_printf(result_output_level, "| Weighted contribution from each k:\n");

        const auto old_precision = std::cout.precision();
        std::cout << std::setprecision(15);
        for (int i_irk = 0; i_irk < n_ibz_kpoints; i_irk++)
        {
            if (i_irk < static_cast<int>(ibz_kpoints.size()))
            {
                std::cout << "| " << ibz_kpoints[i_irk] << ": " << corr_irk[i_irk] << std::endl;
            }
            else
            {
                std::cout << "| q" << i_irk + 1 << ": " << corr_irk[i_irk] << std::endl;
            }
        }
        std::cout.precision(old_precision);
        if (have_q_coordinates)
        {
            lib_printf(result_output_level,
                       "| Gamma EcRPA contribution: %20.12e %20.12e\n",
                       qtotals.gamma.real(), qtotals.gamma.imag());
            lib_printf(result_output_level, "| Total EcRPA including q=0: %20.12e\n", corr);
            lib_printf(result_output_level, "| Total EcRPA excluding q=0: %20.12e\n",
                       corr_excluding_gamma);
        }
        lib_printf(result_output_level, "| use_rpa_gamma = %s\n",
                   driver_params.use_rpa_gamma ? "true" : "false");
        lib_printf(result_output_level, "| Total EcRPA: %18.9f\n", selected_corr);
    }
    if (mpi_comm_global_h.is_root())
    {
        for (int i_irk = 0; i_irk < n_ibz_kpoints; i_irk++)
        {
            const auto &im = corr_irk[i_irk].imag();
            if (std::abs(im) > 1.e-3)
                lib_printf(
                    LIBRPA_VERBOSE_WARN,
                    "Warning: considerable imaginary part of EcRPA = %f\n at IBZ k-point %d\n", im,
                    i_irk + 1);
        }
    }
    mpi_comm_global_h.barrier();
}
