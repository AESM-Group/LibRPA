#include "librpa.hpp"

#include "driver.h"
#include "read_data.h"
#include "inputfile.h"
#include "task.h"

#include <mpi.h>
#include <omp.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

// Internal headers, used here only for printing formation and some consistency check
// May move to public API later
#include "../src/utils/profiler.h"
#include "../src/utils/utils_mem.h"
#include "../src/io/fs.h"
// #include "task_qsgw.h"
// #include "task_qsgwA.h"
// #include "task_qsgw_band.h"
// #include "task_hf_band.h"
// #include "task_scRPA.h"
// #include "task_scRPA_band.h"

using namespace driver;
using namespace librpa_int;
using namespace librpa_int::global;

// Default thread support level.
// Detailed control is set in project CMakeLists.txt
#ifndef LIBRPA_MPI_THREAD_LEVEL
#define LIBRPA_MPI_THREAD_LEVEL MPI_THREAD_MULTIPLE
#endif

static const char *mpi_thread_level_name(const int level)
{
    switch (level)
    {
        case MPI_THREAD_SINGLE: return "MPI_THREAD_SINGLE";
        case MPI_THREAD_FUNNELED: return "MPI_THREAD_FUNNELED";
        case MPI_THREAD_SERIALIZED: return "MPI_THREAD_SERIALIZED";
        case MPI_THREAD_MULTIPLE: return "MPI_THREAD_MULTIPLE";
        default: return "unknown";
    }
}

static void initialize_mpi_env(int argc, char **argv)
{
    // MPI Initialization
    int provided;
    MPI_Init_thread(&argc, &argv, LIBRPA_MPI_THREAD_LEVEL, &provided);
    if (provided < LIBRPA_MPI_THREAD_LEVEL)
    {
        throw std::runtime_error("Error: MPI_Init_thread provide " + std::to_string(provided) +
                                 " < required " + std::to_string(LIBRPA_MPI_THREAD_LEVEL));
    }
}

static void initialize_librpa(bool &initialized)
{
    librpa::set_output_level(driver::driver_params.output_level);
    librpa::init_global(LIBRPA_SWITCH_OFF);
    initialized = true;

    // Global profiler begins right after MPI is initialized
    profiler.start("driver_total", "Total for driver");

    // Use critical so that MPI inforation are always printed when not silent
    constexpr auto level = LIBRPA_VERBOSE_CRITICAL;
    lib_printf_root(level, "Total number of tasks    : %5d\n", size_global);
    lib_printf_root(level, "Total number of nodes    : %5d\n", size_inter);
    lib_printf_root(level, "Maximal number of threads: %3d\n", omp_get_max_threads());
    lib_printf_root(level, "MPI thread support level : %s\n",
                    mpi_thread_level_name(LIBRPA_MPI_THREAD_LEVEL));
    mpi_comm_global_h.barrier();
    lib_printf_root(level, "MPI tasks information:\n");
    mpi_comm_global_h.barrier();
    lib_printf_coll(level, "| %s\n", mpi_comm_global_h.str().c_str());
    mpi_comm_global_h.barrier();
    if (mpi_comm_intra_h.is_root())
    {
        lib_printf(level,
                   "| Global ID of master process of node %5d : %5d\n",
                   mpi_comm_inter_h.myid, mpi_comm_global_h.myid);
    }
    mpi_comm_global_h.barrier();

    // Print cmake infomation
    if (mpi_comm_global_h.is_root())
    {
        lib_printf(level, "\n");
        lib_printf(level, "%s", librpa::get_build_info());
        lib_printf(level, "\n");
    }
    mpi_comm_global_h.barrier();

    // Create the handler before parsing any data or computation
    driver::h.init(MPI_COMM_WORLD);
}

static void report_driver_error(const char *message)
{
    int initialized = 0;
    int finalized = 0;
    int rank = 0;
    MPI_Initialized(&initialized);
    if (initialized)
    {
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    if (rank == 0) std::cerr << "Error: " << message << std::endl;
}

static void finalize_mpi()
{
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if (initialized)
    {
        MPI_Finalized(&finalized);
        if (!finalized) MPI_Finalize();
    }
}

static int run_task_and_catch(const driver::task_t &task)
{
    int local_failed = 0;
    try
    {
        run_task(task);
    }
    catch (const std::exception &e)
    {
        local_failed = 1;
        lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error on MPI rank %d: %s\n", mpi_comm_global_h.myid,
                   e.what());
    }
    catch (...)
    {
        local_failed = 1;
        lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error on MPI rank %d: unknown exception\n",
                   mpi_comm_global_h.myid);
    }

    int any_failed = 0;
    mpi_comm_global_h.allreduce(&local_failed, &any_failed, 1, MPI_MAX);

    if (!any_failed)
    {
        mpi_comm_global_h.barrier();
    }

    return any_failed;
}

static void finalize_librpa(bool success)
{
    // Free the memory space
    driver::h.free();
    profiler.stop("driver_total");

    bool is_root = mpi_comm_global_h.myid == 0;
    mpi_comm_global_h.barrier();
    librpa::finalize_global();

    // If outputs have been redirected, they are now restored
    if (is_root)
    {
        librpa::print_profile();
        if (success)
        {
            lib_printf(LIBRPA_VERBOSE_CRITICAL, "libRPA finished successfully\n");
        }
        else
        {
            lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error: libRPA failed\n");
        }
    }
}

int main(int argc, char **argv)
{
    bool librpa_initialized = false;
    try
    {
        // Initialize MPI environment
        initialize_mpi_env(argc, argv);

        // Parse the main input file
        parse_inputfile_to_params(input_filename);
        // Early check of task to fail quickly in case
        task_t task = get_task(driver_params.task);

        // Initialize LibRPA global environment and handler
        initialize_librpa(librpa_initialized);

        // Echo input file runtime options.
        profiler.start("driver_read_params", "Driver Read Input Parameters");
        if (mpi_comm_global_h.is_root())
        {
            lib_printf("===== Begin driver parameters  =====\n");
            lib_printf(driver_params.format().c_str());
            lib_printf("===== End driver parameters    =====\n\n");
            lib_printf("===== Begin control parameters =====\n");
            lib_printf(format_runtime_options(opts).c_str());
            lib_printf("===== End control parameters   =====\n\n");
        }
        create_directories(driver::opts.output_dir, mpi_comm_global_h.myid);
        mpi_comm_global_h.barrier();
        profiler.stop("driver_read_params");

        const string path_stru = driver_params.input_dir + driver_params.fn_stru;
        const string path_bz_sampling = driver_params.input_dir + driver_params.fn_bz_sampling;
        const string path_eigocc_scf = driver_params.input_dir + driver_params.fn_eigocc_scf;

        profiler.start("driver_read_common_input_data", "Driver Read Task-Common Input Data");
        const bool sternheimer_analytic_headwing =
            task == task_t::SternheimerRPA && driver::get_bool(driver::opts.replace_w_head) &&
            (driver::opts.option_dielect_func == 3 || driver::opts.option_dielect_func == 4);
        const bool needs_scf_eigenvalues =
            task != task_t::SternheimerRPA || sternheimer_analytic_headwing;
        const bool needs_standard_meanfield_data =
            task != task_t::print_minimax &&
            (task != task_t::SternheimerRPA || sternheimer_analytic_headwing);
        const bool needs_sternheimer_symmetry_metadata =
            task == task_t::SternheimerRPA && !driver_params.fn_sternheimer_partial_manifest.empty();
        const bool needs_structure_bz_basis =
            needs_standard_meanfield_data || needs_sternheimer_symmetry_metadata;

        if (needs_scf_eigenvalues)
        {
            profiler.start("driver_band_out", "DFT SCF eigenvalues/occupations");
            read_scf_occ_eigenvalues(path_eigocc_scf);
            profiler.stop("driver_band_out");
        }

        if (needs_structure_bz_basis)
        {
            profiler.start("driver_struct", "Structure");
            read_stru(path_stru);
            profiler.stop("driver_struct");
            lib_printf_root("\n");

            profiler.start("driver_bz", "BZ sampling");
            if (!path_exists(path_bz_sampling.c_str()))
            {
                read_bz_sampling_from_stru(path_stru);
            }
            else
            {
                read_bz_sampling(path_bz_sampling);
            }
            lib_printf_root("\n");
            profiler.stop("driver_bz");

            profiler.start("driver_basis", "Basis (wave-function and auxiliary)");
            read_basis_wfc_aux(driver_params.input_dir, driver_params.fn_basis,
                               driver_params.fn_basis_wfc, driver_params.fn_basis_aux);
            lib_printf_root("\n");
            profiler.stop("driver_basis");

            if (needs_standard_meanfield_data)
            {
                // Direct k-BLACS eigenvector input is a LibRI path.  Resolve AUTO before
                // reading wave functions so the reader can choose the final ownership
                // layout instead of first materializing dense matrices on k-group roots.
                if (driver::opts.parallel_routing == LIBRPA_ROUTING_AUTO)
                {
                    driver::opts.parallel_routing = decide_auto_routing(
                        driver::n_atoms, driver::opts.nfreq * driver::n_kpoints);
                }

                profiler.start("driver_read_eigenvector", "SCF eigenvectors");
                int ret_eigenvec = read_eigenvector(driver_params.input_dir);
                mpi_comm_global_h.barrier();
                if (ret_eigenvec == 0)
                {
                    lib_printf_root("Successfully read eigenvector files\n");
                }
                else
                {
                    if (ret_eigenvec > 0)
                    {
                        throw LIBRPA_RUNTIME_ERROR(
                            "Error in reading eigenvector files (return code " +
                            std::to_string(ret_eigenvec) + ")");
                    }
                    throw LIBRPA_RUNTIME_ERROR(
                        "No eigenvector files found; check the KS_eigenvector input files");
                }
                profiler.stop("driver_read_eigenvector");

                profiler.start("driver_read_ri");
                read_ri(driver_params.input_dir, driver::opts.parallel_routing);
                lib_printf_root("Actual parallel routing used: %s\n",
                                get_routing_string(driver::opts.parallel_routing).c_str());
                profiler.stop("driver_read_ri");

                // Vq distributed using the same strategy
                // There should be no duplicate for V
            }
        }

        mpi_comm_global_h.barrier();
        if (mpi_comm_intra_h.myid == 0)
        {
            if (mpi_comm_global_h.myid == 0)
            {
                const auto cputime =
                    profiler.get_cpu_time_last("driver_read_common_input_data") / 60.0;
                const auto walltime =
                    profiler.get_wall_time_last("driver_read_common_input_data") / 60.0;
                lib_printf("Initialization finished, Wall/CPU time [min]: %12.4f %12.4f\n",
                           walltime, cputime);
            }
            double freemem;
            auto flag = get_node_free_mem(freemem);
            if (flag == 0)
            {
                lib_printf("Free memory on node %5d [GB]: %8.3f\n", mpi_comm_inter_h.myid, freemem);
            }
        }
        lib_printf_root("Common data parsed, task %s will begin\n", driver_params.task.c_str());
        lib_printf_root("%s: %s\n", driver_params.task.c_str(), get_task_string(task).c_str());
        mpi_comm_global_h.barrier();
        profiler.stop("driver_read_common_input_data");

        int any_failed = run_task_and_catch(task);
        librpa_initialized = false;
        finalize_librpa(!any_failed);
        finalize_mpi();

        return any_failed ? EXIT_FAILURE : EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        report_driver_error(e.what());
    }
    catch (...)
    {
        report_driver_error("unknown exception");
    }

    if (librpa_initialized)
    {
        librpa_initialized = false;
        try
        {
            finalize_librpa(false);
        }
        catch (const std::exception &e)
        {
            report_driver_error(e.what());
        }
        catch (...)
        {
            report_driver_error("unknown exception while finalizing LibRPA");
        }
    }
    finalize_mpi();
    return EXIT_FAILURE;
}
