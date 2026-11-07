#include "global_io.h"

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>

#include "../utils/error.h"

namespace librpa_int
{

namespace global
{

std::ofstream ofs_myid;
FILE *pfile_redirect = nullptr;
LibrpaVerbose output_level = LIBRPA_VERBOSE_INFO;

/* ==============================================
 * Private variables starts
 */
static std::ofstream ofs;
static std::streambuf *cout_buf_old = nullptr;
static bool librpa_io_initialized = false;

/*
 * Private variables ends
 ============================================== */

void init_global_io(bool redirect_stdout, const char *redirect_path, bool enable_task_output)
{
    using namespace librpa_int::global;
    if (!is_mpi_initialized())
    {
        throw LIBRPA_RUNTIME_ERROR("init_global_mpi must be called before init_global_io");
    }

    std::string s_fn(redirect_path);
    // Invalid redirect path
    if (redirect_stdout && s_fn == "")
    {
        throw LIBRPA_RUNTIME_ERROR("stdout redirection specified, but redirect path is empty");
    }

    if (enable_task_output)
    {
        std::string fn = "librpa_para_nprocs_" + std::to_string(size_global) +  "_myid_" + std::to_string(myid_global) + ".out";
        ofs_myid.open(fn);
    }

    // Redirect output
    if (redirect_stdout && s_fn != "stdout")
    {
        // Rank 0 touches the output file
        MPI_Barrier(mpi_comm_global);
        if (myid_global == 0)
        {
            std::ofstream ofs(redirect_path);
            ofs.close();
        }
        MPI_Barrier(mpi_comm_global);

        // All process appending to the refreshed buffer
        ofs.open(redirect_path, std::ios::out | std::ios::app);
        if (!ofs.is_open()) throw LIBRPA_RUNTIME_ERROR("failed to redirect stdout to " + s_fn);
        cout_buf_old = std::cout.rdbuf();
        std::cout.rdbuf(ofs.rdbuf());

        // redirecting printf
        pfile_redirect = fopen(redirect_path, "a");
    }

    librpa_io_initialized = true;
}

inline bool is_io_initialized() { return librpa_io_initialized; }

namespace detail
{

void lib_printf_coll_msg_impl(const librpa_int::MpiCommHandler &comm_h,
                              const LibrpaVerbose verbose_level,
                              const char *local_msg,
                              const int count) noexcept
{
    std::vector<int> counts;
    if (comm_h.myid == 0) counts.resize(comm_h.nprocs);
    MPI_Gather(&count, 1, MPI_INT, comm_h.myid == 0 ? counts.data() : nullptr, 1, MPI_INT, 0, comm_h.comm);

    std::vector<int> displs;
    std::vector<char> all_msgs;
    int total_count = 0;
    if (comm_h.myid == 0)
    {
        displs.resize(comm_h.nprocs);
        for (int i = 0; i < comm_h.nprocs; ++i)
        {
            displs[i] = total_count;
            total_count += counts[i];
        }
        all_msgs.resize(total_count);
    }

    MPI_Gatherv(local_msg, count, MPI_CHAR,
                comm_h.myid == 0 ? all_msgs.data() : nullptr,
                comm_h.myid == 0 ? counts.data() : nullptr,
                comm_h.myid == 0 ? displs.data() : nullptr,
                MPI_CHAR, 0, comm_h.comm);

    // Print all messages at once, so that a redirected output is flushed once per call
    if (comm_h.myid == 0 && total_count > 0 && should_output(verbose_level))
        lib_printf(verbose_level, "%.*s", total_count, all_msgs.data());
    comm_h.barrier();
}

} /* end of namespace detail */

void finalize_global_io()
{
    if (ofs_myid.is_open()) ofs_myid.close();

    // Recover original stdout behavior
    if (pfile_redirect != nullptr)
    {
        std::cout.rdbuf(cout_buf_old);
        fclose(pfile_redirect);
        pfile_redirect = nullptr;
    }
    librpa_io_initialized = false;
}

} /* end of namespace global */

} /* end of namespace librpa_int */
