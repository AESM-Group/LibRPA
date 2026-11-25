#include "profiler.h"

#include <ctime>
#include <memory>
#include <ostream>
#include <string>
#include <cstring>
// #include <iostream>
#include <omp.h>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <stdexcept>

#include "../io/global_io.h"
#ifdef LIBRPA_VERBOSE
#include "utils_mem.h"
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#ifdef LIBRPA_USE_CUDA
#include <cuda_runtime_api.h>
#else
#include <hip/hip_runtime_api.h>
#endif
#endif
#endif

namespace librpa_int {

#if defined(LIBRPA_VERBOSE) && (defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP))
namespace {
size_t get_free_device_memory()
{
    size_t free_bytes = 0, total_bytes = 0;
#ifdef LIBRPA_USE_CUDA
    const auto status = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
#else
    const auto status = hipMemGetInfo(&free_bytes, &total_bytes);
    if (status != hipSuccess)
        throw std::runtime_error(hipGetErrorString(status));
#endif
    return free_bytes;
}
} // namespace
#endif

double cpu_time_from_clocks_diff(const std::clock_t& ct_start,
                                 const std::clock_t& ct_end)
{
    return double(ct_end - ct_start) / CLOCKS_PER_SEC;
}

std::string get_timestamp()
{
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::stringstream ss;
    ss << "[" <<std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S")
       << '.' << std::setfill('0') << std::setw(3) << milliseconds.count() << "]";
    return ss.str();
}

void Profiler::Timer::start() noexcept
{
    if(is_on())
        stop();
    ncalls++;
    clock_start = clock();
    wt_start = omp_get_wtime();
    cpu_time_last = 0.0;
    wall_time_last = 0.0;
    // librpa_int::global::lib_printf("start: %zu %zu %f\n", ncalls, clock_start, wt_start);
}

void Profiler::Timer::stop() noexcept
{
    if(!is_on()) return;
    cpu_time_accu += (cpu_time_last = cpu_time_from_clocks_diff(clock_start, clock()));
    wall_time_accu += (wall_time_last = omp_get_wtime() - wt_start);
    // librpa_int::global::lib_printf("stop: %f %f %f\n", wt_start, wall_time, cpu_time);
    wt_start = 0.;
    clock_start = 0;
}

std::shared_ptr<Profiler::Timer> Profiler::add(const std::string &tname,
                                               const std::string &tnote,
                                               LibrpaVerbose verbose_level) noexcept
{
    auto new_timer = std::make_shared<Timer>(tname, tnote, verbose_level);

    if (!root)
    {
        root = new_timer;
    }
    else
    {
        std::shared_ptr<Timer> sibling;
        if (current)
        {
            if (current->child)
            {
                sibling = current->child;
                while (sibling->next) sibling = sibling->next;
                sibling->next = new_timer;
                new_timer->prev = sibling;
            }
            else
            {
                current->child = new_timer;
            }
        }
        else
        {
            // root exists while current is null => sibling of root
            sibling = root;
            while (sibling->next) sibling = sibling->next;
            sibling->next = new_timer;
            new_timer->prev = sibling;
        }
        new_timer->parent = current;
    }

    return new_timer;
}

void Profiler::start(const std::string &tname, const std::string &tnote,
                     LibrpaVerbose verbose) noexcept
{
    if (omp_get_thread_num() != 0) return;
    auto timer = find_timer_in_hierarchy(tname);
    // create a new timer if it is not found, otherwise we move to that timer and start
    if (!timer)
    {
        timer = add(tname, tnote, verbose);
    }
    timer->verbose_level = verbose;
    // move to it
    current = timer;
#ifdef LIBRPA_VERBOSE
    if (global::should_output(verbose))
    {
        double free_mem_gb;
        get_node_free_mem(free_mem_gb);
        global::ofs_myid << get_timestamp() <<" Timer start: " << tname << ". "
                         << "Free memory on node [GB]: " << free_mem_gb;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
        const auto free_mem_gpu_bt = get_free_device_memory();
        global::ofs_myid << "  on GPU [GB]: " << free_mem_gpu_bt/1024./1024./1024.;
#endif
        global::ofs_myid << std::endl;
    }
#endif
    current->start();
}

void Profiler::stop(const std::string &tname) noexcept
{
    if (omp_get_thread_num() != 0) return;
    if (current)
    {
        // Check if the current timer matches the given timer name
        if (current->name == tname)
        {
#ifdef LIBRPA_VERBOSE
            if (global::should_output(current->verbose_level))
            {
                double free_mem_gb;
                get_node_free_mem(free_mem_gb);
                global::ofs_myid << get_timestamp() << " Timer stop:  " << tname << ". "
                                 << "Free memory on node [GB]: " << free_mem_gb;
#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
                const auto free_mem_gpu_bt = get_free_device_memory();
                global::ofs_myid << "  on GPU [GB]: " << free_mem_gpu_bt/1024./1024./1024.;
#endif
                global::ofs_myid << std::endl;
            }
#endif
            current->stop();
            current = current->parent;
        }
        else
        {
             global::ofs_myid << "Warning: Attempting to stop timer '" << tname
                              << "' but current active timer is '" << current->name << "'" << std::endl;
        }
    }
    else
    {
        global::ofs_myid << "Warning: No timer is currently active" << std::endl;
    }
}

void Profiler::terminate() noexcept
{
    if (omp_get_thread_num() != 0) return;
    while (current)
    {
        const auto tname = current->name;
        stop(tname);
    }
}

std::shared_ptr<Profiler::Timer> Profiler::find_timer_in_hierarchy(const std::string &tname)
{
    if (current)
        return search_timer_in_hierarchy(current, tname);
    return search_timer_in_hierarchy(root, tname);
}

std::shared_ptr<Profiler::Timer> Profiler::search_timer_in_hierarchy(std::shared_ptr<Timer> timer,
                                                                     const std::string &tname)
{
    if (!timer) return nullptr;
    if (timer->name == tname) return timer;

    // Recursively check child timers
    if (timer->child)
    {
        auto found_timer = search_timer_in_hierarchy(timer->child, tname);
        if (found_timer)
        {
            return found_timer;
        }
    }

    // Check the next sibling timer
    if (timer->next)
    {
        return search_timer_in_hierarchy(timer->next, tname);
    }

    return nullptr;
}

double Profiler::get_cpu_time_last(const std::string &tname) noexcept
{
    auto timer = this->find_timer_in_hierarchy(tname);
    if (timer)
        return timer->get_cpu_time_last();
    return -1.0;
}

double Profiler::get_wall_time_last(const std::string &tname) noexcept
{
    auto timer = this->find_timer_in_hierarchy(tname);
    if (timer)
        return timer->get_wall_time_last();
    return 0.0;
}

std::string Profiler::get_profile_string_of_timer(std::shared_ptr<Profiler::Timer> timer, int level,
                                                  LibrpaVerbose verbose_level)
{
    std::ostringstream ss;
    if (timer->verbose_level > verbose_level)
    {
        if (timer->child)
            ss << get_profile_string_of_timer(timer->child, level, verbose_level);
        if (timer->next)
            ss << get_profile_string_of_timer(timer->next, level, verbose_level);
        return ss.str();
    }

    // std::string indent(2 * level, ' ');
    std::string indent(level, ' ');
    const auto note = indent + (timer->note == "" ? timer->name : timer->note);
    std::ostringstream cstr_cputime, cstr_walltime;
    cstr_cputime << std::fixed << std::setprecision(4) << timer->get_cpu_time();
    cstr_walltime << std::fixed << std::setprecision(4) << timer->get_wall_time();

    // Print self
    ss << std::left;
    ss << std::setw(49) << note << " " << std::setw(12) << timer->ncalls << " "
        << std::setw(18) << (indent + cstr_cputime.str()) << " "
        << std::setw(18) << (indent + cstr_walltime.str()) << "\n";
    // Print child, then sibling
    if (timer->child)
        ss << get_profile_string_of_timer(timer->child, level + 1, verbose_level);
    if (timer->next)
        ss << get_profile_string_of_timer(timer->next, level, verbose_level);
    return ss.str();
}

static std::string banner(char c, int n)
{
    std::string s = "";
    while(n--) s += c;
    return s;
}

void Profiler::display(int verbose) noexcept
{
    // Leading critical marker ensures that this message is always printed.
    global::lib_printf(LIBRPA_VERBOSE_CRITICAL, "%s", this->get_profile_string(verbose).c_str());
}

std::string Profiler::get_profile_string(int verbose) noexcept
{
    std::ostringstream output;
    output << std::left;

    output << std::setw(49) << "Entry" << " " << std::setw(12) << "#calls" << " "
        << std::setw(18) << "CPU time (s)" << " " << std::setw(18) << "Wall time (s)" << "\n";
    output << banner('-', 100) << "\n";
    if (root)
        output << get_profile_string_of_timer(root, 0, verbose);

    return output.str();
}

namespace global
{

Profiler profiler;

}

}
