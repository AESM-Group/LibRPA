#include "task.h"

#include <map>
#include <stdexcept>
#include <algorithm>
#include <functional>

namespace driver
{

static std::map<std::string, task_t> map_lowstr_task{
    {"rpa",           task_t::RPA},
    {"sternheimer_rpa", task_t::SternheimerRPA},
    {"st_rpa",        task_t::SternheimerRPA},
    {"g0w0",          task_t::G0W0},
    {"g0w0_band",     task_t::G0W0_band},
    {"exx",           task_t::EXX},
    {"exx_band",      task_t::EXX_band},
    {"wc_rf",         task_t::Wc_Rf},
    {"print_minimax", task_t::print_minimax},
    {"qsgw", task_t::QSGW},
    {"qsgwa", task_t::QSGWA},
    {"qsgw_band", task_t::QSGW_band},
    {"hf_band", task_t::HF_band},
    {"scrpa", task_t::scRPA},
    {"scrpa_band", task_t::scRPA_band},
    {"test",          task_t::test},
};

static std::map<task_t, std::string> map_task_lowstr{
    {task_t::RPA,           "RPA correlation energy"},
    {task_t::SternheimerRPA, "Sternheimer RPA correlation energy from response matrices"},
    {task_t::G0W0,          "One-shot GW for quasi-paricle energies"},
    {task_t::G0W0_band,     "One-shot GW for quasi-paricle energies"},
    {task_t::EXX,           "Non-self-consistent exact-exchange (EXX) calculation"},
    {task_t::EXX_band,      "Non-self-consistent exact-exchange (EXX) calculation for band structure"},
    {task_t::Wc_Rf,         "Real-space frequency-domain screened Coulomb (correlation part)"},
    {task_t::print_minimax, "Printing minimax time-frequency grids"},
    {task_t::QSGW,          "QSGW"},
    {task_t::QSGWA,         "QSGW, mode A"},
    {task_t::QSGW_band,     "QSGW band structure"},
    {task_t::HF_band,       "HF band structure"},
    {task_t::scRPA,         "SCRPA"},
    {task_t::scRPA_band,    "SCRPA band structure"},
    {task_t::test,          "Test task"},
};

task_t get_task(const std::string &task_string)
{
    auto task_lower = task_string;
    transform(task_lower.begin(), task_lower.end(), task_lower.begin(), ::tolower);
    auto it = map_lowstr_task.find(task_lower);
    if (it == map_lowstr_task.cend())
    {
        throw std::runtime_error("Unknown task (" + task_string + "). Please check your input");
    }
    return it->second;
}

std::string get_task_string(const task_t &task)
{
    auto it = map_task_lowstr.find(task);
    if (it == map_task_lowstr.cend())
    {
        throw std::runtime_error("Internal error, task not defined");
    }
    return it->second;
}

static std::map<task_t, std::function<void(void)>> map_task_func_impl{
    {task_t::print_minimax, task_print_minimax},
    {task_t::G0W0, task_g0w0},
    {task_t::G0W0_band, task_g0w0_band},
    {task_t::RPA, task_rpa},
    {task_t::SternheimerRPA, task_sternheimer_rpa},
    {task_t::EXX, task_exx},
    {task_t::EXX_band, task_exx_band},
    {task_t::QSGW, task_qsgw},
    {task_t::QSGW_band, task_qsgw_band},
};

void run_task(const task_t &task)
{
    auto it = map_task_func_impl.find(task);
    if (it == map_task_func_impl.cend())
    {
        throw std::runtime_error("Internal error, requested task not implemented");
    }
    it->second();
}

}
