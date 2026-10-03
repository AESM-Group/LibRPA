#pragma once
#include <string>

namespace driver
{

enum class task_t {
    RPA,
    SternheimerRPA,
    EXX,
    EXX_band,
    G0W0,
    G0W0_band,
    Wc_Rf,
    print_minimax,
    QSGW,
    QSGWA,
    QSGW_band,
    HF_band,
    scRPA,
    scRPA_band,
    test,  // a task for convenience of test
};

task_t get_task(const std::string &task_string);

std::string get_task_string(const task_t &task);

void task_rpa();
void task_sternheimer_rpa();
void task_g0w0();
void task_g0w0_band();
void task_exx();
void task_exx_band();
void task_screened_coulomb_real_freq();
void task_print_minimax();
void task_qsgw();
void task_qsgw_band();
void task_test();

void run_task(const task_t &task);
}
