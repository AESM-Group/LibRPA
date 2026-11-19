#pragma once
#include <string>
namespace librpa_int {
class Chi0;
void export_chi0_symmetry(const Chi0 &chi, const std::string &directory, double omega);
}
