#pragma once
#include <string>
namespace librpa_int {
class Chi0;
//! Export the representative chi0(q, i*omega) and the symmetry operators Delta for
//! FULL_CRYSTAL restore mode. chi0_grid_index selects the frequency node of chi.tfg;
//! chi0_source_tag is written to chi0_q.info to record which object was exported.
//! Collective over chi.comm_h: an I/O error on rank 0 is raised on every rank.
void export_chi0_symmetry(const Chi0 &chi, const std::string &directory, int chi0_grid_index,
                          const std::string &chi0_source_tag);
}
