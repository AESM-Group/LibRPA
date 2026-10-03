#pragma once
#include <string>
#include <vector>

#include "../../src/math/vector3_order.h"
#include "../../src/mpi/base_mpi.h"
#include "librpa.hpp"

namespace librpa::reader
{
// Reader settings supplied by the driver or the public file loader.
struct ReaderParameters
{
    std::string prefix_eigvecs_scf = "KS_eigenvector";
    std::string prefix_lri_coeff = "Cs_data";
    std::string prefix_lri_coeff_shrink = "Cs_shrinked_data";
    bool use_spinor_wfc = false;
    bool sternheimer_partial_input = false;
};

// Parsing metadata owned by one input operation; never library-owned data.
struct ReaderState
{
    std::vector<int> atom_types;
    std::size_t n_atoms = 0;
    int n_spins = 0, n_kpoints = 0, n_ibz_kpoints = 0, n_kpoints_band = 0;
    int n_states = 0, n_basis_wfc = 0, n_basis_ao = 0, n_spinor = 1;
    std::vector<std::size_t> nbs_wfc, nbs_aux, nbs_aux_shrink;
    std::vector<int> iks_eigvec_this, iks_band_eigvec_this;
    std::vector<librpa_int::Vector3_Order<double>> ibz_kpoints, kfrac_band;
    bool is_basis_convention_read = false;
    std::string basis_convention_label = "unknown";
};

struct ReaderContext
{
    Handler &h;
    ReaderState &state;
    ReaderParameters params;
    Options opts;
    librpa_int::MpiCommHandler comm;
};

inline bool reader_switch(LibrpaSwitch value) { return value == LIBRPA_SWITCH_ON; }

}  // namespace librpa::reader
