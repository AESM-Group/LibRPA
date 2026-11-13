/*!
 * Common (small) helper functions to initialize dataset components from parsed options.
 */
// Public API headers
#include "librpa_enums.h"
#include "librpa_options.h"

#include <memory>

// Internal headers
#include "../core/dielecmodel.h"
#include "dataset.h"

namespace librpa_int
{

void initialize_ds_tfgrids(Dataset &ds, const LibrpaOptions &opts);

// Initialize the global CPU or GPU owner selected by the options. Owners are
// cached independently so subsequent calls can switch between CPU and GPU work.
void initialize_ds_global_ddla(Dataset &ds, const LibrpaOptions &opts);

void initialize_ds_atpairs_local(Dataset &ds, LibrpaParallelRouting routing);

// Initialize response function component
void initialize_ds_chi0(Dataset &ds, const LibrpaOptions &opts);
void initialize_ds_chi0_qstar(Dataset &ds, const LibrpaOptions &opts);

// Initialize EXX component
void initialize_ds_exx(Dataset &ds, const LibrpaOptions &opts);

// Resolve whether the EXX stage uses the symmetry restore, rejecting
// configuration errors: exchange from a symmetry-reduced SCF k-grid without
// star restore is inexact (weighted-IBZ density matrix plus the
// inversion-only Coulomb star expansion), so a reduced grid with
// use_symmetry_exx off throws instead of computing wrong numbers silently.
bool resolve_use_symmetry_exx(const LibrpaOptions &opts, const PeriodicBoundaryData &pbc);

// Initialize G0W0 component
void initialize_ds_g0w0(Dataset &ds, const LibrpaOptions &opts);

// Build a fresh analytic head/wing from the Dataset, optionally excluding
// weighted band pairs. Does not change Dataset::p_headwing or its output snapshots.
// The returned object borrows Dataset inputs, which must outlive it.
std::unique_ptr<diele_func> make_analytic_headwing(Dataset &ds, const LibrpaOptions &opts,
                                                   bool need_wing,
                                                   const HeadwingBandSelection *excluded = nullptr);

// Initialize/reuse the ordinary GW head/wing cache and its output snapshots.
void initialize_ds_headwing(Dataset &ds, const LibrpaOptions &opts, bool need_wing);

void initialize_symmetry_context(Dataset &ds, bool build_shell_rotations);
void require_symmetry_shell_layouts(const Dataset &ds, const char *calculation);
void reject_spinor_symmetry_speedup(const Dataset &ds, const char *calculation);

}
