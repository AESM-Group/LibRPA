/*!
 * Common (small) helper functions to initialize dataset components from parsed options.
 */
// Public API headers
#include "librpa_enums.h"
#include "librpa_options.h"

// Internal headers
#include "dataset.h"

namespace librpa_int
{

void initialize_ds_tfgrids(Dataset &ds, const LibrpaOptions &opts);

// Initialize the global/world BLACS DDLA handle once when GPU ScaLAPACK
// replacement is requested. This binds each MPI rank to its local GPU before
// any k-local BLACS/DDLA handles are created.
void initialize_ds_global_ddla(Dataset &ds, const LibrpaOptions &opts);

void initialize_ds_atpairs_local(Dataset &ds, LibrpaParallelRouting routing);

// Initialize response function component
void initialize_ds_chi0(Dataset &ds, const LibrpaOptions &opts);
void export_ds_chi0(Dataset &ds, const LibrpaOptions &opts, LibrpaParallelRouting routing);

// Initialize EXX component
void initialize_ds_exx(Dataset &ds, const LibrpaOptions &opts);

// Initialize G0W0 component
void initialize_ds_g0w0(Dataset &ds, const LibrpaOptions &opts);

// Initialize analytic dielectric head/wing component from Dataset::velocity_matrix.
void initialize_ds_headwing(Dataset &ds, const LibrpaOptions &opts, bool need_wing);

void initialize_symmetry_context(Dataset &ds, bool build_shell_rotations);
void require_symmetry_shell_layouts(const Dataset &ds, const char *calculation);
void reject_spinor_symmetry_speedup(const Dataset &ds, const char *calculation);

}
