#pragma once

#include "crpa.h"
#include "crpa_model.h"
#include "crpa_task_context.h"
#include "librpa_options.h"

namespace librpa_int
{
//! Preserve the caller's resolved backend policy in the numerical workflow.
CrpaContext make_crpa_context(CrpaTaskContext& dataset, const LibrpaOptions& options);

struct CrpaTaskInput
{
    std::vector<std::string> species_labels;
    std::vector<ComplexMatrix> spatial_overlap;
    CrpaWindowOptions window;
    bool retain_raw_model = true;
};

enum class CrpaTaskMode
{
    Standalone,
    SharedGw,
    ResponseOnly
};

//! Owning numerical result and the geometry defining its orbital frame.
struct CrpaTaskResult
{
    CrpaResult values;
    std::vector<std::vector<std::vector<int>>> response_bands, orbital_bands;
    Chi0::BandSelection response_weights;
    CrpaModelSnapshot model;
    std::vector<double> lattice, positions, k_fractional, k_weights;
    std::vector<int> atom_types, ao_counts;
    std::vector<int> shell_offsets{0}, shell_l, basis_convention;
};

struct CrpaTrackingState
{
    int n_aos = 0, n_spinor = 0;
    std::vector<std::vector<ComplexMatrix>> parent_phi;
    std::vector<double> parent_capture;
};

CrpaTrackingState snapshot_crpa_tracking(const CrpaModelSnapshot& model);
std::vector<double> crpa_tracking_energies(const MeanField& meanfield);
void write_crpa_tracking_checkpoint(const std::string& filename, int iteration,
                                    const std::string& definition,
                                    const std::vector<double>& checkpoint_energies,
                                    const CrpaTrackingState& state);
CrpaTrackingState read_crpa_tracking_checkpoint(const std::string& filename, int iteration,
                                                const std::string& definition,
                                                const MeanField& meanfield);

CrpaPreparedModel prepare_crpa_task_model(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                          const CrpaTaskInput& input,
                                          CrpaTaskMode mode = CrpaTaskMode::Standalone);
CrpaTaskResult snapshot_crpa_task_model(CrpaTaskContext& dataset, CrpaPreparedModel& model);
CrpaTaskResult evaluate_crpa_task_from_response(CrpaTaskContext& dataset,
                                                const LibrpaOptions& options,
                                                CrpaPreparedModel& model,
                                                const Chi0QMap& full_response);
CrpaTaskResult compute_crpa_task(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                 const CrpaTaskInput& input);
CrpaTaskResult compute_crpa_u_task(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                   const CrpaTaskInput& input);
CrpaTaskResult prepare_crpa_task(CrpaTaskContext& dataset, const LibrpaOptions& options,
                                 const CrpaTaskInput& input);
}  // namespace librpa_int
