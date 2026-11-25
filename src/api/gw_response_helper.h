#pragma once

#include "dataset.h"
#include "../core/qsgw/task_context.h"

namespace librpa_int::api
{
//! Reuse the existing API GW build for a core task, with an optional live-response consumer.
//! export_bare_chi0 enables output_chi0_rf / output_chi0_static for this build. Only the
//! single-shot G0W0 entry point sets it; QSGW and cRPA rebuild the response per update
//! and therefore never export it.
void build_gw_sigma_for_task(Dataset& dataset, const LibrpaOptions& options,
                             const qsgw::ResponseObserver& on_response = {},
                             bool export_bare_chi0 = false);
}  // namespace librpa_int::api
