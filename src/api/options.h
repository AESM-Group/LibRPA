#pragma once

#include "librpa_options.h"

namespace librpa_int
{
// Resolve backend availability and GPU promotion before initializing handles.
// Returns a copy so C/Fortran callers' input options are not modified.
LibrpaOptions resolve_linear_algebra_options(const LibrpaOptions& input);
}
