#pragma once

#include <cstdlib>

#if defined(__GLIBC__)
#include <malloc.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <malloc/malloc.h>
#endif

namespace RI
{
namespace Global_Func
{
// Ask the allocator to return unused memory without changing live allocations.
// Allocators without a supported release API need no explicit action.
inline void release_free_memory()
{
#if defined(__GLIBC__)
    malloc_trim(0);
#elif defined(__APPLE__) && defined(__MACH__)
    malloc_zone_pressure_relief(malloc_default_zone(), 0);
#endif
}
} // namespace Global_Func
} // namespace RI
