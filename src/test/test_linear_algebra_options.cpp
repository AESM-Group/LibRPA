#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>

#include "../api/options.h"

using namespace librpa_int;

int main()
{
    // No device or MPI allocation is needed to resolve the requested policy.
    for (bool gpu : {false, true})
            for (const char* linear : {"scalapack", "ddla"})
                for (const char* eigen : {"scalapack", "ddla", "elpa"})
                {
                    LibrpaOptions input;
                    librpa_init_options(&input);
                    input.use_gpu_replace_scalapack = gpu ? LIBRPA_SWITCH_ON : LIBRPA_SWITCH_OFF;
                    input.use_scalapack_gw_wc = LIBRPA_SWITCH_OFF;
                    std::strcpy(input.linear_algorithm, linear);
                    std::strcpy(input.eigensolver_method, eigen);
                    bool threw = false;
                    try
                    {
                        const auto resolved = resolve_linear_algebra_options(input);
                        const auto again = resolve_linear_algebra_options(resolved);
                        assert(resolved.use_gpu_replace_scalapack == input.use_gpu_replace_scalapack);
                        assert(resolved.use_scalapack_gw_wc == LIBRPA_SWITCH_ON);
                        std::string want_linear = linear, want_eigen = eigen;
#ifndef LIBRPA_USE_DDLA
                        if (want_linear == "ddla") want_linear = "scalapack";
                        if (want_eigen == "ddla") want_eigen = "scalapack";
#endif
#ifndef LIBRPA_USE_ELPA
                        if (want_eigen == "elpa") want_eigen = "scalapack";
#endif
                        if (gpu)
                        {
                            want_linear = "ddla";
                            if (want_eigen == "scalapack")
#ifdef LIBRPA_USE_ELPA
                                want_eigen = "elpa";
#else
                                want_eigen = "ddla";
#endif
                        }
                        assert(resolved.linear_algorithm == want_linear);
                        assert(resolved.eigensolver_method == want_eigen);
                        assert(std::strcmp(again.linear_algorithm, resolved.linear_algorithm) == 0);
                        assert(std::strcmp(again.eigensolver_method, resolved.eigensolver_method) == 0);
                    }
                    catch (const std::runtime_error&)
                    {
                        threw = true;
                    }
#if (defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)) && defined(LIBRPA_USE_DDLA)
                    assert(!threw);
#else
                    assert(threw == gpu);
#endif
                    assert(std::strcmp(input.linear_algorithm, linear) == 0);
                    assert(input.use_scalapack_gw_wc == LIBRPA_SWITCH_OFF);
                    assert(std::strcmp(input.eigensolver_method, eigen) == 0);
                }
}
