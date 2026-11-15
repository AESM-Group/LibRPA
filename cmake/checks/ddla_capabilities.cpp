#include <ddla/ddla.h>
#include <complex>
#include <cstddef>

using H = ddla::DdlaHandle_t;
using Z = std::complex<double>;
template <class T>
using LocalLogdet = void (*)(const H&, int, const T*, int, Z&);
template <class T>
using DistributedLogdet = void (*)(const H&, int, const T*, const int*, Z&);
template <class A, class B>
using MixedGemm = void (*)(const H&, char, char, int, int, int, const Z&,
                           const A*, int, const B*, int, const Z&, Z*, int, void*, std::size_t);
template <class A, class B>
using MixedBatch = void (*)(const H&, char, char, int*, int*, int*, Z,
                            const A* const*, int*, const B* const*, int*, Z, Z**, int*, int);
template <class A, class B, class E>
using MixedTwoStage = void (*)(const H&, char, char, int*, int*, int*, Z,
                               const A* const*, int*, const B* const*, int*, Z, Z**, int*,
                               char, char, int*, int*, int*, Z, const E* const*, int*, int*,
                               Z, Z**, int*, bool, int, const int*, int);

int main()
{
    // Volatile pointer stores preserve link references even in optimized builds.
    LocalLogdet<double> volatile local_real = &ddla::diag_logdet<double>;
    LocalLogdet<Z> volatile local_complex = &ddla::diag_logdet<Z>;
    DistributedLogdet<double> volatile distributed_real = &ddla::pdiag_logdet<double>;
    DistributedLogdet<Z> volatile distributed_complex = &ddla::pdiag_logdet<Z>;
    using Query = std::size_t (*)(const H&, int, int, int, bool);
    Query volatile query = &ddla::gemm_mixed_workspace_bytes<double>;
    MixedGemm<double, Z> volatile gemm_rc = &ddla::gemm<double>;
    MixedGemm<Z, double> volatile gemm_cr = &ddla::gemm<double>;
    MixedBatch<double, Z> volatile batch_rc = &ddla::gemmVbatched<double>;
    MixedBatch<Z, double> volatile batch_cr = &ddla::gemmVbatched<double>;
    MixedTwoStage<double, Z, double> volatile two_rc_r = &ddla::gemmVbatched2s<double>;
    MixedTwoStage<Z, double, double> volatile two_cr_r = &ddla::gemmVbatched2s<double>;
    MixedTwoStage<double, Z, Z> volatile two_rc_c = &ddla::gemmVbatched2s<double>;
    MixedTwoStage<Z, double, Z> volatile two_cr_c = &ddla::gemmVbatched2s<double>;
    return !(local_real && local_complex && distributed_real && distributed_complex && query &&
             gemm_rc && gemm_cr && batch_rc && batch_cr && two_rc_r && two_cr_r && two_rc_c && two_cr_c);
}
