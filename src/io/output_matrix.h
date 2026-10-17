#pragma once

#include <fstream>
#include <optional>
#include <string>
#include <type_traits>

#include "../core/atomic_basis.h"
#include "../math/matrix_m.h"
#include "../math/vector3_order.h"
#include "../mpi/base_blacs.h"

namespace librpa_int
{

//! Export one set of complete real-space AO blocks, without threshold filtering.
//! Native fieldwise layout: optional double frequency, size_t block count, int dtype,
//! then records of int[7] (R1,R2,R3,I,J,rows,columns) and size_t absolute payload offset.
//! Row-major payloads follow the complete index table. Types 0/1 retain the driver
//! double/complex<double> flags; 2/3 denote float/complex<float>.
//! Returns false on I/O failure; callers handle MPI error propagation.
template <typename T>
bool write_rspace_matrices_binary(const ap_p_map<std::map<Vector3_Order<int>, matrix_m<T>>> &blocks,
                                  const AtomicBasis &basis, const std::string &fn,
                                  const std::optional<double> omega = std::nullopt)
{
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, cplxdb> ||
                      std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>,
                  "unsupported real-space matrix output type");
    constexpr int dtype = std::is_same_v<T, double>   ? 0
                          : std::is_same_v<T, cplxdb> ? 1
                          : std::is_same_v<T, float>  ? 2
                                                      : 3;
    std::ofstream ofs(fn, std::ios::binary);
    if (!ofs) return false;
    if (omega) ofs.write(reinterpret_cast<const char *>(&*omega), sizeof(double));

    std::size_t n_blocks = 0;
    for (const auto &[IJ, R_blocks] : blocks) n_blocks += R_blocks.size();
    ofs.write(reinterpret_cast<const char *>(&n_blocks), sizeof(n_blocks));
    ofs.write(reinterpret_cast<const char *>(&dtype), sizeof(dtype));
    std::size_t offset = (omega ? sizeof(double) : 0) + sizeof(n_blocks) + sizeof(dtype) +
                         n_blocks * (7 * sizeof(int) + sizeof(std::size_t));
    for (const auto &[IJ, R_blocks] : blocks)
        for (const auto &[R, mat] : R_blocks)
        {
            assert(mat.nr() == as_int(basis.get_atom_nb(IJ.first)) &&
                   mat.nc() == as_int(basis.get_atom_nb(IJ.second)));
            assert(mat.major() == MAJOR::ROW);
            const int dims[7] = {R.x,      R.y,     R.z, as_int(IJ.first), as_int(IJ.second),
                                 mat.nr(), mat.nc()};
            ofs.write(reinterpret_cast<const char *>(dims), sizeof(dims));
            ofs.write(reinterpret_cast<const char *>(&offset), sizeof(offset));
            offset += mat.size() * sizeof(T);
        }
    // Two map traversals keep both the index and payload writes sequential.
    for (const auto &[IJ, R_blocks] : blocks)
        for (const auto &[R, mat] : R_blocks)
        {
            ofs.write(reinterpret_cast<const char *>(mat.ptr()),
                      static_cast<std::streamsize>(mat.size() * sizeof(T)));
            if (!ofs) return false;
        }
    ofs.close();
    return ofs.good();
}

std::string rspace_matrix_filename(const std::string &dir, const std::string &prefix, int ispin,
                                   int bra, int ket, int myid, int iomega = -1);

//! Collectively export a square complex matrix, or the same half-open index
//! range in both dimensions. A negative index_end selects all remaining indices.
//! Local blocks must use column-major ScaLAPACK storage. The descriptor's source
//! rank writes the file; I/O failure is reported on every rank in desc.comm().
//! Format (native byte order): int32 dimension, int32 sizeof(double), then
//! row-major pairs of real/imaginary doubles, with no threshold filtering.
void write_matrix_binary_parallel(const Matz &mat_loc, const ArrayDesc &desc, const std::string &fn,
                                  int index_start = 0, int index_end = -1);

}  // namespace librpa_int
