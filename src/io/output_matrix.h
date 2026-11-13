#pragma once

#include <mpi.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <optional>
#include <string>
#include <type_traits>

#include "../core/atomic_basis.h"
#include "../math/matrix_m.h"
#include "../math/vector3_order.h"
#include "../mpi/base_blacs.h"

namespace librpa_int
{

//! Kind of a real-space block file; stored in the file header.
enum class RspaceKind : std::int32_t
{
    ExxR = 0,    //!< exact-exchange blocks, no frequency
    SigcRF = 1,  //!< correlation self-energy blocks at one imaginary frequency
};

//! Magic bytes, byte-order marker and layout version of a real-space block file.
inline constexpr char RSPACE_MAGIC[8] = {'L', 'I', 'B', 'R', 'P', 'A', 'I', 'J'};
inline constexpr std::uint32_t RSPACE_ENDIAN_MARKER = 0x01020304;
inline constexpr std::int32_t RSPACE_VERSION = 1;

//! Header of one real-space block file, written by one MPI rank of one output event.
//! Fieldwise native layout; every 8-byte field is 8-byte aligned:
//!   char[8] magic, uint32 endian marker, int32 version, int32 kind, int32 nprocs,
//!   int32 myid, int32 dtype, uint64 run_id, uint64 n_blocks, [double omega, SigcRF only].
//! The header is 48 bytes (56 with omega). nprocs and myid describe the writing
//! communicator, so a reader can tell which files belong together. run_id is nonzero and
//! equal in all files of one output event, whatever the channel, so files left over from
//! another run are detected. dtype 0/1 are double/complex<double>; 2/3 are float/complex<float>.
struct RspaceFileHeader
{
    RspaceKind kind = RspaceKind::ExxR;
    std::int32_t nprocs = 0;
    std::int32_t myid = 0;
    std::int32_t dtype = 0;
    std::uint64_t run_id = 0;
    std::uint64_t n_blocks = 0;
    std::optional<double> omega;  //!< set exactly for SigcRF
};

//! Number of bytes of the header of the given kind.
constexpr std::size_t rspace_header_bytes(const RspaceKind kind)
{
    return 48 + (kind == RspaceKind::SigcRF ? sizeof(double) : 0);
}

//! Generate a nonzero identifier of one output event. Collective on comm: rank 0 draws it
//! from the random device and the clock, and broadcasts it.
std::uint64_t make_rspace_run_id(MPI_Comm comm);

//! Write the header. Returns false on I/O failure.
bool write_rspace_header(std::ostream &os, const RspaceFileHeader &header);

//! Read and validate the header (magic, byte order, version, kind, dtype, run_id, omega).
//! Leaves the stream behind the header. Throws on a malformed or foreign file.
RspaceFileHeader read_rspace_header(std::istream &is, const std::string &fn);

//! Export one set of complete real-space AO blocks, without threshold filtering.
//! The file starts with the header (see RspaceFileHeader), whose kind, nprocs, myid, run_id
//! and omega come from info; dtype and n_blocks are filled in here. Then follow the records
//! of int[7] (R1,R2,R3,I,J,rows,columns) and uint64 absolute payload offset. Row-major
//! payloads follow the complete index table.
//! Returns false on I/O failure; callers handle MPI error propagation.
template <typename T>
bool write_rspace_matrices_binary(const ap_p_map<std::map<Vector3_Order<int>, matrix_m<T>>> &blocks,
                                  const AtomicBasis &basis, const std::string &fn,
                                  RspaceFileHeader info)
{
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, cplxdb> ||
                      std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>,
                  "unsupported real-space matrix output type");
    static_assert(sizeof(std::size_t) == sizeof(std::uint64_t), "64-bit size_t is required");
    constexpr int dtype = std::is_same_v<T, double>   ? 0
                          : std::is_same_v<T, cplxdb> ? 1
                          : std::is_same_v<T, float>  ? 2
                                                      : 3;
    assert(info.omega.has_value() == (info.kind == RspaceKind::SigcRF));
    assert(info.run_id != 0);
    info.dtype = dtype;
    info.n_blocks = 0;
    for (const auto &[IJ, R_blocks] : blocks) info.n_blocks += R_blocks.size();
    std::ofstream ofs(fn, std::ios::binary);
    if (!ofs) return false;
    if (!write_rspace_header(ofs, info)) return false;
    const auto n_blocks = static_cast<std::size_t>(info.n_blocks);
    std::size_t offset =
        rspace_header_bytes(info.kind) + n_blocks * (7 * sizeof(int) + sizeof(std::size_t));
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
