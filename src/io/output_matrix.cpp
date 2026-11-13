#include "output_matrix.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <vector>

#include "../utils/error.h"
#include "fs.h"

namespace librpa_int
{

std::uint64_t make_rspace_run_id(MPI_Comm comm)
{
    int myid = 0;
    MPI_Comm_rank(comm, &myid);
    std::uint64_t run_id = 0;
    if (myid == 0)
    {
        // The clock alone can repeat within one process (several writes in the same
        // tick), so mix in the random device; the identifier must not be zero.
        std::random_device device;
        std::mt19937_64 generator(
            (static_cast<std::uint64_t>(device()) << 32) ^ device() ^
            static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()));
        while (run_id == 0) run_id = generator();
    }
    MPI_Bcast(&run_id, 1, MPI_UINT64_T, 0, comm);
    return run_id;
}

template <typename T>
static void put(std::ostream &os, const T &value)
{
    os.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

bool write_rspace_header(std::ostream &os, const RspaceFileHeader &header)
{
    os.write(RSPACE_MAGIC, sizeof(RSPACE_MAGIC));
    put(os, RSPACE_ENDIAN_MARKER);
    put(os, RSPACE_VERSION);
    put(os, static_cast<std::int32_t>(header.kind));
    put(os, header.nprocs);
    put(os, header.myid);
    put(os, header.dtype);
    put(os, header.run_id);
    put(os, header.n_blocks);
    if (header.kind == RspaceKind::SigcRF) put(os, header.omega.value());
    return static_cast<bool>(os);
}

RspaceFileHeader read_rspace_header(std::istream &is, const std::string &fn)
{
    auto get = [&](auto &value)
    {
        if (!is.read(reinterpret_cast<char *>(&value), sizeof(value)))
            throw LIBRPA_RUNTIME_ERROR("truncated real-space matrix file header: " + fn);
    };
    char magic[sizeof(RSPACE_MAGIC)];
    if (!is.read(magic, sizeof(magic)) || std::memcmp(magic, RSPACE_MAGIC, sizeof(magic)) != 0)
        throw LIBRPA_RUNTIME_ERROR("not a real-space matrix file (bad magic): " + fn);
    std::uint32_t endian;
    std::int32_t version, kind;
    RspaceFileHeader header;
    get(endian);
    if (endian != RSPACE_ENDIAN_MARKER)
        throw LIBRPA_RUNTIME_ERROR("real-space matrix file has a different byte order: " + fn);
    get(version);
    if (version != RSPACE_VERSION)
        throw LIBRPA_RUNTIME_ERROR("unsupported real-space matrix file version " +
                                   std::to_string(version) + ": " + fn);
    get(kind);
    if (kind != static_cast<std::int32_t>(RspaceKind::ExxR) &&
        kind != static_cast<std::int32_t>(RspaceKind::SigcRF))
        throw LIBRPA_RUNTIME_ERROR("unknown real-space matrix file kind: " + fn);
    header.kind = static_cast<RspaceKind>(kind);
    get(header.nprocs);
    get(header.myid);
    get(header.dtype);
    get(header.run_id);
    get(header.n_blocks);
    if (header.nprocs < 1 || header.myid < 0 || header.myid >= header.nprocs)
        throw LIBRPA_RUNTIME_ERROR("invalid rank information in real-space matrix file: " + fn);
    if (header.dtype < 0 || header.dtype > 3)
        throw LIBRPA_RUNTIME_ERROR("unknown scalar type in real-space matrix file: " + fn);
    if (header.run_id == 0)
        throw LIBRPA_RUNTIME_ERROR("invalid run identifier in real-space matrix file: " + fn);
    if (header.kind == RspaceKind::SigcRF)
    {
        double omega;
        get(omega);
        if (!std::isfinite(omega))
            throw LIBRPA_RUNTIME_ERROR("nonfinite frequency in real-space matrix file: " + fn);
        header.omega = omega;
    }
    return header;
}

static bool write_matrix_binary(const Matz &mat, const std::string &fn)
{
    std::ofstream ofs(fn, std::ios::binary);
    if (!ofs) return false;

    const std::int32_t dimension = mat.nr();
    const std::int32_t type_bytes = sizeof(double);
    ofs.write(reinterpret_cast<const char *>(&dimension), sizeof(dimension));
    ofs.write(reinterpret_cast<const char *>(&type_bytes), sizeof(type_bytes));
    std::vector<double> row(2 * static_cast<std::size_t>(dimension));
    for (int i = 0; i != dimension; ++i)
    {
        for (int j = 0; j != dimension; ++j)
        {
            const auto v = mat(i, j);
            row[2 * static_cast<std::size_t>(j)] = v.real();
            row[2 * static_cast<std::size_t>(j) + 1] = v.imag();
        }
        ofs.write(reinterpret_cast<const char *>(row.data()),
                  static_cast<std::streamsize>(row.size() * sizeof(double)));
        if (!ofs) return false;
    }
    ofs.close();
    return ofs.good();
}

std::string rspace_matrix_filename(const std::string &dir, const std::string &prefix,
                                   const int ispin, const int bra, const int ket, const int myid,
                                   const int iomega)
{
    std::ostringstream name;
    name << path_as_directory(dir) << prefix << "_ispin_" << std::setfill('0') << std::setw(2)
         << ispin << "_s_" << bra << ket;
    if (iomega >= 0) name << "_iomega_" << std::setw(3) << iomega;
    name << "_myid_" << std::setw(5) << myid << ".bin";
    return name.str();
}

void write_matrix_binary_parallel(const Matz &mat_loc, const ArrayDesc &desc, const std::string &fn,
                                  const int index_start, const int index_end_option)
{
    if (!desc.is_initialized())
        throw LIBRPA_RUNTIME_ERROR("binary matrix output descriptor is not initialized");
    if (desc.m() != desc.n())
        throw LIBRPA_RUNTIME_ERROR("binary matrix output expects a square matrix");
    const int index_end = index_end_option < 0 ? desc.m() : index_end_option;
    if (index_start < 0 || index_start >= index_end || index_end > desc.m())
        throw LIBRPA_RUNTIME_ERROR("binary matrix output index range is outside the matrix");

    int bad_layout = mat_loc.nr() != desc.m_loc() || mat_loc.nc() != desc.n_loc() ||
                     mat_loc.major() != MAJOR::COL;
    MPI_Allreduce(MPI_IN_PLACE, &bad_layout, 1, MPI_INT, MPI_MAX, desc.comm());
    if (bad_layout)
        throw LIBRPA_RUNTIME_ERROR("binary matrix local block does not match its descriptor");

    ArrayDesc desc_full(desc.ictxt());
    const int dimension = index_end - index_start;
    desc_full.init(dimension, dimension, dimension, dimension, desc.irsrc(), desc.icsrc());
    Matz mat_full(desc_full.m_loc(), desc_full.n_loc(), MAJOR::COL);
    ScalapackConnector::pgemr2d_f(dimension, dimension, mat_loc.ptr(), index_start + 1,
                                  index_start + 1, desc.desc, mat_full.ptr(), 1, 1, desc_full.desc,
                                  desc.ictxt());

    int write_ok = 1;
    if (desc_full.is_src()) write_ok = write_matrix_binary(mat_full, fn);
    MPI_Allreduce(MPI_IN_PLACE, &write_ok, 1, MPI_INT, MPI_MIN, desc.comm());
    if (!write_ok) throw LIBRPA_RUNTIME_ERROR("failed to write binary matrix output file: " + fn);
}

}  // namespace librpa_int
