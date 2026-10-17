#include "output_matrix.h"

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "../utils/error.h"
#include "fs.h"

namespace librpa_int
{

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
