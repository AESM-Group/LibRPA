#include "reader_coulomb.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <complex>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reader_context.h"
#include "../../src/api/instance_manager.h"
#include "../../src/core/atomic_basis.h"
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/math/matrix.h"
#include "../../src/mpi/global_mpi.h"
#include "../../src/utils/profiler.h"

namespace librpa::reader
{

#define READER_COULOMB_V1_MARKER -20129433

using std::ifstream;
using std::string;
using librpa_int::atpair_t;
using librpa_int::matrix;

static const librpa_int::AtomicBasis &target_coulomb_basis(ReaderContext &ctx, const bool use_shrink_basis)
{
    auto ds = librpa_int::api::get_dataset_instance(ctx.h.get_c_handler());
    return use_shrink_basis ? ds->basis_aux_shrink : ds->basis_aux;
}

//! Check if Coulomb matrix data file is in ASCII text or unformatted binary format
bool check_coulomb_file_binary(const string &file_path)
{
    librpa_int::require_readable_file(file_path);
    bool is_binary = true;
    ifstream infile;
    int nirk;
    infile.open(file_path, std::ios::in);
    infile >> nirk;
    if (infile.good())
    {
        is_binary = false;
    }
    // cout << nirk << " " << is_binary << endl;
    infile.close();
    return is_binary;
}

// TODO work-in-progress: 2025-11-30
void read_coulomb(const string &dir_path, const librpa::ParallelRouting routing, bool is_cut)
{
}

static int handle_Vq_full_file(const string &file_path, std::map<int, librpa_int::ComplexMatrix> &Vq_full, bool binary)
{
    // cout << "Begin to read Vq from " << file_path << endl;
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    int n_irk_points_local;
    int n_irk_points;

    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
        infile.read((char *)&n_irk_points, sizeof(int));
        infile.read((char *)&n_irk_points_local, sizeof(int));
    }
    else
    {
        infile.open(file_path);
        infile >> n_irk_points;
    }

    if (!infile.good()) return 1;

    if (binary)
    {
        int nbasbas, brow, erow, bcol, ecol, iq;
        double q_weight;

        for (int i_irk = 0; i_irk < n_irk_points_local; i_irk++)
        {
            infile.read((char *)&nbasbas, sizeof(int));
            infile.read((char *)&brow, sizeof(int));
            infile.read((char *)&erow, sizeof(int));
            infile.read((char *)&bcol, sizeof(int));
            infile.read((char *)&ecol, sizeof(int));
            infile.read((char *)&iq, sizeof(int));
            infile.read((char *)&q_weight, sizeof(double));

            brow--;
            erow--;
            bcol--;
            ecol--;
            iq--;

            if (!Vq_full.count(iq))
            {
                Vq_full[iq].create(nbasbas, nbasbas);
            }

            const int nrow = erow - brow + 1;
            const int ncol = ecol - bcol + 1;
            const size_t n = nrow * ncol;
            std::vector<std::complex<double>> tmp(n);
            infile.read((char *) tmp.data(), 2 * n * sizeof(double));
            for (int i = 0; i < nrow; i++)
            {
                for (int j = 0; j < ncol; j++)
                {
                    const auto i_mu = i + brow;
                    const auto i_nu = j + bcol;
                    Vq_full[iq](i_mu, i_nu) = tmp[i * ncol + j];
                }
            }
        }
    }
    else
    {
        string nbasbas, begin_row, end_row, begin_col, end_col, q1, q2, q3, vq_r, vq_i, q_num,
            q_weight;
        while (infile.peek() != EOF)
        {
            infile >> nbasbas >> begin_row >> end_row >> begin_col >> end_col;
            if (infile.peek() == EOF) break;
            if (!infile.good()) return 2;
            // cout << "vq range: " << begin_row << " ~ " << end_row << "  ,   " << begin_col << " ~
            // " << end_col << endl;
            infile >> q_num >> q_weight;
            if (!infile.good()) return 3;
            int mu = stoi(nbasbas);
            int nu = stoi(nbasbas);
            int brow = stoi(begin_row) - 1;
            int erow = stoi(end_row) - 1;
            int bcol = stoi(begin_col) - 1;
            int ecol = stoi(end_col) - 1;
            int iq = stoi(q_num) - 1;

            //skip empty coulumb_file
            if ((erow - brow < 0) || (ecol - bcol < 0) || iq < 0) return 4;

            if (!Vq_full.count(iq))
            {
                Vq_full[iq].create(mu, nu);
            }
            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                for (int i_nu = bcol; i_nu <= ecol; i_nu++)
                {
                    infile >> vq_r >> vq_i;
                    Vq_full[iq](i_mu, i_nu) = std::complex<double>(stod(vq_r), stod(vq_i));
                }
            }
        }
    }
    return 0;
}

namespace
{

constexpr int COULOMB_V1_REAL_FLAG = 0;
constexpr int COULOMB_V1_COMPLEX_FLAG = 1;
constexpr MPI_Offset COULOMB_V1_HEADER_BASE_SIZE =
    6 * static_cast<MPI_Offset>(sizeof(int));
constexpr MPI_Offset COULOMB_V1_BLOCK_RECORD_SIZE =
    static_cast<MPI_Offset>(sizeof(int) + sizeof(std::int64_t));
constexpr MPI_Offset COULOMB_V1_MISSING_BLOCK = -1;
constexpr std::size_t COULOMB_V1_MAX_COLLECTED_READ_BYTES =
    256ULL * 1024ULL * 1024ULL;

static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
              "Coulomb atom-pair IO expects std::complex<double> as two doubles");

std::size_t coulomb_atom_pair_index(const std::size_t I, const std::size_t J,
                                    const std::size_t natoms)
{
    assert(I <= J);
    return I * natoms - I * (I - 1) / 2 + (J - I);
}

std::string posix_io_error(const std::string &operation, const std::string &path,
                           const int err)
{
    std::ostringstream ss;
    ss << operation << " Coulomb v1 file " << path << ": " << std::strerror(err);
    return ss.str();
}

std::vector<atpair_t> sort_atom_pairs_by_coulomb_v1_order(
    const std::vector<atpair_t> &atom_pairs,
    const std::size_t natoms)
{
    std::vector<atpair_t> sorted_pairs = atom_pairs;
    std::sort(sorted_pairs.begin(), sorted_pairs.end(),
              [natoms](const auto &a, const auto &b)
              {
                  const auto ia = coulomb_atom_pair_index(a.first, a.second, natoms);
                  const auto ib = coulomb_atom_pair_index(b.first, b.second, natoms);
                  return ia < ib;
              });
    return sorted_pairs;
}

void append_atom_pair(std::ostringstream &ss, const atpair_t &pair)
{
    ss << "(" << pair.first << "," << pair.second << ")";
}

std::string compact_atom_pair_ranges(const std::vector<atpair_t> &sorted_pairs,
                                     const std::size_t natoms)
{
    if (sorted_pairs.empty()) return "(empty)";

    std::ostringstream ss;
    std::size_t nranges = 0;
    std::size_t range_begin = 0;
    auto previous_index = coulomb_atom_pair_index(
        sorted_pairs.front().first, sorted_pairs.front().second, natoms);

    auto append_range = [&](const std::size_t begin, const std::size_t end) {
        if (nranges != 0) ss << "; ";
        append_atom_pair(ss, sorted_pairs[begin]);
        if (begin != end)
        {
            ss << " ... ";
            append_atom_pair(ss, sorted_pairs[end]);
        }
        ++nranges;
    };

    for (std::size_t i = 1; i != sorted_pairs.size(); ++i)
    {
        const auto index = coulomb_atom_pair_index(
            sorted_pairs[i].first, sorted_pairs[i].second, natoms);
        if (index != previous_index + 1)
        {
            append_range(range_begin, i - 1);
            range_begin = i;
        }
        previous_index = index;
    }
    append_range(range_begin, sorted_pairs.size() - 1);
    return ss.str();
}

std::size_t checked_size_from_offset(const MPI_Offset value,
                                     const std::string &context)
{
    if (value < 0 ||
        static_cast<unsigned long long>(value) >
            static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max()))
    {
        throw std::logic_error(context + ": byte count is out of range");
    }
    return static_cast<std::size_t>(value);
}

class CoulombV1File
{
public:
    explicit CoulombV1File(const std::string &path_in): path(path_in)
    {
        librpa_int::require_readable_file(path);
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            throw std::logic_error(posix_io_error("Failed to open", path, errno));
        }

        int header[6];
        read_bytes(0, header, sizeof(header));
        const int marker = header[0];
        iq = header[1];
        naux = header[2];
        value_flag = header[3];
        natoms = header[4];
        nblocks = header[5];

        if (marker != READER_COULOMB_V1_MARKER)
        {
            std::ostringstream ss;
            ss << path << ": invalid Coulomb reader v1 marker " << marker
               << ", expected " << READER_COULOMB_V1_MARKER;
            throw std::logic_error(ss.str());
        }
        if (iq <= 0)
        {
            throw std::logic_error(path + ": invalid 1-based q-point index");
        }
        if (naux <= 0)
        {
            throw std::logic_error(path + ": invalid Coulomb v1 Naux");
        }
        if (natoms <= 0)
        {
            throw std::logic_error(path + ": invalid Coulomb v1 atom count");
        }
        if (nblocks < 0)
        {
            throw std::logic_error(path + ": invalid Coulomb v1 block count");
        }
        if (value_flag != COULOMB_V1_REAL_FLAG &&
            value_flag != COULOMB_V1_COMPLEX_FLAG)
        {
            throw std::logic_error(path + ": invalid Coulomb v1 value-type flag");
        }

        value_bytes = value_flag == COULOMB_V1_COMPLEX_FLAG ?
            2 * sizeof(double) : sizeof(double);

        atom_naux.resize(static_cast<std::size_t>(natoms));
        read_bytes(COULOMB_V1_HEADER_BASE_SIZE, atom_naux.data(),
                   atom_naux.size() * sizeof(int));

        int naux_sum = 0;
        for (const auto nb: atom_naux)
        {
            if (nb <= 0)
            {
                throw std::logic_error(path + ": invalid per-atom auxiliary size");
            }
            naux_sum += nb;
        }
        if (naux_sum != naux)
        {
            std::ostringstream ss;
            ss << path << ": Coulomb v1 per-atom auxiliary sizes sum to "
               << naux_sum << ", expected " << naux;
            throw std::logic_error(ss.str());
        }

        npairs = static_cast<std::size_t>(natoms) *
            (static_cast<std::size_t>(natoms) + 1) / 2;
        if (static_cast<std::size_t>(nblocks) > npairs)
        {
            throw std::logic_error(path + ": Coulomb v1 block count exceeds atom-pair count");
        }
        atom_pairs.reserve(npairs);
        for (std::size_t I = 0; I != static_cast<std::size_t>(natoms); ++I)
        {
            for (std::size_t J = I; J != static_cast<std::size_t>(natoms); ++J)
            {
                atom_pairs.emplace_back(I, J);
            }
        }

        struct stat st;
        if (::fstat(fd, &st) != 0)
        {
            throw std::logic_error(posix_io_error("Failed to stat", path, errno));
        }
        if (st.st_size < 0)
        {
            throw std::logic_error(path + ": invalid Coulomb v1 file size");
        }
        const MPI_Offset file_size = static_cast<MPI_Offset>(st.st_size);
        block_offsets.assign(npairs, COULOMB_V1_MISSING_BLOCK);
        const MPI_Offset table_end = header_size();
        std::vector<char> block_table(
            static_cast<std::size_t>(nblocks) *
            static_cast<std::size_t>(COULOMB_V1_BLOCK_RECORD_SIZE));
        if (!block_table.empty())
        {
            read_bytes(block_table_offset(), block_table.data(), block_table.size());
        }
        for (int iblock = 0; iblock != nblocks; ++iblock)
        {
            const char *record = block_table.data() +
                static_cast<std::size_t>(iblock) *
                static_cast<std::size_t>(COULOMB_V1_BLOCK_RECORD_SIZE);
            int ipair_i32 = -1;
            std::int64_t block_offset_i64 = 0;
            std::memcpy(&ipair_i32, record, sizeof(ipair_i32));
            std::memcpy(&block_offset_i64, record + sizeof(ipair_i32),
                        sizeof(block_offset_i64));
            if (ipair_i32 < 0 || static_cast<std::size_t>(ipair_i32) >= npairs)
            {
                std::ostringstream ss;
                ss << path << ": invalid Coulomb v1 atom-pair index " << ipair_i32;
                throw std::logic_error(ss.str());
            }

            const auto ipair = static_cast<std::size_t>(ipair_i32);
            if (block_offsets[ipair] != COULOMB_V1_MISSING_BLOCK)
            {
                std::ostringstream ss;
                ss << path << ": duplicated Coulomb v1 atom-pair index " << ipair_i32;
                throw std::logic_error(ss.str());
            }
            const auto block_offset = static_cast<MPI_Offset>(block_offset_i64);
            const auto [I, J] = atom_pair_from_index(ipair);
            const MPI_Offset block_bytes =
                static_cast<MPI_Offset>(atom_naux[I]) *
                static_cast<MPI_Offset>(atom_naux[J]) * value_bytes;
            if (block_offset < table_end || block_offset > file_size ||
                block_bytes > file_size - block_offset)
            {
                std::ostringstream ss;
                ss << path << ": invalid Coulomb v1 byte offset " << block_offset
                   << " for atom-pair index " << ipair_i32;
                throw std::logic_error(ss.str());
            }
            block_offsets[ipair] = block_offset;
        }

        std::vector<std::pair<MPI_Offset, MPI_Offset>> ranges;
        ranges.reserve(static_cast<std::size_t>(nblocks));
        for (std::size_t ipair = 0; ipair != npairs; ++ipair)
        {
            if (block_offsets[ipair] == COULOMB_V1_MISSING_BLOCK) continue;
            const auto [I, J] = atom_pair_from_index(ipair);
            const MPI_Offset block_bytes =
                static_cast<MPI_Offset>(atom_naux[I]) *
                static_cast<MPI_Offset>(atom_naux[J]) * value_bytes;
            ranges.emplace_back(block_offsets[ipair], block_offsets[ipair] + block_bytes);
        }
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t i = 1; i < ranges.size(); ++i)
        {
            if (ranges[i].first < ranges[i - 1].second)
            {
                throw std::logic_error(path + ": overlapping Coulomb v1 atom-pair blocks");
            }
        }
    }

    ~CoulombV1File()
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }

    CoulombV1File(const CoulombV1File &) = delete;
    CoulombV1File &operator=(const CoulombV1File &) = delete;

    void read_bytes(MPI_Offset offset, void *buffer, std::size_t nbytes) const
    {
        char *ptr = static_cast<char *>(buffer);
        std::size_t bytes_left = nbytes;
        while (bytes_left > 0)
        {
            if (offset < 0 || offset > static_cast<MPI_Offset>(
                    std::numeric_limits<off_t>::max()))
            {
                throw std::logic_error(path + ": Coulomb v1 read offset is out of range");
            }
            const std::size_t chunk = std::min<std::size_t>(
                bytes_left,
                static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t bytes_read = ::pread(
                fd, ptr, chunk, static_cast<off_t>(offset));
            if (bytes_read < 0)
            {
                throw std::logic_error(posix_io_error("Failed to read", path, errno));
            }
            if (bytes_read == 0)
            {
                throw std::logic_error("Truncated Coulomb v1 file " + path);
            }
            offset += static_cast<MPI_Offset>(bytes_read);
            ptr += bytes_read;
            bytes_left -= static_cast<std::size_t>(bytes_read);
        }
    }

    MPI_Offset header_size() const
    {
        return COULOMB_V1_HEADER_BASE_SIZE +
            static_cast<MPI_Offset>(natoms) * sizeof(int) +
            static_cast<MPI_Offset>(nblocks) * COULOMB_V1_BLOCK_RECORD_SIZE;
    }

    MPI_Offset block_table_offset() const
    {
        return COULOMB_V1_HEADER_BASE_SIZE +
            static_cast<MPI_Offset>(natoms) * sizeof(int);
    }

    std::pair<std::size_t, std::size_t> atom_pair_from_index(std::size_t ipair) const
    {
        if (ipair >= atom_pairs.size())
            throw std::logic_error(path + ": atom-pair index is out of range");
        return atom_pairs[ipair];
    }

    MPI_Offset block_offset(const std::size_t I, const std::size_t J) const
    {
        return block_offsets.at(coulomb_atom_pair_index(I, J, atom_naux.size()));
    }

    bool has_block(const std::size_t I, const std::size_t J) const
    {
        return block_offset(I, J) != COULOMB_V1_MISSING_BLOCK;
    }

    std::string path;
    int fd = -1;
    int iq = 0;
    int naux = 0;
    int value_flag = COULOMB_V1_COMPLEX_FLAG;
    int natoms = 0;
    int nblocks = 0;
    std::size_t npairs = 0;
    MPI_Offset value_bytes = 2 * static_cast<MPI_Offset>(sizeof(double));
    std::vector<int> atom_naux;
    std::vector<std::pair<std::size_t, std::size_t>> atom_pairs;
    std::vector<MPI_Offset> block_offsets;
};

struct CoulombV1BlockRead
{
    std::size_t I = 0;
    std::size_t J = 0;
    MPI_Offset offset = 0;
    MPI_Offset nbytes = 0;
    std::size_t nvalues = 0;
};

struct CoulombV1CollectedRead
{
    MPI_Offset offset = 0;
    MPI_Offset nbytes = 0;
    std::vector<CoulombV1BlockRead> blocks;
};

std::vector<std::complex<double>> read_coulomb_v1_atom_pair(
    const CoulombV1File &file,
    const std::size_t I,
    const std::size_t J)
{
    if (I > J)
    {
        throw std::logic_error("Coulomb reader v1 expects upper-triangular atom pairs");
    }

    if (!file.has_block(I, J))
    {
        return {};
    }

    const auto nrow = static_cast<std::size_t>(file.atom_naux[I]);
    const auto ncol = static_cast<std::size_t>(file.atom_naux[J]);
    const auto n = nrow * ncol;
    std::vector<std::complex<double>> block(n);

    if (file.value_flag == COULOMB_V1_COMPLEX_FLAG)
    {
        file.read_bytes(file.block_offset(I, J), block.data(),
                        n * sizeof(std::complex<double>));
    }
    else
    {
        std::vector<double> buffer(n);
        file.read_bytes(file.block_offset(I, J), buffer.data(),
                        n * sizeof(double));
        for (std::size_t i = 0; i != n; ++i)
        {
            block[i] = std::complex<double>(buffer[i], 0.0);
        }
    }
    return block;
}

MPI_Offset coulomb_v1_block_bytes(const CoulombV1File &file,
                                  const std::size_t I,
                                  const std::size_t J)
{
    return static_cast<MPI_Offset>(file.atom_naux[I]) *
        static_cast<MPI_Offset>(file.atom_naux[J]) * file.value_bytes;
}

std::vector<CoulombV1BlockRead> make_coulomb_v1_block_reads(
    const CoulombV1File &file,
    const std::vector<atpair_t> &sorted_atom_pairs)
{
    std::vector<CoulombV1BlockRead> reads;
    reads.reserve(sorted_atom_pairs.size());
    for (const auto &ap: sorted_atom_pairs)
    {
        const auto I = static_cast<std::size_t>(ap.first);
        const auto J = static_cast<std::size_t>(ap.second);
        if (!file.has_block(I, J)) continue;

        const auto nrow = static_cast<std::size_t>(file.atom_naux[I]);
        const auto ncol = static_cast<std::size_t>(file.atom_naux[J]);
        reads.push_back(
            {I, J, file.block_offset(I, J), coulomb_v1_block_bytes(file, I, J),
             nrow * ncol});
    }
    return reads;
}

std::vector<CoulombV1CollectedRead> collect_coulomb_v1_reads(
    const std::vector<CoulombV1BlockRead> &blocks)
{
    std::vector<CoulombV1CollectedRead> collected;
    for (const auto &block: blocks)
    {
        const bool need_new_read =
            collected.empty() ||
            block.offset != collected.back().offset + collected.back().nbytes ||
            (!collected.back().blocks.empty() &&
             block.nbytes >
                 static_cast<MPI_Offset>(COULOMB_V1_MAX_COLLECTED_READ_BYTES) -
                     collected.back().nbytes);

        if (need_new_read)
        {
            collected.push_back({block.offset, block.nbytes, {block}});
        }
        else
        {
            auto &read = collected.back();
            read.nbytes += block.nbytes;
            read.blocks.push_back(block);
        }
    }
    return collected;
}

void set_coulomb_v1_atom_pair(ReaderContext &ctx, const int iq0,
                              const std::size_t I,
                              const std::size_t J,
                              const int naux_i,
                              const int naux_j,
                              const std::complex<double> *block,
                              const double threshold,
                              const bool is_cut_coulomb)
{
    if (is_cut_coulomb)
    {
        ctx.h.set_aux_cut_coulomb_k_atom_pair_packed(
            iq0, I, J, naux_i, naux_j, block, threshold);
    }
    else
    {
        ctx.h.set_aux_bare_coulomb_k_atom_pair_packed(
            iq0, I, J, naux_i, naux_j, block, threshold);
    }
}

void process_complex_coulomb_v1_read(ReaderContext &ctx, const CoulombV1File &file,
                                     const CoulombV1CollectedRead &read,
                                     const int iq0,
                                     const double threshold,
                                     const bool is_cut_coulomb)
{
    if (read.nbytes % static_cast<MPI_Offset>(sizeof(std::complex<double>)) != 0)
    {
        throw std::logic_error(file.path + ": collected complex read is misaligned");
    }
    const auto nvalues = checked_size_from_offset(
        read.nbytes / static_cast<MPI_Offset>(sizeof(std::complex<double>)),
        file.path);
    std::vector<std::complex<double>> buffer(nvalues);
    file.read_bytes(read.offset, buffer.data(),
                    checked_size_from_offset(read.nbytes, file.path));

    for (const auto &block: read.blocks)
    {
        const auto value_offset = checked_size_from_offset(
            (block.offset - read.offset) /
                static_cast<MPI_Offset>(sizeof(std::complex<double>)),
            file.path);
        set_coulomb_v1_atom_pair(ctx, iq0, block.I, block.J,
                                 file.atom_naux[block.I],
                                 file.atom_naux[block.J],
                                 buffer.data() + value_offset,
                                 threshold, is_cut_coulomb);
    }
}

void process_real_coulomb_v1_read(ReaderContext &ctx, const CoulombV1File &file,
                                  const CoulombV1CollectedRead &read,
                                  const int iq0,
                                  const double threshold,
                                  const bool is_cut_coulomb)
{
    if (read.nbytes % static_cast<MPI_Offset>(sizeof(double)) != 0)
    {
        throw std::logic_error(file.path + ": collected real read is misaligned");
    }
    const auto nvalues = checked_size_from_offset(
        read.nbytes / static_cast<MPI_Offset>(sizeof(double)), file.path);
    std::vector<double> buffer(nvalues);
    file.read_bytes(read.offset, buffer.data(),
                    checked_size_from_offset(read.nbytes, file.path));

    for (const auto &block: read.blocks)
    {
        const auto value_offset = checked_size_from_offset(
            (block.offset - read.offset) / static_cast<MPI_Offset>(sizeof(double)),
            file.path);
        std::vector<std::complex<double>> complex_block(block.nvalues);
        for (std::size_t i = 0; i != block.nvalues; ++i)
        {
            complex_block[i] = std::complex<double>(buffer[value_offset + i], 0.0);
        }
        set_coulomb_v1_atom_pair(ctx, iq0, block.I, block.J,
                                 file.atom_naux[block.I],
                                 file.atom_naux[block.J],
                                 complex_block.data(),
                                 threshold, is_cut_coulomb);
    }
}

void read_coulomb_v1_atom_pairs_collected(ReaderContext &ctx,
    const CoulombV1File &file,
    const int iq0,
    const std::vector<atpair_t> &sorted_atom_pairs,
    const double threshold,
    const bool is_cut_coulomb)
{
    using librpa_int::global::ofs_myid;

    const auto block_reads = make_coulomb_v1_block_reads(file, sorted_atom_pairs);
    const auto collected_reads = collect_coulomb_v1_reads(block_reads);
    if (librpa_int::global::should_output(LIBRPA_VERBOSE_DEBUG))
    {
        MPI_Offset payload_bytes = 0;
        for (const auto &block: block_reads) payload_bytes += block.nbytes;
        std::ostringstream payload_mb;
        payload_mb << std::fixed << std::setprecision(2)
                   << static_cast<double>(payload_bytes) * 1.0e-6;
        ofs_myid << "read_Vq_row_v1 "
                 << (is_cut_coulomb ? "cut" : "bare")
                 << " collected payload reads for " << file.path
                 << ": blocks=" << block_reads.size()
                 << ", reads=" << collected_reads.size()
                 << ", payload=" << payload_mb.str() << " MB" << std::endl;
    }

    for (const auto &read: collected_reads)
    {
        if (file.value_flag == COULOMB_V1_COMPLEX_FLAG)
        {
            process_complex_coulomb_v1_read(ctx,
                file, read, iq0, threshold, is_cut_coulomb);
        }
        else
        {
            process_real_coulomb_v1_read(ctx,
                file, read, iq0, threshold, is_cut_coulomb);
        }
    }
}

void validate_coulomb_v1_basis(const CoulombV1File &file,
                               const librpa_int::AtomicBasis &basis_aux)
{
    if (static_cast<std::size_t>(file.naux) != basis_aux.nb_total)
    {
        std::ostringstream ss;
        ss << file.path << ": Coulomb v1 Naux=" << file.naux
           << " does not match basis_aux.nb_total=" << basis_aux.nb_total;
        throw std::logic_error(ss.str());
    }
    if (static_cast<std::size_t>(file.natoms) != basis_aux.n_atoms)
    {
        std::ostringstream ss;
        ss << file.path << ": Coulomb v1 natoms=" << file.natoms
           << " does not match basis_aux.n_atoms=" << basis_aux.n_atoms;
        throw std::logic_error(ss.str());
    }
    for (std::size_t I = 0; I != basis_aux.n_atoms; ++I)
    {
        if (static_cast<std::size_t>(file.atom_naux[I]) != basis_aux[I])
        {
            std::ostringstream ss;
            ss << file.path << ": Coulomb v1 atom " << I
               << " Naux=" << file.atom_naux[I]
               << " does not match basis_aux=" << basis_aux[I];
            throw std::logic_error(ss.str());
        }
    }
}

bool is_legacy_coulomb_filename(const string &filename, const string &prefix)
{
    const string suffix = ".txt";
    return filename.find(prefix) == 0 &&
           filename.size() >= suffix.size() &&
           filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0;
}

size_t read_Vq_full_v1(ReaderContext &ctx, const string &dir_path, const string &vq_fprefix,
                       bool is_cut_coulomb, const bool use_shrink_basis)
{
    using namespace librpa_int::global;

    profiler.start(__FUNCTION__);
    const auto &basis_aux = target_coulomb_basis(ctx, use_shrink_basis);
    const auto files = librpa_int::discover_files_with_prefix(dir_path, vq_fprefix);
    if (files.empty())
    {
        throw std::logic_error(
            "No Coulomb reader v1 files found with prefix " + vq_fprefix);
    }

    for (const auto &path: files)
    {
        CoulombV1File file(path);
        validate_coulomb_v1_basis(file, basis_aux);
        const int iq0 = file.iq - 1;

        for (std::size_t I = 0; I != basis_aux.n_atoms; ++I)
        {
            for (std::size_t J = I; J != basis_aux.n_atoms; ++J)
            {
                auto block = read_coulomb_v1_atom_pair(file, I, J);
                if (block.empty()) continue;
                if (is_cut_coulomb)
                {
                    ctx.h.set_aux_cut_coulomb_k_atom_pair_packed(
                        iq0, I, J, basis_aux[I], basis_aux[J],
                        block.data(), ctx.opts.vq_threshold);
                }
                else
                {
                    ctx.h.set_aux_bare_coulomb_k_atom_pair_packed(
                        iq0, I, J, basis_aux[I], basis_aux[J],
                        block.data(), ctx.opts.vq_threshold);
                }
            }
        }
    }
    profiler.stop(__FUNCTION__);
    return 0;
}

size_t read_Vq_row_v1(ReaderContext &ctx, const string &dir_path, const string &vq_fprefix, double threshold,
                      const std::vector<atpair_t> &local_atpair,
                      bool is_cut_coulomb, const bool use_shrink_basis)
{
    using namespace librpa_int::global;

    profiler.start(__FUNCTION__);
    const auto &basis_aux = target_coulomb_basis(ctx, use_shrink_basis);
    const auto files = librpa_int::discover_files_with_prefix(dir_path, vq_fprefix);
    if (files.empty())
    {
        throw std::logic_error(
            "No Coulomb reader v1 files found with prefix " + vq_fprefix);
    }

    const auto sorted_local_atpair = sort_atom_pairs_by_coulomb_v1_order(
        local_atpair, basis_aux.n_atoms);
    if (librpa_int::global::should_output(LIBRPA_VERBOSE_DEBUG))
    {
        ofs_myid << "read_Vq_row_v1 "
                 << (is_cut_coulomb ? "cut" : "bare")
                 << " local atom-pairs in v1 file order"
                 << " (natoms=" << basis_aux.n_atoms
                 << ", count=" << sorted_local_atpair.size()
                 << "): "
                 << compact_atom_pair_ranges(sorted_local_atpair, basis_aux.n_atoms)
                 << std::endl;
    }

    for (const auto &path: files)
    {
        CoulombV1File file(path);
        validate_coulomb_v1_basis(file, basis_aux);
        const int iq0 = file.iq - 1;
        read_coulomb_v1_atom_pairs_collected(ctx,
            file, iq0, sorted_local_atpair, threshold, is_cut_coulomb);
    }

    profiler.stop(__FUNCTION__);
    return 0;
}

} // namespace

int detect_coulomb_reader_version(const string &dir_path, const string &vq_fprefix)
{
    const auto files = librpa_int::discover_files_with_prefix(dir_path, vq_fprefix);
    if (files.empty())
    {
        throw std::logic_error("No Coulomb files found with prefix " + vq_fprefix);
    }

    ifstream infile(files.front(), std::ios::in | std::ios::binary);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + files.front());
    }

    int marker_or_count = 0;
    infile.read((char *) &marker_or_count, sizeof(int));
    if (infile.good())
    {
        if (marker_or_count >= 0) return 0;
        if (marker_or_count == READER_COULOMB_V1_MARKER) return 1;

        throw std::logic_error(
            files.front() + ": unknown Coulomb reader marker " +
            std::to_string(marker_or_count));
    }

    int text_value = 0;
    ifstream text_infile(files.front(), std::ios::in);
    if (text_infile >> text_value && text_value >= 0)
    {
        return 0;
    }

    throw std::logic_error("Failed to detect Coulomb reader version from " +
                           files.front());
}

HartreeCoulombQ0 read_hartree_coulomb_q0(ReaderContext &ctx, const std::string &dir_path,
                                         const std::string &prefix, int reader_version,
                                         const bool use_shrink_basis)
{
    if (prefix.empty())
        throw std::invalid_argument("A dedicated Hartree metric prefix is required");
    if (reader_version < 0) reader_version = detect_coulomb_reader_version(dir_path, prefix);
    const auto &basis = target_coulomb_basis(ctx, use_shrink_basis);
    if (basis.nb_total == 0 ||
        basis.nb_total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Invalid Hartree auxiliary dimension");
    int nspins, nkpoints, nstates, naos, nspinor;
    librpa_get_meanfield_dimensions(ctx.h.get_c_handler(), 0, &nspins, &nkpoints, &nstates,
                                     &naos, &nspinor);
    std::vector<double> fractional_k(static_cast<std::size_t>(nkpoints) * 3);
    librpa_get_crpa_kgrid(ctx.h.get_c_handler(), nkpoints, fractional_k.data());
    int iq_gamma = -1;
    for (int k = 0; k < nkpoints; ++k)
    {
        const auto offset = 3 * k;
        if (std::abs(fractional_k[offset]) +
                std::abs(fractional_k[offset + 1]) +
                std::abs(fractional_k[offset + 2]) <
            1.0e-10)
            iq_gamma = k;
    }
    if (iq_gamma < 0)
        throw std::invalid_argument("Hartree metric input requires a Gamma-centred SCF grid");
    HartreeCoulombQ0 result;
    result.dimension = static_cast<int>(basis.nb_total);
    const auto n = basis.nb_total;
    result.values.resize(n * n);
    const auto files = librpa_int::discover_files_with_prefix(dir_path, prefix);
    if (files.empty())
        throw std::runtime_error("No Hartree metric files found with prefix " + prefix);
    bool found_gamma = false;
    if (reader_version == 1)
    {
        std::vector<std::size_t> offsets(basis.n_atoms + 1);
        for (std::size_t atom = 0; atom < basis.n_atoms; ++atom)
            offsets[atom + 1] = offsets[atom] + basis[atom];
        std::set<std::pair<std::size_t, std::size_t>> seen;
        for (const auto &path : files)
        {
            CoulombV1File file(path);
            validate_coulomb_v1_basis(file, basis);
            if (file.iq - 1 != iq_gamma) continue;
            found_gamma = true;
            for (std::size_t I = 0; I < basis.n_atoms; ++I)
                for (std::size_t J = I; J < basis.n_atoms; ++J)
                {
                    const auto block = read_coulomb_v1_atom_pair(file, I, J);
                    if (block.empty()) continue;  // legal omitted zero blocks
                    const bool duplicate = !seen.emplace(I, J).second;
                    for (std::size_t i = 0; i < basis[I]; ++i)
                        for (std::size_t j = 0; j < basis[J]; ++j)
                        {
                            const auto value = block[i * basis[J] + j];
                            auto &target = result.values[(offsets[I] + i) * n + offsets[J] + j];
                            if (duplicate &&
                                std::abs(target - value) >
                                    1.0e-11 * std::max({1.0, std::abs(target), std::abs(value)}))
                                throw std::runtime_error(
                                    "Conflicting duplicate Hartree metric atom-pair blocks");
                            target = value;
                            if (I != J)
                                result.values[(offsets[J] + j) * n + offsets[I] + i] =
                                    std::conj(value);
                        }
                }
        }
    }
    else if (reader_version == 0)
    {
        std::map<int, librpa_int::ComplexMatrix> matrices;
        for (const auto &path : files)
        {
            if (!is_legacy_coulomb_filename(path.substr(path.find_last_of('/') + 1), prefix))
                continue;
            const auto status =
                handle_Vq_full_file(path, matrices, check_coulomb_file_binary(path));
            if (status != 0)
                throw std::runtime_error("Failed to read Hartree metric " + path + ": code " +
                                         std::to_string(status));
        }
        const auto gamma = matrices.find(iq_gamma);
        if (gamma != matrices.end())
        {
            if (gamma->second.nr != result.dimension || gamma->second.nc != result.dimension)
                throw std::runtime_error(
                    "Hartree metric dimension differs from selected auxiliary basis");
            found_gamma = true;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j) result.values[i * n + j] = gamma->second(i, j);
        }
    }
    else
        throw std::invalid_argument("Unknown Hartree Coulomb reader version " +
                                    std::to_string(reader_version));
    if (!found_gamma)
        throw std::runtime_error("Dedicated Hartree metric contains no Gamma q=0 matrix");
    for (const auto value : result.values)
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
            throw std::runtime_error("Hartree metric contains a nonfinite value");
    return result;
}

size_t read_Vq_full(ReaderContext &ctx, const string &dir_path, const string &vq_fprefix, bool is_cut_coulomb,
                    int reader_version, const bool use_shrink_basis)
{
    using std::cout;
    using std::endl;
    using librpa_int::ComplexMatrix;
    using namespace librpa_int::global;

    if (reader_version < 0)
    {
        reader_version = detect_coulomb_reader_version(dir_path, vq_fprefix);
    }
    lib_printf_root("Coulomb reader (%s): %s\n", vq_fprefix.c_str(),
                    reader_version == 1 ? "binary v1" : "legacy");

    if (reader_version == 1)
    {
        return read_Vq_full_v1(ctx, dir_path, vq_fprefix, is_cut_coulomb, use_shrink_basis);
    }
    if (reader_version != 0)
    {
        throw std::logic_error("Unknown Coulomb reader version " +
                               std::to_string(reader_version));
    }

    const auto &basis_aux = target_coulomb_basis(ctx, use_shrink_basis);
    const auto atom_mu_part_range = basis_aux.get_part_range();

    size_t vq_save = 0;
    size_t vq_discard = 0;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    std::vector<string> files;
    std::map<int, ComplexMatrix> Vq_full;

    bool binary;
    bool binary_checked = false;

    profiler.start("handle_Vq_full_file");
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        if (is_legacy_coulomb_filename(fm, vq_fprefix))
        {
            string file_path = dir_path + fm;
            if (!binary_checked)
            {
                binary = check_coulomb_file_binary(file_path);
                binary_checked = true;
                if (ctx.comm.myid == 0)
                {
                    if (binary)
                    {
                        cout << "Unformatted binary V files detected" << endl;
                    }
                    else
                    {
                        cout << "ASCII format V files detected" << endl;
                    }
                }
            }

            int retcode = handle_Vq_full_file(file_path, Vq_full, binary);
            if (retcode != 0)
            {
                librpa_int::global::lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error encountered when reading %s, return code %d", fm.c_str(), retcode);
            }
        }
    }
    profiler.stop("handle_Vq_full_file");

    // cout << "FINISH coulomb files reading!" << endl;
    profiler.start("set_aux_coulomb_k_atom_pair_out");
    for (auto &vf_p : Vq_full)
    {
        int iq = vf_p.first;

        // cout << "Qvec:" << qvec << endl;
        for (size_t I = 0; I != basis_aux.n_atoms; I++)
        {
            for (size_t J = 0; J != basis_aux.n_atoms; J++)
            {
                // Coulomb is Hermitian, only parse upper half
                if (I > J)
                {
                    continue;
                }

                // Vq_full stores the full matrix, parse by I-J block
                // The matrices have to be duplicated ...
                matrix re(basis_aux[I], basis_aux[J]), im(basis_aux[I], basis_aux[J]);

                // vq_ptr_tran->create(atom_mu[J],atom_mu[I]);
                // cout << "I J: " << I << "  " << J << "   mu,nu: " << atom_mu[I] << "  " << atom_mu[J] << endl;
                for (size_t i_mu = 0; i_mu != basis_aux[I]; i_mu++)
                {
                    for (size_t i_nu = 0; i_nu != basis_aux[J]; i_nu++)
                    {
                        const auto ir = basis_aux.get_global_index(I, i_mu);
                        const auto ic = basis_aux.get_global_index(J, i_nu);
                        re(i_mu, i_nu) = vf_p.second(ir, ic).real();
                        im(i_mu, i_nu) = vf_p.second(ir, ic).imag();
                    }
                }

                if (is_cut_coulomb)
                {
                    ctx.h.set_aux_cut_coulomb_k_atom_pair(iq, I, J, basis_aux[I], basis_aux[J], re.c, im.c, ctx.opts.vq_threshold);
                }
                else
                {
                    ctx.h.set_aux_bare_coulomb_k_atom_pair(iq, I, J, basis_aux[I], basis_aux[J], re.c, im.c, ctx.opts.vq_threshold);
                }
                // if (I == J)
                // {
                //     (*vq_ptr).set_as_identity_matrix();
                // }

                // if ((*vq_ptr).real().absmax() >= threshold)
                // {
                //     coulomb_mat[I][J][qvec] = vq_ptr;
                //     vq_save++;
                // }
                // else
                // {
                //     vq_discard++;
                // }
            }
        }
    }
    profiler.stop("set_aux_coulomb_k_atom_pair_out");
    closedir(dir);
    dir = NULL;
    // cout << "vq threshold: " << threshold << endl;
    // cout << "vq_save:    " << vq_save << endl;
    // cout << "vq_dicard:  " << vq_discard << endl;
    // cout << "  Vq_dim   " << coulomb_mat.size() << "    " << coulomb_mat[0].size() << "   "
    // << coulomb_mat[0][0].size() << endl; for (auto &irk : irk_weight)
    // {
    //     cout << " irk_vec and weight: " << irk.first << "  " << irk.second << endl;
    // }
    // cout << "Finish reading Vq" << endl;
    return vq_discard;
}


static int handle_Vq_row_file(const string &file_path, double threshold,
        librpa_int::atom_mapping<std::map<int, std::shared_ptr<librpa_int::ComplexMatrix>>>::pair_t_old &coulomb,
        const std::vector<atpair_t> &local_atpair, bool binary,
        const librpa_int::AtomicBasis &basis_aux)
{
    using librpa_int::ComplexMatrix;
    // cout << "Begin to read Vq from " << file_path << endl;
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    int n_irk_points_local;
    int n_irk_points;

    const auto atom_mu_part_range = basis_aux.get_part_range();

    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
        infile.read((char *)&n_irk_points, sizeof(int));
        infile.read((char *)&n_irk_points_local, sizeof(int));
    }
    else
    {
        infile.open(file_path);
        infile >> n_irk_points;
    }
    if (!infile.good()) return 1;

    if (binary)
    {
        std::set<int> coulomb_row_need;
        for (const auto &[I, _]: local_atpair)
        {
            const auto brow = atom_mu_part_range[I];
            const auto nb = basis_aux[I];
            for (size_t ir = 0; ir < nb; ir++)
            {
                coulomb_row_need.insert(brow + ir);
            }
        }

        int nbasbas, brow, erow, bcol, ecol, iq;
        double q_weight;
        for (int i_irk = 0; i_irk < n_irk_points_local; i_irk++)
        {
            infile.read((char *)&nbasbas, sizeof(int));
            infile.read((char *)&brow, sizeof(int));
            infile.read((char *)&erow, sizeof(int));
            infile.read((char *)&bcol, sizeof(int));
            infile.read((char *)&ecol, sizeof(int));
            infile.read((char *)&iq, sizeof(int));
            infile.read((char *)&q_weight, sizeof(double));

            brow--;
            erow--;
            bcol--;
            ecol--;
            iq--;

            for (const auto &ap : local_atpair)
            {
                auto I = ap.first;
                auto J = ap.second;
                if (coulomb[I][J].count(iq) == 0)
                {
                    std::shared_ptr<ComplexMatrix> vq_ptr = std::make_shared<ComplexMatrix>();
                    vq_ptr->create(basis_aux[I], basis_aux[J]);
                    coulomb[I][J][iq] = vq_ptr;
                }
            }

            const auto ncol = ecol - bcol + 1;

            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                std::vector<std::complex<double>> tmp_row(ncol);
                infile.read((char *) tmp_row.data(), 2 * ncol * sizeof(double));

                if (coulomb_row_need.count(i_mu))
                {
                    int I_loc, mu_loc;
                    basis_aux.get_local_index(i_mu, I_loc, mu_loc);
                    for (auto &Jp : coulomb[I_loc])
                    {
                        int J = Jp.first;
                        int Jb = atom_mu_part_range[J];
                        int Je = atom_mu_part_range[J] + basis_aux[J] - 1;

                        if (ecol >= Jb && bcol <= Je)
                        {
                            int start_point = (bcol <= Jb ? Jb : bcol);
                            int end_point = (ecol <= Je ? ecol : Je);
                            for (int i = start_point; i <= end_point; i++)
                            {
                                int J_loc, nu_loc;
                                basis_aux.get_local_index(i, J_loc, nu_loc);
                                // printf("|i: %d   J: %d   J_loc: %d, nu_loc:
                                // %d\n",i,J,J_loc,nu_loc);
                                assert(J == J_loc);
                                (*coulomb[I_loc][J_loc][iq])(mu_loc, nu_loc) = tmp_row[i - bcol];
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        string nbasbas, begin_row, end_row, begin_col, end_col, q1, q2, q3, vq_r, vq_i, q_num,
            q_weight;
        while (infile.peek() != EOF)
        {
            infile >> nbasbas >> begin_row >> end_row >> begin_col >> end_col;
            if (infile.peek() == EOF) break;
            if (!infile.good()) return 2;
            // cout << "vq range: " << begin_row << " ~ " << end_row << "  ,   " << begin_col << " ~
            // " << end_col << endl;
            infile >> q_num >> q_weight;
            if (!infile.good()) return 3;
            // int mu = stoi(nbasbas);
            // int nu = stoi(nbasbas);
            int brow = stoi(begin_row) - 1;
            int erow = stoi(end_row) - 1;
            int bcol = stoi(begin_col) - 1;
            int ecol = stoi(end_col) - 1;
            int iq = stoi(q_num) - 1;
            // cout<<file_path<<" iq:"<<iq<<"  qweight:"<<stod(q_weight)<<endl;

            //skip empty coulumb_file
            if ((erow - brow < 0) || (ecol - bcol < 0) || iq < 0) return 4;

            for (const auto &ap : local_atpair)
            {
                auto I = ap.first;
                auto J = ap.second;
                if (!coulomb[I][J].count(iq))
                {
                    std::shared_ptr<ComplexMatrix> vq_ptr = std::make_shared<ComplexMatrix>();
                    vq_ptr->create(basis_aux[I], basis_aux[J]);
                    // cout<<"  create  IJ: "<<I<<"  "<<J<<"   "<<atom_mu[I]<<"  "<<atom_mu[J];
                    coulomb[I][J][iq] = vq_ptr;
                }
            }

            std::set<int> coulomb_row_need;
            for (auto &[I, _] : coulomb)
            {
                const int st = atom_mu_part_range[I];
                const int ed = atom_mu_part_range[I] + basis_aux[I];
                for (int ir = st; ir < ed; ir++) coulomb_row_need.insert(ir);
            }

            // printf("   |process %d, coulomb_begin:  %d, size:
            // %d\n",para_mpi.get_myid(),*coulomb_row_need.begin(),coulomb_row_need.size());
            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                std::vector<std::complex<double>> tmp_row(ecol - bcol + 1);
                for (int i_nu = bcol; i_nu <= ecol; i_nu++)
                {
                    infile >> vq_r >> vq_i;
                    if (!infile.good()) return 4;

                    tmp_row[i_nu - bcol] = std::complex<double>(stod(vq_r), stod(vq_i));

                }
                if (coulomb_row_need.count(i_mu))
                {
                    int I_loc,mu_loc;
                    basis_aux.get_local_index(i_mu, I_loc,mu_loc);
                    // int bI=atom_mu_part_range[I_loc];
                    for(auto &Jp:coulomb[I_loc] )
                    {
                        auto J = Jp.first;
                        int Jb = atom_mu_part_range[J];
                        int Je = atom_mu_part_range[J] + basis_aux[J] - 1;

                        if (ecol >= Jb && bcol <= Je)
                        {
                            int start_point = (bcol <= Jb ? Jb : bcol);
                            int end_point = (ecol <= Je ? ecol : Je);
                            for (int i = start_point; i <= end_point; i++)
                            {
                                int J_loc, nu_loc;
                                basis_aux.get_local_index(i,J_loc, nu_loc);
                                //printf("|i: %d   J: %d   J_loc: %d, nu_loc: %d\n",i,J,J_loc,nu_loc);
                                assert(J == static_cast<size_t>(J_loc));
                                (*coulomb[I_loc][J_loc][iq])(mu_loc, nu_loc) = tmp_row[i - bcol];
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}

size_t read_Vq_row(ReaderContext &ctx, const string &dir_path, const string &vq_fprefix, double threshold,
                   const std::vector<atpair_t> &local_atpair, bool is_cut_coulomb,
                   int reader_version, const bool use_shrink_basis)
{
    using std::cout;
    using std::endl;
    using librpa_int::ComplexMatrix;
    using librpa_int::atom_mapping;
    using namespace librpa_int::global;

    if (reader_version < 0)
    {
        reader_version = detect_coulomb_reader_version(dir_path, vq_fprefix);
    }
    lib_printf_root("Coulomb reader (%s): %s\n", vq_fprefix.c_str(),
                    reader_version == 1 ? "binary v1" : "legacy");

    if (reader_version == 1)
    {
        return read_Vq_row_v1(ctx,
            dir_path, vq_fprefix, threshold, local_atpair, is_cut_coulomb,
            use_shrink_basis);
    }
    if (reader_version != 0)
    {
        throw std::logic_error("Unknown Coulomb reader version " +
                               std::to_string(reader_version));
    }

    if (ctx.comm.is_root() && should_output()) cout << "Begin READ_Vq_Row" << endl;
    const auto &basis_aux = target_coulomb_basis(ctx, use_shrink_basis);
    std::set<int> local_I_set;
    for(auto &lap:local_atpair)
    {
        local_I_set.insert(lap.first);
        local_I_set.insert(lap.second);
    }

    size_t vq_save = 0;
    size_t vq_discard = 0;
    atom_mapping<std::map<int, std::shared_ptr<ComplexMatrix>>>::pair_t_old coulomb;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dir_path.c_str());
    std::vector<string> files;
    bool binary;
    bool binary_checked = false;

    //map<Vector3_Order<double>, ComplexMatrix> Vq_full;
    profiler.start("handle_Vq_row_file");
    while ((ptr = readdir(dir)) != NULL)
    {
        string fm(ptr->d_name);
        if (is_legacy_coulomb_filename(fm, vq_fprefix))
        {
            string file_path = dir_path + fm;
            if (!binary_checked)
            {
                binary = check_coulomb_file_binary(file_path);
                binary_checked = true;
                if (ctx.comm.myid == 0)
                {
                    const char *info = binary ? "Unformatted binary" : "ASCII format";
                    if (should_output()) cout << info << " V files detected" << endl;
                }
            }
            handle_Vq_row_file(file_path, threshold, coulomb, local_atpair, binary, basis_aux);
        }
    }
    profiler.stop("handle_Vq_row_file");

    // MYZ: now the map coulomb contains the complete atom-pair matrix.
    // Call the API to parse the data.
    // To reduce memory consumption during this process, we erase the data in temporary object
    // once it is parsed.
    auto it_I = coulomb.begin();
    profiler.start("set_aux_coulomb_k_atom_pair_out");
    while (it_I != coulomb.end())
    {
        auto I = it_I->first;
        auto it_J = it_I->second.begin();
        while (it_J != it_I->second.end())
        {
            auto J = it_J->first;
            auto it_iq = it_J->second.begin();
            while (it_iq != it_J->second.end())
            {
                auto iq = it_iq->first;
                auto &vq_ptr = it_iq->second;
                if (is_cut_coulomb)
                {
                    ctx.h.set_aux_cut_coulomb_k_atom_pair(iq, I, J, vq_ptr->nr, vq_ptr->nc, vq_ptr->real().c, vq_ptr->imag().c, threshold);
                }
                else
                {
                    ctx.h.set_aux_bare_coulomb_k_atom_pair(iq, I, J, vq_ptr->nr, vq_ptr->nc, vq_ptr->real().c, vq_ptr->imag().c, threshold);
                }
                it_iq = it_J->second.erase(it_iq);
            }
            it_J = it_I->second.erase(it_J);
        }
        it_I = coulomb.erase(it_I);
    }
    profiler.stop("set_aux_coulomb_k_atom_pair_out");

    // cout << "FINISH coulomb files reading!" << endl;

    closedir(dir);
    dir = NULL;

    // ofstream fs;
    // std::stringstream ss;
    // ss<<"out_coulomb_rank_"<<para_mpi.get_myid()<<".txt";
    // fs.open(ss.str());
    // for(auto &Ip:coulomb_mat)
    // {
    //     for(auto &Jp:Ip.second)
    //         for(auto &qp:Jp.second)
    //         {
    //             std::stringstream sm;
    //             sm<<"I,J "<<Ip.first<<"  "<<Jp.first;
    //             //printf("|process %d  I J: %d, %d\n",para_mpi.get_myid(),
    //             Ip.first,Jp.first);
    //             print_complex_matrix_file(sm.str().c_str(),(*qp.second),fs,false);
    //         }

    // }
    // fs.close();
    return vq_discard;
}

} // namespace librpa::reader
