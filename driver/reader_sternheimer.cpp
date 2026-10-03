#include "reader_sternheimer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "../src/io/fs.h"

namespace driver
{
namespace
{

constexpr std::int32_t kCoulombV1Marker = -20129433;
constexpr std::int32_t kSternheimerChi0V1Marker = -41073291;
constexpr std::int32_t kRealFlag = 0;
constexpr std::int32_t kComplexFlag = 1;
constexpr std::int64_t kMissingBlock = -1;

struct BlockRecord
{
    int pair_index = -1;
    std::int64_t offset = kMissingBlock;
};

struct BlockedMatrixFile
{
    std::string path;
    int marker = 0;
    int iq = 0;
    int ifreq = 0;
    int naux = 0;
    int value_flag = kComplexFlag;
    int natoms = 0;
    int nblocks = 0;
    double omega = 0.0;
    double weight = 1.0;
    std::vector<int> atom_naux;
    std::vector<BlockRecord> blocks;
};

template <typename Value>
Value read_scalar(std::ifstream &input, const std::string &path)
{
    Value value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(Value));
    if (!input.good())
    {
        throw std::runtime_error("Failed to read binary scalar from " + path);
    }
    return value;
}

template <typename Value>
void write_scalar(std::ofstream &output, const Value &value, const std::string &path)
{
    output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
    if (!output.good())
    {
        throw std::runtime_error("Failed to write binary scalar to " + path);
    }
}

std::size_t checked_matrix_size(const int nrow, const int ncol, const std::string &context)
{
    if (nrow < 0 || ncol < 0)
    {
        throw std::runtime_error(context + ": negative matrix dimension");
    }
    const auto rows = static_cast<std::size_t>(nrow);
    const auto cols = static_cast<std::size_t>(ncol);
    if (rows != 0 && cols > std::numeric_limits<std::size_t>::max() / rows)
    {
        throw std::runtime_error(context + ": matrix size overflows size_t");
    }
    return rows * cols;
}

std::size_t checked_upper_pair_count(const int natoms, const std::string &context)
{
    if (natoms <= 0)
    {
        throw std::runtime_error(context + ": non-positive atom count");
    }
    const auto n = static_cast<std::size_t>(natoms);
    if (n + 1 > std::numeric_limits<std::size_t>::max() / n)
    {
        throw std::runtime_error(context + ": atom-pair count overflows size_t");
    }
    return n * (n + 1) / 2;
}

std::streamoff checked_payload_size(const BlockedMatrixFile &file, const std::size_t nvalues)
{
    const auto value_bytes =
        file.value_flag == kComplexFlag ? sizeof(std::complex<double>) : sizeof(double);
    const auto max_bytes =
        static_cast<unsigned long long>(std::numeric_limits<std::streamoff>::max());
    if (static_cast<unsigned long long>(nvalues) > max_bytes / value_bytes)
    {
        throw std::runtime_error(file.path + ": v1 payload size overflows streamoff");
    }
    return static_cast<std::streamoff>(nvalues * value_bytes);
}

std::size_t upper_pair_index(const std::size_t iatom, const std::size_t jatom,
                             const std::size_t natoms)
{
    if (iatom > jatom)
    {
        throw std::runtime_error("upper_pair_index expects I <= J");
    }
    return iatom * natoms - iatom * (iatom - 1) / 2 + (jatom - iatom);
}

std::vector<std::pair<std::size_t, std::size_t>> make_atom_pairs(const int natoms)
{
    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    const auto n = static_cast<std::size_t>(natoms);
    pairs.reserve(checked_upper_pair_count(natoms, "v1 matrix"));
    for (std::size_t iatom = 0; iatom != n; ++iatom)
    {
        for (std::size_t jatom = iatom; jatom != n; ++jatom)
        {
            pairs.emplace_back(iatom, jatom);
        }
    }
    return pairs;
}

std::vector<int> make_atom_offsets(const std::vector<int> &atom_naux)
{
    std::vector<int> offsets(atom_naux.size() + 1, 0);
    for (std::size_t iatom = 0; iatom != atom_naux.size(); ++iatom)
    {
        if (atom_naux[iatom] <= 0)
        {
            throw std::runtime_error("Invalid non-positive per-atom auxiliary size");
        }
        offsets[iatom + 1] = offsets[iatom] + atom_naux[iatom];
    }
    return offsets;
}

BlockedMatrixFile read_coulomb_header(const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path.c_str(), std::ios::binary);
    BlockedMatrixFile file;
    file.path = path;
    file.marker = read_scalar<std::int32_t>(input, path);
    file.iq = read_scalar<std::int32_t>(input, path);
    file.naux = read_scalar<std::int32_t>(input, path);
    file.value_flag = read_scalar<std::int32_t>(input, path);
    file.natoms = read_scalar<std::int32_t>(input, path);
    file.nblocks = read_scalar<std::int32_t>(input, path);
    if (file.marker != kCoulombV1Marker)
    {
        throw std::runtime_error(path + ": invalid Coulomb v1 marker");
    }
    return file;
}

BlockedMatrixFile read_sternheimer_header(const std::string &path)
{
    librpa_int::require_readable_file(path);
    std::ifstream input(path.c_str(), std::ios::binary);
    BlockedMatrixFile file;
    file.path = path;
    file.marker = read_scalar<std::int32_t>(input, path);
    file.iq = read_scalar<std::int32_t>(input, path);
    file.ifreq = read_scalar<std::int32_t>(input, path);
    file.naux = read_scalar<std::int32_t>(input, path);
    file.value_flag = read_scalar<std::int32_t>(input, path);
    file.natoms = read_scalar<std::int32_t>(input, path);
    file.omega = read_scalar<double>(input, path);
    file.weight = read_scalar<double>(input, path);
    file.nblocks = read_scalar<std::int32_t>(input, path);
    if (file.marker != kSternheimerChi0V1Marker)
    {
        throw std::runtime_error(path + ": invalid Sternheimer chi0 v1 marker");
    }
    return file;
}

void validate_sternheimer_frequency_metadata(const BlockedMatrixFile &file)
{
    if (file.ifreq <= 0)
    {
        throw std::runtime_error(file.path +
                                 ": Sternheimer chi0 v1 requires a positive ifrequency");
    }
    if (!std::isfinite(file.omega) || file.omega < 0.0)
    {
        throw std::runtime_error(file.path +
                                 ": Sternheimer chi0 v1 requires a finite non-negative omega");
    }
    if (!std::isfinite(file.weight) || file.weight <= 0.0)
    {
        throw std::runtime_error(
            file.path +
            ": Sternheimer chi0 v1 requires a finite positive frequency weight");
    }
}

void read_atom_sizes_and_blocks(BlockedMatrixFile &file, std::ifstream &input)
{
    if (file.iq <= 0 || file.naux <= 0 || file.natoms <= 0 || file.nblocks < 0)
    {
        throw std::runtime_error(file.path + ": invalid v1 matrix dimensions");
    }
    if (file.value_flag != kRealFlag && file.value_flag != kComplexFlag)
    {
        throw std::runtime_error(file.path + ": invalid v1 value flag");
    }

    file.atom_naux.resize(static_cast<std::size_t>(file.natoms));
    std::int64_t naux_sum = 0;
    for (int &atom_aux : file.atom_naux)
    {
        atom_aux = read_scalar<std::int32_t>(input, file.path);
        if (atom_aux <= 0)
        {
            throw std::runtime_error(file.path + ": non-positive atom_naux entry");
        }
        naux_sum += atom_aux;
    }
    if (naux_sum != file.naux)
    {
        throw std::runtime_error(file.path + ": atom_naux does not sum to naux");
    }

    const auto npairs = checked_upper_pair_count(file.natoms, file.path);
    if (static_cast<std::size_t>(file.nblocks) > npairs)
    {
        throw std::runtime_error(file.path + ": block count exceeds atom-pair count");
    }
    file.blocks.resize(static_cast<std::size_t>(file.nblocks));
    std::vector<bool> seen(npairs, false);
    for (BlockRecord &block : file.blocks)
    {
        block.pair_index = read_scalar<std::int32_t>(input, file.path);
        block.offset = read_scalar<std::int64_t>(input, file.path);
        if (block.pair_index < 0 || static_cast<std::size_t>(block.pair_index) >= npairs)
        {
            throw std::runtime_error(file.path + ": invalid atom-pair index");
        }
        if (seen[static_cast<std::size_t>(block.pair_index)])
        {
            throw std::runtime_error(file.path + ": duplicate atom-pair block");
        }
        seen[static_cast<std::size_t>(block.pair_index)] = true;
    }

    const auto table_end_position = input.tellg();
    if (table_end_position == std::streampos(-1))
    {
        throw std::runtime_error(file.path + ": failed to determine v1 block-table end");
    }
    const auto table_end = static_cast<std::streamoff>(table_end_position);
    input.seekg(0, std::ios::end);
    const auto file_end_position = input.tellg();
    if (file_end_position == std::streampos(-1))
    {
        throw std::runtime_error(file.path + ": failed to determine v1 file size");
    }
    const auto file_size = static_cast<std::streamoff>(file_end_position);

    const auto atom_pairs = make_atom_pairs(file.natoms);
    const auto max_streamoff = std::numeric_limits<std::streamoff>::max();
    std::vector<std::pair<std::streamoff, std::streamoff>> payload_ranges;
    payload_ranges.reserve(file.blocks.size());
    for (const BlockRecord &block : file.blocks)
    {
        if (block.offset < 0 || static_cast<unsigned long long>(block.offset) >
                                    static_cast<unsigned long long>(max_streamoff))
        {
            throw std::runtime_error(file.path + ": invalid v1 byte offset");
        }
        const auto [iatom, jatom] = atom_pairs[static_cast<std::size_t>(block.pair_index)];
        const auto nvalues =
            checked_matrix_size(file.atom_naux[iatom], file.atom_naux[jatom], file.path);
        const auto payload_size = checked_payload_size(file, nvalues);
        const auto payload_offset = static_cast<std::streamoff>(block.offset);
        if (payload_offset < table_end || payload_offset > file_size ||
            payload_size > file_size - payload_offset)
        {
            std::ostringstream message;
            message << file.path << ": invalid v1 byte offset " << block.offset
                    << " for atom-pair index " << block.pair_index;
            throw std::runtime_error(message.str());
        }
        payload_ranges.emplace_back(payload_offset, payload_offset + payload_size);
    }
    std::sort(payload_ranges.begin(), payload_ranges.end());
    for (std::size_t index = 1; index != payload_ranges.size(); ++index)
    {
        if (payload_ranges[index].first < payload_ranges[index - 1].second)
        {
            throw std::runtime_error(file.path + ": overlapping v1 atom-pair blocks");
        }
    }
}

BlockedMatrixFile read_coulomb_file_metadata(const std::string &path)
{
    auto file = read_coulomb_header(path);
    std::ifstream input(path.c_str(), std::ios::binary);
    input.seekg(6 * static_cast<std::streamoff>(sizeof(std::int32_t)));
    read_atom_sizes_and_blocks(file, input);
    return file;
}

BlockedMatrixFile read_sternheimer_file_metadata(const std::string &path)
{
    auto file = read_sternheimer_header(path);
    validate_sternheimer_frequency_metadata(file);
    std::ifstream input(path.c_str(), std::ios::binary);
    input.seekg(6 * static_cast<std::streamoff>(sizeof(std::int32_t)) +
                2 * static_cast<std::streamoff>(sizeof(double)) +
                static_cast<std::streamoff>(sizeof(std::int32_t)));
    read_atom_sizes_and_blocks(file, input);
    return file;
}

std::vector<std::complex<double>> read_block_payload(std::ifstream &input,
                                                     const BlockedMatrixFile &file,
                                                     const BlockRecord &block,
                                                     const std::size_t nvalues)
{
    input.seekg(static_cast<std::streamoff>(block.offset));
    if (!input.good())
    {
        throw std::runtime_error(file.path + ": failed to seek atom-pair block");
    }

    std::vector<std::complex<double>> values(nvalues);
    if (file.value_flag == kComplexFlag)
    {
        input.read(reinterpret_cast<char *>(values.data()),
                   static_cast<std::streamsize>(nvalues * sizeof(std::complex<double>)));
    }
    else
    {
        std::vector<double> buffer(nvalues);
        input.read(reinterpret_cast<char *>(buffer.data()),
                   static_cast<std::streamsize>(nvalues * sizeof(double)));
        for (std::size_t i = 0; i != nvalues; ++i)
        {
            values[i] = {buffer[i], 0.0};
        }
    }
    if (!input.good())
    {
        throw std::runtime_error(file.path + ": failed to read atom-pair block payload");
    }
    return values;
}

librpa_int::ComplexMatrix read_dense_blocked_matrix(const BlockedMatrixFile &file)
{
    const auto atom_pairs = make_atom_pairs(file.natoms);
    const auto atom_offsets = make_atom_offsets(file.atom_naux);
    librpa_int::ComplexMatrix matrix(file.naux, file.naux);
    std::ifstream input(file.path.c_str(), std::ios::binary);

    for (const BlockRecord &block : file.blocks)
    {
        const auto [iatom, jatom] = atom_pairs[static_cast<std::size_t>(block.pair_index)];
        const int ioffset = atom_offsets[iatom];
        const int joffset = atom_offsets[jatom];
        const int inaux = file.atom_naux[iatom];
        const int jnaux = file.atom_naux[jatom];
        const auto nvalues = checked_matrix_size(inaux, jnaux, file.path);
        const auto values = read_block_payload(input, file, block, nvalues);

        for (int imu = 0; imu != inaux; ++imu)
        {
            for (int jmu = 0; jmu != jnaux; ++jmu)
            {
                const auto value =
                    values[static_cast<std::size_t>(imu) * static_cast<std::size_t>(jnaux) +
                           static_cast<std::size_t>(jmu)];
                matrix(ioffset + imu, joffset + jmu) = value;
                if (iatom != jatom)
                {
                    matrix(joffset + jmu, ioffset + imu) = std::conj(value);
                }
            }
        }
    }
    return matrix;
}

std::vector<BlockedMatrixFile> find_coulomb_files(const std::string &dir_path,
                                                  const std::string &prefix, const int iq)
{
    const auto files = librpa_int::discover_files_with_prefix(dir_path, prefix);
    if (files.empty())
    {
        throw std::runtime_error("No Coulomb v1 files found with prefix " + prefix);
    }

    std::vector<BlockedMatrixFile> matches;
    for (const auto &path : files)
    {
        const auto file = read_coulomb_file_metadata(path);
        if (file.iq == iq)
        {
            matches.push_back(file);
        }
    }
    if (matches.empty())
    {
        throw std::runtime_error("No Coulomb v1 file found for iq=" + std::to_string(iq));
    }

    std::sort(matches.begin(), matches.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.path < rhs.path; });
    const auto &reference = matches.front();
    const auto npairs = checked_upper_pair_count(reference.natoms, reference.path);
    std::vector<int> block_owner(npairs, -1);
    for (std::size_t ifile = 0; ifile != matches.size(); ++ifile)
    {
        const auto &file = matches[ifile];
        if (file.naux != reference.naux || file.value_flag != reference.value_flag ||
            file.natoms != reference.natoms || file.atom_naux != reference.atom_naux)
        {
            throw std::runtime_error(
                "Inconsistent metadata across Coulomb v1 shards for iq=" + std::to_string(iq));
        }
        for (const auto &block : file.blocks)
        {
            auto &owner = block_owner[static_cast<std::size_t>(block.pair_index)];
            if (owner >= 0)
            {
                throw std::runtime_error(
                    "duplicate atom-pair block across Coulomb v1 shards for iq=" +
                    std::to_string(iq));
            }
            owner = static_cast<int>(ifile);
        }
    }
    if (std::find(block_owner.begin(), block_owner.end(), -1) != block_owner.end())
    {
        throw std::runtime_error("missing atom-pair block across Coulomb v1 shards for iq=" +
                                 std::to_string(iq));
    }
    return matches;
}

librpa_int::ComplexMatrix read_dense_blocked_matrix(
    const std::vector<BlockedMatrixFile> &files)
{
    if (files.empty())
    {
        throw std::runtime_error("Cannot assemble a dense matrix from zero v1 shards");
    }
    const auto atom_pairs = make_atom_pairs(files.front().natoms);
    const auto atom_offsets = make_atom_offsets(files.front().atom_naux);
    librpa_int::ComplexMatrix matrix(files.front().naux, files.front().naux);
    for (const auto &file : files)
    {
        std::ifstream input(file.path.c_str(), std::ios::binary);
        for (const BlockRecord &block : file.blocks)
        {
            const auto [iatom, jatom] =
                atom_pairs[static_cast<std::size_t>(block.pair_index)];
            const int ioffset = atom_offsets[iatom];
            const int joffset = atom_offsets[jatom];
            const int inaux = file.atom_naux[iatom];
            const int jnaux = file.atom_naux[jatom];
            const auto nvalues = checked_matrix_size(inaux, jnaux, file.path);
            const auto values = read_block_payload(input, file, block, nvalues);

            for (int imu = 0; imu != inaux; ++imu)
            {
                for (int jmu = 0; jmu != jnaux; ++jmu)
                {
                    const auto value =
                        values[static_cast<std::size_t>(imu) *
                                   static_cast<std::size_t>(jnaux) +
                               static_cast<std::size_t>(jmu)];
                    matrix(ioffset + imu, joffset + jmu) = value;
                    if (iatom != jatom)
                    {
                        matrix(joffset + jmu, ioffset + imu) = std::conj(value);
                    }
                }
            }
        }
    }
    return matrix;
}

std::vector<BlockedMatrixFile> find_sternheimer_files(const std::string &dir_path,
                                                      const std::string &prefix, const int iq)
{
    const auto files = librpa_int::discover_files_with_prefix(dir_path, prefix);
    if (files.empty())
    {
        throw std::runtime_error("No Sternheimer chi0 v1 files found with prefix " + prefix);
    }

    std::vector<BlockedMatrixFile> matches;
    for (const auto &path : files)
    {
        auto metadata = read_sternheimer_file_metadata(path);
        if (metadata.iq != iq)
        {
            continue;
        }
        if (metadata.value_flag != kComplexFlag)
        {
            throw std::runtime_error(path + ": Sternheimer chi0 v1 must be complex-valued");
        }
        matches.push_back(std::move(metadata));
    }
    if (matches.empty())
    {
        throw std::runtime_error("No Sternheimer chi0 v1 file found for iq=" + std::to_string(iq));
    }

    std::sort(matches.begin(), matches.end(),
              [](const auto &lhs, const auto &rhs) { return lhs.ifreq < rhs.ifreq; });
    const auto &reference = matches.front();
    for (const auto &file : matches)
    {
        if (file.naux != reference.naux || file.natoms != reference.natoms ||
            file.atom_naux != reference.atom_naux)
        {
            throw std::runtime_error(
                "Inconsistent metadata across Sternheimer chi0 v1 files for iq=" +
                std::to_string(iq));
        }
    }
    for (std::size_t i = 0; i != matches.size(); ++i)
    {
        if (matches[i].ifreq != static_cast<int>(i + 1))
        {
            throw std::runtime_error("Sternheimer chi0 v1 files are not contiguous in ifrequency");
        }
    }
    return matches;
}

}  // namespace

librpa_int::ComplexMatrix read_coulomb_v1_full_matrix(const std::string &dir_path,
                                                      const std::string &prefix, const int iq)
{
    return read_dense_blocked_matrix(find_coulomb_files(dir_path, prefix, iq));
}

void validate_coulomb_v1_full_matrix_file(const std::string &dir_path, const std::string &prefix,
                                          const int iq)
{
    static_cast<void>(find_coulomb_files(dir_path, prefix, iq));
}

SternheimerChi0V1Matrix read_sternheimer_chi0_v1_matrix_file(const std::string &path)
{
    const auto metadata = read_sternheimer_file_metadata(path);
    if (metadata.value_flag != kComplexFlag)
    {
        throw std::runtime_error(path + ": Sternheimer chi0 v1 must be complex-valued");
    }

    SternheimerChi0V1Matrix response;
    response.path = metadata.path;
    response.iq = metadata.iq;
    response.ifreq = metadata.ifreq;
    response.omega = metadata.omega;
    response.weight = metadata.weight;
    response.atom_naux = metadata.atom_naux;
    response.matrix = read_dense_blocked_matrix(metadata);
    return response;
}

void write_sternheimer_chi0_v1_matrix_file(const std::string &path,
                                            const SternheimerChi0V1Matrix &response)
{
    if (response.iq <= 0 || response.ifreq <= 0 || !std::isfinite(response.omega)
        || response.omega < 0.0 || !std::isfinite(response.weight) || response.weight <= 0.0
        || response.atom_naux.empty())
    {
        throw std::runtime_error(path + ": invalid Sternheimer chi0 v1 metadata");
    }
    const auto atom_offsets = make_atom_offsets(response.atom_naux);
    const int naux = atom_offsets.back();
    if (response.matrix.nr != naux || response.matrix.nc != naux)
    {
        throw std::runtime_error(path + ": Sternheimer chi0 matrix dimension does not match atom_naux");
    }
    double matrix_scale = 0.0;
    double hermiticity_residual = 0.0;
    for (int row = 0; row != naux; ++row)
    {
        for (int column = 0; column != naux; ++column)
        {
            matrix_scale = std::max(matrix_scale, std::abs(response.matrix(row, column)));
            hermiticity_residual =
                std::max(hermiticity_residual,
                         std::abs(response.matrix(row, column)
                                  - std::conj(response.matrix(column, row))));
        }
    }
    if (hermiticity_residual > 1.0e-10 * std::max(1.0, matrix_scale))
    {
        throw std::runtime_error(path + ": Sternheimer chi0 v1 output matrix is not Hermitian");
    }

    const auto atom_pairs = make_atom_pairs(static_cast<int>(response.atom_naux.size()));
    const std::int32_t nblocks = static_cast<std::int32_t>(atom_pairs.size());
    const std::int64_t payload_start =
        7 * static_cast<std::int64_t>(sizeof(std::int32_t))
        + 2 * static_cast<std::int64_t>(sizeof(double))
        + static_cast<std::int64_t>(response.atom_naux.size()) * sizeof(std::int32_t)
        + static_cast<std::int64_t>(atom_pairs.size())
              * (sizeof(std::int32_t) + sizeof(std::int64_t));
    std::vector<std::int64_t> payload_offsets;
    payload_offsets.reserve(atom_pairs.size());
    std::int64_t next_offset = payload_start;
    for (const auto &[iatom, jatom] : atom_pairs)
    {
        payload_offsets.push_back(next_offset);
        next_offset += static_cast<std::int64_t>(response.atom_naux[iatom])
                       * static_cast<std::int64_t>(response.atom_naux[jatom])
                       * static_cast<std::int64_t>(sizeof(std::complex<double>));
    }

    std::ofstream output(path.c_str(), std::ios::binary | std::ios::trunc);
    if (!output)
    {
        throw std::runtime_error("Cannot open Sternheimer chi0 v1 output file " + path);
    }
    write_scalar(output, kSternheimerChi0V1Marker, path);
    write_scalar(output, static_cast<std::int32_t>(response.iq), path);
    write_scalar(output, static_cast<std::int32_t>(response.ifreq), path);
    write_scalar(output, static_cast<std::int32_t>(naux), path);
    write_scalar(output, kComplexFlag, path);
    write_scalar(output, static_cast<std::int32_t>(response.atom_naux.size()), path);
    write_scalar(output, response.omega, path);
    write_scalar(output, response.weight, path);
    write_scalar(output, nblocks, path);
    for (const int atom_aux : response.atom_naux)
    {
        write_scalar(output, static_cast<std::int32_t>(atom_aux), path);
    }
    for (std::size_t pair_index = 0; pair_index != atom_pairs.size(); ++pair_index)
    {
        write_scalar(output, static_cast<std::int32_t>(pair_index), path);
        write_scalar(output, payload_offsets[pair_index], path);
    }
    for (const auto &[iatom, jatom] : atom_pairs)
    {
        const int ioffset = atom_offsets[iatom];
        const int joffset = atom_offsets[jatom];
        for (int imu = 0; imu != response.atom_naux[iatom]; ++imu)
        {
            for (int jmu = 0; jmu != response.atom_naux[jatom]; ++jmu)
            {
                write_scalar(output, response.matrix(ioffset + imu, joffset + jmu), path);
            }
        }
    }
}

std::vector<SternheimerChi0V1Matrix> read_sternheimer_chi0_v1_matrices(const std::string &dir_path,
                                                                       const std::string &prefix,
                                                                       const int iq)
{
    const auto files = find_sternheimer_files(dir_path, prefix, iq);
    std::vector<SternheimerChi0V1Matrix> responses;
    responses.reserve(files.size());
    for (const auto &metadata : files)
    {
        responses.push_back(read_sternheimer_chi0_v1_matrix_file(metadata.path));
    }
    return responses;
}

void validate_sternheimer_chi0_v1_files(const std::string &dir_path, const std::string &prefix,
                                        const int iq, const int expected_nfreq)
{
    const auto files = find_sternheimer_files(dir_path, prefix, iq);
    if (static_cast<int>(files.size()) != expected_nfreq)
    {
        throw std::runtime_error("Sternheimer chi0 frequency count (" +
                                 std::to_string(files.size()) + ") for iq=" + std::to_string(iq) +
                                 " does not match nfreq (" + std::to_string(expected_nfreq) + ")");
    }
}

}  // namespace driver
