#include <array>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../reader_sternheimer.h"
#include "../reader_sternheimer_qpoints.h"

namespace
{

class TempDirectory
{
public:
    TempDirectory()
    {
        const auto suffix = std::to_string(std::rand());
        path = std::filesystem::temp_directory_path() / ("librpa_st_qpoints_" + suffix);
        std::filesystem::create_directories(path);
    }

    ~TempDirectory() { std::filesystem::remove_all(path); }

    std::filesystem::path path;
};

void write_text(const std::filesystem::path &path, const std::string &text)
{
    std::ofstream output(path);
    output << text;
}

void require_condition(const bool condition, const std::string &message)
{
    if (!condition)
    {
        std::cerr << "requirement failed: " << message << std::endl;
        std::abort();
    }
}

template <typename Value>
void write_binary(std::ofstream &output, const Value &value)
{
    output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
}

template <typename Value>
void overwrite_binary(const std::filesystem::path &path, const std::streamoff offset,
                      const Value &value)
{
    std::fstream output(path, std::ios::in | std::ios::out | std::ios::binary);
    output.seekp(offset);
    output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
    require_condition(output.good(), "failed to overwrite binary fixture");
}

void write_minimal_coulomb_v1(const std::filesystem::path &path, const int iq)
{
    constexpr std::int32_t marker = -20129433;
    constexpr std::int32_t naux = 1;
    constexpr std::int32_t complex_flag = 1;
    constexpr std::int32_t natoms = 1;
    constexpr std::int32_t nblocks = 1;
    constexpr std::int32_t pair_index = 0;
    constexpr std::int64_t payload_offset = 40;
    const std::complex<double> payload = {1.0, 0.0};

    std::ofstream output(path, std::ios::binary);
    write_binary(output, marker);
    write_binary(output, static_cast<std::int32_t>(iq));
    write_binary(output, naux);
    write_binary(output, complex_flag);
    write_binary(output, natoms);
    write_binary(output, nblocks);
    write_binary(output, naux);
    write_binary(output, pair_index);
    write_binary(output, payload_offset);
    write_binary(output, payload);
}

void write_minimal_sternheimer_v1(const std::filesystem::path &path, const int iq, const int ifreq,
                                  const double omega, const double weight,
                                  const std::complex<double> payload)
{
    constexpr std::int32_t marker = -41073291;
    constexpr std::int32_t naux = 1;
    constexpr std::int32_t complex_flag = 1;
    constexpr std::int32_t natoms = 1;
    constexpr std::int32_t nblocks = 1;
    constexpr std::int32_t pair_index = 0;
    constexpr std::int64_t payload_offset = 60;

    std::ofstream output(path, std::ios::binary);
    write_binary(output, marker);
    write_binary(output, static_cast<std::int32_t>(iq));
    write_binary(output, static_cast<std::int32_t>(ifreq));
    write_binary(output, naux);
    write_binary(output, complex_flag);
    write_binary(output, natoms);
    write_binary(output, omega);
    write_binary(output, weight);
    write_binary(output, nblocks);
    write_binary(output, naux);
    write_binary(output, pair_index);
    write_binary(output, payload_offset);
    write_binary(output, payload);
}

struct CoulombBlock
{
    std::int32_t pair_index;
    std::complex<double> value;
};

struct DenseCoulombBlock
{
    std::int32_t pair_index;
    std::vector<std::complex<double>> values;
};

void write_dense_coulomb_v1_shard(const std::filesystem::path &path, const int iq,
                                  const std::vector<std::int32_t> &atom_naux,
                                  const std::int32_t value_flag,
                                  const std::vector<DenseCoulombBlock> &blocks)
{
    constexpr std::int32_t marker = -20129433;
    const auto natoms = static_cast<std::int32_t>(atom_naux.size());
    std::int32_t naux = 0;
    for (const auto atom_size : atom_naux)
    {
        naux += atom_size;
    }
    const auto nblocks = static_cast<std::int32_t>(blocks.size());
    const auto value_bytes = value_flag == 1 ? sizeof(std::complex<double>) : sizeof(double);
    std::int64_t payload_offset = 6 * sizeof(std::int32_t) + natoms * sizeof(std::int32_t) +
                                  nblocks * (sizeof(std::int32_t) + sizeof(std::int64_t));

    std::ofstream output(path, std::ios::binary);
    write_binary(output, marker);
    write_binary(output, static_cast<std::int32_t>(iq));
    write_binary(output, naux);
    write_binary(output, value_flag);
    write_binary(output, natoms);
    write_binary(output, nblocks);
    for (const auto atom_size : atom_naux)
    {
        write_binary(output, atom_size);
    }
    for (const auto &block : blocks)
    {
        write_binary(output, block.pair_index);
        write_binary(output, payload_offset);
        payload_offset += static_cast<std::int64_t>(block.values.size() * value_bytes);
    }
    for (const auto &block : blocks)
    {
        for (const auto value : block.values)
        {
            if (value_flag == 1)
            {
                write_binary(output, value);
            }
            else
            {
                require_condition(std::abs(value.imag()) < 1.0e-15,
                                  "real fixture has an imaginary component");
                write_binary(output, value.real());
            }
        }
    }
    require_condition(output.good(), "failed to write dense Coulomb fixture");
}

void write_two_atom_coulomb_v1_shard(const std::filesystem::path &path, const int iq,
                                     const std::vector<CoulombBlock> &blocks)
{
    constexpr std::int32_t marker = -20129433;
    constexpr std::int32_t naux = 2;
    constexpr std::int32_t complex_flag = 1;
    constexpr std::int32_t natoms = 2;
    constexpr std::int32_t atom_naux = 1;
    const auto nblocks = static_cast<std::int32_t>(blocks.size());
    const std::int64_t payload_start = 6 * sizeof(std::int32_t) + natoms * sizeof(std::int32_t) +
                                       nblocks * (sizeof(std::int32_t) + sizeof(std::int64_t));

    std::ofstream output(path, std::ios::binary);
    write_binary(output, marker);
    write_binary(output, static_cast<std::int32_t>(iq));
    write_binary(output, naux);
    write_binary(output, complex_flag);
    write_binary(output, natoms);
    write_binary(output, nblocks);
    write_binary(output, atom_naux);
    write_binary(output, atom_naux);
    for (std::size_t iblock = 0; iblock != blocks.size(); ++iblock)
    {
        write_binary(output, blocks[iblock].pair_index);
        write_binary(output, payload_start +
                                 static_cast<std::int64_t>(iblock * sizeof(std::complex<double>)));
    }
    for (const auto &block : blocks)
    {
        write_binary(output, block.value);
    }
}

void require_throws(const std::function<void()> &operation, const std::string &message_fragment)
{
    try
    {
        operation();
    }
    catch (const std::exception &error)
    {
        if (std::string(error.what()).find(message_fragment) != std::string::npos)
        {
            return;
        }
        std::cerr << "unexpected error: " << error.what() << std::endl;
        std::abort();
    }
    std::cerr << "expected an exception containing: " << message_fragment << std::endl;
    std::abort();
}

void test_reads_normalized_manifest()
{
    TempDirectory temp;
    const auto manifest = temp.path / "qpoints.dat";
    write_text(manifest, "# iq qx qy qz qweight\n2 0.5 0 0 0.25\n3 0 0.5 0 0.75 ! row\n");

    const auto qpoints = driver::read_sternheimer_qpoint_manifest(manifest.string());
    require_condition(qpoints.size() == 2, "manifest q-point count");
    require_condition(qpoints[0].iq == 2, "first manifest iq");
    require_condition(qpoints[0].q == std::array<double, 3>{0.5, 0.0, 0.0},
                      "first manifest q coordinate");
    require_condition(std::abs(qpoints[0].weight - 0.25) < 1.0e-15,
                      "first manifest q weight");
    require_condition(qpoints[1].iq == 3, "second manifest iq");
    require_condition(qpoints[1].q == std::array<double, 3>{0.0, 0.5, 0.0},
                      "second manifest q coordinate");
    require_condition(std::abs(qpoints[1].weight - 0.75) < 1.0e-15,
                      "second manifest q weight");
}

void test_rejects_duplicate_iq()
{
    TempDirectory temp;
    const auto manifest = temp.path / "qpoints.dat";
    write_text(manifest, "2 0.5 0 0 0.5\n2 -0.5 0 0 0.5\n");
    require_throws([&]() { driver::read_sternheimer_qpoint_manifest(manifest.string()); },
                   "duplicate iq=2");
}

void test_rejects_unnormalized_or_nonpositive_weights()
{
    TempDirectory temp;
    const auto unnormalized = temp.path / "unnormalized.dat";
    write_text(unnormalized, "2 0.5 0 0 0.25\n3 0 0.5 0 0.5\n");
    require_throws([&]() { driver::read_sternheimer_qpoint_manifest(unnormalized.string()); },
                   "weights must sum to 1");

    const auto nonpositive = temp.path / "nonpositive.dat";
    write_text(nonpositive, "2 0.5 0 0 1.0\n3 0 0.5 0 0.0\n");
    require_throws([&]() { driver::read_sternheimer_qpoint_manifest(nonpositive.string()); },
                   "positive q weight");
}

void test_rejects_missing_coulomb_iq()
{
    TempDirectory temp;
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.5, 0.0, 0.0}, 1.0}};
    require_throws(
        [&]()
        {
            driver::validate_sternheimer_qpoint_input_files(
                qpoints, temp.path.string(), "v1_coulomb_full_iq_", "v1_sternheimer_chi0_iq_", 1);
        },
        "No Coulomb v1 files found");
}

void test_rejects_missing_frequency_files()
{
    TempDirectory temp;
    write_minimal_coulomb_v1(temp.path / "v1_coulomb_full_iq_2", 2);
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.5, 0.0, 0.0}, 1.0}};
    require_throws(
        [&]()
        {
            driver::validate_sternheimer_qpoint_input_files(
                qpoints, temp.path.string(), "v1_coulomb_full_iq_", "v1_sternheimer_chi0_iq_", 1);
        },
        "No Sternheimer chi0 v1 files found");
}

void test_skips_missing_gamma_files_when_gamma_is_excluded()
{
    TempDirectory temp;
    const std::vector<driver::SternheimerQPoint> qpoints{{1, {0.0, 0.0, 0.0}, 1.0}};

    driver::validate_sternheimer_qpoint_input_files(
        qpoints, temp.path.string(), "v1_coulomb_full_iq_", "v1_sternheimer_chi0_iq_", 1, false);
}

void test_partial_mode_requires_coulomb_but_not_aggregate_response_files()
{
    TempDirectory temp;
    write_minimal_coulomb_v1(temp.path / "v1_coulomb_full_iq_2", 2);
    const std::vector<driver::SternheimerQPoint> qpoints{{2, {0.5, 0.0, 0.0}, 1.0}};

    driver::validate_sternheimer_partial_qpoint_input_files(qpoints, temp.path.string(),
                                                            "v1_coulomb_full_iq_");
}

void test_requires_gamma_manifest_row_when_gamma_is_excluded()
{
    const std::vector<driver::SternheimerQPoint> without_gamma{{2, {0.5, 0.0, 0.0}, 1.0}};
    require_throws([&]() { driver::validate_sternheimer_gamma_contract(without_gamma, false); },
                   "exactly one Gamma row");

    const std::vector<driver::SternheimerQPoint> with_gamma{{1, {0.0, 0.0, 0.0}, 0.25},
                                                            {2, {0.5, 0.0, 0.0}, 0.75}};
    driver::validate_sternheimer_gamma_contract(with_gamma, false);
}

void test_reads_one_sternheimer_response_from_explicit_path()
{
    TempDirectory temp;
    const auto response_path = temp.path / "representative_k17_ifreq2.bin";
    write_minimal_sternheimer_v1(response_path, 3, 2, 0.75, 0.125, {-4.0, 0.0});

    const auto response = driver::read_sternheimer_chi0_v1_matrix_file(response_path.string());
    assert(response.path == response_path.string());
    assert(response.iq == 3);
    assert(response.ifreq == 2);
    assert(std::abs(response.omega - 0.75) < 1.0e-15);
    assert(std::abs(response.weight - 0.125) < 1.0e-15);
    assert((response.atom_naux == std::vector<int>{1}));
    assert(response.matrix.nr == 1);
    assert(response.matrix.nc == 1);
    assert(std::abs(response.matrix(0, 0) - std::complex<double>(-4.0, 0.0)) < 1.0e-15);
}

void test_rejects_invalid_sternheimer_frequency_metadata()
{
    TempDirectory temp;
    const auto negative_omega = temp.path / "negative_omega.bin";
    write_minimal_sternheimer_v1(negative_omega, 3, 1, -0.25, 1.0, {1.0, 0.0});
    require_throws(
        [&]()
        {
            static_cast<void>(
                driver::read_sternheimer_chi0_v1_matrix_file(negative_omega.string()));
        },
        "finite non-negative omega");

    const auto zero_weight = temp.path / "zero_weight.bin";
    write_minimal_sternheimer_v1(zero_weight, 3, 1, 0.25, 0.0, {1.0, 0.0});
    require_throws(
        [&]()
        {
            static_cast<void>(driver::read_sternheimer_chi0_v1_matrix_file(zero_weight.string()));
        },
        "finite positive frequency weight");
}

void test_merges_coulomb_atom_pair_blocks_across_rank_shards()
{
    TempDirectory temp;
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2,
                                    {{0, {2.0, 0.0}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank3.dat", 2,
                                    {{1, {0.5, 0.25}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank9.dat", 2,
                                    {{2, {3.0, 0.0}}});

    const auto matrix =
        driver::read_coulomb_v1_full_matrix(temp.path.string(), "v1_coulomb_full_iq_", 2);
    require_condition(matrix.nr == 2, "merged matrix row count");
    require_condition(matrix.nc == 2, "merged matrix column count");
    require_condition(std::abs(matrix(0, 0) - std::complex<double>(2.0, 0.0)) < 1.0e-15,
                      "merged matrix first diagonal");
    require_condition(std::abs(matrix(0, 1) - std::complex<double>(0.5, 0.25)) < 1.0e-15,
                      "merged matrix upper off-diagonal");
    require_condition(std::abs(matrix(1, 0) - std::complex<double>(0.5, -0.25)) < 1.0e-15,
                      "merged matrix lower off-diagonal");
    require_condition(std::abs(matrix(1, 1) - std::complex<double>(3.0, 0.0)) < 1.0e-15,
                      "merged matrix second diagonal");
}

void test_reads_single_coulomb_v1_file()
{
    TempDirectory temp;
    write_minimal_coulomb_v1(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2);
    const auto matrix =
        driver::read_coulomb_v1_full_matrix(temp.path.string(), "v1_coulomb_full_iq_", 2);
    require_condition(matrix.nr == 1 && matrix.nc == 1, "single-file matrix dimensions");
    require_condition(std::abs(matrix(0, 0) - std::complex<double>(1.0, 0.0)) < 1.0e-15,
                      "single-file matrix value");
}

void test_reads_rectangular_complex_atom_blocks()
{
    TempDirectory temp;
    write_dense_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2, {2, 1}, 1,
                                 {{0, {{1.0, 0.0}, {0.25, 0.5}, {0.25, -0.5}, {2.0, 0.0}}},
                                  {1, {{0.75, 0.125}, {-0.5, 0.25}}},
                                  {2, {{3.0, 0.0}}}});

    const auto matrix =
        driver::read_coulomb_v1_full_matrix(temp.path.string(), "v1_coulomb_full_iq_", 2);
    require_condition(matrix.nr == 3 && matrix.nc == 3, "rectangular-block matrix dimensions");
    require_condition(std::abs(matrix(0, 0) - std::complex<double>(1.0, 0.0)) < 1.0e-15,
                      "rectangular-block first diagonal");
    require_condition(std::abs(matrix(0, 1) - std::complex<double>(0.25, 0.5)) < 1.0e-15,
                      "rectangular-block intra-atom upper element");
    require_condition(std::abs(matrix(1, 0) - std::complex<double>(0.25, -0.5)) < 1.0e-15,
                      "rectangular-block intra-atom lower element");
    require_condition(std::abs(matrix(0, 2) - std::complex<double>(0.75, 0.125)) < 1.0e-15,
                      "rectangular-block first inter-atom element");
    require_condition(std::abs(matrix(1, 2) - std::complex<double>(-0.5, 0.25)) < 1.0e-15,
                      "rectangular-block second inter-atom element");
    require_condition(std::abs(matrix(2, 0) - std::complex<double>(0.75, -0.125)) < 1.0e-15,
                      "rectangular-block conjugate first element");
    require_condition(std::abs(matrix(2, 1) - std::complex<double>(-0.5, -0.25)) < 1.0e-15,
                      "rectangular-block conjugate second element");
    require_condition(std::abs(matrix(2, 2) - std::complex<double>(3.0, 0.0)) < 1.0e-15,
                      "rectangular-block final diagonal");
}

void test_reads_real_coulomb_payload()
{
    TempDirectory temp;
    write_dense_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2, {1, 1}, 0,
                                 {{0, {{2.0, 0.0}}}, {1, {{-0.75, 0.0}}}, {2, {{4.0, 0.0}}}});

    const auto matrix =
        driver::read_coulomb_v1_full_matrix(temp.path.string(), "v1_coulomb_full_iq_", 2);
    require_condition(matrix.nr == 2 && matrix.nc == 2, "real matrix dimensions");
    require_condition(std::abs(matrix(0, 0) - std::complex<double>(2.0, 0.0)) < 1.0e-15,
                      "real matrix first diagonal");
    require_condition(std::abs(matrix(0, 1) - std::complex<double>(-0.75, 0.0)) < 1.0e-15,
                      "real matrix upper off-diagonal");
    require_condition(std::abs(matrix(1, 0) - std::complex<double>(-0.75, 0.0)) < 1.0e-15,
                      "real matrix lower off-diagonal");
    require_condition(std::abs(matrix(1, 1) - std::complex<double>(4.0, 0.0)) < 1.0e-15,
                      "real matrix second diagonal");
}

void test_rejects_duplicate_coulomb_block_across_rank_shards()
{
    TempDirectory temp;
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2,
                                    {{0, {2.0, 0.0}}, {1, {0.5, 0.25}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank1.dat", 2,
                                    {{1, {0.5, 0.25}}, {2, {3.0, 0.0}}});

    require_throws(
        [&]()
        {
            static_cast<void>(
                driver::read_coulomb_v1_full_matrix(temp.path.string(), "v1_coulomb_full_iq_", 2));
        },
        "duplicate atom-pair block across Coulomb v1 shards");
}

void test_rejects_missing_coulomb_block_across_rank_shards()
{
    TempDirectory temp;
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2,
                                    {{0, {2.0, 0.0}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank1.dat", 2,
                                    {{1, {0.5, 0.25}}});

    require_throws(
        [&]() {
            driver::validate_coulomb_v1_full_matrix_file(temp.path.string(), "v1_coulomb_full_iq_",
                                                         2);
        },
        "missing atom-pair block across Coulomb v1 shards");
}

void test_rejects_invalid_coulomb_payload_offset()
{
    TempDirectory temp;
    const auto path = temp.path / "v1_coulomb_full_iq_2_rank0.dat";
    write_minimal_coulomb_v1(path, 2);
    constexpr std::streamoff offset_field =
        7 * sizeof(std::int32_t) + sizeof(std::int32_t);
    overwrite_binary(path, offset_field, std::int64_t{0});

    require_throws(
        [&]()
        {
            driver::validate_coulomb_v1_full_matrix_file(
                temp.path.string(), "v1_coulomb_full_iq_", 2);
        },
        "invalid v1 byte offset");
}

void test_rejects_truncated_coulomb_payload()
{
    TempDirectory temp;
    const auto path = temp.path / "v1_coulomb_full_iq_2_rank0.dat";
    write_minimal_coulomb_v1(path, 2);
    std::filesystem::resize_file(path, 40 + sizeof(double));

    require_throws(
        [&]()
        {
            driver::validate_coulomb_v1_full_matrix_file(
                temp.path.string(), "v1_coulomb_full_iq_", 2);
        },
        "invalid v1 byte offset");
}

void test_rejects_overlapping_coulomb_payloads()
{
    TempDirectory temp;
    const auto first = temp.path / "v1_coulomb_full_iq_2_rank0.dat";
    write_two_atom_coulomb_v1_shard(first, 2,
                                    {{0, {2.0, 0.0}}, {1, {0.5, 0.25}}});
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank1.dat", 2,
                                    {{2, {3.0, 0.0}}});
    constexpr std::int64_t first_payload =
        6 * sizeof(std::int32_t) + 2 * sizeof(std::int32_t) +
        2 * (sizeof(std::int32_t) + sizeof(std::int64_t));
    constexpr std::streamoff second_offset_field =
        6 * sizeof(std::int32_t) + 2 * sizeof(std::int32_t) +
        (sizeof(std::int32_t) + sizeof(std::int64_t)) + sizeof(std::int32_t);
    overwrite_binary(first, second_offset_field, first_payload);

    require_throws(
        [&]()
        {
            driver::validate_coulomb_v1_full_matrix_file(
                temp.path.string(), "v1_coulomb_full_iq_", 2);
        },
        "overlapping v1 atom-pair blocks");
}

void test_rejects_invalid_coulomb_pair_index()
{
    TempDirectory temp;
    const auto path = temp.path / "v1_coulomb_full_iq_2_rank0.dat";
    write_minimal_coulomb_v1(path, 2);
    constexpr std::streamoff pair_index_field = 7 * sizeof(std::int32_t);
    overwrite_binary(path, pair_index_field, std::int32_t{1});

    require_throws(
        [&]()
        {
            driver::validate_coulomb_v1_full_matrix_file(
                temp.path.string(), "v1_coulomb_full_iq_", 2);
        },
        "invalid atom-pair index");
}

void test_rejects_inconsistent_coulomb_shard_metadata()
{
    TempDirectory temp;
    write_two_atom_coulomb_v1_shard(temp.path / "v1_coulomb_full_iq_2_rank0.dat", 2,
                                    {{0, {2.0, 0.0}}});
    const auto second = temp.path / "v1_coulomb_full_iq_2_rank1.dat";
    write_two_atom_coulomb_v1_shard(second, 2,
                                    {{1, {0.5, 0.25}}, {2, {3.0, 0.0}}});
    constexpr std::streamoff value_flag_field = 3 * sizeof(std::int32_t);
    overwrite_binary(second, value_flag_field, std::int32_t{0});

    require_throws(
        [&]()
        {
            driver::validate_coulomb_v1_full_matrix_file(
                temp.path.string(), "v1_coulomb_full_iq_", 2);
        },
        "Inconsistent metadata across Coulomb v1 shards");
}

}  // namespace

int main()
{
    test_reads_normalized_manifest();
    test_rejects_duplicate_iq();
    test_rejects_unnormalized_or_nonpositive_weights();
    test_rejects_missing_coulomb_iq();
    test_rejects_missing_frequency_files();
    test_skips_missing_gamma_files_when_gamma_is_excluded();
    test_partial_mode_requires_coulomb_but_not_aggregate_response_files();
    test_requires_gamma_manifest_row_when_gamma_is_excluded();
    test_reads_one_sternheimer_response_from_explicit_path();
    test_rejects_invalid_sternheimer_frequency_metadata();
    test_merges_coulomb_atom_pair_blocks_across_rank_shards();
    test_reads_single_coulomb_v1_file();
    test_reads_rectangular_complex_atom_blocks();
    test_reads_real_coulomb_payload();
    test_rejects_duplicate_coulomb_block_across_rank_shards();
    test_rejects_missing_coulomb_block_across_rank_shards();
    test_rejects_invalid_coulomb_payload_offset();
    test_rejects_truncated_coulomb_payload();
    test_rejects_overlapping_coulomb_payloads();
    test_rejects_invalid_coulomb_pair_index();
    test_rejects_inconsistent_coulomb_shard_metadata();
    return 0;
}
