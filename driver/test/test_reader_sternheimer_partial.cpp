#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../reader_sternheimer_partial.h"

namespace
{

class TempDirectory
{
public:
    TempDirectory()
    {
        const auto suffix = std::to_string(std::rand());
        path = std::filesystem::temp_directory_path() / ("librpa_st_partial_" + suffix);
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

void touch(const std::filesystem::path &path) { std::ofstream output(path, std::ios::binary); }

template <typename Value>
void write_binary(std::ofstream &output, const Value &value)
{
    output.write(reinterpret_cast<const char *>(&value), sizeof(Value));
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

void test_reads_comments_blank_lines_and_relative_paths()
{
    TempDirectory temp;
    std::filesystem::create_directories(temp.path / "responses");
    touch(temp.path / "responses" / "q2_k7_f0.bin");
    touch(temp.path / "responses" / "q2_k7_f1.bin");
    const auto manifest = temp.path / "v1_sternheimer_partial_manifest.dat";
    write_text(manifest,
               "# iq ik_full ifreq response_file\n"
               "\n"
               "2 7 1 responses/q2_k7_f0.bin\n"
               "2 7 2 responses/q2_k7_f1.bin ! second frequency\n");

    const auto records = driver::read_sternheimer_partial_manifest(manifest.string());
    assert(records.size() == 2);
    assert(records[0].iq == 2);
    assert(records[0].ik_full == 7);
    assert(records[0].ifreq == 1);
    assert(records[0].response_path ==
           std::filesystem::weakly_canonical(temp.path / "responses" / "q2_k7_f0.bin").string());
    assert(records[1].ifreq == 2);
}

void test_accepts_absolute_response_path()
{
    TempDirectory temp;
    const auto response = temp.path / "absolute.bin";
    touch(response);
    const auto manifest = temp.path / "manifest.dat";
    write_text(manifest, "1 0 1 " + response.string() + "\n");

    const auto records = driver::read_sternheimer_partial_manifest(manifest.string());
    assert(records.size() == 1);
    assert(records[0].response_path == std::filesystem::weakly_canonical(response).string());
}

void test_rejects_duplicate_key_with_line_number()
{
    TempDirectory temp;
    touch(temp.path / "a.bin");
    touch(temp.path / "b.bin");
    const auto manifest = temp.path / "manifest.dat";
    write_text(manifest, "2 7 1 a.bin\n2 7 1 b.bin\n");

    require_throws([&]() { driver::read_sternheimer_partial_manifest(manifest.string()); },
                   "line 2: duplicate (iq, ik_full, ifreq)=(2, 7, 1)");
}

void test_rejects_malformed_and_extra_fields()
{
    TempDirectory temp;
    const auto malformed = temp.path / "malformed.dat";
    write_text(malformed, "2 7 response.bin\n");
    require_throws([&]() { driver::read_sternheimer_partial_manifest(malformed.string()); },
                   "line 1 must contain iq ik_full ifreq response_file");

    touch(temp.path / "response.bin");
    const auto extra = temp.path / "extra.dat";
    write_text(extra, "2 7 1 response.bin extra\n");
    require_throws([&]() { driver::read_sternheimer_partial_manifest(extra.string()); },
                   "line 1 contains extra fields");
}

void test_rejects_invalid_indices()
{
    TempDirectory temp;
    touch(temp.path / "response.bin");

    const auto bad_iq = temp.path / "bad_iq.dat";
    write_text(bad_iq, "0 7 1 response.bin\n");
    require_throws([&]() { driver::read_sternheimer_partial_manifest(bad_iq.string()); },
                   "positive one-based iq");

    const auto bad_k = temp.path / "bad_k.dat";
    write_text(bad_k, "1 -1 1 response.bin\n");
    require_throws([&]() { driver::read_sternheimer_partial_manifest(bad_k.string()); },
                   "non-negative zero-based ik_full");

    const auto bad_freq = temp.path / "bad_freq.dat";
    write_text(bad_freq, "1 0 0 response.bin\n");
    require_throws([&]() { driver::read_sternheimer_partial_manifest(bad_freq.string()); },
                   "positive one-based ifreq");
}

void test_rejects_missing_response_file()
{
    TempDirectory temp;
    const auto manifest = temp.path / "manifest.dat";
    write_text(manifest, "1 0 1 missing.bin\n");
    require_throws([&]() { driver::read_sternheimer_partial_manifest(manifest.string()); },
                   "missing.bin");
}

void test_rejects_empty_manifest()
{
    TempDirectory temp;
    const auto manifest = temp.path / "manifest.dat";
    write_text(manifest, "# no data\n\n");
    require_throws([&]() { driver::read_sternheimer_partial_manifest(manifest.string()); },
                   "partial manifest is empty");
}

void test_partial_task_requires_explicit_qpoint_manifest()
{
    driver::validate_sternheimer_partial_task_contract("qpoints.dat", "partial.dat");
    driver::validate_sternheimer_partial_task_contract(
        "qpoints.dat", "partial.dat", "routes.dat");
    driver::validate_sternheimer_partial_task_contract("", "");
    require_throws([]() { driver::validate_sternheimer_partial_task_contract("", "partial.dat"); },
                   "requires fn_sternheimer_qpoints");
    require_throws(
        []() { driver::validate_sternheimer_partial_task_contract("qpoints.dat", "", "routes.dat"); },
        "requires fn_sternheimer_partial_manifest");
}

void test_reads_full_kpoint_manifest_in_abacus_index_order()
{
    TempDirectory temp;
    const auto manifest = temp.path / "v1_sternheimer_full_kpoints.dat";
    write_text(manifest,
               "# ik_full kx ky kz\n"
               "2 0 0.5 0\n"
               "0 0 0 0\n"
               "1 0.5 0 0\n");

    const auto points = driver::read_sternheimer_full_kpoint_manifest(manifest.string());
    assert(points.size() == 3);
    assert(points[0].ik_full == 0);
    assert((points[0].k == std::array<double, 3>{0.0, 0.0, 0.0}));
    assert(points[1].ik_full == 1);
    assert((points[1].k == std::array<double, 3>{0.5, 0.0, 0.0}));
    assert(points[2].ik_full == 2);
    assert((points[2].k == std::array<double, 3>{0.0, 0.5, 0.0}));
}

void test_rejects_noncontiguous_full_kpoint_manifest()
{
    TempDirectory temp;
    const auto manifest = temp.path / "v1_sternheimer_full_kpoints.dat";
    write_text(manifest, "0 0 0 0\n2 0.5 0 0\n");
    require_throws(
        [&]() { driver::read_sternheimer_full_kpoint_manifest(manifest.string()); },
        "indices are not contiguous from zero");
}

void test_reads_versioned_fixed_q_inverse_routes()
{
    TempDirectory temp;
    const auto manifest = temp.path / "v1_sternheimer_symmetry_routes.dat";
    write_text(manifest,
               "version 1\n"
               "# iq representative_ik member_ik spatial_isym time_reversal fold_Gx fold_Gy fold_Gz\n"
               "1 3 6 2 1 0 -1 0\n"
               "1 0 0 0 0 0 0 0\n"
               "4 1 1 0 0 0 0 0\n");

    const auto routes = driver::read_sternheimer_fixed_q_route_manifest(manifest.string());
    assert(routes.size() == 3);
    assert(routes[0].iq == 1);
    assert(routes[0].representative_ik_full == 0);
    assert(routes[0].member_ik_full == 0);
    assert(routes[0].inverse_route.spatial_isym == 0);
    assert(!routes[0].inverse_route.time_reversal);
    assert(routes[1].iq == 1);
    assert(routes[1].representative_ik_full == 3);
    assert(routes[1].member_ik_full == 6);
    assert(routes[1].inverse_route.spatial_isym == 2);
    assert(routes[1].inverse_route.time_reversal);
    assert(routes[1].inverse_route.fold_G.x == 0);
    assert(routes[1].inverse_route.fold_G.y == -1);
    assert(routes[1].inverse_route.fold_G.z == 0);
    assert(routes[2].iq == 4);
}

void test_rejects_route_version_and_duplicate_q_member()
{
    TempDirectory temp;
    const auto bad_version = temp.path / "bad_version.dat";
    write_text(bad_version, "version 2\n1 0 0 0 0 0 0 0\n");
    require_throws(
        [&]() { driver::read_sternheimer_fixed_q_route_manifest(bad_version.string()); },
        "unsupported Sternheimer fixed-q route version");

    const auto duplicate = temp.path / "duplicate.dat";
    write_text(duplicate,
               "version 1\n"
               "1 0 0 0 0 0 0 0\n"
               "1 1 0 0 0 0 0 0\n");
    require_throws(
        [&]() { driver::read_sternheimer_fixed_q_route_manifest(duplicate.string()); },
        "duplicate (iq, member_ik)");
}

void test_reads_versioned_discrete_qstar_inverse_routes()
{
    TempDirectory temp;
    const auto manifest = temp.path / "v1_sternheimer_qstar_routes.dat";
    write_text(manifest,
               "version 1\n"
               "# representative_iq member_iq spatial_isym time_reversal fold_Gx fold_Gy fold_Gz\n"
               "1 1 0 0 0 0 0\n"
               "2 2 0 0 0 0 0\n"
               "2 3 4 1 1 -1 0\n");

    const auto routes = driver::read_sternheimer_qstar_route_manifest(manifest.string());
    assert(routes.size() == 3);
    assert(routes[0].representative_iq == 1);
    assert(routes[0].member_iq == 1);
    assert(routes[1].representative_iq == 2);
    assert(routes[1].member_iq == 2);
    assert(routes[2].representative_iq == 2);
    assert(routes[2].member_iq == 3);
    assert(routes[2].inverse_route.spatial_isym == 4);
    assert(routes[2].inverse_route.time_reversal);
    assert(routes[2].inverse_route.fold_G.x == 1);
    assert(routes[2].inverse_route.fold_G.y == -1);
    assert(routes[2].inverse_route.fold_G.z == 0);

    const auto duplicate = temp.path / "duplicate_qstar.dat";
    write_text(duplicate,
               "version 1\n"
               "1 1 0 0 0 0 0\n"
               "2 1 0 0 0 0 0\n");
    require_throws(
        [&]() { driver::read_sternheimer_qstar_route_manifest(duplicate.string()); },
        "duplicate member_iq");
}

void test_groups_explicit_response_files_by_q_and_frequency()
{
    TempDirectory temp;
    const auto k0 = temp.path / "q2_k0_f1.bin";
    const auto k3 = temp.path / "q2_k3_f1.bin";
    const auto f2 = temp.path / "q2_k0_f2.bin";
    write_minimal_sternheimer_v1(k0, 2, 1, 0.5, 0.125, {-1.0, 0.0});
    write_minimal_sternheimer_v1(k3, 2, 1, 0.5, 0.125, {-2.0, 0.0});
    write_minimal_sternheimer_v1(f2, 2, 2, 1.5, 0.25, {-3.0, 0.0});

    const std::vector<driver::SternheimerPartialResponse> records{
        {2, 0, 1, k0.string()},
        {2, 3, 1, k3.string()},
        {2, 0, 2, f2.string()},
    };
    const auto groups = driver::read_sternheimer_partial_response_groups(records);
    assert(groups.size() == 2);
    const auto &first = groups.at({2, 1});
    assert(first.iq == 2);
    assert(first.ifreq == 1);
    assert(std::abs(first.omega - 0.5) < 1.0e-15);
    assert(std::abs(first.weight - 0.125) < 1.0e-15);
    assert((first.atom_naux == std::vector<int>{1}));
    assert(first.representatives.size() == 2);
    assert(std::abs(first.representatives.at(0)(0, 0) - std::complex<double>(-1.0, 0.0)) < 1.0e-15);
    assert(std::abs(first.representatives.at(3)(0, 0) - std::complex<double>(-2.0, 0.0)) < 1.0e-15);
    assert(groups.at({2, 2}).representatives.size() == 1);
}

void test_grouping_rejects_binary_header_and_frequency_metadata_mismatch()
{
    TempDirectory temp;
    const auto wrong_iq = temp.path / "wrong_iq.bin";
    write_minimal_sternheimer_v1(wrong_iq, 3, 1, 0.5, 0.125, {-1.0, 0.0});
    require_throws(
        [&]() {
            driver::read_sternheimer_partial_response_groups({{2, 0, 1, wrong_iq.string()}});
        },
        "binary iq=3 does not match manifest iq=2");

    const auto k0 = temp.path / "q2_k0.bin";
    const auto k1 = temp.path / "q2_k1.bin";
    write_minimal_sternheimer_v1(k0, 2, 1, 0.5, 0.125, {-1.0, 0.0});
    write_minimal_sternheimer_v1(k1, 2, 1, 0.75, 0.125, {-2.0, 0.0});
    require_throws(
        [&]()
        {
            driver::read_sternheimer_partial_response_groups(
                {{2, 0, 1, k0.string()}, {2, 1, 1, k1.string()}});
        },
        "inconsistent omega");
}

void test_sternheimer_v1_writer_round_trips_complex_atom_blocks()
{
    TempDirectory temp;
    driver::SternheimerChi0V1Matrix expected;
    expected.iq = 4;
    expected.ifreq = 2;
    expected.omega = 0.75;
    expected.weight = 0.125;
    expected.atom_naux = {1, 2};
    expected.matrix = librpa_int::ComplexMatrix(3, 3);
    expected.matrix(0, 0) = {-1.0, 0.0};
    expected.matrix(0, 1) = {2.0, 3.0};
    expected.matrix(0, 2) = {-4.0, 5.0};
    expected.matrix(1, 0) = std::conj(expected.matrix(0, 1));
    expected.matrix(2, 0) = std::conj(expected.matrix(0, 2));
    expected.matrix(1, 1) = {6.0, 0.0};
    expected.matrix(1, 2) = {7.0, -8.0};
    expected.matrix(2, 1) = std::conj(expected.matrix(1, 2));
    expected.matrix(2, 2) = {9.0, 0.0};

    const auto path = temp.path / "roundtrip.bin";
    driver::write_sternheimer_chi0_v1_matrix_file(path.string(), expected);
    const auto actual = driver::read_sternheimer_chi0_v1_matrix_file(path.string());
    assert(actual.iq == expected.iq);
    assert(actual.ifreq == expected.ifreq);
    assert(actual.omega == expected.omega);
    assert(actual.weight == expected.weight);
    assert(actual.atom_naux == expected.atom_naux);
    for (int row = 0; row != 3; ++row)
    {
        for (int column = 0; column != 3; ++column)
        {
            assert(std::abs(actual.matrix(row, column) - expected.matrix(row, column)) < 1.0e-15);
        }
    }
}

void test_sternheimer_v1_writer_rejects_nonhermitian_matrix()
{
    TempDirectory temp;
    driver::SternheimerChi0V1Matrix response;
    response.iq = 1;
    response.ifreq = 1;
    response.omega = 0.5;
    response.weight = 0.25;
    response.atom_naux = {2};
    response.matrix = librpa_int::ComplexMatrix(2, 2);
    response.matrix(0, 0) = {-1.0, 0.0};
    response.matrix(0, 1) = {2.0, 3.0};
    response.matrix(1, 0) = {2.0, 3.0};
    response.matrix(1, 1) = {-4.0, 0.0};
    require_throws(
        [&]() {
            driver::write_sternheimer_chi0_v1_matrix_file(
                (temp.path / "nonhermitian.bin").string(), response);
        },
        "not Hermitian");
}

}  // namespace

int main()
{
    test_reads_comments_blank_lines_and_relative_paths();
    test_accepts_absolute_response_path();
    test_rejects_duplicate_key_with_line_number();
    test_rejects_malformed_and_extra_fields();
    test_rejects_invalid_indices();
    test_rejects_missing_response_file();
    test_rejects_empty_manifest();
    test_partial_task_requires_explicit_qpoint_manifest();
    test_reads_full_kpoint_manifest_in_abacus_index_order();
    test_rejects_noncontiguous_full_kpoint_manifest();
    test_reads_versioned_fixed_q_inverse_routes();
    test_rejects_route_version_and_duplicate_q_member();
    test_reads_versioned_discrete_qstar_inverse_routes();
    test_groups_explicit_response_files_by_q_and_frequency();
    test_grouping_rejects_binary_header_and_frequency_metadata_mismatch();
    test_sternheimer_v1_writer_round_trips_complex_atom_blocks();
    test_sternheimer_v1_writer_rejects_nonhermitian_matrix();
    return 0;
}
