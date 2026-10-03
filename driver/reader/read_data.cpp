#include "read_data.h"
#include "reader_context.h"
#include "../../src/api/instance_manager.h"
#include "../../src/api/dataset_helper.h"
#include <librpa_enums.h>

#include "reader_basis.h"
#include "reader_eigenvec.h"
#include "reader_lri.h"
#include "reader_coulomb.h"
#include "reader_structure.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <librpa.hpp>

#include "../../src/mpi/global_mpi.h"
#include "../../src/math/matrix.h"
#include "../../src/utils/constants.h"
#include "../../src/core/meanfield_mpi.h"
#include "../../src/io/aux_basis_summary.h"
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/io/stl_io_helper.h"
#include "../../src/utils/error.h"
#include "../../src/utils/profiler.h"
#include "../../src/utils/utils_mem.h"

#include "librpa_input.h"

namespace librpa::reader
{
using namespace librpa_int;
using std::string;
using std::ifstream;
constexpr double kBzSamplingWeightSumTol = 1e-6;
constexpr double kSymmetryKpointMatchTol = 1e-5;

bool nearly_same_kpoint(const librpa_int::Vector3_Order<double> &lhs,
                       const librpa_int::Vector3_Order<double> &rhs,
                       const double tol = kSymmetryKpointMatchTol)
{
    return std::abs(lhs.x - rhs.x) <= tol && std::abs(lhs.y - rhs.y) <= tol &&
           std::abs(lhs.z - rhs.z) <= tol;
}

librpa_int::Vector3_Order<double> convert_fractional_kpoint_to_klist_units(
    const librpa_int::Vector3_Order<double> &kfrac, const librpa_int::PeriodicBoundaryData &pbc)
{
    const auto &G = pbc.G;
    return {kfrac.x * G.e11 + kfrac.y * G.e21 + kfrac.z * G.e31,
            kfrac.x * G.e12 + kfrac.y * G.e22 + kfrac.z * G.e32,
            kfrac.x * G.e13 + kfrac.y * G.e23 + kfrac.z * G.e33};
}

std::string lowercase_token(std::string token)
{
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return token;
}

bool is_stru_symop_convention(const std::string &token)
{
    const auto convention = lowercase_token(token);
    return convention == "row" || convention == "col";
}

bool legacy_stru_tail_ends_at(const std::vector<std::string> &tokens, const std::size_t pos)
{
    return pos == tokens.size()
           || (pos + 1 < tokens.size() && is_stru_symop_convention(tokens[pos + 1]));
}

int parse_stru_int_token(const std::string &token, const std::string &context)
{
    try
    {
        std::size_t used = 0;
        const int value = std::stoi(token, &used);
        if (used == token.size())
        {
            return value;
        }
    }
    catch (const std::exception &)
    {
    }
    throw LIBRPA_RUNTIME_ERROR("Invalid integer in " + context + ": " + token);
}

double parse_stru_double_token(const std::string &token, const std::string &context)
{
    try
    {
        std::size_t used = 0;
        const double value = std::stod(token, &used);
        if (used == token.size())
        {
            return value;
        }
    }
    catch (const std::exception &)
    {
    }
    throw LIBRPA_RUNTIME_ERROR("Invalid floating-point value in " + context + ": " + token);
}

void sync_ibz_kpoints_from_mapping(ReaderContext &ctx, const std::vector<double> &kvecs,
                                          const std::vector<int> &map_q_ks,
                                          const int n_kpoints)
{
    if (kvecs.size() != static_cast<std::size_t>(3 * n_kpoints)
        || map_q_ks.size() != static_cast<std::size_t>(n_kpoints))
    {
        throw LIBRPA_RUNTIME_ERROR("invalid k-point buffer or k-to-q mapping size");
    }

    std::vector<int> q_ids;
    ctx.state.ibz_kpoints.clear();
    for (const int iq : map_q_ks)
    {
        if (iq < 0 || iq >= n_kpoints)
        {
            throw LIBRPA_RUNTIME_ERROR("k-to-q mapping index out of range");
        }
        if (std::find(q_ids.cbegin(), q_ids.cend(), iq) != q_ids.cend())
        {
            continue;
        }
        q_ids.emplace_back(iq);
        // ponytail: print-only cache from parsed kvecs; use pbc.klist_coul if exact
        // shifted q-vectors become required.
        ctx.state.ibz_kpoints.push_back({
            kvecs[3 * iq],
            kvecs[3 * iq + 1],
            kvecs[3 * iq + 2]});
    }
    ctx.state.n_ibz_kpoints = static_cast<int>(ctx.state.ibz_kpoints.size());
}

bool try_legacy_stru_kpoint_layout(const std::vector<std::string> &tokens,
                                   const std::size_t kvec_pos,
                                   const int n_k_rows,
                                   const int nk_full,
                                   const bool with_mapping,
                                   int &selected_n_k_rows,
                                   bool &selected_has_mapping)
{
    if (n_k_rows <= 0)
    {
        return false;
    }
    const auto after_k_rows = kvec_pos + static_cast<std::size_t>(3 * n_k_rows);
    if (after_k_rows > tokens.size())
    {
        return false;
    }
    const auto end = with_mapping
                         ? after_k_rows + static_cast<std::size_t>(nk_full)
                         : after_k_rows;
    if (end > tokens.size() || !legacy_stru_tail_ends_at(tokens, end))
    {
        return false;
    }

    selected_n_k_rows = n_k_rows;
    selected_has_mapping = with_mapping;
    return true;
}

void read_scf_occ_eigenvalues(const string &file_path, MeanField &mf, bool use_spinor_wfc)
{
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string ks, ss, a, ws, es, d;
    int n_kpoints, n_spins, n_states, n_basis_wfc;
    double efermi;
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_states;
    infile >> n_basis_wfc;
    infile >> efermi;

    int n_spinor = 1;
    int n_basis_ao = n_basis_wfc;
    if (use_spinor_wfc)
    {
        assert(n_spins == 1);
        assert(n_basis_wfc % 2 == 0 && "Error: nbasis is not even when SOC!");
        n_spinor = 2;
        n_basis_ao = n_basis_wfc / 2;
    }

    mf = MeanField();
    mf.set(n_spins, n_kpoints, n_states, n_basis_ao, n_spinor);
    mf.get_efermi() = efermi;

    auto &eskb = mf.get_eigenvals();
    auto &wskb = mf.get_weight();

    int iline = 6;
    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            infile >> ks >> ss;
            if (!infile.good())
            {
                throw std::logic_error("Error in reading k- and spin- index: line " +
                                       std::to_string(iline) + ", file: " + file_path);
            }
            iline++;
            int k_index = stoi(ks) - 1;
            for (int i = 0; i != n_states; i++)
            {
                infile >> a >> ws >> es >> d;
                if (!infile.good())
                {
                    throw std::logic_error("Error in reading band energy and occupation: line " +
                                           std::to_string(iline) + ", file: " + file_path);
                }
                iline++;
                wskb[is](k_index, i) = stod(ws) / n_kpoints;
                eskb[is](k_index, i) = stod(es);
            }
        }
    }
}

void read_scf_occ_eigenvalues(const string &file_path, MeanField &mf, bool use_spinor_wfc,
                              const std::vector<int> &source_ik_for_target,
                              const int source_n_kpoints_expected)
{
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string ks, ss, a, ws, es, d;
    int n_kpoints_source, n_spins, n_states, n_basis_wfc;
    double efermi;
    infile >> n_kpoints_source;
    infile >> n_spins;
    infile >> n_states;
    infile >> n_basis_wfc;
    infile >> efermi;
    if (!infile.good())
    {
        throw std::logic_error("Failed to read band_out header from " + file_path);
    }
    if (n_kpoints_source != source_n_kpoints_expected)
    {
        throw std::logic_error("band_out k-point count is inconsistent with k_path_info");
    }

    int n_spinor = 1;
    int n_basis_ao = n_basis_wfc;
    if (use_spinor_wfc)
    {
        assert(n_spins == 1);
        assert(n_basis_wfc % 2 == 0 && "Error: nbasis is not even when SOC!");
        n_spinor = 2;
        n_basis_ao = n_basis_wfc / 2;
    }

    const int n_kpoints_target = static_cast<int>(source_ik_for_target.size());
    mf = MeanField();
    mf.set(n_spins, n_kpoints_target, n_states, n_basis_ao, n_spinor);
    mf.get_efermi() = efermi;

    std::vector<int> source_to_target(n_kpoints_source, -1);
    for (int ik_target = 0; ik_target != n_kpoints_target; ++ik_target)
    {
        const int ik_source = source_ik_for_target[ik_target];
        if (ik_source < 0 || ik_source >= n_kpoints_source)
        {
            throw std::logic_error("Invalid PyATB source k-point index selected for band_out");
        }
        if (source_to_target[ik_source] >= 0)
        {
            throw std::logic_error("Duplicate PyATB source k-point selected for band_out");
        }
        source_to_target[ik_source] = ik_target;
    }

    auto &eskb = mf.get_eigenvals();
    auto &wskb = mf.get_weight();
    std::vector<int> target_hits(librpa_int::as_size(n_kpoints_target) *
                                     librpa_int::as_size(n_spins),
                                 0);

    int iline = 6;
    for (int ik_read = 0; ik_read != n_kpoints_source; ++ik_read)
    {
        for (int is = 0; is != n_spins; ++is)
        {
            infile >> ks >> ss;
            if (!infile.good())
            {
                throw std::logic_error("Error in reading k- and spin- index: line " +
                                       std::to_string(iline) + ", file: " + file_path);
            }
            iline++;
            const int k_index_source = stoi(ks) - 1;
            const int s_index = stoi(ss) - 1;
            if (s_index != is)
            {
                throw std::logic_error("band_out spin index is not in the expected order");
            }
            if (k_index_source < 0 || k_index_source >= n_kpoints_source)
            {
                throw std::logic_error("band_out k-point index is out of range");
            }
            const int k_index_target = source_to_target[k_index_source];
            for (int ib = 0; ib != n_states; ++ib)
            {
                infile >> a >> ws >> es >> d;
                if (!infile.good())
                {
                    throw std::logic_error("Error in reading band energy and occupation: line " +
                                           std::to_string(iline) + ", file: " + file_path);
                }
                iline++;
                if (k_index_target < 0) continue;
                target_hits[librpa_int::as_size(is) * librpa_int::as_size(n_kpoints_target) +
                            librpa_int::as_size(k_index_target)] = 1;
                wskb[is](k_index_target, ib) = stod(ws) / n_kpoints_target;
                eskb[is](k_index_target, ib) = stod(es);
            }
        }
    }
    for (int is = 0; is != n_spins; ++is)
    {
        for (int ik = 0; ik != n_kpoints_target; ++ik)
        {
            if (target_hits[librpa_int::as_size(is) * librpa_int::as_size(n_kpoints_target) +
                            librpa_int::as_size(ik)] == 0)
            {
                throw std::logic_error("band_out is missing a selected head/wing k-point");
            }
        }
    }
}

void read_bz_sampling(ReaderContext &ctx, const std::string &file_path)
{
    using namespace librpa_int;

    global::lib_printf_root("Reading Brillouin zone sampling file: %s\n", file_path.c_str());

    require_readable_file(file_path);
    ifstream infile;
    infile.open(file_path);
    if (!infile.good()) throw LIBRPA_RUNTIME_ERROR("Fail to open BZ sampling file " + file_path);

    int nk[3];
    for (int i = 0; i < 3; i++)
    {
        infile >> nk[i];
    }
    if (!infile.good() || nk[0] <= 0 || nk[1] <= 0 || nk[2] <= 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Invalid BZ sampling k-grid in " + file_path);
    }
    const int nk_full = nk[0] * nk[1] * nk[2];

    int n_kpoints_scf, nk_ibz;
    infile >> n_kpoints_scf >> nk_ibz;
    if (!infile.good())
    {
        throw LIBRPA_RUNTIME_ERROR("Fail to read BZ sampling k-point counts from " + file_path);
    }
    if (n_kpoints_scf <= 0 || nk_ibz <= 0)
        throw LIBRPA_RUNTIME_ERROR("BZ sampling k-point counts must be positive");
    if (n_kpoints_scf > nk_full)
        throw LIBRPA_RUNTIME_ERROR("SCF k-point count exceeds the full BZ grid size");
    if (nk_ibz > n_kpoints_scf)
        throw LIBRPA_RUNTIME_ERROR("Coulomb IBZ k-point count exceeds the SCF k-point count");
    if (ctx.state.n_kpoints > 0 && n_kpoints_scf != ctx.state.n_kpoints)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "BZ sampling SCF k-point count does not match band_out: "
            + std::to_string(n_kpoints_scf) + " != " + std::to_string(ctx.state.n_kpoints));
    }

    std::vector<double> kvecs(3 * n_kpoints_scf);
    std::vector<double> kweights(n_kpoints_scf);
    std::vector<int> map_q_ks(n_kpoints_scf, -1);
    std::vector<int> ibz_label_to_rep(nk_ibz, -1);
    std::vector<int> ibz_representatives;
    double weight_sum = 0.0;

    for (int i = 0; i != n_kpoints_scf; i++)
    {
        int ik_read, ik_ibz, ik_rep;
        double kfrac_x, kfrac_y, kfrac_z;
        infile >> ik_read >> kweights[i];
        infile >> kfrac_x >> kfrac_y >> kfrac_z;
        infile >> kvecs[3 * i] >> kvecs[3 * i + 1] >> kvecs[3 * i + 2];
        infile >> ik_ibz >> ik_rep;
        if (!infile.good())
        {
            throw LIBRPA_RUNTIME_ERROR(
                "Fail to read BZ sampling k-point row " + std::to_string(i + 1)
                + " from " + file_path);
        }
        if (ik_read != i + 1)
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling k-point index does not match row order");
        }
        if (!std::isfinite(kweights[i]) || kweights[i] < 0.0
            || !std::isfinite(kfrac_x) || !std::isfinite(kfrac_y) || !std::isfinite(kfrac_z)
            || !std::isfinite(kvecs[3 * i])
            || !std::isfinite(kvecs[3 * i + 1])
            || !std::isfinite(kvecs[3 * i + 2]))
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling k-point row contains an invalid number");
        }
        if (ik_ibz <= 0 || ik_ibz > nk_ibz)
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling IBZ index out of range");
        }
        if (ik_rep <= 0 || ik_rep > n_kpoints_scf)
        {
            throw LIBRPA_RUNTIME_ERROR("BZ sampling representative k-point index out of range");
        }
        map_q_ks[i] = ik_rep - 1;
        auto &label_rep = ibz_label_to_rep[static_cast<std::size_t>(ik_ibz - 1)];
        if (label_rep < 0)
        {
            label_rep = map_q_ks[i];
        }
        else if (label_rep != map_q_ks[i])
        {
            throw LIBRPA_RUNTIME_ERROR(
                "BZ sampling irreducible Coulomb k-point label maps to multiple representatives");
        }
        if (std::find(ibz_representatives.cbegin(), ibz_representatives.cend(), map_q_ks[i])
            == ibz_representatives.cend())
        {
            ibz_representatives.emplace_back(map_q_ks[i]);
        }
        weight_sum += kweights[i];
    }
    infile.close();

    if (std::abs(weight_sum - 1.0) > kBzSamplingWeightSumTol)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "BZ sampling SCF k-point weights do not sum to 1: "
            + std::to_string(weight_sum));
    }
    const bool partial_sternheimer_input = ctx.params.sternheimer_partial_input;
    if (!partial_sternheimer_input)
    {
        if (ibz_representatives.size() != static_cast<std::size_t>(nk_ibz))
        {
            throw LIBRPA_RUNTIME_ERROR(
                "BZ sampling representative count does not match Coulomb IBZ count");
        }
        if (std::find(ibz_label_to_rep.cbegin(), ibz_label_to_rep.cend(), -1)
            != ibz_label_to_rep.cend())
        {
            throw LIBRPA_RUNTIME_ERROR(
                "BZ sampling does not contain every irreducible Coulomb k-point label");
        }
    }
    else if (ibz_representatives.size() != static_cast<std::size_t>(nk_ibz)
             || std::find(ibz_label_to_rep.cbegin(), ibz_label_to_rep.cend(), -1)
                    != ibz_label_to_rep.cend())
    {
        global::lib_printf_root(
            "Sternheimer partial-response input accepts incomplete generic Coulomb-IBZ labels; "
            "the explicit q-point manifest selects the Coulomb matrices.\n");
    }

    if (partial_sternheimer_input)
    {
        std::iota(map_q_ks.begin(), map_q_ks.end(), 0);
        global::lib_printf_root(
            "Sternheimer partial-response input uses all listed SCF k-points as "
            "symmetry-context representatives.\n");
    }

    ctx.h.set_kgrids_kvec(nk[0], nk[1], nk[2], kvecs, kweights);
    ctx.h.set_kq_mapping(map_q_ks);
    sync_ibz_kpoints_from_mapping(ctx, kvecs, map_q_ks, n_kpoints_scf);
}

void read_bz_sampling_from_stru(ReaderContext &ctx, const std::string &file_path)
{
    using namespace librpa_int;

    global::lib_printf_root("Fallback reading Brillouin zone sampling from structure file: %s\n",
                            file_path.c_str());

    require_readable_file(file_path);
    ifstream infile(file_path);
    if (!infile.good())
    {
        throw LIBRPA_RUNTIME_ERROR("Fail to open structure file " + file_path);
    }

    std::string token;
    for (int i = 0; i != 6 * 3; ++i)
    {
        infile >> token;
    }

    int n_atoms = 0;
    infile >> n_atoms;
    if (!infile.good() || n_atoms < 0)
    {
        throw LIBRPA_RUNTIME_ERROR("Fail to read atom count from " + file_path);
    }
    for (int i = 0; i != n_atoms * 4; ++i)
    {
        infile >> token;
    }

    std::vector<std::string> tokens;
    while (infile >> token)
    {
        tokens.emplace_back(token);
    }
    if (tokens.empty() || (tokens.size() >= 2 && is_stru_symop_convention(tokens[1])))
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Structure file does not contain legacy k-point sampling data: " + file_path);
    }

    std::size_t pos = 0;
    int nk[3];
    for (int i = 0; i != 3; ++i)
    {
        if (pos >= tokens.size())
        {
            throw LIBRPA_RUNTIME_ERROR("Unexpected end of stru_out while reading legacy k-point grid");
        }
        nk[i] = parse_stru_int_token(tokens[pos++], file_path);
        if (nk[i] <= 0)
        {
            throw LIBRPA_RUNTIME_ERROR("Invalid legacy k-point grid in " + file_path);
        }
    }

    const int nk_full = nk[0] * nk[1] * nk[2];
    int n_k_rows = 0;
    bool has_full_mapping = false;
    const bool can_try_band_k_count = ctx.state.n_kpoints > 0 && ctx.state.n_kpoints <= nk_full;
    if (can_try_band_k_count
        && (!try_legacy_stru_kpoint_layout(tokens, pos, ctx.state.n_kpoints, nk_full, true,
                                           n_k_rows, has_full_mapping)
            && !try_legacy_stru_kpoint_layout(tokens, pos, ctx.state.n_kpoints, nk_full, false,
                                              n_k_rows, has_full_mapping)))
    {
        n_k_rows = 0;
    }
    if (n_k_rows == 0
        && !try_legacy_stru_kpoint_layout(tokens, pos, nk_full, nk_full, true,
                                          n_k_rows, has_full_mapping)
        && !try_legacy_stru_kpoint_layout(tokens, pos, nk_full, nk_full, false,
                                          n_k_rows, has_full_mapping))
    {
        throw LIBRPA_RUNTIME_ERROR("Fail to locate legacy k-point rows in " + file_path);
    }
    if (ctx.state.n_kpoints > 0 && n_k_rows != ctx.state.n_kpoints)
    {
        throw LIBRPA_RUNTIME_ERROR(
            "Legacy stru_out k-point count does not match band_out: "
            + std::to_string(n_k_rows) + " != " + std::to_string(ctx.state.n_kpoints));
    }

    std::vector<double> kvecs(static_cast<std::size_t>(3 * n_k_rows));
    for (int i = 0; i != 3 * n_k_rows; ++i)
    {
        kvecs[static_cast<std::size_t>(i)] =
            parse_stru_double_token(tokens[pos++], file_path);
        if (!std::isfinite(kvecs[static_cast<std::size_t>(i)]))
        {
            throw LIBRPA_RUNTIME_ERROR("Legacy stru_out k-point row contains an invalid number");
        }
    }

    std::vector<int> map_q_ks(static_cast<std::size_t>(n_k_rows));
    for (int ik = 0; ik != n_k_rows; ++ik)
    {
        map_q_ks[static_cast<std::size_t>(ik)] = ik;
    }
    std::vector<double> kweights;

    if (n_k_rows < nk_full && !has_full_mapping)
    {
        auto pds = librpa_int::api::get_dataset_instance(ctx.h);
        auto &pbc = pds->pbc;
        pbc.set_kgrids_kvec(nk[0], nk[1], nk[2], kvecs);
        pbc.set_kq_mapping(map_q_ks);
        librpa_int::initialize_symmetry_context(*pds, false);
        const auto &symmetry_ctx = pds->symmetry_context;
        if (!symmetry_ctx.available || symmetry_ctx.kstars.size() != static_cast<std::size_t>(n_k_rows))
        {
            throw LIBRPA_RUNTIME_ERROR(
                "Legacy stru_out symmetry-reduced k-point list requires the full-to-q mapping "
                "or matching input symmetry k-stars");
        }

        std::vector<std::vector<Vector3_Order<double>>> full_kstars;
        full_kstars.reserve(symmetry_ctx.kstars.size());
        for (int ik = 0; ik != n_k_rows; ++ik)
        {
            Vector3_Order<double> k_ibz_read{
                kvecs[static_cast<std::size_t>(3 * ik)] / TWO_PI,
                kvecs[static_cast<std::size_t>(3 * ik + 1)] / TWO_PI,
                kvecs[static_cast<std::size_t>(3 * ik + 2)] / TWO_PI};
            const auto &star = symmetry_ctx.kstars[static_cast<std::size_t>(ik)];
            const auto k_ibz_expected = convert_fractional_kpoint_to_klist_units(star.k_ibz, pbc);
            if (!nearly_same_kpoint(k_ibz_read, k_ibz_expected))
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "Legacy stru_out symmetry-reduced k-point row does not match input symmetry k-star");
            }

            auto &members = full_kstars.emplace_back();
            members.reserve(star.members.size());
            for (const auto &member : star.members)
            {
                const auto k_member = convert_fractional_kpoint_to_klist_units(member.k_bz, pbc);
                if (std::find(members.begin(), members.end(), k_member) == members.end())
                {
                    members.emplace_back(k_member);
                }
            }
        }

        pbc.set_irreducible_kgrids_kvec(nk[0], nk[1], nk[2], kvecs, full_kstars);
        sync_ibz_kpoints_from_mapping(ctx, kvecs, map_q_ks, n_k_rows);
        return;
    }

    if (has_full_mapping)
    {
        std::vector<int> full_to_q(static_cast<std::size_t>(nk_full));
        for (int ik = 0; ik != nk_full; ++ik)
        {
            full_to_q[static_cast<std::size_t>(ik)] =
                parse_stru_int_token(tokens[pos++], file_path) - 1;
            if (full_to_q[static_cast<std::size_t>(ik)] < 0
                || full_to_q[static_cast<std::size_t>(ik)] >= n_k_rows)
            {
                throw LIBRPA_RUNTIME_ERROR(
                    "Legacy stru_out full-to-q mapping index out of range");
            }
        }

        if (n_k_rows == nk_full)
        {
            map_q_ks = std::move(full_to_q);
        }
        else
        {
            kweights.assign(static_cast<std::size_t>(n_k_rows), 0.0);
            for (const int iq : full_to_q)
            {
                kweights[static_cast<std::size_t>(iq)] += 1.0 / nk_full;
            }
            for (double weight : kweights)
            {
                if (weight <= 0.0)
                {
                    throw LIBRPA_RUNTIME_ERROR(
                        "Legacy stru_out irreducible k-point mapping misses a listed k-point");
                }
            }
        }
    }

    ctx.h.set_kgrids_kvec(nk[0], nk[1], nk[2], kvecs, kweights);
    ctx.h.set_kq_mapping(map_q_ks);
    sync_ibz_kpoints_from_mapping(ctx, kvecs, map_q_ks, n_k_rows);
}

void read_basis_wfc_aux(ReaderContext &ctx, const std::string &input_dir,
                        const std::string &fn_basis,
                        const std::string &fn_basis_wfc,
                        const std::string &fn_basis_aux)
{
    using librpa_int::join_path;
    using librpa_int::path_exists;

    const string path_basis = join_path(input_dir, fn_basis);
    const string path_basis_wfc = join_path(input_dir, fn_basis_wfc);
    const string path_basis_aux = join_path(input_dir, fn_basis_aux);
    if (path_exists(path_basis_wfc.c_str()) && path_exists(path_basis_aux.c_str()))
    {
        reader_basis_wfc(ctx, path_basis_wfc);
        reader_basis_aux(ctx, path_basis_aux);
    }
    else if (path_exists(path_basis.c_str()))
    {
        reader_basis(ctx, path_basis);
    }
    else
    {
        read_basis_from_Cs(ctx, input_dir);
    }
}

void read_band_kpath_info(ReaderContext &ctx, const string &file_path)
{
    auto &kfrac_band = ctx.state.kfrac_band;
    auto &n_basis_ao = ctx.state.n_basis_ao;
    auto &n_basis_wfc = ctx.state.n_basis_wfc;
    auto &n_kpoints_band = ctx.state.n_kpoints_band;
    auto &n_spins = ctx.state.n_spins;
    auto &n_states = ctx.state.n_states;

    int n_basis_band, n_states_band, n_spin_band;

    librpa_int::require_readable_file(file_path);
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string x, y, z;

    // Read dimensions in the first row
    infile >> x;
    n_basis_band = stoi(x);
    if (n_basis_band != n_basis_wfc) throw LIBRPA_RUNTIME_ERROR("band & SCF #basis inconsistent");
    infile >> x;
    n_states_band = stoi(x);
    if (n_states_band != n_states) throw LIBRPA_RUNTIME_ERROR("band & SCF #state inconsistent");
    infile >> x;
    n_spin_band = stoi(x);
    if (n_spin_band != n_spins) throw LIBRPA_RUNTIME_ERROR("band & SCF #spin inconsistent");
    infile >> x;
    n_kpoints_band = stoi(x);

    kfrac_band.clear();
    std::vector<double> vector_kfrac_band(n_kpoints_band * 3);  // For API parsing
    for (int i = 0; i < n_kpoints_band; i++)
    {
        infile >> x >> y >> z;
        Vector3_Order<double> kfrac{stod(x), stod(y), stod(z)};
        kfrac_band.emplace_back(kfrac);
        vector_kfrac_band[3 * i] = kfrac.x;
        vector_kfrac_band[3 * i + 1] = kfrac.y;
        vector_kfrac_band[3 * i + 2] = kfrac.z;
    }

    infile.close();

    ctx.h.set_band_kvec(n_kpoints_band, vector_kfrac_band.data());
}

namespace
{

constexpr std::int32_t READER_VELOCITY_MATRIX_V1_MARKER = -12345680;
constexpr std::int32_t VELOCITY_MATRIX_V1_KIND_COMPLEX_DOUBLE = 29;

static_assert(sizeof(std::complex<double>) == 2 * sizeof(double),
              "velocity_matrix v1 expects std::complex<double> as two doubles");

struct VelocityKpointBlock
{
    std::int32_t ik;
    std::int64_t offset;
};

template <typename T>
bool read_binary_value(std::ifstream &infile, T &value)
{
    infile.read(reinterpret_cast<char *>(&value), sizeof(value));
    return infile.good();
}

int check_velocity_file_version(const std::string &file_path)
{
    std::ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good()) return 0;
    std::int32_t marker = 0;
    infile.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    if (!infile.good() || marker > 0) return 0;
    if (marker == READER_VELOCITY_MATRIX_V1_MARKER) return 1;
    throw std::runtime_error("Unsupported velocity_matrix file marker: " + std::to_string(marker) +
                             " in " + file_path);
}

std::size_t velocity_hit_index(const int is, const int ik, const int ia, const int n_kpoints)
{
    return (librpa_int::as_size(is) * librpa_int::as_size(n_kpoints) + librpa_int::as_size(ik)) *
               3 +
           librpa_int::as_size(ia);
}

void check_velocity_target_hits(const std::vector<int> &target_hits, const int n_spins,
                                const int n_kpoints)
{
    for (int is = 0; is != n_spins; ++is)
    {
        for (int ik = 0; ik != n_kpoints; ++ik)
        {
            for (int ia = 0; ia != 3; ++ia)
            {
                if (target_hits[velocity_hit_index(is, ik, ia, n_kpoints)] == 0)
                {
                    throw std::logic_error(
                        "velocity_matrix is missing a selected head/wing k-point");
                }
            }
        }
    }
}

void validate_velocity_dimensions(const std::string &file_path, const MeanField &mf,
                                  const int n_spins, const int n_bands)
{
    if (n_spins != mf.get_n_spins() || n_bands != mf.get_n_states())
    {
        std::stringstream ss;
        ss << "velocity_matrix dimensions are inconsistent with meanfield: velocity=(" << n_spins
           << "," << n_bands << "), meanfield=(" << mf.get_n_spins() << "," << mf.get_n_kpoints()
           << "," << mf.get_n_states() << "), file=" << file_path;
        throw std::logic_error(ss.str());
    }
}

std::vector<std::string> discover_velocity_binary_v1_files(const string &file_path)
{
    std::vector<std::string> files{file_path};
    const auto extra_files = librpa_int::discover_files_with_prefix(
        librpa_int::parent_path(file_path), librpa_int::base_name(file_path) + "_");
    files.insert(files.end(), extra_files.begin(), extra_files.end());
    return files;
}

void read_velocity_binary_v1_file(const string &file_path, const MeanField &mf,
                                  velocity_matrix_t &velocity,
                                  const std::vector<int> &source_to_target_ik,
                                  std::vector<int> &target_hits)
{
    using librpa_int::ANG2BOHR;
    using librpa_int::HA2EV;

    std::ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good()) throw std::logic_error("Failed to open velocity_matrix file " + file_path);

    std::int32_t marker = 0;
    std::int32_t kind_raw = 0;
    std::int32_t n_kpoints_source = 0;
    std::int32_t n_spins = 0;
    std::int32_t n_bands = 0;
    std::int32_t n_aos = 0;
    std::int32_t n_alpha = 0;
    if (!read_binary_value(infile, marker) || !read_binary_value(infile, kind_raw) ||
        !read_binary_value(infile, n_kpoints_source) || !read_binary_value(infile, n_spins) ||
        !read_binary_value(infile, n_bands) || !read_binary_value(infile, n_aos) ||
        !read_binary_value(infile, n_alpha))
    {
        throw std::logic_error("Failed to read velocity_matrix v1 header from " + file_path);
    }
    if (marker != READER_VELOCITY_MATRIX_V1_MARKER ||
        kind_raw != VELOCITY_MATRIX_V1_KIND_COMPLEX_DOUBLE || n_kpoints_source < 0 ||
        n_spins <= 0 || n_bands <= 0 || n_aos <= 0 || n_alpha != 3)
    {
        throw std::logic_error("Invalid velocity_matrix v1 header in " + file_path);
    }

    validate_velocity_dimensions(file_path, mf, n_spins, n_bands);

    std::vector<VelocityKpointBlock> blocks(librpa_int::as_size(n_kpoints_source));
    for (auto &block : blocks)
    {
        if (!read_binary_value(infile, block.ik) || !read_binary_value(infile, block.offset))
            throw std::logic_error("Failed to read velocity_matrix v1 block table");
        if (block.offset < 0) throw std::logic_error("Invalid velocity_matrix v1 block offset");
    }

    const auto n_block = librpa_int::as_size(n_spins) * 3 * librpa_int::as_size(n_bands) *
                         librpa_int::as_size(n_bands);
    std::vector<std::complex<double>> block_data(n_block);

    for (const auto &block : blocks)
    {
        const int ik_source = block.ik - 1;
        if (ik_source < 0 || ik_source >= static_cast<int>(source_to_target_ik.size()))
            throw std::logic_error("velocity_matrix v1 k-point index is out of range");
        const int ik_target = source_to_target_ik[ik_source];
        if (ik_target < 0) continue;

        infile.clear();
        infile.seekg(static_cast<std::streamoff>(block.offset), std::ios::beg);
        if (!infile.good()) throw std::logic_error("Failed to seek velocity_matrix v1 payload");
        infile.read(reinterpret_cast<char *>(block_data.data()),
                    static_cast<std::streamsize>(block_data.size() * sizeof(std::complex<double>)));
        if (!infile.good()) throw std::logic_error("Failed to read velocity_matrix v1 payload");

        for (int is = 0; is != n_spins; ++is)
        {
            for (int ia = 0; ia != 3; ++ia)
            {
                target_hits[velocity_hit_index(is, ik_target, ia, mf.get_n_kpoints())] = 1;
                for (int i = 0; i != n_bands; ++i)
                {
                    for (int j = 0; j != n_bands; ++j)
                    {
                        const auto index =
                            (((librpa_int::as_size(is) * 3 + librpa_int::as_size(ia)) *
                                  librpa_int::as_size(n_bands) +
                              librpa_int::as_size(i)) *
                                 librpa_int::as_size(n_bands) +
                             librpa_int::as_size(j));
                        velocity.at(is).at(ik_target).at(ia)(i, j) =
                            ANG2BOHR * block_data[index] / HA2EV;
                    }
                }
            }
        }
    }
}

}  // namespace

void read_velocity(ReaderContext &ctx, const string &file_path, const MeanField &mf, velocity_matrix_t &velocity)
{
    using librpa_int::ANG2BOHR;
    using librpa_int::HA2EV;

    librpa_int::require_readable_file(file_path);
    if (check_velocity_file_version(file_path) == 1)
    {
        std::vector<int> source_to_target_ik(librpa_int::as_size(mf.get_n_kpoints()));
        for (int ik = 0; ik != mf.get_n_kpoints(); ++ik) source_to_target_ik[ik] = ik;
        librpa_int::global::lib_printf_root("velocity_matrix reader: binary v1\n");
        initialize_velocity_matrix(velocity, mf.get_n_spins(), mf.get_n_kpoints(),
                                   mf.get_n_states());
        std::vector<int> target_hits(
            librpa_int::as_size(mf.get_n_kpoints()) * librpa_int::as_size(mf.get_n_spins()) * 3, 0);
        for (const auto &velocity_file : discover_velocity_binary_v1_files(file_path))
        {
            read_velocity_binary_v1_file(velocity_file, mf, velocity, source_to_target_ik,
                                         target_hits);
        }
        check_velocity_target_hits(target_hits, mf.get_n_spins(), mf.get_n_kpoints());
        if (ctx.comm.is_root())
            std::cout << "* Success: read velocity from pyatb_librpa_df(ABACUS)." << std::endl;
        return;
    }

    ifstream infile;
    infile.open(file_path);
    string alpha, kk, ss, single_re, single_im;
    int n_kpoints, n_spins, n_bands, n_aos;
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_bands;
    infile >> n_aos;
    if (!infile.good())
        throw std::logic_error("Failed to read velocity dimensions from " + file_path);
    if (n_kpoints != mf.get_n_kpoints() || n_spins != mf.get_n_spins() ||
        n_bands != mf.get_n_states())
    {
        std::stringstream ss;
        ss << "velocity_matrix dimensions are inconsistent with meanfield: velocity=(" << n_spins
           << "," << n_kpoints << "," << n_bands << "), meanfield=(" << mf.get_n_spins() << ","
           << mf.get_n_kpoints() << "," << mf.get_n_states() << ")";
        throw std::logic_error(ss.str());
    }

    initialize_velocity_matrix(velocity, n_spins, n_kpoints, n_bands);
    for (int is = 0; is != n_spins; is++)
    {
        for (int ik = 0; ik != n_kpoints; ik++)
        {
            for (int ia = 0; ia != 3; ia++)
            {
                infile >> alpha >> kk >> ss;
                int k_index = stoi(kk) - 1;
                int a_index = stoi(alpha) - 1;
                int s_index = stoi(ss) - 1;
                assert(k_index == ik);
                assert(a_index == ia);
                assert(s_index == is);
                for (int i = 0; i != n_bands; i++)
                {
                    for (int j = 0; j != n_bands; j++)
                    {
                        infile >> single_re >> single_im;
                        velocity.at(is).at(ik).at(ia)(i, j) =
                            ANG2BOHR * std::complex<double>(stod(single_re), stod(single_im)) /
                            HA2EV;
                    }
                }
            }
        }
    }
    if (ctx.comm.is_root())
        std::cout << "* Success: read velocity from pyatb_librpa_df(ABACUS)." << std::endl;
}

void read_velocity_abacus(ReaderContext &ctx, const MeanField &mf, const string &dir_path,
                          const string &file_prefix, velocity_matrix_t &velocity)
{
    const auto files = librpa_int::discover_files_with_prefix(dir_path, file_prefix);
    for (const auto &name : {file_prefix + ".txt", file_prefix})
    {
        const auto file = std::find_if(files.begin(), files.end(), [&name](const auto &path) {
            return librpa_int::base_name(path) == name;
        });
        if (file != files.end())
        {
            std::cout << "* Read ABACUS velocity file: " << *file << std::endl;
            read_velocity(ctx, *file, mf, velocity);
            return;
        }
    }

    throw std::logic_error("Cannot find ABACUS velocity file with prefix " + file_prefix
                           + " under " + dir_path);
}

void read_velocity(ReaderContext &ctx, const string &file_path, const MeanField &mf, velocity_matrix_t &velocity,
                   const std::vector<int> &source_to_target_ik,
                   const int source_n_kpoints_expected)
{
    using librpa_int::ANG2BOHR;
    using librpa_int::HA2EV;

    librpa_int::require_readable_file(file_path);
    if (check_velocity_file_version(file_path) == 1)
    {
        librpa_int::global::lib_printf_root("velocity_matrix reader: binary v1\n");
        if (source_n_kpoints_expected != static_cast<int>(source_to_target_ik.size()))
        {
            throw std::logic_error("velocity_matrix k-point count is inconsistent with k_path_info");
        }
        initialize_velocity_matrix(velocity, mf.get_n_spins(), mf.get_n_kpoints(),
                                   mf.get_n_states());
        std::vector<int> target_hits(librpa_int::as_size(mf.get_n_kpoints()) *
                                         librpa_int::as_size(mf.get_n_spins()) * 3,
                                     0);
        for (const auto &velocity_file : discover_velocity_binary_v1_files(file_path))
        {
            read_velocity_binary_v1_file(velocity_file, mf, velocity, source_to_target_ik,
                                         target_hits);
        }
        check_velocity_target_hits(target_hits, mf.get_n_spins(), mf.get_n_kpoints());
        if (ctx.comm.is_root())
            std::cout << "* Success: read velocity from pyatb_librpa_df(ABACUS)." << std::endl;
        return;
    }

    ifstream infile;
    infile.open(file_path);
    string alpha, kk, ss, single_re, single_im;
    int n_kpoints_source, n_spins, n_bands, n_aos;
    infile >> n_kpoints_source;
    infile >> n_spins;
    infile >> n_bands;
    infile >> n_aos;
    if (!infile.good())
        throw std::logic_error("Failed to read velocity dimensions from " + file_path);
    if (n_kpoints_source != source_n_kpoints_expected ||
        n_kpoints_source != static_cast<int>(source_to_target_ik.size()))
    {
        throw std::logic_error("velocity_matrix k-point count is inconsistent with k_path_info");
    }
    if (n_spins != mf.get_n_spins() || n_bands != mf.get_n_states())
    {
        std::stringstream ss;
        ss << "velocity_matrix dimensions are inconsistent with meanfield: velocity=(" << n_spins
           << "," << n_kpoints_source << "," << n_bands << "), meanfield=(" << mf.get_n_spins()
           << "," << mf.get_n_kpoints() << "," << mf.get_n_states() << ")";
        throw std::logic_error(ss.str());
    }

    initialize_velocity_matrix(velocity, n_spins, mf.get_n_kpoints(), n_bands);
    std::vector<int> target_hits(librpa_int::as_size(mf.get_n_kpoints()) *
                                     librpa_int::as_size(n_spins) * 3,
                                 0);
    for (int is = 0; is != n_spins; is++)
    {
        for (int ik_read = 0; ik_read != n_kpoints_source; ik_read++)
        {
            for (int ia = 0; ia != 3; ia++)
            {
                infile >> alpha >> kk >> ss;
                if (!infile.good())
                    throw std::logic_error("Failed to read velocity_matrix block header");
                const int k_index_source = stoi(kk) - 1;
                const int a_index = stoi(alpha) - 1;
                const int s_index = stoi(ss) - 1;
                if (k_index_source < 0 || k_index_source >= n_kpoints_source)
                    throw std::logic_error("velocity_matrix k-point index is out of range");
                if (a_index != ia || s_index != is)
                    throw std::logic_error("velocity_matrix block order is inconsistent");
                const int k_index_target = source_to_target_ik[k_index_source];
                for (int i = 0; i != n_bands; i++)
                {
                    for (int j = 0; j != n_bands; j++)
                    {
                        infile >> single_re >> single_im;
                        if (!infile.good())
                            throw std::logic_error("Failed to read velocity_matrix element");
                        if (k_index_target < 0) continue;
                        target_hits[(librpa_int::as_size(is) *
                                         librpa_int::as_size(mf.get_n_kpoints()) +
                                     librpa_int::as_size(k_index_target)) *
                                        3 +
                                    librpa_int::as_size(ia)] = 1;
                        velocity.at(is).at(k_index_target).at(ia)(i, j) =
                            ANG2BOHR * std::complex<double>(stod(single_re), stod(single_im)) /
                            HA2EV;
                    }
                }
            }
        }
    }
    for (int is = 0; is != n_spins; ++is)
    {
        for (int ik = 0; ik != mf.get_n_kpoints(); ++ik)
        {
            for (int ia = 0; ia != 3; ++ia)
            {
                if (target_hits[(librpa_int::as_size(is) *
                                     librpa_int::as_size(mf.get_n_kpoints()) +
                                 librpa_int::as_size(ik)) *
                                    3 +
                                librpa_int::as_size(ia)] == 0)
                {
                    throw std::logic_error(
                        "velocity_matrix is missing a selected head/wing k-point");
                }
            }
        }
    }
    if (ctx.comm.is_root())
        std::cout << "* Success: read velocity from pyatb_librpa_df(ABACUS)." << std::endl;
}

void read_velocity_aims(ReaderContext &ctx, const MeanField &mf, const string &file_path,
                        const string &file_prefix, velocity_matrix_t &velocity)
{

    using std::cerr;
    using std::complex;
    using std::endl;
    using std::vector;

    int nk = mf.get_n_kpoints();
    int n_spins = mf.get_n_spins();
    int nbands = mf.get_n_bands();
    initialize_velocity_matrix(velocity, n_spins, nk, nbands);

    for (int ik = 0; ik < nk; ik++)
    {
        std::stringstream ss;
        ss << file_path << file_prefix << std::setfill('0') << std::setw(6) << ik + 1 << ".dat";

        if (librpa_int::file_exists(ss.str()))
        {
            librpa_int::require_readable_file(ss.str());
        }
        std::ifstream infile(ss.str(), std::ios::binary);
        if (!infile.is_open())
        {
            std::cerr << "Failed to open file: " << ss.str() << std::endl;
            continue;
        }

        int i_k_point, n_state_min, n_state_max, ld, n_spin_in, n_pol_dir;
        infile.read(reinterpret_cast<char *>(&i_k_point), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_state_min), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_state_max), sizeof(int));
        infile.read(reinterpret_cast<char *>(&ld), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_spin_in), sizeof(int));
        infile.read(reinterpret_cast<char *>(&n_pol_dir), sizeof(int));

        int n_pairs = ld * n_spin_in * n_pol_dir;
        std::vector<std::complex<double>> mommat(n_pairs);
        infile.read(reinterpret_cast<char *>(mommat.data()),
                    n_pairs * sizeof(std::complex<double>));
        infile.close();

        int iline = 0;
        for (int ipol = 0; ipol < n_pol_dir; ipol++)
        {
            for (int is = 0; is < n_spins; is++)
            {
                for (int im = 0; im < nbands; im++)
                {
                    for (int in = im; in < nbands; in++)
                    {
                        velocity.at(is).at(ik).at(ipol)(in, im) = mommat[iline];
                        velocity.at(is).at(ik).at(ipol)(im, in) = std::conj(mommat[iline]);
                        iline++;
                    }
                }
            }
        }
    }

    if (ctx.comm.is_root())
        std::cout << "* Success: read moment from mommat_ks_kpt_*.dat (FHI-aims)." << std::endl;
}

} // namespace librpa::reader
