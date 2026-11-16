#include <librpa_enums.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "../../src/api/instance_manager.h"
#include "../../src/core/symmetry_context.h"
#include "../../src/io/fs.h"
#include "../../src/io/global_io.h"
#include "../../src/io/stl_io_helper.h"
#include "../../src/math/complexmatrix.h"
#include "../../src/mpi/collective_stage.h"
#include "../../src/mpi/global_mpi.h"
#include "../../src/utils/constants.h"
#include "../../src/utils/profiler.h"
#include "../driver.h"
#include "../reader/localized_operator_io.h"
#include "../read_data.h"
#include "../reader_2d_coulomb_head.h"
#include "../task.h"
#include "crpa.h"
#include "librpa_input.h"
#include "task_helper.h"

using librpa_int::collective_all_stage;

namespace
{
struct LocalizedOperatorView
{
    std::vector<LibrpaLocalizedBlock> blocks;
    LibrpaLocalizedOperator view{};

    explicit LocalizedOperatorView(const librpa_int::LocalizedMatrix &matrix)
    {
        for (const auto &[translation, values] : matrix)
        {
            if (!view.dimension) view.dimension = values.nr();
            if (values.nr() != view.dimension || values.nc() != view.dimension)
                throw LIBRPA_RUNTIME_ERROR("Native operator contains inconsistent block shapes");
            blocks.push_back({{translation.x, translation.y, translation.z},
                              reinterpret_cast<const double *>(values.ptr()),
                              values.is_row_major() ? 1 : 0});
        }
        view.n_blocks = blocks.size();
        view.blocks = blocks.data();
    }
};
}  // namespace

static void prepare_g0w0_native_velocity(bool prepare_optical = true)
{
    using namespace librpa_int;
    const auto &comm = global::mpi_comm_global_h;
    const auto &params = driver::driver_params;
    LocalizedAoOperators operators;
    std::vector<LocalizedMatrix> hamiltonians;
    std::vector<LocalizedOperatorView> storage;
    std::vector<LibrpaLocalizedOperator> h_views;
    LibrpaNativeVelocityInput input{};
    auto input_path = [&](const std::string &filename)
    { return is_absolute_path(filename) ? filename : join_path(params.input_dir, filename); };
    collective_all_stage(
        comm, "G0W0 native operator input",
        [&]
        {
            if (!comm.is_root()) return;
            const double energy_scale = params.g0w0_hr_unit == "Ry"   ? 0.5
                                        : params.g0w0_hr_unit == "eV" ? 1.0 / HA2EV
                                                                      : 1.0;
            const double length_scale = params.g0w0_rr_unit == "Angstrom" ? ANG2BOHR : 1.0;
            operators = librpa::reader::read_headwing_ao_operators(
                input_path(params.g0w0_fn_sr), input_path(params.g0w0_fn_rr), length_scale);
            hamiltonians.push_back(librpa::reader::read_headwing_hamiltonian_csr(
                input_path(params.g0w0_fn_hr), energy_scale));
            if (!params.g0w0_fn_hr_spin1.empty())
                hamiltonians.push_back(librpa::reader::read_headwing_hamiltonian_csr(
                    input_path(params.g0w0_fn_hr_spin1), energy_scale));
            storage.reserve(4 + hamiltonians.size());
            storage.emplace_back(operators.overlap);
            input.overlap = storage.back().view;
            for (int axis = 0; axis < 3; ++axis)
            {
                storage.emplace_back(operators.position[axis]);
                input.position[axis] = storage.back().view;
            }
            for (const auto &hamiltonian : hamiltonians)
            {
                storage.emplace_back(hamiltonian);
                h_views.push_back(storage.back().view);
            }
            input.n_spins = h_views.size();
            input.hamiltonian = h_views.data();
            if (prepare_optical && driver::get_bool(driver::opts.use_2d_dielectric))
            {
                const auto metadata = read_strict_2d_coulomb_head_metadata(
                    join_path(params.input_dir, "librpa_2d_coulomb_head.txt"));
                input.inplane_area_bohr2 = metadata.inplane_area_bohr2;
                input.auxiliary_monopole_norm_squared = metadata.auxiliary_monopole_norm_squared;
            }
        });
    driver::h.prepare_native_velocity(driver::opts, comm.is_root() ? &input : nullptr,
                                      prepare_optical);
}

void driver::prepare_crpa_optical_velocity()
{
    if (driver_params.option_velocity == "native")
    {
        prepare_g0w0_native_velocity(false);
        return;
    }
    int n_spins, n_kpoints, n_states, n_aos, n_spinor;
    librpa_get_meanfield_dimensions(h.get_c_handler(), 0, &n_spins, &n_kpoints, &n_states,
                                     &n_aos, &n_spinor);
    const auto &name = driver_params.prefix_velocity;
    const auto path = librpa_int::is_absolute_path(name)
                          ? name
                          : librpa_int::join_path(driver_params.input_dir, name);
    collective_all_stage(
        librpa_int::global::mpi_comm_global_h, "cRPA optical velocity input",
        [&]
        {
            std::vector<std::vector<std::vector<librpa_int::ComplexMatrix>>> velocity;
            read_velocity(path, n_spins, n_kpoints, n_states,
                          velocity);
            std::vector<std::complex<double>> packed;
            for (const auto &spin : velocity)
                for (const auto &k : spin)
                    for (const auto &axis : k)
                        for (int row = 0; row < axis.nr; ++row)
                            for (int column = 0; column < axis.nc; ++column)
                                packed.push_back(axis(row, column));
            h.set_velocity_matrix_packed(n_spins, n_kpoints, n_states,
                                         packed.data());
        });
}

static void write_g0w0_headwing_diagnostics()
{
    using namespace librpa_int;
    if (!global::should_output(LIBRPA_VERBOSE_DEBUG)) return;
    const auto &comm = global::mpi_comm_global_h;
    collective_all_stage(
        comm, "G0W0 head/wing diagnostics",
        [&]
        {
            if (!comm.is_root()) return;
            const auto result = driver::h.get_optical_result();
            const auto view = librpa_get_optical_view(result.get());
            create_directories(driver::opts.output_dir, 0);
            std::ofstream velocity(join_path(driver::opts.output_dir, "g0w0_velocity.csv"));
            std::ofstream optical(join_path(driver::opts.output_dir, "g0w0_headwing.csv"));
            if (!velocity || !optical)
                throw LIBRPA_RUNTIME_ERROR("Cannot open G0W0 head/wing diagnostic output");
            velocity << "spin,k,alpha,row,col,real_HaBohr,imag_HaBohr\n"
                     << std::scientific << std::setprecision(17);
            std::size_t index = 0;
            for (int spin = 0; spin < view.n_spins; ++spin)
                for (int k = 0; k < view.n_kpoints; ++k)
                    for (int axis = 0; axis < 3; ++axis)
                        for (int row = 0; row < view.n_states; ++row)
                            for (int column = 0; column < view.n_states; ++column, index += 2)
                                velocity << spin << ',' << k << ',' << axis << ',' << row << ','
                                         << column << ',' << view.velocity_ri[index] << ','
                                         << view.velocity_ri[index + 1] << '\n';
            optical << "quantity,frequency,omega_Ha,row,col,real,imag\n"
                    << std::scientific << std::setprecision(17);
            const auto write = [&](const char *name, const double *values, int rows, int columns)
            {
                std::size_t entry = 0;
                for (int frequency = 0; frequency < view.n_frequencies; ++frequency)
                    for (int row = 0; row < rows; ++row)
                        for (int column = 0; column < columns; ++column, entry += 2)
                            optical << name << ',' << frequency << ','
                                    << view.frequencies_ha[frequency] << ',' << row << ',' << column
                                    << ',' << values[entry] << ',' << values[entry + 1] << '\n';
            };
            write("head", view.head_ri, view.head_rows, view.head_columns);
            write("wing_auxiliary", view.wing_ri, view.wing_rows, view.wing_columns);
            velocity.flush();
            optical.flush();
            if (!velocity || !optical)
                throw LIBRPA_RUNTIME_ERROR("Cannot flush G0W0 head/wing diagnostic output");
        });
}

static std::vector<double> make_spectral_omegas_ha()
{
    const double start = driver::driver_params.sf_omega_start;
    const double end = driver::driver_params.sf_omega_end;
    const double step = driver::driver_params.sf_omega_step;

    std::vector<double> omegas;
    const double guard = 1e-12 * std::max(1.0, std::max(std::abs(start), std::abs(end)));
    for (double omega = start; omega <= end + guard; omega += step)
        omegas.push_back(omega / librpa_int::HA2EV);
    return omegas;
}

static std::vector<double> slice_state_window(
    const std::vector<double> &src,
    const int n_spins,
    const int n_kpoints_local,
    const int src_state_low,
    const int src_state_high,
    const int dst_state_low,
    const int dst_state_high)
{
    if (dst_state_low < src_state_low || dst_state_high > src_state_high
        || dst_state_high <= dst_state_low)
    {
        throw LIBRPA_RUNTIME_ERROR("invalid spectral-function state range");
    }

    const int n_states_src = src_state_high - src_state_low;
    const int n_states_dst = dst_state_high - dst_state_low;
    std::vector<double> dst(n_spins * n_kpoints_local * n_states_dst);
    for (int isp = 0; isp != n_spins; ++isp)
    {
        for (int ik_local = 0; ik_local != n_kpoints_local; ++ik_local)
        {
            const int src0 = (isp * n_kpoints_local + ik_local) * n_states_src
                + dst_state_low - src_state_low;
            const int dst0 = (isp * n_kpoints_local + ik_local) * n_states_dst;
            std::copy_n(src.data() + src0, n_states_dst, dst.data() + dst0);
        }
    }
    return dst;
}

template <typename T>
void write_binary_value(std::ofstream &ofs, const T &value)
{
    ofs.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

static void write_binary_doubles(std::ofstream &ofs, const std::vector<double> &values)
{
    if (values.empty()) return;
    ofs.write(reinterpret_cast<const char *>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
}

static void write_spectral_function_binary(
    const std::string &output_dir,
    const std::string &filename,
    const std::vector<double> &omegas_ha,
    const librpa::G0W0SpectralFunctionResult &sf,
    const librpa_int::MeanField &mf,
    const std::vector<double> &vxc,
    const std::vector<double> &vexx,
    const std::vector<int> &iks,
    const int n_spins,
    const int i_state_low,
    const int i_state_high)
{
    librpa_int::create_directories(output_dir.c_str(), 0);
    const auto path = librpa_int::join_path(output_dir, filename);
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    if (!ofs)
        throw LIBRPA_RUNTIME_ERROR("Failed to open spectral-function output file " + path);

    const int n_states = i_state_high - i_state_low;
    const int n_omegas = static_cast<int>(omegas_ha.size());
    const int n_kpoints_local = static_cast<int>(iks.size());
    const auto n_values = static_cast<size_t>(n_spins) * n_kpoints_local * n_states;
    const auto n_spectral_values = n_values * n_omegas;

    write_binary_value(ofs, static_cast<std::int32_t>(n_spins));
    write_binary_value(ofs, static_cast<std::int32_t>(n_kpoints_local));
    write_binary_value(ofs, static_cast<std::int32_t>(i_state_low));
    write_binary_value(ofs, static_cast<std::int32_t>(i_state_high));
    const double efermi_ev = mf.get_efermi() * librpa_int::HA2EV;
    write_binary_value(ofs, efermi_ev);
    write_binary_value(ofs, static_cast<std::int32_t>(n_omegas));

    std::vector<double> values(n_values);
    for (int isp = 0; isp != n_spins; ++isp)
    {
        for (int ik_local = 0; ik_local != n_kpoints_local; ++ik_local)
        {
            const int ik = iks[ik_local];
            for (int i = 0; i != n_states; ++i)
            {
                const int i_state = i_state_low + i;
                const int state_idx = (isp * n_kpoints_local + ik_local) * n_states + i;
                values[state_idx] =
                    mf.get_eigenvals()[isp](ik, i_state) * librpa_int::HA2EV;
            }
        }
    }
    write_binary_doubles(ofs, values);

    for (size_t i = 0; i != n_values; ++i)
        values[i] = vxc[i] * librpa_int::HA2EV;
    write_binary_doubles(ofs, values);

    for (size_t i = 0; i != n_values; ++i)
        values[i] = vexx[i] * librpa_int::HA2EV;
    write_binary_doubles(ofs, values);

    std::vector<double> omega_values(n_omegas);
    for (int iomega = 0; iomega != n_omegas; ++iomega)
        omega_values[iomega] = omegas_ha[iomega] * librpa_int::HA2EV;
    write_binary_doubles(ofs, omega_values);

    values.resize(n_spectral_values);
    for (size_t i = 0; i != n_spectral_values; ++i)
        values[i] = sf.spectral_function[i] / librpa_int::HA2EV;
    write_binary_doubles(ofs, values);

    for (size_t i = 0; i != n_spectral_values; ++i)
        values[i] = sf.sigc[i].real() * librpa_int::HA2EV;
    write_binary_doubles(ofs, values);

    for (size_t i = 0; i != n_spectral_values; ++i)
        values[i] = sf.sigc[i].imag() * librpa_int::HA2EV;
    write_binary_doubles(ofs, values);

    if (!ofs)
        throw LIBRPA_RUNTIME_ERROR("Failed to write spectral-function output file " + path);
}

static std::size_t checked_mul_size(const std::size_t a,
                                    const std::size_t b,
                                    const std::string &label)
{
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a)
        throw LIBRPA_RUNTIME_ERROR("spectral-function output size overflow: " + label);
    return a * b;
}

static MPI_Offset checked_add_offset(const MPI_Offset offset,
                                     const std::size_t bytes,
                                     const std::string &label)
{
    const auto max_delta =
        static_cast<std::size_t>(std::numeric_limits<MPI_Offset>::max() - offset);
    if (bytes > max_delta)
        throw LIBRPA_RUNTIME_ERROR("spectral-function MPI file offset overflow: " + label);
    return offset + static_cast<MPI_Offset>(bytes);
}

static MPI_Offset values_offset(const MPI_Offset section_offset,
                                const std::size_t value_index,
                                const std::string &label)
{
    const auto bytes = checked_mul_size(value_index, sizeof(double), label);
    return checked_add_offset(section_offset, bytes, label);
}

struct SpectralFunctionBinaryLayout
{
    MPI_Offset eigenvalues = 0;
    MPI_Offset vxc = 0;
    MPI_Offset vexx = 0;
    MPI_Offset omegas = 0;
    MPI_Offset spectral = 0;
    MPI_Offset sigc_re = 0;
    MPI_Offset sigc_im = 0;
    MPI_Offset file_size = 0;
};

static SpectralFunctionBinaryLayout make_spectral_binary_layout(
    const int n_spins,
    const int n_kpoints,
    const int n_states,
    const int n_omegas)
{
    if (n_spins <= 0 || n_kpoints <= 0 || n_states <= 0 || n_omegas <= 0)
        throw LIBRPA_RUNTIME_ERROR("invalid spectral-function output dimensions");

    SpectralFunctionBinaryLayout layout;

    const auto nk_spin = checked_mul_size(
        static_cast<std::size_t>(n_spins), static_cast<std::size_t>(n_kpoints),
        "spin-k section");
    const auto n_state_values = checked_mul_size(
        nk_spin, static_cast<std::size_t>(n_states), "state section");
    const auto n_spectral_values = checked_mul_size(
        n_state_values, static_cast<std::size_t>(n_omegas), "spectral section");

    const auto state_bytes =
        checked_mul_size(n_state_values, sizeof(double), "state bytes");
    const auto omega_bytes =
        checked_mul_size(static_cast<std::size_t>(n_omegas), sizeof(double),
                         "omega bytes");
    const auto spectral_bytes =
        checked_mul_size(n_spectral_values, sizeof(double), "spectral bytes");

    MPI_Offset offset = 5 * static_cast<MPI_Offset>(sizeof(std::int32_t))
        + static_cast<MPI_Offset>(sizeof(double));
    layout.eigenvalues = offset;
    offset = checked_add_offset(offset, state_bytes, "eigenvalue section");
    layout.vxc = offset;
    offset = checked_add_offset(offset, state_bytes, "vxc section");
    layout.vexx = offset;
    offset = checked_add_offset(offset, state_bytes, "vexx section");
    layout.omegas = offset;
    offset = checked_add_offset(offset, omega_bytes, "omega section");
    layout.spectral = offset;
    offset = checked_add_offset(offset, spectral_bytes, "spectral section");
    layout.sigc_re = offset;
    offset = checked_add_offset(offset, spectral_bytes, "ReSigc section");
    layout.sigc_im = offset;
    offset = checked_add_offset(offset, spectral_bytes, "ImSigc section");
    layout.file_size = offset;
    return layout;
}

static void mpi_file_write_exact(MPI_File file,
                                 MPI_Offset offset,
                                 const void *buffer,
                                 std::size_t nbytes,
                                 const std::string &label)
{
    constexpr std::size_t chunk_bytes = 64ULL * 1024ULL * 1024ULL;
    const char *ptr = static_cast<const char *>(buffer);
    while (nbytes > 0)
    {
        const int chunk =
            static_cast<int>(std::min(chunk_bytes, nbytes));
        const int ierr = MPI_File_write_at(file, offset, const_cast<char *>(ptr),
                                           chunk, MPI_BYTE, MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS)
            throw LIBRPA_RUNTIME_ERROR("MPI write failed for spectral-function " + label);
        offset += static_cast<MPI_Offset>(chunk);
        ptr += chunk;
        nbytes -= static_cast<std::size_t>(chunk);
    }
}

template <typename T>
static void mpi_file_write_value(MPI_File file,
                                 MPI_Offset &offset,
                                 const T &value,
                                 const std::string &label)
{
    mpi_file_write_exact(file, offset, &value, sizeof(T), label);
    offset += static_cast<MPI_Offset>(sizeof(T));
}

static void validate_spectral_local_buffers(
    const librpa::G0W0SpectralFunctionResult &sf,
    const std::vector<double> &vxc,
    const std::vector<double> &vexx,
    const std::vector<int> &iks,
    const int n_spins,
    const int n_kpoints_global,
    const int n_states,
    const int n_omegas)
{
    const int n_kpoints_local = static_cast<int>(iks.size());
    const auto n_local_expected =
        checked_mul_size(
            checked_mul_size(
                checked_mul_size(static_cast<std::size_t>(n_spins),
                                 static_cast<std::size_t>(n_kpoints_local),
                                 "local spin-k section"),
                static_cast<std::size_t>(n_states), "local state section"),
            static_cast<std::size_t>(n_omegas), "local spectral section");
    const auto n_state_expected =
        checked_mul_size(
            checked_mul_size(static_cast<std::size_t>(n_spins),
                             static_cast<std::size_t>(n_kpoints_local),
                             "local spin-k state section"),
            static_cast<std::size_t>(n_states), "local state values");

    if (vxc.size() != n_state_expected || vexx.size() != n_state_expected)
        throw LIBRPA_RUNTIME_ERROR("invalid local spectral state-value buffer size");
    if (sf.spectral_function.size() != n_local_expected
        || sf.sigc.size() != n_local_expected)
        throw LIBRPA_RUNTIME_ERROR("invalid local spectral-function buffer size");
    for (const int ik : iks)
        if (ik < 0 || ik >= n_kpoints_global)
            throw LIBRPA_RUNTIME_ERROR("spectral-function k-point index out of range");
}

template <typename LocalValue>
static void write_mpi_state_section(MPI_File file,
                                    const MPI_Offset section_offset,
                                    const std::vector<int> &iks,
                                    const int n_spins,
                                    const int n_kpoints_global,
                                    const int n_states,
                                    LocalValue local_value,
                                    const std::string &label)
{
    const int n_kpoints_local = static_cast<int>(iks.size());
    std::vector<double> row(n_states);
    for (int isp = 0; isp != n_spins; ++isp)
    {
        for (int ik_local = 0; ik_local != n_kpoints_local; ++ik_local)
        {
            const int ik = iks[ik_local];
            for (int ist = 0; ist != n_states; ++ist)
            {
                const auto idx_local =
                    (static_cast<std::size_t>(isp) * n_kpoints_local + ik_local)
                    * n_states + ist;
                row[ist] = local_value(idx_local, isp, ik, ist);
            }

            const auto idx_global =
                (static_cast<std::size_t>(isp) * n_kpoints_global + ik) * n_states;
            mpi_file_write_exact(
                file, values_offset(section_offset, idx_global, label),
                row.data(), row.size() * sizeof(double), label);
        }
    }
}

template <typename LocalValue>
static void write_mpi_spectral_section(MPI_File file,
                                       const MPI_Offset section_offset,
                                       const std::vector<int> &iks,
                                       const int n_spins,
                                       const int n_kpoints_global,
                                       const int n_states,
                                       const int n_omegas,
                                       LocalValue local_value,
                                       const std::string &label)
{
    const int n_kpoints_local = static_cast<int>(iks.size());
    std::vector<double> row(n_omegas);
    for (int isp = 0; isp != n_spins; ++isp)
    {
        for (int ik_local = 0; ik_local != n_kpoints_local; ++ik_local)
        {
            const int ik = iks[ik_local];
            for (int ist = 0; ist != n_states; ++ist)
            {
                const auto row_local =
                    ((static_cast<std::size_t>(isp) * n_kpoints_local + ik_local)
                     * n_states + ist) * n_omegas;
                for (int iomega = 0; iomega != n_omegas; ++iomega)
                    row[iomega] = local_value(row_local + iomega);

                const auto row_global =
                    ((static_cast<std::size_t>(isp) * n_kpoints_global + ik)
                     * n_states + ist) * n_omegas;
                mpi_file_write_exact(
                    file, values_offset(section_offset, row_global, label),
                    row.data(), row.size() * sizeof(double), label);
            }
        }
    }
}

static void write_spectral_function_binary_kdistributed(
    const std::string &output_dir,
    const std::string &filename,
    const std::vector<double> &omegas_ha,
    const librpa::G0W0SpectralFunctionResult &sf,
    const librpa_int::MeanField &mf,
    const std::vector<double> &vxc,
    const std::vector<double> &vexx,
    const std::vector<int> &iks,
    const int n_spins,
    const int n_kpoints_global,
    const int i_state_low,
    const int i_state_high)
{
    const int n_states = i_state_high - i_state_low;
    if (omegas_ha.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw LIBRPA_RUNTIME_ERROR("too many spectral-function frequency points");
    const int n_omegas = static_cast<int>(omegas_ha.size());

    std::string local_error;
    int input_ok_local = 1;
    try
    {
        make_spectral_binary_layout(n_spins, n_kpoints_global, n_states, n_omegas);
        validate_spectral_local_buffers(
            sf, vxc, vexx, iks, n_spins, n_kpoints_global, n_states, n_omegas);
    }
    catch (const std::exception &err)
    {
        input_ok_local = 0;
        local_error = err.what();
    }
    int input_ok = 0;
    librpa_int::global::mpi_comm_global_h.allreduce(
        &input_ok_local, &input_ok, 1, MPI_MIN);
    if (!input_ok)
    {
        if (local_error.empty())
            local_error = "invalid spectral-function MPI output input on another rank";
        throw LIBRPA_RUNTIME_ERROR(local_error);
    }

    const auto layout =
        make_spectral_binary_layout(n_spins, n_kpoints_global, n_states, n_omegas);
    const auto path = librpa_int::join_path(output_dir, filename);

    std::string mkdir_error;
    int output_ready_local = 1;
    if (librpa_int::global::myid_global == 0)
    {
        try
        {
            librpa_int::create_directories(output_dir.c_str(), 0);
        }
        catch (const std::exception &err)
        {
            output_ready_local = 0;
            mkdir_error = err.what();
        }
    }
    int output_ready = 0;
    librpa_int::global::mpi_comm_global_h.allreduce(
        &output_ready_local, &output_ready, 1, MPI_MIN);
    if (!output_ready)
    {
        if (librpa_int::global::myid_global == 0)
            throw LIBRPA_RUNTIME_ERROR(mkdir_error);
        throw LIBRPA_RUNTIME_ERROR("Failed to prepare spectral-function output directory");
    }
    librpa_int::global::mpi_comm_global_h.barrier();

    MPI_File file = MPI_FILE_NULL;
    int ierr = MPI_File_open(
        librpa_int::global::mpi_comm_global_h.comm, path.c_str(),
        MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    if (ierr != MPI_SUCCESS)
        throw LIBRPA_RUNTIME_ERROR("Failed to open spectral-function MPI output file " + path);

    ierr = MPI_File_set_size(file, layout.file_size);
    if (ierr != MPI_SUCCESS)
        throw LIBRPA_RUNTIME_ERROR("Failed to size spectral-function MPI output file " + path);

    if (librpa_int::global::myid_global == 0)
    {
        MPI_Offset header_offset = 0;
        mpi_file_write_value(file, header_offset,
                             static_cast<std::int32_t>(n_spins), "n_spins");
        mpi_file_write_value(file, header_offset,
                             static_cast<std::int32_t>(n_kpoints_global), "n_kpoints");
        mpi_file_write_value(file, header_offset,
                             static_cast<std::int32_t>(i_state_low), "i_state_low");
        mpi_file_write_value(file, header_offset,
                             static_cast<std::int32_t>(i_state_high), "i_state_high");
        const double efermi_ev = mf.get_efermi() * librpa_int::HA2EV;
        mpi_file_write_value(file, header_offset, efermi_ev, "efermi");
        mpi_file_write_value(file, header_offset,
                             static_cast<std::int32_t>(n_omegas), "n_omegas");

        std::vector<double> omega_values(n_omegas);
        for (int iomega = 0; iomega != n_omegas; ++iomega)
            omega_values[iomega] = omegas_ha[iomega] * librpa_int::HA2EV;
        mpi_file_write_exact(file, layout.omegas, omega_values.data(),
                             omega_values.size() * sizeof(double), "omega grid");
    }

    write_mpi_state_section(
        file, layout.eigenvalues, iks, n_spins, n_kpoints_global, n_states,
        [&](const std::size_t, const int isp, const int ik, const int ist)
        {
            const int i_state = i_state_low + ist;
            return mf.get_eigenvals()[isp](ik, i_state) * librpa_int::HA2EV;
        },
        "eigenvalue section");
    write_mpi_state_section(
        file, layout.vxc, iks, n_spins, n_kpoints_global, n_states,
        [&](const std::size_t idx_local, const int, const int, const int)
        {
            return vxc[idx_local] * librpa_int::HA2EV;
        },
        "vxc section");
    write_mpi_state_section(
        file, layout.vexx, iks, n_spins, n_kpoints_global, n_states,
        [&](const std::size_t idx_local, const int, const int, const int)
        {
            return vexx[idx_local] * librpa_int::HA2EV;
        },
        "vexx section");
    write_mpi_spectral_section(
        file, layout.spectral, iks, n_spins, n_kpoints_global, n_states, n_omegas,
        [&](const std::size_t idx)
        {
            return sf.spectral_function[idx] / librpa_int::HA2EV;
        },
        "spectral section");
    write_mpi_spectral_section(
        file, layout.sigc_re, iks, n_spins, n_kpoints_global, n_states, n_omegas,
        [&](const std::size_t idx)
        {
            return sf.sigc[idx].real() * librpa_int::HA2EV;
        },
        "ReSigc section");
    write_mpi_spectral_section(
        file, layout.sigc_im, iks, n_spins, n_kpoints_global, n_states, n_omegas,
        [&](const std::size_t idx)
        {
            return sf.sigc[idx].imag() * librpa_int::HA2EV;
        },
        "ImSigc section");

    ierr = MPI_File_close(&file);
    if (ierr != MPI_SUCCESS)
        throw LIBRPA_RUNTIME_ERROR("Failed to close spectral-function MPI output file " + path);
}

void driver::task_g0w0()
{
    using std::cout;
    using std::endl;
    using std::setw;
    using std::setprecision;
    using std::fixed;
    using std::map;
    using namespace librpa_int;
    using namespace librpa_int::global;

    profiler.start("g0w0", "G0W0 quasi-particle calculation");

    profiler.start("read_vq_cut", "Load truncated Coulomb");

    // Prepare input specific to G0W0
    auto routing = opts.parallel_routing;
    if (routing == LIBRPA_ROUTING_AUTO) routing = librpa_int::decide_auto_routing(n_atoms, n_kpoints * opts.nfreq);

    if (routing == LIBRPA_ROUTING_RTAU)
    {
        read_Vq_full(driver_params.input_dir, driver_params.prefix_coul_cut, true,
                     driver_params.version_coul_reader,
                     driver::get_bool(driver::opts.use_shrink_abfs));
    }
    else
    {
        // NOTE: local_atpair set during read_data::read_ri.
        read_Vq_row(driver_params.input_dir, driver_params.prefix_coul_cut, opts.vq_threshold,
                    local_atpair, true, driver_params.version_coul_reader,
                    driver::get_bool(driver::opts.use_shrink_abfs));
    }
    profiler.stop("read_vq_cut");

    const auto file_df = driver_params.input_dir + driver_params.fn_dielfunc;
    const bool need_wing = opts.option_dielect_func == 3 || opts.option_dielect_func == 53;
    const bool compute_headwing =
        driver::get_bool(opts.replace_w_head) &&
        (need_wing || opts.option_dielect_func == 4);
    if (compute_headwing)
    {
        if (driver_params.option_velocity == "native")
            prepare_g0w0_native_velocity();
        else
            read_headwing_input(driver_params.input_dir, need_wing);
        write_g0w0_headwing_diagnostics();
    }
    else if (driver::get_bool(opts.replace_w_head) && librpa_int::path_exists(file_df.c_str()))
    {
        if (mpi_comm_global_h.is_root() && should_output())
            std::cout << "Reading dielectric function for head correction" << std::endl;
        std::vector<double> omegas_dielect;
        std::vector<double> dielect_func;
        read_dielec_func(file_df, omegas_dielect, dielect_func);
        ofs_myid << "Dielectric functions read:" << std::endl;
        ofs_myid << "omegas_dielect: " << omegas_dielect << std::endl;
        ofs_myid << "dielect_func:   " << dielect_func << std::endl;
        h.set_dielect_func_imagfreq(omegas_dielect, dielect_func);
    }

    profiler.start("read_vxc", "Load DFT xc potential");
    std::vector<matrix> vxc;
    int flag_read_vxc = read_vxc(driver_params.input_dir + driver_params.fn_vxc_scf, vxc);
    if (flag_read_vxc == 0)
    {
        if (mpi_comm_global_h.is_root() && should_output())
            std::cout << "* Success: Read DFT xc potential, solve quasi-particle equation" << std::endl;
    }
    else
        throw LIBRPA_RUNTIME_ERROR("Failed to read DFT xc potential");
    profiler.stop("read_vxc");

    // Build the self-energy matrix (including exchange and correlation)
    if (driver_params.crpa_enabled)
    {
        auto config = read_crpa_driver_input();
        auto input = config.input();
        auto result = h.build_g0w0_sigma_crpa(opts, input);
        write_crpa_state_outputs(result.get(), config,
                                 join_path(opts.output_dir, "crpa/state_0000"), 0, 0, true);
    }
    else
    {
        h.build_g0w0_sigma(opts);
    }

    auto pds = librpa_int::api::get_dataset_instance(h);
    const std::string band_kpath_file =
        driver_params.input_dir + driver_params.fn_band_kpath_info;
    const bool has_band_kpath = librpa_int::path_exists(band_kpath_file.c_str());

    const int i_state_low = driver_params.i_state_low;
    const int i_state_high = driver_params.i_state_high;
    const int n_states_calc = i_state_high - i_state_low;
    std::vector<double> vexx_all;
    std::vector<cplxdb> sigc_all;
    {
        const size_t n_local = n_states_calc * n_spins * iks_eigvec_this.size();
        const auto vexx = h.get_exx_pot_kgrid(opts, n_spins, iks_eigvec_this, i_state_low, i_state_high);
        std::vector<double> vxc_flat(n_local);
        ofs_myid << "k_para " << opts.use_kpara_scf_eigvec << endl;
        ofs_myid << "iks_eigvec_this " << iks_eigvec_this << endl;
        ofs_myid << "n_local " << n_local << endl;
        for (int isp = 0; isp != n_spins; isp++)
        {
            const auto start_isp = isp * iks_eigvec_this.size() * n_states_calc;
            for (size_t ik_local = 0; ik_local < iks_eigvec_this.size(); ik_local++)
            {
                const auto ik = iks_eigvec_this[ik_local];
                const auto start_k = start_isp + ik_local * n_states_calc;
                for (int i = 0; i < n_states_calc; i++)
                {
                    vxc_flat[start_k+i] = vxc[isp](ik, i+i_state_low);
                }
            }
        }

        const auto qpe = h.get_g0w0_qpe_kgrid(opts, n_spins, iks_eigvec_this,
                                              i_state_low, i_state_high, vxc_flat, vexx);

        if (driver_params.output_gw_spec_func && !has_band_kpath)
        {
            const int sf_state_low = driver_params.sf_state_start;
            const int sf_state_high = driver_params.sf_state_end;

            const int n_kpoints_local = static_cast<int>(iks_eigvec_this.size());
            const auto vxc_sf = slice_state_window(
                vxc_flat, n_spins, n_kpoints_local, i_state_low, i_state_high,
                sf_state_low, sf_state_high);
            const auto vexx_sf = slice_state_window(
                vexx, n_spins, n_kpoints_local, i_state_low, i_state_high,
                sf_state_low, sf_state_high);

            const auto omegas_sf = make_spectral_omegas_ha();
            const auto sf = h.get_g0w0_spectral_function_with_sigc_kgrid(
                opts, n_spins, iks_eigvec_this, sf_state_low, sf_state_high,
                omegas_sf, vxc_sf, vexx_sf);
            if (driver::get_bool(opts.use_kpara_scf_eigvec))
            {
                write_spectral_function_binary_kdistributed(
                    opts.output_dir, "spectral_function_kgrid.dat", omegas_sf, sf,
                    pds->mf, vxc_sf, vexx_sf, iks_eigvec_this, n_spins,
                    n_kpoints, sf_state_low, sf_state_high);
            }
            else if (myid_global == 0)
            {
                write_spectral_function_binary(
                    opts.output_dir, "spectral_function_kgrid.dat", omegas_sf, sf,
                    pds->mf, vxc_sf, vexx_sf, iks_eigvec_this, n_spins,
                    sf_state_low, sf_state_high);
            }
        }

        if (!opts.use_kpara_scf_eigvec)
        {
            // master process already has all data
            if (myid_global == 0)
            {
                vexx_all = vexx;
                sigc_all = qpe.sigc;
            }
        }
        else
        {
            // data parallelized over k-point, collect them to master process
            const size_t n_all = n_states_calc * n_spins * n_kpoints;
            vexx_all.resize(n_all);
            sigc_all.resize(n_all);
            for (int isp = 0; isp != n_spins; isp++)
            {
                const auto st_isp_local = isp * iks_eigvec_this.size() * n_states_calc;
                const auto st_isp = isp * n_kpoints * n_states_calc;
                for (size_t ik_local = 0; ik_local < iks_eigvec_this.size(); ik_local++)
                {
                    const auto ik = iks_eigvec_this[ik_local];
                    const auto st_local = st_isp_local + ik_local * n_states_calc;
                    const auto st = st_isp + ik * n_states_calc;
                    memcpy(sigc_all.data() + st, qpe.sigc.data() + st_local,
                           n_states_calc * sizeof(cplxdb));
                    memcpy(vexx_all.data() + st, vexx.data() + st_local, n_states_calc * sizeof(double));
                }
            }
            mpi_comm_global_h.reduce(MPI_IN_PLACE, vexx_all.data(), n_all, 0, MPI_SUM);
            mpi_comm_global_h.reduce(MPI_IN_PLACE, sigc_all.data(), n_all, 0, MPI_SUM);
            if (myid_global != 0)
            {
                vexx_all.clear();
                sigc_all.clear();
            }
        }
    }

    const std::string banner(124, '-');
    const auto &kfrac_list = pds->pbc.kfrac_list;
    const auto &kpoint_weights = pds->pbc.weight_k;
    const auto &mf = pds->mf;
    const auto &symmetry_context = pds->symmetry_context;
    const auto& full_k_members = symmetry_context.full_kpoint_members;
    const bool output_full_kgrid_from_symmetry =
        driver::get_bool(driver::opts.use_symmetry_gw)
        && symmetry_context.available
        && full_k_members.size() > kfrac_list.size();
    const int n_kpoints_output = output_full_kgrid_from_symmetry
        ? as_int(full_k_members.size())
        : n_kpoints;
    if (flag_read_vxc == 0)
    {
        if (myid_global == 0)
        {
            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++)
                {
                    const size_t start_k = (i_spin * n_kpoints + i_kpoint) * n_states_calc;
                    for (int i = 0; i < n_states_calc; i++)
                    {
                        const int i_state = i + i_state_low;
                        if (std::isnan(sigc_all[start_k+i].real()))
                        {
                            lib_printf(LIBRPA_VERBOSE_WARN, "Warning! QPE solver failed for spin %d, kpoint %d, state %d\n",
                                       i_spin+1, i_kpoint+1, i_state+1);
                        }
                    }
                }
            }

            // Final scientific results remain visible at every non-silent output level.
            constexpr auto result_output_level = LIBRPA_VERBOSE_CRITICAL;
            lib_printf(result_output_level, "Printing quasi-particle energy [unit: eV]\n\n");
            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                for (int i_kpoint = 0; i_kpoint < n_kpoints_output; i_kpoint++)
                {
                    const int i_kpoint_ibz = output_full_kgrid_from_symmetry
                        ? full_k_members[as_size(i_kpoint)].ik_ibz
                        : i_kpoint;
                    const auto &k = output_full_kgrid_from_symmetry
                        ? full_k_members[as_size(i_kpoint)].k_bz
                        : kfrac_list[i_kpoint_ibz];
                    lib_printf(result_output_level,
                               "spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                               i_spin+1, i_kpoint+1, k.x, k.y, k.z);
                    lib_printf(result_output_level, "%124s\n", banner.c_str());
                    lib_printf(result_output_level,
                               "%5s %16s %16s %16s %16s %16s %16s %16s\n",
                               "State", "occ", "e_mf", "v_xc", "v_exx", "ReSigc",
                               "ImSigc", "e_qp");
                    lib_printf(result_output_level, "%124s\n", banner.c_str());
                    const size_t start_k = (i_spin * n_kpoints + i_kpoint_ibz) * n_states_calc;
                    for (int i = 0; i < n_states_calc; i++)
                    {
                        const int i_state = i + i_state_low;
                        const auto occ_state = occupation_number_from_kpoint_weight(
                            mf.get_weight()[i_spin](i_kpoint_ibz, i_state),
                            kpoint_weights.at(as_size(i_kpoint_ibz)));
                        const auto &eks_state = mf.get_eigenvals()[i_spin](i_kpoint_ibz, i_state) * HA2EV;
                        const auto &vxc_state = vxc[i_spin](i_kpoint_ibz, i_state) * HA2EV;
                        const auto &exx_state = vexx_all[start_k+i] * HA2EV;
                        const auto &resigc = sigc_all[start_k+i].real() * HA2EV;
                        const auto &imsigc = sigc_all[start_k+i].imag() * HA2EV;
                        const auto &eqp = eks_state - vxc_state + exx_state + resigc;
                        lib_printf(result_output_level,
                                   "%5d %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f %16.5f\n",
                                   i_state+1, occ_state, eks_state, vxc_state, exx_state, resigc, imsigc, eqp);
                    }
                    lib_printf(result_output_level, "\n");
                }
            }

            std::vector<librpa_int::Vector3_Order<double>> kfrac_energy_qp;
            std::vector<int> output_to_input_kpoint;
            if (output_full_kgrid_from_symmetry)
            {
                kfrac_energy_qp.reserve(full_k_members.size());
                output_to_input_kpoint.reserve(full_k_members.size());
                for (const auto &member : full_k_members)
                {
                    kfrac_energy_qp.push_back(member.k_bz);
                    output_to_input_kpoint.push_back(member.ik_ibz);
                }
            }
            lib_printf(result_output_level, "Band analysis on the k-grid\n");
            print_band_analysis(
                mf, output_full_kgrid_from_symmetry ? kfrac_energy_qp : kfrac_list,
                output_to_input_kpoint, vxc, vexx_all, sigc_all, i_state_low, n_states_calc);

            if (driver_params.output_energy_qp)
            {
                write_energy_qp(
                    mf, output_full_kgrid_from_symmetry ? kfrac_energy_qp : kfrac_list,
                    output_to_input_kpoint, vxc, vexx_all, sigc_all, n_kpoints, i_state_low,
                    n_states_calc, kpoint_weights);
            }
        }
    }

    /* Below we handle the band k-points data
     * First load the information of k-points along the k-path */
    if (!has_band_kpath)
    {
        lib_printf_root(LIBRPA_VERBOSE_WARN,
                        "Band k-path file %s is not found under %s, skip band calculation\n",
                        driver_params.fn_band_kpath_info.c_str(), driver_params.input_dir.c_str());
        profiler.stop("g0w0");
        return;
    }

    try
    {
        read_band_kpath_info(band_kpath_file);
        lib_printf_root("Loaded band k-path file %s, starting band calculation\n",
                        band_kpath_file.c_str());
    }
    catch (const std::exception& err)
    {
        if (mpi_comm_global_h.is_root() && should_output(LIBRPA_VERBOSE_CRITICAL))
        {
            std::cout << "Error in reading band k-path file " << band_kpath_file << " (" << err.what() << ")"
                      << std::endl;
        }
        throw err;
    }

    mpi_comm_global_h.barrier();
    const int nkpts_band = kfrac_band.size();

    if (mpi_comm_global_h.is_root() && should_output())
    {
        std::cout << "Band k-points to compute:\n";
        for (int ik = 0; ik < nkpts_band; ik++)
        {
            const auto &k = kfrac_band[ik];
            lib_printf("%5d %12.7f %12.7f %12.7f\n", ik + 1, k.x, k.y, k.z);
        }
    }
    mpi_comm_global_h.barrier();
    profiler.start("g0w0_load_band_mf", "Read eigen solutions at band kpoints");
    read_band_meanfield_data(driver_params.input_dir);
    const int nkpts_band_this = iks_band_eigvec_this.size();
    ofs_myid << "Number of band k-points on this process: " << nkpts_band_this << endl;
    ofs_myid << "iks_band_eigvec_this: " << iks_band_eigvec_this << endl;
    profiler.stop("g0w0_load_band_mf");

    const int i_state_low_band = driver_params.i_state_low;
    const int i_state_high_band = driver_params.i_state_high;
    const int n_states_band_calc = i_state_high_band - i_state_low_band;

    profiler.start("read_vxc_band", "Load DFT xc potential for band");
    const auto vxc_band_all = read_vxc_band(driver_params.input_dir, n_states, n_spins, kfrac_band.size());
    profiler.stop("read_vxc_band");

    // Call APIs to get band EXX potentials and correlation self-energies,
    // then collect them to the master process
    std::vector<double> vexx_band_all;
    std::vector<cplxdb> sigc_band_all;
    {
        const auto &iks_this = iks_band_eigvec_this;
        const size_t n_local = n_states_band_calc * n_spins * iks_this.size();
        const auto vexx_band =
            h.get_exx_pot_band_k(opts, n_spins, iks_this, i_state_low_band, i_state_high_band);

        std::vector<double> vxc_this(n_local);
        for (int isp = 0; isp != n_spins; isp++)
        {
            const auto start_isp = isp * iks_this.size() * n_states_band_calc;
            for (size_t ik_this = 0; ik_this < iks_this.size(); ik_this++)
            {
                const auto ik = iks_this[ik_this];
                const auto start_k = start_isp + ik_this * n_states_band_calc;
                for (int i = 0; i < n_states_band_calc; i++)
                {
                    vxc_this[start_k+i] = vxc_band_all[isp](ik, i+i_state_low_band);
                }
            }
        }
        const auto qpe_band =
            h.get_g0w0_qpe_band_k(opts, n_spins, iks_this, i_state_low_band,
                                  i_state_high_band, vxc_this, vexx_band);

        if (driver_params.output_gw_spec_func)
        {
            const int sf_state_low = driver_params.sf_state_start;
            const int sf_state_high = driver_params.sf_state_end;

            const int n_kpoints_local = static_cast<int>(iks_this.size());
            const auto vxc_sf = slice_state_window(
                vxc_this, n_spins, n_kpoints_local, i_state_low_band, i_state_high_band,
                sf_state_low, sf_state_high);
            const auto vexx_sf = slice_state_window(
                vexx_band, n_spins, n_kpoints_local, i_state_low_band, i_state_high_band,
                sf_state_low, sf_state_high);

            const auto omegas_sf = make_spectral_omegas_ha();
            const auto sf = h.get_g0w0_spectral_function_with_sigc_band_k(
                opts, n_spins, iks_this, sf_state_low, sf_state_high,
                omegas_sf, vxc_sf, vexx_sf);
            if (driver::get_bool(opts.use_kpara_scf_eigvec))
            {
                write_spectral_function_binary_kdistributed(
                    opts.output_dir, "spectral_function_band.dat", omegas_sf, sf,
                    pds->mf_band, vxc_sf, vexx_sf, iks_this, n_spins,
                    n_kpoints_band, sf_state_low, sf_state_high);
            }
            else if (myid_global == 0)
            {
                write_spectral_function_binary(
                    opts.output_dir, "spectral_function_band.dat", omegas_sf, sf,
                    pds->mf_band, vxc_sf, vexx_sf, iks_this, n_spins,
                    sf_state_low, sf_state_high);
            }
        }

        profiler.start("collect_exx_sigc_band");
        if (!opts.use_kpara_scf_eigvec)
        {
            // master process already has all data
            if (myid_global == 0)
            {
                vexx_band_all = vexx_band;
                sigc_band_all = qpe_band.sigc;
            }
        }
        else
        {
            // data parallelized over k-point, collect them to master process
            const auto n_kpts = n_kpoints_band;
            const size_t n_all = n_states_band_calc * n_spins * n_kpts;
            vexx_band_all.resize(n_all);
            sigc_band_all.resize(n_all);
            for (int isp = 0; isp != n_spins; isp++)
            {
                const auto st_isp_local = isp * iks_this.size() * n_states_band_calc;
                const auto st_isp = isp * n_kpts * n_states_band_calc;
                for (size_t ik_this = 0; ik_this < iks_this.size(); ik_this++)
                {
                    const auto ik = iks_this[ik_this];
                    const auto st_local = st_isp_local + ik_this * n_states_band_calc;
                    const auto st = st_isp + ik * n_states_band_calc;
                    memcpy(sigc_band_all.data() + st, qpe_band.sigc.data() + st_local,
                           n_states_band_calc * sizeof(cplxdb));
                    memcpy(vexx_band_all.data() + st, vexx_band.data() + st_local,
                           n_states_band_calc * sizeof(double));
                }
            }
            mpi_comm_global_h.reduce(MPI_IN_PLACE, vexx_band_all.data(), n_all, 0, MPI_SUM);
            mpi_comm_global_h.reduce(MPI_IN_PLACE, sigc_band_all.data(), n_all, 0, MPI_SUM);
            if (myid_global != 0)
            {
                vexx_band_all.clear();
                sigc_band_all.clear();
            }
        }
        profiler.stop("collect_exx_sigc_band");
    }

    profiler.start("output_g0w0_band");
    const auto &mf_band = pds->mf_band;
    {
        const int n_kpts = n_kpoints_band;
        if (myid_global == 0)
        {
            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                for (int i_kpoint = 0; i_kpoint < n_kpts; i_kpoint++)
                {
                    const int start_k = (i_spin * n_kpts + i_kpoint) * n_states_band_calc;
                    for (int i = 0; i < n_states_band_calc; i++)
                    {
                        const int i_state = i + i_state_low_band;
                        if (std::isnan(sigc_band_all[start_k+i].real()))
                        {
                            lib_printf(LIBRPA_VERBOSE_WARN, "Warning! QPE solver failed for spin %d, kpoint %d, state %d\n",
                                       i_spin+1, i_kpoint+1, i_state+1);
                        }
                    }
                }
            }

            for (int i_spin = 0; i_spin < n_spins; i_spin++)
            {
                std::ofstream ofs_ks, ofs_hf, ofs_gw;
                std::stringstream fn_ks, fn_hf, fn_gw;

                fn_gw << "GW_band_spin_" << i_spin + 1 << ".dat";
                fn_hf << "EXX_band_spin_" << i_spin + 1 << ".dat";
                fn_ks << "KS_band_spin_" << i_spin + 1 << ".dat";

                ofs_gw.open(fn_gw.str());
                ofs_hf.open(fn_hf.str());
                ofs_ks.open(fn_ks.str());

                ofs_gw << fixed;
                ofs_hf << fixed;
                ofs_ks << fixed;

                for (int i_kpoint = 0; i_kpoint < n_kpts; i_kpoint++)
                {
                    const auto &k = kfrac_band[i_kpoint];
                    const int start_k = (i_spin * n_kpts + i_kpoint) * n_states_band_calc;

                    ofs_ks << setw(5) << i_kpoint + 1 << setw(15) << setprecision(7) << k.x << setw(15) << setprecision(7) << k.y << std::setw(15) << std::setprecision(7) << k.z;
                    ofs_gw << setw(5) << i_kpoint + 1 << setw(15) << setprecision(7) << k.x << setw(15) << setprecision(7) << k.y << std::setw(15) << std::setprecision(7) << k.z;
                    ofs_hf << setw(5) << i_kpoint + 1 << setw(15) << setprecision(7) << k.x << setw(15) << setprecision(7) << k.y << std::setw(15) << std::setprecision(7) << k.z;
                    for (int i = 0; i < n_states_band_calc; i++)
                    {
                        const int i_state = i + i_state_low_band;
                        const auto occ_state = mf_band.get_weight()[i_spin](i_kpoint, i_state) * mf_band.get_n_kpoints();
                        const auto eks_state = mf_band.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                        const auto vxc_state = vxc_band_all[i_spin](i_kpoint, i_state) * HA2EV;
                        const auto exx_state = vexx_band_all[start_k+i] * HA2EV;
                        const auto resigc = sigc_band_all[start_k+i].real() * HA2EV;
                        // const auto &imsigc = sigc_band_all[start_k+i].imag() * HA2EV;
                        const auto eqp = eks_state - vxc_state + exx_state + resigc;
                        ofs_ks << setw(15) << std::setprecision(5) << occ_state << setw(15) << setprecision(5) << eks_state;
                        ofs_gw << setw(15) << std::setprecision(5) << occ_state << setw(15) << setprecision(5) << eqp;
                        ofs_hf << setw(15) << std::setprecision(5) << occ_state << setw(15) << setprecision(5) << eks_state - vxc_state + exx_state;
                    }
                    ofs_gw << "\n";
                    ofs_hf << "\n";
                    ofs_ks << "\n";
                }
            }
        }
    }
    if (myid_global == 0)
    {
        lib_printf(LIBRPA_VERBOSE_CRITICAL, "Band analysis along the k-path\n");
        print_band_analysis(mf_band, kfrac_band, {}, vxc_band_all, vexx_band_all,
                            sigc_band_all, i_state_low_band, n_states_band_calc);
    }
    profiler.stop("output_g0w0_band");

    // // TODO: parallelize analytic continuation and QPE solver among tasks
    // if (mpi_comm_global_h.is_root())
    // {
    //     const auto &mf = meanfield_band;
    //     const auto n_spin = mf.get_n_spins();
    //     const auto n_kpt = mf.get_n_kpoints();
    //     map<int, map<int, map<int, double>>> e_qp_all;
    //     map<int, map<int, map<int, cplxdb>>> sigc_all;
    //     const auto efermi = mf.get_efermi();
    //
    //     // Spectral function IO handler
    //     std::ofstream wf_sf;
    //     std::vector<cplxdb> omegas;
    //     // FIXME: should check negative n_omegas_sf beforehand for safety and clarity
    //     int n_omegas_sf = (driver_params.sf_omega_end - driver_params.sf_omega_start) / driver_params.sf_omega_step + 1;
    //     const double gf_shift_ha = opts.sf_gf_omega_shift;
    //     const double sigc_shift_ha = opts.sf_sigc_omega_shift;
    //
    //     if (driver_params.output_gw_spec_func && n_omegas_sf > 0)
    //     {
    //         wf_sf.open("spectral_function_band.dat", std::ios::out | std::ios::binary);
    //
    //         // Initialize the real frequencies
    //         std::vector<double> omegas_db(n_omegas_sf, 0.0);
    //         const double start = driver_params.sf_omega_start;
    //         const double step = driver_params.sf_omega_step;
    //         std::generate(omegas_db.begin(), omegas_db.end(),
    //                       [i = 0, start, step]() mutable
    //                       {
    //                           return start + (i++) * step;
    //                       });
    //         // Output frequency setup
    //         wf_sf.write((char *) &driver_params.sf_omega_start, sizeof(double));
    //         wf_sf.write((char *) &driver_params.sf_omega_end, sizeof(double));
    //         wf_sf.write((char *) &driver_params.sf_omega_step, sizeof(double));
    //         wf_sf.write((char *) &opts.sf_gf_omega_shift, sizeof(double));
    //         wf_sf.write((char *) &opts.sf_sigc_omega_shift, sizeof(double));
    //         // Output dimension information and Fermi level
    //         wf_sf.write((char *) &n_omegas_sf, sizeof(int));
    //         wf_sf.write((char *) &n_spin, sizeof(int));
    //         wf_sf.write((char *) &n_kpt, sizeof(int));
    //         // Check actual start and end indices of band
    //         int sf_state_start = max(driver_params.sf_state_start, 0);
    //         int sf_state_end = min(driver_params.sf_state_end, mf.get_n_bands() - 1);
    //         wf_sf.write((char *) &sf_state_start, sizeof(int));
    //         wf_sf.write((char *) &sf_state_end, sizeof(int));
    //         wf_sf.write((char *) &efermi, sizeof(double));
    //         // Output the frequencies
    //         wf_sf.write((char *) omegas_db.data(), sizeof(double) * n_omegas_sf);
    //
    //         omegas.resize(n_omegas_sf);
    //         std::transform(omegas_db.cbegin(), omegas_db.cend(), omegas.begin(),
    //                [](double x) { return std::complex<double>(x / HA2EV, 0.0); });
    //     }
    //
    //     for (int i_spin = 0; i_spin < n_spin; i_spin++)
    //     {
    //         for (int i_kpoint = 0; i_kpoint < n_kpt; i_kpoint++)
    //         {
    //             const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];
    //             for (int i_state = 0; i_state < mf.get_n_bands(); i_state++)
    //             {
    //                 const auto &eks_state = mf.get_eigenvals()[i_spin](i_kpoint, i_state);
    //                 const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state];
    //                 const auto &vxc_state = vxc_band[i_spin](i_kpoint, i_state);
    //                 std::vector<cplxdb> sigc_state;
    //                 for (const auto &freq: chi0.tfg.get_freq_nodes())
    //                 {
    //                     sigc_state.push_back(sigc_sk.at(freq)(i_state, i_state));
    //                 }
    //                 librpa_int::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_state);
    //
    //                 if (driver_params.output_gw_spec_func && n_omegas_sf > 0)
    //                 {
    //                     if (i_state >= driver_params.sf_state_start &&
    //                         i_state <= driver_params.sf_state_end)
    //                     {
    //                         const auto sf = librpa_int::get_specfunc(
    //                             pade, omegas, efermi, eks_state, vxc_state, exx_state,
    //                             sigc_shift_ha, gf_shift_ha);
    //                         wf_sf.write((char *) &eks_state, sizeof(double));
    //                         wf_sf.write((char *) &exx_state, sizeof(double));
    //                         wf_sf.write((char *) &vxc_state, sizeof(double));
    //                         wf_sf.write((char *) sf.data(), sizeof(double) * n_omegas_sf);
    //                     }
    //                 }
    //
    //                 double e_qp;
    //                 cplxdb sigc;
    //                 int flag_qpe_solver = librpa_int::qpe_solver_pade_self_consistent(
    //                     pade, eks_state, efermi, vxc_state, exx_state, e_qp, sigc);
    //                 if (flag_qpe_solver == 0)
    //                 {
    //                     e_qp_all[i_spin][i_kpoint][i_state] = e_qp;
    //                     sigc_all[i_spin][i_kpoint][i_state] = sigc;
    //                 }
    //                 else
    //                 {
    //                     printf("Warning! QPE solver failed for spin %d, kpoint %d, state %d\n",
    //                             i_spin+1, i_kpoint+1, i_state+1);
    //                     e_qp_all[i_spin][i_kpoint][i_state] = std::numeric_limits<double>::quiet_NaN();
    //                     sigc_all[i_spin][i_kpoint][i_state] = std::numeric_limits<cplxdb>::quiet_NaN();
    //                 }
    //             }
    //         }
    //     }
    //
    //     if (driver_params.output_gw_spec_func)
    //     {
    //         wf_sf.close();
    //     }
    //
    //     // display results
    //     for (int i_spin = 0; i_spin < mf.get_n_spins(); i_spin++)
    //     {
    //         std::ofstream ofs_ks;
    //         std::ofstream ofs_hf;
    //         std::ofstream ofs_gw;
    //         std::stringstream fn;
    //
    //         fn << "GW_band_spin_" << i_spin + 1 << ".dat";
    //         ofs_gw.open(fn.str());
    //
    //         fn.str("");
    //         fn.clear();
    //         fn << "EXX_band_spin_" << i_spin + 1 << ".dat";
    //         ofs_hf.open(fn.str());
    //
    //         fn.str("");
    //         fn.clear();
    //         fn << "KS_band_spin_" << i_spin + 1 << ".dat";
    //         ofs_ks.open(fn.str());
    //
    //         ofs_gw << std::fixed;
    //         ofs_hf << std::fixed;
    //         ofs_ks << std::fixed;
    //
    //         for (int i_kpoint = 0; i_kpoint < mf.get_n_kpoints(); i_kpoint++)
    //         {
    //             const auto &k = kfrac_band[i_kpoint];
    //             ofs_ks << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7) << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15) << std::setprecision(7) << k.z;
    //             ofs_gw << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7) << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15) << std::setprecision(7) << k.z;
    //             ofs_hf << std::setw(5) << i_kpoint + 1 << std::setw(15) << std::setprecision(7) << k.x << std::setw(15) << std::setprecision(7) << k.y << std::setw(15) << std::setprecision(7) << k.z;
    //             for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
    //             {
    //                 const auto &occ_state = mf.get_weight()[i_spin](i_kpoint, i_state);
    //                 const auto &eks_state = mf.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
    //                 const auto &exx_state = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
    //                 const auto &vxc_state = vxc_band[i_spin](i_kpoint, i_state) * HA2EV;
    //                 // const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
    //                 // const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
    //                 const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
    //                 ofs_ks << std::setw(15) << std::setprecision(5) << occ_state << std::setw(15) << std::setprecision(5) << eks_state;
    //                 ofs_gw << std::setw(15) << std::setprecision(5) << occ_state << std::setw(15) << std::setprecision(5) << eqp;
    //                 ofs_hf << std::setw(15) << std::setprecision(5) << occ_state << std::setw(15) << std::setprecision(5) << eks_state - vxc_state + exx_state;
    //             }
    //             ofs_gw << "\n";
    //             ofs_hf << "\n";
    //             ofs_ks << "\n";
    //         }
    //     }
    // }
    // profiler.stop("g0w0_solve_qpe");


    profiler.stop("g0w0");
}
