#include "native_velocity.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

#include "../io/global_io.h"
#include "../mpi/collective_stage.h"
#include "../utils/constants.h"
#include "../utils/error.h"
#include "headwing_velocity.h"

namespace
{
using namespace librpa_int;

LocalizedMatrix decode_operator(const LibrpaLocalizedOperator &input, int dimension)
{
    if (dimension <= 0 || input.dimension != dimension || input.n_blocks <= 0 || !input.blocks)
        throw LIBRPA_INVALID_ARGUMENT("Native operator dimensions or blocks are incomplete");
    LocalizedMatrix result;
    for (int block = 0; block < input.n_blocks; ++block)
    {
        const auto &source = input.blocks[block];
        if (!source.values_ri || (source.row_major != 0 && source.row_major != 1))
            throw LIBRPA_INVALID_ARGUMENT("Native operator storage is invalid");
        Matz matrix(dimension, dimension, source.row_major ? MAJOR::ROW : MAJOR::COL);
        for (std::size_t i = 0; i < matrix.size(); ++i)
            matrix.ptr()[i] = {source.values_ri[2 * i], source.values_ri[2 * i + 1]};
        const Vector3_Order<int> translation{source.translation[0], source.translation[1],
                                             source.translation[2]};
        if (!result.emplace(translation, std::move(matrix)).second)
            throw LIBRPA_INVALID_ARGUMENT("Native operator repeats a lattice translation");
    }
    return result;
}

}  // namespace

librpa_int::NativeVelocityResult librpa_int::build_native_velocity(
    const MeanField &mf, const PeriodicBoundaryData &pbc, const MpiCommHandler &comm,
    const LibrpaNativeVelocityInput *root_input, const SymmetryContext *symmetry)
{
    const int n = mf.get_n_states(), nk = mf.get_n_kpoints(), ns = mf.get_n_spins();
    const int nao = mf.get_n_aos(), nspinor = mf.get_n_spinor();
    const std::size_t count = static_cast<std::size_t>(ns) * nk * 3 * n * n;
    NativeVelocityResult result;
    auto &velocity = result.velocity;
    HeadwingAoOperators operators;
    std::vector<LocalizedMatrix> hamiltonians;
    std::array<double, 2> coulomb_normalization{};

    collective_all_stage(
        comm, "G0W0 native velocity input",
        [&]
        {
            const auto period = pbc.period;
            const long long nk_full = static_cast<long long>(period.x) * period.y * period.z;
            const bool collinear = nspinor == 1 && (ns == 1 || ns == 2);
            const bool spinor = nspinor == 2 && ns == 1;
            if ((!collinear && !spinor) || n <= 0 || nao * nspinor != n)
                throw LIBRPA_RUNTIME_ERROR(
                    "G0W0 native velocity requires one/two collinear spins or one two-component "
                    "spinor channel and complete AO-spinor/band frames");
            if (nk <= 0 || static_cast<int>(pbc.kfrac_list.size()) != nk ||
                static_cast<int>(pbc.weight_k.size()) != nk)
                throw LIBRPA_RUNTIME_ERROR(
                    "G0W0 native velocity requires matching SCF k points and weights");
            const bool reduced_grid = nk != nk_full;
            if (reduced_grid &&
                (!spinor || !symmetry || symmetry->kstars.size() != static_cast<std::size_t>(nk) ||
                 symmetry->count_kstar_members() != static_cast<std::size_t>(nk_full) ||
                 symmetry->spin_operations.empty()))
                throw LIBRPA_RUNTIME_ERROR(
                    "IBZ native SOC velocity requires magnetic k-stars covering the full mesh");
            for (int ik = 0; ik != nk; ++ik)
            {
                const auto weight = pbc.weight_k[ik];
                const double expected =
                    reduced_grid
                        ? static_cast<double>(find_symmetry_kstar_for_ibz_kpoint(
                                                  *symmetry, pbc.kfrac_list[ik]).members.size()) /
                              nk_full
                        : 1.0 / nk;
                if (!std::isfinite(weight) || std::abs(weight - expected) > 1e-8)
                    throw LIBRPA_RUNTIME_ERROR(
                        "G0W0 native velocity k weights disagree with the full-mesh star counts");
            }
            if (count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
                throw LIBRPA_RUNTIME_ERROR("Native velocity exceeds the MPI buffer-count limit");
            velocity.resize(count);
            if (!comm.is_root()) return;
            if (!root_input || root_input->n_spins != ns || !root_input->hamiltonian ||
                root_input->overlap.dimension != n)
                throw LIBRPA_INVALID_ARGUMENT(
                    "Native operators do not match the complete SCF frame");
            operators.dimension = n;
            operators.overlap = decode_operator(root_input->overlap, n);
            for (int axis = 0; axis < 3; ++axis)
                operators.position[axis] = decode_operator(root_input->position[axis], n);
            for (int spin = 0; spin < ns; ++spin)
                hamiltonians.push_back(decode_operator(root_input->hamiltonian[spin], n));
            coulomb_normalization = {root_input->inplane_area_bohr2,
                                     root_input->auxiliary_monopole_norm_squared};
        });

    auto &residuals = result.residuals;
    for (int spin = 0; spin < ns; ++spin)
        for (int ik = 0; ik < nk; ++ik)
        {
            Matz ket, component;
            collective_all_stage(comm, "G0W0 native eigenframe shape",
                                 [&]
                                 {
                                     ket = Matz(n, n, MAJOR::ROW);
                                     component = Matz(nao, n, MAJOR::ROW);
                                 });
            for (int spinor = 0; spinor < nspinor; ++spinor)
            {
                const auto *wfc = mf.find_wfc(spin, spinor, ik);
                collective_all_stage(
                    comm, "G0W0 native eigenframe component shape",
                    [&]
                    {
                        if (wfc != nullptr && wfc->size != 0 && (wfc->nr != n || wfc->nc != nao))
                            throw LIBRPA_RUNTIME_ERROR(
                                "G0W0 native velocity found a partial SCF spinor block");
                    });
                const int owner = find_mpi_owner_rank(wfc != nullptr && wfc->size != 0, comm.comm);
                if (owner < 0)
                    throw LIBRPA_RUNTIME_ERROR(
                        "G0W0 native velocity is missing a SCF spinor block");
                collective_all_stage(comm, "G0W0 native eigenframe transfer",
                                     [&]
                                     {
                                         if (comm.myid != owner) return;
                                         // MeanField has band rows, spatial AO columns. Native
                                         // H/S/r use AO-major, spinor-minor ket rows.
                                         for (int ao = 0; ao < nao; ++ao)
                                             for (int band = 0; band < n; ++band)
                                                 component(ao, band) = (*wfc)(band, ao);
                                     });
                comm.bcast(component.ptr(), nao * n, owner);
                for (int ao = 0; ao < nao; ++ao)
                    for (int band = 0; band < n; ++band)
                        ket(ao * nspinor + spinor, band) = component(ao, band);
            }
            collective_all_stage(
                comm, "G0W0 native velocity frame",
                [&]
                {
                    if (!comm.is_root()) return;
                    std::vector<double> eigenvalues(n);
                    for (int band = 0; band < n; ++band)
                        eigenvalues[band] = mf.get_eigenvals().at(spin)(ik, band);
                    const auto result = compute_localized_velocity_in_frame(
                        hamiltonians.at(spin), operators, pbc.kfrac_list.at(ik), pbc.latvec,
                        eigenvalues, ket);
                    residuals[0] = std::max(residuals[0], result.metric_residual);
                    residuals[1] = std::max(residuals[1], result.hamiltonian_frame_residual_ha);
                    residuals[2] =
                        std::max(residuals[2], result.hamiltonian_hermiticity_residual_ha);
                    residuals[3] =
                        std::max(residuals[3], result.velocity_upper_diagonal_imaginary_ha_bohr);
                    if (result.metric_residual > 1e-6 ||
                        result.hamiltonian_frame_residual_ha > 1e-4 / HA2EV ||
                        result.hamiltonian_hermiticity_residual_ha > 1e-4 / HA2EV ||
                        result.velocity_upper_diagonal_imaginary_ha_bohr > 1e-8)
                    {
                        std::ostringstream message;
                        message << std::scientific << "spin=" << spin << " k=" << ik
                                << " metric=" << result.metric_residual
                                << " frame_Ha=" << result.hamiltonian_frame_residual_ha
                                << " hermiticity_Ha=" << result.hamiltonian_hermiticity_residual_ha
                                << " diagonal_imag_HaBohr="
                                << result.velocity_upper_diagonal_imaginary_ha_bohr;
                        throw LIBRPA_RUNTIME_ERROR(message.str());
                    }
                    for (int alpha = 0; alpha < 3; ++alpha)
                        for (int i = 0; i < n; ++i)
                            for (int j = 0; j < n; ++j)
                            {
                                const std::size_t offset =
                                    ((((static_cast<std::size_t>(spin) * nk + ik) * 3 + alpha) * n +
                                      i) *
                                         n +
                                     j);
                                velocity[offset] = result.velocity_ha_bohr[alpha](i, j);
                            }
                });
        }
    // The full-grid buffer can exceed the MPI int count limit on a large
    // symmetry-reduced case, so split the transfer.
    comm.bcast_large(velocity.data(), velocity.size(), 0);
    result.coulomb_normalization = coulomb_normalization;
    return result;
}
