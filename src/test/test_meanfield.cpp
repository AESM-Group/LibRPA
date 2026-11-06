#include "../core/meanfield.h"
#include "../core/symmetry_context.h"
#include <cassert>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

#include "testutils.h"

void test_BCC_He_gamma_minimal_basis_aims()
{
    using namespace librpa_int;

    const int nk = 1;
    MeanField mf(1, 1, 8, 8);
    mf.get_efermi() = 0.240386888648512;
    mf.get_weight()[0].zero_out();
    mf.get_weight()[0](0, 0) = mf.get_weight()[0](0, 1) = 2.0 / nk;
    std::vector<double> eig {
         -0.649240864,
         -0.577333356,
          0.783349882,
          0.783349885,
          0.783349885,
          1.130014638,
          1.130014642,
          1.130014642,
    };
    std::vector<complex<double>> wkc_gamma_T
    {
       0.635395184437372,  0.747426907197624, -0.000000000000000,  0.000000000000000,  0.000000000000000, -0.000000000000000,  0.000000000000000, -0.000000000000000,
       0.000000000000000,  0.000000000000000,  0.000000046406191, -0.379780880349053, -0.566034156121301, -0.000000021962747, -0.074956542623863,  0.783355936987987,
       0.000000000000000, -0.000000000000000,  0.000000013452817,  0.566034156121300, -0.379780880349051, -0.000000008499556,  0.783355936987989,  0.074956542623864,
       0.000000000000000,  0.000000000000000, -0.681636400858004, -0.000000014684412, -0.000000046031303, -0.786933928164525, -0.000000006368929, -0.000000022672482,
       0.635395184437320, -0.747426907197668, -0.000000000000000, -0.000000000000000,  0.000000000000000,  0.000000000000000,  0.000000000000000,  0.000000000000001,
      -0.000000000000000,  0.000000000000000, -0.000000046406191,  0.379780880349045,  0.566034156121291, -0.000000021962747, -0.074956542623865,  0.783355936987997,
      -0.000000000000000,  0.000000000000000, -0.000000013452817, -0.566034156121295,  0.379780880349048, -0.000000008499557,  0.783355936987995,  0.074956542623863,
       0.000000000000000, -0.000000000000000,  0.681636400857993,  0.000000014684413,  0.000000046031304, -0.786933928164535, -0.000000006368930, -0.000000022672483,
    };
    mf.get_eigenvectors()[0][0][0].create(8, 8);
    for (int ib = 0; ib < 8; ib++)
        mf.get_eigenvals()[0](0, ib) = eig[ib];
    for (int iw = 0; iw < 8; iw++)
        for (int ib = 0; ib < 8; ib++)
            mf.get_eigenvectors()[0][0][0](ib, iw) = wkc_gamma_T[iw * 8 + ib];

    // test density matrix
    const auto dmat_gamma = mf.get_dmat_cplx(0, 0, 0, 0);
    const complex<double> thres = 1e-10;
    assert(fequal(dmat_gamma(0, 0), { 0.962374022009208e-00, 0}, thres));
    assert(fequal(dmat_gamma(4, 0), {-1.549199411968699e-01, 0}, thres));
    assert(fequal(dmat_gamma(0, 4), {-1.549199411968699e-01, 0}, thres));
    assert(fequal(dmat_gamma(4, 4), { 0.962374022009208e+00, 0}, thres));

    // test Green's function G(i \tau). approaches to (minus) density matrix for \tau -> 0^-
    const auto gf_gamma = mf.get_gf_cplx_imagtime(0, 0, 0, 0, -1e-12);
    assert(fequal(gf_gamma(0, 0), {-0.962374022009208e-00, 0}, thres));
    assert(fequal(gf_gamma(4, 0), { 1.549199411968699e-01, 0}, thres));
    assert(fequal(gf_gamma(0, 4), { 1.549199411968699e-01, 0}, thres));
    assert(fequal(gf_gamma(4, 4), {-0.962374022009208e+00, 0}, thres));
}

void test_state_index_energy_bounds()
{
    using namespace librpa_int;

    MeanField mf(1, 2, 4, 4);
    const std::vector<std::vector<double>> eig {
        {-3.0, -1.0, 0.5, 2.0},
        {-2.0, -0.5, 1.5, 3.0},
    };
    for (int ik = 0; ik != 2; ++ik)
    {
        for (int ist = 0; ist != 4; ++ist)
        {
            mf.get_eigenvals()[0](ik, ist) = eig[ik][ist];
        }
    }

    assert(mf.get_max_state_below_energy(-3.5) == -1);
    assert(mf.get_max_state_below_energy(-1.5) == 0);
    assert(mf.get_max_state_below_energy(0.75) == 1);
    assert(mf.get_max_state_below_energy(4.0) == 3);

    assert(mf.get_min_state_above_energy(4.0) == 4);
    assert(mf.get_min_state_above_energy(1.0) == 3);
    assert(mf.get_min_state_above_energy(-0.75) == 2);
    assert(mf.get_min_state_above_energy(-4.0) == 0);
}

void test_find_highest_occupied_state()
{
    using namespace librpa_int;

    MeanField mf(1, 2, 4, 1);
    mf.get_eigenvals()[0](0, 0) = -4.0;
    mf.get_eigenvals()[0](0, 1) = -1.0;
    mf.get_eigenvals()[0](0, 2) = 0.0;
    mf.get_eigenvals()[0](0, 3) = 1.0;
    mf.get_eigenvals()[0](1, 0) = -3.0;
    mf.get_eigenvals()[0](1, 1) = -0.5;
    mf.get_eigenvals()[0](1, 2) = 0.5;
    mf.get_eigenvals()[0](1, 3) = 2.0;
    mf.get_weight()[0].zero_out();
    mf.get_weight()[0](0, 1) = 1.0;
    mf.get_weight()[0](1, 0) = 1.0;

    assert(mf.find_highest_occupied_state(0) == std::make_pair(0, 1));
    assert(mf.find_highest_occupied_state(0, 1) == std::make_pair(1, 0));

    mf.get_weight()[0].zero_out();
    assert(mf.find_highest_occupied_state(0) == std::make_pair(-1, -1));

    mf.get_weight()[0](1, 2) = 0.75e-8;
    assert(mf.find_highest_occupied_state(0) == std::make_pair(1, 2));
}

void test_dmat_cplx_Rs_matches_single_R_accumulation()
{
    using namespace librpa_int;

    const int nk = 2;
    MeanField mf(1, nk, 1, 1);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](1, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0 / nk;
    mf.get_weight()[0](1, 0) = 2.0 / nk;
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][1].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {1.0, 0.0};
    mf.get_eigenvectors()[0][0][1](0, 0) = {1.0, 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list {
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const std::vector<Vector3_Order<int>> Rs {
        {0, 0, 0},
        {1, 0, 0},
    };

    const auto dmat_Rs = mf.get_dmat_cplx_Rs(0, 0, 0, kfrac_list, Rs);
    if (dmat_Rs.size() != Rs.size())
        throw std::runtime_error("get_dmat_cplx_Rs returned an unexpected number of R blocks");
    for (const auto &R : Rs)
    {
        const auto dmat_R = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
        if (!fequal(dmat_Rs.at(R)(0, 0), dmat_R(0, 0), {1e-12, 0.0}))
            throw std::runtime_error("get_dmat_cplx_Rs differs from get_dmat_cplx_R");
    }
}

void test_symmetry_context_kstar_restored_dmat_uses_full_star_phases()
{
    using namespace librpa_int;

    librpa_int::SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    librpa_int::SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    librpa_int::SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};

    librpa_int::SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(atom_rotation);
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(atom_rotation);
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 1);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {1.0, 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const Vector3_Order<int> R{1, 0, 0};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    const auto direct_ibz = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
    const auto restored = get_symmetry_restored_dmat_cplx_R(
        ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, R, atom_nw);

    if (std::abs(direct_ibz(0, 0)) < 1e-12)
        throw std::runtime_error("direct IBZ density matrix unexpectedly vanished");
    if (std::abs(restored(0, 0)) > 1e-12)
        throw std::runtime_error("ABACUS k-star restored density matrix did not use full-star phases");
}

void test_symmetry_context_kstar_restore_skips_full_grid()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    SymmetryKStar star;
    star.members.resize(2);
    ctx.kstars.push_back(star);

    MeanField mf(1, 2, 1, 1);
    const std::vector<Vector3_Order<double>> kfrac_list{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };

    assert(!can_restore_symmetry_kstar_meanfield(
        ctx, wfc_layouts, mf, kfrac_list, {{0, 1}}));
}

void test_symmetry_context_full_grid_kstar_route_matches_direct_full_k()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};

    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = 0;
    atom_rotation.atom_to = 0;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(atom_rotation);
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(atom_rotation);
    ctx.kstars.push_back(star);

    MeanField mf(1, 2, 2, 1);
    mf.get_efermi() = 0.0;
    for (int ik = 0; ik != 2; ++ik)
    {
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvals()[0](ik, 1) = 1.0;
        mf.get_weight()[0](ik, 0) = 1.0;
        mf.get_weight()[0](ik, 1) = 0.0;
        mf.get_eigenvectors()[0][0][ik].create(2, 1);
        mf.get_eigenvectors()[0][0][ik](0, 0) = {1.0, 0.0};
        mf.get_eigenvectors()[0][0][ik](1, 0) = {1.0, 0.0};
    }

    const std::vector<Vector3_Order<double>> kfrac_list{
        {0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0},
    };
    const std::vector<Vector3_Order<int>> Rs{
        {0, 0, 0},
        {1, 0, 0},
    };
    const std::map<atom_t, size_t> atom_nw{{0, 1}};
    const auto representative_indices =
        build_symmetry_full_grid_kstar_representative_indices(
            ctx, kfrac_list);
    if (representative_indices.size() != 1 || representative_indices[0] != 0)
        throw std::runtime_error("full-grid k-star representative lookup failed");
    const auto member_kfrac_targets =
        build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, kfrac_list);

    for (const auto &R : Rs)
    {
        const auto direct = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, R);
        const auto restored = get_symmetry_restored_dmat_cplx_R(
            ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, R, atom_nw,
            &member_kfrac_targets, &representative_indices);
        if (!fequal(direct(0, 0), restored(0, 0), {1e-12, 0.0}))
            throw std::runtime_error("full-grid k-star density matrix route differs from direct full-k");
    }

    const std::vector<double> taus{-1e-12, 1e-12};
    const auto direct_gf = mf.get_gf_cplx_imagtimes_Rs(0, 0, 0, kfrac_list, taus, Rs);
    const auto restored_gf = get_symmetry_restored_gf_cplx_imagtimes_Rs(
        ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, taus, Rs, atom_nw, std::vector<bool>{},
        &member_kfrac_targets, &representative_indices);
    for (const auto tau : taus)
    {
        for (const auto &R : Rs)
        {
            if (!fequal(direct_gf.at(tau).at(R)(0, 0),
                        restored_gf.at(tau).at(R)(0, 0), {1e-12, 0.0}))
            {
                throw std::runtime_error(
                    "full-grid k-star Green's-function route differs from direct full-k");
            }
        }
    }
}

void test_symmetry_context_kstar_restored_dmat_uses_target_kpoint_gauge()
{
    using namespace librpa_int;

    librpa_int::SymmetryContext ctx;
    ctx.set_available();
    ctx.basis_convention = {-1,
                            1,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.atom_to_type[1] = 0;
    ctx.input_coord_frac = {
        {0, {0.0, 0.0, 0.0}},
        {1, {0.25, 0.0, 0.0}},
    };

    librpa_int::SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};

    auto make_atom_rotation = [](const atom_t atom) {
        librpa_int::SymmetryKAtomRotation atom_rotation;
        atom_rotation.atom_from = static_cast<int>(atom);
        atom_rotation.atom_to = static_cast<int>(atom);
        atom_rotation.atom_type = 0;
        atom_rotation.lmax = 0;
        atom_rotation.bloch_rsh_rotations[0] = ComplexMatrix(1, 1);
        atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};
        return atom_rotation;
    };

    librpa_int::SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(make_atom_rotation(0));
    star.members[0].atom_rotations.push_back(make_atom_rotation(1));
    star.members[1].spatial_isym = 0;
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(make_atom_rotation(0));
    star.members[1].atom_rotations.push_back(make_atom_rotation(1));
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 2.0;
    mf.get_eigenvectors()[0][0][0].create(1, 2);
    mf.get_eigenvectors()[0][0][0](0, 0) = {std::sqrt(0.5), 0.0};
    mf.get_eigenvectors()[0][0][0](0, 1) = {std::sqrt(0.5), 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const Vector3_Order<int> R{0, 0, 0};
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const std::vector<std::vector<Vector3_Order<double>>> target_kfrac_list{
        {{0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}}};

    const auto restored = get_symmetry_restored_dmat_cplx_R(
        ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, R, atom_nw, &target_kfrac_list);

    const std::complex<double> expected_offdiag{0.25, -0.25};
    if (std::abs(restored(0, 1) - expected_offdiag) > 1e-12)
        throw std::runtime_error("ABACUS k-star restored density matrix ignored target k-point gauge");
}

//! Lock the spinor storage layout (Phase 0 convention C8):
//! AO matrices carry no interleaved spin dimension; the spinor degree of
//! freedom lives entirely in the outer (ispinor_bra, ispinor_ket) channels,
//! and each spin block is the outer product of the two channel wfc blocks,
//! D^{ab}(i,j) = sum_n occ_n * C_a(n,i) * conj(C_b(n,j)).
void test_spinor_channel_layout_uses_outer_blocks_no_interleave()
{
    using namespace librpa_int;

    const int n_spins = 1, nk = 1, nb = 4, nao = 3, n_spinor = 2;
    MeanField mf(n_spins, nk, nb, nao, n_spinor);

    for (int ib = 0; ib != nb; ++ib)
        mf.get_eigenvals()[0](0, ib) = -1.0 + 0.5 * ib;
    mf.get_weight()[0].zero_out();
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_weight()[0](0, 1) = 0.5;

    // deterministic non-symmetric wfc, distinct per spinor channel
    for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
    {
        mf.get_eigenvectors()[0][ispinor][0].create(nb, nao);
        for (int ib = 0; ib != nb; ++ib)
            for (int iw = 0; iw != nao; ++iw)
                mf.get_eigenvectors()[0][ispinor][0](ib, iw) =
                    std::complex<double>(0.11 * (ib + 1) + 0.07 * iw + 0.31 * ispinor,
                                         0.05 * ib - 0.13 * iw + 0.17 * ispinor);
    }

    const std::complex<double> thres = 1e-13;
    const double occ[2] = {1.0, 0.5};
    for (int bra = 0; bra != n_spinor; ++bra)
    {
        for (int ket = 0; ket != n_spinor; ++ket)
        {
            const auto block = mf.get_dmat_cplx(0, bra, ket, 0);
            assert(block.nr == nao && block.nc == nao);
            for (int i = 0; i != nao; ++i)
                for (int j = 0; j != nao; ++j)
                {
                    std::complex<double> expected = 0.0;
                    for (int n = 0; n != 2; ++n)
                        expected += occ[n]
                            * mf.get_eigenvectors()[0][bra][0](n, i)
                            * std::conj(mf.get_eigenvectors()[0][ket][0](n, j));
                    if (!fequal(block(i, j), expected, thres))
                        throw std::runtime_error(
                            "spinor density block deviates from channel outer product");
                }
        }
    }

    // off-diagonal blocks must be genuinely non-zero: channels are mixed
    // only through the (bra, ket) indices, never through an interleaved layout
    const auto d01 = mf.get_dmat_cplx(0, 0, 1, 0);
    double max_abs = 0.0;
    for (int i = 0; i != nao; ++i)
        for (int j = 0; j != nao; ++j)
            if (std::abs(d01(i, j)) > max_abs) max_abs = std::abs(d01(i, j));
    if (max_abs < 1e-3)
        throw std::runtime_error("spinor off-diagonal density block unexpectedly zero");

    // G(tau -> 0^-) approaches minus the density block, per channel
    const auto gf00 = mf.get_gf_cplx_imagtime(0, 0, 0, 0, -1e-12);
    const auto d00 = mf.get_dmat_cplx(0, 0, 0, 0);
    for (int i = 0; i != nao; ++i)
        for (int j = 0; j != nao; ++j)
            if (!fequal(gf00(i, j), -d00(i, j), std::complex<double>(1e-9, 0.0)))
                throw std::runtime_error("spinor GF tau->0- limit mismatch");
}

//! Build a one-operation identity spatial pool entry (s-orbital identity RSH).
static void add_identity_spatial_op(librpa_int::SymmetryContext &ctx)
{
    using namespace librpa_int;
    SymmetryOperation identity_operation;
    identity_operation.rotation.Identity();
    identity_operation.translation = {0.0, 0.0, 0.0};
    ctx.rspace_operations.push_back(identity_operation);
    ctx.rsh_rotations.emplace_back();
    ctx.rsh_rotations.back()[0] = ComplexMatrix(1, 1);
    ctx.rsh_rotations.back()[0](0, 0) = {1.0, 0.0};
}

static librpa_int::SymmetryKAtomRotation make_identity_atom_rotation(const int atom)
{
    librpa_int::SymmetryKAtomRotation atom_rotation;
    atom_rotation.atom_from = atom;
    atom_rotation.atom_to = atom;
    atom_rotation.atom_type = 0;
    atom_rotation.lmax = 0;
    atom_rotation.bloch_rsh_rotations[0] = librpa_int::ComplexMatrix(1, 1);
    atom_rotation.bloch_rsh_rotations[0](0, 0) = {1.0, 0.0};
    return atom_rotation;
}

static void assert_complex_matrix_near(const librpa_int::ComplexMatrix &got,
                                       const librpa_int::ComplexMatrix &expected,
                                       const double tol, const char *label)
{
    if (got.nr != expected.nr || got.nc != expected.nc)
        throw std::runtime_error(std::string(label) + ": matrix shape mismatch");
    for (int i = 0; i != got.nr; ++i)
        for (int j = 0; j != got.nc; ++j)
            if (std::abs(got(i, j) - expected(i, j)) > tol)
                throw std::runtime_error(std::string(label) + ": element mismatch");
}

//! Phase 4: spinor k-star GF restore on a full two-kpoint grid related by
//! time reversal must reproduce the direct full-grid GF in all four channels.
//! k1 = (1/4,0,0), k2 = (-1/4,0,0) = Theta k1, with Kramers-paired wfc
//! C(k2) = i sigma_y conj(C(k1)).
void test_spinor_kstar_gf_restore_full_grid_tr_round_trip()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    add_identity_spatial_op(ctx);

    // (E, I, unitary) and (E, I, antiunitary): grey-group Theta
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, false, SymmetrySpinActionSource::Identity});
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, true, SymmetrySpinActionSource::Identity});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{1, {1}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.25, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].time_reversal = false;
    star.members[0].action_id = 0;
    star.members[0].k_bz = {0.25, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(make_identity_atom_rotation(0));
    star.members[1].spatial_isym = 0;
    star.members[1].time_reversal = true;
    star.members[1].action_id = 1;
    star.members[1].k_bz = {-0.25, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(make_identity_atom_rotation(0));
    ctx.kstars.push_back(star);

    const std::complex<double> c0{0.6, 0.1}, c1{-0.2, 0.7};
    const std::complex<double> d0{0.5, -0.3}, d1{0.1, 0.4};
    MeanField mf(1, 2, 2, 1, 2);
    mf.get_efermi() = 0.0;
    for (int ik = 0; ik != 2; ++ik)
    {
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvals()[0](ik, 1) = 1.0;
        mf.get_weight()[0](ik, 0) = 0.5;
        mf.get_weight()[0](ik, 1) = 0.0;
    }
    mf.get_eigenvectors()[0][0][0].create(2, 1);
    mf.get_eigenvectors()[0][1][0].create(2, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = c0;
    mf.get_eigenvectors()[0][0][0](1, 0) = c1;
    mf.get_eigenvectors()[0][1][0](0, 0) = d0;
    mf.get_eigenvectors()[0][1][0](1, 0) = d1;
    // Kramers partner at k2: i sigma_y = [[0, 1], [-1, 0]]
    mf.get_eigenvectors()[0][0][1].create(2, 1);
    mf.get_eigenvectors()[0][1][1].create(2, 1);
    mf.get_eigenvectors()[0][0][1](0, 0) = std::conj(d0);
    mf.get_eigenvectors()[0][0][1](1, 0) = std::conj(d1);
    mf.get_eigenvectors()[0][1][1](0, 0) = -std::conj(c0);
    mf.get_eigenvectors()[0][1][1](1, 0) = -std::conj(c1);

    const std::vector<Vector3_Order<double>> kfrac_list{
        {0.25, 0.0, 0.0},
        {-0.25, 0.0, 0.0},
    };
    const std::vector<double> taus{0.3, -0.3};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}, {-2, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};

    const auto representative_indices =
        build_symmetry_full_grid_kstar_representative_indices(ctx, kfrac_list);
    if (representative_indices.size() != 1 || representative_indices[0] != 0)
        throw std::runtime_error("spinor full-grid representative lookup failed");
    const auto member_kfrac_targets =
        build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, kfrac_list);

    const auto restored = get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
        ctx, wfc_layouts, mf, 0, kfrac_list, taus, Rs, atom_nw, std::vector<bool>{},
        &member_kfrac_targets, &representative_indices);

    for (const auto tau : taus)
    {
        for (const auto &R : Rs)
        {
            const auto &blocks = restored.at(tau).at(R);
            const ComplexMatrix *channel_blocks[4] = {
                &blocks.b00, &blocks.b01, &blocks.b10, &blocks.b11};
            for (int s = 0; s != 4; ++s)
            {
                const auto direct = mf.get_gf_cplx_imagtimes_Rs(
                    0, s / 2, s % 2, kfrac_list, {tau}, {R});
                assert_complex_matrix_near(*channel_blocks[s], direct.at(tau).at(R), 1e-12,
                                           "spinor TR round trip vs direct full grid");
            }
        }
    }
}

//! Phase 4: for unitary U_s = I members the four-channel spinor restore must
//! agree channel-by-channel with the validated scalar restore, including the
//! target-kpoint gauge (member k_bz = (1/2,0,0) re-gauged to (3/2,0,0), one
//! atom at a non-special position).
void test_spinor_kstar_gf_restore_unitary_matches_scalar_with_gauge()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    ctx.basis_convention = {-1,
                            1,
                            LIBRPA_ANGULAR_ORDER_NATURAL,
                            LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.atom_to_type[1] = 0;
    ctx.input_coord_frac = {
        {0, {0.0, 0.0, 0.0}},
        {1, {0.25, 0.0, 0.0}},
    };
    add_identity_spatial_op(ctx);
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, false, SymmetrySpinActionSource::Identity});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    for (int im = 0; im != 2; ++im)
    {
        star.members[im].spatial_isym = 0;
        star.members[im].action_id = 0;
        star.members[im].atom_rotations.push_back(make_identity_atom_rotation(0));
        star.members[im].atom_rotations.push_back(make_identity_atom_rotation(1));
    }
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 2, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_eigenvectors()[0][0][0].create(1, 2);
    mf.get_eigenvectors()[0][1][0].create(1, 2);
    mf.get_eigenvectors()[0][0][0](0, 0) = {std::sqrt(0.5), 0.0};
    mf.get_eigenvectors()[0][0][0](0, 1) = {std::sqrt(0.5), 0.0};
    mf.get_eigenvectors()[0][1][0](0, 0) = {0.0, 0.6};
    mf.get_eigenvectors()[0][1][0](0, 1) = {0.8, 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<double> taus{0.25, -0.25};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const symmetry_kstar_member_kfrac_targets_t member_kfrac_targets{
        {{0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}}};

    const auto restored_spinor = get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
        ctx, wfc_layouts, mf, 0, kfrac_list, taus, Rs, atom_nw, std::vector<bool>{},
        &member_kfrac_targets);

    for (const auto tau : taus)
    {
        for (const auto &R : Rs)
        {
            const auto &blocks = restored_spinor.at(tau).at(R);
            const ComplexMatrix *channel_blocks[4] = {
                &blocks.b00, &blocks.b01, &blocks.b10, &blocks.b11};
            for (int s = 0; s != 4; ++s)
            {
                const auto scalar = get_symmetry_restored_gf_cplx_imagtimes_Rs(
                    ctx, wfc_layouts, mf, 0, s / 2, s % 2, kfrac_list, {tau}, {R},
                    atom_nw, std::vector<bool>{}, &member_kfrac_targets);
                assert_complex_matrix_near(*channel_blocks[s], scalar.at(tau).at(R), 1e-12,
                                           "spinor unitary restore vs scalar restore with gauge");
            }
        }
    }
    // off-diagonal channels must be genuinely non-zero, otherwise the
    // comparison above is vacuous (tau > 0 is identically zero here because
    // the single band is fully occupied; check the tau < 0 leg)
    if (restored_spinor.at(taus[1]).at(Rs[0]).b01.get_max_abs() < 1e-3)
        throw std::runtime_error("spinor unitary restore: off-diagonal block unexpectedly zero");
}

//! Phase 4: a pure-spin operation (U_s = sigma_x, identity spatial part) must
//! act through the same k-star restore: two members mapping Gamma to itself
//! give the spin-averaged GF (G + sigma_x G sigma_x) / 2.
void test_spinor_kstar_gf_restore_pure_spin_average()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    add_identity_spatial_op(ctx);
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, false, SymmetrySpinActionSource::Identity});
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {0.0, 1.0, 1.0, 0.0}, false, SymmetrySpinActionSource::ExplicitSpinSpace});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{1, {1}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    for (int im = 0; im != 2; ++im)
    {
        star.members[im].spatial_isym = 0;
        star.members[im].action_id = static_cast<std::size_t>(im);
        star.members[im].k_bz = {0.0, 0.0, 0.0};
        star.members[im].atom_rotations.push_back(make_identity_atom_rotation(0));
    }
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 2, 1, 2);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](0, 1) = 1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_weight()[0](0, 1) = 0.0;
    mf.get_eigenvectors()[0][0][0].create(2, 1);
    mf.get_eigenvectors()[0][1][0].create(2, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {0.6, 0.1};
    mf.get_eigenvectors()[0][0][0](1, 0) = {-0.2, 0.7};
    mf.get_eigenvectors()[0][1][0](0, 0) = {0.5, -0.3};
    mf.get_eigenvectors()[0][1][0](1, 0) = {0.1, 0.4};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<double> taus{0.4, -0.4};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};

    const auto restored = get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
        ctx, wfc_layouts, mf, 0, kfrac_list, taus, Rs, atom_nw, std::vector<bool>{});

    for (const auto tau : taus)
    {
        const auto g00 = mf.get_gf_cplx_imagtime(0, 0, 0, 0, tau);
        const auto g01 = mf.get_gf_cplx_imagtime(0, 0, 1, 0, tau);
        const auto g10 = mf.get_gf_cplx_imagtime(0, 1, 0, 0, tau);
        const auto g11 = mf.get_gf_cplx_imagtime(0, 1, 1, 0, tau);
        const auto &blocks = restored.at(tau).at(Rs[0]);
        assert_complex_matrix_near(blocks.b00, 0.5 * (g00 + g11), 1e-12,
                                   "pure-spin average b00");
        assert_complex_matrix_near(blocks.b01, 0.5 * (g01 + g10), 1e-12,
                                   "pure-spin average b01");
        assert_complex_matrix_near(blocks.b10, 0.5 * (g10 + g01), 1e-12,
                                   "pure-spin average b10");
        assert_complex_matrix_near(blocks.b11, 0.5 * (g11 + g00), 1e-12,
                                   "pure-spin average b11");
    }
}

//! Phase 4: a single antiunitary member mapping Gamma to itself applies the
//! Theta remap at the restore level: G'00 = conj(G11), G'01 = -conj(G10),
//! G'10 = -conj(G01), G'11 = conj(G00), for a non-Kramers-symmetric wfc.
void test_spinor_kstar_gf_restore_tr_remap_at_gamma()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    add_identity_spatial_op(ctx);
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, true, SymmetrySpinActionSource::Identity});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(1);
    star.members[0].spatial_isym = 0;
    star.members[0].time_reversal = true;
    star.members[0].action_id = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(make_identity_atom_rotation(0));
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 2, 1, 2);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](0, 1) = 1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_weight()[0](0, 1) = 0.0;
    mf.get_eigenvectors()[0][0][0].create(2, 1);
    mf.get_eigenvectors()[0][1][0].create(2, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {0.6, 0.1};
    mf.get_eigenvectors()[0][0][0](1, 0) = {-0.2, 0.7};
    mf.get_eigenvectors()[0][1][0](0, 0) = {0.5, -0.3};
    mf.get_eigenvectors()[0][1][0](1, 0) = {0.1, 0.4};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<double> taus{0.4, -0.4};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};

    const auto restored = get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
        ctx, wfc_layouts, mf, 0, kfrac_list, taus, Rs, atom_nw, std::vector<bool>{});

    for (const auto tau : taus)
    {
        const auto g00 = mf.get_gf_cplx_imagtime(0, 0, 0, 0, tau);
        const auto g01 = mf.get_gf_cplx_imagtime(0, 0, 1, 0, tau);
        const auto g10 = mf.get_gf_cplx_imagtime(0, 1, 0, 0, tau);
        const auto g11 = mf.get_gf_cplx_imagtime(0, 1, 1, 0, tau);
        const auto &blocks = restored.at(tau).at(Rs[0]);
        assert_complex_matrix_near(blocks.b00, conj(g11), 1e-12, "TR remap b00");
        assert_complex_matrix_near(blocks.b01, (-1.0) * conj(g10), 1e-12, "TR remap b01");
        assert_complex_matrix_near(blocks.b10, (-1.0) * conj(g01), 1e-12, "TR remap b10");
        assert_complex_matrix_near(blocks.b11, conj(g00), 1e-12, "TR remap b11");
    }
}

//! Phase 4 rule A: a band-mask boundary slicing through a degenerate multiplet is
//! rejected, both directly and through the spinor restore entry point.
void test_validate_kstar_band_cutoff_closure_rejects_degenerate_cut()
{
    using namespace librpa_int;

    MeanField mf(1, 1, 3, 1, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](0, 1) = -1.0 + 5e-9;
    mf.get_eigenvals()[0](0, 2) = 1.0;

    SymmetryContext ctx;

    const auto expect_throw = [&ctx, &mf](const std::vector<bool> &band_mask) {
        try
        {
            validate_kstar_band_cutoff_closure(ctx, mf, band_mask);
        }
        catch (const std::runtime_error &)
        {
            return;
        }
        throw std::runtime_error("degenerate band-mask boundary was not rejected");
    };
    expect_throw(std::vector<bool>{true, false, false});
    expect_throw(std::vector<bool>{false, true, true});

    // no throw: gap above tolerance, no truncation, and the full band window
    validate_kstar_band_cutoff_closure(ctx, mf, std::vector<bool>{true, true, false});
    validate_kstar_band_cutoff_closure(ctx, mf, std::vector<bool>{});
    validate_kstar_band_cutoff_closure(ctx, mf, std::vector<bool>{true, true, true});
}

//! Phase 4: missing (bra, ket) wfc channels are zero-filled; with only the
//! channel-0 wfc present, the restored off-diagonal and (1,1) blocks vanish
//! while (0,0) matches the scalar restore.
void test_spinor_kstar_gf_restore_zero_fills_missing_channels()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    add_identity_spatial_op(ctx);
    ctx.spin_operations.push_back(SymmetrySpinOperation{
        0, {1.0, 0.0, 0.0, 1.0}, false, SymmetrySpinActionSource::Identity});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    for (int im = 0; im != 2; ++im)
    {
        star.members[im].spatial_isym = 0;
        star.members[im].action_id = 0;
        star.members[im].atom_rotations.push_back(make_identity_atom_rotation(0));
    }
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 1, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    // only the channel-0 wfc exists; channel 1 is left absent
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {1.0, 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<double> taus{0.2, -0.2};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};

    const auto restored = get_symmetry_restored_gf_cplx_imagtimes_Rs_spinor(
        ctx, wfc_layouts, mf, 0, kfrac_list, taus, Rs, atom_nw, std::vector<bool>{});

    for (const auto tau : taus)
    {
        const auto scalar = get_symmetry_restored_gf_cplx_imagtimes_Rs(
            ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, {tau}, Rs, atom_nw,
            std::vector<bool>{});
        for (const auto &R : Rs)
        {
            const auto &blocks = restored.at(tau).at(R);
            assert_complex_matrix_near(blocks.b00, scalar.at(tau).at(R), 1e-12,
                                       "zero-fill restore b00 vs scalar");
            // tau > 0 is identically zero here (single fully-occupied band),
            // and the two-member star cancels at odd R through the full-star
            // phase; the nonzero sanity check applies to tau < 0, R = 0 only
            if (tau < 0.0 && R == Vector3_Order<int>{0, 0, 0}
                && blocks.b00.get_max_abs() < 1e-3)
                throw std::runtime_error("zero-fill restore b00 unexpectedly zero");
            if (blocks.b01.get_max_abs() > 1e-14 || blocks.b10.get_max_abs() > 1e-14
                || blocks.b11.get_max_abs() > 1e-14)
                throw std::runtime_error("zero-fill restore did not zero missing channels");
        }
    }
}

void test_gf_band_mask()
{
    using namespace librpa_int;
    MeanField mf(1, 1, 3, 3);
    mf.get_efermi() = 0.0;
    auto &wfc = mf.get_eigenvectors()[0][0][0];
    wfc.create(3, 3);
    for (int ib = 0; ib < 3; ++ib)
    {
        wfc(ib, ib) = 1.0;
        mf.get_eigenvals()[0](0, ib) = ib - 1.0;
        mf.get_weight()[0](0, ib) = 0.5;
    }
    const std::vector<Vector3_Order<double>> ks{{0, 0, 0}};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}};
    for (const std::vector<bool> &mask :
         std::initializer_list<std::vector<bool>>{{},
                                                  {true, false, false},
                                                  {false, true, true},
                                                  {false, true, false},
                                                  {false, false, false}})
        for (double tau : {-0.5, 0.5})
        {
            const auto single = mf.get_gf_cplx_imagtime(0, 0, 0, 0, tau, mask);
            const auto summed = mf.get_gf_cplx_imagtimes_Rs(0, 0, 0, ks, {tau}, Rs, mask);
            for (int ib = 0; ib < 3; ++ib)
            {
                const double expected = !mask.empty() && !mask[ib] ? 0.0
                    : (tau > 0 ? 0.75 : -0.25) * std::exp(std::min(0.0, -tau * (ib - 1.0)));
                assert(std::abs(single(ib, ib) - expected) < 1e-12);
                assert(std::abs(summed.at(tau).at(Rs.front())(ib, ib) - expected) < 1e-12);
            }
        }
    bool rejected = false;
    try { mf.get_gf_cplx_imagtime(0, 0, 0, 0, -0.5, {true}); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
    assert(std::abs(mf.get_dmat_cplx(0, 0, 0, 0)(1, 1) - 0.25) < 1e-12);
}

void test_spinor_kstar_dmat_restore_complex_us_antiunitary_full_grid_round_trip()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    ctx.basis_convention = {-1, 1, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.atom_to_type[1] = 0;
    ctx.input_coord_frac = {
        {0, {0.0, 0.0, 0.0}},
        {1, {0.2, 0.0, 0.0}},
    };
    add_identity_spatial_op(ctx);

    // U_s = (1/sqrt(2)) [[1, i], [i, 1]]: complex, off-diagonal, unitary.
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const std::array<std::complex<double>, 4> U_s{inv_sqrt2, std::complex<double>(0.0, inv_sqrt2),
                                                  std::complex<double>(0.0, inv_sqrt2), inv_sqrt2};
    // U_s above is the physical representative-to-member action used by the
    // independently generated full-grid wavefunctions below. Generated k-star
    // metadata records the member-to-representative operation; the production
    // resolve_symmetry_kstar_member_spin_operation reverses it for restoration.
    const std::array<std::complex<double>, 4> U_s_inverse{std::conj(U_s[0]), std::conj(U_s[2]),
                                                          std::conj(U_s[1]), std::conj(U_s[3])};
    ctx.spin_operations.push_back(
        SymmetrySpinOperation{0, {1.0, 0.0, 0.0, 1.0}, false, SymmetrySpinActionSource::Identity});
    ctx.spin_operations.push_back(
        SymmetrySpinOperation{0, U_s_inverse, true, SymmetrySpinActionSource::ExplicitSpinSpace});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{1, {1}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.25, 0.0, 0.0};
    star.members.resize(2);
    star.members[0].spatial_isym = 0;
    star.members[0].time_reversal = false;
    star.members[0].action_id = 0;
    star.members[0].k_bz = {0.25, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(make_identity_atom_rotation(0));
    star.members[0].atom_rotations.push_back(make_identity_atom_rotation(1));
    star.members[1].spatial_isym = 0;
    star.members[1].time_reversal = true;
    star.members[1].action_id = 1;
    star.members[1].k_bz = {-0.25, 0.0, 0.0};
    star.members[1].atom_rotations.push_back(make_identity_atom_rotation(0));
    star.members[1].atom_rotations.push_back(make_identity_atom_rotation(1));
    ctx.kstars.push_back(star);

    // per-band, per-AO channel values: c = channel 0, d = channel 1
    const std::complex<double> c[2][2] = {{{0.6, 0.1}, {-0.3, 0.2}}, {{-0.2, 0.7}, {0.4, 0.5}}};
    const std::complex<double> d[2][2] = {{{0.5, -0.3}, {0.1, 0.4}}, {{0.1, 0.4}, {-0.6, 0.2}}};
    MeanField mf(1, 2, 2, 2, 2);
    mf.get_efermi() = 0.0;
    for (int ik = 0; ik != 2; ++ik)
    {
        mf.get_eigenvals()[0](ik, 0) = -1.0;
        mf.get_eigenvals()[0](ik, 1) = 1.0;
        mf.get_weight()[0](ik, 0) = 0.5;
        mf.get_weight()[0](ik, 1) = 0.0;
    }
    mf.get_eigenvectors()[0][0][0].create(2, 2);
    mf.get_eigenvectors()[0][1][0].create(2, 2);
    for (int ib = 0; ib != 2; ++ib)
        for (int iao = 0; iao != 2; ++iao)
        {
            mf.get_eigenvectors()[0][0][0](ib, iao) = c[ib][iao];
            mf.get_eigenvectors()[0][1][0](ib, iao) = d[ib][iao];
        }
    // Theta image at k2, applied per band and per AO to the spinor (channel)
    // index: c'(k2) = i sigma_y conj(U_s) conj(c(k1)) with
    // i sigma_y = [[0, 1], [-1, 0]]. The wfc first index is the band index
    // and the second the AO index, so each (band, AO) spinor pair is imaged
    // separately.
    const auto image = [&](const std::complex<double> &s0, const std::complex<double> &s1)
    {
        const std::complex<double> t0 =
            std::conj(U_s[0]) * std::conj(s0) + std::conj(U_s[1]) * std::conj(s1);
        const std::complex<double> t1 =
            std::conj(U_s[2]) * std::conj(s0) + std::conj(U_s[3]) * std::conj(s1);
        return std::make_pair(t1, -t0);
    };
    // Reciprocal-gauge phase of the stored k2' = (3/4,0,0) wfc relative to
    // the member k_bz = (-1/4,0,0): k_shift = (1,0,0), phase[atom] =
    // exp(-i 2 pi bloch_phase bloch_ratom (k_shift . tau)) with
    // (bloch_phase, bloch_ratom) = (-1, 1). Atom 0 at the origin keeps
    // phase 1; atom 1 at tau = (0.2,0,0) gets exp(+i 2 pi 0.2).
    const double gauge_angle = 2.0 * std::acos(-1.0) * 0.2;
    const std::complex<double> gauge_phase_atom1(std::cos(gauge_angle), std::sin(gauge_angle));
    mf.get_eigenvectors()[0][0][1].create(2, 2);
    mf.get_eigenvectors()[0][1][1].create(2, 2);
    for (int ib = 0; ib != 2; ++ib)
    {
        const auto img0 = image(c[ib][0], d[ib][0]);  // AO 0 = atom 0, phase 1
        const auto img1 = image(c[ib][1], d[ib][1]);  // AO 1 = atom 1
        mf.get_eigenvectors()[0][0][1](ib, 0) = img0.first;
        mf.get_eigenvectors()[0][1][1](ib, 0) = img0.second;
        mf.get_eigenvectors()[0][0][1](ib, 1) = gauge_phase_atom1 * img1.first;
        mf.get_eigenvectors()[0][1][1](ib, 1) = gauge_phase_atom1 * img1.second;
    }

    // k2' = -1/4 + (1,0,0) = 3/4: the full grid stores the Theta image in the
    // re-gauged basis, so the restore must fold the member k_bz = -1/4 onto
    // the target 3/4 and apply the non-trivial reciprocal-gauge phases.
    const std::vector<Vector3_Order<double>> kfrac_list{
        {0.25, 0.0, 0.0},
        {0.75, 0.0, 0.0},
    };
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};

    const auto representative_indices =
        build_symmetry_full_grid_kstar_representative_indices(ctx, kfrac_list);
    if (representative_indices.size() != 1 || representative_indices[0] != 0)
        throw std::runtime_error("spinor dmat full-grid representative lookup failed");
    const auto member_kfrac_targets =
        build_symmetry_full_grid_kstar_member_kfrac_targets(ctx, kfrac_list);

    for (const auto &R : Rs)
    {
        const auto restored = get_symmetry_restored_dmat_cplx_R_spinor(
            ctx, wfc_layouts, mf, 0, kfrac_list, R, atom_nw, &member_kfrac_targets,
            &representative_indices);
        const ComplexMatrix *channel_blocks[4] = {&restored.b00, &restored.b01, &restored.b10,
                                                  &restored.b11};
        for (int s = 0; s != 4; ++s)
        {
            const auto direct = mf.get_dmat_cplx_R(0, s / 2, s % 2, kfrac_list, R);
            assert_complex_matrix_near(*channel_blocks[s], direct, 1e-12,
                                       "spinor complex-U_s TR round trip vs direct full grid");
        }
    }

    // off-diagonal channels must be genuinely non-zero, otherwise the
    // four-channel comparison above is vacuous
    const auto direct01 = mf.get_dmat_cplx_R(0, 0, 1, kfrac_list, Rs.front());
    if (direct01.get_max_abs() < 1e-3)
        throw std::runtime_error("spinor dmat round trip: off-diagonal channel unexpectedly zero");

    // the old per-channel scalar restore must differ from the reference:
    // it conjugates each channel elementwise without the SU(2) mixing and
    // the Theta channel exchange
    const auto scalar00 =
        get_symmetry_restored_dmat_cplx_R(ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, Rs.front(),
                                          atom_nw, &member_kfrac_targets, &representative_indices);
    const auto direct00 = mf.get_dmat_cplx_R(0, 0, 0, kfrac_list, Rs.front());
    const auto scalar_diff = scalar00 - direct00;
    if (scalar_diff.get_max_abs() < 1e-8)
        throw std::runtime_error(
            "scalar per-channel restore accidentally matches the spinor reference");
}

void test_spinor_kstar_dmat_restore_antiunitary_matches_kronecker_reference()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    add_identity_spatial_op(ctx);

    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    const std::array<std::complex<double>, 4> U_s{inv_sqrt2, std::complex<double>(0.0, inv_sqrt2),
                                                  std::complex<double>(0.0, inv_sqrt2), inv_sqrt2};
    // Keep the physical forward action for both independent references. Only
    // the stored k-star action runs from member to representative, as in
    // generated metadata and resolve_symmetry_kstar_member_spin_operation.
    const SymmetrySpinOperation forward_op{0, U_s, true,
                                           SymmetrySpinActionSource::ExplicitSpinSpace};
    const std::array<std::complex<double>, 4> U_s_inverse{std::conj(U_s[0]), std::conj(U_s[2]),
                                                          std::conj(U_s[1]), std::conj(U_s[3])};
    ctx.spin_operations.push_back(
        SymmetrySpinOperation{0, U_s_inverse, true, SymmetrySpinActionSource::ExplicitSpinSpace});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(1);
    star.members[0].spatial_isym = 0;
    star.members[0].time_reversal = true;
    star.members[0].action_id = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(make_identity_atom_rotation(0));
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 2, 1, 2);
    mf.get_efermi() = 0.0;
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_eigenvals()[0](0, 1) = 1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_weight()[0](0, 1) = 0.0;
    mf.get_eigenvectors()[0][0][0].create(2, 1);
    mf.get_eigenvectors()[0][1][0].create(2, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {0.6, 0.1};
    mf.get_eigenvectors()[0][0][0](1, 0) = {-0.2, 0.7};
    mf.get_eigenvectors()[0][1][0](0, 0) = {0.5, -0.3};
    mf.get_eigenvectors()[0][1][0](1, 0) = {0.1, 0.4};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const Vector3_Order<int> R{0, 0, 0};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};

    const auto restored =
        get_symmetry_restored_dmat_cplx_R_spinor(ctx, wfc_layouts, mf, 0, kfrac_list, R, atom_nw);

    // Reference 1: the kernel applied directly to the four source blocks
    // (the restore adds only the star factor and the k phase, both unity here).
    const SpinorBlocks4<ComplexMatrix> source{
        mf.get_dmat_cplx(0, 0, 0, 0), mf.get_dmat_cplx(0, 0, 1, 0), mf.get_dmat_cplx(0, 1, 0, 0),
        mf.get_dmat_cplx(0, 1, 1, 0)};
    const auto identity_orbit = [](std::size_t spatial_id, const ComplexMatrix &block)
    {
        (void)spatial_id;
        return block;
    };
    const auto kernel_ref = transform_spinor_bilinear(forward_op, source, identity_orbit,
                                                      BilinearConvention::SourceToTarget_DXDdag);
    assert_complex_matrix_near(restored.b00, kernel_ref.b00, 1e-12,
                               "antiunitary dmat restore vs kernel b00");
    assert_complex_matrix_near(restored.b01, kernel_ref.b01, 1e-12,
                               "antiunitary dmat restore vs kernel b01");
    assert_complex_matrix_near(restored.b10, kernel_ref.b10, 1e-12,
                               "antiunitary dmat restore vs kernel b10");
    assert_complex_matrix_near(restored.b11, kernel_ref.b11, 1e-12,
                               "antiunitary dmat restore vs kernel b11");

    // Reference 2: explicit Kronecker formula on the merged channel-outermost
    // matrix M = [[D00, D01], [D10, D11]] (N = 1 AO here):
    // Y = (U_s kron I) M (U_s kron I)^dagger,
    // X' = (J_s kron I) conj(Y) (J_s kron I)^dagger, J_s = i sigma_y =
    // [[0, 1], [-1, 0]].
    ComplexMatrix M(2, 2);
    M(0, 0) = source.b00(0, 0);
    M(0, 1) = source.b01(0, 0);
    M(1, 0) = source.b10(0, 0);
    M(1, 1) = source.b11(0, 0);
    ComplexMatrix Us(2, 2);
    Us(0, 0) = U_s[0];
    Us(0, 1) = U_s[1];
    Us(1, 0) = U_s[2];
    Us(1, 1) = U_s[3];
    const auto Y = Us * M * transpose(Us, true);
    ComplexMatrix Js(2, 2);
    Js(0, 1) = {1.0, 0.0};
    Js(1, 0) = {-1.0, 0.0};
    const auto Xp = Js * conj(Y) * transpose(Js, true);
    const std::complex<double> ref_blocks[4] = {Xp(0, 0), Xp(0, 1), Xp(1, 0), Xp(1, 1)};
    const ComplexMatrix *out_blocks[4] = {&restored.b00, &restored.b01, &restored.b10,
                                          &restored.b11};
    for (int s = 0; s != 4; ++s)
    {
        if (std::abs((*out_blocks[s])(0, 0) - ref_blocks[s]) > 1e-12)
            throw std::runtime_error(
                "spinor dmat antiunitary restore deviates from Kronecker reference");
    }

    // the old per-channel scalar restore must differ from the Kronecker
    // reference (it would return conj(D00) for the (0,0) channel)
    const auto scalar00 =
        get_symmetry_restored_dmat_cplx_R(ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, R, atom_nw);
    if (std::abs(scalar00(0, 0) - ref_blocks[0]) < 1e-8)
        throw std::runtime_error(
            "scalar per-channel restore accidentally matches the Kronecker reference");
}

void test_spinor_kstar_dmat_restore_zero_fills_missing_channels()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.input_coord_frac[0] = {0.0, 0.0, 0.0};
    add_identity_spatial_op(ctx);
    ctx.spin_operations.push_back(
        SymmetrySpinOperation{0, {1.0, 0.0, 0.0, 1.0}, true, SymmetrySpinActionSource::Identity});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(1);
    star.members[0].spatial_isym = 0;
    star.members[0].time_reversal = true;
    star.members[0].action_id = 0;
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[0].atom_rotations.push_back(make_identity_atom_rotation(0));
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 1, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    // only the channel-0 wfc exists; channel 1 is left absent
    mf.get_eigenvectors()[0][0][0].create(1, 1);
    mf.get_eigenvectors()[0][0][0](0, 0) = {0.7, -0.2};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const Vector3_Order<int> R{0, 0, 0};
    const std::map<atom_t, size_t> atom_nw{{0, 1}};

    const auto restored =
        get_symmetry_restored_dmat_cplx_R_spinor(ctx, wfc_layouts, mf, 0, kfrac_list, R, atom_nw);

    if (restored.b00.get_max_abs() > 1e-14 || restored.b01.get_max_abs() > 1e-14 ||
        restored.b10.get_max_abs() > 1e-14)
        throw std::runtime_error("zero-fill restore did not zero the missing channels");
    // the occupied channel lands in b11 through the Theta remap: conj(D00)
    const auto d00 = mf.get_dmat_cplx(0, 0, 0, 0);
    if (std::abs(restored.b11(0, 0) - std::conj(d00(0, 0))) > 1e-12)
        throw std::runtime_error("zero-fill restore: Theta remap did not map channel 0 into b11");
    // the scalar per-channel restore would report a spurious non-zero b00
    const auto scalar00 =
        get_symmetry_restored_dmat_cplx_R(ctx, wfc_layouts, mf, 0, 0, 0, kfrac_list, R, atom_nw);
    if (std::abs(scalar00(0, 0)) < 1e-12)
        throw std::runtime_error("scalar restore unexpectedly vanishes: test is vacuous");
}

void test_spinor_kstar_dmat_restore_unitary_matches_scalar_with_gauge()
{
    using namespace librpa_int;

    SymmetryContext ctx;
    ctx.set_available();
    ctx.basis_convention = {-1, 1, LIBRPA_ANGULAR_ORDER_NATURAL, LIBRPA_RSH_COEFF_1_M,
                            LIBRPA_RSH_COEFF_1_M};
    const std::vector<SpeciesBasisLayout> wfc_layouts{{"X", {0}}};
    ctx.atom_to_type[0] = 0;
    ctx.atom_to_type[1] = 0;
    ctx.input_coord_frac = {
        {0, {0.0, 0.0, 0.0}},
        {1, {0.25, 0.0, 0.0}},
    };
    add_identity_spatial_op(ctx);
    ctx.spin_operations.push_back(
        SymmetrySpinOperation{0, {1.0, 0.0, 0.0, 1.0}, false, SymmetrySpinActionSource::Identity});
    ctx.kspace_actions.push_back(SymmetryGeometricAction{0, {0}});

    SymmetryKStar star;
    star.star_index = 0;
    star.k_ibz = {0.0, 0.0, 0.0};
    star.members.resize(2);
    for (int im = 0; im != 2; ++im)
    {
        star.members[im].spatial_isym = 0;
        star.members[im].action_id = 0;
        star.members[im].atom_rotations.push_back(make_identity_atom_rotation(0));
        star.members[im].atom_rotations.push_back(make_identity_atom_rotation(1));
    }
    star.members[0].k_bz = {0.0, 0.0, 0.0};
    star.members[1].k_bz = {0.5, 0.0, 0.0};
    ctx.kstars.push_back(star);

    MeanField mf(1, 1, 1, 2, 2);
    mf.get_eigenvals()[0](0, 0) = -1.0;
    mf.get_weight()[0](0, 0) = 1.0;
    mf.get_eigenvectors()[0][0][0].create(1, 2);
    mf.get_eigenvectors()[0][1][0].create(1, 2);
    mf.get_eigenvectors()[0][0][0](0, 0) = {std::sqrt(0.5), 0.0};
    mf.get_eigenvectors()[0][0][0](0, 1) = {std::sqrt(0.5), 0.0};
    mf.get_eigenvectors()[0][1][0](0, 0) = {0.0, 0.6};
    mf.get_eigenvectors()[0][1][0](0, 1) = {0.8, 0.0};

    const std::vector<Vector3_Order<double>> kfrac_list{{0.0, 0.0, 0.0}};
    const std::vector<Vector3_Order<int>> Rs{{0, 0, 0}, {1, 0, 0}};
    const std::map<atom_t, size_t> atom_nw{{0, 1}, {1, 1}};
    const symmetry_kstar_member_kfrac_targets_t member_kfrac_targets{
        {{0.0, 0.0, 0.0}, {1.5, 0.0, 0.0}}};

    for (const auto &R : Rs)
    {
        const auto restored_spinor = get_symmetry_restored_dmat_cplx_R_spinor(
            ctx, wfc_layouts, mf, 0, kfrac_list, R, atom_nw, &member_kfrac_targets);
        const ComplexMatrix *channel_blocks[4] = {&restored_spinor.b00, &restored_spinor.b01,
                                                  &restored_spinor.b10, &restored_spinor.b11};
        for (int s = 0; s != 4; ++s)
        {
            const auto scalar =
                get_symmetry_restored_dmat_cplx_R(ctx, wfc_layouts, mf, 0, s / 2, s % 2, kfrac_list,
                                                  R, atom_nw, &member_kfrac_targets);
            assert_complex_matrix_near(*channel_blocks[s], scalar, 1e-12,
                                       "spinor unitary dmat restore vs scalar restore with gauge");
        }
    }
    // off-diagonal channels must be genuinely non-zero, otherwise the
    // comparison above is vacuous
    if (get_symmetry_restored_dmat_cplx_R_spinor(ctx, wfc_layouts, mf, 0, kfrac_list, Rs.front(),
                                                 atom_nw, &member_kfrac_targets)
            .b01.get_max_abs() < 1e-3)
        throw std::runtime_error(
            "spinor unitary dmat restore: off-diagonal block unexpectedly zero");
}

int main(int argc, char *argv[])
{
    test_BCC_He_gamma_minimal_basis_aims();
    test_gf_band_mask();
    test_state_index_energy_bounds();
    test_find_highest_occupied_state();
    test_dmat_cplx_Rs_matches_single_R_accumulation();
    test_symmetry_context_kstar_restored_dmat_uses_full_star_phases();
    test_symmetry_context_kstar_restore_skips_full_grid();
    test_symmetry_context_full_grid_kstar_route_matches_direct_full_k();
    test_symmetry_context_kstar_restored_dmat_uses_target_kpoint_gauge();
    test_spinor_channel_layout_uses_outer_blocks_no_interleave();
    test_spinor_kstar_gf_restore_full_grid_tr_round_trip();
    test_spinor_kstar_gf_restore_unitary_matches_scalar_with_gauge();
    test_spinor_kstar_gf_restore_pure_spin_average();
    test_spinor_kstar_gf_restore_tr_remap_at_gamma();
    test_validate_kstar_band_cutoff_closure_rejects_degenerate_cut();
    test_spinor_kstar_gf_restore_zero_fills_missing_channels();
    test_spinor_kstar_dmat_restore_complex_us_antiunitary_full_grid_round_trip();
    test_spinor_kstar_dmat_restore_antiunitary_matches_kronecker_reference();
    test_spinor_kstar_dmat_restore_zero_fills_missing_channels();
    test_spinor_kstar_dmat_restore_unitary_matches_scalar_with_gauge();
    return 0;
}
