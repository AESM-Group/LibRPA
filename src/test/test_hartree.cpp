#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../api/hartree_input.h"
#include "../core/hartree.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"

using namespace librpa_int;

namespace
{
constexpr double two_pi = 6.283185307179586476925286766559;
constexpr int nk = 3, nao = 3, naux = 3;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

double matrix_error(const Matz& a, const Matz& b)
{
    double error = 0.0;
    for (int i = 0; i != a.nr(); ++i)
        for (int j = 0; j != a.nc(); ++j) error = std::max(error, std::abs(a(i, j) - b(i, j)));
    return error;
}

struct Fixture
{
    AtomicBasis ao{std::vector<std::size_t>{2, 1}};
    AtomicBasis aux{std::vector<std::size_t>{1, 2}};
    PeriodicBoundaryData pbc;
    Cs_LRI all_cs;
    Matz metric{naux, naux, MAJOR::COL};
    std::vector<Matz> density;

    explicit Fixture(double shift)
    {
        pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        std::vector<double> kvecs;
        for (int k = 0; k != nk; ++k)
        {
            kvecs.push_back(two_pi * static_cast<double>(k) / nk);
            kvecs.push_back(0.0);
            kvecs.push_back(0.0);
        }
        pbc.set_kgrids_kvec(nk, 1, 1, kvecs);
        // The general PBC setter currently accepts only an unshifted mesh.
        // Supply the shifted fractional coordinates directly to exercise the
        // core adapter's twist handling without changing that public setter.
        for (auto& k : pbc.kfrac_list) k.x += shift;
        // Nonseparable, unequal atom dimensions and raw R outside the BvK box
        // distinguish endpoint transposes, phase signs and periodic image sums.
        for (int i = 0; i != 2; ++i)
            for (int j = 0; j != 2; ++j)
                for (int r = -4; r <= 4; ++r)
                {
                    RI::Tensor<double> block(
                        {aux.get_atom_nb(i), ao.get_atom_nb(i), ao.get_atom_nb(j)});
                    for (std::size_t p = 0; p != block.shape[0]; ++p)
                        for (std::size_t a = 0; a != block.shape[1]; ++a)
                            for (std::size_t b = 0; b != block.shape[2]; ++b)
                                block(p, a, b) = 0.12 *
                                                 std::sin(0.41 + 0.73 * i + 0.31 * j + 0.57 * p +
                                                          0.23 * a + 0.87 * b + 0.46 * r) /
                                                 (1.0 + r * r);
                    all_cs.data_libri[i][{j, {r, 0, 0}}] = std::move(block);
                }
        for (int p = 0; p != naux; ++p)
            for (int q = 0; q != naux; ++q)
                metric(p, q) = (p == q ? 1.3 + 0.4 * p : 0.11 * (1 + p + q));
        const double weights[nk] = {0.2, 0.3, 0.5};
        for (int k = 0; k != nk; ++k)
        {
            Matz d(nao, nao, MAJOR::COL);
            // Trace-zero signed density differences with nonzero complex
            // offdiagonal entries at every k; no time-reversal assumption.
            for (int a = 0; a != nao; ++a)
                for (int b = 0; b != nao; ++b)
                {
                    const cplxdb u = std::polar(1.0 / std::sqrt(3.0), 0.39 * (k + 1) * (a - b));
                    const cplxdb v = std::polar(1.0 / std::sqrt(3.0), 0.71 * (k + 2) * (a - b));
                    d(a, b) = weights[k] * (u - v);
                }
            density.push_back(std::move(d));
        }
    }

    Cs_LRI distributed_cs(const MpiCommHandler& comm) const
    {
        Cs_LRI cs;
        int block = 0;
        for (const auto& i : all_cs.data_libri)
            for (const auto& jr : i.second)
            {
                if (block % comm.nprocs == comm.myid) cs.data_libri[i.first][jr.first] = jr.second;
                ++block;
            }
        return cs;
    }
};

// Independent finite-supercell AO oracle.  Explicitly builds an AO-pair
// product tensor, a dense four-AO ERI, contracts the real-space density, then
// Bloch-transforms the resulting operator.  No LibRI Fourier/contraction helper
// is used.  The prescribed RI C/V model is the input: this checks exact RI
// algebra and normalization, not the physical accuracy of a fitted RI basis.
std::vector<Matz> finite_supercell_reference(const Fixture& f)
{
    constexpr int n = nao * nk, m = naux * nk;
    const auto pair_index = [](int p, int a, int b) { return (p * n + a) * n + b; };
    std::vector<cplxdb> product(m * n * n, 0.0);
    const double shift = f.pbc.kfrac_list[0].x;
    for (const auto& i : f.all_cs.data_libri)
        for (const auto& jr : i.second)
        {
            const int j = jr.first.first, r = jr.first.second[0];
            const auto& c = jr.second;
            for (int cell = 0; cell != nk; ++cell)
            {
                const int other = ((cell + r) % nk + nk) % nk;
                // Finite twisted boundary condition for a shifted mesh.
                const cplxdb winding = std::polar(1.0, two_pi * shift * (r - other + cell));
                for (std::size_t p = 0; p != c.shape[0]; ++p)
                    for (std::size_t a = 0; a != c.shape[1]; ++a)
                        for (std::size_t b = 0; b != c.shape[2]; ++b)
                        {
                            const int pp = cell * naux + f.aux.get_global_index(i.first, p);
                            const int aa = cell * nao + f.ao.get_global_index(i.first, a);
                            const int bb = other * nao + f.ao.get_global_index(j, b);
                            product[pair_index(pp, aa, bb)] += winding * c(p, a, b);
                            product[pair_index(pp, bb, aa)] += std::conj(winding) * c(p, a, b);
                        }
            }
        }
    const auto eri_index = [](int a, int b, int c, int d) { return ((a * n + b) * n + c) * n + d; };
    std::vector<cplxdb> eri(n * n * n * n, 0.0);
    for (int a = 0; a != n; ++a)
        for (int b = 0; b != n; ++b)
            for (int c = 0; c != n; ++c)
                for (int d = 0; d != n; ++d)
                    for (int p = 0; p != m; ++p)
                        for (int q = 0; q != m; ++q)
                            eri[eri_index(a, b, c, d)] +=
                                product[pair_index(p, a, b)] *
                                (f.metric(p % naux, q % naux) / static_cast<double>(nk)) *
                                product[pair_index(q, c, d)];
    Matz real_density(n, n, MAJOR::ROW), h(n, n, MAJOR::ROW);
    for (int a = 0; a != n; ++a)
        for (int b = 0; b != n; ++b)
            for (int k = 0; k != nk; ++k)
                real_density(a, b) +=
                    f.density[k](a % nao, b % nao) *
                    std::polar(1.0, two_pi * f.pbc.kfrac_list[k].x * (a / nao - b / nao));
    for (int a = 0; a != n; ++a)
        for (int b = 0; b != n; ++b)
            for (int c = 0; c != n; ++c)
                for (int d = 0; d != n; ++d)
                    h(a, b) += eri[eri_index(a, b, c, d)] * real_density(d, c);
    std::vector<Matz> result;
    for (int k = 0; k != nk; ++k)
    {
        Matz hk(nao, nao, MAJOR::COL);
        for (int a = 0; a != n; ++a)
            for (int b = 0; b != n; ++b)
                hk(a % nao, b % nao) +=
                    h(a, b) *
                    std::polar(1.0 / nk, two_pi * f.pbc.kfrac_list[k].x * (b / nao - a / nao));
        result.push_back(std::move(hk));
    }
    return result;
}

// Explicit four-AO integrals between an arbitrary bra Bloch pair and each
// integration-mesh ket pair. This does not use the kernel's auxiliary charge
// or its backprojection. Physical unwrapped translations enter before forming
// the four-index integral, so folding R incorrectly changes the off-mesh oracle.
std::vector<Matz> arbitrary_k_eri_reference(const Fixture& f,
                                            const std::vector<Vector3_Order<double>>& targets)
{
    const auto pair_at = [&](const Vector3_Order<double>& k)
    {
        std::vector<cplxdb> q(naux * nao * nao, 0.0);
        for (const auto& i : f.all_cs.data_libri)
            for (const auto& jr : i.second)
            {
                const auto& c = jr.second;
                const auto& r = jr.first.second;
                const cplxdb phase =
                    std::polar(1.0, two_pi * (k.x * r[0] + k.y * r[1] + k.z * r[2]));
                for (std::size_t mu = 0; mu != c.shape[0]; ++mu)
                    for (std::size_t a = 0; a != c.shape[1]; ++a)
                        for (std::size_t b = 0; b != c.shape[2]; ++b)
                        {
                            const int p = f.aux.get_global_index(i.first, mu);
                            const int ia = f.ao.get_global_index(i.first, a);
                            const int jb = f.ao.get_global_index(jr.first.first, b);
                            q[(p * nao + ia) * nao + jb] += phase * c(mu, a, b);
                            q[(p * nao + jb) * nao + ia] += std::conj(phase) * c(mu, a, b);
                        }
            }
        return q;
    };
    std::vector<Matz> result;
    for (const auto& target : targets)
    {
        Matz h(nao, nao, MAJOR::COL);
        const auto bra = pair_at(target);
        for (int k = 0; k != nk; ++k)
        {
            const auto ket = pair_at(f.pbc.kfrac_list[k]);
            std::vector<cplxdb> eri(nao * nao * nao * nao, 0.0);
            for (int a = 0; a != nao; ++a)
                for (int b = 0; b != nao; ++b)
                    for (int c = 0; c != nao; ++c)
                        for (int d = 0; d != nao; ++d)
                            for (int p = 0; p != naux; ++p)
                                for (int q = 0; q != naux; ++q)
                                    eri[((a * nao + b) * nao + c) * nao + d] +=
                                        bra[(p * nao + a) * nao + b] * f.metric(p, q) *
                                        ket[(q * nao + c) * nao + d];
            for (int a = 0; a != nao; ++a)
                for (int b = 0; b != nao; ++b)
                    for (int c = 0; c != nao; ++c)
                        for (int d = 0; d != nao; ++d)
                            h(a, b) +=
                                eri[((a * nao + b) * nao + c) * nao + d] * f.density[k](d, c);
        }
        result.push_back(std::move(h));
    }
    return result;
}

void check_density(int nspin, int nspinor, bool replicated, bool kblacs)
{
    const auto& comm = global::mpi_comm_global_h;
    Dataset ds(comm.comm);
    ds.basis_wfc.set(std::vector<std::size_t>{2, 1});
    ds.mf.set(nspin, nk, 3, nao, nspinor);
    if (kblacs)
    {
        ds.scfk_blacs_ctxt.init({1, comm.nprocs}, comm.comm, nk);
        ds.desc_wfc_kb = ds.scfk_blacs_ctxt.create_array_desc(nao, 3, 1, 1);
        ds.mark_eigvecs_kpara_2d_ready();
    }
    std::vector<Matz> expected;
    for (int k = 0; k != nk; ++k)
    {
        Matz d(nao, nao, MAJOR::COL);
        for (int spin = 0; spin != nspin; ++spin)
        {
            // Empty first state and tiny later occupation expose the old EXX
            // break/cutoff; scalar capacity 2 is already in these occupations.
            const double capacity = 2.0 / (nspin * nspinor * nk);
            for (int band = 0; band != 3; ++band)
                ds.mf.get_weight()[spin](k, band) = capacity * (band == 0   ? 0.0
                                                                : band == 1 ? 0.7
                                                                            : 1e-7);
            for (int spinor = 0; spinor != nspinor; ++spinor)
            {
                ComplexMatrix full(3, nao);
                for (int band = 0; band != 3; ++band)
                    for (int a = 0; a != nao; ++a)
                        full(band, a) = std::polar(
                            1.0 / std::sqrt(3.0 * nspinor),
                            0.43 * a * (band + 1) + 0.27 * k + 0.19 * spin + 0.31 * spinor * a);
                for (int a = 0; a != nao; ++a)
                    for (int b = 0; b != nao; ++b)
                        for (int band = 0; band != 3; ++band)
                            d(a, b) += ds.mf.get_weight()[spin](k, band) * full(band, a) *
                                       std::conj(full(band, b));
                if (kblacs)
                {
                    ComplexMatrix block(ds.desc_wfc_kb.n_loc(), ds.desc_wfc_kb.m_loc());
                    for (int b = 0; b != block.nr; ++b)
                        for (int a = 0; a != block.nc; ++a)
                            block(b, a) =
                                full(ds.desc_wfc_kb.indx_l2g_c(b), ds.desc_wfc_kb.indx_l2g_r(a));
                    ds.mf.get_eigenvectors()[spin][spinor][k] = std::move(block);
                }
                else if (replicated || k % comm.nprocs == comm.myid)
                    ds.mf.get_eigenvectors()[spin][spinor][k] = std::move(full);
            }
        }
        expected.push_back(std::move(d));
    }
    const auto actual = build_hartree_density_k(ds);
    for (int k = 0; k != nk; ++k)
        require(matrix_error(actual[k], expected[k]) < 2e-14,
                "Physical Hartree density spin/occupation/ownership mismatch");
    // A frozen DFT density remains immutable when the same Dataset advances.
    for (auto& spin : ds.mf.get_eigenvectors())
        for (auto& spinor : spin.second)
            for (auto& k : spinor.second)
                for (int a = 0; a != k.second.nc; ++a)
                    for (int band = 0; band != k.second.nr; ++band)
                        k.second(band, a) *= std::polar(1.0, 0.29 * a);
    const auto live = build_hartree_density_k(ds);
    double changed = 0.0;
    for (int k = 0; k != nk; ++k)
    {
        require(matrix_error(actual[k], expected[k]) < 2e-14, "Frozen DFT Hartree density mutated");
        changed = std::max(changed, matrix_error(live[k], actual[k]));
    }
    // A distributed BLACS row may carry global AO phases differently; the
    // immutable-density check applies to all layouts, nonzero control to dense.
    if (!kblacs) require(changed > 1e-4, "Live Hartree density did not change");
}
// An even-mesh boundary needs opposite BvK images. The closed-form Hartree
// values are H(0)=1.026, H(1/2)=0.2052 and H(1/4)=their mean.
void check_even_mesh(const MpiCommHandler& comm)
{
    AtomicBasis basis{std::vector<std::size_t>{1}};
    PeriodicBoundaryData pbc;
    pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
    pbc.set_kgrids_kvec(2, 1, 1, {0, 0, 0, two_pi / 2, 0, 0});
    Cs_LRI cs;
    if (comm.is_root())
        for (int r = 0; r != 2; ++r)
        {
            RI::Tensor<double> c({1, 1, 1});
            c(0, 0, 0) = r == 0 ? 0.3 : 0.2;
            cs.data_libri[0][{0, {r, 0, 0}}] = std::move(c);
        }
    Matz metric(1, 1, MAJOR::COL);
    metric(0, 0) = 2.7;
    std::vector<Matz> density;
    for (double d : {0.3, 0.4})
    {
        Matz block(1, 1, MAJOR::COL);
        block(0, 0) = d;
        density.push_back(std::move(block));
    }
    const std::vector<Vector3_Order<double>> points{{0, 0, 0}, {0.5, 0, 0}, {0.25, 0, 0}};
    const std::map<atom_t, Vector3_Order<double>> positions{{0, {0, 0, 0}}};
    for (int convention : {0, 1})
    {
        const AtomPairBvKRemap<atom_t> remap(positions, pbc.Rlist, pbc.period, pbc.latvec,
                                             convention);
        const auto result =
            compute_hartree_ao_k(basis, basis, cs, metric, pbc, density, comm, &points, remap);
        for (int k = 0; k != 3; ++k)
            require(std::abs(result.at(k)(0, 0) - (k == 0   ? 1.026
                                                   : k == 1 ? 0.2052
                                                            : 0.6156)) < 2e-12,
                    "Even-mesh Hartree BvK boundary differs from closed form");
    }
}

}  // namespace

int main(int argc, char** argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io(false, "stdout", false);
    int failure = 0;
    try
    {
        const auto& comm = global::mpi_comm_global_h;
        check_even_mesh(comm);
        double error = 0.0;
        for (double shift : {0.0, 1.0 / 6.0})
        {
            Fixture fixture(shift);
            const auto reference = finite_supercell_reference(fixture);
            const auto cs = fixture.distributed_cs(comm);
            const auto actual = compute_hartree_ao_k(fixture.ao, fixture.aux, cs, fixture.metric,
                                                     fixture.pbc, fixture.density, comm);
            double norm = 0.0;
            for (int k = 0; k != nk; ++k)
            {
                error = std::max(error, matrix_error(actual.at(k), reference[k]));
                for (std::size_t n = 0; n != reference[k].size(); ++n)
                    norm = std::max(norm, std::abs(reference[k].ptr()[n]));
            }
            require(norm > 1e-4, "Hartree oracle fixture is accidentally zero");
            require(error < 2e-12, "LibRI Hartree differs from finite-supercell AO ERI oracle");
            std::vector<Vector3_Order<double>> targets = fixture.pbc.kfrac_list;
            targets.push_back({0.173, 0.0, 0.0});
            targets.push_back({-0.219, 0.0, 0.0});
            const std::map<atom_t, Vector3_Order<double>> positions{{0, {0.0, 0.0, 0.0}},
                                                                    {1, {0.25, 0.0, 0.0}}};
            double direct_difference = 0.0;
            const auto direct = arbitrary_k_eri_reference(fixture, targets);
            for (int convention : {0, 1})
            {
                const AtomPairBvKRemap<atom_t> remap(positions, fixture.pbc.Rlist,
                                                     fixture.pbc.period, fixture.pbc.latvec,
                                                     convention);
                const auto path =
                    compute_hartree_ao_k(fixture.ao, fixture.aux, cs, fixture.metric, fixture.pbc,
                                         fixture.density, comm, &targets, remap);
                for (std::size_t k = 0; k != targets.size(); ++k)
                {
                    // Independent three-point cardinal trigonometric interpolant
                    // of the explicit four-AO mesh oracle (not kernel output).
                    Matz expected(nao, nao, MAJOR::COL);
                    for (int mesh = 0; mesh != nk; ++mesh)
                    {
                        const double weight =
                            (1.0 + 2.0 * std::cos(two_pi * (targets[k].x -
                                                            fixture.pbc.kfrac_list[mesh].x))) /
                            nk;
                        for (int a = 0; a != nao; ++a)
                            for (int b = 0; b != nao; ++b)
                                expected(a, b) += weight * reference[mesh](a, b);
                    }
                    require(matrix_error(path.at(static_cast<int>(k)), expected) < 2e-12,
                            "Hartree path differs from finite-mesh trigonometric oracle");
                    if (k < nk)
                        require(matrix_error(path.at(static_cast<int>(k)), actual.at(k)) < 2e-12,
                                "Hartree BvK interpolation failed mesh roundtrip");
                    direct_difference = std::max(
                        direct_difference, matrix_error(path.at(static_cast<int>(k)), direct[k]));
                }
            }
            require(
                direct_difference > 1e-4,
                "Hartree control does not distinguish mesh interpolation from raw-R projection");

            std::vector<Matz> zero;
            for (int k = 0; k != nk; ++k) zero.emplace_back(nao, nao, MAJOR::COL);
            const auto zero_result = compute_hartree_ao_k(fixture.ao, fixture.aux, cs,
                                                          fixture.metric, fixture.pbc, zero, comm);
            for (int k = 0; k != nk; ++k)
                require(matrix_error(zero_result.at(k), zero[k]) < 1e-15,
                        "Zero Hartree feedback changed the reference");
        }
        for (const auto& spins : std::initializer_list<std::pair<int, int>>{{1, 1}, {2, 1}, {1, 2}})
        {
            check_density(spins.first, spins.second, true, false);
            check_density(spins.first, spins.second, false, false);
            check_density(spins.first, spins.second, false, true);
        }
        if (comm.is_root())
            std::cout << "HARTREE_RESULT ranks=" << comm.nprocs << " max_ao_eri_error=" << error
                      << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << "Hartree control failed: " << e.what() << '\n';
        failure = 1;
    }
    int any_failure = 0;
    MPI_Allreduce(&failure, &any_failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    return any_failure;
}
