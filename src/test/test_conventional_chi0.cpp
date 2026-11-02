#include <array>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../api/dataset.h"
#include "../core/chi0.h"
#include "../core/epsilon.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"
#include "../utils/constants.h"

using namespace librpa_int;
namespace
{
using Complex = std::complex<double>;
void close(Complex actual, Complex expected, double tolerance = 2.e-11)
{
    if (!std::isfinite(std::abs(actual)) || std::abs(actual - expected) > tolerance)
    {
        std::cerr << "actual=" << actual << " expected=" << expected << " tolerance=" << tolerance
                  << '\n';
        throw std::runtime_error("conventional response mismatch");
    }
}

struct Coefficient
{
    int I, J, R;
    std::vector<double> values;  // mu,i,j
};

struct Fixture
{
    Dataset ds{MPI_COMM_WORLD};
    std::unique_ptr<Chi0> chi;
    std::vector<atpair_t> pairs;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    std::vector<Coefficient> cs{{0, 0, 0, {0.2}},
                                {0, 1, 1, {0.12, -0.17}},
                                {1, 0, -1, {0.21, 0.13, -0.05, 0.18}},
                                {1, 1, 0, {0.12, 0.03, 0.07, 0.22, 0.31, -0.09, -0.04, 0.25}}};
    // Deliberately shuffled full k mesh; k+q cannot be found by adding indices.
    std::array<int, 4> mesh{2, 0, 3, 1};

    explicit Fixture(int nspin = 1)
    {
        ds.basis_wfc.set(std::vector<std::size_t>{1, 2});
        ds.basis_aux.set(std::vector<std::size_t>{1, 2});
        ds.mf.set(nspin, 4, 3, 3, 1);
        ds.mf.get_efermi() = 0;
        ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        std::vector<double> kv;
        for (const auto k : mesh)
        {
            kv.push_back(TWO_PI * k / 4);
            kv.push_back(0);
            kv.push_back(0);
        }
        ds.pbc.set_kgrids_kvec(4, 1, 1, kv);
        ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD,
                                4);
        ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(3, 3, 3, 3);
        ds.desc_abf.reset_handler(ds.blacs_h);
        ds.desc_abf.init_1b1p(3, 3, 0, 0);
        ds.tfg.reset(3);
        ds.tfg.generate_evenspaced(0.17, 0.61);
        const double u[3][3] = {{1 / std::sqrt(3.), 1 / std::sqrt(3.), 1 / std::sqrt(3.)},
                                {1 / std::sqrt(2.), -1 / std::sqrt(2.), 0},
                                {1 / std::sqrt(6.), 1 / std::sqrt(6.), -2 / std::sqrt(6.)}};
        for (int s = 0; s < nspin; ++s)
            for (int k = 0; k < 4; ++k)
            {
                const double angle = TWO_PI * mesh[k] / 4;
                const double dispersion = std::cos(angle);
                const std::array<double, 3> e{-0.7 + 0.1 * dispersion, 0.5 + 0.2 * dispersion,
                                              1.1 - 0.1 * dispersion};
                ComplexMatrix c(3, 3);
                for (int n = 0; n < 3; ++n)
                {
                    ds.mf.get_eigenvals()[s](k, n) = e[n] + 0.1 * s * n;
                    ds.mf.get_weight()[s](k, n) = n == 0 ? 2.0 / nspin / 4 : 0;
                    for (int a = 0; a < 3; ++a) c(n, a) = u[n][a] * std::polar(1.0, angle * a);
                }
                ds.mf.get_eigenvectors()[s][0][k] = c;
            }
        ds.cs_data.use_libri = true;
        for (std::size_t b = 0; b < cs.size(); ++b)
        {
            if (static_cast<int>(b) % ds.comm_h.nprocs != ds.comm_h.myid) continue;
            const auto &c = cs[b];
            auto data = std::make_shared<std::valarray<double>>(c.values.data(), c.values.size());
            ds.cs_data.data_libri[c.I][{c.J, {c.R, 0, 0}}] =
                RI::Tensor<double>({ds.basis_aux[c.I], ds.basis_wfc[c.I], ds.basis_wfc[c.J]}, data);
        }
        const auto assigned = dispatch_upper_triangular_tasks(2, ds.blacs_h.myid, ds.blacs_h.nprows,
                                                              ds.blacs_h.npcols, ds.blacs_h.myprow,
                                                              ds.blacs_h.mypcol);
        pairs.assign(assigned.begin(), assigned.end());
        chi = std::make_unique<Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
                                     ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
        chi->gf_threshold = chi->libri_threshold_C = chi->libri_threshold_G = 0;
    }

    void build()
    {
        chi->build(LIBRPA_ROUTING_LIBRI, ds.cs_data, pairs, ds.basis_aux, no_shrink, ds.blacs_h);
    }

    ComplexMatrix oracle(double omega, const Vector3_Order<double> &q) const
    {
        ComplexMatrix result(3, 3);
        const int shift = std::llround(4 * (ds.pbc.latvec * q).x);
        const int last = chi->nbands_G < 0 ? 3 : chi->nbands_G;
        for (int s = 0; s < ds.mf.get_n_spins(); ++s)
            for (int k = 0; k < 4; ++k)
            {
                int kq = -1;
                for (int l = 0; l < 4; ++l)
                    if ((mesh[l] - mesh[k] - shift + 12) % 4 == 0) kq = l;
                if (kq < 0) throw std::runtime_error("bad oracle mesh");
                const auto &left = ds.mf.get_eigenvectors().at(s).at(0).at(k);
                const auto &right = ds.mf.get_eigenvectors().at(s).at(0).at(kq);
                for (int n = chi->n_bands_exclude; n < last; ++n)
                    for (int m = chi->n_bands_exclude; m < last; ++m)
                    {
                        std::array<Complex, 3> D{};
                        // Literal real-space pair-density sum, independent of the Fourier/GEMM
                        // kernel.
                        for (const auto &c : cs)
                        {
                            const int ni = ds.basis_wfc[c.I], nj = ds.basis_wfc[c.J];
                            const int i0 = ds.basis_wfc.get_part_range()[c.I],
                                      j0 = ds.basis_wfc.get_part_range()[c.J];
                            const int mu0 = ds.basis_aux.get_part_range()[c.I];
                            for (int mu = 0; mu < static_cast<int>(ds.basis_aux[c.I]); ++mu)
                                for (int i = 0; i < ni; ++i)
                                    for (int j = 0; j < nj; ++j)
                                        D[mu0 + mu] +=
                                            c.values[(mu * ni + i) * nj + j] *
                                            (std::conj(left(n, i0 + i)) * right(m, j0 + j) *
                                                 std::polar(1.0, TWO_PI * mesh[kq] * c.R / 4.) +
                                             std::conj(left(n, j0 + j)) * right(m, i0 + i) *
                                                 std::polar(1.0, -TWO_PI * mesh[k] * c.R / 4.));
                        }
                        const auto f =
                            (ds.mf.get_weight()[s](k, n) - ds.mf.get_weight()[s](kq, m)) /
                            Complex(
                                ds.mf.get_eigenvals()[s](k, n) - ds.mf.get_eigenvals()[s](kq, m),
                                omega);
                        for (int i = 0; i < 3; ++i)
                            for (int j = 0; j < 3; ++j) result(i, j) += f * D[i] * std::conj(D[j]);
                    }
            }
        return result;
    }

    void compare(double tolerance = 2.e-11) const
    {
        for (double w : ds.tfg.get_freq_nodes())
            for (const auto &q : chi->active_qpoints())
            {
                const auto ref = oracle(w, q);
                for (const auto &[I, J] : pairs)
                {
                    const auto &actual = chi->get_chi0_q().at(w).at(q).at(I).at(J);
                    const int i0 = ds.basis_aux.get_part_range()[I],
                              j0 = ds.basis_aux.get_part_range()[J];
                    for (int i = 0; i < actual.nr; ++i)
                        for (int j = 0; j < actual.nc; ++j)
                            close(actual(i, j), ref(i0 + i, j0 + j), tolerance);
                }
            }
    }
};

void test_response()
{
    for (int spins : {1, 2})
    {
        Fixture f(spins);
        f.build();
        f.compare();
        f.chi->nbands_G = 2;
        f.build();
        f.compare();
        f.chi->n_bands_exclude = 1;
        f.build();
        f.compare();
        f.chi->nbands_G = -1;
        f.chi->n_bands_exclude = 0;
        // Fractional occupations at positive frequencies, including equal-energy levels.
        for (int s = 0; s < spins; ++s)
            for (int k = 0; k < 4; ++k)
            {
                f.ds.mf.get_weight()[s](k, 1) = 0.13 / spins;
                f.ds.mf.get_eigenvals()[s](k, 2) = f.ds.mf.get_eigenvals()[s](k, 1);
            }
        f.build();
        f.compare();
    }
}

void test_space_time()
{
    Fixture f;
    f.ds.tfg.reset(16);
    f.ds.tfg.generate_minimax(0.8, 3.0);
    f.build();
    f.compare(2.e-5);
    const auto st = f.chi->take_chi0_q();
    const auto frequencies = f.ds.tfg.get_freq_nodes();
    for (double w : frequencies)
    {
        f.ds.tfg.reset(1);
        f.ds.tfg.generate_evenspaced(w, 0);
        f.build();
        f.compare();
        for (const auto &q : f.chi->active_qpoints())
            for (const auto &[I, J] : f.pairs)
            {
                const auto &a = f.chi->get_chi0_q().at(w).at(q).at(I).at(J);
                const auto &b = st.at(w).at(q).at(I).at(J);
                for (int i = 0; i < a.size; ++i) close(a.c[i], b.c[i], 2.e-5);
            }
    }
}

void test_energy()
{
    Fixture f;
    f.ds.tfg.reset(24);
    f.ds.tfg.generate_GaussLegendre();
    f.build();
    f.compare();
    atpair_k_cplx_mat_t v;
    for (const auto &q : f.chi->active_qpoints())
        for (const auto &[I, J] : f.pairs)
        {
            auto b = std::make_shared<ComplexMatrix>(f.ds.basis_aux[I], f.ds.basis_aux[J]);
            if (I == J) b->set_as_identity_matrix();
            v[I][J][q] = b;
        }
    // Independent 3x3 determinant and trace at V=identity, with the actual q/frequency weights.
    Complex expected = 0;
    const auto frequencies = f.ds.tfg.get_freq_nodes(), weights = f.ds.tfg.get_freq_weights();
    for (const auto &q : f.chi->active_qpoints())
        for (std::size_t iw = 0; iw < frequencies.size(); ++iw)
        {
            const auto x = f.oracle(frequencies[iw], q);
            ComplexMatrix a(3, 3);
            a.set_as_identity_matrix();
            a -= x;
            const auto det = a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) -
                             a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0)) +
                             a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0));
            expected += (std::log(det) + trace(x)) * weights[iw] * f.chi->q_weight(q) / TWO_PI;
        }
    const auto energy = compute_RPA_correlation_blacs_2d(*f.chi, v, f.pairs, f.ds.blacs_h);
    if (f.ds.comm_h.is_root()) close(energy.value, expected);
    if (f.ds.comm_h.is_root())
        std::cout << "Energy=" << energy.value << " reference=" << expected << '\n';
}

void test_rejections()
{
    Fixture f;
    auto rejected = [&](const std::string &message)
    {
        bool failed = false;
        try
        {
            f.build();
        }
        catch (const std::exception &e)
        {
            failed = std::string(e.what()).find(message) != std::string::npos ||
                     std::string(e.what()).find("another MPI rank rejected") != std::string::npos;
        }
        if (!failed) throw std::runtime_error("unsupported input accepted");
    };
    f.ds.tfg.reset(1);
    f.ds.tfg.generate_evenspaced(std::numeric_limits<double>::infinity(), 0);
    rejected("strictly positive imaginary frequencies");
    f.ds.tfg.generate_evenspaced(0.4, 0);
    // A spectrum that breaks k <-> -k symmetry has a non-Hermitian response.
    f.ds.mf.get_eigenvals()[0](3, 1) += 0.137;
    rejected("non-Hermitian imaginary-frequency response");
    f.ds.mf.get_eigenvals()[0](3, 1) -= 0.137;
    // Inconsistent data on one MPI owner must fail collectively before communication.
    if (f.ds.comm_h.is_root()) f.ds.mf.get_eigenvectors()[0][0][0].create(1, 1);
    rejected("missing full band/AO eigenvector matrix");
}
}  // namespace
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    test_response();
    test_space_time();
    test_energy();
    test_rejections();
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    std::cout << "test_conventional_chi0: passed\n";
}
