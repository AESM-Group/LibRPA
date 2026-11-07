#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <valarray>

#include "../core/chi0.h"
#include "../api/dataset.h"
#include "../io/global_io.h"
#include "../mpi/global_mpi.h"

using namespace librpa_int;

namespace
{
using Complex = std::complex<double>;

void close(Complex actual, Complex expected, double tolerance = 2e-5)
{
    if (!std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
        std::abs(actual - expected) > tolerance)
    {
        std::cerr << std::setprecision(17) << "response " << actual << ", reference " << expected
                  << ", tolerance " << tolerance << '\n';
        throw std::runtime_error("cRPA selected-band response mismatch");
    }
}

struct Fixture
{
    Dataset ds{MPI_COMM_WORLD};
    std::unique_ptr<Chi0> chi;
    std::vector<atpair_t> pairs;
    std::map<Vector3_Order<double>, ComplexMatrix> no_shrink;
    std::array<double, 3> energies{-0.7, 0.3, 1.1};
    std::array<double, 3> occupations{1.0, 0.0, 0.0};
    // The RI pair-density vertex is F = C + transpose(C).
    const std::array<double, 9> vertex{0.4, 0.3, 0.4, 0.3, 0.7, 0.2, 0.4, 0.2, 0.9};

    Fixture()
    {
        ds.basis_wfc.set(std::vector<std::size_t>{3});
        ds.basis_aux.set(std::vector<std::size_t>{1});
        ds.mf.set(1, 4, 3, 3, 1);
        ds.mf.get_efermi() = 0.0;
        ds.pbc.set_latvec({1, 0, 0, 0, 1, 0, 0, 0, 1});
        const double pi = std::acos(-1.0);
        ds.pbc.set_kgrids_kvec(4, 1, 1, {0, 0, 0, pi / 2, 0, 0, pi, 0, 0, -pi / 2, 0, 0});
        ds.scfk_blacs_ctxt.init(KPointBlacsProcessShape(1, ds.comm_h.nprocs, true), MPI_COMM_WORLD,
                                4);
        ds.desc_wfc_kb_full = ds.scfk_blacs_ctxt.create_array_desc(3, 3, 3, 3);
        ds.desc_abf.reset_handler(ds.blacs_h);
        ds.desc_abf.init_1b1p(1, 1, 0, 0);
        ds.tfg.reset(16);
        ds.tfg.generate_minimax(0.4, 4.0);
        for (int k = 0; k < 4; ++k)
        {
            ComplexMatrix eigenvectors(3, 3);
            eigenvectors.set_as_identity_matrix();
            ds.mf.get_eigenvectors()[0][0][k] = eigenvectors;
        }
        update_spectrum();
        ds.cs_data.use_libri = true;
        if (ds.comm_h.is_root())
        {
            auto data = std::make_shared<std::valarray<double>>(0.0, 9);
            for (int i = 0; i < 9; ++i) (*data)[i] = 0.5 * vertex[i];
            ds.cs_data.data_libri[0][{0, {0, 0, 0}}] = RI::Tensor<double>({1, 3, 3}, data);
        }
        const auto assigned = dispatch_upper_triangular_tasks(1, ds.blacs_h.myid, ds.blacs_h.nprows,
                                                              ds.blacs_h.npcols, ds.blacs_h.myprow,
                                                              ds.blacs_h.mypcol);
        pairs.assign(assigned.begin(), assigned.end());
        chi = std::make_unique<Chi0>(ds.mf, ds.basis_wfc, ds.basis_aux, ds.pbc, ds.symmetry_context,
                                     ds.tfg, ds.scfk_blacs_ctxt, ds.desc_wfc_kb_full, false, false);
        chi->gf_threshold = chi->libri_threshold_C = chi->libri_threshold_G = 0.0;
        chi->nbands_G = 3;
    }

    void update_spectrum()
    {
        for (int k = 0; k < 4; ++k)
            for (int b = 0; b < 3; ++b)
            {
                ds.mf.get_eigenvals()[0](k, b) = energies[b];
                ds.mf.get_weight()[0](k, b) = 0.5 * occupations[b];  // 2 f / Nk
            }
    }

    Chi0::BandSelection selection(std::array<double, 3> row) const
    {
        return Chi0::BandSelection(
            1, std::vector<std::vector<double>>(4, std::vector<double>(row.begin(), row.end())));
    }

    void build()
    {
        chi->build(LIBRPA_ROUTING_LIBRI, LIBRPA_METHOD_AUTO, ds.cs_data, pairs, ds.basis_aux,
                   no_shrink, ds.blacs_h);
    }

    Complex value(const Chi0QMap &response, double omega, const Vector3_Order<double> &q) const
    {
        Complex local = 0.0, total = 0.0;
        const auto frequency = response.find(omega);
        if (frequency != response.end())
        {
            const auto point = frequency->second.find(q);
            if (point != frequency->second.end())
                for (const auto &[i, columns] : point->second)
                    for (const auto &[j, block] : columns) local += block(0, 0);
        }
        ds.comm_h.allreduce(&local, &total, 1, MPI_SUM);
        return total;
    }

    Complex lindhard(double omega, std::array<double, 3> mask) const
    {
        Complex result = 0.0;
        for (int n = 0; n < 3; ++n)
            for (int m = 0; m < 3; ++m)
                if (mask[n] && mask[m])
                    result += 2.0 * mask[n] * mask[m] * vertex[3 * n + m] * vertex[3 * n + m] *
                              (occupations[n] - occupations[m]) /
                              Complex(energies[n] - energies[m], omega);
        return result;
    }
};

void test_selected_response()
{
    Fixture f;
    // The mask-reset comparison below tests state reuse at 1e-12. Keep its
    // two unmasked builds in the same reduction order: threaded GF/LibRI
    // summation roundoff is amplified by the minimax cosine transform.
    // The selected responses still run with the requested thread count and
    // are checked against the independent Lindhard sum.
    const int requested_threads = omp_get_max_threads();
    omp_set_num_threads(1);
    f.build();
    omp_set_num_threads(requested_threads);
    auto full = f.chi->take_chi0_q();
    const std::array<std::array<double, 3>, 6> cases{
        {{0, 0, 0}, {1, 0, 0}, {1, 1, 1}, {1, 1, 0}, {0.17, 0.63, 0.82}, {1, 1.e-7, 0.2}}};
    for (const auto &mask : cases)
    {
        f.chi->set_band_selection(f.selection(mask));
        f.build();
        for (const auto omega : f.ds.tfg.get_freq_nodes())
            for (const auto &q : f.chi->active_qpoints())
            {
                close(f.value(full, omega, q), f.lindhard(omega, {1, 1, 1}));
                close(f.value(f.chi->get_chi0_q(), omega, q), f.lindhard(omega, mask));
            }
        f.chi->replace_chi0_q_by_difference(full);
        for (const auto omega : f.ds.tfg.get_freq_nodes())
            for (const auto &q : f.chi->active_qpoints())
                close(f.value(f.chi->get_chi0_q(), omega, q),
                      f.lindhard(omega, {1, 1, 1}) - f.lindhard(omega, mask));
    }
    // Different masks at even and odd k must act on both endpoints (k,k+q).
    // Even q retains half of the 0->1 transitions; odd q additionally retains
    // half of 0->2. This catches a selector applied at the wrong endpoint.
    auto alternating = f.selection({1, 1, 0});
    alternating[0][1] = alternating[0][3] = {0, 1, 1};
    f.chi->set_band_selection(alternating);
    f.build();
    for (const auto omega : f.ds.tfg.get_freq_nodes())
        for (const auto &q : f.chi->active_qpoints())
        {
            const auto fractional_q = f.ds.pbc.latvec * q;
            const auto shift = std::llround(4 * fractional_q.x);
            const auto expected =
                0.5 * f.lindhard(omega, shift % 2 ? std::array<double, 3>{1, 1, 1}
                                                  : std::array<double, 3>{1, 1, 0});
            close(f.value(f.chi->get_chi0_q(), omega, q), expected);
        }
    // An independent explicit k sum also checks fractional weights on unequal
    // k and k+q endpoints, including the reverse (empty-to-occupied) term.
    // Retain w(k)=w(-k), as required by this real scalar/even-frequency fixture.
    alternating[0] = {
        {0.17, 0.63, 0.82}, {0.91, 0.24, 0.35}, {0.42, 0.73, 0.08}, {0.91, 0.24, 0.35}};
    f.chi->set_band_selection(alternating);
    f.build();
    for (const auto omega : f.ds.tfg.get_freq_nodes())
        for (const auto &q : f.chi->active_qpoints())
        {
            const int shift = std::llround(4 * (f.ds.pbc.latvec * q).x);
            Complex expected = 0;
            for (int k = 0; k < 4; ++k)
                for (int n = 0; n < 3; ++n)
                    for (int m = 0; m < 3; ++m)
                    {
                        const int kq = (k + shift + 4) % 4;
                        expected += 0.5 * alternating[0][k][n] * alternating[0][kq][m] *
                                    std::pow(f.vertex[3 * n + m], 2) *
                                    (f.occupations[n] - f.occupations[m]) /
                                    Complex(f.energies[n] - f.energies[m], omega);
                    }
            close(f.value(f.chi->get_chi0_q(), omega, q), expected);
        }
    // Disabling a mask must recover the original full response on the same
    // Chi0 instance. Repeated builds also exercise real-space task reset.
    f.chi->clear_band_selection();
    omp_set_num_threads(1);
    f.build();
    omp_set_num_threads(requested_threads);
    for (const auto omega : f.ds.tfg.get_freq_nodes())
        for (const auto &q : f.chi->active_qpoints())
            close(f.value(f.chi->get_chi0_q(), omega, q), f.value(full, omega, q), 1e-12);
    auto current = f.chi->take_chi0_q();
    f.chi->swap_chi0_q(current);
    if (!current.empty()) throw std::runtime_error("Chi0 ownership exchange left a duplicate map");
}

void test_fractional_manifold_cancellation()
{
    Fixture f;
    f.energies = {-1.0, 0.0, 1.5};
    f.occupations = {1.0, 0.25, 0.0};
    f.update_spectrum();
    f.build();
    auto full = f.chi->take_chi0_q();
    // A partially occupied level at EF has a spurious constant-time diagonal
    // term in the inherited zero-temperature GF construction. Excluding that
    // entire level cancels the same term in P0-Pd. This controlled case is not
    // a validation of finite smearing or of all metallic response functions.
    f.chi->set_band_selection(f.selection({0, 1, 0}));
    f.build();
    f.chi->replace_chi0_q_by_difference(full);
    double full_discrepancy = 0.0;
    for (const auto omega : f.ds.tfg.get_freq_nodes())
        for (const auto &q : f.chi->active_qpoints())
        {
            const auto exact = f.lindhard(omega, {1, 1, 1});
            full_discrepancy =
                std::max(full_discrepancy, std::abs(f.value(full, omega, q) - exact));
            close(f.value(f.chi->get_chi0_q(), omega, q), exact);
        }
    if (full_discrepancy < 1e-4)
        throw std::runtime_error(
            "fractional-occupation control did not expose the full-response discrepancy");
    if (f.ds.comm_h.is_root())
        std::cout << "Fractional control: P0-Lindhard maximum " << full_discrepancy
                  << "; selected-manifold Pr agrees within 2e-5 Ha^-1\n";
}
}  // namespace

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    global::init_global_mpi(MPI_COMM_WORLD);
    global::init_global_io();
    test_selected_response();
    test_fractional_manifold_cancellation();
    global::finalize_global_io();
    global::finalize_global_mpi();
    MPI_Finalize();
    std::cout << "test_crpa_band_response: passed\n";
}
