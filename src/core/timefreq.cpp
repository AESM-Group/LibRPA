#include "timefreq.h"

#include <omp.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#include "../io/global_io.h"
// #include "../io/stl_io_helper.h"
#include "../math/mathtools.h"
#include "../utils/base_utility.h"
#include "../utils/error.h"
#include "librpa_enums.h"
#include "minimax.h"

namespace librpa_int {

const string TFGrids::GRID_TYPES_NOTES[LIBRPA_TFGRID_COUNT] =
    {
        "Gauss-Legendre grids",
        "Gauss-Chebyshev grids of the first kind",
        "Gauss-Chebyshev grids of the second kind",
        "Minimax time-frequency grids",
        "Even-spaced frequency grids",
        "Even-spaced time-frequency grids (debug use)",
        "Split Gauss-Legendre grids",
    };

const bool TFGrids::SUPPORT_TIME_GRIDS[LIBRPA_TFGRID_COUNT] =
    { false, false, false, true, false, true, false };

void TFGrids::set_freq()
{
    freq_nodes.resize(n_grids);
    freq_weights.resize(n_grids);
}

void TFGrids::set_time()
{
    time_nodes.resize(n_grids);
    time_weights.resize(n_grids);
    costrans_t2f.create(n_grids, n_grids);
    sintrans_t2f.create(n_grids, n_grids);
    costrans_f2t.create(n_grids, n_grids);
    sintrans_f2t.create(n_grids, n_grids);
    // fourier_t2f.create(n_grids, n_grids);
}

void TFGrids::show() const
{
    using std::endl;
    using std::cout;
    using librpa_int::global::lib_printf;
    using librpa_int::global::should_output;

    if (!should_output()) return;

    cout << "Grid type: " << TFGrids::GRID_TYPES_NOTES[grid_type] << endl;
    cout << "Grid size: " << n_grids << endl;
    cout << "Frequency node & weight: " << endl;
    for ( size_t i = 0; i != n_grids; i++ )
        lib_printf("%2d %23.16f %23.16f\n", i, freq_nodes[i], freq_weights[i]);
    if (has_time_grids())
    {
        cout << "Time node & weight: " << endl;
        for ( size_t i = 0; i != n_grids; i++ )
            lib_printf("%2d %23.16f %23.16f\n", i, time_nodes[i], time_weights[i]);
        cout << "t->f transform: " << endl;
        if (costrans_t2f.size)
        {
            print_matrix("Cosine transform matrix (row: freq, col: time)", costrans_t2f);
        }
        if (sintrans_t2f.size)
        {
            print_matrix("Sine transform matrix (row: freq, col: time)", sintrans_t2f);
        }
        cout << "f->t transform: " << endl;
        if (costrans_f2t.size)
        {
            print_matrix("Cosine transform matrix (row: time, col: freq)", costrans_f2t);
        }
    }
    lib_printf("\n");
}

void TFGrids::unset()
{
    n_grids = 0;
    freq_nodes.clear();
    freq_weights.clear();
    time_nodes.clear();
    time_weights.clear();
    costrans_t2f.create(0, 0);
    sintrans_t2f.create(0, 0);
    costrans_f2t.create(0, 0);
    sintrans_f2t.create(0, 0);
    // fourier_t2f.create(0, 0);
}

TFGrids::TFGrids(const unsigned &N)
{
    n_grids = N;
    set_freq();
}

void TFGrids::reset(const unsigned &N)
{
    unset();
    n_grids = N;
    set_freq();
}

TFGrids::~TFGrids()
{
    /* unset(); */
}

double TFGrids::generate(LibrpaTimeFreqGrid gtype, double emin, double eintveral, double emax,
                         double tmin, double tinterval, double regulation)
{
    double retval = -1;
    switch (gtype)
    {
        case (LIBRPA_TFGRID_GAUSS_LEGENDRE):
        {
            this->generate_GaussLegendre();
            break;
        }
        case (LIBRPA_TFGRID_GAUSS_CHEBYSHEV_I):
        {
            this->generate_GaussChebyshevI();
            break;
        }
        case (LIBRPA_TFGRID_GAUSS_CHEBYSHEV_II):
        {
            this->generate_GaussChebyshevII();
            break;
        }
        case (LIBRPA_TFGRID_MINIMAX):
        {
            retval = this->generate_minimax(emin, emax, regulation);
            break;
        }
        case (LIBRPA_TFGRID_EVEN_SPACED):
        {
            this->generate_evenspaced(emin, eintveral);
            break;
        }
        case (LIBRPA_TFGRID_EVEN_SPACED_TF):
        {
            this->generate_evenspaced_tf(emin, eintveral, tmin, tinterval);
            break;
        }
        case (LIBRPA_TFGRID_SPLIT_GAUSS_LEGENDRE):
        {
            this->generate_SplitGaussLegendre(emax);
            break;
        }
        default:
            throw LIBRPA_RUNTIME_ERROR("requested time-frequency grid is not implemented");
    }
    return retval;
}

void TFGrids::generate_static_export(const TFGrids &source)
{
    if (source.grid_type != LIBRPA_TFGRID_MINIMAX || !source.has_time_grids())
        throw LIBRPA_RUNTIME_ERROR("static chi0 export requires minimax time quadrature");
    reset(source.n_grids);
    grid_type = source.grid_type;
    set_time();
    time_nodes = source.time_nodes; time_weights = source.time_weights;
    // GreenX scales its tabulated 1/x quadrature by 2*E_min for products
    // of responses. A single static response spans [E_min,E_max], so undo
    // that factor two here: integral exp(-E*tau) d tau must equal 1/E.
    for (auto &tau : time_nodes) tau *= 2.0;
    for (auto &weight : time_weights) weight *= 2.0;
    freq_nodes = source.freq_nodes; freq_weights = source.freq_weights;
    costrans_t2f.zero_out();
    freq_nodes[0] = 0.0;
    // GreenX time weights approximate integral_0^infty exp(-E*tau) d tau.
    // The even imaginary-time response requires both positive and negative tau.
    for (std::size_t it=0; it<n_grids; ++it) costrans_t2f(0,it)=2.0*time_weights[it];
}

void TFGrids::generate_single_frequency(double omega)
{
    if (!std::isfinite(omega) || omega < 0.)
        throw LIBRPA_RUNTIME_ERROR("imported imaginary frequency must be finite and nonnegative");
    reset(1);
    grid_type = LIBRPA_TFGRID_EVEN_SPACED;
    freq_nodes[0] = omega;
    freq_weights[0] = 0.; // No quadrature weight: this is a sample, not an integral.
}

void TFGrids::generate_evenspaced(double emin, double interval)
{
    if ( emin <= 0 )
        throw LIBRPA_RUNTIME_ERROR("emin must be positive");
    if ( interval < 0 )
        throw LIBRPA_RUNTIME_ERROR("emin must be non-negative");
    double weight = 1.0 / n_grids;
    for (size_t i = 0; i != n_grids; i++)
    {
        freq_nodes[i] = emin + interval * i;
        freq_weights[i] = weight;
    }
    grid_type = LIBRPA_TFGRID_EVEN_SPACED;
}

void TFGrids::generate_evenspaced_tf(double emin, double eintv, double tmin, double tintv)
{
    generate_evenspaced(emin, eintv);
    set_time();
    if ( tmin <= 0 )
        throw LIBRPA_RUNTIME_ERROR("tmin must be positive");
    if ( tintv < 0 )
        throw LIBRPA_RUNTIME_ERROR("tintv must be non-negative");
    double weight = 1.0 / n_grids;
    for (size_t i = 0; i != n_grids; i++)
    {
        time_nodes[i] = tmin + tintv * i;
        time_weights[i] = weight;
        // WARN: fake transform matrices
        costrans_t2f(i, i) = weight;
        sintrans_t2f(i, i) = weight;
        costrans_f2t(i, i) = 1/weight;
        sintrans_f2t(i, i) = 1/weight;
    }
    grid_type = LIBRPA_TFGRID_EVEN_SPACED_TF;
}

double TFGrids::generate_minimax(double emin, double emax, double regulation)
{
    grid_type = LIBRPA_TFGRID_MINIMAX;
    set_time();

    double max_errors[3];
    double cosft_duality_error = 1e8;
    int ierr = -1;

    // global::ofs_myid << "Before generate_minimax" << std::endl;
    auto n = as_int(n_grids);
    get_minimax_grid(n, emin, emax, time_nodes.data(), time_weights.data(), freq_nodes.data(),
                     freq_weights.data(), costrans_t2f.c, costrans_f2t.c, sintrans_t2f.c,
                     max_errors, cosft_duality_error, ierr, false, regulation);
    // global::ofs_myid << "After generate_minimax" << std::endl;
    // global::ofs_myid << freq_nodes << std::endl;

    switch (ierr) {
        case 0: /* success */
            break;
        case -1:
            throw LIBRPA_RUNTIME_ERROR("get_minimax_grid not properly called");
        case 1: /* GreenX internal code */
            throw LIBRPA_RUNTIME_ERROR(string("unsupported minimax grids size: ") + std::to_string(n_grids));
        default:
            throw LIBRPA_RUNTIME_ERROR(string("minimax grids failed, return code: ") + std::to_string(ierr));
    }

    return cosft_duality_error;

    // zmy debug
    // cout << "Cos transform time -> freq (freq each row)" << endl;
    // cout << costrans_t2f << endl;
    // cout << "Cos transform freq -> time (time each row)" << endl;
    // cout << costrans_f2t << endl;
    // cout << "Cos transform Delta" << endl;
    // cout << costrans_t2f * costrans_f2t << endl;
    // cout << "Sin transform time -> freq (freq each row)" << endl;
    // cout << sintrans_t2f << endl;
    // cout << "Sin transform freq -> time (time each row)" << endl;
    // cout << sintrans_f2t << endl;
    // cout << "Sin transform Delta" << endl;
    // cout << sintrans_t2f * sintrans_f2t << endl;
}

void TFGrids::generate_GaussChebyshevI()
{
    grid_type = LIBRPA_TFGRID_GAUSS_CHEBYSHEV_I;
    std::vector<double> nodes(n_grids), weights(n_grids);
    GaussChebyshevI_unit(n_grids, nodes.data(), weights.data());
    // transform from [-1,1] to [0, infinity]
    transform_GaussQuad_unit2x0inf(0.0, n_grids, nodes.data(), weights.data());
    for (size_t i = 0; i != n_grids; i++ )
    {
        freq_nodes[i] = nodes[i];
        freq_weights[i] = weights[i];
    }
}

void TFGrids::generate_GaussChebyshevII()
{
    grid_type = LIBRPA_TFGRID_GAUSS_CHEBYSHEV_II;
    std::vector<double> nodes(n_grids), weights(n_grids);
    GaussChebyshevII_unit(n_grids, nodes.data(), weights.data());
    // transform from [-1,1] to [0, infinity]
    transform_GaussQuad_unit2x0inf(0.0, n_grids, nodes.data(), weights.data());
    for (size_t i = 0; i != n_grids; i++ )
    {
        freq_nodes[i] = nodes[i];
        freq_weights[i] = weights[i];
    }
}

void TFGrids::generate_GaussLegendre()
{
    grid_type = LIBRPA_TFGRID_GAUSS_LEGENDRE;
    std::vector<double> nodes(n_grids), weights(n_grids);
    GaussLegendre_unit(n_grids, nodes.data(), weights.data());
    // transform from [-1,1] to [0, infinity]
    transform_GaussQuad_unit2x0inf(0.0, n_grids, nodes.data(), weights.data());
    for (size_t i = 0; i != n_grids; i++ )
    {
        freq_nodes[i] = nodes[i];
        freq_weights[i] = weights[i];
    }
}

void TFGrids::generate_SplitGaussLegendre(double emax)
{
    if (n_grids < 2)
        throw LIBRPA_RUNTIME_ERROR("grid size must be at least 2 for split-GL grids");
    if (n_grids % 2 != 0)
        throw LIBRPA_RUNTIME_ERROR("grid size must be even for split-GL grids");
    if (emax <= 0)
        throw LIBRPA_RUNTIME_ERROR("emax must be positive for split-GL grids");

    const size_t n_finite = n_grids / 2;
    const size_t n_tail_output = n_grids - n_finite + 1;

    std::vector<double> finite_nodes(n_finite), finite_weights(n_finite);
    GaussLegendre_unit(n_finite, finite_nodes.data(), finite_weights.data());
    transform_GaussQuad_unit2x0x1(0.0, emax, n_finite, finite_nodes.data(), finite_weights.data());
    for (size_t i = 0; i + 1 < n_finite; i++ )
    {
        freq_nodes[i] = finite_nodes[i];
        freq_weights[i] = finite_weights[i];
    }

    // Match FHI-aims tf_ini: gauleg1 writes omega(ntau:nomega), replacing the last finite GL point.
    std::vector<double> tail_nodes(n_tail_output), tail_weights(n_tail_output);
    GaussLegendre_unit(n_tail_output, tail_nodes.data(), tail_weights.data());
    for (size_t i = 0; i != n_tail_output; i++ )
    {
        const double denom = 1.0 - tail_nodes[i];
        const size_t out = n_finite - 1 + i;
        freq_nodes[out] = 2.0 * emax / denom;
        freq_weights[out] = tail_weights[i] * 2.0 * emax / (denom * denom);
    }
    grid_type = LIBRPA_TFGRID_SPLIT_GAUSS_LEGENDRE;
}

int TFGrids::get_time_index(const double &time) const
{
    if (!has_time_grids())
        throw LIBRPA_RUNTIME_ERROR("time grids not available");
    auto itr = std::find(time_nodes.cbegin(), time_nodes.cend(), time);
    if ( itr == time_nodes.cend() )
        throw LIBRPA_RUNTIME_ERROR("time not found");
    int i = std::distance(time_nodes.cbegin(), itr);
    return i;
}

int TFGrids::get_freq_index(const double &freq) const
{
    auto itr = std::find(freq_nodes.cbegin(), freq_nodes.cend(), freq);
    if ( itr == freq_nodes.cend() )
        throw LIBRPA_RUNTIME_ERROR("frequency not found");
    int i = std::distance(freq_nodes.cbegin(), itr);
    return i;
}

const std::pair<int, int> TFGrids::get_tf_index(const std::pair<double, double> &tf) const
{
    auto itime = get_time_index(tf.first);
    auto ifreq = get_freq_index(tf.second);
    return std::pair<int, int>{itime, ifreq};
}

double TFGrids::find_freq_weight(const double & freq) const
{
    return freq_weights[get_freq_index(freq)];
}

void TFGrids::write_cos_sin_trans_matrices(const std::string &filename) const
{
    std::ofstream ofs;
    const int n_grids_int = as_int(n_grids);
    std::vector<int> is_authentic(4, 0);
    const size_t gg_bytes_double = n_grids * n_grids * sizeof(double);

    std::vector<double> dummy(n_grids * n_grids, 0.0);

    if (n_grids > 0 && has_time_grids())
        std::fill(is_authentic.begin(), is_authentic.end(), 1);
    // ST freq->time matrix is not extracted now, so use dummy
    is_authentic[3] = 0;

    auto write_if_authentic = [&](const int authentic, const matrix &mat)
    {
        double *p = (authentic && n_grids > 0) ? mat.c : dummy.data();
        ofs.write((char *) p, gg_bytes_double);
    };

    ofs.open(filename, std::ios::out | std::ios::binary);
    ofs.write((char *) &n_grids_int, sizeof(int));
    ofs.write((char *) &is_authentic, 4 * sizeof(int));
    write_if_authentic(is_authentic[0], costrans_t2f);
    write_if_authentic(is_authentic[1], sintrans_t2f);
    write_if_authentic(is_authentic[2], costrans_f2t);
    write_if_authentic(is_authentic[3], sintrans_f2t);
    ofs.close();
}

}
