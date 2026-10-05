/*!
 @file: timefreq.h
 @brief Utilities related to quadrature on time/frequency domain
 */
#ifndef TIMEFREQ_H
#define TIMEFREQ_H

#include <string>
#include <vector>

#include "../math/complexmatrix.h"
#include "../math/matrix.h"
#include "librpa_enums.h"

namespace librpa_int {

//! Object to handle time/frequency grids for quadrature
/*!
 * Not necessay have a time grids, unless the space-time minimax grid is used.
 * @note Only pure imaginary grid method is implemented.
 * @note The time grids are always generated when available, since they are cheap
 */
class TFGrids
{
    public:
        static const string GRID_TYPES_NOTES[LIBRPA_TFGRID_COUNT];
        static const bool SUPPORT_TIME_GRIDS[LIBRPA_TFGRID_COUNT];
    private:
        //! Internal storage of grid type
        LibrpaTimeFreqGrid grid_type;
        //! whether to use time grids, i.e. space-time method
        bool _has_time_grids;
        size_t n_grids;
        std::vector<double> freq_nodes;
        std::vector<double> freq_weights;
        std::vector<double> time_nodes;
        std::vector<double> time_weights;
        /*! Cosine transformation matrix from (imaginary) time to frequency, i.e. gamma in Eq.4 of LiuP16
         The row-column convention also follows the equation, i.e.
         Each row corresponds to a certain frequency k index.
         */
        matrix costrans_t2f;
        /*! Inverse Cosine transform matrix.
         */
        matrix costrans_f2t;
        //! Sine transformation matrix from (imaginary) time to frequency, i.e. lambda in Eq.7 of LiuP16
        matrix sintrans_t2f;
        /*! Inverse Cosine transform matrix.
         */
        matrix sintrans_f2t;
        //! General Fourier transformation matrix. Not implemented, maybe needed to for non-minimax grid approach
        // ComplexMatrix fourier_t2f;
        // ComplexMatrix fourier_f2t;
        //! allocate the pointers of array, e.g. nodes and weights
        void set_freq();
        void set_time();
        //! delete the pointers
        void unset();
    public:
        LibrpaTimeFreqGrid get_grid_type() const { return grid_type; }
        TFGrids(): n_grids(0) {};
        TFGrids(const unsigned &N);
        // disable copy at present
        TFGrids(const TFGrids &tfg) {};
        void reset(const unsigned &N);
        void show() const;
        //! get the number of grid points
        size_t get_n_grids() const { return n_grids; }
        //! alias to get_n_grids
        size_t size() const { return n_grids; }
        const std::vector<double> get_freq_nodes() const { return freq_nodes; }
        const std::vector<double> get_freq_weights() const { return freq_weights; }
        const std::vector<double> get_time_nodes() const { return time_nodes; }
        const std::vector<double> get_time_weights() const { return time_weights; }
        // NOTE: use reference for return matrix?
        const matrix &get_costrans_t2f() const { return costrans_t2f; }
        const matrix &get_sintrans_t2f() const { return sintrans_t2f; }
        const matrix &get_costrans_f2t() const { return costrans_f2t; }
        const matrix &get_sintrans_f2t() const { return sintrans_f2t; }
        int get_time_index(const double &time) const;
        int get_freq_index(const double &freq) const;
        const std::pair<int, int> get_tf_index(const std::pair<double, double> &tf) const;
        //! obtain the integral weight from the frequency value
        // NOTE:(ZMY) attempt to use a map<double, double> to store,
        //      but will lead to a segfault in chi0tauR calculation, not knowing why
        double find_freq_weight(const double &freq) const;
        void write_cos_sin_trans_matrices(const std::string &filename) const;

        //! A wrapper around all grids generators
        double generate(LibrpaTimeFreqGrid gtype,
                        double emin = -1, double eintveral = -1,
                        double emax = -1, double tmin = -1, double tinterval = -1,
                        double regulation = 0.0);

        //! Dedicated export grid: rescale the minimax time quadrature to the
        //! single-response energy interval and set row 0 by chi(i0)=2*integral_0^infty chi(tau) d tau. Never use
        //! this grid for GW self-energy integration/inverse transforms.
        void generate_static_export(const TFGrids &source);
        //! One imported response frequency (including zero), for analytic
        //! dielectric head/wing evaluation only; not a GW quadrature grid.
        void generate_single_frequency(double omega);
        //! Generate the even-spaced frequency grid
        void generate_evenspaced(double emin, double interval);
        //! Generate the even-spaced time-frequency grid. @note Currently only for debug use
        void generate_evenspaced_tf(double emin, double eintv, double tmin, double tintv);
        //! Generate the minimax time-frequency grid
        double generate_minimax(double emin, double emax, double regulation = 0.0);
        //! Generate Gauss-Chebyshev quadrature of first kind on [0, infty)
        void generate_GaussChebyshevI();
        //! Generate Gauss-Chebyshev quadrature of second kind on [0, infty)
        void generate_GaussChebyshevII();
        void generate_GaussLegendre();
        void generate_SplitGaussLegendre(double emax);
        bool has_time_grids() const { return time_nodes.size() > 0; }
        ~TFGrids();
};

}
#endif
