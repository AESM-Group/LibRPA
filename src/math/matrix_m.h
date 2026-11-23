/*
 * @file matrix_m.h
 * @brief implementing a matrix template class with support of column-major or row-major style
 */
#pragma once
#include <array>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iomanip>
#include <ios>
#include <limits>
#include <memory>
#include <ostream>
#include <random>
#include <utility>
#include <valarray>
#include <vector>

#include "../io/global_io.h"
#include "../utils/base_utility.h"
#include "../utils/error.h"
#include "lapack_connector.h"
#include "vec.h"

namespace librpa_int {

//! type alias for functors to map 2D array index to flatten array index
typedef std::function<int(const int& nr, const int& ir, const int& nc, const int& ic)> Indx_picker_2d;
typedef std::function<void(const int& indx, const int& nr, int& ir, const int& nc, int& ic)> Indx_folder_2d;
// major dependent flatted index of 2D matrix
//! row-major 2D index picker
inline int flatid_rm(const int &nr, const int &ir, const int &nc, const int &ic) { return ir*nc+ic; }
//! column-major 2D index picker
inline int flatid_cm(const int &nr, const int &ir, const int &nc, const int &ic) { return ic*nr+ir; }

// major dependent folding of 1D index to 2D index of matrix
//! row-major 1D index folder to 2D
inline void foldid_rm(const int& indx, const int &nr, int &ir, const int &nc, int &ic) { ir = indx / nc; ic = indx % nc; }
//! column-major 1D index folder to 2D
inline void foldid_cm(const int& indx, const int &nr, int &ir, const int &nc, int &ic) { ir = indx % nr; ic = indx / nr; }


enum MAJOR
{
    ROW = 0,
    COL = 1,
    AUTO = 2
};
const Indx_picker_2d Indx_pickers_2d[]
{
    flatid_rm,
    flatid_cm
};
const Indx_folder_2d Indx_folders_2d[]
{
    foldid_rm,
    foldid_cm
};


template <typename T>
void get_nr_nc_from_nested_vector(const std::vector<std::vector<T>> &nested_vec, int &nr, int &nc)
{
    nr = nested_vec.size();
    int nc_ = 0;
    for (const auto& v: nested_vec)
        nc_ = std::max(as_int(v.size()), nc_);
    nc = nc_;
}

template <typename T>
void expand_nested_vector_to_pointer(const std::vector<std::vector<T>> &nested_vector, const int &nr, const int &nc, T* c, bool row_major = true)
{
    int nc_, nr_;
    get_nr_nc_from_nested_vector(nested_vector, nr_, nc_);
    if (nr < nr_) nr_ = nr;
    if (nc < nc_) nc_ = nc;

    const Indx_picker_2d picker = Indx_pickers_2d[int(!row_major)];

    if (nr&&nc)
    {
        for (int ir = 0; ir < nr_; ir++)
        {
            const auto &ir_size = as_int(nested_vector[ir].size());
            for (int ic = 0; ic < std::min(nc_, ir_size); ic++)
                c[picker(nr, ir, nc, ic)] = nested_vector[ir][ic];
        }
    }
}


template <typename T>
class matrix_m
{
public:
    //! Flag of whether the instantialized matrix class is complex-typed
    static constexpr bool is_complex = is_complex_t<T>::value;
    static constexpr bool is_double = std::is_same<T, double>::value;
    using container_t = std::shared_ptr<std::valarray<T>>;
private:
    //! Major of matrix data storage, either row- or column-major
    MAJOR major_;
    int nr_;
    int nc_;
    size_t size_;
    //! The leading dimension of the matrix
    int ld_;
    int mrank_;
    //! Flag whether a dummy data of size 1 is allocated to avoid null pointer when size is 0
    bool data_dummy_;
    container_t data_;

public:
    // access to private variables
    inline int nr() const noexcept { return nr_; }
    inline int nc() const noexcept { return nc_; }
    // leading dimension
    inline int ld() const noexcept { return ld_; }
    inline MAJOR major() const noexcept { return major_; }
    inline size_t size() const noexcept { return size_; }
    inline int mrank() const noexcept { return mrank_; }
    inline bool is_data_dummy() const noexcept { return data_dummy_; }

    inline bool is_row_major() const noexcept { return major_ == MAJOR::ROW; }
    inline bool is_col_major() const noexcept { return major_ == MAJOR::COL; }

    // resizing
    matrix_m<T>& resize(int nrows_new, int ncols_new)
    {
        reset_dimensions_(nrows_new, ncols_new);
        // data not initialized
        if (data_ == nullptr)
        {
            if (size_ > 0)
            {
                data_ = std::make_shared<std::valarray<T>>(0, size_);
            }
            else
            {
                assert(data_dummy_);
                data_ = std::make_shared<std::valarray<T>>(0, 1); // dummy
            }
        }
        else
        {
            if (size_ > 0)
            {
                data_->resize(size_);
            }
            else
            {
                assert(data_dummy_);
                data_->resize(1); // dummy
            }
        }
        zero_out();
        return *this;
    }

    matrix_m<T>& resize(int nrows_new, int ncols_new, MAJOR major_new)
    {
        // Keep the original major if AUTO is specified; this is a safeguard.
        // In case that major should be kept, just drop the major argument.
        major_ = major_new == MAJOR::AUTO? major_: major_new;
        return this->resize(nrows_new, ncols_new);
    }

    // indexing
private:
    inline std::size_t index(const int &nr, const int &i, const int &nc, const int &j) const noexcept
    {
        return major_ == ROW? i * nc + j : j * nr + i;
    }
    inline void fold(const size_t& indx, const int &nr, int &ir, const int &nc, int &ic) const noexcept
    {
        if (major_ == ROW)
        {
            ir = indx / nc;
            ic = indx % nc;
        }
        else
        {
            ir = indx % nr;
            ic = indx / nr;
        };
    }

public:
    inline T &operator()(const int ir, const int ic) noexcept { return this->at(ir, ic); }
    inline const T &operator()(const int ir, const int ic) const noexcept { return this->at(ir, ic); }
    inline T &at(const int ir, const int ic) noexcept { return (*data_)[index(nr_, ir, nc_, ic)]; }
    inline const T &at(const int ir, const int ic) const noexcept { return (*data_)[index(nr_, ir, nc_, ic)]; }

    inline T* ptr() noexcept { return &(*this->data_)[0]; }
    inline const T* ptr() const noexcept { return &(*this->data_)[0]; }
    inline std::shared_ptr<std::valarray<T>> sptr() noexcept { return this->data_; }
    inline const std::shared_ptr<std::valarray<T>> sptr() const noexcept { return this->data_; }

private:
    void assign_value_(const T* const pv, MAJOR major_pv) noexcept
    {
        if (major_ == major_pv)
        {
            for (size_t i = 0; i < size_; i++) (*data_)[i] = pv[i];
        }
        else
        {
            const Indx_picker_2d pv_picker = Indx_pickers_2d[major_pv];
            for (int ir = 0; ir < nr_; ir++)
                for (int ic = 0; ic < nc_; ic++)
                    this->at(ir, ic) = pv[pv_picker(nr_, ir, nc_, ic)];
        }
    }

    void reset_dimensions_(int nrows, int ncols)
    {
        assert(nrows >= 0 && ncols >= 0);
        nr_ = nrows;
        nc_ = ncols;
        ld_ = (major_ == MAJOR::ROW ? nc_ : nr_);
        mrank_ = std::min(nr_, nc_);
        size_ = nr_ * nc_;
        data_dummy_ = size_ > 0 ? false : true;
    }

public:
    using type = T;
    using real_t = typename to_real<T>::type;
    using cplx_t = typename to_cplx<T>::type;

    // constructors
    matrix_m(): major_(MAJOR::ROW), nr_(0), nc_(0), size_(0), ld_(0), mrank_(0), data_dummy_(true), data_(nullptr) {}
    matrix_m(const int &nrows, const int &ncols, MAJOR major = MAJOR::ROW): major_(major), data_(nullptr)
    {
        resize(nrows, ncols);
    }
    matrix_m(const int &nrows, const int &ncols, const T * const parr, MAJOR major = MAJOR::ROW): major_(major), ld_(0), data_(nullptr)
    {
        resize(nrows, ncols);
        assign_value_(parr, major);
    }
    matrix_m(const int &nrows, const int &ncols, const T * const parr, MAJOR major_arr, MAJOR major): major_(major), nr_(nrows), nc_(ncols), ld_(0), data_(nullptr)
    {
        resize(nrows, ncols);
        assign_value_(parr, major_arr);
    }
    matrix_m(const int &nrows, const int &ncols, const std::shared_ptr<std::valarray<T>> &valarr, MAJOR major_valarr): major_(major_valarr), nr_(nrows), nc_(ncols), ld_(0), data_(valarr)
    {
        reset_dimensions_(nrows, ncols);
    }
    // constructor from a nested vector
    matrix_m(const std::vector<std::vector<T>> &nested_vector, MAJOR major = MAJOR::ROW): major_(major), ld_(0), data_(nullptr)
    {
        get_nr_nc_from_nested_vector(nested_vector, nr_, nc_);
        resize(nr_, nc_);
        expand_nested_vector_to_pointer(nested_vector, nr_, nc_, this->ptr(), is_row_major());
    }

    // copy constructor
    matrix_m(const matrix_m<T> &m) = default;

    matrix_m(matrix_m &&m) = default;

    // destructor
    ~matrix_m() {}

    inline void zero_out() noexcept { if (size_ > 0) *(this->data_) = 0; }

    inline void clear() noexcept { this->resize(0, 0); }

    inline void set_diag(const T& v) noexcept
    {
        for (int i = 0; i < mrank(); i++) this->at(i, i) = v;
    }

    inline void scale_row(int irow, const T &scale) noexcept
    {
        for (int ic = 0; ic < nc_; ic++) this->at(irow, ic) *= scale;
    }

    inline void scale_col(int icol, const T &scale) noexcept
    {
        for (int ir = 0; ir < nr_; ir++) this->at(ir, icol) *= scale;
    }

    //! Randomize the matrix elements with lower and upper bound and symmetry constraint
    matrix_m& randomize(const T &lb = 0, const T &ub = 1, bool symmetrize = false, bool hermitian = false) noexcept
    {
        static std::default_random_engine e(time(0));
        const auto nr = nr_;
        const auto nc = nc_;
        const auto mrank = mrank_;
        if constexpr (matrix_m<T>::is_complex)
        {
            std::uniform_real_distribution<real_t> dr(librpa_int::get_real(lb), librpa_int::get_real(ub));
            std::uniform_real_distribution<real_t> di(librpa_int::get_imag(lb), librpa_int::get_imag(ub));
            if (symmetrize)
            {
                for (int i = 0; i != mrank; i++)
                {
                    join_re_im((*this)(i, i), dr(e), di(e));
                    for (int j = i + 1; j < mrank; j++)
                    {
                        join_re_im((*this)(i, j), dr(e), di(e));
                        (*this)(j, i) = (*this)(i, j);
                    }
                }
            }
            else if (hermitian)
            {
                for (int i = 0; i != mrank; i++)
                {
                    join_re_im((*this)(i, i), dr(e), real_t(0.0));
                    for (int j = i + 1; j < mrank; j++)
                    {
                        join_re_im((*this)(i, j), dr(e), di(e));
                        (*this)(j, i) = librpa_int::get_conj((*this)(i, j));
                    }
                }
            }
            else
                for (size_t i = 0; i != size(); i++)
                    join_re_im(this->ptr()[i], dr(e), di(e));
        }
        else
        {
            std::uniform_real_distribution<real_t> dr(lb, ub);
            if (symmetrize)
            {
                if (major_ == ROW)
                {
                    // std::cout << "real symmetrize row" << std::endl; // debug
                    for (int i = 0; i != mrank; i++)
                    {
                        this->ptr()[nc * i + i] = dr(e);
                        for (int j = i + 1; j < mrank; j++)
                        {
                            this->ptr()[nc * i + j] = dr(e);
                            this->ptr()[nc * j + i] = this->ptr()[nc * i + j];
                        }
                    }
                }
                else
                {
                    // std::cout << "real symmetrize col" << std::endl; // debug
                    for (int j = 0; j < mrank; j++)
                    {
                        this->ptr()[nr * j + j] = dr(e);
                        for (int i = j + 1; i < mrank; i++)
                        {
                            this->ptr()[nr * j + i] = dr(e);
                            this->ptr()[nr * i + j] = this->ptr()[nr * j + i];
                        }
                    }
                }
            }
            else
            {
                for (size_t i = 0; i != size(); i++)
                    (*this->data_)[i] = dr(e);
            }
        }
        return *this;
    }

    inline matrix_m<T> copy() const
    {
        matrix_m<T> m_copy(nr_, nc_, ptr(), major_);
        return m_copy;
    }

    void reshape(int nrows_new, int ncols_new)
    {
        assert(size_ == as_size(nrows_new * ncols_new));
        reset_dimensions_(nrows_new, ncols_new);
    }

    // swap major
    void swap_to_row_major()
    {
        if (is_col_major())
        {
            int nr_old = nr_, nc_old = nc_;
            this->transpose();
            reset_dimensions_(nr_old, nc_old);
            major_ = MAJOR::ROW;
        }
    }
    void swap_to_col_major()
    {
        if (is_row_major())
        {
            int nr_old = nr_, nc_old = nc_;
            this->transpose();
            reset_dimensions_(nr_old, nc_old);
            major_ = MAJOR::COL;
        }
    }
    void swap_major(MAJOR major_to)
    {
        assert(major_to != MAJOR::AUTO);
        if (major_to == MAJOR::ROW) swap_to_row_major();
        else swap_to_col_major();
    }

    // assignment copy
    matrix_m<T> & operator=(const matrix_m<T> &m) = default;
    matrix_m<T> & operator=(matrix_m<T> &&m) = default;

    // operator overload
    inline matrix_m<T> & operator=(const T &cnum)
    {
        *(this->data_) = cnum;
        return *this;
    }

    // uniary operations
    inline matrix_m<T> operator-() const
    {
        auto m = this->copy();
        *(this->data_) = - *(this->data_);
        return m;
    }

    // binary operations
    template <typename T1>
    void operator+=(const matrix_m<T1> &m)
    {
        assert(this->size() == m.size() && this->major_ == m.major_);
        *(this->data_) += *(m.data_);
    }
    template <typename T1>
    void operator+=(const std::vector<T1> &v)
    {
        assert(nc_ == v.size());
        for (int i = 0; i < nr_; i++)
            for (int ic = 0; ic < nc_; ic++)
                this->at(i, ic) += v[ic];
    }
    template <typename T1>
    void operator+=(const T1 &cnum)
    {
        *(this->data_) += cnum;
    }

    template <typename T1>
    void operator-=(const matrix_m<T1> &m)
    {
        assert(size() == m.size() && major_ == m.major_);
        *(this->data_) -= *(m.data_);
    }
    template <typename T1>
    void operator-=(const std::vector<T1> &v)
    {
        assert(nc() == v.size());
        for (int i = 0; i < nr(); i++)
            for (int ic = 1; ic < nc(); ic++)
                this->at(i, ic) += v[ic];
    }
    template <typename T1>
    void operator-=(const T1 &cnum)
    {
        assert(this->size_ > 0);
        *(this->data_) -= cnum;
    }
    template <typename T1>
    void operator*=(const T1 &cnum)
    {
        *(this->data_) *= cnum;
    }
    template <typename T1>
    void operator/=(const T1 &cnum)
    {
        *(this->data_) /= cnum;
    }

    matrix_m<T>& conj()
    {
        if constexpr (matrix_m<T>::is_complex)
        {
            for (auto &z : *data_)
            {
                z = std::conj(z);
            }
        }
        return *this;
    };

    matrix_m<T> get_transpose(bool conjugate = false) const
    {
        matrix_m<T> m_trans(nc(), nr(), major_);
        for (int i = 0; i < nr(); i++)
            for (int j = 0; j < nc(); j++)
                m_trans(j, i) = this->at(i, j);
        if (conjugate) m_trans.conj();
        return m_trans;
    }

    matrix_m<T>& transpose(bool conjugate = false)
    {
        if (nr_ == nc_)
        {
            // handling within the memory of c pointer for square matrix
            int n = nr_;
            for (int i = 0; i != n; i++)
                for (int j = i + 1; j < n; j++)
                {
                    T temp = (*data_)[i * n + j];
                    (*data_)[i * n + j] = (*data_)[j * n + i];
                    (*data_)[j * n + i] = temp;
                }
        }
        else
        {
            // NOTE: may have memory issue for large matrix as extra memory is requested
            if (size() > 0)
            {
                std::valarray<T> a_new(size());
                for (int i = 0; i < nr_; i++)
                    for (int j = 0; j < nc_; j++)
                        a_new[index(nc_, j, nr_, i)] = this->at(i, j);
                *(data_) = a_new;
            }
        }
        reset_dimensions_(nc_, nr_);
        if (conjugate) conj();
        return *this;
    }

    matrix_m<cplx_t> to_complex() const
    {
        if constexpr (matrix_m<T>::is_complex) return this->copy();
        matrix_m<cplx_t> m(nr_, nc_, major_);
        for (size_t i = 0; i < size_; i++)
            (*m.sptr())[i] = (*data_)[i];
        return m;
    }

    matrix_m<real_t> get_real() const
    {
        if constexpr (matrix_m<T>::is_complex)
        {
            matrix_m<real_t> m(nr(), nc(), this->major_);
            for (size_t i = 0; i < size(); i++)
                (*m.sptr())[i] = librpa_int::get_real((*data_)[i]);
            return m;
        }
        else
        {
            return this->copy();
        }
    }

    matrix_m<real_t> get_imag() const
    {
        matrix_m<real_t> m(nr(), nc(), major_);
        if constexpr (matrix_m<T>::is_complex)
        {
            for (size_t i = 0; i < size(); i++)
                (*m.sptr())[i] = librpa_int::get_imag((*data_)[i]);
        }
        return m;
    }

    size_t count_absle(real_t thres) const
    {
        size_t count = 0;
        for (size_t i = 0; i < size(); i++)
            if (std::abs((*data_)[i]) <= thres) count++;
        return count;
    }

    size_t count_absge(real_t thres) const
    {
        size_t count = 0;
        for (size_t i = 0; i < size(); i++)
            if (std::abs((*data_)[i]) >= thres) count++;
        return count;
    }

    void _argmaxmin(int &ir, int &ic, bool max) const
    {
        real_t value = max? std::numeric_limits<real_t>::min() : std::numeric_limits<real_t>::max();
        size_t indx = 0;
        if constexpr (matrix_m<T>::is_complex)
        {
            for (size_t i = 0; i < size_; i++)
            {
                const real_t a = std::abs((*data_)[i]);
                if ((!max && a < value) || (max && a > value))
                {
                    indx = i;
                    value = a;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < size_; i++)
            {
                const auto &a = (*data_)[i];
                if ((!max && a < value) || (max && a > value))
                {
                    indx = i;
                    value = (*data_)[i];
                }
            }
        }
        fold(indx, nr(), ir, nc(), ic);
    }

    void argmin(int &ir, int &ic) const
    {
        _argmaxmin(ir, ic, false);
    }

    void argmax(int &ir, int &ic) const
    {
        _argmaxmin(ir, ic, true);
    }

    real_t min() const
    {
        // for complex type, compare the absolute value
        const std::function<real_t(T)> filter =
            matrix_m<T>::is_complex ? [](T a) { return std::abs(a); } : [](T a) { return librpa_int::get_real(a); };
        real_t value = std::numeric_limits<real_t>::max();
        for (size_t i = 0; i < size_; i++)
            if (filter((*data_)[i]) < value)
            {
                value = filter((*data_)[i]);
            }
        return value;
    }

    real_t max() const
    {
        // for complex type, compare the absolute value
        const std::function<real_t(T)> filter =
            matrix_m<T>::is_complex ? [](T a) { return std::abs(a); } : [](T a) { return librpa_int::get_real(a); };
        real_t value = std::numeric_limits<real_t>::min();
        for (size_t i = 0; i < size_; i++)
            if (filter((*data_)[i]) > value)
            {
                value = filter((*data_)[i]);
            }
        return value;
    }

    real_t absmin() const
    {
        real_t value = 0;
        for (size_t i = 0; i < size(); i++)
            value = std::min(value, std::abs((*data_)[i]));
        return value;
    }

    real_t absmax() const
    {
        real_t value = 0;
        for (size_t i = 0; i < size(); i++)
            value = std::max(value, std::abs((*data_)[i]));
        return value;
    }
};

//! typedef of double matrix
typedef matrix_m<double> Matd;
//! typedef of complex double matrix
typedef matrix_m<cplxdb> Matz;

template <typename T1, typename T2>
inline bool same_major(const matrix_m<T1> &m1, const matrix_m<T2> &m2)
{
    return m1.major() == m2.major();
}

template <typename T1, typename T2>
inline bool same_type(const matrix_m<T1> &m1, matrix_m<T2> &m2)
{
    return std::is_same<T1, T2>::value;
}

// copy matrix elements of type T1 src to matrix of a different type
template <typename T1, typename T2>
void copy(const matrix_m<T1> &src, matrix_m<T2> &dest)
{
    assert(src.size() == dest.size() && same_major(src, dest));
    *dest.sptr() = *src.sptr();
}

// copy matrix elements of src to matrix of the same type
template <typename T>
void copy(const matrix_m<T> &src, matrix_m<T> &dest)
{
    assert(src.size() == dest.size() && same_major(src, dest));
    memcpy(dest.ptr(), src.ptr(), src.size() * sizeof(T));
}

template <typename T1, typename T2>
matrix_m<T1> operator+(const matrix_m<T1> &m1, const matrix_m<T2> &m2)
{
    assert(m1.nc() == m2.nc() && m1.nr() == m2.nr() && same_major(m1, m2));
    auto sum = m1.copy();
    sum += m2;
    return sum;
}

template <typename T1, typename T2>
inline matrix_m<T1> operator+(const matrix_m<T1> &m, const std::vector<T2> &v)
{
    assert(m.nc() == v.size());
    auto mnew = m.copy();
    mnew += v;
    return mnew;
}

template <typename T, typename T2>
inline matrix_m<T> operator+(const matrix_m<T> &m, const T2 &cnum)
{
    auto sum = m.copy();
    sum += cnum;
    return sum;
}

template <typename T1, typename T2>
inline matrix_m<T2> operator+(const T1 &cnum, const matrix_m<T2> &m)
{
    return m + cnum;
}

template <typename T1, typename T2>
inline matrix_m<T1> operator-(const matrix_m<T1> &m1, const matrix_m<T2> &m2)
{
    assert(m1.nc() == m2.nc() && m1.nr() == m2.nr());
    auto mnew = m1.copy();
    mnew -= m2;
    return mnew;
}

template <typename T1, typename T2>
inline matrix_m<T1> operator-(const matrix_m<T1> &m, const std::vector<T2> &v)
{
    assert(m.nc() == v.size());
    auto mnew = m.copy();
    mnew -= v;
    return mnew;
}

//! memory-retaining matrix multiply
template <typename T>
void matmul(const matrix_m<T> &m1, const matrix_m<T> &m2, matrix_m<T> &prod)
{
    assert(m1.nc() == m2.nr());
    assert(m1.nr() == prod.nr());
    assert(m2.nc() == prod.nc());
    assert(m1.major() == m2.major() && m1.major() == prod.major());

    // Nothing needs to be done if any of the dimension is zero.
    if (m1.nr() == 0 || m1.nc() == 0 || m2.nc() == 0) return;

    if (m1.is_row_major())
    {
        LapackConnector::gemm('N', 'N', m1.nr(), m2.nc(), m1.nc(),
                              T(1.0), m1.ptr(), m1.nc(), m2.ptr(), m2.nc(), T(0.0), prod.ptr(), prod.nc());
    }
    else
    {
        LapackConnector::gemm_f('N', 'N', m1.nr(), m2.nc(), m1.nc(),
                                T(1.0), m1.ptr(), m1.nr(), m2.ptr(), m2.nr(), T(0.0), prod.ptr(), prod.nr());
    }
}

template <typename T>
inline matrix_m<T> operator*(const matrix_m<T> &m1, const matrix_m<T> &m2)
{
    assert(m1.nc() == m2.nr());
    assert(m1.major() == m2.major());

    auto major = m1.major();

    matrix_m<T> prod(m1.nr(), m2.nc(), major);
    matmul<T>(m1, m2, prod);
    return prod;
}

// a naive implementation for integer matrix, e.g. Spglib rotation matrices
template <>
inline matrix_m<int> operator*(const matrix_m<int> &m1, const matrix_m<int> &m2)
{
    assert(m1.nc() == m2.nr());
    matrix_m<int> prod(m1.nr(), m2.nc(), m1.major());
    for (int ir = 0; ir < m1.nr(); ir++)
        for (int ik = 0; ik < m1.nc(); ik++)
            for (int ic = 0; ic < m2.nc(); ic++)
                prod(ir, ic) += m1(ir, ik) * m2(ik, ic);
    return prod;
}

template <typename T1, typename T2>
inline matrix_m<T1> operator*(const matrix_m<T1> &m, const T2 &cnum)
{
    auto sum = m.copy();
    sum *= cnum;
    return sum;
}

template <typename T1, typename T2>
inline matrix_m<T2> operator*(const T1 &cnum, const matrix_m<T2> &m)
{
    return m * cnum;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const matrix_m<T> &m)
{
    os << str(m);
    return os;
}

//! invert matrix m and store in m_inv
template <typename T>
void inverse(const matrix_m<T> &m, matrix_m<T> &m_inv)
{
    assert(m.major() == m_inv.major());
    int lwork = m.nr();
    int info = 0;
    std::vector<T> work(lwork);
    int ipiv[std::min(m.nr(), m.nc())];
    m_inv.resize(m.nr(), m.nc());
    copy(m, m_inv);
    // debug
    // std::cout << m_inv.size() << " " << m_inv(0, 0) << " " << m_inv(m.nr-1, m.nc-1) << std::endl;
    if (m.is_row_major())
    {
        LapackConnector::getrf(m_inv.nr(), m_inv.nc(), m_inv.ptr(), m_inv.nr(), ipiv, info);
        LapackConnector::getri(m_inv.nr(), m_inv.ptr(), m_inv.nr(), ipiv, work, lwork, info);
    }
    else
    {
        LapackConnector::getrf_f(m_inv.nr(), m_inv.nc(), m_inv.ptr(), m_inv.nr(), ipiv, info);
        LapackConnector::getri_f(m_inv.nr(), m_inv.ptr(), m_inv.nr(), ipiv, work, lwork, info);
    }
    m_inv.reshape(m_inv.nc(), m_inv.nr());
}

//! get the transpose of matrix
template <typename T>
inline matrix_m<T> transpose(const matrix_m<T> &m, bool conjugate = false) noexcept
{
    return m.get_transpose(conjugate);
}

//! get the conjugate of matrix
template <typename T>
inline matrix_m<T> conj(const matrix_m<T> &m) noexcept
{
    return m.copy().conj();
}

//! compute the determinant of a matrix
template <typename T>
T get_determinant(const matrix_m<T> &m)
{
    matrix_m<T> m_copy = m.copy();
    int info = 0;
    int mrank = std::min(m_copy.nr(), m_copy.nc());
    std::vector<int> ipiv(mrank);
    // debug
    if (m_copy.is_row_major())
        LapackConnector::getrf(m_copy.nr(), m_copy.nc(), m_copy.ptr(), m_copy.nr(), ipiv.data(), info);
    else
        LapackConnector::getrf_f(m_copy.nr(), m_copy.nc(), m_copy.ptr(), m_copy.nr(), ipiv.data(), info);
    T det = 1;
    for (int i = 0; i < mrank; i++)
    {
        /* std::cout << i << " " << ipiv[i] << " " << m.c[i*m.nc+i] << " "; */
        det *= static_cast<T>(2*int(ipiv[i] == (i+1))-1) * m_copy(i, i);
    }
    return det;
}

template <typename T>
void eigsh(const matrix_m<std::complex<T>> &mat, std::vector<T> &w, matrix_m<std::complex<T>> &v)
{
    assert (mat.nr() == mat.nc());
    const char jobz = 'V';
    const char uplo = 'U';
    const int n = mat.nr();
    int nb;
    if (matrix_m<T>::is_double)
        nb = LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1);
    else
        nb = LapackConnector::ilaenv(1, "cheev", "VU", n, -1, -1, -1);

    // Copy matrix data to output
    v = mat.copy();

    int lwork = mat.nc() * (nb+1);
    int info = 0;
    w.resize(n);
    T wpow[n];
    T rwork[3*n-2];
    std::complex<T> work[lwork];
    if (v.is_row_major())
        LapackConnector::heev(jobz, uplo, n, v.ptr(), n, w.data(), work, lwork, rwork, info);
    else
        LapackConnector::heev_f(jobz, uplo, n, v.ptr(), n, w.data(), work, lwork, rwork, info);
}

template <typename T> matrix_m<std::complex<T>>
power_hemat(matrix_m<std::complex<T>> &mat,
            T power, bool keep_ev, bool filter_original,
            const T &threshold = -1.e5)
{
    using global::lib_printf;
    assert (mat.nr() == mat.nc());
    const char jobz = 'V';
    const char uplo = 'U';
    const int n = mat.nr();
    int nb;
    if (matrix_m<T>::is_double)
        nb = LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1);
    else
        nb = LapackConnector::ilaenv(1, "cheev", "VU", n, -1, -1, -1);

    int lwork = mat.nc() * (nb+1);
    int info = 0;
    std::vector<T> w(n), wpow(n), rwork(3*n-2);
    std::vector<std::complex<T>> work(lwork);
    if (mat.is_row_major())
        LapackConnector::heev(jobz, uplo, n, mat.ptr(), n, w.data(), work.data(), lwork, rwork.data(), info);
    else
        LapackConnector::heev_f(jobz, uplo, n, mat.ptr(), n, w.data(), work.data(), lwork, rwork.data(), info);
    bool is_int_power = fabs(power - int(power)) < 1e-8;

    for ( int i = 0; i != n; i++ )
    {
        if (w[i] < 0 && w[i] > threshold && !is_int_power)
            lib_printf(LIBRPA_VERBOSE_WARN, "Warning! kept negative eigenvalue with non-integer power: # %d ev = %f , pow = %f\n", i, w[i], power);
        if (fabs(w[i]) < 1e-10 && power < 0)
            lib_printf(LIBRPA_VERBOSE_WARN, "Warning! nearly-zero eigenvalue with negative power: # %d ev = %f , pow = %f\n", i, w[i], power);
        if (w[i] < threshold)
        {
            wpow[i] = 0;
            if (filter_original)
                w[i] = 0;
        }
        else
            wpow[i] = w[i];
        wpow[i] = pow(wpow[i], power);
    }

    auto evconj = transpose(mat, true);
    auto temp = evconj.copy();
    for (int i = 0; i != n; i++)
        temp.scale_row(i, wpow[i]);
    auto pmat = mat * temp;
    // recover the original matrix here
    // if eigenvectors are requested, return now
    if (keep_ev)
        return pmat;
    temp = mat.copy();
    for (int i = 0; i != n; i++)
        evconj.scale_row(i, w[i]);
    matmul(temp, evconj, mat);
    return pmat;
}

template <typename T>
void power_hemat_onsite(matrix_m<std::complex<T>> &mat,
                        const T &power, const T &threshold = -1.e5)
{
    using global::lib_printf;
    assert (mat.nr() == mat.nc());
    const char jobz = 'V';
    const char uplo = 'U';
    const int n = mat.nr();
    int nb;
    if (matrix_m<T>::is_double)
        nb = LapackConnector::ilaenv(1, "zheev", "VU", n, -1, -1, -1);
    else
        nb = LapackConnector::ilaenv(1, "cheev", "VU", n, -1, -1, -1);

    int lwork = mat.nc() * (nb+1);
    int info = 0;
    std::vector<T> w(n), wpow(n), rwork(3*n-2);
    std::vector<std::complex<T>> work(lwork);
    if (mat.is_row_major())
        LapackConnector::heev(jobz, uplo, n, mat.ptr(), n, w.data(), work.data(), lwork, rwork.data(), info);
    else
        LapackConnector::heev_f(jobz, uplo, n, mat.ptr(), n, w.data(), work.data(), lwork, rwork.data(), info);
    bool is_int_power = fabs(power - int(power)) < 1e-8;

    for ( int i = 0; i != n; i++ )
    {
        if (w[i] < 0 && w[i] > threshold && !is_int_power)
            lib_printf(LIBRPA_VERBOSE_WARN, "Warning! kept negative eigenvalue with non-integer power: # %d ev = %f , pow = %f\n", i, w[i], power);
        if (fabs(w[i]) < 1e-10 && power < 0)
            lib_printf(LIBRPA_VERBOSE_WARN, "Warning! nearly-zero eigenvalue with negative power: # %d ev = %f , pow = %f\n", i, w[i], power);
        if (w[i] < threshold)
        {
            wpow[i] = 0;
        }
        else
            wpow[i] = w[i];
        wpow[i] = pow(wpow[i], power);
    }

    auto evconj = transpose(mat, true);
    auto temp = evconj.copy();
    for (int i = 0; i != n; i++)
        temp.scale_row(i, wpow[i]);
    auto pmat = mat * temp;
    // parse back to the original matrix here
    memcpy(mat.ptr(), pmat.ptr(), mat.size() * sizeof(T) * 2);
}

template <typename T>
inline bool operator==(const matrix_m<T> &m1, const matrix_m<T> &m2) noexcept
{
    if (m1.nr() != m2.nr() || m1.nc() != m2.nc())
        return false;
    if (m1.sptr() == nullptr || m2.sptr() == nullptr)
        return false;
    if (m1.is_data_dummy() || m2.is_data_dummy()) return false;
    using real_t = typename to_real<T>::type;
    const bool is_double = std::is_same<T, double>::value;
    const real_t thres = is_double ? 1e-14 : 1e-6;
    auto diff = std::abs(*(m1.sptr()) - *(m2.sptr()));
    const auto size = m1.size();
    for (size_t i = 0; i < size; i++)
        if (librpa_int::get_real(diff[i]) > thres)
            return false;
    return true;
}

//! convert matrix into a string
template <typename T>
std::string str(const matrix_m<T> &m)
{
    std::string s = "";
    for (int i = 0; i < m.nr(); i++)
    {
        if (m.nc() > 0) s = s + std::to_string(m(i, 0));
        for (int j = 1; j < m.nc(); j++)
            s = s + " " + std::to_string(m(i, j));
        s = s + "\n";
    }
    return s;
}

//! convert matrix into a string
template <typename T>
std::string str(const matrix_m<std::complex<T>> &m)
{
    std::string s = "";
    for (int i = 0; i != m.nr(); i++)
    {
        if (m.nc() > 0)
            s = s + "(" + std::to_string(m(i, 0).real()) + "," + std::to_string(m(i, 0).imag()) + ")";
        for (int j = 1; j != m.nc(); j++)
            s = s + " (" + std::to_string(m(i, j).real()) + "," + std::to_string(m(i, j).imag()) + ")";
        s = s + "\n";
    }
    return s;
}

//! generate a random matrix
template <typename T>
inline matrix_m<T> random(int nr, int nc, const T &lb, const T &ub, MAJOR major = MAJOR::ROW)
{
    matrix_m<T> m(nr, nc, major);
    m.randomize(lb, ub);
    return m;
}

//! generate a random symmetric matrix
template <typename T>
inline matrix_m<T> random_sy(int n, const T &lb, const T &ub, MAJOR major = MAJOR::ROW)
{
    matrix_m<T> m(n, n, major);
    m.randomize(lb, ub, true, false);
    return m;
}

//! generate a random Hermitian matrix
template <typename T>
inline matrix_m<std::complex<T>> random_he(int n, const std::complex<T> &lb, const std::complex<T> &ub, MAJOR major = MAJOR::ROW)
{
    matrix_m<std::complex<T>> m(n, n, major);
    m.randomize(lb, ub, false, true);
    return m;
}

template <typename T>
inline matrix_m<T> random_orthogonal(int n, MAJOR major = MAJOR::ROW)
{
    int info;
    // always use column major here to ensure that the eigvec construction is correct
    auto mat = random_sy<T>(n, -1.0, 1.0, MAJOR::COL);
    const char *name = mat.is_double? "DSYTRD" : "SSYTRD";
    int nb = LapackConnector::ilaenv(1, name, "VU", n, -1, -1, -1);
    if (nb <= 1) nb = std::max(1, n);
    const int lwork = n * (nb + 2);

    T *w = new T[n];
    T *work = new T[lwork];

    LapackConnector::syev_f('V', 'U', n, mat.ptr(), mat.nr(), w, work, lwork, info);
    delete [] w;
    delete [] work;
    return matrix_m<T>(n, n, mat.ptr(), MAJOR::COL, major);
}

//! generate a random unitary matrix, implemented as using ?heev upon hermitian matrix
template <typename T>
inline matrix_m<T> random_unitary(int n, MAJOR major = MAJOR::ROW)
{
    typedef typename to_real<T>::type Treal;
    constexpr bool is_complex = is_complex_t<T>::value;

    if constexpr (is_complex)
    {
        int info;
        // always use column major here to ensure that the eigvec construction is correct
        auto mat = random_he<Treal>(n, {-1.0, -1.0}, {1.0, 1.0}, MAJOR::COL);
        const char *name = mat.is_double? "ZHETRD" : "CHETRD";
        // printf("%s\n", name);
        int nb = LapackConnector::ilaenv(1, name, "VU", n, -1, -1, -1);
        if (nb <= 1) nb = std::max(1, n);
        const int lwork = n * (nb + 1);

        Treal *w = new Treal[n];
        T *work = new T[lwork];
        Treal *rwork = new Treal[std::max(1, 3 * n - 2)];

        LapackConnector::heev_f('V', 'U', n, mat.ptr(), mat.nr(), w, work, lwork, rwork, info);
        delete [] rwork;
        delete [] w;
        delete [] work;
        return matrix_m<T>(n, n, mat.ptr(), MAJOR::COL, major);
    }
    else
        return random_orthogonal<T>(n, major);
}

//! generate a random Hermitian matrix with selected eigenvalues.
//! when the number of provided eigenvalues is smaller than the dimension,
//! padding_zero will be used as the missing eigenvalues
template <typename T>
inline matrix_m<std::complex<T>> random_he_selected_ev(int n, const std::vector<T> &evs, MAJOR major = MAJOR::ROW, const T& padding_zero = 1e-14)
{
    const int nvec = evs.size();
    if (nvec == 0) return matrix_m<std::complex<T>>{0, 0, major};

    auto eigvec = random_unitary<std::complex<T>>(n, major);
    auto eigvec_H = eigvec.get_transpose(true);
    for (int ie = 0; ie != n; ie++)
        eigvec_H.scale_row(ie, ie < nvec? evs[ie]: padding_zero);
    return eigvec * eigvec_H;
}

//! generate a random symmetric matrix with selected eigenvalues.
//! when the number of provided eigenvalues is smaller than the dimension,
//! padding_zero will be used as the missing eigenvalues
template <typename T>
inline matrix_m<T> random_sy_selected_ev(int n, const std::vector<T> &evs, MAJOR major = MAJOR::ROW, const T& padding_zero = 1e-14)
{
    const int nvec = evs.size();
    if (nvec == 0) return matrix_m<T>{0, 0, major};

    auto eigvec = random_orthogonal<T>(n, major);
    auto eigvec_T = eigvec.get_transpose();
    for (int ie = 0; ie != n; ie++)
        eigvec_T.scale_row(ie, ie < nvec? evs[ie]: padding_zero);
    return eigvec * eigvec_T;
}

//! print the matrix in matrix-market style
template <typename T, typename Treal = typename to_real<T>::type>
void print_matrix_mm(const matrix_m<T> &mat, std::ostream &os, const std::string &comment = "",
                     Treal threshold = 1e-15, bool row_first = true)
{
    using std::endl;
    using std::showpoint;
    using std::scientific;
    using std::setprecision;

    const int nr = mat.nr(), nc = mat.nc();
    size_t nnz = 0;
    for (size_t i = 0; i < mat.size(); i++)
    {
        if (std::abs((*mat.sptr())[i]) > threshold)
            nnz++;
    }
    os << "%%MatrixMarket matrix coordinate "
       << (is_complex<T>()? "complex" : "real") << " general" << endl
       << "%" << endl;
    if (!comment.empty())
        os << "% " << comment << endl << "%" << endl;
    os << nr << " " << nc << " " << nnz << endl;
    // NOTE: better to remove duplicate code?
    if (row_first)
    {
        for (int i = 0; i != nr; i++)
            for (int j = 0; j != nc; j++)
            {
                auto &v = mat(i, j);
                if (std::abs(v) > threshold)
                {
                    os << i + 1 << " " << j + 1 << " " << showpoint << scientific << setprecision(15) << get_real(v);
                    if (is_complex<T>())
                        os << " " << showpoint << scientific << setprecision(15) << get_imag(v);
                    os << endl;
                }
            }
    }
    else
    {
        for (int j = 0; j != nc; j++)
            for (int i = 0; i != nr; i++)
            {
                auto &v = mat(i, j);
                if (std::abs(v) > threshold)
                {
                    os << i + 1 << " " << j + 1 << " " << showpoint << scientific << setprecision(15) << get_real(v);
                    if (is_complex<T>())
                        os << " " << showpoint << scientific << setprecision(15) << get_imag(v);
                    os << endl;
                }
            }
    }
}

template <typename T, typename Treal = typename to_real<T>::type>
void print_whole_matrix(const char *desc, const matrix_m<T> &mat)
{
    using global::lib_printf;
    int nr = mat.nr();
    int nc = mat.nc();
    lib_printf("\n %s\n", desc);
    lib_printf("nr = %d, nc = %d\n", nr, nc);
    if(is_complex<T>())
    {
        for (int i = 0; i < nr; i++)
        {
            for (int j = 0; j < nc; j++)
                lib_printf("%10.6f,%9.6f ", mat.c[i * nc + j].real(), mat.c[i * nc + j].imag());
            lib_printf("\n");
        }
    }
    else
    {
        for (int i = 0; i < nr; i++)
        {
            for (int j = 0; j < nc; j++)
                lib_printf("%10.6f", mat.c[i * nc + j].real());
            lib_printf("\n");
        }
    }
}

template <typename T, typename Treal = typename to_real<T>::type>
void print_matrix_mm_file(const matrix_m<T> &mat, const std::string &fn,
                          const std::string& header_comment = "",
                          Treal threshold = 1e-15, bool row_first = true)
{
    ofstream fs;
    fs.open(fn);
    print_matrix_mm(mat, fs, header_comment, threshold, row_first);
    fs.close();
}

//! Write the matrix to file in ELSI CSC format
template <typename T, typename Treal = typename to_real<T>::type>
void write_matrix_elsi_csc(const matrix_m<T> &mat, const std::string &fn, Treal threshold = 1e-15)
{
    const int nr = mat.nr(), nc = mat.nc();
    const int step =  is_complex<T>()? 2 : 1;
    // ELSI CSC only support square matrices
    assert (nr == nc);

    size_t nnz = 0;
    for (size_t i = 0; i != mat.size(); i++)
    {
        if (std::abs(mat.ptr()[i]) > threshold)
            nnz++;
    }
    // cout << nnz << endl;
    std::vector<int64_t> colptr(nc+1);
    std::vector<int32_t> rowptr(nnz);
    std::vector<Treal> nnz_val(nnz * step);

    int64_t idx = 0;
    colptr[0] = 1;  // Indices begin from 1
    for (int i = 0; i < nc; i++)
    {
        int64_t nnz_this_col = 0;
        for (int j = 0; j < nr; j++)
        {
            if (std::abs(mat(j, i)) > threshold)
            {
                const T val = mat(j,i);
                // cout << val << endl;
                std::memcpy(nnz_val.data() + idx * step, &val, sizeof(T));
                rowptr[idx] = j + 1;  // Indices begin from 1
                nnz_this_col++;
                idx++;
            }
        }
        colptr[i+1] = colptr[i] + nnz_this_col;
    }

    // Ensure correct non-zero values counting are consistent in the two runs
    // cout << colptr[nc] << " " << nnz + 1 << endl;
    assert (colptr[nc] == int64_t(nnz + 1));

    int64_t header[16];
    header[2] = is_complex<T>()? 1 : 0;
    header[3] = nr;
    header[5] = nnz;

    std::ofstream wf;
    wf.open(fn, std::ios::out | std::ios::binary);
    // check open status
    if (!wf)
        throw LIBRPA_RUNTIME_ERROR("Fail to read " + fn);
    wf.write((char *) header, 16 * sizeof(int64_t));
    wf.write((char *) colptr.data(), nc * sizeof(int64_t));  // only write the first nc elements
    wf.write((char *) rowptr.data(), nnz * sizeof(int32_t));
    wf.write((char *) nnz_val.data(), nnz * sizeof(T));
    wf.close();
}

}
