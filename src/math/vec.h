#pragma once
#include "../utils/base_utility.h"
#include "lapack_connector.h"
#include <cstring>
#include <string>
#include <complex>
#include <cassert>
#include <cmath>
#include <vector>

namespace librpa_int {

template <typename T>
class vec
{
public:
    using type = T;
    using real_t = typename to_real<T>::type;
    using cplx_t = typename to_cplx<T>::type;
    const bool is_complex = is_complex_t<T>::value;

    int n;
    T* c;
    constexpr static const double EQUAL_THRES = DOUBLE_EQUAL_THRES;

    vec(): n(0), c(nullptr) { zero_out(); }
    vec(const int &n_in): n(n_in), c(nullptr) { if (n_in > 0) c = new T [n]; zero_out(); }
    vec(const int &n_in, const T * const valarr): n(n_in), c(nullptr)
    {
        if (n>0)
        {
            c = new T [n];
        }
        zero_out();
        // do not manually check out of bound
        for (int i = 0; i < size(); i++)
            c[i] = valarr[i];
    }
    vec(const std::vector<T> &v): n(v.size()), c(nullptr)
    {
        if (n>0)
        {
            c = new T [n];
        }
        for (int i = 0; i < size(); i++)
            c[i] = v[i];
    }
    vec(const vec<T> &v): n(v.n), c(nullptr)
    {
        if(n > 0)
        {
            c = new T[n];
            memcpy(c, v.c, n*sizeof(T));
        }
    }
    vec(vec<T> &&v) : n(v.n)
    {
        c = v.c;
        v.n = 0;
        v.c = nullptr;
    }
    ~vec() { n = 0; if (c) { delete [] c; c = nullptr;}}

    int size() const { return n; }

    void zero_out() { for (int i = 0; i < size(); i++) c[i] = 0; }

    void conj()
    {
        if (!this->is_complex) return;
        for (int i = 0; i < this->size(); i++)
            this->c[i] = get_conj(this->c[i]);
    };
    T &operator[](const int i) { return c[i]; }
    const T &operator[](const int i) const { return c[i]; }

    vec<T> & operator=(const vec<T> &v)
    {
        if (this == &v) return *this;
        resize(v.n);
        memcpy(c, v.c, n*sizeof(T));
        return *this;
    }

    vec<T> & operator=(vec<T> &&v)
    {
        if (this == &v) return *this;
        n = v.n;
        if(c) delete [] c;
        c = v.c;
        v.c = nullptr;
        return *this;
    }

    vec<T> operator-() const
    {
        vec<T> newv(*this);
        for (int i = 0; i < this->size(); i++)
            newv[i] = -this->c[i];
        return newv;
    }

    bool operator<(const vec<T> &v) const
    {
        return norm2(*this) < norm2(v);
    }

    bool operator>(const vec<T> &v) const
    {
        return norm2(*this) > norm2(v);
    }

    void resize(const int &n_new)
    {
        const int size_new = n_new;
        if (size_new > 0)
        {
            if (c)
            {
                if ( size_new != size() )
                {
                    delete [] c;
                    c = new T[size_new];
                }
            }
            else
                c = new T[size_new];
        }
        else
        {
            if(c) delete [] c;
            c = nullptr;
        }
        n = n_new;
        zero_out();
    }

    bool operator==(const vec<T> &v) const
    {
        return vec_equal(*this, v, vec<T>::EQUAL_THRES);
    }

    void operator+=(const T &cnum)
    {
        for (int i = 0; i < size(); i++)
            c[i] += cnum;
    }

    void operator+=(const vec<T> &v)
    {
        assert(n == v.n);
        for (int i = 0; i < size(); i++)
            c[i] += v.c[i];
    }

    void operator-=(const T &cnum)
    {
        for (int i = 0; i < size(); i++)
            c[i] -= cnum;
    }

    void operator-=(const vec<T> &v)
    {
        assert(n == v.n);
        for (int i = 0; i < size(); i++)
            c[i] -= v.c[i];
    }

    void operator*=(const T &cnum)
    {
        for (int i = 0; i < size(); i++)
            c[i] *= cnum;
    }

    void operator/=(const T &cnum)
    {
        assert(fabs(cnum) > 0);
        for (int i = 0; i < size(); i++)
            c[i] /= cnum;
    }
};

template <typename T>
vec<T> conj(const vec<T> &v)
{
    vec<T> conjv(v);
    conjv.conj();
    return conjv;
}

template <typename T>
vec<T> operator+(const vec<T> &v, const T &cnum)
{
    vec<T> sum = v;
    sum += cnum;
    return sum;
}

template <typename T>
vec<T> operator+(const T &cnum, const vec<T> &v)
{
    return v + cnum;
}

template <typename T>
vec<T> operator+(const vec<T> &v1, const vec<T> &v2)
{
    assert(v1.n == v2.n);
    vec<T> sum = v1;
    sum += v2;
    return sum;
}

template <typename T>
vec<T> operator-(const vec<T> &v1, const vec<T> &v2)
{
    assert(v1.n == v2.n);
    vec<T> vnew = v1;
    vnew -= v2;
    return vnew;
}

template <typename T>
vec<T> operator-(const vec<T> &v, const T &cnum)
{
    vec<T> mnew = v;
    mnew -= cnum;
    return mnew;
}

template <typename T>
vec<T> operator-(const T &cnum, const vec<T> &v)
{
    return - v + cnum;
}

template <typename T>
std::ostream & operator<<(std::ostream & os, const vec<T> &v)
{
    for (int i = 0; i < v.n - 1; i++)
        os << v[i] << " ";
    if (v.n > 0)
        os << v[v.n-1];
    return os;
}

template <typename T>
vec<T> operator*(const vec<T> &v, const T &cnum)
{
    vec<T> mnew = v;
    mnew *= cnum;
    return mnew;
}


template <typename T>
vec<T> operator*(const T &cnum, const vec<T> &v)
{
    return v * cnum;
}

template <typename T, typename TR = typename to_real<T>::type>
TR norm2(vec<T> v)
{
    return norm(v.c, v.n, 2);
}

template <typename T>
bool vec_equal(const vec<T> &v1, const vec<T> &v2, double thres)
{
    // NOTE: size is not necessarily equal
    if (v1.size() == 0 || v2.size() == 0)
        return false;
    int size = std::min(v1.size(), v2.size());
    for (int i = 0; i < size; i++)
        if (fabs(v1.c[i] - v2.c[i]) > thres) return false;
    return true;
}

template <typename T>
T dot(const vec<T> &v1, const vec<T> &v2)
{
    assert(v1.size() == v2.size());
    return LapackConnector::dot(v1.size(), v1.c, 1, v2.c, 1);
}

template <typename T>
std::string str(const vec<T> &v)
{
    std::string s;
    if (v.size() > 0) s = std::to_string(v[0]);
    for (int i = 1; i < v.size(); i++)
        s = s + " " + std::to_string(v[i]);
    return s;
}

template <typename T>
std::string str(const vec<std::complex<T>> &v)
{
    std::string s;
    if (v.size() > 0) s = "(" + std::to_string(v[0].real()) + "," + std::to_string(v[0].imag()) + ")";
    for (int i = 1; i < v.size(); i++)
        s = s + " (" + std::to_string(v[0].real()) + "," + std::to_string(v[0].imag()) + ")";
    return s;
}

}
