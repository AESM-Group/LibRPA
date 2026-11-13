#ifndef UTILS_MATRIX_M_MPI
#define UTILS_MATRIX_M_MPI

#pragma once
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../interface/la_connector.h"

#include "../core/utils_atomic_basis_blacs.h"
#include "../utils/profiler.h"
#include "matrix_m.h"
#include "../interface/scalapack/scalapack_connector.h"
#ifdef LIBRPA_USE_LIBRI
#include <RI/global/Tensor.h>
#else
#include "../utils/libri_stub.h"
#endif

namespace librpa_int
{

template <typename T>
matrix_m<T> init_local_mat(const ArrayDesc &ad, MAJOR major)
{
    matrix_m<T> mat_lo(ad.m_loc(), ad.n_loc(), major);
    return mat_lo;
}

template <typename T>
matrix_m<T> get_local_mat(const matrix_m<T> &mat_go, const ArrayDesc &ad,
                          MAJOR major = MAJOR::AUTO)
{
    // assert the shape of global matrix conforms with the array descriptor
    assert(mat_go.nr() == ad.m() && mat_go.nc() == ad.n());

    if (major == MAJOR::AUTO) major = mat_go.major();
    const auto major_src = mat_go.major();
    matrix_m<T> mat_lo(ad.m_loc(), ad.n_loc(), major);
    const auto nr = mat_lo.nr();
    const auto nc = mat_lo.nc();

    if (major_src == MAJOR::ROW)
    {
        for (int i = 0; i != nr; i++)
        {
            const auto i_go = ad.indx_l2g_r(i);
            for (int j = 0; j != nc; j++)
            {
                const auto j_go = ad.indx_l2g_c(j);
                mat_lo(i, j) = mat_go(i_go, j_go);
            }
        }
    }
    else
    {
        for (int j = 0; j != nc; j++)
        {
            const auto j_go = ad.indx_l2g_c(j);
            for (int i = 0; i != nr; i++)
            {
                const auto i_go = ad.indx_l2g_r(i);
                mat_lo(i, j) = mat_go(i_go, j_go);
            }
        }
    }
    return mat_lo;
}

template <typename T>
matrix_m<T> get_local_mat(const T *pv, MAJOR major_pv, const ArrayDesc &ad, MAJOR major)
{
    const int nr = ad.m(), nc = ad.n();
    matrix_m<T> mat_lo(ad.m_loc(), ad.n_loc(), major);
    const auto picker = Indx_pickers_2d[major_pv];
    for (int i = 0; i != nr; i++)
    {
        auto i_lo = ad.indx_g2l_r(i);
        if (i_lo < 0) continue;
        for (int j = 0; j != nc; j++)
        {
            auto j_lo = ad.indx_g2l_c(j);
            if (j_lo < 0) continue;
            mat_lo(i_lo, j_lo) = pv[picker(nr, i, nc, j)];
        }
    }
    if (mat_lo.size() == 0)
        mat_lo.resize(1, 1);
    return mat_lo;
}

template <typename Tdst, typename Tsrc>
void collect_block_from_IJ_storage(matrix_m<Tdst> &mat_lo, const ArrayDesc &ad,
                                   const AtomicBasis &atbasis_row,
                                   const AtomicBasis &atbasis_col, const int &I,
                                   const int &J, Tdst alpha, const Tsrc *pvIJ, MAJOR major_pv)
{
    // assert(mat_lo.nr() == ad.m_loc() && mat_lo.nc() == ad.n_loc());
    assert(as_size(ad.m()) == atbasis_row.nb_total && as_size(ad.n()) == atbasis_col.nb_total);
    const int row_start_id = atbasis_row.get_part_range()[I];
    const int col_start_id = atbasis_col.get_part_range()[J];
    const int row_nb = atbasis_row.get_atom_nb(I);
    const int col_nb = atbasis_col.get_atom_nb(J);
    const auto picker = Indx_pickers_2d[major_pv];
    // cout << str(mat_lo);
    for (int iI = 0; iI != row_nb; iI++)
    {
        int ilo = ad.indx_g2l_r(row_start_id + iI);
        // librpa_int::global::lib_printf("row igo %d ilo %d\n", row_start_id + iI, ilo);
        if (ilo < 0) continue;
        for (int jJ = 0; jJ != col_nb; jJ++)
        {
            int jlo = ad.indx_g2l_c(col_start_id + jJ);
            // librpa_int::global::lib_printf("col jgo %d jlo %d\n", col_start_id + jJ, jlo);
            if (jlo < 0) continue;
            // cout << "pickerID " << picker(row_nb, iI, col_nb, jJ) << " " << pvIJ[picker(row_nb, iI, col_nb, jJ)] << " matloc befor " << mat_lo(ilo, jlo);
            mat_lo(ilo, jlo) += alpha * pvIJ[picker(row_nb, iI, col_nb, jJ)];
            // cout << " after " << mat_lo(ilo, jlo) << endl;
        }
    }
}

template <typename Tdst, typename Tsrc, typename TA, typename TC, typename TAC = std::pair<TA, TC>>
void collect_block_from_IJ_storage_tensor(matrix_m<Tdst> &mat_lo, const ArrayDesc &ad,
                                          const AtomicBasis &atbasis_row,
                                          const AtomicBasis &atbasis_col, const TC &cell,
                                          Tdst alpha,
                                          const std::map<TA, std::map<TAC, RI::Tensor<Tsrc>>> &TMAP)
{
    // assert(mat_lo.nr() == ad.m_loc() && mat_lo.nc() == ad.n_loc());
    assert(as_size(ad.m()) == atbasis_row.nb_total && as_size(ad.n()) == atbasis_col.nb_total);

    matrix_m<Tdst> tmp_loc(mat_lo.nr(), mat_lo.nc(), MAJOR::ROW);
    size_t cp_size = ad.n_loc() * sizeof(Tdst);

    omp_lock_t mat_lock;
    omp_init_lock(&mat_lock);

    #pragma omp parallel for
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
    {
        int I_loc, J_loc, i_ab, j_ab;
        int i_gl = ad.indx_l2g_r(ilo);
        atbasis_row.get_local_index(i_gl, I_loc, i_ab);
        std::vector<Tdst> tmp_loc_row(ad.n_loc());
        for (int jlo = 0; jlo != ad.n_loc(); jlo++)
        {
            int j_gl = ad.indx_l2g_c(jlo);
            atbasis_col.get_local_index(j_gl, J_loc, j_ab);
            //Tdst temp;
            // librpa_int::global::lib_printf("i_gl I_loc i_ab %d %d %d j_gl J_loc j_ab %d %d %d\n", i_gl, I_loc, i_ab, j_gl, J_loc, j_ab);
            tmp_loc_row[jlo] = TMAP.at(I_loc).at({J_loc, cell})(i_ab, j_ab);
        }
        Tdst *row_ptr = tmp_loc.ptr() + ilo * ad.n_loc();
        omp_set_lock(&mat_lock);
        if (cp_size > 0) memcpy(row_ptr, tmp_loc_row.data(), cp_size);
        omp_unset_lock(&mat_lock);
    }
    #pragma omp barrier
    omp_destroy_lock(&mat_lock);
    if (mat_lo.is_col_major())
    {
        tmp_loc.swap_to_col_major();
    }
    mat_lo = tmp_loc;
}

// Assume the matrix_m in TMAP are row major
// TODO: merge it with collect_block_from_IJ_storage_tensor_transform
template <typename Tdst, typename Tsrc, typename TA, typename TC>
void collect_block_from_IJ_storage_matrix_transform(
    matrix_m<Tdst> &mat_lo,
    const ArrayDesc &ad,
    const AtomicBasis &atbasis_row,
    const AtomicBasis &atbasis_col,
    const std::function<Tdst(const TA, const TA, const TC &)> &transform,
    const std::map<TA, std::map<TA, std::map<TC, matrix_m<Tsrc>>>> &TMAP)
{
    // assert(mat_lo.nr() == ad.m_loc() && mat_lo.nc() == ad.n_loc());
    assert(as_size(ad.m()) == atbasis_row.nb_total && as_size(ad.n()) == atbasis_col.nb_total );
    if (TMAP.size() == 0) return;

    matrix_m<Tdst> tmp_loc(mat_lo.nr(),mat_lo.nc(), MAJOR::ROW);
    size_t cp_size= ad.n_loc()*sizeof(Tdst);

    omp_lock_t mat_lock;
    omp_init_lock(&mat_lock);

    #pragma omp parallel for
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
    {
        int I_loc, J_loc, i_ab, j_ab;
        int i_gl = ad.indx_l2g_r(ilo);
        atbasis_row.get_local_index(i_gl, I_loc, i_ab);
        std::vector<Tdst> tmp_loc_row(ad.n_loc(), 0);
        for (int jlo = 0; jlo != ad.n_loc(); jlo++)
        {
            int j_gl = ad.indx_l2g_c(jlo);
            atbasis_col.get_local_index(j_gl, J_loc, j_ab);
            auto it_I = TMAP.find(I_loc);
            if (it_I != TMAP.cend())
            {
                const auto &JMAP = it_I->second;
                auto it_J = JMAP.find(J_loc);
                if (it_J != JMAP.cend())
                {
                    for (const auto &cell_mat: it_J->second)
                    {
                        const auto &cell = cell_mat.first;
                        const auto &mat = cell_mat.second;
                        tmp_loc_row[jlo] += mat(i_ab, j_ab) * transform(I_loc, J_loc, cell);
                    }
                }
            }
        }
        Tdst *row_ptr=tmp_loc.ptr() + ilo* ad.n_loc();
        omp_set_lock(&mat_lock);
        if (cp_size > 0) memcpy(row_ptr, tmp_loc_row.data(), cp_size);
        omp_unset_lock(&mat_lock);
    }
    #pragma omp barrier
    omp_destroy_lock(&mat_lock);
    if(mat_lo.is_col_major())
    {
        tmp_loc.swap_to_col_major();
    }
    mat_lo = tmp_loc;
}

// TMAP tensor is always row major
template <typename Tdst, typename Tsrc, typename TA, typename TAC>
void collect_block_from_IJ_storage_tensor_transform(
    matrix_m<Tdst> &mat_lo, const ArrayDesc &ad, const AtomicBasis &atbasis_row,
    const AtomicBasis &atbasis_col, const std::function<Tdst(const TA &, const TAC &)> &transform,
    const std::map<TA, std::map<TAC, RI::Tensor<Tsrc>>> &TMAP)
{
    // assert(mat_lo.nr() == ad.m_loc() && mat_lo.nc() == ad.n_loc());
    assert(as_size(ad.m()) == atbasis_row.nb_total && as_size(ad.n()) == atbasis_col.nb_total);

    matrix_m<Tdst> tmp_loc(mat_lo.nr(), mat_lo.nc(), MAJOR::ROW);
    size_t cp_size = ad.n_loc() * sizeof(Tdst);

    omp_lock_t mat_lock;
    omp_init_lock(&mat_lock);

    #pragma omp parallel for
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
    {
        int I_loc, J_loc, i_ab, j_ab;
        int i_gl = ad.indx_l2g_r(ilo);
        atbasis_row.get_local_index(i_gl, I_loc, i_ab);
        std::vector<Tdst> tmp_loc_row(ad.n_loc(), 0);
        for (int jlo = 0; jlo != ad.n_loc(); jlo++)
        {
            int j_gl = ad.indx_l2g_c(jlo);
            atbasis_col.get_local_index(j_gl, J_loc, j_ab);
            if (TMAP.count(I_loc) > 0)
            {
                for (const auto &acell_mat: TMAP.at(I_loc))
                {
                    const auto &acell = acell_mat.first;
                    const auto &mat = acell_mat.second;
                    if (acell.first != J_loc)
                    {
                        continue;
                    }
                    tmp_loc_row[jlo] += mat(i_ab, j_ab) * transform(I_loc, acell);
                }
            }
        }
        Tdst *row_ptr = tmp_loc.ptr() + ilo * ad.n_loc();
        omp_set_lock(&mat_lock);
        if (cp_size > 0) memcpy(row_ptr, tmp_loc_row.data(), cp_size);
        omp_unset_lock(&mat_lock);
    }
    #pragma omp barrier
    omp_destroy_lock(&mat_lock);
    if (mat_lo.is_col_major())
    {
        tmp_loc.swap_to_col_major();
    }
    mat_lo = tmp_loc;
}

// Used for tranformation from Cs[I][{J, R}](i,j,mu)
// to 2d-block Cs[I][{J,k}](i,j)
template <typename Tdst, typename Tsrc, typename TA, typename TAC>
void collect_block_from_IJ_storage_tensor_transform_triple(
    matrix_m<Tdst> &mat_lo, const ArrayDesc &ad, const AtomicBasis &atbasis_row,
    const AtomicBasis &atbasis_col, const std::function<Tdst(const TA &, const TAC &)> &transform,
    const std::map<TA, std::map<TAC, RI::Tensor<Tsrc>>> &TMAP, const TA &Mu, const int &mu_local)
{
    // assert(mat_lo.nr() == ad.m_loc() && mat_lo.nc() == ad.n_loc());
    assert(as_size(ad.m()) == atbasis_row.nb_total && as_size(ad.n()) == atbasis_col.nb_total);

    matrix_m<Tdst> tmp_loc(mat_lo.nr(), mat_lo.nc(), MAJOR::ROW);
    size_t cp_size = ad.n_loc() * sizeof(Tdst);

    omp_lock_t mat_lock;
    omp_init_lock(&mat_lock);

#pragma omp parallel for
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
    {
        int I_loc, J_loc, i_ab, j_ab;
        int i_gl = ad.indx_l2g_r(ilo);
        atbasis_row.get_local_index(i_gl, I_loc, i_ab);
        std::vector<Tdst> tmp_loc_row(ad.n_loc(), 0);
        for (int jlo = 0; jlo != ad.n_loc(); jlo++)
        {
            int j_gl = ad.indx_l2g_c(jlo);
            atbasis_col.get_local_index(j_gl, J_loc, j_ab);
            if (TMAP.count(I_loc) > 0 && I_loc == Mu)
            {
                for (const auto &acell_mat : TMAP.at(I_loc))
                {
                    const auto &acell = acell_mat.first;
                    const auto &mat = acell_mat.second;
                    if (acell.first != J_loc)
                    {
                        continue;
                    }
                    tmp_loc_row[jlo] += mat(mu_local, i_ab, j_ab) * transform(Mu, acell);
                }
            }
        }
        Tdst *row_ptr = tmp_loc.ptr() + ilo * ad.n_loc();
        omp_set_lock(&mat_lock);
        if (cp_size > 0) memcpy(row_ptr, tmp_loc_row.data(), cp_size);
        omp_unset_lock(&mat_lock);
    }
#pragma omp barrier
    omp_destroy_lock(&mat_lock);
    if (mat_lo.is_col_major())
    {
        tmp_loc.swap_to_col_major();
    }
    mat_lo = tmp_loc;
}

//! collect 2D block from a IJ-pair storage exploiting its symmetric/Hermitian property
template <typename Tdst, typename Tsrc>
void collect_block_from_IJ_storage_syhe(matrix_m<Tdst> &mat_lo, const ArrayDesc &ad,
                                        const AtomicBasis &atbasis, const int &I, const int &J,
                                        bool conjugate, Tdst alpha, const Tsrc *pvIJ,
                                        MAJOR major_pv)
{
    // assert(mat_lo.nr() == ad.m_loc() && mat_lo.nc() == ad.n_loc());
    assert(as_size(ad.m()) == atbasis.nb_total && as_size(ad.n()) == atbasis.nb_total);
    // blocks on diagonal is trivial
    if (I == J)
    {
        collect_block_from_IJ_storage(mat_lo, ad, atbasis, atbasis, I, J, alpha, pvIJ, major_pv);
        return;
    }
    const std::function<Tdst(Tdst)> filter =
        conjugate ? [](Tdst a) { return get_conj(a); } : [](Tdst a) { return a; };
    const int I_nb = atbasis.get_atom_nb(I);
    const int J_nb = atbasis.get_atom_nb(J);
    const auto picker = Indx_pickers_2d[major_pv];
    int I_loc, J_loc, i_ab, j_ab;
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
        for (int jlo = 0; jlo != ad.n_loc(); jlo++)
        {
            int i_gl = ad.indx_l2g_r(ilo);
            int j_gl = ad.indx_l2g_c(jlo);
            atbasis.get_local_index(i_gl, I_loc, i_ab);
            atbasis.get_local_index(j_gl, J_loc, j_ab);
            Tdst temp;
            // librpa_int::global::lib_printf("i_gl I_loc i_ab %d %d %d j_gl J_loc j_ab %d %d %d\n", i_gl, I_loc, i_ab, j_gl, J_loc, j_ab);
            if ((I_loc == I) && (J_loc == J))
            {
                // in the same block
                temp = pvIJ[picker(I_nb, i_ab, J_nb, j_ab)];
                // cout << "same block pickerID " << picker(I_nb, i_ab, J_nb, j_ab) << " " << temp << " matloc befor " << mat_lo(ilo, jlo);
            }
            else if ((I_loc == J) && (J_loc == I))
            {
                // in the opposite block
                temp = filter(pvIJ[picker(I_nb, j_ab, J_nb, i_ab)]);
                // cout << "opposite block pickerID " << picker(I_nb, j_ab, J_nb, i_ab) << " " << temp << " matloc befor " << mat_lo(ilo, jlo);
            }
            else
                continue;
            mat_lo(ilo, jlo) += alpha * temp;
            // cout << " after " << mat_lo(ilo, jlo) << endl;
        }
}

template <typename Tdst, typename TA, typename TC, typename TAC = std::pair<TA, TC>>
void collect_block_from_ALL_IJ_Tensor(matrix_m<Tdst> &mat_lo, const ArrayDesc &ad,
                                      const AtomicBasis &atbasis, const TC &cell, bool conjugate,
                                      Tdst alpha,
                                      const std::map<TA, std::map<TAC, RI::Tensor<Tdst>>> &TMAP,
                                      MAJOR major_pv)
{
    // assert(mat_lo.nr() == ad.m_loc() && mat_lo.nc() == ad.n_loc());
    assert(as_size(ad.m()) == atbasis.nb_total && as_size(ad.n()) == atbasis.nb_total);
    matrix_m<Tdst> tmp_loc(mat_lo.nr(), mat_lo.nc(), MAJOR::ROW);
    size_t cp_size = ad.n_loc() * sizeof(Tdst);

// #ifdef LIBRPA_DEBUG
//     librpa_int::global::ofs_myid << "cp_size: " << cp_size << std::endl;
//     librpa_int::global::ofs_myid << "Available TMAP keys: ";
//     print_keys(librpa_int::global::ofs_myid, TMAP);
//     librpa_int::global::ofs_myid << std::endl;
//     librpa_int::global::ofs_myid << "mat_lo dims: " << mat_lo.nr() << " " << mat_lo.nc() <<
//     std::endl; librpa_int::global::ofs_myid << "ad_loc dims: " << ad.m_loc() << " " <<
//     ad.n_loc() << std::endl;
// #endif

    omp_lock_t mat_lock;
    omp_init_lock(&mat_lock);

    #pragma omp parallel for
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
    {
        int I_loc, J_loc, i_ab, j_ab;
        int i_gl = ad.indx_l2g_r(ilo);
        atbasis.get_local_index(i_gl, I_loc, i_ab);
        std::vector<Tdst> tmp_loc_row(ad.n_loc());
        for (int jlo = 0; jlo != ad.n_loc(); jlo++)
        {

            int j_gl = ad.indx_l2g_c(jlo);

            atbasis.get_local_index(j_gl, J_loc, j_ab);
            //Tdst temp;
            // librpa_int::global::lib_printf("i_gl I_loc i_ab %d %d %d j_gl J_loc j_ab %d %d %d\n", i_gl, I_loc, i_ab, j_gl, J_loc, j_ab);
            if(I_loc<=J_loc)
            {
                tmp_loc_row[jlo]= TMAP.at(I_loc).at({J_loc, cell})(i_ab,j_ab);
            }
            else
            {
                Tdst tmp_ele;
                tmp_ele = TMAP.at(J_loc).at({I_loc, cell})(j_ab, i_ab);
                if (conjugate) tmp_ele = get_conj(tmp_ele);
                tmp_loc_row[jlo] = tmp_ele;
            }
        }
        Tdst *row_ptr = tmp_loc.ptr() + ilo * ad.n_loc();
        omp_set_lock(&mat_lock);
        if (cp_size > 0) memcpy(row_ptr, tmp_loc_row.data(), cp_size);
        omp_unset_lock(&mat_lock);
    }
    #pragma omp barrier
    omp_destroy_lock(&mat_lock);
    if (mat_lo.is_col_major())
    {
        tmp_loc.swap_to_col_major();
    }
    mat_lo=tmp_loc;
}

template <typename Tdst, typename TA, typename TC, typename TAC = std::pair<TA, TC>>
void collect_block_from_ALL_IJ_Tensor_sparse_zero_missing(
    matrix_m<Tdst> &mat_lo, const ArrayDesc &ad, const AtomicBasis &atbasis, const TC &cell,
    bool conjugate, Tdst alpha,
    const std::map<TA, std::map<TAC, RI::Tensor<Tdst>>> &TMAP,
    MAJOR major_pv)
{
    (void)major_pv;
    assert(as_size(ad.m()) == atbasis.nb_total && as_size(ad.n()) == atbasis.nb_total);
    matrix_m<Tdst> tmp_loc(mat_lo.nr(), mat_lo.nc(), MAJOR::ROW);
    const size_t cp_size = ad.n_loc() * sizeof(Tdst);

    omp_lock_t mat_lock;
    omp_init_lock(&mat_lock);

    #pragma omp parallel for
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
    {
        int I_loc, J_loc, i_ab, j_ab;
        const int i_gl = ad.indx_l2g_r(ilo);
        atbasis.get_local_index(i_gl, I_loc, i_ab);
        std::vector<Tdst> tmp_loc_row(ad.n_loc(), Tdst(0));
        for (int jlo = 0; jlo != ad.n_loc(); jlo++)
        {
            const int j_gl = ad.indx_l2g_c(jlo);
            atbasis.get_local_index(j_gl, J_loc, j_ab);

            const int I_key = std::min(I_loc, J_loc);
            const int J_key = std::max(I_loc, J_loc);
            auto it_I = TMAP.find(I_key);
            if (it_I == TMAP.end()) continue;
            auto it_JR = it_I->second.find({J_key, cell});
            if (it_JR == it_I->second.end()) continue;

            if (I_loc <= J_loc)
            {
                tmp_loc_row[jlo] = alpha * it_JR->second(i_ab, j_ab);
            }
            else
            {
                auto tmp_ele = it_JR->second(j_ab, i_ab);
                if (conjugate) tmp_ele = get_conj(tmp_ele);
                tmp_loc_row[jlo] = alpha * tmp_ele;
            }
        }
        Tdst *row_ptr = tmp_loc.ptr() + ilo * ad.n_loc();
        omp_set_lock(&mat_lock);
        if (cp_size > 0) memcpy(row_ptr, tmp_loc_row.data(), cp_size);
        omp_unset_lock(&mat_lock);
    }
    #pragma omp barrier
    omp_destroy_lock(&mat_lock);
    if (mat_lo.is_col_major())
    {
        tmp_loc.swap_to_col_major();
    }
    mat_lo = tmp_loc;
}

template <typename Tdst, typename TA, typename TC, typename TAC = std::pair<TA, TC>>
void collect_block_from_ordered_IJ_Tensor_sparse_zero_missing(
    matrix_m<Tdst> &mat_lo, const ArrayDesc &ad, const AtomicBasis &atbasis, const TC &cell,
    Tdst alpha, const std::map<TA, std::map<TAC, RI::Tensor<Tdst>>> &TMAP)
{
    assert(as_size(ad.m()) == atbasis.nb_total && as_size(ad.n()) == atbasis.nb_total);
    matrix_m<Tdst> tmp_loc(mat_lo.nr(), mat_lo.nc(), MAJOR::ROW);
    const size_t cp_size = ad.n_loc() * sizeof(Tdst);

    omp_lock_t mat_lock;
    omp_init_lock(&mat_lock);

    #pragma omp parallel for
    for (int ilo = 0; ilo != ad.m_loc(); ilo++)
    {
        int I_loc, J_loc, i_ab, j_ab;
        const int i_gl = ad.indx_l2g_r(ilo);
        atbasis.get_local_index(i_gl, I_loc, i_ab);
        std::vector<Tdst> tmp_loc_row(ad.n_loc(), Tdst(0));
        const auto I_key = static_cast<TA>(I_loc);
        auto it_I = TMAP.find(I_key);
        if (it_I != TMAP.end())
        {
            for (int jlo = 0; jlo != ad.n_loc(); jlo++)
            {
                const int j_gl = ad.indx_l2g_c(jlo);
                atbasis.get_local_index(j_gl, J_loc, j_ab);
                auto it_JR = it_I->second.find({static_cast<TA>(J_loc), cell});
                if (it_JR == it_I->second.end()) continue;
                tmp_loc_row[jlo] = alpha * it_JR->second(i_ab, j_ab);
            }
        }
        Tdst *row_ptr = tmp_loc.ptr() + ilo * ad.n_loc();
        omp_set_lock(&mat_lock);
        if (cp_size > 0) memcpy(row_ptr, tmp_loc_row.data(), cp_size);
        omp_unset_lock(&mat_lock);
    }
    #pragma omp barrier
    omp_destroy_lock(&mat_lock);
    if (mat_lo.is_col_major())
    {
        tmp_loc.swap_to_col_major();
    }
    mat_lo = tmp_loc;
}


//! collect a IJ-pair storage from a 2D block local matrix
template <typename T>
void map_block_to_IJ_storage(std::map<int, std::map<int, matrix_m<T>>> &IJmap,
                             const AtomicBasis &atbasis_row,
                             const AtomicBasis &atbasis_col,
                             const matrix_m<T> &mat_lo,
                             const ArrayDesc &desc, MAJOR major_map)
{
    assert(as_size(desc.m()) == atbasis_row.nb_total && as_size(desc.n()) == atbasis_col.nb_total);
    int I, J, iI, jJ;
    for (int i_lo = 0; i_lo != desc.m_loc(); i_lo++)
    {
        int i_glo = desc.indx_l2g_r(i_lo);
        atbasis_row.get_local_index(i_glo, I, iI);
        for (int j_lo = 0; j_lo != desc.n_loc(); j_lo++)
        {
            int j_glo = desc.indx_l2g_c(j_lo);
            atbasis_col.get_local_index(j_glo, J, jJ);
            if (IJmap.count(I) == 0 || IJmap.at(I).count(J) == 0 || IJmap.at(I).at(J).size() == 0)
                IJmap[I][J] = matrix_m<T>{as_int(atbasis_row.get_atom_nb(I)),
                                          as_int(atbasis_col.get_atom_nb(J)), major_map};
            IJmap[I][J](iI, jJ) = mat_lo(i_lo, j_lo);
        }
    }
}


template <typename T>
void map_block_to_IJ_storage_new(std::map<int, std::map<int, matrix_m<T>>> &IJmap,
                                 const AtomicBasis &atbasis,
                                 const std::map<int, std::vector<int>> &map_lor_I_is,
                                 const std::map<int, std::vector<int>> &map_loc_J_js,
                                 const matrix_m<T> &mat_lo, const ArrayDesc &desc, MAJOR major_map)
{
    // map<int, map<int, matrix_m<T>>> IJmap_local;
    int max_threads = omp_get_max_threads(); // Get the total number of available threads

    std::vector<int> Is;
    std::vector<int> Js;
    for (const auto &I_is: map_lor_I_is)
    {
        Is.push_back(I_is.first);
    }
    for (const auto &J_js: map_loc_J_js)
    {
        Js.push_back(J_js.first);
    }
    const int n_pairs_total = as_int(Is.size() * Js.size());
    const int n_Is = as_int(Is.size());
    const int n_Js = as_int(Js.size());

    // Empty BLACS ranks own no entries to map. This routine is rank-local;
    // returning here preserves the collectives performed by its caller.
    if (n_pairs_total == 0) return;

    // Create necessay atom-pair blocks
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int i_I = 0; i_I != n_Is; i_I++)
    {
        for (int j_J = 0; j_J != n_Js; j_J++)
        {
            const auto &I = Is[i_I];
            const auto &J = Js[j_J];
            const auto n_i = as_int(atbasis.get_atom_nb(I));
            const auto n_j = as_int(atbasis.get_atom_nb(J));
            auto mat = matrix_m<T>(n_i, n_j, major_map);
#pragma omp critical
            {
                // IJmap_local[I][J] = std::move(mat);
                IJmap[I][J] = std::move(mat);
            }
        }
    }

    int subdiv = max_threads / n_pairs_total;
    if (subdiv * n_pairs_total < max_threads)
    {
        subdiv += 1;
    }

    // Compute starting and ending j indices
    std::vector<std::array<int, 4>> blocks;
    for (const auto &I: Is)
    {
        for (const auto &J: Js)
        {
            const int n_j_loc = map_loc_J_js.at(J).size();
            const auto step = n_j_loc / subdiv;
            const auto resi = n_j_loc % subdiv;
            for (int j_subdiv = 0; j_subdiv < subdiv; j_subdiv++)
            {
                int st = j_subdiv * step + j_subdiv * int(j_subdiv < resi) + resi * int(j_subdiv >= resi);
                int ed = std::min(n_j_loc, st + step + int(j_subdiv < resi));
                blocks.push_back({I, J, st, ed});
            }
        }
    }

    const int n_blocks = as_int(blocks.size());
#pragma omp parallel for schedule(dynamic)
    for (int i_block = 0; i_block != n_blocks; i_block++)
    {
        const auto &index = blocks[i_block];
        const auto &I = index[0];
        const auto &J = index[1];
        const auto &i_indices = map_lor_I_is.at(I);
        const auto &j_indices = map_loc_J_js.at(J);

        // auto mat = IJmap_local[I][J];
        auto &mat = IJmap[I][J];

        const auto &j_st = index[2];
        const auto &j_ed = index[3];
        for (int i_idx = 0; i_idx < as_int(i_indices.size()); i_idx++)
        {
            const int i = i_indices[i_idx];
            const auto i_loc = desc.indx_g2l_r(atbasis.get_global_index(I, i));
            for (int j_idx = j_st; j_idx < j_ed; j_idx++)
            {
                const int j = j_indices[j_idx];
                const auto l_loc = desc.indx_g2l_c(atbasis.get_global_index(J, j));
                mat(i, j) = mat_lo(i_loc, l_loc);
            }
        }
    }

    // Merge local map into the global map
    // IJmap.merge(IJmap_local);
}

template <typename T, typename Treal = typename to_real<T>::type>
void print_matrix_mm_parallel(std::ostream &os, const matrix_m<T> &mat_loc, const ArrayDesc &ad,
                              const std::string &comment = "", Treal threshold = 1e-15, bool row_first = true)
{
    ArrayDesc ad_fb(ad.ictxt());
    const int nr = ad.m(), nc = ad.n();
    const int irsrc = ad.irsrc(), icsrc = ad.icsrc();
    ad_fb.init(nr, nc, nr, nc, irsrc, icsrc);
    matrix_m<T> mat_glo = init_local_mat<T>(ad_fb, mat_loc.major());

    ScalapackConnector::pgemr2d_f(nr, nc, mat_loc.ptr(), 1, 1, ad.desc, mat_glo.ptr(), 1, 1,
                                  ad_fb.desc, ad.ictxt());
    if (ad_fb.is_src() && os.good())
    {
        print_matrix_mm(mat_glo, os, comment, threshold, row_first);
    }
}

template <typename T, typename Treal = typename to_real<T>::type>
void print_matrix_mm_file_parallel(const string &fn, const matrix_m<T> &mat_loc,
                                   const ArrayDesc &ad, const std::string &comment = "",
                                   Treal threshold = 1e-15, bool row_first = true)
{
    ofstream fs;
    if (ad.is_src())
        fs.open(fn);
    print_matrix_mm_parallel(fs, mat_loc, ad, comment, threshold, row_first);
    if (ad.is_src())
        fs.close();
}

template <typename T, typename Treal = typename to_real<T>::type>
void write_matrix_elsi_csc_parallel(const string &fn, const matrix_m<T> &mat_loc, const ArrayDesc &ad, Treal threshold = 1e-15)
{
    ArrayDesc ad_fb(ad.ictxt());
    const int nr = ad.m(), nc = ad.n();
    const int irsrc = ad.irsrc(), icsrc = ad.icsrc();
    ad_fb.init(nr, nc, nr, nc, irsrc, icsrc);
    matrix_m<T> mat_glo = init_local_mat<T>(ad_fb, mat_loc.major());

    ScalapackConnector::pgemr2d_f(nr, nc, mat_loc.ptr(), 1, 1, ad.desc, mat_glo.ptr(), 1, 1, ad_fb.desc, ad.ictxt());
    if (ad_fb.is_src()) write_matrix_elsi_csc(mat_glo, fn, threshold);
    ad.barrier();
}

template <typename T>
matrix_m<T> multiply_scalapack(const matrix_m<T> &m1_loc, const ArrayDesc &desc_m1,
                               const matrix_m<T> &m2_loc, const ArrayDesc &desc_m2,
                               const ArrayDesc &desc_prod)
{
    if (m1_loc.major() != m2_loc.major())
    {
        throw std::invalid_argument("m1 and m2 in different major");
    }

    const int m = desc_m1.m();
    const int n = desc_m2.n();
    const int k = desc_m1.n();

    const int m2 = desc_m2.m();
    if (k != m2)
    {
        throw std::invalid_argument("m1.col != m2.row: " + std::to_string(k) + " != " + std::to_string(m2));
    }

    matrix_m<T> prod_loc = init_local_mat<T>(desc_prod, m1_loc.major());
    if (m1_loc.is_col_major())
    {
        ScalapackConnector::pgemm_f('N', 'N', m, n, k, 1.0,
                m1_loc.ptr(), 1, 1, desc_m1.desc,
                m2_loc.ptr(), 1, 1, desc_m2.desc,
                0.0,
                prod_loc.ptr(), 1, 1, desc_prod.desc);
    }
    else
    {
        throw std::logic_error("row-major parallel multiply is not implemented");
    }
    return prod_loc;
}

template <typename T>
void invert_scalapack(matrix_m<T> &m_loc, const ArrayDesc &desc_m)
{
    assert(m_loc.is_col_major());
    int info = 0;
    if (m_loc.is_col_major())
    {
        // NOTE: use local leading dimension desc_m.lld() will lead to invalid pointer error
        int *ipiv = new int [desc_m.m()];
        // librpa_int::global::lib_printf("invert_scalpack desc %s\n", desc_m.info_desc().c_str());
        ScalapackConnector::pgetrf_f(desc_m.m(), desc_m.n(), m_loc.ptr(), 1, 1, desc_m.desc, ipiv,
                                     info);
        // get optimized work size
        int lwork = -1, liwork = -1;
        {
            T *work = new T [1];
            int *iwork = new int [1];
            ScalapackConnector::pgetri_f(desc_m.m(), m_loc.ptr(), 1, 1, desc_m.desc, ipiv, work,
                                         lwork, iwork, liwork, info);
            lwork = std::max(as_int(get_real(work[0])), 1);
            liwork = std::max(iwork[0], 1);
            // librpa_int::global::lib_printf("lwork %d liwork %d\n", lwork, liwork);
            delete [] work;
            delete [] iwork;
        }

        T *work = new T[lwork];
        int *iwork = new int[liwork];
        ScalapackConnector::pgetri_f(desc_m.m(), m_loc.ptr(), 1, 1, desc_m.desc, ipiv, work, lwork,
                                     iwork, liwork, info);
        delete [] iwork;
        delete [] work;
        delete [] ipiv;
        // librpa_int::global::lib_printf("done free\n");
    }
    else
    {
        throw std::logic_error("row-major invert is not implemented");
    }
}

template <typename T>
void fill_local_mat_from_ap_dist_scheduler(matrix_m<T> &m_loc,
                                           const ap_p_map<matrix_m<T>> &data,
                                           const IndexScheduler &sched,
                                           const AtomicBasis &atbasis_r,
                                           const AtomicBasis &atbasis_c,
                                           const ArrayDesc &ad)
{
    global::profiler.start(__FUNCTION__, LIBRPA_VERBOSE_DEBUG);
    assert(sched.initialized());
    assert(ad.initialized());
    const auto major_data = m_loc.major();
    const auto row_major = major_data == MAJOR::ROW ? true : false;
    if (sched.row_major() != row_major)
    {
        throw LIBRPA_RUNTIME_ERROR("IndexScheduler storage order does not match the BLACS local matrix");
    }

    // Resolve data type for communication
    MPI_Datatype dtype = mpi_datatype<T>::value;

    const auto nr = m_loc.nr();
    const auto nc = m_loc.nc();

    // Fill in data that is already available before communication
    // global::profiler.start("assign_self_ap2blacs");
    if (row_major)
    {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ir = 0; ir < nr; ir++)
        {
            for (int ic = 0; ic < nc; ic++)
            {
                int I, J, i, j;
                atbasis_r.get_local_index(ad.indx_l2g_r(ir), I, i);
                atbasis_c.get_local_index(ad.indx_l2g_c(ic), J, j);
                const atpair_t atpair{static_cast<atom_t>(I), static_cast<atom_t>(J)};
                auto it = data.find(atpair);
                if (it != data.end()) m_loc(ir, ic) = it->second(i, j);
            }
        }
    }
    else
    {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ic = 0; ic < nc; ic++)
        {
            for (int ir = 0; ir < nr; ir++)
            {
                int I, J, i, j;
                atbasis_c.get_local_index(ad.indx_l2g_c(ic), J, j);
                atbasis_r.get_local_index(ad.indx_l2g_r(ir), I, i);
                const atpair_t atpair{static_cast<atom_t>(I), static_cast<atom_t>(J)};
                auto it = data.find(atpair);
                if (it != data.end()) m_loc(ir, ic) = it->second(i, j);
            }
        }
    }
    // global::profiler.stop("assign_self_ap2blacs");

    // prepare send buffer (from AP blocks)
    // global::profiler.start("prep_send_ap2blacs");
    // global::ofs_myid << "prep_send_ap2blacs total_count_ap " << sched.total_count_ap << endl;
    std::vector<T> sendbuff(sched.total_count_ap);
    #pragma omp parallel for
    for (MPI_Count i = 0; i < sched.total_count_ap; i++)
    {
        const auto &atpair = sched.atpairs[sched.ids_ap_ipair[i]];
        const auto &locid = sched.ids_ap_locid[i];
        sendbuff[i] = data.at(atpair).ptr()[locid];
    }
    // global::profiler.stop("prep_send_ap2blacs");

    // prepare recv buffer (to BLACS mapping), just allocate
    std::vector<T> recvbuff(sched.total_count_blacs);

    // Begin communication
    const auto nprocs = ad.nprocs();
    std::vector<MPI_Request> reqs;
    MPI_Request req;

    // global::profiler.start("comm_ap2blacs");
    // First non-blocking receive
    for (int pid = 0; pid < nprocs; pid++)
    {
        const auto &count = sched.counts_blacs[pid];
        if (count == 0) continue;
        const auto &disp = sched.disp_blacs[pid];
        // ofs << "MPI_Irecv dist " << dist << " count " << count << " from pid " << pid << endl;
        MPI_Irecv(recvbuff.data() + disp, count, dtype, pid, 0, ad.comm(), &req);
        reqs.push_back(req);
    }
    // Then non-blocking send
    for (int pid = 0; pid < nprocs; pid++)
    {
        const auto &count = sched.counts_ap[pid];
        if (count == 0) continue;
        const auto &disp = sched.disp_ap[pid];
        // ofs << "MPI_Isend count " << count << " to pid " << pid << endl;
        MPI_Isend(sendbuff.data() + disp, count, dtype, pid, 0, ad.comm(), &req);
        reqs.push_back(req);
    }

    // Wait for all non-blocking communication to finish
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    // global::profiler.stop("comm_ap2blacs");

    // Map the received data to the correct location in the BLACS sub-block
    // global::profiler.start("assign_ap2blacs");
    // global::ofs_myid << "assign_ap2blacs total_count_blacs " << sched.total_count_blacs << endl;
    #pragma omp parallel for
    for (MPI_Count i = 0; i < sched.total_count_blacs; i++)
    {
        const auto &locid = sched.ids_blacs_locid[i];
        m_loc.ptr()[locid] = recvbuff[i];
    }
    // global::profiler.stop("assign_ap2blacs");
    global::profiler.stop(__FUNCTION__);
}

/*!
 * @brief Fill the BLACS local matrix on each process by redistributing atom-pair mapping data
 *
 * @param  [in,out] m_loc.                The BLACS local matrix to be filled.
 * @param  [in]     data                  Full matrix data distributed in atom-pair mapping
 * @param  [in]     map_proc_IJs_avail    Distribution of atom-pair blocks on all processes.
 *                                        The keys of the map are process ranks within
 *                                        the context of the array descriptor.
 * @param  [in]     atbasis_r             AtomicBasis object for row
 * @param  [in]     atbasis_c             AtomicBasis object for column
 * @param  [in]     ad                    Array descriptor for BLACS distribution
 */
template <typename T>
void fill_local_mat_from_ap_dist(matrix_m<T> &m_loc,
                                 const ap_p_map<matrix_m<T>> &data,
                                 const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                 const AtomicBasis &atbasis_r,
                                 const AtomicBasis &atbasis_c,
                                 const ArrayDesc &ad)
{
    assert(ad.initialized());
    const auto major_data = m_loc.major();
    const auto row_major = major_data == MAJOR::ROW ? true : false;

    // clean up
    m_loc.zero_out();

    IndexScheduler sched;
    std::unordered_map<int, std::set<atpair_t>> map_proc_set_IJs;
    for (const auto &proc_IJs: map_proc_IJs_avail)
    {
        const auto &IJs = proc_IJs.second;
        map_proc_set_IJs[proc_IJs.first] = std::set<atpair_t>{IJs.cbegin(), IJs.cend()};
    }
    sched.init(map_proc_set_IJs, atbasis_r, atbasis_c, ad, row_major);

    fill_local_mat_from_ap_dist_scheduler(m_loc, data, sched, atbasis_r, atbasis_c, ad);
}

/*!
 * @brief Get the BLACS local matrix on each process by redistributing atom-pair mapping data
 *
 * @param  [in]  data                  Full matrix data distributed in atom-pair mapping
 * @param  [in]  map_proc_IJs_avail    Distribution of atom-pair blocks on all processes.
 *                                     The keys of the map are process ranks within
 *                                     the context of the array descriptor.
 * @param  [in]  atbasis_r             AtomicBasis object for row
 * @param  [in]  atbasis_c             AtomicBasis object for column
 * @param  [in]  ad                    Array descriptor for BLACS distribution
 * @param  [in]  major_data            Major of the input atom-pair matrices
 *
 * @retval    matrix_m. The major is the same as the input, major_data
 */
template <typename T>
matrix_m<T> get_local_mat_from_ap_dist(const std::unordered_map<atpair_t, matrix_m<T>, atpair_hash> &data,
                                       const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                       const AtomicBasis &atbasis_r,
                                       const AtomicBasis &atbasis_c,
                                       const ArrayDesc &ad, MAJOR major_data)
{
    assert(ad.initialized());

    // Initialize return matrix
    auto m_loc = init_local_mat<T>(ad, major_data);
    // cout << global::myid_global << " " << m_loc.size() << " " << ad.m_loc() << " " << ad.n_loc()<< endl;
    fill_local_mat_from_ap_dist(m_loc, data, map_proc_IJs_avail, atbasis_r, atbasis_c, ad);
    return m_loc;
}

template <typename T> matrix_m<T>
get_local_mat_from_ap_dist_scheduler(const std::unordered_map<atpair_t, matrix_m<T>, atpair_hash> &data, // FIXME: do not know why cannot use ap_p_map<matrix_m<T>> here
                                     const IndexScheduler &sched,
                                     const AtomicBasis &atbasis_r,
                                     const AtomicBasis &atbasis_c,
                                     const ArrayDesc &ad, MAJOR major_data)
{
    assert(sched.initialized());
    assert(ad.initialized());

    // Initialize return matrix
    auto m_loc = init_local_mat<T>(ad, major_data);
    // cout << global::myid_global << " " << m_loc.size() << " " << ad.m_loc() << " " << ad.n_loc()<< endl;
    fill_local_mat_from_ap_dist_scheduler(m_loc, data, sched, atbasis_r, atbasis_c, ad);
    return m_loc;
}

/*!
 * @brief Fill the BLACS local matrix on each process by redistributing atom-pair mapping data.
 *        Similar to fill_local_mat_from_ap_dist, but it only requires input data on either lower
 *        or upper half of the global matrix.
 *
 * @param  [in,out] m_loc.                The BLACS local matrix to be filled.
 * @param  [in]     data                  Global matrix data distributed in atom-pair mapping,
 *                                        either upper or lower half depending on uplo.
 * @param  [in]     uplo                  whether the matrix data are available in upper ('U', 'u') or lower ('L', 'l') half.
 * @param  [in]     map_proc_IJs_avail    Distribution of atom-pair blocks on all processes.
 *                                        The keys of the map are process ranks within
 *                                        the context of the array descriptor.
 * @param  [in]     atbasis               AtomicBasis object for row and column
 * @param  [in]     ad                    Array descriptor for BLACS distribution
 * @param  [in]     apply_conjugate       Flag to apply conjugate when necessary, useful when the global matrix
 *                                        is a complex Hermitian matrix.
 */
template <typename T>
void fill_local_mat_from_ap_dist_sy(matrix_m<T> &m_loc,
                                    const std::unordered_map<atpair_t, matrix_m<T>, atpair_hash> data,
                                    const char &uplo,
                                    const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                    const AtomicBasis &atbasis,
                                    const ArrayDesc &ad, bool apply_conjugate)
{
    assert(ad.initialized());
    const auto major_data = m_loc.major();

    // Resolve data type for communication
    MPI_Datatype dtype = mpi_datatype<T>::value;

    const auto &myid = ad.myid();
    const auto &row_first = major_data == MAJOR::ROW ? false : true;
    const auto &row_major = major_data == MAJOR::ROW ? true : false;

    // Fill in data that is already available before communication
    int I, J, i, j;
    for (int ir = 0; ir < m_loc.nr(); ir++)
    {
        atbasis.get_local_index(ad.indx_l2g_r(ir), I, i);
        for (int ic = 0; ic < m_loc.nc(); ic++)
        {
            atbasis.get_local_index(ad.indx_l2g_c(ic), J, j);
            const atpair_t atpair{static_cast<atom_t>(I), static_cast<atom_t>(J)};
            const atpair_t atpair_T{static_cast<atom_t>(J), static_cast<atom_t>(I)};
            if (I == J && data.count(atpair))
            {
                // diagonal block
                m_loc(ir, ic) = data.at(atpair)(i, j);
            }
            else
            {
                // off-diagonal block, obtain element from existing conjugate block
                if (data.count(atpair))
                {
                    m_loc(ir, ic) = data.at(atpair)(i, j);
                }
                else if (data.count(atpair_T))
                {
                    const auto &mat = data.at(atpair_T);
                    m_loc(ir, ic) = is_complex<T>() && apply_conjugate? get_conj(mat(i, j)) : mat(i, j);
                }
            }
        }
    }

    // Compute indices of matrix elements that should be communicated
    const auto proc2idlist = get_communicate_local_ids_list_ap_to_blacs_sy(
        myid, uplo, map_proc_IJs_avail, atbasis, ad, row_first, row_major);
    const auto &pid_ids_send = proc2idlist.first;
    const auto &pid_ids_lconj_recv = proc2idlist.second;

    // prepare recv buffer
    // computer recv buffer size and displacements
    std::map<int, std::pair<int, MPI_Count>> pid_recv_disp_count;
    std::vector<T> recvbuff(0);
    MPI_Count recvcount = 0;
    for (int pid = 0; pid < ad.nprocs(); pid++)
    {
        if (pid_ids_lconj_recv.count(pid))
        {
            const auto &size = pid_ids_lconj_recv.at(pid).first.size();
            pid_recv_disp_count[pid] = {as_int(recvcount), size};
            recvcount += size;
        }
    }
    if (recvcount > 0) recvbuff.resize(recvcount);

    // prepare send buffer
    // first run, compute send buffer size and displacements, allocate the whole buffer
    std::map<int, std::pair<int, MPI_Count>> pid_send_disp_count;
    std::vector<T> sendbuff(0);
    MPI_Count sendcount = 0;
    for (int pid = 0; pid < ad.nprocs(); pid++)
    {
        if (pid_ids_send.count(pid))
        {
            const auto &ids_send = pid_ids_send.at(pid);
            MPI_Count sendcount_pid = 0;
            for (const auto &pair_ids: ids_send)
            {
                sendcount_pid += pair_ids.second.size();
            }
            pid_send_disp_count[pid] = {as_int(sendcount), sendcount_pid};
            sendcount += sendcount_pid;
        }
    }
    if (sendcount > 0) sendbuff.resize(sendcount);

    // second run, fill in the data to be sent
    for (int pid = 0; pid < ad.nprocs(); pid++)
    {
        if (pid_ids_send.count(pid))
        {
            const auto &ids_send = pid_ids_send.at(pid);
            MPI_Count disp = pid_send_disp_count.at(pid).first;
            for (const auto &pair_ids: ids_send)
            {
                const auto &atpair = pair_ids.first;
                const auto &atmat = data.at(atpair);
                const auto &ids = pair_ids.second;
                for (size_t i = 0; i < ids.size(); i++)
                {
                    const auto &id = ids[i];
                    sendbuff[disp+i] = atmat.ptr()[id];
                }
                disp += ids.size();
            }
        }
    }

    // Begin communication
    std::vector<MPI_Request> reqs;
    MPI_Request req;

    // First non-blocking receive
    for (const auto &pid_disp_count: pid_recv_disp_count)
    {
        const auto &pid = pid_disp_count.first;
        const auto &disp_count = pid_disp_count.second;
        const auto &disp = disp_count.first;
        const auto &count = disp_count.second;
        // ofs << "MPI_Irecv dist " << dist << " count " << count << " from pid " << pid << endl;
        MPI_Irecv(recvbuff.data() + disp, count, dtype, pid, 0, ad.comm(), &req);
        reqs.push_back(req);
    }
    // Then non-blocking send
    for (const auto &pid_disp_count: pid_send_disp_count)
    {
        const auto &pid = pid_disp_count.first;
        const auto &disp_count = pid_disp_count.second;
        const auto &disp = disp_count.first;
        const auto &count = disp_count.second;
        // ofs << "MPI_Isend count " << count << " to pid " << pid << endl;
        MPI_Isend(sendbuff.data() + disp, count, dtype, pid, 0, ad.comm(), &req);
        reqs.push_back(req);
    }

    // Wait for all non-blocking communication to finish
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Map the received data to the correct location in the BLACS sub-block
    for (const auto &pid_disp_count: pid_recv_disp_count)
    {
        const auto &pid = pid_disp_count.first;
        const auto &ids = pid_ids_lconj_recv.at(pid).first;
        const auto &conjs = pid_ids_lconj_recv.at(pid).second;
        const auto &disp_count = pid_disp_count.second;
        const auto &disp = disp_count.first;
        const auto &count = disp_count.second;
        if (is_complex<T>() && apply_conjugate)
        {
            for (auto i = 0; i < count; i++)
            {
                m_loc.ptr()[ids[i]] = conjs[i] == 'c' ? get_conj(recvbuff[disp+i]) : recvbuff[disp+i];
            }
        }
        else
        {
            for (auto i = 0; i < count; i++)
            {
                m_loc.ptr()[ids[i]] = recvbuff[disp+i];
            }
        }
    }
}

/*!
 * @brief Get the BLACS local matrix on each process by redistributed atom-pair mapping data.
 *        Similar to get_local_mat_from_ap_dist, but it only requires input data on either lower
 *        or upper half of the global matrix.
 *
 * @param  [in]  data                  Global matrix data distributed in atom-pair mapping,
 *                                     either upper or lower half depending on uplo.
 * @param  [in]  uplo                  whether the matrix data are available in upper ('U', 'u') or lower ('L', 'l') half.
 * @param  [in]  map_proc_IJs_avail    Distribution of atom-pair blocks on all processes.
 *                                     The keys of the map are process ranks within
 *                                     the context of the array descriptor.
 * @param  [in]  atbasis               AtomicBasis object for row and column
 * @param  [in]  ad                    Array descriptor for BLACS distribution
 * @param  [in]  major_data            Major of the input atom-pair matrices
 * @param  [in]  apply_conjugate       Flag to apply conjugate when necessary, useful when the global matrix
 *                                     is a complex Hermitian matrix.
 *
 * @retval    matrix_m. The major is the same as the input, major_data
 */
template <typename T>
matrix_m<T> get_local_mat_from_ap_dist_sy(const std::unordered_map<atpair_t, matrix_m<T>, atpair_hash> data,
                                          const char &uplo,
                                          const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_avail,
                                          const AtomicBasis &atbasis,
                                          const ArrayDesc &ad, bool apply_conjugate, MAJOR major_data)
{
    assert(ad.initialized());

    // Initialize return matrix
    auto m_loc = init_local_mat<T>(ad, major_data);
    fill_local_mat_from_ap_dist_sy<T>(m_loc, data, uplo, map_proc_IJs_avail, atbasis, ad, apply_conjugate);
    return m_loc;
}

template <typename T>
void fill_ap_map_from_blacs_dist_scheduler(ap_p_map<matrix_m<T>> &data,
                                           const matrix_m<T> &m_loc,
                                           const IndexScheduler &sched,
                                           const AtomicBasis &atbasis_r,
                                           const AtomicBasis &atbasis_c,
                                           const ArrayDesc &ad)
{
    global::profiler.start(__FUNCTION__, LIBRPA_VERBOSE_DEBUG);

    assert(ad.initialized());
    assert(sched.initialized());
    const auto major_data = m_loc.major();

    // clean up input
    data.clear();

    // Resolve data type for communication
    MPI_Datatype dtype = mpi_datatype<T>::value;

    const auto &row_major = major_data == MAJOR::ROW ? true : false;
    if (sched.row_major() != row_major)
    {
        throw LIBRPA_RUNTIME_ERROR("IndexScheduler storage order does not match the BLACS local matrix");
    }

    // global::profiler.start("assign_self_blacs2ap");
    // Fill in data that is already available before communication
    // atom pairs required by this process
    const auto IJs = std::unordered_set<atpair_t, atpair_hash>(sched.atpairs.cbegin(), sched.atpairs.cend());
    const auto nr = m_loc.nr();
    const auto nc = m_loc.nc();
    // Allocate necessary blocks first
    for (const auto &atpair: sched.atpairs)
    {
        const auto I = atpair.first;
        const auto J = atpair.second;
        const auto &nI = atbasis_r.get_atom_nb(I);
        const auto &nJ = atbasis_c.get_atom_nb(J);
        data[atpair] = matrix_m<T>(nI, nJ, major_data);
    }
    if (row_major)
    {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ir = 0; ir < nr; ir++)
        {
            for (int ic = 0; ic < nc; ic++)
            {
                int I, J, i, j;
                atbasis_r.get_local_index(ad.indx_l2g_r(ir), I, i);
                atbasis_c.get_local_index(ad.indx_l2g_c(ic), J, j);
                const atpair_t atpair{static_cast<atom_t>(I), static_cast<atom_t>(J)};
                if (IJs.find(atpair) == IJs.cend()) continue; // not required
                const auto &nJ = atbasis_c.get_atom_nb(J);
                data.at(atpair).ptr()[i*nJ+j] = m_loc.ptr()[ir*nc+ic];
            }
        }
    }
    else
    {
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ic = 0; ic < nc; ic++)
        {
            for (int ir = 0; ir < nr; ir++)
            {
                int I, J, i, j;
                atbasis_c.get_local_index(ad.indx_l2g_c(ic), J, j);
                atbasis_r.get_local_index(ad.indx_l2g_r(ir), I, i);
                const atpair_t atpair{static_cast<atom_t>(I), static_cast<atom_t>(J)};
                if (IJs.find(atpair) == IJs.cend()) continue; // not required
                const auto nI = atbasis_r.get_atom_nb(I);
                data.at(atpair).ptr()[j*nI+i] = m_loc.ptr()[ic*nr+ir];
            }
        }
    }
    // global::profiler.stop("assign_self_blacs2ap");
    // return;

    // prepare send buffer (from BLACS block)
    // global::profiler.start("prep_send_blacs2ap");
    std::vector<T> sendbuff(sched.total_count_blacs);
    #pragma omp parallel for
    for (MPI_Count i = 0; i < sched.total_count_blacs; i++)
    {
        const auto &id = sched.ids_blacs_locid[i];
        sendbuff[i] = m_loc.ptr()[id];
    }
    // global::profiler.stop("prep_send_blacs2ap");

    // prepare recv buffer (to AP mapping)
    std::vector<T> recvbuff(sched.total_count_ap);

    // Begin communication
    // global::profiler.start("comm_blacs2ap");
    std::vector<MPI_Request> reqs;
    MPI_Request req;

    const auto nprocs = ad.nprocs();

    // First non-blocking receive
    for (int pid = 0; pid < nprocs; pid++)
    {
        const auto &count = sched.counts_ap[pid];
        if (count == 0) continue;
        const auto &disp = sched.disp_ap[pid];
        // ofs << "MPI_Irecv dist " << dist << " count " << count << " from pid " << pid << endl;
        MPI_Irecv(recvbuff.data() + disp, count, dtype, pid, 0, ad.comm(), &req);
        reqs.push_back(req);
    }
    // Then non-blocking send
    for (int pid = 0; pid < nprocs; pid++)
    {
        const auto &count = sched.counts_blacs[pid];
        if (count == 0) continue;
        const auto &disp = sched.disp_blacs[pid];
        // ofs << "MPI_Isend count " << count << " to pid " << pid << endl;
        MPI_Isend(sendbuff.data() + disp, count, dtype, pid, 0, ad.comm(), &req);
        reqs.push_back(req);
    }

    // Wait for all non-blocking communication to finish
    if (!reqs.empty()) MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    // global::profiler.stop("comm_blacs2ap");

    // Map the received data to the correct location in the atom-pair mapping
    // global::profiler.start("assign_blacs2ap");
    // Allocate the required atom-pair block first
    // global::ofs_myid << "sched.total_count_ap " << sched.total_count_ap << std::endl;
    #pragma omp parallel for
    for (MPI_Count i = 0; i < sched.total_count_ap; i++)
    {
        const auto &atpair = sched.atpairs[sched.ids_ap_ipair[i]];
        const auto &locid = sched.ids_ap_locid[i];
        // global::ofs_myid << "pair " << atpair.first << " " << atpair.second << " locid " << locid << " at? " << std::boolalpha << data.count(atpair) << std::endl;
        data.at(atpair).ptr()[locid] = recvbuff[i];
    }
    // global::profiler.stop("assign_blacs2ap");
    global::profiler.stop(__FUNCTION__);
}

/*!
 * @brief Fill atom-pair mapping data by redistributing the BLACS local matrix on each process.
 *
 * @param  [in,out] data                  Matrix data distributed in atom-pair mapping to be completed
 * @param  [in]     m_loc                 The BLACS local matrix owned by each process
 * @param  [in]     map_proc_IJs_require  Atom-pair blocks required by each processes.
 * @param  [in]     atbasis_r             AtomicBasis object for row
 * @param  [in]     atbasis_c             AtomicBasis object for column
 * @param  [in]     ad                    Array descriptor for BLACS distribution
 */
template <typename T>
void fill_ap_map_from_blacs_dist(ap_p_map<matrix_m<T>> &data,
                                 const matrix_m<T> &m_loc,
                                 const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_require,
                                 const AtomicBasis &atbasis_r,
                                 const AtomicBasis &atbasis_c,
                                 const ArrayDesc &ad)
{
    assert(ad.initialized());
    const auto row_major = m_loc.major() == MAJOR::ROW ? true : false;

    IndexScheduler sched;
    std::unordered_map<int, std::set<atpair_t>> map_proc_set_IJs;
    for (const auto &proc_IJs: map_proc_IJs_require)
    {
        const auto &IJs = proc_IJs.second;
        map_proc_set_IJs[proc_IJs.first] = std::set<atpair_t>{IJs.cbegin(), IJs.cend()};
    }
    sched.init(map_proc_set_IJs, atbasis_r, atbasis_c, ad, row_major);
    fill_ap_map_from_blacs_dist_scheduler(data, m_loc, sched, atbasis_r, atbasis_c, ad);
}

/*!
 * @brief Get atom-pair mapping data by redistributing the BLACS local matrix on each process.
 *
 * @param  [in]     m_loc                 The BLACS local matrix owned by each process
 * @param  [in]     map_proc_IJs_require  Atom-pair blocks required by each processes.
 * @param  [in]     atbasis_r             AtomicBasis object for row
 * @param  [in]     atbasis_c             AtomicBasis object for column
 * @param  [in]     ad                    Array descriptor for BLACS distribution
 * @param  [in]     major_data            Major of the input atom-pair matrices
 *
 * @retval atom-pair mapping matrix data. The major is the same as the input m_loc, major_data
 */
template <typename T> ap_p_map<matrix_m<T>>
get_ap_map_from_blacs_dist(const matrix_m<T> &m_loc,
                           const std::unordered_map<int, std::vector<atpair_t>> &map_proc_IJs_require,
                           const AtomicBasis &atbasis_r,
                           const AtomicBasis &atbasis_c,
                           const ArrayDesc &ad,
                           MAJOR major_data = MAJOR::AUTO)
{
    assert(ad.initialized());

    // cout << global::myid_global
    //      << " " << m_loc.size()
    //      << " " << m_loc.nr() << " " << m_loc.nc()
    //      << " " << ad.m_loc() << " " << ad.n_loc()<< endl;
    if (major_data != MAJOR::AUTO && major_data != m_loc.major())
    {
        throw std::logic_error("major passed but not consistent with m_loc");
    }

    ap_p_map<matrix_m<T>> data;
    fill_ap_map_from_blacs_dist<T>(data, m_loc, map_proc_IJs_require, atbasis_r, atbasis_c, ad);
    return data;
}

template <typename T> ap_p_map<matrix_m<T>>
get_ap_map_from_blacs_dist_scheduler(const matrix_m<T> &m_loc,
                                     const IndexScheduler &sched,
                                     const AtomicBasis &atbasis_r,
                                     const AtomicBasis &atbasis_c,
                                     const ArrayDesc &ad,
                                     MAJOR major_data = MAJOR::AUTO)
{
    assert(sched.initialized());
    assert(ad.initialized());

    // cout << global::myid_global
    //      << " " << m_loc.size()
    //      << " " << m_loc.nr() << " " << m_loc.nc()
    //      << " " << ad.m_loc() << " " << ad.n_loc()<< endl;
    if (major_data != MAJOR::AUTO && major_data != m_loc.major())
    {
        throw std::logic_error("major passed but not consistent with m_loc");
    }

    ap_p_map<matrix_m<T>> data;
    fill_ap_map_from_blacs_dist_scheduler(data, m_loc, sched, atbasis_r, atbasis_c, ad);
    return data;
}


} /* end of namespace librpa_int */


#endif // UTILS_MATRIX_M_MPI_H
