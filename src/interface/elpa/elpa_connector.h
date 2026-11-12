#ifndef ELPA_CONNECTOR_H
#define ELPA_CONNECTOR_H

#pragma once
#include <omp.h>
#include "../../math/matrix_m.h"
#include "../../mpi/base_blacs.h"


namespace librpa_int
{

namespace ElpaConnector
{

/*!
 * @brief Compute power of Hermitian matrix using BLACS
 *
 * @param  [in,out]  A_local       Process-local part of global matrix A to be powered
 * @param  [in]      ad_A          Array descriptor of A
 * @param  [out]     Z_local       Process-local part of the eigenvector matrix (Z) of A
 * @param  [in]      ad_Z          Array descriptor of Z
 * @param  [out]     n_filtered    Array descriptor of Z
 * @param  [out]     W             Eigenvalues of A, including those smaller than threshold
 * @param  [in]      power         Power to perform
 * @param  [in]      threshold     The threshold to filter the eigenvalues
 *
 * @retval           scale_Z       Eigenvectors scaled by the power of eigenvalues, using ad_Z
 */
template <typename T>
matrix_m<std::complex<T>> power_hemat_elpa(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, matrix_m<std::complex<T>> &Z_local,
    const ArrayDesc &ad_Z, size_t &n_filtered, T *W, T power, const T &threshold = -1.e5, bool use_gpu_replace_scalapack = false,
    std::complex<T>* d_A = nullptr, std::complex<T>* d_Z = nullptr, std::complex<T>* d_C = nullptr);


template <typename T>
matrix_m<std::complex<T>> power_hemat_elpa_real(
    matrix_m<std::complex<T>> &A_local, const ArrayDesc &ad_A, matrix_m<std::complex<T>> &Z_local,
    const ArrayDesc &ad_Z, size_t &n_filtered, T *W, T power, const T &threshold = -1.e5,
    bool use_gpu_replace_scalapack = false, T* d_A = nullptr, T* d_Z = nullptr, T* d_C = nullptr);



}

}

#endif // ELPA_CONNECTOR_H
