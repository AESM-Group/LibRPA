#pragma once
/**
 * @file file_reader.hpp
 * @brief Experimental file input interface for LibBSE only.
 *
 * Link against librpa_file_reader, which depends on rpa_lib and is also built
 * with LIBRPA_ENABLE_DRIVER=OFF. This interface retains the existing Dataset
 * return type, input defaults, and replicated loading behavior.
 */

// Public API headers
#include "librpa.hpp"

// Internal data types exposed by this C++ interface
#include "../src/api/dataset.h"
#include "../src/core/coulmat.h"
#include "../src/math/lapack_connector.h"
#include "../src/math/scalapack_connector.h"

// Standard headers
#include <memory>
#include <string>

namespace librpa
{

/** @brief Options controlling dataset input from LibRPA files. */
struct FileReaderOptions
{
    std::string input_dir;               ///< Directory containing LibRPA input files.
    double cs_threshold = 1.0e-12;       ///< Screening threshold for RI coefficients.
    double coulomb_threshold = 1.0e-12;  ///< Screening threshold for Coulomb matrices.
    bool read_ri = true;                 ///< Whether to read RI and Coulomb data.
    bool read_band_data = true;          ///< Read separate band files; false copies SCF data.
    bool distribute_lri = false;        ///< Give each RI block one MPI owner (required for analytic wings).
    bool read_full_coulomb = false;      ///< Also read coulomb_mat for dielectric screening.

    // Defaults preserve the original LibBSE input filenames.
    std::string fn_stru = "stru_out";
    std::string fn_bz_sampling = "bz_sampling_out";
    std::string fn_basis = "basis_out";
    std::string fn_basis_wfc = "basis_wfc_out";
    std::string fn_basis_aux = "basis_aux_out";
    std::string fn_eigocc_scf = "band_out";
    std::string fn_velocity = "velocity_matrix";
    std::string fn_band_kpath_info = "band_kpath_info";
    std::string prefix_eigvecs_scf = "KS_eigenvector";
    std::string prefix_lri_coeff = "Cs_data";
    std::string prefix_lri_coeff_shrink = "Cs_shrinked_data";
    std::string prefix_coul_cut = "coulomb_unshrinked_cut";
};

/**
 * @brief Read a LibRPA dataset from files.
 *
 * @param[in] comm MPI communicator used to initialize the dataset.
 * @param[in] opts File reader options.
 * @return Shared pointer to the populated LibRPA dataset.
 *
 * Initialize MPI and LibRPA before calling. Input is replicated on each rank
 * by default; distribute_lri=true assigns each RI block to one rank.
 * With read_band_data=false, SCF data is
 * copied to the band data. Release the dataset before global finalization.
 *
 * @code
 * librpa::FileReaderOptions input;
 * input.input_dir = "dataset";
 * input.fn_eigocc_scf = "band_out_custom"; // optional filename
 * auto dataset = librpa::read_dataset_from_files(MPI_COMM_WORLD, input);
 * // Use dataset in LibBSE.
 * dataset.reset();
 * @endcode
 */
std::shared_ptr<librpa_int::Dataset> read_dataset_from_files(
    MPI_Comm comm, const FileReaderOptions &opts);

} // namespace librpa
