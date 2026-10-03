#pragma once

#include <complex>
#include <string>
#include <vector>

#include "../src/math/complexmatrix.h"

namespace driver
{

struct SternheimerChi0V1Matrix
{
    std::string path;
    int iq = 0;
    int ifreq = 0;
    double omega = 0.0;
    double weight = 0.0;
    std::vector<int> atom_naux;
    librpa_int::ComplexMatrix matrix;
};

SternheimerChi0V1Matrix read_sternheimer_chi0_v1_matrix_file(const std::string &path);

void write_sternheimer_chi0_v1_matrix_file(const std::string &path,
                                            const SternheimerChi0V1Matrix &response);

librpa_int::ComplexMatrix read_coulomb_v1_full_matrix(const std::string &dir_path,
                                                      const std::string &prefix, int iq);

void validate_coulomb_v1_full_matrix_file(const std::string &dir_path, const std::string &prefix,
                                          int iq);

std::vector<SternheimerChi0V1Matrix> read_sternheimer_chi0_v1_matrices(const std::string &dir_path,
                                                                       const std::string &prefix,
                                                                       int iq);

void validate_sternheimer_chi0_v1_files(const std::string &dir_path, const std::string &prefix,
                                        int iq, int expected_nfreq);

}  // namespace driver
