#include "read_data.h"
#include <librpa_enums.h>

#include "reader/reader_structure.h"
#include "reader/reader_basis.h"
#include "reader/read_data.h"
#include "reader_2d_coulomb_head.h"
#include "../src/api/dataset_helper.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <librpa.hpp>

#include "driver.h"
#include "../src/mpi/global_mpi.h"
#include "../src/math/matrix.h"
#include "../src/utils/constants.h"
#include "../src/api/instance_manager.h"
#include "../src/core/meanfield_mpi.h"
#include "../src/io/aux_basis_summary.h"
#include "../src/io/fs.h"
#include "../src/io/global_io.h"
#include "../src/io/stl_io_helper.h"
#include "../src/utils/error.h"
#include "../src/utils/profiler.h"
#include "../src/utils/utils_mem.h"

using std::ifstream;
using std::string;

namespace
{

librpa::reader::ReaderContext driver_reader_context()
{
    librpa::reader::ReaderParameters params;
    params.prefix_eigvecs_scf = driver::driver_params.prefix_eigvecs_scf;
    params.prefix_lri_coeff = driver::driver_params.prefix_lri_coeff;
    params.prefix_lri_coeff_shrink = driver::driver_params.prefix_lri_coeff_shrink;
    params.use_spinor_wfc = driver::driver_params.use_spinor_wfc;
    return {driver::h, driver::reader_state, params, driver::opts,
            librpa_int::global::mpi_comm_global_h};
}


constexpr double kSymmetryKpointMatchTol = 1e-5;
constexpr std::int32_t READER_SHRINK_SINVS_V1_MARKER = -30241621;

}  // namespace

struct QpStateRange
{
    int low;
    int high;
};

static QpStateRange automatic_qp_state_range(const librpa_int::MeanField &mf)
{
    const int n_states_mf = mf.get_n_states();
    if (n_states_mf <= 0)
    {
        throw std::runtime_error("Cannot resolve QP state range from an empty meanfield object");
    }

    const double gap = mf.get_band_gap();
    const double efermi = mf.get_efermi();
    const double e_low = efermi - 0.5 * gap - 0.5;
    const double e_high = efermi + 0.5 * gap + 0.5;
    const int i_state_low = mf.get_max_state_below_energy(e_low) + 1;
    const int i_state_high = mf.get_min_state_above_energy(e_high);

    if (i_state_high <= i_state_low)
    {
        return {0, n_states_mf};
    }
    return {i_state_low, i_state_high};
}

static void normalize_qp_state_range_from_kgrid_mf(const librpa_int::MeanField &mf)
{
    auto &params = driver::driver_params;
    const int n_states_mf = mf.get_n_states();
    if (n_states_mf <= 0)
    {
        throw std::runtime_error("Cannot resolve QP state range from an empty meanfield object");
    }

    const bool automatic_low = params.i_state_low < 0;
    const bool automatic_high = params.i_state_high < 0;
    const bool use_automatic_default_high =
        params.i_state_high == driver::DriverParams::default_i_state_high &&
        (automatic_low || automatic_high);
    const QpStateRange automatic_range =
        (automatic_low || automatic_high || use_automatic_default_high)
            ? automatic_qp_state_range(mf)
            : QpStateRange{0, n_states_mf};

    if (automatic_low)
    {
        params.i_state_low = automatic_range.low;
    }
    else if (params.i_state_low > n_states_mf)
    {
        std::stringstream ss;
        ss << "i_state_low = " << params.i_state_low << " exceeds the maximum number of states ("
           << n_states_mf << ")";
        throw std::runtime_error(ss.str());
    }

    if (automatic_high || use_automatic_default_high)
    {
        params.i_state_high = automatic_range.high;
    }
    else if (params.i_state_high > n_states_mf)
    {
        params.i_state_high = n_states_mf;
    }

    if (params.i_state_high <= params.i_state_low)
    {
        std::stringstream ss;
        ss << "Empty QP state range: i_state_low = " << params.i_state_low
           << ", i_state_high = " << params.i_state_high << ". The high state index is exclusive.";
        throw std::runtime_error(ss.str());
    }

    if (params.output_gw_spec_func)
    {
        if (params.sf_state_start < 0) params.sf_state_start = params.i_state_low;
        if (params.sf_state_end < 0) params.sf_state_end = params.i_state_high;
        params.sf_state_start = std::max(params.i_state_low, params.sf_state_start);
        params.sf_state_end = std::min(params.i_state_high, params.sf_state_end);
        if (params.sf_state_end <= params.sf_state_start)
        {
            std::stringstream ss;
            ss << "Empty spectral-function state range: sf_state_start = "
               << params.sf_state_start << ", sf_state_end = " << params.sf_state_end
               << ". The high state index is exclusive.";
            throw std::runtime_error(ss.str());
        }
    }
}

void read_scf_occ_eigenvalues(const string &file_path)
{
    using driver::iks_eigvec_this;
    using driver::n_basis_ao;
    using driver::n_basis_wfc;
    using driver::n_kpoints;
    using driver::n_spinor;
    using driver::n_spins;
    using driver::n_states;
    using librpa_int::global::myid_global;
    using librpa_int::global::size_global;
    using std::to_string;

    // cout << "Begin to read band_out" << endl;
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    string ks, ss, a, ws, es, d;
    double efermi;
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_states;
    infile >> n_basis_wfc;
    infile >> efermi;

    const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;

    if (use_spinor_wfc)
    {
        assert(n_spins == 1);
        assert(n_basis_wfc % 2 == 0 && "Error: nbasis is not even when SOC!");
        n_spinor = 2;
        n_basis_ao = n_basis_wfc / 2;
    }
    else
    {
        n_spinor = 1;
        n_basis_ao = n_basis_wfc;
    }

    driver::h.set_scf_dimension(n_spins, n_kpoints, n_states, n_basis_ao, n_spinor);
    auto pds = librpa_int::api::get_dataset_instance(driver::h);
    const auto &kbctxt = pds->scfk_blacs_ctxt;

    iks_eigvec_this.clear();
    if (driver::get_bool(driver::opts.use_kpara_scf_eigvec))
    {
        // reusing the internal distribution
        if (kbctxt.comm_blacs_h.myid == 0) iks_eigvec_this = kbctxt.kpoints_local();
    }
    else
    {
        for (int ik = 0; ik < driver::n_kpoints; ik++) iks_eigvec_this.emplace_back(ik);
    }

    driver::n_ibz_kpoints = n_kpoints;

    // Load the file data
    auto eskb = new double[n_spins * n_kpoints * n_states];
    auto wskb = new double[n_spins * n_kpoints * n_states];

    const int n_kb = n_kpoints * n_states;

    int iline = 6;

    // cout<<"|eskb: "<<endl;
    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            infile >> ks >> ss;
            if (!infile.good())
            {
                throw std::logic_error("Error in reading k- and spin- index: line " +
                                       to_string(iline) + ", file: " + file_path);
            }
            iline++;
            // cout<<ik<<is<<endl;
            int k_index = stoi(ks) - 1;
            // int s_index = stoi(ss) - 1;
            for (int i = 0; i != n_states; i++)
            {
                // iband weight energy(Ha) energy(eV)
                infile >> a >> ws >> es >> d;
                if (!infile.good())
                {
                    throw std::logic_error("Error in reading band energy and occupation: line " +
                                           to_string(iline) + ", file: " + file_path);
                }
                iline++;
                wskb[is * n_kb + k_index * n_states + i] = stod(ws);
                eskb[is * n_kb + k_index * n_states + i] = stod(es);
                // cout<<" i_band: "<<i<<"    eskb: "<<eskb[is](k_index, i)<<endl;
            }
        }
    }
    // for (int is = 0; is != n_spins; is++)
    //     print_matrix("eskb_mat",eskb[is]);

    driver::h.set_wg_ekb_efermi(n_spins, n_kpoints, n_states, wskb, eskb, efermi);

    // free buffer
    delete[] eskb;
    delete[] wskb;

    normalize_qp_state_range_from_kgrid_mf(pds->mf);
}

int read_vxc(const string &file_path, std::vector<matrix> &vxc)
{
    if (!librpa_int::file_exists(file_path)) return 1;
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    infile.open(file_path);
    double ha, ev;
    int n_spins, n_kpoints, n_states;
    // int retcode;

    // dimension information
    infile >> n_kpoints;
    infile >> n_spins;
    infile >> n_states;
    if (!infile.good())
    {
        return 1;
    }

    vxc.clear();
    vxc.resize(n_spins);
    for (int is = 0; is != n_spins; is++)
    {
        vxc[is].create(n_kpoints, n_states);
    }

    for (int ik = 0; ik != n_kpoints; ik++)
    {
        for (int is = 0; is != n_spins; is++)
        {
            for (int i = 0; i != n_states; i++)
            {
                infile >> ha >> ev;
                if (!infile.good())
                {
                    return 2;
                }
                vxc[is](ik, i) = ha;
            }
        }
    }
    return 0;
}

void read_ri(const string &dir_path, librpa::ParallelRouting &routing)
{
    using driver::local_atpair;
    using driver::n_atoms;
    using driver::n_kpoints;
    using librpa_int::decide_auto_routing;
    using librpa_int::dispatch_upper_triangular_tasks;
    using librpa_int::generate_atom_pair_from_nat;
    using namespace librpa_int::global;

    mpi_comm_global_h.barrier();
    lib_printf_root("Loading RI file from directory: %s\n", dir_path.c_str());

    const auto tot_atpair = generate_atom_pair_from_nat(n_atoms, false);
    const auto tot_atpair_ordered = generate_atom_pair_from_nat(n_atoms, true);

    if (routing == LIBRPA_ROUTING_AUTO)
    {
        routing = decide_auto_routing(n_atoms, driver::opts.nfreq * n_kpoints);
    }

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    const auto &Cs_data = pds->cs_data;
    const auto &blacs_h = pds->blacs_h;
    const bool use_shrink_abfs = driver::get_bool(driver::opts.use_shrink_abfs);

    local_atpair.clear();

    // HACK: local_atpair should be set in the same mechanism as inside the dataset object,
    //       which is implemented in initialize_ds_atpairs_local in dataset_helper.cpp.
    //       It consists of distributed atom pairs of only upper half, since repsonse function
    //       matrix is Hermitian.
    if (routing == LIBRPA_ROUTING_ATOMPAIR)
    {
        lib_printf_root("Triangular dispatching of atom pairs\n");
        auto tri_local_atpair = librpa_int::dispatch_upper_triangular_tasks(
            n_atoms, blacs_h.myid, blacs_h.nprows, blacs_h.npcols, blacs_h.myprow, blacs_h.mypcol);
        for (const auto &p : tri_local_atpair) local_atpair.push_back(p);
        profiler.start("driver_read_Cs");
        read_Cs(dir_path, driver::driver_params.cs_threshold, local_atpair,
                driver::driver_params.prefix_lri_coeff, driver::driver_params.version_lri_reader);
        profiler.stop("driver_read_Cs");

        if (use_shrink_abfs) read_ri_shrink(dir_path);

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        read_Vq_row(dir_path, driver::driver_params.prefix_coul_full, driver::opts.vq_threshold,
                    local_atpair, false, driver::driver_params.version_coul_reader,
                    use_shrink_abfs);
        profiler.stop("driver_read_Vq");
    }
    else if (routing == LIBRPA_ROUTING_LIBRI)
    {
        lib_printf_root("Evenly distributed Cs and V for LibRI\n");
        profiler.start("driver_read_Cs");
        read_Cs_evenly_distribute(dir_path, driver::driver_params.cs_threshold,
                                  mpi_comm_global_h.myid, mpi_comm_global_h.nprocs,
                                  driver::driver_params.prefix_lri_coeff,
                                  driver::driver_params.version_lri_reader);
        profiler.stop("driver_read_Cs");
        if (use_shrink_abfs) read_ri_shrink(dir_path);
        // Vq distributed using the same strategy
        // There should be no duplicate for V

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        auto trangular_loc_atpair = librpa_int::dispatch_upper_triangular_tasks(
            n_atoms, blacs_h.myid, blacs_h.nprows, blacs_h.npcols, blacs_h.myprow, blacs_h.mypcol);
        for (auto &iap : trangular_loc_atpair) local_atpair.push_back(iap);
        read_Vq_row(dir_path, driver::driver_params.prefix_coul_full, driver::opts.vq_threshold,
                    local_atpair, false, driver::driver_params.version_coul_reader,
                    use_shrink_abfs);
        profiler.stop("driver_read_Vq");
    }
    else
    {
        lib_printf_root("Complete copy of Cs and V on each process\n");
        local_atpair = generate_atom_pair_from_nat(n_atoms, false);
        profiler.start("driver_read_Cs");
        read_Cs(dir_path, driver::driver_params.cs_threshold, local_atpair,
                driver::driver_params.prefix_lri_coeff, driver::driver_params.version_lri_reader);
        profiler.stop("driver_read_Cs");

        if (use_shrink_abfs) read_ri_shrink(dir_path);

        mpi_comm_global_h.barrier();
        profiler.start("driver_read_Vq");
        read_Vq_full(dir_path, driver::driver_params.prefix_coul_full, false,
                     driver::driver_params.version_coul_reader, use_shrink_abfs);
        profiler.stop("driver_read_Vq");
    }

    mpi_comm_global_h.barrier();
    lib_printf_coll("| Process %5d: coulomb_mat read. Wall/CPU time [min]: %12.4f %12.4f\n",
                    mpi_comm_global_h.myid, profiler.get_wall_time_last("driver_read_Vq") / 60.0,
                    profiler.get_cpu_time_last("driver_read_Vq") / 60.0);
    mpi_comm_global_h.barrier();
    lib_printf_coll(
        "| Process %5d: Cs with %14zu non-zero keys from local atpair size %7zu. "
        "Data memory: %10.2f MB. Wall/CPU time [min]: %12.4f %12.4f\n",
        mpi_comm_global_h.myid, Cs_data.n_keys(), local_atpair.size(),
        Cs_data.n_data_bytes() * 8.0e-6, profiler.get_wall_time_last("driver_read_Cs") / 60.0,
        profiler.get_cpu_time_last("driver_read_Cs") / 60.0);
    mpi_comm_global_h.barrier();
}

static std::vector<Vector3_Order<double>> read_headwing_k_path_info(const string &file_path,
                                                                    int &n_basis, int &n_states,
                                                                    int &n_spin)
{
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    infile.open(file_path);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    int n_kpoints;
    infile >> n_basis >> n_states >> n_spin >> n_kpoints;
    if (!infile.good())
    {
        throw std::logic_error("Failed to read headwing k_path_info header from " + file_path);
    }

    std::vector<Vector3_Order<double>> kfrac_list;
    kfrac_list.reserve(n_kpoints);
    string x, y, z;
    for (int ik = 0; ik < n_kpoints; ++ik)
    {
        infile >> x >> y >> z;
        if (!infile.good())
        {
            throw std::logic_error("Failed to read headwing k point from " + file_path);
        }
        kfrac_list.push_back({stod(x), stod(y), stod(z)});
    }
    return kfrac_list;
}

void read_headwing_input(const string &dir_path, bool need_wing)
{
    using namespace librpa_int;
    using namespace librpa_int::global;

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    auto &mf = pds->mf;
    auto &velocity_matrix = pds->velocity_matrix;
    velocity_matrix.clear();
    const int scf_nk = mf.get_n_kpoints();
    if (static_cast<int>(pds->pbc.kfrac_list.size()) != scf_nk)
    {
        throw std::runtime_error("SCF k-point list is inconsistent with meanfield");
    }
    struct MfRestore
    {
        MeanField &mf;
        MeanField original;
        bool active = false;

        explicit MfRestore(MeanField &mf_in) : mf(mf_in) {}
        void capture()
        {
            original = mf;
            active = true;
        }
        ~MfRestore()
        {
            if (active)
            {
                mf = std::move(original);
            }
        }
    } restore_mf(mf);

    const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;
    const bool wants_headwing_symmetry =
        driver::get_bool(driver::opts.use_symmetry_rpa) ||
        driver::get_bool(driver::opts.use_symmetry_gw);
    const bool use_kpara_eigvec = driver::get_bool(driver::opts.use_kpara_scf_eigvec);
    const string pyatb_dir = path_as_directory(dir_path) + "pyatb_librpa_df/";
    const string pyatb_velocity = pyatb_dir + "velocity_matrix";

    const auto &active_kfrac_list = pds->pbc.kfrac_list;
    std::vector<Vector3_Order<double>> kfrac_pyatb;
    velocity_matrix_t direct_full_bz_velocity;
    MeanField direct_full_bz_wfc;
    std::vector<std::vector<int>> direct_full_bz_velocity_member_source_ik;
    int n_basis = 0;
    int n_states = 0;
    int n_spin = 0;

    std::vector<double> freq_weights;
    driver::h.get_imaginary_frequency_grids(driver::opts, pds->omegas_imagfreq, freq_weights);
    const auto &freqs = pds->tfg.get_freq_nodes();

    if (path_exists(pyatb_velocity.c_str()))
    {
        // PyATB may write a full-BZ k grid even when LibRPA runs with symmetry
        // and the active mean-field grid is IBZ.  Treat k_path_info as the
        // source-coordinate table, then read only the active LibRPA k-list.
        if (mpi_comm_global_h.is_root())
        {
            std::cout << "Reading head/wing input from " << pyatb_dir << std::endl;
        }
        kfrac_pyatb = read_headwing_k_path_info(pyatb_dir + "k_path_info", n_basis, n_states,
                                                 n_spin);
        if (use_spinor_wfc)
        {
            if (n_basis % 2 != 0)
                throw std::runtime_error("Head/wing spinor basis size is not even");
            n_basis /= 2;
        }
        const auto target_to_source_ik =
            map_kpoints_by_coordinates(active_kfrac_list, kfrac_pyatb,
                                       kSymmetryKpointMatchTol);
        std::vector<int> source_to_target_ik(kfrac_pyatb.size(), -1);
        for (int ik_target = 0; ik_target != static_cast<int>(target_to_source_ik.size());
             ++ik_target)
        {
            source_to_target_ik.at(target_to_source_ik[ik_target]) = ik_target;
        }

        restore_mf.capture();
        read_scf_occ_eigenvalues(pyatb_dir + "band_out", mf, use_spinor_wfc,
                                  target_to_source_ik, static_cast<int>(kfrac_pyatb.size()));
        const bool direct_headwing_kblacs_2d =
            use_kpara_eigvec && driver::opts.parallel_routing == LIBRPA_ROUTING_LIBRI;
        std::vector<int> iks_headwing_eigvec_this;
        if (use_kpara_eigvec && !direct_headwing_kblacs_2d &&
            pds->scfk_blacs_ctxt.is_initialized())
        {
            if (pds->scfk_blacs_ctxt.n_kpoints() != mf.get_n_kpoints())
                throw std::runtime_error(
                    "SCF k-point parallel context is inconsistent with head/wing meanfield");
            if (pds->scfk_blacs_ctxt.comm_blacs_h.myid == 0)
            {
                for (const int ik_target : pds->scfk_blacs_ctxt.kpoints_local())
                    iks_headwing_eigvec_this.emplace_back(target_to_source_ik.at(ik_target));
            }
        }
        const std::vector<int> *source_iks_headwing_eigvec_selected =
            use_kpara_eigvec && pds->scfk_blacs_ctxt.is_initialized()
                ? &iks_headwing_eigvec_this
                : nullptr;
        const int ret_eigenvec =
            direct_headwing_kblacs_2d
                ? read_eigenvector_kblacs_2d(
                      pyatb_dir, mf, use_spinor_wfc, pds->scfk_blacs_ctxt,
                      pds->desc_wfc_kb, &source_to_target_ik,
                      LegacyTextWfcOrder::SpinBasisBand)
                : read_eigenvector(pyatb_dir, mf, use_spinor_wfc, source_to_target_ik,
                                   source_iks_headwing_eigvec_selected,
                                   LegacyTextWfcOrder::SpinBasisBand);
        if (ret_eigenvec != 0)
        {
            throw std::runtime_error("Failed to read pyatb head/wing eigenvectors from " +
                                     pyatb_dir);
        }
        if (direct_headwing_kblacs_2d)
            pds->mark_eigvecs_kpara_2d_ready();
        read_velocity(pyatb_velocity, mf, velocity_matrix, source_to_target_ik,
                      static_cast<int>(kfrac_pyatb.size()));
        if (n_basis != mf.get_n_aos() || n_states != mf.get_n_states() ||
            n_spin != mf.get_n_spins())
        {
            throw std::runtime_error(
                "Head/wing k_path_info dimensions are inconsistent with band_out");
        }
    }
    else
    {
        // ABACUS/FHI-aims package outputs use the SCF k grid for head/wing.
        // Only the velocity/momentum matrix is head/wing-specific here.
        n_basis = mf.get_n_aos();
        n_states = mf.get_n_states();
        n_spin = mf.get_n_spins();

        const string input_path = path_as_directory(dir_path);
        if (driver::driver_params.input_preset == "abacus" ||
            driver::driver_params.input_preset == "abacus-legacy")
        {
            read_velocity_abacus(mf, input_path, driver::driver_params.prefix_velocity,
                                velocity_matrix);
        }
        else if (driver::driver_params.input_preset == "fhi-aims")
        {
            read_velocity_aims(mf, input_path, driver::driver_params.prefix_velocity,
                               velocity_matrix);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported head/wing input preset: " + driver::driver_params.input_preset);
        }
    }

    if (wants_headwing_symmetry &&
        (!pds->symmetry_context.available || pds->symmetry_context.kstars.empty() ||
         pds->symmetry_context.rsh_rotations.empty()))
    {
        librpa_int::initialize_symmetry_context(*pds, true);
    }
    if (path_exists(pyatb_velocity.c_str()) && pds->symmetry_context.available &&
        !pds->symmetry_context.kstars.empty())
    {
        direct_full_bz_velocity_member_source_ik =
            map_symmetry_kstar_members_to_source_kpoints(
                pds->symmetry_context, active_kfrac_list, kfrac_pyatb,
                kSymmetryKpointMatchTol);
        MeanField full_bz_headwing_mf(
            mf.get_n_spins(), static_cast<int>(kfrac_pyatb.size()), mf.get_n_states(),
            mf.get_n_aos(), mf.get_n_spinor());
        std::vector<int> full_bz_identity_map(kfrac_pyatb.size());
        for (int ik = 0; ik != static_cast<int>(full_bz_identity_map.size()); ++ik)
            full_bz_identity_map[ik] = ik;
        const int ret_full_wfc = read_eigenvector(
            pyatb_dir, full_bz_headwing_mf, use_spinor_wfc, full_bz_identity_map, nullptr,
            LegacyTextWfcOrder::SpinBasisBand);
        if (ret_full_wfc != 0)
        {
            throw std::runtime_error(
                "Failed to read full-BZ PyATB head/wing eigenvectors from " + pyatb_dir);
        }
        read_velocity(pyatb_velocity, full_bz_headwing_mf, direct_full_bz_velocity);
        direct_full_bz_wfc = std::move(full_bz_headwing_mf);
        librpa_int::global::lib_printf_root(
            "Head/wing symmetry: use direct full-BZ PyATB WFC and velocity for %zu k-star members.\n",
            pds->symmetry_context.count_kstar_members());
    }
    const auto &headwing_basis_aux =
        driver::get_bool(driver::opts.use_shrink_abfs) ? pds->basis_aux_shrink : pds->basis_aux;
    if (!headwing_basis_aux.initialized())
        throw std::runtime_error("Head/wing auxiliary basis is not initialized");

    const bool use_headwing_symmetry = wants_headwing_symmetry;
    if (use_headwing_symmetry)
    {
        reject_spinor_symmetry_speedup(*pds, "analytic head/wing");
        initialize_symmetry_context(*pds, true);
        require_symmetry_shell_layouts(*pds, "analytic head/wing");
    }
    pds->p_headwing = std::make_unique<diele_func>(
        mf, velocity_matrix, pds->pbc.kfrac_list, pds->basis_wfc, headwing_basis_aux, freqs,
        n_basis, n_states, n_spin, headwing_basis_aux.nb_total, pds->pbc, pds->comm_h,
        pds->blacs_h, use_kpara_eigvec, &pds->scfk_blacs_ctxt,
        &(pds->eigvecs_kpara_2d_ready() ? pds->desc_wfc_kb : pds->desc_wfc_kb_full));
    const bool use_strict_2d = driver::get_bool(driver::opts.use_2d_dielectric);
    if (use_strict_2d)
    {
        const auto metadata = read_strict_2d_coulomb_head_metadata(
            librpa_int::join_path(dir_path, "librpa_2d_coulomb_head.txt"));
        const auto normalization = librpa_int::strict_2d_coulomb_head_normalization(
            pds->pbc, metadata.auxiliary_monopole_norm_squared);
        const double area_scale = std::max(normalization.inplane_area_bohr2,
                                           metadata.inplane_area_bohr2);
        if (std::abs(normalization.inplane_area_bohr2 - metadata.inplane_area_bohr2) >
            1.0e-10 * area_scale)
            throw std::runtime_error(
                "Strict 2D Coulomb metadata area is inconsistent with the input lattice");

        librpa_int::global::lib_printf_root(
            "Strict 2D Coulomb normalization computed from reader-v1 metadata: "
            "area=%20.12e Bohr^2, multipole_norm_squared=%20.12e, "
            "A_lambda=%20.12e, PW_to_auxiliary_scale=%20.12e\n",
            normalization.inplane_area_bohr2,
            normalization.auxiliary_monopole_norm_squared,
            normalization.auxiliary_head_coefficient,
            normalization.pw_to_auxiliary_scale);
        pds->p_headwing->configure_strict_2d_coulomb_head(
            true, metadata.auxiliary_monopole_norm_squared);
    }
    else
    {
        pds->p_headwing->configure_strict_2d_coulomb_head(false);
    }
    pds->p_headwing->use_soc = mf.get_n_spinor() > 1;
    pds->p_headwing->debug = librpa_int::global::should_output(LIBRPA_VERBOSE_DEBUG);
    // Symmetry-aware head/wing uses the same active k-list as the main LibRPA
    // path: full BZ without symmetry, IBZ with input k-star metadata.
    if (use_headwing_symmetry && pds->symmetry_context.available &&
        !pds->symmetry_context.kstars.empty())
    {
        pds->p_headwing->set_symmetry_context(pds->symmetry_context);
        pds->p_headwing->use_symmetry = true;
        for (atom_t atom = 0; atom != static_cast<atom_t>(pds->basis_wfc.n_atoms); ++atom)
            pds->p_headwing->atom_nw[atom] = pds->basis_wfc.get_atom_nb(atom);
        pds->p_headwing->coord_frac = pds->symmetry_context.input_coord_frac;
    }
    if (!direct_full_bz_velocity.empty())
    {
        pds->p_headwing->set_direct_full_bz_headwing_inputs(
            std::move(direct_full_bz_velocity),
            std::move(direct_full_bz_wfc),
            std::move(direct_full_bz_velocity_member_source_ik));
    }
    if (restore_mf.active)
    {
        const int n_spinor = mf.get_n_spinor();
        auto &hw_mf = pds->p_headwing->get_meanfield_df();
        for (int ispin = 0; ispin != n_spin; ++ispin)
        {
            for (int ispinor = 0; ispinor != n_spinor; ++ispinor)
            {
                std::vector<int> missing_k;
                for (int ik = 0; ik != mf.get_n_kpoints(); ++ik)
                {
                    const auto *C = hw_mf.find_wfc(ispin, ispinor, ik);
                    const int have_local = (C != nullptr && C->nr > 0) ? 1 : 0;
                    int have_any = 0;
                    MPI_Allreduce(&have_local, &have_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
                    if (have_any == 0) missing_k.push_back(ik);
                }
                if (!missing_k.empty())
                {
                    std::ostringstream oss;
                    oss << "PyATB head/wing input is missing eigenvectors for "
                        << missing_k.size() << " k-points in spin " << ispin << " spinor "
                        << ispinor << ". Check k_path_info and KS_eigenvector_* coverage.";
                    throw std::runtime_error(oss.str());
                }
            }
        }
        if (mpi_comm_global_h.is_root())
            std::cout << "Using PyATB head/wing input on the active LibRPA k-list with "
                      << mf.get_n_kpoints() << " k-points" << std::endl;
    }
    pds->p_headwing->init(driver::opts.sqrt_coulomb_threshold, pds->vq);
    pds->p_headwing->cal_head();
    pds->epsmacs_imagfreq = pds->p_headwing->get_head_vec();
    pds->omegas_imagfreq = freqs;
    pds->p_headwing->test_head();
    if (need_wing)
    {
        const auto &headwing_cs =
            driver::get_bool(driver::opts.use_shrink_abfs) ? pds->cs_data_shrink : pds->cs_data;
        pds->p_headwing->cal_wing(headwing_cs, driver::opts.sqrt_coulomb_threshold, pds->vq);
        pds->p_headwing->test_wing();
    }
}

void read_dielec_func(const string &file_path, std::vector<double> &omegas,
                      std::vector<double> &dielec_func_imagfreq)
{
    std::ifstream ifs;
    double omega, re, im;
    librpa_int::require_readable_file(file_path);
    ifs.open(file_path);

    if (!ifs.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    while (ifs >> omega >> re >> im)
    {
        omegas.push_back(omega);
        dielec_func_imagfreq.push_back(re);
    }
    ifs.close();
}

void erase_Cs_from_local_atp(atpair_R_mat_t &Cs, std::vector<atpair_t> &local_atpair)
{
    using namespace std;
    using namespace librpa_int;
    // erase no need Cs

    set<size_t> loc_atp_index;
    for (auto &lap : local_atpair)
    {
        loc_atp_index.insert(lap.first);
        loc_atp_index.insert(lap.second);
    }
    std::vector<atom_t> Cs_first;
    for (const auto &Ip : Cs) Cs_first.push_back(Ip.first);
    for (const auto &I : Cs_first)
    {
        if (!loc_atp_index.count(I)) Cs.erase(I);
    }
    // for(auto &Ip:Cs)
    //     if(!loc_atp_index.count(Ip.first))
    //     {
    //         Cs.erase(Ip.first);
    //     }
    release_free_mem();
    global::lib_printf("| process %d, size of Cs after erase: %lu\n",
                       librpa_int::global::mpi_comm_global_h.myid, Cs.size());
}

void read_stru(const std::string &file_path)
{
    reader_structure(file_path);
}

void read_band_meanfield_data(const string &dir_path)
{
    if (!driver::get_bool(driver::opts.use_kpara_scf_eigvec) ||
        driver::opts.parallel_routing != LIBRPA_ROUTING_LIBRI)
    {
        auto ctx = driver_reader_context();
        librpa::reader::read_band_meanfield_data(ctx, dir_path);
        return;
    }

    using namespace librpa_int;
    using namespace librpa_int::global;
    using driver::iks_band_eigvec_this;
    using driver::n_basis_ao;
    using driver::n_basis_wfc;
    using driver::n_kpoints_band;
    using driver::n_spins;
    using driver::n_states;
    using std::endl;

    if (driver::n_kpoints_band == 0)
        throw LIBRPA_RUNTIME_ERROR(
            "Number of band k-points not set, run read_band_kpath_info first");

    iks_band_eigvec_this.clear();

    const bool direct_kblacs_2d =
        driver::get_bool(driver::opts.use_kpara_scf_eigvec) &&
        driver::opts.parallel_routing == LIBRPA_ROUTING_LIBRI;
    if (driver::get_bool(driver::opts.use_kpara_scf_eigvec) && !direct_kblacs_2d)
    {
        for (int ik = 0; ik < driver::n_kpoints_band; ik++)
        {
            if (ik % size_global == myid_global) iks_band_eigvec_this.emplace_back(ik);
        }
    }
    else
    {
        for (int ik = 0; ik < driver::n_kpoints_band; ik++) iks_band_eigvec_this.emplace_back(ik);
    }

    auto reader_ctx = driver_reader_context();
    librpa::reader::read_band_eigenvalues(reader_ctx, dir_path);

    if (direct_kblacs_2d)
    {
        auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
        pds->initialize_band_kblacs_wfc_layout();
        auto &band_kctx = pds->bandk_blacs_ctxt;
        const auto &desc_wfc = pds->desc_band_wfc_kb;
        auto &mf_band = pds->mf_band;
        const auto desc_wfc_io = band_kctx.create_array_desc(n_basis_ao, n_states);
        if (!desc_wfc_io.is_row_consec() || !desc_wfc_io.is_col_consec())
            throw LIBRPA_RUNTIME_ERROR(
                "temporary direct band eigenvector input descriptor must be contiguous");

        iks_band_eigvec_this.clear();
        if (band_kctx.comm_blacs_h.is_root())
            iks_band_eigvec_this = band_kctx.kpoints_local();

        profiler.start("driver_read_band_eigenvector_kblacs_2d");
        mf_band.get_eigenvectors().clear();
        const bool use_spinor_wfc = driver::driver_params.use_spinor_wfc;
        const int n_soc = use_spinor_wfc ? 2 : 1;
        const int local_count = desc_wfc_io.m_loc() * desc_wfc_io.n_loc();
        std::vector<std::complex<double>> dummy(1, {0.0, 0.0});

        for (const int ik : band_kctx.kpoints_local())
        {
            std::stringstream ss;
            ss << dir_path << "band_KS_eigenvector_k_" << std::setfill('0') << std::setw(5)
               << ik + 1 << ".txt";
            const auto file_path = ss.str();
            librpa_int::require_readable_file(file_path);
            ofs_myid << "Loading local 2D band eigenvector block from " << file_path << endl;

            MPI_File file = MPI_FILE_NULL;
            if (MPI_File_open(band_kctx.comm_blacs_h.comm,
                              const_cast<char *>(file_path.c_str()), MPI_MODE_RDONLY,
                              MPI_INFO_NULL, &file) != MPI_SUCCESS)
                throw LIBRPA_RUNTIME_ERROR("Fail to open band eigenvector file " + file_path);

            MPI_Offset file_size = 0;
            MPI_File_get_size(file, &file_size);
            const MPI_Offset expected_size =
                static_cast<MPI_Offset>(n_spins) * n_states * n_basis_ao * n_soc *
                sizeof(std::complex<double>);
            if (file_size < expected_size)
            {
                MPI_File_close(&file);
                throw LIBRPA_RUNTIME_ERROR("Band eigenvector file is shorter than expected: " +
                                          file_path);
            }

            int read_error = 0;
            for (int ispin = 0; ispin != n_spins; ++ispin)
            {
                for (int ispinor = 0; ispinor != n_soc; ++ispinor)
                {
                    auto &wfc = mf_band.get_eigenvectors()[ispin][ispinor][ik];
                    wfc.create(desc_wfc_io.n_loc(), desc_wfc_io.m_loc(), false);
                    MPI_Datatype filetype = MPI_C_DOUBLE_COMPLEX;
                    bool free_filetype = false;
                    MPI_Offset displacement = 0;
                    if (use_spinor_wfc)
                    {
                        const int global_sizes[3]{n_states, n_basis_ao, n_soc};
                        const int local_sizes[3]{desc_wfc_io.n_loc(), desc_wfc_io.m_loc(), 1};
                        const int starts[3]{
                            desc_wfc_io.n_loc() == 0 ? 0 : desc_wfc_io.indx_l2g_c(0),
                            desc_wfc_io.m_loc() == 0 ? 0 : desc_wfc_io.indx_l2g_r(0),
                            ispinor};
                        if (local_count > 0)
                        {
                            MPI_Type_create_subarray(3, global_sizes, local_sizes, starts,
                                                     MPI_ORDER_C, MPI_C_DOUBLE_COMPLEX,
                                                     &filetype);
                            MPI_Type_commit(&filetype);
                            free_filetype = true;
                        }
                    }
                    else
                    {
                        const int global_sizes[2]{n_states, n_basis_ao};
                        const int local_sizes[2]{desc_wfc_io.n_loc(), desc_wfc_io.m_loc()};
                        const int starts[2]{
                            desc_wfc_io.n_loc() == 0 ? 0 : desc_wfc_io.indx_l2g_c(0),
                            desc_wfc_io.m_loc() == 0 ? 0 : desc_wfc_io.indx_l2g_r(0)};
                        if (local_count > 0)
                        {
                            MPI_Type_create_subarray(2, global_sizes, local_sizes, starts,
                                                     MPI_ORDER_C, MPI_C_DOUBLE_COMPLEX,
                                                     &filetype);
                            MPI_Type_commit(&filetype);
                            free_filetype = true;
                        }
                        displacement = static_cast<MPI_Offset>(ispin) * n_states * n_basis_ao *
                                       sizeof(std::complex<double>);
                    }
                    MPI_File_set_view(file, displacement, MPI_C_DOUBLE_COMPLEX, filetype,
                                      const_cast<char *>("native"), MPI_INFO_NULL);
                    MPI_Status status;
                    auto *dst = wfc.c == nullptr ? dummy.data() : wfc.c;
                    if (MPI_File_read_all(file, dst, local_count, MPI_C_DOUBLE_COMPLEX,
                                          &status) != MPI_SUCCESS)
                        read_error = 1;
                    if (free_filetype) MPI_Type_free(&filetype);
                }
            }
            MPI_File_close(&file);
            MPI_Allreduce(MPI_IN_PLACE, &read_error, 1, MPI_INT, MPI_MAX,
                          band_kctx.comm_blacs_h.comm);
            if (read_error)
                throw LIBRPA_RUNTIME_ERROR("Error reading local band eigenvector blocks from " +
                                          file_path);
        }
        redistribute_meanfield_eigvecs_kblacs(
            mf_band, band_kctx, desc_wfc_io, desc_wfc, "direct band-path");
        pds->mark_band_eigvecs_kpara_2d_ready();
        profiler.stop("driver_read_band_eigenvector_kblacs_2d");
        return;
    }

}

std::vector<matrix> read_vxc_band(const string &dir_path, int n_states, int n_spin,
                                  int n_kpoints_band)
{
    std::vector<matrix> vxc_band(n_spin);
    for (int i_spin = 0; i_spin < n_spin; i_spin++)
    {
        vxc_band[i_spin].create(n_kpoints_band, n_states);
    }
    std::string s1, s2, s3;

    for (int ik = 0; ik < n_kpoints_band; ik++)
    {
        // Load occupation weights and eigenvalues
        std::stringstream ss;
        ss << dir_path << "band_vxc_k_" << std::setfill('0') << std::setw(5) << ik + 1 << ".txt";
        const auto file_path = ss.str();
        librpa_int::require_readable_file(file_path);
        ifstream infile;
        infile.open(file_path);
        ss.clear();

        for (int i_spin = 0; i_spin < n_spin; i_spin++)
        {
            for (int i_state = 0; i_state < n_states; i_state++)
            {
                infile >> s1 >> s2 >> s3;
                vxc_band[i_spin](ik, i_state) = stod(s3);
            }
        }

        infile.close();
    }
    return vxc_band;
}

void read_elsi_csc(const string &file_path, bool save_row_major, std::vector<double> &mat,
                   int &n_basis, bool &is_real)
{
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    infile.open(file_path, std::ios::binary);
    if (!infile.good())
    {
        throw std::logic_error("Failed to open " + file_path);
    }

    // Read the whole buffer
    infile.seekg(0, std::ios::end);
    std::streampos size = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    infile.read(buffer.data(), size);
    infile.close();

    int64_t header[16];
    std::memcpy(header, buffer.data(), 128);

    n_basis = header[3];
    int64_t nnz = header[5];
    // cout << n_basis << " " << nnz << endl;

    int64_t *col_ptr_raw = reinterpret_cast<int64_t *>(buffer.data() + 128);
    std::vector<int> col_ptr;
    col_ptr.assign(col_ptr_raw, col_ptr_raw + n_basis);
    // Trailing column index to mark the end. +1 for index starting from 1 in ELSI CSC
    col_ptr.push_back(nnz + 1);

    int32_t *row_idx_raw = reinterpret_cast<int32_t *>(buffer.data() + 128 + n_basis * 8);

    char *nnz_val_raw = buffer.data() + 128 + n_basis * 8 + nnz * 4;
    double *nnz_val_double = reinterpret_cast<double *>(nnz_val_raw);

    if (header[2] == 0)
    {
        // Real valued
        is_real = true;
        mat.resize(n_basis * n_basis);
    }
    else
    {
        // Complex valued
        is_real = false;
        mat.resize(2 * n_basis * n_basis);
    }

    for (auto col = 0; col < n_basis; ++col)
    {
        for (auto idx = col_ptr[col]; idx < col_ptr[col + 1]; ++idx)
        {
            int row = row_idx_raw[idx - 1] - 1;
            int index = save_row_major ? row * n_basis + col : col * n_basis + row;
            // cout << idx - 1 << " " << col << " " << row << " " << index << endl;
            if (is_real)
            {
                mat[index] = nnz_val_double[idx - 1];
            }
            else
            {
                mat[2 * index] = nnz_val_double[2 * idx - 2];
                mat[2 * index + 1] = nnz_val_double[2 * idx - 1];
            }
        }
    }
}

static int handle_sinvS_file(const std::string &file_path,
                             std::map<Vector3_Order<double>, ComplexMatrix> &sinvS, bool binary)
{
    librpa_int::require_readable_file(file_path);
    ifstream infile;
    int n_irk_points_local;
    // TODO: variables that needs to be adapted into pbc object
    std::map<Vector3_Order<double>, double> irk_weight;
    int n_irk_points;

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    auto &pbc = pds->pbc;

    if (binary)
    {
        infile.open(file_path, std::ios::in | std::ios::binary);
        infile.read((char *)&n_irk_points, sizeof(int));
        infile.read((char *)&n_irk_points_local, sizeof(int));
    }
    else
    {
        infile.open(file_path);
        infile >> n_irk_points;
    }

    if (!infile.good()) return 1;

    const int nk_ibz = pbc.klist_coul.size();

    if (binary)
    {
        int nbasbas_s, nbasbas, brow, erow, bcol, ecol, iq;
        double q_weight;

        for (int i_irk = 0; i_irk < n_irk_points_local; i_irk++)
        {
            infile.read((char *)&nbasbas_s, sizeof(int));
            infile.read((char *)&nbasbas, sizeof(int));
            infile.read((char *)&brow, sizeof(int));
            infile.read((char *)&erow, sizeof(int));
            infile.read((char *)&bcol, sizeof(int));
            infile.read((char *)&ecol, sizeof(int));
            infile.read((char *)&iq, sizeof(int));
            infile.read((char *)&q_weight, sizeof(double));

            brow--;
            erow--;
            bcol--;
            ecol--;
            iq--;
            if ((erow - brow < 0) || (ecol - bcol < 0) || iq < 0 || iq >= nk_ibz) return 4;
            const auto qvec = pbc.klist_coul[iq];

            if (!sinvS.count(qvec))
            {
                sinvS[qvec].create(nbasbas_s, nbasbas);
            }

            const int nrow = erow - brow + 1;
            const int ncol = ecol - bcol + 1;
            const size_t n = nrow * ncol;
            std::vector<std::complex<double>> tmp(n);
            infile.read((char *)tmp.data(), 2 * n * sizeof(double));
            for (int i = 0; i < nrow; i++)
            {
                for (int j = 0; j < ncol; j++)
                {
                    const auto i_mu = i + brow;
                    const auto i_nu = j + bcol;
                    sinvS[qvec](i_mu, i_nu) = tmp[i * ncol + j];
                }
            }
        }
    }
    else
    {
        string nbasbas_s, nbasbas, begin_row, end_row, begin_col, end_col, q1, q2, q3, vq_r, vq_i,
            q_num, q_weight;
        while (infile.peek() != EOF)
        {
            // row is mu_s, col is mu
            infile >> nbasbas_s >> nbasbas >> begin_row >> end_row >> begin_col >> end_col;
            if (infile.peek() == EOF) break;
            if (!infile.good()) return 2;

            infile >> q_num >> q_weight;
            if (!infile.good()) return 3;
            int mu = stoi(nbasbas_s);
            int nu = stoi(nbasbas);
            int brow = stoi(begin_row) - 1;
            int erow = stoi(end_row) - 1;
            int bcol = stoi(begin_col) - 1;
            int ecol = stoi(end_col) - 1;
            int iq = stoi(q_num) - 1;

            // skip empty coulumb_file
            if ((erow - brow < 0) || (ecol - bcol < 0) || iq < 0 || iq >= nk_ibz) return 4;
            const auto qvec = pbc.klist_coul[iq];
            if (!sinvS.count(qvec))
            {
                sinvS[qvec].create(mu, nu);
            }
            for (int i_mu = brow; i_mu <= erow; i_mu++)
            {
                for (int i_nu = bcol; i_nu <= ecol; i_nu++)
                {
                    infile >> vq_r >> vq_i;
                    sinvS[qvec](i_mu, i_nu) =
                        std::complex<double>(stod(vq_r), stod(vq_i));
                }
            }
        }
    }

    return 0;
}

static bool sinvS_file_has_v1_marker(const std::string &file_path)
{
    librpa_int::require_readable_file(file_path);
    ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good())
    {
        throw LIBRPA_RUNTIME_ERROR("Failed to open " + file_path);
    }
    std::int32_t marker = 0;
    infile.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    return infile.good() && marker == READER_SHRINK_SINVS_V1_MARKER;
}

static std::streamoff checked_streamoff_from_i64(const std::int64_t value,
                                                 const std::string &context)
{
    if (value < 0 ||
        static_cast<unsigned long long>(value) >
            static_cast<unsigned long long>(std::numeric_limits<std::streamoff>::max()))
    {
        throw LIBRPA_RUNTIME_ERROR(context + ": invalid file offset");
    }
    return static_cast<std::streamoff>(value);
}

static std::size_t checked_sinvS_payload_bytes(const std::int32_t nrow, const std::int32_t ncol,
                                               const std::string &file_path)
{
    if (nrow <= 0 || ncol <= 0)
    {
        throw LIBRPA_RUNTIME_ERROR(file_path + ": invalid shrink_sinvS v1 block dimensions");
    }
    const auto max_count = std::numeric_limits<std::size_t>::max() / sizeof(std::complex<double>);
    const auto nrow_size = static_cast<std::size_t>(nrow);
    const auto ncol_size = static_cast<std::size_t>(ncol);
    if (nrow_size > max_count / ncol_size)
    {
        throw LIBRPA_RUNTIME_ERROR(file_path + ": shrink_sinvS v1 block is too large");
    }
    return nrow_size * ncol_size * sizeof(std::complex<double>);
}

static int handle_sinvS_v1_file(const std::string &file_path,
                                std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    librpa_int::require_readable_file(file_path);
    struct Record
    {
        std::int32_t iq = 0;
        std::int32_t nrow_total = 0;
        std::int32_t ncol_total = 0;
        std::int32_t begin_row = 0;
        std::int32_t end_row = 0;
        std::int32_t begin_col = 0;
        std::int32_t end_col = 0;
        double weight = 0.0;
        std::int64_t offset = 0;
    };

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    auto &pbc = pds->pbc;
    const int nk_ibz = pbc.klist_coul.size();

    ifstream infile(file_path, std::ios::in | std::ios::binary);
    if (!infile.good())
    {
        return 1;
    }

    std::int32_t marker = 0;
    std::int32_t nrecords_i32 = 0;
    infile.read(reinterpret_cast<char *>(&marker), sizeof(marker));
    infile.read(reinterpret_cast<char *>(&nrecords_i32), sizeof(nrecords_i32));
    if (!infile.good() || marker != READER_SHRINK_SINVS_V1_MARKER || nrecords_i32 < 0)
    {
        return 2;
    }

    infile.seekg(0, std::ios::end);
    const auto end_pos = infile.tellg();
    if (end_pos == std::streampos(-1))
    {
        return 3;
    }
    const auto file_size = static_cast<std::streamoff>(end_pos);
    infile.seekg(2 * static_cast<std::streamoff>(sizeof(std::int32_t)), std::ios::beg);

    std::vector<Record> records(static_cast<std::size_t>(nrecords_i32));
    for (auto &record : records)
    {
        infile.read(reinterpret_cast<char *>(&record.iq), sizeof(record.iq));
        infile.read(reinterpret_cast<char *>(&record.nrow_total), sizeof(record.nrow_total));
        infile.read(reinterpret_cast<char *>(&record.ncol_total), sizeof(record.ncol_total));
        infile.read(reinterpret_cast<char *>(&record.begin_row), sizeof(record.begin_row));
        infile.read(reinterpret_cast<char *>(&record.end_row), sizeof(record.end_row));
        infile.read(reinterpret_cast<char *>(&record.begin_col), sizeof(record.begin_col));
        infile.read(reinterpret_cast<char *>(&record.end_col), sizeof(record.end_col));
        infile.read(reinterpret_cast<char *>(&record.weight), sizeof(record.weight));
        infile.read(reinterpret_cast<char *>(&record.offset), sizeof(record.offset));
        if (!infile.good())
        {
            return 4;
        }
    }

    for (const auto &record : records)
    {
        const int iq = record.iq - 1;
        if (iq < 0 || iq >= nk_ibz)
        {
            return 5;
        }
        if (record.begin_row < 1 || record.begin_col < 1 || record.end_row < record.begin_row ||
            record.end_col < record.begin_col || record.end_row > record.nrow_total ||
            record.end_col > record.ncol_total)
        {
            return 6;
        }
        const auto nrow_block = record.end_row - record.begin_row + 1;
        const auto ncol_block = record.end_col - record.begin_col + 1;
        const auto bytes = checked_sinvS_payload_bytes(nrow_block, ncol_block, file_path);
        const auto offset = checked_streamoff_from_i64(record.offset, file_path);
        if (file_size - offset < static_cast<std::streamoff>(bytes))
        {
            return 7;
        }

        std::vector<std::complex<double>> tmp(static_cast<std::size_t>(nrow_block) *
                                              static_cast<std::size_t>(ncol_block));
        infile.seekg(offset, std::ios::beg);
        infile.read(reinterpret_cast<char *>(tmp.data()), static_cast<std::streamsize>(bytes));
        if (!infile.good())
        {
            return 8;
        }

        const auto qvec = pbc.klist_coul[iq];
        if (!sinvS.count(qvec))
        {
            sinvS[qvec].create(record.nrow_total, record.ncol_total);
        }
        if (sinvS[qvec].nr != record.nrow_total || sinvS[qvec].nc != record.ncol_total)
        {
            return 9;
        }

        for (int i = 0; i != nrow_block; ++i)
        {
            for (int j = 0; j != ncol_block; ++j)
            {
                sinvS[qvec](record.begin_row - 1 + i, record.begin_col - 1 + j) =
                    tmp[static_cast<std::size_t>(i) * static_cast<std::size_t>(ncol_block) +
                        static_cast<std::size_t>(j)];
            }
        }
    }

    return 0;
}

void read_ri_shrink(const string &dir_path)
{
    using librpa_int::global::mpi_comm_global_h;
    using librpa_int::global::profiler;
    using librpa_int::join_path;
    using librpa_int::path_exists;
    using driver::driver_params;

    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());

    const auto shrink_basis_path =
        join_path(driver_params.input_dir, driver_params.fn_basis_aux_shrink);
    const auto legacy_shrink_basis_path =
        join_path(driver_params.input_dir, "basis_out_shrink");
    const auto legacy_backup_basis_path =
        join_path(driver_params.input_dir, "basis_out.shrink_backup");
    if (path_exists(shrink_basis_path.c_str()))
    {
        reader_basis_aux_shrink(shrink_basis_path);
    }
    else if (path_exists(legacy_shrink_basis_path.c_str()))
    {
        reader_basis_aux_shrink(legacy_shrink_basis_path);
    }
    else if (path_exists(legacy_backup_basis_path.c_str()))
    {
        reader_basis_aux_shrink(legacy_backup_basis_path);
    }
    else
    {
        driver::h.set_ao_basis_aux_shrink(
            read_aux_basis_from_Cs(driver_params.input_dir, driver_params.prefix_lri_coeff_shrink));
    }
    // TODO: get rid of pds call here
    pds->desc_abf_shrink.reset_handler(pds->blacs_h);
    pds->desc_abf_shrink.init_1b1p(pds->basis_aux_shrink.nb_total, pds->basis_aux_shrink.nb_total,
                                   0, 0);

    profiler.start("read_Cs_shrink");
    read_Cs_evenly_distribute(driver_params.input_dir, driver_params.cs_threshold,
                              mpi_comm_global_h.myid, mpi_comm_global_h.nprocs,
                              driver_params.prefix_lri_coeff_shrink,
                              driver_params.version_lri_reader);
    profiler.stop("read_Cs_shrink");

    profiler.start("read_shrink_sinvS_fold", "Load shrink transformation");
    pds->sinvS.clear();
    read_shrink_sinvS(driver_params.input_dir, driver_params.prefix_shrink_sinvS, pds->sinvS);

    if (!pds->sinvS.empty())
    {
        const auto &first_sinvS = pds->sinvS.begin()->second;
        if (static_cast<size_t>(first_sinvS.nr) != pds->basis_aux_shrink.nb_total ||
            static_cast<size_t>(first_sinvS.nc) != pds->basis_aux.nb_total)
        {
            throw std::runtime_error(
                "shrink_sinvS dimensions are inconsistent with auxiliary bases");
        }
    }

    if (mpi_comm_global_h.is_root())
    {
        std::cout << librpa_int::format_aux_basis_compression_summary(
            pds->basis_aux.get_atom_nbs(),
            pds->basis_aux_shrink.get_atom_nbs(),
            pds->atoms.types);
    }
    profiler.stop("read_shrink_sinvS_fold");
}

size_t read_shrink_sinvS(const string &dir_path, const string &vq_fprefix,
                         std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using librpa_int::global::lib_printf;
    using librpa_int::global::myid_global;
    using librpa_int::global::profiler;
    using std::cout;
    using std::endl;

    size_t vq_discard = 0;
    auto files = librpa_int::discover_files_with_prefix(dir_path, vq_fprefix);
    if (files.empty())
    {
        throw LIBRPA_RUNTIME_ERROR("No shrink_sinvS files found with prefix " + vq_fprefix +
                                   " under: " + dir_path);
    }

    profiler.start("handle_sinvS_file");
    for (const auto &file_path : files)
    {
        int retcode = 0;
        if (sinvS_file_has_v1_marker(file_path))
        {
            if (myid_global == 0)
            {
                cout << "sinvS: reader v1 binary files detected" << endl;
            }
            retcode = handle_sinvS_v1_file(file_path, sinvS);
        }
        else
        {
            const bool binary = check_coulomb_file_binary(file_path);
            if (myid_global == 0)
            {
                cout << "sinvS: " << (binary ? "Unformatted binary" : "ASCII")
                     << " legacy files detected" << endl;
            }
            retcode = handle_sinvS_file(file_path, sinvS, binary);
        }
        if (retcode != 0)
        {
            lib_printf(LIBRPA_VERBOSE_CRITICAL, "Error encountered when reading %s, return code %d",
                       file_path.c_str(), retcode);
            throw LIBRPA_RUNTIME_ERROR("Failed to read shrink_sinvS file " + file_path +
                                       ", return code " + std::to_string(retcode));
        }
    }
    profiler.stop("handle_sinvS_file");
    return vq_discard;
}

void reader_structure(const std::string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::reader_structure(ctx, file_path);
}

void reader_basis(const std::string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::reader_basis(ctx, file_path);
}

void reader_basis_wfc(const std::string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::reader_basis_wfc(ctx, file_path);
}

void reader_basis_aux(const std::string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::reader_basis_aux(ctx, file_path);
}

void reader_basis_aux_shrink(const std::string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::reader_basis_aux_shrink(ctx, file_path);
}

int detect_Cs_reader_version(const std::string &dir_path, const std::string keyword)
{
    auto ctx = driver_reader_context();
    return librpa::reader::detect_Cs_reader_version(ctx, dir_path, keyword);
}

size_t read_Cs(const std::string &dir_path, double threshold,
               const std::vector<librpa_int::atpair_t> &local_atpair, const std::string keyword,
               int reader_version)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_Cs(ctx, dir_path, threshold, local_atpair, keyword, reader_version);
}

size_t read_Cs_evenly_distribute(const std::string &dir_path, double threshold, int myid,
                                 int nprocs, const std::string keyword, int reader_version)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_Cs_evenly_distribute(ctx, dir_path, threshold, myid, nprocs,
                                                     keyword, reader_version);
}

void get_natom_ncell_from_first_Cs_file(int &n_atom, int &n_cell, const std::string &dir_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::get_natom_ncell_from_first_Cs_file(ctx, n_atom, n_cell, dir_path);
}

std::vector<size_t> read_aux_basis_from_Cs(const std::string &dir_path, const std::string &keyword)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_aux_basis_from_Cs(ctx, dir_path, keyword);
}

void read_basis_from_Cs(const std::string &dir_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_basis_from_Cs(ctx, dir_path);
}

size_t read_Vq_full(const std::string &dir_path, const std::string &vq_fprefix, bool is_cut_coulomb,
                    int reader_version, bool use_shrink_basis)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_Vq_full(ctx, dir_path, vq_fprefix, is_cut_coulomb, reader_version,
                                        use_shrink_basis);
}

size_t read_Vq_row(const std::string &dir_path, const std::string &vq_fprefix, double threshold,
                   const std::vector<librpa_int::atpair_t> &local_atpair, bool is_cut_coulomb,
                   int reader_version, bool use_shrink_basis)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_Vq_row(ctx, dir_path, vq_fprefix, threshold, local_atpair,
                                       is_cut_coulomb, reader_version, use_shrink_basis);
}

int read_eigenvector(const std::string &dir_path)
{
    auto ctx = driver_reader_context();
    auto pds = librpa_int::api::get_dataset_instance(driver::h.get_c_handler());
    const bool direct_kblacs_2d = driver::get_bool(driver::opts.use_kpara_scf_eigvec) &&
                                  driver::opts.parallel_routing == LIBRPA_ROUTING_LIBRI;
    if (direct_kblacs_2d)
    {
        librpa_int::global::profiler.start("driver_read_eigenvector_kblacs_2d");
        const int ret =
            read_eigenvector_kblacs_2d(dir_path, pds->mf, driver::driver_params.use_spinor_wfc,
                                       pds->scfk_blacs_ctxt, pds->desc_wfc_kb);
        if (ret == 0) pds->mark_eigvecs_kpara_2d_ready();
        librpa_int::global::profiler.stop("driver_read_eigenvector_kblacs_2d");
        return ret;
    }

    return librpa::reader::read_eigenvector(ctx, dir_path);
}

int read_eigenvector(const std::string &dir_path, librpa_int::MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> *iks_selected)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_eigenvector(ctx, dir_path, mf, use_spinor_wfc, iks_selected);
}

int read_eigenvector(const std::string &dir_path, librpa_int::MeanField &mf, bool use_spinor_wfc,
                     const std::vector<int> &source_to_target_ik,
                     const std::vector<int> *source_iks_selected, LegacyTextWfcOrder text_order)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_eigenvector(ctx, dir_path, mf, use_spinor_wfc, source_to_target_ik,
                                            source_iks_selected, text_order);
}

int read_eigenvector_kblacs_2d(const std::string &dir_path, librpa_int::MeanField &mf,
                               bool use_spinor_wfc,
                               const librpa_int::KPointBlacsParallelContext &kblacs_ctxt,
                               const librpa_int::ArrayDesc &desc_wfc,
                               const std::vector<int> *source_to_target_ik,
                               LegacyTextWfcOrder text_order)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_eigenvector_kblacs_2d(
        ctx, dir_path, mf, use_spinor_wfc, kblacs_ctxt, desc_wfc, source_to_target_ik, text_order);
}

void read_velocity(const string &file_path, const MeanField &mf, velocity_matrix_t &velocity)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_velocity(ctx, file_path, mf, velocity);
}

void read_velocity(const string &file_path, const MeanField &mf, velocity_matrix_t &velocity,
                   const std::vector<int> &source_to_target_ik, int source_n_kpoints)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_velocity(ctx, file_path, mf, velocity, source_to_target_ik,
                                         source_n_kpoints);
}

void read_velocity_abacus(const MeanField &mf, const std::string &dir_path,
                          const std::string &file_prefix, velocity_matrix_t &velocity)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_velocity_abacus(ctx, mf, dir_path, file_prefix, velocity);
}

void read_velocity_aims(const MeanField &mf, const std::string &file_path,
                        const std::string &file_prefix, velocity_matrix_t &velocity)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_velocity_aims(ctx, mf, file_path, file_prefix, velocity);
}

void read_bz_sampling(const std::string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_bz_sampling(ctx, file_path);
}

void read_bz_sampling_from_stru(const std::string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_bz_sampling_from_stru(ctx, file_path);
}

void read_basis_wfc_aux(const std::string &input_dir, const std::string &fn_basis,
                        const std::string &fn_basis_wfc, const std::string &fn_basis_aux)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_basis_wfc_aux(ctx, input_dir, fn_basis, fn_basis_wfc, fn_basis_aux);
}

void read_band_kpath_info(const string &file_path)
{
    auto ctx = driver_reader_context();
    return librpa::reader::read_band_kpath_info(ctx, file_path);
}

void read_scf_occ_eigenvalues(const string &file_path, MeanField &mf, bool use_spinor_wfc)
{
    librpa::reader::read_scf_occ_eigenvalues(file_path, mf, use_spinor_wfc);
}

void read_scf_occ_eigenvalues(const string &file_path, MeanField &mf, bool use_spinor_wfc,
                              const std::vector<int> &source_ik_for_target, int source_n_kpoints)
{
    librpa::reader::read_scf_occ_eigenvalues(file_path, mf, use_spinor_wfc,
                                           source_ik_for_target, source_n_kpoints);
}
