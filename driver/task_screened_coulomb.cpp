#include "task_screened_coulomb.h"

#include "chi0.h"
#include "driver_params.h"
#include "driver_utils.h"
#include "envs_mpi.h"
#include "epsilon.h"
#include "params.h"
#include "pbc.h"
#include "profiler.h"
#include "read_data.h"
#include "utils_io.h"
#include "utils_timefreq.h"

void task_screened_coulomb_real_freq(std::map<Vector3_Order<double>, ComplexMatrix> &sinvS)
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::utils::lib_printf;

    Profiler::start("Wc_Rf", "Build Screened Coulomb: R and freq. space");

    Vector3_Order<int> period{kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    vector<Vector3_Order<double>> qlist;
    for (auto q_weight : irk_weight)
    {
        qlist.push_back(q_weight.first);
    }

    // Prepare time-frequency grids
    auto tfg =
        LIBRPA::utils::generate_timefreq_grids(Params::nfreq, Params::tfgrids_type, meanfield);

    Chi0 chi0(meanfield, klist, tfg);
    chi0.gf_R_threshold = Params::gf_R_threshold;

    Profiler::start("chi0_build", "Build response function chi0");
    chi0.build(Cs_data, Rlist, period, local_atpair, qlist, sinvS);
    Profiler::stop("chi0_build");

    Profiler::start("read_vq_cut", "Load truncated Coulomb");
    read_Vq_full(driver_params.input_dir, "coulomb_cut_", true,
                 driver_params.version_coul_reader, Params::use_shrink_abfs);
    Profiler::stop("read_vq_cut");

    std::vector<double> epsmac_LF_imagfreq_re;
    if (Params::replace_w_head)
    {
        std::vector<double> omegas_dielect;
        std::vector<double> dielect_func;
        read_dielec_func(driver_params.input_dir + "dielecfunc_out", omegas_dielect, dielect_func);

        epsmac_LF_imagfreq_re = interpolate_dielec_func(Params::option_dielect_func, omegas_dielect,
                                                        dielect_func, chi0.tfg.get_freq_nodes());

        if (Params::debug)
        {
            if (mpi_comm_global_h.is_root())
            {
                lib_printf("Dielection function parsed:\n");
                for (int i = 0; i < chi0.tfg.get_freq_nodes().size(); i++)
                    lib_printf("%d %f %f\n", i + 1, chi0.tfg.get_freq_nodes()[i],
                               epsmac_LF_imagfreq_re[i]);
            }
            mpi_comm_global_h.barrier();
        }
    }

    Profiler::start("compute_Wc_freq_q", "Build upper half of Wc(q,w)");
    vector<std::complex<double>> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(),
                                                    epsmac_LF_imagfreq_re.cend());
    map<double,
        atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old>
        Wc_freq_q;
    if (Params::use_scalapack_gw_wc)
        Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
    else
        Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
    Profiler::stop("compute_Wc_freq_q");

    Profiler::start("construct_Wc_lower_half", "Construct Lower Half of Wc(q,w)");
    // NOTE: only upper half of Wc is built now
    //       here we recover the other half before transform to R space using the Hermitian property
    for (int ifreq = 0; ifreq < chi0.tfg.get_n_grids(); ifreq++)
    {
        const auto freq = chi0.tfg.get_freq_nodes()[ifreq];
        auto &Wc = Wc_freq_q.at(freq);
        vector<atom_t> iatoms_row;
        for (const auto &Wc_IJ : Wc) iatoms_row.push_back(Wc_IJ.first);
        for (auto iatom_row : iatoms_row)
        {
            vector<atom_t> iatoms_col;
            for (const auto &Wc_J : Wc.at(iatom_row))
            {
                iatoms_col.push_back(Wc_J.first);
            }
            for (auto iatom_col : iatoms_col)
            {
                if (iatom_row == iatom_col) continue;
                for (const auto &Wc_q : Wc.at(iatom_row).at(iatom_col))
                {
                    Wc[iatom_col][iatom_row][Wc_q.first] = Wc_q.second.get_transpose(true);
                }
            }
        }
    }
    Profiler::stop("construct_Wc_lower_half");

    Profiler::start("FT_Wc_freq_q", "Fourier Transform Wc(q,w) -> Wc(R,w)");
    const auto Wc_freq_MN_R = FT_Wc_freq_q(Wc_freq_q, chi0.tfg, meanfield.get_n_kpoints(), Rlist);
    Profiler::stop("FT_Wc_freq_q");

    Profiler::start("write_Wc_freq_R", "Export Wc(R,w) to file");
    for (const auto &freq_MuNuRWc : Wc_freq_MN_R)
    {
        char fn[80];
        auto freq = freq_MuNuRWc.first;
        auto ifreq = chi0.tfg.get_freq_index(freq);
        for (const auto &Mu_NuRWc : freq_MuNuRWc.second)
        {
            auto Mu = Mu_NuRWc.first;
            // const int n_mu = atom_mu[Mu];
            for (const auto &Nu_RWc : Mu_NuRWc.second)
            {
                auto Nu = Nu_RWc.first;
                // const int n_nu = atom_mu[Nu];
                for (const auto &R_Wc : Nu_RWc.second)
                {
                    auto R = R_Wc.first;
                    auto Wc = R_Wc.second;
                    auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
                    auto iR = std::distance(Rlist.cbegin(), iteR);
                    sprintf(fn, "Wc_Mu_%zu_Nu_%zu_iR_%zu_ifreq_%d.mtx", Mu, Nu, iR, ifreq);
                    print_matrix_mm_file(Wc, Params::output_dir + "/" + fn, "", 1e-10);
                }
            }
        }
    }
    Profiler::stop("write_Wc_freq_R");
    Profiler::stop("Wc_Rf");
}
