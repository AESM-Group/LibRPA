// Public API headers
#include "librpa_enums.h"
#include "librpa_compute.h"

// Internal headers
#include "../core/epsilon.h"
#include "../io/fs.h"
#include "../io/global_io.h"
// #include "../io/stl_io_helper.h"
#include "../math/complexmatrix.h"
#include "../utils/profiler.h"
#include "../utils/utils_mem.h"
#include "dataset_helper.h"
#include "instance_manager.h"

#include <string>

void librpa_get_imaginary_frequency_grids(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                          double *omegas, double *weights)
{
    using namespace librpa_int;
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;

    profiler.start("api_get_imaginary_freq_grids");

    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &opts = *p_opts;
    initialize_ds_tfgrids(*pds, opts);
    const auto v_freqs = pds->tfg.get_freq_nodes();
    const auto v_wegihts = pds->tfg.get_freq_weights();
    // global::ofs_myid << v_freqs << std::endl;
    memcpy(omegas, v_freqs.data(), opts.nfreq * sizeof(double));
    memcpy(weights, v_wegihts.data(), opts.nfreq * sizeof(double));

    profiler.stop("api_get_imaginary_freq_grids");
}

double librpa_get_rpa_correlation_energy(LibrpaHandler *h, const LibrpaOptions *p_opts,
                                         int n_ibz_kpoints, double *rpa_corr_ibzk_contrib_re,
                                         double *rpa_corr_ibzk_contrib_im)
{
    using namespace librpa_int;
    using librpa_int::global::lib_printf;
    using librpa_int::global::profiler;

    double rpa_corr = 0.0;

    profiler.start("api_get_rpa_correlation_energy");
    auto pds = librpa_int::api::get_dataset_instance(h);
    const auto &opts = *p_opts;
    initialize_ds_global_ddla(*pds, opts);

    const auto ks_local = pds->mf.get_iks_local();
    const bool seems_k_para = as_int(ks_local.size()) < pds->mf.get_n_kpoints();
    if (seems_k_para)
    {
        global::ofs_myid << "KS eigenvectors of SCF start point seem distributed over k-points" << std::endl;
        if (opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON)
        {
            global::ofs_myid << "Option use_kpara_scf_eigvec is on: input consistent" << std::endl;
        }
        else
        {
            global::ofs_myid << "Option use_kpara_scf_eigvec is off while the eigenvectors are distributed" << std::endl;
            throw LIBRPA_RUNTIME_ERROR("inconsistent input");
        }
    }

    const bool debug = global::should_output(LIBRPA_VERBOSE_DEBUG);

    // Prepare time-frequency grids
    initialize_ds_tfgrids(*pds, opts);

    // Decide actual routing
    LibrpaParallelRouting routing = opts.parallel_routing;
    if (routing == LIBRPA_ROUTING_AUTO)
    {
        const int n_atoms = pds->atoms.size();
        routing = decide_auto_routing(n_atoms, opts.nfreq * pds->pbc.get_n_cells_bvk());
    }

    if (opts.use_kpara_scf_eigvec == LIBRPA_SWITCH_ON)
    {
        if (routing == LIBRPA_ROUTING_LIBRI)
            pds->redistribute_eigvecs_kpara_2d();
        else
            pds->redistribute_eigvecs_kpara();
    }

    // Determine the atom pairs that this process is responsible for
    initialize_ds_atpairs_local(*pds, routing);

    // Redistribute 2D Coulomb matrices to atom-pair blocks if they are parsed
    pds->redistribute_coulomb_blacs2ap();

    const std::string rpa_headwing_mode(opts.rpa_headwing_mode);
    if (rpa_headwing_mode != "qavg" && rpa_headwing_mode != "head_only")
    {
        throw LIBRPA_RUNTIME_ERROR("rpa_headwing_mode must be qavg or head_only");
    }
    const bool need_rpa_wing =
        opts.option_dielect_func == 3 && rpa_headwing_mode != "head_only";
    initialize_ds_headwing(*pds, opts, need_rpa_wing);

    // Initialize response function object
    initialize_ds_chi0(*pds, opts);
    auto &chi0 = *(pds->p_chi0);

    // std::cout << "n_abf & " << chi0.atbasis_abf.nb_total;
    // std::cout << "n_abf * " << pds->p_chi0->atbasis_abf.nb_total;

    profiler.start("chi0_build", "Build response function chi0");
    chi0.build(routing, pds->cs_data, pds->atpairs_local, pds->basis_aux, pds->sinvS,
               pds->blacs_h);
    profiler.stop("chi0_build");
    export_ds_chi0(*pds, opts, routing);

    if (debug)
    { // debug, check chi0
        profiler.start("chi0_write_matrix_mm", "Export chi0 matrices");
        char fn[80];
        for (const auto &[freq, q_IJchi]: chi0.get_chi0_q())
        {
            const int ifreq = chi0.tfg.get_freq_index(freq);
            for (const auto &[q, I_Jchi]: q_IJchi)
            {
                const int iq = pds->pbc.get_k_index_ibz(q);
                for (const auto &[I, J_chi]: I_Jchi)
                {
                    for (const auto &[J, chi]: J_chi)
                    {
                        sprintf(fn, "chi0fq_ifreq_%d_iq_%d_I_%zu_J_%zu_id_%d.mtx", ifreq, iq, I, J, pds->comm_h.myid);
                        print_complex_matrix_mm(chi, path_as_directory(opts.output_dir) + fn);
                    }
                }
            }
        }
        profiler.stop("chi0_write_matrix_mm");
    }

    profiler.start("chi0_release_free");
    release_free_mem();
    pds->comm_h.barrier();
    profiler.stop("chi0_release_free");

    profiler.start("EcRPA", "Compute RPA correlation Energy");
    CorrEnergy corr;

    const bool use_blacs = opts.use_scalapack_ecrpa && (routing == LIBRPA_ROUTING_ATOMPAIR || routing == LIBRPA_ROUTING_LIBRI);
    const bool use_rpa_headwing =
        opts.replace_w_head == LIBRPA_SWITCH_ON && opts.option_dielect_func == 3 &&
        pds->p_headwing != nullptr;

    if (use_blacs)
    {
        if(pds->mf.get_n_kpoints() == 1 && !use_rpa_headwing)
            corr = compute_RPA_correlation_blacs_2d_gamma_only(chi0, pds->vq, pds->atpairs_local, pds->blacs_h, opts.use_gpu_replace_scalapack);
        else
        {
            RpaHeadwingSettings headwing_settings;
            headwing_settings.enabled = use_rpa_headwing;
            headwing_settings.option_dielect_func = opts.option_dielect_func;
            headwing_settings.use_2d_dielectric = opts.use_2d_dielectric == LIBRPA_SWITCH_ON;
            headwing_settings.rpa_headwing_body_start = opts.rpa_headwing_body_start;
            headwing_settings.rpa_headwing_mode = rpa_headwing_mode;
            headwing_settings.sqrt_coulomb_threshold = opts.sqrt_coulomb_threshold;
            corr = compute_RPA_correlation_blacs_2d(chi0, pds->vq, pds->atpairs_local,
                                                    pds->blacs_h, headwing_settings,
                                                    pds->p_headwing.get());
        }
    }
    else
        corr = compute_RPA_correlation(routing, *(pds->p_chi0), pds->vq);

    rpa_corr = corr.value.real();
    if (corr.qcontrib.size() != as_size(n_ibz_kpoints))
    {
        if (pds->comm_h.is_root())
        {
            lib_printf(
                "WARNING: parsed n_ibz_kpoints is not consistent with generated CorrEne object: %d != %zu\n",
                as_size(n_ibz_kpoints), corr.qcontrib.size());
        }
    }

    for (const auto &[q, corr_q]: corr.qcontrib)
    {
        const auto iq = pds->pbc.get_k_index_ibz(q);
        if (iq == -1)
            throw LIBRPA_RUNTIME_ERROR("Internal error, failed to find irreducible k-point");
        rpa_corr_ibzk_contrib_re[iq] = corr_q.real();
        rpa_corr_ibzk_contrib_im[iq] = corr_q.imag();
    }
    profiler.stop("EcRPA");

    // Works done, free the object and reset the pointer to nullptr
    pds->p_chi0.reset();
    profiler.stop("api_get_rpa_correlation_energy");

    return rpa_corr;
}
