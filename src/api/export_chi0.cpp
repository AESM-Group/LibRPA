#include "dataset_helper.h"
#include "../core/chi0.h"
#include "../core/epsilon.h"
#include "../io/global_io.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

namespace librpa_int {
namespace {
void write_response(const Chi0 &chi, const std::string &directory, double omega) {
    const auto &basis=chi.atbasis_abf; const int n=basis.nb_total;
    auto pbc=chi.pbc; pbc.map_irk_ks.clear();
    const auto &qlist=chi.active_qpoints();
    auto same=[](const auto &a,const auto &b){return std::abs(a.x-b.x)+std::abs(a.y-b.y)+std::abs(a.z-b.z)<1.e-8;};
    // The export format is deliberately in the ORIGINAL real auxiliary basis.
    // Without rotation metadata only a full mesh or time-reversal pairs can
    // be restored. Do not apply conjugation to arbitrary space-group stars.
    for(const auto &q:chi.pbc.klist_full) {
        auto rep=std::find_if(qlist.begin(),qlist.end(),[&](const auto &qr){return same(q,qr);});
        if(rep==qlist.end()) {
            rep=std::find_if(qlist.begin(),qlist.end(),[&](const auto &qr){
                const auto f=(q+qr)*pbc.latvec.Transpose();
                return std::abs(f.x-std::round(f.x))+std::abs(f.y-std::round(f.y))+std::abs(f.z-std::round(f.z))<1.e-8;
            });
        }
        if(rep==qlist.end()) throw std::runtime_error("chi0 export requires full/inverse q coverage; disable general chi0 symmetry");
        pbc.map_irk_ks[*rep].push_back(q);
    }
    atom_mapping<std::map<Vector3_Order<double>,Matz>>::pair_t_old cq;
    for(const auto &q:qlist) {
        Matz dense(n,n,MAJOR::ROW); std::vector<int> counts(n*n,0);
        const auto fq=chi.get_chi0_q().find(omega);
        if(fq!=chi.get_chi0_q().end()) {
            const auto qi=fq->second.find(q);
            if(qi!=fq->second.end()) for(const auto &[i,js]:qi->second) for(const auto &[j,block]:js)
                for(int a=0;a<block.nr;++a) for(int b=0;b<block.nc;++b) {
                    const int mu=basis.get_global_index(i,a),nu=basis.get_global_index(j,b);
                    dense(mu,nu)=block(a,b); counts[mu*n+nu]=1;
                    if(i!=j){dense(nu,mu)=std::conj(block(a,b));counts[nu*n+mu]=1;}
                }
        }
        MPI_Allreduce(MPI_IN_PLACE,dense.ptr(),n*n,MPI_C_DOUBLE_COMPLEX,MPI_SUM,chi.comm_h.comm);
        MPI_Allreduce(MPI_IN_PLACE,counts.data(),n*n,MPI_INT,MPI_SUM,chi.comm_h.comm);
        for(int k=0;k<n*n;++k){
            if(!counts[k]) throw std::runtime_error("missing chi0 atom block during real-space export");
            dense.ptr()[k]/=counts[k];
        }
        if(chi.comm_h.is_root()) for(int i=0;i<basis.n_atoms;++i) for(int j=0;j<basis.n_atoms;++j){
            Matz block(basis[i],basis[j],MAJOR::ROW);
            for(std::size_t a=0;a<block.nr();++a) for(std::size_t b=0;b<block.nc();++b)
                block(a,b)=dense(basis.get_global_index(i,a),basis.get_global_index(j,b));
            cq[i][j][q]=std::move(block);
        }
    }
    // The existing LibRPA linear spatial FT applies equally to bare chi0.
    // chi0(R,iw) = (1/Nq) sum_q exp(-2*pi*i*q.R) chi0(q,iw).
    // These are unscreened response matrices, not epsilon, W, or Wc.
    const auto cr=FT_Wc_q2R(chi.comm_h,basis,chi.symmetry_context,cq,chi.tfg,pbc,pbc.Rlist,true,"",false);
    if(!chi.comm_h.is_root()) return;
    std::filesystem::create_directories(directory);
    for(const auto &[i,js]:cr) for(const auto &[j,rs]:js) for(const auto &[r,m]:rs) {
        const auto path=std::filesystem::path(directory)/("Chi0_Mu_"+std::to_string(i)+"_Nu_"+std::to_string(j)+"_iR_"+std::to_string(pbc.get_R_index(r))+"_ifreq_0.mtx");
        std::ofstream out(path);
        out<<"%%MatrixMarket matrix coordinate complex general\n%\n% bare chi0; R = ( "<<r.x<<' '<<r.y<<' '<<r.z<<" )\n"
           <<m.nr()<<' '<<m.nc()<<' '<<m.nr()*m.nc()<<'\n'<<std::scientific<<std::setprecision(17);
        for(std::size_t a=0;a<m.nr();++a) for(std::size_t b=0;b<m.nc();++b)
            out<<a+1<<' '<<b+1<<' '<<m(a,b).real()<<' '<<m(a,b).imag()<<'\n';
        if(!out) throw std::runtime_error("failed writing chi0 real-space matrix");
    }
    // ifreq_0 labels the single exported node; physical omega is authoritative.
    std::ofstream info(std::filesystem::path(directory)/"chi0_rf.info");
    info<<"LIBRPA_CHI0_RF 1\n"<<std::setprecision(17)<<omega<<' '<<n<<' '<<pbc.period.x<<' '<<pbc.period.y<<' '<<pbc.period.z<<'\n';
    if(!info) throw std::runtime_error("failed writing chi0_rf.info");
    global::lib_printf_root("Exported bare chi0(R,iw), omega = %.16e Ha, to %s\n",omega,directory.c_str());
}
}
void export_ds_chi0(Dataset &ds,const LibrpaOptions &opts,LibrpaParallelRouting routing) {
    if(opts.output_chi0_rf!=LIBRPA_SWITCH_ON && opts.output_chi0_static!=LIBRPA_SWITCH_ON) return;
    if(opts.use_shrink_abfs==LIBRPA_SWITCH_ON)
        throw std::runtime_error("chi0 export currently requires uncompressed auxiliary basis");
    auto &chi=*ds.p_chi0;
    if(opts.output_chi0_rf==LIBRPA_SWITCH_ON) {
        const auto frequencies=chi.tfg.get_freq_nodes();
        const double omega=*std::min_element(frequencies.begin(),frequencies.end(),[](double a,double b){return std::abs(a)<std::abs(b);});
        write_response(chi,std::string(opts.output_dir)+"/chi0_lowest",omega);
    }
    if(opts.output_chi0_static==LIBRPA_SWITCH_ON) {
        // A dedicated response build leaves the GW quadrature and chi0 intact.
        // omega=0 is evaluated by imaginary-time integration, not by renaming
        // the first positive minimax frequency or extrapolating matrix entries.
        TFGrids grid; grid.generate_static_export(ds.tfg);
        Chi0 stat(ds.mf,ds.basis_wfc,ds.basis_aux,ds.pbc,ds.symmetry_context,grid,
                  ds.scfk_blacs_ctxt,chi.desc_wfc,opts.use_kpara_scf_eigvec==LIBRPA_SWITCH_ON,
                  opts.use_symmetry_rpa==LIBRPA_SWITCH_ON);
        stat.gf_threshold=chi.gf_threshold; stat.nbands_G=chi.nbands_G;
        stat.n_bands_exclude=chi.n_bands_exclude;
        stat.libri_threshold_C=chi.libri_threshold_C; stat.libri_threshold_G=chi.libri_threshold_G;
        stat.libri_collect_s0_chunk=chi.libri_collect_s0_chunk; stat.libri_collect_max_bytes=chi.libri_collect_max_bytes;
        std::map<Vector3_Order<double>,ComplexMatrix> empty;
        stat.build(routing,ds.cs_data,ds.atpairs_local,ds.basis_aux,empty,ds.blacs_h);
        write_response(stat,std::string(opts.output_dir)+"/chi0_static",0.0);
    }
}
}
