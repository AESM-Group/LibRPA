// ===================
//  Author: Ziqing Guan
//  date: 2026.08.02
// ===================

#pragma once

#include "LRI_k.h"
#include "LRI_Cal_Aux.h"
#include "../global/Array_Operator.h"
#include "../global/Tensor_Multiply.h"
#include <cmath>
#include <omp.h>
#include "../global/Memory.h"
#ifdef __MKL_RI
#include <mkl_service.h>
#endif

namespace RI
{
template<typename TA, typename Tcell, std::size_t Ndim, typename Tdata>
std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>> // H(s,t)[k]
LRI_k<TA, Tcell, Ndim, Tdata>::cal_cvcd_k_hartree(
	const std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>>& Ds,  // D(s,t)[k]
	const std::vector<Tk>& kindex_map,// k index to direct coordinate array<double, Ndim>
	const std::vector<int>& k_indices,
	const std::vector<TA>& list_I,
	const std::vector<TA>& list_J,
	const std::vector<TA>& list_IJ,
	const std::string& save_name_C,
	const std::string& save_name_V)
{
#ifdef __MKL_RI
	const std::size_t mkl_threads = mkl_get_max_threads();
	mkl_set_num_threads(1);
#endif
	const int nk = static_cast<int>(kindex_map.size());
	assert (nk == this->period[0] * this->period[1] * this->period[2]);
    std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>> hartree_k;
    std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>> Csk;
	std::map<TA, std::map<TA, Tensor<Tdata>>> Vq; // has only one q=0
	std::map<TA, Tensor<Tdata>> M_nu; // M_nu = \sum_{uvk} (C^nu_u_v[k] + C^nu*_v_u[k]) D_v_u[k]

	const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Vs = this->data_pool.at(save_name_V).Ds_ab;
	const std::map<TA, std::map<TAC, Tensor<Tdata>>>& Cs = this->data_pool.at(save_name_C).Ds_ab;

	// add thread lock for the TA key of Vq
	std::map<TA, omp_lock_t> lock_vq_result_add_map = LRI_Cal_Aux::init_lock_result(Vq, list_IJ);
	const std::vector<TAC> list_IJR = Divide_Atoms::traversal_atom_period(list_IJ, this->period);
	// add thread lock for the TA key of Csk
	std::map<TA, omp_lock_t> lock_csk_result_add_map = LRI_Cal_Aux::init_lock_result(Csk, list_IJ);
	
	// add thread lock for the TA key of M_nu
	std::map<TA, omp_lock_t> lock_m_result_add_map = LRI_Cal_Aux::init_lock_result(M_nu, list_IJ);
	#pragma omp parallel
	{
		Tk q{0.0, 0.0, 0.0};
		// 1. FT V_mu_nu <I,<J,R>> to V_mu_nu <q=0,<I,J>>
		std::map<TA, std::map<TA, Tensor<Tdata>>> Vq_thread;

		for (const TA mu : list_I)
		{
			const auto& V_mu = Vs.at(mu);
			auto& Mu_Vq_nu_thread = Vq_thread[mu];
	#pragma omp for schedule(dynamic) nowait
			for (int nu_R_index=0; nu_R_index<list_IJR.size(); nu_R_index++)
			{
				const TAC& nu_R = list_IJR[nu_R_index];
				const Tensor<Tdata>& V_mu_nu_R = Global_Func::find(V_mu, nu_R);
				if (V_mu_nu_R.empty()) continue;
				const TA nu = nu_R.first;
				const TC R = nu_R.second;
				double arg = 2.0 * M_PI * (q[0] * R[0] + q[1] * R[1] + q[2] * R[2]);
				std::complex<double> fac (cos(arg), sin(arg));
				LRI_Cal_Aux::add_Ds(V_mu_nu_R, Mu_Vq_nu_thread[nu], Global_Func::convert<Tdata>(fac));
			}
			LRI_Cal_Aux::add_Ds_omp_try_map(Vq_thread, Vq, lock_vq_result_add_map, 1.0);
		}

        // 2. FT CsR <I,<J,R>> to Csk <I,<J,<k tensor{nabf, nwt1, nwt2}>>>
        std::map<TA, std::map<TA, std::map<int, Tensor<Tdata>>>> Csk_thread;
		for (const TA mu : list_IJ)
		{
			const auto& Cs_mu = Cs.at(mu);
			auto& Mu_C_nu_k_thread = Csk_thread[mu];
	#pragma omp for schedule(dynamic) nowait
			for (int nu_R_index=0; nu_R_index<list_IJR.size(); nu_R_index++)
			{
				const TAC& nu_R = list_IJR[nu_R_index];
				const Tensor<Tdata>& Cs_mu_nu_R = Global_Func::find(Cs_mu, nu_R);
				if (Cs_mu_nu_R.empty()) continue;
				const TA nu = nu_R.first;
				const TC R = nu_R.second;
				auto& Nu_C_k_thread = Mu_C_nu_k_thread[nu];
				for (const int ik: k_indices)
				{
					const Tk k = kindex_map.at(ik);
                    double arg = 2.0 * M_PI * (k[0] * R[0] + k[1] * R[1] + k[2] * R[2]);
                    std::complex<double> fac (cos(arg), sin(arg));
                    LRI_Cal_Aux::add_Ds(Cs_mu_nu_R, Nu_C_k_thread[ik], Global_Func::convert<Tdata>(fac));
                }
            }   
            LRI_Cal_Aux::add_Ds_omp_try_map(Csk_thread, Csk, lock_csk_result_add_map, 1.0);         
        }
        LRI_Cal_Aux::add_Ds_omp_wait_map(Vq_thread, Vq, lock_vq_result_add_map, 1.0);
        LRI_Cal_Aux::add_Ds_omp_wait_map(Csk_thread, Csk, lock_csk_result_add_map, 1.0);
        #pragma omp barrier
		#pragma omp master
		{
            LRI_Cal_Aux::destroy_lock(lock_vq_result_add_map);
            LRI_Cal_Aux::destroy_Ds_result(Vq);
            this->free_tensors_map2(save_name_V);
			LRI_Cal_Aux::destroy_lock(lock_csk_result_add_map);
			LRI_Cal_Aux::destroy_Ds_result(Csk);
            this->free_tensors_map2(save_name_C);
		}
        #pragma omp barrier

		// 3. calculate M_nu =\sum_{uvk} (C^nu_u_v[k] + C^nu*_v_u[k]) D_v_u[k]		
        std::map<TA, Tensor<Tdata>> M_nu_thread;
#pragma omp for schedule(dynamic) collapse(2)
		for (int v_index=0; v_index<list_I.size(); v_index++)
		{
			for (int u_index=0; u_index<list_J.size(); u_index++)
			{
				const TA v = list_I[v_index];
				const TA u = list_J[u_index];
				const std::map<int, Tensor<Tdata>>& D_v_u = Global_Func::find(Ds, v, u);
				if (D_v_u.empty()) continue;
				const std::size_t nwt1 = D_v_u.begin()->second.shape[0];
				const std::size_t nwt2 = D_v_u.begin()->second.shape[1];
				const std::map<int, Tensor<Tdata>>& Csk_u_v = Global_Func::find(Csk, u, v);
				const std::map<int, Tensor<Tdata>>& Csk_v_u = Global_Func::find(Csk, v, u);
				for (const int ik: k_indices)
				{
					if (!Csk_u_v.empty())
					{
						const Tensor<Tdata>& D_v_u_k = Global_Func::find(D_v_u, ik);
						if (D_v_u_k.empty()) continue;
						const Tensor<Tdata>& C_u_v_k = Global_Func::find(Csk_u_v, ik);
						if (C_u_v_k.empty()) continue;
						assert(C_u_v_k.shape[1]==nwt2);
						assert(C_u_v_k.shape[2]==nwt1);
						// M_nu += C^nu_u_v[k] * D_v_u[k]
						LRI_Cal_Aux::add_Ds(Tensor_Multiply::gemv(C_u_v_k, D_v_u_k.transpose()), M_nu_thread[u]);
					}
					if (!Csk_v_u.empty())
					{
						const Tensor<Tdata>& D_v_u_k = Global_Func::find(D_v_u, ik);
						if (D_v_u_k.empty()) continue;
						const Tensor<Tdata>& C_v_u_k = Global_Func::find(Csk_v_u, ik);
						if (C_v_u_k.empty()) continue;
						assert(C_v_u_k.shape[1]==nwt1);
						assert(C_v_u_k.shape[2]==nwt2);
						// M_nu += C^nu*_v_u[k] * D_v_u[k]
						LRI_Cal_Aux::add_Ds(Tensor_Multiply::gemv(C_v_u_k.conjugate(), D_v_u_k), M_nu_thread[v]);
					}
				}
			}
		}
		LRI_Cal_Aux::add_Ds_omp_wait_map(M_nu_thread, M_nu, lock_m_result_add_map, 1.0);
	}// end #pragma omp parallel
	LRI_Cal_Aux::destroy_lock(lock_m_result_add_map);
	LRI_Cal_Aux::destroy_Ds_result(M_nu);
#ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
#endif

	// 4. calculate N_mu = 1/Nk \sum_nu Vq_mu_nu * M_nu, and communicate N_mu
	std::map<TA, Tensor<Tdata>> N_mu;
	for (int mu : list_I)
	{
		const auto& Vq_mu = Vq.at(mu);
		Tensor<Tdata>& N_Mu = N_mu[mu];
		for (const TA nu : list_IJ)
		{
			const Tensor<Tdata>& Vq_mu_nu = Global_Func::find(Vq_mu, nu);
			if (Vq_mu_nu.empty()) continue;
			const Tensor<Tdata>& M_Nu = M_nu.at(nu);
			LRI_Cal_Aux::add_Ds(Tensor_Multiply::gemv(Vq_mu_nu, M_Nu), N_Mu);
		}
		const Tdata fac = Tdata(1.0 / static_cast<double>(nk));
		N_Mu = fac * N_Mu;
	}
	std::set<TA> set_IJ(list_IJ.begin(), list_IJ.end());
	N_mu = Communicate_Tensors_Map_Judge::comm_map(this->mpi_comm, std::move(N_mu), set_IJ);
	
	// 5. calculate H_st[k] = \sum_mu (C^mu_s_t[k] + C^mu*_t_s[k]) N_mu
	for (const TA s : list_I)
	{
		std::map<TA, std::map<int, Tensor<Tdata>>>& hartree_s = hartree_k[s];
		for (const TA t : list_J)
		{
			std::map<int, Tensor<Tdata>>& hartree_st = hartree_s[t];
			for (const int ik: k_indices)
			{
				hartree_st[ik];
			}
		}
	}
#ifdef __MKL_RI
	mkl_set_num_threads(1);
#endif

#pragma omp parallel for schedule(dynamic) collapse(2)
	for (int s_index=0; s_index<list_I.size(); s_index++)
	{
		for (int t_index=0; t_index<list_J.size(); t_index++)
		{
			const TA s = list_I[s_index];
			const TA t = list_J[t_index];
			auto& hartree_st = hartree_k.at(s).at(t);
			const std::map<int, Tensor<Tdata>>& Csk_s_t = Global_Func::find(Csk, s, t);
			const std::map<int, Tensor<Tdata>>& Csk_t_s = Global_Func::find(Csk, t, s);
			const Tensor<Tdata>& N_Mu_at_s = N_mu.at(s);
			const Tensor<Tdata>& N_Mu_at_t = N_mu.at(t);

			for (const int ik: k_indices)
			{
				if (!Csk_s_t.empty())
				{
					const Tensor<Tdata>& C_s_t_k = Global_Func::find(Csk_s_t, ik);
					if (C_s_t_k.empty()) continue;
					// H_st[k] += C^mu_s_t[k] * N_mu
					LRI_Cal_Aux::add_Ds(Tensor_Multiply::gemv_trans(C_s_t_k, N_Mu_at_s), hartree_st.at(ik));
				}
				if (!Csk_t_s.empty())
				{
					const Tensor<Tdata>& C_t_s_k = Global_Func::find(Csk_t_s, ik);
					if (C_t_s_k.empty()) continue;
					// H_st[k] += C^mu*_t_s[k] * N_mu
					Tensor<Tdata> tmp = Tensor_Multiply::gemv_trans(C_t_s_k.conjugate(), N_Mu_at_t);
					LRI_Cal_Aux::add_Ds(tmp.transpose(), hartree_st.at(ik));
				}
			}
		}
	}

#ifdef __MKL_RI
	mkl_set_num_threads(mkl_threads);
#endif

	Global_Func::release_free_memory();
	return hartree_k;
}

}	// end namespace RI
