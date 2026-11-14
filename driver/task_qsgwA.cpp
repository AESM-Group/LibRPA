#include "task_qsgwA.h"
#include "task_qsgw.h"
// 标准库头文件
#include <iostream>         // 用于输入输出操作
#include <map>              // 用于std::map容器
#include <string>           // 用于std::string类
#include <fstream> // 用于文件存在检查
#include <sstream>
#include <iomanip> // 用于格式化
#include <vector>
#include <cmath>
// 自定义头文件     
#include "envs_mpi.h"     
#include "envs_io.h"        
#include "utils_io.h"
#include "meanfield.h"              // MeanField类相关
#include "params.h"                 // 参数设置相关
#include "pbc.h"                    // 周期性边界条件相关
#include "chi0.h"                   // 响应函数相关
#include "gw.h"                     // GW计算相关
#include "analycont.h"              // 分析延拓相关
#include "qpe_solver.h"             // 准粒子方程求解器
#include "epsilon.h"                // 介电函数相关
#include "exx.h"                    // Exact exchange相关
#include "constants.h"              // 常量定义
#include "coulmat.h"                // 库仑矩阵相关
#include "profiler.h"               // 性能分析工具
#include "ri.h"     

#include "envs_mpi.h"
#include "envs_io.h"
#include "utils_timefreq.h"

#include "read_data.h"       
#include "write_aims.h"    
#include "driver_params.h"      
#include "driver_utils.h"
#include "matrix.h"     
#include "fermi_energy_occupation.h"// 费米能和占据数计算相关
#include "convert_csc.h"
#include "Hamiltonian.h"            // 哈密顿量相关




void task_qsgwA()
{
    using LIBRPA::envs::mpi_comm_global_h;
    using LIBRPA::utils::lib_printf;
 

    Profiler::start("qsgwA", "QSGW quasi-particle calculation");

    Vector3_Order<int> period {kv_nmp[0], kv_nmp[1], kv_nmp[2]};
    auto Rlist = construct_R_grid(period);

    vector<Vector3_Order<double>> qlist;
    for (auto q_weight: irk_weight)
    {
        qlist.push_back(q_weight.first);
    }

    

    
    const auto n_spins = meanfield.get_n_spins();
    const auto n_bands = meanfield.get_n_bands();
    const auto n_kpoints = meanfield.get_n_kpoints();
    const auto n_aos = meanfield.get_n_aos();


    // 初始化
    Profiler::start("read_vxc_HKS");
    std::map<int, std::map<int, Matz>> hf_nao;  
    std::map<int, std::map<int, Matz>> vxc;  
    std::map<int, std::map<int, Matz>> hf;
    std::map<int, std::map<int, Matz>> vxc0;
    std::map<int, std::map<int, Matz>> vxc1;
    std::map<int, std::map<int, Matz>> exx0;
    std::map<int,std::map<int, std::map<int, Matz>>> Hexx_matrix_temp;
    std::map<int, std::map<int, Matz>> H_KS; // H_KS矩阵
    std::map<int, std::map<int, Matz>> H_KS0;
    std::map<int, std::map<int, Matz>> H_KS1;//用于混合迭代
    bool all_files_processed_successfully = true;
    const std::string final_banner(90, '-');

        // 自旋和 k 点的循环，读取初始数据
    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            std::map<std::string, Matz> arrays;
            std::string key_hf, key_vxc;

            // 使用 ostringstream 构建文件名
            std::ostringstream oss_hf, oss_vxc;
            oss_hf << "hf_exchange_spin_0" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";
            oss_vxc << "xc_matr_spin_" << (ispin + 1) << "_kpt_" << std::setw(6) << std::setfill('0') << (ikpt + 1) << ".csc";

            std::string hfFilePath = oss_hf.str();
            std::string vxcFilePath = oss_vxc.str();
            
            Matz wfc1(n_bands, n_aos, MAJOR::COL);
            for (int ib = 0; ib < n_bands; ++ib) {
                for (int iao = 0; iao < n_aos; iao++) {
                    wfc1(ib, iao) = meanfield.get_eigenvectors()[ispin][ikpt](ib, iao);
                    meanfield.get_eigenvectors0()[ispin][ikpt](ib, iao) = wfc1(ib, iao);
                    
                }
            }
            //check 正交性
            // Matz wfc7(n_bands, n_aos, MAJOR::COL);
            // wfc7 = transpose(wfc1,false) ;
            // Matz wfc8(n_bands, n_aos, MAJOR::COL);
            // wfc8 = conj(wfc1);
            // Matz wfc0(n_bands, n_aos, MAJOR::COL);
            // wfc0 = wfc8 * wfc7 ;
            // printf("wfc0_real:\n");
            // for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         const auto &wfc0_value = wfc0(i,j) ;
            //         printf("%16.6f ", wfc0_value.real()); 
            //     }
            //     printf("\n");
            // }
            // printf("wfc0_imag:\n");
            // printf("\n");
            // for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         const auto &wfc0_value = wfc0(i,j) ;
            //         printf("%16.6f ", wfc0_value.imag()); 
            //     }
            //     printf("\n");
            // }
            // printf("\n"); // 换行

            hf_nao[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);  
            vxc0[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);     
            // 初始化 hf 和 vxc 矩阵为零矩阵
            for (int i = 0; i < n_aos; ++i) {
                for (int j = 0; j < n_aos; ++j) {
                    hf_nao[ispin][ikpt](i, j) = 0.0;
                    vxc0[ispin][ikpt](i, j) = 0.0;
                }
            }
          
            

            bool hf_file_found = false;
            bool vxc_file_found = false;

            // 读取 hf 文件
            std::ifstream hf_file(hfFilePath.c_str());
            if (hf_file.good()) {
                if (!convert_csc(hfFilePath, arrays, key_hf)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << hfFilePath << std::endl;
                } else {
                    hf_nao[ispin][ikpt] = arrays[key_hf];
                    hf_file_found = true;
                }
            } else {
                std::cerr << "HF file not found: " << hfFilePath << std::endl;
            }

            // 读取 vxc 文件
            std::ifstream vxc_file(vxcFilePath.c_str());
            if (vxc_file.good()) {
                if (!convert_csc(vxcFilePath, arrays, key_vxc)) {
                    all_files_processed_successfully = false;
                    std::cerr << "Failed to process file: " << vxcFilePath << std::endl;
                } else {
                    vxc0[ispin][ikpt] = arrays[key_vxc]; 
                    vxc_file_found = true;
                }
            } else {
                std::cerr << "VXC file not found: " << vxcFilePath << std::endl;
            }

            // 如果两个文件都不存在，报错并跳过该 k 点
            if (!hf_file_found && !vxc_file_found) {
                all_files_processed_successfully = false;
                std::cerr << "Both HF and VXC files not found for spin " << ispin + 1 << ", k-point " << ikpt + 1 << std::endl;
                continue;
            }


            // 生成 H_KS 和 H_KS0 矩阵
            hf[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);     
            hf[ispin][ikpt] = conj(wfc1) * hf_nao[ispin][ikpt] * transpose(wfc1);//row hf,KS basis
            
            // 将 hf 和 vxc 在 KS 基下相加，生成最终的 vxc 矩阵
            
            vxc[ispin][ikpt] = vxc0[ispin][ikpt] + hf[ispin][ikpt];
            vxc0[ispin][ikpt] = vxc[ispin][ikpt];
            // vxc1[ispin][ikpt] = Matz(n_aos, n_aos, MAJOR::COL);    
            // vxc1[ispin][ikpt] = transpose(wfc1) * vxc[ispin][ikpt] * conj(wfc1);//for check
            // vxc1[ispin][ikpt] = conj(wfc1) * vxc1[ispin][ikpt] * transpose(wfc1);
            
            // printf("vxc_diff:\n");
            // for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         const auto &vxc_value = vxc[ispin][ikpt](i,j) ;
            //         const auto &vxc_trans_value = vxc1[ispin][ikpt](i,j) ;
            //         printf("%16.6f ", vxc_value.real()-vxc_trans_value.real()); 
            //     }
            //     printf("\n");
            // }
            // printf("\n"); // 换行
            // for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         const auto &vxc_value = vxc[ispin][ikpt](i,j) ;
            //         printf("%16.6f ", vxc_value.imag()); 
            //     }
            //     printf("\n");
            // }
            // printf("\n"); // 换行
            // 构建 H_KS 矩阵，使用哈密顿量中的本征值
            H_KS[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            H_KS0[ispin][ikpt] = Matz(n_bands, n_bands, MAJOR::COL);
            for (int i_band = 0; i_band < n_bands; ++i_band) {
                H_KS[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
                H_KS0[ispin][ikpt](i_band, i_band) = meanfield.get_eigenvals()[ispin](ikpt, i_band);
            }
            // H_KS[ispin][ikpt]= transpose(wfc1) * H_KS[ispin][ikpt] * conj(wfc1);//construct H 
            // H_KS0[ispin][ikpt] = H_KS[ispin][ikpt];
            // const auto &h = H_KS0.at(ispin).at(ikpt).copy();
            // std::vector<double> v;
            // Matz eigvec_H;
            // eigsh(h, v, eigvec_H);
            // // 打印本征值 w
            // printf("Eigenvalues (v):\n");
            // for (int i = 0; i < n_bands ; ++i) {
            //     printf("%20.16f\n", v[i]);
            // }
            // printf("%77s\n", final_banner.c_str());
            // printf("Eigenvalues before:\n");
            // for (int i = 0; i < n_bands ; ++i) {
            //     printf("%20.16f\n", meanfield.get_eigenvals()[ispin](ikpt, i));
            // }
            // printf("%77s\n", final_banner.c_str());
        }
        // check Hamiltonian real-space
        // std::map<int, Matz> HKS_IR = FT_K_TO_R(meanfield, H_KS[ispin], Rlist);
        // printf("%77s\n", final_banner.c_str());
        // printf("HKS_IR_imag:\n");
        // for (auto R : Rlist) {
        //     // if(R.x==0&R.y==0&R.z==0)
        //     // {
        //     //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        //     //     auto iR = std::distance(Rlist.cbegin(), iteR);   
        //     //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
        //     //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //     //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //     //             const auto &vxc_IR_value = vxc_IR[iR](i,j) ;
        //     //             printf("%16.6f ", vxc_IR_value.real()); 
        //     //         }
        //     //     }
        //     // printf("\n"); // 换行
        //     // }
        //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        //     auto iR = std::distance(Rlist.cbegin(), iteR);   
        //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &HKS_IR_value = HKS_IR[iR](i,j) ;
        //             printf("%16.6f ", HKS_IR_value.imag()); 
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }
        // printf("\n");
        // printf("HKS_IR_real:\n");
        // for (auto R : Rlist) {
        //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        //     auto iR = std::distance(Rlist.cbegin(), iteR);   
        //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &HKS_IR_value = HKS_IR[iR](i,j) ;
        //             printf("%16.6f ", HKS_IR_value.real()); 
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }

        // vxc FT real-space check
        // std::map<int, Matz> vxc_IR = FT_K_TO_R(meanfield, vxc1[ispin], Rlist);
        // for (auto R : Rlist) {
        //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        //     auto iR = std::distance(Rlist.cbegin(), iteR);   
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             vxc_IR[iR](i,j)=std::real(vxc_IR[iR](i,j));   
        //         }
        //     }
        // }
        // printf("%77s\n", final_banner.c_str());
        // printf("vxc_IR_imag:\n");
        // for (auto R : Rlist) {
        //     // if(R.x==0&R.y==0&R.z==0)
        //     // {
        //     //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        //     //     auto iR = std::distance(Rlist.cbegin(), iteR);   
        //     //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
        //     //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //     //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //     //             const auto &vxc_IR_value = vxc_IR[iR](i,j) ;
        //     //             printf("%16.6f ", vxc_IR_value.real()); 
        //     //         }
        //     //     }
        //     // printf("\n"); // 换行
        //     // }
        //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        //     auto iR = std::distance(Rlist.cbegin(), iteR);   
        //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &vxc_IR_value = vxc_IR[iR](i,j) ;
        //             printf("%16.6f ", vxc_IR_value.imag()); 
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }
        // printf("\n");
        // printf("vxc_IR_real:\n");
        // for (auto R : Rlist) {
        //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
        //     auto iR = std::distance(Rlist.cbegin(), iteR);   
        //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &vxc_IR_value = vxc_IR[iR](i,j) ;
        //             printf("%16.6f ", vxc_IR_value.real()); 
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }

        // std::map<int, Matz> vxc_IK = FT_R_TO_K(meanfield, vxc_IR, Rlist);

        // printf("%77s\n", final_banner.c_str());
        // printf("vxc_IK_REAL:\n");
        // for (int r = 0; r < meanfield.get_n_kpoints(); r++){
        //     printf("Ik: %d\n",r);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &vxc_IK_value = vxc_IK[r](i,j) ;
        //             const auto &vxc_value = vxc[ispin][r](i,j);
        //             printf("%16.6f ", vxc_IK_value.real()); 
                    
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }
        // printf("%77s\n", final_banner.c_str());
        // printf("\n");

        // printf("%77s\n", final_banner.c_str());
        // printf("vxc_IK_IMAG:\n");
        // for (int r = 0; r < meanfield.get_n_kpoints(); r++){
        //     printf("Ik: %d\n",r);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &vxc_IK_value = vxc_IK[r](i,j) ;
        //             const auto &vxc_value = vxc[ispin][r](i,j);
        //             printf("%16.6f ", vxc_IK_value.imag()); 
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }
        // printf("%77s\n", final_banner.c_str());
        // printf("\n");

        // printf("%77s\n", final_banner.c_str());
        // printf("vxc_REAL:\n");
        // for (int r = 0; r < meanfield.get_n_kpoints(); r++){
        //     printf("Ik: %d\n",r);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &vxc_IK_value = vxc_IK[r](i,j) ;
        //             const auto &vxc_value = vxc[ispin][r](i,j);
        //             printf("%16.6f ", vxc_value.real()-vxc_IK_value.real()); 
                    
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }
        // printf("%77s\n", final_banner.c_str());
        // printf("\n");

        // printf("%77s\n", final_banner.c_str());
        // printf("vxc_IMAG:\n");
        // for (int r = 0; r < meanfield.get_n_kpoints(); r++){
        //     printf("Ik: %d\n",r);
        //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //             const auto &vxc_IK_value = vxc_IK[r](i,j) ;
        //             const auto &vxc_value = vxc[ispin][r](i,j);
        //             printf("%16.6f ", vxc_value.imag()-vxc_IK_value.imag()); 
        //         }
        //         printf("\n");
        //     }
        //     printf("\n"); // 换行
        // }
        // printf("%77s\n", final_banner.c_str());
        // printf("\n");

        // vxc[ispin] = vxc_IK;//realize
        //这里
    }
    


    // // 判断是否所有文件都成功处理
    // if (mpi_comm_global_h.myid == 0) {
    //     if (all_files_processed_successfully) {
    //         std::cout << "* Success: Read DFT xc potential, will solve quasi-particle equation\n";
    //     } else {
    //         std::cout << "* Error: Some files failed to process, switch off solving quasi-particle equation\n";
    //     }
    // }

    Profiler::stop("read_vxc_HKS");
    mpi_comm_global_h.barrier();

    
    // 在迭代开始前计算初始 HOMO, LUMO 和费米能级
    double efermi = meanfield.get_efermi();
    double homo = -1e6;
    double lumo = 1e6;

    for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            int homo_level = -1;
            for (int ib = 0; ib < meanfield.get_n_bands(); ++ib) {
                double weight = meanfield.get_weight()[ispin](ikpt, ib);
                double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                if (weight >= 1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints())) {
                    homo_level = ib;
                }
            }

            if (homo_level != -1) {
                homo = std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                lumo = std::min(lumo, meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
            }
        }
    }

    // 保存初始状态数据
    homo_values.push_back(homo * HA2EV);  // 初始 HOMO 值
    lumo_values.push_back(lumo * HA2EV);  // 初始 LUMO 值
    efermi_values.push_back(efermi * HA2EV);  // 初始费米能级
    iteration_numbers.push_back(0);  // 初始迭代次数为 0

    std::cout << "Initial HOMO = " << homo * HA2EV << " eV, "
              << "LUMO = " << lumo * HA2EV << " eV, "
              << "Fermi Energy = " << efermi * HA2EV << " eV\n";
    plot_homo_lumo_vs_iterations();
    
    //计算初始体系总电子数/初始总占据数
    double total_electrons = meanfield.get_total_weight();
    printf("%5s\n","Total_electrons");
    printf("%5f\n",total_electrons);
    

    // // //check input eigenvector
    
    // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
    // {
    //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
    //     {
    //         const auto &k = kfrac_list[i_kpoint];

    //         // Output the k-point vector components
    //         printf("k-point %d: (%20.15f, %20.15f, %20.15f)\n", i_kpoint, k.x, k.y, k.z);
    //         printf("%77s\n", final_banner.c_str());
    //         printf("eigenvectors_real:\n");
    //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
    //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
    //                 const auto &eigenvectors = meanfield.get_eigenvectors()[i_spin][i_kpoint](i, j) ;
    //                 printf("%20.15f ", eigenvectors.real()); 
    //             }
    //             printf("\n"); // 换行
    //         }
    //         printf("%77s\n", final_banner.c_str());
    //         printf("\n");
    //         printf("eigenvectors_imag:\n");
    //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
    //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
    //                 const auto &eigenvectors = meanfield.get_eigenvectors()[i_spin][i_kpoint](i, j) ;
    //                 printf("%20.15f ", eigenvectors.imag()); 
    //             }
    //             printf("\n"); // 换行
    //         }
    //         printf("%77s\n", final_banner.c_str());
    //         printf("\n");
    //     }
    // }
    // 设置收敛条件
    double eigenvalue_tolerance = 1e-3; // 设置一个适当的小值，作为本征值收敛的判断标准
    int max_iterations =20;           // 最大迭代次数
    int iteration = 0;
    const double temperature = 0.0001;
    bool converged = false;
    int frequency = n_bands + 1; 
    std::vector<std::pair<int, int>> significant_positions;
    // 定义存储前一轮的本征值以检查收敛性
    std::vector<matrix> previous_eigenvalues(n_spins);
    mpi_comm_global_h.barrier();
    std::ofstream file("homo_lumo_vs_iterations.dat", std::ios::trunc);
    file.close();

    // 初始化完毕，开始循环
    while (!converged && iteration < max_iterations) {
        iteration++;

        // 更新前一轮的本征值
        for (int i_spin = 0; i_spin < n_spins; i_spin++)
        {
            previous_eigenvalues[i_spin] = meanfield.get_eigenvals()[i_spin];
        }

        // Prepare time-frequency grids
        auto tfg = LIBRPA::utils::generate_timefreq_grids(Params::nfreq, Params::tfgrids_type, meanfield);

        Chi0 chi0(meanfield, klist, tfg);
        chi0.gf_R_threshold = Params::gf_R_threshold;

        Profiler::start("chi0_build", "Build response function chi0");
        chi0.build(Cs_data, Rlist, period, local_atpair, qlist);
        Profiler::stop("chi0_build"); 
        mpi_comm_global_h.barrier();

        // 读取库伦相互作用
        Profiler::start("read_vq_cut", "Load truncated Coulomb");
        read_Vq_full(driver_params.input_dir, "coulomb_cut_", true,
                     driver_params.version_coul_reader, Params::use_shrink_abfs);
        Profiler::stop("read_vq_cut");

        // 读取和处理介电函数
        std::vector<double> epsmac_LF_imagfreq_re;
        if (Params::replace_w_head)
        {
            std::vector<double> omegas_dielect;
            std::vector<double> dielect_func;
            read_dielec_func(driver_params.input_dir + "dielecfunc_out", omegas_dielect, dielect_func);
    
            epsmac_LF_imagfreq_re = interpolate_dielec_func(
                    Params::option_dielect_func, omegas_dielect, dielect_func,
                    chi0.tfg.get_freq_nodes());
        }

        // 构建V^{exx}矩阵,得到Hexx_nband_nband: exx.exx_is_ik_KS

        Profiler::start("qsgw_exx", "Build exchange self-energy");
        auto exx = LIBRPA::Exx(meanfield, kfrac_list, period);
        {
            Profiler::start("ft_vq_cut", "Fourier transform truncated Coulomb");
            const auto VR = FT_Vq(Vq_cut, meanfield.get_n_kpoints(), Rlist, true);
            Profiler::stop("ft_vq_cut");

            Profiler::start("g0w0_exx_real_work");
            exx.build(Cs_data, Rlist, VR);
            exx.build_KS_kgrid0();//rotate  
            Profiler::stop("g0w0_exx_real_work");
            // for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            //     for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
            //         exx0[ispin][ikpt] = Matz(n_bands, n_aos, MAJOR::COL);
            //         exx0[ispin][ikpt] = exx.exx_is_ik_KS[ispin][ikpt];
            //         Matz wfc2(n_bands, n_aos, MAJOR::COL);
            //         for (int ib = 0; ib < n_bands; ++ib) {
            //             for (int iao = 0; iao < n_aos; iao++) {
            //                 wfc2(ib, iao) = meanfield.get_eigenvectors0()[ispin][ikpt](ib, iao); 
            //             }
            //         }
            //         exx0[ispin][ikpt] = transpose(wfc2) * exx0[ispin][ikpt] * conj(wfc2);//for check
            //     }
            // }
            // //exx FT real-space check
            // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
            // {
            //     std::map<int, Matz> exx_IR = FT_K_TO_R(meanfield, exx0[i_spin], Rlist);
            //     printf("%77s\n", final_banner.c_str());
            //     printf("exx_IR_imag:\n");
            //     for (auto R : Rlist) {
            //         // if(R.x==0&R.y==0&R.z==0)
            //         // {
            //         //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
            //         //     auto iR = std::distance(Rlist.cbegin(), iteR);   
            //         //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
            //         //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //         //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //         //             const auto &vxc_IR_value = vxc_IR[iR](i,j) ;
            //         //             printf("%16.6f ", vxc_IR_value.real()); 
            //         //         }
            //         //     }
            //         // printf("\n"); // 换行
            //         // }
            //         auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
            //         auto iR = std::distance(Rlist.cbegin(), iteR);   
            //         printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
            //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
            //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
            //                 const auto &exx_IR_value = exx_IR[iR](i,j) ;
            //                 printf("%16.6f ", exx_IR_value.imag()); 
            //                 // exx_IR[iR](i,j)=std::real(exx_IR[iR](i,j));//realize
            //             }
            //             printf("\n");
            //         }
            //         printf("\n"); // 换行
            //     }
            //     // std::map<int, Matz> exx_IK = FT_R_TO_K(meanfield, exx_IR, Rlist);
            //     // exx.exx_is_ik_KS[i_spin] = exx_IK;
            // }
        }
        Profiler::stop("qsgw_exx");
        Hexx_matrix_temp[iteration] = exx.exx_is_ik_KS ;
        Hexx_matrix_temp[0] = exx.exx_is_ik_KS ;

        mpi_comm_global_h.barrier();

        
        // //check
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
        //             const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
        //             printf("%5d %16.5f %16.5f\n",
        //                 i_state + 1, eks_state, vxc_state.real());
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("1exx_real Matrix0:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc[i_spin][i_kpoint](i, j) ;
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc-exx_real Matrix0:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc[i_spin][i_kpoint](i, j) ;
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.real()-exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc-exx_imag Matrix0:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc[i_spin][i_kpoint](i, j) ;
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.imag()-exx_value.imag()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
              
        //     }
        // }
        
        
        
        // Build screened interaction
        Profiler::start("qsgw_wc", "Build screened interaction");
        vector<std::complex<double>> epsmac_LF_imagfreq(epsmac_LF_imagfreq_re.cbegin(), epsmac_LF_imagfreq_re.cend());
        map<double, atom_mapping<std::map<Vector3_Order<double>, matrix_m<complex<double>>>>::pair_t_old> Wc_freq_q;
        if (Params::use_scalapack_gw_wc) {
            Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        } else {
            Wc_freq_q = compute_Wc_freq_q(chi0, Vq, Vq_cut, epsmac_LF_imagfreq);
        }
        Profiler::stop("qsgw_wc");


        LIBRPA::G0W0 s_g0w0(meanfield, kfrac_list, chi0.tfg, period);
        Profiler::start("g0w0_sigc_IJ", "Build correlation self-energy");
        s_g0w0.build_spacetime(Cs_data, Wc_freq_q, Rlist);
        Profiler::stop("g0w0_sigc_IJ");

        Profiler::start("g0w0_sigc_rotate_KS", "Rotate self-energy, IJ -> ij -> KS");
        s_g0w0.build_sigc_matrix_KS_kgrid0();//rotate
        Profiler::stop("g0w0_sigc_rotate_KS");

        // 构建哈密顿量矩阵并对角化，旋转基底，并存储本征值，本征矢量
        // 第一步：构建关联势矩阵
        std::map<int, std::map<int, Matz>> Vc_all;

        // 构建虚频点列表
        std::vector<cplxdb> imagfreqs;
        for (const auto &freq : chi0.tfg.get_freq_nodes()) {
            imagfreqs.push_back(cplxdb{0.0, freq});
        }

        std::map<int, std::map<int, std::map<int, double>>> e_qp_all;
        std::map<int, std::map<int, std::map<int, cplxdb>>> sigc_all;
        
        if (all_files_processed_successfully)
        {
            Profiler::start("qsgw_solve_qpe", "Solve quasi-particle equation");

            if (mpi_comm_global_h.is_root()) {
                std::cout << "Solving quasi-particle equation\n";
            }

            if (mpi_comm_global_h.is_root()) {
                // 遍历自旋、k点和能带状态
                for (int i_spin = 0; i_spin < n_spins; i_spin++) {
                    for (int i_kpoint = 0; i_kpoint < n_kpoints; i_kpoint++) {
                        std::vector<std::vector<std::vector<cplxdb>>> sigcmat(
                            n_bands, std::vector<std::vector<cplxdb>>(n_bands, std::vector<cplxdb>(n_bands + 1))
                        );                  
                        const auto &sigc_sk = s_g0w0.sigc_is_ik_f_KS[i_spin][i_kpoint];
                        for (int i_state_row = 0; i_state_row < n_bands; i_state_row++) {                 
                            for (int i_state_col = 0; i_state_col < meanfield.get_n_bands(); i_state_col++) {
                                
                                std::vector<cplxdb> sigc_mn;
                                for (const auto &freq : chi0.tfg.get_freq_nodes()) {
                                    sigc_mn.push_back(sigc_sk.at(freq)(i_state_row, i_state_col));
                            
                                }    
                                LIBRPA::AnalyContPade pade(Params::n_params_anacon, imagfreqs, sigc_mn);

                                auto energy0 = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state_row); 
                                efermi = meanfield.get_efermi();
                                // 计算得到的值
                                auto result = pade.get(energy0 - efermi);
                                auto result1 = pade.get(0.0);
                                // 存储值到 sigcmat
                                sigcmat[i_state_row][i_state_col][i_state_row] = result;
                                sigcmat[i_state_row][i_state_col][n_bands] = result1;
                                
                                // // 输出当前计算结果
                                // std::cout << "sigcmat[" << i_state_row << "][" << i_state_col << "][" << i_state_row 
                                //         << "] = " << result << std::endl;
                            }
                        }

                        Vc_all[i_spin][i_kpoint] = build_correlation_potential_spin_k_modeA(sigcmat,n_bands);
                        Matz wfc3(n_bands, n_aos, MAJOR::COL);
                        std::cout << "VC_KS_1_real " << std::endl;
                        for (int ib = 0; ib < n_bands; ++ib) {
                            for (int iao = 0; iao < n_aos; iao++) {
                                wfc3(ib, iao) = meanfield.get_eigenvectors0()[i_spin][i_kpoint](ib, iao); 
                                const auto &Vc_k_ks_value = Vc_all[i_spin][i_kpoint](ib,iao) ;
                                printf("%16.6f ", Vc_k_ks_value.real()* HA2EV); 
                            }
                            printf("\n");
                        }
                        printf("\n");
                        std::cout << "VC_KS_1_imag " << std::endl;
                        for (int ib = 0; ib < n_bands; ++ib) {
                            for (int iao = 0; iao < n_aos; iao++) {
                                wfc3(ib, iao) = meanfield.get_eigenvectors0()[i_spin][i_kpoint](ib, iao); 
                                const auto &Vc_k_ks_value = Vc_all[i_spin][i_kpoint](ib,iao) ;
                                printf("%16.6f ", Vc_k_ks_value.imag()* HA2EV); 
                            }
                            printf("\n");
                        }
                        printf("\n");
                        // Vc_all[i_spin][i_kpoint] = transpose(wfc3) * Vc_all[i_spin][i_kpoint] * conj(wfc3);//to NAO 
                        // std::cout << "VC_NAO_1 " << std::endl;
                        // for (int ib = 0; ib < n_bands; ++ib) {
                        //     for (int iao = 0; iao < n_aos; iao++) {
                        //         const auto &Vc_k_NAO_value = Vc_all[i_spin][i_kpoint](ib,iao);
                        //         printf("%16.6f ", Vc_k_NAO_value.real()* HA2EV); 
                        //     }
                        //     printf("\n");
                        // }
                        // printf("\n");
                        // // FT
                    }

                    // //Vc FT real-space check

                    // std::map<int, Matz> Vc_IR = FT_K_TO_R(meanfield, Vc_all[i_spin], Rlist);
                    // // printf("%77s\n", final_banner.c_str());
                    // // printf("Vc_IR_imag:\n");
                    // for (auto R : Rlist) {
                    //     // if(R.x==0&R.y==0&R.z==0)
                    //     // {
                    //     //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
                    //     //     auto iR = std::distance(Rlist.cbegin(), iteR);   
                    //     //     printf("Rlist %d: (%3d %3d %3d)\n", iR, R.x, R.y, R.z);
                    //     //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    //     //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
                    //     //             const auto &vxc_IR_value = vxc_IR[iR](i,j) ;
                    //     //             printf("%16.6f ", vxc_IR_value.real()); 
                    //     //         }
                    //     //     }
                    //     // printf("\n"); // 换行
                    //     // }
                    //     auto iteR = std::find(Rlist.cbegin(), Rlist.cend(), R);
                    //     auto iR = std::distance(Rlist.cbegin(), iteR);   
                    //     for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    //         for (int j = 0; j < meanfield.get_n_bands(); j++) {
                    //             const auto &Vc_IR_value = Vc_IR[iR](i,j) ;
                    //             // printf("%16.6f ", Vc_IR_value.imag()); 
                    //             // Vc_IR[iR](i,j)=std::real(Vc_IR_value);//realize
                    //         }
                    //         // printf("\n");
                    //     }
                    //     // printf("\n"); // 换行
                    // }
                    // std::map<int, Matz> Vc_IK = FT_R_TO_K(meanfield, Vc_IR, Rlist);
                    // std::cout << "VC_NAO_2 " << std::endl;
                    // Vc_all[i_spin] = Vc_IK;
                    // for (int ikpt = 0; ikpt < n_kpoints; ikpt++) {
                    //     Matz wfc4(n_bands, n_aos, MAJOR::COL);
                    //     for (int ib = 0; ib < n_bands; ++ib) {
                    //         for (int iao = 0; iao < n_aos; iao++) {
                    //             wfc4(ib, iao) = meanfield.get_eigenvectors0()[i_spin][ikpt](ib, iao); 
                    //             const auto &Vc_k2_nao_value = Vc_all[i_spin][ikpt](ib,iao) ;
                    //             printf("%16.6f ", Vc_k2_nao_value.real()* HA2EV); 
                    //         }
                    //         printf("\n");
                    //     }
                    //     printf("\n");
                    //     std::cout << "VC_KS_2 " << std::endl;
                    //     Matz Vc_temp = Vc_all[i_spin][ikpt];
                    //     Vc_all[i_spin][ikpt] = conj(wfc4) * Vc_temp * transpose(wfc4);
                    //     for (int ib = 0; ib < n_bands; ++ib) {
                    //         for (int iao = 0; iao < n_aos; iao++) {
                    //             const auto &Vc_k2_ks_value = Vc_all[i_spin][ikpt](ib,iao) ;
                    //             printf("%16.6f ", Vc_k2_ks_value.real()* HA2EV); 
                    //         }
                    //         printf("\n");
                    //     }
                    //     printf("\n");
                    //     std::cout << "check112 " << std::endl;
                    // }
                    // //FT
                }

            }
            Profiler::stop("qsgw_solve_qpe");
        }

        //检查输入
        
        for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        {
            for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
            {
                const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
        //             const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
        //             printf("%5d %20.15f %20.15f\n",
        //                 i_state + 1, eks_state, vxc_state.real());
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("H Matrix_real:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &H_value = H_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", H_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("H Matrix_image:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &H_value = H_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", H_value.imag()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
                // 输出 exx_is_ik_KS 矩阵
                printf("%77s\n", final_banner.c_str());
                printf("exx Matrix:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", exx_value.real()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");

                printf("%77s\n", final_banner.c_str());
                printf("exx Matrix:\n");
                for (int i = 0; i < meanfield.get_n_bands(); i++) {
                    for (int j = 0; j < meanfield.get_n_bands(); j++) {
                        const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
                        printf("%16.6f ", exx_value.imag()); 
                    }
                    printf("\n"); // 换行
                }
                printf("%77s\n", final_banner.c_str());
                printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("Vc_all\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &Vc_value = Vc_all[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%16.6f ", Vc_value.real()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
            }
        }
        
            
        
        auto H0_GW_all = construct_H0_GW(meanfield, H_KS0, vxc0, exx.exx_is_ik_KS, Vc_all, n_spins, n_kpoints, n_bands);
        
        mpi_comm_global_h.barrier();
        //混合
        // if(iteration > 1){
        //     for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
        //         for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
        //             H0_GW_all[ispin][ikpt] = 0.2 * H0_GW_all[ispin][ikpt] + 0.8 * H_KS[ispin][ikpt];
        //         }
        //     }
        // }
        //检查输入
        
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("H Matrix_real:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &H_value = H0_GW_all[i_spin][i_kpoint](i, j) ;
        //                 printf("%20.15f ", H_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("H Matrix_image:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &H_value = H0_GW_all[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", H_value.imag()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
                
        //     }
        // }

        // 第三步：对 Hamiltonian 进行对角化并存储本征值
        diagonalize_and_store(meanfield, H0_GW_all, n_spins, n_kpoints, n_bands);
        
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
  
        //         // 输出 exx_is_ik_KS 矩阵
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix_real:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");

        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix_imag:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.imag()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix_real:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &vxc_value = vxc[i_spin][i_kpoint](i, j) ;
        //                 printf("%20.15f ", vxc_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");

        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix_imag:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &vxc_value = vxc[i_spin][i_kpoint](i, j) * HA2EV;
        //                 printf("%20.15f ", vxc_value.imag()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");

        //         printf("%77s\n", final_banner.c_str());
        //         printf("Vc Matrix_real:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &Vc_value = Vc_all[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%20.15f ", Vc_value.real()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");

        //         printf("%77s\n", final_banner.c_str());
        //         printf("Vc Matrix_imag:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &Vc_value = Vc_all[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%20.15f ", Vc_value.imag()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");

                // printf("%77s\n", final_banner.c_str());
                // printf("Vc+exx Matrix-vxc Matrix:\n");
                // for (int i = 0; i < meanfield.get_n_bands(); i++) {
                //     for (int j = 0; j < meanfield.get_n_bands(); j++) {
                //         const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
                //         const auto &vxc_value = vxc1[i_spin][i_kpoint](i, j)* HA2EV ;
                //         const auto &Vc_value = Vc_all[i_spin][i_kpoint](i, j)* HA2EV ;
                //         printf("%20.15f ", Vc_value.real() + exx_value.real() - vxc_value.real()); 
                //     }
                //     printf("\n"); 
                // }
                // printf("%77s\n", final_banner.c_str());
                // printf("\n");
            // }
        // }

        // 计算全局费米能和占据数
        const auto &Efermi0 = meanfield.get_efermi() ;
        printf("%5s\n","efermi0");
        printf("%5f\n",Efermi0);
        // 计算费米能级
        
        double efermi = calculate_fermi_energy(meanfield, temperature, total_electrons);
        printf("%5s\n","efermi0");
        printf("%5f\n",efermi);
        double eqp_gap = calculate_eqp_fermi_energy(meanfield, e_qp_all, temperature, total_electrons);
        printf("%5s\n","eqp_gap:");
        printf("%5f\n",eqp_gap);
         //将占据数和费米能级更新到 MeanField 对象中
        update_fermi_energy_and_occupations(meanfield, temperature, efermi);
        efermi_values.push_back(efermi * HA2EV);  
        // 比较本轮和前一轮的本征值判断是否收敛
        converged = true;
        for (int ispin = 0; ispin < n_spins; ++ispin) {
            const auto &current_eigenvals = meanfield.get_eigenvals()[ispin];
            const auto max_diff = (current_eigenvals - previous_eigenvalues[ispin]).absmax();
            if (max_diff > eigenvalue_tolerance) {
                converged = false;
                break;
            }
        }
        std::cout << "Converged after " << iteration << " iterations.\n";
        // const std::string final_banner(90, '-');
        lib_printf("Final Quasi-Particle Energy after QSGW Iterations [unit: eV]\n\n");
        const auto &Efermi = meanfield.get_efermi() ;
        printf("%5s\n","efermi");
        printf("%5f\n",Efermi);
        for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        {
            for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
            {
                const auto &k = kfrac_list[i_kpoint];
                printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
                        i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
                printf("%77s\n", final_banner.c_str());
                printf("%5s %16s %16s %16s %16s %16s %16s %16s\n", "State", "e_mf", "v_xc", "v_exx1", "v_exx2","ReSigc", "ImSigc", "e_qp");
                printf("%77s\n", final_banner.c_str());
                for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
                {
                    const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    const auto &exx_state1 = exx.Eexx[i_spin][i_kpoint][i_state] * HA2EV;
                    const auto &exx_state2 = exx.exx_is_ik_KS[i_spin][i_kpoint](i_state, i_state)* HA2EV;
                    const auto &vxc_state = vxc0[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                    const auto &resigc = sigc_all[i_spin][i_kpoint][i_state].real() * HA2EV;
                    const auto &imsigc = sigc_all[i_spin][i_kpoint][i_state].imag() * HA2EV;
                    const auto &eqp = e_qp_all[i_spin][i_kpoint][i_state] * HA2EV;
                    printf("%5d %20.15f %16.5f %16.5f %16.5f %16.5f %16.5f %20.15f\n",
                           i_state + 1, eks_state, vxc_state.real(), exx_state1, exx_state2.real(), resigc, imsigc, eqp);
                }
                printf("\n");
            }
        }

        



        
        // std::ofstream exx_output_file("exx_output_all_iterations.dat", std::ios::app);
        // if (!exx_output_file.is_open()) {
        //     std::cerr << "Error: Unable to open file for writing." << std::endl;
        //     return;
        // }
        // // 写入当前迭代次数标识
        // exx_output_file << "Iteration " << iteration << "\n";
        // for (int ispin = 0; ispin < n_spins; ++ispin) { 
        //     for (int ikpt = 0; ikpt < n_kpoints; ++ikpt) {
        //         exx_output_file << "Spin " << ispin << ", K-point " << ikpt << ":\n";

        //         // 获取 exx 矩阵并写入文件
        //         for (int i = 0; i < meanfield.get_n_bands(); ++i) {
        //             for (int j = 0; j < meanfield.get_n_bands(); ++j) {
        //                 const auto& exx_value = exx.exx_is_ik_KS[ispin][ikpt](i, j);
        //                 exx_output_file << exx_value.real() << " "; // 假设只写入实部
        //             }
        //             exx_output_file << "\n";
        //         }
        //         exx_output_file << "\n"; // 分隔不同自旋或 k 点

              
        //     }
        // }

        // 计算 HOMO 和 LUMO
        homo = -1e6;  // 
        lumo = 1e6;   // 
        for (int ispin = 0; ispin < meanfield.get_n_spins(); ++ispin) {
            for (int ikpt = 0; ikpt < meanfield.get_n_kpoints(); ++ikpt) {
                int homo_level = -1;
                for (int ib = 0; ib < meanfield.get_n_bands(); ++ib) {
                    double weight = meanfield.get_weight()[ispin](ikpt, ib);
                    double energy = meanfield.get_eigenvals()[ispin](ikpt, ib);

                    // 
                    if (weight >= 1.0 / (meanfield.get_n_spins() * meanfield.get_n_kpoints())) {
                        homo_level = ib;
                    }
                }

                // 
                if (homo_level != -1) {
                    // 
                    homo = std::max(homo, meanfield.get_eigenvals()[ispin](ikpt, homo_level));
                    // 
                    lumo = std::min(lumo, meanfield.get_eigenvals()[ispin](ikpt, homo_level + 1));
                }
            }
        }

        // 
        homo_values.push_back(homo * HA2EV);  // 
        lumo_values.push_back(lumo * HA2EV);  // 
        iteration_numbers.push_back(iteration);

        // 输出当前 HOMO 和 LUMO 值
        std::cout << "Iteration " << iteration
          << ": HOMO = " << homo * HA2EV << " eV, "
          << "LUMO = " << lumo * HA2EV << " eV, "
          << "Efermi = " << efermi * HA2EV << " eV\n";
        
        // for (int i_spin = 0; i_spin < meanfield.get_n_spins(); i_spin++)
        // {
        //     for (int i_kpoint = 0; i_kpoint < meanfield.get_n_kpoints(); i_kpoint++)
        //     {
        //         const auto &k = kfrac_list[i_kpoint];
        //         printf("spin %2d, k-point %4d: (%.5f, %.5f, %.5f) \n",
        //                 i_spin + 1, i_kpoint + 1, k.x, k.y, k.z);
        //         printf("%77s\n", final_banner.c_str());
        //         printf("%5s %16s %16s\n", "State", "e_mf", "v_xc");
        //         printf("%77s\n", final_banner.c_str());
        //         for (int i_state = 0; i_state < meanfield.get_n_bands(); i_state++)
        //         {
        //             const auto &eks_state = meanfield.get_eigenvals()[i_spin](i_kpoint, i_state) * HA2EV;
                    
        //             const auto &vxc_state = vxc[i_spin][i_kpoint](i_state, i_state) * HA2EV;
                        
        //             printf("%5d %16.5f %16.5f\n",
        //                 i_state + 1, eks_state, vxc_state.real());
        //         }
        //         // 输出 exx_is_ik_KS 矩阵
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix5:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", exx_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("vxc Matrix5:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j) ;
        //                 printf("%16.6f ", vxc_value.real()); 
        //             }
        //             printf("\n"); // 换行
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //         printf("%77s\n", final_banner.c_str());
        //         printf("exx Matrix-vxc Matrix5:\n");
        //         for (int i = 0; i < meanfield.get_n_bands(); i++) {
        //             for (int j = 0; j < meanfield.get_n_bands(); j++) {
        //                 const auto &exx_value = exx.exx_is_ik_KS[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 const auto &vxc_value = vxc0[i_spin][i_kpoint](i, j)* HA2EV ;
        //                 printf("%16.6f ", exx_value.real()-vxc_value.real()); 
        //             }
        //             printf("\n"); 
        //         }
        //         printf("%77s\n", final_banner.c_str());
        //         printf("\n");
        //     }
        // }
        // 保存 HOMO、LUMO 和费米能级数据
        {
            std::ofstream file("homo_lumo_vs_iterations.dat", std::ios::app); // 使用 std::ios::app 以追加模式打开文件
            file << iteration << " "
                << homo_values[iteration] << " "
                << lumo_values[iteration] << " "
                << efermi_values[iteration] << std::endl;
        }

        

        // 如果已经收敛或达到最大迭代次数，输出最终的QSGW迭代结果，退出循环
        if (converged) {
            break;
        }

        if (iteration == max_iterations) {
            std::cout << "Reached maximum number of iterations.\n";
        }
        mpi_comm_global_h.barrier();
        
    }

    Profiler::stop("qsgwA");
}
