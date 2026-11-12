#ifndef DEVICE_CONNECTOR_H
#define DEVICE_CONNECTOR_H

#include "../../mpi/base_blacs.h"

namespace librpa_int{
namespace DeviceConnector{
    // static void float_to_double_device(float* d_A, double* d_B, const int64_t& n);
    // static void double_to_float_device(double* d_A, float* d_B, const int64_t& n);
    // static void 
bool check_device_ptr(void* A);

template<typename T1, typename T2>
void pdam(const T1& num, T2* d_A, const ArrayDesc& array_desc);

}
}




#endif // DEVICE_CONNECTOR_H
