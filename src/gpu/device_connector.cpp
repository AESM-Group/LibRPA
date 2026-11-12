#include "device_connector.h"

#if defined(LIBRPA_USE_CUDA) || defined(LIBRPA_USE_HIP)
#include <ddla/ddla.h>

#endif

#include <cassert>

namespace librpa_int{
namespace DeviceConnector{
bool check_device_ptr(void* ptr){
#if defined(LIBRPA_USE_CUDA)
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    
    if (err != cudaSuccess) {
        std::cerr << "cudaPointerGetAttributes failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return (attr.type == cudaMemoryTypeDevice);
#elif defined(LIBRPA_USE_HIP)
    hipPointerAttribute_t attr;
    hipError_t status = hipPointerGetAttributes(&attr, ptr);
    
    if (status != hipSuccess) {
        std::cerr << "Error: " << hipGetErrorString(status) << std::endl;
        return false;
    }
    return (attr.memoryType == hipMemoryTypeDevice);
#else
    return false;
#endif
}


} // namespace DeviceConnector

}
