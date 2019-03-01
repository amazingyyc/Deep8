#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/AXPBY.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void AXPBYKernel(const T* x, T alpha, T* y, T beta, T *z, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        z[i] = alpha * x[i] + beta * y[i];
    }
}

void AXPBYGPU(const Tensor& x, float alpha, const Tensor& y, float beta, Tensor& z) {
    int N = (int) x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        AXPBYKernel<float> <<<grideSize, blockSize >>>(x.data<float>(), alpha, y.data<float>(), beta, z.data<float>(), N);
        break;
    case DType::Float64:
        AXPBYKernel<double> << <grideSize, blockSize >> > (x.data<double>(), double(alpha), y.data<double>(), double(beta), z.data<double>(), N);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        AXPBYKernel<half> << <grideSize, blockSize >> > (x.data<half>(), __float2half(alpha), y.data<half>(), __float2half(beta), z.data<half>(), N);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }

}



}
}