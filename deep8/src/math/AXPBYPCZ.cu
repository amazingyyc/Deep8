#include "math/AXPBYPCZ.h"
#include "basic/GPUBasic.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void AXPBYPCZKernel(const T* x, T a, const T* y, T b, const T* z, T c, T *w, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        w[i] = a * x[i] + b * y[i] + c * z[i];
    }
}

void AXPBYPCZGPU(const Tensor& x, float a, const Tensor& y, float b, const Tensor& z, float c, Tensor& w) {
    int N = (int)x.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (x.elementType.id) {
    case DType::Float32:
        AXPBYPCZKernel<float> << <grideSize, blockSize >> > (x.data<float>(), a, y.data<float>(), b, z.data<float>(), c, w.data<float>(), N);
        break;
    case DType::Float64:
        AXPBYPCZKernel<double> << <grideSize, blockSize >> > (x.data<double>(), double(a), y.data<double>(), double(b), z.data<double>(), (double)c, w.data<double>(), N);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        AXPBYPCZKernel<half> << <grideSize, blockSize >> > (x.data<half>(), __float2half(a), y.data<half>(), __float2half(b), z.data<half>(), __float2half(c), w.data<half>(), N);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}




}
}