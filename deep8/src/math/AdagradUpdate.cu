#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/AdagradUpdate.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void AdagradUpdateKernel(T* value, T* gradient, T* accumulate, T epsilon, T learningRate, T weightDecay, int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        gradient[i]   += value[i]     * weightDecay;
        accumulate[i] += gradient[i]  * gradient[i];
        value[i]      -= learningRate * gradient[i] / cudaSqrt(accumulate[i] + epsilon);
    }
}

void AdagradUpdateGPU(Tensor& value, Tensor& gradient, Tensor& accumulate, float epsilon, float learningRate, float weightDecay) {
    int N = (int)value.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    switch (value.elementType.id) {
    case DType::Float32:
        AdagradUpdateKernel<float> << <grideSize, blockSize >> > (value.data<float>(), gradient.data<float>(), accumulate.data<float>(), epsilon, learningRate, weightDecay, N);
        break;
    case DType::Float64:
        AdagradUpdateKernel<double> << <grideSize, blockSize >> > (value.data<double>(), gradient.data<double>(), accumulate.data<double>(), double(epsilon), double(learningRate), double(weightDecay), N);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        AdagradUpdateKernel<half> << <grideSize, blockSize >> > (value.data<half>(), gradient.data<half>(), accumulate.data<half>(), __float2half(epsilon), __float2half(learningRate), __float2half(weightDecay), N);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << value.elementType.name << " is not support");
        break;
    }
}



}
}