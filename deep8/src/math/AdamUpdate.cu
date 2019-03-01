#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/AdamUpdate.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void AdamUpdateKernel(T* value, T* gradient, T* mt, T* vt, T beta1, T beta2, T epsilon, T weightDecay, T ratio, int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        gradient[i] += value[i] * weightDecay;

        mt[i] = mt[i] * beta1 + (T(1.0) - beta1) * gradient[i];
        vt[i] = vt[i] * beta2 + gradient[i] * gradient[i] * (T(1.0) - beta2);

        value[i] -= mt[i] / (cudaSqrt(vt[i]) + epsilon) * ratio;
    }
}

void AdamUpdateGPU(Tensor& value, Tensor& gradient, Tensor& mt, Tensor& vt, float beta1, float beta2, float epsilon, float learningRate, float weightDecay, int64_t steps) {
    int N = (int)value.shape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    float ratio = learningRate * std::sqrt(1.0 - std::pow(beta2, (float)steps)) / (1.0 - std::pow(beta1, (float)steps));

    switch (value.elementType.id) {
    case DType::Float32:
        AdamUpdateKernel<float> <<<grideSize, blockSize >>>(value.data<float>(), gradient.data<float>(), mt.data<float>(), vt.data<float>(), beta1, beta2, epsilon, weightDecay, ratio, N);
        break;
    case DType::Float64:
        AdamUpdateKernel<double> << <grideSize, blockSize >> > (value.data<double>(), gradient.data<double>(), mt.data<double>(), vt.data<double>(), (double)beta1, (double)beta2, (double)epsilon, (double)weightDecay, (double)ratio, N);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        AdamUpdateKernel<half> << <grideSize, blockSize >> > (value.data<half>(), gradient.data<half>(), mt.data<half>(), vt.data<half>(), __float2half(beta1), __float2half(beta2), __float2half(epsilon), __float2half(weightDecay), __float2half(ratio), N);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << value.elementType.name << " is not support");
        break;
    }

}

}
}