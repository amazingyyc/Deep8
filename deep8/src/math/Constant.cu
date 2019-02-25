#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUUnaryElementWise.h"
#include "math/GPUMath.h"
#include "math/Constant.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void ConstantKernel(T *value, T scalar, int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		value[i] = scalar;
	}
}

void ConstantGPU(Tensor &x, float scalar) {
    int N = (int)tensor.size();

	int blockSize = DEEP8_GPU_BLOCK_SIZE;
	int grideSize = (N + blockSize - 1) / blockSize;

    switch (x.elementType.id) {
    case DType::Float32:
        ConstantKernel<float> << <grideSize, blockSize >> > (tensor.data<float>(), scalar, N);
        break;
    case DType::Float64:
        ConstantKernel<float> << <grideSize, blockSize >> > (tensor.data<double>(), double(scalar), N);
        break;
#ifdef HAVE_HALF
    case DType::Float16:
        ConstantKernel<float> << <grideSize, blockSize >> > (tensor.data<half>(), __float2half(scalar), N); 
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}