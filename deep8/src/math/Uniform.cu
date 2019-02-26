#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUUnaryElementWise.h"
#include "math/GPUMath.h"
#include "math/Uniform.h"

namespace Deep8 {
namespace Math {

template <typename T>
__global__ void UniformKernel(T *x, T a, T b, int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		x[i] = a * x[i] + b;
    }
}

template <typename T>
void UniformGPUImpl(GPUDevice *device, T *tensor, T left, T right, int size) {
    DEEP8_RUNTIME_ERROR("the type is not suppport");
}

template <>
void UniformGPUImpl<float>(GPUDevice *device, float *x, float left, float right, int size) {
	CURAND_CHECK(curandGenerateUniform(device->curandGenerator, x, size));

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
	int grideSize = (size + blockSize - 1) / blockSize;

    UniformKernel<float> << <grideSize, blockSize >> > (x, (right - left), left, size);
}

template <>
void UniformGPUImpl<double>(GPUDevice *device, double *x, double left, double right, int size) {
	CURAND_CHECK(curandGenerateUniformDouble(device->curandGenerator, x, size));

    UniformKernel<double> << <grideSize, blockSize >> > (x, (right - left), left, size);
}

void UniformGPU(Tensor &x, float left = 0.0, float right = 1.0) {
    auto device = (GPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        UniformGPUImpl<float>(device, x.data<float>(), left, right, (int)x.size());
        break;
    case DType::Float64:
        UniformGPUImpl<double>(device, x.data<double>(), left, right, (int)x.size());
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}