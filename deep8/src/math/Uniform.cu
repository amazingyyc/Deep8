#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUUnaryElementWise.h"
#include "math/GPUMath.h"
#include "math/Uniform.h"

namespace Deep8 {
namespace Math {

template <typename T>
void UniformGPUImpl(GPUDevice *device, T *tensor, int size) {
    DEEP8_RUNTIME_ERROR("the type is not suppport");
}

template <>
void UniformGPUImpl<float>(GPUDevice *device, float *tensor, int size) {
	CURAND_CHECK(curandGenerateUniform(device->curandGenerator, tensor, size));
}

template <>
void UniformGPUImpl<double>(GPUDevice *device, double *tensor, int size) {
	CURAND_CHECK(curandGenerateUniformDouble(device->curandGenerator, tensor, size));
}

void UniformGPU(Tensor &tensor) {
    auto device = (GPUDevice*)tensor.device();

    switch (x.elementType.id) {
    case DType::Float32:
        UniformGPUImpl<float>(device, tensor.data<float>(), (int)tensor.size());
        break;
    case DType::Float64:
        UniformGPUImpl<double>(device, tensor.data<double>(), (int)tensor.size());
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}