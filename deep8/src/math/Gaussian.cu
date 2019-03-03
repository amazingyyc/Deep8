#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUUnaryElementWise.h"
#include "math/GPUMath.h"
#include "math/Gaussian.h"

namespace Deep8 {
namespace Math {

template <typename T>
void GaussianGPUImpl(GPUDevice *device, T *x, int size, T mean, T stddev) {
    DEEP8_RUNTIME_ERROR("the type is not suppport");
}

template <>
void GaussianGPUImpl<float>(GPUDevice *device, float *x, int size, float mean, float stddev) {
    CURAND_CHECK(curandGenerateNormal(device->curandGenerator, x, size, mean, stddev));
}

template <>
void GaussianGPUImpl<double>(GPUDevice *device, double *x, int size, double mean, double stddev) {
    CURAND_CHECK(curandGenerateNormalDouble(device->curandGenerator, x, size, mean, stddev));
}

void GaussianGPU(Tensor &x, float mean, float stddev) {
    auto device = (GPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        GaussianGPUImpl<float>(device, x.data<float>(), (int)x.size(), mean, stddev);
        break;
    case DType::Float64:
        GaussianGPUImpl<double>(device, x.data<double>(), (int)x.size(), double(mean), double(stddev));
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}