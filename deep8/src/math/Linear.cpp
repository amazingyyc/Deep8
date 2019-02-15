#include "math/Linear.h"

namespace Deep8 {
namespace Math {

/**
 * y = Linear(x)
 */
void Linear(const Tensor &x, const float a, const float b, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == y.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        LinearCPU(x, a, b, y);
    } else {
#ifdef HAVE_CUDA
        LinearGPU(x, a, b, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

/**
 * calculate the grad(x) of Linear
 */
void LinearGrad(const Tensor &x, Tensor &dx, const float a, const float b, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == dx.type  && x.type == y.type && x.type  == dy.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        LinearGradCPU(x, dx, a, b, y, dy);
    } else {
#ifdef HAVE_CUDA
        LinearGradGPU(x, dx, a, b, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void LinearCPUImpl(CPUDevice *device, T *x, const Shape &xshape, const T a, const T b, T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int)yshape.size());

    yvec.device(*eigenDevice) = xvec * a + b;
}

void LinearCPU(const Tensor &x, const float a, const float b, Tensor &y) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        LinearCPUImpl<float>(device, x.data<float>(), x.shape, a, b, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        LinearCPUImpl<double>(device, x.data<double>(), x.shape, double(a), double(b), y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
void LinearGradCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, const T a, const T b, T *y, T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());

    dxvec.device(*eigenDevice) += dyvec * a;
}

void LinearGradCPU(const Tensor &x, Tensor &dx, const float a, const float b, const Tensor &y, const Tensor &dy) {
    auto device = (CPUDevice*) x.device();

    switch (x.type.id) {
    case DType::Float32:
        LinearGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, a, b, y.data<float>(), dy.data<float>(), y.shape);
        break;
    case DType::Float64:
        LinearGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, double(a), double(b), y.data<double>(), dy.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}



}
}