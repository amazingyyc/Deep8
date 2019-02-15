#include "math/Exp.h"

namespace Deep8 {
namespace Math {

/**
 * y = exp(x)
 */
void Exp(const Tensor &x, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == y.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        ExpCPU(x, y);
    } else {
#ifdef HAVE_CUDA
        ExpGPU(x, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

/**
 * calculate the grad(x) of Exp
 */
void ExpGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == dx.type  && x.type == y.type && x.type  == dy.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        ExpGradCPU(x, dx, y, dy);
    } else {
#ifdef HAVE_CUDA
        ExpGradGPU(x, dx, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}


template <typename T>
void ExpCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int)yshape.size());

    yvec.device(*eigenDevice) = xvec.exp();
}

void ExpCPU(const Tensor &x, Tensor &y) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        ExpCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        ExpCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
void ExpGradCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, T *y, T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  yvec( y, (int)yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());

    dxvec.device(*eigenDevice) += yvec * dyvec;
}

void ExpGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto device = (CPUDevice*) x.device();

    switch (x.type.id) {
    case DType::Float32:
        ExpGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape);
        break;
    case DType::Float64:
        ExpGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}





}
}