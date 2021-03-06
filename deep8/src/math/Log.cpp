#include "math/Log.h"

namespace Deep8 {
namespace Math {

/**
 * y = log(x)
 */
void Log(const Tensor &x, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        LogCPU(x, y);
    } else {
#ifdef HAVE_CUDA
        LogGPU(x, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

/**
 * calculate the grad(x) of Log
 */
void LogGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == dx.elementType  && x.elementType == y.elementType && x.elementType  == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        LogGradCPU(x, dx, y, dy);
    } else {
#ifdef HAVE_CUDA
        LogGradGPU(x, dx, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}


template <typename T>
void LogCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int)yshape.size());

    yvec.device(*eigenDevice) = xvec.log();
}

void LogCPU(const Tensor &x, Tensor &y) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        LogCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        LogCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct LogGradEigenExpr {
    inline T operator()(T dy, T x) const {
        return dy/ x;
    }
};

template <typename T>
void LogGradCPUImpl(CPUDevice *device, T *x, T * dx, const Shape &xshape, T *y, T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());

    dxvec.device(*eigenDevice) += dyvec.binaryExpr(xvec, LogGradEigenExpr<T>());
}

void LogGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto device = (CPUDevice*) x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        LogGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape);
        break;
    case DType::Float64:
        LogGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}




}
}