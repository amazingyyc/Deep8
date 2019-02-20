#include "math/Abs.h"

namespace Deep8 {
namespace Math {

void Abs(const Tensor &x, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        AbsCPU(x, y);
    } else {
#ifdef HAVE_CUDA
        AbsGPU(x, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

/**
 * calculate the grad(x)
 */
void AbsGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == dx.elementType && x.elementType == y.elementType && x.elementType == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        AbsGradCPU(x, dx, y, dy);
    } else {
#ifdef HAVE_CUDA
        AbsGradGPU(x, dx, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}


template <typename T>
void AbsCPUImpl(CPUDevice *device, T *x, T *y, int n) {
    auto eigenDevice = device->eigenDevice;
    
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

    yvec.device(*eigenDevice) = xvec.abs();
}

void AbsCPU(const Tensor &x, Tensor &y) {
    auto device = (CPUDevice*) x.device();
    auto n      = (int) x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        AbsCPUImpl<float>(device, x.data<float>(), y.data<float>(), n);
        break;
    case DType::Float64:
        AbsCPUImpl<double>(device, x.data<double>(), y.data<double>(), n);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct AbsGradEigenExpr {
    inline T operator()(T dy, T x) const {
        if (x >= T(0)) {
            return dy;
        } else {
            return -dy;
        }
    }
};

template <typename T>
void AbsGradCPUImpl(CPUDevice *device, T *x, T *dx, T *dy, int n) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, n);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, n);

    dxvec.device(*eigenDevice) += dyvec.binaryExpr(xvec, AbsGradEigenExpr<T>());
}

void AbsGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto device = (CPUDevice*)x.device();
    auto n      = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        AbsGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), dy.data<float>(), n);
        break;
    case DType::Float64:
        AbsGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), dy.data<double>(), n);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}



}
}