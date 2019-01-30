#include "math/Sigmoid.h"

namespace Deep8 {
namespace Math {

void Sigmoid(const Tensor &x, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == y.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        SigmoidCPU(x, y);
    } else {
#ifdef HAVE_CUDA
        SigmoidGPU(x, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void SigmoidGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == dx.type  && x.type == y.type && x.type  == dy.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        SigmoidGradCPU(x, dx, y, dy);
    } else {
#ifdef HAVE_CUDA
        SigmoidGradGPU(x, dx, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
struct SigmoidEigenExpr {
	inline T operator()(T x) const {
		return 0.5 + 0.5 * tanh(0.5 * x);
	}
};

template <typename T>
void SigmoidCPUImpl(CPUDevice *device, const T *x, const Shape &xshape, T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int)yshape.size());

    yvec.device(*eigenDevice) = xvec.unaryExpr(SigmoidEigenExpr<T>());
}

void SigmoidCPU(const Tensor &x, Tensor &y) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        SigmoidCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>, y.shape);
        break;
    case DType::Float64:
        SigmoidCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>, y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
struct SigmoidGradEigenExpr {
	inline T operator()(T dy, T y) const {
		return dy * y * (1.0 - y);
	}
};

template <typename T>
void SigmoidGradCPUImpl(CPUDevice *device, const T *x, T *dx, const Shape &xshape, const T *y, const T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  yvec( y, (int)yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());

    dxvec.device(*eigenDevice) += dyvec.binaryExpr(yvec, SigmoidGradEigenExpr<T>());
}

void SigmoidGradCPU(const Tensor &x, Tensor &dx, const float a, const float b, const Tensor &y, const Tensor &dy) {
    auto device = x.device();

    case DType::Float32:
        SigmoidGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>, y.shape);
        break;
    case DType::Float64:
        SigmoidGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>, y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}



}
}