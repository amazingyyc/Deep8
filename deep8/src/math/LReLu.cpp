#include "math/LReLu.h"

namespace Deep8 {
namespace Math {

void LReLu(const Tensor &x, const float a, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == y.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        LReLuCPU(x, a, y);
    } else {
#ifdef HAVE_CUDA
        LReLuGPU(x, a, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void LReLuGrad(const Tensor &x, Tensor &dx, const float a, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == dx.type  && x.type == y.type && x.type  == dy.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        LReLuGradCPU(x, dx, a, y, dy);
    } else {
#ifdef HAVE_CUDA
        LReLuGradGPU(x, dx, a, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
struct LReLuEigenExpr {
    T a;

    LReLuEigenExpr(T aa): a(aa) {
    } 

	inline T operator()(T x) const {
		return (x > 0 ? x : a * x);
	}
};

template <typename T>
void LReLuCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T a,T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int)yshape.size());

    yvec.device(*eigenDevice) = xvec.unaryExpr(LReLuEigenExpr<T>(a));
}

void LReLuCPU(const Tensor &x, const float a, Tensor &y) {
    auto device = (CPUDevice*) x.device();

    switch (x.type.id) {
    case DType::Float32:
        LReLuCPUImpl<float>(device, x.data<float>(), x.shape, a, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        LReLuCPUImpl<double>(device, x.data<double>(), x.shape, double(a), y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
struct LReLuGradEigenExpr {
    T a;

    LReLuGradEigenExpr(T aa): a(aa) {
    }

	inline T operator()(T dy, T x) const {
		return (x > 0 ? dy : a * dy);
	}
};

template <typename T>
void LReLuGradCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, T a, T *y, T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());

    dxvec.device(*eigenDevice) += dyvec.binaryExpr(xvec, LReLuGradEigenExpr<T>(a));
}

void LReLuGradCPU(const Tensor &x, Tensor &dx, const float a, const Tensor &y, const Tensor &dy) {
    auto device = (CPUDevice*) x.device();

    switch (x.type.id) {
    case DType::Float32:
        LReLuGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, a, y.data<float>(), dy.data<float>(), y.shape);
        break;
    case DType::Float64:
        LReLuGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, double(a), y.data<double>(), dy.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


}
}