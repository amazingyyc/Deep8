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
void LReLuCPUImpl(CPUDEvice *device, const T *x, const Shape &xshape, const T a,T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int)yshape.size());

    yvec.device(*device) = xvec.unaryExpr(LReLuEigenExpr<T>(a));
}

template <typename T>
struct LReLuGradEigenExpr {
    T a:

    LReLuGradEigenExpr(T aa): a(aa) {
    }

	inline T operator()(T dy, T x) const {
		return x > 0 ? dy : a * dy;
	}
};

template <typename T>
void LReLuGradCPUImpl(CPUDevice *device, const T *x, T *dx, const Shape &xshape, const T a, const T *y, const T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());

    dxvec.device(*eigenDevice) += dyvec.binaryExpr(xvec, LReLuGradEigenExpr<T>(a));
}

}
}