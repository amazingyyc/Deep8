#include "math/PReLu.h"

namespace Deep8 {
namespace Math {

void PReLu(const Tensor& x, const Tensor& y, Tensor& z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType() && x.deviceType() == z.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType && x.elementType == z.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape && x.shape == z.shape, "the x/y/z shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        PReLuCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        PReLuGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void PReLuGradX(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() &&
                         x.deviceType() == y.deviceType() &&
                         x.deviceType() == z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.elementType == dx.elementType &&
                         x.elementType == y.elementType &&
                         x.elementType == z.elementType &&
                         x.elementType == dz.elementType, "the param data type must be same");

    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == z.shape && x.shape == dz.shape, "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        PReLuGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        PReLuGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void PReLuGradY(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType() &&
                         x.deviceType() == dy.deviceType() &&
                         x.deviceType() == z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType &&
                         x.elementType == dy.elementType &&
                         x.elementType == z.elementType &&
                         x.elementType == dz.elementType, "the param data type must be same");

    DEEP8_ARGUMENT_CHECK(x.shape == y.shape && x.shape == dy.shape && x.shape == z.shape && x.shape == dz.shape, "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        PReLuGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        PReLuGradYGPU(x, y, dy, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

template <typename T>
struct PReLuEigenExpr {
    inline T operator()(T x, T y) const {
        return (x > 0 ? x : x * y);
    }
};

template <typename T>
void PReLuCPUImpl(CPUDevice* device, T* x, const Shape& xshape, T* y, const Shape& yshape, T* z, const Shape& zshape) {
    auto eigenDevice = device->eigenDevice;

    int size = (int) xshape.size();

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, size);

    zvec.device(*eigenDevice) = xvec.binaryExpr(yvec, PReLuEigenExpr<T>());
}

void PReLuCPU(const Tensor& x, const Tensor& y, Tensor& z) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        PReLuCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), z.shape);
        break;
    case DType::Float64:
        PReLuCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct PReLuGradXEigenExpr {
    inline T operator()(T x, T y) const {
        return (x > 0 ? 1 : y);
    }
};

template <typename T>
void PReLuGradXCPUImpl(CPUDevice* device, T* x, T* dx, const Shape& xshape, T* y, const Shape& yshape, T* z, T* dz, const Shape& zshape) {
    auto eigenDevice = device->eigenDevice;

    int size = (int) xshape.size();

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  yvec( y, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dzvec(dz, size);

    dxvec.device(*eigenDevice) += xvec.binaryExpr(yvec, PReLuGradXEigenExpr<T>()) * dzvec;
}

void PReLuGradXCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& z, const Tensor& dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        PReLuGradXCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), dz.data<float>(), dz.shape);
        break;
    case DType::Float64:
        PReLuGradXCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), dz.data<double>(), dz.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct PReLuGradYEigenExpr {
    inline T operator()(T dz, T x) const {
        return (x > 0 ? 0 : dz * x);
    }
};

template <typename T>
void PReLuGradYCPUImpl(CPUDevice* device, T* x, const Shape& xshape, T* y, T *dy, const Shape& yshape, T* z, T* dz, const Shape& zshape) {
    auto eigenDevice = device->eigenDevice;

    int size = (int)xshape.size();

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dzvec(dz, size);

    dyvec.device(*eigenDevice) += dzvec.binaryExpr(xvec, PReLuGradYEigenExpr<T>());
}

void PReLuGradYCPU(const Tensor& x, const Tensor& y, Tensor& dy, const Tensor& z, const Tensor& dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        PReLuGradYCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape, z.data<float>(), dz.data<float>(), dz.shape);
        break;
    case DType::Float64:
        PReLuGradYCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, z.data<double>(), dz.data<double>(), dz.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}



}
}