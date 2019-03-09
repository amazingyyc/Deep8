#include "math/Mean.h"

namespace Deep8 {
namespace Math {

void Mean(const Tensor& x, Tensor& y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(1 == y.shape.size(), "the y size must be 1");

    if (DeviceType::CPU == x.deviceType()) {
        MeanCPU(x, y);
    } else {
#ifdef HAVE_CUDA
        MeanGPU(x, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void MeanGrad(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == dx.elementType && x.elementType == y.elementType && x.elementType == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && y.shape == dy.shape && 1 == y.shape.size(), "the param shape error");

    if (DeviceType::CPU == x.deviceType()) {
        MeanGradCPU(x, dx, y, dy);
    } else {
#ifdef HAVE_CUDA
        MeanGradGPU(x, dx, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void MeanCPUImpl(CPUDevice* device, T* x, const Shape& xshape, T* y, const Shape& yshape) {
    auto eigenDevice = device->eigenDevice;

    int xsize = (int)xshape.size();

    Eigen::array<int, 1> reshapeDims = { 1 };
    Eigen::array<int, 1> sumDims = { 0 };

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, xsize);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, 1);

    yvec.device(*eigenDevice) = xvec.sum(sumDims).reshape(reshapeDims) / T(xsize);
}

void MeanCPU(const Tensor& x, Tensor& y) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        MeanCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        MeanCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void MeanGradCPUImpl(CPUDevice* device, T* x, T* dx, const Shape& xshape, T* dy, const Shape& yshape) {
    auto eigenDevice = device->eigenDevice;

    int xsize = (int)xshape.size();

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, xsize);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, 1);

    Eigen::array<int, 1> broad = { xsize };

    dxvec.device(*eigenDevice) += dyvec.broadcast(broad) / T(xsize);
}

void MeanGradCPU(const Tensor& x, Tensor& dx, const Tensor& y, const Tensor& dy) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        MeanGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, dy.data<float>(), y.shape);
        break;
    case DType::Float64:
        MeanGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, dy.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}