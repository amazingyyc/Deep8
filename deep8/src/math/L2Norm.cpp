#include "math/L2Norm.h"

namespace Deep8 {
namespace Math {

/*
 * y = l2norm(x)
 */
void L2Norm(const Tensor &x, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.batch() == y.batch(), "the shape is error");
    DEEP8_ARGUMENT_CHECK(1 == y.batchSize(), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        L2NormCPU(x, y);
    } else {
#ifdef HAVE_CUDA
        L2NormGPU(x, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void L2NormGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == dx.elementType  && x.elementType == y.elementType && x.elementType  == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && y.shape == dy.shape, "the param shape error");
    DEEP8_ARGUMENT_CHECK(x.batch() == y.batch(), "the shape is error");
    DEEP8_ARGUMENT_CHECK(1 == y.batchSize(), "the shape is error");
    
    if (DeviceType::CPU == x.deviceType()) {
        L2NormGradCPU(x, dx, y, dy);
    } else {
#ifdef HAVE_CUDA
        L2NormGradGPU(x, dx, y, dy);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void L2NormCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

	Eigen::array<int, 1> reshapeDims = { batch };
	Eigen::array<int, 1> sumDims = { 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> xvec(x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, batch);

    yvec.device(*eigenDevice) = xvec.square().sum(sumDims).sqrt().reshape(reshapeDims);
}

void L2NormCPU(const Tensor &x, Tensor &y) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        L2NormCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        L2NormCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void L2NormGradCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, T *y, T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

    Eigen::array<int, 2> broad = { 1, size};

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  xvec( x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dxvec(dx, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  yvec( y, batch, 1);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dyvec(dy, batch, 1);

    dxvec.device(*eigenDevice) += (dyvec / yvec).broadcast(broad) * xvec;
}

void L2NormGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        L2NormGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape);
        break;
    case DType::Float64:
        L2NormGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}