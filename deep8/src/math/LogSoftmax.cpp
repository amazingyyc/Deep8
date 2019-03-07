#include "math/LogSoftmax.h"

namespace Deep8 {
namespace Math {

void LogSoftmax(const Tensor &x, Tensor &y, int axis, void *maxptr, void *sumptr) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape  == y.shape, "the shape must be same");

    if (-1 == axis) {
        axis = (int)x.shape.nDims - 1;
    }

    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int)x.shape.nDims, "the axis is error");

    if (DeviceType::CPU == x.deviceType()) {
        LogSoftmaxCPU(x, y, axis, maxptr, sumptr);
    } else {
#ifdef HAVE_CUDA
        LogSoftmaxGPU(x, y, axis, maxptr, sumptr);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void LogSoftmaxGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis, void *sumptr) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == dx.elementType  && x.elementType == y.elementType && x.elementType  == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (-1 == axis) {
        axis = (int)x.shape.nDims - 1;
}

    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int)x.shape.nDims, "the axis is error");

    if (DeviceType::CPU == x.deviceType()) {
        LogSoftmaxGradCPU(x, dx, y, dy, axis, sumptr);
    } else {
#ifdef HAVE_CUDA
        LogSoftmaxGradGPU(x, dx, y, dy, axis, sumptr);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void LogSoftmaxCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape, int axis, T *maxptr, T *sumptr) {
    auto eigenDevice = device->eigenDevice;

    int dim0, dim1, dim2;

    dim0 = (int) xshape.batch;
    dim1 = (int) xshape.dim(axis);
    dim2 = 1;

    for (int i = 0; i < axis; ++i) {
        dim0 *= (int) xshape.dim(i);
    }

    for (int i = axis + 1; i < xshape.nDims; ++i) {
        dim2 *= (int) xshape.dim(i);
    }

    Eigen::array<int, 1> reduceDims = { 1 };
    Eigen::array<int, 3> reshape    = { dim0, 1, dim2 };
    Eigen::array<int, 3> broad      = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> xvec(x, dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> yvec(y, dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> maxvec(maxptr, dim0, 1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> sumvec(sumptr, dim0, 1, dim2);

    maxvec.device(*eigenDevice) = xvec.maximum(reduceDims).reshape(reshape);
    sumvec.device(*eigenDevice) = (xvec - maxvec.broadcast(broad)).exp().sum(reduceDims).reshape(reshape);
    yvec.device(*eigenDevice)   = xvec - maxvec.broadcast(broad) - sumvec.log().broadcast(broad);
}

void LogSoftmaxCPU(const Tensor &x, Tensor &y, int axis, void *maxptr, void *sumptr) {
    auto device = (CPUDevice*) x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        LogSoftmaxCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, axis, (float*)maxptr, (float*)sumptr);
        break;
    case DType::Float64:
        LogSoftmaxCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, axis, (double*)maxptr, (double*)sumptr);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void LogSoftmaxGradCPUImpl( CPUDevice *device, 
                            T *x, 
                            T *dx, 
                            const Shape &xshape, 
                            T *y, 
                            T *dy, 
                            const Shape &yshape,
                            int axis, 
                            T *sumptr) {
    auto eigenDevice = device->eigenDevice;

    int dim0, dim1, dim2;

    dim0 = (int) xshape.batch;
    dim1 = (int) xshape.dim(axis);
    dim2 = 1;

    for (int i = 0; i < axis; ++i) {
        dim0 *= (int) xshape.dim(i);
    }

    for (int i = axis + 1; i < xshape.nDims; ++i) {
        dim2 *= (int) xshape.dim(i);
    }

    Eigen::array<int, 1> sumDims = { 1 };
    Eigen::array<int, 3> reshape = { dim0, 1, dim2 };
    Eigen::array<int, 3> broad   = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> sumvec(sumptr, dim0, 1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>>  dxvec(dx,     dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>>   yvec(y,      dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>>  dyvec(dy,     dim0, dim1, dim2);

    sumvec.device(*eigenDevice) = dyvec.sum(sumDims).reshape(reshape);
    dxvec.device(*eigenDevice) += dyvec - yvec.exp() * sumvec.broadcast(broad);
}

void LogSoftmaxGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis, void *sumptr) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        LogSoftmaxGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape, axis, (float*)sumptr);
        break;
    case DType::Float64:
        LogSoftmaxGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, axis, (double*)sumptr);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}
}
