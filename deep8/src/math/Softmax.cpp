#include "math/Softmax.h"

namespace Deep8 {
namespace Math {

void Softmax(const Tensor &x, Tensor &y, int axis, void *ptr) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the shape must be same");

    if (-1 == axis) {
        axis = (int)x.shape.nDims - 1;
    }

    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int) x.shape.nDims, "the axis is error");

    if (DeviceType::CPU == x.deviceType()) {
        SoftmaxCPU(x, y, axis, ptr);
    } else {
#ifdef HAVE_CUDA
        SoftmaxGPU(x, y, axis, ptr);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void SoftmaxGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis, void *ptr) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == dx.elementType  && x.elementType == y.elementType && x.elementType  == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape && x.shape == dy.shape, "the param shape must be same");

    if (-1 == axis) {
        axis = (int)x.shape.nDims - 1;
    }

    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int) x.shape.nDims, "the axis is error");

    if (DeviceType::CPU == x.deviceType()) {
        SoftmaxGradCPU(x, dx, y, dy, axis, ptr);
    } else {
#ifdef HAVE_CUDA
        SoftmaxGradGPU(x, dx, y, dy, axis, ptr);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void SoftmaxCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape, int axis, T *ptr) {
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
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> tvec(ptr, dim0, 1, dim2);

    tvec.device(*eigenDevice) = xvec.maximum(reduceDims).reshape(reshape);
    yvec.device(*eigenDevice) = (xvec - tvec.broadcast(broad)).exp();
    tvec.device(*eigenDevice) = yvec.sum(reduceDims).reshape(reshape);
    yvec.device(*eigenDevice) = yvec / tvec.broadcast(broad);
}

void SoftmaxCPU(const Tensor &x, Tensor &y, int axis, void *ptr) {
    auto device = (CPUDevice*) x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        SoftmaxCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, axis, (float*)ptr);
        break;
    case DType::Float64:
        SoftmaxCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, axis, (double*)ptr);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void SoftmaxGradCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, T *y, T *dy, const Shape &yshape, int axis, T *ptr) {
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

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>>  tvec(ptr, dim0, 1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dxvec(dx,  dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>>  yvec(y,   dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dyvec(dy,  dim0, dim1, dim2);

    tvec.device(*eigenDevice) = (yvec * dyvec).sum(sumDims).reshape(reshape);
    dxvec.device(*eigenDevice) += (dyvec - tvec.broadcast(broad)) * yvec;
}

void SoftmaxGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis, void *ptr) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        SoftmaxGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape, axis, (float*)ptr);
        break;
    case DType::Float64:
        SoftmaxGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, axis, (double*)ptr);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}