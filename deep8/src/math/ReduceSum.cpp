#include "math/ReduceSum.h"

namespace Deep8 {
namespace Math {

void ReduceSum(const Tensor &x, Tensor &y, int axis) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.batch() == y.batch(), "the shape is error");

    if (-1 == axis) {
        axis = x.shape.nDims - 1;
    }

    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int)x.shape.nDims, "the axis is error");

    int size = x.shape.batch;

    for (int i = 0; i < axis; ++i) {
        size *= (int)x.shape.dim(i);
    }

    for (int i = axis + 1; i < x.shape.nDims; ++i) {
        size *= (int)x.shape.dim(i);
    }

    DEEP8_ARGUMENT_CHECK(size == (int)y.size(), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        ReduceSumCPU(x, y, axis);
    } else {
#ifdef HAVE_CUDA
        ReduceSumGPU(x, y, axis);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

void ReduceSumGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType  == dx.elementType  && x.elementType == y.elementType && x.elementType  == dy.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && y.shape == dy.shape, "the x/dx or y/dy shape must be same");
    DEEP8_ARGUMENT_CHECK(x.batch() == y.batch(), "the shape is error");

    if (-1 == axis) {
        axis = x.shape.nDims - 1;
    }

    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int)x.shape.nDims, "the axis is error");

    int size = x.shape.batch;

    for (int i = 0; i < axis; ++i) {
        size *= (int)x.shape.dim(i);
    }

    for (int i = axis + 1; i < x.shape.nDims; ++i) {
        size *= (int)x.shape.dim(i);
    }

    DEEP8_ARGUMENT_CHECK(size == (int)y.size(), "the shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        ReduceSumGradCPU(x, dx, y, dy, axis);
    } else {
#ifdef HAVE_CUDA
        ReduceSumGradGPU(x, dx, y, dy, axis);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}


template <typename T>
void ReduceSumCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape, int axis) {
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
    Eigen::array<int, 2> reshape    = { dim0, dim2 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> xvec(x, dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> yvec(y, dim0, dim2);

    yvec.device(*eigenDevice) = xvec.sum(reduceDims).reshape(reshape);
}

void ReduceSumCPU(const Tensor &x, Tensor &y, int axis) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        ReduceSumCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, axis);
        break;
    case DType::Float64:
        ReduceSumCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, axis);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void ReduceSumGradCPUImpl(CPUDevice*device, T *x, T *dx, const Shape &xshape, T *y, T *dy, const Shape &yshape, int axis) {
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
    
    Eigen::array<int, 3> broad = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dxvec(dx, dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dyvec(dy, dim0,    1, dim2);

    dxvec.device(*eigenDevice) += dyvec.broadcast(broad);
}

void ReduceSumGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        ReduceSumGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape, axis);
        break;
    case DType::Float64:
        ReduceSumGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, axis);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}    
}
