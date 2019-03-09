#include "math/Dot.h"

namespace Deep8 {
namespace Math {

void Dot(const Tensor &x, const Tensor &y, Tensor &z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType() && x.deviceType()  == z.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType && x.elementType == z.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the x/y shape must be same");
    DEEP8_ARGUMENT_CHECK(x.shape.batch == z.shape.batch && 1 == z.shape.batchSize(), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        DotCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        DotGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void DotGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && 
                         x.deviceType() ==  y.deviceType() &&
                         x.deviceType() ==  z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.elementType == dx.elementType &&
                         x.elementType ==  y.elementType &&
                         x.elementType ==  z.elementType &&
                         x.elementType == dz.elementType, "the param data type must be same");

    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && x.shape == y.shape, "the shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape, "the z shape is error");
    DEEP8_ARGUMENT_CHECK(x.shape.batch == z.shape.batch && 1 == z.shape.batchSize(), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        DotGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        DotGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void DotGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() ==  y.deviceType() &&
                         x.deviceType() == dy.deviceType() &&
                         x.deviceType() ==  z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.elementType ==  y.elementType &&
                         x.elementType == dy.elementType &&
                         x.elementType ==  z.elementType &&
                         x.elementType == dz.elementType, "the param data type must be same");

    DEEP8_ARGUMENT_CHECK(x.shape == y.shape && y.shape == dy.shape, "the y shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape, "the z shape is error");
    DEEP8_ARGUMENT_CHECK(x.shape.batch == z.shape.batch && 1 == z.shape.batchSize(), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        DotGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        DotGradYGPU(x, y, dy, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

template <typename T>
void DotCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> xvec(x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> yvec(y, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, batch);

    Eigen::array<int, 1> sumdims = { 1 };

    zvec.device(*eigenDevice) = (xvec * yvec).sum(sumdims);
}

void DotCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        DotCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), z.shape);
        break;
    case DType::Float64:
        DotCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void DotGradXCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, T *y, const Shape &yshape, T *dz, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dxvec(dx, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  yvec( y, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dzvec(dz, batch, 1);

    Eigen::array<int, 2> broad = { 1, size };
    
    dxvec.device(*eigenDevice) += dzvec.broadcast(broad) * yvec;
}

void DotGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        DotGradXCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
        break;
    case DType::Float64:
        DotGradXCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void DotGradYCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, T *dy, const Shape &yshape, T *dz, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  xvec( x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dyvec(dy, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dzvec(dz, batch, 1);

    Eigen::array<int, 2> broad = { 1, size };

    dyvec.device(*eigenDevice) += dzvec.broadcast(broad) * xvec;
}

void DotGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        DotGradYCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape, dz.data<float>(), dz.shape);
        break;
    case DType::Float64:
        DotGradYCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, dz.data<double>(), dz.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}   