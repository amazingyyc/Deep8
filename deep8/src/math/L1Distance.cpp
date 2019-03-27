#include "math/L1Distance.h"

namespace Deep8 {
namespace Math {

void L1Distance(const Tensor &x, const Tensor &y, Tensor &z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType() && x.deviceType()  == z.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType && x.elementType == z.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the x/y shape must be same");
    DEEP8_ARGUMENT_CHECK(x.shape.batch == z.shape.batch && 1 == z.shape.batchSize(), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        L1DistanceCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        L1DistanceGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void L1DistanceGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
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
        L1DistanceGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        L1DistanceGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void L1DistanceGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
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
        L1DistanceGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        L1DistanceGradYGPU(x, y, dy, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

template <typename T>
void L1DistanceCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape, T *z, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> xvec(x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> yvec(y, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, batch);

    Eigen::array<int, 1> sumdims = { 1 };

    zvec.device(*eigenDevice) = (xvec - yvec).abs().sum(sumdims);
}

void L1DistanceCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        L1DistanceCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, z.data<float>(), z.shape);
        break;
    case DType::Float64:
        L1DistanceCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, z.data<double>(), z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L1DistanceGradXEigenExpr {
    inline T operator()(T x, T y) const {
        if (x > y) {
            return 1;
        } else if (x == y) {
            return 0;
        } else {
            return -1;
        }
    }
};

template <typename T>
void L1DistanceGradXCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, T *y, const Shape &yshape, T *dz, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  xvec( x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dxvec(dx, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  yvec( y, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dzvec(dz, batch, 1);

    Eigen::array<int, 2> broad = { 1, size };

    dxvec.device(*eigenDevice) += dzvec.broadcast(broad) * xvec.binaryExpr(yvec, L1DistanceGradXEigenExpr<T>());
}

void L1DistanceGradXCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        L1DistanceGradXCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), y.shape, dz.data<float>(), dz.shape);
        break;
    case DType::Float64:
        L1DistanceGradXCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), y.shape, dz.data<double>(), dz.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
struct L1DistanceGradYEigenExpr {
    inline T operator()(T x, T y) const {
        if (y > x) {
            return 1;
        } else if (y == x) {
            return 0;
        } else {
            return -1;
        }
    }
};

template <typename T>
void L1DistanceGradYCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, T *dy, const Shape &yshape, T *dz, const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    int batch = (int) xshape.batch;
    int size  = (int) xshape.batchSize();

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  xvec( x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  yvec( y, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dyvec(dy, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dzvec(dz, batch, 1);

    Eigen::array<int, 2> broad = { 1, size };

    dyvec.device(*eigenDevice) += dzvec.broadcast(broad) * xvec.binaryExpr(yvec, L1DistanceGradYEigenExpr<T>());
}

void L1DistanceGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        L1DistanceGradYCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape, dz.data<float>(), dz.shape);
        break;
    case DType::Float64:
        L1DistanceGradYCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape, dz.data<double>(), dz.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}