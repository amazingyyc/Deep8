#include "math/CrossEntropy.h"

namespace Deep8 {
namespace Math {

void CrossEntropy(const Tensor &x, const Tensor &y, Tensor &z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType() && x.deviceType()  == z.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType && x.elementType == z.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.size() == y.size() && x.batch() == y.batch(), "the x/y shape error");
    DEEP8_ARGUMENT_CHECK(2 == z.shape.nDims && x.batch() == z.batch() && 1 == z.dim(0), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        CrossEntropyCPU(x, y, z);
    } else {
#ifdef HAVE_CUDA
        CrossEntropyGPU(x, y, z);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void CrossEntropyGradX(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && 
                         x.deviceType() ==  y.deviceType() &&
                         x.deviceType() ==  z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.elementType == dx.elementType &&
                         x.elementType ==  y.elementType &&
                         x.elementType ==  z.elementType &&
                         x.elementType == dz.elementType, "the param data type must be same");
    
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape, "the x shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape, "the z shape is error");

    DEEP8_ARGUMENT_CHECK(x.size() == y.size() && x.batch() == y.batch(), "the x/y shape error");
    DEEP8_ARGUMENT_CHECK(2 == z.shape.nDims && x.batch() == z.batch() && 1 == z.dim(0), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        CrossEntropyGradXCPU(x, dx, y, z, dz);
    } else {
#ifdef HAVE_CUDA
        CrossEntropyGradXGPU(x, dx, y, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

void CrossEntropyGradY(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dy.deviceType() && 
                         x.deviceType() ==  y.deviceType() &&
                         x.deviceType() ==  z.deviceType() &&
                         x.deviceType() == dz.deviceType(), "the param device type must be same");

    DEEP8_ARGUMENT_CHECK(x.elementType == dy.elementType &&
                         x.elementType ==  y.elementType &&
                         x.elementType ==  z.elementType &&
                         x.elementType == dz.elementType, "the param data type must be same");
    
    DEEP8_ARGUMENT_CHECK(y.shape == dy.shape, "the y shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape, "the z shape is error");

    DEEP8_ARGUMENT_CHECK(x.size() == y.size() && x.batch() == y.batch(), "the x/y shape error");
    DEEP8_ARGUMENT_CHECK(2 == z.shape.nDims && x.batch() == z.batch() && 1 == z.dim(0), "the z shape is error");

    if (DeviceType::CPU == x.deviceType()) {
        CrossEntropyGradYCPU(x, y, dy, z, dz);
    } else {
#ifdef HAVE_CUDA
        CrossEntropyGradYGPU(x, y, dy, z, dz);
#else 
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif
    }
}

template <typename T>
void CrossEntropyCPUImpl(CPUDevice *device, 
                        T *x, 
                        const Shape &xshape, 
                        T *y,
                        const Shape &yshape, 
                        T *z, 
                        const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    auto batch = (int) xshape.batch;
    auto size  = (int) xshape.batchSize();

    Eigen::array<int, 1> sumDims = { 1 };
    Eigen::array<int, 1> reshape = { batch };

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> xvec(x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> yvec(y, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, batch);

    zvec.device(*eigenDevice) = (-yvec * xvec.log()).sum(sumDims).reshape(reshape);
}

void CrossEntropyCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        CrossEntropyCPUImpl<float>( device, 
                                    x.data<float>(), 
                                    x.shape, 
                                    y.data<float>(), 
                                    y.shape, 
                                    z.data<float>(), 
                                    z.shape);
        break;
    case DType::Float64:
        CrossEntropyCPUImpl<double>( device, 
                                    x.data<double>(), 
                                    x.shape, 
                                    y.data<double>(), 
                                    y.shape, 
                                    z.data<double>(), 
                                    z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void CrossEntropyGradXCPUImpl(CPUDevice *device, 
                              T *x, 
                              T *dx, 
                              const Shape &xshape, 
                              T *y, 
                              const Shape &yshape, 
                              T *z, 
                              T *dz, 
                              const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    auto batch = (int) xshape.batch;
    auto size  = (int) xshape.batchSize();

    Eigen::array<int, 2> broad = { 1, size };

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  xvec( x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dxvec(dx, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  yvec( y, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dzvec(dz, batch, 1);

    dxvec.device(*eigenDevice) -= dzvec.broadcast(broad) * yvec / xvec;
}

void CrossEntropyGradXCPU(const Tensor &x, 
                        Tensor &dx, 
                        const Tensor &y, 
                        const Tensor &z, 
                        const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        CrossEntropyGradXCPUImpl<float>(device, 
                x.data<float>(), 
                dx.data<float>(), 
                x.shape, 
                y.data<float>(),
                y.shape,
                z.data<float>(),
                dz.data<float>(),
                z.shape);
        break;
    case DType::Float64:
        CrossEntropyGradXCPUImpl<double>(device, 
                x.data<double>(), 
                dx.data<double>(), 
                x.shape, 
                y.data<double>(),
                y.shape,
                z.data<double>(),
                dz.data<double>(),
                z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
void CrossEntropyGradYCPUImpl(CPUDevice *device, 
                              T *x, 
                              const Shape &xshape, 
                              T *y, 
                              T *dy,
                              const Shape &yshape, 
                              T *z,
                              T *dz, 
                              const Shape &zshape) {
    auto eigenDevice = device->eigenDevice;

    auto batch = (int)yshape.batch;
    auto size  = (int)yshape.batchSize();

    Eigen::array<int, 2> broad = { 1, size };

    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>>  xvec( x, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dyvec(dy, batch, size);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> dzvec(dz, batch, 1);

    dyvec.device(*eigenDevice) -= xvec.log() * dzvec.broadcast(broad);
}

void CrossEntropyGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        CrossEntropyGradYCPUImpl<float>(device, 
                    x.data<float>(), 
                    x.shape, 
                    y.data<float>(), 
                    dy.data<float>(), 
                    y.shape, 
                    z.data<float>(), 
                    dz.data<float>(), 
                    z.shape);
        break;
    case DType::Float64:
        CrossEntropyGradYCPUImpl<double>(device, 
                    x.data<double>(), 
                    x.shape, 
                    y.data<double>(), 
                    dy.data<double>(), 
                    y.shape, 
                    z.data<double>(), 
                    dz.data<double>(), 
                    z.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}