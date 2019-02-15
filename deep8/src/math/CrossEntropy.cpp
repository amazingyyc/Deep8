#include "math/CrossEntropy.h"

namespace Deep8 {
namespace Math {

void CrossEntropy(const Tensor &x, const Tensor &y, Tensor &z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType() && x.deviceType()  == z.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == y.type && x.type  == z.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.size() == y.size() && x.batch() == y.batch(), "the x/y shape error");
    DEEP8_ARGUMENT_CHECK(1 == z.size(), "the z size must be 1");

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

    DEEP8_ARGUMENT_CHECK(x.type == dx.type &&
                         x.type ==  y.type &&
                         x.type ==  z.type &&
                         x.type == dz.type, "the param data type must be same");
    
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape, "the x shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape && 1 == z.size(), "the z shape is error");

    DEEP8_ARGUMENT_CHECK(x.size() == y.size() && x.batch() == y.batch(), "the x/y shape error");

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

    DEEP8_ARGUMENT_CHECK(x.type == dy.type &&
                         x.type ==  y.type &&
                         x.type ==  z.type &&
                         x.type == dz.type, "the param data type must be same");
    
    DEEP8_ARGUMENT_CHECK(y.shape == dy.shape, "the y shape is error");
    DEEP8_ARGUMENT_CHECK(z.shape == dz.shape && 1 == z.size(), "the z shape is error");

    DEEP8_ARGUMENT_CHECK(x.size() == y.size() && x.batch() == y.batch(), "the x/y shape error");

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

    auto batch = xshape.batch;
    auto scale = -T(1) / T(batch);

    Eigen::array<int, 1> sumDims = { 0 };
    Eigen::array<int, 1> reshape = { 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int) xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int) yshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, (int) zshape.size());

    zvec.device(*eigenDevice) = (yvec * xvec.log()).sum(sumDims).reshape(reshape) * scale;
}

void CrossEntropyCPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
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
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
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

    auto batch = xshape.batch;
    auto scale = -T(1) * dz[0] / T(batch);

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  yvec( y, (int)yshape.size());

    dxvec.device(*eigenDevice) += (yvec / xvec) * scale;
}

void CrossEntropyGradXCPU(const Tensor &x, 
                        Tensor &dx, 
                        const Tensor &y, 
                        const Tensor &z, 
                        const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
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
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
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

    auto batch = yshape.batch;
    auto scale = -T(1) * dz[0] / T(batch);

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, (int)yshape.size());

    dyvec.device(*eigenDevice) += xvec.log() * scale;
}

void CrossEntropyGradYCPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
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
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

}
}