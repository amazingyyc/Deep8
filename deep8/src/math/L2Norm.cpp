#include "math/L2Norm.h"

namespace Deep8 {
namespace Math {

/*
 * y = l2norm(x)
 */
void L2Norm(const Tensor &x, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType()  == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == y.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(1 == y.shape.size(), "the y size must be 1");

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

/*
 * calculate the grad(x) of L2Norm
 */
void L2NormGrad(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == dx.deviceType() && x.deviceType() == y.deviceType() && x.deviceType() == dy.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.type  == dx.type  && x.type == y.type && x.type  == dy.type, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == dx.shape && y.shape == dy.shape && 1 == y.shape.size(), "the param shape error");

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

	Eigen::array<int, 1> reshapeDims = { 1 };
	Eigen::array<int, 1> sumDims = { 0 };

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, (int)xshape.size());
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, (int)yshape.size());

    yvec.device(*eigenDevice) = xvec.square().sum(sumDims).sqrt().reshape(reshapeDims);
}

void L2NormCPU(const Tensor &x, Tensor &y) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        L2NormCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        L2NormCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

template <typename T>
void L2NormGradCPUImpl(CPUDevice *device, T *x, T *dx, const Shape &xshape, T *y, T *dy, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    int xsize = (int)xshape.size();
    int ysize = (int)yshape.size();

    Eigen::array<int, 1> broad = { xsize };

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  xvec( x, xsize);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dxvec(dx, xsize);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>>  yvec( y, ysize);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> dyvec(dy, ysize);

    dxvec.device(*eigenDevice) += (dyvec / yvec).broadcast(broad) * xvec;
}

void L2NormGradCPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy) {
    auto device = (CPUDevice*)x.device();

    switch (x.type.id) {
    case DType::Float32:
        L2NormGradCPUImpl<float>(device, x.data<float>(), dx.data<float>(), x.shape, y.data<float>(), dy.data<float>(), y.shape);
        break;
    case DType::Float64:
        L2NormGradCPUImpl<double>(device, x.data<double>(), dx.data<double>(), x.shape, y.data<double>(), dy.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }

}


}   
}
