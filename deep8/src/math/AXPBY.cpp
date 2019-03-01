#include "math/AXPBY.h"

namespace Deep8 {
namespace Math {

void AXPBY(const Tensor& x, float alpha, const Tensor& y, float beta, Tensor& z) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType() && x.deviceType() == z.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType && x.elementType == z.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape.size() == y.shape.size() && x.shape.size() == z.shape.size(), "the param shape size must be same");

    if (DeviceType::CPU == x.deviceType()) {
        AXPBYCPU(x, alpha, y, beta, z);
    } else {
#ifdef HAVE_CUDA
        AXPBYGPU(x, alpha, y, beta, z);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void AXPBYCPUImpl(CPUDevice *device, T *x, T alpha, T *y, T beta, T *z, int size) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, size);

    zvec.device(*eigenDevice) = xvec * alpha + yvec * beta;
}

void AXPBYCPU(const Tensor& x, float alpha, const Tensor& y, float beta, Tensor& z) {
    auto device = (CPUDevice*)x.device();
    auto size   = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        AXPBYCPUImpl<float>(device, x.data<float>(), alpha, y.data<float>(), beta, z.data<float>(), size);
        break;
    case DType::Float64:
        AXPBYCPUImpl<double>(device, x.data<double>(), (double)alpha, y.data<double>(), (double)beta, z.data<double>(), size);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}
}