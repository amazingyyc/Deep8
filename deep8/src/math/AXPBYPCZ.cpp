#include "math/AXPBYPCZ.h"

namespace Deep8 {
namespace Math {

void AXPBYPCZ(const Tensor& x, float a, const Tensor& y, float b, const Tensor& z, float c, Tensor& w) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType() && x.deviceType() == z.deviceType() && x.deviceType() == w.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType && x.elementType == z.elementType && x.elementType == w.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape.size() == y.shape.size() && x.shape.size() == z.shape.size() && x.shape.size() == w.shape.size(), "the param shape size must be same");

    if (DeviceType::CPU == x.deviceType()) {
        AXPBYPCZCPU(x, a, y, b, z, c, w);
    } else {
#ifdef HAVE_CUDA
        AXPBYPCZGPU(x, a, y, b, z, c, w);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void AXPBYPCZCPUImpl(CPUDevice* device, T *x, T a, T *y, T b, T *z, T c, T *w, int size) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> zvec(z, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> wvec(w, size);

    wvec.device(*eigenDevice) = xvec * a + yvec * b + zvec * c;
}

void AXPBYPCZCPU(const Tensor& x, float a, const Tensor& y, float b, const Tensor& z, float c, Tensor& w) {
    auto device = (CPUDevice*)x.device();
    auto size = (int)x.shape.size();

    switch (x.elementType.id) {
    case DType::Float32:
        AXPBYPCZCPUImpl<float>(device, x.data<float>(), a, y.data<float>(), b, z.data<float>(), c, w.data<float>(), size);
        break;
    case DType::Float64:
        AXPBYPCZCPUImpl<double>(device, x.data<double>(), (double)a, y.data<double>(), (double)b, z.data<double>(), (double)c, w.data<double>(), size);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}
}