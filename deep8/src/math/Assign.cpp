#include "math/Assign.h"

namespace Deep8 {
namespace Math {

void Assign(const Tensor &x, Tensor &y) {
    DEEP8_ARGUMENT_CHECK(x.deviceType() == y.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(x.elementType == y.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(x.shape == y.shape, "the param shape must be same");

    if (DeviceType::CPU == x.deviceType()) {
        AssignCPU(x, y);
    } else {
#ifdef HAVE_CUDA
        AssignGPU(x, y);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void AssignCPUImpl(CPUDevice *device, T *x, const Shape &xshape, T *y, const Shape &yshape) {
    auto eigenDevice = device->eigenDevice;

    int n = (int) xshape.size();

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> xvec(x, n);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> yvec(y, n);

    yvec.device(*eigenDevice) = xvec;
}

void AssignCPU(const Tensor &x, Tensor &y) {
    auto device = (CPUDevice*) x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        AssignCPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape);
        break;
    case DType::Float64:
        AssignCPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


}
}