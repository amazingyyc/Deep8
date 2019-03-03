#include "math/RMSPropUpdate.h"

namespace Deep8 {
namespace Math {

void RMSPropUpdate(Tensor& value, Tensor& gradient, Tensor& accumulate, float rho, float epsilon, float learningRate, float weightDecay) {
    DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType() && value.deviceType() == accumulate.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(value.elementType == gradient.elementType && value.elementType == accumulate.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(value.shape.size() == gradient.shape.size() && value.shape.size() == accumulate.shape.size(), "the param shape size must be same");

    if (DeviceType::CPU == value.deviceType()) {
        RMSPropUpdateCPU(value, gradient, accumulate, rho, epsilon, learningRate, weightDecay);
    } else {
#ifdef HAVE_CUDA
        RMSPropUpdateGPU(value, gradient, accumulate, rho, epsilon, learningRate, weightDecay);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void RMSPropUpdateCPUImpl(CPUDevice* device, T* value, T* gradient, T* accumulate, int size, T rho, T epsilon, T learningRate, T weightDecay) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> valuevec(value, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> gradientvec(gradient, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> accumulatevec(accumulate, size);

    gradientvec.device(*eigenDevice) += valuevec * weightDecay;
    accumulatevec.device(*eigenDevice) = accumulatevec * rho + gradientvec.square() * (1 - rho);
    valuevec.device(*eigenDevice) -= gradientvec / (accumulatevec + epsilon).sqrt() * learningRate;
}

void RMSPropUpdateCPU(Tensor& value, Tensor& gradient, Tensor& accumulate, float rho, float epsilon, float learningRate, float weightDecay) {
    auto device = (CPUDevice*)value.device();
    auto size   = (int)value.shape.size();

    switch (value.elementType.id) {
    case DType::Float32:
        RMSPropUpdateCPUImpl<float>(device, value.data<float>(), gradient.data<float>(), accumulate.data<float>(), size, rho, epsilon, learningRate, weightDecay);
        break;
    case DType::Float64:
        RMSPropUpdateCPUImpl<double>(device, value.data<double>(), gradient.data<double>(), accumulate.data<double>(), size, (double)rho, (double)epsilon, (double)learningRate, (double)weightDecay);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << value.elementType.name << " is not support");
        break;
    }
}

}
}