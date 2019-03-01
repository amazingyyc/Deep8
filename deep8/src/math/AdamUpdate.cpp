#include "math/AdamUpdate.h"

namespace Deep8 {
namespace Math {

void AdamUpdate(Tensor& value, Tensor& gradient, Tensor& m, Tensor& v, float beta1, float beta2, float epsilon, float learningRate, float weightDecay, int64_t steps) {
    DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType() && value.deviceType() == m.deviceType() && value.deviceType() == v.deviceType(), "the param device type must be same");
    DEEP8_ARGUMENT_CHECK(value.elementType == gradient.elementType && value.elementType == m.elementType && value.elementType == v.elementType, "the param data type must be same");
    DEEP8_ARGUMENT_CHECK(value.shape.size() == gradient.shape.size() && value.shape.size() == m.shape.size() && value.shape.size() == v.shape.size(), "the param shape size must be same");

    if (DeviceType::CPU == value.deviceType()) {
        AdamUpdateCPU(value, gradient, m, v, beta1, beta2, epsilon, learningRate, weightDecay, steps);
    } else {
#ifdef HAVE_CUDA
        AdamUpdateGPU(value, gradient, m, v, beta1, beta2, epsilon, learningRate, weightDecay, steps);
#else
        DEEP8_RUNTIME_ERROR("do not have a GPU");
#endif  
    }
}

template <typename T>
void AdamUpdateCPUImpl(CPUDevice* device, T* value, T* gradient, T* m, T* v, int size, T beta1, T beta2, T epsilon, T learningRate, T weightDecay, int64_t steps) {
    auto eigenDevice = device->eigenDevice;

    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> valuevec(value, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> gradientvec(gradient, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> mvec(m, size);
    Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> vvec(v, size);

    gradientvec.device(*eigenDevice) += valuevec * weightDecay;

    mvec.device(*eigenDevice) = mvec * beta1 + gradientvec * (1 - beta1);
    vvec.device(*eigenDevice) = vvec * beta2 + gradientvec.square() * (1 - beta2);

    auto ratio = learningRate * std::sqrt(1 - std::pow(beta2, T(steps))) / (1 - std::pow(beta1, T(steps)));

    valuevec.device(*eigenDevice) -= mvec / (vvec.sqrt() + epsilon) * ratio;
}

void AdamUpdateCPU(Tensor& value, Tensor& gradient, Tensor& m, Tensor& v, float beta1, float beta2, float epsilon, float learningRate, float weightDecay, int64_t steps) {
    auto device = (CPUDevice*)value.device();
    auto size = (int)value.shape.size();

    switch (value.elementType.id) {
    case DType::Float32:
        AdamUpdateCPUImpl<float>(device, value.data<float>(), gradient.data<float>(), m.data<float>(), v.data<float>(), size, beta1, beta2, epsilon, learningRate, weightDecay, steps);
        break;
    case DType::Float64:
        AdamUpdateCPUImpl<double>(device, value.data<double>(), gradient.data<double>(), m.data<double>(), v.data<double>(), size, (double)beta1, (double)beta2, (double)epsilon, (double)learningRate, (double)weightDecay, steps);
        break;
    default:
        DEEP8_RUNTIME_ERROR("type " << value.elementType.name << " is not support");
        break;
    }
}


}
}