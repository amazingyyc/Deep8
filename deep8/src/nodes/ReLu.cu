#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.h"
#include "ReLu.h"

namespace Deep8 {

template <typename T>
struct ReLuOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T forward(const T &x) {
		return x >= T(0) ? x : T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return x >= T(0) ? dy : T(0);
	}
};

template <typename T>
void ReLu<T>::forwardGPUImpl(const T *x, T *y, const int N) {
	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, ReLuOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, y, ReLuOP<T>(), N);
}


#ifdef HAVE_CUDNN
template <>
void ReLu<float>::forwardGPUCUDNNImpl(Device *d, const float *X, float *Y, Shape &shape) {
    auto device = (GPUDevice*)d;

    float alpha = 1;
    float beta  = 0;

    int size = static_cast<int>(shape.size());

    cudnnActivationDescriptor_t activationDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

    CUDNN_CHECK(cudnnActivationForward(device->cudnnHandle, activationDesc, &alpha, xDesc, X, &beta, yDesc, Y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
}

template <>
void ReLu<double>::forwardGPUCUDNNImpl(Device *d, const double *X, double *Y, Shape &shape) {
    auto device = (GPUDevice*)d;

    double alpha = 1;
    double beta = 0;

    int size = static_cast<int>(shape.size());

    cudnnActivationDescriptor_t activationDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

    CUDNN_CHECK(cudnnActivationForward(device->cudnnHandle, activationDesc, &alpha, xDesc, X, &beta, yDesc, Y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
}

#ifdef HAVE_HALF
template <>
void ReLu<half>::forwardGPUCUDNNImpl(Device *d, const half *X, half *Y, Shape &shape) {
    auto device = (GPUDevice*)d;

    half alpha(1.0);
    half beta(0.0);

    int size = static_cast<int>(shape.size());

    cudnnActivationDescriptor_t activationDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

    CUDNN_CHECK(cudnnActivationForward(device->cudnnHandle, activationDesc, &alpha, xDesc, X, &beta, yDesc, Y));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
}
#endif
#endif

template <typename T>
void ReLu<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
#ifdef HAVE_CUDNN
		forwardGPUCUDNNImpl(output->device(), inputs[0]->data(), output->data(), output->shape);
#else
		forwardGPUImpl(inputs[0]->data(), output->data(), (int)(output->size()));
#endif
}

template <typename T>
void ReLu<T>::backwardGPUImpl(T *dx, const T *x, const T *y, const T *dy, const int N) {
	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, ReLuOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, dx, y, dy, ReLuOP<T>(), N);
}

#ifdef HAVE_CUDNN
template <>
void ReLu<float>::backwardGPUCUDNNImpl(Device *d, const float *x, float *dx, const float *y, const float *dy, Shape &shape) {
    auto device = (GPUDevice*)d;

    float alpha = 1;
    float beta  = 1;

    int size = static_cast<int>(shape.size());

    cudnnActivationDescriptor_t activationDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

    CUDNN_CHECK(cudnnActivationBackward(device->cudnnHandle, activationDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
}

template <>
void ReLu<double>::backwardGPUCUDNNImpl(Device *d, const double *x, double *dx, const double *y, const double *dy, Shape &shape) {
    auto device = (GPUDevice*)d;

    double alpha = 1;
    double beta = 1;

    int size = static_cast<int>(shape.size());

    cudnnActivationDescriptor_t activationDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

    CUDNN_CHECK(cudnnActivationBackward(device->cudnnHandle, activationDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
}

#ifdef HAVE_HALF
template <>
void ReLu<half>::backwardGPUCUDNNImpl(Device *d, const half *x, half *dx, const half *y, const half *dy, Shape &shape) {
    auto device = (GPUDevice*)d;

    half alpha(1.0);
    half beta(1.0);

    int size = static_cast<int>(shape.size());

    cudnnActivationDescriptor_t activationDesc;
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
    CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

    cudnnTensorDescriptor_t xDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

    cudnnTensorDescriptor_t dxDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

    cudnnTensorDescriptor_t yDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

    cudnnTensorDescriptor_t dyDesc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

    CUDNN_CHECK(cudnnActivationBackward(device->cudnnHandle, activationDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
    CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
}
#endif
#endif

template <typename T>
void ReLu<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

#ifdef HAVE_CUDNN
    backwardGPUCUDNNImpl(iGradient->device(),
                        inputs[0]->data(),
                        iGradient->data(),
                        output->data(),
                        outputGradient->data(),
                        iGradient->shape);
#else
		backwardGPUImpl(iGradient->data(), inputs[0]->data(), output->data(), outputGradient->data(), (int)(iGradient->size()));
#endif
}

DEEP8_DECLARATION_GPU_FUNC(ReLu);

template void ReLu<float>::forwardGPUImpl(const float *x, float *y, const int N);
template void ReLu<double>::forwardGPUImpl(const double *x, double *y, const int N);
#ifdef HAVE_HALF
template void ReLu<half>::forwardGPUImpl(const half *x, half *y, const int N);
#endif

#ifdef HAVE_CUDNN
template void ReLu<float>::forwardGPUCUDNNImpl(Device *device, const float *X, float *Y, Shape &shape);
template void ReLu<double>::forwardGPUCUDNNImpl(Device *device, const double *X, double *Y, Shape &shape);
#ifdef HAVE_HALF
template void ReLu<half>::forwardGPUCUDNNImpl(Device *device, const half *X, half *Y, Shape &shape);
#endif
#endif

template void ReLu<float>::backwardGPUImpl(float *dx, const float *x, const float *y, const float *dy, const int N);
template void ReLu<double>::backwardGPUImpl(double *dx, const double *x, const double *y, const double *dy, const int N);
#ifdef HAVE_HALF
template void ReLu<half>::backwardGPUImpl(half *dx, const half *x, const half *y, const half *dy, const int N);
#endif

#ifdef HAVE_CUDNN
template void ReLu<float>::backwardGPUCUDNNImpl(Device *device, const float *x, float *dx, const float *y, const float *dy, Shape &shape);
template void ReLu<double>::backwardGPUCUDNNImpl(Device *device, const double *x, double *dx, const double *y, const double *dy, Shape &shape);
#ifdef HAVE_HALF
template void ReLu<half>::backwardGPUCUDNNImpl(Device *device, const half *x, half *dx, const half *y, const half *dy, Shape &shape);
#endif
#endif

}