#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "ReLu.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void ReLuForwardKernel(const real *X, real *Y, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = X[i] > real(0.0) ? X[i] : real(0.0);
    }
}

template <typename real>
__global__ void ReLuBackwardKernel(real *xGrad, const real *X, const real *yGrad, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += (X[i] > real(0.0) ? yGrad[i] : real(0.0));
    }
}

template <typename T>
void ReLu<T>::forwardGPUImpl(const T *x, T *y, const int N) {
    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ReLuForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    ReLuForwardKernel<T> << <grideSize, blockSize >> > (x, y, N);
}

#ifdef HAVE_HALF
template <>
void ReLu<half>::forwardGPUImpl(const half *x, half *y, const int N) {
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    ReLuForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
}
#endif

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
void ReLu<T>::backwardGPUImpl(T *dx, const T *x, const T *dy, const int N) {
    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ReLuBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    ReLuBackwardKernel<T> << <grideSize, blockSize >> > (dx, x, dy, N);
}

#ifdef HAVE_HALF
template <>
void ReLu<half>::backwardGPUImpl(half *dx, const half *x, const half *dy, const int N) {
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    ReLuBackwardKernel<half> << <grideSize, blockSize >> > (dx, x, dy, N);
}
#endif

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
		backwardGPUImpl(iGradient->data(), inputs[0]->data(), outputGradient->data(), (int)(iGradient->size()));
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

template void ReLu<float>::backwardGPUImpl(float *dx, const float *x, const float *dy, const int N);
template void ReLu<double>::backwardGPUImpl(double *dx, const double *x, const double *dy, const int N);
#ifdef HAVE_HALF
template void ReLu<half>::backwardGPUImpl(half *dx, const half *x, const half *dy, const int N);
#endif

#ifdef HAVE_CUDNN
template void ReLu<float>::backwardGPUCUDNNImpl(Device *device, const float *x, float *dx, const float *y, const float *dy, Shape &shape);
template void ReLu<double>::backwardGPUCUDNNImpl(Device *device, const double *x, double *dx, const double *y, const double *dy, Shape &shape);
#ifdef HAVE_HALF
template void ReLu<half>::backwardGPUCUDNNImpl(Device *device, const half *x, half *dx, const half *y, const half *dy, Shape &shape);
#endif
#endif

#endif

}