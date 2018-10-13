#include "ScalarMinus.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void ScalarMinusForwardKernel(const real scalar, const real *X, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = scalar - X[i];
    }
}

template <typename real>
__global__ void ScalarMinusBackwardKernel(real *dx, const real *dy, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        dx[i] -= dy[i];
    }
}

template <typename T>
void ScalarMinus<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto X = inputs[0]->data();
    auto Y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarMinusForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    ScalarMinusForwardKernel<T> << <grideSize, blockSize >> > (scalar, X, Y, N);
}

#ifdef HAVE_HALF
template <>
void ScalarMinus<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto X = inputs[0]->data();
    auto Y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    ScalarMinusForwardKernel<half> << <grideSize, blockSize >> > (scalar, X, Y, N);
}
#endif

template <>
void ScalarMinus<float>::backwardGPU(const std::vector<const Tensor<float>*> &inputs,
                                     const Tensor<float> *output,
                                     const Tensor<float> *outputGradient,
                                     size_t index,
                                     Tensor<float> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device);

    float alpha = -1;

    CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)iGradient->size(), &alpha, outputGradient->data(), 1, iGradient->data(), 1));
}

template <>
void ScalarMinus<double>::backwardGPU(const std::vector<const Tensor<double>*> &inputs,
                                     const Tensor<double> *output,
                                     const Tensor<double> *outputGradient,
                                     size_t index,
                                     Tensor<double> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device);

    double alpha = -1;

    CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)iGradient->size(), &alpha, outputGradient->data(), 1, iGradient->data(), 1));
}

#ifdef HAVE_HALF
template <>
void ScalarMinus<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                                      const Tensor<half> *output,
                                      const Tensor<half> *outputGradient,
                                      size_t index,
                                      Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    int N = (int)iGradient->size();
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    ScalarMinusBackwardKernel<half> << <grideSize, blockSize >> > (iGradient->data(), outputGradient->data(), N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(ScalarMinus);

#endif

}