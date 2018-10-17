#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "MinusScalar.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void MinusScalarForwardKernel(const real *X, const real scalar, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = X[i] - scalar;
    }
}

template <typename real>
__global__ void MinusScalarBackwardKernel(real *xGrad, const real *yGrad, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += yGrad[i];
    }
}

template <typename T>
void MinusScalar<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto X = inputs[0]->data();
    auto Y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MinusScalarForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    MinusScalarForwardKernel<T> << <grideSize, blockSize >> > (X, scalar, Y, N);
}

#ifdef HAVE_HALF
template <>
void MinusScalar<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto X = inputs[0]->data();
    auto Y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    MinusScalarForwardKernel<half> << <grideSize, blockSize >> > (X, scalar, Y, N);
}
#endif

template <>
void MinusScalar<float>::backwardGPU(const std::vector<const Tensor<float>*> &inputs,
                                     const Tensor<float> *output,
                                     const Tensor<float> *outputGradient,
                                     size_t index,
                                     Tensor<float> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    float alpha = 1;
    CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)iGradient->size(), &alpha, outputGradient->data(), 1, iGradient->data(), 1));
}

template <>
void MinusScalar<double>::backwardGPU(const std::vector<const Tensor<double>*> &inputs,
                                      const Tensor<double> *output,
                                      const Tensor<double> *outputGradient,
                                      size_t index,
                                      Tensor<double> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    double alpha = 1;

    CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)iGradient->size(), &alpha, outputGradient->data(), 1, iGradient->data(), 1));
}

#ifdef HAVE_HALF
template <>
void MinusScalar<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                                      const Tensor<half> *output,
                                      const Tensor<half> *outputGradient,
                                      size_t index,
                                      Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    int N = (int)iGradient->size();
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    MinusScalarBackwardKernel<half> << <grideSize, blockSize >> > (iGradient->data(), outputGradient->data(), N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(MinusScalar);

#endif

}