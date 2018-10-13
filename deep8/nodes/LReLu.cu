#include "LReLu.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void LReLuForwardKernel(const real *X, const real a, real *Y, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = X[i] > real(0) ? X[i] : a * X[i];
    }
}

template <typename real>
__global__ void LReLuBackwardKernel(real *xGrad, const real *X, const real a, const real *yGrad, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += yGrad[i] * (X[i] > real(0) ? real(1) : a);
    }
}

template <typename T>
void LReLu<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int) output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LReLuForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    LReLuForwardKernel<T> << <grideSize, blockSize >> > (x, a, y, N);
}

#ifdef HAVE_HALF
template <>
void LReLu<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int) output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    LReLuForwardKernel<half> << <grideSize, blockSize >> > (x, a, y, N);
}
#endif

template <typename T>
void LReLu<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of LReLu backwardCPU is error");

    auto dx = iGradient->data();
    auto x  = inputs[0]->data();
    auto dy = outputGradient->data();
    auto N  = (int)iGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LReLuBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    LReLuBackwardKernel<T> << <grideSize, blockSize >> > (dx, x, a, dy, N);
}

#ifdef HAVE_HALF
template <>
void LReLu<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                         const Tensor<half> *output,
                         const Tensor<half> *outputGradient,
                         size_t index,
                         Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of LReLu backwardCPU is error");

    auto dx = iGradient->data();
    auto x  = inputs[0]->data();
    auto dy = outputGradient->data();
    auto N  = (int)iGradient->size();

    int minGrideSize;
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    LReLuBackwardKernel<half> << <grideSize, blockSize >> > (dx, x, a, dy, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(LReLu);

#endif
}