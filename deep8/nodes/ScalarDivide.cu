#include "ScalarDivide.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void ScalarDivideForwardKernel(const real scalar, const real *X, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = scalar / X[i];
    }
}

template <typename real>
__global__ void ScalarDivideBackwardKernel(const real scalar, real *xGrad, const real *X, const real *yGrad, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] = -scalar * yGrad[i] / (X[i] * X[i]);
    }
}

template <typename T>
void ScalarDivide<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarDivideForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    ScalarDivideForwardKernel<T> << <grideSize, blockSize >> > (scalar, x, y, N);
}

#ifdef HAVE_HALF
template <>
void ScalarDivide<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    ScalarDivideForwardKernel<half> << <grideSize, blockSize >> > (scalar, x, y, N);
}
#endif

template <typename T>
void ScalarDivide<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                                 const Tensor<T> *output,
                                 const Tensor<T> *outputGradient,
                                 size_t index,
                                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto dx = iGradient->data();
    auto x  = inputs[0]->data();
    auto dy = outputGradient->data();
    auto N  = (int)iGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarDivideBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    ScalarDivideBackwardKernel<T> << <grideSize, blockSize >> > (scalar, dx, x, dy, N);
}

#ifdef HAVE_HALF
template <>
void ScalarDivide<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                                  const Tensor<half> *output,
                                  const Tensor<half> *outputGradient,
                                  size_t index,
                                  Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto dx = iGradient->data();
    auto x  = inputs[0]->data();
    auto dy = outputGradient->data();
    auto N  = (int)iGradient->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    ScalarDivideBackwardKernel<half> << <grideSize, blockSize >> > (scalar, dx, x, dy, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(ScalarDivide);

#endif

}