#include "Square.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void SquareForwardKernel(const real *X, real *Y, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = X[i] * X[i];
    }
}

template <typename real>
__global__ void SquareBackwardKernel(real *xGrad, const real *X, const real *yGrad, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += real(2.0) * yGrad[i] * X[i];
    }
}

template <typename T>
void Square<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SquareForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    SquareForwardKernel<T> << <grideSize, blockSize >> > (x, y, N);
}

#ifdef HAVE_HALF
template <>
void Square<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    SquareForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
}
#endif

template <typename T>
void Square<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto dx = iGradient->data();
    auto x  = inputs[0]->data();
    auto dy = outputGradient->data();
    auto N  = (int) iGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SquareBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    SquareBackwardKernel<T> << <grideSize, blockSize >> > (dx, x, dy, N);
}

#ifdef HAVE_HALF
template <>
void Square<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                             const Tensor<half> *output,
                             const Tensor<half> *outputGradient,
                             size_t index,
                             Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto dx = iGradient->data();
    auto x  = inputs[0]->data();
    auto dy = outputGradient->data();
    auto N  = (int) iGradient->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    SquareBackwardKernel<half> << <grideSize, blockSize >> > (dx, x, dy, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(Square);

#endif

}