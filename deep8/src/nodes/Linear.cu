#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "Linear.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void LinearForwardKernel(const real *X, const real a, const real b, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = a * X[i] + b;
    }
}

template <typename real>
__global__ void LinearBackwardKernel(real *xGrad, const real a, const real *yGrad, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += a * yGrad[i];
    }
}

template <typename T>
void Linear<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LinearForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    LinearForwardKernel<T> << <grideSize, blockSize >> > (x, a, b, y, N);
}

#ifdef HAVE_HALF
template <>
void Linear<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    LinearForwardKernel<half> << <grideSize, blockSize >> > (x, a, b, y, N);
}
#endif

template <typename T>
void Linear<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                        const Tensor<T> *output,
                        const Tensor<T> *outputGradient,
                        size_t index,
                        Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto N = (int)outputGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LinearBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    LinearBackwardKernel<T> << <grideSize, blockSize >> > (dx, a, dy, N);
}

#ifdef HAVE_HALF
template <>
void Linear<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                        const Tensor<half> *output,
                        const Tensor<half> *outputGradient,
                        size_t index,
                        Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto N = (int)outputGradient->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    LinearBackwardKernel<half> << <grideSize, blockSize >> > (dx, a, dy, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(Linear);

#endif

}