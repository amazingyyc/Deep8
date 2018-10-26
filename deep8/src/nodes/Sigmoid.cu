#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "Sigmoid.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void SigmoidForwardKernel(const real *X, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = real(0.5) + real(0.5) * cuTanh(real(0.5) * X[i]);
    }
}

template <typename real>
__global__ void SigmoidBackwardKernel(real *xGrad, const real *yGrad, const real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += yGrad[i] * Y[i] * (real(1) - Y[i]);
    }
}

template <typename T>
void Sigmoid<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SigmoidForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    SigmoidForwardKernel<T> << <grideSize, blockSize >> > (x, y, N);
}

#ifdef HAVE_HALF
template <>
void Sigmoid<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    SigmoidForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
}
#endif

template <typename T>
void Sigmoid<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto y  = output->data();
    auto N  = (int)iGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SigmoidBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    SigmoidBackwardKernel<T> << <grideSize, blockSize >> > (dx, dy, y, N);
}

#ifdef HAVE_HALF
template <>
void Sigmoid<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                             const Tensor<half> *output,
                             const Tensor<half> *outputGradient,
                             size_t index,
                             Tensor<half> *iGradient) {
    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto y  = output->data();
    auto N  = (int)iGradient->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    SigmoidBackwardKernel<half> << <grideSize, blockSize >> > (dx, dy, y, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(Sigmoid);

#endif

}