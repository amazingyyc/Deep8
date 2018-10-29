#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "Tanh.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void TanhForwardKernel(const real *X, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = cuTanh(X[i]);
    }
}

template <typename real>
__global__ void TanhBackwardKernel(real *xGrad, const real *yGrad, const real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += yGrad[i] * (real(1.0) - Y[i] * Y[i]);
    }
}

template <typename T>
void Tanh<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanhForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    TanhForwardKernel<T> << <grideSize, blockSize >> > (x, y, N);
}

#ifdef HAVE_HALF
template <>
void Tanh<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    TanhForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
}
#endif

template <typename T>
void Tanh<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Tanh backwardCPU is error");

    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto y  = output->data();
    auto N  = (int)iGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanhBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    TanhBackwardKernel<T> << <grideSize, blockSize >> > (dx, dy, y, N);
}

#ifdef HAVE_HALF
template <>
void Tanh<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                         const Tensor<half> *output,
                         const Tensor<half> *outputGradient,
                         size_t index,
                         Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Tanh backwardCPU is error");

    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto y  = output->data();
    auto N  = (int)iGradient->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    TanhBackwardKernel<half> << <grideSize, blockSize >> > (dx, dy, y, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(Tanh);

#endif

}