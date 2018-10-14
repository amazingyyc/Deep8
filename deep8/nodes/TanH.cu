#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "TanH.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void TanHForwardKernel(const real *X, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = cuTanh(X[i]);
    }
}

template <typename real>
__global__ void TanHBackwardKernel(real *xGrad, const real *yGrad, const real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += yGrad[i] * (real(1.0) - Y[i] * Y[i]);
    }
}

template <typename T>
void TanH<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanHForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    TanHForwardKernel<T> << <grideSize, blockSize >> > (x, y, N);
}

#ifdef HAVE_HALF
template <>
void TanH<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    TanHForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
}
#endif

template <typename T>
void TanH<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of TanH backwardCPU is error");

    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto y  = output->data();
    auto N  = (int)iGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanHBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    TanHBackwardKernel<T> << <grideSize, blockSize >> > (dx, dy, y, N);
}

#ifdef HAVE_HALF
template <>
void TanH<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                         const Tensor<half> *output,
                         const Tensor<half> *outputGradient,
                         size_t index,
                         Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of TanH backwardCPU is error");

    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto y  = output->data();
    auto N  = (int)iGradient->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    TanHBackwardKernel<half> << <grideSize, blockSize >> > (dx, dy, y, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(TanH);

#endif

}