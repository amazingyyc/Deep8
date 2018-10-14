#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "Pow.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void PowForwardKernel(const real *X, const real scalar, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = cuPow(X[i], scalar);
    }
}

template <typename real>
__global__ void PowBackwardKernel(real *xGrad, const real *X, const real scalar, const real *yGrad, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    real realScalar = scalar - real(1);

    for (int i = start; i < N; i += stride) {
        xGrad[i] += yGrad[i] * cuPow(X[i], realScalar) * scalar;
    }
}

template <typename T>
void Pow<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, PowForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    PowForwardKernel<T> << <grideSize, blockSize >> > (x, scalar, y, N);
}

#ifdef HAVE_HALF
template <>
void Pow<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    PowForwardKernel<T> << <grideSize, blockSize >> > (x, scalar, y, N);
}
#endif

template <typename T>
void Pow<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Pow backwardCPU is error");

    auto x = inputs[0]->data();
    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto N  = (int)iGradient->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, PowBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    PowBackwardKernel<T> << <grideSize, blockSize >> > (xGrad, x, scalar, yGrad, N);
}

#ifdef HAVE_HALF
template <>
void Pow<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                         const Tensor<half> *output,
                         const Tensor<half> *outputGradient,
                         size_t index,
                         Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Pow backwardCPU is error");

    auto x = inputs[0]->data();
    auto dx = iGradient->data();
    auto dy = outputGradient->data();
    auto N  = (int)iGradient->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    PowBackwardKernel<half> << <grideSize, blockSize >> > (xGrad, x, scalar, yGrad, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(Pow);

#endif
}