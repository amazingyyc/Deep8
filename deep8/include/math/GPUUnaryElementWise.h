#ifndef DEEP8_MATH_GPUUNARYELEMENTWISE_H
#define DEEP8_MATH_GPUUNARYELEMENTWISE_H

#include "GPUBasic.h"

namespace Deep8 {
namespace Math {

/**
 * Unary Element Wise Kernel
 */
template <typename T, typename UnaryOp>
__global__ void UnaryElementWiseKernel(const T *x, T *y, UnaryOp op, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        y[i] = op(x[i]);
    }
}

/**
 * Unary Element Wise Grad Kernel
 */
template <typename T, typename UnaryGradOp>
__global__ void UnaryElementWiseGradKernel(const T *x, T *dx, const T *y, const T *dy, UnaryGradOp op, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        dx[i] += op(x[i], y[i], dy[i]);
    }
}


}
}

#endif