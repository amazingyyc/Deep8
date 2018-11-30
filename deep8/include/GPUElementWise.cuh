#ifndef DEEP8_GPUELEMENTWISE_H
#define DEEP8_GPUELEMENTWISE_H

#include "GPUBasic.h"

namespace Deep8 {

template <typename real, typename UnaryOp>
__global__ void UnaryElementWiseForward(const real *x, real *y, UnaryOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		y[i] = op.forward(x[i]);
	}
}

template <typename real, typename UnaryOp>
__global__ void UnaryElementWiseBackward(const real *x, real *dx, const real *y, const real *dy, UnaryOp op, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		dx[i] += op.backward(x[i], y[i], dy[i]);
	}
}

}

#endif