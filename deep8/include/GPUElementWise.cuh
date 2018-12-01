#ifndef DEEP8_GPUELEMENTWISE_H
#define DEEP8_GPUELEMENTWISE_H

#include "Shape.h"
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
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		dx[i] += op.backward(x[i], y[i], dy[i]);
	}
}

template <typename real, typename BinaryOp, int NumDims>
__global__ void BinaryElementWiseForward(const real *x, const NVShape<NumDims> xshape,
										 const real *y, const NVShape<NumDims> yshape,
											   real *z, const NVShape<NumDims> zshape, BinaryOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		int xI = 0;
		int yI = 0;

		for (int k = 0, index = i; k < NumDims; ++k) {
			auto zD = index / zshape.strides[k];

			if (xshape.dims[k] == zshape.dims[k]) {
				xI += zD * xshape.strides[k];
			}

			if (yshape.dims[k] == zshape.dims[k]) {
				yI += zD * yshape.strides[k];
			}

			index %= zshape.strides[k];
		}

		z[i] = op.forward(x[xI], y[yI]);
	}
}

template <typename real, typename BinaryOp, int NumDims>
__global__ void BinaryElementWiseBackwardX(const real *x,       real *dx, const NVShape<NumDims> xshape, 
	                                       const real *y,                 const NVShape<NumDims> yshape, 
										   const real *z, const real *dz, const NVShape<NumDims> zshape, 
										   BinaryOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int xindex[NumDims];
	int zindex[NumDims];

	for (int i = start; i < N; i += stride) {
		for (int k = 0, index = i; k < NumDims; ++k) {
			xindex[k] = index / xshape.strides[k];
			zindex[k] = xindex[k];

			index %= xshape.strides[k];
		}

		int j = NumDims - 1;

		while (j >= 0) {
			if (j == NumDims - 1) {
				int yi = 0;
				int zi = 0;

				for (int l = 0; l < NumDims; ++l) {
					zi += zindex[l] * zshape.strides[l];

					if (yshape.dims[l] == zshape.dims[l]) {
						yi += zindex[l] * yshape.strides[l];
					}
				}

				dx[i] += op.backwardX(x[i], y[yi], z[zi], dz[zi]);
			}

			if (xshape.dims[j] == zshape.dims[j]) {
				j--;
			} else {
				zindex[j]++;

				if (zindex[j] >= zshape.dims[j]) {
					j--;
				} else {
					for (int l = j + 1; l < NumDims; ++l) {
						zindex[l] = xindex[l];
					}

					j = NumDims - 1;
				}
			}
		}
	}
}

template <typename real, typename BinaryOp, int NumDims>
__global__ void BinaryElementWiseBackwardY(const real *x,                 const NVShape<NumDims> xshape,
										   const real *y,       real *dy, const NVShape<NumDims> yshape,
										   const real *z, const real *dz, const NVShape<NumDims> zshape,
										   BinaryOp op, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int yindex[NumDims];
	int zindex[NumDims];

	for (int i = start; i < N; i += stride) {
		for (int k = 0, index = i; k < NumDims; ++k) {
			yindex[k] = index / yshape.strides[k];
			zindex[k] = yindex[k];

			index %= yshape.strides[k];
		}

		int j = NumDims - 1;

		while (j >= 0) {
			if (j == NumDims - 1) {
				int xi = 0;
				int zi = 0;

				for (int l = 0; l < NumDims; ++l) {
					zi += zindex[l] * zshape.strides[l];

					if (xshape.dims[l] == zshape.dims[l]) {
						xi += zindex[l] * xshape.strides[l];
					}
				}

				dy[i] += op.backwardY(x[xi], y[i], z[zi], dz[zi]);
			}

			if (yshape.dims[j] == zshape.dims[j]) {
				j--;
			} else {
				zindex[j]++;

				if (zindex[j] >= zshape.dims[j]) {
					j--;
				} else {
					for (int l = j + 1; l < NumDims; ++l) {
						zindex[l] = yindex[l];
					}

					j = NumDims - 1;
				}
			}
		}
	}
}

template <typename real, typename BinaryOp>
void callBinaryElementWiseForward(const real *x, const Shape &xshape, const real *y, const Shape &yshape, real *z, const Shape &zshape, BinaryOp op) {
	int N = (int) zshape.size();
	int numDims = (int)zshape.nDims();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	switch (numDims) {
	case 1:
		BinaryElementWiseForward<real, BinaryOp, 1> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > 
			(
				x, xshape.convertToNVShape<1>(),
				y, yshape.convertToNVShape<1>(),
				z, zshape.convertToNVShape<1>(),
				op, N
			);
		break;
	case 2:
		BinaryElementWiseForward<real, BinaryOp, 2> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, xshape.convertToNVShape<2>(),
				y, yshape.convertToNVShape<2>(),
				z, zshape.convertToNVShape<2>(),
				op, N
				);
		break;
	case 3:
		BinaryElementWiseForward<real, BinaryOp, 3> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, xshape.convertToNVShape<3>(),
				y, yshape.convertToNVShape<3>(),
				z, zshape.convertToNVShape<3>(),
				op, N
				);
		break;
	case 4:
		BinaryElementWiseForward<real, BinaryOp, 4> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, xshape.convertToNVShape<4>(),
				y, yshape.convertToNVShape<4>(),
				z, zshape.convertToNVShape<4>(),
				op, N
				);
		break;
	case 5:
		BinaryElementWiseForward<real, BinaryOp, 5> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, xshape.convertToNVShape<5>(),
				y, yshape.convertToNVShape<5>(),
				z, zshape.convertToNVShape<5>(),
				op, N
				);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the shape is error");
		break;
	}
}

template <typename real, typename BinaryOp>
void callBinaryElementWiseBackwardX(const real *x, real *dx, const Shape &xshape, const real *y, const Shape &yshape, const real *z, const real *dz, const Shape &zshape, BinaryOp op) {
	int N       = (int)xshape.size();
	int numDims = (int)zshape.nDims();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	switch (numDims) {
	case 1:
		BinaryElementWiseBackwardX<real, BinaryOp, 1> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, dx, xshape.convertToNVShape<1>(),
				y,     yshape.convertToNVShape<1>(),
				z, dz, zshape.convertToNVShape<1>(),
				op, N
				);
		break;
	case 2:
		BinaryElementWiseBackwardX<real, BinaryOp, 2> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, dx, xshape.convertToNVShape<2>(),
				y,     yshape.convertToNVShape<2>(),
				z, dz, zshape.convertToNVShape<2>(),
				op, N
				);
		break;
	case 3:
		BinaryElementWiseBackwardX<real, BinaryOp, 3> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, dx, xshape.convertToNVShape<3>(),
				y,     yshape.convertToNVShape<3>(),
				z, dz, zshape.convertToNVShape<3>(),
				op, N
				);
		break;
	case 4:
		BinaryElementWiseBackwardX<real, BinaryOp, 4> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, dx, xshape.convertToNVShape<4>(),
				y,     yshape.convertToNVShape<4>(),
				z, dz, zshape.convertToNVShape<4>(),
				op, N
				);
		break;
	case 5:
		BinaryElementWiseBackwardX<real, BinaryOp, 5> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x, dx, xshape.convertToNVShape<5>(),
				y,     yshape.convertToNVShape<5>(),
				z, dz, zshape.convertToNVShape<5>(),
				op, N
				);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the shape is error");
		break;
	}
}

template <typename real, typename BinaryOp>
void callBinaryElementWiseBackwardY(const real *x, const Shape &xshape, const real *y, real *dy, const Shape &yshape, const real *z, const real *dz, const Shape &zshape, BinaryOp op) {
	int N = (int)yshape.size();
	int numDims = (int)zshape.nDims();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	switch (numDims) {
	case 1:
		BinaryElementWiseBackwardY<real, BinaryOp, 1> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x,     xshape.convertToNVShape<1>(),
				y, dy, yshape.convertToNVShape<1>(),
				z, dz, zshape.convertToNVShape<1>(),
				op, N);
		break;
	case 2:
		BinaryElementWiseBackwardY<real, BinaryOp, 2> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x,     xshape.convertToNVShape<2>(),
				y, dy, yshape.convertToNVShape<2>(),
				z, dz, zshape.convertToNVShape<2>(),
				op, N);
		break;
	case 3:
		BinaryElementWiseBackwardY<real, BinaryOp, 3> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x,	   xshape.convertToNVShape<3>(),
				y, dy, yshape.convertToNVShape<3>(),
				z, dz, zshape.convertToNVShape<3>(),
				op, N);
		break;
	case 4:
		BinaryElementWiseBackwardY<real, BinaryOp, 4> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x,     xshape.convertToNVShape<4>(),
				y, dy, yshape.convertToNVShape<4>(),
				z, dz, zshape.convertToNVShape<4>(),
				op, N);
		break;
	case 5:
		BinaryElementWiseBackwardY<real, BinaryOp, 5> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> >
			(
				x,     xshape.convertToNVShape<5>(),
				y, dy, yshape.convertToNVShape<5>(),
				z, dz, zshape.convertToNVShape<5>(),
				op, N);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the shape is error");
		break;
	}
}

}

#endif