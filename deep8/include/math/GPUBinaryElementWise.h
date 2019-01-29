#ifndef DEEP8_MATH_GPUBINARYELEMENTWISE_H
#define DEEP8_MATH_GPUBINARYELEMENTWISE_H

#include "GPUBasic.h"

namespace Deep8 {
namespace Math {

template <typename T, typename BinaryOp, int NumDims>
__global__ void BinaryElementWiseKernel(const T *x, const NVShape<NumDims> xshape,
                                        const T *y, const NVShape<NumDims> yshape,
                                              T *z, const NVShape<NumDims> zshape,
                                              BinaryOp op, const int N) {
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

		z[i] = op(x[xI], y[yI]);
	}
}

template <typename T, typename BinaryOp, int NumDims>
__global__ void BinaryElementWiseGradXKernel(const T *x,       T *dx, const NVShape<NumDims> xshape, 
	                                   		 const T *y,              const NVShape<NumDims> yshape, 
									         const T *z, const T *dz, const NVShape<NumDims> zshape, 
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

				dx[i] += op(x[i], y[yi], z[zi], dz[zi]);
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

template <typename T, typename BinaryOp, int NumDims>
__global__ void BinaryElementWiseGradYKernel(const T *x,               const NVShape<NumDims> xshape,
										     const T *y,        T *dy, const NVShape<NumDims> yshape,
										     const T *z,  const T *dz, const NVShape<NumDims> zshape,
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

				dy[i] += op(x[xi], y[i], z[zi], dz[zi]);
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

template <typename T, typename BinaryOp>
void CallBinaryElementWiseKernel(const T *x, const Shape &xshape, const T *y, const Shape &yshape, T *z, const Shape &zshape, BinaryOp op) {
	int N       = (int) zshape.size();
	int numDims = (int) zshape.nDims + 1;

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	switch (numDims) {
	case 1:
		BinaryElementWiseKernel<T, BinaryOp, 1> << <grideSize, blockSize >> > 
			(
				x, convertToNVShape<1>(xshape),
				y, convertToNVShape<1>(yshape),
				z, convertToNVShape<1>(zshape),
				op, N
			);
		break;
	case 2:
		BinaryElementWiseKernel<T, BinaryOp, 2> << <grideSize, blockSize >> >
			(
				x, convertToNVShape<2>(xshape),
				y, convertToNVShape<2>(yshape),
				z, convertToNVShape<2>(zshape),
				op, N
				);
		break;
	case 3:
		BinaryElementWiseKernel<T, BinaryOp, 3> << <grideSize, blockSize >> >
			(
				x, convertToNVShape<3>(xshape),
				y, convertToNVShape<3>(yshape),
				z, convertToNVShape<3>(zshape),
				op, N
				);
		break;
	case 4:
		BinaryElementWiseKernel<T, BinaryOp, 4> << <grideSize, blockSize >> >
			(
				x, convertToNVShape<4>(xshape),
				y, convertToNVShape<4>(yshape),
				z, convertToNVShape<4>(zshape),
				op, N
				);
		break;
	case 5:
		BinaryElementWiseKernel<real, BinaryOp, 5> << <grideSize, blockSize >> >
			(
				x, convertToNVShape<5>(xshape),
				y, convertToNVShape<5>(yshape),
				z, convertToNVShape<5>(zshape),
				op, N
				);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the shape is error");
		break;
	}
}

template <typename T, typename BinaryOp>
void CallBinaryElementWiseGradXKernel(const T *x, T *dx, const Shape &xshape, const T *y, const Shape &yshape, const T *z, const T *dz, const Shape &zshape, BinaryOp op) {
	int N       = (int)xshape.size();
	int numDims = (int)zshape.nDims + 1;

	int blockSize = DEEP8_GPU_BLOCK_SIZE;
	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	switch (numDims) {
	case 1:
		BinaryElementWiseGradXKernel<T, BinaryOp, 1> << <grideSize, blockSize >> >
			(
				x, dx, convertToNVShape<1>(xshape),
				y,     convertToNVShape<1>(yshape),
				z, dz, convertToNVShape<1>(zshape),
				op, N
				);
		break;
	case 2:
		BinaryElementWiseGradXKernel<T, BinaryOp, 2> << <grideSize, blockSize >> >
			(
				x, dx, convertToNVShape<2>(xshape),
				y,     convertToNVShape<2>(yshape),
				z, dz, convertToNVShape<2>(zshape),
				op, N
				);
		break;
	case 3:
		BinaryElementWiseGradXKernel<T, BinaryOp, 3> << <grideSize, blockSize >> >
			(
				x, dx, convertToNVShape<3>(xshape),
				y,     convertToNVShape<3>(yshape),
				z, dz, convertToNVShape<3>(zshape),
				op, N
				);
		break;
	case 4:
		BinaryElementWiseGradXKernel<T, BinaryOp, 4> << <grideSize, blockSize >> >
			(
				x, dx, convertToNVShape<4>(xshape),
				y,     convertToNVShape<4>(yshape),
				z, dz, convertToNVShape<4>(zshape),
				op, N
				);
		break;
	case 5:
		BinaryElementWiseGradXKernel<T, BinaryOp, 5> << <grideSize, blockSize >> >
			(
				x, dx, convertToNVShape<5>(xshape),
				y,     convertToNVShape<5>(yshape),
				z, dz, convertToNVShape<5>(zshape),
				op, N
				);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the shape is error");
		break;
	}
}


template <typename T, typename BinaryOp>
void CallBinaryElementWiseGradYKernel(const T *x, const Shape &xshape, const T *y, T *dy, const Shape &yshape, const T *z, const T *dz, const Shape &zshape, BinaryOp op) {
	int N       = (int)yshape.size();
	int numDims = (int)zshape.nDims + 1;

	int blockSize = DEEP8_GPU_BLOCK_SIZE;
	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	switch (numDims) {
	case 1:
		BinaryElementWiseGradYKernel<T, BinaryOp, 1> << <grideSize, blockSize >> >
			(
				x,     convertToNVShape<1>(xshape),
				y, dy, convertToNVShape<1>(yshape),
				z, dz, convertToNVShape<1>(zshape),
				op, N);
		break;
	case 2:
		BinaryElementWiseGradYKernel<T, BinaryOp, 2> << <grideSize, blockSize >> >
			(
				x,     convertToNVShape<2>(xshape),
				y, dy, convertToNVShape<2>(yshape),
				z, dz, convertToNVShape<2>(zshape),
				op, N);
		break;
	case 3:
		BinaryElementWiseGradYKernel<T, BinaryOp, 3> << <grideSize, blockSize >> >
			(
				x,	   convertToNVShape<3>(xshape),
				y, dy, convertToNVShape<3>(yshape),
				z, dz, convertToNVShape<3>(zshape),
				op, N);
		break;
	case 4:
		BinaryElementWiseGradYKernel<T, BinaryOp, 4> << <grideSize, blockSize >> >
			(
				x,     convertToNVShape<4>(xshape),
				y, dy, convertToNVShape<4>(yshape),
				z, dz, convertToNVShape<4>(zshape),
				op, N);
		break;
	case 5:
		BinaryElementWiseGradYKernel<T, BinaryOp, 5> << <grideSize, blockSize >> >
			(
				x,     convertToNVShape<5>(xshape),
				y, dy, convertToNVShape<5>(yshape),
				z, dz, convertToNVShape<5>(zshape),
				op, N);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the shape is error");
		break;
	}
}

}
}

#endif
