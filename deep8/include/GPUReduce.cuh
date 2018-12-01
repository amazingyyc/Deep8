#ifndef DEEP8_GPUREDUCE_H
#define DEEP8_GPUREDUCE_H

#include "Shape.h"
#include "GPUBasic.h"

namespace Deep8 {

/**
 * the common reduce kernel
 */
template <typename T, typename ReduceOp, int NumDims>
__global__ void CommonReduceForward(const T *x, const NVShape<NumDims> xshape, T *y, const NVShape<NumDims> yshape, ReduceOp op, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    int xindex[NumDims];
    int yindex[NumDims];

	for (int i = start; i < N; i += stride) {
		T ret = op.init();

        for (int k = 0, index = i; k < NumDims; ++k) {
            yindex[k] = index / yshape.strides[k];
            xindex[k] = yindex[k];

            index %= yshape.strides[k];
        }

        int j = NumDims - 1;

        while (j >= 0) {
            if (j == NumDims - 1) {
                int xi = 0;

                for (int l = 0; l < NumDims; ++l) {
                    xi += xindex[l] * xshape.strides[l];
                }

                ret = op.step(ret, x[xi]);
            }

            if (xshape.dims[j] == yshape.dims[j]) {
                j--;
            } else {
                xindex[j]++;

                if (xindex[j] >= x.shape.dims[j]) {
                    j--;
                } else {
                    for (int l = j + 1; l < NumDims; ++l) {
                        xindex[l] = yindex[l];
                    }

                    j = NumDims - 1;
                }
            }
        }

        y[i] = op.complete(ret);
	}
}

template <typename T, typename ReduceOp, int NumDims>
__global__ void CommonReduceBackward(
	const T *x,       T *dx, const NVShape<NumDims> xshape, 
	const T *y, const T *dy, const NVShape<NumDims> yshape, 
	ReduceOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int xindex[NumDims];

	for (int xi = start; xi < N; xi += stride) {
		int yi = 0;

		for (int k = 0, index = xi; k < NumDims; ++k) {
			int xd = index / xshape.stride[k];

			if (xshape.dims[k] == yshape.dims[k]) {
				yi += xd * yshape.strides[k];
			}

			index %= xshape.strides[k];
		}

		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
	}
}

/**
 * special for middle reduce
 */
template <typename T, typename ReduceOp>
__global__ void MiddleReduceForward(const T *x, T *y, const int preDim, const int middleDim, const int nextDim, ReduceOp op, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    for (int yi = start; yi < N; yi += stride) {
        T ret = op.init();

        int row = yi / nextDim;
        int col = yi % nextDim;

        int xi = row * middleDim * nextDim + col;

        for (int k = 0; k < middleDim; ++k) {
            ret = op.step(ret, x[xi]);

            xi += nextDim;
        }

        y[yi] = op.complete(ret);
    }
}

template <typename T, typename ReduceOp> 
__global__ void MiddleReduceBackward(
	const T *x, T *dx, const T *y, const T *dy, 
	const int preDim, const int middleDim, const int nextDim, ReduceOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int d1 = i / (middleDim * nextDim);
		int d2 = i % nextDim;

		int yi = d1 * nextDim + d2;

		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
	}
}

template <typename T, typename ReduceOp, int blockSize>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE void warp32ReduceStep(volatile T *shared, int threadId, ReduceOp op) {
	if (blockSize >= 64) shared[threadId] = op.step(shared[threadId], shared[threadId + 32]);
	if (blockSize >= 32) shared[threadId] = op.step(shared[threadId], shared[threadId + 16]);
	if (blockSize >= 16) shared[threadId] = op.step(shared[threadId], shared[threadId +  8]);
	if (blockSize >=  8) shared[threadId] = op.step(shared[threadId], shared[threadId +  4]);
	if (blockSize >=  4) shared[threadId] = op.step(shared[threadId], shared[threadId +  2]);
	if (blockSize >=  2) shared[threadId] = op.step(shared[threadId], shared[threadId +  1]);
}

/**
 * special for head reduce
 * the grideSize equal col
 */
template <typename T, typename ReduceOp, int blockSize>
__global__ void HeadReduceForward(const T *x, T *y, const int row, const int col, ReduceOp op) {
    SharedMemory<T> shareMemory;
	T *shared = shareMemory.pointer();

    int threaId = threadIdx.x;
	int blockId = blockIdx.x;

    int i = blockId + threaId * col;
    int j = threaId;

    shared[threaId] = op.init();

    while (j < row) {
        shared[threaId] = op.step(shared[threaId], x[i]);

        j += blockSize;
        i += blockSize * col;
    }

    __syncthreads();

	if (blockSize >= 1024) {
		if (threaId < 512) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 512]);
		}

		__syncthreads();
	}

    if (blockSize >= 512) {
		if (threaId < 256) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threaId < 128) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threaId < 64) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 64]);
		}

		__syncthreads();
	}

    if (threaId < 32) {
        warp32ReduceStep<T, ReduceOp, blockSize>(shared, threaId, op);
	}

    if (0 == threadId) {
        y[blockId] = op.complete(shared[0]);
    }
}

template <typename T, typename ReduceOp>
__global__ void HeadReduceBackward(const T *x, T *dx, const T *y, const T *dy, const int row, const int col, ReduceOp op) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int yi = xi % col;

		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
	}
}

/**
 * special for tail reduce
 */
template <typename T, typename ReduceOp, int blockSize>
__global__ void TailReduceForward(const T *x, T *y, const int row, const int col, ReduceOp op) {
    SharedMemory<T> shareMemory;
	T *shared = shareMemory.pointer();

	int threaId = threadIdx.x;
	int blockId = blockIdx.x;

	int i = blockId * col + threaId;
	int j = threaId;

	shared[threaId] = op.init();

    while (j < col) {
		shared[threaId] = op.step(shared[threaId], x[i]);

		j += blockSize;
		i += blockSize;
	}

    __syncthreads();

	if (blockSize >= 1024) {
		if (threaId < 512) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 512]);
		}

		__syncthreads();
	}

    if (blockSize >= 512) {
		if (threaId < 256) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threaId < 128) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threaId < 64) {
			shared[threaId] = op.step(shared[threaId], shared[threaId + 64]);
		}

		__syncthreads();
	}

    if (threaId < 32) {
        warp32ReduceStep<T, ReduceOp, blockSize>(shared, threaId, op);
	}

    if (0 == threadId) {
        y[blockId] = op.complete(shared[0]);
    }
}

template <typename T, typename ReduceOp>
__global__ void TailReduceBackward(const T *x, T *dx, const T *y, const T *dy, const int row, const int col, ReduceOp op) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int yi = xi / col;

		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
	}
}

template <typename T, typename ReduceOp>
void callTailReduceForward(const T *x, T *y, const int row, const int col, int blockSize) {
	int sharedSize = sizeof(T) * blockSize;

	switch (blockSize) {
	case 1024:
		TailReduceForward<T, ReduceOp<T>, 1024> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 512:
		TailReduceForward<T, ReduceOp<T>,  512> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 256:
		TailReduceForward<T, ReduceOp<T>,  256> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 128:
		TailReduceForward<T, ReduceOp<T>,  128> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 64:
		TailReduceForward<T, ReduceOp<T>,   64> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 32:
		TailReduceForward<T, ReduceOp<T>,   32> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 16:
		TailReduceForward<T, ReduceOp<T>,   16> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 8:
		TailReduceForward<T, ReduceOp<T>,    8> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 4:
		TailReduceForward<T, ReduceOp<T>,    4> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 2:
		TailReduceForward<T, ReduceOp<T>,    2> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	case 1:
		TailReduceForward<T, ReduceOp<T>,    1> <<<row, blockSize, sharedSize>>>(x, maxPtr, row, col, ReduceOp<T>());
		break;
	default:
		DEEP8_RUNTIME_ERROR("the block size is error");
		break
	}
}

}

#endif