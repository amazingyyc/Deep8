#ifndef DEEP8_MATH_REDUCE_H
#define DEEP8_MATH_REDUCE_H

#include "GPUBasic.h"

namespace Deep8 {
namespace Math {

template <typename T, typename ReduceOp>
__global__ void MiddleReduceKernel( const T *x, 
                                    T *y, 
                                    const int dim0, 
                                    const int dim1, 
                                    const int dim2, 
                                    ReduceOp op, 
                                    const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    for (int yi = start; yi < N; yi += stride) {
        T ret = op.commense();

        int d0 = yi / dim2;
		int d2 = yi % dim2;

		int xi = d0 * dim1 * dim2 + d2;

        for (int k = 0; k < dim1; ++k) {
            ret = op.init(ret, x[xi]);

            xi += dim2;
        }

        y[yi] = op.complete(ret);
    }
}

template <typename T, typename ReduceOp>
__global__ void MiddleReduceGradKernel( const T *x, 
                                        T *dx, 
                                        const T *y, 
                                        const T *dy, 
                                        const int dim0,
                                        const int dim1, 
                                        const int dim2,
                                        ReduceOP op, 
                                        const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    for (int xi = start; xi < N; xi += stride) {
		int d0 = xi / (dim1 * dim2);
		int d2 = xi % dim2;

		int yi = d0 * dim2 + d2;

		dx[xi] += op(x[xi], y[yi], dy[yi]);
	}
}

/**
 * the CUDN does not support template shared memory
 * ref:https://wrf.ecse.rpi.edu//wiki/ParallelComputingSpring2015/cuda/nvidia/samples/0_Simple/simpleTemplates/sharedmem.cuh
 */
template <typename T>
struct GPUSharedMemory {
	__device__ T *pointer() {
		return nullptr;
	}
};

template <>
struct GPUSharedMemory<float> {
	__device__ float *pointer() {
		extern __shared__ float sharedFloat[];
		return sharedFloat;
	}
};

template <>
struct GPUSharedMemory<double> {
	__device__ double *pointer() {
		extern __shared__ double sharedDouble[];
		return sharedDouble;
	}
};

#ifdef HAVE_HALF
template <>
struct GPUSharedMemory<half> {
	__device__ half *pointer() {
		extern __shared__ half sharedHalf[];
		return sharedHalf;
	}
};
#endif

template <typename T, typename ReduceOp, int blockSize>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE void Warp32ReduceStep(volatile T *shared, int threadId, ReduceOp op) {
	if (blockSize >= 64) {
		shared[threadId] = op.step(shared[threadId], shared[threadId + 32]);
	}
	
	if (blockSize >= 32) {
		shared[threadId] = op.step(shared[threadId], shared[threadId + 16]);
	}

	if (blockSize >= 16) {
		shared[threadId] = op.step(shared[threadId], shared[threadId + 8]);
	}

	if (blockSize >= 8) {
		shared[threadId] = op.step(shared[threadId], shared[threadId + 4]);
	}

	if (blockSize >= 4) {
		shared[threadId] = op.step(shared[threadId], shared[threadId + 2]);
	}

	if (blockSize >= 2) {
		shared[threadId] = op.step(shared[threadId], shared[threadId + 1]);
	}
}

/**
 * special for head reduce
 * the grideSize equal col
 */
template <typename T, typename ReduceOp, int blockSize>
__global__ void HeadReduceKernel(const T *x, T *y, const int row, const int col, ReduceOp op) {
    GPUSharedMemory<T> shareMemory;
	T *shared = shareMemory.pointer();

    int threadId = threadIdx.x;
	int blockId  = blockIdx.x;

    int i = blockId + threadId * col;
    int j = threadId;

    shared[threadId] = op.commense();

    while (j < row) {
        shared[threadId] = op.init(shared[threadId], x[i]);

        j += blockSize;
        i += blockSize * col;
    }

    __syncthreads();

	if (blockSize >= 1024) {
		if (threadId < 512) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 512]);
		}

		__syncthreads();
	}

    if (blockSize >= 512) {
		if (threadId < 256) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadId < 128) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadId < 64) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 64]);
		}

		__syncthreads();
	}

    if (threadId < 32) {
        Warp32ReduceStep<T, ReduceOp, blockSize>(shared, threadId, op);
	}

    if (0 == threadId) {
        y[blockId] = op.complete(shared[threadId]);
    }
}

template <typename T, typename ReduceOp>
__global__ void HeadReduceGradKernel(const T *x, T *dx, const T *y, const T *dy, const int row, const int col, ReduceOp op, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int yi = xi % col;

		dx[xi] += op(x[xi], y[yi], dy[yi]);
	}
}

template <typename T, typename ReduceOp>
void CallHeadReduceKernel(const T *x, T *y, const int row, const int col, ReduceOp op) {
    int blockSize = 1024;

	if (blockSize > row) {
		blockSize = prevPowerOf2(row);
	}

    int sharedSize = sizeof(T) * blockSize;

    switch (blockSize) {
	case 1024:
		HeadReduceKernel<T, ReduceOp, 1024> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 512:
		HeadReduceKernel<T, ReduceOp, 512> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 256:
		HeadReduceKernel<T, ReduceOp, 256> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 128:
		HeadReduceKernel<T, ReduceOp, 128> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 64:
		HeadReduceKernel<T, ReduceOp, 64> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 32:
		HeadReduceKernel<T, ReduceOp, 32> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 16:
		HeadReduceKernel<T, ReduceOp, 16> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 8:
		HeadReduceKernel<T, ReduceOp, 8> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 4:
		HeadReduceKernel<T, ReduceOp, 4> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 2:
		HeadReduceKernel<T, ReduceOp, 2> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 1:
		HeadReduceKernel<T, ReduceOp, 1> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the block size is error");
		break;
	}
}

/**
 * special for tail reduce 
 */
template <typename T, typename ReduceOp, int blockSize>
__global__ void TailReduceKernel(const T *x, T *y, const int row, const int col, ReduceOp op) {
    GPUSharedMemory<T> shareMemory;
	T *shared = shareMemory.pointer();

	int threadId = threadIdx.x;
	int blockId  = blockIdx.x;

	int i = blockId * col + threadId;
	int j = threadId;

	shared[threadId] = op.commense();

    while (j < col) {
		shared[threadId] = op.init(shared[threadId], x[i]);

		j += blockSize;
		i += blockSize;
	}

    __syncthreads();

	if (blockSize >= 1024) {
		if (threadId < 512) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 512]);
		}

		__syncthreads();
	}

    if (blockSize >= 512) {
		if (threadId < 256) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threadId < 128) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threadId < 64) {
			shared[threadId] = op.step(shared[threadId], shared[threadId + 64]);
		}

		__syncthreads();
	}

    if (threadId < 32) {
        Warp32ReduceStep<T, ReduceOp, blockSize>(shared, threadId, op);
	}

    if (0 == threadId) {
        y[blockId] = op.complete(shared[threadId]);
    }
}

template <typename T, typename ReduceOp>
__global__ void TailReduceGradKernel(const T *x, T *dx, const T *y, const T *dy, const int row, const int col, ReduceOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int yi = xi / col;

		dx[xi] += op(x[xi], y[yi], dy[yi]);
	}
}

template <typename T, typename ReduceOp>
void CallTailReduceKernel(const T *x, T *y, const int row, const int col, ReduceOp op) {
	int blockSize = 1024;

	if (col < blockSize) {
		blockSize = prevPowerOf2(col);
	}

	int sharedSize = sizeof(T) * blockSize;

    switch (blockSize) {
	case 1024:
		TailReduceKernel<T, ReduceOp, 1024> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 512:
		TailReduceKernel<T, ReduceOp, 512> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 256:
		TailReduceKernel<T, ReduceOp, 256> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 128:
		TailReduceKernel<T, ReduceOp, 128> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 64:
		TailReduceKernel<T, ReduceOp, 64> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 32:
		TailReduceKernel<T, ReduceOp, 32> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 16:
		TailReduceKernel<T, ReduceOp, 16> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 8:
		TailReduceKernel<T, ReduceOp, 8> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 4:
		TailReduceKernel<T, ReduceOp, 4> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 2:
		TailReduceKernel<T, ReduceOp, 2> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 1:
		TailReduceKernel<T, ReduceOp, 1> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the block size is error");
		break;
	}
}

/**
 * for Reduce 
 */
template <typename T, typename ReduceOp>
void CallReduceKernel(const T *x, T *y, const int size, ReduceOp op) {
    CallTailReduceKernel<T, ReduceOp>(x, y, 1, size, op);
}

template <typename T, typename ReduceOp>
void CallReduceGradKernel(const T *x, T *dx, const T *y, const T *dy, const int size, ReduceOp op) {
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    TailReduceGradKernel<T, ReduceOp> <<< grideSize, blockSize >>> (x, dx, y, dy, 1, size, op, size);
}

}
}

#endif