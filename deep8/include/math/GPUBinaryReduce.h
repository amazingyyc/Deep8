#ifndef DEEP8_MATH_GPUBINARYREDUCE_H
#define DEEP8_MATH_GPUBINARYREDUCE_H

#include "basic/GPUBasic.h"
#include "math/GPUReduce.h"

namespace Deep8 {
namespace Math {

/**
 * special for tail reduce 
 */
template <typename T, typename ReduceOp, int blockSize>
__global__ void TailBinaryReduceKernel(const T *x, const T *y, T *z, const int row, const int col, ReduceOp op) {
    GPUSharedMemory<T> shareMemory;
	T *shared = shareMemory.pointer();

	int threadId = threadIdx.x;
	int blockId  = blockIdx.x;

	int i = blockId * col + threadId;
	int j = threadId;

    shared[threadId] = op.commense();

    while (j < col) {
		shared[threadId] = op.init(shared[threadId], x[i], y[i]);

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
        z[blockId] = op.complete(shared[threadId]);
    }
}

template <typename T, typename ReduceOp>
void CallTailBinaryReduceKernel(const T *x, const T *y, T *z, const int row, const int col, ReduceOp op) {
    int blockSize = DEEP8_GPU_BLOCK_SIZE;

	if (blockSize > col) {
		blockSize = prevPowerOf2(col);
	}

    int sharedSize = sizeof(T) * blockSize;

    switch (blockSize) {
	case 1024:
		TailBinaryReduceKernel<T, ReduceOp, 1024> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 512:
		TailBinaryReduceKernel<T, ReduceOp, 512> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 256:
		TailBinaryReduceKernel<T, ReduceOp, 256> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 128:
		TailBinaryReduceKernel<T, ReduceOp, 128> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 64:
		TailBinaryReduceKernel<T, ReduceOp, 64> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 32:
		TailBinaryReduceKernel<T, ReduceOp, 32> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 16:
		TailBinaryReduceKernel<T, ReduceOp, 16> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 8:
		TailBinaryReduceKernel<T, ReduceOp, 8> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 4:
		TailBinaryReduceKernel<T, ReduceOp, 4> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 2:
		TailBinaryReduceKernel<T, ReduceOp, 2> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	case 1:
		TailBinaryReduceKernel<T, ReduceOp, 1> << <row, blockSize, sharedSize >> > (x, y, z, row, col, op);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the block size is error");
		break;
	}
}

template <typename T, typename ReduceOp>
__global__ void TailBinaryReduceGradXKernel(const T *x, 
                                            T *dx, 
                                            const T *y, 
                                            const T *z, 
                                            const T *dz, 
                                            const int row, 
                                            const int col, 
                                            ReduceOp op, 
                                            const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int j = i / col;

        dx[i] += op(x[i], y[i], z[j], dz[j]);
	}
}

template <typename T, typename ReduceOp>
void CallTailBinaryReduceGradXKernel(const T *x, T *dx, const T *y, const T *z, const T *dz, const int row, const int col, ReduceOp op) {
    int N = row * col;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    TailBinaryReduceGradXKernel<T, ReduceOp> <<<blockSize, grideSize>>> (x, dx, y, z, dz, row, col, op, N);
}

template <typename T, typename ReduceOp>
__global__ void TailBinaryReduceGradYKernel(const T *x, 
                                            const T *y,
                                            T *dy, 
                                            const T *z, 
                                            const T *dz, 
                                            const int row, 
                                            const int col, 
                                            ReduceOp op, 
                                            const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int j = i / col;

        dy[i] += op(x[i], y[i], z[j], dz[j]);
	}
}

template <typename T, typename ReduceOp>
void CallTailBinaryReduceGradYKernel(const T *x, const T *y, T *dy, const T *z, const T *dz, const int row, const int col, ReduceOp op) {
    int N = row * col;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    TailBinaryReduceGradYKernel<T, ReduceOp> <<<blockSize, grideSize>>> (x, y, dy, z, dz, row, col, op, N);
}


}
}

#endif