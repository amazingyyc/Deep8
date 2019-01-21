#ifndef DEEP8_GPUREDUCE_H
#define DEEP8_GPUREDUCE_H

#include "Shape.h"
#include "GPUBasic.h"

namespace Deep8 {

/**********************************************************************************************************
 *the common reduce forward kernel
**********************************************************************************************************/
//template <typename T, typename ReduceOp, int NumDims>
//__global__ void CommonReduceForward(const T *x, const NVShape<NumDims> xshape, T *y, const NVShape<NumDims> yshape, ReduceOp op, const int N) {
//    int start  = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//    int xindex[NumDims];
//    int yindex[NumDims];
//
//	for (int i = start; i < N; i += stride) {
//		T ret = op.init();
//
//        for (int k = 0, index = i; k < NumDims; ++k) {
//            yindex[k] = index / yshape.strides[k];
//            xindex[k] = yindex[k];
//
//            index %= yshape.strides[k];
//        }
//
//        int j = NumDims - 1;
//
//        while (j >= 0) {
//            if (j == NumDims - 1) {
//                int xi = 0;
//
//                for (int l = 0; l < NumDims; ++l) {
//                    xi += xindex[l] * xshape.strides[l];
//                }
//
//                ret = op.step(ret, x[xi]);
//            }
//
//            if (xshape.dims[j] == yshape.dims[j]) {
//                j--;
//            } else {
//                xindex[j]++;
//
//                if (xindex[j] >= xshape.dims[j]) {
//                    j--;
//                } else {
//                    for (int l = j + 1; l < NumDims; ++l) {
//                        xindex[l] = yindex[l];
//                    }
//
//                    j = NumDims - 1;
//                }
//            }
//        }
//
//        y[i] = op.complete(ret);
//	}
//}

/**********************************************************************************************************
 *the common reduce backward kernel
**********************************************************************************************************/
//template <typename T, typename ReduceOp, int NumDims>
//__global__ void CommonReduceBackward(
//	const T *x,       T *dx, const NVShape<NumDims> xshape, 
//	const T *y, const T *dy, const NVShape<NumDims> yshape, 
//	ReduceOp op, const int N) {
//	int start  = blockIdx.x * blockDim.x + threadIdx.x;
//	int stride = blockDim.x * gridDim.x;
//
//	int xindex[NumDims];
//
//	for (int xi = start; xi < N; xi += stride) {
//		int yi = 0;
//
//		for (int k = 0, index = xi; k < NumDims; ++k) {
//			int xd = index / xshape.stride[k];
//
//			if (xshape.dims[k] == yshape.dims[k]) {
//				yi += xd * yshape.strides[k];
//			}
//
//			index %= xshape.strides[k];
//		}
//
//		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
//	}
//}

/**********************************************************************************************************
 *the middle reduce forward kernel
**********************************************************************************************************/
/**
 * support the x dimesnison is [dim0, dim1, dim2]
 * then y diemsnions is [dim0, 1, dim2]
 */
template <typename T, typename ReduceOp>
__global__ void MiddleReduceForward(const T *x, T *y, const int dim0, const int dim1, const int dim2, ReduceOp op, const int N) {
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

/**********************************************************************************************************
 *the middle reduce Backward kernel
**********************************************************************************************************/
template <typename T, typename ReduceOp> 
__global__ void MiddleReduceBackward(
	const T *x, T *dx, const T *y, const T *dy, 
	const int dim0, const int dim1, const int dim2, ReduceOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int d0 = xi / (dim1 * dim2);
		int d2 = xi % dim2;

		int yi = d0 * dim2 + d2;

		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
	}
}

/**********************************************************************************************************
 *the tail and head reduce kernel
**********************************************************************************************************/
template <typename T, typename ReduceOp, int blockSize>
DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE void warp32ReduceStep(volatile T *shared, int threadId, ReduceOp op) {
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
__global__ void HeadReduceForward(const T *x, T *y, const int row, const int col, ReduceOp op) {
    SharedMemory<T> shareMemory;
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
        warp32ReduceStep<T, ReduceOp, blockSize>(shared, threadId, op);
	}

    if (0 == threadId) {
        y[blockId] = op.complete(shared[threadId]);
    }
}

template <typename T, typename ReduceOp>
__global__ void HeadReduceBackward(const T *x, T *dx, const T *y, const T *dy, const int row, const int col, ReduceOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int yi = xi % col;

		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
	}
}

template <typename T, typename ReduceOp>
void callHeadReduceForward(const T *x, T *y, const int row, const int col, ReduceOp op) {
	int blockSize = 1024;

	if (blockSize > row) {
		blockSize = prevPowerOf2(row);
	}

	int sharedSize = sizeof(T) * blockSize;

	switch (blockSize) {
	case 1024:
		HeadReduceForward<T, ReduceOp, 1024> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 512:
		HeadReduceForward<T, ReduceOp, 512> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 256:
		HeadReduceForward<T, ReduceOp, 256> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 128:
		HeadReduceForward<T, ReduceOp, 128> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 64:
		HeadReduceForward<T, ReduceOp, 64> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 32:
		HeadReduceForward<T, ReduceOp, 32> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 16:
		HeadReduceForward<T, ReduceOp, 16> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 8:
		HeadReduceForward<T, ReduceOp, 8> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 4:
		HeadReduceForward<T, ReduceOp, 4> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 2:
		HeadReduceForward<T, ReduceOp, 2> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 1:
		HeadReduceForward<T, ReduceOp, 1> << <col, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the block size is error");
		break;
	}
}

/********************************************************************************
 * special for tail reduce
 *******************************************************************************/
template <typename T, typename ReduceOp, int blockSize>
__global__ void TailReduceForward(const T *x, T *y, const int row, const int col, ReduceOp op) {
    SharedMemory<T> shareMemory;
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
        warp32ReduceStep<T, ReduceOp, blockSize>(shared, threadId, op);
	}

    if (0 == threadId) {
        y[blockId] = op.complete(shared[threadId]);
    }
}

template <typename T, typename ReduceOp>
__global__ void TailReduceBackward(const T *x, T *dx, const T *y, const T *dy, const int row, const int col, ReduceOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int yi = xi / col;

		dx[xi] += op.backward(x[xi], y[yi], dy[yi]);
	}
}

template <typename T, typename ReduceOp>
void callTailReduceForward(const T *x, T *y, const int row, const int col, ReduceOp op) {
	int blockSize = 1024;

	if (col < blockSize) {
		blockSize = prevPowerOf2(col);
	}

	int sharedSize = sizeof(T) * blockSize;

	switch (blockSize) {
	case 1024:
		TailReduceForward<T, ReduceOp, 1024> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 512:
		TailReduceForward<T, ReduceOp, 512> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 256:
		TailReduceForward<T, ReduceOp, 256> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 128:
		TailReduceForward<T, ReduceOp, 128> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 64:
		TailReduceForward<T, ReduceOp, 64> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 32:
		TailReduceForward<T, ReduceOp, 32> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 16:
		TailReduceForward<T, ReduceOp, 16> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 8:
		TailReduceForward<T, ReduceOp, 8> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 4:
		TailReduceForward<T, ReduceOp, 4> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 2:
		TailReduceForward<T, ReduceOp, 2> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	case 1:
		TailReduceForward<T, ReduceOp, 1> << <row, blockSize, sharedSize >> > (x, y, row, col, op);
		break;
	default:
		DEEP8_RUNTIME_ERROR("the block size is error");
		break;
	}
}

/******************************************************************************************************
 * all reduce  
 *****************************************************************************************************/
template <typename T, typename ReduceOp>
void callAllReduceForward(const T *x, T *y, const int size, ReduceOp op) {
	callTailReduceForward<T, ReduceOp>(x, y, 1, size, op);
}

template <typename T, typename ReduceOp>
void callAllReduceBackward(const T *x, T *dx, const T *y, const T *dy, const int size, ReduceOp op) {
	int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	TailReduceBackward<T, ReduceOp> <<< grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, dx, y, dy, 1, size, op, size);
}

}

#endif