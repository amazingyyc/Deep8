#ifndef DEEP8_GPUREDUCE_H
#define DEEP8_GPUREDUCE_H

#include "Shape.h"
#include "GPUBasic.h"

namespace Deep8 {

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
 * the common reduce kernel
 */
template <typename T, typename ReduceOp, int NumDims>
__global__ void CommonReduce(const T *x, const NVShape<NumDims> xshape, T *y, NVShape<NumDims> yshape, ReduceOp op) {
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

/**
 * special for tail reduce
 */
template <typename T, typename ReduceOp, int blockSize>
__global__ void TailReduce(const T *x, T *y, const int row, const int col, ReduceOp op) {
    SharedMemory<T> shareMemory;
	T *shared = shareMemory.pointer();

	int threaId = threadIdx.x;
	int blockId = blockIdx.x;

	int i = blockId * col + threaId;
	int j = threaId;

	shared[threaId] = op.init();

    while (j < size) {
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



}

#endif