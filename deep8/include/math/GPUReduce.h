#ifndef DEEP8_MATH_REDUCE_H
#define DEEP8_MATH_REDUCE_H

#include "model/Shape.h"
#include "utils/ShapeUtils.h"
#include "basic/GPUBasic.h"

namespace Deep8 {
namespace Math {

template <typename T, typename ReduceOp, int NumDims>
__global__ void ReduceKernel(const T *x, const NVShape<NumDims> xshape, T *y, const NVShape<NumDims> yshape, ReduceOp op, const int N) {
    int start   = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

    int xindex[NumDims];
    int yindex[NumDims];

	for (int i = start; i < N; i += stride) {
		T ret = op.commense();

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

               ret = op.init(ret, x[xi]);
           }

           if (xshape.dims[j] == yshape.dims[j]) {
               j--;
           } else {
               xindex[j]++;

               if (xindex[j] >= xshape.dims[j]) {
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
__global__ void ReduceGradKernel(const T *x, T *dx, const NVShape<NumDims> xshape, const T *y, const T *dy, const NVShape<NumDims> yshape, ReduceOp op, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int xi = start; xi < N; xi += stride) {
		int yi = 0;

		for (int k = 0, index = xi; k < NumDims; ++k) {
			int xd = index / xshape.strides[k];

			if (xshape.dims[k] == yshape.dims[k]) {
				yi += xd * yshape.strides[k];
			}

			index %= xshape.strides[k];
		}

		dx[xi] += op(x[xi], y[yi], dy[yi]);
	}
}

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
                                        ReduceOp op,
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
void CallAllReduceKernel(const T *x, T *y, const int size, ReduceOp op) {
    CallTailReduceKernel<T, ReduceOp>(x, y, 1, size, op);
}

template <typename T, typename ReduceOp>
void CallAllReduceGradKernel(const T *x, T *dx, const T *y, const T *dy, const int size, ReduceOp op) {
    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (size + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    TailReduceGradKernel<T, ReduceOp> <<< grideSize, blockSize >>> (x, dx, y, dy, 1, size, op, size);
}

template <typename T, typename ReduceOp>
void CallReduceKernel(const T *x, std::vector<int> &xshape, T *y, std::vector<int> &yshape, ReduceOp op) {
	DEEP8_ARGUMENT_CHECK(xshape.size() == yshape.size(), "the reduce shape rank must be same");
	DEEP8_ARGUMENT_CHECK(1 <= xshape.size() && xshape.size() <= MAX_TENSOR_DIMS + 1, "the reduce shape rank error");

	int rank = xshape.size();

	std::vector<bool> reduceAxis(rank);

	for (int i = 0; i < rank; ++i) {
		DEEP8_ARGUMENT_CHECK(xshape[i] >= 1 && yshape[i] >= 1, "the shape is error");
		DEEP8_ARGUMENT_CHECK(1 == yshape[i] || xshape[i] == yshape[i], "the shape is error");

		if (1 == yshape[i]) {
            reduceAxis[i] = true;
		} else {
            reduceAxis[i] = false;
		}
	}

	std::vector<bool> shrikAxis;
	shrikAxis.emplace_back(reduceAxis[0]);

	for (int i = 1; i < rank; ++i) {
		if (reduceAxis[i] != reduceAxis[i - 1]) {
			shrikAxis.emplace_back(reduceAxis[i]);
		}
	}

	if (1 == shrikAxis.size()) {
		DEEP8_ARGUMENT_CHECK(shrikAxis[0], "error, must be set the reduce dimension");

		/**reduce all dimension*/
		int size = 1;

		for (auto i : xshape) {
			size *= i;
		}

		CallAllReduceKernel<T, ReduceOp>(x, y, size, op);
	} else if (2 == shrikAxis.size() && shrikAxis[0] && !shrikAxis[1]) {
		/**head reduce*/
		int row = 1;
		int col = 1;

		for (int i = 0; i < rank; ++i) {
			if (reduceAxis[i]) {
				row *= xshape[i];
			} else {
				col *= xshape[i];
			}
		}

		CallHeadReduceKernel<T, ReduceOp>(x, y, row, col, op);
	} else if (2 == shrikAxis.size() && !shrikAxis[0] && shrikAxis[1]) {
		/**tail reduce*/
		int row = 1;
		int col = 1;

		for (int i = 0; i < rank; ++i) {
			if (!reduceAxis[i]) {
				row *= xshape[i];
			} else {
				col *= xshape[i];
			}
		}

		CallTailReduceKernel<T, ReduceOp>(x, y, row, col, op);
	} else if (3 == shrikAxis.size() && !shrikAxis[0] && shrikAxis[1] && !shrikAxis[2]) {
		/**middle reduce*/
		int dim0 = 1;
		int dim1 = 1;
		int dim2 = 1;

		int i = 0; 
		for (; i < rank && !reduceAxis[i]; ++i) {
			dim0 *= xshape[i];
		}

		for (; i < rank && reduceAxis[i]; ++i) {
			dim1 *= xshape[i];
		}

		for (; i < rank; ++i) {
			dim2 *= xshape[i];
		}

		int N         = dim0 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		MiddleReduceKernel<T, ReduceOp> <<<grideSize, blockSize>>>(x, y, dim0, dim1, dim2, op, N);
	} else {
		int N = 1;

		for (int i = 0; i < rank; ++i) {
			N *= yshape[i];
		}

		int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		if (1 == rank) {
			auto xnvshape = convertVectorToNVShape<1>(xshape);
			auto ynvshape = convertVectorToNVShape<1>(yshape);

			ReduceKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, xnvshape, y, ynvshape, op, N);
		} else if (2 == rank) {
			auto xnvshape = convertVectorToNVShape<2>(xshape);
			auto ynvshape = convertVectorToNVShape<2>(yshape);

			ReduceKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, xnvshape, y, ynvshape, op, N);
		} else if (3 == rank) {
			auto xnvshape = convertVectorToNVShape<3>(xshape);
			auto ynvshape = convertVectorToNVShape<3>(yshape);

			ReduceKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, xnvshape, y, ynvshape, op, N);
		} else if (4 == rank) {
			auto xnvshape = convertVectorToNVShape<4>(xshape);
			auto ynvshape = convertVectorToNVShape<4>(yshape);

			ReduceKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, xnvshape, y, ynvshape, op, N);
		} else if (5 == rank) {
			auto xnvshape = convertVectorToNVShape<5>(xshape);
			auto ynvshape = convertVectorToNVShape<5>(yshape);

			ReduceKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, xnvshape, y, ynvshape, op, N);
		} else {
			DEEP8_RUNTIME_ERROR("the shape's rank is error");
		}
	}
}

template <typename T, typename ReduceOp>
void CallReduceGradKernel(const T *x, T *dx, std::vector<int> &xshape, const T *y, const T *dy, std::vector<int> &yshape, ReduceOp op) {
	DEEP8_ARGUMENT_CHECK(xshape.size() == yshape.size(), "the reduce shape rank must be same");
	DEEP8_ARGUMENT_CHECK(1 <= xshape.size() && xshape.size() <= MAX_TENSOR_DIMS + 1, "the reduce shape rank error");

	int rank = xshape.size();

	std::vector<bool> reduceAxis(rank);

	for (int i = 0; i < rank; ++i) {
		DEEP8_ARGUMENT_CHECK(xshape[i] >= 1 && yshape[i] >= 1, "the shape is error");
		DEEP8_ARGUMENT_CHECK(1 == yshape[i] || xshape[i] == yshape[i], "the shape is error");

		if (1 == yshape[i]) {
            reduceAxis[i] = true;
		} else {
            reduceAxis[i] = false;
		}
	}

	std::vector<bool> shrikAxis;
	shrikAxis.emplace_back(reduceAxis[0]);

	for (int i = 1; i < rank; ++i) {
		if (reduceAxis[i] != reduceAxis[i - 1]) {
			shrikAxis.emplace_back(reduceAxis[i]);
		}
	}

	if (1 == shrikAxis.size()) {
		DEEP8_ARGUMENT_CHECK(shrikAxis[0], "error, must be set the reduce dimension");

		/**reduce all dimension*/
		int size = 1;

		for (auto i : xshape) {
			size *= i;
		}

		CallAllReduceGradKernel<T, ReduceOp>(x, dx, y, dy, size, op);
	} else if (2 == shrikAxis.size() && shrikAxis[0] && !shrikAxis[1]) {
		/**head reduce*/
		int row = 1;
		int col = 1;

		for (int i = 0; i < rank; ++i) {
			if (reduceAxis[i]) {
				row *= xshape[i];
			} else {
				col *= xshape[i];
			}
		}

		int N = row * col;

		int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		HeadReduceGradKernel<T, ReduceOp> <<<grideSize, blockSize>>>(x, dx, y, dy, row, col, op, N);
	} else if (2 == shrikAxis.size() && !shrikAxis[0] && shrikAxis[1]) {
		/**tail reduce*/
		int row = 1;
		int col = 1;

		for (int i = 0; i < rank; ++i) {
			if (!reduceAxis[i]) {
				row *= xshape[i];
			} else {
				col *= xshape[i];
			}
		}

		int N = row * col;

		int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		TailReduceGradKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, dx, y, dy, row, col, op, N);
	} else if (3 == shrikAxis.size() && !shrikAxis[0] && shrikAxis[1] && !shrikAxis[2]) {
		/**middle reduce*/
		int dim0 = 1;
		int dim1 = 1;
		int dim2 = 1;

		int i = 0; 
		for (; i < rank && !reduceAxis[i]; ++i) {
			dim0 *= xshape[i];
		}

		for (; i < rank && reduceAxis[i]; ++i) {
			dim1 *= xshape[i];
		}

		for (; i < rank; ++i) {
			dim2 *= xshape[i];
		}

		int N = dim0 * dim1 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		MiddleReduceGradKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, dx, y, dy, dim0, dim1, dim2, op, N);
	} else {
		int N = 1;

		for (int i = 0; i < rank; ++i) {
			N *= xshape[i];
		}

		int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

		if (1 == rank) {
			auto xnvshape = convertVectorToNVShape<1>(xshape);
			auto ynvshape = convertVectorToNVShape<1>(yshape);

			ReduceGradKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, dx, xnvshape, y, dy, ynvshape, op, N);
		} else if (2 == rank) {
			auto xnvshape = convertVectorToNVShape<2>(xshape);
			auto ynvshape = convertVectorToNVShape<2>(yshape);

			ReduceGradKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, dx, xnvshape, y, dy, ynvshape, op, N);
		} else if (3 == rank) {
			auto xnvshape = convertVectorToNVShape<3>(xshape);
			auto ynvshape = convertVectorToNVShape<3>(yshape);

			ReduceGradKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, dx, xnvshape, y, dy, ynvshape, op, N);
		} else if (4 == rank) {
			auto xnvshape = convertVectorToNVShape<4>(xshape);
			auto ynvshape = convertVectorToNVShape<4>(yshape);

			ReduceGradKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, dx, xnvshape, y, dy, ynvshape, op, N);
		} else if (5 == rank) {
			auto xnvshape = convertVectorToNVShape<5>(xshape);
			auto ynvshape = convertVectorToNVShape<5>(yshape);
			
			ReduceGradKernel<T, ReduceOp><<<grideSize, blockSize>>>(x, dx, xnvshape, y, dy, ynvshape, op, N);
		} else {
			DEEP8_RUNTIME_ERROR("the shape's rank is error");
		}
	} 
}

}
}

#endif