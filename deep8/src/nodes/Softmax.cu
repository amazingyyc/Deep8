#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.h"
#include "GPUReduce.h"
#include "Softmax.h"

namespace Deep8 {

/**
 * find the max value and put it in y
 */
template <typename T>
struct SoftmaxFindMaxOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return CuMath::cuMinValue<T>();
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
		return ret >= cur ? ret : cur;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 >= ret2 ? ret1 : ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return ret;
	}
};

template <typename real>
struct SoftmaxExpMinusOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real forward(const real &x, const real &y) {
		return CuMath::cuExp(x - y);
	}
};

template <typename T>
struct SoftmaxSumOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
		return ret + cur;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
        return ret1 + ret2;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
        return ret;
    }
};

template <typename real>
struct SoftmaxDivideOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real forward(const real &x, const real &y) {
		return x / y;
	}
};

/**
 * support the x/y dimension is [dim0, dim1, dim2]
 * the dotptr dimension is [dim0, 1, dim2]
 * dotptr[i, 0, j] = sum(y[i, l, j] * dy[i, l, j]), l = (0..dim1)
 */
template <typename T>
struct SoftmaxBackwardDotOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}
};

template <int blockSize, typename real>
__global__ void SoftmaxBackwardDotKernel(const real *y, const real *dy, real *dotptr, const int dim0, const int dim1, const int dim2) {
     SharedMemory<real> shareMemory;
     real *shared = shareMemory.pointer();

     int threaId = threadIdx.x;
     int blockId = blockIdx.x;

     int d0 = blockId / dim2;
     int d2 = blockId % dim2;

     int i = threaId;
     int j = d0 * dim1 * dim2 + i * dim2 + d2;

     shared[threaId] = 0;

     while (i < dim1) {
        shared[threaId] = y[j] * dy[j];

        i += blockSize;
        j += blockSize * dim2;
     }

     __syncthreads();

     if (blockSize >= 1024) {
         if (threaId < 512) {
             shared[threaId] += shared[threaId + 512];
         }

         __syncthreads();
     }

     if (blockSize >= 512) {
         if (threaId < 256) {
             shared[threaId] += shared[threaId + 256];
         }

         __syncthreads();
     }

     if (blockSize >= 256) {
         if (threaId < 128) {
             shared[threaId] += shared[threaId + 128];
         }

         __syncthreads();
     }

     if (blockSize >= 128) {
         if (threaId < 64) {
             shared[threaId] += shared[threaId + 64];
         }

         __syncthreads();
     }

     if (threaId < 32) {
 		warp32ReduceStep<real, SoftmaxBackwardDotOp<real>, blockSize>(shared, threaId, SoftmaxBackwardDotOp<real>());
     }

     if (0 == threaId) {
         dotptr[blockId] = shared[threaId];
     }
}

template <typename real>
__global__ void SoftmaxBackwardKernel(real *dx, const real *y, const real *dy, const real *dotptr, const int dim0, const int dim1, const int dim2, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int d0 = i / (dim1 * dim2);
        int d2 = i % dim2;

        int j = d0 * dim2 + d2;

        dx[i] += (dy[i] - dotptr[j]) * y[i];
    }
}

template <typename T>
void Softmax<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = (GPUDevice*)output->device();

	auto x = inputs[0]->data();
	auto y = output->data();

    auto shape = inputs[0]->shape;
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) shape.dim(i);
        }

        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int) shape.dim(i);
        }
    }

	auto tempptr = (T*)device->malloc(sizeof(T) * dim0 * dim2);

	/**find max value*/
	if (1 == dim2) {
		/**tail reduce*/
        callTailReduceForward<T, SoftmaxFindMaxOp<T>>(x, tempptr, dim0, dim1);
	} else if (1 == dim0) {
		/**head reduce*/
        callHeadReduceForward<T, SoftmaxFindMaxOp<T>>(x, tempptr, dim1, dim2);
	} else {
		/**middle reduce*/
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;        
        MiddleReduceForward<T, SoftmaxFindMaxOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, tempptr, dim0, dim1, dim2, SoftmaxFindMaxOp<T>(), N);
	}

	/**y = exp(x - max)*/
    if (1 == dim2) {
        /**tail reduce*/
        NVShape<2> xNVShape;
        NVShape<2> maxNVShape;
        NVShape<2> yNVShape;

        xNVShape.dims[0] = dim0;
        xNVShape.dims[1] = dim1;
        xNVShape.strides[0] = dim1;
        xNVShape.strides[1] = 1;

        maxNVShape.dims[0] = dim0;
        maxNVShape.dims[1] = 1;
        maxNVShape.strides[0] = 1;
        maxNVShape.strides[1] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.strides[0] = dim1;
        yNVShape.strides[1] = 1;

       int N = dim0 * dim1;
       int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

       BinaryElementWiseForward<T, SoftmaxExpMinusOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, tempptr, maxNVShape, y, yNVShape, SoftmaxExpMinusOp<T>(), N);
    } else if (1 == dim0) {
        /**head reduce*/
        NVShape<2> xNVShape;
        NVShape<2> maxNVShape;
        NVShape<2> yNVShape;

        xNVShape.dims[0] = dim1;
        xNVShape.dims[1] = dim2;
        xNVShape.strides[0] = dim2;
        xNVShape.strides[1] = 1;

        maxNVShape.dims[0] = 1;
        maxNVShape.dims[1] = dim2;
        maxNVShape.strides[0] = dim2;
        maxNVShape.strides[1] = 1;

        yNVShape.dims[0] = dim1;
        yNVShape.dims[1] = dim2;
        yNVShape.strides[0] = dim2;
        yNVShape.strides[1] = 1;

        int N = dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxExpMinusOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, tempptr, maxNVShape, y, yNVShape, SoftmaxExpMinusOp<T>(), N);
    } else {
        /**middle reduce*/
        NVShape<3> xNVShape;
        NVShape<3> maxNVShape;
        NVShape<3> yNVShape;

        xNVShape.dims[0] = dim0;
        xNVShape.dims[1] = dim1;
        xNVShape.dims[2] = dim2;
        xNVShape.strides[0] = dim1 * dim2;
        xNVShape.strides[1] = dim2;
        xNVShape.strides[2] = 1;

        maxNVShape.dims[0] = dim0;
        maxNVShape.dims[1] = 1;
        maxNVShape.dims[2] = dim2;
        maxNVShape.strides[0] = dim2;
        maxNVShape.strides[1] = dim2;
        maxNVShape.strides[2] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.dims[2] = dim2;
        yNVShape.strides[0] = dim1 * dim2;
        yNVShape.strides[1] = dim2;
        yNVShape.strides[2] = 1;

        int N = dim0 * dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxExpMinusOp<T>, 3> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, tempptr, maxNVShape, y, yNVShape, SoftmaxExpMinusOp<T>(), N);
    }

    /**calculate sum*/
	if (1 == dim2) {
		/**tail reduce*/
        callTailReduceForward<T, SoftmaxSumOp<T>>(y, tempptr, dim0, dim1);
	} else if (1 == dim0) {
		/**head reduce*/
        callHeadReduceForward<T, SoftmaxSumOp<T>>(y, tempptr, dim1, dim2);
	} else {
		/**middle reduce*/
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;        
        MiddleReduceForward<T, SoftmaxSumOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, tempptr, dim0, dim1, dim2, SoftmaxSumOp<T>(), N);
	}

    /**calculate result*/
    if (1 == dim2) {
        /**tail reduce*/
        NVShape<2> sumNVShape;
        NVShape<2> yNVShape;

        sumNVShape.dims[0] = dim0;
        sumNVShape.dims[1] = 1;
        sumNVShape.strides[0] = 1;
        sumNVShape.strides[1] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.strides[0] = dim1;
        yNVShape.strides[1] = 1;

       int N = dim0 * dim1;
       int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

       BinaryElementWiseForward<T, SoftmaxDivideOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, yNVShape, tempptr, sumNVShape, y, yNVShape, SoftmaxDivideOp<T>(), N);
    } else if (1 == dim0) {
        /**head reduce*/
        NVShape<2> sumNVShape;
        NVShape<2> yNVShape;

        sumNVShape.dims[0] = 1;
        sumNVShape.dims[1] = dim2;
        sumNVShape.strides[0] = dim2;
        sumNVShape.strides[1] = 1;

        yNVShape.dims[0] = dim1;
        yNVShape.dims[1] = dim2;
        yNVShape.strides[0] = dim2;
        yNVShape.strides[1] = 1;

        int N = dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxDivideOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, yNVShape, tempptr, sumNVShape, y, yNVShape, SoftmaxDivideOp<T>(), N);
    } else {
        /**middle reduce*/
        NVShape<3> sumNVShape;
        NVShape<3> yNVShape;

        sumNVShape.dims[0] = dim0;
        sumNVShape.dims[1] = 1;
        sumNVShape.dims[2] = dim2;
        sumNVShape.strides[0] = dim2;
        sumNVShape.strides[1] = dim2;
        sumNVShape.strides[2] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.dims[2] = dim2;
        yNVShape.strides[0] = dim1 * dim2;
        yNVShape.strides[1] = dim2;
        yNVShape.strides[2] = 1;

        int N = dim0 * dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxDivideOp<T>, 3> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, yNVShape, tempptr, sumNVShape, y, yNVShape, SoftmaxDivideOp<T>(), N);
    }

    device->free(tempptr);
}

template <typename T>
void Softmax<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

    auto device = (GPUDevice*)iGradient->device();

    auto x  = inputs[0]->data();
    auto dx = iGradient->data();
	auto y  = output->data();
	auto dy = outputGradient->data();
	
    auto shape = iGradient->shape;
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) shape.dim(i);
        }

        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int) shape.dim(i);
        }
    }

    /**store the temp data*/
    auto dotptr = (T*)device->malloc(sizeof(T) * dim0 * dim2);

    int gridSize  = dim0 * dim2;
    int blockSize = 1024;

    if (blockSize > gridSize) {
        blockSize = prevPowerOf2(gridSize);
    }

    int sharedSize = sizeof(T) * blockSize;

    if (1024 == blockSize) {
        SoftmaxBackwardDotKernel<1024, T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (512 == blockSize) {
        SoftmaxBackwardDotKernel<512,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (256 == blockSize) {
        SoftmaxBackwardDotKernel<256,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (128 == blockSize) {
        SoftmaxBackwardDotKernel<128,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (64 == blockSize) {
        SoftmaxBackwardDotKernel<64,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (32 == blockSize) {
        SoftmaxBackwardDotKernel<32,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (16 == blockSize) {
        SoftmaxBackwardDotKernel<16,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (8 == blockSize) {
        SoftmaxBackwardDotKernel<8,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (4 == blockSize) {
        SoftmaxBackwardDotKernel<4,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (2 == blockSize) {
        SoftmaxBackwardDotKernel<2,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (1 == blockSize) {
        SoftmaxBackwardDotKernel<1,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
	}

    int N = (int)iGradient->shape.size();

    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    SoftmaxBackwardKernel<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(dx, y, dy, dotptr, dim0, dim1, dim2, N);

    device->free(dotptr);
}

DEEP8_DECLARATION_GPU_FUNC(Softmax);

}