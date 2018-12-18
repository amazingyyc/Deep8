#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
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
 * Y[i] = X[i] / scalar[0];
 */
// template <typename real>
// __global__ void SoftmaxDivideScalar(real *y, const real *scalar, const int size, const int N) {
//     int start = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;

//     for (int i = start; i < N; i += stride) {
//         y[i] = y[i] / scalar[i / size];
//     }
// }

// template <int blockSize, typename real>
// __global__ void SoftmaxBackwardDotKernel(const real *y, const real *dy, real *dotPtr, const int batch, const int size) {
//     SharedMemory<real> shareMemory;
//     real *shared = shareMemory.pointer();

//     int threaId = threadIdx.x;
//     int blockId = blockIdx.x;

//     int i = blockId * size + threaId;
//     int j = threaId;

//     shared[threaId] = 0;

//     while (j < size) {
//         shared[threaId] += y[i] * dy[i];

//         i += blockSize;
//         j += blockSize;
//     }

//     __syncthreads();

//     if (blockSize >= 1024) {
//         if (threaId < 512) {
//             shared[threaId] += shared[threaId + 512];
//         }

//         __syncthreads();
//     }

//     if (blockSize >= 512) {
//         if (threaId < 256) {
//             shared[threaId] += shared[threaId + 256];
//         }

//         __syncthreads();
//     }

//     if (blockSize >= 256) {
//         if (threaId < 128) {
//             shared[threaId] += shared[threaId + 128];
//         }

//         __syncthreads();
//     }

//     if (blockSize >= 128) {
//         if (threaId < 64) {
//             shared[threaId] += shared[threaId + 64];
//         }

//         __syncthreads();
//     }

//     if (threaId < 32) {
// 		warpSumReduce<blockSize, real>(shared, threaId);
//     }

//     if (0 == threaId) {
//         dotPtr[blockId] = shared[threaId];
//     }
// }

// template <typename real>
// __global__ void SoftmaxBackwardKernel(real *dx, const real *y, const real *dy, const real *scalar, const int size, const int N) {
//     int start  = blockIdx.x * blockDim.x + threadIdx.x;
//     int stride = blockDim.x * gridDim.x;

//     for (int i = start; i < N; i += stride) {
//         dx[i] += (dy[i] - scalar[i / size]) * y[i];
//     }
// }

template <typename T>
void Softmax<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = (GPUDevice*)output->device();

	auto xshape = inputs[0]->shape;

	auto x = inputs[0]->data();
	auto y = output->data();

	int dim0 = xshape.batch;
	int dim1 = xshape.dim(axis);
	int dim2 = 1;

	for (int i = 0; i < axis; ++i) {
		dim0 *= xshape.dim(i);
	}

	for (int i = axis + 1; i < xshape.nDims; ++i) {
		dim2 *= xshape.dim(i);
	}

	auto maxptr = (T*)device->malloc(sizeof(T) * dim0 * dim2);
	auto sumptr = (T*)device->malloc(sizeof(T) * dim0 * dim2);

	/**find max value*/
	if (1 == dim2) {
		/**tail reduce*/
        callTailReduceForward<T, SoftmaxFindMaxOp<T>>(x, maxptr, dim0, dim1);
	} else if (1 == dim0) {
		/**head reduce*/
        callHeadReduceForward<T, SoftmaxFindMaxOp<T>>(x, maxptr, dim1, dim2);
	} else {
		/**middle reduce*/
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;        
        MiddleReduceForward<T, SoftmaxFindMaxOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, maxptr, dim0, dim1, dim2, SoftmaxFindMaxOp<T>(), N);
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
        maxNVShape.stride[0] = 1;
        maxNVShape.stride[1] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.strides[0] = dim1;
        yNVShape.strides[1] = 1;

       int N = dim0 * dim1;
       int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

       BinaryElementWiseForward<T, SoftmaxExpMinusOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, maxptr, maxNVShape, y, yNVShape, SoftmaxExpMinusOp<T>(), N);
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
        maxNVShape.stride[0] = dim2;
        maxNVShape.stride[1] = 1;

        yNVShape.dims[0] = dim1;
        yNVShape.dims[1] = dim2;
        yNVShape.strides[0] = dim2;
        yNVShape.strides[1] = 1;

        int N = dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxExpMinusOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, maxptr, maxNVShape, y, yNVShape, SoftmaxExpMinusOp<T>(), N);
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
        maxNVShape.stride[0] = dim2;
        maxNVShape.stride[1] = dim2;
        maxNVShape.stride[2] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.dims[2] = dim2;
        yNVShape.strides[0] = dim1 * dim2;
        yNVShape.strides[1] = dim2;
        yNVShape.strides[2] = 1;

        int N = dim0 * dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxExpMinusOp<T>, 3> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(x, xNVShape, maxptr, maxNVShape, y, yNVShape, SoftmaxExpMinusOp<T>(), N);
    }

    /**calculate sum*/
	if (1 == dim2) {
		/**tail reduce*/
        callTailReduceForward<T, SoftmaxSumOp<T>>(y, sumptr, dim0, dim1);
	} else if (1 == dim0) {
		/**head reduce*/
        callHeadReduceForward<T, SoftmaxSumOp<T>>(y, sumptr, dim1, dim2);
	} else {
		/**middle reduce*/
        int N = dim0 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;        
        MiddleReduceForward<T, SoftmaxSumOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, sumptr, dim0, dim1, dim2, SoftmaxSumOp<T>(), N);
	}

    /**calculate result*/
    /**y = exp(x - max)*/
    if (1 == dim2) {
        /**tail reduce*/
        NVShape<2> sumNVShape;
        NVShape<2> yNVShape;

        sumNVShape.dims[0] = dim0;
        sumNVShape.dims[1] = 1;
        sumNVShape.stride[0] = 1;
        sumNVShape.stride[1] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.strides[0] = dim1;
        yNVShape.strides[1] = 1;

       int N = dim0 * dim1;
       int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

       BinaryElementWiseForward<T, SoftmaxDivideOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, yNVShape, sumptr, sumNVShape, y, yNVShape, SoftmaxDivideOp<T>(), N);
    } else if (1 == dim0) {
        /**head reduce*/
        NVShape<2> sumNVShape;
        NVShape<2> yNVShape;

        sumNVShape.dims[0] = 1;
        sumNVShape.dims[1] = dim2;
        sumNVShape.stride[0] = dim2;
        sumNVShape.stride[1] = 1;

        yNVShape.dims[0] = dim1;
        yNVShape.dims[1] = dim2;
        yNVShape.strides[0] = dim2;
        yNVShape.strides[1] = 1;

        int N = dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxDivideOp<T>, 2> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, yNVShape, sumptr, sumNVShape, y, yNVShape, SoftmaxDivideOp<T>(), N);
    } else {
        /**middle reduce*/
        NVShape<3> sumNVShape;
        NVShape<3> yNVShape;

        sumNVShape.dims[0] = dim0;
        sumNVShape.dims[1] = 1;
        sumNVShape.dims[2] = dim2;
        sumNVShape.stride[0] = dim2;
        sumNVShape.stride[1] = dim2;
        sumNVShape.stride[2] = 1;

        yNVShape.dims[0] = dim0;
        yNVShape.dims[1] = dim1;
        yNVShape.dims[2] = dim2;
        yNVShape.strides[0] = dim1 * dim2;
        yNVShape.strides[1] = dim2;
        yNVShape.strides[2] = 1;

        int N = dim0 * dim1 * dim2;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseForward<T, SoftmaxDivideOp<T>, 3> <<<grideSize, DEEP8_GPU_BLOCK_SIZE>>>(y, yNVShape, sumptr, sumNVShape, y, yNVShape, SoftmaxDivideOp<T>(), N);
    }

    device->free(sumptr);
    device->free(maxptr);

	// /*
    // auto device = (GPUDevice*)output->device();

    // auto x = inputs[0]->data();
    // auto y = output->data();

    // int N      = (int)output->shape.size();
    // int batch  = (int)output->shape.batch;
    // int size   = N / batch;

    // int blockSize = 1024;

    // if (size < blockSize) {
    //     blockSize = prevPowerOf2(size);
    // }

    // auto maxPtr = (T*)device->malloc(sizeof(T) * batch);
    // auto sumPtr = (T*)device->malloc(sizeof(T) * batch);
	// */


    // /**find max*/

	// /*
    // callTailReduceForward<T, SoftmaxFindMaxOp<T>>(x, maxPtr, batch, size, blockSize);

    // int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    // SoftmaxExpMinusScalar<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(x, maxPtr, y, size, N);
	// */

    // /***calculate sum*/
    // /*
	// callTailReduceForward<T, SoftmaxSumOp<T>>(y, sumPtr, batch, size, blockSize);

    // SoftmaxDivideScalar<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(y, sumPtr, size, N);

    // device->free(sumPtr);
    // device->free(maxPtr);
	// */
}

template <typename T>
void Softmax<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	/*
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

	auto device = (GPUDevice*)iGradient->device();

	auto dx = iGradient->data();
	auto y  = output->data();
	auto dy = outputGradient->data();

    int N      = (int)iGradient->shape.size();
    int batch  = (int)iGradient->shape.batch;
    int size   = N / batch;

    int blockSize = 1024;

    if (size < blockSize) {
        blockSize = prevPowerOf2(size);
    }

    int sharedSize = sizeof(T) * blockSize;

    auto dotPtr = (T*)device->malloc(sizeof(T) * batch);

    if (1024 == blockSize) {
        SoftmaxBackwardDotKernel<1024, T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (512 == blockSize) {
        SoftmaxBackwardDotKernel<512,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (256 == blockSize) {
        SoftmaxBackwardDotKernel<256,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (128 == blockSize) {
        SoftmaxBackwardDotKernel<128,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (64 == blockSize) {
        SoftmaxBackwardDotKernel<64,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (32 == blockSize) {
        SoftmaxBackwardDotKernel<32,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (16 == blockSize) {
        SoftmaxBackwardDotKernel<16,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (8 == blockSize) {
        SoftmaxBackwardDotKernel<8,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (4 == blockSize) {
        SoftmaxBackwardDotKernel<4,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (2 == blockSize) {
        SoftmaxBackwardDotKernel<2,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else if (1 == blockSize) {
        SoftmaxBackwardDotKernel<1,  T> << <batch, blockSize, sharedSize >> > (y, dy, dotPtr, batch, size);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
	}

    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    SoftmaxBackwardKernel<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(dx, y, dy, dotPtr, size, N);

    device->free(dotPtr);
	*/
}

DEEP8_DECLARATION_GPU_FUNC(Softmax);

}