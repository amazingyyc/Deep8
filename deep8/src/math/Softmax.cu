#include "math/Softmax.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct SoftmaxMaxKernelOp {
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

template <typename T>
struct SoftmaxExpMinusKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y) {
		return CuMath::cuExp(x - y);
	}
};

template <typename T>
struct SoftmaxSumKernelOp {
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

template <typename T>
struct SoftmaxDivideKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T operator()(const T &x, const T &y) {
		return x / y;
	}
};

template <typename T>
void SoftmaxGPUImpl(GPUDevice *device, 
                    const T *x, 
                    const Shape &xshape, 
                    T *y, 
                    const Shape &yshape, 
                    int axis, 
                    T *ptr) {
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) xshape.dim(i);
        }

        for (int i = axis + 1; i < xshape.nDims; ++i) {
            dim2 *= (int) xshape.dim(i);
        }
    }

    /**find max value*/
	if (1 == dim2) {
		/**tail reduce*/
        CallTailReduceKernel<T, SoftmaxMaxKernelOp<T>>(x, ptr, dim0, dim1, SoftmaxMaxKernelOp<T>());
	} else if (1 == dim0) {
		/**head reduce*/
        CallHeadReduceKernel<T, SoftmaxMaxKernelOp<T>>(x, ptr, dim1, dim2, SoftmaxMaxKernelOp<T>());
	} else {
		/**middle reduce*/
        int N = dim0 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;        

        MiddleReduceKernel<T, SoftmaxMaxKernelOp<T>> <<<grideSize, blockSize>>>(x, ptr, dim0, dim1, dim2, SoftmaxMaxKernelOp<T>(), N);
	}

    {
        /**y = exp(x - max)*/
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
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseKernel<T, SoftmaxExpMinusKernelOp<T>, 3> <<<grideSize, blockSize>>>(x, xNVShape, ptr, maxNVShape, y, yNVShape, SoftmaxExpMinusKernelOp<T>(), N);
    }

    /**calculate sum*/
	if (1 == dim2) {
		/**tail reduce*/
        CallTailReduceKernel<T, SoftmaxSumKernelOp<T>>(y, ptr, dim0, dim1, SoftmaxSumKernelOp<T>());
	} else if (1 == dim0) {
		/**head reduce*/
        CallHeadReduceKernel<T, SoftmaxSumKernelOp<T>>(y, ptr, dim1, dim2, SoftmaxSumKernelOp<T>());
	} else {
		/**middle reduce*/
        int N = dim0 * dim2;
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        MiddleReduceKernel<T, SoftmaxSumKernelOp<T>> <<<grideSize, blockSize>>>(y, ptr, dim0, dim1, dim2, SoftmaxSumKernelOp<T>(), N);
	}

    {
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
        int blockSize = DEEP8_GPU_BLOCK_SIZE;
        int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

        BinaryElementWiseKernel<T, SoftmaxDivideKernelOp<T>, 3> <<<grideSize, blockSize>>>(y, yNVShape, ptr, sumNVShape, y, yNVShape, SoftmaxDivideKernelOp<T>(), N);
    }
}

void SoftmaxGPU(const Tensor &x, Tensor &y, int axis, void *ptr) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        SoftmaxGPUImpl<float>(device, x.data<float>(), x.shape, y.data<float>(), y.shape, axis, (float*)ptr);
        break;
    case DType::Float64:
        SoftmaxGPUImpl<double>(device, x.data<double>(), x.shape, y.data<double>(), y.shape, axis, (double*)ptr);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        SoftmaxGPUImpl<half>(device, x.data<half>(), x.shape, y.data<half>(), y.shape, axis, (half*)ptr);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}

/**
 * support the x/y dimension is [dim0, dim1, dim2]
 * the dotptr dimension is [dim0, 1, dim2]
 * dotptr[i, 0, j] = sum(y[i, l, j] * dy[i, l, j]), l = (0..dim1)
 */
template <typename T>
struct SoftmaxGradDotKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}
};

template <int blockSize, typename T>
__global__ void SoftmaxGradDotKernel(const T *y, const T *dy, T *dotptr, const int dim0, const int dim1, const int dim2) {
    GPUSharedMemory<T> shareMemory;
    T *shared = shareMemory.pointer();

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
        Warp32ReduceStep<T, SoftmaxGradDotKernelOp<T>, blockSize>(shared, threaId, SoftmaxGradDotKernelOp<T>());
    }

    if (0 == threaId) {
        dotptr[blockId] = shared[threaId];
    }
}

template <typename T>
__global__ void SoftmaxGradKernel(real *dx, const T *y, const T *dy, const T *dotptr, const int dim0, const int dim1, const int dim2, const int N) {
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
void SoftmaxGradGPUImpl(GPUDevice *device, 
                        const T *x, 
                        T *dx, 
                        const Shape &xshape, 
                        const T *y, 
                        const T *dy, 
                        const Shape &yshape, 
                        int axis, 
                        T *dotptr) {
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) xshape.batch;
        dim1 = (int) xshape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) xshape.dim(i);
        }

        for (int i = axis + 1; i < xshape.nDims; ++i) {
            dim2 *= (int) xshape.dim(i);
        }
    }

    int gridSize  = dim0 * dim2;
    int blockSize = 1024;

    if (blockSize > dim1) {
        blockSize = prevPowerOf2(dim1);
    }

    if (1024 == blockSize) {
        SoftmaxGradDotKernel<1024, T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (512 == blockSize) {
        SoftmaxGradDotKernel<512,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (256 == blockSize) {
        SoftmaxGradDotKernel<256,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (128 == blockSize) {
        SoftmaxGradDotKernel<128,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (64 == blockSize) {
        SoftmaxGradDotKernel<64,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (32 == blockSize) {
        SoftmaxGradDotKernel<32,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (16 == blockSize) {
        SoftmaxGradDotKernel<16,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (8 == blockSize) {
        SoftmaxGradDotKernel<8,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (4 == blockSize) {
        SoftmaxGradDotKernel<4,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (2 == blockSize) {
        SoftmaxGradDotKernel<2,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else if (1 == blockSize) {
        SoftmaxGradDotKernel<1,  T> << <gridSize, blockSize, sharedSize >> > (y, dy, dotptr, dim0, dim1, dim2);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
	}

    int N = (int)xshape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    SoftmaxGradKernel<T><<<grideSize, blockSize >>>(dx, y, dy, dotptr, dim0, dim1, dim2, N);
}

void SoftmaxGradGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &dy, int axis, void *ptr) {
    auto device = x.device();

    switch (x.type.id) {
    case DType::Float32:
        SoftmaxGradGPUImpl<float>(device,
                                 x.data<float>(),
                                 dx.data<float>(),
                                 x.shape,
                                 y.data<float>(),
                                 dy.data<float>(),
                                 y.shape,
                                 axis,
                                 (float*)ptr);
        break;
    case DType::Float64:
        SoftmaxGradGPUImpl<double>(device,
                                 x.data<double>(),
                                 dx.data<double>(),
                                 x.shape,
                                 y.data<double>(),
                                 dy.data<double>(),
                                 y.shape,
                                 axis,
                                 (double*)ptr);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        SoftmaxGradGPUImpl<half>(device,
                                 x.data<half>(),
                                 dx.data<half>(),
                                 x.shape,
                                 y.data<half>(),
                                 dy.data<half>(),
                                 y.shape,
                                 axis,
                                 (half*)ptr);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.type.name << " is not support");
        break;
    }
}


}
}