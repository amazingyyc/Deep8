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
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {}
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init() {}
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret, const T &cur) {}
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {}
};

template <>
struct SoftmaxFindMaxOp<float> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float commense() {
		return -FLT_MAX;
	}

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float init(float ret, float cur) {
        return ret >= cur ? ret : cur;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float step(float ret1, float ret2) {
        return ret1 >= ret2 ? ret1 : ret2;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float complete(float ret) {
        return ret;
    }
};

template <>
struct SoftmaxFindMaxOp<double> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double commense() {
		return -DBL_MAX;
	}

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double init(double ret, double cur) {
        return ret >= cur ? ret : cur;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double step(double ret1, double ret2) {
        return ret1 >= ret2 ? ret1 : ret2;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double complete(double ret) {
        return ret;
    }
};

#ifdef HAVE_HALF
template <>
struct SoftmaxFindMaxOp<half> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half commense() {
		return -65504.0;
	}

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half init(half ret, half cur) {
        return ret >= cur ? ret : cur;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half step(half ret1, half ret2) {
        return ret1 >= ret2 ? ret1 : ret2;
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half complete(half ret) {
        return ret;
    }
};
#endif

/**
 * Y[i] = exp(X[i] - scalar);
 */
template <typename real>
__global__ void SoftmaxExpMinusScalar(const real *x, const real *scalar, real *y, const int size, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        y[i] = CuMath::cuExp(x[i] - scalar[i / size]);
    }
}

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

/**
 * Y[i] = X[i] / scalar[0];
 */
template <typename real>
__global__ void SoftmaxDivideScalar(real *y, const real *scalar, const int size, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        y[i] = y[i] / scalar[i / size];
    }
}

template <int blockSize, typename real>
__global__ void SoftmaxBackwardDotKernel(const real *y, const real *dy, real *dotPtr, const int batch, const int size) {
    SharedMemory<real> shareMemory;
    real *shared = shareMemory.pointer();

    int threaId = threadIdx.x;
    int blockId = blockIdx.x;

    int i = blockId * size + threaId;
    int j = threaId;

    shared[threaId] = 0;

    while (j < size) {
        shared[threaId] += y[i] * dy[i];

        i += blockSize;
        j += blockSize;
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
		warpSumReduce<blockSize, real>(shared, threaId);
    }

    if (0 == threaId) {
        dotPtr[blockId] = shared[threaId];
    }
}

template <typename real>
__global__ void SoftmaxBackwardKernel(real *dx, const real *y, const real *dy, const real *scalar, const int size, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        dx[i] += (dy[i] - scalar[i / size]) * y[i];
    }
}

template <typename T>
void Softmax<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = (GPUDevice*)output->device();

    auto x = inputs[0]->data();
    auto y = output->data();

    int N      = (int)output->shape.size();
    int batch  = (int)output->shape.batch;
    int size   = N / batch;

    int blockSize = 1024;

    if (size < blockSize) {
        blockSize = prevPowerOf2(size);
    }

    auto maxPtr = (T*)device->malloc(sizeof(T) * batch);
    auto sumPtr = (T*)device->malloc(sizeof(T) * batch);

    /**find max*/
    callTailReduceForward<T, SoftmaxFindMaxOp<T>>(x, maxPtr, batch, size, blockSize);

    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    SoftmaxExpMinusScalar<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(x, maxPtr, y, size, N);

    /***calculate sum*/
    callTailReduceForward<T, SoftmaxSumOp<T>>(y, sumPtr, batch, size, blockSize);

    SoftmaxDivideScalar<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(y, sumPtr, size, N);

    device->free(sumPtr);
    device->free(maxPtr);
}

template <typename T>
void Softmax<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
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
}

DEEP8_DECLARATION_GPU_FUNC(Softmax);

}