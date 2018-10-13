#include "Softmax.h"

namespace Deep8 {

#ifdef HAVE_CUDA

/**
 * find the max value and put it in y
 */
template <int blockSize, typename real>
__global__ void SoftmaxForwardFindMaxKernel(const real *x, real *y, const int batch, const int size) {
    SharedMemory<real> shareMemory;
    real *shared = shareMemory.pointer();

    int threaId = threadIdx.x;
    int blockId = blockIdx.x;

    int i = blockId * size + threaId;
    int j = threaId;

    shared[threaId] =  x[i];

    while (j < size) {
        shared[threaId] = cuMax(shared[threaId], x[i]);

        i += blockSize;
        j += blockSize;
    }

    __syncthreads();

    if (blockSize >= 1024) {
        if (threaId < 512) {
            shared[threaId] = cuMax(shared[threaId], shared[threaId + 512]);
        }

        __syncthreads();
    }

    if (blockSize >= 512) {
        if (threaId < 256) {
            shared[threaId] = cuMax(shared[threaId], shared[threaId + 256]);
        }

        __syncthreads();
    }

    if (blockSize >= 256) {
        if (threaId < 128) {
            shared[threaId] = cuMax(shared[threaId], shared[threaId + 128]);
        }

        __syncthreads();
    }

    if (blockSize >= 128) {
        if (threaId < 64) {
            shared[threaId] = cuMax(shared[threaId], shared[threaId + 64]);
        }

        __syncthreads();
    }

    if (threaId < 32) {
        warpMaxReduce<blockSize, real>(shared, threaId);
    }

    if (0 == threaId) {
        y[blockId] = shared[threaId];
    }
}

/**
 * Y[i] = exp(X[i] - scalar);
 */
template <typename real>
__global__ void SoftmaxForwardExpMinusScalar(const real *x, const real *scalar, real *y, const int size, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        y[i] = cuExp(x[i] - scalar[i / size]);
    }
}

template <int blockSize, typename real>
__global__ void SoftmaxForwardSumKernel(const real *x, real *y, const int batch, const int size) {
    SharedMemory<real> shareMemory;
    real *shared = shareMemory.pointer();

    int threaId = threadIdx.x;
    int blockId = blockIdx.x;

    int i = blockId * size + threaId;
    int j = threaId;

    shared[threaId] = 0;

    while (j < size) {
        shared[threaId] += x[i];

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
        y[blockId] = shared[threaId];
    }
}

/**
 * Y[i] = X[i] / scalar[0];
 */
template <typename real>
__global__ void SoftmaxForwardDivideScalar(real *Y, const real *scalar, const int size, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = Y[i] / scalar[i / size];
    }
}

template <int blockSize, typename real>
__global__ void SoftmaxBackwardDotKernel(const real *y, const real *yGrad, real *dotPtr, const int batch, const int size) {
    SharedMemory<real> shareMemory;
    real *shared = shareMemory.pointer();

    int threaId = threadIdx.x;
    int blockId = blockIdx.x;

    int i = blockId * size + threaId;
    int j = threaId;

    shared[threaId] = 0;

    while (j < size) {
        shared[threaId] += y[i] * yGrad[i];

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
__global__ void SoftmaxBackwardKernel(real *xGrad, const real *Y, const real *yGrad, const real *scalar, const int size, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += (yGrad[i] - scalar[i / size]) * Y[i];
    }
}

template <typename T>
void Softmax<T>::forwardGPUImpl(Device *device, const T *x, T *y, Shape &shape) {
    int N      = (int)shape.size();
    int batch  = (int)shape.batch();
    int size   = N / batch;

    int blockSize       = 1024;
    int maxBlockSize    = 1024;
    int divideBlockSize = 1024;

    int grideSize;

    if (size < blockSize) {
        blockSize = prevPowerOf2(size);
    }

    int sharedSize = sizeof(T) * blockSize;

    auto maxPtr = (T*)device->malloc(sizeof(T) * batch);
    auto sumPtr = (T*)device->malloc(sizeof(T) * batch);

    if (1024 == blockSize) {
        SoftmaxForwardFindMaxKernel<1024, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (512 == blockSize) {
        SoftmaxForwardFindMaxKernel<512, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (256 == blockSize) {
        SoftmaxForwardFindMaxKernel<256, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (128 == blockSize) {
        SoftmaxForwardFindMaxKernel<128, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (64 == blockSize) {
        SoftmaxForwardFindMaxKernel<64, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (32 == blockSize) {
        SoftmaxForwardFindMaxKernel<32, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (16 == blockSize) {
        SoftmaxForwardFindMaxKernel<16, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (8 == blockSize) {
        SoftmaxForwardFindMaxKernel<8, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (4 == blockSize) {
        SoftmaxForwardFindMaxKernel<4, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (2 == blockSize) {
        SoftmaxForwardFindMaxKernel<2, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else if (1 == blockSize) {
        SoftmaxForwardFindMaxKernel<1, T> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
    }

    grideSize = (N + maxBlockSize - 1) / maxBlockSize;

    SoftmaxForwardExpMinusScalar<T><<<grideSize, maxBlockSize >>>(x, maxPtr, y, size, N);

    if (1024 == blockSize) {
        SoftmaxForwardSumKernel<1024, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (512 == blockSize) {
        SoftmaxForwardSumKernel<512, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (256 == blockSize) {
        SoftmaxForwardSumKernel<256, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (128 == blockSize) {
        SoftmaxForwardSumKernel<128, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (64 == blockSize) {
        SoftmaxForwardSumKernel<64, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (32 == blockSize) {
        SoftmaxForwardSumKernel<32, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (16 == blockSize) {
        SoftmaxForwardSumKernel<16, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (8 == blockSize) {
        SoftmaxForwardSumKernel<8, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (4 == blockSize) {
        SoftmaxForwardSumKernel<4, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (2 == blockSize) {
        SoftmaxForwardSumKernel<2, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else if (1 == blockSize) {
        SoftmaxForwardSumKernel<1, T> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
    }

    grideSize = (N + divideBlockSize - 1) / divideBlockSize;

    SoftmaxForwardDivideScalar<T><<<grideSize, divideBlockSize >>>(y, sumPtr, size, N);

    device->free(sumPtr);
    device->free(maxPtr);
}

template <typename T>
void Softmax<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    forwardGPUImpl(output->device(), inputs[0]->data(), output->data(), output->shape);
}

template <typename T>
void Softmax<T>::backwardGPUImpl(Device *device, T *xGrad, const T *y, const T *yGrad, Shape &shape) {
    int N      = (int)shape.size();
    int batch  = (int)shape.batch();
    int size   = N / batch;

    int blockSize     = 1024;
    int backBlockSize = 1024;

    int grideSize;

    if (size < blockSize) {
        blockSize = prevPowerOf2(size);
    }

    int sharedSize = sizeof(T) * blockSize;

    auto dotPtr = (T*)device->malloc(sizeof(T) * batch);

    if (1024 == blockSize) {
        SoftmaxBackwardDotKernel<1024, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (512 == blockSize) {
        SoftmaxBackwardDotKernel<512, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (256 == blockSize) {
        SoftmaxBackwardDotKernel<256, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (128 == blockSize) {
        SoftmaxBackwardDotKernel<128, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (64 == blockSize) {
        SoftmaxBackwardDotKernel<64, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (32 == blockSize) {
        SoftmaxBackwardDotKernel<32, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (16 == blockSize) {
        SoftmaxBackwardDotKernel<16, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (8 == blockSize) {
        SoftmaxBackwardDotKernel<8, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (4 == blockSize) {
        SoftmaxBackwardDotKernel<4, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (2 == blockSize) {
        SoftmaxBackwardDotKernel<2, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else if (1 == blockSize) {
        SoftmaxBackwardDotKernel<1, T> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
    }

    grideSize = (N + backBlockSize - 1) / backBlockSize;

    SoftmaxBackwardKernel<T><<<grideSize, backBlockSize >>>(xGrad, y, yGrad, dotPtr, size, N);

    device->free(dotPtr);
}

template <typename T>
void Softmax<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

    backwardGPUImpl(iGradient->device(), iGradient->data(), output->data(), outputGradient->data(), iGradient->shape);
}

DEEP8_DECLARATION_GPU_FUNC(Softmax);

#endif

}