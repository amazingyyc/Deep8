#include "basic/GPUBasic.h"
#include "model/GPUDevice.h"
#include "math/GPUMath.h"
#include "math/GPUReduce.h"
#include "math/CrossEntropy.h"

namespace Deep8 {
namespace Math {

template <typename T>
struct CrossEntropyKernelOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
        return ret1 + ret2;
    }
};

template <int blockSize, typename T>
__global__ void CrossEntropyKernel(const T *x, const T *y, T *z, const int batch, const int size) {
    GPUSharedMemory<T> shareMemory;
    T *shared = shareMemory.pointer();

    int blockId = blockIdx.x;
    int threaId = threadIdx.x;

    int i = threaId;

    shared[threaId] = 0;

    while (i < size) {
        shared[threaId] += (-y[i] * cudaLog(x[i]));

        i += blockSize;
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
        Warp32ReduceStep<T, CrossEntropyKernelOp<T>, blockSize>(shared, threaId, CrossEntropyKernelOp<T>());
    }

    if (0 == threaId) {
        z[blockId] = shared[threaId];
    }
}

template <typename T>
void CrossEntropyGPUImpl(GPUDevice *device, 
                         const T *x, 
                         const Shape &xshape, 
                         const T *y, 
                         const Shape &yshape, 
                         T *z, 
                         const Shape &zshape) {
    auto batch = (int) xshape.batch;
    auto size  = (int) xshape.batchSize();

    int gridSize  = batch;
    int blockSize = DEEP8_GPU_BLOCK_SIZE;

    if (blockSize > size) {
        blockSize = prevPowerOf2(size);
    }

    int sharedSize = sizeof(T) * blockSize;

    if (1024 == blockSize) {
        CrossEntropyKernel<1024, T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (512 == blockSize) {
        CrossEntropyKernel<512,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (256 == blockSize) {
        CrossEntropyKernel<256,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (128 == blockSize) {
        CrossEntropyKernel<128,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (64 == blockSize) {
        CrossEntropyKernel<64,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (32 == blockSize) {
        CrossEntropyKernel<32,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (16 == blockSize) {
        CrossEntropyKernel<16,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (8 == blockSize) {
        CrossEntropyKernel<8,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (4 == blockSize) {
        CrossEntropyKernel<4,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (2 == blockSize) {
        CrossEntropyKernel<2,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else if (1 == blockSize) {
        CrossEntropyKernel<1,  T> << <gridSize, blockSize, sharedSize >> > (x, y, z, batch, size);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
	}
}

void CrossEntropyGPU(const Tensor &x, const Tensor &y, Tensor &z) {
    auto device = (GPUDevice*) x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        CrossEntropyGPUImpl<float>(device, 
                                x.data<float>(), 
                                x.shape, 
                                y.data<float>(), 
                                y.shape, 
                                z.data<float>(), 
                                z.shape);
        break;
    case DType::Float64:
        CrossEntropyGPUImpl<double>(device, 
                                x.data<double>(), 
                                x.shape, 
                                y.data<double>(), 
                                y.shape, 
                                z.data<double>(), 
                                z.shape);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CrossEntropyGPUImpl<half>(device, 
                                x.data<half>(), 
                                x.shape, 
                                y.data<half>(), 
                                y.shape, 
                                z.data<half>(), 
                                z.shape);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

template <typename T>
__global__ void CrossEntropyGradXKernel(const T *x, T *dx, const T *y, const T *dz, const int batch, const int size, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
        int row = i / size;

		dx[i] -= dz[row] * y[i] / x[i];
	}
}

template <typename T>
void CrossEntropyGradXGPUImpl(GPUDevice *device, 
                              const T *x, 
                              T *dx, 
                              const Shape &xshape, 
                              const T *y, 
                              const Shape &yshape, 
                              const T *z, 
                              const T *dz, 
                              const Shape &zshape) {
    auto batch = (int) xshape.batch;
    auto size  = (int) xshape.batchSize();

    int N = (int) xshape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;
    
    CrossEntropyGradXKernel<T><<<grideSize, blockSize >>>(x, dx, y, dz, batch, size, N);
}

void CrossEntropyGradXGPU(const Tensor &x, Tensor &dx, const Tensor &y, const Tensor &z, const Tensor &dz) {
    auto device = (GPUDevice*) x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        CrossEntropyGradXGPUImpl<float>(device, 
            x.data<float>(), 
            dx.data<float>(), 
            x.shape,
            y.data<float>(),
            y.shape,
            z.data<float>(),
            dz.data<float>(),
            z.shape);
        break;
    case DType::Float64:
        CrossEntropyGradXGPUImpl<double>(device, 
            x.data<double>(), 
            dx.data<double>(), 
            x.shape,
            y.data<double>(),
            y.shape,
            z.data<double>(),
            dz.data<double>(),
            z.shape);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CrossEntropyGradXGPUImpl<half>(device, 
            x.data<half>(), 
            dx.data<half>(), 
            x.shape,
            y.data<half>(),
            y.shape,
            z.data<half>(),
            dz.data<half>(),
            z.shape);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}


template <typename T>
__global__ void CrossEntropyGradYKernel(const T *x, const T *y, T *dy, const T *dz, const int batch, const int size, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
        int row = i / size;

		dy[i] -= dz[row] * cudaLog(x[i]);
	}
}

template <typename T>
void CrossEntropyGradYGPUImpl(GPUDevice *device, 
                              const T *x, 
                              const Shape &xshape, 
                              const T *y, 
                              T *dy,
                              const Shape &yshape, 
                              const T *z, 
                              const T *dz, 
                              const Shape &zshape) {
    auto batch = (int) yshape.batch;
    auto size  = (int) yshape.batchSize();

    int N = (int) yshape.size();

    int blockSize = DEEP8_GPU_BLOCK_SIZE;
    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    CrossEntropyGradYKernel<T><<<grideSize, blockSize >>>(x, y, dy, dz, batch, size, N);
}

void CrossEntropyGradYGPU(const Tensor &x, const Tensor &y, Tensor &dy, const Tensor &z, const Tensor &dz) {
    auto device = (GPUDevice*) x.device();

    switch (x.elementType.id) {
    case DType::Float32:
        CrossEntropyGradYGPUImpl<float>(device, 
            x.data<float>(), 
            x.shape,
            y.data<float>(),
            dy.data<float>(),
            y.shape,
            z.data<float>(),
            dz.data<float>(),
            z.shape);
        break;
    case DType::Float64:
        CrossEntropyGradYGPUImpl<double>(device, 
            x.data<double>(), 
            x.shape,
            y.data<double>(),
            dy.data<double>(),
            y.shape,
            z.data<double>(),
            dz.data<double>(),
            z.shape);
        break;

#ifdef HAVE_HALF
    case DType::Float16:
        CrossEntropyGradYGPUImpl<half>(device, 
            x.data<half>(), 
            x.shape,
            y.data<half>(),
            dy.data<half>(),
            y.shape,
            z.data<half>(),
            dz.data<half>(),
            z.shape);
        break;
#endif

    default:
        DEEP8_RUNTIME_ERROR("type " << x.elementType.name << " is not support");
        break;
    }
}

}
}