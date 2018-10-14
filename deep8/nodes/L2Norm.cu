#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "L2Norm.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <int blockSize, typename real>
__global__ void L2NormForwardKernel(const real *x, real *y, const int batch, const int size) {
    SharedMemory<real> shareMemory;
    real *shared = shareMemory.pointer();

    int threaId = threadIdx.x;
    int blockId = blockIdx.x;

    int i = blockId * size + threaId;
    int j = threaId;

    shared[threaId] = 0;

    while (j < size) {
        shared[threaId] += x[i] * x[i];

        j += blockSize;
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
        warpSumReduce<blockSize, real>(shared, threaId);
    }

    if (0 == threaId) {
        y[blockId] = cuSqrt(shared[threaId]);
    }
}

template <typename real>
__global__ void L2NormBackwardKernel(const real *x, real *xGrad, const real *y, const real *yGrad, const int size, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        int yI = i / size;

        xGrad[i] += x[i] * yGrad[yI] / y[yI];
    }
}

template <typename T>
void L2Norm<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto shape = inputs[0]->shape;
    int batch = (int)shape.batch();
    int size  = (int)shape.size() / batch;

    auto x = inputs[0]->data();
    auto y = output->data();

    int blockSize = 1024;

    if (size < blockSize) {
        blockSize = prevPowerOf2(size);
    }

    int sharedSize = sizeof(T) * blockSize;

    if (1024 == blockSize) {
        L2NormForwardKernel<1024, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (512 == blockSize) {
        L2NormForwardKernel<512, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (256 == blockSize) {
        L2NormForwardKernel<256, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (128 == blockSize) {
        L2NormForwardKernel<128, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (64 == blockSize) {
        L2NormForwardKernel<64, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (32 == blockSize) {
        L2NormForwardKernel<32, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (16 == blockSize) {
        L2NormForwardKernel<16, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (8 == blockSize) {
        L2NormForwardKernel<8, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (4 == blockSize) {
        L2NormForwardKernel<4, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (2 == blockSize) {
        L2NormForwardKernel<2, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else if (1 == blockSize) {
        L2NormForwardKernel<1, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
    }
}

template <typename T>
void L2Norm<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

    auto shape = iGradient->shape;
    int batch  = (int)shape.batch();
    int size   = (int)shape.size() / batch;

    auto x  = inputs[0]->data();
    auto dx = iGradient->data();
    auto y  = output->data();
    auto dy = outputGradient->data();

    int N = batch * size;

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, L2NormBackwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    L2NormBackwardKernel<T> << <grideSize, blockSize >> > (x, dx, y, dy, size, N);
}

#ifdef HAVE_HALF
template <>
void L2Norm<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs, const Tensor<half> *output, const Tensor<half> *outputGradient, size_t index, Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

    auto shape = iGradient->shape;
    int batch  = (int)shape.batch();
    int size   = (int)shape.size() / batch;

    auto x  = inputs[0]->data();
    auto dx = iGradient->data();
    auto y  = output->data();
    auto dy = outputGradient->data();

    int N = batch * size;

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    L2NormBackwardKernel<half> << <grideSize, blockSize >> > (x, dx, y, dy, size, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(L2Norm);

#endif

}