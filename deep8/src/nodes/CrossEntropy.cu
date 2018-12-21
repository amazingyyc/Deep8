#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.h"
#include "CrossEntropy.h"

namespace Deep8 {

template <typename T>
struct CrossEntropyForwardOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
        return ret1 + ret2;
    }
};

template <int blockSize, typename real>
__global__ void CrossEntropyForwardKernel(const real *x, const real *y, real *z, const real scale, const int N) {
    SharedMemory<real> shareMemory;
    real *shared = shareMemory.pointer();

    int threaId = threadIdx.x;
    int i = threaId;

    shared[threaId] = 0;

    while (i < N) {
        ret += y[i] * CuMath::cuLog(x[i]);

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
 		warp32ReduceStep<real, CrossEntropyForwardOp<real>, blockSize>(shared, threaId, CrossEntropyForwardOp<real>());
     }

     if (0 == threaId) {
         z[0] = shared[threaId] * scale;
     }
}

template <typename real>
__global__ void CrossEntropyBackwardXKernel(const real *x, real *dx, const real *y, const real *dz, const real scale, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		dx[i] += scale * dz[0] * y[i] / x[i];
	}
}

template <typename real>
__global__ void CrossEntropyBackwardYKernel(const real *x, const real *y, real *dy, const real *dz, const real scale, const int N) {
    int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		dy[i] += scale * dz[0] * CuMath::cuLog(x[i]);
	}
}

template <typename T>
void CrossEntropy<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto batch = inputs[0]->shape.batch;
    auto scale = -1 / T(batch);

    auto x = inputs[0]->data();
    auto y = inputs[1]->data();
    auto z = output->data();

    int N = (int) inputs[0]->size();
    
    int blockSize = 1024;

    if (blockSize > N) {
        blockSize = prevPowerOf2(N);
    }

    int sharedSize = sizeof(T) * blockSize;

    if (1024 == blockSize) {
        CrossEntropyForwardKernel<1024, T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (512 == blockSize) {
        CrossEntropyForwardKernel<512,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (256 == blockSize) {
        CrossEntropyForwardKernel<256,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (128 == blockSize) {
        CrossEntropyForwardKernel<128,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (64 == blockSize) {
        CrossEntropyForwardKernel<64,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (32 == blockSize) {
        CrossEntropyForwardKernel<32,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (16 == blockSize) {
        CrossEntropyForwardKernel<16,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (8 == blockSize) {
        CrossEntropyForwardKernel<8,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (4 == blockSize) {
        CrossEntropyForwardKernel<4,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (2 == blockSize) {
        CrossEntropyForwardKernel<2,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else if (1 == blockSize) {
        CrossEntropyForwardKernel<1,  T> << <1, blockSize, sharedSize >> > (x, y, z, scale, N);
    } else {
        DEEP8_RUNTIME_ERROR("the block size is error");
	}
}

template <typename T>
void CrossEntropy<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
    auto batch = iGradient->shape.batch;
    auto scale = -T(1) / T(batch);

    int N = (int)iGradient->shape.size();

    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

    if (0 == index) {
        CrossEntropyBackwardXKernel<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(inputs[0]->data(), iGradient->data(), inputs[1]->data(), outputGradient->data(), scale, N);
    } else if (1 == index) {
        CrossEntropyBackwardYKernel<T><<<grideSize, DEEP8_GPU_BLOCK_SIZE >>>(inputs[0]->data(), inputs[1]->data(), iGradient->data(), outputGradient->data(), scale, N);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}


DEEP8_DECLARATION_GPU_FUNC(CrossEntropy);

}