#ifndef DEEP8_SOFTMAX_H
#define DEEP8_SOFTMAX_H

#include "Function.h"

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

#endif

template <typename T>
class Softmax: public Function<T> {
public:
    explicit Softmax(std::vector<Node *> &inputs): Function<T>(inputs) {
        check();
    }

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

	template <typename real>
	void forwardGPUImpl(GPUDevice *device, const real *x, real *y, Shape &shape) {
		int N      = (int)shape.size();
		int batch  = (int)shape.batch();
		int size   = N / batch;

		int blockSize = 1024;

		int minGrideSize;
		int grideSize;
		int maxBlockSize;
		int divideBlockSize;

		if (size < blockSize) {
			blockSize = prevPowerOf2(size);
		}

		int sharedSize = sizeof(real) * blockSize;

		auto maxPtr = (real*)device->malloc(sizeof(real) * batch);
		auto sumPtr = (real*)device->malloc(sizeof(real) * batch);

		if (1024 == blockSize) {
			SoftmaxForwardFindMaxKernel<1024, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (512 == blockSize) {
			SoftmaxForwardFindMaxKernel<512, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (256 == blockSize) {
			SoftmaxForwardFindMaxKernel<256, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (128 == blockSize) {
			SoftmaxForwardFindMaxKernel<128, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (64 == blockSize) {
			SoftmaxForwardFindMaxKernel<64, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (32 == blockSize) {
			SoftmaxForwardFindMaxKernel<32, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (16 == blockSize) {
			SoftmaxForwardFindMaxKernel<16, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (8 == blockSize) {
			SoftmaxForwardFindMaxKernel<8, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (4 == blockSize) {
			SoftmaxForwardFindMaxKernel<4, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (2 == blockSize) {
			SoftmaxForwardFindMaxKernel<2, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (1 == blockSize) {
			SoftmaxForwardFindMaxKernel<1, real> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &maxBlockSize, SoftmaxForwardExpMinusScalar<real>, 0, N));
		grideSize = (N + maxBlockSize - 1) / maxBlockSize;

		SoftmaxForwardExpMinusScalar<real><<<grideSize, maxBlockSize >>>(x, maxPtr, y, size, N);
												       
		if (1024 == blockSize) {
			SoftmaxForwardSumKernel<1024, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (512 == blockSize) {
			SoftmaxForwardSumKernel<512, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (256 == blockSize) {
			SoftmaxForwardSumKernel<256, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (128 == blockSize) {
			SoftmaxForwardSumKernel<128, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (64 == blockSize) {
			SoftmaxForwardSumKernel<64, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (32 == blockSize) {
			SoftmaxForwardSumKernel<32, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (16 == blockSize) {
			SoftmaxForwardSumKernel<16, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (8 == blockSize) {
			SoftmaxForwardSumKernel<8, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (4 == blockSize) {
			SoftmaxForwardSumKernel<4, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (2 == blockSize) {
			SoftmaxForwardSumKernel<2, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (1 == blockSize) {
			SoftmaxForwardSumKernel<1, real> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &divideBlockSize, SoftmaxForwardDivideScalar<real>, 0, N));
		grideSize = (N + divideBlockSize - 1) / divideBlockSize;

		SoftmaxForwardDivideScalar<real><<<grideSize, divideBlockSize >>>(y, sumPtr, size, N);
										  
		device->free(sumPtr);
		device->free(maxPtr);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(GPUDevice *device, const half *x, half *y, Shape &shape) {
		int N      = (int)shape.size();
		int batch  = (int)shape.batch();
		int size   = N / batch;

		int blockSize = 1024;
		int maxBlockSize = 1024;
		int divideBlockSize = 1024;

		int grideSize;

		if (size < blockSize) {
			blockSize = prevPowerOf2(size);
		}

		int sharedSize = sizeof(half) * blockSize;

		auto maxPtr = (half*)device->malloc(sizeof(half) * batch);
		auto sumPtr = (half*)device->malloc(sizeof(half) * batch);

		if (1024 == blockSize) {
			SoftmaxForwardFindMaxKernel<1024, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (512 == blockSize) {
			SoftmaxForwardFindMaxKernel<512, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (256 == blockSize) {
			SoftmaxForwardFindMaxKernel<256, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (128 == blockSize) {
			SoftmaxForwardFindMaxKernel<128, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (64 == blockSize) {
			SoftmaxForwardFindMaxKernel<64, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (32 == blockSize) {
			SoftmaxForwardFindMaxKernel<32, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (16 == blockSize) {
			SoftmaxForwardFindMaxKernel<16, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (8 == blockSize) {
			SoftmaxForwardFindMaxKernel<8, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (4 == blockSize) {
			SoftmaxForwardFindMaxKernel<4, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (2 == blockSize) {
			SoftmaxForwardFindMaxKernel<2, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else if (1 == blockSize) {
			SoftmaxForwardFindMaxKernel<1, half> << <batch, blockSize, sharedSize >> > (x, maxPtr, batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}

		grideSize = (N + maxBlockSize - 1) / maxBlockSize;

		SoftmaxForwardExpMinusScalar<half><<<grideSize, maxBlockSize >>>(x, maxPtr, y, size, N);
												       
		if (1024 == blockSize) {
			SoftmaxForwardSumKernel<1024, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (512 == blockSize) {
			SoftmaxForwardSumKernel<512, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (256 == blockSize) {
			SoftmaxForwardSumKernel<256, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (128 == blockSize) {
			SoftmaxForwardSumKernel<128, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (64 == blockSize) {
			SoftmaxForwardSumKernel<64, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (32 == blockSize) {
			SoftmaxForwardSumKernel<32, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (16 == blockSize) {
			SoftmaxForwardSumKernel<16, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (8 == blockSize) {
			SoftmaxForwardSumKernel<8, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (4 == blockSize) {
			SoftmaxForwardSumKernel<4, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (2 == blockSize) {
			SoftmaxForwardSumKernel<2, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else if (1 == blockSize) {
			SoftmaxForwardSumKernel<1, half> << <batch, blockSize, sharedSize >> > (y, sumPtr, batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}

		grideSize = (N + divideBlockSize - 1) / divideBlockSize;

		SoftmaxForwardDivideScalar<half><<<grideSize, divideBlockSize >>>(y, sumPtr, size, N);
										  
		device->free(sumPtr);
		device->free(maxPtr);
	}
#endif
#endif // HAVE_CUDA

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(static_cast<GPUDevice*>(output->device()), inputs[0]->data(), output->data(), output->shape);
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }


#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(GPUDevice *device, real *xGrad, const real *y, const real *yGrad, Shape &shape) {
		int N      = (int)shape.size();
		int batch  = (int)shape.batch();
		int size   = N / batch;

		int blockSize = 1024;
		
		int minGrideSize;
		int grideSize;
		int backBlockSize;

		if (size < blockSize) {
			blockSize = prevPowerOf2(size);
		}

		int sharedSize = sizeof(real) * blockSize;

		auto dotPtr = (real*)device->malloc(sizeof(real) * batch);

		if (1024 == blockSize) {
			SoftmaxBackwardDotKernel<1024, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (512 == blockSize) {
			SoftmaxBackwardDotKernel<512, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (256 == blockSize) {
			SoftmaxBackwardDotKernel<256, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (128 == blockSize) {
			SoftmaxBackwardDotKernel<128, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (64 == blockSize) {
			SoftmaxBackwardDotKernel<64, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (32 == blockSize) {
			SoftmaxBackwardDotKernel<32, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (16 == blockSize) {
			SoftmaxBackwardDotKernel<16, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (8 == blockSize) {
			SoftmaxBackwardDotKernel<8, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (4 == blockSize) {
			SoftmaxBackwardDotKernel<4, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (2 == blockSize) {
			SoftmaxBackwardDotKernel<2, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (1 == blockSize) {
			SoftmaxBackwardDotKernel<1, real> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &backBlockSize, SoftmaxBackwardKernel<real>, 0, N));
		grideSize = (N + backBlockSize - 1) / backBlockSize;

		SoftmaxBackwardKernel<real><<<grideSize, backBlockSize >>>(xGrad, y, yGrad, dotPtr, size, N);

		device->free(dotPtr);
	}

#ifdef HAVE_HALF

	template <>
	void backwardGPUImpl<half>(GPUDevice *device, half *xGrad, const half *y, const half *yGrad, Shape &shape) {
		int N      = (int)shape.size();
		int batch  = (int)shape.batch();
		int size   = N / batch;

		int blockSize = 1024;
		
		int grideSize;
		int backBlockSize = 1024;

		if (size < blockSize) {
			blockSize = prevPowerOf2(size);
		}

		int sharedSize = sizeof(half) * blockSize;

		auto dotPtr = (half*)device->malloc(sizeof(half) * batch);

		if (1024 == blockSize) {
			SoftmaxBackwardDotKernel<1024, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (512 == blockSize) {
			SoftmaxBackwardDotKernel<512, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (256 == blockSize) {
			SoftmaxBackwardDotKernel<256, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (128 == blockSize) {
			SoftmaxBackwardDotKernel<128, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (64 == blockSize) {
			SoftmaxBackwardDotKernel<64, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (32 == blockSize) {
			SoftmaxBackwardDotKernel<32, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (16 == blockSize) {
			SoftmaxBackwardDotKernel<16, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (8 == blockSize) {
			SoftmaxBackwardDotKernel<8, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (4 == blockSize) {
			SoftmaxBackwardDotKernel<4, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (2 == blockSize) {
			SoftmaxBackwardDotKernel<2, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else if (1 == blockSize) {
			SoftmaxBackwardDotKernel<1, half> << <batch, blockSize, sharedSize >> > (y, yGrad, dotPtr, batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}

		grideSize = (N + backBlockSize - 1) / backBlockSize;

		SoftmaxBackwardKernel<half><<<grideSize, backBlockSize >>>(xGrad, y, yGrad, dotPtr, size, N);

		device->free(dotPtr);
	}
#endif
#endif // HAVE_CUDA

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

		backwardGPUImpl(static_cast<GPUDevice*>(iGradient->device()), iGradient->data(), output->data(), outputGradient->data(), iGradient->shape);
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_SOFTMAX_H