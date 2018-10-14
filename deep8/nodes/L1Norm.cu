#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "L1Norm.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <int blockSize, typename real>
__global__ void L1NormForwardKernel(const real *x, real *y, const int batch, const int size) {
	SharedMemory<real> shareMemory;
	real *shared = shareMemory.pointer();

	int threaId = threadIdx.x;
	int blockId = blockIdx.x;

	int i = blockId * size + threaId;
	int j = threaId;

	shared[threaId] = 0;

	while (j < size) {
		shared[threaId] += cuAbs(x[i]);

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
		y[blockId] = shared[0];
	}
}

template <typename real>
__global__ void L1NormBackwardKernel(const real *x, real *xGrad, const real *yGrad, const int size, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		int y = i / size;

		if (x[i] > real(0.0)) {
			xGrad[i] += yGrad[y];
		} else if (x[i] < real(0.0)) {
			xGrad[i] -= yGrad[y];
		}
	}
}

template <typename T>
void L1Norm<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto shape = inputs[0]->shape;
	int batch  = (int)shape.batch();
	int size   = (int)shape.size() / batch;

	auto x = inputs[0]->data();
	auto y = output->data();

	int blockSize = 1024;

	if (size < blockSize) {
		blockSize = prevPowerOf2(size);
	}

	int sharedSize = sizeof(T) * blockSize;

	if (1024 == blockSize) {
		L1NormForwardKernel<1024, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (512 == blockSize) {
		L1NormForwardKernel<512, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (256 == blockSize) {
		L1NormForwardKernel<256, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (128 == blockSize) {
		L1NormForwardKernel<128, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (64 == blockSize) {
		L1NormForwardKernel<64, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (32 == blockSize) {
		L1NormForwardKernel<32, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (16 == blockSize) {
		L1NormForwardKernel<16, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (8 == blockSize) {
		L1NormForwardKernel<8, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (4 == blockSize) {
		L1NormForwardKernel<4, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (2 == blockSize) {
		L1NormForwardKernel<2, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else if (1 == blockSize) {
		L1NormForwardKernel<1, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
	} else {
		DEEP8_RUNTIME_ERROR("the block size is error");
	}
}

template <typename T>
void L1Norm<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

	auto shape = iGradient->shape;

	int N     = (int)shape.size();
	int batch = (int)shape.batch();
	int size  = N / batch;

	auto x = inputs[0]->data();
	auto dx = iGradient->data(); 
	auto dy = outputGradient->data();

	int minGrideSize;
	int blockSize;
	int grideSize;

	CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, L1NormBackwardKernel<T>, 0, N));

	grideSize = (N + blockSize - 1) / blockSize;

	L1NormBackwardKernel<T> << <grideSize, blockSize >> > (x, dx, dy, size, N);
}

#ifdef HAVE_HALF
template <>
void L1Norm<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs, const Tensor<half> *output, const Tensor<half> *outputGradient, size_t index, Tensor<half> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

	auto shape = iGradient->shape;

	int N = (int)shape.size();
	int batch = (int)shape.batch();
	int size = N / batch;

	auto x = inputs[0]->data();
	auto dx = iGradient->data();
	auto dy = outputGradient->data();

	int blockSize = 1024;
	int grideSize = (N + blockSize - 1) / blockSize;

	L1NormBackwardKernel<half> << <grideSize, blockSize >> > (x, dx, dy, size, N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(L1Norm)

#endif

}