#ifndef DEEP8_SOFTMAX_H
#define DEEP8_SOFTMAX_H

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
			shared[threaId] = max(shared[threaId], shared[threaId + 512]);
		}

		__syncthreads();
	}

	if (blockSize >= 512) {
		if (threaId < 256) {
			shared[threaId] = max(shared[threaId], shared[threaId + 256]);
		}

		__syncthreads();
	}

	if (blockSize >= 256) {
		if (threaId < 128) {
			shared[threaId] = max(shared[threaId], shared[threaId + 128]);
		}

		__syncthreads();
	}

	if (blockSize >= 128) {
		if (threaId < 64) {
			shared[threaId] = max(shared[threaId], shared[threaId + 64]);
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

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Softmax Function needs only 1 input");
        DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.nDims() >= 2, "the input dimension must be >= 2");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:

	template <typename real>
	void forwardCPUImpl(const std::vector<const Tensor<real>*> &inputs, Tensor<real> *output) {
		auto cpuDevice = static_cast<CPUDevice*>(output->device());
		auto eigenDevice = cpuDevice->eigenDevice;

		auto shape = output->shape;

		auto batch = (int) shape.batch();
		auto size  = (int) shape.size() / batch;

		Eigen::array<int, 1> reduceDims = { 1 };
		Eigen::array<int, 2> reshape    = { batch, 1 };
		Eigen::array<int, 2> broad      = { 1, size };

		Eigen::TensorMap<Eigen::Tensor<real, 2, Eigen::RowMajor>> inputTensor(inputs[0]->data(), batch, size);
		Eigen::TensorMap<Eigen::Tensor<real, 2, Eigen::RowMajor>> outputTensor(output->data(), batch, size);

		outputTensor.device(*eigenDevice) = (inputTensor - inputTensor.maximum(reduceDims).reshape(reshape).broadcast(broad)).exp();
		outputTensor.device(*eigenDevice) = outputTensor / outputTensor.sum(reduceDims).reshape(reshape).broadcast(broad);
	}

#ifdef HAVE_HALF
	template <>
	void forwardCPUImpl<half>(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

	template <typename real>
	void backwardCPUImpl(const std::vector<const Tensor<real>*> &inputs,
		const Tensor<real> *output,
		const Tensor<real> *outputGradient,
		size_t index,
		Tensor<real> *iGradient) {
		//DEEP8_ARGUMENT_CHECK(0 == index, "the index of Softmax backwardCPU is error");

		//auto cpuDevice = static_cast<CPUDevice*>(iGradient->device());
		//auto eigenDevice = cpuDevice->eigenDevice;

		//auto shape = outputGradient->shape;

		//auto batch = shape.batch();
		//auto size = shape.size() / batch;

		//auto sumPtr = static_cast<real*>(cpuDevice->malloc(sizeof(T) * batch));

		//// Tensor<real> sum(sumPtr, { batch }, cpuDevice);

		//Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor>> sumTensor(sumPtr, (int)batch);

		//Eigen::array<size_t, 1> sumDims = { 1 };
		//Eigen::array<size_t, 2> reShapeDims = { batch, size };
		//Eigen::array<size_t, 1> sumReShapeDims = { batch };

		//sumTensor.device(*eigenDevice) = (eTVec(outputGradient).reshape(reShapeDims) * eTVec(output).reshape(reShapeDims)).sum(sumDims).reshape(sumReShapeDims);

		//Tensor<real> outputGradientRow(outputGradient->data(), { size }, cpuDevice);
		//Tensor<real> outputValueRow(output->data(), { size }, cpuDevice);
		//Tensor<real> iGradientRow(iGradient->data(), { size }, cpuDevice);

		//for (size_t b = 0; b < batch; ++b) {
		//	eTVec(iGradientRow).device(*eigenDevice) += (eTVec(outputGradientRow) - sumPtr[b]) * eTVec(outputValueRow);

		//	outputGradientRow.pointer = (byte*)(outputGradientRow.raw()) + size * sizeof(real);
		//	outputValueRow.pointer    = (byte*)(outputValueRow.raw()) + size * sizeof(real);
		//	iGradientRow.pointer      = (byte*)(iGradientRow.raw()) + size * sizeof(real);
		//}

		//cpuDevice->free(sumPtr);
	}

#ifdef HAVE_HALF
	template <>
	void backwardCPUImpl<half>(const std::vector<const Tensor<half>*> &inputs,
		const Tensor<half> *output,
		const Tensor<half> *outputGradient,
		size_t index,
		Tensor<half> *iGradient) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		forwardCPUImpl(inputs, output);
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
		backwardCPUImpl(inputs, output, outputGradient, index, iGradient);
    }

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
		forwardGPUImpl(static_cast<GPUDevice*>(output->device), inputs[0]->data(), output->data(), output->shape);
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

		backwardGPUImpl(static_cast<GPUDevice*>(iGradient->device), iGradient->data(), output->data(), outputGradient->data(), iGradient->shape);
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_SOFTMAX_H
