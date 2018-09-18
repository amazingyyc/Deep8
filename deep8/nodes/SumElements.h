#ifndef DEEP8_SUMELEMENTS_H
#define DEEP8_SUMELEMENTS_H

namespace Deep8 {


/**
 * y = sum(x)
 * the y is scalar
 */

#ifdef HAVE_CUDA

template <int blockSize, typename real>
__global__ void SumElementsForwardKernel(const real *x, real *y, const int batch, const int size) {
	SharedMemory<real> shareMemory;
	real *shared = shareMemory.pointer();

	int threaId = threadIdx.x;
	int blockId = blockIdx.x;

	int i = blockId * size + threaId;
	int j = threaId;
	
	shared[threaId] = 0;

	while (j < size) {
		shared[threaId] += x[i];

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
		y[blockId] = shared[threaId];
	}
}

template <typename real>
__global__ void SumElementsBackwardKernel(real *xGrad, const real *yGrad, const int size, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i / size];
	}
}

#endif

template <typename T>
class SumElements: public Function<T> {
public:
    explicit SumElements(std::vector<Node *> &inputs): Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the SumElements Function needs only 1 input");

        this->outputShape = Shape({this->inputs[0]->outputShape.batch(), 1});
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto eigenDevice = static_cast<CPUDevice*>(output->device)->eigenDevice;

        auto input = inputs[0];
        auto batch = input->batch();
        auto size  = input->size() / batch;

        Eigen::array<size_t, 2> reshapeDims = {batch, size};
        Eigen::array<size_t, 1> sumDims = {1};

        eTVec(output).device(*eigenDevice) = eTVec(input).reshape(reshapeDims).sum(sumDims);
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
        if (0 != index) {
            DEEP8_RUNTIME_ERROR("the index of SumElements backwardCPU is error");
        }

        auto eigenDevice = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;

        auto batch = iGradient->batch();
        auto size  = iGradient->size() / batch;

        Eigen::array<size_t, 2> iGradientDims = {batch, size};
        Eigen::array<size_t, 2> outputGradientDims = {batch, 1};
        Eigen::array<size_t, 2> broadDims = {1, size};

        eTVec(iGradient).reshape(iGradientDims).device(*eigenDevice) += eTVec(outputGradient).reshape(outputGradientDims).broadcast(broadDims);
    }

#ifdef HAVE_CUDA

	void forwardGPUImpl(GPUDevice *device, const T *x, T *y, const int batch, const int size) {
		int blockSize = 1024;

		if (size < blockSize) {
			blockSize = prevPowerOf2(size);
		}

		int sharedSize = sizeof(T) * blockSize;

		if (1024 == blockSize) {
			SumElementsForwardKernel<1024, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (512 == blockSize) {
			SumElementsForwardKernel<512, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (256 == blockSize) {
			SumElementsForwardKernel<256, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (128 == blockSize) {
			SumElementsForwardKernel<128, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (64 == blockSize) {
			SumElementsForwardKernel<64, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (32 == blockSize) {
			SumElementsForwardKernel<32, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (16 == blockSize) {
			SumElementsForwardKernel<16, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (8 == blockSize) {
			SumElementsForwardKernel<8, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (4 == blockSize) {
			SumElementsForwardKernel<4, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (2 == blockSize) {
			SumElementsForwardKernel<2, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else if (1 == blockSize) {
			SumElementsForwardKernel<1, T> << <batch, blockSize, sharedSize >> > (x, y, batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}
	}

#endif

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		int batch = (int)inputs[0]->shape.batch();
		int size  = (int)inputs[0]->shape.size() / batch;

		forwardGPUImpl(static_cast<GPUDevice*>(output->device), inputs[0]->data(), output->data(), batch, size);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of SumElements backwardCPU is error");

		auto shape = iGradient->shape;
		int batch = (int)shape.batch();
		int size = (int)shape.size() / batch;

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = (int) shape.size();

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SumElementsBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		SumElementsBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), outputGradient->data(), size, N);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_SUMELEMENTS_H
