#ifndef DEEP8_L2NORM_H
#define DEEP8_L2NORM_H

#include "Node.h"
#include "Function.h"

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

#endif

template <typename T>
class L2Norm: public Function<T> {
public:
    explicit L2Norm(std::vector<Node *> &inputs): Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L2Norm Function needs only 1 input");

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

        eTVec(output).device(*eigenDevice) = eTVec(input).square().reshape(reshapeDims).sum(sumDims).sqrt();
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of L2Norm backwardCPU is error");

        auto eigenDevice = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;

        auto batch = iGradient->batch();
        auto size  = iGradient->batchSize();

        Eigen::array<size_t, 2> outputDims = {batch, 1};
        Eigen::array<size_t, 2> outputBroad = {1, size};
        Eigen::array<size_t, 2> iGradientDims = {batch, size};

        eTVec(iGradient).reshape(iGradientDims).device(*eigenDevice) +=
                (eTVec(outputGradient).reshape(outputDims) / eTVec(output).reshape(outputDims)).broadcast(outputBroad) * eTVec(inputs[0]).reshape(iGradientDims);
    }

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		
		auto shape = inputs[0]->shape;
		int batch  = (int)shape.batch();
		int size   = (int)shape.size() / batch;

		int blockSize = 1024;

		if (size < blockSize) {
			blockSize = prevPowerOf2(size);
		}

		int sharedSize = sizeof(T) * blockSize;

		if (1024 == blockSize) {
			L2NormForwardKernel<1024, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (512 == blockSize) {
			L2NormForwardKernel<512, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (256 == blockSize) {
			L2NormForwardKernel<256, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (128 == blockSize) {
			L2NormForwardKernel<128, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (64 == blockSize) {
			L2NormForwardKernel<64, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (32 == blockSize) {
			L2NormForwardKernel<32, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (16 == blockSize) {
			L2NormForwardKernel<16, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (8 == blockSize) {
			L2NormForwardKernel<8, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (4 == blockSize) {
			L2NormForwardKernel<4, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (2 == blockSize) {
			L2NormForwardKernel<2, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (1 == blockSize) {
			L2NormForwardKernel<1, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else {
			DEEP8_RUNTIME_ERROR("the block size is error");
		}

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
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

		auto shape = iGradient->shape;
		int batch  = (int)shape.batch();
		int size   = (int)shape.size() / batch;

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = batch * size;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, L2NormBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		L2NormBackwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), iGradient->data(), output->data(), outputGradient->data(), size, N);
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_L2NORM_H
