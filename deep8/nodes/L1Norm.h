#ifndef DEEP8_L1NORM_H
#define DEEP8_L1NORM_H

#include "Node.h"
#include "Function.h"
#include "CudaMathUtils.h"

namespace Deep8 {

template <typename T>
struct L1NormBackwardExpr {
    inline T operator()(T outputGrad, T input) const {
        if (input > 0) {
            return outputGrad;
        } else if (input < 0) {
            return -outputGrad;
        } else {
            return 0;
        }
    }
};

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
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		int y = i / size;

		if (x[i] > 0) {
			xGrad[i] += yGrad[y];
		} else if (x[i] < 0) {
			xGrad[i] -= yGrad[y];
		}
	}
}


#endif

template <typename T>
class L1Norm: public Function<T> {
public:
    explicit L1Norm(std::vector<Node *> &inputs): Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1Norm Function needs only 1 input");

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

        eTVec(output).device(*eigenDevice) = eTVec(input).abs().reshape(reshapeDims).sum(sumDims);
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

        auto eigenDevice = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;

        auto batch = iGradient->batch();
        auto size  = iGradient->size() / batch;

        Eigen::array<size_t, 2> iGradientDims = {batch, size};
        Eigen::array<size_t, 2> outputGradientDims = {batch, 1};
        Eigen::array<size_t, 2> broadDims = {1, size};

        eTVec(iGradient).reshape(iGradientDims).device(*eigenDevice) +=
                eTVec(outputGradient).reshape(outputGradientDims).broadcast(broadDims).binaryExpr(eTVec(inputs[0]).reshape(iGradientDims), L1NormBackwardExpr<T>());
    }

#ifdef HAVE_CUDA

	void forwardGPUImpl(GPUDevice *device, const float *x, float *y, Shape &shape) {
		auto batch = (int)shape.batch();
		auto size  = (int)shape.size() / batch;

		for (int b = 0; b < batch; ++b) {
			CUBLAS_CHECK(cublasSasum(device->cublasHandle, size, x, 1, y));

			x += size;
			y += 1;
		}
	}

	void forwardGPUImpl(GPUDevice *device, const double *x, double *y, Shape &shape) {
		auto batch = (int)shape.batch();
		auto size = (int)shape.size() / batch;

		for (int b = 0; b < batch; ++b) {
			CUBLAS_CHECK(cublasDasum(device->cublasHandle, size, x, 1, y));

			x += size;
			y += 1;
		}
	}

#endif

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		auto shape = inputs[0]->shape;
		int batch = (int)shape.batch();
		int size  = (int)shape.size() / batch;

		int blockSize = 1024;

		if (size < blockSize) {
			blockSize = prevPowerOf2(size);
		}

		int sharedSize = sizeof(T) * blockSize;

		if (1024 == blockSize) {
			L1NormForwardKernel<1024, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (512 == blockSize) {
			L1NormForwardKernel<512, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (256 == blockSize) {
			L1NormForwardKernel<256, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (128 == blockSize) {
			L1NormForwardKernel<128, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (64 == blockSize) {
			L1NormForwardKernel<64, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (32 == blockSize) {
			L1NormForwardKernel<32, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (16 == blockSize) {
			L1NormForwardKernel<16, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (8 == blockSize) {
			L1NormForwardKernel<8, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (4 == blockSize) {
			L1NormForwardKernel<4, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (2 == blockSize) {
			L1NormForwardKernel<2, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
		} else if (1 == blockSize) {
			L1NormForwardKernel<1, T> << <batch, blockSize, sharedSize >> > (inputs[0]->data(), output->data(), batch, size);
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

		int N = (int)shape.size();

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, L1NormBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		L1NormBackwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), iGradient->data(), outputGradient->data(), size, N);
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_L1NORM_H
