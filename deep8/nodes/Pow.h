#ifndef DEEP8_POW_H
#define DEEP8_POW_H

#include "Function.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void PowForwardKernel(const real *X, const real scalar, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = cuPow(X[i], scalar);
	}
}

template <typename real>
__global__ void PowBackwardKernel(real *xGrad, const real *X, const real scalar, const real *yGrad, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	real realScalar = scalar - 1;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i] * cuPow(X[i], realScalar) * scalar;
	}
}

#endif

template <typename T> 
class Pow : public Function<T> {
public:
	T scalar;

	explicit Pow(std::vector<Node*> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
		check();
	}

	void check() override {
		Function<T>::check();

		DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in Pow Function");

		this->outputShape = this->inputs[0]->outputShape;
	}

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

		eTVec(output).device(*device) = eTVec(inputs[0]).pow(scalar);
	}

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
		const Tensor<T> *output,
		const Tensor<T> *outputGradient,
		size_t index,
		Tensor<T> *iGradient) override {
		if (0 != index) {
			DEEP8_RUNTIME_ERROR("the index of Linear backwardCPU is error");
		}

		auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

		eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(inputs[0]).pow(scalar - T(1)) * scalar;
	}

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, PowForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		PowForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), scalar, output->data(), N);
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
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of Pow backwardCPU is error");

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(iGradient->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, PowBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		PowBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), inputs[0]->data(), scalar, outputGradient->data(), N);
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}
};


}

#endif
