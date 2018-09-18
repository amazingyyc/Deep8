#ifndef DEEP8_SQUARE_H
#define DEEP8_SQUARE_H

namespace Deep8 {

/**
 * Y = X * X
 */

#ifdef HAVE_CUDA

template <typename real>
__global__ void SquareForwardKernel(const real *X, real *Y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = X[i] * X[i];
	}
}

template <typename real>
__global__ void SquareBackwardKernel(real *xGrad, const real *X, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += 2 * yGrad[i] * X[i];
	}
}

#endif

template <typename T>
class Square : public Function<T> {
public:

	Square(std::vector<Node*> &inputs): Function<T>(inputs) {
		check();
	}

	void check() override {
		Function<T>::check();

		DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Square Function needs only 1 input");

		this->outputShape = this->inputs[0]->outputShape;
	}

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

		eTVec(output).device(*device) = eTVec(inputs[0]).square();
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

		eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(inputs[0]) * T(2);
	}

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SquareForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		SquareForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), output->data(), N);

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

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(iGradient->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SquareBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		SquareBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), inputs[0]->data(), outputGradient->data(), N);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}

};


}

#endif
