#ifndef DEEP8_SCALARDIVIDE_H
#define DEEP8_SCALARDIVIDE_H

namespace Deep8 {

/*****************************************************************************/
 /**Y = scalar / X*/
 /*****************************************************************************/

template <typename T>
struct ScalarDivideForwardExpr {
	T scalar;

	explicit ScalarDivideForwardExpr(T s) : scalar(s) {
	}

	inline T operator()(T in) const {
		return scalar / in;
	}
};

template <typename T>
struct ScalarDivideBackwardExpr {
	T scalar;

	explicit ScalarDivideBackwardExpr(T s) : scalar(s) {
	}

	inline T operator()(T in) const {
		return -scalar / (in * in);
	}
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void ScalarDivideForwardKernel(const real scalar, const real *X, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = scalar / X[i];
	}
}

template <typename real>
__global__ void ScalarDivideBackwardKernel(const real scalar, real *xGrad, const real *X, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] = -scalar * yGrad[i] / (X[i] * X[i]);
	}
}

#endif

template <typename T>
class ScalarDivide : public Function<T> {
public:
	T scalar;

	explicit ScalarDivide(std::vector<Node*> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
		check();
	}

	void check() override {
		Function<T>::check();

		DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in ScalarDivide Function");

		this->outputShape = this->inputs[0]->outputShape;
	}

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

		eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(ScalarDivideForwardExpr<T>(scalar));
	}

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

		eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(inputs[0]).unaryExpr(ScalarDivideBackwardExpr<T>(scalar));
	}

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarDivideForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ScalarDivideForwardKernel<T> << <grideSize, blockSize >> > (scalar, inputs[0]->data(), output->data(), N);

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
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(iGradient->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarDivideBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ScalarDivideBackwardKernel<T> << <grideSize, blockSize >> > (scalar, iGradient->data(), inputs[0]->data(), outputGradient->data(), N);
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}
};



}

#endif