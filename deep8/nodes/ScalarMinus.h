#ifndef DEEP8_SCALARMINUS_H
#define DEEP8_SCALARMINUS_H

namespace Deep8 {

/*****************************************************************************************************************/
/**a scalar minus Tensor */
/*****************************************************************************************************************/

#ifdef HAVE_CUDA

template <typename real>
__global__ void ScalarMinusForwardKernel(const real scalar, const real *X, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = scalar - X[i];
	}
}

template <typename real>
__global__ void ScalarMinusBackwardKernel(real *dx, const real *dy, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		dx[i] -= dy[i];
	}
}

#endif


template <typename T>
class ScalarMinus : public Function<T> {
public:
	T scalar;

	explicit ScalarMinus(std::vector<Node*> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
		check();
	}

	void check() override {
		Function<T>::check();

		DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in ScalarMinus Function");

		this->outputShape = this->inputs[0]->outputShape;
	}

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

		eTVec(output).device(*device) = -eTVec(inputs[0]) + scalar;
	}


	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
		const Tensor<T> *output,
		const Tensor<T> *outputGradient,
		size_t index,
		Tensor<T> *iGradient) override {
		auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		eTVec(iGradient).device(*device) -= eTVec(outputGradient);
	}
#ifdef HAVE_CUDA

	template<typename real>
	void forwardGPUImpl(const real scalar, const real *X, real *Y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarMinusForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ScalarMinusForwardKernel<real> << <grideSize, blockSize >> > (scalar, X, Y, N);
	}

#ifdef HAVE_HALF

	template<>
	void forwardGPUImpl<half>(const half scalar, const half *X, half *Y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ScalarMinusForwardKernel<half> << <grideSize, blockSize >> > (scalar, X, Y, N);
	}
#endif
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(scalar, inputs[0]->data(), output->data(), static_cast<int>(output->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif // HAVE_CUDA
	}

#ifdef HAVE_CUDA

	void backwardGPUImpl(cublasHandle_t &cublasHandle, float *inGrad, const float *outGrad, const int N) {
		float alpha = -1;
		CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &alpha, outGrad, 1, inGrad, 1));
	}

	void backwardGPUImpl(cublasHandle_t &cublasHandle, double *inGrad, const double *outGrad, const int N) {
		double alpha = -1;
		CUBLAS_CHECK(cublasDaxpy(cublasHandle, N, &alpha, outGrad, 1, inGrad, 1));
	}

#ifdef HAVE_HALF

	void backwardGPUImpl(cublasHandle_t &cublasHandle, half *inGrad, const half *outGrad, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ScalarMinusBackwardKernel<half> << <grideSize, blockSize >> > (inGrad, outGrad, N);
	}
#endif
#endif

	void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
		const Tensor<T> *output,
		const Tensor<T> *outputGradient,
		size_t index,
		Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA

		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<GPUDevice*>(iGradient->device);

		backwardGPUImpl(device->cublasHandle, iGradient->data(), outputGradient->data(), static_cast<int>(iGradient->size()));

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif // HAVE_CUDA
	}
};


}

#endif