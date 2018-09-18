#ifndef DEEP8_DIVIDESCALAR_H
#define DEEP8_DIVIDESCALAR_H

namespace Deep8 {

/*****************************************************************************/
/**Tensor divide a scalar*/
/*****************************************************************************/

#ifdef HAVE_CUDA

template <typename real>
__global__ void DivideScalarForwardKernel(const real *X, const real scalar, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = X[i] / scalar;
	}
}

#endif

template <typename T>
class DivideScalar : public Function<T> {
public:
	T scalar;

	explicit DivideScalar(std::vector<Node *> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
		check();
	}

	void check() override {
		Function<T>::check();

		DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in DivideScalar Function");
		DEEP8_ARGUMENT_CHECK(0 != scalar, "the divide scalar can no be 0");

		this->outputShape = this->inputs[0]->outputShape;
	}

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

		eTVec(output).device(*device) = eTVec(inputs[0]) / scalar;
	}
	
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
		const Tensor<T> *output,
		const Tensor<T> *outputGradient,
		size_t index,
		Tensor<T> *iGradient) override {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

		eTVec(iGradient).device(*device) += eTVec(outputGradient) / scalar;
	}

#ifdef HAVE_CUDA

	void forwardGPUImpl(const T *X, const T scalar, T *Y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, DivideScalarForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		DivideScalarForwardKernel<T> << <grideSize, blockSize >> > (X, scalar, Y, N);
	}

#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(inputs[0]->data(), scalar, output->data(), static_cast<int>(output->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}

#ifdef HAVE_CUDA

	void backwardGPUImpl(cublasHandle_t &cublasHandle, float *inGrad, const float scalar, const float *outGrad, const int N) {
		CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &scalar, outGrad, 1, inGrad, 1));
	}

	void backwardGPUImpl(cublasHandle_t &cublasHandle, double *inGrad, const double scalar, const double *outGrad, const int N) {
		CUBLAS_CHECK(cublasDaxpy(cublasHandle, N, &scalar, outGrad, 1, inGrad, 1));
	}

#endif

	void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
		const Tensor<T> *output,
		const Tensor<T> *outputGradient,
		size_t index,
		Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<GPUDevice*>(iGradient->device);

		backwardGPUImpl(device->cublasHandle, iGradient->data(), 1.0 / scalar, outputGradient->data(), static_cast<int>(iGradient->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}
};


}

#endif