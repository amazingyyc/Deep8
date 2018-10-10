#ifndef DEEP8_MULTIPLYSCALAR_H
#define DEEP8_MULTIPLYSCALAR_H

#include "Function.h"

namespace Deep8 {

/******************************************************************************************/
/**a Tensor multiply Scalar*/
/******************************************************************************************/

#ifdef HAVE_CUDA

template <typename real>
__global__ void MultiplyScalarForwardKernel(const real *X, const real scalar, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = X[i] * scalar;
	}
}

template <typename real>
__global__ void MultiplyScalarBackwardKernel(real *xGrad, const real scalar, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += scalar * yGrad[i];
	}
}

#endif

template <typename T>
class MultiplyScalar : public Function<T> {
public:
	T scalar;

	explicit MultiplyScalar(std::vector<Node *> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
		check();
	}

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
	template <typename real>
	void forwardGPUImpl(const real *X, const real scalar, real *Y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MultiplyScalarForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		MultiplyScalarForwardKernel<real> << <grideSize, blockSize >> > (X, scalar, Y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *X, const half scalar, half *Y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		MultiplyScalarForwardKernel<half> << <grideSize, blockSize >> > (X, scalar, Y, N);
	}
#endif // HAVE_HALF
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

#ifdef HAVE_HALF
	void backwardGPUImpl(cublasHandle_t &cublasHandle, half *inGrad, const half scalar, const half *outGrad, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		MultiplyScalarBackwardKernel<half> << <grideSize, blockSize >> > (inGrad, scalar, outGrad, N);
	}
#endif // HAVE_HALF
#endif

	void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<GPUDevice*>(iGradient->device());

		backwardGPUImpl(device->cublasHandle, iGradient->data(), scalar, outputGradient->data(), static_cast<int>(iGradient->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}
};


}

#endif