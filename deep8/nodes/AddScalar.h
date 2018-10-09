#ifndef DEEP8_ADDSCALAR_H
#define DEEP8_ADDSCALAR_H

#include "Function.h"

namespace Deep8 {

/**
 * Y = X(tensor) + scalar
 */

#ifdef HAVE_CUDA

template <typename real>
__global__ void AddScalarForwardKernel(const real *x, const real scalar, real *y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		y[i] = x[i] + scalar;
	}
}

template <typename real>
__global__ void AddScalarBackwardKernel(real *xGrad, const real *yGrad, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i];
	}
}

#endif

template <typename T>
class AddScalar : public Function<T> {
public:
	T scalar;

	explicit AddScalar(std::vector<Node *> &inputs, T scalar) : Function<T>(inputs), scalar(scalar) {
		check();
	}

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
		const Tensor<T> *output,
		const Tensor<T> *outputGradient,
		size_t index,
		Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
	
	template <typename real>
	void forwardGPUImpl(const real *X, const real scalar, real *Y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AddScalarForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AddScalarForwardKernel<real> << <grideSize, blockSize >> > (X, scalar, Y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *X, const half scalar, half *Y, const int N) {
		int blockSize = 1024;
		int grideSize;

		grideSize = (N + blockSize - 1) / blockSize;

		AddScalarForwardKernel<half> << <grideSize, blockSize >> > (X, scalar, Y, N);
	}

#endif // HVE_HALF
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(inputs[0]->data(), scalar, output->data(), static_cast<int>(output->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}

#ifdef HAVE_CUDA

	void backwardGPUImpl(cublasHandle_t &handle, float *inGrad, const float *outGrad, const int N) {
		float alpha = 1;
		CUBLAS_CHECK(cublasSaxpy(handle, N, &alpha, outGrad, 1, inGrad, 1));
	}

	void backwardGPUImpl(cublasHandle_t &handle, double *inGrad, const double *outGrad, const int N) {
		double alpha = 1;
		CUBLAS_CHECK(cublasDaxpy(handle, N, &alpha, outGrad, 1, inGrad, 1));
	}

#ifdef HAVE_HALF

	void backwardGPUImpl(cublasHandle_t &handle, half *inGrad, const half *outGrad, const int N) {
		int blockSize = 1024;
		int grideSize;

		grideSize = (N + blockSize - 1) / blockSize;

		AddScalarBackwardKernel<half> << <grideSize, blockSize >> > (inGrad, outGrad, N);
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

		auto device = static_cast<GPUDevice*>(iGradient->device());

		backwardGPUImpl(device->cublasHandle, iGradient->data(), outputGradient->data(), static_cast<int>(iGradient->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}
};


}

#endif