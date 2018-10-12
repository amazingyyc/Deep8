#ifndef DEEP8_SCALARDIVIDE_H
#define DEEP8_SCALARDIVIDE_H

#include "Function.h"

namespace Deep8 {

/*****************************************************************************/
 /**Y = scalar / X*/
 /*****************************************************************************/



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
	void forwardGPUImpl(const real scalar, const real *x, real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarDivideForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ScalarDivideForwardKernel<real> << <grideSize, blockSize >> > (scalar, x, y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half scalar, const half *x, half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ScalarDivideForwardKernel<half> << <grideSize, blockSize >> > (scalar, x, y, N);
	}

#endif
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(scalar, inputs[0]->data(), output->data(), static_cast<int>(output->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}


#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(const real scalar, real *dx, const real *x, const real *dy, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ScalarDivideBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ScalarDivideBackwardKernel<real> << <grideSize, blockSize >> > (scalar, dx, x, dy, N);
	}

#ifdef HAVE_HALF

	template <>
	void backwardGPUImpl<half>(const half scalar, half *dx, const half *x, const half *dy, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ScalarDivideBackwardKernel<half> << <grideSize, blockSize >> > (scalar, dx, x, dy, N);
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

		backwardGPUImpl(scalar, iGradient->data(), inputs[0]->data(), outputGradient->data(), static_cast<int>(iGradient->size()));
		
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}
};



}

#endif