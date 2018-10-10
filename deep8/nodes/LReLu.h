#ifndef DEEP8_LRELU_H
#define DEEP8_LRELU_H

#include "Function.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void LReLuForwardKernel(const real *X, const real a, real *Y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = X[i] > real(0) ? X[i] : a * X[i];
	}
}

template <typename real>
__global__ void LReLuBackwardKernel(real *xGrad, const real *X, const real a, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i] * (X[i] > real(0) ? real(1) : a);
	}
}


#endif

template <typename T>
class LReLu: public Function<T> {
public:
    T a;

    explicit LReLu(std::vector<Node*> &inputs, T a): Function<T>(inputs), a(a) {
        check();
    }

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

	template <typename real>
	void forwardGPUImpl(const real *x, const real a, real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LReLuForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LReLuForwardKernel<real> << <grideSize, blockSize >> > (x, a, y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *x, const half a, half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		LReLuForwardKernel<half> << <grideSize, blockSize >> > (x, a, y, N);
	}

#endif // HAVE_HALF
#endif // HAVE_CUDA

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(inputs[0]->data(), a, output->data(), static_cast<int>(output->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}


#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(real *xGrad, const real *x, const real a, const real *yGrad, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LReLuBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LReLuBackwardKernel<real> << <grideSize, blockSize >> > (xGrad, x, a, yGrad, N);
	}

#ifdef HAVE_HALF

	template <>
	void backwardGPUImpl<half>(half *xGrad, const half *x, const half a, const half *yGrad, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		LReLuBackwardKernel<half> << <grideSize, blockSize >> > (xGrad, x, a, yGrad, N);
	}
#endif // HAVE_HALF
#endif // HAVE_CUDA

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of LReLu backwardCPU is error");

		backwardGPUImpl(iGradient->data(), inputs[0]->data(), a, outputGradient->data(), static_cast<int>(iGradient->size()));

#else
        DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif
