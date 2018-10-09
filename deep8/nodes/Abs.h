#ifndef DEEP8_ABS_H
#define DEEP8_ABS_H

#include "Function.h"

namespace Deep8 {

/**
 * y = |x|
 */


#ifdef HAVE_CUDA

template <typename real>
__global__ void AbsForwardKernel(const real *x, real *y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		y[i] = cuAbs(x[i]);
	}
}

template <typename real>
__global__ void AbsBackwardKernel(const real *x, real *xGrad, const real *yGrad, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		if (x[i] > real(0)) {
			xGrad[i] += yGrad[i];
		} else if (x[i] < real(0)) {
			xGrad[i] -= yGrad[i];
		}
	}
}

#endif

template <typename T>
class Abs: public Function<T> {
public:
    explicit Abs(std::vector<Node *> &inputs): Function<T>(inputs) {
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
	void forwardGPUImpl(const real *x, real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AbsForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AbsForwardKernel<real> << <grideSize, blockSize >> > (x, y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *x, half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		AbsForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
	}

#endif
#endif // HAVE_CUDA

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(inputs[0]->data(), output->data(), static_cast<int>(output->shape.size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }


#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(const real *x, real *xGrad, const real *yGrad, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AbsBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AbsBackwardKernel<real> << <grideSize, blockSize >> > (x, xGrad, yGrad, N);
	}

#ifdef HAVE_HALF

	template <>
	void backwardGPUImpl<half>(const half *x, half *xGrad, const half *yGrad, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		AbsBackwardKernel<half> << <grideSize, blockSize >> > (x, xGrad, yGrad, N);
	}

#endif
#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {

#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backward is error!");

		backwardGPUImpl(inputs[0]->data(), iGradient->data(), outputGradient->data(), static_cast<int>(iGradient->shape.size()));
			
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
		
    }
};

}

#endif //DEEP8_ABS_H
