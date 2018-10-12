#ifndef DEEP8_TANH_H
#define DEEP8_TANH_H

#include "Function.h"

namespace Deep8 {




#ifdef HAVE_CUDA

template <typename real>
__global__ void TanHForwardKernel(const real *X, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = cuTanh(X[i]);
	}
}

template <typename real>
__global__ void TanHBackwardKernel(real *xGrad, const real *yGrad, const real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i] * (real(1.0) - Y[i] * Y[i]);
	}
}

#endif

template <typename T>
class TanH: public Function<T> {
public:
    explicit TanH(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
    }

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

	template <typename real>
	void forwardGPUImpl(const real *x, real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanHForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		TanHForwardKernel<real> << <grideSize, blockSize >> > (x, y, N);
	}

#ifdef HAVE_HALF
	template <>
	void forwardGPUImpl<half>(const half *x, half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TanHForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
	}
#endif
#endif
	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(inputs[0]->data(), output->data(), static_cast<int>(output->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}


#ifdef HAVE_CUDA

	template <typename real> 
	void backwardGPUImpl(real *dx, const real *dy, const real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanHBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		TanHBackwardKernel<real> << <grideSize, blockSize >> > (dx, dy, y, N);
	}

#ifdef HAVE_HALF

	template <> 
	void backwardGPUImpl<half>(half *dx, const half *dy, const half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		TanHBackwardKernel<half> << <grideSize, blockSize >> > (dx, dy, y, N);
	}

#endif
#endif
    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					 const Tensor<T> *output,
					 const Tensor<T> *outputGradient,
				     size_t index,
					 Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of TanH backwardCPU is error");

		backwardGPUImpl(iGradient->data(), outputGradient->data(), output->data(), static_cast<int>(iGradient->size()));
		
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif

    }
};


}

#endif //DEEP8_TANH_H
