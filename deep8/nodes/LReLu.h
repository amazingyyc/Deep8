#ifndef DEEP8_LRELU_H
#define DEEP8_LRELU_H

#include "TensorUtils.h"
#include "Function.h"

namespace Deep8 {

template <typename T>
struct LReLuForwardExpr {
    T a;

    explicit LReLuForwardExpr(T p): a(p) {
    }

    inline T operator()(T in) const {
        return ((in > 0.0) ? in : a * in);
    }
};

template <typename T>
struct LReLuBackwardExpr {
    T a;

    explicit LReLuBackwardExpr(T p): a(p) {
    }

    inline T operator()(T outputGrad, T in) const {
        return outputGrad * (in > 0 ? 1.0 : a);
    }
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void LReLuForwardKernel(const real *X, const real a, real *Y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = X[i] > 0 ? X[i] : a * X[i];
	}
}

template <typename real>
__global__ void LReLuBackwardKernel(real *xGrad, const real *X, const real a, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i] * (X[i] > 0 ? 1 : a);
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

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the LReLu Function needs only 1 input");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(LReLuForwardExpr<T>(a));
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

        auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

        eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), LReLuBackwardExpr<T>(a));
    }

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LReLuForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LReLuForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), a, output->data(), N);

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

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LReLuBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LReLuBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), inputs[0]->data(), a, outputGradient->data(), N);

#else
        DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif
