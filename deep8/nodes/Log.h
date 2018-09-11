#ifndef DEEP8_LOG_H
#define DEEP8_LOG_H

#include "Node.h"
#include "CudaMathUtils.h"
#include "Function.h"

namespace Deep8 {

/**
 * y = Log(x)
 * x must be > 0
 */

template <typename T>
struct LogBackwardExpr {
    inline T operator()(T outputGrad, T input) const {
        return outputGrad / input;
    }
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void LogForwardKernel(const real *X, real *Y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = cuLog(X[i]);
	}
}

template <typename real>
__global__ void LogBackwardKernel(real *xGrad, const real *X, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i] / X[i];
	}
}

#endif

template <typename T>
class Log: public Function<T> {
public:
    explicit Log(std::vector<Node *> &inputs): Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Log Function needs only 1 input");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(output).device(*device) = eTVec(inputs[0]).log();
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

        auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

        eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), LogBackwardExpr<T>());
    }

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LogForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LogForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), output->data(), N);
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
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of Log backwardCPU is error");

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(iGradient->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LogBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LogBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), inputs[0]->data(), outputGradient->data(), N);
#else
        DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_LOG_H
