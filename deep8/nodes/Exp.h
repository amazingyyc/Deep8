#ifndef DEEP8_EXP_H
#define DEEP8_EXP_H

namespace Deep8 {

/**
 * y = exp(x)
 */

#ifdef HAVE_CUDA

template <typename real>
__global__ void ExpForwardKernel(const real *X, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = cuExp(X[i]);
	}
}

template <typename real>
__global__ void ExpBackwardKernel(real *xGrad, const real *yGrad, const real *Y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i] * Y[i];
	}
}

#endif

template <typename T>
class Exp: public Function<T> {
public:
    explicit Exp(std::vector<Node *> &inputs): Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Exp Function needs only 1 input");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

        eTVec(output).device(*device) = eTVec(inputs[0]).exp();
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        DEEP8_ARGUMENT_CHECK(0 == index, "the index of Exp backwardCPU is error");

        auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

        eTVec(iGradient).device(*device) += (eTVec(output) * eTVec(outputGradient));
    }


#ifdef HAVE_CUDA

	template <typename real>
	void forwardGPUImpl(const real *x, real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ExpForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ExpForwardKernel<real> << <grideSize, blockSize >> > (x, y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *x, half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ExpForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
	}
#endif // HAVE_HALF
#endif // HAVE_CUDA

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(inputs[0]->data(), output->data(), static_cast<int>(output->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}


#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(real *xGrad, const real *yGrad, const real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ExpBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ExpBackwardKernel<real> << <grideSize, blockSize >> > (xGrad, yGrad, y, N);
	}

#ifdef HAVE_HALF

	template <>
	void backwardGPUImpl<half>(half *xGrad, const half *yGrad, const half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ExpBackwardKernel<half> << <grideSize, blockSize >> > (xGrad, yGrad, y, N);
	}
#endif // HAVE_HALF
#endif // HAVE_CUDA

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
        DEEP8_ARGUMENT_CHECK(0 == index, "the index of Exp backwardCPU is error");

		backwardGPUImpl(iGradient->data(), outputGradient->data(), output->data(), static_cast<int>(iGradient->size()));
#else
        DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_EXP_H
