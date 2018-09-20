#ifndef DEEP8_SIGMOID_H
#define DEEP8_SIGMOID_H

namespace Deep8 {

template <typename T>
struct SigmoidForwardExpr {
    inline T operator()(T in) const {
        return T(0.5) + T(0.5) * tanh(T(0.5) * in);
    }
};

template <typename T>
struct SigmoidBackwardExpr {
    inline T operator()(T outputGrad, T output) const {
        return outputGrad * output * (T(1) - output);
    }
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void SigmoidForwardKernel(const real *X, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = 0.5 + 0.5 * cuTanh(0.5 * X[i]);
	}
}

template <typename real>
__global__ void SigmoidBackwardKernel(real *xGrad, const real *yGrad, const real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += yGrad[i] * Y[i] * (1 - Y[i]);
	}
}

#endif

template <typename T>
class Sigmoid: public Function<T> {
public:
    explicit Sigmoid(std::vector<Node*> &inputs): Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Sigmoid Function needs only 1 input");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
	template <typename real>
	void forwardCPUImpl(const std::vector<const Tensor<real>*> &inputs, Tensor<real> *output) {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;
		eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(SigmoidForwardExpr<T>());
	}

#ifdef HAVE_HALF
	template <>
	void forwardCPUImpl<half>(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		forwardCPUImpl(inputs, output);
    }


	template <typename real>
	void backwardCPUImpl(const std::vector<const Tensor<real>*> &inputs,
		const Tensor<real> *output,
		const Tensor<real> *outputGradient,
		size_t index,
		Tensor<real> *iGradient) {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of Sigmoid backwardCPU is error");

		auto device = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;
		eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(output), SigmoidBackwardExpr<T>());
	}

#ifdef HAVE_HALF
	template <>
	void backwardCPUImpl<half>(const std::vector<const Tensor<half>*> &inputs,
		const Tensor<half> *output,
		const Tensor<half> *outputGradient,
		size_t index,
		Tensor<half> *iGradient) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	}
#endif // HAVE_HALF

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
		backwardCPUImpl(inputs, output, outputGradient, index, iGradient);
    }

#ifdef HAVE_CUDA

	template <typename real>
	void forwardGPUImpl(const real *x, real *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SigmoidForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		SigmoidForwardKernel<real> << <grideSize, blockSize >> > (x, y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *x, half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		SigmoidForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
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

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SigmoidBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		SigmoidBackwardKernel<real> << <grideSize, blockSize >> > (dx, dy, y, N);
	}

#ifdef HAVE_HALF
	template <>
	void backwardGPUImpl<half>(half *dx, const half *dy, const half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		SigmoidBackwardKernel<half> << <grideSize, blockSize >> > (dx, dy, y, N);
	}
	
#endif
#endif
	void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		backwardGPUImpl(iGradient->data(), outputGradient->data(), output->data(), static_cast<int>(iGradient->size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_SIGMOID_H
