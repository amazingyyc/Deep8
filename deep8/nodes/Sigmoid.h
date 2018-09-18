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

protected:protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(SigmoidForwardExpr<T>());
    }

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override {
        if (0 != index) {
            DEEP8_RUNTIME_ERROR("the index of Sigmoid backwardCPU is error");
        }

		auto device = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;

        eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(output), SigmoidBackwardExpr<T>());
    }

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SigmoidForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		SigmoidForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), output->data(), N);

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

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, SigmoidBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		SigmoidBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), outputGradient->data(), output->data(), N);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_SIGMOID_H
