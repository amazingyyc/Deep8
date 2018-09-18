#ifndef DEEP8_TANH_H
#define DEEP8_TANH_H

namespace Deep8 {

template <typename T>
struct TanHForwardExpr {
    inline T operator()(T in) const {
        return tanh(in);
    }
};

template <typename T>
struct TanHBackwardExpr {
    inline T operator()(T outputGrad, T output) const {
        return outputGrad * (T(1.0) - output * output);
    }
};


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
		xGrad[i] += yGrad[i] * (1.0 - Y[i] * Y[i]);
	}
}

#endif

template <typename T>
class TanH: public Function<T> {
public:
    explicit TanH(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the TanH Function needs only 1 input");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(TanHForwardExpr<T>());
    }

    
    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					 const Tensor<T> *output,
					 const Tensor<T> *outputGradient,
					 size_t index,
					 Tensor<T> *iGradient) override {
        if (0 != index) {
            DEEP8_RUNTIME_ERROR("the index of TanH backwardCPU is error");
        }

		auto device = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;

        eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(output), TanHBackwardExpr<T>());
    }

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanHForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		TanHForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), output->data(), N);

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
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of TanH backwardCPU is error");

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(iGradient->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, TanHBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		TanHBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), outputGradient->data(), output->data(), N);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif

    }
};


}

#endif //DEEP8_TANH_H
