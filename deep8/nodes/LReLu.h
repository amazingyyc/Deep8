#ifndef DEEP8_LRELU_H
#define DEEP8_LRELU_H

namespace Deep8 {

template <typename T>
struct LReLuForwardExpr {
    T a;

    explicit LReLuForwardExpr(T p): a(p) {
    }

    inline T operator()(T in) const {
        return ((in > T(0)) ? in : a * in);
    }
};

template <typename T>
struct LReLuBackwardExpr {
    T a;

    explicit LReLuBackwardExpr(T p): a(p) {
    }

    inline T operator()(T outputGrad, T in) const {
        return outputGrad * (in > T(0) ? 1.0 : a);
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
	template <typename real>
	void forwardCPUImpl(const std::vector<const Tensor<real>*> &inputs, Tensor<real> *output) {
		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;
		eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(LReLuForwardExpr<T>(a));
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
	void backwardCPUImpl(const std::vector<const Tensor<real>*> &inputs, const Tensor<real> *output, const Tensor<real> *outputGradient, size_t index, Tensor<real> *iGradient) {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;
		eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), LReLuBackwardExpr<T>(a));
	}

#ifdef HAVE_HALF
	template <>
	void backwardCPUImpl<half>(const std::vector<const Tensor<half>*> &inputs, const Tensor<half> *output, const Tensor<half> *outputGradient, size_t index, Tensor<half> *iGradient) {
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
