#ifndef DEEP8_ABS_H
#define DEEP8_ABS_H

namespace Deep8 {

/**
 * y = |x|
 */

template <typename T>
struct AbsBackwardExpr {
    inline T operator()(T outputGrad, T input) const {
        if (input > T(0)) {
            return outputGrad;
        } else if (input < T(0)) {
            return -outputGrad;
        } else {
            return 0;
        }
    }
};

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

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Abs Function needs only 1 input");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto eigenDevice = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(output).device(*eigenDevice) = eTVec(inputs[0]).abs();
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        if (0 != index) {
            DEEP8_RUNTIME_ERROR("the index of Abs backwardCPU is error");
        }

        auto eigenDevice = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(iGradient).device(*eigenDevice) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), AbsBackwardExpr<T>());
    }

#ifdef HAVE_CUDA

	void forwardGPUImpl(const T *x, T *y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AbsForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AbsForwardKernel<T> << <grideSize, blockSize >> > (x, y, N);
	}

#endif // HAVE_CUDA

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		forwardGPUImpl(inputs[0]->data(), output->data(), static_cast<int>(output->shape.size()));
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }


#ifdef HAVE_CUDA

	void backwardGPUImpl(const T *x, T *xGrad, const T *yGrad, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AbsBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AbsBackwardKernel<T> << <grideSize, blockSize >> > (x, xGrad, yGrad, N);
	}

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
