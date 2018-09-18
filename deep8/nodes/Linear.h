#ifndef DEEP8_LINEAR_H
#define DEEP8_LINEAR_H

namespace Deep8 {

/**
 * @brief y = a * x + b
 */

#ifdef HAVE_CUDA

template <typename real>
__global__ void LinearForwardKernel(const real *X, const real a, const real b, real *Y, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = a * X[i] + b;
	}
}

template <typename real>
__global__ void LinearBackwardKernel(real *xGrad, const real a, const real *yGrad, const int N) {
	int start = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += a * yGrad[i];
	}
}

#endif

template <typename T>
class Linear: public Function<T> {
public:
    T a;
    T b;

    explicit Linear(std::vector<Node*> &inputs, T a, T b):Function<T>(inputs), a(a), b(b) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Linear Function needs only 1 input");

        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(output).device(*device) = eTVec(inputs[0]) * a + b;
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

        auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

        eTVec(iGradient).device(*device) += eTVec(outputGradient) * a;
    }


	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
		
		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LinearForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LinearForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), a, b, output->data(), N);

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
        DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(iGradient->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, LinearBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		LinearBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), a, outputGradient->data(), N);
#else
        DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_LINEAR_H
