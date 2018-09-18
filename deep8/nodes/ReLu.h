#ifndef DEEP8_RELU_H
#define DEEP8_RELU_H

namespace Deep8 {

template <typename T>
struct ReLuBackwardExpr {
    inline T operator()(T outputGrad, T in) const {
        return outputGrad * (in > 0 ? 1.0 : 0);
    }
};

#ifdef HAVE_CUDA

template <typename real>
__global__ void ReLuForwardKernel(const real *X, real *Y, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		Y[i] = X[i] > 0 ? X[i] : 0;
	}
}

template <typename real>
__global__ void ReLuBackwardKernel(real *xGrad, const real *X, const real *yGrad, const int N) {
	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		xGrad[i] += (X[i] > 0 ? yGrad[i] : 0);
	}
}

#endif

template <typename T>
class ReLu: public Function<T> {
public:
    explicit ReLu(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReLu Function needs only 1 input");

        /**the ReLu output shape equal the input*/
        this->outputShape = this->inputs[0]->outputShape;
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
        auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        eTVec(output).device(*device) = eTVec(inputs[0]).cwiseMax(T(0));
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
        if (0 != index) {
            DEEP8_RUNTIME_ERROR("the index of ReLu backwardCPU is error");
        }

        auto device = static_cast<CPUDevice*>(outputGradient->device)->eigenDevice;

        eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), ReLuBackwardExpr<T>());
    }

#ifdef HAVE_CUDA
#ifdef HAVE_CUDNN

	void forwardGPUCUDNNImpl(GPUDevice *device, const float *X, float *Y, Shape &shape) {
		float alpha = 1;
		float beta  = 0;

		int size = static_cast<int>(shape.size());

		cudnnActivationDescriptor_t activationDesc;
		CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
		CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

		CUDNN_CHECK(cudnnActivationForward(device->cudnnHandle, activationDesc, &alpha, xDesc, X, &beta, yDesc, Y));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
	}

	void forwardGPUCUDNNImpl(GPUDevice *device, const double *X, double *Y, Shape &shape) {
		double alpha = 1;
		double beta = 0;

		int size = static_cast<int>(shape.size());

		cudnnActivationDescriptor_t activationDesc;
		CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
		CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

		CUDNN_CHECK(cudnnActivationForward(device->cudnnHandle, activationDesc, &alpha, xDesc, X, &beta, yDesc, Y));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
	}

#endif
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
#ifdef HAVE_CUDNN
		forwardGPUCUDNNImpl(static_cast<GPUDevice*>(output->device), inputs[0]->data(), output->data(), output->shape);
#else
		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(output->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ReLuForwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ReLuForwardKernel<T> << <grideSize, blockSize >> > (inputs[0]->data(), output->data(), N);
#endif
		
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}


#ifdef HAVE_CUDA
#ifdef HAVE_CUDNN

	void backwardGPUCUDNNImpl(GPUDevice *device, const float *x, float *dx, const float *y, const float *dy, Shape &shape) {
		float alpha = 1;
		float beta  = 1;

		int size = static_cast<int>(shape.size());

		cudnnActivationDescriptor_t activationDesc;
		CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
		CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

		cudnnTensorDescriptor_t dxDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

		cudnnTensorDescriptor_t dyDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, size, 1, 1, 1));

		CUDNN_CHECK(cudnnActivationBackward(device->cudnnHandle, activationDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
	}

	void backwardGPUCUDNNImpl(GPUDevice *device, const double *x, double *dx, const double *y, const double *dy, Shape &shape) {
		double alpha = 1;
		double beta = 1;

		int size = static_cast<int>(shape.size());

		cudnnActivationDescriptor_t activationDesc;
		CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
		CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

		cudnnTensorDescriptor_t dxDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

		cudnnTensorDescriptor_t dyDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, size, 1, 1, 1));

		CUDNN_CHECK(cudnnActivationBackward(device->cudnnHandle, activationDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
	}

#endif
#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

#ifdef HAVE_CUDA
#ifdef HAVE_CUDNN

		backwardGPUCUDNNImpl(static_cast<GPUDevice*>(iGradient->device), 
			inputs[0]->data(), 
			iGradient->data(), 
			output->data(), 
			outputGradient->data(), 
			iGradient->shape);

#else
		int minGrideSize;
		int blockSize;
		int grideSize;

		int N = static_cast<int>(iGradient->size());

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ReLuBackwardKernel<T>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ReLuBackwardKernel<T> << <grideSize, blockSize >> > (iGradient->data(), inputs[0]->data(), outputGradient->data(), N);
#endif

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_RELU_H
