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
	template <typename real>
	void forwardCPUImpl(const std::vector<const Tensor<real>*> &inputs, Tensor<real> *output) {
		auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;
		eTVec(output).device(*device) = eTVec(inputs[0]).cwiseMax(T(0));
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
		DEEP8_ARGUMENT_CHECK(0 == index, "the index of ReLu backwardCPU is error");

		auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;
		eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), ReLuBackwardExpr<real>());
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

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ReLuForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ReLuForwardKernel<real> << <grideSize, blockSize >> > (x, y, N);
	}

#ifdef HAVE_HALF
	template <>
	void forwardGPUImpl<half>(const half *x, half *y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ReLuForwardKernel<half> << <grideSize, blockSize >> > (x, y, N);
	}
#endif // HAVE_HALF


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

#ifdef HAVE_HALF
	void forwardGPUCUDNNImpl(GPUDevice *device, const half *X, half *Y, Shape &shape) {
		half alpha = 1;
		half beta = 0;

		int size = static_cast<int>(shape.size());

		cudnnActivationDescriptor_t activationDesc;
		CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
		CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

		CUDNN_CHECK(cudnnActivationForward(device->cudnnHandle, activationDesc, &alpha, xDesc, X, &beta, yDesc, Y));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
	}
#endif // HAVE_HALF
#endif
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
#ifdef HAVE_CUDNN
		forwardGPUCUDNNImpl(static_cast<GPUDevice*>(output->device()), inputs[0]->data(), output->data(), output->shape);
#else
		forwardGPUImpl(inputs[0]->data(), output->data(), static_cast<int>(output->size()));
#endif
		
#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}


#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(real *dx, const real *x, const real *dy, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, ReLuBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		ReLuBackwardKernel<real> << <grideSize, blockSize >> > (dx, x, dy, N);
	}

#ifdef HAVE_HALF
	template <>
	void backwardGPUImpl<half>(half *dx, const half *x, const half *dy, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		ReLuBackwardKernel<half> << <grideSize, blockSize >> > (dx, x, dy, N);
	}
#endif // HAVE_HALF

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

#ifdef HAVE_HALF
	void backwardGPUCUDNNImpl(GPUDevice *device, const half *x, half *dx, const half *y, const half *dy, Shape &shape) {
		half alpha = 1;
		half beta = 1;

		int size = static_cast<int>(shape.size());

		cudnnActivationDescriptor_t activationDesc;
		CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
		CUDNN_CHECK(cudnnSetActivationDescriptor(activationDesc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

		cudnnTensorDescriptor_t dxDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

		cudnnTensorDescriptor_t dyDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, size, 1, 1, 1));

		CUDNN_CHECK(cudnnActivationBackward(device->cudnnHandle, activationDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
	}
#endif // HAVE_HALF
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

		backwardGPUCUDNNImpl(static_cast<GPUDevice*>(iGradient->device()), 
			inputs[0]->data(), 
			iGradient->data(), 
			output->data(), 
			outputGradient->data(), 
			iGradient->shape);

#else
		backwardGPUImpl(iGradient->data(), inputs[0]->data(), outputGradient->data(), static_cast<int>(iGradient->size()));
#endif

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_RELU_H
