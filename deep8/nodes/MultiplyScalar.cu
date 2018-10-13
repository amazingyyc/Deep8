#include "MultiplyScalar.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void MultiplyScalarForwardKernel(const real *X, const real scalar, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = X[i] * scalar;
    }
}

template <typename real>
__global__ void MultiplyScalarBackwardKernel(real *xGrad, const real scalar, const real *yGrad, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += scalar * yGrad[i];
    }
}

template <typename T>
void MultiplyScalar<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto X = inputs[0]->data();
    auto Y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MultiplyScalarForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    MultiplyScalarForwardKernel<T> << <grideSize, blockSize >> > (X, scalar, Y, N);
}

#ifdef HAVE_HALF
template <>
void MultiplyScalar<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto X = inputs[0]->data();
    auto Y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    MultiplyScalarForwardKernel<half> << <grideSize, blockSize >> > (X, scalar, Y, N);
}
#endif

#ifdef HAVE_CUDA
void backwardGPUImpl(cublasHandle_t &cublasHandle, float *inGrad, const float scalar, const float *outGrad, const int N) {
		CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &scalar, outGrad, 1, inGrad, 1));
	}

	void backwardGPUImpl(cublasHandle_t &cublasHandle, double *inGrad, const double scalar, const double *outGrad, const int N) {
		CUBLAS_CHECK(cublasDaxpy(cublasHandle, N, &scalar, outGrad, 1, inGrad, 1));
	}

#ifdef HAVE_HALF
	void backwardGPUImpl(cublasHandle_t &cublasHandle, half *inGrad, const half scalar, const half *outGrad, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		MultiplyScalarBackwardKernel<half> << <grideSize, blockSize >> > (inGrad, scalar, outGrad, N);
	}
#endif // HAVE_HALF
#endif

template <>
void MultiplyScalar<float>::backwardGPU(const std::vector<const Tensor<float>*> &inputs,
                                     const Tensor<float> *output,
                                     const Tensor<float> *outputGradient,
                                     size_t index,
                                     Tensor<float> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)iGradient->size(), &scalar, outputGradient->data(), 1, iGradient->data(), 1));
}

template <>
void MultiplyScalar<double>::backwardGPU(const std::vector<const Tensor<double>*> &inputs,
                                        const Tensor<double> *output,
                                        const Tensor<double> *outputGradient,
                                        size_t index,
                                        Tensor<double> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)iGradient->size(), &scalar, outputGradient->data(), 1, iGradient->data(), 1));
}

#ifdef HAVE_HALF
template <>
void MultiplyScalar<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                                        const Tensor<half> *output,
                                        const Tensor<half> *outputGradient,
                                        size_t index,
                                        Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    MultiplyScalarBackwardKernel<half> << <grideSize, blockSize >> > (iGradient->data(), scalar, outputGradient->data(), (int)iGradient->size());
}
#endif

DEEP8_DECLARATION_GPU_FUNC(MultiplyScalar);

#endif

}