#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "MinusScalar.h"

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void MinusScalarForwardKernel(const real *X, const real scalar, real *Y, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        Y[i] = X[i] - scalar;
    }
}

template <typename real>
__global__ void MinusScalarBackwardKernel(real *xGrad, const real *yGrad, const int N) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < N; i += stride) {
        xGrad[i] += yGrad[i];
    }
}

#ifdef HAVE_CUDA
template <typename real>
	void forwardGPUImpl(const real *X, const real scalar, real *Y, const int N) {
		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MinusScalarForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		MinusScalarForwardKernel<real> << <grideSize, blockSize >> > (X, scalar, Y, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(const half *X, const half scalar, half *Y, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		MinusScalarForwardKernel<half> << <grideSize, blockSize >> > (X, scalar, Y, N);
	}
#endif // HAVE_HALF
#endif

template <typename T>
void MinusScalar<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int minGrideSize;
    int blockSize;
    int grideSize;

    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, MinusScalarForwardKernel<T>, 0, N));

    grideSize = (N + blockSize - 1) / blockSize;

    MinusScalarForwardKernel<T> << <grideSize, blockSize >> > (X, scalar, Y, N);
}

#ifdef HAVE_HALF
template <>
void MinusScalar<half>::forwardGPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = (int)output->size();

    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    MinusScalarForwardKernel<half> << <grideSize, blockSize >> > (X, scalar, Y, N);
}
#endif

void backwardGPUImpl(cublasHandle_t &cublasHandle, float *inGrad, const float *outGrad, const int N) {
    float alpha = 1;
    CUBLAS_CHECK(cublasSaxpy(cublasHandle, N, &alpha, outGrad, 1, inGrad, 1));
}

void backwardGPUImpl(cublasHandle_t &cublasHandle, double *inGrad, const double *outGrad, const int N) {
    double alpha = 1;
    CUBLAS_CHECK(cublasDaxpy(cublasHandle, N, &alpha, outGrad, 1, inGrad, 1));
}

#ifdef HAVE_HALF

void backwardGPUImpl(cublasHandle_t &cublasHandle, half *inGrad, const half *outGrad, const int N) {
		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		MinusScalarBackwardKernel<half> << <grideSize, blockSize >> > (inGrad, outGrad, N);
	}

#endif // HAVE_HALF
#endif

template <>
void MinusScalar<float>::backwardGPU(const std::vector<const Tensor<double>*> &inputs,
                                     const Tensor<double> *output,
                                     const Tensor<double> *outputGradient,
                                     size_t index,
                                     Tensor<double> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    float alpha = 1;
    CUBLAS_CHECK(cublasSaxpy(device->cublasHandle, (int)iGradient->size(), &alpha, outputGradient->data(), 1, iGradient->data(), 1));
}

template <>
void MinusScalar<double>::backwardGPU(const std::vector<const Tensor<double>*> &inputs,
                                      const Tensor<double> *output,
                                      const Tensor<double> *outputGradient,
                                      size_t index,
                                      Tensor<double> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto device = static_cast<GPUDevice*>(iGradient->device());

    double alpha = 1;

    CUBLAS_CHECK(cublasDaxpy(device->cublasHandle, (int)iGradient->size(), &alpha, outputGradient->data(), 1, iGradient->data(), 1));
}

#ifdef HAVE_HALF
template <>
void MinusScalar<half>::backwardGPU(const std::vector<const Tensor<half>*> &inputs,
                                      const Tensor<half> *output,
                                      const Tensor<half> *outputGradient,
                                      size_t index,
                                      Tensor<half> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    int N = (int)iGradient->size();
    int blockSize = 1024;
    int grideSize = (N + blockSize - 1) / blockSize;

    MinusScalarBackwardKernel<half> << <grideSize, blockSize >> > (iGradient->data(), outputGradient->data(), N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(MinusScalar);

#endif

}