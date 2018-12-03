#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.cuh"
#include "MinusScalar.h"

namespace Deep8 {

template <typename real>
struct MinusScalarOp {
    real scalar;

    MinusScalarOp(real s): scalar(s) {
    }

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real forward(const real &x) {
		return x - scalar;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real backward(const real &x, const real &y, const real &dy) {
		return dy;
	}
}; 

template <typename T>
void MinusScalar<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = static_cast<int>(output->size());

    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, MinusScalarOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, y, MinusScalarOp<T>(scalar), N);
}

template <>
void MinusScalar<float>::backwardGPU(const std::vector<const Tensor<float>*> &inputs,
                                     const Tensor<float> *output,
                                     const Tensor<float> *outputGradient,
                                     size_t index,
                                     Tensor<float> *iGradient) {
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

    auto x  = inputs[0]->data();
	auto dx = iGradient->data();
	auto y  = output->data();
	auto dy = outputGradient->data();

	int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<half, MinusScalarOp<half>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, dx, y, dy, MinusScalarOp<half>(scalar), N);
}
#endif

DEEP8_DECLARATION_GPU_FUNC(MinusScalar);

}