#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.cuh"
#include "ScalarDivide.h"

namespace Deep8 {

template <typename real> 
struct ScalarDivideOp {
    real scalar;

    ScalarDivideOp(real s): scalar(s) {
    }

    DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real forward(const real &x) {
		return scalar / x;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE real backward(const real &x, const real &y, const real &dy) {
		return -scalar * dy / (x * x);
	}
};

template <typename T>
void ScalarDivide<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto x = inputs[0]->data();
    auto y = output->data();
    auto N = static_cast<int>(output->size());

    int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, ScalarDivideOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, y, ScalarDivideOp<T>(scalar), N);
}

template <typename T>
void ScalarDivide<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                                 const Tensor<T> *output,
                                 const Tensor<T> *outputGradient,
                                 size_t index,
                                 Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    auto x  = inputs[0]->data();
	auto dx = iGradient->data();
	auto y  = output->data();
	auto dy = outputGradient->data();

	int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, ScalarDivideOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, dx, y, dy, ScalarDivideOp<T>(scalar), N);
}

DEEP8_DECLARATION_GPU_FUNC(ScalarDivide);

}