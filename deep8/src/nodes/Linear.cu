#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.cuh"
#include "Linear.h"

namespace Deep8 {

template <typename T>
struct LinearOP {
	T a;
	T b;

	LinearOP(T aa, T bb) : a(aa), b(bb) {
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T forward(const T &x) {
		return a * x + b;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return a * dy;
	}
};

template <typename T>
void Linear<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto x = inputs[0]->data();
	auto y = output->data();
	auto N = (int)output->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, LinearOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, y, LinearOP<T>(a, b), N);
}


template <typename T>
void Linear<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                        const Tensor<T> *output,
                        const Tensor<T> *outputGradient,
                        size_t index,
                        Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	auto x = inputs[0]->data();
	auto dx = iGradient->data();
	auto y = output->data();
	auto dy = outputGradient->data();

	int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, LinearOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, dx, y, dy, LinearOP<T>(a, b), N);
}

DEEP8_DECLARATION_GPU_FUNC(Linear);


}