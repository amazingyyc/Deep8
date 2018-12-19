#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.h"
#include "Abs.h"

namespace Deep8 {

template <typename T>
struct AbsOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T forward(const T &x) {
		return CuMath::cuAbs(x);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return x >= T(0) ? dy : -dy;
	}
};

template <typename T>
void Abs<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto x = inputs[0]->data();
	auto y = output->data();
	auto N = (int)output->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, AbsOP<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, y, AbsOP<T>(), N);
}

template <typename T>
void Abs<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
						 const Tensor<T> *output,
						 const Tensor<T> *outputGradient,
						 size_t index,
						 Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backward is error!");

	auto x  = inputs[0]->data();
	auto dx = iGradient->data();
	auto y  = output->data();
	auto dy = outputGradient->data();

	int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, AbsOP<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, dx, y, dy, AbsOP<T>(), N);
}


DEEP8_DECLARATION_GPU_FUNC(Abs);

}