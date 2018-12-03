#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.cuh"
#include "Sigmoid.h"

namespace Deep8 {

template <typename T>
struct SigmoidForwardOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T forward(const T &x) {
		return 0.5 + 0.5 * CuMath::cuTanh(0.5 * x);
	}
};

#ifdef HAVE_HALF
template <>
struct SigmoidForwardOP<half> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half forward(const half &x) {
		return __float2half(0.5 + 0.5 * tanhf(0.5 * __half2float(x)));
	}
};
#endif

template <typename T>
struct SigmoidBackwardOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return dy * y * (T(1) - y);
	}
};

template <typename T>
void Sigmoid<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto x = inputs[0]->data();
	auto y = output->data();
	auto N = (int)output->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, SigmoidForwardOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, y, SigmoidForwardOP<T>(), N);
}

template <typename T>
void Sigmoid<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                             const Tensor<T> *output,
                             const Tensor<T> *outputGradient,
                             size_t index,
                             Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of is error");

	auto x  = inputs[0]->data();
	auto dx = iGradient->data();
	auto y  = output->data();
	auto dy = outputGradient->data();

	int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, SigmoidBackwardOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, dx, y, dy, SigmoidBackwardOP<T>(), N);
}


DEEP8_DECLARATION_GPU_FUNC(Sigmoid);

}