#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.cuh"
#include "Exp.h"

namespace Deep8 {

template <typename T>
struct ExpForwardOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T forward(const T &x) {}
};

template <>
struct ExpForwardOP<float> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float forward(const float &x) {
		return expf(x);
	}
};

template <>
struct ExpForwardOP<double> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double forward(const double &x) {
		return exp(x);
	}
};

#ifdef HAVE_HALF
template <>
struct ExpForwardOP<half> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half forward(const half &x) {
		return hexp(x);
	}
};
#endif

template <typename T>
struct ExpBackwardOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return y * dy;
	}
};

template <typename T>
void Exp<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto x = inputs[0]->data();
	auto y = output->data();
	auto N = (int)output->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, ExpForwardOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, y, ExpForwardOP<T>(), N);
}

template <typename T>
void Exp<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                         const Tensor<T> *output,
                         const Tensor<T> *outputGradient,
                         size_t index,
                         Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Exp backwardCPU is error");

	auto x  = inputs[0]->data();
	auto dx = iGradient->data();
	auto y  = output->data();
	auto dy = outputGradient->data();

	int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, ExpBackwardOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, dx, y, dy, ExpBackwardOP<T>(), N);
}

DEEP8_DECLARATION_GPU_FUNC(Exp);

}