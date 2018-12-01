#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUElementWise.cuh"
#include "Tanh.h"

namespace Deep8 {

template <typename T>
struct TanhForwardOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T forward(const T &x) {}
};

template <>
struct TanhForwardOP<float> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE float forward(const float &x) {
		return tanhf(x);
	}
};

template <>
struct TanhForwardOP<double> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE double forward(const double &x) {
		return tanh(x);
	}
};

#ifdef HAVE_HALF
template <>
struct TanhForwardOP<half> {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE half forward(const half &x) {
		return __float2half(tanh(__half2float(x)));
	}
};
#endif

template <typename T>
struct TanhBackwardOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return dy * (T(1) - y * y);
	}
};

template <typename T>
void Tanh<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto x = inputs[0]->data();
	auto y = output->data();
	auto N = (int)output->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseForward<T, TanhForwardOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, y, TanhForwardOP<T>(), N);
}

template <typename T>
void Tanh<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Tanh backwardCPU is error");

	auto x = inputs[0]->data();
	auto dx = iGradient->data();
	auto y = output->data();
	auto dy = outputGradient->data();

	int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, TanhBackwardOP<T>> << <grideSize, DEEP8_GPU_BLOCK_SIZE >> > (x, dx, y, dy, TanhBackwardOP<T>(), N);
}


DEEP8_DECLARATION_GPU_FUNC(Tanh);

}