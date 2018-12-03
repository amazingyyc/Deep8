#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUReduce.cuh"
#include "L1Norm.h"

namespace Deep8 {

template <typename T>
struct L1NormForwardOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T commense() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init(T ret, T cur) {
		return ret + CuMath::cuAbs(cur);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(T ret1, T ret2) {
		return ret1 + ret2;
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(T ret) {
		return ret;
	}
};

template <typename T>
struct L1NormBackwardOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return x >= T (0) ? dy : -dy;
	}
};

template <typename T>
void L1Norm<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto shape = inputs[0]->shape;
	int batch  = (int)shape.batch();
	int size   = (int)shape.size() / batch;

	auto x = inputs[0]->data();
	auto y = output->data();

	int blockSize = 1024;

	if (size < blockSize) {
		blockSize = prevPowerOf2(size);
	}

	callTailReduceForward<T, L1NormForwardOp<T>>(x, y, batch, size, blockSize);
}

template <typename T>
void L1Norm<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backwardCPU is error");

	auto x = inputs[0]->data();
	auto dx = iGradient->data(); 
	auto y  = output->data();
	auto dy = outputGradient->data();

	int N     = (int)iGradient->shape.size();
	int batch = (int)iGradient->shape.batch();
	int size  = N / batch;

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	TailReduceBackward<T, L1NormBackwardOp<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, dx, y, dy, batch, size, L1NormBackwardOp<T>(), N);
}

DEEP8_DECLARATION_GPU_FUNC(L1Norm)

}