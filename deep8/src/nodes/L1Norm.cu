#include "Exception.h"
#include "GPUException.h"
#include "GPUMathUtils.h"
#include "GPUDevice.h"
#include "GPUReduce.cuh"
#include "L1Norm.h"

namespace Deep8 {

template <typename T>
struct L1NormForwardOp {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T init() {
		return T(0);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T step(const T &ret, const T &cur) {
		return ret + (cur >= T(0) ? cur : -cur);
	}

	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T complete(const T &ret) {
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

	int sharedSize = sizeof(T) * blockSize;

	switch (blockSize) {
	case 1024:
		TailReduceForward<T, L1NormForwardOp<T>, 1024> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 512:
		TailReduceForward<T, L1NormForwardOp<T>,  512> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 256:
		TailReduceForward<T, L1NormForwardOp<T>,  256> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 128:
		TailReduceForward<T, L1NormForwardOp<T>,  128> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 64:
		TailReduceForward<T, L1NormForwardOp<T>,   64> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 32:
		TailReduceForward<T, L1NormForwardOp<T>,   32> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 16:
		TailReduceForward<T, L1NormForwardOp<T>,   16> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 8:
		TailReduceForward<T, L1NormForwardOp<T>,    8> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 4:
		TailReduceForward<T, L1NormForwardOp<T>,    4> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 2:
		TailReduceForward<T, L1NormForwardOp<T>,    2> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	case 1:
		TailReduceForward<T, L1NormForwardOp<T>,    1> <<<batch, blockSize, sharedSize>>>(x, y, batch, size, L1NormForwardOp<T>());
		break;
	default:
		DEEP8_RUNTIME_ERROR("the block size is error");
		break
	}
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