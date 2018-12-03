#include "Batch.h"
#include "GPUElementWise.cuh"
#include "GPUDevice.h"

namespace Deep8 {

template <typename T>
struct BatchOP {
	DEEP8_CUDA_FUNC DEEP8_CUDA_INLINE T backward(const T &x, const T &y, const T &dy) {
		return dy;
	}
};

template <typename T>
void Batch<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<GPUDevice *>(output->device());

    /**copy memory*/
    size_t offset = 0;

    for (auto item : inputs) {
        device->copyFromGPUToGPU(item->raw(), (byte*)(output->raw()) + offset, sizeof(T) * item->size());

        offset += sizeof(T) * item->size();
    }
}

template <typename T>
void Batch<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 <= index && index < inputs.size(), "the index of Batch backward is error!");

    size_t offset = 0;

	for (size_t i = 0; i < index; ++i) {
		offset += sizeof(T) * inputs[i]->size();
	}

	auto x  = inputs[0]->data();
	auto dx = iGradient->data();
	auto y  = (T*)((byte*)(output->raw()) + offset);
	auto dy = (T*)((byte*)(outputGradient->raw()) + offset);

    int N = (int)iGradient->shape.size();

	int grideSize = (N + DEEP8_GPU_BLOCK_SIZE - 1) / DEEP8_GPU_BLOCK_SIZE;

	UnaryElementWiseBackward<T, BatchOP<T>> <<<grideSize, DEEP8_GPU_BLOCK_SIZE >>> (x, dx, y, dy, BatchOP<T>(), N);
}

DEEP8_DECLARATION_GPU_FUNC(Batch);

}