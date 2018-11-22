#include "UnBatch.h"
#include "GPUDevice.h"

namespace Deep8 {

template <typename T>
void UnBatch<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	output->storage = inputs[0]->storage;
	output->offset  = inputs[0]->offset + sizeof(T) * offset;
}

template <typename T>
void UnBatch<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	/**do nothing*/
}

DEEP8_DECLARATION_GPU_FUNC(UnBatch);

}