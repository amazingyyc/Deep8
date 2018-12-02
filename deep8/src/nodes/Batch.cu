#include "Batch.h"
#include "GPUDevice.h"

namespace Deep8 {

template <typename T>
void Batch<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
//	auto device = static_cast<GPUDevice *>(output->device());
//
//	/**if the inputs tensor's memory is continue*/
//	continuous = true;
//	for (size_t i = 1; i < inputs.size(); ++i) {
//		void *prePtr = (byte*)(inputs[i - 1]->raw()) + inputs[i - 1]->size() * sizeof(T);
//
//		if (prePtr != inputs[i]->raw()) {
//			continuous = false;
//			break;
//		}
//	}
//
//	if (continuous) {
//		/**if the memory is continuous, than do not copy*/
//		output->storage = inputs[0]->storage;
//		output->offset  = inputs[0]->offset;
//		output->shape   = this->outputShape;
//	} else {
//		/**copy memory*/
//		size_t offset = 0;
//
//		for (auto item : inputs) {
//			device->copyFromGPUToGPU(item->raw(), (byte*)(output->raw()) + offset, sizeof(T) * item->size());
//
//			offset += sizeof(T) * item->size();
//		}
//	}
}

template <typename T>
void Batch<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
//	if (continuous) {
//		return;
//	}
//
//	auto device = static_cast<GPUDevice *>(iGradient->device());
//
//	size_t offset = 0;
//
//	for (int i = 0; i < index; ++i) {
//		offset += sizeof(T) * inputs[i]->size();
//	}
//
//	device->copyFromGPUToGPU((byte*)(outputGradient->raw()) + offset, iGradient->raw(), sizeof(T) * iGradient->size());
}

DEEP8_DECLARATION_GPU_FUNC(Batch);

}