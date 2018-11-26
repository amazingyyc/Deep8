#include "Batch.h"

namespace Deep8 {

template <typename T>
Batch<T>::Batch(std::vector<Node *> &inputs, Shape &outputShape) : Function<T>(inputs), continuous(false) {
	this->outputShape = outputShape;
	check();
}

template <typename T>
void Batch<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(!this->inputs.empty(), "the input can not be empty");

	size_t totalSize = 0;
	for (auto item : this->inputs) {
		totalSize += item->outputShape.size();
	}

	DEEP8_ARGUMENT_CHECK(totalSize == this->outputShape.size(), "the inputs's output shape is error");
}

template <typename T>
void Batch<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice *>(output->device());

	/**if the inputs tensor's memory is continue*/
	continuous = true;
	for (size_t i = 1; i < inputs.size(); ++i) {
		void *prePtr = (byte*)(inputs[i - 1]->raw()) + inputs[i - 1]->size() * sizeof(T);

		if (prePtr != inputs[i]->raw()) {
			continuous = false;
			break;
		}
	}

	if (continuous) {
		/**if the memory is continuous, than do not copy*/
		output->storage = inputs[0]->storage;
		output->offset  = inputs[0]->offset;
		output->shape   = this->outputShape;
	} else {
		/**copy memory*/
		size_t offset = 0;

		for (auto item : inputs) {
			device->copy(item->raw(), (byte*)(output->raw()) + offset, sizeof(T) * item->size());

			offset += sizeof(T) * item->size();
		}
	}
}

template <typename T>
void Batch<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	if (continuous) {
		return;
	}

	auto device = static_cast<CPUDevice *>(iGradient->device());

	size_t offset = 0;

	for (int i = 0; i < index; ++i) {
		offset += sizeof(T) * inputs[i]->size();
	}

	device->copy((byte*)(outputGradient->raw()) + offset, iGradient->raw(), sizeof(T) * iGradient->size());
}

DEEP8_RE_DECLARATION_HALF_FUNC(Batch);
DEEP8_DECLARATION_INSTANCE(Batch)

}