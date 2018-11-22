#include "UnBatch.h"

namespace Deep8 {

template <typename T>
UnBatch<T>::UnBatch(std::vector<Node*> &inputs, size_t o, Shape &shape) : Function<T>(inputs), offset(o) {
	this->outputShape = shape;
	check();
}

template <typename T>
void UnBatch<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.size() >= offset + this->outputShape.size(), "the shape is error");
}

template <typename T>
void UnBatch<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	output->storage = inputs[0]->storage;
	output->offset  = inputs[0]->offset + sizeof(T) * offset;
}

template <typename T>
void UnBatch<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	/**do nothing*/
}

DEEP8_RE_DECLARATION_HALF_FUNC(UnBatch);
DEEP8_DECLARATION_INSTANCE(UnBatch)

}