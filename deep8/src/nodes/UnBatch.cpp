#include "UnBatch.h"

namespace Deep8 {

template <typename T>
UnBatch<T>::UnBatch(std::vector<Node*> &inputs, size_t o, Shape &shape) : Function<T>(inputs), offset(o) {
	this->shared      = true;
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
void UnBatch<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {}

template <typename T>
void UnBatch<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {}

#ifdef HAVE_CUDA
template <typename T>
void UnBatch<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {}

template <typename T>
void UnBatch<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {}
#endif

template <typename T>
void UnBatch<T>::forward() {
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == this->outputs.size(), "the outputs size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");

	auto x = static_cast<Variable<T>*>(this->inputs[0]);
	auto y = static_cast<Variable<T>*>(this->outputs.first());

	y->outputShape    = this->outputShape;
	y->updateGradient = x->updateGradient;

	y->value.storage = x->value.storage;
	y->value.offset  = x->value.offset + sizeof(T) * offset;
	y->value.shape   = this->outputShape;

	if (x->updateGradient) {
		y->gradient.storage = x->gradient.storage;
		y->gradient.offset  = x->gradient.offset + sizeof(T) * offset;
		y->gradient.shape   = this->outputShape;
	}
}

template <typename T>
void UnBatch<T>::backward() {
}

DEEP8_RE_DECLARATION_HALF_FUNC(UnBatch);
DEEP8_DECLARATION_INSTANCE(UnBatch)

}