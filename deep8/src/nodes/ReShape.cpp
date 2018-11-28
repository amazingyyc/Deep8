#include "ReShape.h"

namespace Deep8 {

template <typename T>
ReShape<T>::ReShape(std::vector<Node *> &inputs, Shape &shape): Function<T>(inputs) {
	this->outputShape = shape;
	this->shared = true;
	check();
}

template <typename T>
ReShape<T>::ReShape(std::vector<Node *> &inputs, std::vector<size_t> &list): Function<T>(inputs) {
	this->outputShape.reShape(list);
	this->shared = true;
	check();
}

template <typename T>
void ReShape<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	DEEP8_ARGUMENT_CHECK(this->outputShape.nDims() <= MAX_TENSOR_DIMS, "the reShape is error");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.size() == this->outputShape.size(), "the input's Shape size must be equal to reShape size");
}

template <typename T>
void ReShape<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {}

template <typename T>
void ReShape<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {}

#ifdef HAVE_CUDA
template <typename T>
void ReShape<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {}

template <typename T>
void ReShape<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {}
#endif

template <typename T>
void ReShape<T>::forward() {
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == this->outputs.size(), "the outputs size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");

	auto x = static_cast<Variable<T>*>(this->inputs[0]);
	auto y = static_cast<Variable<T>*>(this->outputs.first());

	y->outputShape = this->outputShape;
	y->updateGradient = x->updateGradient;

	y->value.storage  = x->value.storage;
	y->value.offset   = x->value.offset;
	y->value.shape    = this->outputShape;

	if (x->updateGradient) {
		y->gradient.storage = x->gradient.storage;
		y->gradient.offset  = x->gradient.offset;
		y->gradient.shape   = this->outputShape;
	}
}

template <typename T>
void ReShape<T>::backward() {
}

DEEP8_DECLARATION_INSTANCE(ReShape)

}