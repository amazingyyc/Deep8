#include "ReShape.h"

namespace Deep8 {

template <typename T>
void ReShape<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	DEEP8_ARGUMENT_CHECK(reShape.nDims() < MAX_TENSOR_DIMS, "the reShape is error");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.size() == reShape.size(), "the input's Shape size must be equal to reShape size");

	this->outputShape = reShape;
}

template <typename T>
void ReShape<T>::forward() {
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->output->type, "the output must be Variable type");

	auto x = static_cast<Variable<T>*>(this->inputs[0]);
	auto y = static_cast<Variable<T>*>(this->output);

	y->outputShape = this->outputShape;

	y->updateGradient = x->updateGradient;
	y->value.storage  = x->value.storage;
	y->value.offset   = x->value.offset;
	y->value.shape    = this->outputShape;

	if (x->updateGradient) {
		y->gradient.storage = x->gradient.storage;
		y->gradient.offset = x->gradient.offset;
		y->gradient.shape = this->outputShape;
	}
}

template <typename T>
void ReShape<T>::backward() {
}

DEEP8_DECLARATION_INSTANCE(ReShape)

}