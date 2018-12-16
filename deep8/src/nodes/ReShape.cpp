#include "ReShape.h"

namespace Deep8 {

template <typename T>
ReShape<T>::ReShape(std::vector<Node *> &inputs, Shape &shape): Function<T>(inputs) {
	/**the outputShape's batch equal to inputs[0]'s*/
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.batchSize() == shape.batchSize(), "the shape is error");

	this->outputShape = Shape(this->inputs[0]->outputShape.batch, shape);
}

template <typename T>
ReShape<T>::ReShape(std::vector<Node *> &inputs, std::vector<size_t> &list): Function<T>(inputs) {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");

	this->outputShape = Shape(this->inputs[0]->outputShape.batch, list);

	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.batchSize() == this->outputShape.batchSize(), "the shape is error");
}

template <typename T>
bool ReShape<T>::isShared() {
	return true;
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
	DEEP8_ARGUMENT_CHECK(this->outputShape  == this->outputs.first()->outputShape, "the output shape is error");

	auto x = static_cast<Variable<T>*>(this->inputs[0]);
	auto y = static_cast<Variable<T>*>(this->outputs.first());

	y->outputShape    = this->outputShape;
	y->updateGradient = x->updateGradient;

	/**value*/
	y->value.storage = x->value.storage;
	y->value.offset  = x->value.offset;
	y->value.shape   = this->outputShape;

	if (x->updateGradient) {
		y->gradient.storage = x->gradient.storage;
		y->gradient.offset  = x->gradient.offset;
		y->gradient.shape   = this->outputShape;
	} else {
		/**set the output gradient is empty*/
		y->releaseGradient();
	}
}

template <typename T>
void ReShape<T>::backward() {
	/**do nothing*/
}

DEEP8_DECLARATION_INSTANCE(ReShape)

}