#include "ReShape.h"

namespace Deep8 {

template <typename T>
ReShape<T>::ReShape(std::vector<Node *> &inputs, Shape &shape): Function<T>(inputs), reShape(shape) {
	this->shared = true;
	check();
}

template <typename T>
ReShape<T>::ReShape(std::vector<Node *> &inputs, std::vector<size_t> &shape): Function<T>(inputs), reShape(shape) {
	this->shared = true;
	check();
}

template <typename T>
void ReShape<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	DEEP8_ARGUMENT_CHECK(reShape.nDims() < MAX_TENSOR_DIMS, "the reShape is error");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.size() == reShape.size(), "the input's Shape size must be equal to reShape size");

	this->outputShape = reShape;
}

template <typename T>
void ReShape<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {}

template <typename T>
void ReShape<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
				 const Tensor<T> *output,
				 const Tensor<T> *outputGradient,
				 size_t index,
				 Tensor<T> *iGradient) {}

#ifdef HAVE_CUDA
template <typename T>
void ReShape<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {}

template <typename T>
void ReShape<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs,
							 const Tensor<T> *output,
							 const Tensor<T> *outputGradient,
							 size_t index,
							 Tensor<T> *iGradient) {}
#endif

template <typename T>
void ReShape<T>::forward() {
	DEEP8_ARGUMENT_CHECK(1 == this->outputs.size(), "the outputs size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.outputs[0]->type, "the output must be Variable type");

	auto x = static_cast<Variable<T>*>(this->inputs[0]);
	auto y = static_cast<Variable<T>*>(this->outputs.outputs[0]);

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