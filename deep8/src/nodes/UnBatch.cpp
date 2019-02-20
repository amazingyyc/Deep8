#include "nodes/UnBatch.h"

namespace Deep8 {

UnBatch::UnBatch(std::vector<Node*> &inputs, size_t o, Shape &shape) : Function(inputs), offset(o) {
	this->shape = shape;
    this->isShared = true;

    DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape.size() >= offset + this->shape.size(), "the shape is error");
}

void UnBatch::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	
    this->elementType = this->inputs[0]->elementType;
}

void UnBatch::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
}

void UnBatch::backward(const std::vector<const Tensor*> &inputs, 
						const Tensor *output, 
						const Tensor *outputGradient, 
						size_t index, 
						Tensor *iGradient) {
}

void UnBatch::forward() {
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == this->outputs.size(), "the outputs size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");
	DEEP8_ARGUMENT_CHECK(this->shape == this->outputs.first()->shape, "the output shape is error");

	auto x = (Variable*)(this->inputs[0]);
	auto y = (Variable*)(this->outputs.first());

	y->shape          = this->shape;
	y->updateGradient = this->updateGradient;
    y->elementType    = this->elementType;

	y->value.storage     = x->value.storage;
	y->value.offset      = x->value.offset + x->value.elementType.byteWidth * this->offset;
	y->value.elementType = x->value.elementType;
	y->value.shape       = this->shape;

	if (x->updateGradient) {
		y->gradient.storage     = x->gradient.storage;
		y->gradient.offset      = x->gradient.offset + x->gradient.elementType.byteWidth * this->offset;
		y->gradient.elementType = x->gradient.elementType;
		y->gradient.shape       = this->shape;
	} else {
		/**set the output gradient is empty to save the memory*/
		y->releaseGradient();
	}
}

void UnBatch::backward() {
	/**do nothing*/
}



}