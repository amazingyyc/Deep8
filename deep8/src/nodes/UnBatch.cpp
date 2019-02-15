#include "nodes/UnBatch.h"

namespace Deep8 {

UnBatch::UnBatch(std::vector<Node*> &inputs, size_t o, Shape &shape) : Function(inputs), offset(o) {
	this->outputShape = shape;
	check();
}

void UnBatch::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.size() >= offset + this->outputShape.size(), "the shape is error");
}

bool UnBatch::isShared() {
	return true;
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
	DEEP8_ARGUMENT_CHECK(this->outputShape  == this->outputs.first()->outputShape, "the output shape is error");

	auto x = (Variable*)(this->inputs[0]);
	auto y = (Variable*)(this->outputs.first());

	y->outputShape    = this->outputShape;
	y->updateGradient = x->updateGradient;

	y->value.storage = x->value.storage;
	y->value.offset  = x->value.offset + x->value.type.byteWidth * offset;
	y->value.type    = x->value.type;
	y->value.shape   = this->outputShape;

	if (x->updateGradient) {
		y->gradient.storage = x->gradient.storage;
		y->gradient.offset  = x->gradient.offset + x->gradient.type.byteWidth * offset;
		y->gradient.type    = x->gradient.type;
		y->gradient.shape   = this->outputShape;
	} else {
		/**set the output gradient is empty to save the memory*/
		y->releaseGradient();
	}
}

void UnBatch::backward() {
	/**do nothing*/
}



}