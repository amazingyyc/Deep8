#include "nodes/ReShape.h"

namespace Deep8 {

ReShape::ReShape(std::vector<Node *> &inputs, Shape &shape): Function(inputs) {
	/**the outputShape's batch equal to inputs[0]'s*/
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.batchSize() == shape.batchSize(), "the shape is error");

	this->outputShape = Shape(this->inputs[0]->outputShape.batch, shape);
}

ReShape::ReShape(std::vector<Node *> &inputs, std::vector<size_t> &list): Function(inputs) {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the input size must be 1");

	this->outputShape = Shape(this->inputs[0]->outputShape.batch, list);

	DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.batchSize() == this->outputShape.batchSize(), "the shape is error");
}

bool ReShape::isShared() {
	return true;
}

void ReShape::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
}

void ReShape::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
}

void ReShape::forward() {
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

	/**value*/
	y->value.storage = x->value.storage;
	y->value.offset  = x->value.offset;
	y->value.type    = x->value.type;
	y->value.shape   = this->outputShape;

	if (x->updateGradient) {
		y->gradient.storage = x->gradient.storage;
		y->gradient.offset  = x->gradient.offset;
		y->gradient.type    = x->gradient.type;
		y->gradient.shape   = this->outputShape;
	} else {
		/**set the output gradient is empty*/
		y->releaseGradient();
	}
}

void ReShape::backward() {
	/**do nothing*/
}



}