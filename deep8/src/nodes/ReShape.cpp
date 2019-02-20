#include "nodes/ReShape.h"

namespace Deep8 {

ReShape::ReShape(std::vector<Node *> &inputs, Shape &shape): Function(inputs) {
	/**the outputShape's batch equal to inputs[0]'s*/
	DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape.batchSize() == shape.batchSize(), "the shape is error");

	this->shape    = Shape(this->inputs[0]->shape.batch, shape);
    this->isShared = true;
}

ReShape::ReShape(std::vector<Node *> &inputs, std::vector<size_t> &list): Function(inputs) {
	this->shape    = Shape(this->inputs[0]->shape.batch, list);
    this->isShared = true;

	DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape.batchSize() == this->shape.batchSize(), "the shape is error");
}

void ReShape::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReShape Function needs only 1 input");

    this->elementType = this->inputs[0]->elementType;
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

	auto x = (Variable*)(this->inputs[0]);
	auto y = (Variable*)(this->outputs.first());

	y->shape          = this->shape;
	y->updateGradient = this->updateGradient;
    y->elementType    = this->elementType;

	/**value*/
	y->value.storage     = x->value.storage;
	y->value.offset      = x->value.offset;
	y->value.elementType = x->value.elementType;
	y->value.shape       = this->shape;

	if (x->updateGradient) {
		y->gradient.storage     = x->gradient.storage;
		y->gradient.offset      = x->gradient.offset;
		y->gradient.elementType = x->gradient.elementType;
		y->gradient.shape       = this->shape;
	} else {
		/**set the output gradient is empty*/
		y->releaseGradient();
	}
}

void ReShape::backward() {
	/**do nothing*/
}



}