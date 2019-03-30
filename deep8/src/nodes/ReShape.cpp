#include "nodes/ReShape.h"

namespace Deep8 {

ReShape::ReShape(std::vector<Node *> &inputs, Shape &shape): Function(inputs), reShape(shape) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReShape Function needs 1 input");
}

ReShape::ReShape(std::vector<Node *> &inputs, std::vector<size_t> &list): Function(inputs), reShape(list) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReShape Function needs 1 input");
}

bool ReShape::isShared() {
	return true;
}

Shape ReShape::checkShape(std::vector<Shape> &inputShapes) {
	DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");
	DEEP8_ARGUMENT_CHECK(inputShapes[0].batchSize() == this->reShape.batchSize(), "the input shape is error");

	return Shape(inputShapes[0].batch, this->reShape);
}

ElementType ReShape::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void ReShape::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
}

void ReShape::backward(const std::vector<const Tensor*> &inputs, const Tensor *output, const Tensor *outputGradient, size_t index, Tensor *iGradient) {
}

void ReShape::forward() {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1");
	DEEP8_ARGUMENT_CHECK(1 == this->outputs.size(), "the outputs size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->inputs[0]->type, "the inputs must be Variable type");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");

	auto x = (Variable*)(this->inputs[0]);
	auto y = (Variable*)(this->outputs.first());

	auto xshape = x->shape();

	DEEP8_ARGUMENT_CHECK(xshape.batchSize() == this->reShape.batchSize(), "the input shape is error");

	auto yshape = Shape(xshape.batch, this->reShape);

	y->updateGradient = x->updateGradient;

	/**value*/
	y->value.storage     = x->value.storage;
	y->value.offset      = x->value.offset;
	y->value.elementType = x->value.elementType;
	y->value.shape       = yshape;

	if (x->updateGradient) {
		y->gradient.storage     = x->gradient.storage;
		y->gradient.offset      = x->gradient.offset;
		y->gradient.elementType = x->gradient.elementType;
		y->gradient.shape       = yshape;
	} else {
		/**set the output gradient is empty*/
		y->removeGradient();
	}
}

void ReShape::backward() {
	/**do nothing*/
}



}