#include "nodes/Function.h"

namespace Deep8 {

Function::Function(std::vector<Node*> &inputs): Node(inputs) {
	this->type = NodeType::Function;
}

/**is the function output shared memory with input*/
bool Function::isShared() {
	return false;
}

/**return the shape and elementtype by the input's*/
Shape Function::checkShape(std::vector<Shape> &inputShapes) {
    return Shape();
}

ElementType Function::checkElementType(std::vector<ElementType> &inputTypes) {
	if (inputTypes.empty()) {
		return ElementType::unknown();
	}

	auto elementType = inputTypes[0];

	for (int i = 1; i < inputTypes.size(); ++i) {
		DEEP8_ARGUMENT_CHECK(elementType == inputTypes[i], "the input's ElementType must be same");
	}

    return elementType;
}

void Function::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	DEEP8_RUNTIME_ERROR("can not call this forward by Function class");
}

void Function::backward(const std::vector<const Tensor*> &inputs, const Tensor *output, const Tensor *outputGradient, size_t index, Tensor *iGradient) {
	DEEP8_RUNTIME_ERROR("can not call this backward by Function class");
}

void Function::forward() {
	/**the inputs and outputs must be Variable*/
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be a Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == outputs.size(), "the output size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be a Variable type");

	auto outputVar = (Variable*)this->outputs.first();

	auto outputValue = &(outputVar->value);
	auto deviceType  = outputVar->deviceType();

	std::vector<const Tensor*> inputValues;

	for (auto item : inputs) {
		auto inputVar = (Variable*)(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVar->deviceType(), "the device of input must be same with output");

		inputValues.emplace_back(&(inputVar->value));
	}

	/**calculate the output*/
	this->forward(inputValues, outputValue);
}

void Function::backward() {
	/**the inputs and outputs must be Variable*/
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be a Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == outputs.size(), "the output size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");

	auto outputVar = (Variable*)(this->outputs.first());

	auto outputValue    = &(outputVar->value);
	auto outputGradient = &(outputVar->gradient);

	auto deviceType = outputVar->deviceType();

	std::vector<const Tensor*> inputValues;

	for (auto item : inputs) {
		auto inputVar = (Variable*)(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVar->deviceType(), "the device of the input and output must have same device type");

		inputValues.emplace_back(&(inputVar->value));
	}

	for (size_t i = 0; i < inputs.size(); ++i) {
		auto inputVar = (Variable*)(inputs[i]);

		if (inputVar->updateGradient) {
			this->backward(inputValues, outputValue, outputGradient, i, &(inputVar->gradient));
		}
	}
}

}