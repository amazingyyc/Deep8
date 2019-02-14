#include "Function.h"

namespace Deep8 {

Function::Function(): Node() {
	this->type = NodeType::Function;
}

Function::Function(std::vector<Node*> &inputs): Node(inputs) {
	this->type = NodeType::Function;
}

void Function::check() {
}

bool Function::isShared() {
	return false;
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
	DEEP8_ARGUMENT_CHECK(this->outputShape  == this->outputs.first()->outputShape, "the output shape is error");

	auto outputVariable = (Variable*)this->outputs.first();

	auto outputValue = &(outputVariable->value);
	auto deviceType  = outputVariable->deviceType();

	std::vector<const Tensor*> inputValues;

	for (auto item : inputs) {
		auto inputVariable = (Variable*)(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of input must be same with output");

		inputValues.emplace_back(&(inputVariable->value));
	}

	/**calculate the output*/
	this->forward(inputValues, outputValue);

	// /**if all inputs's updateGradient is false than set the output's updateGradient is false and releae the memory*/
	// bool updateGradient = false;

	// for (auto item : this->inputs) {
	// 	auto var = (Variable*)(item);

	// 	if (var->updateGradient) {
	// 		updateGradient = true;
	// 		break;
	// 	}
	// }

	// if (!updateGradient) {
	// 	outputVariable->releaseGradient();
	// 	outputVariable->updateGradient = false;
	// }
}

void Function::backward() {
	/**the inputs and outputs must be Variable*/
	for (auto item : this->inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be a Variable type");
	}

	DEEP8_ARGUMENT_CHECK(1 == outputs.size(), "the output size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.first()->type, "the output must be Variable type");

	auto outputVarialbe = (Variable*)(this->outputs.first());

	auto outputValue    = &(outputVarialbe->value);
	auto outputGradient = &(outputVarialbe->gradient);

	auto deviceType = outputVarialbe->deviceType();

	std::vector<const Tensor*> inputValues;

	for (auto item : inputs) {
		auto inputVariable = (Variable*)(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of the input and output must have same device type");

		inputValues.emplace_back(&(inputVariable->value));
	}

	for (size_t i = 0; i < inputs.size; ++i) {
		auto inputVariable = (Variable*)(inputs[i]);

		if (inputVariable->updateGradient) {
			this->backward(inputValues, outputValue, outputGradient, i, &(inputVariable->gradient));
		}
	}
}

}