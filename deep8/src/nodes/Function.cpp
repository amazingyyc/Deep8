#include "Function.h"

namespace Deep8 {

FunctionBase::FunctionBase(): Node(), shared(false) {
	this->type = NodeType::Function;
}

FunctionBase::FunctionBase(std::vector<Node*> &inputs): Node(inputs), shared(false) {
	this->type = NodeType::Function;
}

void FunctionBase::check() {
}

void FunctionBase::forward() {
}

void FunctionBase::backward() {
}


template <typename T>
Function<T>::Function(): FunctionBase() {
}

template <typename T>
Function<T>::Function(std::vector<Node*> &inputs): FunctionBase(inputs) {
}

template <typename T>
void Function<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	DEEP8_RUNTIME_ERROR("can not call this forwardCPU by Function class");
}

template <typename T>
void Function<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_RUNTIME_ERROR("can not call this backwardCPU by Function class");
}

#ifdef HAVE_CUDA
template <typename T>
void Function<T>::forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	DEEP8_RUNTIME_ERROR("can not call this forwardGPU by Function class");
}

template <typename T>
void Function<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_RUNTIME_ERROR("can not call this backwardGPU by Function class");
}
#endif

template <typename T>
void Function<T>::forward() {
	DEEP8_ARGUMENT_CHECK(1 == outputs.size(), "the output size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.outputs[0]->type, "the output must be a Variable type");

	auto outputVar = static_cast<Variable<T>*>(this->outputs.outputs[0]);

	DEEP8_ARGUMENT_CHECK(this->outputShape == outputVar->value.shape, "the output shape is error");

	auto outputValue = &(outputVar->value);
	auto deviceType = outputVar->deviceType();

	std::vector<const Tensor<T>*> inputValues;

	for (auto item : inputs) {
		auto inputVar = static_cast<Variable<T>*>(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVar->deviceType(), "the device of input must be same with output");

		inputValues.emplace_back(&(inputVar->value));
	}

	if (DeviceType::CPU == deviceType) {
		this->forwardCPU(inputValues, outputValue);
	} else {
#ifdef HAVE_CUDA
		this->forwardGPU(inputValues, outputValue);
#else
		DEEP8_RUNTIME_ERROR("not have a GPU");
#endif
	}
}

template <typename T>
void Function<T>::backward() {
	DEEP8_ARGUMENT_CHECK(1 == outputs.size(), "the output size must be 1");
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->outputs.outputs[0]->type, "the output must be Variable type");

	auto outputVar = static_cast<Variable<T>*>(this->outputs.outputs[0]);

	auto outputValue    = &(outputVar->value);
	auto outputGradient = &(outputVar->gradient);

	auto deviceType = outputVar->deviceType();

	std::vector<const Tensor<T>*> inputValues;

	for (auto item : inputs) {
		auto inputVar = static_cast<Variable<T>*>(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVar->deviceType(), "the device of the input and output must have same device type");

		inputValues.emplace_back(&(inputVar->value));
	}

	if (DeviceType::CPU == deviceType) {
		for (size_t i = 0; i < inputs.size(); ++i) {
			auto inputVar = static_cast<Variable<T>*>(inputs[i]);

			if (inputVar->updateGradient) {
				this->backwardCPU(inputValues, outputValue, outputGradient, i, &(inputVar->gradient));
			}
		}
	} else {
		for (size_t i = 0; i < inputs.size(); ++i) {
			auto inputVar = static_cast<Variable<T>*>(inputs[i]);

			if (inputVar->updateGradient) {
#ifdef HAVE_CUDA
				this->backwardGPU(inputValues, outputValue, outputGradient, i, &(inputVar->gradient));
#else
				DEEP8_RUNTIME_ERROR("not have a GPU");
#endif
			}
		}
	}
}

DEEP8_DECLARATION_INSTANCE(Function)

}