#include "Function.h"

namespace Deep8 {

FunctionBase::FunctionBase(): Node(), output(nullptr), shared(false) {
	this->type = NodeType::Function;
}

FunctionBase::FunctionBase(std::vector<Node*> &inputs): Node(inputs), output(nullptr), shared(false) {
	this->type = NodeType::Function;
}

void FunctionBase::check() {
	for (auto item : inputs) {
		DEEP8_ARGUMENT_CHECK(NodeType::Variable == item->type, "the inputs must be Variable type");
	}
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
void Function<T>::backwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_RUNTIME_ERROR("can not call this backwardGPU by Function class");
}
#endif

template <typename T>
void Function<T>::forward() {
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == this->output->type, "the output must be a Variable type");

	auto outputVariable = static_cast<Variable<T>*>(this->output);

	DEEP8_ARGUMENT_CHECK(this->outputShape == outputVariable->value.shape, "the output shape is error");

	auto outputValue = &(outputVariable->value);
	auto deviceType = outputVariable->deviceType();

	std::vector<const Tensor<T>*> inputValues;

	for (auto item : inputs) {
		auto inputVariable = static_cast<Variable<T>*>(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of input must be same with output");

		inputValues.emplace_back(&(inputVariable->value));
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
	DEEP8_ARGUMENT_CHECK(NodeType::Variable == output->type, "the output must be Variable type");

	auto outputVariable = static_cast<Variable<T>*>(output);

	auto outputValue = &(outputVariable->value);
	auto outputGradient = &(outputVariable->gradient);

	auto deviceType = outputVariable->deviceType();

	std::vector<const Tensor<T>*> inputValues;

	for (auto item : inputs) {
		auto inputVariable = static_cast<Variable<T>*>(item);

		DEEP8_ARGUMENT_CHECK(deviceType == inputVariable->deviceType(), "the device of the input and output must have same device type");

		inputValues.emplace_back(&(inputVariable->value));
	}

	if (DeviceType::CPU == deviceType) {
		for (size_t i = 0; i < inputs.size(); ++i) {
			auto inputVariable = static_cast<Variable<T>*>(inputs[i]);

			if (inputVariable->updateGradient) {
				this->backwardCPU(inputValues, outputValue, outputGradient, i, &(inputVariable->gradient));
			}
		}
	} else {
		for (size_t i = 0; i < inputs.size(); ++i) {
			auto inputVariable = static_cast<Variable<T>*>(inputs[i]);

			if (inputVariable->updateGradient) {
#ifdef HAVE_CUDA
				this->backwardGPU(inputValues, outputValue, outputGradient, i, &(inputVariable->gradient));
#else
				DEEP8_RUNTIME_ERROR("not have a GPU");
#endif
			}
		}
	}
}

DEEP8_DECLARATION_INSTANCE(Function)

}