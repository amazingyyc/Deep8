#include "Variable.h"

namespace Deep8 {

VariableBase::VariableBase(): updateGradient(false) {
	this->type = NodeType::Variable;
}

VariableBase::VariableBase(bool update) : updateGradient(update) {
	this->type = NodeType::Variable;
}

VariableBase::VariableBase(Node* input, bool update) : Node(input), updateGradient(update) {
	this->type = NodeType::Variable;
}

VariableBase::VariableBase(std::vector<Node*> &inputs, bool update) : Node(inputs), updateGradient(update) {
	this->type = NodeType::Variable;
}

/**
 * @brief the Variable do nothing in forward and backward process
 */
void VariableBase::forward() {
}

void VariableBase::backward() {
}

template <typename T>
Variable<T>::Variable(): VariableBase(false) {
}

template <typename T>
Variable<T>::Variable(Tensor<T> &v): value(v), VariableBase(false) {
}

template <typename T>
Variable<T>::Variable(Tensor<T> &v, Tensor<T> &g): value(v), gradient(g), VariableBase(true) {
	DEEP8_ARGUMENT_CHECK(this->value.device()->type == this->gradient.device()->type, "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(this->value.shape == this->gradient.shape, "the shape if Value and Gradient must be same");

	this->outputShape = this->value.shape;
}

template <typename T>
Variable<T>::Variable(Node *input, Shape &shape) : VariableBase(input, false) {
	DEEP8_ARGUMENT_CHECK(1 == inputs.size(), "the Variable Node must need 1 input");

	for (auto i : inputs) {
		DEEP8_ARGUMENT_CHECK(nullptr != i, "the input can not be null");
		DEEP8_ARGUMENT_CHECK(i->outputShape == this->outputShape, "the shape of the input, pointer and gradient must be same")
	}

	this->outputShape = shape;
}

template <typename T>
Variable<T>::Variable(Node *input, Tensor<T> &v, Tensor<T> &g): VariableBase(input, true), value(v), gradient(g) {
	DEEP8_ARGUMENT_CHECK(1 == inputs.size(), "the Variable Node must need 1 input");

	DEEP8_ARGUMENT_CHECK(value.device()->type == gradient.device()->type, "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the shape if Value and Gradient must be same");

	for (auto i : inputs) {
		DEEP8_ARGUMENT_CHECK(nullptr != i, "the input can not be null");
		DEEP8_ARGUMENT_CHECK(i->outputShape == value.shape, "the shape of the input, pointer and gradient must be same")
	}

	this->outputShape = value.shape;
}

template <typename T>
void Variable<T>::check() {
}

/**
 * set the Gradient to be 0
 */
template <typename T>
void Variable<T>::zeroGradient() {
	if (this->updateGradient) {
		gradient.zero();
	}
}

/**
 * get the device type
 */
template <typename T>
DeviceType Variable<T>::deviceType() {
	return value.device()->type;
}

/**
 * set the gradient be 1 for backward process
 */
template <typename T>
void Variable<T>::setGradientOne() {
	DEEP8_RUNTIME_ERROR("must definition the function");
}

template <>
void Variable<float>::setGradientOne() {
	DEEP8_ARGUMENT_CHECK(this->updateGradient, "this variable does not update gradient");
	DEEP8_ARGUMENT_CHECK(gradient.isScalar(), "the gradient is  not scalar");

	if (DeviceType::CPU == gradient.device()->type) {
		gradient.data()[0] = 1;
	} else {
#ifdef HAVE_CUDA
		auto device = gradient.device();
		device->copyFromGPUToGPU(device->gpuOneFloat(), gradient.raw(), sizeof(float));
#else
		DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
	}
}

template <>
void Variable<double>::setGradientOne() {
	DEEP8_ARGUMENT_CHECK(this->updateGradient, "this variable does not update gradient");
	DEEP8_ARGUMENT_CHECK(gradient.isScalar(), "the gradient is  not scalar");

	if (DeviceType::CPU == gradient.device()->type) {
		gradient.data()[0] = 1;
	} else {
#ifdef HAVE_CUDA
		auto device = gradient.device();
		device->copyFromGPUToGPU(device->gpuOneDouble(), gradient.raw(), sizeof(double));
#else
		DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
	}
}

#ifdef HAVE_HALF
template <>
void Variable<half>::setGradientOne() {
	DEEP8_ARGUMENT_CHECK(this->updateGradient, "this variable does not update gradient");
	DEEP8_ARGUMENT_CHECK(gradient.isScalar(), "the gradient is  not scalar");

	if (DeviceType::CPU == gradient.device()->type) {
		DEEP8_RUNTIME_ERROR("CPU not support half");
	} else {
#ifdef HAVE_CUDA
		auto device = gradient.device();
		device->copyFromGPUToGPU(device->gpuOneHalf(), gradient.raw(), sizeof(half));
#else
		DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
	}
}
#endif

template <typename T>
bool Variable<T>::isScalar() {
	if (this->updateGradient) {
		return value.isScalar() && gradient.isScalar();
	} else {
		return value.isScalar();
	}
}

template <typename T>
std::string Variable<T>::toString() {
	std::stringstream ss;
	ss << "Value is " << this->value.toString();

	if (this->updateGradient) {
		ss << ", Gradient is " << this->gradient.toString();
	}

	return ss.str();
}

DEEP8_DECLARATION_INSTANCE(Variable)

}