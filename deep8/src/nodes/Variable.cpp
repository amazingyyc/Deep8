#include "nodes/Variable.h"

namespace Deep8 {

Variable::Variable(): Node(), updateGradient(false) {
	this->type = NodeType::Variable;
}

Variable::Variable(Tensor &v): Node(), updateGradient(false), value(v)  {
	this->type = NodeType::Variable;
	this->outputShape = this->value.shape;
}

Variable::Variable(Tensor &v, Tensor &g): Node(), updateGradient(true), value(v), gradient(g) {
	DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType(), "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(value.type == gradient.type, "the values and gradient data type must be the same");
	DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the shape of Value and Gradient must be same");

	this->type = NodeType::Variable;
	this->outputShape = this->value.shape;
}

Variable::Variable(Node *input, Shape &shape) : Node(input), updateGradient(false) {
	DEEP8_ARGUMENT_CHECK(1 == inputs.size(), "the Variable Node must need 1 input");

	for (auto i : inputs) {
		DEEP8_ARGUMENT_CHECK(i->outputShape == shape, "the shape of the input is error")
	}

	this->type = NodeType::Variable;
	this->outputShape = shape;
}

Variable::Variable(Node *input, Tensor &v): Node(input), updateGradient(false), value(v) {
	DEEP8_ARGUMENT_CHECK(1 == inputs.size(), "the Variable Node must need 1 input");

	for (auto i : inputs) {
		DEEP8_ARGUMENT_CHECK(i->outputShape == value.shape, "the shape of the input and value must be same")
	}

	this->type = NodeType::Variable;
	this->outputShape = value.shape;
}

Variable::Variable(Node *input, Tensor &v, Tensor &g): Node(input), updateGradient(true), value(v), gradient(g) {
	DEEP8_ARGUMENT_CHECK(1 == inputs.size(), "the Variable Node must need 1 input");

	DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType(), "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(value.type == gradient.type, "the values and gradient data type must be the same");
	DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the shape of Value and Gradient must be same");

	for (auto i : inputs) {
		DEEP8_ARGUMENT_CHECK(i->outputShape == value.shape, "the shape of the inputs, value and gradient must be same")
	}

	this->type = NodeType::Variable;
	this->outputShape = value.shape;
}

void Variable::check() {
}

/**
 * set the Gradient to be 0
 */
void Variable::zeroGradient() {
	DEEP8_ARGUMENT_CHECK(this->updateGradient, "this variable does not update gradient");

	gradient.zero();
}

/**release the gradient*/
void Variable::releaseGradient() {
	DEEP8_ARGUMENT_CHECK(this->updateGradient, "this variable does not update gradient");

	this->gradient.release();
}

/**
 * get the device type
 */
DeviceType Variable::deviceType() {
	return value.deviceType();
}

/**
 * set the gradient be 1 for backward process
 */
void Variable::setGradientOne() {
	DEEP8_ARGUMENT_CHECK(this->updateGradient, "this variable does not update gradient");

	/**set gradient to one*/
	gradient.one();
}

bool Variable::isScalar() {
	if (this->updateGradient) {
		return value.isScalar() && gradient.isScalar();
	} else {
		return value.isScalar();
	}
}

/**feed data to value*/
void Variable::feed(const void *ptr) {
	DEEP8_ARGUMENT_CHECK(nullptr != ptr, "the pointer can not be null");

	if (DeviceType::CPU == value.deviceType()) {
		value.device()->copy(ptr, value.raw(), value.byteCount());
	} else {
#ifdef HAVE_CUDA
		value.device()->copyFromCPUToGPU(ptr, value.raw(), value.byteCount());
#else
		DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
	}
}

/**fetch data from value*/
void Variable::fetch(void *ptr) {
	DEEP8_ARGUMENT_CHECK(nullptr != ptr, "the pointer can not be null");

	if (DeviceType::CPU == value.deviceType()) {
		value.device()->copy(value.raw(), ptr, value.byteCount());
	} else {
#ifdef HAVE_CUDA
		value.device()->copyFromGPUToCPU(value.raw(), ptr, value.byteCount());
#else
		DEEP8_RUNTIME_ERROR("can not call a GPU function without a GPU");
#endif
	}
}

void Variable::forward() {
	/**do nothing*/
}

void Variable::backward() {
	/**do nothing*/
}

}