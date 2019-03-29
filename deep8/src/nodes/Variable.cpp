#include "math/Constant.h"
#include "nodes/Variable.h"

namespace Deep8 {

Variable::Variable(int64_t id, std::string name): Node(id, name), updateGradient(false) {
	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Tensor &v): Node(id, name), value(v), updateGradient(false) {
	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Tensor &v, Tensor &g)
    : Node(id, name), value(v), gradient(g), updateGradient(true) {
	DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType(), "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(value.elementType  == gradient.elementType, "the values and gradient data type must be the same");
	DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the shape of Value and Gradient must be same");

	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Node *input, Tensor &v)
    : Node(id, name, input), value(v), updateGradient(false) {
	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Node *input, Tensor &v, Tensor &g)
    : Node(id, name, input), value(v), gradient(g), updateGradient(true) {
	DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType(), "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(value.elementType  == gradient.elementType, "the values and gradient data type must be the same");
	DEEP8_ARGUMENT_CHECK(value.shape        == gradient.shape, "the shape of Value and Gradient must be same");

    this->type = NodeType::Variable;
}

Shape Variable::shape() {
    if (updateGradient) {
        DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the value and gradient shape must be same");
    }

    return value.shape;
}

/**get the element type*/
ElementType Variable::elementType() {
    if (updateGradient) {
        DEEP8_ARGUMENT_CHECK(value.elementType == gradient.elementType, "the value and gradient elementType must be same");
    }

    return value.elementType;
}

/**
 * get the device type
 */
DeviceType Variable::deviceType() {
    if (updateGradient) {
        DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType(), "the value and gradient deviceType must be same");
    }

    return value.deviceType();
}

bool Variable::isScalar() {
    if (updateGradient) {
        return value.isScalar() && gradient.isScalar();
    } else {
        return value.isScalar();
    }
}

/**zero value*/
void Variable::zero() {
    Math::Constant(value, 0);
}

/**
 * set the Gradient to be 0
 */
void Variable::zeroGradient() {
	DEEP8_ARGUMENT_CHECK(updateGradient, "this variable does not have gradient");

    Math::Constant(gradient, 0);
}

/**set value to one*/
void Variable::one() {
    Math::Constant(value, 1);
}

/**set gradient to one*/
void Variable::oneGradient() {
    DEEP8_ARGUMENT_CHECK(updateGradient, "this variable does not have gradient");

    Math::Constant(gradient, 1);
}

/**release the gradient*/
void Variable::removeGradient() {
    if (updateGradient) {
        this->gradient = Tensor();
        this->updateGradient = false;
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