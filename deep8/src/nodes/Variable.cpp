#include "math/Constant.h"
#include "math/Gaussian.h"
#include "math/PositiveUnitball.h"
#include "math/Uniform.h"
#include "math/Assign.h"

#include "nodes/Variable.h"

namespace Deep8 {

Variable::Variable(int64_t id, std::string name, Executor *exe): Node(id, name, exe), updateGradient(false) {
	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Executor *exe, Tensor &v): Node(id, name, exe), value(v), updateGradient(false) {
	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Executor *exe, Tensor &v, Tensor &g)
    : Node(id, name, exe), value(v), gradient(g), updateGradient(true) {
	DEEP8_ARGUMENT_CHECK(value.deviceType() == gradient.deviceType(), "the values and gradient must be the same type");
	DEEP8_ARGUMENT_CHECK(value.elementType  == gradient.elementType, "the values and gradient data type must be the same");
	DEEP8_ARGUMENT_CHECK(value.shape == gradient.shape, "the shape of Value and Gradient must be same");

	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Executor *exe, Node *input)
    : Node(id, name, exe, input), updateGradient(false) {
    this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Executor *exe, Node *input, Tensor &v)
    : Node(id, name, exe, input), value(v), updateGradient(false) {
	this->type = NodeType::Variable;
}

Variable::Variable(int64_t id, std::string name, Executor *exe, Node *input, Tensor &v, Tensor &g)
    : Node(id, name, exe, input), value(v), gradient(g), updateGradient(true) {
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

/**
 * set the Gradient to be 0
 */
void Variable::zeroGradient() {
	DEEP8_ARGUMENT_CHECK(updateGradient, "this variable does not have gradient");

    Math::Constant(gradient, 0);
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

void Variable::forward() {
	/**do nothing*/
}

void Variable::backward() {
	/**do nothing*/
}

/**return a string for print value*/
std::string Variable::valueStr() {
    return value.valueStr();
}

/**feed data to value from CPU memory*/
Variable& Variable::feed(const void *ptr) {
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

    return (*this);
}

/**copy memory from value to CPU memory*/
Variable& Variable::fetch(void *ptr) {
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

    return (*this);
}

Variable& Variable::constant(float scalar) {
    Math::Constant(this->value, scalar);

    return (*this);
}

Variable& Variable::zero() {
    return constant(0);
}

Variable& Variable::one() {
    return constant(1);
}

Variable& Variable::gaussian(float mean, float stddev) {
    Math::Gaussian(this->value, mean, stddev);

    return (*this);
}

Variable& Variable::positiveUnitball() {
    Math::positiveUnitball(this->value);

    return (*this);
}

Variable& Variable::random(float lower, float upper) {
    return uniform(lower, upper);
}

Variable& Variable::uniform(float left, float right) {
    Math::Uniform(this->value, left, right);

    return (*this);
}

Variable& Variable::assign(Variable& v) {
    Math::Assign(v.value, this->value);

    return (*this);
}

}