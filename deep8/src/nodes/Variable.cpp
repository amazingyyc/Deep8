#include "math/Constant.h"
#include "math/Gaussian.h"
#include "math/PositiveUnitball.h"
#include "math/Uniform.h"
#include "math/Assign.h"

#include "nodes/Abs.h"
#include "nodes/Add.h"
#include "nodes/AvgPooling2d.h"
#include "nodes/Conv2d.h"
#include "nodes/CrossEntropy.h"
#include "nodes/DeConv2d.h"
#include "nodes/Divide.h"
#include "nodes/Dot.h"
#include "nodes/Exp.h"
#include "nodes/L1Distance.h"
#include "nodes/L1Norm.h"
#include "nodes/L2Distance.h"
#include "nodes/L2Norm.h"
#include "nodes/Linear.h"
#include "nodes/Log.h"
#include "nodes/LogSoftmax.h"
#include "nodes/LReLu.h"
#include "nodes/MatrixMultiply.h"
#include "nodes/MaxPooling2d.h"
#include "nodes/MaxPooling2dWithIndex.h"
#include "nodes/MaxUnPooling2d.h"
#include "nodes/Minus.h"
#include "nodes/Multiply.h"
#include "nodes/PReLu.h"
#include "nodes/ReduceMean.h"
#include "nodes/ReduceSum.h"
#include "nodes/ReLu.h"
#include "nodes/ReShape.h"
#include "nodes/Sigmoid.h"
#include "nodes/Softmax.h"
#include "nodes/Square.h"
#include "nodes/Tanh.h"
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

Variable& Variable::add(Variable &y) {
    std::vector<Node*> inputs = {this, &y};

    return *(this->executor->addFunction(new Add(inputs)));
}

Variable& Variable::minus(Variable &y) {
    std::vector<Node*> inputs = {this, &y};

    return *(this->executor->addFunction(new Minus(inputs)));
}

Variable& Variable::multiply(Variable &y) {
    std::vector<Node*> inputs = { this, &y };

    return *(this->executor->addFunction(new Multiply(inputs)));
}

Variable& Variable::divide(Variable &y) {
    std::vector<Node*> inputs = {this, &y};

    return *(this->executor->addFunction(new Divide(inputs)));
}

Variable& Variable::addConstant(float c) {
    return this->linear(1, c);
}

Variable& Variable::minusConstant(float c) {
    return this->linear(1, -c);
}

Variable& Variable::multiplyConstant(float c) {
    return this->linear(c, 0);
}

Variable& Variable::divideConstant(float c) {
    return this->linear(1.0 / c, 0);
}

Variable& Variable::abs() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Abs(inputs)));
}

Variable& Variable::avgPooling2d(bool covered, 
                                int filterHeight, 
                                int filterWidth, 
                                int strideY, 
                                int strideX) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new AvgPooling2d(inputs, 
                                                            covered, 
                                                            filterHeight, 
                                                            filterWidth, 
                                                            strideY, 
                                                            strideX)));
}

Variable& Variable::conv2d(Variable &filter,
                        bool covered, 
                        int strideY, 
                        int strideX, 
                        int dilationY, 
                        int dilationX) {
    std::vector<Node*> inputs = { this, &filter };

    return *(this->executor->addFunction(new Conv2d(inputs, 
                                                    covered, 
                                                    strideY, 
                                                    strideX, 
                                                    dilationY, 
                                                    dilationX)));
}

Variable& Variable::crossEntropy(Variable &y) {
    std::vector<Node*> inputs = { this, &y };
    
    return *(this->executor->addFunction(new CrossEntropy(inputs)));
}

Variable& Variable::deConv2d( Variable &filter,
                            bool covered, 
                            int strideY, 
                            int strideX) {
    std::vector<Node*> inputs = { this, &filter };

    return *(this->executor->addFunction(new DeConv2d(inputs, 
                                                    covered, 
                                                    strideY, 
                                                    strideX)));
}

Variable& Variable::dot(Variable &y) {
    std::vector<Node*> inputs = { this, &y };

    return *(this->executor->addFunction(new Dot(inputs)));
}

Variable& Variable::exp() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Exp(inputs)));
}

Variable& Variable::l1Distance(Variable &y) {
    std::vector<Node*> inputs = { this, &y };

    return *(this->executor->addFunction(new L1Distance(inputs)));
}

Variable& Variable::l1Norm() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new L1Norm(inputs)));
}

Variable& Variable::l2Distance(Variable &y) {
    std::vector<Node*> inputs = { this, &y };

    return *(this->executor->addFunction(new L2Distance(inputs)));
}

Variable& Variable::l2Norm() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new L2Norm(inputs)));
}

Variable& Variable::linear(float a, float b) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Linear(inputs, a, b)));
}

Variable& Variable::log() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Log(inputs)));
}

Variable& Variable::logSoftmax(int axis) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new LogSoftmax(inputs, axis)));
}

Variable& Variable::lRelu(float a) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new LReLu(inputs, a)));
}

Variable& Variable::matrixMultiply(Variable &y) {
    std::vector<Node*> inputs = { this, &y };

    return *(this->executor->addFunction(new MatrixMultiply(inputs)));
}

Variable& Variable::maxPooling2d( bool covered, 
                                int filterHeight, 
                                int filterWidth, 
                                int strideY, 
                                int strideX) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new MaxPooling2d(inputs, 
                                                covered, 
                                                filterHeight, 
                                                filterWidth, 
                                                strideY, 
                                                strideX)));
}

Variable& Variable::maxPooling2dWithIndex(Variable& index,
                                        bool covered, 
                                        int filterHeight, 
                                        int filterWidth, 
                                        int strideY, 
                                        int strideX) {
    std::vector<Node*> inputs = { this, &index };

    return *(this->executor->addFunction(new MaxPooling2dWithIndex(inputs, 
                                                        covered, 
                                                        filterHeight, 
                                                        filterWidth, 
                                                        strideY, 
                                                        strideX)));
}

Variable& Variable::maxUnPooling2d(Variable& index,
                                    bool covered , 
                                    int filterHeight, 
                                    int filterWidth, 
                                    int strideY, 
                                    int strideX) {
    std::vector<Node*> inputs = { this, &index };

    return *(this->executor->addFunction(new MaxUnPooling2d(inputs, 
                                                            covered, 
                                                            filterHeight, 
                                                            filterWidth, 
                                                            strideY, 
                                                            strideX)));
}

Variable& Variable::pRelu(Variable &p) {
    std::vector<Node*> inputs = { this, &p };

    return *(this->executor->addFunction(new PReLu(inputs)));
}

Variable& Variable::reduceMean(std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new ReduceMean(inputs, axis, keepDims)));
}

Variable& Variable::reduceSum(std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new ReduceSum(inputs, axis, keepDims)));
}

Variable& Variable::relu() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new ReLu(inputs)));
}

Variable& Variable::reShape(Shape &shape) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new ReShape(inputs, shape)));
}

Variable& Variable::reShape(std::vector<size_t> list) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new ReShape(inputs, list)));
}

Variable& Variable::sigmoid() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Sigmoid(inputs)));
}

Variable& Variable::softmax(int axis) {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Softmax(inputs, axis)));
}

Variable& Variable::square() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Square(inputs)));
}

Variable& Variable::tanh() {
    std::vector<Node*> inputs = { this };

    return *(this->executor->addFunction(new Tanh(inputs)));
}

Variable& Variable::l1Loss(Variable &y) {
    return minus(y).abs().reduceMean({}, false);
}

Variable& Variable::l1NormLoss() {
    return l1Norm().reduceMean({}, false);
}

Variable& Variable::l2Loss(Variable &y) {
    return minus(y).square().reduceMean({}, false);
}

Variable& Variable::l2NormLoss() {
    return l2Norm().reduceMean({}, false);
}

Variable& Variable::softmaxCrossEntropyLoss(Variable &y) {
    auto xshape = this->shape();
    auto yshape = y.shape();

    DEEP8_ARGUMENT_CHECK(xshape == yshape, "the shape of SoftmaxCrossEntropyLoss must be same");
    DEEP8_ARGUMENT_CHECK(1 == xshape.nDims, "the shape's ndims must be 1");

    Variable &pred = logSoftmax();

    return y.multiplyConstant(-1).dot(pred).reduceMean({}, false);
}   

}