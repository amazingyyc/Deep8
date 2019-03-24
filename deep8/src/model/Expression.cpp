#include "math/Constant.h"
#include "math/Gaussian.h"
#include "math/PositiveUnitball.h"
#include "math/Uniform.h"
#include "model/Expression.h"

namespace Deep8 {

Expression parameter(Executor *executor, std::vector<size_t> list, bool updateGradient, DType type) {
	return Expression(executor, executor->addVariable(list, type, updateGradient));
}

Expression parameter(Executor *executor, size_t batch, std::vector<size_t> list, bool updateGradient, DType type) {
	return Expression(executor, executor->addVariable(batch, list, type, updateGradient));
}

Expression parameter(Executor *executor, Shape &shape, bool updateGradient, DType type) {
	return Expression(executor, executor->addVariable(shape, type, updateGradient));
}

/**create input paramater, the input parameter will deleted after bakward, it is temp*/
Expression inputParameter(Executor *executor, std::vector<size_t> list, DType type) {
    return Expression(executor, executor->addVariable(list, type, false, false));
}

Expression inputParameter(Executor *executor, size_t batch, std::vector<size_t> list, DType type) {
    return Expression(executor, executor->addVariable(batch, list, type, false, false));
}

Expression inputParameter(Executor *executor, Shape &shape, DType type) {
    return Expression(executor, executor->addVariable(shape, type, false, false));
}

Expression::Expression() : executor(nullptr), node(nullptr) {
}

Expression::Expression(Executor *exe, Node *n) : executor(exe), node(n) {
}

void Expression::forward() {
    executor->forward(node);
}

void Expression::backward() {
    executor->backward(node);
}

std::string Expression::valueStr() {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call valueStr function in no-variable node");

    return ((Variable*)node)->value.valueStr();
}

/**feed the data only work for Parameter*/
Expression Expression::feed(const void *ptr) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call feed function in no-variable node");

    ((Variable*)node)->feed(ptr);

    return *this;
}

/**fetch data*/
Expression Expression::fetch(void *ptr) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call fetch function in no-variable node");

    ((Variable*)node)->fetch(ptr);

    return *this;
}

/**init the variable's value*/
Expression Expression::constant(float scalar) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::Constant(variable->value, scalar);

    return (*this);
}

Expression Expression::zero() {
    constant(0);

    return (*this);
}

Expression Expression::one() {
    constant(1);

    return (*this);
}

Expression Expression::gaussian(float mean, float stddev) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::Gaussian(variable->value, mean, stddev);

    return (*this);
}

Expression Expression::positiveUnitball() {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::positiveUnitball(variable->value);

    return (*this);
}

Expression Expression::random(float lower, float upper) {
    uniform(lower, upper);

    return (*this);
}

Expression Expression::uniform(float left, float right) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::Uniform(variable->value, left, right);

    return (*this);
}

Expression Expression::operator + (const Expression &y) const {
    std::vector<Node*> inputs = {node, y.node};
    return Expression(executor, executor->addFunction(new Add(inputs)));
}

Expression Expression::operator - (const Expression &y) const {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Minus(inputs)));
}

Expression Expression::operator * (const Expression &y) const {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new MatrixMultiply(inputs)));
}

Expression Expression::operator / (const Expression &y) const {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Divide(inputs)));
}

Expression Expression::operator + (float c) const {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Linear(inputs, 1.0, c)));
}

Expression Expression::operator - (float c) const {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Linear(inputs, 1.0, -c)));
}

Expression Expression::operator * (float c) const {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Linear(inputs, c, 0.0)));
}

Expression Expression::operator / (float c) const {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Linear(inputs, 1.0 / c, 0.0)));
}

Expression Expression::add(Expression &y) {
    std::vector<Node*> inputs = {node, y.node};
    return Expression(executor, executor->addFunction(new Add(inputs)));
}

Expression Expression::minus(Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Minus(inputs)));
}

Expression Expression::multiply(Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Multiply(inputs)));
}

Expression Expression::divide(Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Divide(inputs)));
}

Expression Expression::addConstant(float c) {
    return this->linear(1.0, c);
}

Expression Expression::minusConstant(float c) {
    return this->linear(1.0, -c);
}

Expression Expression::multiplyConstant(float c) {
    return this->linear(c, 0.0);
}

Expression Expression::divideConstant(float c) {
    return this->linear(1.0 / c, 0.0);
}

Expression Expression::dot(Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Dot(inputs)));
}

/**one operand function*/
Expression Expression::abs() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Abs(inputs)));
}

Expression Expression::avgPooling2d(bool covered, 
                            size_t filterHeight, 
                            size_t filterWidth, 
                            size_t strideY, 
                            size_t strideX) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new AvgPooling2d(inputs, covered, filterHeight, filterWidth, strideY, strideX)));
}

Expression Expression::conv2d(Expression &filter, 
                        bool covered, 
                        size_t strideY, 
                        size_t strideX, 
                        size_t dilationY, 
                        size_t dilationX) {
    std::vector<Node*> inputs = { node, filter.node };

    return Expression(executor, executor->addFunction(new Conv2d(inputs, covered, strideY, strideX, dilationY, dilationX)));
}

Expression Expression::crossEntropy(Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new CrossEntropy(inputs)));
}

Expression Expression::deConv2d(Expression &filter, 
                        bool covered, 
                        size_t strideY, 
                        size_t strideX) {
    std::vector<Node*> inputs = { node, filter.node };
    return Expression(executor, executor->addFunction(new DeConv2d(inputs, covered, strideY, strideX)));
}

Expression Expression::exp() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Exp(inputs)));
}

Expression Expression::l1Distance(Expression &y) {
    std::vector<Node*> inputs = { node, y.node};
    return Expression(executor, executor->addFunction(new L1Distance(inputs)));
}

Expression Expression::l1Norm() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new L1Norm(inputs)));
}

Expression Expression::l2Distance(Expression &y) {
    std::vector<Node*> inputs = { node, y.node};
    return Expression(executor, executor->addFunction(new L2Distance(inputs)));
}

Expression Expression::l2Norm() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new L2Norm(inputs)));
}

Expression Expression::linear(float a, float b) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Linear(inputs, a, b)));
}

Expression Expression::log() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Log(inputs)));
}

Expression Expression::logSoftmax(int axis) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new LogSoftmax(inputs, axis)));
}

Expression Expression::lRelu(float a) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new LReLu(inputs, a)));
}

Expression Expression::matrixMultiply(Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new MatrixMultiply(inputs)));
}

Expression Expression::maxPooling2d(bool covered, 
                        size_t filterHeight, 
                        size_t filterWidth, 
                        size_t strideY, 
                        size_t strideX) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new MaxPooling2d(inputs, covered, filterHeight, filterWidth, strideY, strideX)));
}

Expression Expression::pRelu(Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new PReLu(inputs)));
}

Expression Expression::reduceMean(std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new ReduceMean(inputs, axis, keepDims)));
}

Expression Expression::reduceSum(std::vector<int> axis, bool keepDims) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new ReduceSum(inputs, axis, keepDims)));
}

Expression Expression::relu() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new ReLu(inputs)));
}

Expression Expression::reShape(Shape &shape) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new ReShape(inputs, shape)));
}

Expression Expression::reShape(std::vector<size_t> list) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new ReShape(inputs, list)));
}

Expression Expression::sigmoid() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Sigmoid(inputs)));
}

Expression Expression::softmax(int axis) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Softmax(inputs, axis)));
}

Expression Expression::square() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Square(inputs)));
}

Expression Expression::tanh() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new Tanh(inputs)));
}

Expression Expression::l1DistanceLoss(Expression &y) {
    return this->l1Distance(y).reduceMean({}, false);
}

Expression Expression::l1NormLoss() {
    return this->l1Norm().reduceMean({}, false);
}

Expression Expression::l2DistanceLoss(Expression &y) {
    return this->l2Distance(y).reduceMean({}, false);
}

Expression Expression::l2NormLoss() {
    return this->l2Norm().reduceMean({}, false);
}

Expression Expression::softmaxCrossEntropyLoss(Expression &y) {
    DEEP8_ARGUMENT_CHECK(this->node->shape == y.node->shape, "the shape of SoftmaxCrossEntropyLoss must be same");
    DEEP8_ARGUMENT_CHECK(1 == this->node->shape.nDims, "the shape's ndims must be 1");

    auto pred = this->logSoftmax();
    return y.linear(-1).dot(pred).reduceMean({}, false);
}


}