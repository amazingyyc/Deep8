#include "math/Constant.h"
#include "math/Gaussian.h"
#include "math/PositiveUnitball.h"
#include "math/Uniform.h"
#include "model/Expression.h"

namespace Deep8 {

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

/**feed the data only work for Parameter*/
void Expression::feed(const void *ptr) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call feed function in no-variable node");

    ((Variable*)node)->feed(ptr);
}

/**fetch data*/
void Expression::fetch(void *ptr) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call fetch function in no-variable node");

    ((Variable*)node)->fetch(ptr);
}

std::string Expression::valueStr() {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call valueStr function in no-variable node");

    return ((Variable*)node)->value.valueStr();
}


/**init the variable's value*/
void Expression::constant(float scalar) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::Constant(variable->value, scalar);
}

void Expression::zero() {
    constant(0);
}

void Expression::one() {
    constant(1);
}

void Expression::gaussian(float mean, float stddev) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::Gaussian(variable->value, mean, stddev);
}

void Expression::positiveUnitball() {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::positiveUnitball(variable->value);
}

void Expression::random(float lower, float upper) {
    uniform(lower, upper);
}

void Expression::uniform(float left, float right) {
    DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "the Node must be a Variable");

    auto variable = (Variable*) node;

    Math::Uniform(variable->value, left, right);
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

Expression Expression::add(const Expression &y) {
    std::vector<Node*> inputs = {node, y.node};
    return Expression(executor, executor->addFunction(new Add(inputs)));
}

Expression Expression::minus(const Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Minus(inputs)));
}

Expression Expression::multiply(const Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Multiply(inputs)));
}

Expression Expression::divide(const Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new Divide(inputs)));
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

Expression Expression::conv2d(const Expression &filter, 
                        bool covered, 
                        size_t strideY, 
                        size_t strideX, 
                        size_t dilationY, 
                        size_t dilationX) {
    std::vector<Node*> inputs = { node, filter.node };

    return Expression(executor, executor->addFunction(new Conv2d(inputs, covered, strideY, strideX, dilationY, dilationX)));
}

Expression Expression::crossEntropy(const Expression &y) {
    std::vector<Node*> inputs = { node, y.node };
    return Expression(executor, executor->addFunction(new CrossEntropy(inputs)));
}

Expression Expression::deConv2d(const Expression &filter, 
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

Expression Expression::l1NormLoss() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new L1NormLoss(inputs)));
}

Expression Expression::l2NormLoss() {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new L2NormLoss(inputs)));
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

Expression Expression::matrixMultiply(const Expression &y) {
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

Expression Expression::reduceMean(int axis, bool keep) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new ReduceMean(inputs, axis, keep)));
}

Expression Expression::reduceSum(int axis, bool keep) {
    std::vector<Node*> inputs = { node };
    return Expression(executor, executor->addFunction(new ReduceSum(inputs, axis, keep)));
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

Expression parameter(Executor *executor, std::vector<size_t> list, bool updateGradient, DType type) {
	return Expression(executor, executor->addVariable(list, type, updateGradient));
}

Expression parameter(Executor *executor, size_t batch, std::vector<size_t> list, bool updateGradient, DType type) {
	return Expression(executor, executor->addVariable(batch, list, type, updateGradient));
}

Expression parameter(Executor *executor, Shape &shape, bool updateGradient, DType type) {
	return Expression(executor, executor->addVariable(shape, type, updateGradient));
}

}