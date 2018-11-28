#ifndef DEEP8_EXPRESSION_H
#define DEEP8_EXPRESSION_H

#include "Node.h"
#include "Variable.h"
#include "Parameter.h"
#include "Abs.h"
#include "Add.h"
#include "AddScalar.h"
#include "AvgPooling2d.h"
#include "Conv2d.h"
#include "DeConv2d.h"
#include "Divide.h"
#include "DivideScalar.h"
#include "Exp.h"
#include "L1Norm.h"
#include "L2Norm.h"
#include "Linear.h"
#include "Log.h"
#include "LReLu.h"
#include "MatrixMultiply.h"
#include "MaxPooling2d.h"
#include "Minus.h"
#include "MinusScalar.h"
#include "Multiply.h"
#include "MultiplyScalar.h"
#include "ReLu.h"
#include "ReShape.h"
#include "ScalarDivide.h"
#include "ScalarMinus.h"
#include "Sigmoid.h"
#include "Softmax.h"
#include "Square.h"
#include "Tanh.h"

#include "Executor.h"
#include "EagerExecutor.h"

namespace Deep8 {

template <typename T>
class Expression {
public:
	/**
	 * @brief the compute executor
	 */
	Executor<T> *executor;

	/**
	 * @brief the Node pointer that contacted to this Expression
	 */
	Node *node;

	explicit Expression() : executor(nullptr), node(nullptr) {
	}

	explicit Expression(Executor<T> *exe, Node *n) : executor(exe), node(n) {
	}

	void forward() {
		executor->forward(node);
	}

	void backward() {
		executor->backward(node);
	}

	std::string valueString() {
		if (NodeType::Variable == node->type) {
			return static_cast<Variable<T> *>(node)->value.valueString();
		} else {
			DEEP8_RUNTIME_ERROR("This is not a Variable Expression, can not call valueString function");
		}
	}

	/**one operand function*/
	Expression<T> abs() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Abs<T>(inputs)));
	}

	Expression<T> exp() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Exp<T>(inputs)));
	}

	Expression<T> l1Norm() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new L1Norm<T>(inputs)));
	}

	Expression<T> l2Norm() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new L2Norm<T>(inputs)));
	}

	Expression<T> linear(T a, T b) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Linear<T>(inputs, a, b)));
	}

	Expression<T> log() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Log<T>(inputs)));
	}

	Expression<T> lReLu(T a) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new LReLu<T>(inputs, a)));
	}

	Expression<T> reLu() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new ReLu<T>(inputs)));
	}

	Expression<T> reShape(Shape &shape) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new ReShape<T>(inputs, shape)));
	}

	Expression<T> reShape(std::vector<size_t> list) {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new ReShape<T>(inputs, list)));
	}

	Expression<T> sigmoid() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Sigmoid<T>(inputs)));
	}

	Expression<T> softmax() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Softmax<T>(inputs)));
	}

	Expression<T> square() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Square<T>(inputs)));
	}

	Expression<T> tanh() {
		std::vector<Node*> inputs = { node };
		return Expression<T>(executor, executor->addFunction(new Tanh<T>(inputs)));
	}
};

/**add operator*/
template <typename T>
Expression<T> operator + (const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new Add<T>(inputs)));
}

template <typename T>
Expression<T> operator + (const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new AddScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> operator + (T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new AddScalar<T>(inputs, scalar)));
}

/**minus operator*/
template <typename T>
Expression<T> operator - (const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new Minus<T>(inputs)));
}

template <typename T>
Expression<T> operator - (const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new MinusScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> operator - (T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new ScalarMinus<T>(inputs, scalar)));
}

/**
 * the * operator will be a Matrix Multiply not a CWise Multiply
 */
template <typename T>
Expression<T> operator * (const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new MatrixMultiply<T>(inputs)));
}

template <typename T>
Expression<T> operator * (const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> operator * (T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
}

/**divide operator*/
template <typename T>
Expression<T> operator / (const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new Divide<T>(inputs)));
}

template <typename T>
Expression<T> operator / (const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new DivideScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> operator / (T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new ScalarDivide<T>(inputs, scalar)));
}

template <typename T>
Expression<T> parameter(Executor<T> *executor, std::vector<size_t> list, bool updateGradient = true, void *ptr = nullptr) {
	return Expression<T>(executor, executor->addParameter(list, updateGradient, ptr));
}

template <typename T>
Expression<T> parameter(Executor<T> *executor, size_t batch, std::vector<size_t> list, bool updateGradient = true, void *ptr = nullptr) {
	return Expression<T>(executor, executor->addParameter(batch, list, updateGradient, ptr));
}

template <typename T>
Expression<T> parameter(Executor<T> *executor, Shape &shape, bool updateGradient = true, void *ptr = nullptr) {
	return Expression<T>(executor, executor->addParameter(shape, updateGradient, ptr));
}

/**function*/
template <typename T>
Expression<T> add(const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new Add<T>(inputs)));
}

template <typename T>
Expression<T> add(const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new AddScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> add(T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new AddScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> minus(const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new Minus<T>(inputs)));
}

template <typename T>
Expression<T> minus(const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new MinusScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> minus(T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new ScalarMinus<T>(inputs, scalar)));
}

template <typename T>
Expression<T> multiply(const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new Multiply<T>(inputs)));
}

template <typename T>
Expression<T> multiply(const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> multiply(T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> divide(const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
	return Expression<T>(x.executor, x.executor->addFunction(new Divide<T>(inputs)));
}

template <typename T>
Expression<T> divide(const Expression<T> &x, T scalar) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new DivideScalar<T>(inputs, scalar)));
}

template <typename T>
Expression<T> divide(T scalar, const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
	return Expression<T>(x.executor, x.executor->addFunction(new ScalarDivide<T>(inputs, scalar)));
}

template <typename T>
Expression<T> abs(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Abs<T>(inputs)));
}

template <typename T>
Expression<T> avgPooling2d(const Expression<T> &x, bool covered = false, size_t filterHeight = 1, size_t filterWidth = 1, size_t strideY = 1, size_t strideX = 1) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new AvgPooling2d<T>(inputs, covered, filterHeight, filterWidth, strideY, strideX)));
}

template <typename T>
Expression<T> conv2d(const Expression<T> &x, const Expression<T> &y, bool covered = false, size_t strideH = 1, size_t strideW = 1, size_t dilationY = 1, size_t dilationX = 1) {
	std::vector<Node*> inputs = { x.node, y.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Conv2d<T>(inputs, covered, strideH, strideW, dilationY, dilationX)));
}

template <typename T>
Expression<T> deConv2d(const Expression<T> &x, const Expression<T> &y, bool covered = false, size_t strideY = 1, size_t strideX = 1) {
	std::vector<Node*> inputs = { x.node, y.node };
    return Expression<T>(x.executor, x.executor->addFunction(new DeConv2d<T>(inputs, covered, strideY, strideX)));
}

template <typename T>
Expression<T> exp(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Exp<T>(inputs)));
}

template <typename T>
Expression<T> l1Norm(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new L1Norm<T>(inputs)));
}

template <typename T>
Expression<T> l2Norm(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new L2Norm<T>(inputs)));
}

template <typename T>
Expression<T> linear(const Expression<T> &x, T a, T b) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Linear<T>(inputs, a, b)));
}

template <typename T>
Expression<T> log(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Log<T>(inputs)));
}

template <typename T>
Expression<T> lReLu(const Expression<T> &x, T a) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new LReLu<T>(inputs, a)));
}

template <typename T>
Expression<T> matrixMultiply(const Expression<T> &x, const Expression<T> &y) {
	std::vector<Node*> inputs = { x.node, y.node };
    return Expression<T>(x.executor, x.executor->addFunction(new MatrixMultiply<T>(inputs)));
}

template <typename T>
Expression<T> maxPooling2d(const Expression<T> &x, bool covered = false, size_t filterHeight = 1, size_t filterWidth = 1, size_t strideY = 1, size_t strideX = 1) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new MaxPooling2d<T>(inputs, covered, filterHeight, filterWidth, strideY, strideX)));
}

template <typename T>
Expression<T> reLu(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new ReLu<T>(inputs)));
}

template <typename T>
Expression<T> reShape(const Expression<T> &x, Shape &shape) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new ReShape<T>(inputs, shape)));
}

template <typename T>
Expression<T> reShape(const Expression<T> &x, std::vector<size_t> list) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new ReShape<T>(inputs, list)));
}

template <typename T>
Expression<T> sigmoid(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Sigmoid<T>(inputs)));
}

template <typename T>
Expression<T> softmax(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Softmax<T>(inputs)));
}

template <typename T>
Expression<T> square(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Square<T>(inputs)));
}

template <typename T>
Expression<T> tanh(const Expression<T> &x) {
	std::vector<Node*> inputs = { x.node };
    return Expression<T>(x.executor, x.executor->addFunction(new Tanh<T>(inputs)));
}



}

#endif