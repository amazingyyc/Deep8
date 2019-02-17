#ifndef DEEP8_EXPRESSION_H
#define DEEP8_EXPRESSION_H

#include "nodes/Node.h"
#include "nodes/Variable.h"
#include "nodes/Parameter.h"
#include "nodes/Abs.h"
#include "nodes/Add.h"
#include "nodes/AvgPooling2d.h"
#include "nodes/Conv2d.h"
#include "nodes/CrossEntropy.h"
#include "nodes/DeConv2d.h"
#include "nodes/Divide.h"
#include "nodes/Exp.h"
#include "nodes/L1Norm.h"
#include "nodes/L2Norm.h"
#include "nodes/Linear.h"
#include "nodes/Log.h"
#include "nodes/LogSoftmax.h"
#include "nodes/LReLu.h"
#include "nodes/MatrixMultiply.h"
#include "nodes/MaxPooling2d.h"
#include "nodes/Minus.h"
#include "nodes/Multiply.h"
#include "nodes/ReLu.h"
#include "nodes/ReShape.h"
#include "nodes/Sigmoid.h"
#include "nodes/Softmax.h"
#include "nodes/Square.h"
#include "nodes/Tanh.h"

#include "model/Executor.h"

namespace Deep8 {

class Executor;

class Expression {
public:
    /**the executor*/
    Executor *executor;

    /**the Node pointer that contacted to this Expression*/
    Node *node;

    explicit Expression() : executor(nullptr), node(nullptr) {
	}

	explicit Expression(Executor *exe, Node *n) : executor(exe), node(n) {
	}
};


}


#endif

//namespace Deep8 {
//
//template <typename T>
//class Expression {
//public:
//	/**
//	 * @brief the compute executor
//	 */
//	Executor<T> *executor;
//
//	/**
//	 * @brief the Node pointer that contacted to this Expression
//	 */
//	Node *node;
//
//	explicit Expression() : executor(nullptr), node(nullptr) {
//	}
//
//	explicit Expression(Executor<T> *exe, Node *n) : executor(exe), node(n) {
//	}
//
//	/**feed the data only work for Parameter*/
//	void feed(const void *ptr) {
//		DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call feed function in no-variable node");
//
//		static_cast<Variable<T>*>(node)->feed(ptr);
//	}
//
//	/**fetch data*/
//	void fetch(void *ptr) {
//		DEEP8_ARGUMENT_CHECK(NodeType::Variable == node->type, "can not call fetch function in no-variable node");
//
//		static_cast<Variable<T>*>(node)->fetch(ptr);
//	}
//
//	void forward() {
//		executor->forward(node);
//	}
//
//	void backward() {
//		executor->backward(node);
//	}
//
//	std::string valueString() {
//		if (NodeType::Variable == node->type) {
//			return static_cast<Variable<T> *>(node)->value.valueString();
//		} else {
//			DEEP8_RUNTIME_ERROR("This is not a Variable Expression, can not call valueString function");
//		}
//	}
//
//	Expression<T> add(const Expression<T> &y) {
//		std::vector<Node*> inputs = { node, y.node };
//		return Expression<T>(executor, executor->addFunction(new Add<T>(inputs)));
//	}
//
//	Expression<T> add(T scalar) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new AddScalar<T>(inputs, scalar)));
//	}
//
//	Expression<T> minus(const Expression<T> &y) {
//		std::vector<Node*> inputs = { node, y.node };
//		return Expression<T>(executor, executor->addFunction(new Minus<T>(inputs)));
//	}
//
//	Expression<T> minus(T scalar) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new MinusScalar<T>(inputs, scalar)));
//	}
//
//	Expression<T> multiply(const Expression<T> &y) {
//		std::vector<Node*> inputs = { node, y.node };
//		return Expression<T>(executor, executor->addFunction(new Multiply<T>(inputs)));
//	}
//
//	Expression<T> multiply(T scalar) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
//	}
//
//	Expression<T> divide(const Expression<T> &y) {
//		std::vector<Node*> inputs = { node, y.node };
//		return Expression<T>(executor, executor->addFunction(new Divide<T>(inputs)));
//	}
//
//	Expression<T> divide(T scalar) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new DivideScalar<T>(inputs, scalar)));
//	}
//
//	/**one operand function*/
//	Expression<T> abs() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Abs<T>(inputs)));
//	}
//
//	Expression<T> avgPooling2d(bool covered = false, 
//								size_t filterHeight = 1, 
//								size_t filterWidth = 1, 
//								size_t strideY = 1, 
//								size_t strideX = 1) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new AvgPooling2d<T>(inputs, covered, filterHeight, filterWidth, strideY, strideX)));
//	}
//
//	Expression<T> conv2d(const Expression<T> &filter, 
//							bool covered = false, 
//							size_t strideH = 1, 
//							size_t strideW = 1, 
//							size_t dilationY = 1, 
//							size_t dilationX = 1) {
//		std::vector<Node*> inputs = { node, filter.node };
//
//		return Expression<T>(executor, executor->addFunction(new Conv2d<T>(inputs, covered, strideH, strideW, dilationY, dilationX)));
//	}
//
//	Expression<T> crossEntropy(const Expression<T> &y) {
//		std::vector<Node*> inputs = { node, y.node };
//		return Expression<T>(executor, executor->addFunction(new CrossEntropy<T>(inputs)));
//	}
//
//	Expression<T> deConv2d(const Expression<T> &filter, 
//							bool covered = false, 
//							size_t strideY = 1, 
//							size_t strideX = 1) {
//		std::vector<Node*> inputs = { node, filter.node };
//    	return Expression<T>(executor, executor->addFunction(new DeConv2d<T>(inputs, covered, strideY, strideX)));
//	}
//
//	Expression<T> exp() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Exp<T>(inputs)));
//	}
//
//	Expression<T> l1Norm() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new L1Norm<T>(inputs)));
//	}
//
//	Expression<T> l2Norm() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new L2Norm<T>(inputs)));
//	}
//
//	Expression<T> linear(T a, T b) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Linear<T>(inputs, a, b)));
//	}
//
//	Expression<T> log() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Log<T>(inputs)));
//	}
//
//	Expression<T> logSoftmax(int axis = -1) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new LogSoftmax<T>(inputs, axis)));
//	}
//
//	Expression<T> lRelu(T a) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new LReLu<T>(inputs, a)));
//	}
//
//	Expression<T> matrixMultiply(const Expression<T> &y) {
//		std::vector<Node*> inputs = { node, y.node };
//    	return Expression<T>(executor, executor->addFunction(new MatrixMultiply<T>(inputs)));
//	}
//
//	Expression<T> maxPooling2d(bool covered = false, 
//							size_t filterHeight = 1, 
//							size_t filterWidth = 1, 
//							size_t strideY = 1, 
//							size_t strideX = 1) {
//		std::vector<Node*> inputs = { node };
//    	return Expression<T>(executor, executor->addFunction(new MaxPooling2d<T>(inputs, covered, filterHeight, filterWidth, strideY, strideX)));
//	}
//
//	Expression<T> relu() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new ReLu<T>(inputs)));
//	}
//
//	Expression<T> reShape(Shape &shape) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new ReShape<T>(inputs, shape)));
//	}
//
//	Expression<T> reShape(std::vector<size_t> list) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new ReShape<T>(inputs, list)));
//	}
//
//	Expression<T> sigmoid() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Sigmoid<T>(inputs)));
//	}
//
//	Expression<T> softmax(int axis = -1) {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Softmax<T>(inputs, axis)));
//	}
//
//	Expression<T> square() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Square<T>(inputs)));
//	}
//
//	Expression<T> tanh() {
//		std::vector<Node*> inputs = { node };
//		return Expression<T>(executor, executor->addFunction(new Tanh<T>(inputs)));
//	}
//};
//
///**add operator*/
//template <typename T>
//Expression<T> operator + (const Expression<T> &x, const Expression<T> &y) {
//	std::vector<Node*> inputs = { x.node, y.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new Add<T>(inputs)));
//}
//
//template <typename T>
//Expression<T> operator + (const Expression<T> &x, T scalar) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new AddScalar<T>(inputs, scalar)));
//}
//
//template <typename T>
//Expression<T> operator + (T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new AddScalar<T>(inputs, scalar)));
//}
//
///**minus operator*/
//template <typename T>
//Expression<T> operator - (const Expression<T> &x, const Expression<T> &y) {
//	std::vector<Node*> inputs = { x.node, y.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new Minus<T>(inputs)));
//}
//
//template <typename T>
//Expression<T> operator - (const Expression<T> &x, T scalar) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new MinusScalar<T>(inputs, scalar)));
//}
//
//template <typename T>
//Expression<T> operator - (T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new ScalarMinus<T>(inputs, scalar)));
//}
//
///**
// * the * operator will be a Matrix Multiply not a CWise Multiply
// */
//template <typename T>
//Expression<T> operator * (const Expression<T> &x, const Expression<T> &y) {
//	std::vector<Node*> inputs = { x.node, y.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new MatrixMultiply<T>(inputs)));
//}
//
//template <typename T>
//Expression<T> operator * (const Expression<T> &x, T scalar) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
//}
//
//template <typename T>
//Expression<T> operator * (T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
//}
//
///**divide operator*/
//template <typename T>
//Expression<T> operator / (const Expression<T> &x, const Expression<T> &y) {
//	std::vector<Node*> inputs = { x.node, y.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new Divide<T>(inputs)));
//}
//
//template <typename T>
//Expression<T> operator / (const Expression<T> &x, T scalar) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new DivideScalar<T>(inputs, scalar)));
//}
//
//template <typename T>
//Expression<T> operator / (T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new ScalarDivide<T>(inputs, scalar)));
//}
//
//template <typename T>
//Expression<T> parameter(Executor<T> *executor, std::vector<size_t> list, bool updateGradient = true, void *ptr = nullptr) {
//	return Expression<T>(executor, executor->addParameter(list, updateGradient, ptr));
//}
//
//template <typename T>
//Expression<T> parameter(Executor<T> *executor, size_t batch, std::vector<size_t> list, bool updateGradient = true, void *ptr = nullptr) {
//	return Expression<T>(executor, executor->addParameter(batch, list, updateGradient, ptr));
//}
//
//template <typename T>
//Expression<T> parameter(Executor<T> *executor, Shape &shape, bool updateGradient = true, void *ptr = nullptr) {
//	return Expression<T>(executor, executor->addParameter(shape, updateGradient, ptr));
//}
//
///**scalar + Node*/
//template <typename T>
//Expression<T> add(T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new AddScalar<T>(inputs, scalar)));
//}
//
///**scalar - Node*/
//template <typename T>
//Expression<T> minus(T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new ScalarMinus<T>(inputs, scalar)));
//}
//
///**scalar * Node*/
//template <typename T>
//Expression<T> multiply(T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new MultiplyScalar<T>(inputs, scalar)));
//}
//
///**scalar / Node*/
//template <typename T>
//Expression<T> divide(T scalar, const Expression<T> &x) {
//	std::vector<Node*> inputs = { x.node };
//	return Expression<T>(x.executor, x.executor->addFunction(new ScalarDivide<T>(inputs, scalar)));
//}
//
//}
//
//#endif