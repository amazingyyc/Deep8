#ifndef DEEP8_EXPRESSION_H
#define DEEP8_EXPRESSION_H

#include "Executor.h"
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
#include "Pow.h"
#include "ReLu.h"
#include "ReShape.h"
#include "ScalarDivide.h"
#include "ScalarMinus.h"
#include "Sigmoid.h"
#include "Softmax.h"
#include "Square.h"
#include "SumElements.h"
#include "TanH.h"

namespace Deep8 {

/**
 * the Expression class to build the Executor is reference from dynet
 * a Expression represent a Node (Parameter, Variable) it used to build mathematical expression;
 */
class Expression {
public:
    /**
     * @brief the compute executor
     */
    Executor *executor;

    /**
     * @brief the Node pointer that contacted to this Expression
     */
    Node *node;

    explicit Expression(): executor(nullptr), node(nullptr) {
    }

    explicit Expression(Executor *exe, Node *n): executor(exe), node(n) {
    }
};

template <typename T>
Expression parameter(Executor *executor, std::initializer_list<size_t> list) {
    return Expression(executor, executor->addParameter<T>(list));
}

template <typename T>
Expression parameter(Executor *executor, Shape &shape) {
    return Expression(executor, executor->addParameter<T>(shape));
}

template <typename T>
Expression inputParameter(Executor *executor, std::initializer_list<size_t> list) {
    return Expression(executor, executor->addInputParameter<T>(list));
}

template <typename T>
Expression inputParameter(Executor *executor, Shape &shape) {
    return Expression(executor, executor->addInputParameter<T>(shape));
}

template <typename T>
Expression add(const Expression &x, const Expression &y) {
    return Expression(x.executor, x.executor->addFunction<T, Add<T>>({x.node, y.node}));
}

template <typename T>
Expression add(const Expression &x, T scalar) {
    return Expression(x.executor, x.executor->addFunction<T, AddScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression add(T scalar, const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, AddScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression minus(const Expression &x, const Expression &y) {
    return Expression(x.executor, x.executor->addFunction<T, Minus<T>>({x.node, y.node}));
}

template <typename T>
Expression minus(const Expression &x, T scalar) {
    return Expression(x.executor, x.executor->addFunction<T, MinusScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression minus(T scalar, const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, ScalarMinus<T>>({x.node}, scalar));
}

template <typename T>
Expression multiply(const Expression &x, const Expression &y) {
    return Expression(x.executor, x.executor->addFunction<T, Multiply<T>>({x.node, y.node}));
}

template <typename T>
Expression multiply(const Expression &x, T scalar) {
    return Expression(x.executor, x.executor->addFunction<T, MultiplyScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression multiply(T scalar, const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, MultiplyScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression divide(const Expression &x, const Expression &y) {
    return Expression(x.executor, x.executor->addFunction<T, Divide<T>>({x.node, y.node}));
}

template <typename T>
Expression divide(const Expression &x, T scalar) {
    return Expression(x.executor, x.executor->addFunction<T, DivideScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression divide(T scalar, const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, ScalarDivide<T>>({x.node}, scalar));
}

template <typename T>
Expression abs(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, Abs<T>>({x.node}));
}

template <typename T>
Expression avgPooling2d(const Expression &x, bool covered = false, size_t filterHeight = 1, size_t filterWidth = 1, size_t strideY = 1, size_t strideX = 1) {
    return Expression(x.executor, x.executor->addFunction<T, AvgPooling2d<T>>({x.node}, covered, filterHeight, filterWidth, strideY, strideX));
}

template <typename T>
Expression conv2d(const Expression &x, const Expression &y, bool covered = false, size_t strideH = 1, size_t strideW = 1, size_t dilationY = 1, size_t dilationX = 1) {
    return Expression(x.executor, x.executor->addFunction<T, Conv2d<T>>({x.node, y.node}, covered, strideH, strideW, dilationY, dilationX));
}

template <typename T>
Expression deConv2d(const Expression &x, const Expression &y, bool covered = false, size_t strideY = 1, size_t strideX = 1) {
    return Expression(x.executor, x.executor->addFunction<T, DeConv2d<T>>({x.node, y.node}, covered, strideY, strideX));
}

template <typename T>
Expression exp(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, Exp<T>>({x.node}));
}

template <typename T>
Expression l1Norm(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, L1Norm<T>>({x.node}));
}

template <typename T>
Expression l2Norm(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, L2Norm<T>>({x.node}));
}

template <typename T>
Expression linear(const Expression &x, T a, T b) {
    return Expression(x.executor, x.executor->addFunction<T, Linear<T>>({x.node}, a, b));
}

template <typename T>
Expression log(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, Log<T>>({x.node}));
}

template <typename T>
Expression lReLu(const Expression &x, T a) {
    return Expression(x.executor, x.executor->addFunction<T, LReLu<T>>({x.node}, a));
}

template <typename T>
Expression matrixMultiply(const Expression &x, const Expression &y) {
    return Expression(x.executor, x.executor->addFunction<T, MatrixMultiply<T>>({x.node, y.node}));
}

template <typename T>
Expression maxPooling2d(const Expression &x, bool covered = false, size_t filterHeight = 1, size_t filterWidth = 1, size_t strideY = 1, size_t strideX = 1) {
    return Expression(x.executor, x.executor->addFunction<T, MaxPooling2d<T>>({x.node}, covered, filterHeight, filterWidth, strideY, strideX));
}

template <typename T>
Expression pow(const Expression &x, T scalar) {
    return Expression(x.executor, x.executor->addFunction<T, Pow<T>>({x.node}, scalar));
}

template <typename T>
Expression reLu(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, ReLu<T>>({x.node}));
}

template <typename T>
Expression reShape(const Expression &x, Shape &shape) {
    return Expression(x.executor, x.executor->addFunction<T, ReShape<T>>({x.node}, shape));
}

template <typename T>
Expression reShape(const Expression &x, std::initializer_list<size_t> list) {
    return Expression(x.executor, x.executor->addFunction<T, ReShape<T>>({x.node}, list));
}

template <typename T>
Expression sigmoid(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, Sigmoid<T>>({x.node}));
}

template <typename T>
Expression softmax(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, Softmax<T>>({x.node}));
}

template <typename T>
Expression square(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, Square<T>>({x.node}));
}

template <typename T>
Expression sumElements(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, SumElements<T>>({x.node}));
}

template <typename T>
Expression tanH(const Expression &x) {
    return Expression(x.executor, x.executor->addFunction<T, TanH<T>>({x.node}));
}



}

#endif