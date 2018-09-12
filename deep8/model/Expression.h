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


template <typename T>
Expression<T> parameter(Executor<T> *executor, std::initializer_list<size_t> list) {
    return Expression<T>(executor, executor->addParameter(list));
}

template <typename T>
Expression<T> parameter(Executor<T> *executor, Shape &shape) {
    return Expression<T>(executor, executor->addParameter(shape));
}

template <typename T>
Expression<T> inputParameter(Executor<T> *executor, std::initializer_list<size_t> list) {
    return Expression<T>(executor, executor->addInputParameter(list));
}

template <typename T>
Expression<T> inputParameter(Executor<T> *executor, Shape &shape) {
    return Expression<T>(executor, executor->addInputParamete(shape));
}

/**add operator*/
template <typename T>
Expression<T> operator + (const Expression<T> &x, const Expression<T> &y) {
	return Expression<T>(x.executor, x.executor->addFunction<Add<T>>({ x.node, y.node }));
}

template <typename T>
Expression<T> operator + (const Expression<T> &x, T scalar) {
	return Expression<T>(x.executor, x.executor->addFunction<AddScalar<T>>({ x.node }, scalar));
}

template <typename T>
Expression<T> operator + (T scalar, const Expression<T> &x) {
	return Expression<T>(x.executor, x.executor->addFunction<AddScalar<T>>({ x.node }, scalar));
}

/**minus operator*/
template <typename T>
Expression<T> operator - (const Expression<T> &x, const Expression<T> &y) {
	return Expression<T>(x.executor, x.executor->addFunction<Minus<T>>({ x.node, y.node }));
}

template <typename T>
Expression<T> operator - (const Expression<T> &x, T scalar) {
	return Expression<T>(x.executor, x.executor->addFunction<MinusScalar<T>>({ x.node }, scalar));
}

template <typename T>
Expression<T> operator - (T scalar, const Expression<T> &x) {
	return Expression<T>(x.executor, x.executor->addFunction<ScalarMinus<T>>({ x.node }, scalar));
}

/**multiply operator*/
template <typename T>
Expression<T> operator * (const Expression<T> &x, const Expression<T> &y) {
	return Expression<T>(x.executor, x.executor->addFunction<Multiply<T>>({ x.node, y.node }));
}

template <typename T>
Expression<T> operator * (const Expression<T> &x, T scalar) {
	return Expression<T>(x.executor, x.executor->addFunction<MultiplyScalar<T>>({ x.node }, scalar));
}

template <typename T>
Expression<T> operator * (T scalar, const Expression<T> &x) {
	return Expression<T>(x.executor, x.executor->addFunction<MultiplyScalar<T>>({ x.node }, scalar));
}

/**divide operator*/
template <typename T>
Expression<T> operator / (const Expression<T> &x, const Expression<T> &y) {
	return Expression<T>(x.executor, x.executor->addFunction<Divide<T>>({ x.node, y.node }));
}

template <typename T>
Expression<T> operator / (const Expression<T> &x, T scalar) {
	return Expression<T>(x.executor, x.executor->addFunction<DivideScalar<T>>({ x.node }, scalar));
}

template <typename T>
Expression<T> operator / (T scalar, const Expression<T> &x) {
	return Expression<T>(x.executor, x.executor->addFunction<ScalarDivide<T>>({ x.node }, scalar));
}

/**function*/
template <typename T>
Expression<T> add(const Expression<T> &x, const Expression<T> &y) {
    return Expression<T>(x.executor, x.executor->addFunction<Add<T>>({x.node, y.node}));
}

template <typename T>
Expression<T> add(const Expression<T> &x, T scalar) {
    return Expression<T>(x.executor, x.executor->addFunction<AddScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> add(T scalar, const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<AddScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> minus(const Expression<T> &x, const Expression<T> &y) {
    return Expression<T>(x.executor, x.executor->addFunction<Minus<T>>({x.node, y.node}));
}

template <typename T>
Expression<T> minus(const Expression<T> &x, T scalar) {
    return Expression<T>(x.executor, x.executor->addFunction<MinusScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> minus(T scalar, const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<ScalarMinus<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> multiply(const Expression<T> &x, const Expression<T> &y) {
    return Expression<T>(x.executor, x.executor->addFunction<Multiply<T>>({x.node, y.node}));
}

template <typename T>
Expression<T> multiply(const Expression<T> &x, T scalar) {
    return Expression<T>(x.executor, x.executor->addFunction<MultiplyScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> multiply(T scalar, const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<MultiplyScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> divide(const Expression<T> &x, const Expression<T> &y) {
    return Expression<T>(x.executor, x.executor->addFunction<Divide<T>>({x.node, y.node}));
}

template <typename T>
Expression<T> divide(const Expression<T> &x, T scalar) {
    return Expression<T>(x.executor, x.executor->addFunction<DivideScalar<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> divide(T scalar, const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<ScalarDivide<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> abs(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<Abs<T>>({x.node}));
}

template <typename T>
Expression<T> avgPooling2d(const Expression<T> &x, bool covered = false, size_t filterHeight = 1, size_t filterWidth = 1, size_t strideY = 1, size_t strideX = 1) {
    return Expression<T>(x.executor, x.executor->addFunction<AvgPooling2d<T>>({x.node}, covered, filterHeight, filterWidth, strideY, strideX));
}

template <typename T>
Expression<T> conv2d(const Expression<T> &x, const Expression<T> &y, bool covered = false, size_t strideH = 1, size_t strideW = 1, size_t dilationY = 1, size_t dilationX = 1) {
    return Expression<T>(x.executor, x.executor->addFunction<Conv2d<T>>({x.node, y.node}, covered, strideH, strideW, dilationY, dilationX));
}

template <typename T>
Expression<T> deConv2d(const Expression<T> &x, const Expression<T> &y, bool covered = false, size_t strideY = 1, size_t strideX = 1) {
    return Expression<T>(x.executor, x.executor->addFunction<DeConv2d<T>>({x.node, y.node}, covered, strideY, strideX));
}

template <typename T>
Expression<T> exp(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<Exp<T>>({x.node}));
}

template <typename T>
Expression<T> l1Norm(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<L1Norm<T>>({x.node}));
}

template <typename T>
Expression<T> l2Norm(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<L2Norm<T>>({x.node}));
}

template <typename T>
Expression<T> linear(const Expression<T> &x, T a, T b) {
    return Expression<T>(x.executor, x.executor->addFunction<Linear<T>>({x.node}, a, b));
}

template <typename T>
Expression<T> log(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<Log<T>>({x.node}));
}

template <typename T>
Expression<T> lReLu(const Expression<T> &x, T a) {
    return Expression<T>(x.executor, x.executor->addFunction<LReLu<T>>({x.node}, a));
}

template <typename T>
Expression<T> matrixMultiply(const Expression<T> &x, const Expression<T> &y) {
    return Expression<T>(x.executor, x.executor->addFunction<MatrixMultiply<T>>({x.node, y.node}));
}

template <typename T>
Expression<T> maxPooling2d(const Expression<T> &x, bool covered = false, size_t filterHeight = 1, size_t filterWidth = 1, size_t strideY = 1, size_t strideX = 1) {
    return Expression<T>(x.executor, x.executor->addFunction<MaxPooling2d<T>>({x.node}, covered, filterHeight, filterWidth, strideY, strideX));
}

template <typename T>
Expression<T> pow(const Expression<T> &x, T scalar) {
    return Expression<T>(x.executor, x.executor->addFunction<Pow<T>>({x.node}, scalar));
}

template <typename T>
Expression<T> reLu(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<ReLu<T>>({x.node}));
}

template <typename T>
Expression<T> reShape(const Expression<T> &x, Shape &shape) {
    return Expression<T>(x.executor, x.executor->addFunction<ReShape<T>>({x.node}, shape));
}

template <typename T>
Expression<T> reShape(const Expression<T> &x, std::initializer_list<size_t> list) {
    return Expression<T>(x.executor, x.executor->addFunction<ReShape<T>>({x.node}, list));
}

template <typename T>
Expression<T> sigmoid(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<Sigmoid<T>>({x.node}));
}

template <typename T>
Expression<T> softmax(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<Softmax<T>>({x.node}));
}

template <typename T>
Expression<T> square(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<Square<T>>({x.node}));
}

template <typename T>
Expression<T> sumElements(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<SumElements<T>>({x.node}));
}

template <typename T>
Expression<T> tanH(const Expression<T> &x) {
    return Expression<T>(x.executor, x.executor->addFunction<TanH<T>>({x.node}));
}



}

#endif