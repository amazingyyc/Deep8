#ifndef DEEP8_EXPRESSION_H
#define DEEP8_EXPRESSION_H

#include "nodes/Node.h"
#include "nodes/Variable.h"
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

#include "model/Executor.h"

namespace Deep8 {

class Executor;

class Expression {
public:
    /**the executor*/
    Executor *executor;

    /**the Node pointer that contacted to this Expression*/
    Node *node;

    explicit Expression();
	explicit Expression(Executor *exe, Node *n);

    void forward();
    void backward();

	std::string valueStr();

	Expression feed(const void *ptr);
	Expression fetch(void *ptr);

	/**init the variable's value*/
	Expression constant(float scalar = 0);
	Expression zero();
	Expression one();
	Expression gaussian(float mean = 0.0, float stddev = 0.01);
	Expression positiveUnitball();
	Expression random(float lower = 0.0, float upper = 1.0);
	Expression uniform(float left = 0.0, float right = 1.0);

	Expression operator + (const Expression &y) const;
	Expression operator - (const Expression &y) const;
	Expression operator * (const Expression &y) const;
	Expression operator / (const Expression &y) const;

    Expression operator + (float c) const;
    Expression operator - (float c) const;
    Expression operator * (float c) const;
    Expression operator / (float c) const;

    Expression add(Expression &y);
	Expression minus(Expression &y);
	Expression multiply(Expression &y);
	Expression divide(Expression &y);
	Expression dot(Expression &y);

    Expression addConstant(float c);
    Expression minusConstant(float c);
    Expression multiplyConstant(float c);
    Expression divideConstant(float c);

	/**one operand function*/
	Expression abs();

	Expression avgPooling2d(bool covered = false, 
								size_t filterHeight = 1, 
								size_t filterWidth = 1, 
								size_t strideY = 1, 
								size_t strideX = 1);

	Expression conv2d(Expression &filter, 
							bool covered = false, 
							size_t strideH = 1, 
							size_t strideW = 1, 
							size_t dilationY = 1, 
							size_t dilationX = 1);

	Expression crossEntropy(Expression &y);
	Expression deConv2d(Expression &filter, 
							bool covered = false, 
							size_t strideY = 1, 
							size_t strideX = 1);

	Expression exp();
	Expression l1Distance(Expression &y);
	Expression l1Norm();
	Expression l2Distance(Expression &y);
	Expression l2Norm();
	Expression linear(float a = 1, float b = 0);
	Expression log();
	Expression logSoftmax(int axis = -1);
	Expression lRelu(float a);
	Expression matrixMultiply(Expression &y);

	Expression maxPooling2d(bool covered = false, 
							size_t filterHeight = 1, 
							size_t filterWidth = 1, 
							size_t strideY = 1, 
							size_t strideX = 1);

    Expression pRelu(Expression &y);
    Expression reduceMean(std::vector<int> axis = {-1}, bool keepDims = true);
	Expression reduceSum(std::vector<int> axis = {-1}, bool keepDims = true);
	Expression relu();
	Expression reShape(Shape &shape);
	Expression reShape(std::vector<size_t> list);
	Expression sigmoid();
	Expression softmax(int axis = -1);
	Expression square();
	Expression tanh();

	Expression l1DistanceLoss(Expression &y);
	Expression l1NormLoss();
	Expression l2DistanceLoss(Expression &y);
	Expression l2NormLoss();
	Expression softmaxCrossEntropyLoss(Expression &y);
};

/**create parameter, the parameter will be in memory*/
Expression parameter(Executor *executor, std::vector<size_t> list, bool updateGradient = true, DType type = DType::Float32);
Expression parameter(Executor *executor, size_t batch, std::vector<size_t> list, bool updateGradient = true, DType type = DType::Float32);
Expression parameter(Executor *executor, Shape &shape, bool updateGradient = true, DType type = DType::Float32);

/**create input paramater, the input parameter will deleted after bakward, it is temp*/
Expression inputParameter(Executor *executor, std::vector<size_t> list, DType type = DType::Float32);
Expression inputParameter(Executor *executor, size_t batch, std::vector<size_t> list, DType type = DType::Float32);
Expression inputParameter(Executor *executor, Shape &shape, DType type = DType::Float32);

}


#endif
