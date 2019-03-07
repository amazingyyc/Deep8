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
#include "nodes/Exp.h"
#include "nodes/L1NormLoss.h"
#include "nodes/L2NormLoss.h"
#include "nodes/Linear.h"
#include "nodes/Log.h"
#include "nodes/LogSoftmax.h"
#include "nodes/LReLu.h"
#include "nodes/MatrixMultiply.h"
#include "nodes/MaxPooling2d.h"
#include "nodes/MeanLoss.h"
#include "nodes/ReduceMean.h"
#include "nodes/ReduceSum.h"
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

    explicit Expression();
	explicit Expression(Executor *exe, Node *n);

    void forward();
    void backward();

	void feed(const void *ptr);
	void fetch(void *ptr);

	std::string valueStr();

	/**init the variable's value*/
	void constant(float scalar = 0);
	void zero();
	void one();
	void gaussian(float mean = 0.0, float stddev = 0.01);
	void positiveUnitball();
	void random(float lower = 0.0, float upper = 1.0);
	void uniform(float left = 0.0, float right = 1.0);

	Expression operator + (const Expression &y) const;
	Expression operator - (const Expression &y) const;
	Expression operator * (const Expression &y) const;
	Expression operator / (const Expression &y) const;

    Expression add(Expression &y);
	Expression minus(Expression &y);
	Expression multiply(Expression &y);
	Expression divide(Expression &y);

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
    Expression reduceMean(int axis = -1, bool keep = true);
	Expression reduceSum(int axis = -1, bool keep = true);
	Expression relu();
	Expression reShape(Shape &shape);
	Expression reShape(std::vector<size_t> list);
	Expression sigmoid();
	Expression softmax(int axis = -1);
	Expression square();
	Expression tanh();

	Expression meanLoss();
	Expression l1NormLoss();
	Expression l2NormLoss();
	Expression softmaxCrossEntropyLoss(Expression &y);
};

/**create parameter*/
Expression parameter(Executor *executor, std::vector<size_t> list, bool updateGradient = true, DType type = DType::Float32);
Expression parameter(Executor *executor, size_t batch, std::vector<size_t> list, bool updateGradient = true, DType type = DType::Float32);
Expression parameter(Executor *executor, Shape &shape, bool updateGradient = true, DType type = DType::Float32);


}


#endif
