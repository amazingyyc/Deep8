#ifndef DEEP8_NET_H
#define DEEP8_NET_H

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
#include "model/Executor.h"

namespace Deep8 {

void backward(Variable&, bool clearInterim = true);
void backward(Variable*, bool clearInterim = true);

Variable& parameter(Executor *, std::vector<size_t> ,         bool updateGradient = true, DType type = DType::Float32);
Variable& parameter(Executor *, size_t, std::vector<size_t> , bool updateGradient = true, DType type = DType::Float32);
Variable& parameter(Executor *, Shape& ,                      bool updateGradient = true, DType type = DType::Float32);

Variable& inputParameter(Executor *, std::vector<size_t> ,         DType type = DType::Float32);
Variable& inputParameter(Executor *, size_t, std::vector<size_t> , DType type = DType::Float32);
Variable& inputParameter(Executor *, Shape& ,                      DType type = DType::Float32);

Variable& feed(Variable&, const void*);

Variable& fetch(Variable&, void*);

Variable& constant(Variable&, float scalar = 0);
Variable& zero(Variable&);
Variable& one(Variable&);
Variable& gaussian(Variable&, float mean = 0.0, float stddev = 0.01);
Variable& positiveUnitball(Variable&);
Variable& random(Variable&, float lower = 0.0, float upper = 1.0);
Variable& uniform(Variable&, float left = 0.0, float right = 1.0);
Variable& assign(Variable &x, Variable &v);

Variable& operator + (Variable &x, Variable &y);
Variable& operator - (Variable &x, Variable &y);
Variable& operator * (Variable &x, Variable &y);
Variable& operator / (Variable &x, Variable &y);

Variable& operator += (Variable &x, Variable &y);
Variable& operator -= (Variable &x, Variable &y);
Variable& operator *= (Variable &x, Variable &y);
Variable& operator /= (Variable &x, Variable &y);

Variable& operator + (Variable &x, float c);
Variable& operator - (Variable &x, float c);
Variable& operator * (Variable &x, float c);
Variable& operator / (Variable &x, float c);

Variable& add(Variable &x, Variable &y);
Variable& minus(Variable &x, Variable &y);
Variable& multiply(Variable &x, Variable &y);
Variable& divide(Variable &x, Variable &y);

Variable& addConstant(Variable &x, float c);
Variable& minusConstant(Variable &x, float c);
Variable& multiplyConstant(Variable &x, float c);
Variable& divideConstant(Variable &x, float c);

Variable& abs(Variable &x);

Variable& avgPooling2d( Variable &x,
                        bool covered = true, 
                        int filterHeight = 1, 
                        int filterWidth = 1, 
                        int strideY = 1, 
                        int strideX = 1);

Variable& conv2d(Variable &x,
                Variable &filter,
                bool covered = true, 
                int strideY = 1, 
                int strideX = 1, 
                int dilationY = 1, 
                int dilationX = 1);

Variable& crossEntropy(Variable &x, Variable &y);

Variable& deConv2d( Variable &x,
                    Variable &filter,
                    bool covered = false, 
                    int strideY = 1, 
                    int strideX = 1);

Variable& dot(Variable &x, Variable &y);
Variable& exp(Variable &x);
Variable& l1Distance(Variable &x, Variable &y);
Variable& l1Norm(Variable &x);
Variable& l2Distance(Variable &x, Variable &y);
Variable& l2Norm(Variable &x);
Variable& linear(Variable &x, float a = 1, float b = 0);
Variable& log(Variable &x);
Variable& logSoftmax(Variable &x, int axis = -1);
Variable& lRelu(Variable &x, float a);

Variable& matrixMultiply(Variable &x, Variable &y);

Variable& maxPooling2d( Variable &x,
                        bool covered = false, 
                        int filterHeight = 1, 
                        int filterWidth = 1, 
                        int strideY = 1, 
                        int strideX = 1);

Variable& maxPooling2dWithIndex(Variable &x,
                                Variable& index,
                                bool covered = false, 
                                int filterHeight = 1, 
                                int filterWidth = 1, 
                                int strideY = 1, 
                                int strideX = 1);

Variable& maxUnPooling2d(Variable &x,
                        Variable& index,
                        bool covered = false, 
                        int filterHeight = 1, 
                        int filterWidth = 1, 
                        int strideY = 1, 
                        int strideX = 1);

Variable& pRelu(Variable &x, Variable &p);

Variable& reduceMean(Variable &x, std::vector<int> axis = {-1}, bool keepDims = true);
Variable& reduceSum(Variable &x, std::vector<int> axis = {-1}, bool keepDims = true);
Variable& relu(Variable &x);
Variable& reShape(Variable &x, Shape &shape);
Variable& reShape(Variable &x, std::vector<size_t> list);
Variable& sigmoid(Variable &x);
Variable& softmax(Variable &x, int axis = -1);
Variable& square(Variable &x);
Variable& tanh(Variable &x);

Variable& l1Loss(Variable &x, Variable &y);
Variable& l1NormLoss(Variable &x);
Variable& l2Loss(Variable &x, Variable &y);
Variable& l2NormLoss(Variable &x);
Variable& softmaxCrossEntropyLoss(Variable &x, Variable &y);

}

#endif