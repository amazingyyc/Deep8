#ifndef DEEP8_FUNCTION_H
#define DEEP8_FUNCTION_H

#include "Node.h"
#include "Variable.h"

namespace Deep8 {

/**
 * the function Type
 */
enum class FunctionType {
	Unkown = 0,
	Abs,
	Add,
	AvgPooling2d,
    Batch,
	Conv2d,
	DeConv2d,
    CrossEntropy,
	Divide,
	Exp,
    L1Distance,
	L1Norm,
    L2Distance,
	L2Norm,
	Linear,
	Log,
	LogSoftmax,
	LReLu,
	MatrixMultiply,
	MaxPooling2d,
	Minus,
	Multiply,
	PReLu,
	ReduceMean,
	ReduceSum,
	ReLu,
	ReShape,
	Sigmoid,
	Softmax,
	Square,
	Tanh,
};

class Function: public Node {
protected:
    /**
     * if a Function shared is true means the output Variable shared the memory with the input Variable
     * default is false
     */
	bool isShared;
	
protected:
    explicit Function(int64_t id, std::string name, std::vector<Node*>& inputs);

public:

    /**return the shape and elementtype by the input's*/
    Shape outputShape(std::vector<Shape&> inputShapes);
    ElementType outputElementType(std::vector<ElementType&> inputTypes);

    virtual void forward(const std::vector<const Tensor*> &inputs, Tensor *output);
    virtual void backward(const std::vector<const Tensor*> &inputs, const Tensor *output, const Tensor *outputGradient, size_t index, Tensor *iGradient);

	void forward() override;
	void backward() override;
};

}

#endif
