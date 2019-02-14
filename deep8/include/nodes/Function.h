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
	Conv2d,
	DeConv2d,
	Divide,
	Exp,
	L1Norm,
	L2Norm,
	Linear,
	Log,
	LReLu,
	MatrixMultiply,
	MaxPooling2d,
	Minus,
	Multiply,
	ReLu,
	ReShape,
	Sigmoid,
	Softmax,
	Square,
	Tanh,
};

class Function: public Node {
protected:
	explicit Function();
    explicit Function(std::vector<Node*> &inputs);
	
protected:
	virtual void check();

	virtual void forward(const std::vector<const Tensor*> &inputs, Tensor *output);
	virtual void backward(const std::vector<const Tensor*> &inputs, const Tensor *output, const Tensor *outputGradient, size_t index, Tensor *iGradient);

public:
	/**
	 * if a Function shared is true means the output Variable shared the memory with the input Variable
	 * default is false
	 */
	virtual bool isShared();

	void forward() override;
	void backward() override;
};

}

#endif
