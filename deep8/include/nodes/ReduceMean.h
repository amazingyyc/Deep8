#ifndef DEEP8_REDUCEMEAN_H
#define DEEP8_REDUCEMEAN_H


#include "Function.h"

namespace Deep8 {

class ReduceMean : public Function {
public:
    std::vector<int> axis;
    bool keepDims;

    explicit ReduceMean(std::vector<Node *> &inputs, std::vector<int> reduceAxis = {}, bool keep = true);

	Shape checkShape(std::vector<Shape> &inputShapes) override;

	ElementType checkElementType(std::vector<ElementType> &inputTypes) override;

	void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) override;
};

}


#endif