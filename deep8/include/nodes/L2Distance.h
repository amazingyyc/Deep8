#ifndef DEEP8_L2DISTANCE_H
#define DEEP8_L2DISTANCE_H

#include "Function.h"

namespace Deep8 {

class L2Distance : public Function {
public:
    explicit L2Distance(std::vector<Node *> &inputs);

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