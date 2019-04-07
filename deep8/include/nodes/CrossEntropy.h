#ifndef DEEP8_CROSSENTROPY_H
#define DEEP8_CROSSENTROPY_H

#include "Function.h"

namespace Deep8 {

/**
 * in training
 * inputs[0]: the predict output
 * inputs[1]: the target output
 */
class CrossEntropy: public Function {
public:
    CrossEntropy(std::vector<Node*> &inputs);

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