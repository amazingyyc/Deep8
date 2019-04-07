#ifndef DEEP8_ABS_H
#define DEEP8_ABS_H

#include "Function.h"

namespace Deep8 {

/**
 * y = |x|
 */
class Abs: public Function {
public:
	explicit Abs(std::vector<Node *> &inputs);

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

#endif //DEEP8_ABS_H
