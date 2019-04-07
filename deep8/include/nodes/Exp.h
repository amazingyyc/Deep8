#ifndef DEEP8_EXP_H
#define DEEP8_EXP_H

#include "Function.h"

namespace Deep8 {

/**
 * y = exp(x)
 */
class Exp: public Function {
public:
    explicit Exp(std::vector<Node *> &inputs);

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

#endif //DEEP8_EXP_H
