#ifndef DEEP8_LINEAR_H
#define DEEP8_LINEAR_H

#include "Function.h"

namespace Deep8 {

/**
 * @brief y = a * x + b
 */
class Linear: public Function {
public:
    float a;
    float b;

    explicit Linear(std::vector<Node*> &inputs, float a, float b);

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

#endif //DEEP8_LINEAR_H
