#ifndef DEEP8_LOG_H
#define DEEP8_LOG_H

#include "Function.h"


namespace Deep8 {

/**
 * y = Log(x)
 * x must be > 0
 */
class Log: public Function {
public:
    explicit Log(std::vector<Node *> &inputs);

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

#endif //DEEP8_LOG_H
