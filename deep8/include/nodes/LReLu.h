#ifndef DEEP8_LRELU_H
#define DEEP8_LRELU_H

#include "Function.h"

namespace Deep8 {

class LReLu: public Function {
public:
    float a;

    explicit LReLu(std::vector<Node*> &inputs, float a);
	
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
