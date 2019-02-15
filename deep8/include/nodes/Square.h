#ifndef DEEP8_SQUARE_H
#define DEEP8_SQUARE_H

#include "Function.h"

namespace Deep8 {

/**
 * Y = X * X
 */
class Square : public Function {
public:

	Square(std::vector<Node*> &inputs);

	void check() override;

protected:
	void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) override;
				  
};


}

#endif
