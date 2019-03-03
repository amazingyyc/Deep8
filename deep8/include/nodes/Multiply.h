#ifndef DEEP8_MULTIPLY_H
#define DEEP8_MULTIPLY_H

#include "Function.h"

namespace Deep8 {

/**
 * @brief this is a element-wise multiply it will BroadCast the input
 */
class Multiply: public Function {
public:
    explicit Multiply(std::vector<Node *> &inputs);

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



#endif //DEEP8_WISEMULTIPLY_H
