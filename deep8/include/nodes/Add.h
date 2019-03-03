#ifndef DEEP8_ADD_H
#define DEEP8_ADD_H

#include "Function.h"

namespace Deep8 {

/**
 * Z = X + Y
 */
class Add: public Function {
public:
	explicit Add(std::vector<Node *> &inputs);

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

#endif //DEEP8_ADD_H
