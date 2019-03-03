#ifndef DEEP8_L1NORM_H
#define DEEP8_L1NORM_H

#include "Function.h"

namespace Deep8 {

class L1Norm: public Function {
public:
    explicit L1Norm(std::vector<Node *> &inputs);

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

#endif //DEEP8_L1NORM_H
