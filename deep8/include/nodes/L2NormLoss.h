#ifndef DEEP8_L2NORMLOSS_H
#define DEEP8_L2NORMLOSS_H

#include "Function.h"

namespace Deep8 {

class L2NormLoss : public Function {
public:
    explicit L2NormLoss(std::vector<Node *> &inputs);

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

#endif //DEEP8_L2NORM_H
