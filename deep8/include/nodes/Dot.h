#ifndef DEEP8_DOT_H
#define DEEP8_DOT_H

#include "Function.h"

namespace Deep8 {

class Dot : public Function {
public:
    explicit Dot(std::vector<Node *> &inputs);

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