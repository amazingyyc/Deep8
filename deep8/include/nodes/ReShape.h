#ifndef DEEP8_RESHAPE_H
#define DEEP8_RESHAPE_H

#include "Function.h"

namespace Deep8 {

class ReShape: public Function {
public:
    explicit ReShape(std::vector<Node *> &inputs, Shape &shape);
    explicit ReShape(std::vector<Node *> &inputs, std::vector<size_t> &shape);
    
    virtual void check() override;

protected:
	void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) override;

public:
	void forward() override;
	void backward() override;
};

}

#endif //DEEP8_RESHAPE_H
