#ifndef DEEP8_CONV2D_H
#define DEEP8_CONV2D_H

#include "Function.h"

namespace Deep8 {

class Conv2d: public Function {
public:
    int strideY;
    int strideX;

    int dilationY;
    int dilationX;

    /**
     * if true the slide filter will cover all the input
     */
    bool covered;

    Conv2d(std::vector<Node *> &inputs, bool covered = false, int strideY = 1, int strideX = 1, int dilationY = 1, int dilationX = 1);

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

#endif //DEEP8_CONV2D_H
