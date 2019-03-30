#ifndef DEEP8_MAXPOOLING2D_H
#define DEEP8_MAXPOOLING2D_H

#include "Function.h"

namespace Deep8 {

class MaxPooling2d: public Function {
public:
    int filterHeight;
    int filterWidth;

    int strideY;
    int strideX;

    /**
     * if the slide filter will cover all the input
     */
    bool covered;

    explicit MaxPooling2d(std::vector<Node *> &inputs, bool covered = false, int fh = 1, int fw = 1, int sy = 1, int sx = 1);

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

#endif //DEEP8_MAXPOOLING2D_H
