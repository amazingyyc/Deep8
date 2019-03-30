#ifndef DEEP8_AVGPOOLING2D_H
#define DEEP8_AVGPOOLING2D_H

#include "Function.h"

namespace Deep8 {

class AvgPooling2d: public Function {
public:
    int filterHeight;
    int filterWidth;

    int strideY;
    int strideX;

    /**
     * if the slide filter will cover all the input
     */
    bool covered;

    explicit AvgPooling2d(  std::vector<Node *> &inputs, 
                            bool covered = false, 
                            int filterHeight = 1, 
                            int filterWidth = 1, 
                            int strideY = 1, 
                            int strideX = 1);

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

#endif //DEEP8_AVGPOOL2D_H
