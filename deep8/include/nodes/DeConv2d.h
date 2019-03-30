#ifndef DEEP8_DECONV2D_H
#define DEEP8_DECONV2D_H

#include "Function.h"

namespace Deep8 {

/**
 * @brief the DeConv2d is the reverse of Conv2d
 * for Conv2d
 * the input dimension is (batch, inputHeight, inputWidth, inputChannel)
 * the filter dimension is (outputchannel, filterHeight, filterWidth, inputChannel)
 * the stride is (strideH, strideW)
 * dilation is (dilationH, dilationW)
 *
 * realFilterH = filterH + (filterH - 1) * (dilationH - 1);
 * realFilterW = filterW + (filterW - 1) * (dilationW - 1);
 *
 * so the output is
 * the covered is false: (batch, (inputH - realFilterH) / strideH + 1, (inputW - realFilterW) / strideW + 1, outputChannel)
 * the covered is true: (batch, (inputH - 1) / strideH + 1, (inputW - 1) / strideW + 1, outputChannel)
 *
 * ref: A guide to convolution arithmetic for deep learning
 * the reverse of Conv2d TransposeConv2d:
 * the inputs dimension is (Batch, inputHeight, inputWidth, inputChannel)
 * the filter dimension is (outputChannel, filterHeight, filterWidth, inputChannel)
 * the stride is (1, 1) (the stride of the TransposeConv2d always 1)
 * inset (stride - 1) 0-unit into the input
 */

class DeConv2d: public Function {
public:
    /**
     * the forwardStride and forwardCovered is the property of the forward Conv2d
     */
    int forwardStrideY;
    int forwardStrideX;

    /**
     * if the slide filter will cover the input of the Conv2d
     */
    bool forwardCovered;

    DeConv2d(std::vector<Node *> &inputs, bool covered = false, int strideY = 1, int strideX = 1);

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

#endif //DEEP8_TRANSPOSECONV2D_H
