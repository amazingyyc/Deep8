#include "math/MaxPooling2d.h"
#include "nodes/MaxPooling2d.h"

namespace Deep8 {

MaxPooling2d::MaxPooling2d(std::vector<Node *> &inputs, bool covered, int fh, int fw, int sy, int sx )
		:Function(inputs), covered(covered), filterHeight(fh), filterWidth(fw), strideY(sy), strideX(sx) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the MaxPooling2d Function needs 1 input");
	DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX >= 1, "the filter size or stride is error");
}

Shape MaxPooling2d::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");
    DEEP8_ARGUMENT_CHECK(3 == inputShapes[0].nDims, "AvgPooling2d needs inputs nDims is 3");

    if (!covered) {
        DEEP8_ARGUMENT_CHECK(filterHeight <= inputShapes[0].dim(0) && filterWidth <= inputShapes[0].dim(1),
                             "not forward Covered mode type needs filter smaller than input");
    }

    auto inputHeight = (int) inputShapes[0].dim(0);
    auto inputWidth  = (int) inputShapes[0].dim(1);

    std::vector<size_t> outputDim(3);
    outputDim[2] = inputShapes[0].dim(2);

    if (!covered) {
        int outputHeight = (inputHeight - filterHeight) / strideY + 1;
        int outputWidth  = (inputWidth  - filterWidth)  / strideX + 1;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height or width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    } else {
        int outputHeight = (inputHeight - 1) / strideY + 1;
        int outputWidth  = (inputWidth  - 1) / strideX + 1;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height or width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    }

    return Shape(inputShapes[0].batch, outputDim);
}

ElementType MaxPooling2d::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void MaxPooling2d::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::MaxPooling2d(*(inputs[0]), *output, covered, filterHeight, filterWidth, strideY, strideX);
}
void MaxPooling2d::backward(const std::vector<const Tensor*> &inputs, 
							const Tensor *output, 
							const Tensor *outputGradient, 
							size_t index, 
							Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	Math::MaxPooling2dGrad(*(inputs[0]), *iGradient, *output, *outputGradient, covered, filterHeight, filterWidth, strideY, strideX);
}



}