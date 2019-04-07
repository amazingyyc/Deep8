#include "math/MaxUnPooling2d.h"
#include "nodes/MaxUnPooling2d.h"

namespace Deep8 {

MaxUnPooling2d::MaxUnPooling2d(std::vector<Node *> &inputs, bool covered, int fh, int fw, int sy, int sx )
		:Function(inputs), covered(covered), filterHeight(fh), filterWidth(fw), strideY(sy), strideX(sx) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the MaxUnPooling2d Function needs 2 input");
	DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX >= 1, "the filter size or stride is error");
}

Shape MaxUnPooling2d::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the input count must be 2");
    DEEP8_ARGUMENT_CHECK(inputShapes[0] == inputShapes[1], "the input shape must be same");
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
        int outputHeight = (inputHeight - 1) * strideY + filterHeight;
        int outputWidth  = (inputWidth  - 1) * strideX + filterWidth;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height or width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    } else {
        int outputHeight = (inputHeight - 1) * strideY + 1;
        int outputWidth  = (inputWidth  - 1) * strideX + 1;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height or width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    }

    return Shape(inputShapes[0].batch, outputDim);
}

ElementType MaxUnPooling2d::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 2");
    DEEP8_ARGUMENT_CHECK(DType::Int32 == inputTypes[1].id, "the second input must be a int32 element type");

    return inputTypes[0];
}

void MaxUnPooling2d::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::MaxUnPooling2d(*(inputs[0]), *(inputs[1]), *output, covered, filterHeight, filterWidth, strideY, strideX);
}
void MaxUnPooling2d::backward(const std::vector<const Tensor*> &inputs,
							const Tensor *output, 
							const Tensor *outputGradient, 
							size_t index, 
							Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	Math::MaxUnPooling2dGrad(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient, covered, filterHeight, filterWidth, strideY, strideX);
}



}