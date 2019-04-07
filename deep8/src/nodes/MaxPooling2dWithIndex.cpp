#include "math/MaxPooling2dWithIndex.h"
#include "nodes/MaxPooling2dWithIndex.h"

namespace Deep8 {

MaxPooling2dWithIndex::MaxPooling2dWithIndex(std::vector<Node *> &inputs, bool covered, int fh, int fw, int sy, int sx )
		:Function(inputs), covered(covered), filterHeight(fh), filterWidth(fw), strideY(sy), strideX(sx) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the MaxPooling2dWithIndex Function needs 2 input");
	DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX >= 1, "the filter size or stride is error");
}

Shape MaxPooling2dWithIndex::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the input count must be 2");
    DEEP8_ARGUMENT_CHECK(3 == inputShapes[0].nDims, "MaxPooling2dWithIndex needs inputs nDims is 3");

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

    Shape outputShape(inputShapes[0].batch, outputDim);

    DEEP8_ARGUMENT_CHECK(outputShape == inputShapes[1], "the output shape must be same with index shape");

    return outputShape;
}

ElementType MaxPooling2dWithIndex::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 1");
    DEEP8_ARGUMENT_CHECK(DType::Int32 == inputTypes[1].id, "the second input must be a int32 element type");

    return inputTypes[0];
}

void MaxPooling2dWithIndex::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::MaxPooling2dWithIndex(*(inputs[0]), *(inputs[1]), *output, covered, filterHeight, filterWidth, strideY, strideX);
}

void MaxPooling2dWithIndex::backward(const std::vector<const Tensor*> &inputs,
							const Tensor *output, 
							const Tensor *outputGradient, 
							size_t index, 
							Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	Math::MaxPooling2dWithIndexGrad(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient, covered, filterHeight, filterWidth, strideY, strideX);
}



}