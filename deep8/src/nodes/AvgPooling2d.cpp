#include "math/AvgPooling2d.h"
#include "nodes/AvgPooling2d.h"

namespace Deep8 {

AvgPooling2d::AvgPooling2d(std::vector<Node *> &inputs, bool covered , int fh, int fw, int sy, int sx): 
    Function(inputs), covered(covered), filterHeight(fh), filterWidth(fw), strideY(sy), strideX(sx) {
        check();
}

void AvgPooling2d::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the AvgPooling2d only need 1 input");
    DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX >= 1, "the filter size or stride is error");
    DEEP8_ARGUMENT_CHECK(3 == this->inputs[0]->shape.nDims, "AvgPooling2d needs inputs nDims is 3");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the inputs elementtype must be same");

    auto inputShape = this->inputs[0]->shape;

    if (!covered) {
        DEEP8_ARGUMENT_CHECK(filterHeight <= inputShape.dim(0) && filterWidth <= inputShape.dim(1),
                             "the not forwardCovered mode type needs filter smaller than input");
    }

    auto inputHeight = (int) inputShape.dim(0);
    auto inputWidth  = (int) inputShape.dim(1);

    std::vector<size_t> outputDim(3);
    outputDim[2] = inputShape.dim(2);

    if (!covered) {
        int outputHeight = (inputHeight - filterHeight) / strideY + 1;
        int outputWidth  = (inputWidth  - filterWidth)  / strideX + 1;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height or width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    } else {
        int outputHeight = (inputHeight - filterHeight + strideY - 1) / strideY + 1;
        int outputWidth  = (inputWidth  - filterWidth  + strideX - 1) / strideX + 1;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height or width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    }

    this->shape       = Shape(inputShape.batch, outputDim);
    this->elementType = this->inputs[0]->elementType;
}

void AvgPooling2d::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::AvgPooling2d(*(inputs[0]), *output, covered, filterHeight, filterWidth, strideY, strideX);
}

void AvgPooling2d::backward(const std::vector<const Tensor*> &inputs, 
                            const Tensor *output, 
                            const Tensor *outputGradient, 
                            size_t index, 
                            Tensor *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

    Math::AvgPooling2dGrad(*(inputs[0]), *iGradient, *output, *outputGradient, covered, filterHeight, filterWidth, strideY, strideX);
}


}