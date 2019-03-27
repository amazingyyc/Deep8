#include "math/Conv2d.h"
#include "nodes/Conv2d.h"

namespace Deep8 {

Conv2d::Conv2d(std::vector<Node *> &inputs, bool covered, int sy, int sx, int dy , int dx)
        :Function(inputs), covered(covered), strideY(sy), strideX(sx), dilationY(dy), dilationX(dx) {
    check();
}

void Conv2d::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "need 2 inputs node");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the inputs elementtype must be same");
    DEEP8_ARGUMENT_CHECK(strideY >= 1 && strideX >= 1, "the stride can not smaller than 1");
    DEEP8_ARGUMENT_CHECK(dilationY >= 1 && dilationX >= 1, "the dilation can not smaller than 1");

    auto inputShape  = this->inputs[0]->shape;
    auto filterShape = this->inputs[1]->shape;

    DEEP8_ARGUMENT_CHECK(3 == inputShape.nDims, "Conv2d needs inputs nDims is 3");
    DEEP8_ARGUMENT_CHECK(4 == filterShape.nDims && 1 == filterShape.batch, "Conv2d needs filter nDims is 4, the batch must be 1");

    DEEP8_ARGUMENT_CHECK(inputShape.dim(2) == filterShape.dim(3), "the inputs dimension is error");
    DEEP8_ARGUMENT_CHECK(filterShape.dim(1) > 0 && filterShape.dim(2) > 0,
                         "the filter width and height must bigger than 0");

    if (!covered) {
        DEEP8_ARGUMENT_CHECK(filterShape.dim(1) <= inputShape.dim(0) && filterShape.dim(2) <= inputShape.dim(1),
                "the not covered mode Padding type needs filter smaller than input");
    }

    auto filterHeight = (int)(filterShape.dim(1));
    auto filterWidth  = (int)(filterShape.dim(2));

    auto inputHeight = (int)(inputShape.dim(0));
    auto inputWidth  = (int)(inputShape.dim(1));

    auto realFilterHeight = filterHeight + (filterHeight - 1) * (dilationY - 1);
    auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * (dilationX - 1);

    std::vector<size_t> outputDim(3);
    outputDim[2] = filterShape.dim(0);

    /**
     * the input dimension is (batch, inputHeight, inputWidth, inputChannel)
     * filter dimension is (outputChannel, filterHeight, filterWidth, inputChannel)
     * output dimension is (batch, outputHeight, outputWidth, outputChannel)
     */
    if (!covered) {
        int outputHeight = (inputHeight - realFilterHeight) / strideY + 1;
        int outputWidth  = (inputWidth  - realFilterWidth)  / strideX + 1;

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

    this->shape = Shape(inputShape.batch, outputDim);
    this->elementType = this->inputs[0]->elementType;
}

void Conv2d::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    auto device = output->device();

    auto interimSize = Math::Conv2dInterimSize(*(inputs[0]), *(inputs[1]), *output, covered, strideY, strideX, dilationY, dilationX);
    auto interimPtr  = device->malloc(interimSize);

    Math::Conv2d(*(inputs[0]), *(inputs[1]), *output, covered, strideY, strideX, dilationY, dilationX, interimPtr);

    device->free(interimPtr);
}

void Conv2d::backward(const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    if (0 == index) {
        auto device = iGradient->device();

        auto interimSize = Math::Conv2dGradXInterimSize(*(inputs[0]), 
                                                      *iGradient,
                                                      *(inputs[1]),
                                                      *output,
                                                      *outputGradient,
                                                      covered,
                                                      strideY,
                                                      strideX,
                                                      dilationY,
                                                      dilationX);

        auto interimPtr = device->malloc(interimSize);

        Math::Conv2dGradX(*(inputs[0]),
                          *iGradient,
                          *(inputs[1]),
                          *output,
                          *outputGradient,
                          covered,
                          strideY,
                          strideX,
                          dilationY,
                          dilationX,
                          interimPtr);

        device->free(interimPtr);

    } else if (1 == index) {
        auto device = iGradient->device();

        auto interimSize = Math::Conv2dGradYInterimSize(*(inputs[0]),
                                                      *(inputs[1]),
                                                      *iGradient,
                                                      *output,
                                                      *outputGradient,
                                                      covered,
                                                      strideY,
                                                      strideX,
                                                      dilationY,
                                                      dilationX);

        auto interimPtr = device->malloc(interimSize);

        Math::Conv2dGradY(*(inputs[0]),
                          *(inputs[1]),
                          *iGradient,
                          *output,
                          *outputGradient,
                          covered,
                          strideY,
                          strideX,
                          dilationY,
                          dilationX,
                          interimPtr);

        device->free(interimPtr);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}



}
