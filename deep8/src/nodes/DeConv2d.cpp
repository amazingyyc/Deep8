#include "math/DeConv2d.h"
#include "nodes/DeConv2d.h"

namespace Deep8 {

DeConv2d::DeConv2d(std::vector<Node *> &inputs, bool covered, int strideY, int strideX)
        :Function(inputs), forwardCovered(covered), forwardStrideY(strideY), forwardStrideX(strideX) {
    check();
}

void DeConv2d::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "need 2 inputs node");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the inputs elementtype must be same");
    DEEP8_ARGUMENT_CHECK(forwardStrideY >= 1 && forwardStrideX >= 1, "the stride is error");

    auto inputShape  = this->inputs[0]->shape;
    auto filterShape = this->inputs[1]->shape;

    DEEP8_ARGUMENT_CHECK(3 == inputShape.nDims, "Conv2d needs inputs nDims is 3");
    DEEP8_ARGUMENT_CHECK(4 == filterShape.nDims && 1 == filterShape.batch, "Conv2d needs filter nDims is 4, the batch must be 1");

    DEEP8_ARGUMENT_CHECK(inputShape.dim(2) == filterShape.dim(3), "the inputs dimension is error");
    DEEP8_ARGUMENT_CHECK(filterShape.dim(1) > 0 && filterShape.dim(2) > 0, "the filter must bigger than 0");

    auto filterHeight = (int)(filterShape.dim(1));
    auto filterWidth  = (int)(filterShape.dim(2));

    auto inputHeight = (int)(inputShape.dim(0));
    auto inputWidth  = (int)(inputShape.dim(1));

    std::vector<size_t> outputDim(3);
    outputDim[2] = filterShape.dim(0);

    /**
     * calculate the output dimension is the reverse of the forward Conv2d
     */
    if (!forwardCovered) {
        int outputHeight = (inputHeight - 1) * forwardStrideY + filterHeight;
        int outputWidth  = (inputWidth  - 1) * forwardStrideX + filterWidth;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height/width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    } else {
        int outputHeight = (inputHeight - 1) * forwardStrideY + 1;
        int outputWidth  = (inputWidth  - 1) * forwardStrideX + 1;

        DEEP8_ARGUMENT_CHECK(outputHeight > 0 && outputWidth > 0, "the output height/width must > 0")

        outputDim[0] = outputHeight;
        outputDim[1] = outputWidth;
    }

    this->shape       = Shape(inputShape.batch, outputDim);
    this->elementType = this->inputs[0]->elementType;
}

void DeConv2d::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    auto device = output->device();

    auto interimSize = Math::DeConv2dInterimSize(*(inputs[0]), *(inputs[1]), *output, forwardCovered, forwardStrideY, forwardStrideX);
    auto interimPtr  = device->malloc(interimSize);

    Math::DeConv2d(*(inputs[0]), *(inputs[1]), *output, forwardCovered, forwardStrideY, forwardStrideX, interimPtr);

    device->free(interimPtr);
}

void DeConv2d::backward(const std::vector<const Tensor*> &inputs, 
                        const Tensor *output, 
                        const Tensor *outputGradient, 
                        size_t index, 
                        Tensor *iGradient) {
    if (0 == index) {
        auto device = iGradient->device();

        auto interimSize = Math::DeConv2dGradXInterimSize(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient, forwardCovered, forwardStrideY, forwardStrideX);
        auto interimPtr  = device->malloc(interimSize);

        Math::DeConv2dGradX(*(inputs[0]),
                            *iGradient,
                            *(inputs[1]),
                            *output,
                            *outputGradient,
                            forwardCovered,
                            forwardStrideY,
                            forwardStrideX,
                            interimPtr);

        device->free(interimPtr);

    } else if (1 == index) {
        auto device = iGradient->device();

        auto interimSize = Math::DeConv2dGradYInterimSize(*(inputs[0]),
                                                          *(inputs[1]),
                                                          *iGradient,
                                                          *output,
                                                          *outputGradient,
                                                          forwardCovered,
                                                          forwardStrideY,
                                                          forwardStrideX);
        
        auto interimPtr = device->malloc(interimSize);

        Math::DeConv2dGradY(*(inputs[0]),
                            *(inputs[1]),
                            *iGradient,
                            *output,
                            *outputGradient,
                            forwardCovered,
                            forwardStrideY,
                            forwardStrideX,
                            interimPtr);

        device->free(interimPtr);

    } else {
        DEEP8_RUNTIME_ERROR("the index of DeConv2d backward is error");
    }
}



}