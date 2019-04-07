#include "math/Abs.h"
#include "nodes/Abs.h"

namespace Deep8 {

Abs::Abs(std::vector<Node *> &inputs) : Function(inputs) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Abs Function needs 1 input");
}

Shape Abs::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType Abs::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Abs::forward(const std::vector<const Tensor*> &inputs, Tensor *output)  {
    Math::Abs(*(inputs[0]), *output);
}

void Abs::backward( const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backwardCPU is error");

    Math::AbsGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}