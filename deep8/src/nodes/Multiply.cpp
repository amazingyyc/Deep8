#include "math/Multiply.h"
#include "nodes/Multiply.h"

namespace Deep8 {

Multiply::Multiply(std::vector<Node*> &inputs) : Function(inputs) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the Multiply Function needs 2 input");
}

Shape Multiply::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the input count must be 2");

    return broadcastShape(inputShapes[0], inputShapes[1]);
}

ElementType Multiply::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 2");

    return Function::checkElementType(inputTypes);
}
void Multiply::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::Multiply(*(inputs[0]), *(inputs[1]), *output);
}

void Multiply::backward(const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    if (0 == index) {
        Math::MultiplyGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::MultiplyGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}



}