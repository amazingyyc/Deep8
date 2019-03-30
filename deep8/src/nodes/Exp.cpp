#include "math/Exp.h"
#include "nodes/Exp.h"

namespace Deep8 {

Exp::Exp(std::vector<Node *> &inputs): Function(inputs) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Exp Function needs 1 input");
}

Shape Exp::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType Exp::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Exp::forward(const std::vector<const Tensor*> &inputs, Tensor *output)  {
    Math::Exp(*(inputs[0]), *output);
}

void Exp::backward( const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Exp backward is error");

    Math::ExpGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}