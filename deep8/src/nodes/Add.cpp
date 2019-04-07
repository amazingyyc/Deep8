#include "math/Add.h"
#include "nodes/Add.h"

namespace Deep8 {

Add::Add(std::vector<Node *> &inputs) : Function(inputs) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the Add Function needs 2 input");
}

Shape Add::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the input count must be 2");

    /**
     * the Add Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
     */
    return broadcastShape(inputShapes[0], inputShapes[1]);
}

ElementType Add::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 2");

    return Function::checkElementType(inputTypes);
}

void Add::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::Add(*(inputs[0]), *(inputs[1]), *output);
}

void Add::backward(const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    if (0 == index) {
        Math::AddGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::AddGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}



}
