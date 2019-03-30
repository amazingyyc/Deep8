#include "math/PReLu.h"
#include "nodes/PReLu.h"

namespace Deep8 {

PReLu::PReLu(std::vector<Node*>& inputs) : Function(inputs) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the PReLu Function needs 2 input");
}

Shape PReLu::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the input count must be 2");
    DEEP8_ARGUMENT_CHECK(inputShapes[0] == inputShapes[1], "the inputs shap must be same");

    return inputShapes[0];
}

ElementType PReLu::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 2");

    return Function::checkElementType(inputTypes);
}

void PReLu::forward(const std::vector<const Tensor*>& inputs, Tensor* output) {
    Math::PReLu(*(inputs[0]), *(inputs[1]), *output);
}

void PReLu::backward(const std::vector<const Tensor*>& inputs, const Tensor* output, const Tensor* outputGradient, size_t index, Tensor* iGradient) {
    if (0 == index) {
        Math::PReLuGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::PReLuGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

}
