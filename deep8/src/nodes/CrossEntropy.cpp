#include "math/CrossEntropy.h"
#include "nodes/CrossEntropy.h"

namespace Deep8 {

CrossEntropy::CrossEntropy(std::vector<Node *> &inputs) : Function(inputs) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the CrossEntropy Function needs 2 input");
}

Shape CrossEntropy::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the CrossEntropy Function needs only 2 input");
    DEEP8_ARGUMENT_CHECK(inputShapes[0].size() == inputShapes[1].size(), "inputs's shape size must be equal");
    DEEP8_ARGUMENT_CHECK(inputShapes[0].batch == inputShapes[1].batch, "inputs's batch must be equal");

    return Shape(inputShapes[0].batch, { 1 });
}

ElementType CrossEntropy::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 2");

    return Function::checkElementType(inputTypes);
}

void CrossEntropy::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::CrossEntropy(*(inputs[0]), *(inputs[1]), *output);
}

void CrossEntropy::backward(const std::vector<const Tensor*> &inputs, 
                            const Tensor *output, 
                            const Tensor *outputGradient, 
                            size_t index, 
                            Tensor *iGradient) {
    if (0 == index) {
        Math::CrossEntropyGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::CrossEntropyGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}



}