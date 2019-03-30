#include "math/L1Distance.h"
#include "nodes/L1Distance.h"

namespace Deep8 {

L1Distance::L1Distance(std::vector<Node *> &inputs): Function(inputs) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the L1Distance Function needs 2 input");
}

Shape L1Distance::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the input count must be 2");
    DEEP8_ARGUMENT_CHECK(inputShapes[0] == inputShapes[1], "the input shape must be same");

    return Shape(inputShapes[0].batch, { 1 });
}

ElementType L1Distance::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 2");

    return Function::checkElementType(inputTypes);
}

void L1Distance::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L1Distance(*(inputs[0]), *(inputs[1]), *output);
}

void L1Distance::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
    if (0 == index) {
        Math::L1DistanceGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::L1DistanceGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

}