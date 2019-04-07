#include "math/L1Norm.h"
#include "nodes/L1Norm.h"

namespace Deep8 {

L1Norm::L1Norm(std::vector<Node *> &inputs): Function(inputs) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1Norm Function needs 1 input");
}

Shape L1Norm::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return Shape(inputShapes[0].batch, { 1 });
}

ElementType L1Norm::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void L1Norm::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L1Norm(*(inputs[0]), *output);
}

void L1Norm::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1NormLoss backward is error");

    Math::L1NormGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}

