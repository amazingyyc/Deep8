#include "math/L2Norm.h"
#include "nodes/L2Norm.h"

namespace Deep8 {

L2Norm::L2Norm(std::vector<Node *> &inputs): Function(inputs) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L2Norm Function needs 1 input");
}

Shape L2Norm::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return Shape(inputShapes[0].batch, { 1 });
}

ElementType L2Norm::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void L2Norm::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L2Norm(*(inputs[0]), *output);
}

void L2Norm::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backward is error");

    Math::L2NormGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}