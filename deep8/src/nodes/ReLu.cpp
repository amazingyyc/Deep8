#include "math/ReLu.h"
#include "nodes/ReLu.h"

namespace Deep8 {

ReLu::ReLu(std::vector<Node *> &inputs) : Function(inputs) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReLu Function needs 1 input");
}

Shape ReLu::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType ReLu::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void ReLu::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::ReLu(*(inputs[0]), *output);
}

void ReLu::backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) {
	Math::ReLuGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}