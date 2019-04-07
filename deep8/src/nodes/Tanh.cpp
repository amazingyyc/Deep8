#include "math/Tanh.h"
#include "nodes/Tanh.h"

namespace Deep8 {

Tanh::Tanh(std::vector<Node *> &inputs) : Function(inputs) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Tanh Function needs 1 input");
}

Shape Tanh::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType Tanh::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}
void Tanh::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Tanh(*(inputs[0]), *output);
}

void Tanh::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	Math::TanhGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}