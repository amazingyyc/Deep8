#include "math/Sigmoid.h"
#include "nodes/Sigmoid.h"

namespace Deep8 {

Sigmoid::Sigmoid(std::vector<Node*> &inputs): Function(inputs) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Sigmoid Function needs 1 input");
}

Shape Sigmoid::checkShape(std::vector<Shape> &inputShapes) {
	DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

	return inputShapes[0];
}

ElementType Sigmoid::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Sigmoid::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Sigmoid(*(inputs[0]), *output);
}

void Sigmoid::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	Math::SigmoidGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}