#include "math/Sqrt.h"
#include "nodes/Sqrt.h"

namespace Deep8 {

Sqrt::Sqrt(std::vector<Node*> &inputs): Function(inputs) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Sqrt Function needs 1 input");
}

Shape Sqrt::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType Sqrt::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Sqrt::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Sqrt(*(inputs[0]), *output);
}

void Sqrt::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	Math::SqrtGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}