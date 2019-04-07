#include "math/Linear.h"
#include "nodes/Linear.h"

namespace Deep8 {

Linear::Linear(std::vector<Node*> &inputs, float a, float b):Function(inputs), a(a), b(b) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Linear Function needs 1 input");
}

Shape Linear::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType Linear::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Linear::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Linear(*(inputs[0]), a, b, *output);
}

void Linear::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");
	
	Math::LinearGrad(*(inputs[0]), *iGradient, a, b, *output, *outputGradient);
}

}