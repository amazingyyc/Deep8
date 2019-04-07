#include "math/LReLu.h"
#include "nodes/LReLu.h"

namespace Deep8 {

LReLu::LReLu(std::vector<Node*> &inputs, float a): Function(inputs), a(a) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the LReLu Function needs 1 input");
}

Shape LReLu::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType LReLu::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void LReLu::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::LReLu(*(inputs[0]), a, *output);
}

void LReLu::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	Math::LReLuGrad(*(inputs[0]), *iGradient, a, *output, *outputGradient);
}



}