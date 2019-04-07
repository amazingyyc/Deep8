#include "math/Log.h"
#include "nodes/Log.h"

namespace Deep8 {

Log::Log(std::vector<Node *> &inputs): Function(inputs) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Log Function needs 1 input");
}

Shape Log::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType Log::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Log::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Log(*(inputs[0]), *output);
}

void Log::backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	Math::LogGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}




}