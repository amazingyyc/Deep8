#include "math/Square.h"
#include "nodes/Square.h"

namespace Deep8 {

Square::Square(std::vector<Node*> &inputs): Function(inputs) {
	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Square Function needs 1 input");
}

Shape Square::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");

    return inputShapes[0];
}

ElementType Square::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Square::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Square(*(inputs[0]), *output);
}

void Square::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	Math::SquareGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}

}