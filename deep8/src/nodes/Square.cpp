#include "math/Square.h"
#include "nodes/Square.h"

namespace Deep8 {

Square::Square(std::vector<Node*> &inputs): Function(inputs) {
	check();
}

void Square::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Square Function needs only 1 input");

    this->shape = this->inputs[0]->shape;
    this->elementType = this->inputs[0]->elementType;
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