#include "math/Dot.h"
#include "nodes/Dot.h"

namespace Deep8 {

Dot::Dot(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void Dot::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the Dot Function needs 2 input");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the input elementType must be same");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape == this->inputs[1]->shape, "the input shape must be same");

	this->shape       = Shape(this->inputs[0]->shape.batch, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void Dot::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Dot(*(inputs[0]), *(inputs[1]), *output);
}

void Dot::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
    if (0 == index) {
        Math::DotGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::DotGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

}