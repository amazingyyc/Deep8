#include "math/Multiply.h"
#include "nodes/Multiply.h"

namespace Deep8 {

Multiply::Multiply(std::vector<Node *> &inputs) : Function(inputs) {
	check();
}

void Multiply::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs dim must be 2 in Multiply Function");

	/**
	 * the Add Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
	 */
	this->outputShape = broadcastShape(this->inputs[0]->outputShape, this->inputs[1]->outputShape);
}

void Multiply::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::Multiply(*(inputs[0]), *(inputs[1]), *output);
}

void Multiply::backward(const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    if (0 == index) {
        Math::MultiplyGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::MultiplyGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}



}