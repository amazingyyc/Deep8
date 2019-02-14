#include "math/Minus.h"
#include "Minus.h"

namespace Deep8 {

Minus::Minus(std::vector<Node *> &inputs): Function(inputs) {
		check();
}

void Minus::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs size must be 2 in Add Function");

	/**
	 * the Minus Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
	 */
	this->outputShape = broadcastShape(this->inputs[0]->outputShape, this->inputs[1]->outputShape);
}

void Minus::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::Minus(*(inputs[0]), *(inputs[1]), *output);
}

void Minus::backward(const std::vector<const Tensor*> &inputs, 
                    const Tensor *output, 
                    const Tensor *outputGradient, 
                    size_t index, 
                    Tensor *iGradient) {
    if (0 == index) {
        Math::MinusGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::MinusGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

}