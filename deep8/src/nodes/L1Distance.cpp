#include "math/L1Distance.h"
#include "nodes/L1Distance.h"

namespace Deep8 {

L1Distance::L1Distance(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void L1Distance::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the L1Distance Function needs 2 input");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the input elementType must be same");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape == this->inputs[1]->shape, "the input shape must be same");

	this->shape       = Shape(this->inputs[0]->shape.batch, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void L1Distance::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L1Distance(*(inputs[0]), *(inputs[1]), *output);
}

void L1Distance::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
    if (0 == index) {
        Math::L1DistanceGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::L1DistanceGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

}