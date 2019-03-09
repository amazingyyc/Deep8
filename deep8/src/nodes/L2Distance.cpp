#include "math/L2Distance.h"
#include "nodes/L2Distance.h"

namespace Deep8 {

L2Distance::L2Distance(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void L2Distance::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the L2Distance Function needs 2 input");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->elementType == this->inputs[1]->elementType, "the input elementType must be same");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->shape == this->inputs[1]->shape, "the input shape must be same");

	this->shape       = Shape(this->inputs[0]->shape.batch, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void L2Distance::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L2Distance(*(inputs[0]), *(inputs[1]), *output);
}

void L2Distance::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
    if (0 == index) {
        Math::L2DistanceGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::L2DistanceGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

}