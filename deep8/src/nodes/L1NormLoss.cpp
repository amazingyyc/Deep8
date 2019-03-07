#include "math/L1NormLoss.h"
#include "nodes/L1NormLoss.h"

namespace Deep8 {

L1NormLoss::L1NormLoss(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void L1NormLoss::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1NormLoss Function needs only 1 input");

	this->shape       = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void L1NormLoss::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L1NormLoss(*(inputs[0]), *output);
}

void L1NormLoss::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1NormLoss backward is error");

    Math::L1NormLossGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}

