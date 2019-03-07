#include "math/L2NormLoss.h"
#include "nodes/L2NormLoss.h"

namespace Deep8 {

L2NormLoss::L2NormLoss(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void L2NormLoss::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L2Norm Function needs only 1 input");

	this->shape       = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void L2NormLoss::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L2NormLoss(*(inputs[0]), *output);
}

void L2NormLoss::backward(const std::vector<const Tensor*> &inputs,
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backward is error");

    Math::L2NormLossGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}