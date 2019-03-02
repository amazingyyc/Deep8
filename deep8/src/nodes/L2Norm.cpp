#include "math/L2Norm.h"
#include "nodes/L2Norm.h"

namespace Deep8 {

L2Norm::L2Norm(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void L2Norm::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L2Norm Function needs only 1 input");

	this->shape       = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void L2Norm::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L2Norm(*(inputs[0]), *output);
}

void L2Norm::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backward is error");

    Math::L2NormGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}