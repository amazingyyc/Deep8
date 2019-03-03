#include "math/L1Norm.h"
#include "nodes/L1Norm.h"

namespace Deep8 {

L1Norm::L1Norm(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void L1Norm::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L1Norm Function needs only 1 input");

	this->shape       = Shape(1, { 1 });
    this->elementType = this->inputs[0]->elementType;
}

void L1Norm::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::L1Norm(*(inputs[0]), *output);
}

void L1Norm::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L1Norm backward is error");

    Math::L1NormGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}


}

