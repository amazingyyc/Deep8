#include "math/Log.h"
#include "nodes/Log.h"

namespace Deep8 {


Log::Log(std::vector<Node *> &inputs): Function(inputs) {
	check();
}

void Log::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Log Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

void Log::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
	Math::Log(*(inputs[0]), *output);
}

void Log::backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	Math::LogGrad(*(inputs[0]), *iGradient, *output, *outputGradient);
}




}