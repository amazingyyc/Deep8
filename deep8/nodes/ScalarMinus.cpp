#include "ScalarMinus.h"

namespace Deep8 {

template <typename T>
void ScalarMinus<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in ScalarMinus Function");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void ScalarMinus<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	eTVec(output).device(*device) = -eTVec(inputs[0]) + scalar;
}

template <typename T>
void ScalarMinus<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
								const Tensor<T> *output,
								const Tensor<T> *outputGradient,
								size_t index,
								Tensor<T> *iGradient) {
	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	eTVec(iGradient).device(*device) -= eTVec(outputGradient);
}

DEEP8_DECLARATION_INSTANCE(ScalarMinus)

}