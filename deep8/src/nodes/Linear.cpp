#include "Linear.h"

namespace Deep8 {

template <typename T>
Linear<T>::Linear(std::vector<Node*> &inputs, T a, T b):Function<T>(inputs), a(a), b(b) {
	check();
}

template <typename T>
void Linear<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Linear Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Linear<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	eTVec(output).device(*device) = eTVec(inputs[0]) * a + b;
}

template <typename T>
void Linear<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
							const Tensor<T> *output,
							const Tensor<T> *outputGradient,
							size_t index,
							Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	eTVec(iGradient).device(*device) += eTVec(outputGradient) * a;
}

DEEP8_RE_DECLARATION_HALF_FUNC(Linear);
DEEP8_DECLARATION_INSTANCE(Linear)

}