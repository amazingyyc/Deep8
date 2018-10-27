#include "Square.h"

namespace Deep8 {

template <typename T>
Square<T>::Square(std::vector<Node*> &inputs): Function<T>(inputs) {
	check();
}

template <typename T>
void Square<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Square Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Square<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	eTVec(output).device(*device) = eTVec(inputs[0]).square();
}

template <typename T>
void Square<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	if (0 != index) {
		DEEP8_RUNTIME_ERROR("the index of Linear backwardCPU is error");
	}

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(inputs[0]) * T(2);
}

DEEP8_RE_DECLARATION_HALF_FUNC(Square);
DEEP8_DECLARATION_INSTANCE(Square);

}