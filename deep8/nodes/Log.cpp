#include  "Log.h"

namespace Deep8 {

template <typename T>
void Log<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Log Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Log<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	eTVec(output).device(*device) = eTVec(inputs[0]).log();
}

template <typename T>
void Log<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
						const Tensor<T> *output,
						const Tensor<T> *outputGradient,
						size_t index,
						Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), LogBackwardExpr<T>());
}

DEEP8_DECLARATION_INSTANCE(Log)

}