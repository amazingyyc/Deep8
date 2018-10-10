#include "Pow.h"

namespace Deep8 {

template <typename T>
void Pow<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in Pow Function");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Pow<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;
	eTVec(output).device(*device) = eTVec(inputs[0]).pow(scalar);
}

#ifdef HAVE_HALF
template <>
void Pow<half>::forwardCPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif

template <typename T>
void Pow<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
				const Tensor<T> *output,
				const Tensor<T> *outputGradient,
				size_t index,
				Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of Pow backwardCPU is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(inputs[0]).pow(scalar - T(1)) * scalar;
}

#ifdef HAVE_HALF
template <>
void Pow<half>::backwardCPU<half>(const std::vector<const Tensor<half>*> &inputs, const Tensor<half> *output, const Tensor<half> *outputGradient, size_t index, Tensor<half> *iGradient) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif // HAVE_HALF

DEEP8_DECLARATION_INSTANCE(Pow)

}