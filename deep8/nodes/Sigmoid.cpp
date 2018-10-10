#include "Sigmoid.h"

namespace Deep8 {

template <typename T>
struct SigmoidForwardExpr {
	inline T operator()(T in) const {
		return T(0.5) + T(0.5) * tanh(T(0.5) * in);
	}
};

template <typename T>
struct SigmoidBackwardExpr {
	inline T operator()(T outputGrad, T output) const {
		return outputGrad * output * (T(1) - output);
	}
};

template <typename T>
void Sigmoid<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Sigmoid Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Sigmoid<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;
	eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(SigmoidForwardExpr<T>());
}

#ifdef HAVE_HALF
template <>
void Sigmoid<half>::forwardCPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif // HAVE_HALF


template <typename T>
void Sigmoid<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
				const Tensor<T> *output,
				const Tensor<T> *outputGradient,
				size_t index,
				Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of Sigmoid backwardCPU is error");

	auto device = static_cast<CPUDevice*>(iGradient->device())->eigenDevice;
	eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(output), SigmoidBackwardExpr<T>());
}

#ifdef HAVE_HALF
template <>
void Sigmoid<half>::backwardCPU(const std::vector<const Tensor<half>*> &inputs,
								const Tensor<half> *output,
								const Tensor<half> *outputGradient,
								size_t index,
								Tensor<half> *iGradient) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif // HAVE_HALF

DEEP8_DECLARATION_INSTANCE(Sigmoid)

}