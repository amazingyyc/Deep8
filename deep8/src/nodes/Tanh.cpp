#include "Tanh.h"

namespace Deep8 {

template <typename T>
struct TanHForwardExpr {
	inline T operator()(T in) const {
		return tanh(in);
	}
};

template <typename T>
struct TanHBackwardExpr {
	inline T operator()(T outputGrad, T output) const {
		return outputGrad * (T(1.0) - output * output);
	}
};

template <typename T>
Tanh<T>::Tanh(std::vector<Node *> &inputs) : Function<T>(inputs) {
		check();
}

template <typename T>
void Tanh<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Tanh Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void Tanh<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(TanHForwardExpr<T>());
}


template <typename T>
void Tanh<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
							const Tensor<T> *output,
							const Tensor<T> *outputGradient,
							size_t index,
							Tensor<T> *iGradient) {
	if (0 != index) {
		DEEP8_RUNTIME_ERROR("the index of Tanh backwardCPU is error");
	}

	auto device = static_cast<CPUDevice*>(iGradient->device())->eigenDevice;

	eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(output), TanHBackwardExpr<T>());
}

DEEP8_RE_DECLARATION_HALF_FUNC(Tanh);
DEEP8_DECLARATION_INSTANCE(Tanh);

}