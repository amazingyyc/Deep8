#include "ScalarDivide.h"

namespace Deep8 {

template <typename T>
struct ScalarDivideForwardExpr {
	T scalar;

	explicit ScalarDivideForwardExpr(T s) : scalar(s) {
	}

	inline T operator()(T in) const {
		return scalar / in;
	}
};

template <typename T>
struct ScalarDivideBackwardExpr {
	T scalar;

	explicit ScalarDivideBackwardExpr(T s) : scalar(s) {
	}

	inline T operator()(T in) const {
		return -scalar / (in * in);
	}
};

template <typename T>
void ScalarDivide<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the inputs size must be 1 in ScalarDivide Function");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void ScalarDivide<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(ScalarDivideForwardExpr<T>(scalar));
}

template <typename T>
void ScalarDivide<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
								const Tensor<T> *output,
								const Tensor<T> *outputGradient,
								size_t index,
								Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	eTVec(iGradient).device(*device) += eTVec(outputGradient) * eTVec(inputs[0]).unaryExpr(ScalarDivideBackwardExpr<T>(scalar));
}

DEEP8_DECLARATION_INSTANCE(ScalarDivide)

}