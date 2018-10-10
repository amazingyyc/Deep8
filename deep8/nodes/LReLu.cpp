#include "LReLu.h"

namespace Deep8 {

template <typename T>
struct LReLuForwardExpr {
	T a;

	explicit LReLuForwardExpr(T p) : a(p) {
	}

	inline T operator()(T in) const {
		return ((in > T(0.0)) ? in : a * in);
	}
};

template <typename T>
struct LReLuBackwardExpr {
	T a;

	explicit LReLuBackwardExpr(T p) : a(p) {
	}

	inline T operator()(T outputGrad, T in) const {
		return outputGrad * (in > T(0.0) ? T(1.0) : a);
	}
};

template <typename T>
void LReLu<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the LReLu Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
void LReLu<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;
	eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(LReLuForwardExpr<T>(a));
}

#ifdef HAVE_HALF
template <>
void LReLu<half>::forwardCPU(const std::vector<const Tensor<half>*> &inputs, Tensor<half> *output) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif // HAVE_HALF

template <typename T>
void LReLu<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;
	eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), LReLuBackwardExpr<T>(a));
}

#ifdef HAVE_HALF
template <>
void LReLu<half>::backwardCPU(const std::vector<const Tensor<half>*> &inputs, const Tensor<half> *output, const Tensor<half> *outputGradient, size_t index, Tensor<half> *iGradient) {
	DEEP8_RUNTIME_ERROR("CPU not support half");
}
#endif // HAVE_HALF


DEEP8_DECLARATION_INSTANCE(LReLu)

}