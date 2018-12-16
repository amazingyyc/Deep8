#include "Sigmoid.h"
#include "AutoBatchCodeHelper.h"

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
Sigmoid<T>::Sigmoid(std::vector<Node*> &inputs): Function<T>(inputs) {
		check();
}

template <typename T>
void Sigmoid<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Sigmoid Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
int Sigmoid<T>::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
template <typename T>
size_t Sigmoid<T>::autoBatchCode() {
    AutoBatchCodeHelper helper;

    helper.functionType(FunctionType::Sigmoid);

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
template <typename T>
Shape Sigmoid<T>::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error!");

    /**simple set it to a 1 batch shape*/
    size_t size = 0;

    for (auto item : shapes) {
        size += item.size();
    }

    return Shape({ size });
}

/**
 * return the inputs's index that can be auto batched
 */
template <typename T>
std::vector<size_t> Sigmoid<T>::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
template <typename T>
Node* Sigmoid<T>::autoBatchClone(std::vector<Node*> &inputs) {
	return new Sigmoid<T>(inputs);
}

template <typename T>
void Sigmoid<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;
	eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(SigmoidForwardExpr<T>());
}

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

DEEP8_RE_DECLARATION_HALF_FUNC(Sigmoid);
DEEP8_DECLARATION_INSTANCE(Sigmoid);

}