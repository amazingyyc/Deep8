#include "ReLu.h"
#include "AutoBatchCodeHelper.h"

namespace Deep8 {

template <typename T>
struct ReLuBackwardExpr {
	inline T operator()(T outputGrad, T in) const {
		return outputGrad * (in > 0 ? 1.0 : 0);
	}
};

template <typename T>
ReLu<T>::ReLu(std::vector<Node *> &inputs) : Function<T>(inputs) {
	check();
}

template <typename T>
void ReLu<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReLu Function needs only 1 input");

	/**the ReLu output shape equal the input*/
	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
int ReLu<T>::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
template <typename T>
size_t ReLu<T>::autoBatchCode() {
    AutoBatchCodeHelper helper;

	// todo: aupport half
    helper.functionType(FunctionType::ReLu);

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
template <typename T>
Shape ReLu<T>::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error!");

    /**simple set it to a 1 batch shape*/
    size_t size = 0;

    for (auto item : shapes) {
        size += item.size();
    }

    std::vector<size_t> vec({1});
    return Shape(1, vec);
}

/**
 * return the inputs's index that can be auto batched
 */
template <typename T>
std::vector<size_t> ReLu<T>::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
template <typename T>
Node* ReLu<T>::autoBatchClone(std::vector<Node*> &inputs) {
	return new ReLu<T>(inputs);
}

template <typename T>
void ReLu<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;
	eTVec(output).device(*device) = eTVec(inputs[0]).cwiseMax(T(0));
}

template <typename T>
void ReLu<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
				const Tensor<T> *output,
				const Tensor<T> *outputGradient,
				size_t index,
				Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of ReLu backwardCPU is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;
	eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), ReLuBackwardExpr<T>());
}

DEEP8_RE_DECLARATION_HALF_FUNC(ReLu);
DEEP8_DECLARATION_INSTANCE(ReLu)

}