#include "LReLu.h"
#include "AutoBatchCodeHelper.h"

namespace Deep8 {

template <typename T>
LReLu<T>::LReLu(std::vector<Node*> &inputs, T a): Function<T>(inputs), a(a) {
	check();
}

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
int LReLu<T>::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
template <typename T>
size_t LReLu<T>::autoBatchCode() {
    AutoBatchCodeHelper helper;

	// todo: aupport half
    helper.functionType(FunctionType::LReLu);
	helper.attachBegin();
	helper.put("a", a);
	helper.attachEnd();

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
template <typename T>
Shape LReLu<T>::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
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
std::vector<size_t> LReLu<T>::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
template <typename T>
Node* LReLu<T>::autoBatchClone(std::vector<Node*> &inputs) {
	return new LReLu<T>(inputs, a);
}

template <typename T>
void LReLu<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;
	eTVec(output).device(*device) = eTVec(inputs[0]).unaryExpr(LReLuForwardExpr<T>(a));
}

template <typename T>
void LReLu<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;
	eTVec(iGradient).device(*device) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), LReLuBackwardExpr<T>(a));
}

DEEP8_RE_DECLARATION_HALF_FUNC(LReLu)
DEEP8_DECLARATION_INSTANCE(LReLu)

}