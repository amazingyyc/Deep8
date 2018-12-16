#include "Linear.h"
#include "AutoBatchCodeHelper.h"

namespace Deep8 {

template <typename T>
Linear<T>::Linear(std::vector<Node*> &inputs, T a, T b):Function<T>(inputs), a(a), b(b) {
	check();
}

template <typename T>
void Linear<T>::check() {
	Function<T>::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Linear Function needs only 1 input");

	this->outputShape = this->inputs[0]->outputShape;
}

template <typename T>
int Linear<T>::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
template <typename T>
size_t Linear<T>::autoBatchCode() {
    AutoBatchCodeHelper helper;

	// todo: aupport half
    helper.functionType(FunctionType::Linear);
	helper.attachBegin();
	helper.put("a", a);
	helper.put("b", a);
	helper.attachEnd();

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
template <typename T>
Shape Linear<T>::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
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
std::vector<size_t> Linear<T>::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
template <typename T>
Node* Linear<T>::autoBatchClone(std::vector<Node*> &inputs) {
	return new Linear<T>(inputs, a, b);
} 

template <typename T>
void Linear<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
	auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

	eTVec(output).device(*device) = eTVec(inputs[0]) * a + b;
}

template <typename T>
void Linear<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
							const Tensor<T> *output,
							const Tensor<T> *outputGradient,
							size_t index,
							Tensor<T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

	auto device = static_cast<CPUDevice*>(outputGradient->device())->eigenDevice;

	eTVec(iGradient).device(*device) += eTVec(outputGradient) * a;
}

DEEP8_RE_DECLARATION_HALF_FUNC(Linear);
DEEP8_DECLARATION_INSTANCE(Linear)

}