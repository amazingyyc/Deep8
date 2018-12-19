#include "Abs.h"
#include "AutoBatchCodeHelper.h"

namespace Deep8 {

template <typename T>
struct AbsBackwardExpr {
    inline T operator()(T dy, T x) const {
        if (x >= T(0)) {
            return dy;
        } else {
            return -dy;
        }
    }
};

template <typename T>
Abs<T>::Abs(std::vector<Node *> &inputs) : Function<T>(inputs) {
	check();
}

template <typename T>
void Abs<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Abs Function needs only 1 input");

    this->outputShape = this->inputs[0]->outputShape;
}

/**
 * for Unary Function it can be auto bateched but default set it to not support auto-batch
 */
template <typename T>
int Abs<T>::supportAutoBatch() {
    return -1;
}

/**auto batch code*/
template <typename T>
size_t Abs<T>::autoBatchCode() {
    AutoBatchCodeHelper helper;

    helper.functionType(FunctionType::Abs);

    return helper.autoBatchCode();
}

/**
 * return the inputs[index]'s shape if it is be batched together.
 * the shapes is the inputs[index]'s shape that will be batched.
 */
template <typename T>
Shape Abs<T>::autoBatchShape(size_t index, std::vector<Shape> &shapes) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index is error!");

    /**simple set it to a 1 batch shape*/
    size_t size = 0;

    for (auto item : shapes) {
        size += item.size();
    }

    return Shape({size});
}

/**
 * return the inputs's index that can be auto batched
 */
template <typename T>
std::vector<size_t> Abs<T>::autoBatchIndexes() {
    return std::vector<size_t>({ 0 });
}

/**
 * clone current node for auto batch
 */
template <typename T>
Node* Abs<T>::autoBatchClone(std::vector<Node*> &inputs) {
    return new Abs<T>(inputs);
} 

template <typename T>
void Abs<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output)  {
    auto eigenDevice = static_cast<CPUDevice *>(output->device())->eigenDevice;

    eTVec(output).device(*eigenDevice) = eTVec(inputs[0]).abs();
}

template <typename T>
void Abs<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of Abs backwardCPU is error");

    auto eigenDevice = static_cast<CPUDevice *>(output->device())->eigenDevice;

    eTVec(iGradient).device(*eigenDevice) += eTVec(outputGradient).binaryExpr(eTVec(inputs[0]), AbsBackwardExpr<T>());
}

DEEP8_RE_DECLARATION_HALF_FUNC(Abs);
DEEP8_DECLARATION_INSTANCE(Abs);

}