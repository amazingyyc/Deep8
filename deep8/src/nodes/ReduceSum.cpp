#include "math/ReduceSum.h"
#include "nodes/ReduceSum.h"

namespace Deep8 {

ReduceSum::ReduceSum(std::vector<Node*> &inputs, int a, bool keep) : Function(inputs), axis(a), keepDims(keep) {
    check();
}

void ReduceSum::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "only have 1 input");

    auto inputShape = this->inputs[0]->shape;

    if (axis == -1) {
        axis = (int)inputShape.nDims - 1;
    }

    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int) inputShape.nDims, "the axis is error");

    std::vector<size_t> list;

    for (int i = 0; i < (int)inputShape.nDims; ++i) {
        if (i == axis) {
            if (keepDims) {
                list.emplace_back(1);
            }
        } else {
            list.emplace_back(shape[i]);
        }
    }

    this->shape = Shape(inputShape.batch, list);
    this->elementType = this->inputs[0]->elementType;
}

void ReduceSum::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::ReduceSum(*(inputs[0]), *output, axis);
}

void ReduceSum::backward(const std::vector<const Tensor*> &inputs, 
                        const Tensor *output, 
                        const Tensor *outputGradient, 
                        size_t index, 
                        Tensor *iGradient) {
    Math::ReduceSumGrad(*(inputs[0]), *iGradient, *output, *outputGradient, axis);
}



}