#include "math/ReduceSum.h"
#include "nodes/ReduceSum.h"

namespace Deep8 {

ReduceSum::ReduceSum(std::vector<Node*> &inputs, int a, bool keep) : Function(inputs), axis(a), keepDims(keep) {
}

void ReduceSum::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "only have 1 input");
    DEEP8_ARGUMENT_CHECK(axis < this->inputs[0]->shape.nDims, "the axis is error");

    auto shape = this->inputs[0]->shape;
    std::vector<size_t> list;

    if (axis < 0) {
        if (keepDims) {
            for (int i = 0; i < (int) shape.nDims; ++i) {
                list.emplace_back(1);
            }
        } else {
            list.emplace_back(1);
        }
    } else {
        for (int i = 0; i < (int) shape.nDims; ++i) {
            if (i == axis) {
                if (keepDims) {
                    list.emplace_back(1);
                }
            } else {
                list.emplace_back(shape[i]);
            }
        }
    }

    this->shape = Shape(shape.batch, list);
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