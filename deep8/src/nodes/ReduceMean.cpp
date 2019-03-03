#include "math/ReduceMean.h"
#include "nodes/ReduceMean.h"

namespace Deep8 {

ReduceMean::ReduceMean(std::vector<Node*> &inputs, int a, bool keep) : Function(inputs), axis(a), keepDims(keep) {
    check();
}

void ReduceMean::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "only have 1 input");
    DEEP8_ARGUMENT_CHECK(axis < this->inputs[0]->shape.nDims, "the axis is error");

    auto shape = this->inputs[0]->shape;
    std::vector<size_t> list;

    if (axis < 0) {
        if (keepDims) {
            for (int i = 0; i < (int)shape.nDims; ++i) {
                list.emplace_back(1);
            }
        } else {
            list.emplace_back(1);
        }
    }
    else {
        for (int i = 0; i < (int)shape.nDims; ++i) {
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

void ReduceMean::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::ReduceMean(*(inputs[0]), *output, axis);
}

void ReduceMean::backward(const std::vector<const Tensor*> &inputs, 
                        const Tensor *output, 
                        const Tensor *outputGradient, 
                        size_t index, 
                        Tensor *iGradient) {
    Math::ReduceMeanGrad(*(inputs[0]), *iGradient, *output, *outputGradient, axis);
}


}