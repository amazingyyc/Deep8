#include "utils/ShapeUtils.h"
#include "math/ReduceMean.h"
#include "nodes/ReduceMean.h"

namespace Deep8 {

ReduceMean::ReduceMean(std::vector<Node*> &inputs, std::vector<int> reduceAxis, bool keep) : Function(inputs), axis(reduceAxis), keepDims(keep) {
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the ReduceMean Function needs 1 input");
}

Shape ReduceMean::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "only have 1 input");

    auto inputShape = convertShapeToVector(inputShapes[0]);

    int rank = inputShape.size();

    std::vector<bool> reduceAxis(rank);

    if (axis.empty()) {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = true;
        }
    } else {
        for (int i = 0; i < rank; ++i) {
            reduceAxis[i] = false;
        }

        for (int i = 0; i < axis.size(); ++i) {
            DEEP8_ARGUMENT_CHECK(-rank <= axis[i] && axis[i] < rank, "the reduce dim is error");

            axis[i] = (axis[i] + rank) % rank;
            
            reduceAxis[axis[i]] = true;
        }
    }

    DEEP8_ARGUMENT_CHECK(reduceAxis.size() >= 2, "the shape is error");

    size_t outputBatch = reduceAxis[0] ? 1 : inputShape[0];

    std::vector<size_t> list;

    for(int i = 1; i < rank; ++i) {
        if (reduceAxis[i]) {
            if (keepDims) {
                list.emplace_back(1);
            }
        } else {
            list.emplace_back(inputShape[i]);
        }
    }

    if (list.empty()) {
        list.emplace_back(1);
    }

    return Shape(outputBatch, list);
}

ElementType ReduceMean::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void ReduceMean::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::ReduceMean(*(inputs[0]), *output, axis, keepDims);
}

void ReduceMean::backward(const std::vector<const Tensor*> &inputs, const Tensor *output, const Tensor *outputGradient, size_t index, Tensor *iGradient) {
    Math::ReduceMeanGrad(*(inputs[0]), *iGradient, *output, *outputGradient, axis, keepDims);
}


}