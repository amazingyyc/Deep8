#include "math/LogSoftmax.h"
#include "nodes/LogSoftmax.h"

namespace Deep8 {

LogSoftmax::LogSoftmax(std::vector<Node *> &inputs, int a): Function(inputs), axis(a) {
    check();
}

void LogSoftmax::check() {
    Function::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the LogSoftmax Function needs only 1 input");

    auto inputShape = this->inputs[0]->shape;

    DEEP8_ARGUMENT_CHECK(axis < (int) inputShape.nDims, "the axis is error");

    this->shape = inputShape;
    this->elementType = this->inputs[0]->elementType;
}

void LogSoftmax::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    auto device = output->device();

    auto shape = inputs[0]->shape;
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) shape.dim(i);
        }

        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int) shape.dim(i);
        }
    }

    auto maxptr = device->malloc(output->elementType.byteWidth * dim0 * dim2);
    auto sumptr = device->malloc(output->elementType.byteWidth * dim0 * dim2);

    Math::LogSoftmax(*(inputs[0]), *output, axis, maxptr, sumptr);

    device->free(sumptr);
    device->free(maxptr);
}

void LogSoftmax::backward(const std::vector<const Tensor*> &inputs, 
                        const Tensor *output, 
                        const Tensor *outputGradient, 
                        size_t index, 
                        Tensor *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of LogSoftmax backward is error");

    auto device = iGradient->device();

    auto shape = iGradient->shape;
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.batchSize();
        dim2 = 1;
    } else {
        dim0 = (int) shape.batch;
        dim1 = (int) shape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int) shape.dim(i);
        }

        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int) shape.dim(i);
        }
    }

    auto sumptr = device->malloc(iGradient->elementType.byteWidth * dim0 * dim2);

    Math::LogSoftmaxGrad(*(inputs[0]), *iGradient, *output, *outputGradient, axis, sumptr);

    device->free(sumptr);
}




}   