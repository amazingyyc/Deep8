#include "math/Softmax.h"
#include "nodes/Softmax.h"

namespace Deep8 {

Softmax::Softmax(std::vector<Node *> &inputs, int a): Function(inputs), axis(a) {
    check();
}

void Softmax::check() {
	Function::check();

	DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Softmax Function needs only 1 input");

	auto inputShape = this->inputs[0]->shape;

	DEEP8_ARGUMENT_CHECK(axis < (int) inputShape.nDims, "the axis is error");

    this->shape = this->inputs[0]->shape;
    this->elementType = this->inputs[0]->elementType;
}

void Softmax::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
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

    auto ptr = device->malloc(output->elementType.byteWidth * dim0 * dim2);

    Math::Softmax(*(inputs[0]), *output, axis, ptr);

    device->free(ptr);
}

void Softmax::backward(const std::vector<const Tensor*> &inputs, 
					const Tensor *output, 
					const Tensor *outputGradient, 
					size_t index, 
					Tensor *iGradient) {
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

    auto ptr = device->malloc(iGradient->elementType.byteWidth * dim0 * dim2);

    Math::SoftmaxGrad(*(inputs[0]), *iGradient, *output, *outputGradient, axis, ptr);

    device->free(ptr);
}




}