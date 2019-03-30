#include "math/Softmax.h"
#include "nodes/Softmax.h"

namespace Deep8 {

Softmax::Softmax(std::vector<Node *> &inputs, int a): Function(inputs), axis(a) {    
    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the Softmax Function needs 1 input");
}

Shape Softmax::checkShape(std::vector<Shape> &inputShapes) {
	DEEP8_ARGUMENT_CHECK(1 == inputShapes.size(), "the input count must be 1");
    DEEP8_ARGUMENT_CHECK(-1 <= axis && axis < (int) inputShapes[0].nDims, "the axis is error");

	return inputShapes[0];
}

ElementType Softmax::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(1 == inputTypes.size(), "the input count must be 1");

    return Function::checkElementType(inputTypes);
}

void Softmax::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int)inputs[0]->shape.nDims, "the axis is error");

    auto device = output->device();

    auto shape = inputs[0]->shape;
	int dim0, dim1, dim2;

    dim0 = (int) shape.batch;
    dim1 = (int) shape.dim(axis);
    dim2 = 1;

    for (int i = 0; i < axis; ++i) {
        dim0 *= (int) shape.dim(i);
    }

    for (int i = axis + 1; i < shape.nDims; ++i) {
        dim2 *= (int) shape.dim(i);
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
    DEEP8_ARGUMENT_CHECK(0 <= axis && axis < (int)inputs[0]->shape.nDims, "the axis is error");

    auto device = iGradient->device();

	auto shape = iGradient->shape;
    int dim0, dim1, dim2;

    dim0 = (int) shape.batch;
    dim1 = (int) shape.dim(axis);
    dim2 = 1;

    for (int i = 0; i < axis; ++i) {
        dim0 *= (int) shape.dim(i);
    }

    for (int i = axis + 1; i < shape.nDims; ++i) {
        dim2 *= (int) shape.dim(i);
    }

    auto ptr = device->malloc(iGradient->elementType.byteWidth * dim0 * dim2);

    Math::SoftmaxGrad(*(inputs[0]), *iGradient, *output, *outputGradient, axis, ptr);

    device->free(ptr);
}




}