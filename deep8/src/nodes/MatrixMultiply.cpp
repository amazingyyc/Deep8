#include "math/MatrixMultiply.h"
#include "nodes/MatrixMultiply.h"

namespace Deep8 {

MatrixMultiply::MatrixMultiply(std::vector<Node *> &inputs) : Function(inputs) {
    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the MatrixMultiply Function needs 2 input");
}

Shape MatrixMultiply::checkShape(std::vector<Shape> &inputShapes) {
    DEEP8_ARGUMENT_CHECK(2 == inputShapes.size(), "the input count must be 2");

    auto xShape = inputShapes[0];
    auto yShape = inputShapes[1];

    DEEP8_ARGUMENT_CHECK(xShape.batch == yShape.batch || 1 == xShape.batch || 1 == yShape.batch, "the batch of input is error");
    DEEP8_ARGUMENT_CHECK((1 == xShape.nDims || 2 == xShape.nDims) &&
                         (1 == yShape.nDims || 2 == yShape.nDims),
                         "the inputs dimensions is error");
    DEEP8_ARGUMENT_CHECK(xShape.col() == yShape.row(),
                         "the col of input1 must same to the row of input2");

    if (1 == yShape.col()) {
        return Shape(std::max<size_t>(xShape.batch, yShape.batch), {xShape.row()});
    } else {
        return Shape(std::max<size_t>(xShape.batch, yShape.batch), {xShape.row(), yShape.col()});
    }
}

ElementType MatrixMultiply::checkElementType(std::vector<ElementType> &inputTypes) {
    DEEP8_ARGUMENT_CHECK(2 == inputTypes.size(), "the input count must be 2");

    return Function::checkElementType(inputTypes);
}

void MatrixMultiply::forward(const std::vector<const Tensor*> &inputs, Tensor *output) {
    Math::MatrixMultiply(*(inputs[0]), *(inputs[1]), *output);
}

void MatrixMultiply::backward(const std::vector<const Tensor*> &inputs, 
                            const Tensor *output, 
                            const Tensor *outputGradient, 
                            size_t index, 
                            Tensor *iGradient) {
    if (0 == index) {
        Math::MatrixMultiplyGradX(*(inputs[0]), *iGradient, *(inputs[1]), *output, *outputGradient);
    } else if (1 == index) {
        Math::MatrixMultiplyGradY(*(inputs[0]), *(inputs[1]), *iGradient, *output, *outputGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}





}