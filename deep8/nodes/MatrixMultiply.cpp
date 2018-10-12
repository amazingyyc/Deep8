#include "MatrixMultiply.h"

namespace Deep8 {
template <typename T>
MatrixMultiply<T>::MatrixMultiply(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
}

template <typename T>
void MatrixMultiply<T>::check() {
    Function <T> ::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs dim must be 2");

    auto xValue = static_cast<Variable<T> *>(this->inputs[0])->value;
    auto yValue = static_cast<Variable<T> *>(this->inputs[1])->value;

    auto xShape = xValue.shape;
    auto yShape = yValue.shape;

    DEEP8_ARGUMENT_CHECK(
            xShape.batch() == yShape.batch() || 1 == xShape.batch() || 1 == yShape.batch(),
            "the batch of input is error");
    DEEP8_ARGUMENT_CHECK((2 == xShape.nDims() || 3 == xShape.nDims()) &&
                         (2 == yShape.nDims() || 3 == yShape.nDims()),
                         "the inputs dimensions is error");
    DEEP8_ARGUMENT_CHECK(xShape.col() == yShape.row(),
                         "the col of input1 must same to the row of input2");

    if (1 == yShape.col()) {
        this->outputShape = Shape({std::max<size_t>(xShape.batch(), yShape.batch()), xShape.row()});
    } else {
        this->outputShape = Shape({std::max<size_t>(xShape.batch(), yShape.batch()), xShape.row(), yShape.col()});
    }
}

template <typename T>
void MatrixMultiply<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output)  {
    auto xTensor = inputs[0];
    auto yTensor = inputs[1];

    if (1 == yTensor->batch()) {
        eRowBatchMat(output).noalias() = eRowBatchMat(xTensor) * eMat(yTensor);
    } else if (1 == xTensor->batch() && 1 == yTensor->col()) {
        eBatchSizeMat(output).noalias() = eBatchSizeMat(yTensor) * eMat(xTensor).transpose();
    } else {
        DEEP8_ARGUMENT_CHECK(1 == xTensor->batch() || xTensor->batch() == yTensor->batch(),
                             "the inputs batch error");
        DEEP8_ARGUMENT_CHECK(
                std::max<size_t>(xTensor->batch(), yTensor->batch()) == output->batch(),
                "the output batch is error");

        for (size_t b = 0; b < output->batch(); ++b) {
            eBatchMat(output, b).noalias() = eBatchMat(xTensor, b) * eBatchMat(yTensor, b);
        }
    }
}

template <typename T>
void MatrixMultiply<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                      const Tensor<T> *output,
                      const Tensor<T> *outputGradient,
                      size_t index,
                      Tensor<T> *iGradient) {
    if (0 == index) {
        /**
         * for a MatrixMultiply C = A * B, index is 0 means calculate the grad for A
         * grad(A) = grad(C) * transpose(B)
         */
        if (1 == inputs[1]->batch()) {
            eRowBatchMat(iGradient).noalias() += eRowBatchMat(outputGradient) * eMat(inputs[1]).transpose();
        } else if (1 == inputs[0]->batch() && 1 == inputs[1]->col()) {
            eMat(iGradient).noalias() += eBatchSizeMat(outputGradient).transpose() * eBatchSizeMat(inputs[1]);
        } else {
            for (size_t b = 0; b < outputGradient->batch(); ++b) {
                eBatchMat(iGradient, b).noalias() += eBatchMat(outputGradient, b) * eBatchMat(inputs[1], b).transpose();
            }
        }
    } else if (1 == index) {
        /**
         * for a MatrixMultiply C = A * B, index is 1 means calculate the grad for B
         * grad(B) = transpose(A) * grad(C)
         */
        if (1 == iGradient->batch()) {
            eMat(iGradient).noalias() += eRowBatchMat(inputs[0]).transpose() * eRowBatchMat(outputGradient);
        } else if (1 == inputs[0]->batch() && 1 == inputs[1]->col()) {
            eBatchSizeMat(iGradient).noalias() += eBatchSizeMat(outputGradient) * eMat(inputs[0]);
        } else {
            for (size_t b = 0; b < outputGradient->batch(); ++b) {
                eBatchMat(iGradient, b).noalias() += eBatchMat(inputs[0], b).transpose() * eBatchMat(outputGradient, b);
            }
        }
    }
}

DEEP8_RE_DECLARATION_HALF_FUNC(MatrixMultiply)
DEEP8_DECLARATION_INSTANCE(MatrixMultiply)

}