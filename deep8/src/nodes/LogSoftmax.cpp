#include "LogSoftmax.h"

namespace Deep8 {


template <typename T>
LogSoftmax<T>::LogSoftmax(std::vector<Node *> &inputs, int a): Function<T>(inputs), axis(a) {
    check();
}

template <typename T>
void LogSoftmax<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the LogSoftmax Function needs only 1 input");

    auto inputShape = this->inputs[0]->outputShape;

    DEEP8_ARGUMENT_CHECK(axis < (int) inputShape.nDims, "the axis is error");

    this->outputShape = inputShape;
}

template <typename T>
void LogSoftmax<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto cpuDevice = static_cast<CPUDevice *>(output->device());
    auto eigenDevice = cpuDevice->eigenDevice;

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

    auto maxptr = (T*)cpuDevice->malloc(sizeof(T) * dim0 * dim2);
    auto sumptr = (T*)cpuDevice->malloc(sizeof(T) * dim0 * dim2);

    Eigen::array<int, 1> reduceDims = { 1 };
    Eigen::array<int, 3> reshape    = { dim0, 1, dim2 };
    Eigen::array<int, 3> broad      = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> x(inputs[0]->data(), dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> y(output->data(), dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> maxt(maxptr, dim0, 1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> sumt(sumptr, dim0, 1, dim2);

    maxt.device(*eigenDevice) = x.maximum(reduceDims).reshape(reshape);
    sumt.device(*eigenDevice) = (x - maxt.broadcast(broad)).exp().sum(reduceDims).reshape(reshape);
    y.device(*eigenDevice)    = x - maxt.broadcast(broad) - sumt.log().broadcast(broad);

    cpuDevice->free(maxptr);
    cpuDevice->free(sumptr);
}

template <typename T>
void LogSoftmax<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of LogSoftmax backwardCPU is error");

    auto cpuDevice = static_cast<CPUDevice*>(iGradient->device());
    auto eigenDevice = cpuDevice->eigenDevice;

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

    auto sumptr = (T*)cpuDevice->malloc(sizeof(T) * dim0 * dim2);

    Eigen::array<int, 1> sumDims = { 1 };
    Eigen::array<int, 3> reshape = { dim0, 1, dim2 };
    Eigen::array<int, 3> broad   = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> sumt(sumptr, dim0, 1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dx(iGradient->data(),      dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> y(output->data(),          dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dy(outputGradient->data(), dim0, dim1, dim2);

    sumt.device(*eigenDevice) = dy.sum(sumDims).reshape(reshape);
    dx.device(*eigenDevice)  += dy - y.exp() * sumt.broadcast(broad);

    cpuDevice->free(sumptr);
}

DEEP8_RE_DECLARATION_HALF_FUNC(LogSoftmax);
DEEP8_DECLARATION_INSTANCE(LogSoftmax)

}