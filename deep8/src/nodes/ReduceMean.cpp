#include "ReduceMean.h"

namespace Deep8 {

template <typename T>
ReduceMean<T>::ReduceMean(std::vector<Node*> &inputs, int a, bool keep) : Function<T>(inputs), axis(a), keepDims(keep) {
    check();
}

template <typename T>
void ReduceMean<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "only have 1 input");
    DEEP8_ARGUMENT_CHECK(axis < this->inputs[0]->outputShape.nDims, "the axis is error");

    auto shape = this->inputs[0]->outputShape;
    std::vector<size_t> list;

    if (axis < 0) {
        if (keepDims) {
            for (int i = 0; i < (int)shape.nDims; ++i) {
                list.emplace_back(1);
            }
        }
        else {
            list.emplace_back(1);
        }
    }
    else {
        for (int i = 0; i < (int)shape.nDims; ++i) {
            if (i == axis) {
                if (keepDims) {
                    list.emplace_back(1);
                }
            }
            else {
                list.emplace_back(shape[i]);
            }
        }
    }

    this->outputShape = Shape(shape.batch, list);
}

template <typename T>
void ReduceMean<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

    auto shape = inputs[0]->shape;
    int dim0, dim1, dim2;

    if (axis < 0) {
        dim0 = (int)shape.batch;
        dim1 = (int)shape.batchSize();
        dim2 = 1;
    }
    else {
        dim0 = (int)shape.batch;
        dim1 = (int)shape.dim(axis);
        dim2 = 1;

        for (int i = 0; i < axis; ++i) {
            dim0 *= (int)shape.dim(i);
        }

        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int)shape.dim(i);
        }
    }

    Eigen::array<int, 1> reduceDims = { 1 };
    Eigen::array<int, 2> reshape = { dim0, dim2 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> x(inputs[0]->data(), dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor>> y(output->data(), dim0, dim2);

    y.device(*device) = x.mean(reduceDims).reshape(reshape);
}

template <typename T>
void ReduceMean<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    DEEP8_ARGUMENT_CHECK(0 == index, "the index of ReduceMean backwardCPU is error");

    auto device = static_cast<CPUDevice*>(output->device())->eigenDevice;

    if (axis < 0) {
        dim0 = (int)shape.batch;
        dim1 = (int)shape.batchSize();
        dim2 = 1;
    }
    else {
        dim0 = (int)shape.batch;
        dim1 = (int)shape.dim(axis);
        dim2 = 1;
    
        for (int i = 0; i < axis; ++i) {
            dim0 *= (int)shape.dim(i);
        }
    
        for (int i = axis + 1; i < shape.nDims; ++i) {
            dim2 *= (int)shape.dim(i);
        }
    }

    auto ratio = T(1) / T(dim1);

    Eigen::array<int, 3> broad = { 1, dim1, 1 };

    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dx(iGradient->data(), dim0, dim1, dim2);
    Eigen::TensorMap<Eigen::Tensor<T, 3, Eigen::RowMajor>> dy(outputGradient->data(), dim0, 1, dim2);

    dx.device(*device) += dy.broadcast(broad) * ratio;
}

DEEP8_RE_DECLARATION_HALF_FUNC(ReduceMean);
DEEP8_DECLARATION_INSTANCE(ReduceMean)

}