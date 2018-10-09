#include "Add.h"

namespace Deep8 {

template <typename T>
void Add<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs size must be 2 in Add Function");

    /**
     * the Add Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
     */
    auto xShape = static_cast<Variable<T> *>(this->inputs[0])->value.shape;
    auto yShape = static_cast<Variable<T> *>(this->inputs[1])->value.shape;

    this->outputShape = broadcastShape(xShape, yShape);
}

template <typename T>
void Add<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto xShape = inputs[0]->shape;
    auto yShape = inputs[1]->shape;

    auto zShape = output->shape;

    if (zShape == xShape && zShape == yShape) {
        eTVec(output).device(*device) = eTVec(inputs[0]) + eTVec(inputs[1]);
    } else {
        auto xReshape = enlongateShapeToMaxDim(xShape);
        auto yReshape = enlongateShapeToMaxDim(yShape);
        auto zReshape = enlongateShapeToMaxDim(zShape);

        auto xBroad = xReshape;
        auto yBroad = yReshape;

        for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (xBroad[i] < zReshape[i]) {
                xBroad[i] = zReshape[i];
            } else {
                xBroad[i] = 1;
            }

            if (yBroad[i] < zReshape[i]) {
                yBroad[i] = zReshape[i];
            } else {
                yBroad[i] = 1;
            }
        }

        eTVec(output).reshape(zReshape).device(*device) =
                eTVec(inputs[0]).reshape(xReshape).broadcast(xBroad) + eTVec(inputs[1]).reshape(yReshape).broadcast(yBroad);
    }
}


template <typename T>
template <int diffCount>
void Add<T>::backwardCPUImpl(Eigen::ThreadPoolDevice *device, const Tensor<T> *outputGradient, Tensor<T> *iGradient) {
    auto outputGradShape = enlongateShapeToMaxDim(outputGradient->shape);
    auto iGradShape      = enlongateShapeToMaxDim(iGradient->shape);

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (outputGradShape[i] != iGradShape[i]) {
            sumDims[j++] = i;
        }
    }

    eTVec(iGradient).reshape(iGradShape).device(*device) += eTVec(outputGradient).reshape(outputGradShape).sum(sumDims).reshape(iGradShape);
}

template <typename T>
void Add<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    auto device = static_cast<CPUDevice *>(outputGradient->device())->eigenDevice;

    auto gradShape = iGradient->shape;
    auto outputShape = outputGradient->shape;

    if (gradShape == outputShape) {
        eTVec(iGradient).device(*device) += eTVec(outputGradient);
        return;
    }

    auto outputGradEnlongateShape = enlongateShapeToMaxDim(outputGradient->shape);
    auto iGradEnlongateShape = enlongateShapeToMaxDim(iGradient->shape);

    int diffCount = 0;

    for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (outputGradEnlongateShape[i] != iGradEnlongateShape[i]) {
            diffCount++;
        }
    }

    if (1 == diffCount) {
        backwardCPUImpl<1>(device, outputGradient, iGradient);
    } else if (2 == diffCount) {
        backwardCPUImpl<2>(device, outputGradient, iGradient);
    } else if (3 == diffCount) {
        backwardCPUImpl<3>(device, outputGradient, iGradient);
    } else if (4 == diffCount) {
        backwardCPUImpl<4>(device, outputGradient, iGradient);
    } else {
        DEEP8_RUNTIME_ERROR("the shape is error");
    }
}

DEEP8_DECLARATION_INSTANCE(Add)

}
