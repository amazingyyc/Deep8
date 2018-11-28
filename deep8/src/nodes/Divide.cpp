#include "Divide.h"

namespace Deep8 {

template <typename T>
Divide<T>::Divide(std::vector<Node *> &inputs) : Function<T>(inputs) {
        check();
}

template <typename T>
void Divide<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the inputs size must be 2 in Divide Function");

    /**
     * the Add Function apply to Broadcasting rule: https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
     */
    this->outputShape = broadcastShape(this->inputs[0]->outputShape, this->inputs[1]->outputShape);
}

template <typename T>
void Divide<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) {
    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto xShape = inputs[0]->shape;
    auto yShape = inputs[1]->shape;

    auto zShape = output->shape;

    if (zShape == xShape && zShape == yShape) {
        eTVec(output).device(*device) = eTVec(inputs[0]) / eTVec(inputs[1]);
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
                eTVec(inputs[0]).reshape(xReshape).broadcast(xBroad) / eTVec(inputs[1]).reshape(yReshape).broadcast(yBroad);
    }
}

template <typename T>
template <int diffCount>
void Divide<T>::backwardCPUImpl0(Eigen::ThreadPoolDevice *device,const Tensor<T> *yTensor, const Tensor<T> *outputGradient, Tensor<T> *iGradient) {
    auto yElongateDims = enlongateShapeToMaxDim(yTensor->shape);
    auto iElongateDims = enlongateShapeToMaxDim(iGradient->shape);
    auto outputElongateDims = enlongateShapeToMaxDim(outputGradient->shape);

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (iElongateDims[i] != outputElongateDims[i]) {
            sumDims[j++] = i;
        }
    }

    auto yBroad = yElongateDims;

    for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (yElongateDims[i] != outputElongateDims[i]) {
            yBroad[i] = outputElongateDims[i];
        } else {
            yBroad[i] = 1;
        }
    }

    eTVec(iGradient).reshape(iElongateDims).device(*device) +=
            (eTVec(outputGradient).reshape(outputElongateDims) / eTVec(yTensor).reshape(yElongateDims).broadcast(yBroad)).sum(sumDims).reshape(iElongateDims);
}

template <typename T>
template <int diffCount>
void Divide<T>::backwardCPUImpl1(Eigen::ThreadPoolDevice *device, const Tensor<T> *xTensor, const Tensor<T> *yTensor, const Tensor<T> *outputGradient, Tensor<T> *iGradient) {
    auto xElongateDims = enlongateShapeToMaxDim(xTensor->shape);
    auto yElongateDims = enlongateShapeToMaxDim(yTensor->shape);
    auto outputElongateDims = enlongateShapeToMaxDim(outputGradient->shape);

    auto xBroad = xElongateDims;
    auto yBroad = yElongateDims;

    for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (xElongateDims[i] != outputElongateDims[i]) {
            xBroad[i] = outputElongateDims[i];
        } else {
            xBroad[i] = 1;
        }

        if (yElongateDims[i] != outputElongateDims[i]) {
            yBroad[i] = outputElongateDims[i];
        } else {
            yBroad[i] = 1;
        }
    }

    Eigen::array<int, diffCount> sumDims;

    for (int i = 0, j = 0; i < MAX_TENSOR_DIMS; ++i) {
        if (yElongateDims[i] != outputElongateDims[i]) {
            sumDims[j++] = i;
        }
    }

    eTVec(iGradient).reshape(yElongateDims).device(*device) +=
            ((-eTVec(outputGradient).reshape(outputElongateDims) * eTVec(xTensor).reshape(xElongateDims).broadcast(xBroad))
             / ((eTVec(yTensor).reshape(yElongateDims).broadcast(yBroad)) * (eTVec(yTensor).reshape(yElongateDims).broadcast(yBroad)))).sum(sumDims).reshape(yElongateDims);
}

template <typename T>
void Divide<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                 const Tensor<T> *output,
                 const Tensor<T> *outputGradient,
                 size_t index,
                 Tensor<T> *iGradient) {
    auto device = static_cast<CPUDevice *>(outputGradient->device())->eigenDevice;

/**
 * Z = X / Y
 */
    if (0 == index) {
        auto xShape = iGradient->shape;
        auto yShape = inputs[1]->shape;
        auto zShape = outputGradient->shape;

        auto xEnlongateDims = enlongateShapeToMaxDim(xShape);
        auto zEnlongateDims = enlongateShapeToMaxDim(zShape);

        int diffCount = 0;

        for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (xEnlongateDims[i] != zEnlongateDims[i]) {
                diffCount++;
            }
        }

        if (0 == diffCount) {
            backwardCPUImpl0<0>(device, inputs[1], outputGradient, iGradient);
        } else if (1 == diffCount) {
            backwardCPUImpl0<1>(device, inputs[1], outputGradient, iGradient);
        } else if (2 == diffCount) {
            backwardCPUImpl0<2>(device, inputs[1], outputGradient, iGradient);
        } else if (3 == diffCount) {
            backwardCPUImpl0<3>(device, inputs[1], outputGradient, iGradient);
        } else if (4 == diffCount) {
            backwardCPUImpl0<4>(device, inputs[1], outputGradient, iGradient);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    } else if (1 == index) {
        auto xShape = inputs[0]->shape;
        auto yShape = inputs[1]->shape;
        auto zShape = outputGradient->shape;

        auto yEnlongateDims = enlongateShapeToMaxDim(yShape);
        auto zEnlongateDims = enlongateShapeToMaxDim(zShape);

        int diffCount = 0;

        for (int i = 0; i < MAX_TENSOR_DIMS; ++i) {
            if (yEnlongateDims[i] != zEnlongateDims[i]) {
                diffCount++;
            }
        }

        if (0 == diffCount) {
            backwardCPUImpl1<0>(device, inputs[0], inputs[1], outputGradient, iGradient);
        } else if (1 == diffCount) {
            backwardCPUImpl1<1>(device, inputs[0], inputs[1], outputGradient, iGradient);
        } else if (2 == diffCount) {
            backwardCPUImpl1<2>(device, inputs[0], inputs[1], outputGradient, iGradient);
        } else if (3 == diffCount) {
            backwardCPUImpl1<3>(device, inputs[0], inputs[1], outputGradient, iGradient);
        } else if (4 == diffCount) {
            backwardCPUImpl1<4>(device, inputs[0], inputs[1], outputGradient, iGradient);
        } else {
            DEEP8_RUNTIME_ERROR("the shape is error");
        }
    }
}

DEEP8_RE_DECLARATION_HALF_FUNC(Divide);
DEEP8_DECLARATION_INSTANCE(Divide)

}