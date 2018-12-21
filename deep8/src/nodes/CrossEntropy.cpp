#include "CrossEntropy.h"

namespace Deep8 {

template <typename T>
CrossEntropy<T>::CrossEntropy(std::vector<Node *> &inputs) : Function<T>(inputs) {
	check();
}

template <typename T>
void CrossEntropy<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "the CrossEntropy Function needs only 2 input");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.size() == this->inputs[1]->outputShape.size(), "inputs's shape size must be equal");
    DEEP8_ARGUMENT_CHECK(this->inputs[0]->outputShape.batch == this->inputs[1]->outputShape.batch, "inputs's batch must be equal");

    this->outputShape = Shape(1, { 1 });
}

template <typename T>
void CrossEntropy<T>::forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output)  {
    auto device = static_cast<CPUDevice *>(output->device())->eigenDevice;

    auto batch = inputs[0]->shape.batch;
    auto scale = -1 / T(batch);

    Eigen::array<int, 1> sumDims = { 0 };
    Eigen::array<int, 1> reshape = { 1 };

    eTVec(output).device(*device) = (eTVec(inputs[1]) * eTVec(inputs[0]).log()).sum(sumDims).reshape(reshape) * scale;
}

template <typename T>
void CrossEntropy<T>::backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) {
    auto device = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

    if (0 == index) {
        auto batch = iGradient->shape.batch;
        auto scale = -outputGradient->scalar() / T(batch);

        eTVec(iGradient).device(*device) += (eTVec(inputs[1]) / eTVec(inputs[0])) * scale;
    } else if (1 == index) {
        auto batch = iGradient->shape.batch;
        auto scale = -outputGradient->scalar() / T(batch);

        eTVec(iGradient).device(*device) += eTVec(inputs[0]).log() * scale;
    } else {
        DEEP8_RUNTIME_ERROR("the index is error");
    }
}

DEEP8_RE_DECLARATION_HALF_FUNC(CrossEntropy);
DEEP8_DECLARATION_INSTANCE(CrossEntropy);

}