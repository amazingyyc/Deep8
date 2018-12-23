#include "L2Norm.h"

namespace Deep8 {

template <typename T>
L2Norm<T>::L2Norm(std::vector<Node *> &inputs): Function<T>(inputs) {
    check();
}

template<typename T>
void L2Norm<T>::check() {
    Function<T>::check();

    DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the L2Norm Function needs only 1 input");

	this->outputShape = Shape(1, { 1 });
}

template<typename T>
void L2Norm<T>::forwardCPU(const std::vector<const Tensor <T> *> &inputs, Tensor <T> *output) {
	auto eigenDevice = static_cast<CPUDevice *>(output->device())->eigenDevice;

	auto x = inputs[0];
	auto y = output;

	Eigen::array<int, 1> reshapeDims = { 1 };
	Eigen::array<int, 1> sumDims = { 0 };

	eTVec(y).device(*eigenDevice) = eTVec(x).square().sum(sumDims).sqrt().reshape(reshapeDims);
}

template<typename T>
void L2Norm<T>::backwardCPU(const std::vector<const Tensor <T> *> &inputs,
                            const Tensor <T> *output,
                            const Tensor <T> *outputGradient,
                            size_t index,
                            Tensor <T> *iGradient) {
	DEEP8_ARGUMENT_CHECK(0 == index, "the index of L2Norm backwardCPU is error");

	auto eigenDevice = static_cast<CPUDevice *>(iGradient->device())->eigenDevice;

	int size = (int)iGradient->size();

	Eigen::array<int, 1> outputBroad = { size };

	eTVec(iGradient).device(*eigenDevice) += (eTVec(outputGradient) / eTVec(output)).broadcast(outputBroad) * eTVec(inputs[0]);
}

DEEP8_RE_DECLARATION_HALF_FUNC(L2Norm);
DEEP8_DECLARATION_INSTANCE(L2Norm)

}