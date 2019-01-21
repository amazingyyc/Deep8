#ifndef DEEP8_CROSSENTROPY_H
#define DEEP8_CROSSENTROPY_H

#include "Function.h"

namespace Deep8 {

/**
 * in training
 * inputs[0] means the Graph output
 * inputs[1] means the target output
 */
template <typename T>
class CrossEntropy: public Function<T> {
public:
    CrossEntropy(std::vector<Node *> &inputs);

    void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;
#endif
};

}

#endif