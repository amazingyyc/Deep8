#ifndef DEEP8_ABS_H
#define DEEP8_ABS_H

#include "Function.h"

namespace Deep8 {

/**
 * y = |x|
 */

template <typename T>
class Abs: public Function<T> {
public:
	explicit Abs(std::vector<Node *> &inputs);

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

#endif //DEEP8_ABS_H