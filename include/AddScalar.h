#ifndef DEEP8_ADDSCALAR_H
#define DEEP8_ADDSCALAR_H

#include "Function.h"

namespace Deep8 {

/**
 * Y = X(tensor) + scalar
 */

template <typename T>
class AddScalar : public Function<T> {
public:
	T scalar;

	explicit AddScalar(std::vector<Node *> &inputs, T scalar);

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