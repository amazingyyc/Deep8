#ifndef DEEP8_POW_H
#define DEEP8_POW_H

#include "Function.h"

namespace Deep8 {

template <typename T> 
class Pow : public Function<T> {
public:
	T scalar;

	explicit Pow(std::vector<Node*> &inputs, T scalar);

	void check() override;

protected:

	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;
#endif
};


}

#endif
