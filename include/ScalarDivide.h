#ifndef DEEP8_SCALARDIVIDE_H
#define DEEP8_SCALARDIVIDE_H

#include "Function.h"

namespace Deep8 {

/*****************************************************************************/
 /**Y = scalar / X*/
 /*****************************************************************************/

template <typename T>
class ScalarDivide : public Function<T> {
public:
	T scalar;

	explicit ScalarDivide(std::vector<Node*> &inputs, T scalar);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;

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