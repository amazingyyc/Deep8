#ifndef DEEP8_SUMELEMENTS_H
#define DEEP8_SUMELEMENTS_H

#include "Function.h"

namespace Deep8 {


/**
 * y = sum(x)
 * the y is scalar
 */

template <typename T>
class SumElements: public Function<T> {
public:
    explicit SumElements(std::vector<Node *> &inputs);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

	void forwardGPUImpl(Device *device, const T *x, T *y, const int batch, const int size);

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;
#endif

};


}

#endif //DEEP8_SUMELEMENTS_H
