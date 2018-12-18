#ifndef DEEP8_SOFTMAX_H
#define DEEP8_SOFTMAX_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class Softmax: public Function<T> {
public:
	sizze_t axis;

    explicit Softmax(std::vector<Node *> &inputs, size_t axis = 0);

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

#endif //DEEP8_SOFTMAX_H
