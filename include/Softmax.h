#ifndef DEEP8_SOFTMAX_H
#define DEEP8_SOFTMAX_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class Softmax: public Function<T> {
public:
    explicit Softmax(std::vector<Node *> &inputs);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

	void forwardGPUImpl(Device *device, const T *x, T *y, Shape &shape);

    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardGPUImpl(Device *device, T *xGrad, const T *y, const T *yGrad, Shape &shape);

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;

#endif

};


}

#endif //DEEP8_SOFTMAX_H
