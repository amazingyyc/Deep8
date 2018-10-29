#ifndef DEEP8_RESHAPE_H
#define DEEP8_RESHAPE_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class ReShape: public Function<T> {
public:
    Shape reShape;

    explicit ReShape(std::vector<Node *> &inputs, Shape &shape);
    explicit ReShape(std::vector<Node *> &inputs, std::vector<size_t> &shape);

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

public:
	void forward() override;
	void backward() override;
};

}

#endif //DEEP8_RESHAPE_H
