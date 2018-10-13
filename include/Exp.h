#ifndef DEEP8_EXP_H
#define DEEP8_EXP_H

#include "Function.h"

namespace Deep8 {

/**
 * y = exp(x)
 */
template <typename T>
class Exp: public Function<T> {
public:
    explicit Exp(std::vector<Node *> &inputs);

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

#endif //DEEP8_EXP_H
