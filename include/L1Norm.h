#ifndef DEEP8_L1NORM_H
#define DEEP8_L1NORM_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class L1Norm: public Function<T> {
public:
    explicit L1Norm(std::vector<Node *> &inputs): Function<T>(inputs) {
        check();
    }

    void check() override;

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
    void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;
};


}

#endif //DEEP8_L1NORM_H
