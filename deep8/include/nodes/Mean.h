#ifndef DEEP8_MEAN_H
#define DEEP8_MEAN_H

#include "Function.h"

namespace Deep8 {

class Mean : public Function {
public:
    explicit Mean(std::vector<Node*>& inputs);

    void check() override;

protected:
    void forward(const std::vector<const Tensor*>& inputs, Tensor* output) override;
    void backward(const std::vector<const Tensor*>& inputs,
                  const Tensor* output,
                  const Tensor* outputGradient,
                  size_t index,
                  Tensor* iGradient) override;
};


}

#endif