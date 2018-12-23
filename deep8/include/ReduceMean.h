#ifndef DEEP8_REDUCEMEAN_H
#define DEEP8_REDUCEMEAN_H


#include "Function.h"

namespace Deep8 {

/**
 *  axis < 0 means all reduce
 */
template <typename T>
class ReduceMean : public Function<T> {
public:
    int axis;
    bool keepDims;

    explicit ReduceMean(std::vector<Node *> &inputs, int a = -1, bool keep = false);

    void check() override;

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
    void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
    void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
    void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;
#endif
}

}