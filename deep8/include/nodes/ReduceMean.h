#ifndef DEEP8_REDUCEMEAN_H
#define DEEP8_REDUCEMEAN_H


#include "Function.h"

namespace Deep8 {

/**
 *  axis < 0 means all reduce
 */
class ReduceMean : public Function {
public:
    int axis;
    bool keepDims;

    explicit ReduceMean(std::vector<Node *> &inputs, int a = -1, bool keep = false);

    void check() override;

protected:
	void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) override;
};

}


#endif