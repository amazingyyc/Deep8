#ifndef DEEP8_REDUCEMEAN_H
#define DEEP8_REDUCEMEAN_H


#include "Function.h"

namespace Deep8 {

/**
 * axis empty means all reduce
 * the axis is 0 mean the batch dimension
 * -1 means the last dimension
 */
class ReduceMean : public Function {
public:
    std::vector<int> axis;
    bool keepDims;

    explicit ReduceMean(std::vector<Node *> &inputs, std::vector<int> reduceAxis = {-1}, bool keep = true);

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