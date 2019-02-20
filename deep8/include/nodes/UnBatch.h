#ifndef DEEP8_UNBATCH_H
#define DEEP8_UNBATCH_H

#include "Function.h"

namespace Deep8 {

/**the reverse function of batch, this function does not copy any memory*/
class UnBatch : public Function {
private:
	/**the offset of input*/
	size_t offset;

public:
	explicit UnBatch(std::vector<Node *> &inputs, size_t offset, Shape &outShape);

	void check() override;

protected:
	void forward(const std::vector<const Tensor*> &inputs, Tensor *output) override;
	void backward(const std::vector<const Tensor*> &inputs, 
				  const Tensor *output, 
				  const Tensor *outputGradient, 
				  size_t index, 
				  Tensor *iGradient) override;

public:
	void forward() override;
	void backward() override;
};

}

#endif