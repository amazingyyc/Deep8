#ifndef DEEP8_BATCH_H
#define DEEP8_BATCH_H

#include "Function.h"

namespace Deep8 {

/**
 * a simple batch function. simple copy the input data together
 */
class Batch : public Function {
private:
	/**store if the inputs memory is continuous*/
	bool continuous;

public:
	explicit Batch(std::vector<Node *> &inputs, Shape &outShape);

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