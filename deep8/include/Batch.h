#ifndef DEEP8_BATCH_H
#define DEEP8_BATCH_H

#include "Function.h"

namespace Deep8 {

/**
 * a simple batch function. simple copy the input data together
 */
template <typename T>
class Batch : public Function<T> {
private:
	/**store if the inputs memory is continuous*/
	bool continuous;

public:
	explicit Batch(std::vector<Node *> &inputs, Shape &outShape);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;
	void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;
#endif
};

}

#endif