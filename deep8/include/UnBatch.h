#ifndef DEEP8_UNBATCH_H
#define DEEP8_UNBATCH_H

#include "Function.h"

namespace Deep8 {

/**the reverse function of batch, this function does not copy any memory*/
template <typename T>
class UnBatch : public Function<T> {
private:
	/**the offset of input*/
	size_t offset;

public:
	explicit UnBatch(std::vector<Node *> &inputs, size_t offset, Shape &outShape);

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