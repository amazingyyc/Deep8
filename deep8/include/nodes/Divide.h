#ifndef DEEP8_DIVIDE_H
#define DEEP8_DIVIDE_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class Divide: public Function<T> {
public:
	explicit Divide(std::vector<Node *> &inputs);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	template <int diffCount>
	void backwardCPUImpl0(Eigen::ThreadPoolDevice *device,const Tensor<T> *yTensor, const Tensor<T> *outputGradient, Tensor<T> *iGradient);

	template <int diffCount>
	void backwardCPUImpl1(Eigen::ThreadPoolDevice *device, const Tensor<T> *xTensor, const Tensor<T> *yTensor, const Tensor<T> *outputGradient, Tensor<T> *iGradient);

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

#endif
