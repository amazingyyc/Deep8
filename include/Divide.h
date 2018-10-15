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
	void forwardGPUImpl(const T *x, const int *xshape, const int *xdims,
						const T *y, const int *yshape, const int *ydims,
							  T *z, const int *zshape, const int *zdims, const int N);

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardGPUImplX(T *xGrad,  const int *xshape, const int *xdims,
					const T *y,      const int *yshape, const int *ydims,
					const T *zGrad,  const int *zshape, const int *zdims, const int N);

	void backwardGPUImplY(const T *x, const int *xshape, const int *xdims,
						  const T *y, T *yGrad, const int *yshape, const int *ydims,
						  const T *zGrad, const int *zshape, const int *zdims, const int N);

	void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
					 const Tensor<T> *output,
					 const Tensor<T> *outputGradient,
					 size_t index,
					 Tensor<T> *iGradient) override;
#endif

};




}

#endif
