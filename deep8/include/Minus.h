#ifndef DEEP8_MINUS_H
#define DEEP8_MINUS_H

#include "Function.h"

namespace Deep8 {

/**
 * Z = X - Y
 */
template <typename T>
class Minus: public Function<T> {
public:
    explicit Minus(std::vector<Node *> &inputs);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    template <int diffCount>
	void backwardCPUImpl(Eigen::ThreadPoolDevice *device, const Tensor<T> *outputGradient, Tensor<T> *iGradient, size_t index);

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
	void forwardGPUImpl(const T *x, const int *xshape, const int *xdims,
						const T *y, const int *yshape, const int *ydims,
		                      T *z, const int *zshape, const int *zdims, const int N);

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardGPUImpl0(T *inGrad, const int *inShape, const int *inDims, const T *outGrad, const int *outShape, const int *outDims, const int N);
	void backwardGPUImpl1(T *inGrad, const int *inShape, const int *inDims, const T *outGrad, const int *outShape, const int *outDims, const int N);

	void backwardGPU(const std::vector<const Tensor<T>*> &inputs, const Tensor<T> *output, const Tensor<T> *outputGradient, size_t index, Tensor<T> *iGradient) override;
#endif
};

}

#endif //DEEP8_MINUS_H
