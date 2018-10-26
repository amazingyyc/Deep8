#ifndef DEEP8_MULTIPLY_H
#define DEEP8_MULTIPLY_H

#include "Function.h"

namespace Deep8 {

/**
 * @brief this is a element-wise multiply it will BroadCast the input
 */

template <typename T>
class Multiply: public Function<T> {
public:
    explicit Multiply(std::vector<Node *> &inputs);

	void check() override;

protected:
	void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    template <int diffCount>
	void backwardCPUImpl(Eigen::ThreadPoolDevice *device,
						const Tensor<T> *otherValue,
						const Tensor<T> *outputGradient,
						Tensor<T> *iGradient);

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

	void backwardGPUImpl(T *inGrad,     const int *inShape,    const int *inDims,
				   const T *otherValue, const int *otherShape, const int *otherDims,
				   const T *outGrad,    const int *outShape,   const int *outDims,
		           const int N);

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override;
#endif

};


}



#endif //DEEP8_WISEMULTIPLY_H
