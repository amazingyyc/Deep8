#ifndef DEEP8_CONV2D_H
#define DEEP8_CONV2D_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class Conv2d: public Function<T> {
public:
    size_t strideY;
    size_t strideX;

    size_t dilationY;
    size_t dilationX;

    /**
     * if true the slide filter will cover all the input
     */
    bool covered;

    Conv2d(std::vector<Node *> &inputs, bool covered = false, size_t strideH = 1, size_t strideW = 1, size_t dilationH = 1, size_t dilationW = 1);

    void check() override;

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

	void forwardGPUImpl(Device* device, const T *x, const T *filter, T *y,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	/**for input*/
	void backwardGPUInputImpl(Device* device, T *dx, const T *w, const T *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

	/**for filter*/
	void backwardGPUFilterImpl(Device* device, const T *x, T *dw, const T *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX);

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override;
#endif
};


}

#endif //DEEP8_CONV2D_H
