#ifndef DEEP8_DECONV2D_H
#define DEEP8_DECONV2D_H

#include "Function.h"

namespace Deep8 {

/**
 * @brief the DeConv2d is the reverse of Conv2d
 * for Conv2d
 * the input dimension is (batch, inputHeight, inputWidth, inputChannel)
 * the filter dimension is (outputchannel, filterHeight, filterWidth, inputChannel)
 * the stride is (strideH, strideW)
 * dilation is (dilationH, dilationW)
 *
 * realFilterH = filterH + (filterH - 1) * (dilationH - 1);
 * realFilterW = filterW + (filterW - 1) * (dilationW - 1);
 *
 * so the output is
 * the covered is false: (batch, (inputH - realFilterH) / strideH + 1, (inputW - realFilterW) / strideW + 1, outputChannel)
 * the covered is true: (batch, (inputH - realFilterH + strideH - 1) / strideH + 1, (inputW - realFilterW + strideW - 1) / strideW + 1, outputChannel)
 *
 * ref: A guide to convolution arithmetic for deep learning
 * the reverse of Conv2d TransposeConv2d:
 * the inputs dimension is (Batch, inputHeight, inputWidth, inputChannel)
 * the filter dimension is (outputChannel, filterHeight, filterWidth, inputChannel)
 * the stride is (1, 1) (the stride of the TransposeConv2d always 1)
 * inset (stride - 1) 0-unit into the input
 */

template <typename T>
class DeConv2d: public Function<T> {
public:
    /**
     * the forwardStride and forwardCovered is the property of the forward Conv2d
     */
    size_t forwardStrideY;
    size_t forwardStrideX;

    /**
     * if the slide filter will cover the input of the Conv2d
     */
    bool forwardCovered;

    DeConv2d(std::vector<Node *> &inputs, bool covered = false, size_t strideY = 1, size_t strideX = 1);

    void check() override;

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
					const Tensor<T> *output,
					const Tensor<T> *outputGradient,
					size_t index,
					Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA

    void forwardGPUImpl(Device *device, const T *x, const T *filter, T *y,
        int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
		int padTop, int padLeft);

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    void backwardGPUInputImpl(Device *device, T *dx, const T *filter, const T *dy,
        int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
		int padTop, int padLeft);

    void backwardGPUFilterImpl(Device *device, const T *x, T *dw, const T *dy,
        int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int forwardStrideY, int forwardStrideX,
		int padTop, int padLeft);

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                    const Tensor<T> *output,
                    const Tensor<T> *outputGradient,
                    size_t index,
                    Tensor<T> *iGradient) override;

#endif

};


}

#endif //DEEP8_TRANSPOSECONV2D_H
