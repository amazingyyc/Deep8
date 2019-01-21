#ifndef DEEP8_AVGPOOLING2D_H
#define DEEP8_AVGPOOLING2D_H

#include "Function.h"

namespace Deep8 {

template <typename T>
class AvgPooling2d: public Function<T> {
public:
    size_t filterHeight;
    size_t filterWidth;

    size_t strideY;
    size_t strideX;

    /**
     * if the slide filter will cover all the input
     */
    bool covered;

    explicit AvgPooling2d(std::vector<Node *> &inputs, bool covered = false, size_t filterH = 1, size_t filterW = 1, size_t strideH = 1, size_t strideW = 1);

    void check() override;

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

    void backwardCPUImpl(T *inputGrad,
                         T *outputGrad,
                         int64_t batch,
                         int64_t startChannel,
                         int64_t endChannel,
                         int64_t inputHeight,
                         int64_t inputWidth,
                         int64_t outputHeight,
                         int64_t outputWidth,
                         int64_t channel,
                         int64_t filterH,
                         int64_t filterW,
                         int64_t strideH,
                         int64_t strideW,
                         int64_t padTop,
                         int64_t padLeft);

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override;

#ifdef HAVE_CUDA
    void forwardGPUImpl(const T *x, T *y,
						const int batch,
						const int inputHeight,
						const int inputWidth,
						const int outputHeight,
						const int outputWidth,
						const int channel,
						const int filterHeight,
						const int filterWidth,
						const int padTop,
						const int padLeft,
						const int strideY,
						const int strideX);

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override;

	void backwardGPUImpl(T *dx, const T *dy,
						const int batch,
						const int inputHeight,
						const int inputWidth,
						const int outputHeight,
						const int outputWidth,
						const int channel,
						const int filterHeight,
						const int filterWidth,
						const int padTop,
						const int padLeft,
						const int strideY,
						const int strideX);

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override;

#endif
};


}

#endif //DEEP8_AVGPOOL2D_H
