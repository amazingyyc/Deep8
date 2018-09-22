#ifndef DEEP8_MAXPOOLING2D_H
#define DEEP8_MAXPOOLING2D_H

namespace Deep8 {

template <typename T>
class MaxPooling2d: public Function<T> {
public:
    size_t filterHeight;
    size_t filterWidth;

    size_t strideY;
    size_t strideX;

    /**
     * if the slide filter will cover all the input
     */
    bool covered;

    explicit MaxPooling2d(std::vector<Node *> &inputs, bool covered = false, size_t filterH = 1, size_t filterW = 1, size_t strideH = 1, size_t strideW = 1)
            :Function<T>(inputs), covered(covered), filterHeight(filterH), filterWidth(filterW), strideY(strideH), strideX(strideW) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the MaxPooling2d only need 1 input");
        DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX  >= 1, "the filter size or stride is error");
        DEEP8_ARGUMENT_CHECK(4 == this->inputs[0]->outputShape.nDims(), "MaxPooling2d needs inputs nDims is 4");

        auto inputShape = this->inputs[0]->outputShape;

        if (!covered) {
            DEEP8_ARGUMENT_CHECK(filterHeight <= inputShape.dim(1) && filterWidth <= inputShape.dim(2),
                                 "the not forwardCovered mode type needs filter smaller than input");
        }

        auto inputH = static_cast<int64_t>(inputShape.dim(1));
        auto inputW = static_cast<int64_t>(inputShape.dim(2));

        std::vector<size_t> outputDim(4);
        outputDim[0] = inputShape.dim(0);
        outputDim[3] = inputShape.dim(3);

        if (!covered) {
            int64_t outputH = (inputH - static_cast<int64_t>(filterHeight)) / static_cast<int64_t>(strideY) + 1;
            int64_t outputW = (inputW - static_cast<int64_t>(filterWidth))  / static_cast<int64_t>(strideX) + 1;

            DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

            outputDim[1] = static_cast<size_t>(outputH);
            outputDim[2] = static_cast<size_t>(outputW);
        } else {
            int64_t outputH = (inputH - static_cast<int64_t>(filterHeight) + static_cast<int64_t>(strideY) - 1) / static_cast<int64_t>(strideY) + 1;
            int64_t outputW = (inputW - static_cast<int64_t>(filterWidth)  + static_cast<int64_t>(strideX) - 1) / static_cast<int64_t>(strideX) + 1;

            DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

            outputDim[1] = static_cast<size_t>(outputH);
            outputDim[2] = static_cast<size_t>(outputW);
        }

        this->outputShape = Shape(outputDim);
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        auto input = inputs[0];

        auto batch  = static_cast<TensorIndex>(input->shape.dim(0));
        auto inputH = static_cast<TensorIndex>(input->shape.dim(1));
        auto inputW = static_cast<TensorIndex>(input->shape.dim(2));
        auto inputC = static_cast<TensorIndex>(input->shape.dim(3));

        auto outputH = static_cast<TensorIndex>(output->shape.dim(1));
        auto outputW = static_cast<TensorIndex>(output->shape.dim(2));
        auto outputC = static_cast<TensorIndex>(output->shape.dim(3));

        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                inputTensor(input->data(), batch, inputH, inputW, inputC);

        Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
                outputTensor(output->data(), batch, outputH, outputW, outputC);

        Eigen::DSizes<TensorIndex, 4> preDims;
        preDims[0] = batch * outputH * outputW;
        preDims[1] = filterHeight ;
        preDims[2] = filterWidth;
        preDims[3] = inputC;

        Eigen::DSizes<TensorIndex, 2> reductionDims;
        reductionDims[0] = 1;
        reductionDims[1] = 2;

		auto padY = std::max<TensorIndex>(0, (outputH - 1) * static_cast<TensorIndex>(strideY) + static_cast<TensorIndex>(filterHeight) - inputH);
		auto padX = std::max<TensorIndex>(0, (outputW - 1) * static_cast<TensorIndex>(strideX) + static_cast<TensorIndex>(filterWidth) - inputW);

		auto padTop    = padY / 2;
		auto padBottom = padY - padTop;
		auto padLeft   = padX / 2;
		auto padRight  = padX - padLeft;

        outputTensor.device(*device) = inputTensor.extract_image_patches(filterWidth, filterHeight, strideX, strideY, 1, 1, 1, 1, padTop, padBottom, padLeft, padRight, 0)
                .reshape(preDims)
                .maximum(reductionDims)
                .reshape(outputTensor.dimensions());
    }

    void backwardCPUImpl(T *input,
                         T *inputGrad,
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
                         int64_t padLeft) {
        for (int64_t b = 0; b < batch; ++b) {
            auto inputPtr      = input + b * inputHeight * inputWidth * channel;
            auto inputGradPtr  = inputGrad + b * inputHeight * inputWidth * channel;
            auto outputGradPtr = outputGrad + b * outputHeight * outputWidth * channel;

            for (int64_t k = startChannel; k < endChannel; ++k) {
                for (int64_t y = 0; y < outputHeight; ++y) {
                    for (int64_t x = 0; x < outputWidth; ++x) {
                        auto startH = std::max<int64_t>(0, padTop + y * strideH);
                        auto endH   = std::min<int64_t>(inputHeight, padTop + y * strideH + filterH);

                        auto startW = std::max<int64_t>(0, padLeft + x * strideW);
                        auto endW   = std::min<int64_t>(inputWidth, padLeft + x * strideW + filterW);

                        if (startH >= endH || startW >= endW) {
                            continue;
                        }

                        int64_t maxH = startH;
                        int64_t maxW = startW;
                        T maxValue = inputPtr[startH * inputWidth * channel + startW * channel + k];

                        for (int64_t inH = startH; inH < endH; ++inH) {
                            for (int64_t inW = startW; inW < endW; ++inW) {
                                auto curValue = inputPtr[inH * inputWidth * channel + inW * channel + k];
                                if (curValue > maxValue) {
                                    maxValue = curValue;

                                    maxH = inH;
                                    maxW = inW;
                                }
                            }
                        }

                        inputGradPtr[maxH * inputWidth * channel + maxW * channel + k] += outputGradPtr[y * outputWidth * channel + x * channel + k];
                    }
                }
            }
        }
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");
        DEEP8_ARGUMENT_CHECK(outputGradient->shape.dim(3) == iGradient->shape.dim(3), "the input channel and output channel must be same");

        auto device = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;

        auto input = inputs[0];

        auto batch  = static_cast<int64_t>(input->shape.dim(0));
        auto inputH = static_cast<int64_t>(input->shape.dim(1));
        auto inputW = static_cast<int64_t>(input->shape.dim(2));
        auto inputC = static_cast<int64_t>(input->shape.dim(3));

        auto outputH = static_cast<int64_t>(outputGradient->shape.dim(1));
        auto outputW = static_cast<int64_t>(outputGradient->shape.dim(2));
        auto outputC = static_cast<int64_t>(outputGradient->shape.dim(3));

        int64_t padY = std::max<int64_t>(0, (outputH - 1) * static_cast<int64_t>(strideY) + static_cast<int64_t>(filterHeight) - inputH);
        int64_t padX = std::max<int64_t>(0, (outputW - 1) * static_cast<int64_t>(strideX) + static_cast<int64_t>(filterWidth) - inputW);

        int64_t padTop  = -(padY / 2);
        int64_t padLeft = -(padX / 2);

         /**
          * use the Eigen ThreadPool
          */
        int64_t threadNum = device->numThreads();
        int64_t blockSize = (outputC + threadNum - 1) / threadNum;

        Eigen::Barrier barrier(static_cast<unsigned int>(threadNum));

        auto blockFunc = [this, &barrier] (T *input,
                                           T *inputGrad,
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
                                           int64_t padLeft) {
            this->backwardCPUImpl(input,
                                 inputGrad,
                                 outputGrad,
                                 batch,
                                 startChannel,
                                 endChannel,
                                 inputHeight,
                                 inputWidth,
                                 outputHeight,
                                 outputWidth,
                                 channel,
                                 filterH,
                                 filterW,
                                 strideH,
                                 strideW,
                                 padTop,
                                 padLeft);

            barrier.Notify();
        };

        for (int64_t i = 0; i < threadNum; ++i) {
            int64_t startChannel = i * blockSize;
            int64_t endChannel   = std::min<int64_t>(startChannel + blockSize, outputC);

            device->enqueueNoNotification(blockFunc, input->data(), iGradient->data(),
                                          outputGradient->data(),
				batch, startChannel, endChannel, inputH, inputW, outputH, outputW, outputC, filterHeight, filterWidth, strideY, strideX, padTop, padLeft);
        }

        barrier.Wait();
    }

#ifdef HAVE_CUDNN

	void forwardGPUCUDNNImpl(GPUDevice *device, const float *x, const Shape &xShape, float *y, const Shape &yShape, 
							int windowsHeight,   int windowsWidth, 
							int verticalPadding, int horizontalPadding,
							int verticalStride,  int horizontalStride) {
		float alpha = 1;
		float beta  = 0;

		cudnnPoolingDescriptor_t poolingDesc;
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
		CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN, 
			windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
	}

	void forwardGPUCUDNNImpl(GPUDevice *device, const double *x, const Shape &xShape, double *y, const Shape &yShape,
							int windowsHeight,   int windowsWidth,
							int verticalPadding, int horizontalPadding,
							int verticalStride,  int horizontalStride) {

		double alpha = 1;
		double beta  = 0;

		cudnnPoolingDescriptor_t poolingDesc;
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
		CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
	}

#ifdef HAVE_HALF

	void forwardGPUCUDNNImpl(GPUDevice *device, const half *x, const Shape &xShape, half *y, const Shape &yShape,
		int windowsHeight, int windowsWidth,
		int verticalPadding, int horizontalPadding,
		int verticalStride, int horizontalStride) {
		half alpha = 1;
		half beta = 0;

		cudnnPoolingDescriptor_t poolingDesc;
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
		CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		CUDNN_CHECK(cudnnPoolingForward(device->cudnnHandle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
	}
#endif // HAVE_HALF
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA
#ifdef HAVE_CUDNN

		auto inputH = static_cast<int>(inputs[0]->shape.dim(1));
		auto inputW = static_cast<int>(inputs[0]->shape.dim(2));

		auto outputH = static_cast<int>(output->shape.dim(1));
		auto outputW = static_cast<int>(output->shape.dim(2));

		int padY = std::max<int>(0, (outputH - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputH);
		int padX = std::max<int>(0, (outputW - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth) - inputW);

		int padTop  = (padY / 2);
		int padLeft = (padX / 2);

		forwardGPUCUDNNImpl(static_cast<GPUDevice*>(output->device), inputs[0]->data(), inputs[0]->shape, output->data(), output->shape,
			(int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);
#else
		DEEP8_RUNTIME_ERROR("the MaxPooling2d needs CUDNN");
#endif

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}


#ifdef HAVE_CUDNN

	void backwardGPUCUDNNImpl(GPUDevice *device, const float *x, float *dx, const Shape &xShape, const float *y, const float *dy, const Shape &yShape,
								int windowsHeight, int windowsWidth,
								int verticalPadding, int horizontalPadding,
								int verticalStride, int horizontalStride) {

		float alpha = 1;
		float beta = 1;

		cudnnPoolingDescriptor_t poolingDesc;
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
		CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t dxDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		cudnnTensorDescriptor_t dyDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
	}

	void backwardGPUCUDNNImpl(GPUDevice *device, const double *x, double *dx, const Shape &xShape, const double *y, const double *dy, const Shape &yShape,
		int windowsHeight, int windowsWidth,
		int verticalPadding, int horizontalPadding,
		int verticalStride, int horizontalStride) {

		double alpha = 1;
		double beta = 1;

		cudnnPoolingDescriptor_t poolingDesc;
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
		CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t dxDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		cudnnTensorDescriptor_t dyDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
	}

#ifdef HAVE_HALF
	void backwardGPUCUDNNImpl(GPUDevice *device, const half *x, half *dx, const Shape &xShape, const half *y, const half *dy, const Shape &yShape,
		int windowsHeight, int windowsWidth,
		int verticalPadding, int horizontalPadding,
		int verticalStride, int horizontalStride) {

		half alpha = 1;
		half beta = 1;

		cudnnPoolingDescriptor_t poolingDesc;
		CUDNN_CHECK(cudnnCreatePoolingDescriptor(&poolingDesc));
		CUDNN_CHECK(cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, CUDNN_PROPAGATE_NAN,
			windowsHeight, windowsWidth, verticalPadding, horizontalPadding, verticalStride, horizontalStride));

		cudnnTensorDescriptor_t xDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&xDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t dxDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dxDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dxDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)xShape.dim(0), (int)xShape.dim(3), (int)xShape.dim(1), (int)xShape.dim(2)));

		cudnnTensorDescriptor_t yDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&yDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		cudnnTensorDescriptor_t dyDesc;
		CUDNN_CHECK(cudnnCreateTensorDescriptor(&dyDesc));
		CUDNN_CHECK(cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, (int)yShape.dim(0), (int)yShape.dim(3), (int)yShape.dim(1), (int)yShape.dim(2)));

		CUDNN_CHECK(cudnnPoolingBackward(device->cudnnHandle, poolingDesc, &alpha, yDesc, y, dyDesc, dy, xDesc, x, &beta, dxDesc, dx));

		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dyDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(yDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(dxDesc));
		CUDNN_CHECK(cudnnDestroyTensorDescriptor(xDesc));
		CUDNN_CHECK(cudnnDestroyPoolingDescriptor(poolingDesc));
	}
#endif // HAVE_HALF
#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
#ifdef HAVE_CUDNN

		auto inputH = static_cast<int>(iGradient->shape.dim(1));
		auto inputW = static_cast<int>(iGradient->shape.dim(2));

		auto outputH = static_cast<int>(outputGradient->shape.dim(1));
		auto outputW = static_cast<int>(outputGradient->shape.dim(2));

		int padY = std::max<int>(0, (outputH - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputH);
		int padX = std::max<int>(0, (outputW - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth) - inputW);

		int padTop  = (padY / 2);
		int padLeft = (padX / 2);

		backwardGPUCUDNNImpl(static_cast<GPUDevice*>(output->device),
			inputs[0]->data(), iGradient->data(), iGradient->shape,
			output->data(), outputGradient->data(), output->shape,
			(int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);
		
#else
		DEEP8_RUNTIME_ERROR("the MaxPooling2d needs CUDNN");
#endif

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_MAXPOOLING2D_H
