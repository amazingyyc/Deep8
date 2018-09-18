#ifndef DEEP8_AVGPOOLING2D_H
#define DEEP8_AVGPOOLING2D_H

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void AvgPooling2dForwardKernel(const real *input, real *output,
	const int batch, const int inputHeight, const int inputWidth,
	const int outputHeight, const int outputWidth, const int channel,
	const int filterHeight, const int filterWidth, 
	const int padTop, const int padLeft,
	const int strideY, const int strideX, const int N) {

	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		int b = i / (outputHeight * outputWidth * channel);
		int outputY = (i % (outputHeight * outputWidth * channel)) / (outputWidth * channel);
		int outputX = (i % (outputWidth * channel)) / channel;
		int offset  = i % channel;

		real sum = 0;

		for (int y = 0; y < filterHeight; ++y) {
			for (int x = 0; x < filterWidth; ++x) {
				int inputY = outputY * strideY + padTop + y;
				int inputX = outputX * strideX + padLeft + x;

				if (0 <= inputY && inputY < inputHeight && 0 <= inputX && inputX < inputWidth) {
					sum += input[((b * inputHeight + inputY) * inputWidth + inputX) * channel + offset];
				}
			}
		}

		output[i] = sum / real(filterHeight * filterWidth);
	}
}

template <typename real>
__global__ void AvgPooling2dBackwardKernel(real *dx, const real *dy,
	const int batch, const int inputHeight, const int inputWidth,
	const int outputHeight, const int outputWidth, const int channel,
	const int filterHeight, const int filterWidth,
	const int padTop, const int padLeft,
	const int strideY, const int strideX, const int N) {

	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	real ratio = real(1) / real(filterHeight * filterWidth);

	for (int i = start; i < N; i += stride) {
		int b = i / (inputHeight * inputWidth * channel);
		int inputY = (i % (inputHeight * inputWidth * channel)) / (inputWidth * channel);
		int inputX = (i % (inputWidth * channel)) / channel;
		int offset = i % channel;

		for (int y = 0; y < filterHeight; ++y) {
			for (int x = 0; x < filterWidth; ++x) {
				int outputY = inputY - padTop  - y;
				int outputX = inputX - padLeft - x;

				if (0 == outputY % strideY && 0 == outputX % strideX) {
					outputY /= strideY;
					outputX /= strideX;

					if (0 <= outputY && outputY < outputHeight && 0 <= outputX && outputX < outputWidth) {
						dx[i] += ratio * dy[((b * outputHeight + outputY) * outputWidth + outputX) * channel + offset];
					}
				}
			}
		}
	}
}

#endif

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

    AvgPooling2d(std::vector<Node *> &inputs, bool covered = false, size_t filterH = 1, size_t filterW = 1, size_t strideH = 1, size_t strideW = 1):
            Function<T>(inputs), covered(covered), filterHeight(filterH), filterWidth(filterW), strideY(strideH), strideX(strideW) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(1 == this->inputs.size(), "the AvgPooling2d only need 1 input");
        DEEP8_ARGUMENT_CHECK(filterHeight >= 1 && filterWidth >= 1 && strideY >= 1 && strideX  >= 1, "the filter size or stride is error");
        DEEP8_ARGUMENT_CHECK(4 == this->inputs[0]->outputShape.nDims(), "AvgPooling2d needs inputs nDims is 4");

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
            int64_t outputW = (inputW - static_cast<int64_t>(filterWidth)) / static_cast<int64_t>(strideX) + 1;

            DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

            outputDim[1] = static_cast<size_t>(outputH);
            outputDim[2] = static_cast<size_t>(outputW);
        } else {
            int64_t outputH = (inputH - static_cast<int64_t>(filterHeight) + static_cast<int64_t>(strideY) - 1) / static_cast<int64_t>(strideY) + 1;
            int64_t outputW = (inputW - static_cast<int64_t>(filterWidth) + static_cast<int64_t>(strideX) - 1) / static_cast<int64_t>(strideX) + 1;

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
                .mean(reductionDims)
                .reshape(outputTensor.dimensions());
    }

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
                         int64_t padLeft) {
        T ratio = T(1) / (T(filterH) * T(filterW));

        for (int64_t b = 0; b < batch; ++b) {
            auto inputGradPtr  = inputGrad  + b * inputHeight  * inputWidth  * channel;
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

                        auto grad = outputGradPtr[y * outputWidth * channel + x * channel + k];

                        for (int64_t inH = startH; inH < endH; ++inH) {
                            for (int64_t inW = startW; inW < endW; ++inW) {
                                inputGradPtr[inH * inputWidth * channel + inW * channel + k] += (ratio * grad);
                            }
                        }
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

        auto batch  = static_cast<int64_t>(iGradient->shape.dim(0));
        auto inputH = static_cast<int64_t>(iGradient->shape.dim(1));
        auto inputW = static_cast<int64_t>(iGradient->shape.dim(2));
        auto inputC = static_cast<int64_t>(iGradient->shape.dim(3));

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

        auto blockFunc = [this, &barrier] (T *inputGrad,
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
            this->backwardCPUImpl(inputGrad,
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

            device->enqueueNoNotification(blockFunc,
                                          iGradient->data(),
                                          outputGradient->data(),
                                          batch,
                                          startChannel,
                                          endChannel,
                                          inputH,
                                          inputW,
                                          outputH,
                                          outputW,
                                          outputC,
                                          filterHeight,
                                          filterWidth,
                                          strideY,
                                          strideX,
                                          padTop,
                                          padLeft);
		}

        barrier.Wait();
    }

#ifdef HAVE_CUDNN

	void forwardGPUImpl(GPUDevice *device, const float *x, float *y,
		const int batch,
		const int inputHeight, const int inputWidth,
		const int outputHeight, const int outputWidth,
		const int channel,
		const int filterHeight, const int filterWidth,
		const int padTop, const int padLeft,
		const int strideY, const int strideX) {

		int N = batch * outputHeight * outputWidth * channel;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dForwardKernel<float>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dForwardKernel<float> << <grideSize, blockSize >> > (x, y, 
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}

	void forwardGPUImpl(GPUDevice *device, const double *x, double *y,
		const int batch,
		const int inputHeight, const int inputWidth,
		const int outputHeight, const int outputWidth,
		const int channel,
		const int filterHeight, const int filterWidth,
		const int padTop, const int padLeft,
		const int strideY, const int strideX) {

		int N = batch * outputHeight * outputWidth * channel;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dForwardKernel<double>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dForwardKernel<double> << <grideSize, blockSize >> > (x, y,
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}


#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA

		auto device = static_cast<GPUDevice*>(output->device);

		auto batch       = static_cast<int>(inputs[0]->shape.dim(0));
		auto inputHeight = static_cast<int>(inputs[0]->shape.dim(1));
		auto inputWidth  = static_cast<int>(inputs[0]->shape.dim(2));
		auto channel     = static_cast<int>(inputs[0]->shape.dim(3));

		auto outputHeight = static_cast<int>(output->shape.dim(1));
		auto outputWidth  = static_cast<int>(output->shape.dim(2));

		int padY = std::max<int>(0, (outputHeight - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputHeight);
		int padX = std::max<int>(0, (outputWidth - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth)  - inputWidth);

		int padTop  = -(padY / 2);
		int padLeft = -(padX / 2);

		forwardGPUImpl(device, inputs[0]->data(), output->data(),
			batch, inputHeight, inputWidth, outputHeight, outputWidth, channel, (int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}

#ifdef HAVE_CUDNN

	void backwardGPUImpl(GPUDevice *device, float *dx, const float *dy,
		const int batch, const int inputHeight, const int inputWidth,
		const int outputHeight, const int outputWidth, const int channel,
		const int filterHeight, const int filterWidth,
		const int padTop, const int padLeft,
		const int strideY, const int strideX) {

		int N = batch * inputHeight * inputWidth * channel;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dBackwardKernel<float>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dBackwardKernel<float> << <grideSize, blockSize >> > (dx, dy,
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}

	void backwardGPUImpl(GPUDevice *device, double *dx, const double *dy,
		const int batch, const int inputHeight, const int inputWidth,
		const int outputHeight, const int outputWidth, const int channel,
		const int filterHeight, const int filterWidth,
		const int padTop, const int padLeft,
		const int strideY, const int strideX) {

		int N = batch * inputHeight * inputWidth * channel;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dBackwardKernel<double>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dBackwardKernel<double> << <grideSize, blockSize >> > (dx, dy,
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}

#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA

		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<GPUDevice*>(output->device);

		auto batch       = static_cast<int>(inputs[0]->shape.dim(0));
		auto inputHeight = static_cast<int>(inputs[0]->shape.dim(1));
		auto inputWidth  = static_cast<int>(inputs[0]->shape.dim(2));
		auto channel     = static_cast<int>(inputs[0]->shape.dim(3));

		auto outputHeight = static_cast<int>(output->shape.dim(1));
		auto outputWidth  = static_cast<int>(output->shape.dim(2));

		int padY = std::max<int>(0, (outputHeight - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputHeight);
		int padX = std::max<int>(0, (outputWidth  - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth) - inputWidth);

		int padTop  = -(padY / 2);
		int padLeft = -(padX / 2);

		backwardGPUImpl(device, iGradient->data(), outputGradient->data(),
			batch, inputHeight, inputWidth, outputHeight, outputWidth, channel,
			(int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_AVGPOOL2D_H
