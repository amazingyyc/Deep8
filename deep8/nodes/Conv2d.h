#ifndef DEEP8_CONV2D_H
#define DEEP8_CONV2D_H

namespace Deep8 {

#ifdef HAVE_CUDA

template <typename real>
__global__ void Conv2dIm2ColKernel(const real *im, real *col, 
	const int batch, const int inputHeight, const int inputWidth, const int inputChannel, 
	const int filterHeight, const int filterWidth, const int padTop, const int padLeft, 
	const int strideY, const int strideX, const int dilationY, const int dilationX, 
	const int outputHeight, const int outputWidth, const int N) {

	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = start; i < N; i += stride) {
		int colY = i / (filterHeight * filterWidth);
		int colX = i % (filterHeight * filterWidth);

		int b = colY / (outputHeight * outputWidth);

		int outputIndex = colY % (outputHeight * outputWidth);

		int outputY = outputIndex / outputWidth;
		int outputX = outputIndex % outputWidth;

		int filterY = colX / filterWidth;
		int filterX = colX % filterWidth;

		int inputY = padTop  + outputY * strideY + filterY * dilationY;
		int inputX = padLeft + outputX * strideX + filterX * dilationX;

		real *colPtr = col + colY * filterHeight * filterWidth * inputChannel + colX * inputChannel;

		if (0 > inputY || inputY >= inputHeight || 0 > inputX || inputX >= inputWidth) {
			for (int k = 0; k < inputChannel; ++k) {
				colPtr[k] = 0;
			}
		} else {
			const real *imPtr = im + b * inputHeight * inputWidth * inputChannel + inputY * inputWidth * inputChannel + inputX * inputChannel;

			for (int k = 0; k < inputChannel; ++k) {
				colPtr[k] = imPtr[k];
			}
		}
	}
}

template <typename real>
__global__ void Conv2dCol2ImKernel(const real *col, real *im,
	const int batch, const int inputHeight, const int inputWidth, const int inputChannel,
	const int filterHeight, const int filterWidth, const int padTop, const int padLeft,
	const int strideY, const int strideX, const int dilationY, const int dilationX,
	const int outputHeight, const int outputWidth, const int N) {

	int start  = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	int colWidth = filterHeight * filterWidth * inputChannel;

	for (int i = start; i < N; i += stride) {
		int b = i / (inputHeight * inputWidth);

		int inputIndex = i % (inputHeight * inputWidth);

		int inputY = inputIndex / inputWidth;
		int inputX = inputIndex % inputWidth;

		real *imPtr = im + b * inputHeight * inputWidth * inputChannel + inputY * inputWidth * inputChannel + inputX * inputChannel;

		for (int filterY = 0; filterY < filterHeight; ++filterY) {
			for (int filterX = 0; filterX < filterWidth; ++filterX) {
				int outputY = inputY - padTop - filterY * dilationY;
				int outputX = inputX - padLeft - filterX * dilationX;

				if (0 == (outputY % strideY) && 0 == (outputX % strideX)) {
					outputY /= strideY;
					outputX /= strideX;

					if (0 <= outputY && outputY < outputHeight && 0 <= outputX && outputX < outputWidth) {
						const real *colPtr = col + (b * outputHeight * outputWidth + outputY * outputWidth + outputX) * colWidth
							+ (filterY * filterWidth + filterX) * inputChannel;

						for (int k = 0; k < inputChannel; ++k) {
							imPtr[k] += colPtr[k];
						}
					}
				}
			}
		}
	}
}

#endif

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

    Conv2d(std::vector<Node *> &inputs, bool covered = false, size_t strideH = 1, size_t strideW = 1, size_t dilationH = 1, size_t dilationW = 1):
            Function<T>(inputs), covered(covered), strideY(strideH), strideX(strideW), dilationY(dilationH), dilationX(dilationW) {
        check();
    }

    void check() override {
        Function<T>::check();

        DEEP8_ARGUMENT_CHECK(2 == this->inputs.size(), "need 2 inputs node");
        DEEP8_ARGUMENT_CHECK(strideY >= 1 && strideX >= 1, "the stride can not smaller than 1");
        DEEP8_ARGUMENT_CHECK(dilationY >= 1 && dilationX >= 1, "the dilation can not smaller than 1");

        auto inputShape  = static_cast<Variable<T>*>(this->inputs[0])->value.shape;
        auto filterShape = static_cast<Variable<T>*>(this->inputs[1])->value.shape;

        DEEP8_ARGUMENT_CHECK(4 == inputShape.nDims() && 4 == filterShape.nDims(), "Conv2d needs inputs nDims is 4");
        DEEP8_ARGUMENT_CHECK(inputShape.dim(3) == filterShape.dim(3), "the inputs dimension is error");
        DEEP8_ARGUMENT_CHECK(filterShape.dim(1) > 0 &&  filterShape.dim(2) > 0, "the filter must bigger than 0");

        if (!covered) {
            DEEP8_ARGUMENT_CHECK(filterShape.dim(1) <= inputShape.dim(1) && filterShape.dim(2) <= inputShape.dim(2),
                                 "the not forwardCovered mode Padding type needs filter smaller than input");
        }

        auto filterH = static_cast<int64_t>(filterShape.dim(1));
        auto filterW = static_cast<int64_t>(filterShape.dim(2));

        auto inputH = static_cast<int64_t>(inputShape.dim(1));
        auto inputW = static_cast<int64_t>(inputShape.dim(2));

        auto realFilterH = filterH + (filterH - 1) * (static_cast<int64_t>(dilationY) - 1);
        auto realFilterW = filterW + (filterW - 1) * (static_cast<int64_t>(dilationX) - 1);

        std::vector<size_t> outputDim(4);
        outputDim[0] = inputShape.dim(0);
        outputDim[3] = filterShape.dim(0);

        /**
         * the input dimension is (batch, inputHeight, inputWidth, inputChannel)
         * filter dimension is (outputChannel, filterHeight, filterWidth, inputChannel)
         * output dimension is (batch, outputHeight, outputWidth, outputChannel)
         */
        if (!covered) {
            int64_t outputH = (inputH - realFilterH) / static_cast<int64_t>(strideY) + 1;
            int64_t outputW = (inputW - realFilterW) / static_cast<int64_t>(strideX) + 1;

            DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

            outputDim[1] = static_cast<size_t>(outputH);
            outputDim[2] = static_cast<size_t>(outputW);
        } else {
            int64_t outputH = (inputH - realFilterH + static_cast<int64_t>(strideY) - 1) / static_cast<int64_t>(strideY) + 1;
            int64_t outputW = (inputW - realFilterW + static_cast<int64_t>(strideX) - 1) / static_cast<int64_t>(strideX) + 1;

            DEEP8_ARGUMENT_CHECK(outputH > 0 && outputW > 0, "the output height or width must > 0")

            outputDim[1] = static_cast<size_t>(outputH);
            outputDim[2] = static_cast<size_t>(outputW);
        }

        this->outputShape = Shape(outputDim);
    }

protected:
    void forwardCPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
		/**
		 * the input dimension in Deep8 is NHWC (batch, inputHeight, intputWidth, inputChannel)
		 * the filter dimension is NHWC (outputChannel, filterHeight, filterWidth, inputChannel)
		 * the output dimension is NHWC (batch, outputHeight, outputWidth, outputChannel)
		 */
		typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

		auto device = static_cast<CPUDevice*>(output->device)->eigenDevice;

        auto input  = inputs[0];
        auto filter = inputs[1];

		auto batch = (TensorIndex)input->shape.batch();

		auto inputHeight  = (TensorIndex)input->shape.dim(1);
		auto inputWidth   = (TensorIndex)input->shape.dim(2);
		auto inputChannel = (TensorIndex)input->shape.dim(3);

		auto outputHeight  = (TensorIndex)output->shape.dim(1);
		auto outputWidth   = (TensorIndex)output->shape.dim(2);
		auto outputChannel = (TensorIndex)output->shape.dim(3);

		auto filterHeight = (TensorIndex)filter->shape.dim(1);
		auto filterWidth  = (TensorIndex)filter->shape.dim(2);

		Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
			inputTensor(input->data(), batch, inputHeight, inputWidth, inputChannel);

		Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
			filterTensor(filter->data(), outputChannel, filterHeight, filterWidth, inputChannel);

		Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
			outputTensor(output->data(), batch, outputHeight, outputWidth, outputChannel);

		Eigen::DSizes<TensorIndex, 2> preContractDims;
		preContractDims[0] = batch * outputHeight * outputWidth;
		preContractDims[1] = filterHeight * filterWidth * inputChannel;

		Eigen::DSizes<TensorIndex, 2> shuffleDims;
		shuffleDims[0] = 1;
		shuffleDims[1] = 0;

		Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contractDims;
		contractDims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

		Eigen::DSizes<TensorIndex, 2> kernelDims;
		kernelDims[0] = outputChannel;
		kernelDims[1] = filterHeight * filterWidth * inputChannel;

		auto realFilterHeight = filterHeight + (filterHeight - 1) * ((TensorIndex)(dilationY) - 1);
		auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * ((TensorIndex)(dilationX) - 1);

		auto padY = std::max<TensorIndex>(0, (outputHeight - 1) * (TensorIndex)(strideY) + realFilterHeight - inputHeight);
		auto padX = std::max<TensorIndex>(0, (outputWidth  - 1) * (TensorIndex)(strideX) + realFilterWidth  - inputWidth);

		auto padTop    = padY / 2;
		auto padBottom = padY - padTop;
		auto padLeft   = padX / 2;
		auto padRight  = padX - padLeft;

		outputTensor.device(*device) = inputTensor.extract_image_patches(filterWidth, filterHeight, strideX, strideY, dilationX, dilationY, 1, 1, padTop, padBottom, padLeft, padRight, 0)
			.reshape(preContractDims)
			.contract(filterTensor.reshape(kernelDims).shuffle(shuffleDims), contractDims)
			.reshape(outputTensor.dimensions());
    }

    void backwardCPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
		DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

		typedef typename Eigen::internal::traits<Eigen::Tensor<T, 4, Eigen::RowMajor>>::Index TensorIndex;

		auto device = static_cast<CPUDevice*>(iGradient->device)->eigenDevice;

		auto inputShape  = inputs[0]->shape;
		auto filterShape = inputs[1]->shape;
		auto outputShape = output->shape;

		auto batch = (TensorIndex)inputShape.batch();

		auto inputHeight  = (TensorIndex)inputShape.dim(1);
		auto inputWidth   = (TensorIndex)inputShape.dim(2);
		auto inputChannel = (TensorIndex)inputShape.dim(3);

		auto outputHeight  = (TensorIndex)outputShape.dim(1);
		auto outputWidth   = (TensorIndex)outputShape.dim(2);
		auto outputChannel = (TensorIndex)outputShape.dim(3);

		auto filterHeight = (TensorIndex)filterShape.dim(1);
		auto filterWidth  = (TensorIndex)filterShape.dim(2);

		if (0 == index) {
			/**
			 * calculate the grad for input
			 */
			Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
				outputGradTensor(outputGradient->data(), batch, outputHeight, outputWidth, outputChannel);

			Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
				inputGradTensor(iGradient->data(), batch, inputHeight, inputWidth, inputChannel);

			Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
				filterTensor(inputs[1]->data(), outputChannel, filterHeight, filterWidth, inputChannel);

			Eigen::DSizes<TensorIndex, 2> preContractDims;
			preContractDims[0] = batch * inputHeight * inputWidth;
			preContractDims[1] = filterHeight * filterWidth * outputChannel;

			Eigen::internal::conditional<false, Eigen::array<bool, 4>, Eigen::array<bool, 4>>::type filterReverse;
			filterReverse[0] = false;
			filterReverse[1] = true;
			filterReverse[2] = true;
			filterReverse[3] = false;

			Eigen::DSizes<TensorIndex, 4> filterShuffle;
			filterShuffle[0] = 1;
			filterShuffle[1] = 2;
			filterShuffle[2] = 0;
			filterShuffle[3] = 3;

			Eigen::DSizes<TensorIndex, 2> filterDim;
			filterDim[0] = filterHeight * filterWidth * outputChannel;
			filterDim[1] = inputChannel;

			Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contractDims;
			contractDims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

			auto realFilterHeight = filterHeight + (filterHeight - 1) * ((TensorIndex)(dilationY) - 1);
			auto realFilterWidth  = filterWidth  + (filterWidth - 1)  * ((TensorIndex)(dilationX) - 1);

			auto forwardPadTop  = std::max<TensorIndex>(0, ((outputHeight - 1) * (TensorIndex)(strideY) +realFilterHeight - inputHeight) / 2);
			auto forwardPadLeft = std::max<TensorIndex>(0, ((outputWidth  - 1) * (TensorIndex)(strideX) +realFilterWidth - inputWidth) / 2);

			auto padTop  = realFilterHeight - 1 - forwardPadTop;
			auto padLeft = realFilterWidth - 1 - forwardPadLeft;
			auto padBottom = inputHeight - (outputHeight - 1) * (TensorIndex)(strideY) - 2 - padTop  + realFilterHeight;
			auto padRight  = inputWidth - (outputWidth - 1) * (TensorIndex)(strideX)   - 2 - padLeft + realFilterWidth;

			inputGradTensor.device(*device) += outputGradTensor.extract_image_patches(filterWidth, filterHeight, 1, 1, dilationX, dilationY, strideX, strideY, padTop, padBottom, padLeft, padRight, 0)
				.reshape(preContractDims)
				.contract(filterTensor.reverse(filterReverse).shuffle(filterShuffle).reshape(filterDim), contractDims)
				.reshape(inputGradTensor.dimensions());
		} else if (1 == index) {
			/**
			 * calculate the filter gradient
			 */
			Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
				outputGradTensor(outputGradient->data(), batch, outputHeight, outputWidth, outputChannel);

			Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
				filterGradTensor(iGradient->data(), outputChannel, filterHeight, filterWidth, inputChannel);

			Eigen::TensorMap<Eigen::Tensor<T, 4, Eigen::RowMajor>>
				inputTensor(inputs[0]->data(), batch, inputHeight, inputWidth, inputChannel);

			Eigen::DSizes<TensorIndex, 2> preContractDims;
			preContractDims[0] = batch * outputHeight * outputWidth;
			preContractDims[1] = filterHeight * filterWidth * inputChannel;

			Eigen::DSizes<TensorIndex, 2> outputGradDim;
			outputGradDim[0] = batch * outputHeight * outputWidth;
			outputGradDim[1] = outputChannel;

			Eigen::array<Eigen::IndexPair<TensorIndex>, 1> contractDims;
			contractDims[0] = Eigen::IndexPair<TensorIndex>(1, 0);

			Eigen::DSizes<TensorIndex, 2> shuffleDims;
			shuffleDims[0] = 1;
			shuffleDims[1] = 0;

			TensorIndex realFilterH = filterHeight + (filterHeight - 1) * ((TensorIndex)(dilationY) - 1);
			TensorIndex realFilterW = filterWidth + (filterWidth - 1) * ((TensorIndex)(dilationX) - 1);

			TensorIndex padY = std::max<TensorIndex>(0, (outputHeight - 1) * (TensorIndex)(strideY) + realFilterH - inputHeight);
			TensorIndex padX = std::max<TensorIndex>(0, (outputWidth - 1) * (TensorIndex)(strideX) + realFilterW - inputWidth);

			TensorIndex padTop    = padY / 2;
			TensorIndex padBottom = padY - padTop;
			TensorIndex padLeft   = padX / 2;
			TensorIndex padRight  = padX - padLeft;

			filterGradTensor.device(*device) += outputGradTensor.reshape(outputGradDim).shuffle(shuffleDims)
				.contract(inputTensor.extract_image_patches(filterWidth, filterHeight, strideX, strideY, dilationX, dilationY, 1, 1, padTop, padBottom, padLeft, padRight, 0).reshape(preContractDims), contractDims)
				.reshape(filterGradTensor.dimensions());
		}
    }
	
#ifdef HAVE_CUDA

	void forwardGPUImpl(GPUDevice* device, const float *x, const float *filter, float *y,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX) {

		int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, Conv2dIm2ColKernel<float>, 0, size));

		grideSize = (size + blockSize - 1) / blockSize;

		auto xCol = (float*)device->malloc(sizeof(float) * size * inputChannel);

		Conv2dIm2ColKernel<float> << <grideSize, blockSize >> > (x, xCol, 
			batch, inputHeight, inputWidth, inputChannel, 
			filterHeight, filterWidth, padTop, padLeft,
			strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

		int m = batch * outputHeight * outputWidth;
		int k = filterHeight * filterWidth * inputChannel;
		int n = outputChannel;

		float alpha = 1;
		float beta  = 0;

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, xCol, k, &beta, y, n));

		device->free(xCol);
	}

	void forwardGPUImpl(GPUDevice* device, const double *x, const double *filter, double *y,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX) {

		int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, Conv2dIm2ColKernel<double>, 0, size));

		grideSize = (N + blockSize - 1) / blockSize;

		auto xCol = (double*)device->malloc(sizeof(double) * size * inputChannel);

		Conv2dIm2ColKernel<double> << <grideSize, blockSize >> > (x, xCol,
			batch, inputHeight, inputWidth, inputChannel,
			filterHeight, filterWidth, padTop, padLeft,
			strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

		int m = batch * outputHeight * outputWidth;
		int k = filterHeight * filterWidth * inputChannel;
		int n = outputChannel;

		double alpha = 1;
		double beta = 0;

		CUBLAS_CHECK(cublasDgemm(device->cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, filter, k, xCol, k, &beta, y, n));

		device->free(xCol);
	}

#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA

		auto device = static_cast<GPUDevice*>(output->device);

		auto x = inputs[0];
		auto filter = inputs[1];

		auto y = output;

		auto batch        = (int)x->shape.dim(0);
		auto inputHeight  = (int)x->shape.dim(1);
		auto inputWidth   = (int)x->shape.dim(2);
		auto inputChannel = (int)x->shape.dim(3);

		auto outputHeight  = (int)y->shape.dim(1);
		auto outputWidth   = (int)y->shape.dim(2);
		auto outputChannel = (int)y->shape.dim(3);

		auto filterHeight = (int)filter->shape.dim(1);
		auto filterWidth  = (int)filter->shape.dim(2);

		auto realFilterHeight = filterHeight + (filterHeight - 1) * ((int)dilationY - 1);
		auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * ((int)dilationX - 1);
		
		auto padTop  = -(std::max<int>(0, (outputHeight - 1) * (int)(strideY) + realFilterHeight - inputHeight) / 2);
		auto padLeft = -(std::max<int>(0, (outputWidth  - 1) * (int)(strideX) + realFilterWidth  - inputWidth)  / 2);

		forwardGPUImpl(device, x->data(), filter->data(), y->data(),
			batch, inputHeight, inputWidth, inputChannel, outputHeight, outputWidth, outputChannel,
			filterHeight, filterWidth, (int)strideY, (int)strideX, padTop, padLeft, (int)dilationY, (int)dilationX);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}

#ifdef HAVE_CUDA

	void backwardGPUInputImpl(GPUDevice* device, float *dx, const float *w, const float *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX) {

		int size = batch * inputHeight * inputWidth;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, Conv2dCol2ImKernel<float>, 0, size));

		grideSize = (size + blockSize - 1) / blockSize;

		auto xCol = (float*)device->malloc(sizeof(float) * batch * outputHeight * outputWidth * filterHeight * filterWidth * inputChannel);

		int m = filterHeight * filterWidth * inputChannel;
		int k = outputChannel;
		int n = batch * outputHeight * outputWidth;

		float alpha = 1;
		float beta  = 0;

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, w, m, dy, k, &beta, xCol, m));

		Conv2dCol2ImKernel<float> << <grideSize, blockSize >> > (xCol, dx,
			batch, inputHeight, inputWidth, inputChannel,
			filterHeight, filterWidth, padTop, padLeft,
			strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

		device->free(xCol);
	}

	void backwardGPUFilterImpl(GPUDevice* device, const float *x, float *dw, const float *dy,
		int batch, int inputHeight, int inputWidth, int inputChannel,
		int outputHeight, int outputWidth, int outputChannel,
		int filterHeight, int filterWidth, int strideY, int strideX,
		int padTop, int padLeft, int dilationY, int dilationX) {

		int size = batch * outputHeight * outputWidth * filterHeight * filterWidth;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, Conv2dIm2ColKernel<float>, 0, size));

		grideSize = (size + blockSize - 1) / blockSize;

		auto xCol = (float*)device->malloc(sizeof(float) * size * inputChannel);

		Conv2dIm2ColKernel<float> << <grideSize, blockSize >> > (x, xCol,
			batch, inputHeight, inputWidth, inputChannel,
			filterHeight, filterWidth, padTop, padLeft,
			strideY, strideX, dilationY, dilationX, outputHeight, outputWidth, size);

		int m = filterHeight * filterWidth * inputChannel;
		int k = batch * outputHeight * outputWidth;
		int n = outputChannel;

		float alpha = 1;
		float beta  = 1;

		CUBLAS_CHECK(cublasSgemm(device->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha, xCol, m, dy, n, &beta, dw, m));

		device->free(xCol);
	}

#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA
		
		DEEP8_ARGUMENT_CHECK(0 == index || 1 == index, "the index is error");

		auto device = static_cast<GPUDevice*>(iGradient->device);

		auto xShape = inputs[0]->shape;
		auto wShape = inputs[1]->shape;
		auto yShape = output->shape;

		auto batch        = (int)xShape.dim(0);
		auto inputHeight  = (int)xShape.dim(1);
		auto inputWidth   = (int)xShape.dim(2);
		auto inputChannel = (int)xShape.dim(3);

		auto outputHeight  = (int)yShape.dim(1);
		auto outputWidth   = (int)yShape.dim(2);
		auto outputChannel = (int)yShape.dim(3);

		auto filterHeight = (int)wShape.dim(1);
		auto filterWidth  = (int)wShape.dim(2);

		auto realFilterHeight = filterHeight + (filterHeight - 1) * ((int)dilationY - 1);
		auto realFilterWidth  = filterWidth  + (filterWidth  - 1) * ((int)dilationX - 1);

		auto padTop  = -(std::max<int>(0, (outputHeight - 1) * (int)(strideY) +realFilterHeight - inputHeight) / 2);
		auto padLeft = -(std::max<int>(0, (outputWidth  - 1) * (int)(strideX) +realFilterWidth  - inputWidth)  / 2);

		if (0 == index) {
				backwardGPUInputImpl(device, iGradient->data(), inputs[1]->data(), outputGradient->data(),
				batch, inputHeight, inputWidth, inputChannel, outputHeight, outputWidth, outputChannel,
				filterHeight, filterWidth, strideY, strideX, padTop, padLeft, dilationY, dilationX);
		} else if (1 == index) {
			backwardGPUFilterImpl(device, inputs[0]->data(), iGradient->data(), outputGradient->data(),
				batch, inputHeight, inputWidth, inputChannel, outputHeight, outputWidth, outputChannel,
				filterHeight, filterWidth, strideY, strideX, padTop, padLeft, dilationY, dilationX);
		}

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
    }
};


}

#endif //DEEP8_CONV2D_H
