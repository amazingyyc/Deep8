#ifndef DEEP8_AVGPOOLING2D_H
#define DEEP8_AVGPOOLING2D_H

#include "Function.h"

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

	template <typename real>
	void forwardGPUImpl(GPUDevice *device, const real *x, real *y,
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

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dForwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dForwardKernel<real> << <grideSize, blockSize >> > (x, y,
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}

#ifdef HAVE_HALF

	template <>
	void forwardGPUImpl<half>(GPUDevice *device, const half *x, half *y,
		const int batch,
		const int inputHeight, const int inputWidth,
		const int outputHeight, const int outputWidth,
		const int channel,
		const int filterHeight, const int filterWidth,
		const int padTop, const int padLeft,
		const int strideY, const int strideX) {

		int N = batch * outputHeight * outputWidth * channel;

		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dForwardKernel<half> << <grideSize, blockSize >> > (x, y,
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}
#endif // HAVE_HALF
#endif

	void forwardGPU(const std::vector<const Tensor<T>*> &inputs, Tensor<T> *output) override {
#ifdef HAVE_CUDA

		auto device = static_cast<GPUDevice*>(output->device());

		auto batch       = static_cast<int>(inputs[0]->shape.dim(0));
		auto inputHeight = static_cast<int>(inputs[0]->shape.dim(1));
		auto inputWidth  = static_cast<int>(inputs[0]->shape.dim(2));
		auto channel     = static_cast<int>(inputs[0]->shape.dim(3));

		auto outputHeight = static_cast<int>(output->shape.dim(1));
		auto outputWidth  = static_cast<int>(output->shape.dim(2));

		int padY = std::max<int>(0, (outputHeight - 1) * static_cast<int>(strideY) + static_cast<int>(filterHeight) - inputHeight);
		int padX = std::max<int>(0, (outputWidth  - 1) * static_cast<int>(strideX) + static_cast<int>(filterWidth)  - inputWidth);

		int padTop  = -(padY / 2);
		int padLeft = -(padX / 2);

		forwardGPUImpl(device, inputs[0]->data(), output->data(),
			batch, inputHeight, inputWidth, outputHeight, outputWidth, channel, (int)filterHeight, (int)filterWidth, padTop, padLeft, (int)strideY, (int)strideX);

#else
		DEEP8_RUNTIME_ERROR("can not call the GPU function without a GPU");
#endif
	}

#ifdef HAVE_CUDA

	template <typename real>
	void backwardGPUImpl(GPUDevice *device, real *dx, const real *dy,
		const int batch, const int inputHeight, const int inputWidth,
		const int outputHeight, const int outputWidth, const int channel,
		const int filterHeight, const int filterWidth,
		const int padTop, const int padLeft,
		const int strideY, const int strideX) {

		int N = batch * inputHeight * inputWidth * channel;

		int minGrideSize;
		int blockSize;
		int grideSize;

		CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGrideSize, &blockSize, AvgPooling2dBackwardKernel<real>, 0, N));

		grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dBackwardKernel<real> << <grideSize, blockSize >> > (dx, dy,
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}

#ifdef HAVE_HALF

	template <>
	void backwardGPUImpl<half>(GPUDevice *device, half *dx, const half *dy,
		const int batch, const int inputHeight, const int inputWidth,
		const int outputHeight, const int outputWidth, const int channel,
		const int filterHeight, const int filterWidth,
		const int padTop, const int padLeft,
		const int strideY, const int strideX) {

		int N = batch * inputHeight * inputWidth * channel;

		int blockSize = 1024;
		int grideSize = (N + blockSize - 1) / blockSize;

		AvgPooling2dBackwardKernel<half> << <grideSize, blockSize >> > (dx, dy,
			batch, inputHeight, inputWidth,
			outputHeight, outputWidth, channel,
			filterHeight, filterWidth,
			padTop, padLeft,
			strideY, strideX, N);
	}

#endif
#endif

    void backwardGPU(const std::vector<const Tensor<T>*> &inputs,
                     const Tensor<T> *output,
                     const Tensor<T> *outputGradient,
                     size_t index,
                     Tensor<T> *iGradient) override {
#ifdef HAVE_CUDA

		DEEP8_ARGUMENT_CHECK(0 == index, "the index is error");

		auto device = static_cast<GPUDevice*>(output->device());

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
