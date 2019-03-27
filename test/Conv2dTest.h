#ifndef DEEP8_CONV2DTEST_H
#define DEEP8_CONV2DTEST_H

#include "nodes/Conv2d.h"

namespace Deep8 {

TEST(Conv2d, forwardCPU_float) {
    bool isCovered = true;

    size_t strideH = 2;
	size_t strideW = 2;

	size_t dilationH = 2;
	size_t dilationW = 2;

	size_t batch = 2;
	size_t inputHeight  = 32;
	size_t inputWidth   = 32;
	size_t inputChannel = 64;

	size_t filterH = 4;
	size_t filterW = 4;

    auto realFilterH = filterH + (filterH - 1) * (dilationH - 1);
    auto realFilterW = filterW + (filterW - 1) * (dilationW - 1);

    size_t outputHeight;
    size_t outputWidth;

    if (!isCovered) {
        outputHeight = (inputHeight - realFilterH) / static_cast<size_t>(strideH) + 1;
        outputWidth  = (inputWidth  - realFilterW) / static_cast<size_t>(strideW) + 1;
    } else {
        outputHeight = (inputHeight - 1) / static_cast<size_t>(strideH) + 1;
        outputWidth  = (inputWidth  - 1) / static_cast<size_t>(strideW) + 1;
    }

	size_t outputChannel = 32;

	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), batch, {inputHeight, inputWidth, inputChannel});
    auto filter = createTensor(device, ElementType::from<float>(), 1, {outputChannel, filterH,  filterW, inputChannel});
    auto output = createTensor(device, ElementType::from<float>(), batch, {outputHeight, outputWidth, outputChannel});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>(), size_t(batch), {size_t(inputHeight), size_t(inputWidth), size_t(inputChannel)});
    auto inputVar2 = createFakeVariable(device, ElementType::from<float>(), 1,  {size_t(outputChannel), size_t(filterH),  size_t(filterW), size_t(inputChannel)});

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Conv2d conv2d(inputs, isCovered, size_t(strideH), size_t(strideW), size_t(dilationH), size_t(dilationW));

    std::vector<const Tensor*> inputTensor = {&input, &filter};

    conv2d.forward(inputTensor, &output);

    int64_t padY = std::max<int64_t>(0, (outputHeight - 1) * static_cast<int64_t>(strideH) + realFilterH - inputHeight);
    int64_t padX = std::max<int64_t>(0, (outputWidth  - 1) * static_cast<int64_t>(strideW) + realFilterW - inputWidth);

    int64_t padTop    = -padY / 2;
    int64_t padLeft   = -padX / 2;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < outputHeight; ++h) {
            for (int w = 0; w < outputWidth; ++w) {
                for (int c = 0; c < outputChannel; ++c) {
                    int inY = padTop  + h * strideH;
                    int inX = padLeft + w * strideW;

                    float temp = 0;

                    for (int y = 0, yy = inY; y < filterH; ++y, yy += dilationH) {
                        for (int x = 0, xx = inX; x < filterW; ++x, xx += dilationW) {
                            if (!(yy >= 0 && yy < inputHeight && xx >= 0 && xx < inputWidth)) {
                                continue;
                            }

							for (int k = 0; k < inputChannel; ++k) {
								temp += input.data<float>()[b * inputHeight * inputWidth * inputChannel + yy * inputWidth * inputChannel + xx * inputChannel + k]
									* filter.data<float>()[c * filterH * filterW * inputChannel + y * filterW * inputChannel + x * inputChannel + k];
							}
                        }
                    }

                    ASSERT_EQ(temp, output.data<float>()[b * outputHeight * outputWidth * outputChannel + h * outputWidth * outputChannel + w * outputChannel + c]);
                }
            }
        }
    }
}

TEST(Conv2d, backwardCPU_float) {
    bool isCovered = true;

	size_t strideH = 1;
	size_t strideW = 1;

	size_t dilationH = 1;
	size_t dilationW = 1;

	size_t batch = 2;
	size_t inputHeight  = 32;
	size_t inputWidth   = 32;
	size_t inputChannel = 64;

	size_t filterH = 5;
	size_t filterW = 5;

    auto realFilterH = filterH + (filterH - 1) * (dilationH - 1);
    auto realFilterW = filterW + (filterW - 1) * (dilationW - 1);

    size_t outputHeight = (inputHeight - realFilterH) / static_cast<size_t>(strideH) + 1;
    size_t outputWidth = (inputWidth - realFilterW) / static_cast<size_t>(strideW) + 1;

    if (!isCovered) {
        outputHeight = (inputHeight - realFilterH) / static_cast<size_t>(strideH) + 1;
        outputWidth = (inputWidth - realFilterW) / static_cast<size_t>(strideW) + 1;
    } else {
        outputHeight = (inputHeight - 1) / static_cast<size_t>(strideH) + 1;
        outputWidth = (inputWidth - 1) / static_cast<size_t>(strideW) + 1;
    }
	
	size_t outputChannel = 32;

	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), batch, {inputHeight, inputWidth, inputChannel});
    auto filter = createTensor(device, ElementType::from<float>(), 1, {outputChannel, filterH,  filterW, inputChannel});
    auto output = createTensor(device, ElementType::from<float>(), batch, {outputHeight, outputWidth, outputChannel});

    auto inputGrad  = createTensor(device, ElementType::from<float>(), batch, {inputHeight, inputWidth, inputChannel});
    auto filterGrad = createTensor(device, ElementType::from<float>(), 1, {outputChannel, filterH,  filterW, inputChannel});
    auto outputGrad = createTensor(device, ElementType::from<float>(), batch, {outputHeight, outputWidth, outputChannel});

    /**create fake Function*/
    auto inputVar0 = createFakeVariable(device, ElementType::from<float>(), size_t(batch), {size_t(inputHeight), size_t(inputWidth), size_t(inputChannel)});
    auto inputVar1 = createFakeVariable(device, ElementType::from<float>(), 1,  {size_t(outputChannel), size_t(filterH),  size_t(filterW), size_t(inputChannel)});

    std::vector<Node*> inputs = {&inputVar0, &inputVar1};
    Conv2d conv2d(inputs, isCovered, size_t(strideH), size_t(strideW), size_t(dilationH), size_t(dilationW));

    zeroTensor(device, inputGrad);
    zeroTensor(device, filterGrad);

    std::vector<const Tensor*> inputValues = {&input, &filter};

    conv2d.backward(inputValues, &output, &outputGrad, 0, &inputGrad);
    conv2d.backward(inputValues, &output, &outputGrad, 1, &filterGrad);

    int64_t padY = std::max<int64_t>(0, (outputHeight - 1) * static_cast<int64_t>(strideH) + realFilterH - inputHeight);
    int64_t padX = std::max<int64_t>(0, (outputWidth  - 1) * static_cast<int64_t>(strideW) + realFilterW - inputWidth);

    int64_t padTop    = -padY / 2;
    int64_t padLeft   = -padX / 2;

    auto *inputMat = (float*)device.malloc(sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);
    auto *inputGradTemp = (float*)device.malloc(sizeof(float) * batch * inputHeight * inputWidth * inputChannel);

    device.zero(inputGradTemp, sizeof(float) * batch * inputHeight * inputWidth * inputChannel);
    device.zero(inputMat, sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);

    /**
     * test the inputGrad
     */
    for (int64_t m = 0; m < batch * outputHeight * outputWidth; ++m) {
        for (int64_t n = 0; n < filterH * filterW * inputChannel; ++n) {
            float temp = 0;

            for (int64_t k = 0; k < outputChannel; ++k) {
                temp += outputGrad.data<float>()[m * outputChannel + k] * filter.data<float>()[k * filterH * filterW * inputChannel + n];
            }

            inputMat[m * filterH * filterW * inputChannel + n] = temp;
        }
    }

    for (int64_t row = 0; row < batch * outputHeight * outputWidth; ++row) {
        int64_t b = row / (outputHeight * outputWidth);
        int64_t outH = (row % (outputHeight * outputWidth)) / outputWidth;
        int64_t outW = (row % (outputHeight * outputWidth)) % outputWidth;

        int64_t inH = padTop  + outH * strideH;
        int64_t inW = padLeft + outW * strideW;

        for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
            int64_t fH = col / (filterW * inputChannel);
            int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
            int64_t inC = (col % (filterW * inputChannel)) % inputChannel;

            int64_t realInH = inH + dilationH * fH;
            int64_t realInW = inW + dilationW * fW;

            if (realInH >= 0 && realInH < inputHeight && realInW >= 0 && realInW < inputWidth) {
                inputGradTemp[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC]
                        += inputMat[row * filterH * filterW * inputChannel + col];
            }
        }
    }

    for (int64_t i = 0; i < batch * inputHeight * inputWidth * inputChannel; ++i) {
        ASSERT_EQ(inputGradTemp[i], inputGrad.data<float>()[i]);
    }

    /**
     * test the filterGrad
     */
    device.zero(inputMat, sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);

    for (int64_t row = 0; row < batch * outputHeight * outputWidth; ++row) {
        int64_t b = row / (outputHeight * outputWidth);
        int64_t outH = (row % (outputHeight * outputWidth)) / outputWidth;
        int64_t outW = (row % (outputHeight * outputWidth)) % outputWidth;

        int64_t inH = padTop  + outH * strideH;
        int64_t inW = padLeft + outW * strideW;

        for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
            int64_t fH = col / (filterW * inputChannel);
            int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
            int64_t inC = (col % (filterW * inputChannel)) % inputChannel;

            int64_t realInH = inH + dilationH * fH;
            int64_t realInW = inW + dilationW * fW;

            if (realInH >= 0 && realInH < inputHeight && realInW >= 0 && realInW < inputWidth) {
                inputMat[row * filterH * filterW * inputChannel + col] =
                     input.data<float>()[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC];
            }
        }
    }

	for (int64_t m = 0; m < outputChannel; ++m) {
		for (int64_t n = 0; n < filterH * filterW * inputChannel; ++n) {
			float temp = 0;

			for (int64_t k = 0; k < batch * outputHeight * outputWidth; ++k) {
				temp += inputMat[k * filterH * filterW * inputChannel + n] * outputGrad.data<float>()[k * outputChannel + m];
			}

			ASSERT_EQ(temp, filterGrad.data<float>()[m * filterH * filterW * inputChannel + n]);
		}
	}

    device.free(inputMat);
    device.free(inputGradTemp);
}

#ifdef HAVE_CUDA

TEST(Conv2d, forwardGPU_float) {
    bool isCovered = false;

    size_t strideH = 2;
	size_t strideW = 2;

	size_t dilationH = 2;
	size_t dilationW = 2;

	size_t batch = 2;
	size_t inputHeight  = 32;
	size_t inputWidth   = 32;
	size_t inputChannel = 64;

	size_t filterH = 4;
	size_t filterW = 4;

    auto realFilterH = filterH + (filterH - 1) * (dilationH - 1);
    auto realFilterW = filterW + (filterW - 1) * (dilationW - 1);

	size_t outputHeight = (inputHeight - realFilterH) / static_cast<size_t>(strideH) + 1;
	size_t outputWidth  = (inputWidth - realFilterW)  / static_cast<size_t>(strideW) + 1;
	size_t outputChannel = 32;

	GPUDevice device;

    auto inputPtr  = (float*)malloc(sizeof(float) * batch * inputHeight * inputWidth * inputChannel);
    auto filterPtr = (float*)malloc(sizeof(float) * outputChannel * filterH * filterW * inputChannel);
    auto outputPtr = (float*)malloc(sizeof(float) * batch * outputHeight * outputWidth * outputChannel);

    auto input  = createTensor(device, inputPtr,  ElementType::from<float>(), batch, {inputHeight, inputWidth, inputChannel});
    auto filter = createTensor(device, filterPtr, ElementType::from<float>(), 1, {outputChannel, filterH,  filterW, inputChannel});
    auto output = createTensor(device, outputPtr, ElementType::from<float>(), batch, {outputHeight, outputWidth, outputChannel});

	auto inputVar1 = createFakeVariable(device, ElementType::from<float>(),batch, {inputHeight, inputWidth, inputChannel });
	auto inputVar2 = createFakeVariable(device, ElementType::from<float>(), 1, {outputChannel, filterH, filterW, inputChannel });

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Conv2d conv2d(inputs, isCovered, size_t(strideH), size_t(strideW), size_t(dilationH), size_t(dilationW));

    std::vector<const Tensor*> inputTensor = {&input, &filter};

    conv2d.forward(inputTensor, &output);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(float) * batch * outputHeight * outputWidth * outputChannel);

    int64_t padY = std::max<int64_t>(0, (outputHeight - 1) * static_cast<int64_t>(strideH) + realFilterH - inputHeight);
    int64_t padX = std::max<int64_t>(0, (outputWidth  - 1) * static_cast<int64_t>(strideW) + realFilterW - inputWidth);

    int64_t padTop    = -padY / 2;
    int64_t padLeft   = -padX / 2;

    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < outputHeight; ++h) {
            for (int w = 0; w < outputWidth; ++w) {
                for (int c = 0; c < outputChannel; ++c) {
                    int inY = padTop  + h * strideH;
                    int inX = padLeft + w * strideW;

                    float temp = 0;

                    for (int y = 0, yy = inY; y < filterH; ++y, yy += dilationH) {
                        for (int x = 0, xx = inX; x < filterW; ++x, xx += dilationW) {
                            if (!(yy >= 0 && yy < inputHeight && xx >= 0 && xx < inputWidth)) {
                                continue;
                            }

                            for (int k = 0; k < inputChannel; ++k) {
                                temp += inputPtr[b * inputHeight * inputWidth * inputChannel + yy * inputWidth * inputChannel + xx * inputChannel + k]
                                        * filterPtr[c * filterH * filterW * inputChannel + y * filterW * inputChannel + x * inputChannel + k];
                            }
                        }
                    }

                    ASSERT_EQ(temp, outputPtr[b * outputHeight * outputWidth * outputChannel + h * outputWidth * outputChannel + w * outputChannel + c]);
                }
            }
        }
    }

	free(inputPtr);
	free(filterPtr);
	free(outputPtr);

}

TEST(Conv2d, backwardGPU_float) {
    bool isCovered = true;

	size_t strideH = 1;
	size_t strideW = 1;

	size_t dilationH = 1;
	size_t dilationW = 1;

	size_t batch = 2;
	size_t inputHeight  = 32;
	size_t inputWidth   = 32;
	size_t inputChannel = 64;

	size_t filterH = 5;
	size_t filterW = 5;

    auto realFilterH = filterH + (filterH - 1) * (dilationH - 1);
    auto realFilterW = filterW + (filterW - 1) * (dilationW - 1);

	size_t outputHeight = (inputHeight - realFilterH) / static_cast<size_t>(strideH) + 1;
	size_t outputWidth  = (inputWidth - realFilterW) / static_cast<size_t>(strideW) + 1;
	size_t outputChannel = 32;

    if (!isCovered) {
        outputHeight = (inputHeight - realFilterH) / static_cast<size_t>(strideH) + 1;
        outputWidth = (inputWidth - realFilterW) / static_cast<size_t>(strideW) + 1;
    } else {
        outputHeight = (inputHeight - 1) / static_cast<size_t>(strideH) + 1;
        outputWidth = (inputWidth - 1) / static_cast<size_t>(strideW) + 1;
    }

	GPUDevice device;

    auto inputValuePtr = (float*)malloc(sizeof(float)*batch * inputHeight * inputWidth * inputChannel);
    auto inputGradPtr = (float*)malloc(sizeof(float)*batch * inputHeight * inputWidth * inputChannel);

    auto filterValuePtr = (float*)malloc(sizeof(float) * outputChannel * filterH * filterW * inputChannel);
    auto filterGradPtr = (float*)malloc(sizeof(float) * outputChannel * filterH * filterW * inputChannel);

    auto outputValuePtr = (float*)malloc(sizeof(float)* batch * outputHeight * outputWidth * outputChannel);
    auto outputGradPtr = (float*)malloc(sizeof(float)* batch * outputHeight * outputWidth * outputChannel);

    auto input      = createTensor(device, inputValuePtr, ElementType::from<float>(),batch, {inputHeight, inputWidth, inputChannel});
    auto inputGrad  = createTensor(device, inputGradPtr,  ElementType::from<float>(), batch, {inputHeight, inputWidth, inputChannel});
    auto filter     = createTensor(device, filterValuePtr,ElementType::from<float>(), 1, {outputChannel, filterH,  filterW, inputChannel});
    auto filterGrad = createTensor(device, filterGradPtr, ElementType::from<float>(), 1, {outputChannel, filterH,  filterW, inputChannel});
    auto output     = createTensor(device, outputValuePtr,ElementType::from<float>(), batch, {outputHeight, outputWidth, outputChannel});
    auto outputGrad = createTensor(device, outputGradPtr, ElementType::from<float>(), batch, {outputHeight, outputWidth, outputChannel});

    auto inputVar1  = createFakeVariable(device, ElementType::from<float>(), batch, {inputHeight, inputWidth, inputChannel });
	auto inputVar2  = createFakeVariable(device, ElementType::from<float>(), 1, {outputChannel, filterH, filterW, inputChannel });

    zeroTensor(device, inputGrad);
    zeroTensor(device, filterGrad);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Conv2d conv2d(inputs, isCovered, size_t(strideH), size_t(strideW), size_t(dilationH), size_t(dilationW));

    std::vector<const Tensor*> inputTensor = {&input, &filter};

    conv2d.forward(inputTensor, &output);

    std::vector<const Tensor*> inputValues = {&input, &filter};

    conv2d.backward(inputValues, &output, &outputGrad, 0, &inputGrad);
    conv2d.backward(inputValues, &output, &outputGrad, 1, &filterGrad);
    
	int64_t padY = std::max<int64_t>(0, (outputHeight - 1) * static_cast<int64_t>(strideH) + realFilterH - inputHeight);
	int64_t padX = std::max<int64_t>(0, (outputWidth - 1) * static_cast<int64_t>(strideW) + realFilterW - inputWidth);

	int64_t padTop = -padY / 2;
	int64_t padLeft = -padX / 2;

    device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(float) * batch * inputHeight * inputWidth * inputChannel);
	device.copyFromGPUToCPU(filterGrad.raw(), filterGradPtr, sizeof(float) * outputChannel * filterH * filterW * inputChannel);

    auto *inputMat = (float*)malloc(sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);
    auto *inputGradTemp = (float*)malloc(sizeof(float) * batch * inputHeight * inputWidth * inputChannel);

    memset(inputGradTemp, 0, sizeof(float) * batch * inputHeight * inputWidth * inputChannel);
    memset(inputMat, 0, sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);

    /**
     * test the inputGrad
     */
    for (int64_t m = 0; m < batch * outputHeight * outputWidth; ++m) {
        for (int64_t n = 0; n < filterH * filterW * inputChannel; ++n) {
            float temp = 0;

            for (int64_t k = 0; k < outputChannel; ++k) {
                temp += outputGradPtr[m * outputChannel + k] * filterValuePtr[k * filterH * filterW * inputChannel + n];
            }

            inputMat[m * filterH * filterW * inputChannel + n] = temp;
        }
    }

    for (int64_t row = 0; row < batch * outputHeight * outputWidth; ++row) {
        int64_t b = row / (outputHeight * outputWidth);
        int64_t outH = (row % (outputHeight * outputWidth)) / outputWidth;
        int64_t outW = (row % (outputHeight * outputWidth)) % outputWidth;

        int64_t inH = padTop  + outH * strideH;
        int64_t inW = padLeft + outW * strideW;

        for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
            int64_t fH = col / (filterW * inputChannel);
            int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
            int64_t inC = col % inputChannel;

            int64_t realInH = inH + dilationH * fH;
            int64_t realInW = inW + dilationW * fW;

            if (realInH >= 0 && realInH < inputHeight && realInW >= 0 && realInW < inputWidth) {
                inputGradTemp[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC]
                        += inputMat[row * filterH * filterW * inputChannel + col];
            }
        }
    }

    for (int64_t i = 0; i < batch * inputHeight * inputWidth * inputChannel; ++i) {
		ASSERT_EQ(inputGradTemp[i], inputGradPtr[i]);
    }

    memset(inputMat, 0, sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);

    for (int64_t row = 0; row < batch * outputHeight * outputWidth; ++row) {
        int64_t b = row / (outputHeight * outputWidth);
        int64_t outH = (row % (outputHeight * outputWidth)) / outputWidth;
        int64_t outW = (row % (outputHeight * outputWidth)) % outputWidth;

        int64_t inH = padTop  + outH * strideH;
        int64_t inW = padLeft + outW * strideW;

        for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
            int64_t fH = col / (filterW * inputChannel);
            int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
            int64_t inC = (col % (filterW * inputChannel)) % inputChannel;

            int64_t realInH = inH + dilationH * fH;
            int64_t realInW = inW + dilationW * fW;

            if (realInH >= 0 && realInH < inputHeight && realInW >= 0 && realInW < inputWidth) {
                inputMat[row * filterH * filterW * inputChannel + col] =
                     inputValuePtr[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC];
            }
        }
    }

    for (int64_t m = 0; m < outputChannel; ++m) {
         for (int64_t n = 0; n < filterH * filterW * inputChannel; ++n) {
             float temp = 0;

             for (int64_t k = 0; k < batch * outputHeight * outputWidth; ++k) {
                 temp += inputMat[k * filterH * filterW * inputChannel + n] * outputGradPtr[k * outputChannel + m];
             }

             ASSERT_EQ(temp, filterGradPtr[m * filterH * filterW * inputChannel + n]);
         }
    }

    free(inputValuePtr);
    free(inputGradPtr);
    free(filterValuePtr);
    free(filterGradPtr);
    free(outputValuePtr);
    free(outputGradPtr);

    free(inputMat);
    free(inputGradTemp);

}


#ifdef HAVE_HALF

TEST(Conv2d, half_GPU) {
	bool isCovered = false;

	size_t strideH = 2;
	size_t strideW = 2;

	size_t dilationH = 2;
	size_t dilationW = 2;

	size_t batch = 2;
	size_t inputHeight = 32;
	size_t inputWidth = 32;
	size_t inputChannel = 64;

	size_t filterH = 4;
	size_t filterW = 4;

	auto realFilterH = filterH + (filterH - 1) * (dilationH - 1);
	auto realFilterW = filterW + (filterW - 1) * (dilationW - 1);

	size_t outputHeight = (inputHeight - realFilterH) / static_cast<size_t>(strideH) + 1;
	size_t outputWidth = (inputWidth - realFilterW) / static_cast<size_t>(strideW) + 1;
	size_t outputChannel = 32;

	GPUDevice device;

    auto input      = createTensor(device,ElementType::from<half>(),  batch, {inputHeight, inputWidth, inputChannel});
    auto inputGrad  = createTensor(device,ElementType::from<half>(), batch, {inputHeight, inputWidth, inputChannel});
    auto filter     = createTensor(device,ElementType::from<half>(), 1, {outputChannel, filterH, filterW, inputChannel});
    auto filterGrad = createTensor(device,ElementType::from<half>(), 1, {outputChannel, filterH, filterW, inputChannel});
    auto output     = createTensor(device,ElementType::from<half>(), batch, {outputHeight, outputWidth, outputChannel});
    auto outputGrad = createTensor(device,ElementType::from<half>(), batch, {outputHeight, outputWidth, outputChannel});

	auto inputVar1 = createFakeVariable(device, ElementType::from<half>(), batch, {inputHeight, inputWidth, inputChannel });
	auto inputVar2 = createFakeVariable(device, ElementType::from<half>(), 1, {outputChannel, filterH, filterW, inputChannel });

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Conv2d conv2d(inputs, isCovered, size_t(strideH), size_t(strideW), size_t(dilationH), size_t(dilationW));

	std::vector<const Tensor*> inputTensor = { &input, &filter };

	conv2d.forward(inputTensor, &output);
	conv2d.backward(inputTensor, &output, &outputGrad, 0, &inputGrad);
	conv2d.backward(inputTensor, &output, &outputGrad, 1, &filterGrad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_CONV2DTEST_H
