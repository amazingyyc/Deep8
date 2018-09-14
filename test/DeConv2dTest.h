#ifndef DEEP8_DECONV2DTEST_H
#define DEEP8_DECONV2DTEST_H

#include "DeConv2d.h"

namespace Deep8 {

TEST(DeConv2d, forwardCPU) {
    bool isCovered = true;

    size_t strideY = 3;
	size_t strideX = 3;

    size_t batch = 2;
    size_t inputHeight  = 32;
    size_t inputWidth   = 32;
    size_t inputChannel = 64;

    size_t filterH = 4;
    size_t filterW = 4;

    size_t outputHeight = (inputHeight - 1) * strideY + 1 - strideY + filterH;
    size_t outputWidth  = (inputWidth  - 1) * strideX + 1 - strideX + filterW;
    size_t outputChannel = 32;

    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, batch, inputHeight, inputWidth, inputChannel);
    auto filter = createTensor<CPUDevice, float>(device, outputChannel, filterH,  filterW, inputChannel);
    auto output = createTensor<CPUDevice, float>(device, batch, outputHeight, outputWidth, outputChannel);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device, {size_t(batch), size_t(inputHeight), size_t(inputWidth), size_t(inputChannel)});
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device, { size_t(outputChannel), size_t(filterH),  size_t(filterW), size_t(inputChannel)});

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    DeConv2d<float> transposeConv2d(inputs, isCovered, size_t(strideY), size_t(strideX));

    std::vector<const Tensor<float>*> inputTensor = {&input, &filter};

    transposeConv2d.forwardCPU(inputTensor, &output);

    int64_t padH = std::max<int64_t>(0, outputHeight + filterH - (inputHeight - 1) * static_cast<int64_t>(strideY) - 2);
    int64_t padW = std::max<int64_t>(0, outputWidth + filterW - (inputWidth - 1) * static_cast<int64_t>(strideX) - 2);

    int64_t padTop    = -padH / 2;
    int64_t padLeft   = -padW / 2;

    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < outputHeight; ++h) {
            for (int64_t w = 0; w < outputWidth; ++w) {
                for (int64_t c = 0; c < outputChannel; ++c) {
                    int64_t inY = padTop  + h;
                    int64_t inX = padLeft + w;

                    float temp = 0;

                    for (int64_t y = 0; y < filterH; ++y) {
                        for (int64_t x = 0; x < filterW; ++x) {
                            int64_t realInY = inY + y;
                            int64_t realInX = inX + x;

                            if (0 > realInY || 0 > realInX || 0 != (realInY % strideY) || 0 != (realInX % strideX)) {
                                continue;
                            }

                            realInY /= strideY;
                            realInX /= strideX;

                            if (!(realInY >= 0 && realInY < inputHeight && realInX >= 0 && realInX < inputWidth)) {
                                continue;
                            }

                            for (int64_t k = 0; k < inputChannel; ++k) {
                                temp += input.data()[b * inputHeight * inputWidth * inputChannel + realInY * inputWidth * inputChannel + realInX * inputChannel + k]
                                        * filter.data()[c * filterH * filterW * inputChannel + y * filterW * inputChannel + x * inputChannel + k];
                            }
                        }
                    }

                    ASSERT_EQ(temp, output.data()[b * outputHeight * outputWidth * outputChannel + h * outputWidth * outputChannel + w * outputChannel + c]);
                }
            }
        }
    }

    freeTensor(device, input);
    freeTensor(device, filter);
    freeTensor(device, output);

	freeFakeVariable(inputVar1);
	freeFakeVariable(inputVar2);

    delete device;
}

TEST(DeConv2d, backwardCPU) {
    bool isCovered = true;

    size_t strideH = 3;
    size_t strideW = 3;
	
    size_t batch = 2;
    size_t inputHeight  = 32;
    size_t inputWidth   = 32;
    size_t inputChannel = 64;
	
    size_t filterH = 4;
    size_t filterW = 4;
	
    size_t outputHeight = (inputHeight - 1) * strideH + 1 - strideH + filterH;
    size_t outputWidth  = (inputWidth  - 1) * strideW + 1 - strideW + filterW;
    size_t outputChannel = 32;

    auto device = new CPUDevice();

	auto input = createTensor<CPUDevice, float>(device, batch, inputHeight, inputWidth, inputChannel);
	auto filter = createTensor<CPUDevice, float>(device, outputChannel, filterH, filterW, inputChannel);
	auto output = createTensor<CPUDevice, float>(device, batch, outputHeight, outputWidth, outputChannel);

    auto inputGrad  = createTensor<CPUDevice, float>(device, batch, inputHeight, inputWidth, inputChannel);
    auto filterGrad = createTensor<CPUDevice, float>(device, outputChannel, filterH,  filterW, inputChannel);
    auto outputGrad = createTensor<CPUDevice, float>(device, batch, outputHeight, outputWidth, outputChannel);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device, {size_t(batch), size_t(inputHeight), size_t(inputWidth), size_t(inputChannel)});
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device, { size_t(outputChannel), size_t(filterH),  size_t(filterW), size_t(inputChannel)});

    zeroTensor(device, inputGrad);
    zeroTensor(device, filterGrad);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    DeConv2d<float> transposeConv2d(inputs, isCovered, size_t(strideH), size_t(strideW));

    std::vector<const Tensor<float>*> inputTensor = {&input, &filter};

    transposeConv2d.backwardCPU(inputTensor, &output, &outputGrad, 0, &inputGrad);
    transposeConv2d.backwardCPU(inputTensor, &output, &outputGrad, 1, &filterGrad);

    int64_t padH = std::max<int64_t>(0, outputHeight + filterH - (inputHeight - 1) * static_cast<int64_t>(strideH) - 2);
    int64_t padW = std::max<int64_t>(0, outputWidth + filterW - (inputWidth - 1) * static_cast<int64_t>(strideW) - 2);

    int64_t padTop    = -padH / 2;
    int64_t padLeft   = -padW / 2;

    auto *inputMat = (float*)device->malloc(sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);
    auto *inputGradTemp = (float*)device->malloc(sizeof(float) * batch * inputHeight * inputWidth * inputChannel);

    device->zero(inputGradTemp, sizeof(float) * batch * inputHeight * inputWidth * inputChannel);
    device->zero(inputMat, sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);

    /**
     * test the inputGrad
     */
    for (int64_t m = 0; m < batch * outputHeight * outputWidth; ++m) {
        for (int64_t n = 0; n < filterH * filterW * inputChannel; ++n) {
            float temp = 0;

            for (int64_t k = 0; k < outputChannel; ++k) {
                temp += outputGrad.data()[m * outputChannel + k] * filter.data()[k * filterH * filterW * inputChannel + n];
            }

            inputMat[m * filterH * filterW * inputChannel + n] = temp;
        }
    }

    for (int64_t row = 0; row < batch * outputHeight * outputWidth; ++row) {
        int64_t b = row / (outputHeight * outputWidth);
        int64_t outH = (row % (outputHeight * outputWidth)) / outputWidth;
        int64_t outW = (row % (outputHeight * outputWidth)) % outputWidth;

        int64_t inH = padTop  + outH;
        int64_t inW = padLeft + outW;

        for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
            int64_t fH = col / (filterW * inputChannel);
            int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
            int64_t inC = (col % (filterW * inputChannel)) % inputChannel;

            int64_t realInH = inH + fH;
            int64_t realInW = inW + fW;

            if (0 > realInH || 0 > realInW) {
                continue;
            }

            if (0 != realInH % strideH || 0 != realInW % strideW) {
                continue;
            }

            realInH /= strideH;
            realInW /= strideW;

            if (realInH >= inputHeight || realInW >= inputWidth) {
                continue;
            }

            inputGradTemp[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC]
                    += inputMat[row * filterH * filterW * inputChannel + col];
        }
    }

    for (int64_t i = 0; i < batch * inputHeight * inputWidth * inputChannel; ++i) {
        ASSERT_EQ(inputGradTemp[i], inputGrad.data()[i]);
    }

    /**
     * test the filterGrad
     */
    device->zero(inputMat, sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);

    for (int64_t row = 0; row < batch * outputHeight * outputWidth; ++row) {
        int64_t b = row / (outputHeight * outputWidth);
        int64_t outH = (row % (outputHeight * outputWidth)) / outputWidth;
        int64_t outW = (row % (outputHeight * outputWidth)) % outputWidth;

        int64_t inH = padTop  + outH;
        int64_t inW = padLeft + outW;

        for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
            int64_t fH = col / (filterW * inputChannel);
            int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
            int64_t inC = (col % (filterW * inputChannel)) % inputChannel;

            int64_t realInH = inH + fH;
            int64_t realInW = inW + fW;

            if (0 > realInH || 0 > realInW) {
                continue;
            }

            if (0 != realInH % strideH || 0 != realInW % strideW) {
                continue;
            }

            realInH /= strideH;
            realInW /= strideW;

            if (realInH >= inputHeight || realInW >= inputWidth) {
                continue;
            }

            inputMat[row * filterH * filterW * inputChannel + col] =
                    input.data()[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC];
        }
    }

	for (int64_t m = 0; m < outputChannel; m++) {
		for (int64_t n = 0; n < filterH * filterW * inputChannel; ++n) {
			float temp = 0;

			for (int64_t k = 0; k < batch * outputHeight * outputWidth; ++k) {
				temp += inputMat[k * filterH * filterW * inputChannel + n] * outputGrad.data()[k * outputChannel + m];
			}

			ASSERT_EQ(temp, filterGrad.data()[m * filterH * filterW * inputChannel + n]);
		}
	}

    freeTensor(device, input);
    freeTensor(device, filter);
    freeTensor(device, output);
    freeTensor(device, inputGrad);
    freeTensor(device, filterGrad);
    freeTensor(device, outputGrad);

	freeFakeVariable(inputVar1);
	freeFakeVariable(inputVar2);

    device->free(inputMat);
    device->free(inputGradTemp);

    delete device;
}


#ifdef HAVE_CUDA

TEST(DeConv2d, forwarGPU_float) {
	auto device = new GPUDevice();

	bool isCovered = true;

	size_t strideY = 3;
	size_t strideX = 3;

	size_t batch = 2;
	size_t inputHeight = 32;
	size_t inputWidth = 32;
	size_t inputChannel = 64;

	size_t filterH = 4;
	size_t filterW = 4;

	size_t outputHeight = (inputHeight - 1) * strideY + 1 - strideY + filterH;
	size_t outputWidth = (inputWidth - 1) * strideX + 1 - strideX + filterW;
	size_t outputChannel = 32;

	auto inputPtr  = (float*)malloc(sizeof(float) * batch * inputHeight * inputWidth * inputChannel);
	auto filterPtr = (float*)malloc(sizeof(float) * outputChannel * filterH * filterW * inputChannel);
	auto outputPtr = (float*)malloc(sizeof(float) * batch * outputHeight * outputWidth * outputChannel);

	auto input = createTensorGPU<float>(device, inputPtr, batch, inputHeight, inputWidth, inputChannel);
	auto filter = createTensorGPU<float>(device, filterPtr, outputChannel, filterH, filterW, inputChannel);
	auto output = createTensorGPU<float>(device, outputPtr, batch, outputHeight, outputWidth, outputChannel);

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device, { batch, inputHeight, inputWidth, inputChannel });
	auto inputVar2 = createFakeVariable<GPUDevice, float>(device, { outputChannel, filterH, filterW, inputChannel });

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	DeConv2d<float> transposeConv2d(inputs, isCovered, size_t(strideY), size_t(strideX));

	std::vector<const Tensor<float>*> inputTensor = { &input, &filter };

	transposeConv2d.forwardGPU(inputTensor, &output);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(float) * batch * outputHeight * outputWidth * outputChannel);

	int64_t padH = std::max<int64_t>(0, outputHeight + filterH - (inputHeight - 1) * static_cast<int64_t>(strideY) - 2);
	int64_t padW = std::max<int64_t>(0, outputWidth + filterW - (inputWidth - 1) * static_cast<int64_t>(strideX) - 2);

	int64_t padTop = -padH / 2;
	int64_t padLeft = -padW / 2;


	for (int64_t b = 0; b < batch; ++b) {
		for (int64_t h = 0; h < outputHeight; ++h) {
			for (int64_t w = 0; w < outputWidth; ++w) {
				for (int64_t c = 0; c < outputChannel; ++c) {
					int64_t inY = padTop + h;
					int64_t inX = padLeft + w;

					float temp = 0;

					for (int64_t y = 0; y < filterH; ++y) {
						for (int64_t x = 0; x < filterW; ++x) {
							int64_t realInY = inY + y;
							int64_t realInX = inX + x;

							if (0 > realInY || 0 > realInX || 0 != (realInY % strideY) || 0 != (realInX % strideX)) {
								continue;
							}

							realInY /= strideY;
							realInX /= strideX;

							if (!(realInY >= 0 && realInY < inputHeight && realInX >= 0 && realInX < inputWidth)) {
								continue;
							}

							for (int64_t k = 0; k < inputChannel; ++k) {
								temp += inputPtr[b * inputHeight * inputWidth * inputChannel + realInY * inputWidth * inputChannel + realInX * inputChannel + k]
									* filterPtr[c * filterH * filterW * inputChannel + y * filterW * inputChannel + x * inputChannel + k];
							}
						}
					}

					ASSERT_EQ(temp, outputPtr[b * outputHeight * outputWidth * outputChannel + h * outputWidth * outputChannel + w * outputChannel + c]);
				}
			}
		}
	}

	freeTensor(device, input);
	freeTensor(device, filter);
	freeTensor(device, output);

	freeFakeVariable(inputVar1);
	freeFakeVariable(inputVar2);

	free(inputPtr);
	free(filterPtr);
	free(outputPtr);

	delete device;
}

TEST(DeConv2d, backwarGPU_float) {
	bool isCovered = true;

    size_t strideH = 3;
    size_t strideW = 3;
	
    size_t batch = 2;
    size_t inputHeight  = 32;
    size_t inputWidth   = 32;
    size_t inputChannel = 64;
	
    size_t filterH = 4;
    size_t filterW = 4;
	
    size_t outputHeight = (inputHeight - 1) * strideH + 1 - strideH + filterH;
    size_t outputWidth  = (inputWidth  - 1) * strideW + 1 - strideW + filterW;
    size_t outputChannel = 32;

    auto device = new GPUDevice();

	auto inputValuePtr = (float*)malloc(sizeof(float)*batch * inputHeight * inputWidth * inputChannel);
	auto inputGradPtr = (float*)malloc(sizeof(float)*batch * inputHeight * inputWidth * inputChannel);

	auto filterValuePtr = (float*)malloc(sizeof(float) * outputChannel * filterH * filterW * inputChannel);
	auto filterGradPtr = (float*)malloc(sizeof(float) * outputChannel * filterH * filterW * inputChannel);

	auto outputValuePtr = (float*)malloc(sizeof(float)* batch * outputHeight * outputWidth * outputChannel);
	auto outputGradPtr = (float*)malloc(sizeof(float)* batch * outputHeight * outputWidth * outputChannel);

	auto input = createTensorGPU<float>(device, inputValuePtr, batch, inputHeight, inputWidth, inputChannel);
	auto inputGrad = createTensorGPU<float>(device, inputGradPtr, batch, inputHeight, inputWidth, inputChannel);

	auto filter = createTensorGPU<float>(device, filterValuePtr, outputChannel, filterH, filterW, inputChannel);
	auto filterGrad = createTensorGPU<float>(device, filterGradPtr, outputChannel, filterH, filterW, inputChannel);

	auto output = createTensorGPU<float>(device, outputValuePtr, batch, outputHeight, outputWidth, outputChannel);
	auto outputGrad = createTensorGPU<float>(device, outputGradPtr, batch, outputHeight, outputWidth, outputChannel);

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device, { batch, inputHeight, inputWidth, inputChannel });
	auto inputVar2 = createFakeVariable<GPUDevice, float>(device, { outputChannel, filterH, filterW, inputChannel });

	zeroTensor(device, inputGrad);
	zeroTensor(device, filterGrad);

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	DeConv2d<float> transposeConv2d(inputs, isCovered, size_t(strideH), size_t(strideW));

	std::vector<const Tensor<float>*> inputTensor = { &input, &filter };

	transposeConv2d.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);
	transposeConv2d.backwardGPU(inputTensor, &output, &outputGrad, 1, &filterGrad);

	device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(float) * batch * inputHeight * inputWidth * inputChannel);
	device->copyFromGPUToCPU(filterGrad.pointer, filterGradPtr, sizeof(float) * outputChannel * filterH * filterW * inputChannel);

	int64_t padH = std::max<int64_t>(0, outputHeight + filterH - (inputHeight - 1) * static_cast<int64_t>(strideH) - 2);
	int64_t padW = std::max<int64_t>(0, outputWidth + filterW - (inputWidth - 1) * static_cast<int64_t>(strideW) - 2);

	int64_t padTop = -padH / 2;
	int64_t padLeft = -padW / 2;

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

		int64_t inH = padTop + outH;
		int64_t inW = padLeft + outW;

		for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
			int64_t fH = col / (filterW * inputChannel);
			int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
			int64_t inC = (col % (filterW * inputChannel)) % inputChannel;

			int64_t realInH = inH + fH;
			int64_t realInW = inW + fW;

			if (0 > realInH || 0 > realInW) {
				continue;
			}

			if (0 != realInH % strideH || 0 != realInW % strideW) {
				continue;
			}

			realInH /= strideH;
			realInW /= strideW;

			if (realInH >= inputHeight || realInW >= inputWidth) {
				continue;
			}

			inputGradTemp[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC]
				+= inputMat[row * filterH * filterW * inputChannel + col];
		}
	}

	for (int64_t i = 0; i < batch * inputHeight * inputWidth * inputChannel; ++i) {
		ASSERT_EQ(inputGradTemp[i], inputGradPtr[i]);
	}

	/**
	* test the filterGrad
	*/
	memset(inputMat, 0, sizeof(float) * batch * outputHeight * outputWidth * filterH * filterW * inputChannel);

	for (int64_t row = 0; row < batch * outputHeight * outputWidth; ++row) {
		int64_t b = row / (outputHeight * outputWidth);
		int64_t outH = (row % (outputHeight * outputWidth)) / outputWidth;
		int64_t outW = (row % (outputHeight * outputWidth)) % outputWidth;

		int64_t inH = padTop + outH;
		int64_t inW = padLeft + outW;

		for (int64_t col = 0; col < filterH * filterW * inputChannel; ++col) {
			int64_t fH = col / (filterW * inputChannel);
			int64_t fW = (col % (filterW * inputChannel)) / inputChannel;
			int64_t inC = (col % (filterW * inputChannel)) % inputChannel;

			int64_t realInH = inH + fH;
			int64_t realInW = inW + fW;

			if (0 > realInH || 0 > realInW) {
				continue;
			}

			if (0 != realInH % strideH || 0 != realInW % strideW) {
				continue;
			}

			realInH /= strideH;
			realInW /= strideW;

			if (realInH >= inputHeight || realInW >= inputWidth) {
				continue;
			}

			inputMat[row * filterH * filterW * inputChannel + col] =
				inputValuePtr[b * inputHeight * inputWidth * inputChannel + realInH * inputWidth * inputChannel + realInW * inputChannel + inC];
		}
	}

	for (int64_t m = 0; m < outputChannel; m++) {
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

	freeTensor(device, input);
	freeTensor(device, filter);
	freeTensor(device, output);
	freeTensor(device, inputGrad);
	freeTensor(device, filterGrad);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar1);
	freeFakeVariable(inputVar2);

	free(inputMat);
	free(inputGradTemp);

	delete device;
}

#endif

}

#endif //DEEP8_TRANSPOSECONV2DTEST_H
