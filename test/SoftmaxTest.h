#ifndef DEEP8_SOFTMAXTEST_H
#define DEEP8_SOFTMAXTEST_H

#include "Softmax.h"

namespace Deep8 {

TEST(Softmax, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, 400, 200);
    auto output = createTensor<CPUDevice, float>(device, 400, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    Softmax<float> softmax(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    softmax.forwardCPU(inputTensor, &output);

    auto temp = (float*)device->malloc(sizeof(float) * 200);

    for (int b = 0; b < 400; ++b) {
        auto ptr = input.data() + b * 200;
        auto outputPtr = output.data() + b * 200;

        auto maxValue = ptr[0];

        for (int i = 0; i < 200; ++i) {
            maxValue = std::max<float>(maxValue, ptr[i]);
        }

        for (int i = 0; i < 200; ++i) {
            temp[i] = exp(ptr[i] - maxValue);
        }

        long double sumValue = 0;

        for (int i = 0; i < 200; ++i) {
            sumValue += temp[i];
        }

        for (int i = 0; i < 200; ++i) {
            temp[i] /= sumValue;

            ASSERT_TRUE(abs(temp[i] - outputPtr[i]) <= 1e-6);
        }
    }

    freeTensor(device, input);
    freeTensor(device, output);

	freeFakeVariable(inputVar1);

    device->free(temp);

    delete device;
}

TEST(Softmax, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, float>(device, 400, 200);
	auto inputGrad = createTensor<CPUDevice, float>(device, 400, 200);

    auto outputValue = createTensor<CPUDevice, float>(device, 400, 200);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 400, 200);

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    Softmax<float> softmax(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    softmax.forwardCPU(inputValues, &outputValue);
    softmax.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int b = 0; b < 400; ++b) {
        auto inputGradPtr  = inputGrad.data() + b * 200;
        auto outputGradPtr = outputGrad.data() + b * 200;
        auto outputValuePtr = outputValue.data() + b * 200;

        float sum = 0;

        for (int i = 0; i < 200; ++i) {
            sum += outputGradPtr[i] * outputValuePtr[i];
        }

        for (int i = 0; i < 200; ++i) {
            auto temp = (outputGradPtr[i] - sum) * outputValuePtr[i];

            ASSERT_TRUE(abs(temp - inputGradPtr[i]) < 1e-6);
        }
    }

	freeTensor(device, inputValue);
	freeTensor(device, inputGrad);
	freeTensor(device, outputValue);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

	delete device;
}

#ifdef HAVE_CUDA

TEST(Softmax, GPU_float) {
	typedef float real;

	auto device = new GPUDevice();

	auto inputPtr = (real*)malloc(sizeof(real) * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 400 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 400 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 400, 200);

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Softmax<real> softmax(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	softmax.forwardGPU(inputTensor, &output);
	softmax.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * 400 * 200);
	device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 400 * 200);

	auto temp = (real*)malloc(sizeof(real) * 200);

	for (int b = 0; b < 400; ++b) {
		auto ptr = inputPtr + b * 200;
		auto tempoutputPtr = outputPtr + b * 200;

		auto maxValue = ptr[0];

		for (int i = 0; i < 200; ++i) {
			maxValue = std::max<float>(maxValue, ptr[i]);
		}

		for (int i = 0; i < 200; ++i) {
			temp[i] = exp(ptr[i] - maxValue);
		}

		float sumValue = 0;

		for (int i = 0; i < 200; ++i) {
			sumValue += temp[i];
		}

		for (int i = 0; i < 200; ++i) {
			temp[i] /= sumValue;

			if (abs(temp[i] - tempoutputPtr[i]) > 1e-6) {
				std::cout << temp[i] << ", " << tempoutputPtr[i] << std::endl;
				ASSERT_TRUE(false);
			}
		}
	}

	for (int b = 0; b < 400; ++b) {
		auto tempinputGradPtr = inputGradPtr + b * 200;
		auto tempoutputGradPtr = outputGradPtr + b * 200;
		auto tempoutputValuePtr = outputPtr + b * 200;

		float sum = 0;

		for (int i = 0; i < 200; ++i) {
			sum += tempoutputGradPtr[i] * tempoutputValuePtr[i];
		}

		for (int i = 0; i < 200; ++i) {
			auto temp = (tempoutputGradPtr[i] - sum) * tempoutputValuePtr[i];

			if (abs(temp - tempinputGradPtr[i]) > 1e-4) {
				std::cout << temp << ", " << tempinputGradPtr[i] << "," << abs(temp - tempinputGradPtr[i]) << std::endl;
				ASSERT_TRUE(false);
			}
		}
	}

	free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);
	free(temp);

	freeTensor(device, input);
	freeTensor(device, inputGrad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	delete device;
}

#endif
}






#endif //DEEP8_SOFTMAXTEST_H
