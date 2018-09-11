#ifndef DEEP8_MINUSSCALARTEST_H
#define DEEP8_MINUSSCALARTEST_H

#include "MinusScalar.h"

namespace Deep8 {

TEST(MinusScalar, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, 10, 500, 200);
    auto output = createTensor<CPUDevice, float>(device, 10, 500, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    MinusScalar<float> add(inputs, 3.0);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    add.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_TRUE(abs(input.data()[i * 500 * 200 + j * 200 + k] - 3.0 - output.data()[i * 500 * 200 + j * 200 + k]) < 1e-6);
            }
        }
    }

    freeTensor<CPUDevice, float>(device, input);
    freeTensor<CPUDevice, float>(device, output);

    freeFakeVariable(inputVar1);

    delete device;
}

TEST(MinusScalar, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue1 = createTensor<CPUDevice, float>(device, 10, 500, 200);
	auto inputGrad1 = createTensor<CPUDevice, float>(device, 10, 500, 200);

    auto outputValue = createTensor<CPUDevice, float>(device, 10, 500, 200);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 10, 500, 200);

    /**create fake Add Function*/
    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    MinusScalar<float> add(inputs, 5.0);

    zeroTensor(device, inputGrad1);

    std::vector<const Tensor<float>*> inputValues = {&inputValue1};

    add.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);

    for (int i = 0; i < 10 * 500 * 200; ++i) {
        ASSERT_TRUE(abs(inputGrad1.data()[i] - outputGrad.data()[i]) < 1e-6);
    }

    freeTensor<CPUDevice, float>(device, inputValue1);
    freeTensor<CPUDevice, float>(device, inputGrad1);
    freeTensor<CPUDevice, float>(device, outputValue);
    freeTensor<CPUDevice, float>(device, outputGrad);

    freeFakeVariable(inputVar1);

    delete device;
}


#ifdef HAVE_CUDA

TEST(MinusScalar, GPU_float) {
	typedef float real;

	auto device = new GPUDevice();

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 10, 500, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 10, 500, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 10, 500, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 10, 500, 200);

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device);

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar1 };
	MinusScalar<real> minusScalar(inputs, 3.0);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	minusScalar.forwardGPU(inputTensor, &output);
	minusScalar.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device->copyToCPU(output.pointer, outputPtr, sizeof(real) * 10 * 500 * 200);
	device->copyToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 10 * 500 * 200);

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 500; ++j) {
			for (int k = 0; k < 200; ++k) {
				ASSERT_TRUE(abs(inputPtr[i * 500 * 200 + j * 200 + k] - 3.0 - outputPtr[i * 500 * 200 + j * 200 + k]) < 1e-6);
			}
		}
	}

	for (int i = 0; i < 10 * 500 * 200; ++i) {
		ASSERT_TRUE(abs(inputGradPtr[i] - outputGradPtr[i]) < 1e-6);
	}

	free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input);
	freeTensor(device, inputGrad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	delete device;
}

#endif

}

#endif //DEEP8_MINUSSCALARTEST_H
