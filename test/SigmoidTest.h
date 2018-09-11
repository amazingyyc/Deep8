#ifndef DEEP8_SIGMOIDTEST_H
#define DEEP8_SIGMOIDTEST_H

#include "Sigmoid.h"

namespace Deep8 {

TEST(Sigmoid, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, 10, 400, 200);
    auto output = createTensor<CPUDevice, float>(device, 10, 400, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    Sigmoid<float> sigmoid(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    sigmoid.forwardCPU(inputTensor,&output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        float temp = 1.f / (1.f + exp(-1.f * input.data()[i]));

        ASSERT_TRUE(abs(temp - output.data()[i]) <= 1e-6);
    }

    freeTensor(device, input);
    freeTensor(device, output);

	freeFakeVariable(inputVar1);

    delete device;
}

TEST(Sigmoid, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, long double>(device, 10, 400, 200);
	auto inputGrad = createTensor<CPUDevice, long double>(device, 10, 400, 200);

    auto outputValue = createTensor<CPUDevice, long double>(device, 10, 400, 200);
    auto outputGrad  = createTensor<CPUDevice, long double>(device, 10, 400, 200);

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, long double>(device);

    std::vector<Node*> inputs = {&inputVar};
    Sigmoid<long double> sigmoid(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<long double>*> inputValues = {&inputValue};

    sigmoid.forwardCPU(inputValues, &outputValue);
    sigmoid.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        long double temp = outputGrad.data()[i] * outputValue.data()[i] * (1 - outputValue.data()[i]);

        ASSERT_EQ(inputGrad.data()[i], temp);
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

    delete device;
}

#ifdef HAVE_CUDA

TEST(Sigmoid, GPU_float) {
	typedef float real;

	auto device = new GPUDevice();

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 10, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 10, 400, 200);

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Sigmoid<real> sigmoid(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	sigmoid.forwardGPU(inputTensor, &output);
	sigmoid.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device->copyToCPU(output.pointer, outputPtr, sizeof(real) * 10 * 400 * 200);
	device->copyToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		float temp = 1.f / (1.f + exp(-1.f * inputPtr[i]));

		ASSERT_TRUE(abs(temp - outputPtr[i]) <= 1e-6);
	}

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		long double temp = outputGradPtr[i] * outputPtr[i] * (1 - outputPtr[i]);

		ASSERT_EQ(inputGradPtr[i], temp);
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



#endif //DEEP8_SIGMOIDTEST_H
