#ifndef DEEP8_L2NORMTEST_H
#define DEEP8_L2NORMTEST_H

#include "L2Norm.h"

namespace Deep8 {

TEST(L2Norm, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, long double>(device, 10, 200);
    auto output = createTensor<CPUDevice, long double>(device, 10, 1);

    auto inputVar1 = createFakeVariable<CPUDevice, long double>(device);

    std::vector<Node*> inputs = {&inputVar1};
    L2Norm<long double> l2Norm(inputs);

    std::vector<const Tensor<long double>*> inputTensor = {&input};

    l2Norm.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10; ++i) {
        long double temp = 0;

        auto inputPtr = input.data() + i * 200;

        for (int j = 0; j < 200; ++j) {
            temp += inputPtr[j] * inputPtr[j];
        }

        ASSERT_EQ(sqrt(temp), output.data()[i]);
    }

    freeTensor(device, input);
    freeTensor(device, output);

	freeFakeVariable(inputVar1);

    delete device;
}

TEST(L2Norm, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, float>(device, 400, 200);
	auto inputGrad = createTensor<CPUDevice, float>(device, 400, 200);

    auto outputValue = createTensor<CPUDevice, float>(device, 400, 1);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 400, 1);

    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    L2Norm<float> l2Norm(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    l2Norm.forwardCPU(inputValues, &outputValue);
    l2Norm.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 400; ++i) {
        auto inputPtr = inputValue.data() + i * 200;
        auto inputGradPtr = inputGrad.data() + i * 200;

        auto temp = outputGrad.data()[i] / outputValue.data()[i];

        for (int j = 0; j < 200; ++j) {
            ASSERT_EQ(temp * inputPtr[j], inputGradPtr[j]);
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

TEST(L2Norm, GPU_float) {
    typedef float real;

    auto device = new GPUDevice();

    auto inputPtr = (float*)malloc(sizeof(float) * 400 * 200);
    auto inputGradPtr = (float*)malloc(sizeof(float) * 400 * 200);

    auto outputPtr = (float*)malloc(sizeof(float) * 400 * 1);
    auto outputGradPtr = (float*)malloc(sizeof(float) * 400 * 1);

    auto input = createTensorGPU<real>(device, inputPtr, 400, 200);
	auto inputGrad  = createTensorGPU<real>(device, inputGradPtr, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 400, 1);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 400, 1);

    /**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

    zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = {&inputVar};
	L2Norm<real> l2norm(inputs);

    std::vector<const Tensor<real>*> inputValues = {&input};

	l2norm.forwardGPU(inputValues, &output);
	l2norm.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

    device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * 400);
    device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 400 * 200);

    for (int i = 0; i < 10; ++i) {
        real temp = 0;

        auto tempInputPtr = inputPtr + i * 200;

        for (int j = 0; j < 200; ++j) {
            temp += tempInputPtr[j] * tempInputPtr[j];
        }

        ASSERT_EQ(sqrt(temp), outputPtr[i]);
    }

    for (int i = 0; i < 400; ++i) {
        auto tempInputPtr = inputPtr + i * 200;
        auto tempInputGradPtr = inputGradPtr + i * 200;

        auto temp = outputGradPtr[i] / outputPtr[i];

        for (int j = 0; j < 200; ++j) {
            ASSERT_TRUE(abs(temp * tempInputPtr[j] - tempInputGradPtr[j]) < 1e-6);
        }
    }

    free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input);
	freeTensor(device, inputGrad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

	delete device;
}

#endif

}

#endif //DEEP8_L2NORMTEST_H
