#ifndef DEEP8_L1NORMTEST_H
#define DEEP8_L1NORMTEST_H

#include "L1Norm.h"

namespace Deep8 {

TEST(L1Norm, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, long double>(device, size_t(10), size_t(200));
    auto output = createTensor<CPUDevice, long double>(device, size_t(10), size_t(1));

    auto inputVar1 = createFakeVariable<CPUDevice, long double>(device);

    std::vector<Node*> inputs = {&inputVar1};
    L1Norm<long double> l1Norm(inputs);

    std::vector<const Tensor<long double>*> inputTensor = {&input};

    l1Norm.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10; ++i) {
        long double temp = 0;

        auto inputPtr = input.data() + i * 200;

        for (int j = 0; j < 200; ++j) {
            temp += abs(inputPtr[j]);
        }

        ASSERT_EQ(temp, output.data()[i]);
    }

    freeTensor(device, input);
    freeTensor(device, output);


	freeFakeVariable(inputVar1);

    delete device;
}

TEST(L1Norm, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, float>(device, size_t(400), size_t(200));
	auto inputGrad = createTensor<CPUDevice, float>(device, size_t(400), size_t(200));

    auto outputValue = createTensor<CPUDevice, float>(device, size_t(400), size_t(1));
    auto outputGrad  = createTensor<CPUDevice, float>(device, size_t(400), size_t(1));

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    L1Norm<float> l1Norm(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    l1Norm.forwardCPU(inputValues, &outputValue);
    l1Norm.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 400; ++i) {
        auto inputPtr = inputValue.data() + i * 200;
        auto inputGradPtr = inputGrad.data() + i * 200;

        for (int j = 0; j < 200; ++j) {
            if (inputPtr[j] > 0) {
                ASSERT_EQ(inputGradPtr[j], outputGrad.data()[i]);
            } else if (inputPtr[j] == 0) {
                ASSERT_EQ(inputGradPtr[j], 0);
            } else {
                ASSERT_EQ(inputGradPtr[j], -outputGrad.data()[i]);
            }
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

TEST(L1Norm, GPU_float) {
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
	L1Norm<real> l1norm(inputs);

    std::vector<const Tensor<real>*> inputValues = {&input};

	l1norm.forwardGPU(inputValues, &output);
	l1norm.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

    device->copyToCPU(output.pointer, outputPtr, sizeof(real) * 400);
    device->copyToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 400 * 200);

    for (int i = 0; i < 400; ++i) {
        real temp = 0;

        auto tempInputPtr = inputPtr + i * 200;

        for (int j = 0; j < 200; ++j) {
            temp += abs(tempInputPtr[j]);
        }

        ASSERT_EQ(temp, outputPtr[i]);
    }

    for (int i = 0; i < 400; ++i) {
        auto tempInputPtr = inputPtr + i * 200;
        auto tempInputGradPtr = inputGradPtr + i * 200;

        for (int j = 0; j < 200; ++j) {
            if (tempInputPtr[j] > 0) {
                ASSERT_EQ(tempInputGradPtr[j], outputGradPtr[i]);
            } else if (tempInputPtr[j] == 0) {
                ASSERT_EQ(tempInputGradPtr[j], 0);
            } else {
                ASSERT_EQ(tempInputGradPtr[j], -outputGradPtr[i]);
            }
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

#endif //DEEP8_L1NORMTEST_H
