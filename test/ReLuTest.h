#ifndef DEEP8_RELUTEST_H
#define DEEP8_RELUTEST_H

#include "ReLu.h"

namespace Deep8 {

TEST(ReLu, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor<CPUDevice, float>(device, 10, 400, 200);
    auto output = createTensor<CPUDevice, float>(device, 10, 400, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    ReLu<float> relu(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    relu.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data()[i] < 0) {
            ASSERT_EQ(0, output.data()[i]);
        } else {
            ASSERT_EQ(input.data()[i], output.data()[i]);
        }
    }

    freeTensor(device, input);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);

}

TEST(ReLu, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor<CPUDevice, float>(device, 10, 400, 200);
	auto inputGrad = createTensor<CPUDevice, float>(device, 10, 400, 200);

	auto outputValue = createTensor<CPUDevice, float>(device, 10, 400, 200);
	auto outputGrad = createTensor<CPUDevice, float>(device, 10, 400, 200);

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    ReLu<float> reLu(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    reLu.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data()[i] > 0) {
            ASSERT_EQ(inputGrad.data()[i], outputGrad.data()[i]);
        } else {
            ASSERT_EQ(inputGrad.data()[i], 0);
        }
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar);

}

#ifdef HAVE_CUDA

TEST(ReLu, GPU_float) {
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

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	ReLu<real> relu(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	relu.forwardGPU(inputTensor, &output);
	relu.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * 10 * 400 * 200);
	device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		if (inputPtr[i] < 0) {
			ASSERT_EQ(0, outputPtr[i]);
		} else {
			ASSERT_EQ(inputPtr[i], outputPtr[i]);
		}
	}

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		if (inputPtr[i] > 0) {
			ASSERT_EQ(inputGradPtr[i], outputGradPtr[i]);
		} else {
			ASSERT_EQ(inputGradPtr[i], 0);
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

	delete device;
}

#ifdef HAVE_HALF

TEST(ReLU, half_GPU) {
	typedef half real;

	auto device = new GPUDevice();

	auto input = createTensorGPU<real>(device, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto output = createTensorGPU<real>(device, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	ReLu<real> relu(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	relu.forwardGPU(inputTensor, &output);
	relu.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	delete device;
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_RELUTEST_H
