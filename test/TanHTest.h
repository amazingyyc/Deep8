#ifndef DEEP8_TANHTEST_H
#define DEEP8_TANHTEST_H

#include "TanH.h"

namespace Deep8 {

TEST(TanH, forwardCPU) {
	CPUDevice device;

	auto input = createTensor<CPUDevice, float>(device, 10, 400, 200);
	auto output = createTensor<CPUDevice, float>(device, 10, 400, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    TanH<float> tanH(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    tanH.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_EQ(tanh(input.data()[i]), output.data()[i]);
    }

    freeTensor(device, input);
    freeTensor(device, output);

	freeFakeVariable(inputVar1);

}

TEST(TanH, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor<CPUDevice, float>(device, 10, 400, 200);
	auto inputGrad = createTensor<CPUDevice, float>(device, 10, 400, 200);

    auto outputValue = createTensor<CPUDevice, float>(device, 10, 400, 200);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 10, 400, 200);

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    TanH<float> tt(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    tt.forwardCPU(inputValues, &outputValue);
    tt.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        auto temp = outputGrad.data()[i] * (1.0 - tanh(inputValue.data()[i]) * tanh(inputValue.data()[i]));

        ASSERT_TRUE(std::abs(temp - inputGrad.data()[i]) <=  1e-6);
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

}



#ifdef HAVE_CUDA

TEST(TanH, GPU_float) {
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
	TanH<real> tanhFunc(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	tanhFunc.forwardGPU(inputTensor, &output);
	tanhFunc.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * 10 * 400 * 200);
	device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(tanh(inputPtr[i]), outputPtr[i]);
	}

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		auto temp = outputGradPtr[i] * (1.0 - tanh(inputPtr[i]) * tanh(inputPtr[i]));

		ASSERT_TRUE(std::abs(temp - inputGradPtr[i]) <= 1e-6);
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

TEST(TanH, half_GPU) {
	typedef half real;

	auto device = new GPUDevice();

	auto input = createTensorGPU<real>(device, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto output = createTensorGPU<real>(device, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	TanH<real> tanhFunc(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	tanhFunc.forwardGPU(inputTensor, &output);
	tanhFunc.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	delete device;
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_TANHTEST_H
