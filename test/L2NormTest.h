#ifndef DEEP8_L2NORMTEST_H
#define DEEP8_L2NORMTEST_H

#include "L2Norm.h"

namespace Deep8 {

TEST(L2Norm, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor<CPUDevice, double>(device, 10, 200);
    auto output = createTensor<CPUDevice, double>(device, 1, 1);

    auto inputVar1 = createFakeVariable<CPUDevice, double>(device);

    std::vector<Node*> inputs = {&inputVar1};
    L2Norm<double> l2Norm(inputs);

    std::vector<const Tensor<double>*> inputTensor = {&input};

    l2Norm.forwardCPU(inputTensor, &output);

	double temp = 0;

	for (int i = 0; i < 10 * 200; ++i) {
		temp += input.data()[i] * input.data()[i];
	}

	ASSERT_EQ(sqrt(temp), output.data()[0]);

    freeTensor(device, input);
    freeTensor(device, output);

	freeFakeVariable(inputVar1);

}

TEST(L2Norm, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor<CPUDevice, float>(device, 400, 200);
	auto inputGrad = createTensor<CPUDevice, float>(device, 400, 200);

    auto outputValue = createTensor<CPUDevice, float>(device, 1, 1);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 1, 1);

    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    L2Norm<float> l2Norm(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    l2Norm.forwardCPU(inputValues, &outputValue);
    l2Norm.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);


	for (int i = 0; i < 400 * 200; ++i) {
		auto temp = outputGrad.data()[0] / outputValue.data()[0];

		ASSERT_EQ(temp * inputValue.data()[i], inputGrad.data()[i]);
	}

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

}

#ifdef HAVE_CUDA

TEST(L2Norm, GPU_float) {
    typedef float real;

	GPUDevice device;

    auto inputPtr = (float*)malloc(sizeof(float) * 400 * 200);
    auto inputGradPtr = (float*)malloc(sizeof(float) * 400 * 200);

    auto outputPtr = (float*)malloc(sizeof(float) * 1 * 1);
    auto outputGradPtr = (float*)malloc(sizeof(float) * 1 * 1);

    auto input = createTensorGPU<real>(device, inputPtr, 400, 200);
	auto inputGrad  = createTensorGPU<real>(device, inputGradPtr, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 1, 1);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 1, 1);

    /**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

    zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = {&inputVar};
	L2Norm<real> l2norm(inputs);

    std::vector<const Tensor<real>*> inputValues = {&input};

	l2norm.forwardGPU(inputValues, &output);
	l2norm.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 1);
    device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 400 * 200);

	real temp = 0;
	for (int i = 0; i < 400 * 200; ++i) {
		temp += inputPtr[i] * inputPtr[i];
	}

	ASSERT_EQ(sqrt(temp), outputPtr[0]);

	for (int i = 0; i < 400 * 200; ++i) {
		auto temp = outputGradPtr[0] / outputPtr[0];

		ASSERT_TRUE(std::abs(temp * inputPtr[i] - inputGradPtr[i]) < 1e-6);
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

}

#ifdef HAVE_HALF

TEST(L2Norm, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input = createTensorGPU<real>(device, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 400, 200);

	auto output = createTensorGPU<real>(device, 1, 1);
	auto outputGrad = createTensorGPU<real>(device, 1, 1);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar };
	L2Norm<real> l2norm(inputs);

	std::vector<const Tensor<real>*> inputValues = { &input };

	l2norm.forwardGPU(inputValues, &output);
	l2norm.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

}

#endif // HAVE_HALF

#endif

}

#endif //DEEP8_L2NORMTEST_H
