#ifndef DEEP8_L2NORMTEST_H
#define DEEP8_L2NORMTEST_H

#include "nodes/L2Norm.h"

namespace Deep8 {

TEST(L2Norm, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<double>(), 10, {200});
    auto output = createTensor(device, ElementType::from<double>(), 1, {1});

    auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar1};
    L2Norm l2Norm(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    l2Norm.forward(inputTensor, &output);

	double temp = 0;

	for (int i = 0; i < 10 * 200; ++i) {
		temp += input.data<double>()[i] * input.data<double>()[i];
	}

	ASSERT_EQ(sqrt(temp), output.data<double>()[0]);

}

TEST(L2Norm, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<float>(), 400, {200});
	auto inputGrad = createTensor(device, ElementType::from<float>(), 400, {200});

    auto outputValue = createTensor(device, ElementType::from<float>(), 1, {1});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 1, {1});

    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar};
    L2Norm l2Norm(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    l2Norm.forward(inputValues, &outputValue);
    l2Norm.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);


	for (int i = 0; i < 400 * 200; ++i) {
		auto temp = outputGrad.data<float>()[0] / outputValue.data<float>()[0];

		ASSERT_EQ(temp * inputValue.data<float>()[i], inputGrad.data<float>()[i]);
	}

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
