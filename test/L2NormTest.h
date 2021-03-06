#ifndef DEEP8_L2NORMTEST_H
#define DEEP8_L2NORMTEST_H

#include "nodes/L2Norm.h"

namespace Deep8 {

TEST(L2NormLoss, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<double>(), 10, {200});
    auto output = createTensor(device, ElementType::from<double>(), 10, {1});

    auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar1};
    L2Norm l2Norm(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    l2Norm.forward(inputTensor, &output);

	for (int i = 0; i < 10; ++i) {
		double temp = 0;

		for (int j = 0; j < 200; ++j) {
			temp += input.data<double>()[i * 200 + j] * input.data<double>()[i * 200 + j];
		}

		ASSERT_EQ(sqrt(temp), output.data<double>()[i]);
	}
}

TEST(L2NormLoss, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<float>(), 400, {200});
	auto inputGrad = createTensor(device, ElementType::from<float>(), 400, {200});

    auto outputValue = createTensor(device, ElementType::from<float>(), 400, {1});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 400, {1});

    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar};
    L2Norm l2Norm(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    l2Norm.forward(inputValues, &outputValue);
    l2Norm.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

	for (int i = 0; i < 400 * 200; ++i) {
		auto temp = outputGrad.data<float>()[i / 200] / outputValue.data<float>()[i / 200];

		ASSERT_TRUE(std::abs(temp * inputValue.data<float>()[i] - inputGrad.data<float>()[i]) < 1e-6);
	}

}

#ifdef HAVE_CUDA

TEST(L2NormLoss, GPU_float) {
    typedef double real;

	GPUDevice device;

    auto inputPtr = (real*)malloc(sizeof(real) * 400 * 200);
    auto inputGradPtr = (real*)malloc(sizeof(real) * 400 * 200);

    auto outputPtr = (real*)malloc(sizeof(real) * 400 * 1);
    auto outputGradPtr = (real*)malloc(sizeof(real) * 400 * 1);


    auto input = createTensor(device, inputPtr, ElementType::from<real>(), 400, { 200 });
    auto inputGrad = createTensor(device, inputGradPtr, ElementType::from<real>(), 400, { 200 });
    auto output = createTensor(device, outputPtr, ElementType::from<real>(), 400, { 1 });
    auto outputGrad = createTensor(device, outputGradPtr, ElementType::from<real>(), 400, { 1 });

    /**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

    zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = {&inputVar};
    L2Norm l2norm(inputs);

    std::vector<const Tensor*> inputValues = {&input};

	l2norm.forward(inputValues, &output);
	l2norm.backward(inputValues, &output, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 1);
    device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 400 * 200);


	for (int i = 0; i < 400; ++i) {
		real temp = 0;

		for (int j = 0; j < 200; ++j) {
			temp += inputPtr[i] * inputPtr[i];
		}

		ASSERT_EQ(sqrt(temp), outputPtr[i]);
	}

	for (int i = 0; i < 400 * 200; ++i) {
		auto temp = outputGradPtr[i / 200] / outputPtr[i / 200];

		ASSERT_TRUE(std::abs(temp * inputPtr[i] - inputGradPtr[i]) < 1e-6);
	}


    free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

}

#ifdef HAVE_HALF

TEST(L2NormLoss, half_GPU) {
	typedef half real;

	GPUDevice device;

    auto input = createTensor(device, ElementType::from<real>(), 400, { 200 });
    auto inputGrad = createTensor(device, ElementType::from<real>(), 400, { 200 });
    auto output = createTensor(device, ElementType::from<real>(), 400, { 1 });
    auto outputGrad = createTensor(device, ElementType::from<real>(), 400, { 1 });

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar };
    L2NormLoss l2norm(inputs);

	std::vector<const Tensor*> inputValues = { &input };

	l2norm.forward(inputValues, &output);
	l2norm.backward(inputValues, &output, &outputGrad, 0, &inputGrad);

}

#endif // HAVE_HALF

#endif

}

#endif //DEEP8_L2NORMTEST_H
