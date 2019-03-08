#ifndef DEEP8_L1NORMTEST_H
#define DEEP8_L1NORMTEST_H

#include "nodes/L1Norm.h"

namespace Deep8 {

TEST(L1Norm, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(200)});
    auto output = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(1)});

    auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar1};
    L1Norm l1Norm(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    l1Norm.forward(inputTensor, &output);

	for (int i = 0; i < 10; ++i) {
		long double temp = 0;

		for (int j = 0; j < 200; ++j) {
			temp += std::abs(input.data<double>()[i * 200 + j]);
		}

		ASSERT_TRUE(std::abs(temp - output.data<double>()[i]) < 1e-6);
	}
}

TEST(L1Norm, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<float>(), size_t(400), {size_t(200)});
	auto inputGrad = createTensor(device, ElementType::from<float>(), size_t(400), {size_t(200)});

    auto outputValue = createTensor(device, ElementType::from<float>(), size_t(400), {size_t(1)});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), size_t(400), {size_t(1)});

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar};
    L1Norm l1Norm(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    l1Norm.forward(inputValues, &outputValue);
    l1Norm.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

	for (int i = 0; i < 400 * 200; ++i) {
		if (inputValue.data<float>()[i] >= 0) {
			ASSERT_EQ(inputGrad.data<float>()[i], outputGrad.data<float>()[i / 200]);
		} else {
			ASSERT_EQ(inputGrad.data<float>()[i], -outputGrad.data<float>()[i / 200]);
		}
	}
}


#ifdef HAVE_CUDA

TEST(L1Norm, GPU_float) {
    typedef float real;

	GPUDevice device;

    auto inputPtr = (float*)malloc(sizeof(float) * 400 * 200);
    auto inputGradPtr = (float*)malloc(sizeof(float) * 400 * 200);

    auto outputPtr = (float*)malloc(sizeof(float) * 400 * 1);
    auto outputGradPtr = (float*)malloc(sizeof(float) * 400 * 1);

    auto input      = createTensor(device, inputPtr,     ElementType::from<real>(), 400, {200});
    auto inputGrad  = createTensor(device, inputGradPtr, ElementType::from<real>(), 400, {200});
    auto output     = createTensor(device, outputPtr,    ElementType::from<real>(), 400, {1});
    auto outputGrad = createTensor(device, outputGradPtr,ElementType::from<real>(), 400, {1});

    /**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

    zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = {&inputVar};
    L1Norm l1norm(inputs);

    std::vector<const Tensor*> inputValues = {&input};

	l1norm.forward(inputValues, &output);
	l1norm.backward(inputValues, &output, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 1);
    device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 400 * 200);

	

	for (int i = 0; i < 400; ++i) {
		real temp = 0;

		for (int j = 0; j < 200; ++j) {
			temp += std::abs(inputPtr[i * 200 + j]);
		}

		ASSERT_EQ(temp, outputPtr[i]);
	}

	for (int i = 0; i < 400 * 200; ++i) {
		if (inputPtr[i] >= 0) {
			ASSERT_EQ(inputGradPtr[i], outputGradPtr[i / 200]);
		} else {
			ASSERT_EQ(inputGradPtr[i], -outputGradPtr[i / 200]);
		}
	}

    free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);


}

#ifdef HAVE_HALF

TEST(L1Norm, half_GPU) {
	typedef half real;

	GPUDevice device;

    auto input      = createTensor(device, ElementType::from<real>(), 400, {200});
    auto inputGrad  = createTensor(device, ElementType::from<real>(), 400, {200});
    auto output     = createTensor(device, ElementType::from<real>(), 400, {1});
    auto outputGrad = createTensor(device, ElementType::from<real>(), 400, {1});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar };
    L1NormLoss l1norm(inputs);

	std::vector<const Tensor*> inputValues = { &input };

	l1norm.forward(inputValues, &output);
	l1norm.backward(inputValues, &output, &outputGrad, 0, &inputGrad);
}

#endif // HAVE_HALF
#endif
}

#endif //DEEP8_L1NORMTEST_H
