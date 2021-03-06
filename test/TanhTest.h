#ifndef DEEP8_TANHTEST_H
#define DEEP8_TANHTEST_H

#include "nodes/Tanh.h"

namespace Deep8 {

TEST(Tanh, forwardCPU) {
	CPUDevice device;

	auto input  = createTensor(device, ElementType::from<float>(), 10, {400, 200});
	auto output = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1};
    Tanh tanH(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    tanH.forward(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_EQ(std::tanh(input.data<float>()[i]), output.data<float>()[i]);
    }


}

TEST(Tanh, backwardCPU) {
	CPUDevice device;

	auto inputValue  = createTensor(device, ElementType::from<float>(), 10, {400, 200});
	auto inputGrad   = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto outputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar};
    Tanh tt(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    tt.forward(inputValues, &outputValue);
    tt.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        auto temp = outputGrad.data<float>()[i] * (1.0 - std::tanh(inputValue.data<float>()[i]) * std::tanh(inputValue.data<float>()[i]));

        ASSERT_TRUE(std::abs(temp - inputGrad.data<float>()[i]) <=  1e-6);
    }


}



#ifdef HAVE_CUDA

TEST(Tanh, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

    auto input      = createTensor(device, inputPtr,     ElementType::from<real>(), 10, {400, 200});
    auto inputGrad  = createTensor(device, inputGradPtr, ElementType::from<real>(), 10, {400, 200});
    auto output     = createTensor(device, outputPtr,    ElementType::from<real>(),  10,{ 400, 200});
    auto outputGrad = createTensor(device, outputGradPtr,ElementType::from<real>(),  10,{ 400, 200});

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar1 };
	Tanh tanhFunc(inputs);

	std::vector<const Tensor*> inputTensor = { &input };

	tanhFunc.forward(inputTensor, &output);
	tanhFunc.backward(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 400 * 200);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(std::tanh(inputPtr[i]), outputPtr[i]);
	}

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		auto temp = outputGradPtr[i] * (1.0 - std::tanh(inputPtr[i]) * std::tanh(inputPtr[i]));

		ASSERT_TRUE(std::abs(temp - inputGradPtr[i]) <= 1e-5);
	}

	free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

}

#ifdef HAVE_HALF

TEST(Tanh, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input      = createTensor(device, ElementType::from<real>(), 10, {400, 200});
	auto inputGrad  = createTensor(device, ElementType::from<real>(), 10, {400, 200});
	auto output     = createTensor(device, ElementType::from<real>(), 10, {400, 200});
	auto outputGrad = createTensor(device, ElementType::from<real>(), 10, {400, 200});

	auto inputVar1 = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar1 };
	Tanh tanhFunc(inputs);

	std::vector<const Tensor*> inputTensor = { &input };

	tanhFunc.forward(inputTensor, &output);
	tanhFunc.backward(inputTensor, &output, &outputGrad, 0, &inputGrad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_TANHTEST_H
