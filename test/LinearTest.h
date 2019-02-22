#ifndef DEEP8_LINEARTEST_H
#define DEEP8_LINEARTEST_H

#include "nodes/Linear.h"

namespace Deep8 {

TEST(Linear, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto output = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    float a = 2.0;
    float b = 3.5;

    std::vector<Node*> inputs = {&inputVar1};
    Linear linear(inputs, a, b);

    std::vector<const Tensor*> inputTensor = {&input};

    linear.forward(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_EQ(input.data<float>()[i] * a + b, output.data<float>()[i]);
    }


}

#ifdef HAVE_CUDA

TEST(Linear, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

    auto input  = createTensor(device, inputPtr, ElementType::from<real>(), 10, {400, 200});
    auto output = createTensor(device, outputPtr,ElementType::from<real>(),  10, {400, 200});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	float a = 2.0;
    float b = 3.5;

	std::vector<Node*> inputs = { &inputVar };
	Linear linear(inputs, a, b);

	std::vector<const Tensor*> inputTensor = { &input };

	linear.forward(inputTensor, &output);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(inputPtr[i] * a + b, outputPtr[i]);
	}

	free(inputPtr);
	free(outputPtr);

}

#ifdef HAVE_HALF

TEST(Linear, half_GPU) {
	typedef half real;

	GPUDevice device;

    auto input  = createTensor(device, ElementType::from<real>(), 10, {400, 200});
    auto output = createTensor(device, ElementType::from<real>(), 10, {400, 200});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	float a = 2.0;
    float b = 3.5;

	std::vector<Node*> inputs = { &inputVar };
	Linear linear(inputs, a, b);

	std::vector<const Tensor*> inputTensor = { &input };

	linear.forward(inputTensor, &output);

}

#endif // HAVE_HALF
#endif // HAVE_CUDA


}

#endif //DEEP8_LINEARTEST_H
