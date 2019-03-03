#ifndef DEEP8_LRELUTEST_H
#define DEEP8_LRELUTEST_H

#include "nodes/LReLu.h"

namespace Deep8 {

TEST(LReLu, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto output = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    float a = 2.3;

    std::vector<Node*> inputs = {&inputVar1};
    LReLu lrelu(inputs, a);

    std::vector<const Tensor*> inputTensor = {&input};

    lrelu.forward(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data<float>()[i] < 0) {
            ASSERT_EQ(input.data<float>()[i] * a, output.data<float>()[i]);
        } else {
            ASSERT_EQ(input.data<float>()[i], output.data<float>()[i]);
        }
    }

}

TEST(LReLu, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
	auto inputGrad  = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto outputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    float a = 2.3;

    std::vector<Node*> inputs = {&inputVar};
    LReLu lreLu(inputs, a);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    lreLu.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data<float>()[i] > 0) {
            ASSERT_EQ(inputGrad.data<float>()[i], outputGrad.data<float>()[i]);
        } else {
            ASSERT_EQ(inputGrad.data<float>()[i], outputGrad.data<float>()[i] * a);
        }
    }
}

#ifdef HAVE_CUDA

TEST(LReLu, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (float*)malloc(sizeof(float) * 400 * 200);
	auto inputGradPtr = (float*)malloc(sizeof(float) * 400 * 200);

	auto outputPtr = (float*)malloc(sizeof(float) * 400 * 200);
	auto outputGradPtr = (float*)malloc(sizeof(float) * 400 * 200);

    auto input      = createTensor(device, inputPtr,      ElementType::from<real>(), 400, {200});
    auto inputGrad  = createTensor(device, inputGradPtr,  ElementType::from<real>(), 400, {200});
    auto output     = createTensor(device, outputPtr,     ElementType::from<real>(), 400, {200});
    auto outputGrad = createTensor(device, outputGradPtr, ElementType::from<real>(), 400, {200});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	zeroTensor(device, inputGrad);

	real a = 2.3;

	std::vector<Node*> inputs = { &inputVar };
	LReLu lrelu(inputs, a);

	std::vector<const Tensor*> inputValues = { &input };

	lrelu.forward(inputValues, &output);
	lrelu.backward(inputValues, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) *  400 * 200);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 400 * 200);

	for (int i = 0; i < 400 * 200; ++i) {
		if (inputPtr[i] < 0) {
			ASSERT_EQ(inputPtr[i] * a, outputPtr[i]);
		} else {
			ASSERT_EQ(inputPtr[i], outputPtr[i]);
		}
	}

	for (int i = 0; i < 400 * 200; ++i) {
		if (inputPtr[i] >= 0) {
			ASSERT_EQ(inputGradPtr[i], outputGradPtr[i]);
		} else {
			ASSERT_EQ(inputGradPtr[i], outputGradPtr[i] * a);
		}
	}

	free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);
}

#ifdef HAVE_HALF

TEST(LReLu, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input      = createTensor(device, ElementType::from<real>(), 400, {200});
	auto inputGrad  = createTensor(device, ElementType::from<real>(), 400, {200});
	auto output     = createTensor(device, ElementType::from<real>(), 400, {200});
	auto outputGrad = createTensor(device, ElementType::from<real>(), 400, {200});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	zeroTensor(device, inputGrad);

	float a = 2.3;

	std::vector<Node*> inputs = { &inputVar };
	LReLu lrelu(inputs, a);

	std::vector<const Tensor*> inputValues = { &input };

	lrelu.forward(inputValues, &output);
	lrelu.backward(inputValues, &output, &outputGrad, 0, &inputGrad);
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_LRELUTEST_H
