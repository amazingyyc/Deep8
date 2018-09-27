#ifndef DEEP8_LINEARTEST_H
#define DEEP8_LINEARTEST_H

#include "Linear.h"

namespace Deep8 {

TEST(Linear, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor<CPUDevice, float>(device, 10, 400, 200);
    auto output = createTensor<CPUDevice, float>(device, 10, 400, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    float a = 2.0;
    float b = 3.5;

    std::vector<Node*> inputs = {&inputVar1};
    Linear<float> linear(inputs, a, b);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    linear.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_EQ(input.data()[i] * a + b, output.data()[i]);
    }

    freeTensor(device, input);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);

}

#ifdef HAVE_CUDA

TEST(Linear, GPU_float) {
	typedef float real;

	auto device = new GPUDevice();

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 10, 400, 200);
	auto output = createTensorGPU<real>(device, outputPtr, 10, 400, 200);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	real a = 2.0;
	real b = 3.5;

	std::vector<Node*> inputs = { &inputVar };
	Linear<real> linear(inputs, a, b);

	std::vector<const Tensor<float>*> inputTensor = { &input };

	linear.forwardGPU(inputTensor, &output);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(inputPtr[i] * a + b, outputPtr[i]);
	}

	free(inputPtr);
	free(outputPtr);

	freeTensor(device, input);
	freeTensor(device, output);

	freeFakeVariable(inputVar);

	delete device;
}

#ifdef HAVE_HALF

TEST(Linear, half_GPU) {
	typedef half real;

	auto device = new GPUDevice();

	auto input = createTensorGPU<real>(device, 10, 400, 200);
	auto output = createTensorGPU<real>(device, 10, 400, 200);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	real a = 2.0;
	real b = 3.5;

	std::vector<Node*> inputs = { &inputVar };
	Linear<real> linear(inputs, a, b);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	linear.forwardGPU(inputTensor, &output);

	delete device;
}

#endif // HAVE_HALF
#endif // HAVE_CUDA


}

#endif //DEEP8_LINEARTEST_H
