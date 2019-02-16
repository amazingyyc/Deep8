#ifndef DEEP8_SIGMOIDTEST_H
#define DEEP8_SIGMOIDTEST_H

#include "nodes/Sigmoid.h"

namespace Deep8 {

TEST(Sigmoid, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto output = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1};
    Sigmoid sigmoid(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    sigmoid.forward(inputTensor,&output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        float temp = 1.f / (1.f + std::exp(-1.f * input.data<float>()[i]));

        ASSERT_TRUE(std::abs(temp - output.data<float>()[i]) <= 1e-6);
    }


}

TEST(Sigmoid, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
	auto inputGrad = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto outputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar};
    Sigmoid sigmoid(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    sigmoid.forward(inputValues, &outputValue);
    sigmoid.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        double temp = outputGrad.data<float>()[i] * outputValue.data<float>()[i] * (1 - outputValue.data<float>()[i]);

        ASSERT_EQ(inputGrad.data<float>()[i], temp);
    }


}

#ifdef HAVE_CUDA

TEST(Sigmoid, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 10, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 10, 400, 200);

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Sigmoid<real> sigmoid(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	sigmoid.forwardGPU(inputTensor, &output);
	sigmoid.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 400 * 200);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		float temp = 1.f / (1.f + std::exp(-1.f * inputPtr[i]));

		ASSERT_TRUE(std::abs(temp - outputPtr[i]) <= 1e-4);
	}

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		real temp = outputGradPtr[i] * outputPtr[i] * (1 - outputPtr[i]);

		ASSERT_EQ(inputGradPtr[i], temp);
	}

	free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input);
	freeTensor(device, inputGrad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);
}

#ifdef HAVE_HALF

TEST(LReLU, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input = createTensorGPU<real>(device, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto output = createTensorGPU<real>(device, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Sigmoid<real> sigmoid(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	sigmoid.forwardGPU(inputTensor, &output);
	sigmoid.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);
}

#endif // HAVE_HALF
#endif

}



#endif //DEEP8_SIGMOIDTEST_H
