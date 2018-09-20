#ifndef DEEP8_SUMELEMENTSTEST_H
#define DEEP8_SUMELEMENTSTEST_H

#include "SumElements.h"

namespace Deep8 {

TEST(SumElements, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, 400, 200);
    auto output = createTensor<CPUDevice, float>(device, 400, 1);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    SumElements<float> sumElements(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    sumElements.forwardCPU(inputTensor, &output);

    float sum = 0;

    for (int i = 0; i < 400; ++i) {
        auto inputPtr = input.data() + i * 200;
        float sum = 0;

        for (int k = 0; k < 200; ++k) {
            sum += inputPtr[k];
        }

        ASSERT_EQ(sum, output.data()[i]);

    }

    freeTensor(device, input);
    freeTensor(device, output);

	freeFakeVariable(inputVar1);

    delete device;
}

TEST(SumElements, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, float>(device, 400, 200);
	auto inputGrad = createTensor<CPUDevice, float>(device, 400, 200);

    auto outputValue = createTensor<CPUDevice, float>(device, 400, 1);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 400, 1);

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    SumElements<float> sumElements(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    sumElements.forwardCPU(inputValues, &outputValue);
    sumElements.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 400; ++i) {
        auto inputGradPtr = inputGrad.data() + i * 200;

        for (int j = 0; j < 200; ++j) {
            ASSERT_EQ(inputGradPtr[j], outputGrad.data()[i]);
        }
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

    delete device;
}

#ifdef HAVE_CUDA

TEST(SumElements, GPU_float) {
	typedef float real;

	auto device = new GPUDevice();

	auto inputPtr = (real*)malloc(sizeof(real) * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 400 * 1);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 400 * 1);

	auto input = createTensorGPU<real>(device, inputPtr, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 400, 1);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 400, 1);

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	SumElements<real> sumElements(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	sumElements.forwardGPU(inputTensor, &output);
	sumElements.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) * 400 * 1);
	device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 400 * 200);


	for (int i = 0; i < 400; ++i) {
		auto tempinputPtr = inputPtr + i * 200;
		float sum = 0;

		for (int k = 0; k < 200; ++k) {
			sum += tempinputPtr[k];
		}

		ASSERT_EQ(sum, outputPtr[i]);
	}

	for (int i = 0; i < 400; ++i) {
		auto tempinputGradPtr = inputGradPtr + i * 200;

		for (int j = 0; j < 200; ++j) {
			ASSERT_EQ(tempinputGradPtr[j], outputGradPtr[i]);
		}
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

TEST(SumElements, half_GPU) {
	typedef half real;

	auto device = new GPUDevice();

	auto input = createTensorGPU<real>(device, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 400, 200);

	auto output = createTensorGPU<real>(device, 400, 1);
	auto outputGrad = createTensorGPU<real>(device, 400, 1);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	SumElements<real> sumElements(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	sumElements.forwardGPU(inputTensor, &output);
	sumElements.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	delete device;
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_SUMELEMENTSTEST_H
