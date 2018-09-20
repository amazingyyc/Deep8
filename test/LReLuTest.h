#ifndef DEEP8_LRELUTEST_H
#define DEEP8_LRELUTEST_H

#include "LReLu.h"

namespace Deep8 {

TEST(LReLu, forwardCPU) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, 10, 400, 200);
    auto output = createTensor<CPUDevice, float>(device, 10, 400, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    float a = 2.3;

    std::vector<Node*> inputs = {&inputVar1};
    LReLu<float> lrelu(inputs, a);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    lrelu.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data()[i] < 0) {
            ASSERT_EQ(input.data()[i] * a, output.data()[i]);
        } else {
            ASSERT_EQ(input.data()[i], output.data()[i]);
        }
    }

    freeTensor(device, input);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);

    delete device;
}

TEST(LReLu, backwardCPU) {
    auto device = new CPUDevice();

	auto inputValue = createTensor<CPUDevice, float>(device, 10, 400, 200);
	auto inputGrad = createTensor<CPUDevice, float>(device, 10, 400, 200);

    auto outputValue = createTensor<CPUDevice, float>(device, 10, 400, 200);
    auto outputGrad  = createTensor<CPUDevice, float>(device, 10, 400, 200);

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    float a = 2.3;

    std::vector<Node*> inputs = {&inputVar};
    LReLu<float> lreLu(inputs, a);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    lreLu.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data()[i] > 0) {
            ASSERT_EQ(inputGrad.data()[i], outputGrad.data()[i]);
        } else {
            ASSERT_EQ(inputGrad.data()[i], outputGrad.data()[i] * a);
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

TEST(LReLu, GPU_float) {
	typedef float real;

	auto device = new GPUDevice();

	auto inputPtr = (float*)malloc(sizeof(float) * 400 * 200);
	auto inputGradPtr = (float*)malloc(sizeof(float) * 400 * 200);

	auto outputPtr = (float*)malloc(sizeof(float) * 400 * 200);
	auto outputGradPtr = (float*)malloc(sizeof(float) * 400 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 400, 200);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	zeroTensor(device, inputGrad);

	real a = 2.3;

	std::vector<Node*> inputs = { &inputVar };
	LReLu<real> lrelu(inputs, a);

	std::vector<const Tensor<real>*> inputValues = { &input };

	lrelu.forwardGPU(inputValues, &output);
	lrelu.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

	device->copyFromGPUToCPU(output.pointer, outputPtr, sizeof(real) *  400 * 200);
	device->copyFromGPUToCPU(inputGrad.pointer, inputGradPtr, sizeof(real) * 400 * 200);

	for (int i = 0; i < 400 * 200; ++i) {
		if (inputPtr[i] < 0) {
			ASSERT_EQ(inputPtr[i] * a, outputPtr[i]);
		} else {
			ASSERT_EQ(inputPtr[i], outputPtr[i]);
		}
	}

	for (int i = 0; i < 400 * 200; ++i) {
		if (inputPtr[i] > 0) {
			ASSERT_EQ(inputGradPtr[i], outputGradPtr[i]);
		} else {
			ASSERT_EQ(inputGradPtr[i], outputGradPtr[i] * a);
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

	freeFakeVariable(inputVar);

	delete device;
}

#ifdef HAVE_HALF

TEST(LReLu, half_GPU) {
	typedef half real;

	auto device = new GPUDevice();

	auto input = createTensorGPU<real>(device, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 400, 200);

	auto output = createTensorGPU<real>(device, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, 400, 200);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	zeroTensor(device, inputGrad);

	real a = 2.3;

	std::vector<Node*> inputs = { &inputVar };
	LReLu<real> lrelu(inputs, a);

	std::vector<const Tensor<real>*> inputValues = { &input };

	lrelu.forwardGPU(inputValues, &output);
	lrelu.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

	delete device;
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_LRELUTEST_H
