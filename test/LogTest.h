#ifndef DEEP8_LOGTEST_H
#define DEEP8_LOGTEST_H

#include "Log.h"

namespace Deep8 {

TEST(Log, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor<CPUDevice, long double>(device, 10, 400, 200);
    auto output = createTensor<CPUDevice, long double>(device, 10, 400, 200);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data()[i] <= 0) {
            input.data()[i] = 1.0;
        }
    }

    auto inputVar1 = createFakeVariable<CPUDevice, long double>(device);

    std::vector<Node*> inputs = {&inputVar1};
    Log<long double> logFunc(inputs);

    std::vector<const Tensor<long double>*> inputTensor = {&input};

    logFunc.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_TRUE(std::abs(std::log(input.data()[i]) - output.data()[i]) < 1e-6);
    }

    freeTensor(device, input);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);

}

TEST(Log, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor<CPUDevice, long double>(device, 10, 400, 200);
	auto inputGrad = createTensor<CPUDevice, long double>(device, 10, 400, 200);

    auto outputValue = createTensor<CPUDevice, long double>(device, 10, 400, 200);
    auto outputGrad  = createTensor<CPUDevice, long double>(device, 10, 400, 200);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data()[i] <= 0) {
            inputValue.data()[i] = 1.0;
        }
    }

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, long double>(device);

    std::vector<Node*> inputs = {&inputVar};
    Log<long double> logFunc(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<long double>*> inputValues = {&inputValue};

    logFunc.forwardCPU(inputValues, &outputValue);
    logFunc.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        long double temp = outputGrad.data()[i] / inputValue.data()[i];

        ASSERT_TRUE(std::abs(temp - inputGrad.data()[i]) < 1e-6);
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar);

}

#ifdef HAVE_CUDA

TEST(Log, GPU_float) {
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

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		if (inputPtr[i] <= 0) {
			inputPtr[i] = 1.0;
		}
	}

	device.copyFromCPUToGPU(inputPtr, input.raw(), sizeof(real) * 10 * 400 * 200);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar };
	Log<real> logFunc(inputs);

	std::vector<const Tensor<real>*> inputValues = { &input };

	logFunc.forwardGPU(inputValues, &output);
	logFunc.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 400* 200);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_TRUE(std::abs(std::log(inputPtr[i]) - outputPtr[i]) < 1e-6);
	}

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		real temp = outputGradPtr[i] / inputPtr[i];

		ASSERT_TRUE(std::abs(temp - inputGradPtr[i]) < 1e-6);
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

TEST(Log, half_GPU) {
	typedef float real;

	GPUDevice device;

	auto input = createTensorGPU<real>(device, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 10, 400, 200);
	auto output = createTensorGPU<real>(device, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 400, 200);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar };
	Log<real> logFunc(inputs);

	std::vector<const Tensor<real>*> inputValues = { &input };

	logFunc.forwardGPU(inputValues, &output);
	logFunc.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);
}

#endif // HAVE_HALF

#endif

}

#endif //DEEP8_LOGTEST_H
