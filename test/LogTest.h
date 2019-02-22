#ifndef DEEP8_LOGTEST_H
#define DEEP8_LOGTEST_H

#include "nodes/Log.h"

namespace Deep8 {

TEST(Log, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<double>(), 10, {400, 200});
    auto output = createTensor(device, ElementType::from<double>(), 10, {400, 200});

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data<double>()[i] <= 0) {
            input.data<double>()[i] = 1.0;
        }
    }

    auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar1};
    Log logFunc(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    logFunc.forward(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_TRUE(std::abs(std::log(input.data<double>()[i]) - output.data<double>()[i]) < 1e-6);
    }

}

TEST(Log, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<double>(), 10, {400, 200});
	auto inputGrad  = createTensor(device, ElementType::from<double>(), 10, {400, 200});

    auto outputValue = createTensor(device, ElementType::from<double>(), 10, {400, 200});
    auto outputGrad  = createTensor(device, ElementType::from<double>(), 10, {400, 200});

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data<double>()[i] <= 0) {
            inputValue.data<double>()[i] = 1.0;
        }
    }

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar};
    Log logFunc(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    logFunc.forward(inputValues, &outputValue);
    logFunc.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        double temp = outputGrad.data<double>()[i] / inputValue.data<double>()[i];

        ASSERT_TRUE(std::abs(temp - inputGrad.data<double>()[i]) < 1e-6);
    }

}

#ifdef HAVE_CUDA

TEST(Log, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

    auto input      = createTensor(device, inputPtr,     ElementType::from<real>(), 10, { 400, 200});
    auto inputGrad  = createTensor(device, inputGradPtr, ElementType::from<real>(), 10, {400, 200});
    auto output     = createTensor(device, outputPtr,    ElementType::from<real>(),  10,{ 400, 200});
    auto outputGrad = createTensor(device, outputGradPtr,ElementType::from<real>(),  10,{ 400, 200});

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		if (inputPtr[i] <= 0) {
			inputPtr[i] = 1.0;
		}
	}

	device.copyFromCPUToGPU(inputPtr, input.raw(), sizeof(real) * 10 * 400 * 200);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar };
	Log logFunc(inputs);

	std::vector<const Tensor*> inputValues = { &input };

	logFunc.forward(inputValues, &output);
	logFunc.backward(inputValues, &output, &outputGrad, 0, &inputGrad);

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
}


#ifdef HAVE_HALF

TEST(Log, half_GPU) {
	typedef float real;

	GPUDevice device;

	auto input      = createTensor(device, ElementType::from<real>(), 10, {400, 200});
	auto inputGrad  = createTensor(device, ElementType::from<real>(), 10, {400, 200});
	auto output     = createTensor(device, ElementType::from<real>(), 10, {400, 200});
	auto outputGrad = createTensor(device, ElementType::from<real>(), 10, {400, 200});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar };
	Log logFunc(inputs);

	std::vector<const Tensor*> inputValues = { &input };

	logFunc.forward(inputValues, &output);
	logFunc.backward(inputValues, &output, &outputGrad, 0, &inputGrad);
}

#endif // HAVE_HALF

#endif

}

#endif //DEEP8_LOGTEST_H
