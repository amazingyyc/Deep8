#ifndef DEEP8_EXPTEST_H
#define DEEP8_EXPTEST_H

#include <cmath>
#include "Exp.h"

namespace Deep8 {

TEST(Exp, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));
    auto output = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data()[i] > 10) {
            input.data()[i] = 10;
        }
    }

    auto inputVar1 = createFakeVariable<CPUDevice, double>(device);

    std::vector<Node*> inputs = {&inputVar1};
    Exp<double> expFunc(inputs);

    std::vector<const Tensor<double>*> inputTensor = {&input};

    expFunc.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_TRUE(std::abs(std::exp(input.data()[i]) - output.data()[i]) < 1e-6);
    }

    freeTensor(device, input);
    freeTensor(device, output);

    freeFakeVariable(inputVar1);
}

TEST(Exp, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));
	auto inputGrad = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));

    auto outputValue = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));
    auto outputGrad  = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data()[i] > 10) {
            inputValue.data()[i] = 10;
        }
    }

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, double>(device);

    std::vector<Node*> inputs = {&inputVar};
    Exp<double> expFunc(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<double>*> inputValues = {&inputValue};

    expFunc.forwardCPU(inputValues, &outputValue);
    expFunc.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        double temp = std::exp(inputValue.data()[i]) * outputGrad.data()[i];

        ASSERT_TRUE(std::abs(temp - inputGrad.data()[i]) < 1e-6);
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar);

}

#ifdef HAVE_CUDA

TEST(Exp, GPU_double) {
    typedef double real;

	GPUDevice device;

    int dim0 = 10, dim1 = 400, dim2 = 200;

    auto inputPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
    auto inputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto outputPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
    auto outputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto input = createTensorGPU<real>(device, inputPtr, dim0, dim1, dim2);
	auto inputGrad  = createTensorGPU<real>(device, inputGradPtr, dim0, dim1, dim2);

	auto output = createTensorGPU<real>(device, outputPtr, dim0, dim1, dim2);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, dim0, dim1, dim2);

    /**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

    zeroTensor(device, inputGrad);

    std::vector<Node*> inputs = {&inputVar};
    Exp<real> expFunc(inputs);

    std::vector<const Tensor<real>*> inputValues = {&input};

	expFunc.forwardGPU(inputValues, &output);
	expFunc.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * dim0 * dim1 * dim2);
    device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * dim0 * dim1 * dim2);

    for (int i = 0; i < dim0 * dim1 * dim2; ++i) {
        ASSERT_TRUE(std::abs(std::exp(inputPtr[i]) - outputPtr[i]) < 1e-4);
    }

    for (int i = 0; i < 10 * 400 * 200; ++i) {
		real temp = std::exp(inputPtr[i]) * outputGradPtr[i];

		ASSERT_TRUE(std::abs(temp - inputGradPtr[i]) < 1e-4);
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

TEST(Exp, half_GPU) {
	typedef half real;

	GPUDevice device;

	int dim0 = 10, dim1 = 400, dim2 = 200;

	auto input = createTensorGPU<real>(device, dim0, dim1, dim2);
	auto inputGrad = createTensorGPU<real>(device, dim0, dim1, dim2);

	auto output = createTensorGPU<real>(device, dim0, dim1, dim2);
	auto outputGrad = createTensorGPU<real>(device, dim0, dim1, dim2);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar };
	Exp<real> expFunc(inputs);

	std::vector<const Tensor<real>*> inputValues = { &input };

	expFunc.forwardGPU(inputValues, &output);
	expFunc.backwardGPU(inputValues, &output, &outputGrad, 0, &inputGrad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_EXPTEST_H
