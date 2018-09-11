#ifndef DEEP8_ABSTEST_H
#define DEEP8_ABSTEST_H

#include "Abs.h"

namespace Deep8 {

TEST(Abs, forwardCPU_double) {
	auto device = new CPUDevice();

	auto input = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));
	auto output = createTensor<CPUDevice, double>(device, size_t(10), size_t(400), size_t(200));

	auto inputVar1 = createFakeVariable<CPUDevice, double>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Abs< double> absFunc(inputs);

	std::vector<const Tensor< double>*> inputTensor = { &input };

	absFunc.forwardCPU(inputTensor, &output);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(abs(input.data()[i]), output.data()[i]);
	}

	freeTensor(device, input);
	freeTensor(device, output);

	freeFakeVariable(inputVar1);

	delete device;
}

TEST(Abs, backwardCPU_float) {
    auto device = new CPUDevice();

    auto inputValue = createTensor<CPUDevice, float>(device, size_t(10), size_t(400), size_t(200));
    auto inputGrad  = createTensor<CPUDevice, float>(device, size_t(10), size_t(400), size_t(200));

    auto outputValue = createTensor<CPUDevice, float>(device, size_t(10), size_t(400), size_t(200));
    auto outputGrad  = createTensor<CPUDevice, float>(device, size_t(10), size_t(400), size_t(200));

    /**create fake Add Function*/
    auto inputVar = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar};
    Abs<float> absFunc(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor<float>*> inputValues = {&inputValue};

    absFunc.forwardCPU(inputValues, &outputValue);
    absFunc.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        float temp;

        if (inputValue.data()[i] > 0) {
            temp = outputGrad.data()[i];
        } else if (inputValue.data()[i] < 0) {
            temp = -outputGrad.data()[i];
        } else {
            temp = 0;
        }

        ASSERT_EQ(temp, inputGrad.data()[i]);
    }

    freeTensor(device, inputValue);
    freeTensor(device, inputGrad);
    freeTensor(device, outputValue);
    freeTensor(device, outputGrad);

    freeFakeVariable(inputVar);

    delete device;
}

#ifdef HAVE_CUDA

TEST(Abs, forwardGPU_double) {
	auto device = new GPUDevice(0);

	size_t dim0 = 10, dim1 = 400, dim2 = 200;

	auto cpuInputPtr = (double*)malloc(sizeof(double) * dim0 * dim1 * dim2);
	auto cpuOuputPtr = (double*)malloc(sizeof(double) * dim0 * dim1 * dim2);

	auto input  = createTensorGPU<double>(device, cpuInputPtr, dim0, dim1, dim2);
	auto output = createTensorGPU<double>(device, cpuOuputPtr, dim0, dim1, dim2);

	auto inputVar1 = createFakeVariable<GPUDevice, double>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Abs<double> absFunc(inputs);

	std::vector<const Tensor<double>*> inputTensor = { &input };

	absFunc.forwardGPU(inputTensor, &output);

	device->copyToCPU(output.pointer, cpuOuputPtr, sizeof(double) * dim0 * dim1 * dim2);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(abs(cpuInputPtr[i]), cpuOuputPtr[i]);
	}

	freeTensor(device, input);
	freeTensor(device, output);

	freeFakeVariable(inputVar1);

	free(cpuInputPtr);
	free(cpuOuputPtr);

	delete device;

}

TEST(Abs, backwardGPU_double) {
	typedef double real;

	auto device = new GPUDevice();

	size_t dim0 = 10, dim1 = 400, dim2 = 200;

	auto cpuInputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
	auto cpuInputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

	auto cpuOutputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
	auto cpuOutputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

	auto inputValue = createTensorGPU<real>(device, cpuInputValuePtr, dim0, dim1, dim2);
	auto inputGrad = createTensorGPU<real>(device, cpuInputGradPtr, dim0, dim1, dim2);

	auto outputValue = createTensorGPU<real>(device, cpuOutputValuePtr, dim0, dim1, dim2);
	auto outputGrad = createTensorGPU<real>(device, cpuOutputGradPtr, dim0, dim1, dim2);

	/**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = {&inputVar};
	Abs<real> absFunc(inputs);

	zeroTensor(device, inputGrad);

	std::vector<const Tensor<real>*> inputValues = {&inputValue};

	absFunc.forwardGPU(inputValues, &outputValue);
	absFunc.backwardGPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

	device->copyToCPU(inputGrad.pointer, cpuInputGradPtr, sizeof(real) * dim0 * dim1 * dim2);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
	    real temp;

		if (cpuInputValuePtr[i] > 0) {
			temp = cpuOutputGradPtr[i];
		} else if (cpuInputValuePtr[i] < 0) {
			temp = -cpuOutputGradPtr[i];
		} else {
			temp = 0;
		}

	    ASSERT_EQ(temp, cpuInputGradPtr[i]);
	}

	free(cpuInputValuePtr);
	free(cpuInputGradPtr);
	free(cpuOutputValuePtr);
	free(cpuOutputGradPtr);

	freeTensor(device, inputValue);
	freeTensor(device, inputGrad);
	freeTensor(device, outputValue);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar);

	delete device;
}

#endif



}


#endif //DEEP8_ABSTEST_H
