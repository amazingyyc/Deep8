#ifndef DEEP8_ADDSCALARTEST_H
#define DEEP8_ADDSCALARTEST_H

#include "AddScalar.h"

namespace Deep8 {

/**
 * test the AddScalar Function
 */
TEST(AddScalar, forwardCPU_float) {
    auto device = new CPUDevice();

    auto input  = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto output = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    AddScalar<float> add(inputs, 3.0);

    std::vector<const Tensor<float>*> inputTensor = {&input};

    add.forwardCPU(inputTensor, &output);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(input.data()[i * 500 * 200 + j * 200 + k] + 3.0, output.data()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }

    freeTensor<CPUDevice, float>(device, input);
    freeTensor<CPUDevice, float>(device, output);

    freeFakeVariable(inputVar1);

    delete device;
}

TEST(AddScalar, backwardCPU_float) {
    auto device = new CPUDevice();
	
    auto inputValue1 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto inputGrad1  = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

	auto outputValue = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto outputGrad  = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    /**create fake Add Function*/
    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1};
    AddScalar<float> add(inputs, 5.0);

    zeroTensor(device, inputGrad1);

    std::vector<const Tensor<float>*> inputValues = {&inputValue1};

    add.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);

    for (int i = 0; i < 10 * 500 * 200; ++i) {
        ASSERT_EQ(inputGrad1.data()[i], outputGrad.data()[i]);
    }

    freeTensor<CPUDevice, float>(device, inputValue1);
    freeTensor<CPUDevice, float>(device, inputGrad1);
    freeTensor<CPUDevice, float>(device, outputValue);
    freeTensor<CPUDevice, float>(device, outputGrad);

    freeFakeVariable(inputVar1);

    delete device;
}

#ifdef HAVE_CUDA

/**Test forward on GPU*/
TEST(AddScalar, forwardGPU_float) {
    auto device = new GPUDevice();

    int dim0 = 10, dim1 = 500, dim2 = 200;
    float scalar = 5.5;

    auto cpuInputPtr  = (float*)malloc(sizeof(float) * dim0 * dim1 * dim2);
    auto cpuOutputPtr = (float*)malloc(sizeof(float) * dim0 * dim1 * dim2);

    auto input  = createTensorGPU<float>(device, cpuInputPtr, dim0, dim1, dim2);
	auto output = createTensorGPU<float>(device, cpuOutputPtr, dim0, dim1, dim2);

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	AddScalar<float> addScalar(inputs, scalar);

    std::vector<const Tensor<float>*> inputTensor = { &input };

	addScalar.forwardGPU(inputTensor, &output);

	device->copyFromGPUToCPU(output.pointer, cpuOutputPtr, sizeof(float) * dim0 * dim1 * dim2);

	for (int i = 0; i < dim0 * dim1 * dim2; ++i) {
		ASSERT_EQ(cpuInputPtr[i] + scalar, cpuOutputPtr[i]);
	}

	freeTensor(device, input);
	freeTensor(device, output);

	freeFakeVariable(inputVar1);

	free(cpuInputPtr);
	free(cpuOutputPtr);

	delete device;
}

/**test backward on GPU*/
TEST(AddScalar, backwardGPU_double) {
    typedef double real;

    auto device = new GPUDevice();

    int dim0 = 10, dim1 = 500, dim2 = 200;
    float scalar = 5.5;

    auto cpuInputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
	auto cpuInputGradPtr  = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

	auto cpuOutputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
	auto cpuOutputGradPtr  = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

	auto inputValue = createTensorGPU<real>(device, cpuInputValuePtr, dim0, dim1, dim2);
	auto inputGrad  = createTensorGPU<real>(device, cpuInputGradPtr, dim0, dim1, dim2);

	auto outputValue = createTensorGPU<real>(device, cpuOutputValuePtr, dim0, dim1, dim2);
	auto outputGrad = createTensorGPU<real>(device, cpuOutputGradPtr, dim0, dim1, dim2);

    /**create fake Add Function*/
	auto inputVar = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = {&inputVar};
	AddScalar<real> addScalar(inputs, scalar);

	zeroTensor(device, inputGrad);

    std::vector<const Tensor<real>*> inputValues = {&inputValue};

	addScalar.forwardGPU(inputValues, &outputValue);
	addScalar.backwardGPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    device->copyFromGPUToCPU(inputGrad.pointer, cpuInputGradPtr, sizeof(real) * dim0 * dim1 * dim2);

    for (int i = 0; i < dim0 * dim1 * dim2; ++i) {
	    ASSERT_EQ(cpuOutputGradPtr[i], cpuInputGradPtr[i]);
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

#ifdef HAVE_HALF

TEST(AddScalar, half_GPU) {
	auto device = new GPUDevice();

	int dim0 = 10, dim1 = 500, dim2 = 200;
	half scalar = 5.5;

	auto input = createTensorGPU<half>(device, dim0, dim1, dim2);
	auto inputGrad = createTensorGPU<half>(device, dim0, dim1, dim2);

	auto output = createTensorGPU<half>(device, dim0, dim1, dim2);
	auto outputGrad = createTensorGPU<half>(device, dim0, dim1, dim2);

	auto inputVar1 = createFakeVariable<GPUDevice, half>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	AddScalar<half> addScalar(inputs, scalar);

	std::vector<const Tensor<half>*> inputTensor = { &input };

	addScalar.forwardGPU(inputTensor, &output);
	addScalar.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	delete device;
}

#endif // HAVE_HALF


#endif

}

#endif //DEEP8_ADDSCALARTEST_H
