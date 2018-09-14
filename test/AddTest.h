#ifndef DEEP8_ADDTEST_H
#define DEEP8_ADDTEST_H

#include "Add.h"

namespace Deep8 {

/**
 * @brief test the Add forwardCPU function
 */
TEST(Add, forwardCPU_float) {
    auto device = new CPUDevice();

    auto t1 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto t2 = createTensor<CPUDevice, float>(device, size_t(1), size_t(200));
    auto t3 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Add<float> add(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&t1, &t2};

    add.forwardCPU(inputTensor, &t3);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(t1.data()[i * 500 * 200 + j * 200 + k] + t2.data()[k], t3.data()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }

    freeTensor<CPUDevice, float>(device, t1);
    freeTensor<CPUDevice, float>(device, t2);
    freeTensor<CPUDevice, float>(device, t3);

    freeFakeVariable(inputVar1);
    freeFakeVariable(inputVar2);

    delete device;
}

TEST(Add, backwardCPU_float) {
    auto device = new CPUDevice();

    auto inputValue1 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto inputValue2 = createTensor<CPUDevice, float>(device, size_t(1), size_t(200));

    auto inputGrad1 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto inputGrad2 = createTensor<CPUDevice, float>(device, size_t(1), size_t(200));

    auto outputValue = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto outputGrad  = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    /**create fake Add Function*/
    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Add<float> add(inputs);

    zeroTensor(device, inputGrad1);
    zeroTensor(device, inputGrad2);

    std::vector<const Tensor<float>*> inputValues = {&inputValue1, &inputValue2};

    add.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);
    add.backwardCPU(inputValues, &outputValue, &outputGrad, 1, &inputGrad2);

    for (int i = 0; i < 10 * 500 * 200; ++i) {
        ASSERT_EQ(inputGrad1.data()[i], outputGrad.data()[i]);
    }

    for (int i = 0; i < 200; ++i) {
        float temp = 0;

        for (int m = 0; m < 10; ++m) {
            for (int n = 0; n < 500; ++n) {
                temp += outputGrad.data()[m * 500 * 200 + n * 200 + i];
            }
        }

        ASSERT_EQ(inputGrad2.data()[i], temp);
    }

    freeTensor<CPUDevice, float>(device, inputValue1);
    freeTensor<CPUDevice, float>(device, inputValue2);
    freeTensor<CPUDevice, float>(device, inputGrad1);
    freeTensor<CPUDevice, float>(device, inputGrad2);
    freeTensor<CPUDevice, float>(device, outputValue);
    freeTensor<CPUDevice, float>(device, outputGrad);

    freeFakeVariable(inputVar1);
    freeFakeVariable(inputVar2);

    delete device;
}

#ifdef HAVE_CUDA

TEST(Add, forwardGPU_float) {
	auto device = new GPUDevice();

	auto t1Ptr = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto t2Ptr = (float*)malloc(sizeof(float) * 1 * 200);
	auto t3Ptr = (float*)malloc(sizeof(float) * 10 * 500 * 200);

	auto t1 = createTensorGPU<float>(device, t1Ptr, (10), (500), (200));
	auto t2 = createTensorGPU<float>(device, t2Ptr, (1), (200));
	auto t3 = createTensorGPU<float>(device, t3Ptr, (10), (500), (200));

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device);
	auto inputVar2 = createFakeVariable<GPUDevice, float>(device);

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Add<float> add(inputs);

	std::vector<const Tensor<float>*> inputTensor = { &t1, &t2 };

	add.forwardGPU(inputTensor, &t3);

	device->copyFromGPUToCPU(t3.pointer, t3Ptr, sizeof(float) * 10 * 500 * 200);

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 500; ++j) {
			for (int k = 0; k < 200; ++k) {
				ASSERT_EQ(t1Ptr[i * 500 * 200 + j * 200 + k] + t2Ptr[k], t3Ptr[i * 500 * 200 + j * 200 + k]);
			}
		}
	}

	freeTensor(device, t1);
	freeTensor(device, t2);
	freeTensor(device, t3);

	freeFakeVariable(inputVar1);
	freeFakeVariable(inputVar2);

	free(t1Ptr);
	free(t2Ptr);
	free(t3Ptr);

	delete device;
}

TEST(Add, backwardGPU_float) {
	auto device = new GPUDevice();

	auto inputValue1CPU = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto inputValue2CPU = (float*)malloc(sizeof(float) * 1  * 200);

	auto inputGrad1CPU = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto inputGrad2CPU = (float*)malloc(sizeof(float) * 1  * 200);

	auto outputValueCPU = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto outputGradCPU  = (float*)malloc(sizeof(float) * 10 * 500 * 200);

	auto inputValue1 = createTensorGPU<float>(device, inputValue1CPU, (10), (500), (200));
	auto inputValue2 = createTensorGPU<float>(device, inputValue2CPU, 1, 200);
	auto inputGrad1  = createTensorGPU<float>(device, inputGrad1CPU, (10), (500), (200));
	auto inputGrad2  = createTensorGPU<float>(device, inputGrad2CPU, 1, 200);

	auto outputValue = createTensorGPU<float>(device, outputValueCPU, size_t(10), size_t(500), size_t(200));
	auto outputGrad  = createTensorGPU<float>(device, outputGradCPU, size_t(10), size_t(500), size_t(200));

	auto inputVar1 = createFakeVariable<GPUDevice, float>(device);
	auto inputVar2 = createFakeVariable<GPUDevice, float>(device);

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Add<float> add(inputs);

	zeroTensor(device, inputGrad1);
	zeroTensor(device, inputGrad2);

	std::vector<const Tensor<float>*> inputValues = { &inputValue1, &inputValue2 };

	add.backwardGPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);
	add.backwardGPU(inputValues, &outputValue, &outputGrad, 1, &inputGrad2);

	device->copyFromGPUToCPU(inputGrad1.pointer, inputGrad1CPU, sizeof(float) * 10 * 500 * 200);
	device->copyFromGPUToCPU(inputGrad2.pointer, inputGrad2CPU, sizeof(float) * 1 * 200);

	for (int i = 0; i < 10 * 500 * 200; ++i) {
		ASSERT_EQ(inputGrad1CPU[i], outputGradCPU[i]);
	}

	for (int i = 0; i < 200; ++i) {
		float temp = 0;

		for (int m = 0; m < 10; ++m) {
			for (int n = 0; n < 500; ++n) {
				temp += outputGradCPU[m * 500 * 200 + n * 200 + i];
			}
		}

		ASSERT_EQ(inputGrad2CPU[i], temp);
	}

	free(inputValue1CPU);
	free(inputValue2CPU);
	free(inputGrad1CPU);
	free(inputGrad2CPU);
	free(outputValueCPU);
	free(outputGradCPU);

	freeTensor(device, inputValue1);
	freeTensor(device, inputValue2);
	freeTensor(device, inputGrad1);
	freeTensor(device, inputGrad2);
	freeTensor(device, outputValue);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar1);
	freeFakeVariable(inputVar2);

	delete device;
}

#endif


}

#endif //DEEP8_ADDTEST_H
