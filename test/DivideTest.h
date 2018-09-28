#ifndef DEEP8_DIVIDETEST_H
#define DEEP8_DIVIDETEST_H

#include "Divide.h"

namespace Deep8 {

TEST(Divide, forwardCPU) {
	CPUDevice device;

    auto t1 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));
    auto t2 = createTensor<CPUDevice, float>(device, size_t(1), size_t(200));
    auto t3 = createTensor<CPUDevice, float>(device, size_t(10), size_t(500), size_t(200));

    for (int i = 0; i < 200; ++i) {
        if (t2.data()[i] == 0) {
            t2.data()[i] = 1.0;
        }
    }

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Divide<float> divide(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&t1, &t2};

    divide.forwardCPU(inputTensor, &t3);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(t1.data()[i * 500 * 200 + j * 200 + k] / t2.data()[k], t3.data()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }

    freeTensor<CPUDevice, float>(device, t1);
    freeTensor<CPUDevice, float>(device, t2);
    freeTensor<CPUDevice, float>(device, t3);

    freeFakeVariable(inputVar1);
    freeFakeVariable(inputVar2);
}

TEST(Divide, backwardCPU) {
	CPUDevice device;

    auto inputValue0 = createTensor<CPUDevice, double>(device, size_t(10), size_t(100), size_t(200));
    auto inputValue1 = createTensor<CPUDevice, double>(device, size_t(1), size_t(200));

    auto inputGrad0 = createTensor<CPUDevice, double>(device, size_t(10), size_t(100), size_t(200));
    auto inputGrad1 = createTensor<CPUDevice, double>(device, size_t(1), size_t(200));

    auto outputValue = createTensor<CPUDevice, double>(device, size_t(10), size_t(100), size_t(200));
    auto outputGrad  = createTensor<CPUDevice, double>(device, size_t(10), size_t(100), size_t(200));

    for (int i = 0; i < 200; ++i) {
        if (inputValue1.data()[i] == 0) {
            inputValue1.data()[i] = 1.0;
        }
    }

    /**create fake Add Function*/
    auto inputVar0 = createFakeVariable<CPUDevice, double>(device);
    auto inputVar1 = createFakeVariable<CPUDevice, double>(device);

    std::vector<Node*> inputs = {&inputVar0, &inputVar1};
    Divide<double> divide(inputs);

    zeroTensor(device, inputGrad0);
    zeroTensor(device, inputGrad1);

    std::vector<const Tensor<double>*> inputValues = {&inputValue0, &inputValue1};

    divide.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad0);
    divide.backwardCPU(inputValues, &outputValue, &outputGrad, 1, &inputGrad1);

    /**
     * test inputGrad0
     */
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(inputGrad0.data()[i * 100 * 200 + j * 200 + k], outputGrad.data()[i * 100 * 200 + j * 200 + k] / inputValue1.data()[k]);
            }
        }
    }

    /**
     * test inputGrad1
     */
    for (int i = 0; i < 200; ++i) {
        double temp = 0;

        for (int m = 0; m < 10; ++m) {
            for (int n = 0; n < 100; ++n) {
                temp += (outputGrad.data()[m * 100 * 200 + n * 200 + i] * -1 * inputValue0.data()[m * 100 * 200 + n * 200 + i] / (inputValue1.data()[i] * inputValue1.data()[i]));
            }
        }

        ASSERT_TRUE(std::abs(inputGrad1.data()[i] - temp) < 1e-6);
    }

    freeTensor<CPUDevice, double>(device, inputValue0);
    freeTensor<CPUDevice, double>(device, inputValue1);
    freeTensor<CPUDevice, double>(device, inputGrad0);
    freeTensor<CPUDevice, double>(device, inputGrad1);
    freeTensor<CPUDevice, double>(device, outputValue);
    freeTensor<CPUDevice, double>(device, outputGrad);

    freeFakeVariable(inputVar0);
    freeFakeVariable(inputVar1);
}

#ifdef HAVE_CUDA

TEST(Divide, GPU_float) {
	GPUDevice device;

    auto input1Ptr = (float*)malloc(sizeof(float) * 10 * 100 * 200);
    auto input1GradPtr = (float*)malloc(sizeof(float) * 10 * 100 * 200);

    auto input2Ptr = (float*)malloc(sizeof(float) * 1 * 200);
    auto input2GradPtr = (float*)malloc(sizeof(float) * 1 * 200);

    auto outputPtr = (float*)malloc(sizeof(float)*10*100*200);
    auto outputGradPtr = (float*)malloc(sizeof(float)*10*100*200);

    auto input1 = createTensorGPU<float>(device, input1Ptr, 10, 100, 200);
    auto input1Grad = createTensorGPU<float>(device, input1GradPtr, 10, 100, 200);

    auto input2 = createTensorGPU<float>(device, input2Ptr, 1, 200);
    auto input2Grad = createTensorGPU<float>(device, input2GradPtr, 1, 200);

    auto output = createTensorGPU<float>(device, outputPtr, 10, 100, 200);
    auto outputGrad = createTensorGPU<float>(device, outputGradPtr, 10, 100, 200);

    auto inputVar1 = createFakeVariable<GPUDevice, float>(device);
	auto inputVar2 = createFakeVariable<GPUDevice, float>(device);

	for (int i = 0; i < 200; ++i) {
		if (input2Ptr[i] == 0) {
			input2Ptr[i] = 2.0;
		}
	}

	device.copyFromCPUToGPU(input2Ptr, input2.raw(), sizeof(float) * 200);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Divide<float> divide(inputs);

    zeroTensor(device, input1Grad);
    zeroTensor(device, input2Grad);

    std::vector<const Tensor<float>*> inputValues = {&input1, &input2};

    divide.forwardGPU(inputValues, &output);
    divide.backwardGPU(inputValues, &output, &outputGrad, 0, &input1Grad);
    divide.backwardGPU(inputValues, &output, &outputGrad, 1, &input2Grad);

    device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(float) * 10 * 100 * 200);
    device.copyFromGPUToCPU(input1Grad.raw(), input1GradPtr, sizeof(float) * 10 * 100 * 200);
    device.copyFromGPUToCPU(input2Grad.raw(), input2GradPtr, sizeof(float) * 1 * 200);

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 100; ++j) {
			for (int k = 0; k < 200; ++k) {
				ASSERT_EQ(input1Ptr[i * 100 * 200 + j * 200 + k] / input2Ptr[k], outputPtr[i * 100 * 200 + j * 200 + k]);				
			}
		}
	}

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 100; ++j) {
			for (int k = 0; k < 200; ++k) {
				ASSERT_EQ(input1GradPtr[i * 100 * 200 + j * 200 + k], outputGradPtr[i * 100 * 200 + j * 200 + k] / input2Ptr[k]);
			}
		}
	}

	/**
	 * test
	 */
	for (int i = 0; i < 200; ++i) {
		float temp = 0;

		for (int m = 0; m < 10; ++m) {
			for (int n = 0; n < 100; ++n) {
				temp += (outputGradPtr[m * 100 * 200 + n * 200 + i] * -1 * input1Ptr[m * 100 * 200 + n * 200 + i] / (input2Ptr[i] * input2Ptr[i]));
			}
		}

		ASSERT_TRUE(std::abs(input2GradPtr[i] - temp) < 1e-6);
	}

    free(input1Ptr);
    free(input1GradPtr);
    free(input2Ptr);
    free(input2GradPtr);
    free(outputPtr);
    free(outputGradPtr);

    freeTensor(device, input1);
	freeTensor(device, input2);
	freeTensor(device, input1Grad);
	freeTensor(device, input2Grad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar1);
	freeFakeVariable(inputVar2);

}

#ifdef HAVE_HALF

TEST(Divide, half_GPU) {
	GPUDevice device;

	auto input1 = createTensorGPU<half>(device, 10, 100, 200);
	auto input1Grad = createTensorGPU<half>(device, 10, 100, 200);

	auto input2 = createTensorGPU<half>(device, 1, 200);
	auto input2Grad = createTensorGPU<half>(device, 1, 200);

	auto output = createTensorGPU<half>(device, 10, 100, 200);
	auto outputGrad = createTensorGPU<half>(device, 10, 100, 200);

	auto inputVar1 = createFakeVariable<GPUDevice, half>(device);
	auto inputVar2 = createFakeVariable<GPUDevice, half>(device);

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Divide<half> divide(inputs);

	zeroTensor(device, input1Grad);
	zeroTensor(device, input2Grad);

	std::vector<const Tensor<half>*> inputValues = { &input1, &input2 };

	divide.forwardGPU(inputValues, &output);
	divide.backwardGPU(inputValues, &output, &outputGrad, 0, &input1Grad);
	divide.backwardGPU(inputValues, &output, &outputGrad, 1, &input2Grad);

}

#endif // HAVE_HALF

#endif

}

#endif //DEEP8_DIVIDETEST_H
