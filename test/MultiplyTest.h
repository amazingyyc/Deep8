#ifndef DEEP8_CWISEMULTIPLYTEST_H
#define DEEP8_CWISEMULTIPLYTEST_H

#include "Multiply.h"

namespace Deep8 {

TEST(Multiply, forwardCPU) {
	CPUDevice device;

    auto t1 = createTensor<CPUDevice, float>(device, 10, 500, 200);
    auto t2 = createTensor<CPUDevice, float>(device, 1, 200);
    auto t3 = createTensor<CPUDevice, float>(device, 10, 500, 200);

    auto inputVar1 = createFakeVariable<CPUDevice, float>(device);
    auto inputVar2 = createFakeVariable<CPUDevice, float>(device);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Multiply<float> cwiseMultiply(inputs);

    std::vector<const Tensor<float>*> inputTensor = {&t1, &t2};

    cwiseMultiply.forwardCPU(inputTensor, &t3);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(t1.data()[i * 500 * 200 + j * 200 + k] * t2.data()[k], t3.data()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }

    freeTensor<CPUDevice, float>(device, t1);
    freeTensor<CPUDevice, float>(device, t2);
    freeTensor<CPUDevice, float>(device, t3);

    freeFakeVariable(inputVar1);
    freeFakeVariable(inputVar2);

}

TEST(Multiply, backwardCPU) {
	CPUDevice device;

	auto inputValue0 = createTensor<CPUDevice, long double>(device, 10, 100, 200);
	auto inputValue1 = createTensor<CPUDevice, long double>(device, 1, 200);

	auto inputGrad0 = createTensor<CPUDevice, long double>(device, 10, 100, 200);
	auto inputGrad1 = createTensor<CPUDevice, long double>(device, 1, 200);

    auto outputValue = createTensor<CPUDevice, long double>(device, 10, 100, 200);
    auto outputGrad  = createTensor<CPUDevice, long double>(device, 10, 100, 200);

    /**create fake Add Function*/
    auto inputVar0 = createFakeVariable<CPUDevice, long double>(device);
    auto inputVar1 = createFakeVariable<CPUDevice, long double>(device);

    std::vector<Node*> inputs = {&inputVar0, &inputVar1};
    Multiply<long double> cwiseMultiply(inputs);

    zeroTensor(device, inputGrad0);
    zeroTensor(device, inputGrad1);

    std::vector<const Tensor<long double>*> inputValues = {&inputValue0, &inputValue1};

    cwiseMultiply.backwardCPU(inputValues, &outputValue, &outputGrad, 0, &inputGrad0);
    cwiseMultiply.backwardCPU(inputValues, &outputValue, &outputGrad, 1, &inputGrad1);

    /**
     * test inputGrad0
     */
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(inputGrad0.data()[i * 100 * 200 + j * 200 + k], outputGrad.data()[i * 100 * 200 + j * 200 + k] * inputValue1.data()[k]);
            }
        }
    }

    /**
     * test inputGrad1
     */
    for (int i = 0; i < 200; ++i) {
        long double temp = 0;

        for (int m = 0; m < 10; ++m) {
            for (int n = 0; n < 100; ++n) {
                temp += (inputValue0.data()[m * 100 * 200 + n * 200 + i]  * outputGrad.data()[m * 100 * 200 + n * 200 + i]);
            }
        }

        ASSERT_EQ(inputGrad1.data()[i], temp);
    }

    freeTensor<CPUDevice, long double>(device, inputValue0);
    freeTensor<CPUDevice, long double>(device, inputValue1);
    freeTensor<CPUDevice, long double>(device, inputGrad0);
    freeTensor<CPUDevice, long double>(device, inputGrad1);
    freeTensor<CPUDevice, long double>(device, outputValue);
    freeTensor<CPUDevice, long double>(device, outputGrad);

    freeFakeVariable(inputVar0);
    freeFakeVariable(inputVar1);

}

#ifdef HAVE_CUDA

TEST(Multiply, GPU_float) {
    typedef float real;

	GPUDevice device;

    auto input0Ptr = (real*)malloc(sizeof(real) * 10 * 100 * 200);
    auto input0GradPtr = (real*)malloc(sizeof(real) * 10 * 100 * 200);

    auto input1Ptr = (real*)malloc(sizeof(real) * 1 * 200);
    auto input1GradPtr = (real*)malloc(sizeof(real) * 1 * 200);

    auto outputPtr = (real*)malloc(sizeof(real) * 10 * 100 * 200);
    auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 100 * 200);

    auto input0 = createTensorGPU<real>(device, input0Ptr, 10, 100, 200);
    auto input0Grad = createTensorGPU<real>(device, input0GradPtr, 10, 100, 200);

    auto input1 = createTensorGPU<real>(device, input1Ptr, 1, 200);
    auto input1Grad = createTensorGPU<real>(device, input1GradPtr, 1, 200);

    auto output = createTensorGPU<real>(device, outputPtr, 10, 100, 200);
    auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 10, 100, 200);

    zeroTensor(device, input0Grad);
	zeroTensor(device, input1Grad);

	/**create fake Add Function*/
	auto inputVar0 = createFakeVariable<GPUDevice, real>(device);
	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	Multiply<float> multiply(inputs);

	std::vector<const Tensor<float>*> inputValues = { &input0, &input1 };

	multiply.forwardGPU(inputValues, &output);
	multiply.backwardGPU(inputValues, &output, &outputGrad, 0, &input0Grad);
	multiply.backwardGPU(inputValues, &output, &outputGrad, 1, &input1Grad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 100 * 200);
	device.copyFromGPUToCPU(input0Grad.raw(), input0GradPtr, sizeof(real) * 10 * 100 * 200);
	device.copyFromGPUToCPU(input1Grad.raw(), input1GradPtr, sizeof(real) * 200);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(input0Ptr[i * 100 * 200 + j * 200 + k] * input1Ptr[k], outputPtr[i * 100 * 200 + j * 200 + k]);
            }
        }
    }

    /**
     * test inputGrad0
     */
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(input0GradPtr[i * 100 * 200 + j * 200 + k], outputGradPtr[i * 100 * 200 + j * 200 + k] * input1Ptr[k]);
            }
        }
    }

    /**
     * test inputGrad1
     */
    for (int i = 0; i < 200; ++i) {
        long double temp = 0;

        for (int m = 0; m < 10; ++m) {
            for (int n = 0; n < 100; ++n) {
                temp += (input0Ptr[m * 100 * 200 + n * 200 + i]  * outputGradPtr[m * 100 * 200 + n * 200 + i]);
            }
        }

        ASSERT_EQ(input1GradPtr[i], temp);
    }

    free(input0Ptr);
	free(input0GradPtr);
	free(input1Ptr);
	free(input1GradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input0);
	freeTensor(device, input0Grad);
	freeTensor(device, input1);
	freeTensor(device, input1Grad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

	freeFakeVariable(inputVar0);
	freeFakeVariable(inputVar1);
}

#ifdef HAVE_HALF

TEST(Multiply, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input0 = createTensorGPU<real>(device, 10, 100, 200);
	auto input0Grad = createTensorGPU<real>(device, 10, 100, 200);

	auto input1 = createTensorGPU<real>(device, 1, 200);
	auto input1Grad = createTensorGPU<real>(device, 1, 200);

	auto output = createTensorGPU<real>(device, 10, 100, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 100, 200);

	/**create fake Add Function*/
	auto inputVar0 = createFakeVariable<GPUDevice, real>(device);
	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	Multiply<real> multiply(inputs);

	std::vector<const Tensor<real>*> inputValues = { &input0, &input1 };

	multiply.forwardGPU(inputValues, &output);
	multiply.backwardGPU(inputValues, &output, &outputGrad, 0, &input0Grad);
	multiply.backwardGPU(inputValues, &output, &outputGrad, 1, &input1Grad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_CWISEMULTIPLY_H
