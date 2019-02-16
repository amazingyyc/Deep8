#ifndef DEEP8_MINUSTEST_H
#define DEEP8_MINUSTEST_H

#include "nodes/Minus.h"

namespace Deep8 {

TEST(Minus, forwardCPU) {
	CPUDevice device;

    auto x = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto y = createTensor(device, ElementType::from<float>(), 1, {200});
    auto z = createTensor(device, ElementType::from<float>(), 10, {500, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
    auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Minus minus(inputs);

    std::vector<const Tensor*> inputTensor = {&x, &y};

    minus.forward(inputTensor, &z);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(x.data<float>()[i * 500 * 200 + j * 200 + k] - y.data<float>()[k], z.data<float>()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }
}

TEST(Minus, backwardCPU) {
	CPUDevice device;


	auto inputValue1 = createTensor(device, ElementType::from<float>(), 10,{ 500, 200});
	auto inputValue2 = createTensor(device, ElementType::from<float>(), 1, {200});
	auto inputGrad1  = createTensor(device, ElementType::from<float>(), 10,{ 500, 200});
	auto inputGrad2  = createTensor(device, ElementType::from<float>(), 1, {200});
    auto outputValue = createTensor(device, ElementType::from<float>(), 10,{ 500, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10,{ 500, 200});

    /**create fake Add Function*/
    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
    auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Minus minus(inputs);

    zeroTensor(device, inputGrad1);
    zeroTensor(device, inputGrad2);

    std::vector<const Tensor*> inputValues = {&inputValue1, &inputValue2};

    minus.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);
    minus.backward(inputValues, &outputValue, &outputGrad, 1, &inputGrad2);

    for (int i = 0; i < 10 * 500 * 200; ++i) {
        ASSERT_EQ(inputGrad1.data<float>()[i], outputGrad.data<float>()[i]);
    }

    for (int i = 0; i < 200; ++i) {
        float temp = 0;

        for (int m = 0; m < 10; ++m) {
            for (int n = 0; n < 500; ++n) {
                temp += outputGrad.data<float>()[m * 500 * 200 + n * 200 + i];
            }
        }

        ASSERT_EQ(inputGrad2.data<float>()[i], -temp);
    }

}
#ifdef HAVE_CUDA

TEST(Minus, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto input0Ptr = (real*)malloc(sizeof(real) * 10 * 500 * 200);
	auto input0GradPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);

	auto input1Ptr = (real*)malloc(sizeof(real) * 1 * 1 * 200);
	auto input1GradPtr = (real*)malloc(sizeof(real) * 1 * 1 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);

	auto input0 = createTensorGPU<real>(device, input0Ptr, 10, 500, 200);
	auto input0Grad = createTensorGPU<real>(device, input0GradPtr, 10, 500, 200);

	auto input1 = createTensorGPU<real>(device, input1Ptr, 1, 200);
	auto input1Grad = createTensorGPU<real>(device, input1GradPtr, 1, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 10, 500, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 10, 500, 200);

	zeroTensor(device, input0Grad);
	zeroTensor(device, input1Grad);

	/**create fake Add Function*/
	auto inputVar0 = createFakeVariable<GPUDevice, real>(device);
	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	Minus<float> minus(inputs);

	std::vector<const Tensor<float>*> inputValues = { &input0, &input1 };

	minus.forwardGPU(inputValues, &output);
	minus.backwardGPU(inputValues, &output, &outputGrad, 0, &input0Grad);
	minus.backwardGPU(inputValues, &output, &outputGrad, 1, &input1Grad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 500 * 200);
	device.copyFromGPUToCPU(input0Grad.raw(), input0GradPtr, sizeof(real) * 10 * 500 * 200);
	device.copyFromGPUToCPU(input1Grad.raw(), input1GradPtr, sizeof(real) * 200);

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 500; ++j) {
			for (int k = 0; k < 200; ++k) {
				ASSERT_EQ(input0Ptr[i * 500 * 200 + j * 200 + k] - input1Ptr[k], outputPtr[i * 500 * 200 + j * 200 + k]);
			}
		}
	}

	for (int i = 0; i < 10 * 500 * 200; ++i) {
		ASSERT_EQ(input0GradPtr[i], outputGradPtr[i]);
	}

	for (int i = 0; i < 200; ++i) {
		float temp = 0;

		for (int m = 0; m < 10; ++m) {
			for (int n = 0; n < 500; ++n) {
				temp += outputGradPtr[m * 500 * 200 + n * 200 + i];
			}
		}

		ASSERT_EQ(input1GradPtr[i], -temp);
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

TEST(Minus, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input0 = createTensorGPU<real>(device, 10, 500, 200);
	auto input0Grad = createTensorGPU<real>(device, 10, 500, 200);

	auto input1 = createTensorGPU<real>(device, 1, 200);
	auto input1Grad = createTensorGPU<real>(device, 1, 200);

	auto output = createTensorGPU<real>(device, 10, 500, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 500, 200);

	/**create fake Add Function*/
	auto inputVar0 = createFakeVariable<GPUDevice, real>(device);
	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	Minus<real> minus(inputs);

	std::vector<const Tensor<real>*> inputValues = { &input0, &input1 };

	minus.forwardGPU(inputValues, &output);
	minus.backwardGPU(inputValues, &output, &outputGrad, 0, &input0Grad);
	minus.backwardGPU(inputValues, &output, &outputGrad, 1, &input1Grad);
}

#endif // HAVE_HALF
#endif // HAVE_CUDA



}

#endif //DEEP8_MINUSTEST_H
