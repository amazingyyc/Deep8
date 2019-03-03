#ifndef DEEP8_CWISEMULTIPLYTEST_H
#define DEEP8_CWISEMULTIPLYTEST_H

#include "nodes/Multiply.h"

namespace Deep8 {

TEST(Multiply, forwardCPU) {
	CPUDevice device;

    auto t1 = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto t2 = createTensor(device, ElementType::from<float>(), 1, {200});
    auto t3 = createTensor(device, ElementType::from<float>(), 10, {500, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
    auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Multiply cwiseMultiply(inputs);

    std::vector<const Tensor*> inputTensor = {&t1, &t2};

    cwiseMultiply.forward(inputTensor, &t3);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(t1.data<float>()[i * 500 * 200 + j * 200 + k] * t2.data<float>()[k], t3.data<float>()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }

}

TEST(Multiply, backwardCPU) {
	CPUDevice device;

	auto inputValue0 = createTensor(device, ElementType::from<float>(), 10, {100, 200});
	auto inputValue1 = createTensor(device, ElementType::from<float>(), 1, {200});
	auto inputGrad0  = createTensor(device, ElementType::from<float>(), 10, {100, 200});
	auto inputGrad1  = createTensor(device, ElementType::from<float>(), 1, {200});
    auto outputValue = createTensor(device, ElementType::from<float>(), 10, {100, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {100, 200});

    /**create fake Add Function*/
    auto inputVar0 = createFakeVariable(device, ElementType::from<float>());
    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar0, &inputVar1};
    Multiply cwiseMultiply(inputs);

    zeroTensor(device, inputGrad0);
    zeroTensor(device, inputGrad1);

    std::vector<const Tensor*> inputValues = {&inputValue0, &inputValue1};

    cwiseMultiply.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad0);
    cwiseMultiply.backward(inputValues, &outputValue, &outputGrad, 1, &inputGrad1);

    /**
     * test inputGrad0
     */
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(inputGrad0.data<float>()[i * 100 * 200 + j * 200 + k], outputGrad.data<float>()[i * 100 * 200 + j * 200 + k] * inputValue1.data<float>()[k]);
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
                temp += (inputValue0.data<float>()[m * 100 * 200 + n * 200 + i]  * outputGrad.data<float>()[m * 100 * 200 + n * 200 + i]);
            }
        }

        ASSERT_EQ(inputGrad1.data<float>()[i], temp);
    }

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

    auto input0     = createTensor(device, input0Ptr,     ElementType::from<real>(), 10, {100, 200});
    auto input0Grad = createTensor(device, input0GradPtr, ElementType::from<real>(), 10, {100, 200});
    auto input1     = createTensor(device, input1Ptr,     ElementType::from<real>(), 1,  {200});
    auto input1Grad = createTensor(device, input1GradPtr, ElementType::from<real>(), 1, {200});
    auto output     = createTensor(device, outputPtr,     ElementType::from<real>(), 10, {100, 200});
    auto outputGrad = createTensor(device, outputGradPtr, ElementType::from<real>(), 10, {100, 200});

    zeroTensor(device, input0Grad);
	zeroTensor(device, input1Grad);

	/**create fake Add Function*/
	auto inputVar0 = createFakeVariable(device, ElementType::from<real>());
	auto inputVar1 = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	Multiply multiply(inputs);

	std::vector<const Tensor*> inputValues = { &input0, &input1 };

	multiply.forward(inputValues, &output);
	multiply.backward(inputValues, &output, &outputGrad, 0, &input0Grad);
	multiply.backward(inputValues, &output, &outputGrad, 1, &input1Grad);

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
        float temp = 0;

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
}

#ifdef HAVE_HALF

TEST(Multiply, half_GPU) {
	typedef half real;

	GPUDevice device;

    auto input0     = createTensor(device, ElementType::from<real>(), 10, {100, 200});
    auto input0Grad = createTensor(device, ElementType::from<real>(), 10, {100, 200});
    auto input1     = createTensor(device, ElementType::from<real>(), 1,  {200});
    auto input1Grad = createTensor(device, ElementType::from<real>(), 1,  {200});
    auto output     = createTensor(device, ElementType::from<real>(), 10, {100, 200});
    auto outputGrad = createTensor(device, ElementType::from<real>(), 10, {100, 200});

	/**create fake Add Function*/
	auto inputVar0 = createFakeVariable(device, ElementType::from<real>());
	auto inputVar1 = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar0, &inputVar1 };
	Multiply multiply(inputs);

	std::vector<const Tensor*> inputValues = { &input0, &input1 };

	multiply.forward(inputValues, &output);
	multiply.backward(inputValues, &output, &outputGrad, 0, &input0Grad);
	multiply.backward(inputValues, &output, &outputGrad, 1, &input1Grad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_CWISEMULTIPLY_H
