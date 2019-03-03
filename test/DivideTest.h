#ifndef DEEP8_DIVIDETEST_H
#define DEEP8_DIVIDETEST_H

#include "nodes/Divide.h"

namespace Deep8 {

TEST(Divide, forwardCPU) {
	CPUDevice device;

    auto t1 = createTensor(device, ElementType::from<float>(), size_t(10), {size_t(500), size_t(200)});
    auto t2 = createTensor(device, ElementType::from<float>(), size_t(1), {size_t(200)});
    auto t3 = createTensor(device, ElementType::from<float>(), size_t(10), {size_t(500), size_t(200)});

    for (int i = 0; i < 200; ++i) {
        if (t2.data<float>()[i] == 0) {
            t2.data<float>()[i] = 1.0;
        }
    }

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
    auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Divide divide(inputs);

    std::vector<const Tensor*> inputTensor = {&t1, &t2};

    divide.forward(inputTensor, &t3);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(t1.data<float>()[i * 500 * 200 + j * 200 + k] / t2.data<float>()[k], t3.data<float>()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }
}

TEST(Divide, backwardCPU) {
	CPUDevice device;

    auto inputValue0 = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(100), size_t(200)});
    auto inputValue1 = createTensor(device, ElementType::from<double>(), size_t(1), {size_t(200)});

    auto inputGrad0 = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(100), size_t(200)});
    auto inputGrad1 = createTensor(device, ElementType::from<double>(), size_t(1), {size_t(200)});

    auto outputValue = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(100), size_t(200)});
    auto outputGrad  = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(100), size_t(200)});

    for (int i = 0; i < 200; ++i) {
        if (inputValue1.data<double>()[i] == 0) {
            inputValue1.data<double>()[i] = 1.0;
        }
    }

    /**create fake Add Function*/
    auto inputVar0 = createFakeVariable(device, ElementType::from<double>());
    auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar0, &inputVar1};
    Divide divide(inputs);

    zeroTensor(device, inputGrad0);
    zeroTensor(device, inputGrad1);

    std::vector<const Tensor*> inputValues = {&inputValue0, &inputValue1};

    divide.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad0);
    divide.backward(inputValues, &outputValue, &outputGrad, 1, &inputGrad1);

    /**
     * test inputGrad0
     */
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 100; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(inputGrad0.data<double>()[i * 100 * 200 + j * 200 + k], outputGrad.data<double>()[i * 100 * 200 + j * 200 + k] / inputValue1.data<double>()[k]);
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
                temp += (outputGrad.data<double>()[m * 100 * 200 + n * 200 + i] * -1 * inputValue0.data<double>()[m * 100 * 200 + n * 200 + i] / (inputValue1.data<double>()[i] * inputValue1.data<double>()[i]));
            }
        }

        ASSERT_TRUE(std::abs(inputGrad1.data<double>()[i] - temp) < 1e-6);
    }
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

    auto input1     = createTensor(device, input1Ptr, ElementType::from<float>(), 10, {100, 200});
    auto input1Grad = createTensor(device, input1GradPtr, ElementType::from<float>(), 10, {100, 200});
    auto input2 = createTensor(device, input2Ptr, ElementType::from<float>(), 1, {200});
    auto input2Grad = createTensor(device, input2GradPtr, ElementType::from<float>(), 1, {200});
    auto output = createTensor(device, outputPtr, ElementType::from<float>(), 10, {100, 200});
    auto outputGrad = createTensor(device, outputGradPtr, ElementType::from<float>(), 10, {100, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
	auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

	for (int i = 0; i < 200; ++i) {
		if (input2Ptr[i] == 0) {
			input2Ptr[i] = 2.0;
		}
	}

	device.copyFromCPUToGPU(input2Ptr, input2.raw(), sizeof(float) * 200);

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Divide divide(inputs);

    zeroTensor(device, input1Grad);
    zeroTensor(device, input2Grad);

    std::vector<const Tensor*> inputValues = {&input1, &input2};

    divide.forward(inputValues, &output);
    divide.backward(inputValues, &output, &outputGrad, 0, &input1Grad);
    divide.backward(inputValues, &output, &outputGrad, 1, &input2Grad);

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

}

#ifdef HAVE_HALF

TEST(Divide, half_GPU) {
	GPUDevice device;

    auto input1     = createTensor(device, ElementType::from<half>(), 10, {100, 200});
    auto input1Grad = createTensor(device, ElementType::from<half>(), 10, {100, 200});
    auto input2     = createTensor(device, ElementType::from<half>(), 1, {200});
    auto input2Grad = createTensor(device, ElementType::from<half>(), 1, {200});
    auto output     = createTensor(device, ElementType::from<half>(), 10, {100, 200});
    auto outputGrad = createTensor(device, ElementType::from<half>(), 10, {100, 200});

	auto inputVar1 = createFakeVariable(device, ElementType::from<half>());
	auto inputVar2 = createFakeVariable(device, ElementType::from<half>());

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Divide divide(inputs);

	zeroTensor(device, input1Grad);
	zeroTensor(device, input2Grad);

	std::vector<const Tensor*> inputValues = { &input1, &input2 };

	divide.forward(inputValues, &output);
	divide.backward(inputValues, &output, &outputGrad, 0, &input1Grad);
	divide.backward(inputValues, &output, &outputGrad, 1, &input2Grad);

}

#endif // HAVE_HALF

#endif

}

#endif //DEEP8_DIVIDETEST_H