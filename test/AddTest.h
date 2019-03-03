#ifndef DEEP8_ADDTEST_H
#define DEEP8_ADDTEST_H

#include "nodes/Add.h"

namespace Deep8 {

/**
 * @brief test the Add forwardCPU function
 */
TEST(Add, forwardCPU_float) {
	CPUDevice device;

    auto t1 = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto t2 = createTensor(device, ElementType::from<float>(),  1, {200});
    auto t3 = createTensor(device, ElementType::from<float>(), 10, {500, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
    auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Add add(inputs);

    std::vector<const Tensor*> inputTensor = {&t1, &t2};

    add.forward(inputTensor, &t3);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                ASSERT_EQ(t1.data<float>()[i * 500 * 200 + j * 200 + k] + t2.data<float>()[k], t3.data<float>()[i * 500 * 200 + j * 200 + k]);
            }
        }
    }
}

TEST(Add, backwardCPU_float) {
	CPUDevice device;

    auto inputValue1 = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto inputValue2 = createTensor(device, ElementType::from<float>(), 1, {200});

    auto inputGrad1 = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto inputGrad2 = createTensor(device, ElementType::from<float>(), 1, {200});

    auto outputValue = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {500, 200});

    /**create fake Add Function*/
    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
    auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Add add(inputs);

    zeroTensor(device, inputGrad1);
    zeroTensor(device, inputGrad2);

    std::vector<const Tensor*> inputValues = {&inputValue1, &inputValue2};

    add.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);
    add.backward(inputValues, &outputValue, &outputGrad, 1, &inputGrad2);

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

        ASSERT_EQ(inputGrad2.data<float>()[i], temp);
    }
}

#ifdef HAVE_CUDA

TEST(Add, forwardGPU_float) {
	GPUDevice device;

	auto t1Ptr = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto t2Ptr = (float*)malloc(sizeof(float) * 1 * 200);
	auto t3Ptr = (float*)malloc(sizeof(float) * 10 * 500 * 200);

    auto t1 = createTensor(device, t1Ptr, ElementType::from<float>(), (10), {(500), (200)});
    auto t2 = createTensor(device, t2Ptr, ElementType::from<float>(), (1), {(200)});
    auto t3 = createTensor(device, t3Ptr, ElementType::from<float>(), (10), {(500), (200)});

	auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
	auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Add add(inputs);

	std::vector<const Tensor*> inputTensor = { &t1, &t2 };

	add.forward(inputTensor, &t3);

	device.copyFromGPUToCPU(t3.raw(), t3Ptr, sizeof(float) * 10 * 500 * 200);

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 500; ++j) {
			for (int k = 0; k < 200; ++k) {
				ASSERT_EQ(t1Ptr[i * 500 * 200 + j * 200 + k] + t2Ptr[k], t3Ptr[i * 500 * 200 + j * 200 + k]);
			}
		}
	}

    free(t1Ptr);
    free(t2Ptr);
    free(t3Ptr);
}

TEST(Add, backwardGPU_float) {
	GPUDevice device;

	auto inputValue1CPU = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto inputValue2CPU = (float*)malloc(sizeof(float) * 1  * 200);

	auto inputGrad1CPU = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto inputGrad2CPU = (float*)malloc(sizeof(float) * 1  * 200);

	auto outputValueCPU = (float*)malloc(sizeof(float) * 10 * 500 * 200);
	auto outputGradCPU  = (float*)malloc(sizeof(float) * 10 * 500 * 200);

    auto inputValue1 = createTensor(device, inputValue1CPU,ElementType::from<float>(),   (10), {(500), (200)});
    auto inputValue2 = createTensor(device, inputValue2CPU,ElementType::from<float>(),   1, {200});
    auto inputGrad1  = createTensor(device, inputGrad1CPU, ElementType::from<float>(),  (10),{ (500), (200)});
    auto inputGrad2  = createTensor(device, inputGrad2CPU, ElementType::from<float>(),  1, {200});
    auto outputValue = createTensor(device, outputValueCPU,ElementType::from<float>(),  size_t(10), {size_t(500), size_t(200)});
    auto outputGrad  = createTensor(device, outputGradCPU, ElementType::from<float>(), size_t(10), {size_t(500), size_t(200)});

	auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
	auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Add add(inputs);

	zeroTensor(device, inputGrad1);
	zeroTensor(device, inputGrad2);

	std::vector<const Tensor*> inputValues = { &inputValue1, &inputValue2 };

	add.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);
	add.backward(inputValues, &outputValue, &outputGrad, 1, &inputGrad2);

	device.copyFromGPUToCPU(inputGrad1.raw(), inputGrad1CPU, sizeof(float) * 10 * 500 * 200);
	device.copyFromGPUToCPU(inputGrad2.raw(), inputGrad2CPU, sizeof(float) * 1 * 200);

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
}


#ifdef HAVE_HALF

TEST(Add, half_GPU) {
	GPUDevice device;

    auto inputValue1 = createTensor(device,ElementType::from<float>(), (10), { (500), (200)});
    auto inputValue2 = createTensor(device,ElementType::from<float>(), 1, {200});
    auto inputGrad1  = createTensor(device,ElementType::from<float>(), (10), {(500), (200)});
    auto inputGrad2  = createTensor(device,ElementType::from<float>(), 1, {200});
    auto outputValue = createTensor(device,ElementType::from<float>(), size_t(10), {size_t(500), size_t(200)});
    auto outputGrad  = createTensor(device,ElementType::from<float>(), size_t(10), {size_t(500), size_t(200)});

	auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
	auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

	std::vector<Node*> inputs = { &inputVar1, &inputVar2 };
	Add add(inputs);

	std::vector<const Tensor*> inputValues = { &inputValue1, &inputValue2 };

	add.forward(inputValues, &outputValue);
	add.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);
	add.backward(inputValues, &outputValue, &outputGrad, 1, &inputGrad2);

}

#endif // HAVE_HALF


#endif


}

#endif //DEEP8_ADDTEST_H
