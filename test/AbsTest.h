#ifndef DEEP8_ABSTEST_H
#define DEEP8_ABSTEST_H

#include "nodes/Abs.h"

using namespace std;

namespace Deep8 {

TEST(Abs, forwardCPU_double) {
	CPUDevice device;

	auto input  = createTensor(device, ElementType::from<double>(), 10, {400, 200});
	auto output = createTensor(device, ElementType::from<double>(), 10, {400, 200});

	auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

	std::vector<Node*> inputs = { &inputVar1 };
	Abs absFunc(inputs);

	std::vector<const Tensor*> inputTensor = { &input };

	absFunc.forward(inputTensor, &output);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(std::abs(input.data<double>()[i]), output.data<double>()[i]);
	}
}

TEST(Abs, backwardCPU_float) {
	CPUDevice device;

    auto inputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto inputGrad  = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto outputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar};
    Abs absFunc(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    absFunc.forward(inputValues, &outputValue);
    absFunc.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        float temp;

        if (inputValue.data<float>()[i] >= 0) {
            temp = outputGrad.data<float>()[i];
        } else {
            temp = -outputGrad.data<float>()[i];
        }

        ASSERT_EQ(temp, inputGrad.data<float>()[i]);
    }
}

#ifdef HAVE_CUDA

TEST(Abs, forwardGPU_double) {
	GPUDevice device;

	size_t dim0 = 10, dim1 = 400, dim2 = 200;

	auto cpuInputPtr = (double*)malloc(sizeof(double) * dim0 * dim1 * dim2);
	auto cpuOuputPtr = (double*)malloc(sizeof(double) * dim0 * dim1 * dim2);

    auto input  = createTensor(device, cpuInputPtr, ElementType::from<double>(), dim0, {dim1, dim2});
    auto output = createTensor(device, cpuOuputPtr, ElementType::from<double>(),  dim0, {dim1, dim2});

	auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

	std::vector<Node*> inputs = { &inputVar1 };
	Abs absFunc(inputs);

	std::vector<const Tensor*> inputTensor = { &input };

	absFunc.forward(inputTensor, &output);

	device.copyFromGPUToCPU(output.raw(), cpuOuputPtr, sizeof(double) * dim0 * dim1 * dim2);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		ASSERT_EQ(std::abs(cpuInputPtr[i]), cpuOuputPtr[i]);
	}

    free(cpuInputPtr);
    free(cpuOuputPtr);
}

TEST(Abs, backwardGPU_double) {
	typedef double real;

	GPUDevice device;

	size_t dim0 = 10, dim1 = 400, dim2 = 200;

	auto cpuInputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
	auto cpuInputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

	auto cpuOutputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
	auto cpuOutputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto inputValue = createTensor(device, cpuInputValuePtr, ElementType::from<real>(), dim0, {dim1, dim2});
    auto inputGrad = createTensor(device, cpuInputGradPtr, ElementType::from<real>(), dim0, {dim1, dim2});

    auto outputValue = createTensor(device, cpuOutputValuePtr, ElementType::from<real>(), dim0, {dim1, dim2});
    auto outputGrad = createTensor(device, cpuOutputGradPtr, ElementType::from<real>(), dim0, {dim1, dim2});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = {&inputVar};
	Abs absFunc(inputs);

	zeroTensor(device, inputGrad);

	std::vector<const Tensor*> inputValues = {&inputValue};

	absFunc.forward(inputValues, &outputValue);
	absFunc.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(inputGrad.raw(), cpuInputGradPtr, sizeof(real) * dim0 * dim1 * dim2);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
	    real temp;

		if (cpuInputValuePtr[i] >= 0) {
			temp = cpuOutputGradPtr[i];
		} else if (cpuInputValuePtr[i] < 0) {
			temp = -cpuOutputGradPtr[i];
		}

	    ASSERT_EQ(temp, cpuInputGradPtr[i]);
	}

    free(cpuInputValuePtr);
    free(cpuInputGradPtr);
    free(cpuOutputValuePtr);
    free(cpuOutputGradPtr);
}


/**half test*/
#ifdef HAVE_HALF

TEST(Abs, half_GPU) {
	typedef half real;

	GPUDevice device;

	size_t dim0 = 10, dim1 = 400, dim2 = 200;

    auto inputValue = createTensor(device, ElementType::from<real>(), dim0, {dim1, dim2});
    auto inputGrad  = createTensor(device, ElementType::from<real>(), dim0, {dim1, dim2});

    auto outputValue = createTensor(device, ElementType::from<real>(), dim0, {dim1, dim2});
    auto outputGrad  = createTensor(device, ElementType::from<real>(), dim0, {dim1, dim2});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar };
	Abs absFunc(inputs);

	zeroTensor(device, inputGrad);

	std::vector<const Tensor*> inputValues = { &inputValue };

	absFunc.forward(inputValues, &outputValue);
	absFunc.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);
}

#endif

#endif



}


#endif //DEEP8_ABSTEST_H
