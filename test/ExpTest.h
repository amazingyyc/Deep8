#ifndef DEEP8_EXPTEST_H
#define DEEP8_EXPTEST_H

#include <cmath>
#include "nodes/Exp.h"

namespace Deep8 {

TEST(Exp, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(400), size_t(200)});
    auto output = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(400), size_t(200)});

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data<double>()[i] > 10) {
            input.data<double>()[i] = 10;
        }
    }

    auto inputVar1 = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar1};
    Exp expFunc(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    expFunc.forward(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        ASSERT_TRUE(std::abs(std::exp(input.data<double>()[i]) - output.data<double>()[i]) < 1e-6);
    }
}

TEST(Exp, backwardCPU) {
	CPUDevice device;

	auto inputValue = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(400), size_t(200)});
	auto inputGrad  = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(400), size_t(200)});

    auto outputValue = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(400), size_t(200)});
    auto outputGrad  = createTensor(device, ElementType::from<double>(), size_t(10), {size_t(400), size_t(200)});

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data<double>()[i] > 10) {
            inputValue.data<double>()[i] = 10;
        }
    }

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<double>());

    std::vector<Node*> inputs = {&inputVar};
    Exp expFunc(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    expFunc.forward(inputValues, &outputValue);
    expFunc.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        double temp = std::exp(inputValue.data<double>()[i]) * outputGrad.data<double>()[i];

        ASSERT_TRUE(std::abs(temp - inputGrad.data<double>()[i]) < 1e-6);
    }
}

#ifdef HAVE_CUDA

TEST(Exp, GPU_double) {
    typedef double real;

	GPUDevice device;

    int dim0 = 10, dim1 = 400, dim2 = 200;

    auto inputPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
    auto inputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto outputPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
    auto outputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto input      = createTensor(device, inputPtr,     ElementType::from<double>(), (size_t)dim0, { (size_t)dim1, (size_t)dim2});
    auto inputGrad  = createTensor(device, inputGradPtr, ElementType::from<double>(), (size_t)dim0, { (size_t)dim1, (size_t)dim2});
    auto output     = createTensor(device, outputPtr,    ElementType::from<double>(), (size_t)dim0, { (size_t)dim1, (size_t)dim2});
    auto outputGrad = createTensor(device, outputGradPtr,ElementType::from<double>(), (size_t)dim0, { (size_t)dim1, (size_t)dim2});

    /**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<double>());

    zeroTensor(device, inputGrad);

    std::vector<Node*> inputs = {&inputVar};
    Exp expFunc(inputs);

    std::vector<const Tensor*> inputValues = {&input};

	expFunc.forward(inputValues, &output);
	expFunc.backward(inputValues, &output, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * dim0 * dim1 * dim2);
    device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * dim0 * dim1 * dim2);

    for (int i = 0; i < dim0 * dim1 * dim2; ++i) {
        ASSERT_TRUE(std::abs(std::exp(inputPtr[i]) - outputPtr[i]) < 1e-4);
    }

    for (int i = 0; i < 10 * 400 * 200; ++i) {
		real temp = std::exp(inputPtr[i]) * outputGradPtr[i];

		ASSERT_TRUE(std::abs(temp - inputGradPtr[i]) < 1e-4);
    }

    free(inputPtr);
    free(inputGradPtr);
    free(outputPtr);
    free(outputGradPtr);

}

#ifdef HAVE_HALF

TEST(Exp, half_GPU) {
	typedef half real;

	GPUDevice device;

	int dim0 = 10, dim1 = 400, dim2 = 200;

    auto input      = createTensor(device, ElementType::from<half>(), (size_t)dim0, {(size_t)dim1, (size_t)dim2});
    auto inputGrad  = createTensor(device, ElementType::from<half>(), (size_t)dim0, {(size_t)dim1, (size_t)dim2});
    auto output     = createTensor(device, ElementType::from<half>(), (size_t)dim0, {(size_t)dim1, (size_t)dim2});
    auto outputGrad = createTensor(device, ElementType::from<half>(), (size_t)dim0, {(size_t)dim1, (size_t)dim2});

	/**create fake Add Function*/
	auto inputVar = createFakeVariable(device, ElementType::from<half>());

	zeroTensor(device, inputGrad);

	std::vector<Node*> inputs = { &inputVar };
	Exp expFunc(inputs);

	std::vector<const Tensor*> inputValues = { &input };

	expFunc.forward(inputValues, &output);
	expFunc.backward(inputValues, &output, &outputGrad, 0, &inputGrad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_EXPTEST_H
