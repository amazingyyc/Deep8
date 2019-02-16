#ifndef DEEP8_SQUARETEST_H
#define DEEP8_SQUARETEST_H

#include "nodes/Square.h"

namespace Deep8 {

TEST(Square, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto output = createTensor(device, ElementType::from<float>(), 10, {500, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1};
    Square add(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    add.forward(inputTensor, &output);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 500; ++j) {
            for (int k = 0; k < 200; ++k) {
                auto temp = input.data<float>()[i * 500 * 200 + j * 200 + k];
                ASSERT_TRUE(std::abs(temp * temp - output.data<float>()[i * 500 * 200 + j * 200 + k]) < 1e-6);
            }
        }
    }


}

TEST(Square, backwardCPU) {
	CPUDevice device;

    auto inputValue1 = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto inputGrad1  = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto outputValue = createTensor(device, ElementType::from<float>(), 10, {500, 200});
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {500, 200});

    /**create fake Add Function*/
    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1};
    Square add(inputs);

    zeroTensor(device, inputGrad1);

    std::vector<const Tensor*> inputValues = {&inputValue1};

    add.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad1);

    for (int i = 0; i < 10 * 500 * 200; ++i) {
        ASSERT_TRUE(std::abs(inputGrad1.data<float>()[i] - outputGrad.data<float>()[i] * 2 * inputValue1.data<float>()[i]) < 1e-6);
    }
}


#ifdef HAVE_CUDA

TEST(Square, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 500 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 10, 500, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 10, 500, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 10, 500, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 10, 500, 200);

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Square<real> square(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	square.forwardGPU(inputTensor, &output);
	square.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 500 * 200);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 10 * 500 * 200);

	for (int i = 0; i < 10; ++i) {
		for (int j = 0; j < 500; ++j) {
			for (int k = 0; k < 200; ++k) {
				auto temp = inputPtr[i * 500 * 200 + j * 200 + k];
				ASSERT_TRUE(std::abs(temp * temp - outputPtr[i * 500 * 200 + j * 200 + k]) < 1e-6);
			}
		}
	}

	for (int i = 0; i < 10 * 500 * 200; ++i) {
		ASSERT_TRUE(std::abs(inputGradPtr[i] - outputGradPtr[i] * 2 * inputPtr[i]) < 1e-6);
	}

	free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);

	freeTensor(device, input);
	freeTensor(device, inputGrad);
	freeTensor(device, output);
	freeTensor(device, outputGrad);

}

#ifdef HAVE_HALF

TEST(Square, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input = createTensorGPU<real>(device, 10, 500, 200);
	auto inputGrad = createTensorGPU<real>(device, 10, 500, 200);

	auto output = createTensorGPU<real>(device, 10, 500, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 500, 200);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	Square<real> square(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	square.forwardGPU(inputTensor, &output);
	square.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_SQUARE_H
