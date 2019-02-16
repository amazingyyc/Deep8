#ifndef DEEP8_RELUTEST_H
#define DEEP8_RELUTEST_H

#include "nodes/ReLu.h"

namespace Deep8 {

TEST(ReLu, forwardCPU) {
	CPUDevice device;

    auto input  = createTensor(device, ElementType::from<float>(), 10, {400, 200});
    auto output = createTensor(device, ElementType::from<float>(), 10, {400, 200});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1};
    ReLu relu(inputs);

    std::vector<const Tensor*> inputTensor = {&input};

    relu.forward(inputTensor, &output);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (input.data<float>()[i] < 0) {
            ASSERT_EQ(0, output.data<float>()[i]);
        } else {
            ASSERT_EQ(input.data<float>()[i], output.data<float>()[i]);
        }
    }


}

TEST(ReLu, backwardCPU) {
	CPUDevice device;

	auto inputValue  = createTensor(device, ElementType::from<float>(), 10, {400, 200});
	auto inputGrad   = createTensor(device, ElementType::from<float>(), 10, {400, 200});
	auto outputValue = createTensor(device, ElementType::from<float>(), 10, {400, 200});
	auto outputGrad  = createTensor(device, ElementType::from<float>(), 10, {400, 200});

   	/**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar};
    ReLu reLu(inputs);

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = {&inputValue};

    reLu.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    for (int i = 0; i < 10 * 400 * 200; ++i) {
        if (inputValue.data<float>()[i] >= 0) {
            ASSERT_EQ(inputGrad.data<float>()[i], outputGrad.data<float>()[i]);
        } else {
            ASSERT_EQ(inputGrad.data<float>()[i], 0);
        }
    }

}

#ifdef HAVE_CUDA

TEST(ReLu, GPU_float) {
	typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 10 * 400 * 200);

	auto input = createTensorGPU<real>(device, inputPtr, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, inputGradPtr, 10, 400, 200);

	auto output = createTensorGPU<real>(device, outputPtr, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, outputGradPtr, 10, 400, 200);

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	ReLu<real> relu(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	relu.forwardGPU(inputTensor, &output);
	relu.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 10 * 400 * 200);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 10 * 400 * 200);

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		if (inputPtr[i] < 0) {
			ASSERT_EQ(0, outputPtr[i]);
		} else {
			ASSERT_EQ(inputPtr[i], outputPtr[i]);
		}
	}

	for (int i = 0; i < 10 * 400 * 200; ++i) {
		if (inputPtr[i] > 0) {
			ASSERT_EQ(inputGradPtr[i], outputGradPtr[i]);
		} else {
			ASSERT_EQ(inputGradPtr[i], 0);
		}
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

TEST(ReLU, half_GPU) {
	typedef half real;

	GPUDevice device;

	auto input = createTensorGPU<real>(device, 10, 400, 200);
	auto inputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto output = createTensorGPU<real>(device, 10, 400, 200);
	auto outputGrad = createTensorGPU<real>(device, 10, 400, 200);

	auto inputVar1 = createFakeVariable<GPUDevice, real>(device);

	std::vector<Node*> inputs = { &inputVar1 };
	ReLu<real> relu(inputs);

	std::vector<const Tensor<real>*> inputTensor = { &input };

	relu.forwardGPU(inputTensor, &output);
	relu.backwardGPU(inputTensor, &output, &outputGrad, 0, &inputGrad);
}

#endif // HAVE_HALF
#endif

}

#endif //DEEP8_RELUTEST_H
