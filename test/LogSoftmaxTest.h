#ifndef DEEP8_LOGSOFTMAXTEST_H
#define DEEP8_LOGSOFTMAXTEST_H

#include "nodes/LogSoftmax.h"

namespace Deep8 {

#ifdef HAVE_CUDA

TEST(LogSoftmax, GPU) {
    typedef float real;

	GPUDevice device;

	auto inputPtr = (real*)malloc(sizeof(real) * 400 * 200);
	auto inputGradPtr = (real*)malloc(sizeof(real) * 400 * 200);

	auto outputPtr = (real*)malloc(sizeof(real) * 400 * 200);
	auto outputGradPtr = (real*)malloc(sizeof(real) * 400 * 200);

	auto input      = createTensor(device, inputPtr,     ElementType::from<real>(), 400, {200});
	auto inputGrad  = createTensor(device, inputGradPtr, ElementType::from<real>(), 400, {200});
	auto output     = createTensor(device, outputPtr,    ElementType::from<real>(), 400, {200});
	auto outputGrad = createTensor(device, outputGradPtr,ElementType::from<real>(), 400, {200});

	zeroTensor(device, inputGrad);

	auto inputVar1 = createFakeVariable(device, ElementType::from<real>());

	std::vector<Node*> inputs = { &inputVar1 };
	LogSoftmax logsoftmax(inputs);

	std::vector<const Tensor*> inputTensor = { &input };

    softmax.forward(inputTensor, &output);
	softmax.backward(inputTensor, &output, &outputGrad, 0, &inputGrad);

	device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(real) * 400 * 200);
	device.copyFromGPUToCPU(inputGrad.raw(), inputGradPtr, sizeof(real) * 400 * 200);

    auto tempsumptr = (real*)malloc(sizeof(real) * 400);

    for (int i = 0; i < 400; ++i) {
        tempsumptr[i] = 0;
        for (int j = 0; j < 200; ++j) {
            tempsumptr[i] += outputGradPtr[i * 200 + j];
        }
    }

    for (int i = 0; i < 400; ++i) {
        for (int j = 0; j < 200; ++j) {
            auto tmp = outputGradPtr[i * 200 + j] - std::exp(outputPtr[i * 200 + j]) * tempsumptr[i];

            if (std::abs(tmp - inputGradPtr[i * 200 + j]) > 1e-6) {
                std::cout << tmp << ", " << inputGradPtr[i * 200 + j] << std::endl;
                ASSERT_TRUE(false);
            }
        }
    }

    free(inputPtr);
	free(inputGradPtr);
	free(outputPtr);
	free(outputGradPtr);
	free(tempsumptr);
}

#endif

}

#endif