#ifndef DEEP8_DOTTEST_H
#define DEEP8_DOTTEST_H

#include "nodes/Dot.h"

namespace Deep8 {

#ifdef HAVE_CUDA

TEST(Divide, GPU) {
    GPUDevice device;

    auto input1Ptr     = (float*)malloc(sizeof(float) * 1 * 100);
    auto input1GradPtr = (float*)malloc(sizeof(float) * 1 * 100);

    auto input2Ptr     = (float*)malloc(sizeof(float) * 1 * 100);
    auto input2GradPtr = (float*)malloc(sizeof(float) * 1 * 100);

    auto outputPtr     = (float*)malloc(sizeof(float)*1);
    auto outputGradPtr = (float*)malloc(sizeof(float)*1);

    auto input1     = createTensor(device, input1Ptr, ElementType::from<float>(), 1, {100});
    auto input1Grad = createTensor(device, input1GradPtr, ElementType::from<float>(), 1, {100});
    auto input2     = createTensor(device, input2Ptr, ElementType::from<float>(), 1, {100});
    auto input2Grad = createTensor(device, input2GradPtr, ElementType::from<float>(), 1, {100});
    auto output     = createTensor(device, outputPtr, ElementType::from<float>(), 1, {1});
    auto outputGrad = createTensor(device, outputGradPtr, ElementType::from<float>(), 1, {1});

    auto inputVar1 = createFakeVariable(device, ElementType::from<float>());
	auto inputVar2 = createFakeVariable(device, ElementType::from<float>());

    std::vector<Node*> inputs = {&inputVar1, &inputVar2};
    Dot dot(inputs);

    zeroTensor(device, input1Grad);
    zeroTensor(device, input2Grad);

    std::vector<const Tensor*> inputValues = { &input1, &input2 };

    dot.forward(inputValues, &output);
    dot.backward(inputValues, &output, &outputGrad, 0, &input1Grad);
    dot.backward(inputValues, &output, &outputGrad, 1, &input2Grad);

    device.copyFromGPUToCPU(output.raw(), outputPtr, sizeof(float) * 1);
    device.copyFromGPUToCPU(input1Grad.raw(), input1GradPtr, sizeof(float) * 1 * 100);
    device.copyFromGPUToCPU(input2Grad.raw(), input2GradPtr, sizeof(float) * 1 * 100);

    float tmp = 0;

    for (int i = 0; i < 100; ++i) {
        tmp += input1Ptr[i] * input2Ptr[i];
    }

    ASSERT_EQ(tmp, outputPtr[0]);

    /**x grad*/
    for (int i = 0; i < 100; ++i) {
        ASSERT_EQ(input1GradPtr[i], input2Ptr[i] * outputGradPtr[0]);
        ASSERT_EQ(input2GradPtr[i], input1Ptr[i] * outputGradPtr[0]);
    }
}

#endif

}

#endif
