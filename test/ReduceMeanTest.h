#ifndef DEEP8_REDUCEMEANTEST_H
#define DEEP8_REDUCEMEANTEST_H

#include "nodes/ReduceMean.h"

namespace Deep8 {

TEST(ReduceMean, CPU) {
    CPUDevice device;

    auto inputValue = createTensor(device, ElementType::from<float>(), 10, { 400, 200 });
    auto inputGrad  = createTensor(device, ElementType::from<float>(), 10, { 400, 200 });

    auto outputValue = createTensor(device, ElementType::from<float>(), 1, { 400, 1 });
    auto outputGrad  = createTensor(device, ElementType::from<float>(), 1, { 400, 1 });

    auto inputVar = createFakeVariable(device, ElementType::from<float>(), 10, { 400, 200 });

    std::vector<Node*> inputs = { &inputVar };
    ReduceMean reduceMean(inputs, {0, 2});

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = { &inputValue };

    reduceMean.forward(inputValues, &outputValue);
    reduceMean.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    /**test forward*/
    for (int dim1 = 0; dim1 < 400; ++dim1) {
        float temp = 0;

        for (int dim0 = 0; dim0 < 10; ++dim0) {
            for (int dim2 = 0; dim2 < 200; ++dim2) {
                temp += inputValue.data<float>()[dim0 * 400 * 200 + dim1 * 200 + dim2];
            }
        }

        ASSERT_TRUE(std::abs(temp / 10.0 / 200.0 - outputValue.data<float>()[dim1]) < 1e-5);
    }

    /**backward test*/
    for (int dim0 = 0; dim0 < 10; ++dim0) {
        for (int dim1 = 0; dim1 < 400; ++dim1) {
            for (int dim2 = 0; dim2 < 200; ++dim2) {
                ASSERT_TRUE(std::abs(inputGrad.data<float>()[dim0 * 400 * 200 + dim1 * 200 + dim2] - outputGrad.data<float>()[dim1] / 10.0 / 200.0) < 1e-5);
            }
        }
    }
}

#ifdef HAVE_CUDA

TEST(ReduceMean, GPU1) {
    typedef float real;

    GPUDevice device;

    size_t dim0 = 10, dim1 = 400, dim2 = 200;

    auto cpuInputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
    auto cpuInputGradPtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto cpuOutputValuePtr = (real*)malloc(sizeof(real) * dim0 * 1 * 1);
    auto cpuOutputGradPtr = (real*)malloc(sizeof(real) * dim0 * 1 * 1);

    auto inputValue = createTensor(device, cpuInputValuePtr, ElementType::from<real>(), dim0, { dim1, dim2 });
    auto inputGrad = createTensor(device, cpuInputGradPtr, ElementType::from<real>(), dim0, { dim1, dim2 });

    auto outputValue = createTensor(device, cpuOutputValuePtr, ElementType::from<real>(), dim0, { 1, 1 });
    auto outputGrad = createTensor(device, cpuOutputGradPtr, ElementType::from<real>(), dim0, { 1, 1 });

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<real>(), dim0, { dim1, dim2 });

    std::vector<Node*> inputs = { &inputVar };
    ReduceMean reduceMean(inputs, { 1, -1 });

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = { &inputValue };

    reduceMean.forward(inputValues, &outputValue);
    reduceMean.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(inputGrad.raw(), cpuInputGradPtr, sizeof(real) * dim0 * dim1 * dim2);
    device.copyFromGPUToCPU(outputValue.raw(), cpuOutputValuePtr, sizeof(real) * dim0 * 1 * 1);

    /**test forward*/
    for (int d0 = 0; d0 < 10; ++d0) {
        float temp = 0;

        for (int d1 = 0; d1 < 400; ++d1) {
            for (int d2 = 0; d2 < 200; ++d2) {
                temp += cpuInputValuePtr[d0 * 400 * 200 + d1 * 200 + d2];
            }
        }

        ASSERT_TRUE(std::abs(temp / 400.0 / 200.0 - cpuOutputValuePtr[d0]) < 1e-5);
    }

    /**backward test*/
    for (int d0 = 0; d0 < 10; ++d0) {
        for (int d1 = 0; d1 < 400; ++d1) {
            for (int d2 = 0; d2 < 200; ++d2) {
                ASSERT_TRUE(std::abs(cpuInputGradPtr[d0 * 400 * 200 + d1 * 200 + d2] - cpuOutputGradPtr[d0] / 400.0 / 200.0) < 1e-5);
            }
        }
    }
}

TEST(ReduceMean, GPU2) {
    typedef float real;

    GPUDevice device;

    size_t dim0 = 10, dim1 = 400, dim2 = 200;

    auto cpuInputValuePtr = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);
    auto cpuInputGradPtr  = (real*)malloc(sizeof(real) * dim0 * dim1 * dim2);

    auto cpuOutputValuePtr = (real*)malloc(sizeof(real) * 1 * dim1 * 1);
    auto cpuOutputGradPtr  = (real*)malloc(sizeof(real) * 1 * dim1 * 1);

    auto inputValue = createTensor(device, cpuInputValuePtr, ElementType::from<real>(), dim0, { dim1, dim2 });
    auto inputGrad  = createTensor(device, cpuInputGradPtr, ElementType::from<real>(), dim0, { dim1, dim2 });

    auto outputValue = createTensor(device, cpuOutputValuePtr, ElementType::from<real>(), 1, { dim1, 1 });
    auto outputGrad = createTensor(device, cpuOutputGradPtr, ElementType::from<real>(), 1, { dim1, 1 });

    /**create fake Add Function*/
    auto inputVar = createFakeVariable(device, ElementType::from<real>(), dim0, { dim1, dim2 });

    std::vector<Node*> inputs = { &inputVar };
    ReduceMean reduceMean(inputs, { 0, 2 });

    zeroTensor(device, inputGrad);

    std::vector<const Tensor*> inputValues = { &inputValue };

    reduceMean.forward(inputValues, &outputValue);
    reduceMean.backward(inputValues, &outputValue, &outputGrad, 0, &inputGrad);

    device.copyFromGPUToCPU(inputGrad.raw(), cpuInputGradPtr, sizeof(real) * dim0 * dim1 * dim2);
    device.copyFromGPUToCPU(outputValue.raw(), cpuOutputValuePtr, sizeof(real) * 1 * dim1 * 1);

    /**test forward*/
    for (int dim1 = 0; dim1 < 400; ++dim1) {
        float temp = 0;

        for (int dim0 = 0; dim0 < 10; ++dim0) {
            for (int dim2 = 0; dim2 < 200; ++dim2) {
                temp += cpuInputValuePtr[dim0 * 400 * 200 + dim1 * 200 + dim2];
            }
        }

        ASSERT_TRUE(std::abs(temp / 10.0 / 200.0 - cpuOutputValuePtr[dim1]) < 1e-5);
}

    /**backward test*/
    for (int dim0 = 0; dim0 < 10; ++dim0) {
        for (int dim1 = 0; dim1 < 400; ++dim1) {
            for (int dim2 = 0; dim2 < 200; ++dim2) {
                ASSERT_TRUE(std::abs(cpuInputGradPtr[dim0 * 400 * 200 + dim1 * 200 + dim2] - cpuOutputGradPtr[dim1] / 10.0 / 200.0) < 1e-5);
            }
        }
    }
}



#endif


}

#endif